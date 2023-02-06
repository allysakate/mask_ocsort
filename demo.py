import os
from typing import List
from functools import reduce
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.ops import masks_to_boxes
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from models.config import get_cfg

from trackers.deep_sort import tracker, detection, nn_matching
from trackers.ocsort_tracker import OCSort
from trackers.tracking_utils.timer import Timer
from utils import initiate_logging, get_config, create_output, get_IOP
from progress_bar import ProgressBar


class ParameterManager:
    """Manages parameter for video processing

    Args:
        yml_path (str): config file path

    Attributes:
        config (Config) :       configuration for video processing
        video (VideoCapture) :  cv2 video capture
        fps (bool) :            frame per second
        output_dir (str):       directory of output for viz and text
        model_device (str):     device parameter for model [cpu, cuda:0]
        frame_skip (int):       number of frames to be skipped when processing
        tracker_name (str):     tracker name based on feature and ocsort
        _tracker (object):      tracker object [OCSort, DeepSORT]
        predictor (object):     model predict
        metadata (object):      testing metadata
    """

    def __init__(self, yml_path: str):
        self.config = get_config(yml_path)
        self.tracker_name = self.config.tracker
        self.output_dir = create_output(
            self.config.output_dir, f"{self.config.tracker}/data"
        )
        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tracker = None
        self.use_ocsort, self.with_feature = self.set_useocsort_feat()

    def set_useocsort_feat(self):
        if self.tracker_name == "DeepOCSORT":
            return True, True
        elif self.tracker_name == "OCSORT":
            return True, False
        elif self.tracker_name == "DeepSORT":
            return False, True


class LayerHook:
    """Taps into the intermediate layer of a CNN Model"""

    def __init__(self):
        self.outputs = []
        self.inputs = []
        self.hookList = []

    def hook(self, module, input, output):
        self.inputs.append(input)
        self.outputs.append(output)

    def node2module(self, network: torch.nn.Module, node: str) -> nn.Module:
        module_list = node.split(".")
        module = network
        if module_list:
            module = reduce(getattr, [module, *module_list])

        return module

    def register(self, network: nn.Module, nodes: List) -> None:
        for node in nodes:
            module = self.node2module(network, node)
            hook = module.register_forward_hook(self.hook)
            self.hookList.append(hook)

    def remove(self) -> None:
        """Removes the hook registrations.
        Useful for re-inserting new hooks
        """
        [hook.remove() for hook in self.hookList]


class VideoProcessor:
    """Video processing
    Attributes:
        param (ParameterManager) :  parameter object
        video_file (str) :          video file path
        video (cv2.VideoCapture) :  cv2 video capture
        fps (float):                frame per second
        logger (logging):           logging object
        predictor (object):         model predict
        metadata (object):          testing metadata
        frame_skip (int):           number of frames to be skipped when processing
    """

    def __init__(self):
        self.param = ParameterManager("configs/demo_config.yaml")
        self.video = cv2.VideoCapture(self.param.config.video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.logger = initiate_logging(self.param.config.predictor)
        cfg = self.setup_config()
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

    @property
    def frame_skip(self):
        """Get number of frames to be skipped when processing"""
        if self.param.config.frame_use == "all":
            frame_use = self.fps
        else:
            frame_use = self.param.config.frame_use
        return int(self.fps / frame_use)

    def setup_config(self):
        """Load config from file and command-line arguments
        Returns:
            configuration
        """
        configuration = get_cfg()
        configuration.merge_from_file(self.param.config.config_file)
        configuration.merge_from_list(["MODEL.WEIGHTS", self.param.config.weight_file])
        configuration.MODEL.DEVICE = self.param.model_device
        # Set score_threshold for builtin models
        configuration.MODEL.RETINANET.SCORE_THRESH_TEST = (
            self.param.config.confidence_threshold
        )
        configuration.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.param.config.confidence_threshold
        )
        if self.param.config.predictor.upper() == "SOLOV2":
            configuration.MODEL.FCOS.INFERENCE_TH_TEST = (
                self.param.config.confidence_threshold
            )
            configuration.MODEL.MEInst.INFERENCE_TH_TEST = (
                self.param.config.confidence_threshold
            )
        # configuration.MODEL.SOLOV2.SCORE_THR = self.param.config.confidence_threshold
        configuration.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            self.param.config.confidence_threshold
        )
        configuration.freeze()
        return configuration

    def set_hooker(self):
        if self.param.config.predictor.upper() == "MASKRCNN":
            hooker = LayerHook()
            hooker.register(self.predictor.model, ["roi_heads.mask_pooler"])
        else:
            hooker = None
        return hooker

    def post_process(self, predictions, mask_pool=None):
        """Filter predictions
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            feat_masks, frame_boxes, scores, classes
        """
        # num_instances = len(predictions)
        # TODO: Filter out non-vehicle classes
        classes = (
            predictions.pred_classes.numpy()
            if predictions.has("pred_classes")
            else None
        )

        scores = predictions.scores.numpy() if predictions.has("scores") else None
        class_mask = np.isin(classes, self.param.config.class_list)
        score_mask = scores >= self.param.config.track_thresh
        filter_mask = np.logical_and(class_mask, score_mask)

        boxes = (
            predictions.pred_boxes.tensor.numpy()
            if predictions.has("pred_boxes")
            else None
        )

        boxes = boxes[filter_mask]
        classes = classes[filter_mask]
        scores = scores[filter_mask]
        if mask_pool is not None:
            mask_pool = mask_pool[filter_mask]

        if scores.any():

            periods = predictions.ID_period if predictions.has("ID_period") else None
            if periods:
                periods = periods[filter_mask]
            period_threshold = self.metadata.get("period_threshold", 0)
            visibilities = (
                [True] * len(boxes)
                if periods is None
                else [x > period_threshold for x in periods]
            )

            if predictions.has("pred_masks"):
                pred_masks = predictions.pred_masks
                pred_masks = pred_masks[filter_mask]

            else:
                pred_masks = None

            pred_masks = None if pred_masks is None else pred_masks[visibilities]
            seg_masks = pred_masks

            if mask_pool is None:
                if predictions.has("feat_masks"):
                    feat_masks = predictions.feat_masks
                    feat_masks = feat_masks[filter_mask]
                else:
                    feat_masks = None
                feat_masks = None if feat_masks is None else feat_masks[visibilities]
            else:
                feat_masks = mask_pool

            # labels = _create_text_labels(
            #     classes, scores, metadata.get("thing_classes", None)
            # )
            # labels = (
            #     None
            #     if labels is None
            #     else [y[0] for y in filter(lambda x: x[1], zip(labels, visibilities))]
            # )  # noqa

            # Display in largest to smallest order to reduce occlusion.
            areas = None
            if boxes is not None:
                areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            elif pred_masks is not None:
                areas = np.asarray([x.area() for x in pred_masks])

            frame_boxes = None
            if areas is not None:
                sorted_idxs = np.argsort(-areas).tolist()
                # Re-order overlapped instances in descending order.
                classes = (
                    [classes[idx] for idx in sorted_idxs]
                    if classes is not None
                    else None
                )
                pred_masks = (
                    [pred_masks[idx] for idx in sorted_idxs]
                    if pred_masks is not None
                    else None
                )
                scores = scores[sorted_idxs]
                seg_masks = seg_masks[sorted_idxs, :, :]
                if feat_masks is not None:
                    feat_masks = feat_masks[sorted_idxs]
                else:
                    feat_masks = np.empty(len(scores))
                if seg_masks is not None:
                    frame_boxes = masks_to_boxes(seg_masks)

            return feat_masks, frame_boxes, scores, classes

    def plot_tracking(
        self,
        image,
        tlwhs,
        obj_ids,
        classes,
        scores=None,
        frame_id=0,
        fps=0.0,
        ids2=None,
    ):
        """Plots predictions"""

        def get_color(idx):
            idx = idx * 3
            color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

            return color

        im = np.ascontiguousarray(np.copy(image))
        # im_h, im_w = im.shape[:2]

        # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        # text_scale = max(1, image.shape[1] / 1600.)
        # text_thickness = 2
        # line_thickness = max(1, int(image.shape[1] / 500.))
        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        # radius = max(5, int(im_w / 140.0))
        # cv2.putText(
        #     im,
        #     "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        #     (0, int(15 * text_scale)),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     (0, 0, 255),
        #     thickness=2,
        # )

        for i, tlwh in enumerate(tlwhs):
            # if classes:
            #     label = classes[i]
            # else:
            #     label = None
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ", {}".format(int(ids2[i]))
            color = get_color(abs(obj_id))
            cv2.rectangle(
                im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
            )
            cv2.putText(
                im,
                id_text,
                (intbox[0], intbox[1]),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=text_thickness,
            )
            # if label:
            #     cv2.putText(
            #         im,
            #         label,
            #         (intbox[0], intbox[1] + 20),
            #         cv2.FONT_HERSHEY_PLAIN,
            #         text_scale,
            #         (0, 0, 255),
            #         thickness=text_thickness,
            #     )
        return im

    def filter_box(self, box, frame=None):
        """Filter Bounding Box
        Args:
            box (List[int]): bounding box
        Returns:
            is_included, frame
        """
        is_included = False
        area = (box[2] - box[0]) * (box[3] - box[1])
        if self.param.config.roi:
            roi_iop = get_IOP(self.param.config.roi, box)
            condition = roi_iop < self.param.config.iop_threshold and area > 5900
        else:
            condition = True
        if condition:
            is_included = True
            # if frame is not None:
            #     intbox = tuple(map(int, box))
            #     cv2.rectangle(
            #         frame, intbox[0:2], intbox[2:4], color=(0, 0, 255), thickness=1
            #     )
        return is_included, frame

    def ocsort(
        self, frame_id, feat_masks, boxes, scores, classes, results, thing_classes
    ):
        detections = []
        for feat, box, score, label in zip(feat_masks, boxes, scores, classes):
            is_included, _ = self.filter_box(box)
            if is_included:
                det = np.insert(box, 4, score)
                det = np.insert(det, 5, label)
                feat = np.array(feat).flatten()
                # feat = np.array(feat)
                # shape = min(feat.shape)
                # feat = np.resize(feat, (shape, shape)).flatten()
                det = np.asarray(np.insert(det, 6, feat))
                detections.append(det)
        targets = self._tracker.update(np.asarray(detections), self.param.with_feature)
        track_tlwhs = []
        track_ids = []
        track_classes = []
        for tlbrs in targets:
            box_width = tlbrs[2] - tlbrs[0]
            box_height = tlbrs[3] - tlbrs[1]
            tlwh = [
                tlbrs[0],
                tlbrs[1],
                tlbrs[2] - tlbrs[0],
                tlbrs[3] - tlbrs[1],
            ]
            tid = tlbrs[5]
            tclass = int(tlbrs[4])
            vertical = box_width / box_height > self.param.config._aspect_ratio_thresh
            if (
                box_width * box_height > self.param.config._min_box_area_thresh
                and not vertical
            ):
                track_tlwhs.append(tlwh)
                track_ids.append(tid)
                if tclass in self.param.config.class_list:
                    track_classes.append(thing_classes[tclass])
                else:
                    track_classes.append(None)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                )
        return track_tlwhs, track_ids, track_classes, results

    def deepsort(
        self, frame_id, feat_masks, boxes, scores, classes, results, thing_classes
    ):
        detections = []
        for feat, box, score, label in zip(feat_masks, boxes, scores, classes):
            is_included, _ = self.filter_box(box)
            if is_included:
                feat = np.array(feat)
                shape = min(feat.shape)
                feature = np.resize(feat, (shape, shape)).flatten()
                det = detection.Detection(
                    np.array(box), score, thing_classes[label], feature
                )
                detections.append(det)
        self._tracker.predict()
        self._tracker.update(detections)
        track_tlwhs = []
        track_ids = []
        track_classes = []
        for track in self._tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            tlwh = track.to_tlwh()
            box_width = tlwh[2]
            box_height = tlwh[3]
            tid = track.track_id
            vertical = box_width / box_height > self.param.config._ASPECT_RATIO_THRESH
            if (
                box_width * box_height > self.param.config._MIN_BOX_AREA_THRESH
                and not vertical
            ):
                track_tlwhs.append(tlwh)
                track_ids.append(tid)
                track_classes.append(track.classification)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                )
        return track_tlwhs, track_ids, track_classes, results

    def process(self):
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename, _ = os.path.splitext(os.path.basename(self.param.config.video_path))
        # current_time = time.localtime()
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        output_name = self.param.config.text_filename.upper()
        save_folder = os.path.join(self.param.output_dir, output_name)
        os.makedirs(save_folder, exist_ok=True)
        if self.param.config.is_save_result:
            save_path = os.path.join(save_folder, basename)
            self.logger.info(f"video save_path is {save_path}")
            vid_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (int(width), int(height)),
            )

        frame_id = 0
        thing_classes = self.metadata.get("thing_classes", None)
        # thing_classes[0] = "car"
        # thing_classes[1] = "person"
        # thing_classes[2] = "ignore"
        if not self.param.use_ocsort:
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
            self._tracker = tracker.Tracker(metric)
        else:
            self._tracker = OCSort()
        timer = Timer()
        frame_id = 0
        results = []

        frame_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        prog_bar = ProgressBar(int(frame_count), fmt=ProgressBar.FULL)
        while True:
            if frame_id % self.fps == 0:
                self.logger.info(
                    "Processing frame {} ({:.2f} fps)".format(
                        frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )
            ret_val, frame = self.video.read()
            if ret_val:
                raw_img = frame
                timer.tic()
                if int(frame_id % self.frame_skip) == 0 and frame_id != 0:
                    hooker = self.set_hooker()
                    predictions = self.predictor(frame)
                    predictions = predictions["instances"].to(torch.device("cpu"))
                    if hooker:
                        mask_pool = hooker.outputs[-1].cpu().numpy()
                        hooker.remove()
                    else:
                        mask_pool = None
                    # try:
                    feat_masks, boxes, scores, classes = self.post_process(
                        predictions, mask_pool
                    )
                    if boxes.any():
                        # TODO: tracking
                        if not self.param.use_ocsort:
                            (
                                track_tlwhs,
                                track_ids,
                                track_classes,
                                results,
                            ) = self.deepsort(
                                frame_id,
                                feat_masks,
                                boxes,
                                scores,
                                classes,
                                results,
                                thing_classes,
                            )
                        else:
                            (
                                track_tlwhs,
                                track_ids,
                                track_classes,
                                results,
                            ) = self.ocsort(
                                frame_id,
                                feat_masks,
                                boxes,
                                scores,
                                classes,
                                results,
                                thing_classes,
                            )
                        timer.toc()
                        res_image = self.plot_tracking(
                            raw_img,
                            track_tlwhs,
                            track_ids,
                            track_classes,
                            frame_id=frame_id + 1,
                            fps=1.0 / timer.average_time,
                        )
                        res_path = os.path.join(save_folder, f"{frame_id}.jpg")
                        for box in boxes:
                            _, res_image = self.filter_box(box, res_image)
                        cv2.imwrite(res_path, res_image)
                    # except Exception as e:
                    #     print(frame_id, e)
                else:
                    timer.toc()
                    res_image = raw_img
                if self.param.config.is_save_result:
                    vid_writer.write(res_image)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1
            # Update progress bar
            prog_bar.current += 1
            prog_bar()
        prog_bar.done()
        if self.param.config.is_save_result:
            res_file = os.path.join(self.param.output_dir, f"{output_name}.txt")
            with open(res_file, "w") as f:
                f.writelines(results)
            self.logger.info(f"save results to {res_file}")


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process()
