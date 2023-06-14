import os
import csv
import math
import yaml
import argparse
from typing import List
from functools import reduce
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from torchvision.ops import masks_to_boxes
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
from models.config import get_cfg
from models.utils.datasets import letterbox
from models.utils.general import non_max_suppression_mask_conf
from models.feature_extractor import FeatExtractor
from trackers.deep_sort import tracker, detection, nn_matching
from trackers.ocsort_tracker import OCSort
from trackers.tracking_utils.timer import Timer
from utils import initiate_logging, get_config, create_output, get_IOP, close_logging, calculate_iou, colors_instance
from progress_bar import ProgressBar

IOU_LIST = []
SCALE_LIST = []


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
        self.matcher = self.config.matcher
        if self.matcher:
            self.matcher = self.config.matcher.upper()
        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predictor_name = None
        self.tracker_name = None
        self.text_filename = None
        self.folder_name = None
        self.use_ocsort = None
        self.with_feature = None
        self.frame_use = self.config.frame_use
        self.iou_weight = self.config.iou_weight
        self.vd_weight = self.config.vd_weight
        self.min_hits = self.config.min_hits
        self.max_age = self.config.max_age

    def set_text_filename(self):
        pred_name = self.predictor_name
        text = f"{pred_name}_FPS{self.frame_use}_{self.matcher}_MA{str(self.max_age)}"
        self.text_filename = text.upper()

    def set_useocsort_feat(self):
        if self.tracker_name == "MASKOCSORT":
            self.folder_name = "CATaft03-test"
            self.use_ocsort = True
            self.with_feature = True
        elif self.tracker_name == "OCSORT":
            self.folder_name = "CATaft02-test"
            self.use_ocsort = True
            self.with_feature = False
        elif self.tracker_name == "DEEPSORT":
            self.folder_name = "CATaft01-test"
            self.use_ocsort = False
            self.with_feature = True


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

    def __init__(self, param, is_yolo):
        self.param = param
        import glob
        self.video = sorted(glob.glob(os.path.join(self.param.config.video_path, "*.jpg")))
        self.fps = 25
        self.device = torch.device(self.param.model_device)
        self.cfg = None
        self.class_names = None
        self.period_threshold = 0
        if not is_yolo:
            self.predictor = self.set_det2model()
        else:
            self.predictor = self.set_yolomodel()
        self.output_dir = create_output(
            param.config.output_dir, f"{param.folder_name}/{param.tracker_name}/data"
        )
        self.det_dir = create_output(
            param.config.det_dir, f"{param.tracker_name}/{param.text_filename}"
        )
        self.image_dir = create_output(
            param.config.det_dir, f"{param.tracker_name}/{param.text_filename}/images"
        )
        self.data_attr = create_output(
            param.config.det_dir, f"{param.tracker_name}/{param.text_filename}/params"
        )
        self.data_assoc = create_output(
            param.config.det_dir, f"{param.tracker_name}/{param.text_filename}/assoc"
        )
        if not param.matcher:
            self.det_res_dir = create_output(
                param.config.det_dir, f"det_results/{param.text_filename}"
            )
        elif param.matcher == "COSINESL":
            if param.model_device == "cuda:0":
                use_cuda = True
            else:
                use_cuda = False
            # Extractor: "checkpoints/CosineMetric_Learning.t7", use_cuda=use_cuda
            self.extractor = FeatExtractor("alexnet-fc7", use_cuda=use_cuda)
        log_dir = create_output(param.config.det_dir, f"{param.tracker_name}/logs")
        self.logger = initiate_logging(log_dir, param.text_filename)
        self.bbox_ids = []
        self.bbox_list = []
        self.prev_img = None


    @property
    def frame_skip(self):
        """Get number of frames to be skipped when processing"""
        if self.param.frame_use == "all":
            frame_use = self.fps
        else:
            frame_use = self.param.frame_use
        return int(self.fps / frame_use)

    def set_yolomodel(self):
        with open(self.param.config.config_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        weights = torch.load(self.param.config.weight_file)
        model = weights["model"]
        model = model.half().to(self.device)
        model.eval()
        self.class_names = model.names
        return model

    def set_frame(self, image):
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()
        return image

    def set_det2model(self):
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
        if self.param.predictor_name == "SOLOV2":
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
        model = DefaultPredictor(configuration)
        metadata = MetadataCatalog.get(
            configuration.DATASETS.TEST[0]
            if len(configuration.DATASETS.TEST)
            else "__unused"
        )
        self.period_threshold = metadata.get("period_threshold", 0)
        self.class_names = metadata.get("thing_classes", None)
        return model

    def set_hooker(self, hooker=None):
        if self.param.predictor_name == "MASKRCNN":
            if (
                self.param.tracker_name in ["DEEPSORT", "MASKOCSORT"]
                and self.param.matcher is None
            ):
                hooker = LayerHook()
                if self.param.predictor_name == "MASKRCNN":
                    node = "roi_heads.mask_pooler"  # (19, 256, 14, 14)
                hooker.register(self.predictor.model, [node])
        return hooker

    def yolo_process(self, predictions, frame_dim, orig_image_dim):
        inf_out, _, attn, _, bases, sem_output = (
            predictions["test"],
            predictions["bbox_and_cls"],
            predictions["attn"],
            predictions["mask_iou"],
            predictions["bases"],
            predictions["sem"],
        )
        bases = torch.cat([bases, sem_output], dim=1)
        height, width = frame_dim
        pooler_scale = self.predictor.pooler_scale
        pooler = ROIPooler(
            output_size=self.cfg["mask_resolution"],
            scales=(pooler_scale,),
            sampling_ratio=1,
            pooler_type="ROIAlignV2",
            canonical_level=2,
        )
        output, output_mask, _, _, _ = non_max_suppression_mask_conf(
            inf_out,
            attn,
            bases,
            pooler,
            self.cfg,
            conf_thres=0.25,
            iou_thres=0.65,
            merge=False,
            mask_iou=None,
        )
        pred, pred_masks = output[0], output_mask[0]
        _ = bases[0]
        bboxes = Boxes(pred[:, :4])
        feat_masks = pred_masks.view(
            -1, self.cfg["mask_resolution"], self.cfg["mask_resolution"]
        )
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            feat_masks, bboxes, (height, width), threshold=0.5
        )
        seg_masks = pred_masks.detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        # TODO: Fix scaling
        raw_h, raw_w = orig_image_dim
        y_scale, x_scale = raw_h / height, raw_w / width
        bboxes = bboxes.tensor.detach().cpu().numpy()
        boxes = []
        for box in bboxes:
            x = int(np.round(box[0] * x_scale))
            y = int(np.round(box[1] * y_scale))
            xmax = int(np.round(box[2] * (x_scale)))
            ymax = int(np.round(box[3] * y_scale))
            boxes.append([x, y, xmax, ymax])
        boxes = np.array(boxes)
        feat_masks = feat_masks.cpu().numpy()
        return feat_masks, seg_masks, boxes, scores, classes

    def det2_process(self, predictions, mask_pool=None, matcher=None):
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
            visibilities = (
                [True] * len(boxes)
                if periods is None
                else [x > self.period_threshold for x in periods]
            )

            if predictions.has("pred_masks"):
                pred_masks = predictions.pred_masks
                pred_masks = pred_masks[filter_mask]

            else:
                pred_masks = None

            pred_masks = None if pred_masks is None else pred_masks[visibilities]
            seg_masks = pred_masks

            if matcher is None:
                if mask_pool is None:
                    if predictions.has("feat_masks"):
                        feat_masks = predictions.feat_masks
                        feat_masks = feat_masks[filter_mask]
                    else:
                        feat_masks = None
                    feat_masks = (
                        None if feat_masks is None else feat_masks[visibilities]
                    )
                else:
                    feat_masks = mask_pool
            else:
                feat_masks = None

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

            return feat_masks, seg_masks, frame_boxes, scores, classes

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
        is_included = True
        area = (box[2] - box[0]) * (box[3] - box[1])
        if self.param.config.roi:
            for roi in self.param.config.roi:
                roi_iop = get_IOP(roi, box)
                condition = roi_iop < self.param.config.iop_threshold # and area > 5900
                if not condition:
                    is_included = False
                    break
        if condition:
            is_included = True
            if frame is not None:
                intbox = tuple(map(int, box))
                cv2.rectangle(
                    frame, intbox[0:2], intbox[2:4], color=(0, 0, 0), thickness=1
                )
        return is_included, frame

    def mask2polygon(self, image, masks):
        masks = masks.numpy()
        image_copy = image.copy()
        image_mask = np.zeros(image.shape, dtype=np.uint8)
        for i in range(3):
            image_mask[:, :, i] = masks.astype(int)
        image_mask[image_mask == 1] = 255
        result = cv2.bitwise_and(image_copy, image_mask)
        # contours, _ = cv2.findContours(image=mask_reformed,mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # # or you can also method=cv2.CHAIN_APPROX_NONE
        # # check the contours
        # # Mask input image with binary mask
        # result = cv2.bitwise_and(image, mask_reformed)
        # # Color background white
        # result[mask_reformed==0] = 255 # Optional
        # cv2.fillPoly(result, pts=[contours], color=(255, 0, 0))
        cv2.imshow("result", result)
        cv2.waitKey()
        return result

    def visualize(self, raw_img, bboxes):
        def point_equation(point1, point2):
            x1, y1 = list(map(int, point1))
            x2, y2 = list(map(int, point2))
            m = (y2 - y1) / (x2 - x1 + 1)
            x = x1 + 30
            return abs(m*(x - x1) - y1)

        text_scale = 1.2
        text_thickness = 2
        line_thickness = 3
        im = raw_img.copy()
        for roi in self.param.config.roi:
            cv2.rectangle(
                im, roi[0:2], roi[2:4], color=(255,255,255), thickness=line_thickness
            )
        boxes = []
        for bb in bboxes:
            ret, _ = self.filter_box(bb)
            if ret:
                boxes.append(bb)

        for i, box in enumerate(boxes):
            intbox = list(map(int, box))
            i_color = colors_instance[i]
            cv2.rectangle(
                im, intbox[0:2], intbox[2:4], color=i_color, thickness=line_thickness
            )
            cv2.rectangle(
                im, (intbox[0], intbox[1]-20), (intbox[0]+180, intbox[1]), color=(0,0,0), thickness=-1
            )
            area = (box[2] - box[0]) * (box[3] - box[1])
            scale = math.sqrt(area)
            SCALE_LIST.append(scale)
            i_text = f"{i+1}, Scale: {scale:.2f}"
            cv2.putText(
                im,
                i_text,
                (intbox[0], intbox[1]-5),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (255, 255, 255),
                thickness=text_thickness,
            )
            dist_list = []
            next_ids = []
            next_bbox = []
            if len(self.bbox_ids):
                for b_id, bbox in zip(self.bbox_ids, self.bbox_list):
                    cx1, cy1 = (intbox[0] + intbox[2]) / 2.0, (intbox[1] + intbox[3]) / 2.0
                    cx2, cy2 = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                    distance = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
                    try:
                        new_cy = point_equation((cx1, cy1), (cx2, cy2))
                        d_text = f"{b_id+1}: {distance:.2f}"
                        dist_list.append([b_id, d_text, [cx1, cy1, cx1+30, new_cy]])
                        next_ids.append(b_id)
                        next_bbox.append(bbox)
                    except ZeroDivisionError:
                        continue

                for d_list in dist_list:
                    idx, d_text, vis_pts = d_list
                    vis_pts = list(map(int, vis_pts))
                    d_color = colors_instance[idx]
                    cv2.line(im, vis_pts[0:2], vis_pts[2:4], d_color, text_thickness)
                    cv2.putText(
                        im,
                        d_text,
                        vis_pts[2:4],
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        d_color,
                        thickness=text_thickness,
                    )
                self.bbox_ids = next_ids
                self.bbox_list = next_bbox
            else:
                self.bbox_ids = range(len(boxes))
                self.bbox_list = boxes

            text_list = []
            for j, box in enumerate(boxes):
                if i != j:
                    j_intbox = list(map(int,box))
                    j_iop = calculate_iou(j_intbox, intbox)
                    if j_iop > 0.01:
                        IOU_LIST.append(j_iop)
                        j_text = f"{j+1}: {j_iop:.2f}"
                        text_list.append(j_text)
            height = int(len(text_list)*20)
            cv2.rectangle(
                im, intbox[0:2], (intbox[0]+100, intbox[1]+height), color=(255,255,255), thickness=-1
            )
            for idx, j_text in enumerate(text_list):
                j_color = colors_instance[idx]
                cv2.putText(
                    im,
                    j_text,
                    (intbox[0], intbox[1]+14*(idx+1)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    j_color,
                    thickness=text_thickness,
                )
        return im


    def ocsort(
        self,
        data,
        raw_img,
        frame_id,
        feat_masks,
        seg_masks,
        boxes,
        scores,
        classes,
        trk_results,
        det_results,
        class_names,
        width,
    ):
        detections = []
        # cnt = 0
        img_v = self.visualize(raw_img, boxes)
        for feat, mask, box, score, label in zip(
            feat_masks, seg_masks, boxes, scores, classes
        ):
            is_included, _ = self.filter_box(box)
            if is_included and int(label) in self.param.config.class_list:
                # class_name = class_names[int(label)]
                intbox = list(map(int, box))
                det_results.append(
                    f"car {float(score)} {intbox[0]} {intbox[1]} {intbox[2]-intbox[0]} {intbox[3]-intbox[1]}\n"
                )
                det = np.insert(box, 4, score)
                det = np.insert(det, 5, label)
                if self.param.matcher is None and self.param.with_feature:
                    if self.param.predictor_name == "SOLOV2":
                        max_pool = nn.MaxPool1d(3, stride=12)
                        feat = max_pool(feat).numpy()
                    elif self.param.predictor_name == "MASKRCNN":
                        max_pool = nn.MaxPool2d(3, stride=3)
                        feat = max_pool(feat).numpy()
                    feat = np.array(feat).flatten()
                elif self.param.matcher == "COSINESL" and self.param.with_feature:
                    # new_image = self.mask2polygon(raw_img, mask)
                    crop = raw_img[intbox[1] : intbox[3], intbox[0] : intbox[2]]
                    is_crop = any(v == 0 for v in crop.shape)
                    if is_crop:
                        is_included = False
                    else:
                        feat = self.extractor(crop)
                else:
                    feat = np.array(feat).flatten()
                if is_included:
                    det = np.asarray(np.insert(det, 6, feat))
                    detections.append(det)
        if detections:
            assoc_img, data, targets = self._tracker.update(
                data, frame_id, raw_img, np.asarray(detections), self.param.with_feature, self.param.matcher, width
            )
        else:
            assoc_img = raw_img
            targets = []
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
            # vertical = box_width / box_height > self.param.config._aspect_ratio_thresh
            if (
                box_width * box_height > self.param.config._min_box_area_thresh
                # and not vertical
            ):
                track_tlwhs.append(tlwh)
                track_ids.append(tid)
                if tclass in self.param.config.class_list:
                    track_classes.append(class_names[tclass])
                else:
                    track_classes.append(None)
                trk_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                )

        return assoc_img, data, img_v, track_tlwhs, track_ids, track_classes, trk_results, det_results

    def deepsort(
        self,
        raw_img,
        frame_id,
        feat_masks,
        boxes,
        scores,
        classes,
        trk_results,
        det_results,
        class_names,
    ):
        detections = []
        for feat, box, score, label in zip(feat_masks, boxes, scores, classes):
            is_included, _ = self.filter_box(box)
            if is_included and int(label) in self.param.config.class_list:
                # class_name = class_names[int(label)]
                intbox = list(map(int, box))
                det_results.append(
                    f"car {float(score)} {intbox[0]} {intbox[1]} {intbox[2]-intbox[0]} {intbox[3]-intbox[1]}\n"
                )
                if self.param.matcher is None:
                    feat = np.array(feat).flatten()
                elif self.param.matcher == "COSINESL":
                    # x_feat = np.array(feat).flatten()
                    crop = raw_img[intbox[1] : intbox[3], intbox[0] : intbox[2]]
                    # crop = cv2.resize(crop, (256, 256))
                    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # feat = np.array(gray).flatten()
                    is_crop = any(v == 0 for v in crop.shape)
                    if is_crop:
                        is_included = False
                    else:
                        feat = self.extractor(crop)
                    # shape = min(feat.shape)
                    # feat = np.resize(feat, (shape, shape)).flatten()
                if is_included:
                    det = detection.Detection(
                        np.array(box), score, class_names[int(label)], feat
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
            # vertical = box_width / box_height > self.param.config._aspect_ratio_thresh
            if (
                box_width * box_height > self.param.config._min_box_area_thresh
                # and not vertical
            ):
                track_tlwhs.append(tlwh)
                track_ids.append(tid)
                track_classes.append(track.classification)
                trk_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                )
        return track_tlwhs, track_ids, track_classes, trk_results, det_results

    def process(self, is_yolo):
        first_frame = cv2.imread(self.video[0])
        height, width, _ = first_frame.shape
        # width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = "MVI_39031"
        # current_time = time.localtime()
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        if self.param.config.is_save_result:
            save_path = os.path.join(self.det_dir, f"{basename}.mp4")
            self.logger.info(f"video save_path is {save_path}")
            vid_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (int(width), int(height)),
            )

        frame_id = 0
        if not self.param.use_ocsort:
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
            self._tracker = tracker.Tracker(metric)
        else:
            self._tracker = OCSort(
                iou_weight=self.param.iou_weight,
                inertia=self.param.vd_weight,
                max_age=self.param.max_age,
                min_hits=self.param.min_hits,
                out_folder=self.data_assoc,
                )
        timer = Timer()
        det_timer = Timer()
        trk_timer = Timer()
        frame_id = 0
        trk_results = []
        data = []
        frame_count = len(self.video)
        prog_bar = ProgressBar(int(frame_count), fmt=ProgressBar.FULL)

        # while True:
        for image in self.video:
            basename, _ = os.path.splitext(os.path.basename(image))
            frame_id = int(basename.replace("img", "")) - 1
            if frame_id % self.fps == 0:
                self.logger.info(
                    "Processing frame {} ({:.2f} fps)".format(
                        frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )
            frame = cv2.imread(image)
            # if ret_val:
            raw_img = frame
            frame_copy = frame.copy()
            det_results = []
            timer.tic()
            if int(frame_id % self.frame_skip) == 0 and frame_id != 0:
                det_timer.tic()
                hooker = self.set_hooker()

                if not is_yolo:
                    predictions = self.predictor(frame)
                    predictions = predictions["instances"].to(torch.device("cpu"))

                    if hooker and self.param.matcher is None:
                        mask_pool = hooker.outputs[-1].cpu()
                        hooker.remove()
                    else:
                        mask_pool = None

                    (
                        feat_masks,
                        seg_masks,
                        boxes,
                        scores,
                        classes,
                    ) = self.det2_process(
                        predictions,
                        mask_pool,
                        self.param.matcher,
                    )

                else:
                    frame = self.set_frame(frame)
                    _, _, f_height, f_width = frame.shape
                    predictions = self.predictor(frame)
                    (
                        feat_masks,
                        seg_masks,
                        boxes,
                        scores,
                        classes,
                    ) = self.yolo_process(
                        predictions, (f_height, f_width), (height, width)
                    )
                det_timer.toc()
                trk_timer.tic()
                if boxes.any():
                    # TODO: tracking
                    if not self.param.use_ocsort:
                        (
                            track_tlwhs,
                            track_ids,
                            track_classes,
                            trk_results,
                            det_results,
                        ) = self.deepsort(
                            frame_copy,
                            frame_id,
                            feat_masks,
                            boxes,
                            scores,
                            classes,
                            trk_results,
                            det_results,
                            self.class_names,
                        )
                    else:
                        (   
                            assoc_img,
                            data,
                            img_v,
                            track_tlwhs,
                            track_ids,
                            track_classes,
                            trk_results,
                            det_results,
                        ) = self.ocsort(
                            data,
                            frame_copy,
                            frame_id,
                            feat_masks,
                            seg_masks,
                            boxes,
                            scores,
                            classes,
                            trk_results,
                            det_results,
                            self.class_names,
                            width,
                        )
                    trk_timer.toc()
                    timer.toc()
                    res_image = self.plot_tracking(
                        raw_img,
                        track_tlwhs,
                        track_ids,
                        track_classes,
                        frame_id=frame_id + 1,
                        fps=1.0 / timer.average_time,
                    )
                    res_path = os.path.join(self.image_dir, f"{frame_id}.jpg")
                    for box in boxes:
                        _, res_image = self.filter_box(box, res_image)
                    cv2.imwrite(res_path, res_image)
                    # Write det lines fro evaluation
                    new_gt_file = open(
                        os.path.join(self.det_dir, f"{frame_id:06d}.txt"), "w"
                    )
                    new_gt_file.writelines(det_results)
                    new_gt_file.close()  # to change file access modes
                else:
                    res_image = raw_img
                if self.param.config.is_save_result:
                    vid_writer.write(res_image)
                    iou_path = os.path.join(self.data_attr, f"{frame_id}.jpg")
                    if img_v.any():
                        # Stack the images horizontally
                        stacked_image = cv2.hconcat([img_v, res_image])
                        cv2.imwrite(iou_path, stacked_image)
            # else:
            #     timer.toc()
            #     self.logger.info(f"Average Time: {timer.average_time}")
            #     self.logger.info(f"Total Time: {timer.total_time}")
            #     self.logger.info(
            #         f"Time: {det_timer.total_time}, {det_timer.average_time}, {trk_timer.total_time}, {trk_timer.average_time}, {timer.total_time}, {timer.average_time}"
            #     )
            #     break
            # frame_id += 1

            # Update progress bar
            prog_bar.current += 1
            prog_bar()
        prog_bar.done()

        if self.param.config.is_save_result:
            res_file = os.path.join(self.output_dir, f"{self.param.text_filename}.txt")
            with open(res_file, "w") as f:
                f.writelines(trk_results)
            self.logger.info(f"save trk_results to {res_file}")


        # Specify the CSV file path
        csv_file = os.path.join(self.data_attr, "output.csv")
        # Extract the field names from the first dictionary
        fieldnames = data[0].keys()
        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames)
            # Write the header row
            writer.writeheader()
            # Write the rows to the CSV file
            writer.writerows(data)
        print("CSV file created successfully.")

def start_process(param, is_yolo=False):
    param.set_text_filename()
    param.set_useocsort_feat()
    processor = VideoProcessor(param, is_yolo)
    processor.process(is_yolo)
    close_logging(processor.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="if batch ")
    args = parser.parse_args()
    is_yolo = False
    param = ParameterManager("configs/demo_config.yaml")
    if args.batch:
        for frame_use in [1, 5, 25]:
            for value in [10,30,50]:
                param.max_age = value
                param.frame_use = frame_use
                for predictor in ["SOLOV2"]:
                    param.predictor_name = predictor.upper()
                    param.config.predictor = param.predictor_name
                    if param.predictor_name == "SOLOV2":
                        param.config.config_file = "configs/SOLOv2/SOLOv2_R50_FPN_3x.yaml"
                        param.config.weight_file = "checkpoints/SOLOv2_R50_3x.pth"
                    elif param.predictor_name == "MASKRCNN":
                        param.config.config_file = (
                            "configs/MaskRCNN/MaskRCNN_R50_FPN_3x.yaml"
                        )
                        param.config.weight_file = "checkpoints/MaskRCNN_R50_3x.pkl"
                    elif param.predictor_name == "FASTERRCNN":
                        param.config.config_file = (
                            "configs/MaskRCNN/FasterRCNN_R50_FPN_3x.yaml"
                        )
                        param.config.weight_file = "checkpoints/FasterRCNN_R50_3x.pkl"
                    elif param.predictor_name == "YOLOV7":
                        is_yolo = True
                        param.config.config_file = "configs/YOLOv7_Mask.yaml"
                        param.config.weight_file = "checkpoints/YOLOv7_Mask.pt"
                    for tracker_name in ["MASKOCSORT"]:
                        param.tracker_name = tracker_name.upper()
                        if tracker_name == "MASKOCSORT":
                            matchers = [None]
                        else:
                            matchers = ["COSINESL"]
                        for matcher in matchers:
                            param.matcher = matcher
                            print(
                                f"Processing... Tracker: {param.tracker_name}, Predictor: {param.predictor_name}, Matcher: {param.matcher} Frame Use: {param.frame_use}"
                            )
                            start_process(param, is_yolo)
                            print()
                    is_yolo = False
    else:
        param.predictor_name = param.config.predictor.upper()
        param.tracker_name = param.config.tracker.upper()
        if param.predictor_name == "YOLOV7":
            is_yolo = True
        start_process(param, is_yolo)
    print(min(IOU_LIST), max(IOU_LIST))
    print(min(SCALE_LIST), max(SCALE_LIST))
