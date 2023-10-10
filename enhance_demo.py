import os
import argparse
import torch
import cv2
import numpy as np
from glob import glob
from torchvision.ops import masks_to_boxes
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from models.config import get_cfg
from utils import (
    initiate_logging,
    get_config,
    create_output,
    colors_instance,
)
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
        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predictor_name = None
        self.text_filename = None
        self.folder_name = None
        self.frame_use = self.config.frame_use


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

    def __init__(self, param):
        self.param = param
        if self.param.config.image_path:
            self.video = sorted(
                glob(os.path.join(self.param.config.image_path, "*.jpg"))
            )
            self.fps = 25
            first_frame_path = self.video[0]
            # self.basename = os.path.dirname(first_frame_path).split("/")[-1]
            self.basename = "ch03_20230601193640"
            first_frame = cv2.imread(first_frame_path)
            self.height, self.width, _ = first_frame.shape
            self.num_frames = len(self.video)
        else:
            self.video = cv2.VideoCapture(self.param.config.video_path)
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.basename, _ = os.path.splitext(
                os.path.basename(self.param.config.video_path)
            )
        self.device = torch.device(self.param.model_device)
        self.cfg = None
        self.class_names = None
        self.period_threshold = 0
        self.predictor = self.set_det2model()
        self.output_dir = create_output(param.config.output_dir, self.basename)
        self.raw_dir = create_output(self.output_dir, "raw")
        self.vis_dir = create_output(self.output_dir, "images")
        self.det_dir = create_output(self.output_dir, "det")
        self.logger = initiate_logging(self.output_dir, param.text_filename)
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

    def det2_process(self, predictions):
        """Filter predictions
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            frame_boxes, scores, classes
        """
        # num_instances = len(predictions)
        # TODO: Filter out non-vehicle classes
        seg_masks, frame_boxes, scores, classes = [], [], [], []
        classes = (
            predictions.pred_classes.numpy()
            if predictions.has("pred_classes")
            else None
        )

        scores = predictions.scores.numpy() if predictions.has("scores") else None
        class_mask = np.isin(classes, self.param.config.class_list)
        score_mask = scores >= self.param.config.confidence_threshold
        filter_mask = np.logical_and(class_mask, score_mask)

        boxes = (
            predictions.pred_boxes.tensor.numpy()
            if predictions.has("pred_boxes")
            else None
        )

        boxes = boxes[filter_mask]
        classes = classes[filter_mask]
        scores = scores[filter_mask]

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
                if seg_masks is not None:
                    frame_boxes = masks_to_boxes(seg_masks)

        return frame_boxes, scores, classes

    def perform_task(
        self,
        frame,
        frame_id,
    ):
        det_results = []
        box_cnt = 0
        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        raw_img = frame.copy()
        vis_img = frame.copy()
        if int(frame_id % self.frame_skip) == 0:
            vis_path = os.path.join(self.vis_dir, f"{self.basename}_{frame_id}.jpg")
            raw_path = os.path.join(self.raw_dir, f"{self.basename}_{frame_id}.jpg")
            cv2.imwrite(raw_path, raw_img)
            predictions = self.predictor(frame)
            predictions = predictions["instances"].to(torch.device("cpu"))
            if predictions:
                (
                    boxes,
                    scores,
                    classes,
                ) = self.det2_process(predictions)
                for box, score, label in zip(boxes, scores, classes):
                    obj_class = self.class_names[label]
                    intbox = list(map(int, box))
                    det_results.append(
                        f"{obj_class} {float(score)} {intbox[0]} {intbox[1]} {intbox[2]-intbox[0]} {intbox[3]-intbox[1]}\n"
                    )
                    i_color = colors_instance[box_cnt]
                    cv2.rectangle(
                        vis_img,
                        intbox[0:2],
                        intbox[2:4],
                        color=i_color,
                        thickness=line_thickness,
                    )
                    cv2.putText(
                        vis_img,
                        obj_class,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 0, 255),
                        thickness=text_thickness,
                    )
                cv2.imwrite(vis_path, vis_img)
                # Write det lines fro evaluation
                text_file = open(
                    os.path.join(self.det_dir, f"{self.basename}_{frame_id}.txt"), "w"
                )
                text_file.writelines(det_results)
                text_file.close()

    def process(self):
        frame_id = 0
        prog_bar = ProgressBar(int(self.num_frames), fmt=ProgressBar.FULL)
        if self.param.config.image_path:
            for image in self.video:
                basename, _ = os.path.splitext(os.path.basename(image))
                basename = basename.split("_")[-1]
                frame_id = int(basename.replace("img", ""))  # - 1
                frame = cv2.imread(image)
                self.perform_task(frame, frame_id)
                # Update progress bar
                prog_bar.current += 1
                prog_bar()
        else:
            frame_id = 0
            while True:
                ret_val, frame = self.video.read()
                if ret_val:
                    self.perform_task(frame, frame_id)
                else:
                    break
                frame_id += 1
                # Update progress bar
                prog_bar.current += 1
                prog_bar()
        prog_bar.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="if batch ")
    args = parser.parse_args()
    param = ParameterManager("configs/enhance_config.yaml")
    param.predictor_name = param.config.predictor.upper()
    processor = VideoProcessor(param)
    processor.process()
