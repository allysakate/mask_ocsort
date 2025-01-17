import os
import csv
import math
import yaml
import json
import tqdm
import argparse
from typing import List
from functools import reduce
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from skimage import measure
from glob import glob
from torchvision.ops import masks_to_boxes
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
from detectron2.config import get_cfg
from models.utils.datasets import letterbox
from models.utils.general import non_max_suppression_mask_conf
from trackers.ocsort_tracker import OCSort
from utils import (
    initiate_logging,
    get_config,
    create_output,
    get_IOP,
    close_logging,
    calculate_iou,
    colors_instance,
)


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
        _tracker (object):      tracker object [OCSort]
        predictor (object):     model predict
        metadata (object):      testing metadata
    """

    def __init__(self, yml_path: str):
        self.config = get_config(yml_path)
        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predictor_name = None
        self.tracker_name = None
        self.text_filename = None
        self.folder_name = None
        self.use_maskocsort = None
        self.with_feature = None
        self.frame_use = self.config.frame_use
        self.iou_weight = self.config.iou_weight
        self.vd_weight = self.config.vd_weight
        self.min_hits = self.config.min_hits
        self.max_age = self.config.max_age

    def set_useocsort_feat(self):
        if self.use_maskocsort:
            self.tracker_name = "MASKOCSORT"
            self.with_feature = True
        else:
            self.tracker_name = "OCSORT"
            self.with_feature = False

    def set_text_filename(self):
        self.text_filename = f"{self.tracker_name}_{self.predictor_name}_{self.frame_use}FPS"

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
        self.is_yolo = is_yolo
        if self.param.config.image_path:
            self.video = sorted(
                glob(os.path.join(self.param.config.image_path, "*.jpg"))
            )
            self.fps = self.param.config.image_fps
            first_frame_path = self.video[0]
            self.basename = os.path.dirname(first_frame_path).split("/")[-1]
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
        if not self.is_yolo:
            self.predictor = self.set_det2model()
        else:
            self.predictor = self.set_yolomodel()
        self.output_dir = create_output(param.config.output_dir, param.text_filename)
        self.image_dir = create_output(self.output_dir, "images")
        self.det_res_dir = create_output(self.output_dir, "det")
        log_dir = create_output(self.output_dir, "logs")
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

    def yolo_set_frame(self, image):
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()
        return image

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
            if self.param.tracker_name in ["MASKOCSORT"]:
                hooker = LayerHook()
                if self.param.predictor_name == "MASKRCNN":
                    node = "roi_heads.mask_pooler"  # (19, 256, 14, 14)
                hooker.register(self.predictor.model, [node])
        return hooker

    def det2_process(self, predictions, mask_pool=None):
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
        feat_masks, seg_masks, frame_boxes, scores, classes = [], [], [], [], []
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
        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = f"Trk: {obj_id}, Cls: {classes[i]}"
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
        return im

    def filter_box(self, box, frame=None):
        """Filter Bounding Box
        Args:
            box (List[int]): bounding box
        Returns:
            is_included, frame
        """
        is_included = True
        if self.param.config.roi:
            for roi in self.param.config.roi:
                roi_iop = get_IOP(roi, box)
                condition = roi_iop < self.param.config.iop_threshold
                if not condition:
                    is_included = False
                    break
            if condition:
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
        cv2.imshow("result", result)
        cv2.waitKey()
        return result

    def visualize(self, raw_img, bboxes):
        def point_equation(point1, point2):
            x1, y1 = list(map(int, point1))
            x2, y2 = list(map(int, point2))
            m = (y2 - y1) / (x2 - x1 + 1)
            x = x1 + 30
            return abs(m * (x - x1) - y1)

        text_scale = 1.2
        text_thickness = 2
        line_thickness = 3
        im = raw_img.copy()
        if self.param.config.roi:
            for roi in self.param.config.roi:
                cv2.rectangle(
                    im,
                    roi[0:2],
                    roi[2:4],
                    color=(255, 255, 255),
                    thickness=line_thickness,
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
                im,
                (intbox[0], intbox[1] - 20),
                (intbox[0] + 180, intbox[1]),
                color=(0, 0, 0),
                thickness=-1,
            )
            area = (box[2] - box[0]) * (box[3] - box[1])
            scale = math.sqrt(area)
            i_text = f"{i+1}, Scale: {scale:.2f}"
            cv2.putText(
                im,
                i_text,
                (intbox[0], intbox[1] - 5),
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
                    cx1, cy1 = (intbox[0] + intbox[2]) / 2.0, (
                        intbox[1] + intbox[3]
                    ) / 2.0
                    cx2, cy2 = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                    distance = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
                    try:
                        new_cy = point_equation((cx1, cy1), (cx2, cy2))
                        d_text = f"{b_id+1}: {distance:.2f}"
                        dist_list.append([b_id, d_text, [cx1, cy1, cx1 + 30, new_cy]])
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
                    j_intbox = list(map(int, box))
                    j_iop = calculate_iou(j_intbox, intbox)
                    if j_iop > 0.01:
                        IOU_LIST.append(j_iop)
                        j_text = f"{j+1}: {j_iop:.2f}"
                        text_list.append(j_text)
            height = int(len(text_list) * 20)
            cv2.rectangle(
                im,
                intbox[0:2],
                (intbox[0] + 100, intbox[1] + height),
                color=(255, 255, 255),
                thickness=-1,
            )
            for idx, j_text in enumerate(text_list):
                j_color = colors_instance[idx]
                cv2.putText(
                    im,
                    j_text,
                    (intbox[0], intbox[1] + 14 * (idx + 1)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    j_color,
                    thickness=text_thickness,
                )
        return im

    def ocsort(
        self,
        frame_id,
        feat_masks,
        boxes,
        scores,
        classes,
        trk_results,
        trk_dict,
        det_results,
        class_names,
    ):
        detections = []
        for feat, box, score, label in zip(
            feat_masks, boxes, scores, classes
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
                if self.param.with_feature:
                    if self.param.predictor_name == "MASKRCNN":
                        max_pool = nn.MaxPool2d(3, stride=3)
                        feat = max_pool(feat).numpy()
                    feat = np.array(feat).flatten()
                else:
                    feat = np.array(feat).flatten()
                if is_included:
                    det = np.asarray(np.insert(det, 6, feat))
                    detections.append(det)
        if detections:
            targets = self._tracker.update(np.asarray(detections),self.param.with_feature)
        else:
            targets = []
        track_tlwhs = []
        track_scores = []
        track_classes = []
        track_ids = []
        for tlbrs in targets:
            box_width = tlbrs[2] - tlbrs[0]
            box_height = tlbrs[3] - tlbrs[1]
            tlwh = [
                tlbrs[0],
                tlbrs[1],
                tlbrs[2] - tlbrs[0],
                tlbrs[3] - tlbrs[1],
            ]
            tscore = tlbrs[4]
            tclass = int(tlbrs[5])
            tid = tlbrs[6]
            if (
                box_width * box_height
                > self.param.config._min_box_area_thresh
            ):
                track_tlwhs.append(tlwh)
                track_ids.append(tid)
                track_scores.append(tscore)
                if tclass in self.param.config.class_list:
                    track_classes.append(class_names[tclass])
                else:
                    track_classes.append(None)
                trk_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{round(tscore, 2)},-1,-1,-1\n"
                )
                if tid not in trk_dict:
                    trk_dict[tid] = []
                trk_dict[tid].append(
                    [
                        frame_id,
                        int(tlbrs[0]),
                        int(tlbrs[1]),
                        int(tlbrs[2]),
                        int(tlbrs[3]),
                    ]
                )

        return (
            track_tlwhs,
            track_ids,
            track_classes,
            trk_results,
            trk_dict,
            det_results,
        )

    def perform_task(
        self,
        frame,
        frame_id,
        det_results,
        trk_results,
        trk_dict,
        vid_writer,
    ):
        raw_img = frame
        res_image = frame.copy()
        if int(frame_id % self.frame_skip) == 0 and frame_id != 0:
            hooker = self.set_hooker()
            # try:
            if not self.is_yolo:
                predictions = self.predictor(frame)
                predictions = predictions["instances"].to(torch.device("cpu"))

                if hooker:
                    mask_pool = hooker.outputs[-1].cpu()
                    hooker.remove()
                else:
                    mask_pool = None
                if predictions:
                    (
                        feat_masks,
                        _,
                        boxes,
                        scores,
                        classes,
                    ) = self.det2_process(
                        predictions,
                        mask_pool,
                    )
            else:
                frame = self.yolo_set_frame(frame)
                _, _, f_height, f_width = frame.shape
                predictions = self.predictor(frame)
                (feat_masks, _, boxes, scores, classes,) = self.yolo_process(
                    predictions, (f_height, f_width), (self.height, self.width)
                )

            if len(boxes):
                (
                    track_tlwhs,
                    track_ids,
                    track_classes,
                    trk_results,
                    trk_dict,
                    det_results,
                ) = self.ocsort(
                    frame_id,
                    feat_masks,
                    boxes,
                    scores,
                    classes,
                    trk_results,
                    trk_dict,
                    det_results,
                    self.class_names,
                )
                res_image = self.plot_tracking(
                    raw_img,
                    track_tlwhs,
                    track_ids,
                    track_classes,
                    frame_id=frame_id + 1,
                )
            else:
                res_image = raw_img
            res_path = os.path.join(self.image_dir, f"{frame_id}.jpg")
            for box in boxes:
                _, res_image = self.filter_box(box, res_image)
            cv2.imwrite(res_path, res_image)
            # Write det lines from evaluation
            new_gt_file = open(
                os.path.join(self.det_res_dir, f"{frame_id:06d}.txt"), "w"
            )
            new_gt_file.writelines(det_results)
            new_gt_file.close()  # to change file access modes
            if self.param.config.is_save_result:
                vid_writer.write(res_image)
        return det_results, trk_results, trk_dict

    def process(self):
        if self.param.config.is_save_result:
            save_path = os.path.join(self.output_dir, f"{self.basename}.mp4")
            self.logger.info(f"video save_path is {save_path}")
            vid_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (int(self.width), int(self.height)),
            )

        self._tracker = OCSort(
            iou_weight=self.param.iou_weight,
            inertia=self.param.vd_weight,
            max_age=self.param.max_age,
            min_hits=self.param.min_hits,
        )

        frame_id = 0
        det_results = []
        trk_results = []
        trk_dict = {}
        if self.param.config.image_path:
            for image in tqdm.tqdm(self.video):
                basename, _ = os.path.splitext(os.path.basename(image))
                frame_id = int(basename.replace("img", ""))  # - 1
                frame = cv2.imread(image)
                det_results, trk_results, trk_dict = self.perform_task(
                    frame,
                    frame_id,
                    det_results,
                    trk_results,
                    trk_dict,
                    vid_writer,
                )
        else:
            frame_id = 0
            total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            prog_bar = tqdm.tqdm(total=int(total_frames))
            while True:
                ret_val, frame = self.video.read()
                if ret_val:
                    det_results, trk_results, trk_dict = self.perform_task(
                        frame,
                        frame_id,
                        det_results,
                        trk_results,
                        trk_dict,
                        vid_writer,
                    )
                else:
                    break
                frame_id += 1
                # Update progress bar
                prog_bar.update(1)
            self.video.release()

        if self.param.config.is_save_result:
            res_file = os.path.join(self.output_dir, f"{self.param.text_filename}.txt")
            with open(res_file, "w") as f:
                f.writelines(trk_results)
            self.logger.info(f"save trk_results to {res_file}")


def start_process(param, is_yolo=False):
    param.set_useocsort_feat()
    param.set_text_filename()
    processor = VideoProcessor(param, is_yolo)
    processor.process()
    close_logging(processor.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    is_yolo = False
    param = ParameterManager("configs/demo_config.yaml")
    param.predictor_name = param.config.predictor.upper()
    if param.predictor_name == "YOLOV7":
        is_yolo = True
    start_process(param, is_yolo)
