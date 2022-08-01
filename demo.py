import os
import time

import torch
import cv2
import numpy as np
from torchvision.ops import masks_to_boxes

from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import _create_text_labels
from models.config import get_cfg

from trackers.ocsort_tracker import OCSort, Detection
from trackers.tracking_utils.timer import Timer


# constants
WINDOW_NAME = "solov2 deep ocsort"
CONFIG_FILE = "configs/SOLOv2/R50_3x.yaml"
OPTS = ["MODEL.WEIGHTS", "SOLOv2_R50_3x.pth"]
CONFIDENCE_THRESHOLD = 0.5
VIDEO_INPUT = "/home/catchall/Videos/projects/ch01_20181212105607_6000.mp4"
VIS_FOLDER = "/home/catchall/Videos/projects/test_solov2_ocsort_aligncorners"
IS_SAVE_RESULT = False
MODEL_DEVICE = "cuda:0"  # cpu
CLASS_LIST = [0, 1, 2, 3, 5, 7]
# VIDEO_OUTPUT = "/home/allysakate/Videos/ffmpeg_capture_6-000_OUT.mp4"
VIDEO_OUTPUT = None
TRACK_THRESH = 0.3
IOU_THRESH = 0.3
USE_BYTE = False
CPU_DEVICE = torch.device("cpu")
_ASPECT_RATIO_THRESH = 1.6
_MIN_BOX_AREA_THRESH = 10


def setup_config():
    """Load config from file and command-line arguments"""
    configuration = get_cfg()
    configuration.merge_from_file(CONFIG_FILE)
    configuration.merge_from_list(OPTS)
    # num_gpu = torch.cuda.device_count()
    # configuration.MODEL.DEVICE = "cuda:0" if num_gpu > 0 else "cpu"
    configuration.MODEL.DEVICE = MODEL_DEVICE
    # Set score_threshold for builtin models
    configuration.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    configuration.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    configuration.MODEL.FCOS.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
    configuration.MODEL.MEInst.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
    # configuration.MODEL.SOLOV2.SCORE_THR = CONFIDENCE_THRESHOLD
    configuration.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        CONFIDENCE_THRESHOLD
    )
    configuration.freeze()
    return configuration


def post_process(frame, predictions):
    """
    Filter predictions

    Args:
        frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        class_list (List): list of class ids

    Returns:
        output (VisImage): image object with visualizations.
    """
    # num_instances = len(predictions)
    # TODO: Filter out non-vehicle classes
    classes = (
        predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    )

    scores = predictions.scores.numpy() if predictions.has("scores") else None
    class_mask = np.isin(classes, CLASS_LIST)
    filter_mask = class_mask
    # score_mask = scores >= _SCORE_THRESHOLD
    # filter_mask = np.logical_and(class_mask, score_mask)

    boxes = (
        predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    )

    boxes = boxes[filter_mask]
    classes = classes[filter_mask]
    scores = scores[filter_mask]

    if scores.any():

        periods = predictions.ID_period if predictions.has("ID_period") else None
        if periods:
            periods = periods[filter_mask]
        period_threshold = metadata.get("period_threshold", 0)
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

        if predictions.has("feat_masks"):
            feat_masks = predictions.feat_masks
            feat_masks = feat_masks[filter_mask]

        else:
            feat_masks = None

        feat_masks = None if feat_masks is None else feat_masks[visibilities]

        labels = _create_text_labels(
            classes, scores, metadata.get("thing_classes", None)
        )
        labels = (
            None
            if labels is None
            else [y[0] for y in filter(lambda x: x[1], zip(labels, visibilities))]
        )  # noqa

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
            labels = (
                [labels[idx] for idx in sorted_idxs] if labels is not None else None
            )
            pred_masks = (
                [pred_masks[idx] for idx in sorted_idxs]
                if pred_masks is not None
                else None
            )
            scores = scores[sorted_idxs]
            seg_masks = seg_masks[sorted_idxs, :, :]
            feat_masks = feat_masks[sorted_idxs, :, :]
            if seg_masks is not None:
                frame_boxes = masks_to_boxes(seg_masks)

        return feat_masks, frame_boxes, scores, labels


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    # radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        thickness=2,
    )

    for i, tlwh in enumerate(tlwhs):
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
    return im


def main(cfg, predictor, metadata):
    video = cv2.VideoCapture(VIDEO_INPUT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(VIDEO_INPUT)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = os.path.join(VIS_FOLDER, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if IS_SAVE_RESULT:
        save_path = os.path.join(save_folder, basename)
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    frame_id = 0
    tracker = OCSort(
        det_thresh=TRACK_THRESH, iou_threshold=IOU_THRESH, use_byte=USE_BYTE
    )
    timer = Timer()
    frame_id = 0
    results = []

    while True:
        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )
        ret_val, frame = video.read()
        if ret_val:
            detections = []
            raw_img = frame
            timer.tic()
            predictions = predictor(frame)
            predictions = predictions["instances"].to(CPU_DEVICE)
            feat_masks, boxes, scores, labels = post_process(frame, predictions)
            if boxes.any():
                # TODO: tracking
                for feat, box, score, label in zip(feat_masks, boxes, scores, labels):
                    det = Detection(np.array(box), score, label, np.array(feat))
                    detections.append(det)
                detections = np.array(detections)
                targets = tracker.update(
                    detections, boxes, scores, [height, width], (1080, 1920)
                )
                track_tlwhs = []
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
                    tid = tlbrs[4]
                    vertical = box_width / box_height > _ASPECT_RATIO_THRESH
                    if box_width * box_height > _MIN_BOX_AREA_THRESH and not vertical:
                        track_tlwhs.append(tlwh)
                        track_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                timer.toc()
                res_image = plot_tracking(
                    raw_img,
                    track_tlwhs,
                    track_ids,
                    frame_id=frame_id + 1,
                    fps=1.0 / timer.average_time,
                )
                res_path = os.path.join(VIS_FOLDER, f"{frame_id}.jpg")
                cv2.imwrite(res_path, res_image)
            else:
                timer.toc()
                res_image = raw_img
            if IS_SAVE_RESULT:
                vid_writer.write(res_image)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if IS_SAVE_RESULT:
        res_file = os.path.join(VIS_FOLDER, f"{timestamp}.txt")
        with open(res_file, "w") as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


if __name__ == "__main__":
    logger = setup_logger()

    if not os.path.exists(VIS_FOLDER):
        os.makedirs(VIS_FOLDER)

    cfg = setup_config()

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    main(cfg, predictor, metadata)
