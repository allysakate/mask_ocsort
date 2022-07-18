import os
import multiprocessing as mp

# import torch
import cv2
import tqdm

from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from models.config import get_cfg

from trackers.ocsort_tracker.ocsort import OCSort


# constants
WINDOW_NAME = "solov2 deep ocsort"
CONFIG_FILE = "configs/SOLOv2/R50_3x.yaml"
OPTS = ["MODEL.WEIGHTS", "SOLOv2_R50_3x.pth"]
CONFIDENCE_THRESHOLD = 0.5
VIDEO_INPUT = "/home/allysakate/Videos/ffmpeg_capture_6-000.mp4"
TEST_FOLDER = "media/test_solov2_classes"
MODEL_DEVICE = "cuda:0"  # cpu
CLASS_LIST = [0, 1, 2, 3, 5, 7]
# VIDEO_OUTPUT = "/home/allysakate/Videos/ffmpeg_capture_6-000_OUT.mp4"
VIDEO_OUTPUT = None
TRACK_THRESH = 0.3
IOU_THRESH = 0.3
USE_BYTE = False


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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)

    cfg = setup_config()

    demo = VisualizationDemo(cfg)

    if VIDEO_INPUT:
        video = cv2.VideoCapture(VIDEO_INPUT)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(VIDEO_INPUT)

        if VIDEO_OUTPUT:
            if os.path.isdir(VIDEO_OUTPUT):
                OUTPUT_FNAME = os.path.join(VIDEO_OUTPUT, basename)
                OUTPUT_FNAME = os.path.splitext(OUTPUT_FNAME)[0] + ".mkv"
            else:
                OUTPUT_FNAME = VIDEO_OUTPUT
            assert not os.path.isfile(OUTPUT_FNAME), OUTPUT_FNAME
            output_file = cv2.VideoWriter(
                filename=OUTPUT_FNAME,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(VIDEO_INPUT)
        frame_id = 0
        tracker = OCSort(
            det_thresh=TRACK_THRESH, iou_threshold=IOU_THRESH, use_byte=USE_BYTE
        )

        for vis_frame in tqdm.tqdm(
            demo.run_on_video(video, tracker, CLASS_LIST), total=num_frames
        ):
            if VIDEO_OUTPUT:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
                img_name = f"{TEST_FOLDER}/{frame_id}.jpg"
                cv2.imwrite(img_name, vis_frame)
            frame_id += 1
        video.release()
        if VIDEO_OUTPUT:
            output_file.release()
        else:
            cv2.destroyAllWindows()
