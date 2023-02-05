import os
import yaml
import logging
import shutil
from datetime import datetime


colors_instance = [
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (0, 0, 0),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    (50, 0, 0),
    (100, 0, 0),
    (50, 128, 0),
    (100, 128, 0),
    (50, 0, 128),
    (100, 0, 128),
    (50, 128, 128),
    (100, 128, 128),
    (0, 50, 0),
    (128, 50, 0),
    (0, 100, 0),
    (128, 100, 0),
    (0, 50, 128),
    (85, 0, 0),
    (170, 0, 0),
    (85, 128, 0),
    (170, 128, 0),
    (85, 0, 128),
    (170, 0, 128),
    (85, 128, 128),
    (170, 128, 128),
    (0, 85, 0),
    (128, 85, 0),
    (0, 170, 0),
    (128, 170, 0),
    (0, 85, 128),
    (36, 0, 0),
    (70, 0, 0),
    (36, 128, 0),
    (70, 128, 0),
    (36, 0, 128),
    (70, 0, 128),
    (36, 128, 128),
    (70, 128, 128),
    (0, 36, 0),
    (128, 36, 0),
    (0, 70, 0),
    (128, 70, 0),
    (0, 36, 128),
    (28, 0, 0),
    (151, 0, 0),
    (28, 128, 0),
    (151, 128, 0),
    (28, 0, 128),
    (151, 0, 128),
    (28, 128, 128),
    (151, 128, 128),
    (0, 28, 0),
    (128, 28, 0),
    (0, 151, 0),
    (128, 151, 0),
    (0, 28, 128),
    (168, 0, 0),
    (222, 0, 0),
    (168, 128, 0),
    (222, 128, 0),
    (168, 0, 128),
    (222, 0, 128),
    (168, 128, 128),
    (222, 128, 128),
    (0, 222, 0),
]


class Config(object):
    def __init__(self, config_data):
        self.__dict__.update(config_data)


def get_config(yml_file: str):
    """Creates log file for script
    Arguments:
        config_path (str): config file path
        script (str):  image path
    Return:
        config (Dict):  Config object
    """
    with open(yml_file, "r", encoding="utf8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    model_cfg = Config(cfg)
    return model_cfg


def initiate_logging(script: str):
    """Creates log file for script
    Arguments:
        log_name (str):  name of log file
    Return:
        logging (object):  logging object
    """
    log_dir = "logs/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now().strftime("%m%d%Y")  # current date and time
    log_name = os.path.join(log_dir, f"{script}_{now}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # our first handler is a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler_format = "%(asctime)s | %(levelname)s: %(message)s"
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    # the second handler is a file handler
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.INFO)
    file_handler_format = "%(asctime)s | %(levelname)s | %(lineno)d: %(message)s"
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    # start logging and show messages
    logger.info("---------------START---------------")
    return logger


def create_output(dir_path: str, base_name: str = None):
    """Removes existing and creates output directory
    Args:
        dir_path (str) : directory of output
        base_name (str) : folder name

    Returns:
        output_dir (str) : output directory
    """
    if base_name:
        output_dir = os.path.join(dir_path, base_name)
    else:
        output_dir = dir_path
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir


def get_IOP(roi, bbox):
    """
    Get intersection over prediction bbox and rois
    Args:
        roi (List[float]): x- & y- coordinates
                    [xmin, ymin, xmax, ymax]
        bbox (List[int]): x- & y- coordinates
            [xmin, ymin, xmax, ymax]

    Returns:
        Tuple(float): x- & y- coordinates of centroids
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(roi[0], bbox[0])
    yA = max(roi[1], bbox[1])
    xB = min(roi[2], bbox[2])
    yB = min(roi[3], bbox[3])

    # compute the area of the prediction rectangle
    bboxArea = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return interArea / bboxArea
