import os
import yaml
import logging
from datetime import datetime
import numpy as np


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


def initiate_logging(log_dir: str, script: str):
    """Creates log file for script
    Arguments:
        log_name (str):  name of log file
    Return:
        logging (object):  logging object
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now().strftime("%m%d%Y%H%M%S")  # current date and time
    log_name = os.path.join(log_dir, f"{script}_{now}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # # our first handler is a console handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.NOTSET)
    # console_handler_format = "%(asctime)s | %(levelname)s: %(message)s"
    # console_handler.setFormatter(logging.Formatter(console_handler_format))
    # logger.addHandler(console_handler)

    # the second handler is a file handler
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.INFO)
    file_handler_format = "%(asctime)s | %(levelname)s | %(lineno)d: %(message)s"
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    # start logging and show messages
    logger.info("---------------START---------------")
    return logger


def close_logging(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


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
    if not os.path.exists(output_dir):
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


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Arguments:
    box1: Tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box.
    box2: Tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box.

    Returns:
    iou: Intersection over Union (IoU) value.
    """

    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x2_max = min(box1[2], box2[2])
    y2_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2_max - x1_min + 1) * max(0, y2_max - y1_min + 1)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the Union area by subtracting the intersection area
    # and adding the areas of both bounding boxes
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou


def calculate_range_output(height, width, overhead_hmatrix, opt=False, verbose=False):
    """Adjust the perspective matrix to correspond to the transformed image size"""
    range_u = np.array([np.inf, -np.inf])
    range_v = np.array([np.inf, -np.inf])

    i = 0
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_upperpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_lowerpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = 0
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])

    range_u = np.array(range_u, dtype=np.int32)
    range_v = np.array(range_v, dtype=np.int32)

    if out_upperpixel > out_lowerpixel and opt:

        # range_v needs to be updated
        max_height = range_v[1]
        upper_range = out_lowerpixel
        best_lower = (
            upper_range  # since out_lowerpixel was lower value than out_upperpixel
        )
        #                           i.e. above in image than out_lowerpixel
        x_best_lower = np.inf
        x_best_upper = -np.inf

        for steps_h in range(2, height):
            temp = np.dot(
                overhead_hmatrix,
                np.vstack(
                    (
                        np.arange(0, width),
                        np.ones((1, width)) * (height - steps_h),
                        np.ones((1, width)),
                    )
                ),
            )
            temp = temp / temp[2, :]

            lower_range = temp.min(axis=1)[1]
            x_lower_range = temp.min(axis=1)[0]
            x_upper_range = temp.max(axis=1)[0]
            if x_lower_range < x_best_lower:
                x_best_lower = x_lower_range
            if x_upper_range > x_best_upper:
                x_best_upper = x_upper_range

            if (
                upper_range - lower_range
            ) > max_height:  # enforcing max_height of destination image
                lower_range = upper_range - max_height
                break
            if lower_range > upper_range:
                lower_range = best_lower
                break
            if lower_range < best_lower:
                best_lower = lower_range
            if verbose:
                print(steps_h, lower_range, x_best_lower, x_best_upper)
        range_v = np.array([lower_range, upper_range], dtype=np.int32)

        # for testing
        range_u = np.array([x_best_lower, x_best_upper], dtype=np.int32)

    return range_u, range_v


def get_scaled_matrix(
    homo_matrix, target_shape, estimated_xrange, estimated_yrange, strict=False
):
    current_height = estimated_yrange[1] - estimated_yrange[0]
    current_width = estimated_xrange[1] - estimated_xrange[0]
    x_scale, y_scale = target_shape[0] / current_width, target_shape[1] / current_height
    print("x_scale, y_scale ", x_scale, y_scale)
    if strict:
        scale = min(x_scale, y_scale)
        scaling_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    else:
        scaling_matrix = np.array([[x_scale, 0, 0], [0, y_scale, 0], [0, 0, 1]])
    scaled_homo = np.dot(scaling_matrix, homo_matrix)
    return scaled_homo
