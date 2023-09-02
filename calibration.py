import os
import math
import json
import pickle
import warnings
import cv2
import torch
import numpy as np
from torchvision.ops import masks_to_boxes
from diamond_space import DiamondSpace
from sklearn.linear_model import RANSACRegressor
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from utils import (
    get_config,
    colors_instance,
    create_output,
    calculate_range_output,
    get_scaled_matrix,
)
from models.config import get_cfg

warnings.filterwarnings("ignore")


class VPDetection(object):
    """
    VP Detection Object

    Args:
        length_thresh: Line segment detector threshold (default=30)
        principal_point: Principal point of the image (in pixels)
        focal_length: Focal length of the camera (in pixels)
        seed: Seed for reproducibility due to RANSAC
    """

    def __init__(self, config, track_points, segm_points):
        self._track_points = track_points
        self._segm_points = segm_points
        self._length_thresh = config.length_thresh
        self._camera_height = config.camera_height
        self._min_samples = config.min_samples
        self._seed = config.seed
        self._principal_point = None
        self._focal_length = None
        self._angle_thresh = np.pi / 30  # For displaying debug image
        self._vps_world = []  # For storing the VPs in 3D space
        self._vps_2d = []  # For storing the VPs in 2D space
        self._img_vps1 = None
        self._img_vps2 = None
        self._img_height = None
        self._img_width = None
        self._out_dir = create_output(config.out_dir)

    @property
    def length_thresh(self):
        """
        Length threshold for line segment detector

        Returns:
            The minimum length required for a line
        """
        return self._length_thresh

    @length_thresh.setter
    def length_thresh(self, value):
        """
        Length threshold for line segment detector

        Args:
            value: The minimum length required for a line

        Raises:
            ValueError: If the threshold is 0 or negative
        """
        if value <= 0:
            raise ValueError("Invalid threshold: {}".format(value))

        self._length_thresh = value

    @property
    def principal_point(self):
        """
        Principal point for VP Detection algorithm

        Returns:
            The minimum length required for a line
        """
        return self._principal_point

    @principal_point.setter
    def principal_point(self, value):
        """
        Principal point for VP Detection algorithm

        Args:
            value: A list or tuple of two elements denoting the x and y
           coordinates

        Raises:
            ValueError: If the input is not a list or tuple and there aren't
            two elements
        """
        try:
            assert isinstance(value, (list, tuple)) and not isinstance(value, str)
            assert len(value) == 2
        except AssertionError:
            raise ValueError("Invalid principal point: {}".format(value))

        self._length_thresh = value

    @property
    def focal_length(self):
        """
        Focal length for VP detection algorithm

        Returns:
            The focal length in pixels
        """
        return self._focal_length

    @property
    def vps_world(self):
        """
        Vanishing points of the image in 3D space.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate
        """
        return self._vps_world

    @property
    def vps_2d(self):
        """
        Vanishing points of the image in 2D image coordinates.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate
        """
        return self._vps_2d

    def get_line_segments(
        self,
        data,
        detected_segments,
        vp1_slopes,
        max_trials=100,
        residual_thresh=0.5,
        diff_angle=80,
    ):
        """Detect line segments from points
        Arguments:
            min_samples - Minimum number of points to fit a line
            max_trials - Maximum number of iterations
            residual_thresh - Threshold for inlier definition
        """

        def angle_between_lines(slope1, slope2):
            angle_radians = abs(math.atan((slope2 - slope1) / (1 + slope1 * slope2)))
            angle_degrees = math.degrees(angle_radians)
            return angle_degrees

        while len(data) >= self._min_samples:
            ransac = RANSACRegressor(
                random_state=self._seed,
                min_samples=self._min_samples,
                max_trials=max_trials,
                residual_threshold=residual_thresh,
            )
            ransac.fit(data[:, 0].reshape(-1, 1), data[:, 1])
            inlier_mask = ransac.inlier_mask_

            if np.sum(inlier_mask) < self._min_samples:
                break  # No valid segment found

            segment = data[inlier_mask]
            for vp1_k in vp1_slopes:
                ransac_m = ransac.estimator_.coef_[0]
                angle_slope = angle_between_lines(vp1_k, ransac_m)
                if angle_slope > diff_angle:
                    detected_segments.append(segment)

            inverted_inlier_mask = np.logical_not(inlier_mask)
            data = data[inverted_inlier_mask]  # Remove detected points from data
        return detected_segments

    def create_edgelines(self, vp1_slopes):
        # Read data from JSON file
        segm_dict = {}
        with open(self._segm_points, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            for idx, sg_masks in enumerate(data):
                segm_dict[idx] = {}
                segments = []
                for mask_pts in sg_masks:
                    mask_coords = np.array(mask_pts).reshape(-1, 2)
                    segments = self.get_line_segments(
                        mask_coords, segments, vp1_slopes=vp1_slopes
                    )
                for seg_idx, segment in enumerate(segments):
                    segm_dict[idx][seg_idx] = segment
        return list(segm_dict.values())

    def create_tracklines(self, height, width, conf_thresh=0.3, margin=20):
        """
        Create lines from track points
        Sample Data: 160,1.0,1340.00,190.00,273.00,195.00,1.0,-1,-1,-1
        Track data: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        track_dict = {}
        with open(self._track_points, "r", encoding="utf-8") as file:
            for line in file:
                trk_data = line.split(",")
                track_id = int(float(trk_data[1]))
                if float(trk_data[6]) > conf_thresh:
                    tlwh = np.array(trk_data[2:6], dtype=np.float32)
                    tlbr = list(tlwh)
                    tlbr[2] = tlwh[0] + tlwh[2]
                    tlbr[3] = tlwh[1] + tlwh[3]
                    idx_dict = {0: [0, 2], 1: [0, 1], 2: [1, 3], 3: [2, 3]}
                    points = [
                        [tlbr[0], tlbr[1]],  # top left
                        [tlbr[2], tlbr[1]],  # top right
                        [tlbr[0], tlbr[3]],  # bottom left
                        [tlbr[2], tlbr[3]],  # bottom right
                    ]
                    idx_to_remove = [
                        index
                        for index, element in enumerate(tlbr[:2])
                        if element < margin
                    ]
                    if tlbr[2] > width - margin:
                        idx_to_remove.append(2)
                    if tlbr[3] > height - margin:
                        idx_to_remove.append(3)

                    for idx in idx_to_remove:
                        for key, value in idx_dict.items():
                            if key == idx:
                                for v_idx in value:
                                    points[v_idx] = []
                    if track_id not in track_dict:
                        track_dict[track_id] = {
                            0: [],
                            1: [],
                            2: [],
                            3: [],
                        }
                    for idx, pt in enumerate(points):
                        if len(pt):
                            track_dict[track_id][idx].append(pt)
        # check if object is moving
        ids_to_remove = []
        for track_id, track_pts in track_dict.items():
            tracks = track_pts[0]
            dx = tracks[-1][0] - tracks[0][0]
            dy = tracks[-1][1] - tracks[0][1]
            length = np.sqrt(dx * dx + dy * dy)
            if length < self._length_thresh:
                ids_to_remove.append(track_id)
        for track_id in ids_to_remove:
            del track_dict[track_id]
        return track_dict

    def cvt_diamond_space(self, line_pts, image):
        """
        :param tracks:
        :return:lines: [ax+by+c=0]
        [a, b, c]
        """
        lines, slopes = [], []
        for point in line_pts:
            for p_idx, coord in point.items():
                coord = np.array(coord)
                # Draw the points on the image
                for idx, point in enumerate(coord):
                    # Draw a small point at the specified location
                    cv2.circle(
                        image,
                        (int(point[0]), int(point[1])),
                        5,
                        colors_instance[p_idx],
                        -1,
                    )
                    if idx:
                        cv2.line(
                            image,
                            (int(coord[idx - 1][0]), int(coord[idx - 1][1])),
                            (int(coord[idx][0]), int(coord[idx][1])),
                            colors_instance[p_idx],
                            2,
                        )
                k, b = np.polyfit(coord[:, 0], coord[:, 1], deg=1)
                lines.append([k, -1, b])
                slopes.append(k)
        return np.array(lines), slopes, image

    def get_focal(self):
        """Compute focal length"""
        vp1 = self._vps_2d[0]
        vp2 = self._vps_2d[1]
        prp = self._principal_point
        return math.sqrt(abs(np.dot(vp1[0:2] - prp[0:2], vp2[0:2] - prp[0:2])))

    def get_vps(self, height, width, space_size=128, ds_scale=256):
        """Detects vanishing points using DiamondSpace Accumulator
        Args:
            size - size of the accumulator subspace
            ds_scale - scaling inside diamond space:

        How to set ds_scale(d) parameter:
        *   area corresponding to square<-d,d> x <-d,d> in Cartesion coordinates occupied
            left/right half of the minus/plus part of the Diamond space
        *   for better precision close to the CC origin, set d smaller
            (around half of image width/height, d < rad)
        *   for better precision of the accumulator far outside (close to the infinity),
            set d bigger ( d >> rad)
        """

        # Initialize diamond space accumulators and accumulate lines
        dmnd_space = DiamondSpace(d=ds_scale, size=space_size)
        # 1st vanishing point
        track_dict = self.create_tracklines(height, width)
        trk_pts = list(track_dict.values())
        vp1_lines, vp1_slopes, vp1_image = self.cvt_diamond_space(
            trk_pts, self._img_vps1
        )
        dmnd_space.insert(vp1_lines)
        vps1, p_values1, _ = dmnd_space.find_peaks(min_dist=2, prominence=2, t=0.9)
        pos1 = np.argmax(p_values1)
        self._vps_2d.append(vps1[pos1][:2])
        cv2.circle(
            vp1_image, (int(vps1[pos1][0]), int(vps1[pos1][1])), 10, (0, 0, 255), -1
        )
        print(f"1st Vanishing Point: {self._vps_2d[0]}")
        cv2.imwrite(os.path.join(self._out_dir, "vp1.jpg"), vp1_image)

        # 2nd vanishing point
        segm_pts = self.create_edgelines(vp1_slopes)
        vp2_lines, _, vp2_image = self.cvt_diamond_space(segm_pts, self._img_vps2)
        dmnd_space.insert(vp2_lines)
        vps2, p_values2, _ = dmnd_space.find_peaks(min_dist=2, prominence=2, t=0.9)
        pos2 = np.argmax(p_values2)
        self._vps_2d.append(vps2[pos2][:2])
        print(f"2nd Vanishing Point: {self._vps_2d[1]}")
        cv2.circle(
            vp2_image, (int(vps2[pos2][0]), int(vps2[pos2][1])), 10, (0, 255, 0), -1
        )
        cv2.imwrite(os.path.join(self._out_dir, "vp2.jpg"), vp2_image)

        return self._vps_2d

    def calibration(self, img):
        """
        Find the vanishing points given the input image

        Args:
            img: Either the path to the image or the image read in with
         `cv2.imread`

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate. Additionally, the VPs are ordered such that
            the right most VP is the first row, the left most VP is the second
            row and the vertical VP is the last row
        """

        # Detect the lines in the image
        if isinstance(img, str):
            img = cv2.imread(img, -1)

        self._img_vps1 = img.copy()  # Keep a copy for later
        self._img_vps2 = img.copy()  # Keep a copy for later

        # Reset principal point if we haven't set it yet
        if self._principal_point is None:
            rows, cols = img.shape[:2]
            self._img_height, self._img_width = rows, cols
            self._principal_point = np.array([cols / 2.0, rows / 2.0])

        # Detect vps
        _ = self.get_vps(rows, cols, ds_scale=min(rows, cols), space_size=1024)
        self._focal_length = self.get_focal()
        (
            road_plane,
            intrinsic_matrix,
            rotation_matrix,
            translation_matrix,
        ) = self.compute_calibration()
        calibration = dict(
            vp1=self.vps_2d[0],
            vp2=self.vps_2d[1],
            vp3=self.vps_2d[2],
            principal_point=self.principal_point,
            road_plane=road_plane,
            focal=self.focal_length,
            intrinsic=intrinsic_matrix,
            rotation=rotation_matrix,
            translation=translation_matrix,
        )
        pkl_path = os.path.join(self._out_dir, "calibration.pkl")
        pikd = open(pkl_path, "wb")
        pickle.dump(calibration, pikd)
        pikd.close()

    def compute_calibration(self):
        """
        Compute camera calibration from two van points and principal point.

        Returns:
            road_plane: The plane equation of the road plane (ax + by + cz + d).
            intrinsic_matrix: K
            rotation_matrix: R
        """

        vp1_world = np.concatenate((self._vps_2d[0], [self.focal_length]))
        vp2_world = np.concatenate((self._vps_2d[1], [self.focal_length]))
        pp_world = np.concatenate((self._principal_point, np.array([0.0])))
        vp1_pp = vp1_world - pp_world
        vp2_pp = vp2_world - pp_world
        print(vp1_pp, vp2_pp)
        vp3_world = np.cross(vp1_pp, vp2_pp)
        vp3 = vp3_world[0:2] / vp3_world[2] * self.focal_length + pp_world[0:2]
        self._vps_2d.append(vp3)
        print(f"3rd Vanishing Point: {vp3}")

        vp3_direction = np.concatenate((vp3, [self.focal_length])) - pp_world
        road_plane = np.concatenate(
            (vp3_direction / np.linalg.norm(vp3_direction), [self._camera_height])
        )
        # Estimate focal lengths from vanishing points and camera height
        # fx = self._camera_height * np.linalg.norm(self._vps_2d[0] - self._vps_2d[1])
        # fy = self._camera_height * np.linalg.norm(self._vps_2d[1] - self._vps_2d[2])
        # intrinsic_matrix = np.array([[fx, 0, pp_world[0]],
        #                             [0, fy, pp_world[1],],
        #                             [0, 0, 1]])
        intrinsic_matrix = np.array(
            [
                [self.focal_length, 0, pp_world[0]],
                [0, self.focal_length, pp_world[1]],
                [0, 0, 1],
            ]
        )

        rotation_matrix = np.stack(
            [
                vp2_pp / np.linalg.norm(vp2_pp),
                vp1_pp / np.linalg.norm(vp1_pp),
                vp3_world / np.linalg.norm(vp3_world),
            ],
            axis=1,
        )

        center_height_matrix = np.array(
            [[pp_world[0], pp_world[1], self._camera_height]]
        ).reshape(3, 1)
        translation_matrix = -rotation_matrix @ center_height_matrix

        return (road_plane, intrinsic_matrix, rotation_matrix, translation_matrix)


class Detection:
    def __init__(self, config):
        self.config = config
        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.period_threshold = 0
        self.class_names = None
        self.class_ids = None
        self.model = self.set_det2model()

    def set_det2model(self):
        """Load config from file and command-line arguments
        Returns:
            configuration
        """
        configuration = get_cfg()
        configuration.merge_from_file(self.config.config_file)
        configuration.merge_from_list(["MODEL.WEIGHTS", self.config.weight_file])
        configuration.MODEL.DEVICE = self.model_device
        # Set score_threshold for builtin models
        configuration.MODEL.RETINANET.SCORE_THRESH_TEST = (
            self.config.confidence_threshold
        )
        configuration.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.config.confidence_threshold
        )
        # SOLOV2
        configuration.MODEL.FCOS.INFERENCE_TH_TEST = self.config.confidence_threshold
        configuration.MODEL.MEInst.INFERENCE_TH_TEST = self.config.confidence_threshold
        # configuration.MODEL.SOLOV2.SCORE_THR = self.config.confidence_threshold
        configuration.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            self.config.confidence_threshold
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
        self.class_ids = [
            index
            for index, element in enumerate(self.class_names)
            if element in self.config.classes
        ]

        return model

    def detect_process(self, predictions):
        """Filter predictions
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            segm_masks, frame_boxes, scores, classes
        """
        classes = (
            predictions.pred_classes.numpy()
            if predictions.has("pred_classes")
            else None
        )

        scores = predictions.scores.numpy() if predictions.has("scores") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        class_mask = np.isin(classes, self.class_ids)
        score_mask = scores >= self.config.detect_thresh
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

            # seg_masks
            return frame_boxes, scores, classes

    def inference(self, image):
        boxes = []
        predictions = self.model(image)
        predictions = predictions["instances"].to(torch.device("cpu"))
        if predictions:
            (
                # seg_masks,
                boxes,
                scores,
                classes,
            ) = self.detect_process(predictions)
        return boxes, scores, classes


class CameraCalibration:
    def __init__(self, class_names, data_dict):
        self.class_names = class_names
        self.height = None
        self.width = None
        self.__dict__.update(data_dict)
        self.perspective_matrix = None

    def load_calibration(self, verbose=False):
        perspective_matrix = (
            self.intrinsic @ self.rotation.T @ np.linalg.inv(self.intrinsic)
        )
        est_range_u, est_range_v = calculate_range_output(
            self.height, self.width, perspective_matrix, opt=True, verbose=verbose
        )
        moveup_camera = np.array(
            [[1, 0, est_range_u[1]], [0, 1, est_range_v[1]], [0, 0, 1]]
        )
        perspective_matrix = np.dot(moveup_camera, perspective_matrix)
        scale_matrix = get_scaled_matrix(
            perspective_matrix,
            [self.width, self.height],
            est_range_u,
            est_range_v,
            strict=False,
        )
        return scale_matrix

    def convert_2d_world(self, x_2d, y_2d):
        image_coords = np.array([x_2d, y_2d, 1])
        world_coords = image_coords @ self.perspective_matrix.T
        return world_coords

    def projector(self, image, boxes, scores, classes):
        text_scale = 1.5
        text_thickness = 2
        line_thickness = 3
        for idx, tlbr in enumerate(boxes):
            score = float(scores[idx])
            class_name = self.class_names[classes[idx]]
            print(f"{score:.2f}, {class_name}")

            intbox = list(map(int, tlbr))
            # x1, y1, x2, y2 = intbox
            first_point = self.convert_2d_world(1377, 413)
            # 1377 413, 1001 907
            second_point = self.convert_2d_world(1001, 907)
            distance = np.linalg.norm(first_point - second_point)
            print(f"1: {first_point} 2: {second_point}")

            i_color = colors_instance[idx]
            cv2.rectangle(
                image, intbox[0:2], intbox[2:4], color=i_color, thickness=line_thickness
            )
            cv2.putText(
                image,
                f"H: {distance:.2f} {score:.2f}, {class_name}",
                (intbox[0], intbox[1] - 5),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 0),
                thickness=text_thickness,
            )
        return image

    def test_calibration(self, detect_model, img_path, out_dir):
        """Convert 2D bbox coords from tracker results to world coords
        Sample Data: 160,1.0,1340.00,190.00,273.00,195.00,1.0,-1,-1,-1
        Track data: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        image = cv2.imread(img_path)
        self.height, self.width, _ = image.shape
        self.perspective_matrix = self.load_calibration()
        boxes, scores, classes = detect_model.inference(image)
        if boxes.any():
            image = self.projector(image, boxes, scores, classes)
            infer_path = os.path.join(out_dir, "inference.jpg")
            cv2.imwrite(infer_path, image)


if __name__ == "__main__":
    config = get_config("configs/calib_config.yaml")
    if config.root_dir:
        track_path = os.path.join(config.root_dir, config.track_path)
        segm_path = os.path.join(config.root_dir, config.segm_path)
        image_path = os.path.join(config.root_dir, config.image_path)
    else:
        track_path = config.track_path
        segm_path = config.segm_path
        image_path = config.image_path

    # Create object
    vpd = VPDetection(
        config,
        track_path,
        segm_path,
    )
    if config.is_calibrate:
        # Run VP detection algorithm
        vpd.calibration(image_path)

    else:
        file_path = os.path.join(config.out_dir, "calibration.pkl")
        file = open(file_path, "rb")
        calib_dict = pickle.load(file)
        file.close()

        print(calib_dict)
        detect = Detection(config)
        calib = CameraCalibration(class_names=detect.class_names, data_dict=calib_dict)
        calib.test_calibration(detect, image_path, config.out_dir)
