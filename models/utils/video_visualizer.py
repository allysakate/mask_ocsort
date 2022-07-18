# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import List
import pycocotools.mask as mask_util
from torchvision.ops import masks_to_boxes

from detectron2.structures import Instances
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)

from detectron2.utils.colormap import random_color, random_colors


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_SCORE_THRESHOLD = 0.5
_KEYPOINT_THRESHOLD = 0.05
_ASPECT_RATIO_THRESH = 1.6
_MIN_BOX_AREA_THRESH = 10


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self._max_num_instances = self.metadata.get("max_num_instances", 74)
        self._assigned_colors = {}
        self._color_pool = random_colors(self._max_num_instances, rgb=True, maximum=1)
        self._color_idx_set = set(range(len(self._color_pool)))

    def draw_instance_predictions(self, frame, predictions, tracker, class_list):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            class_list (List): list of class ids

        Returns:
            output (VisImage): image object with visualizations.
        """
        vis_output = None

        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        # TODO: Filter out non-vehicle classes
        classes = (
            predictions.pred_classes.numpy()
            if predictions.has("pred_classes")
            else None
        )

        scores = predictions.scores.numpy() if predictions.has("scores") else None

        class_mask = np.isin(classes, class_list)
        filter_mask = class_mask
        # score_mask = scores >= _SCORE_THRESHOLD
        # filter_mask = np.logical_and(class_mask, score_mask)

        boxes = (
            predictions.pred_boxes.tensor.numpy()
            if predictions.has("pred_boxes")
            else None
        )

        boxes = boxes[filter_mask]
        classes = classes[filter_mask]
        scores = scores[filter_mask]

        if scores.any():
            keypoints = (
                predictions.pred_keypoints
                if predictions.has("pred_keypoints")
                else None
            )
            if keypoints:
                keypoints = keypoints[filter_mask]

            colors = (
                predictions.COLOR if predictions.has("COLOR") else [None] * len(boxes)
            )

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
                masks = predictions.pred_masks
                masks = masks[filter_mask]
                # mask IOU is not yet enabled
                # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
                # assert len(masks_rles) == num_instances
            else:
                masks = None

            if not predictions.has("COLOR"):
                if predictions.has("ID"):
                    colors = self._assign_colors_by_id(predictions)
                else:
                    # ToDo: clean old assign color method and use a default tracker to assign id
                    detected = [
                        _DetectedInstance(
                            classes[i], boxes[i], mask_rle=None, color=colors[i], ttl=8
                        )
                        for i in range(len(classes))
                    ]
                    colors = self._assign_colors(detected)

            labels = _create_text_labels(
                classes, scores, self.metadata.get("thing_classes", None)
            )

            if self._instance_mode == ColorMode.IMAGE_BW:
                # any() returns uint8 tensor
                frame_visualizer.output.reset_image(
                    frame_visualizer._create_grayscale_image(
                        (masks.any(dim=0) > 0).numpy() if masks is not None else None
                    )
                )
                alpha = 0.3
            else:
                alpha = 0.5

            labels = (
                None
                if labels is None
                else [y[0] for y in filter(lambda x: x[1], zip(labels, visibilities))]
            )  # noqa
            assigned_colors = (
                None
                if colors is None
                else [y[0] for y in filter(lambda x: x[1], zip(colors, visibilities))]
            )  # noqa

            vis_output = self._overlay_instances(
                tracker=tracker,
                visualizer=frame_visualizer,
                boxes=None
                if masks is not None
                else boxes[visibilities],  # boxes are a bit distracting
                scores=scores,
                masks=None if masks is None else masks[visibilities],
                labels=labels,
                keypoints=None if keypoints is None else keypoints[visibilities],
                assigned_colors=assigned_colors,
                alpha=alpha,
            )
        return vis_output

    def draw_sem_seg(self, frame, sem_seg, area_threshold=None):
        """
        Args:
            sem_seg (ndarray or Tensor): semantic segmentation of shape (H, W),
                each value is the integer label.
            area_threshold (Optional[int]): only draw segmentations larger than the threshold
        """
        # don't need to do anything special
        frame_visualizer = Visualizer(frame, self.metadata)
        frame_visualizer.draw_sem_seg(sem_seg, area_threshold=None)
        return frame_visualizer.output

    def draw_panoptic_seg_predictions(
        self, frame, panoptic_seg, segments_info, area_threshold=None, alpha=0.5
    ):
        frame_visualizer = Visualizer(frame, self.metadata)
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            frame_visualizer.output.reset_image(
                frame_visualizer._create_grayscale_image(pred.non_empty_mask())
            )

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            frame_visualizer.draw_binary_mask(
                mask,
                color=mask_color,
                text=self.metadata.stuff_classes[category_idx],
                alpha=alpha,
                area_threshold=area_threshold,
            )

        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return frame_visualizer.output
        # draw mask for all instances second
        masks, sinfo = list(zip(*all_instances))
        num_instances = len(masks)
        masks_rles = mask_util.encode(
            np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
        )
        assert len(masks_rles) == num_instances

        category_ids = [x["category_id"] for x in sinfo]
        detected = [
            _DetectedInstance(
                category_ids[i], bbox=None, mask_rle=masks_rles[i], color=None, ttl=8
            )
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)
        labels = [self.metadata.thing_classes[k] for k in category_ids]

        frame_visualizer.overlay_instances(
            boxes=None,
            masks=masks,
            labels=labels,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return frame_visualizer.output

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.5
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]

    def _assign_colors_by_id(self, instances: Instances) -> List:
        colors = []
        untracked_ids = set(self._assigned_colors.keys())
        for id in instances.ID:
            if id in self._assigned_colors:
                colors.append(self._color_pool[self._assigned_colors[id]])
                untracked_ids.remove(id)
            else:
                assert (
                    len(self._color_idx_set) >= 1
                ), f"Number of id exceeded maximum, \
                    max = {self._max_num_instances}"
                idx = self._color_idx_set.pop()
                color = self._color_pool[idx]
                self._assigned_colors[id] = idx
                colors.append(color)
        for id in untracked_ids:
            self._color_idx_set.add(self._assigned_colors[id])
            del self._assigned_colors[id]
        return colors

    def _overlay_instances(
        self,
        tracker=None,
        visualizer=None,
        boxes=None,
        scores=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        height, width = visualizer.img.shape[:2]
        frame_masks = masks
        num_instances = 0
        if boxes is not None:
            boxes = visualizer._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = visualizer._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = visualizer._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return visualizer.output
        if boxes is not None and boxes.shape[1] == 5:
            return visualizer.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        track_tlbrs = []
        track_ids = []
        frame_boxes = None
        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            frame_masks = frame_masks[sorted_idxs, :, :]
            if frame_masks is not None:
                frame_boxes = masks_to_boxes(frame_masks)

                # TODO: tracking
                targets = tracker.update(
                    frame_boxes, scores, [height, width], (1080, 1920)
                )
                for idx, target in enumerate(targets):
                    box_width = target[2] - target[0]
                    box_height = target[3] - target[1]
                    tlbr = [target[0], target[1], target[2], target[3]]
                    tid = target[4]
                    vertical = box_width / box_height > _ASPECT_RATIO_THRESH
                    if box_width * box_height > _MIN_BOX_AREA_THRESH and not vertical:
                        track_tlbrs.append(tlbr)
                        track_ids.append(tid)
                        color = assigned_colors[idx]
                        visualizer.draw_box(tlbr, edge_color=color)
                        text_pos = (
                            target[0],
                            target[1],
                        )  # if drawing boxes, put text on the box corner.
                        horiz_align = "left"
                        lighter_color = visualizer._change_color_brightness(
                            color, brightness_factor=0.7
                        )
                        visualizer.draw_text(
                            f"ID: {tid}",
                            text_pos,
                            color=lighter_color,
                            horizontal_alignment=horiz_align,
                            font_size=12,
                        )

        for i in range(num_instances):
            color = assigned_colors[i]
            if frame_boxes is not None:
                visualizer.draw_box(frame_boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    visualizer.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * visualizer.output.scale
                    or y1 - y0 < 40 * visualizer.output.scale
                ):
                    if y1 >= visualizer.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(
                    visualizer.output.height * visualizer.output.width
                )
                lighter_color = visualizer._change_color_brightness(
                    color, brightness_factor=0.7
                )
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * visualizer._default_font_size
                )
                visualizer.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                visualizer.draw_and_connect_keypoints(keypoints_per_instance)
        return visualizer.output
