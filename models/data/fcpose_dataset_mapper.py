import copy
import logging

from models.data.detection_utils import HeatmapGenerator
from models.data.dataset_mapper import DatasetMapperWithBasis

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


class FCPoseDatasetMapper(DatasetMapperWithBasis):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.fcpose_on = cfg.MODEL.FCPOSE_ON
        if self.fcpose_on:
            self.gt_heatmap_stride = cfg.MODEL.FCPOSE.GT_HEATMAP_STRIDE
            self.sigma = cfg.MODEL.FCPOSE.HEATMAP_SIGMA
            self.head_sigma = cfg.MODEL.FCPOSE.HEAD_HEATMAP_SIGMA
            self.HeatmapGenerator = HeatmapGenerator(17, self.sigma, self.head_sigma)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        for i in range(100):
            dataset_dict_temp = copy.deepcopy(dataset_dict)
            dataset_dict_temp = super().__call__(dataset_dict_temp)
            if len(dataset_dict_temp["instances"]) != 0:
                if self.is_train:
                    dataset_dict_temp["instances"] = self.HeatmapGenerator(
                        dataset_dict_temp["instances"], self.gt_heatmap_stride
                    )
                return dataset_dict_temp
        raise
