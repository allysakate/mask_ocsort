# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, confidence, classification, feature):
        self.tlbr_arr = tlbr
        self.tlbr = tlbr.tolist()
        self.tlwh = np.asarray(self._to_tlhw(tlbr), dtype=np.float)
        self.confidence = float(confidence)
        self.classification = classification
        self.feature = np.asarray(feature, dtype=np.float32)

    def _to_tlhw(self, tlbr):
        """Convert bounding box to format (min x, min y, max x, max y)
        to (min x, min y, width, height).
        """
        ret = tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret