import math

import cv2
import numpy as np
from numpy.typing import NDArray

from sfh.domain.entities import ExtractedFeatures, Params


class SiftExtractor:
    """OpenCV SIFT: 128-d float32 descriptors, keypoints with scale and orientation."""

    def __init__(self, max_keypoints: int = 8192, contrast_threshold: float = 0.04) -> None:
        self._max_keypoints = max_keypoints
        self._contrast_threshold = contrast_threshold
        self._sift = cv2.SIFT.create(nfeatures=max_keypoints, contrastThreshold=contrast_threshold)

    @property
    def name(self) -> str:
        return "opencv-sift"

    @property
    def params(self) -> Params:
        return {
            "max_keypoints": self._max_keypoints,
            "contrast_threshold": self._contrast_threshold,
        }

    def extract(self, pixels: NDArray[np.uint8]) -> ExtractedFeatures:
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        n = len(keypoints)
        return ExtractedFeatures(
            xy=np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(n, 2),
            descriptors=(
                np.zeros((0, 128), dtype=np.float32)
                if descriptors is None
                else np.asarray(descriptors, dtype=np.float32)
            ),
            scores=np.array([kp.response for kp in keypoints], dtype=np.float32),
            scales=np.array([kp.size for kp in keypoints], dtype=np.float32),
            orientations=np.array([math.radians(kp.angle) for kp in keypoints], dtype=np.float32),
        )
