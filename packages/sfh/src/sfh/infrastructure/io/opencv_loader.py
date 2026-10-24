from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from sfh.domain.errors import ImageReadError


class OpenCvImageLoader:
    def load(self, path: Path) -> NDArray[np.uint8]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageReadError(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb, dtype=np.uint8)

    def size(self, path: Path) -> tuple[int, int]:
        # Decodes the whole image; swap for a header-only reader if registration gets slow.
        height, width = self.load(path).shape[:2]
        return width, height
