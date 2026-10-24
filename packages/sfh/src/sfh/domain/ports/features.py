from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sfh.domain.entities import (
    Descriptors,
    ExtractedFeatures,
    Image,
    Keypoints,
    MatchResult,
    Params,
)
from sfh.domain.ids import ImageId


class ImageLoader(Protocol):
    def load(self, path: Path) -> NDArray[np.uint8]:
        """Pixels as an (H, W, 3) RGB array. Raises ImageReadError."""
        ...

    def size(self, path: Path) -> tuple[int, int]:
        """(width, height). Raises ImageReadError."""
        ...


class FeatureExtractor(Protocol):
    """Detector and descriptor together; learned methods compute both jointly."""

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> Params:
        """Everything that changes the output; it keys the stored results."""
        ...

    def extract(self, pixels: NDArray[np.uint8]) -> ExtractedFeatures: ...


class Matcher(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def params(self) -> Params: ...

    def match(
        self, kp_a: Keypoints, d_a: Descriptors, kp_b: Keypoints, d_b: Descriptors
    ) -> MatchResult: ...


class PairSelector(Protocol):
    """Chooses which image pairs to match: exhaustive, sequential, retrieval, ..."""

    def select(self, images: Sequence[Image]) -> Iterable[tuple[ImageId, ImageId]]: ...
