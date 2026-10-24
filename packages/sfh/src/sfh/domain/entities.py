from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sfh.domain.ids import DatasetId, FeatureConfigId, ImageId

ParamValue = str | int | float | bool | None
Params = Mapping[str, ParamValue]


@dataclass(frozen=True, slots=True)
class Dataset:
    id: DatasetId
    name: str


@dataclass(frozen=True, slots=True)
class Image:
    id: ImageId
    dataset_id: DatasetId
    name: str  # unique within the dataset
    path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Identifies which method produced a result, so results of several methods coexist."""

    id: FeatureConfigId
    name: str
    params: Params


def _check_per_keypoint(name: str, values: NDArray[Any] | None, n: int) -> None:
    if values is not None and values.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {values.shape}")


@dataclass(frozen=True, slots=True, eq=False)
class ExtractedFeatures:
    """What a FeatureExtractor returns: keypoints and descriptors not yet tied to an image."""

    xy: NDArray[np.float32]  # (N, 2) pixel coordinates
    descriptors: NDArray[Any]  # (N, D) float32 for SIFT/learned, uint8 for binary
    scores: NDArray[np.float32] | None = None  # (N,)
    scales: NDArray[np.float32] | None = None  # (N,)
    orientations: NDArray[np.float32] | None = None  # (N,) radians

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2), got {self.xy.shape}")
        n = self.xy.shape[0]
        if self.descriptors.ndim != 2 or self.descriptors.shape[0] != n:
            raise ValueError(f"descriptors must have shape ({n}, D), got {self.descriptors.shape}")
        _check_per_keypoint("scores", self.scores, n)
        _check_per_keypoint("scales", self.scales, n)
        _check_per_keypoint("orientations", self.orientations, n)


@dataclass(frozen=True, slots=True, eq=False)
class Keypoints:
    image_id: ImageId
    config_id: FeatureConfigId
    xy: NDArray[np.float32]  # (N, 2)
    scores: NDArray[np.float32] | None = None
    scales: NDArray[np.float32] | None = None
    orientations: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2), got {self.xy.shape}")
        n = len(self)
        _check_per_keypoint("scores", self.scores, n)
        _check_per_keypoint("scales", self.scales, n)
        _check_per_keypoint("orientations", self.orientations, n)

    def __len__(self) -> int:
        return int(self.xy.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class Descriptors:
    image_id: ImageId
    config_id: FeatureConfigId
    data: NDArray[Any]  # (N, D)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"descriptors must have shape (N, D), got {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class MatchResult:
    """What a Matcher returns for one image pair."""

    indices: NDArray[np.int32]  # (M, 2): keypoint index in image a, index in image b
    scores: NDArray[np.float32] | None = None  # (M,)
