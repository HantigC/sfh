from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from sfh.domain.entities import Dataset, Descriptors, FeatureConfig, Image, Keypoints, Params
from sfh.domain.ids import DatasetId, FeatureConfigId, ImageId


class DatasetRepository(Protocol):
    def add(self, name: str) -> Dataset: ...
    def get_by_name(self, name: str) -> Dataset | None: ...
    def list(self) -> Sequence[Dataset]: ...


class ImageRepository(Protocol):
    def add(
        self, dataset_id: DatasetId, name: str, path: Path, width: int, height: int
    ) -> Image: ...

    def get(self, image_id: ImageId) -> Image:
        """Raises ImageNotFoundError."""
        ...

    def list_by_dataset(self, dataset_id: DatasetId) -> Sequence[Image]:
        """Images ordered by name."""
        ...


class FeatureConfigRepository(Protocol):
    def get_or_create(self, name: str, params: Params) -> FeatureConfig: ...


class KeypointsRepository(Protocol):
    def save(self, keypoints: Keypoints) -> None:
        """Insert, replacing an existing row for the same (image, config)."""
        ...

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Keypoints | None: ...
    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool: ...


class DescriptorRepository(Protocol):
    def save(self, descriptors: Descriptors) -> None:
        """Insert, replacing an existing row for the same (image, config)."""
        ...

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Descriptors | None: ...
    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool: ...
