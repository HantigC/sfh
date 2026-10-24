"""In-memory implementations of the ports. Copy these patterns when testing a new step."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from sfh.domain.entities import (
    Dataset,
    Descriptors,
    ExtractedFeatures,
    FeatureConfig,
    Image,
    Keypoints,
    Params,
)
from sfh.domain.errors import ImageNotFoundError, ImageReadError
from sfh.domain.ids import DatasetId, FeatureConfigId, ImageId


@dataclass
class InMemoryStore:
    """Committed state shared by every FakeUnitOfWork created from one factory."""

    datasets: dict[DatasetId, Dataset] = field(default_factory=dict)
    images: dict[ImageId, Image] = field(default_factory=dict)
    configs: dict[str, FeatureConfig] = field(default_factory=dict)
    keypoints: dict[tuple[ImageId, FeatureConfigId], Keypoints] = field(default_factory=dict)
    descriptors: dict[tuple[ImageId, FeatureConfigId], Descriptors] = field(default_factory=dict)

    def copy(self) -> "InMemoryStore":
        return InMemoryStore(
            dict(self.datasets),
            dict(self.images),
            dict(self.configs),
            dict(self.keypoints),
            dict(self.descriptors),
        )


class FakeDatasetRepository:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def add(self, name: str) -> Dataset:
        if any(d.name == name for d in self._store.datasets.values()):
            raise ValueError(f"duplicate dataset {name!r}")
        dataset = Dataset(DatasetId(len(self._store.datasets) + 1), name)
        self._store.datasets[dataset.id] = dataset
        return dataset

    def get_by_name(self, name: str) -> Dataset | None:
        return next((d for d in self._store.datasets.values() if d.name == name), None)

    def list(self) -> Sequence[Dataset]:
        return sorted(self._store.datasets.values(), key=lambda d: d.name)


class FakeImageRepository:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def add(self, dataset_id: DatasetId, name: str, path: Path, width: int, height: int) -> Image:
        image = Image(ImageId(len(self._store.images) + 1), dataset_id, name, path, width, height)
        self._store.images[image.id] = image
        return image

    def get(self, image_id: ImageId) -> Image:
        if image_id not in self._store.images:
            raise ImageNotFoundError(image_id)
        return self._store.images[image_id]

    def list_by_dataset(self, dataset_id: DatasetId) -> Sequence[Image]:
        images = [i for i in self._store.images.values() if i.dataset_id == dataset_id]
        return sorted(images, key=lambda i: i.name)


class FakeFeatureConfigRepository:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def get_or_create(self, name: str, params: Params) -> FeatureConfig:
        key = json.dumps({"name": name, "params": dict(params)}, sort_keys=True)
        if key not in self._store.configs:
            config_id = FeatureConfigId(len(self._store.configs) + 1)
            self._store.configs[key] = FeatureConfig(config_id, name, dict(params))
        return self._store.configs[key]


class FakeKeypointsRepository:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def save(self, keypoints: Keypoints) -> None:
        self._store.keypoints[keypoints.image_id, keypoints.config_id] = keypoints

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Keypoints | None:
        return self._store.keypoints.get((image_id, config_id))

    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool:
        return (image_id, config_id) in self._store.keypoints


class FakeDescriptorRepository:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def save(self, descriptors: Descriptors) -> None:
        self._store.descriptors[descriptors.image_id, descriptors.config_id] = descriptors

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Descriptors | None:
        return self._store.descriptors.get((image_id, config_id))

    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool:
        return (image_id, config_id) in self._store.descriptors


class FakeUnitOfWork:
    """Works on a copy of the store; commit() publishes the copy."""

    def __init__(self, committed: InMemoryStore) -> None:
        self._committed = committed
        self._pending = committed.copy()
        self.datasets = FakeDatasetRepository(self._pending)
        self.images = FakeImageRepository(self._pending)
        self.configs = FakeFeatureConfigRepository(self._pending)
        self.keypoints = FakeKeypointsRepository(self._pending)
        self.descriptors = FakeDescriptorRepository(self._pending)
        self.commits = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def commit(self) -> None:
        for name in ("datasets", "images", "configs", "keypoints", "descriptors"):
            setattr(self._committed, name, dict(getattr(self._pending, name)))
        self.commits += 1


class FakeUnitOfWorkFactory:
    def __init__(self) -> None:
        self.store = InMemoryStore()

    def __call__(self) -> FakeUnitOfWork:
        return FakeUnitOfWork(self.store)


class FakeImageLoader:
    """Serves a deterministic image per path; paths in `broken` fail to load."""

    def __init__(self, width: int = 64, height: int = 48, broken: set[Path] | None = None) -> None:
        self._width = width
        self._height = height
        self._broken = broken or set()
        self.loaded: list[Path] = []

    def load(self, path: Path) -> NDArray[np.uint8]:
        if path in self._broken:
            raise ImageReadError(str(path))
        self.loaded.append(path)
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def size(self, path: Path) -> tuple[int, int]:
        if path in self._broken:
            raise ImageReadError(str(path))
        return self._width, self._height


class FakeExtractor:
    def __init__(self, n: int = 5, dim: int = 8, version: int = 1) -> None:
        self._n = n
        self._dim = dim
        self._version = version
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def params(self) -> Params:
        return {"n": self._n, "dim": self._dim, "version": self._version}

    def extract(self, pixels: NDArray[np.uint8]) -> ExtractedFeatures:
        self.calls += 1
        return ExtractedFeatures(
            xy=np.arange(self._n * 2, dtype=np.float32).reshape(self._n, 2),
            descriptors=np.ones((self._n, self._dim), dtype=np.float32),
            scores=np.linspace(0, 1, self._n, dtype=np.float32),
        )
