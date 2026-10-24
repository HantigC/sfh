"""Composition root: the only place that chooses concrete adapters."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sfh.application.extract_features import ExtractFeatures
from sfh.application.list_datasets import ListDatasets
from sfh.application.register_dataset import RegisterDataset
from sfh.domain.ports.features import FeatureExtractor
from sfh.infrastructure.features.opencv_sift import SiftExtractor
from sfh.infrastructure.io.opencv_loader import OpenCvImageLoader
from sfh.infrastructure.sqlite.unit_of_work import SqliteUnitOfWork, connect

EXTRACTORS: dict[str, Callable[[], FeatureExtractor]] = {
    "sift": lambda: SiftExtractor(max_keypoints=8192),
}


@dataclass(frozen=True, slots=True)
class Container:
    register_dataset: RegisterDataset
    list_datasets: ListDatasets
    extract_features: ExtractFeatures


@contextmanager
def open_container(db_path: Path, extractor: str = "sift") -> Iterator[Container]:
    if extractor not in EXTRACTORS:
        raise ValueError(f"unknown extractor {extractor!r}; choose from {sorted(EXTRACTORS)}")
    conn = connect(db_path)
    try:

        def uow() -> SqliteUnitOfWork:
            return SqliteUnitOfWork(conn)

        loader = OpenCvImageLoader()
        yield Container(
            register_dataset=RegisterDataset(uow, loader),
            list_datasets=ListDatasets(uow),
            extract_features=ExtractFeatures(uow, loader, EXTRACTORS[extractor]()),
        )
    finally:
        conn.close()
