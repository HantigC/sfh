from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sfh.domain.entities import Dataset
from sfh.domain.errors import DatasetAlreadyExistsError, DuplicateImageNameError
from sfh.domain.ports.features import ImageLoader
from sfh.domain.ports.progress import NullProgress, ProgressReporter
from sfh.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class RegisterDatasetResult:
    dataset: Dataset
    images_added: int


class RegisterDataset:
    """Create a named dataset and record its images (path + size; pixels stay on disk)."""

    def __init__(self, uow: UnitOfWorkFactory, loader: ImageLoader) -> None:
        self._uow = uow
        self._loader = loader

    def __call__(
        self,
        name: str,
        paths: Sequence[Path],
        progress: ProgressReporter | None = None,
    ) -> RegisterDatasetResult:
        progress = progress or NullProgress()
        ordered = sorted(paths, key=lambda p: p.name)
        seen: set[str] = set()
        for path in ordered:
            if path.name in seen:
                raise DuplicateImageNameError(path.name)
            seen.add(path.name)

        with self._uow() as uow:
            if uow.datasets.get_by_name(name) is not None:
                raise DatasetAlreadyExistsError(name)
            dataset = uow.datasets.add(name)
            progress.start(len(ordered))
            for path in ordered:
                width, height = self._loader.size(path)
                uow.images.add(dataset.id, path.name, path, width, height)
                progress.advance()
            uow.commit()
        progress.finish()
        return RegisterDatasetResult(dataset=dataset, images_added=len(ordered))
