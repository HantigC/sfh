from collections.abc import Sequence
from dataclasses import dataclass

from sfh.domain.entities import Dataset
from sfh.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    dataset: Dataset
    image_count: int


class ListDatasets:
    def __init__(self, uow: UnitOfWorkFactory) -> None:
        self._uow = uow

    def __call__(self) -> Sequence[DatasetSummary]:
        with self._uow() as uow:
            return [
                DatasetSummary(d, len(uow.images.list_by_dataset(d.id)))
                for d in uow.datasets.list()
            ]
