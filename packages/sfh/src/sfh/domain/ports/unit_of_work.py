from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from sfh.domain.ports.repositories import (
    DatasetRepository,
    DescriptorRepository,
    FeatureConfigRepository,
    ImageRepository,
    KeypointsRepository,
)


class UnitOfWork(Protocol):
    """Transaction boundary over all repositories.

    Writes become visible only after `commit()`; leaving the `with` block without
    committing rolls them back.
    """

    @property
    def datasets(self) -> DatasetRepository: ...
    @property
    def images(self) -> ImageRepository: ...
    @property
    def configs(self) -> FeatureConfigRepository: ...
    @property
    def keypoints(self) -> KeypointsRepository: ...
    @property
    def descriptors(self) -> DescriptorRepository: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
