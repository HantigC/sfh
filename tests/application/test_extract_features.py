from pathlib import Path

import pytest

from sfh.application.extract_features import ExtractFeatures
from sfh.application.register_dataset import RegisterDataset
from sfh.domain.errors import DatasetNotFoundError, ImageReadError
from tests.fakes import FakeExtractor, FakeImageLoader, FakeUnitOfWorkFactory

PATHS = [Path("a.png"), Path("b.png"), Path("c.png")]


class RecordingProgress:
    def __init__(self) -> None:
        self.total: int | None = None
        self.advanced = 0
        self.finished = False

    def start(self, total: int) -> None:
        self.total = total

    def advance(self, n: int = 1) -> None:
        self.advanced += n

    def finish(self) -> None:
        self.finished = True


@pytest.fixture
def uow() -> FakeUnitOfWorkFactory:
    uow = FakeUnitOfWorkFactory()
    RegisterDataset(uow, FakeImageLoader())("church", PATHS)
    return uow


def test_extracts_and_stores_features_for_every_image(uow: FakeUnitOfWorkFactory) -> None:
    extractor = FakeExtractor(n=5, dim=8)
    progress = RecordingProgress()

    result = ExtractFeatures(uow, FakeImageLoader(), extractor)("church", progress)

    assert (result.extracted, result.skipped) == (3, 0)
    assert result.config.name == "fake"
    assert (progress.total, progress.advanced, progress.finished) == (3, 3, True)
    with uow() as u:
        dataset = u.datasets.get_by_name("church")
        assert dataset is not None
        for image in u.images.list_by_dataset(dataset.id):
            kp = u.keypoints.get(image.id, result.config.id)
            desc = u.descriptors.get(image.id, result.config.id)
            assert kp is not None
            assert desc is not None
            assert len(kp) == 5
            assert desc.data.shape == (5, 8)


def test_second_run_skips_finished_images(uow: FakeUnitOfWorkFactory) -> None:
    extractor = FakeExtractor()
    step = ExtractFeatures(uow, FakeImageLoader(), extractor)
    step("church")

    result = step("church")

    assert (result.extracted, result.skipped) == (0, 3)
    assert extractor.calls == 3


def test_different_params_get_their_own_results(uow: FakeUnitOfWorkFactory) -> None:
    first = ExtractFeatures(uow, FakeImageLoader(), FakeExtractor(version=1))("church")
    second = ExtractFeatures(uow, FakeImageLoader(), FakeExtractor(version=2))("church")

    assert first.config.id != second.config.id
    assert second.extracted == 3


def test_interrupted_run_keeps_finished_images(uow: FakeUnitOfWorkFactory) -> None:
    loader = FakeImageLoader(broken={Path("c.png")})
    with pytest.raises(ImageReadError):
        ExtractFeatures(uow, loader, FakeExtractor())("church")

    result = ExtractFeatures(uow, FakeImageLoader(), FakeExtractor())("church")

    assert (result.extracted, result.skipped) == (1, 2)


def test_unknown_dataset() -> None:
    step = ExtractFeatures(FakeUnitOfWorkFactory(), FakeImageLoader(), FakeExtractor())

    with pytest.raises(DatasetNotFoundError):
        step("missing")
