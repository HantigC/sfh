from pathlib import Path

import pytest

from sfh.application.register_dataset import RegisterDataset
from sfh.domain.errors import (
    DatasetAlreadyExistsError,
    DuplicateImageNameError,
    ImageReadError,
)
from tests.fakes import FakeImageLoader, FakeUnitOfWorkFactory


def test_registers_images_sorted_by_name() -> None:
    uow = FakeUnitOfWorkFactory()
    register = RegisterDataset(uow, FakeImageLoader(width=640, height=480))

    result = register("church", [Path("/d/b.png"), Path("/d/a.png")])

    assert result.images_added == 2
    with uow() as u:
        images = u.images.list_by_dataset(result.dataset.id)
    assert [i.name for i in images] == ["a.png", "b.png"]
    assert (images[0].width, images[0].height) == (640, 480)


def test_rejects_existing_dataset_name() -> None:
    uow = FakeUnitOfWorkFactory()
    register = RegisterDataset(uow, FakeImageLoader())
    register("church", [Path("a.png")])

    with pytest.raises(DatasetAlreadyExistsError):
        register("church", [Path("b.png")])


def test_rejects_duplicate_image_names() -> None:
    register = RegisterDataset(FakeUnitOfWorkFactory(), FakeImageLoader())

    with pytest.raises(DuplicateImageNameError):
        register("church", [Path("/x/a.png"), Path("/y/a.png")])


def test_unreadable_image_leaves_nothing_behind() -> None:
    uow = FakeUnitOfWorkFactory()
    register = RegisterDataset(uow, FakeImageLoader(broken={Path("b.png")}))

    with pytest.raises(ImageReadError):
        register("church", [Path("a.png"), Path("b.png")])

    with uow() as u:
        assert u.datasets.get_by_name("church") is None
