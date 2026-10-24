from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from sfh.domain.entities import Descriptors, Keypoints
from sfh.domain.errors import ImageNotFoundError
from sfh.domain.ids import FeatureConfigId, ImageId
from sfh.domain.ports.unit_of_work import UnitOfWorkFactory


def test_datasets_add_get_list(uow_factory: UnitOfWorkFactory) -> None:
    with uow_factory() as uow:
        b = uow.datasets.add("b")
        a = uow.datasets.add("a")
        uow.commit()

    with uow_factory() as uow:
        assert uow.datasets.get_by_name("a") == a
        assert uow.datasets.get_by_name("missing") is None
        assert [d.name for d in uow.datasets.list()] == ["a", "b"]
        assert a.id != b.id


def test_uncommitted_writes_are_discarded(uow_factory: UnitOfWorkFactory) -> None:
    with uow_factory() as uow:
        uow.datasets.add("never-committed")

    with uow_factory() as uow:
        assert uow.datasets.get_by_name("never-committed") is None


def test_images_add_get_list(uow_factory: UnitOfWorkFactory) -> None:
    with uow_factory() as uow:
        ds = uow.datasets.add("ds")
        other = uow.datasets.add("other")
        img2 = uow.images.add(ds.id, "b.jpg", Path("/x/b.jpg"), 640, 480)
        img1 = uow.images.add(ds.id, "a.jpg", Path("/x/a.jpg"), 320, 240)
        uow.images.add(other.id, "c.jpg", Path("/x/c.jpg"), 1, 1)
        uow.commit()

    with uow_factory() as uow:
        assert uow.images.get(img2.id) == img2
        assert uow.images.list_by_dataset(ds.id) == [img1, img2]
        with pytest.raises(ImageNotFoundError):
            uow.images.get(ImageId(9999))


def test_feature_configs_are_keyed_by_name_and_params(uow_factory: UnitOfWorkFactory) -> None:
    with uow_factory() as uow:
        sift = uow.configs.get_or_create("sift", {"n": 10, "thr": 0.5})
        again = uow.configs.get_or_create("sift", {"thr": 0.5, "n": 10})
        other_params = uow.configs.get_or_create("sift", {"n": 20, "thr": 0.5})
        other_name = uow.configs.get_or_create("orb", {"n": 10, "thr": 0.5})
        uow.commit()

    assert again == sift
    assert dict(sift.params) == {"n": 10, "thr": 0.5}
    assert len({sift.id, other_params.id, other_name.id}) == 3


def _image_and_config(uow_factory: UnitOfWorkFactory) -> tuple[ImageId, FeatureConfigId]:
    with uow_factory() as uow:
        ds = uow.datasets.add("ds")
        image = uow.images.add(ds.id, "a.jpg", Path("a.jpg"), 10, 10)
        config = uow.configs.get_or_create("x", {})
        uow.commit()
    return image.id, config.id


def test_keypoints_round_trip(uow_factory: UnitOfWorkFactory) -> None:
    image_id, config_id = _image_and_config(uow_factory)
    xy = np.random.default_rng(0).random((7, 2), dtype=np.float32)
    scores = np.arange(7, dtype=np.float32)

    with uow_factory() as uow:
        assert not uow.keypoints.exists(image_id, config_id)
        uow.keypoints.save(Keypoints(image_id, config_id, xy, scores=scores))
        uow.commit()

    with uow_factory() as uow:
        assert uow.keypoints.exists(image_id, config_id)
        loaded = uow.keypoints.get(image_id, config_id)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.xy, xy)
    assert loaded.scores is not None
    np.testing.assert_array_equal(loaded.scores, scores)
    assert loaded.scales is None
    assert loaded.orientations is None


def test_keypoints_save_replaces(uow_factory: UnitOfWorkFactory) -> None:
    image_id, config_id = _image_and_config(uow_factory)
    with uow_factory() as uow:
        uow.keypoints.save(Keypoints(image_id, config_id, np.zeros((3, 2), np.float32)))
        uow.keypoints.save(Keypoints(image_id, config_id, np.ones((5, 2), np.float32)))
        uow.commit()

    with uow_factory() as uow:
        loaded = uow.keypoints.get(image_id, config_id)
    assert loaded is not None
    assert len(loaded) == 5


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.float16])
def test_descriptors_round_trip_keeps_dtype(uow_factory: UnitOfWorkFactory, dtype: type) -> None:
    image_id, config_id = _image_and_config(uow_factory)
    data: NDArray[Any] = (np.random.default_rng(1).random((4, 32)) * 255).astype(dtype)

    with uow_factory() as uow:
        assert uow.descriptors.get(image_id, config_id) is None
        uow.descriptors.save(Descriptors(image_id, config_id, data))
        uow.commit()

    with uow_factory() as uow:
        loaded = uow.descriptors.get(image_id, config_id)
    assert loaded is not None
    assert loaded.data.dtype == data.dtype
    np.testing.assert_array_equal(loaded.data, data)


def test_empty_feature_sets_round_trip(uow_factory: UnitOfWorkFactory) -> None:
    image_id, config_id = _image_and_config(uow_factory)
    with uow_factory() as uow:
        uow.keypoints.save(Keypoints(image_id, config_id, np.zeros((0, 2), np.float32)))
        uow.descriptors.save(Descriptors(image_id, config_id, np.zeros((0, 128), np.float32)))
        uow.commit()

    with uow_factory() as uow:
        kp = uow.keypoints.get(image_id, config_id)
        desc = uow.descriptors.get(image_id, config_id)
    assert kp is not None
    assert desc is not None
    assert len(kp) == 0
    assert desc.data.shape == (0, 128)
