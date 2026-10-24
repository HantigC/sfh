from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from sfh.domain.errors import ImageReadError
from sfh.infrastructure.features.opencv_sift import SiftExtractor
from sfh.infrastructure.io.opencv_loader import OpenCvImageLoader


def textured_image(width: int = 320, height: int = 240, seed: int = 0) -> NDArray[np.uint8]:
    """Blurred noise: plenty of corners and blobs for SIFT to find."""
    noise = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(noise, (0, 0), 2.0), dtype=np.uint8)


def test_loader_returns_rgb_and_size(tmp_path: Path) -> None:
    bgr = np.zeros((30, 40, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # red in BGR order
    path = tmp_path / "red.png"
    cv2.imwrite(str(path), bgr)
    loader = OpenCvImageLoader()

    rgb = loader.load(path)

    assert rgb.shape == (30, 40, 3)
    assert tuple(rgb[0, 0]) == (255, 0, 0)
    assert loader.size(path) == (40, 30)


def test_loader_raises_on_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "not-an-image.png"
    path.write_text("nope")

    with pytest.raises(ImageReadError):
        OpenCvImageLoader().load(path)


def test_sift_extracts_consistent_arrays() -> None:
    features = SiftExtractor(max_keypoints=500).extract(textured_image())

    n = features.xy.shape[0]
    assert 0 < n <= 500
    assert features.descriptors.shape == (n, 128)
    assert features.descriptors.dtype == np.float32
    assert features.scales is not None
    assert features.orientations is not None
    assert features.orientations.max() <= 2 * np.pi


def test_sift_on_blank_image_returns_empty_features() -> None:
    features = SiftExtractor().extract(np.zeros((64, 64, 3), dtype=np.uint8))

    assert features.xy.shape == (0, 2)
    assert features.descriptors.shape == (0, 128)


def test_sift_params_identify_the_configuration() -> None:
    assert SiftExtractor(max_keypoints=100).params != SiftExtractor(max_keypoints=200).params
