from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from sfh.infrastructure.sqlite.arrays import dtype_name, from_blob, to_blob


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(12, dtype=np.uint8).reshape(4, 3),
        np.arange(6, dtype=np.float64),
        np.zeros((0, 128), dtype=np.float32),
    ],
)
def test_round_trip(array: NDArray[Any]) -> None:
    restored = from_blob(to_blob(array), np.dtype(dtype_name(array)), array.shape)

    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)
    assert restored.flags.writeable


def test_non_contiguous_input_is_stored_in_c_order() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4).T  # Fortran-ordered view

    restored = from_blob(to_blob(array), np.float32, array.shape)

    np.testing.assert_array_equal(restored, array)


def test_cast_on_write() -> None:
    restored = from_blob(to_blob(np.array([1.5, 2.5]), np.float32), np.float32, (2,))

    assert restored.dtype == np.float32
