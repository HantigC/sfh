from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray


def to_blob(array: NDArray[Any], dtype: DTypeLike | None = None) -> bytes:
    """C-ordered raw bytes of `array`, cast to `dtype` first when given."""
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def from_blob(blob: bytes, dtype: DTypeLike, shape: tuple[int, ...]) -> NDArray[Any]:
    """Inverse of `to_blob`. The result is a writable copy, not a view on `blob`."""
    return np.frombuffer(blob, dtype=dtype).reshape(shape).copy()


def dtype_name(array: NDArray[Any]) -> str:
    """Byte-order explicit dtype string, e.g. '<f4', so blobs stay portable."""
    return array.dtype.str
