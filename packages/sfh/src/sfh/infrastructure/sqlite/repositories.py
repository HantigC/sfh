import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sfh.domain.entities import Dataset, Descriptors, FeatureConfig, Image, Keypoints, Params
from sfh.domain.errors import ImageNotFoundError
from sfh.domain.ids import DatasetId, FeatureConfigId, ImageId
from sfh.infrastructure.sqlite.arrays import dtype_name, from_blob, to_blob


class SqliteDatasetRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, name: str) -> Dataset:
        cur = self._conn.execute(
            "INSERT INTO datasets (name, created_at) VALUES (?, ?)",
            (name, datetime.now(UTC).isoformat()),
        )
        assert cur.lastrowid is not None
        return Dataset(DatasetId(cur.lastrowid), name)

    def get_by_name(self, name: str) -> Dataset | None:
        row = self._conn.execute("SELECT id, name FROM datasets WHERE name = ?", (name,)).fetchone()
        return None if row is None else Dataset(DatasetId(row["id"]), row["name"])

    def list(self) -> Sequence[Dataset]:
        rows = self._conn.execute("SELECT id, name FROM datasets ORDER BY name").fetchall()
        return [Dataset(DatasetId(r["id"]), r["name"]) for r in rows]


def _image(row: sqlite3.Row) -> Image:
    return Image(
        id=ImageId(row["id"]),
        dataset_id=DatasetId(row["dataset_id"]),
        name=row["name"],
        path=Path(row["path"]),
        width=row["width"],
        height=row["height"],
    )


class SqliteImageRepository:
    _COLUMNS = "id, dataset_id, name, path, width, height"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, dataset_id: DatasetId, name: str, path: Path, width: int, height: int) -> Image:
        cur = self._conn.execute(
            "INSERT INTO images (dataset_id, name, path, width, height) VALUES (?, ?, ?, ?, ?)",
            (dataset_id, name, str(path), width, height),
        )
        assert cur.lastrowid is not None
        return Image(ImageId(cur.lastrowid), dataset_id, name, path, width, height)

    def get(self, image_id: ImageId) -> Image:
        row = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            raise ImageNotFoundError(image_id)
        return _image(row)

    def list_by_dataset(self, dataset_id: DatasetId) -> Sequence[Image]:
        rows = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM images WHERE dataset_id = ? ORDER BY name",
            (dataset_id,),
        ).fetchall()
        return [_image(r) for r in rows]


def params_hash(name: str, params: Params) -> str:
    canonical = json.dumps({"name": name, "params": dict(params)}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class SqliteFeatureConfigRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(self, name: str, params: Params) -> FeatureConfig:
        digest = params_hash(name, params)
        self._conn.execute(
            "INSERT INTO feature_configs (name, params_json, params_hash) VALUES (?, ?, ?)"
            " ON CONFLICT (params_hash) DO NOTHING",
            (name, json.dumps(dict(params), sort_keys=True), digest),
        )
        row = self._conn.execute(
            "SELECT id, name, params_json FROM feature_configs WHERE params_hash = ?", (digest,)
        ).fetchone()
        return FeatureConfig(
            FeatureConfigId(row["id"]), row["name"], json.loads(row["params_json"])
        )


def _optional_blob(values: NDArray[np.float32] | None) -> bytes | None:
    return None if values is None else to_blob(values, np.float32)


def _optional_array(blob: bytes | None, n: int) -> NDArray[np.float32] | None:
    return None if blob is None else from_blob(blob, np.float32, (n,))


class SqliteKeypointsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, keypoints: Keypoints) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO keypoints"
            " (image_id, config_id, n, xy, scores, scales, orientations)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                keypoints.image_id,
                keypoints.config_id,
                len(keypoints),
                to_blob(keypoints.xy, np.float32),
                _optional_blob(keypoints.scores),
                _optional_blob(keypoints.scales),
                _optional_blob(keypoints.orientations),
            ),
        )

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Keypoints | None:
        row = self._conn.execute(
            "SELECT n, xy, scores, scales, orientations FROM keypoints"
            " WHERE image_id = ? AND config_id = ?",
            (image_id, config_id),
        ).fetchone()
        if row is None:
            return None
        n = row["n"]
        return Keypoints(
            image_id=image_id,
            config_id=config_id,
            xy=from_blob(row["xy"], np.float32, (n, 2)),
            scores=_optional_array(row["scores"], n),
            scales=_optional_array(row["scales"], n),
            orientations=_optional_array(row["orientations"], n),
        )

    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM keypoints WHERE image_id = ? AND config_id = ?", (image_id, config_id)
        ).fetchone()
        return row is not None


class SqliteDescriptorRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, descriptors: Descriptors) -> None:
        rows, cols = descriptors.data.shape
        self._conn.execute(
            "INSERT OR REPLACE INTO descriptors (image_id, config_id, rows, cols, dtype, data)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                descriptors.image_id,
                descriptors.config_id,
                rows,
                cols,
                dtype_name(descriptors.data),
                to_blob(descriptors.data),
            ),
        )

    def get(self, image_id: ImageId, config_id: FeatureConfigId) -> Descriptors | None:
        row = self._conn.execute(
            "SELECT rows, cols, dtype, data FROM descriptors WHERE image_id = ? AND config_id = ?",
            (image_id, config_id),
        ).fetchone()
        if row is None:
            return None
        data = from_blob(row["data"], np.dtype(row["dtype"]), (row["rows"], row["cols"]))
        return Descriptors(image_id, config_id, data)

    def exists(self, image_id: ImageId, config_id: FeatureConfigId) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM descriptors WHERE image_id = ? AND config_id = ?",
            (image_id, config_id),
        ).fetchone()
        return row is not None
