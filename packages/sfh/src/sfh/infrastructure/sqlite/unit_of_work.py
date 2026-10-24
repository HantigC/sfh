import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

from sfh.infrastructure.sqlite.repositories import (
    SqliteDatasetRepository,
    SqliteDescriptorRepository,
    SqliteFeatureConfigRepository,
    SqliteImageRepository,
    SqliteKeypointsRepository,
)
from sfh.infrastructure.sqlite.schema import migrate


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) and migrate a database. Use ":memory:" for tests."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


class SqliteUnitOfWork:
    """One transaction on a shared connection; the connection's owner closes it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.datasets = SqliteDatasetRepository(conn)
        self.images = SqliteImageRepository(conn)
        self.configs = SqliteFeatureConfigRepository(conn)
        self.keypoints = SqliteKeypointsRepository(conn)
        self.descriptors = SqliteDescriptorRepository(conn)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # No-op after commit(); otherwise discards whatever this unit of work wrote.
        self._conn.rollback()

    def commit(self) -> None:
        self._conn.commit()
