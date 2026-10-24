import sqlite3

# Bump when the schema changes and add the migration to MIGRATIONS.
SCHEMA_VERSION = 1

_V1 = """
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    UNIQUE (dataset_id, name)
);

CREATE TABLE feature_configs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    params_json TEXT NOT NULL,
    params_hash TEXT NOT NULL UNIQUE
);

-- Arrays are stored as raw bytes; shape and dtype live in the neighbouring columns.
CREATE TABLE keypoints (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    config_id INTEGER NOT NULL REFERENCES feature_configs(id),
    n INTEGER NOT NULL,
    xy BLOB NOT NULL,           -- float32 (n, 2)
    scores BLOB,                -- float32 (n,)
    scales BLOB,                -- float32 (n,)
    orientations BLOB,          -- float32 (n,)
    PRIMARY KEY (image_id, config_id)
);

CREATE TABLE descriptors (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    config_id INTEGER NOT NULL REFERENCES feature_configs(id),
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (image_id, config_id)
);
"""

MIGRATIONS: dict[int, str] = {1: _V1}


def migrate(conn: sqlite3.Connection) -> None:
    """Bring the database up to SCHEMA_VERSION, tracked in PRAGMA user_version."""
    (current,) = conn.execute("PRAGMA user_version").fetchone()
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema v{current} is newer than this sfh (v{SCHEMA_VERSION})")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(
            f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version};\nCOMMIT;"
        )
