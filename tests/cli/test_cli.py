import sqlite3
from pathlib import Path

import cv2
import pytest
from typer.testing import CliRunner

from sfh_cli.main import app
from tests.infrastructure.test_opencv_adapters import textured_image

runner = CliRunner()


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for i in range(3):
        rgb = textured_image(seed=i)
        cv2.imwrite(str(directory / f"img{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (directory / "notes.txt").write_text("ignored")
    return directory


def test_register_then_extract_end_to_end(tmp_path: Path, image_dir: Path) -> None:
    db = str(tmp_path / "sfh.db")

    added = runner.invoke(app, ["dataset", "add", "church", str(image_dir), "--db", db])
    assert added.exit_code == 0, added.output
    assert "3 images" in added.output

    listed = runner.invoke(app, ["dataset", "list", "--db", db])
    assert "church\t3 images" in listed.output

    first = runner.invoke(app, ["extract", "church", "--db", db])
    assert first.exit_code == 0, first.output
    assert "extracted 3, skipped 0" in first.output

    second = runner.invoke(app, ["extract", "church", "--db", db])
    assert "extracted 0, skipped 3" in second.output

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM keypoints").fetchone() == (3,)
        assert conn.execute("SELECT count(*) FROM descriptors").fetchone() == (3,)


def test_domain_errors_become_exit_code_1(tmp_path: Path, image_dir: Path) -> None:
    db = str(tmp_path / "sfh.db")

    result = runner.invoke(app, ["extract", "missing", "--db", db])

    assert result.exit_code == 1
    assert "dataset not found" in result.output


def test_unknown_extractor_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["extract", "x", "--extractor", "nope", "--db", str(tmp_path / "d")]
    )

    assert result.exit_code == 2
