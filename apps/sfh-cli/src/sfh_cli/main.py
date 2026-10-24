from pathlib import Path
from typing import Annotated

import typer

from sfh.domain.errors import SfhError
from sfh_cli.container import EXTRACTORS, open_container
from sfh_cli.progress import RichProgress

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

app = typer.Typer(help="Reconstruct 3D scenes from images.", no_args_is_help=True)
dataset_app = typer.Typer(help="Manage datasets.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

DbOption = Annotated[
    Path, typer.Option("--db", envvar="SFH_DB", help="SQLite database file.", dir_okay=False)
]
DEFAULT_DB = Path("sfh.db")


def _fail(error: SfhError) -> typer.Exit:
    typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@dataset_app.command("add")
def dataset_add(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    directory: Annotated[
        Path, typer.Argument(help="Folder with the images.", exists=True, file_okay=False)
    ],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Register every image in DIRECTORY as dataset NAME."""
    paths = [p.resolve() for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        typer.secho(f"no images found in {directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        with open_container(db) as c, RichProgress("Registering images") as progress:
            result = c.register_dataset(name, paths, progress)
    except SfhError as e:
        raise _fail(e) from e
    typer.echo(f"dataset {result.dataset.name!r}: {result.images_added} images")


@dataset_app.command("list")
def dataset_list(db: DbOption = DEFAULT_DB) -> None:
    """List datasets and their image counts."""
    with open_container(db) as c:
        for summary in c.list_datasets():
            typer.echo(f"{summary.dataset.name}\t{summary.image_count} images")


@app.command()
def extract(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    extractor: Annotated[
        str, typer.Option(help=f"One of: {', '.join(sorted(EXTRACTORS))}.")
    ] = "sift",
    db: DbOption = DEFAULT_DB,
) -> None:
    """Extract keypoints and descriptors for every image of DATASET."""
    if extractor not in EXTRACTORS:
        typer.secho(f"unknown extractor {extractor!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        with open_container(db, extractor) as c, RichProgress("Extracting features") as progress:
            result = c.extract_features(dataset, progress)
    except SfhError as e:
        raise _fail(e) from e
    typer.echo(
        f"{result.config.name}: extracted {result.extracted}, "
        f"skipped {result.skipped} (already done)"
    )
