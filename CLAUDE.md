# sfh

Reconstructs 3D scenes from images. A uv workspace with a library (`packages/sfh`) and a CLI bin
(`apps/sfh-cli`). A backend API comes later as another app.

**Read `docs/architecture.md` before adding modules.** It has the layout, the decisions, and
recipes for adding extractors, matchers, mappers, storage backends and steps.

## Rules
- Hexagonal layers: `sfh_cli → sfh.infrastructure → sfh.application → sfh.mapping → sfh.domain`.
  Import only to the right. `domain`, `mapping` and `application` never import sqlite3, cv2,
  pycolmap, torch, typer or rich.
- New pluggable components implement a Protocol from `sfh/domain/ports/` and are wired only in a
  bin's `container.py`.
- Steps in `application/` take dependencies via the constructor, report progress through
  `ProgressReporter`, raise `SfhError` subclasses, and never print.
- Changing the SQLite schema means adding a migration in `infrastructure/sqlite/schema.py`. Never
  edit an existing one.
- Test steps with the fakes in `tests/fakes.py`. Repository behaviour goes in `tests/contracts/`.

## Commands
- `uv sync --all-groups`: install everything
- `uv run pytest`: tests
- `uv run mypy`: strict type check
- `uv run lint-imports`: architecture contracts
- `uv run ruff check --fix && uv run ruff format`
- `uv run sfh --help`: the CLI
