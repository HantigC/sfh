# sfh architecture

sfh reconstructs 3D scenes from images. It is built as a **hexagonal (ports & adapters)**
architecture: the core (domain, reconstruction strategies, pipeline steps) knows nothing about
SQLite, OpenCV, pycolmap or torch. Those are adapters plugged in from the outside.

## Layout

```
packages/sfh/src/sfh/        LIB, installable on its own
├── domain/                  entities + ports. stdlib and numpy only
│   ├── ids.py               DatasetId, ImageId, FeatureConfigId (NewType over int)
│   ├── entities.py          Dataset, Image, FeatureConfig, Keypoints, Descriptors, ...
│   ├── errors.py            SfhError and subclasses. Entrypoints map these to exit codes / HTTP
│   └── ports/               Protocols the adapters implement
│       ├── repositories.py  storage, one repository per aggregate
│       ├── unit_of_work.py  transaction boundary over all repositories
│       ├── features.py      ImageLoader, FeatureExtractor, Matcher, PairSelector
│       └── progress.py      ProgressReporter (+ NullProgress)
├── mapping/                 reconstruction strategies implementing the Mapper port
├── application/             use cases = pipeline steps. Every entrypoint calls these
└── infrastructure/          adapters: the only code that imports sqlite3/cv2/pycolmap/torch
    ├── sqlite/              repositories, unit of work, schema + migrations, numpy<->BLOB
    ├── features/            SiftExtractor, ...
    └── io/                  OpenCvImageLoader

apps/sfh-cli/src/sfh_cli/    BIN
├── container.py             composition root: the only place that picks concrete adapters
├── progress.py              RichProgress (a ProgressReporter)
└── main.py                  typer commands. Parse input, call a step, print the result
```

A future backend is `apps/sfh-api/` with its own `container.py`. It reuses the lib unchanged.

## The dependency rule

```
sfh_cli  →  sfh.infrastructure  →  sfh.application  →  sfh.mapping  →  sfh.domain
```

A module may import only from layers to its right. On top of that:

- `domain`, `mapping` and `application` must not import `sqlite3`, `cv2`, `pycolmap`, `torch`,
  `typer` or `rich`.
- Adapters are independent of each other: `infrastructure.sqlite`, `.features` and `.io` don't
  import one another.

These rules are import-linter contracts in the root `pyproject.toml`. They run in pre-commit and
in the Claude Code hook (`.claude/hooks/py-quality.sh`), so a violation fails immediately. Run them
by hand with `uv run lint-imports`.

## Key decisions

- **Dataset** is a named group of images. An image belongs to exactly one dataset.
- **Images are stored as path + size.** The pixels stay on disk.
- **Arrays are SQLite BLOBs** (raw bytes plus shape/dtype columns, like COLMAP's database).
  Keypoint arrays are always float32. Descriptors keep their dtype (float32 for SIFT and learned
  methods, uint8 for binary ones).
- **Results are keyed by `FeatureConfig`** = (extractor name, params). Several methods coexist for
  one image, and a step skips work that already exists for its config. This is what makes steps
  resumable and lets you compare methods.
- **One unit of work per image** in long steps: an interrupted run keeps what it finished.
- **Steps are entrypoint-agnostic.** They take dependencies in the constructor, input as
  arguments, report progress through `ProgressReporter`, signal problems with `SfhError`, and
  return plain result objects. No printing, no CLI or HTTP types.
- **Schema migrations**: `infrastructure/sqlite/schema.py` tracks the version in
  `PRAGMA user_version`. To change the schema, add the next entry to `MIGRATIONS` and bump
  `SCHEMA_VERSION`. Never edit an existing migration.

## Pipeline

```
dataset add → extract → match (PairSelector + Matcher) → verify geometry → reconstruct → export
  done          done        next                            next              next       next
```

Reconstruction (the next design round): strategies in `sfh.mapping` (incremental, your own)
only orchestrate: initial pair, next view, when to triangulate and when to run bundle
adjustment. The math goes through a `GeometryKernel` port that `infrastructure/colmap/`
implements with pycolmap. COLMAP's own incremental mapper is wrapped behind the same `Mapper`
port as a baseline.

## Recipes

### Add a feature extractor (or matcher, pair selector, mapper)
1. Implement the Protocol from `domain/ports/` in `infrastructure/features/<name>.py`. Strategies
   that are pure orchestration go in `mapping/` instead.
2. Put **every** setting that changes the output into `params`. It keys the stored results.
3. Register it in each bin's container, e.g. `EXTRACTORS` in `apps/sfh-cli/src/sfh_cli/container.py`.
4. Test it in `tests/infrastructure/`. Never import it from `application/`.

### Add a storage backend
1. Implement the repository Protocols and a `UnitOfWork` in `infrastructure/<backend>/`.
2. Add it as a param of the `uow_factory` fixture in `tests/contracts/conftest.py`. The shared
   suite in `tests/contracts/` then runs against it.
3. Choose it in the container.

### Add a pipeline step
1. Add a class in `application/` that depends only on ports (`UnitOfWorkFactory`, `ImageLoader`, ...).
2. Test it with the in-memory fakes in `tests/fakes.py`.
3. Wire it in `Container`, then add the CLI command (and later the API route) as a thin wrapper.

### Add a new entity that needs storage
1. Add the entity to `domain/entities.py` and a repository Protocol to `domain/ports/repositories.py`.
2. Expose it on `UnitOfWork`, then implement it in SQLite (plus a migration) and in `tests/fakes.py`.
3. Add contract tests in `tests/contracts/`.

## Tests

| Folder                  | What it covers                                                  |
|-------------------------|-----------------------------------------------------------------|
| `tests/contracts/`      | repository behaviour, run against every backend and the fakes   |
| `tests/application/`    | steps, with fakes only (fast, no I/O)                           |
| `tests/infrastructure/` | individual adapters (BLOB round trips, OpenCV loader, SIFT)     |
| `tests/cli/`            | end to end through the CLI on generated images                  |
