# sfh
Structure from harness: reconstruct 3D scenes from images.

```sh
uv sync --all-groups
uv run sfh dataset add church data/image-matching-challenge-2025-ongoing/train/ETs
uv run sfh extract church --extractor sift
uv run sfh dataset list
```

Results go to `sfh.db` (override with `--db` or `SFH_DB`). See
[docs/architecture.md](docs/architecture.md) for how the code is organised.
