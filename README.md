# hack-kosice-tears

Initial exploration repo for the crystallized-tears train set.

## Setup

```bash
uv venv
uv run python scripts/explore_dataset.py
```

## Data

The local dataset lives in `TRAIN_SET/` and is intentionally ignored by git because it is large (~4.1G) and likely sensitive.

Folder labels in the train set:
- `Diabetes`
- `PGOV_Glaukom`
- `SklerózaMultiplex`
- `SucheOko`
- `ZdraviLudia`

## Raw files

The non-BMP files are Bruker NanoScope AFM measurement exports. Each raw file contains:
- a text header with acquisition metadata
- multiple embedded image channels
- binary raster data blocks referenced by `Data offset` / `Data length`

The paired `*_1.bmp` files are preview renders exported by the instrument/software.
