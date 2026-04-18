# hack-kosice-tears

Initial exploration repo for the crystallized-tears train set.

Reusable raw-data helpers live in `raw_data.py`.
Reusable higher-level summaries/inspection objects live in `raw_analysis.py`.
Runnable exploration scripts live in `scripts/`.

## Setup

```bash
uv venv
uv run python scripts/explore_dataset.py
uv run python scripts/inspect_raw.py
```

## Data

The local dataset lives in `data/` and is intentionally ignored by git because it is large (~4.1G) and likely sensitive.

Folder labels in the train set:
- `Diabetes`
- `PGOV_Glaukom`
- `SklerózaMultiplex`
- `SucheOko`
- `ZdraviLudia`

## Python usage

```python
from raw_analysis import inspect_measurement, summarize_dataset

measurement = inspect_measurement("data/Diabetes/37_DM.010")
summary = summarize_dataset("data")

measurement.channels[0].stats
summary.class_summaries[0]
```

## Raw files

The non-BMP files are Bruker NanoScope AFM measurement exports. Each raw file contains:
- a text header with acquisition metadata
- multiple embedded image channels
- binary raster data blocks referenced by `Data offset` / `Data length`

The paired `*_1.bmp` files are preview renders exported by the instrument/software.
