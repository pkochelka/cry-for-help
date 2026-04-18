Keep this file minimalistic; note only surprising project-specific things.

## Project-specific environment notes
- `TRAIN_SET/` is local-only, ~4.1G, and should stay out of git.
- Raw measurement files are Bruker NanoScope AFM exports with text headers plus binary image blocks.

## Important commands
- `uv run python scripts/explore_dataset.py`

## Common mistakes
- `_1.bmp` files are previews, not the full raw measurements.
- A few files are irregular: one extra preview in `Diabetes`, one missing preview in `PGOV_Glaukom`, one missing preview in `SucheOko`.

## User preferences
- Be concise.
