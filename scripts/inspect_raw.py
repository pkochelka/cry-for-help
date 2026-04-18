from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_analysis import inspect_measurement, render_measurement_inspection
from raw_data import DEFAULT_DATA_ROOT, iter_raw_measurements


def choose_default_file(root: Path) -> Path:
    files = iter_raw_measurements(root)
    if not files:
        raise SystemExit(f"No raw files found under {root}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one raw AFM tear file")
    parser.add_argument("path", nargs="?", help="Path to one raw file")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Dataset root used when no path is given")
    args = parser.parse_args()

    path = Path(args.path) if args.path else choose_default_file(Path(args.root))
    inspection = inspect_measurement(path)
    print(render_measurement_inspection(inspection))


if __name__ == "__main__":
    main()
