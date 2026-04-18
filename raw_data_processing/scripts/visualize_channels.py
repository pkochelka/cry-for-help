from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_data_processing.raw_data import DEFAULT_DATA_ROOT, iter_raw_measurements
from raw_data_processing.raw_visualization import save_channel_figure


def choose_default_file(root: Path) -> Path:
    files = iter_raw_measurements(root)
    if not files:
        raise SystemExit(f"No raw files found under {root}")
    return files[0]


def default_output_path(raw_path: Path) -> Path:
    safe_name = raw_path.as_posix().replace("/", "__")
    return Path("reports") / f"{safe_name}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RGB preview + raw channels into one figure")
    parser.add_argument("path", nargs="?", help="Path to one raw file")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Dataset root used when no path is given")
    parser.add_argument("--out", help="Output PNG path")
    args = parser.parse_args()

    raw_path = Path(args.path) if args.path else choose_default_file(Path(args.root))
    output_path = Path(args.out) if args.out else default_output_path(raw_path)
    saved = save_channel_figure(raw_path, output_path)
    print(saved)


if __name__ == "__main__":
    main()
