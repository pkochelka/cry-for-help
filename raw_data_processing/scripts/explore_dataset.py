from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_data_processing.raw_analysis import render_dataset_summary, summarize_dataset
from raw_data_processing.raw_data import DEFAULT_DATA_ROOT

ROOT = DEFAULT_DATA_ROOT


def main() -> None:
    summary = summarize_dataset(ROOT)
    print(render_dataset_summary(summary))


if __name__ == "__main__":
    main()
