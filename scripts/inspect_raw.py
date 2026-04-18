from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_data import DEFAULT_DATA_ROOT, get_channel_metadata, iter_raw_measurements, parse_measurement_metadata, read_channel_data


def format_stats(values: np.ndarray) -> str:
    return (
        f"min={values.min()} max={values.max()} "
        f"mean={values.mean():.2f} std={values.std():.2f}"
    )


def inspect_file(path: Path) -> None:
    metadata = parse_measurement_metadata(path)
    print(path)
    print(f"instrument: {metadata.instrument}")
    print(f"scanner: {metadata.scanner_type}")
    print(f"acquired: {metadata.acquired_at}")
    print(f"scan size: {metadata.scan_size_nm}")
    print(f"channels: {len(metadata.channels)}")
    print()

    for channel in get_channel_metadata(path):
        values = read_channel_data(path, channel)
        pixel_x, pixel_y = channel.pixel_size_um
        print(f"[{channel.order}] {channel.name} ({channel.label})")
        print(f"  shape: {channel.number_of_lines} x {channel.samples_per_line}")
        print(f"  bytes/pixel: {channel.bytes_per_pixel}")
        print(f"  offset: {channel.data_offset}")
        print(f"  data length: {channel.data_length}")
        print(f"  scan size (um): {channel.scan_size_um_x} x {channel.scan_size_um_y}")
        print(f"  pixel size (um): {pixel_x} x {pixel_y}")
        print(f"  stats: {format_stats(values)}")
        if channel.z_scale_line:
            print(f"  z-scale: {channel.z_scale_line}")
        if channel.z_offset_line:
            print(f"  z-offset: {channel.z_offset_line}")
        print()


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
    inspect_file(path)


if __name__ == "__main__":
    main()
