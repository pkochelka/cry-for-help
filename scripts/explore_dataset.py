from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_data import DEFAULT_DATA_ROOT, RawMeasurementMetadata, is_raw_measurement, parse_measurement_metadata

ROOT = DEFAULT_DATA_ROOT


def paired_preview_name(path: Path) -> str:
    return f"{path.name}_1.bmp"


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"{ROOT}/ not found")

    measurements_by_class: dict[str, list[RawMeasurementMetadata]] = defaultdict(list)
    previews_by_class: dict[str, set[str]] = defaultdict(set)

    for class_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        for entry in class_dir.iterdir():
            if entry.suffix.lower() == ".bmp":
                previews_by_class[class_dir.name].add(entry.name)
            elif is_raw_measurement(entry):
                measurements_by_class[class_dir.name].append(parse_measurement_metadata(entry))

    total_measurements = sum(len(items) for items in measurements_by_class.values())
    total_previews = sum(len(items) for items in previews_by_class.values())

    print("Dataset summary")
    print("===============")
    print(f"Classes: {len(measurements_by_class)}")
    print(f"Raw measurements: {total_measurements}")
    print(f"BMP previews: {total_previews}")
    print()

    channel_combos = Counter()
    resolutions = Counter()
    scan_sizes = Counter()
    instruments = Counter()
    scanners = Counter()

    for class_name, items in measurements_by_class.items():
        missing_previews: list[str] = []
        previews = previews_by_class[class_name]
        extra_previews = sorted(name for name in previews if not any(name == paired_preview_name(item.path) for item in items))

        for item in items:
            if paired_preview_name(item.path) not in previews:
                missing_previews.append(item.path.name)
            instruments[item.instrument or "?"] += 1
            scanners[item.scanner_type or "?"] += 1
            combo = tuple(channel.name for channel in item.channels)
            channel_combos[combo] += 1
            if item.channels:
                resolutions[(item.channels[0].samples_per_line, item.channels[0].number_of_lines)] += 1
                scan_sizes[(item.channels[0].scan_size_um_x, item.channels[0].scan_size_um_y)] += 1

        print(class_name)
        print(f"  raw files: {len(items)}")
        print(f"  preview bmps: {len(previews)}")
        print(f"  missing previews: {missing_previews or '-'}")
        print(f"  extra previews: {extra_previews or '-'}")
        local_resolutions = Counter((item.channels[0].samples_per_line, item.channels[0].number_of_lines) for item in items if item.channels)
        print(f"  top resolutions: {local_resolutions.most_common(5)}")
        local_combos = Counter(tuple(channel.name for channel in item.channels) for item in items)
        print(f"  channel combos: {local_combos.most_common(3)}")
        print()

    print("Global metadata")
    print("---------------")
    print(f"Instruments: {dict(instruments)}")
    print(f"Scanners: {dict(scanners)}")
    print(f"Top channel combos: {channel_combos.most_common(5)}")
    print(f"Top resolutions: {resolutions.most_common(10)}")
    print(f"Top scan sizes (um): {scan_sizes.most_common(10)}")
    print()

    first = next(iter(next(iter(measurements_by_class.values()))), None)
    if first:
        print("Example raw file")
        print("----------------")
        print(first.path)
        print(f"  acquired: {first.acquired_at}")
        print(f"  instrument: {first.instrument}")
        print(f"  scanner: {first.scanner_type}")
        for channel in first.channels:
            print(
                "  - "
                f"{channel.name} ({channel.label}): "
                f"offset={channel.data_offset}, length={channel.data_length}, "
                f"shape={channel.samples_per_line}x{channel.number_of_lines}, "
                f"bpp={channel.bytes_per_pixel}, "
                f"scan={channel.scan_size_um_x}x{channel.scan_size_um_y} um"
            )
            if channel.z_scale_line:
                print(f"    z-scale: {channel.z_scale_line}")


if __name__ == "__main__":
    main()
