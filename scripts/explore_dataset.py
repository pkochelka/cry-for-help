from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("TRAIN_SET")
IGNORE = {"NsThumbnails.bin", "Thumbs.db"}
HEADER_BYTES = 40960


@dataclass(frozen=True)
class ImageChannel:
    name: str
    label: str
    data_offset: int
    data_length: int
    bytes_per_pixel: int
    samples_per_line: int
    number_of_lines: int
    scan_size_um_x: float | None
    scan_size_um_y: float | None
    line_direction: str | None
    frame_direction: str | None
    z_scale_line: str | None


@dataclass(frozen=True)
class RawMeasurement:
    path: Path
    acquired_at: str | None
    instrument: str | None
    scanner_type: str | None
    scan_size_nm: str | None
    channels: tuple[ImageChannel, ...]


def extract(pattern: str, text: str, cast=str):
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1)
    return cast(value) if cast is not str else value.strip()


def is_raw_measurement(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() == ".bmp" or path.name in IGNORE:
        return False
    with path.open("rb") as fh:
        return fh.read(20).startswith(b"\\*File list")


def parse_measurement(path: Path) -> RawMeasurement:
    header = path.read_bytes()[:HEADER_BYTES].decode("latin1", errors="ignore")
    blocks = header.split("\\*Ciao image list")[1:]

    channels: list[ImageChannel] = []
    for block in blocks:
        image_match = re.search(r'\\@(\d+):Image Data: S \[([^\]]+)\] "([^"]+)"', block)
        if not image_match:
            continue
        scan_match = re.search(r"\\Scan Size: ([0-9.]+) ([0-9.]+) ~m", block)
        scan_x = float(scan_match.group(1)) if scan_match else None
        scan_y = float(scan_match.group(2)) if scan_match else None
        z_scale_match = re.search(r"\\@\d+:Z scale: ([^\r\n]+)", block)
        channels.append(
            ImageChannel(
                name=image_match.group(2),
                label=image_match.group(3),
                data_offset=extract(r"\\Data offset: ([0-9]+)", block, int) or 0,
                data_length=extract(r"\\Data length: ([0-9]+)", block, int) or 0,
                bytes_per_pixel=extract(r"\\Bytes/pixel: ([0-9]+)", block, int) or 0,
                samples_per_line=extract(r"\\Samps/line: ([0-9]+)", block, int) or 0,
                number_of_lines=extract(r"\\Number of lines: ([0-9]+)", block, int) or 0,
                scan_size_um_x=scan_x,
                scan_size_um_y=scan_y,
                line_direction=extract(r"\\Line Direction: ([^\r\n]+)", block),
                frame_direction=extract(r"\\Frame direction: ([^\r\n]+)", block),
                z_scale_line=z_scale_match.group(1).strip() if z_scale_match else None,
            )
        )

    return RawMeasurement(
        path=path,
        acquired_at=extract(r"\\Date: ([^\r\n]+)", header),
        instrument=extract(r"\\Description: ([^\r\n]+)", header),
        scanner_type=extract(r"\\Scanner type: ([^\r\n]+)", header),
        scan_size_nm=extract(r"\\Scan Size: ([^\r\n]+)", header),
        channels=tuple(channels),
    )


def paired_preview_name(path: Path) -> str:
    return f"{path.name}_1.bmp"


def main() -> None:
    if not ROOT.exists():
        raise SystemExit("TRAIN_SET/ not found")

    measurements_by_class: dict[str, list[RawMeasurement]] = defaultdict(list)
    previews_by_class: dict[str, set[str]] = defaultdict(set)

    for class_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        for entry in class_dir.iterdir():
            if entry.suffix.lower() == ".bmp":
                previews_by_class[class_dir.name].add(entry.name)
            elif is_raw_measurement(entry):
                measurements_by_class[class_dir.name].append(parse_measurement(entry))

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
