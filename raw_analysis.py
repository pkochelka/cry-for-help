from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from raw_data import ChannelMetadata, DEFAULT_DATA_ROOT, RawMeasurementMetadata, get_channel_metadata, is_raw_measurement, parse_measurement_metadata, read_channel_data


@dataclass(frozen=True)
class ChannelStats:
    min: int
    max: int
    mean: float
    std: float


@dataclass(frozen=True)
class ChannelInspection:
    metadata: ChannelMetadata
    pixel_size_um_x: float | None
    pixel_size_um_y: float | None
    stats: ChannelStats


@dataclass(frozen=True)
class MeasurementInspection:
    metadata: RawMeasurementMetadata
    channels: tuple[ChannelInspection, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClassSummary:
    name: str
    raw_files: int
    preview_bmps: int
    missing_previews: tuple[str, ...]
    extra_previews: tuple[str, ...]
    top_resolutions: tuple[tuple[tuple[int, int], int], ...]
    channel_combos: tuple[tuple[tuple[str, ...], int], ...]


@dataclass(frozen=True)
class DatasetSummary:
    root: Path
    class_summaries: tuple[ClassSummary, ...]
    total_classes: int
    total_measurements: int
    total_previews: int
    instruments: tuple[tuple[str, int], ...]
    scanners: tuple[tuple[str, int], ...]
    top_channel_combos: tuple[tuple[tuple[str, ...], int], ...]
    top_resolutions: tuple[tuple[tuple[int, int], int], ...]
    top_scan_sizes_um: tuple[tuple[tuple[float | None, float | None], int], ...]
    example_measurement: RawMeasurementMetadata | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stats(values: np.ndarray) -> ChannelStats:
    return ChannelStats(
        min=int(values.min()),
        max=int(values.max()),
        mean=float(values.mean()),
        std=float(values.std()),
    )


def inspect_measurement(path: str | Path) -> MeasurementInspection:
    metadata = parse_measurement_metadata(path)
    channels: list[ChannelInspection] = []

    for channel in get_channel_metadata(path):
        values = read_channel_data(path, channel)
        pixel_x, pixel_y = channel.pixel_size_um
        channels.append(
            ChannelInspection(
                metadata=channel,
                pixel_size_um_x=pixel_x,
                pixel_size_um_y=pixel_y,
                stats=_stats(values),
            )
        )

    return MeasurementInspection(metadata=metadata, channels=tuple(channels))


def paired_preview_name(path: Path) -> str:
    return f"{path.name}_1.bmp"


def summarize_dataset(root: str | Path = DEFAULT_DATA_ROOT) -> DatasetSummary:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root}/ not found")

    measurements_by_class: dict[str, list[RawMeasurementMetadata]] = defaultdict(list)
    previews_by_class: dict[str, set[str]] = defaultdict(set)

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for entry in class_dir.iterdir():
            if entry.suffix.lower() == ".bmp":
                previews_by_class[class_dir.name].add(entry.name)
            elif is_raw_measurement(entry):
                measurements_by_class[class_dir.name].append(parse_measurement_metadata(entry))

    instruments = Counter()
    scanners = Counter()
    channel_combos = Counter()
    resolutions = Counter()
    scan_sizes = Counter()
    class_summaries: list[ClassSummary] = []

    for class_name, items in measurements_by_class.items():
        previews = previews_by_class[class_name]
        missing_previews: list[str] = []
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

        local_resolutions = Counter((item.channels[0].samples_per_line, item.channels[0].number_of_lines) for item in items if item.channels)
        local_combos = Counter(tuple(channel.name for channel in item.channels) for item in items)

        class_summaries.append(
            ClassSummary(
                name=class_name,
                raw_files=len(items),
                preview_bmps=len(previews),
                missing_previews=tuple(sorted(missing_previews)),
                extra_previews=tuple(extra_previews),
                top_resolutions=tuple(local_resolutions.most_common(5)),
                channel_combos=tuple(local_combos.most_common(3)),
            )
        )

    total_measurements = sum(len(items) for items in measurements_by_class.values())
    total_previews = sum(len(items) for items in previews_by_class.values())
    example_measurement = next(iter(next(iter(measurements_by_class.values()))), None) if measurements_by_class else None

    return DatasetSummary(
        root=root,
        class_summaries=tuple(class_summaries),
        total_classes=len(measurements_by_class),
        total_measurements=total_measurements,
        total_previews=total_previews,
        instruments=tuple(instruments.most_common()),
        scanners=tuple(scanners.most_common()),
        top_channel_combos=tuple(channel_combos.most_common(5)),
        top_resolutions=tuple(resolutions.most_common(10)),
        top_scan_sizes_um=tuple(scan_sizes.most_common(10)),
        example_measurement=example_measurement,
    )


def render_measurement_inspection(inspection: MeasurementInspection) -> str:
    lines = [
        str(inspection.metadata.path),
        f"instrument: {inspection.metadata.instrument}",
        f"scanner: {inspection.metadata.scanner_type}",
        f"acquired: {inspection.metadata.acquired_at}",
        f"scan size: {inspection.metadata.scan_size_nm}",
        f"channels: {len(inspection.channels)}",
        "",
    ]

    for channel in inspection.channels:
        meta = channel.metadata
        lines.extend(
            [
                f"[{meta.order}] {meta.name} ({meta.label})",
                f"  shape: {meta.number_of_lines} x {meta.samples_per_line}",
                f"  bytes/pixel: {meta.bytes_per_pixel}",
                f"  offset: {meta.data_offset}",
                f"  data length: {meta.data_length}",
                f"  scan size (um): {meta.scan_size_um_x} x {meta.scan_size_um_y}",
                f"  pixel size (um): {channel.pixel_size_um_x} x {channel.pixel_size_um_y}",
                f"  stats: min={channel.stats.min} max={channel.stats.max} mean={channel.stats.mean:.2f} std={channel.stats.std:.2f}",
            ]
        )
        if meta.z_scale_line:
            lines.append(f"  z-scale: {meta.z_scale_line}")
        if meta.z_offset_line:
            lines.append(f"  z-offset: {meta.z_offset_line}")
        lines.append("")

    return "\n".join(lines).rstrip()


def render_dataset_summary(summary: DatasetSummary) -> str:
    lines = [
        "Dataset summary",
        "===============",
        f"Root: {summary.root}",
        f"Classes: {summary.total_classes}",
        f"Raw measurements: {summary.total_measurements}",
        f"BMP previews: {summary.total_previews}",
        "",
    ]

    for class_summary in summary.class_summaries:
        lines.extend(
            [
                class_summary.name,
                f"  raw files: {class_summary.raw_files}",
                f"  preview bmps: {class_summary.preview_bmps}",
                f"  missing previews: {list(class_summary.missing_previews) or '-'}",
                f"  extra previews: {list(class_summary.extra_previews) or '-'}",
                f"  top resolutions: {list(class_summary.top_resolutions)}",
                f"  channel combos: {list(class_summary.channel_combos)}",
                "",
            ]
        )

    lines.extend(
        [
            "Global metadata",
            "---------------",
            f"Instruments: {dict(summary.instruments)}",
            f"Scanners: {dict(summary.scanners)}",
            f"Top channel combos: {list(summary.top_channel_combos)}",
            f"Top resolutions: {list(summary.top_resolutions)}",
            f"Top scan sizes (um): {list(summary.top_scan_sizes_um)}",
            "",
        ]
    )

    if summary.example_measurement:
        lines.extend(
            [
                "Example raw file",
                "----------------",
                str(summary.example_measurement.path),
                f"  acquired: {summary.example_measurement.acquired_at}",
                f"  instrument: {summary.example_measurement.instrument}",
                f"  scanner: {summary.example_measurement.scanner_type}",
            ]
        )
        for channel in summary.example_measurement.channels:
            lines.append(
                "  - "
                f"{channel.name} ({channel.label}): "
                f"offset={channel.data_offset}, length={channel.data_length}, "
                f"shape={channel.samples_per_line}x{channel.number_of_lines}, "
                f"bpp={channel.bytes_per_pixel}, "
                f"scan={channel.scan_size_um_x}x{channel.scan_size_um_y} um"
            )
            if channel.z_scale_line:
                lines.append(f"    z-scale: {channel.z_scale_line}")

    return "\n".join(lines).rstrip()
