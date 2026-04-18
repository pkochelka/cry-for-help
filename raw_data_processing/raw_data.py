from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLike: TypeAlias = str | Path
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_HEADER_BYTES = 40960
IGNORE_FILENAMES = {"NsThumbnails.bin", "Thumbs.db"}
_DTYPE_BY_BPP = {
    2: np.dtype("<i2"),
    4: np.dtype("<i4"),
}


@dataclass(frozen=True)
class ChannelMetadata:
    order: int
    ciao_id: int
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
    z_offset_line: str | None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.number_of_lines, self.samples_per_line)

    @property
    def dtype(self) -> np.dtype:
        try:
            return _DTYPE_BY_BPP[self.bytes_per_pixel]
        except KeyError as exc:
            raise ValueError(f"Unsupported bytes/pixel: {self.bytes_per_pixel}") from exc

    @property
    def pixel_size_um(self) -> tuple[float | None, float | None]:
        if not self.scan_size_um_x or not self.scan_size_um_y:
            return (None, None)
        return (
            self.scan_size_um_x / self.samples_per_line,
            self.scan_size_um_y / self.number_of_lines,
        )


@dataclass(frozen=True)
class RawMeasurementMetadata:
    path: Path
    acquired_at: str | None
    instrument: str | None
    scanner_type: str | None
    scan_size_nm: str | None
    header_entries: dict[str, str]
    channels: tuple[ChannelMetadata, ...]


def _as_path(path: PathLike) -> Path:
    return Path(path)


def _extract(pattern: str, text: str, cast=str):
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1)
    return cast(value) if cast is not str else value.strip()


def is_raw_measurement(path: PathLike) -> bool:
    candidate = _as_path(path)
    if not candidate.is_file() or candidate.suffix.lower() == ".bmp" or candidate.name in IGNORE_FILENAMES:
        return False
    with candidate.open("rb") as fh:
        return fh.read(20).startswith(b"\\*File list")


def iter_raw_measurements(root: PathLike = DEFAULT_DATA_ROOT) -> list[Path]:
    root_path = _as_path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if is_raw_measurement(path))


def read_header_text(path: PathLike, header_bytes: int = DEFAULT_HEADER_BYTES) -> str:
    raw_path = _as_path(path)
    with raw_path.open("rb") as fh:
        return fh.read(header_bytes).decode("latin1", errors="ignore")


def parse_header_entries(header_text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line.startswith("\\") or line.startswith("\\*") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        entries.setdefault(key.strip(), value.strip())
    return entries


def get_header_entries(path: PathLike, header_bytes: int = DEFAULT_HEADER_BYTES) -> dict[str, str]:
    return parse_header_entries(read_header_text(path, header_bytes=header_bytes))


def parse_measurement_metadata(path: PathLike, header_bytes: int = DEFAULT_HEADER_BYTES) -> RawMeasurementMetadata:
    raw_path = _as_path(path)
    header_text = read_header_text(raw_path, header_bytes=header_bytes)
    header_entries = parse_header_entries(header_text)
    blocks = header_text.split("\\*Ciao image list")[1:]

    channels: list[ChannelMetadata] = []
    for order, block in enumerate(blocks, start=1):
        image_match = re.search(r'\\@(\d+):Image Data: S \[([^\]]+)\] "([^"]+)"', block)
        if not image_match:
            continue

        scan_match = re.search(r"\\Scan Size: ([0-9.]+) ([0-9.]+) ~m", block)
        scan_x = float(scan_match.group(1)) if scan_match else None
        scan_y = float(scan_match.group(2)) if scan_match else None
        ciao_id = int(image_match.group(1))

        z_scale_match = re.search(rf"\\@{ciao_id}:Z scale: ([^\r\n]+)", block)
        z_offset_match = re.search(rf"\\@{ciao_id}:Z offset: ([^\r\n]+)", block)

        channels.append(
            ChannelMetadata(
                order=order,
                ciao_id=ciao_id,
                name=image_match.group(2),
                label=image_match.group(3),
                data_offset=_extract(r"\\Data offset: ([0-9]+)", block, int) or 0,
                data_length=_extract(r"\\Data length: ([0-9]+)", block, int) or 0,
                bytes_per_pixel=_extract(r"\\Bytes/pixel: ([0-9]+)", block, int) or 0,
                samples_per_line=_extract(r"\\Samps/line: ([0-9]+)", block, int) or 0,
                number_of_lines=_extract(r"\\Number of lines: ([0-9]+)", block, int) or 0,
                scan_size_um_x=scan_x,
                scan_size_um_y=scan_y,
                line_direction=_extract(r"\\Line Direction: ([^\r\n]+)", block),
                frame_direction=_extract(r"\\Frame direction: ([^\r\n]+)", block),
                z_scale_line=z_scale_match.group(1).strip() if z_scale_match else None,
                z_offset_line=z_offset_match.group(1).strip() if z_offset_match else None,
            )
        )

    return RawMeasurementMetadata(
        path=raw_path,
        acquired_at=header_entries.get("Date"),
        instrument=header_entries.get("Description"),
        scanner_type=header_entries.get("Scanner type"),
        scan_size_nm=header_entries.get("Scan Size"),
        header_entries=header_entries,
        channels=tuple(channels),
    )


def get_channel_metadata(path: PathLike) -> tuple[ChannelMetadata, ...]:
    return parse_measurement_metadata(path).channels


def read_channel_data(path: PathLike, channel: ChannelMetadata) -> NDArray[np.signedinteger]:
    raw_path = _as_path(path)
    expected_items = channel.samples_per_line * channel.number_of_lines
    expected_bytes = expected_items * channel.bytes_per_pixel
    if expected_bytes != channel.data_length:
        raise ValueError(
            f"Channel byte size mismatch for {raw_path}: "
            f"expected {expected_bytes}, metadata says {channel.data_length}"
        )

    with raw_path.open("rb") as fh:
        fh.seek(channel.data_offset)
        block = fh.read(channel.data_length)

    if len(block) != channel.data_length:
        raise ValueError(
            f"Short read for {raw_path}: wanted {channel.data_length} bytes, got {len(block)}"
        )

    data = np.frombuffer(block, dtype=channel.dtype, count=expected_items)
    return data.reshape(channel.shape)


def get_channel(path: PathLike, channel_name: str, occurrence: int = 0) -> NDArray[np.signedinteger]:
    metadata = parse_measurement_metadata(path)
    matches = [channel for channel in metadata.channels if channel.name == channel_name]
    if not matches:
        available = [channel.name for channel in metadata.channels]
        raise KeyError(f"Channel {channel_name!r} not found. Available: {available}")
    if occurrence < 0 or occurrence >= len(matches):
        raise IndexError(f"Channel {channel_name!r} occurrence {occurrence} out of range")
    return read_channel_data(metadata.path, matches[occurrence])


def get_channel_by_order(path: PathLike, order: int) -> NDArray[np.signedinteger]:
    metadata = parse_measurement_metadata(path)
    for channel in metadata.channels:
        if channel.order == order:
            return read_channel_data(metadata.path, channel)
    raise KeyError(f"Channel order {order} not found")


def get_channels(path: PathLike, unique_names: bool = False) -> dict[str, list[NDArray[np.signedinteger]]] | dict[str, NDArray[np.signedinteger]]:
    metadata = parse_measurement_metadata(path)
    if unique_names:
        counts: dict[str, int] = defaultdict(int)
        out: dict[str, NDArray[np.signedinteger]] = {}
        for channel in metadata.channels:
            counts[channel.name] += 1
            suffix = "" if counts[channel.name] == 1 else f"#{counts[channel.name]}"
            out[f"{channel.name}{suffix}"] = read_channel_data(metadata.path, channel)
        return out

    grouped: dict[str, list[NDArray[np.signedinteger]]] = defaultdict(list)
    for channel in metadata.channels:
        grouped[channel.name].append(read_channel_data(metadata.path, channel))
    return dict(grouped)
