from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from raw_data_processing.raw_data import PathLike, get_channel_metadata, parse_measurement_metadata, read_channel_data

DEFAULT_CHANNEL_CMAPS = {
    "ZSensor": "viridis",
    "Height": "viridis",
    "AmplitudeError": "magma",
    "Phase": "twilight",
}


def find_preview_image(path: PathLike) -> Path | None:
    raw_path = Path(path)
    candidate = raw_path.with_name(f"{raw_path.name}_1.bmp")
    return candidate if candidate.exists() else None


def load_preview_image(path: PathLike) -> np.ndarray | None:
    preview_path = find_preview_image(path)
    if preview_path is None:
        return None
    with Image.open(preview_path) as image:
        return np.asarray(image.convert("RGB"))


def robust_limits(values: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> tuple[float, float]:
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    if low == high:
        low = float(values.min())
        high = float(values.max())
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _channel_title(channel_name: str, order: int, shape: tuple[int, int]) -> str:
    return f"[{order}] {channel_name}\n{shape[1]}x{shape[0]}"


def create_channel_figure(
    path: PathLike,
    *,
    include_preview: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    channel_cmaps: dict[str, str] | None = None,
    columns: int = 3,
) -> Figure:
    measurement = parse_measurement_metadata(path)
    channel_cmaps = channel_cmaps or DEFAULT_CHANNEL_CMAPS
    preview = load_preview_image(path) if include_preview else None

    panels: list[tuple[str, np.ndarray, str | None]] = []
    if preview is not None:
        panels.append(("RGB preview", preview, None))

    for channel in get_channel_metadata(path):
        values = read_channel_data(path, channel)
        cmap = channel_cmaps.get(channel.name, "gray")
        panels.append((_channel_title(channel.name, channel.order, values.shape), values, cmap))

    n_panels = len(panels)
    n_cols = max(1, columns)
    n_rows = ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for axis in axes_flat[n_panels:]:
        axis.axis("off")

    for axis, (title, image, cmap) in zip(axes_flat, panels, strict=False):
        if cmap is None:
            axis.imshow(image)
        else:
            vmin, vmax = robust_limits(image, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
            artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(measurement.path.name)
    fig.tight_layout()
    return fig


def save_channel_figure(
    path: PathLike,
    output_path: PathLike,
    *,
    include_preview: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    channel_cmaps: dict[str, str] | None = None,
    columns: int = 3,
    dpi: int = 180,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = create_channel_figure(
        path,
        include_preview=include_preview,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        channel_cmaps=channel_cmaps,
        columns=columns,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
