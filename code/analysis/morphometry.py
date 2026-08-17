#!/usr/bin/env python3
"""Shared rectus-muscle morphometry used by all JMS analyses.

MCSA is estimated orthogonally to the muscle's principal axis. The principal axis is
fitted on the largest connected component, while all voxels carrying the anatomical
label contribute to volume and cross-sectional area. A measurement is missing when
the labelled volume is below the prespecified 50 mm3 segmentability threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import label, regionprops


MUSCLE_LABELS: dict[str, int] = {
    "medial_rectus_right": 16,
    "inferior_rectus_right": 9,
    "superior_rectus_right": 6,
    "lateral_rectus_right": 3,
    "medial_rectus_left": 7,
    "inferior_rectus_left": 18,
    "superior_rectus_left": 15,
    "lateral_rectus_left": 12,
}
DEFAULT_VOLUME_THRESHOLD_MM3 = 50.0


def compute_principal_axis_mcsa(
    binary_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Return maximum area (mm2) across bins orthogonal to the muscle axis."""
    components = label(binary_mask)
    regions = regionprops(components)
    if not regions:
        return float("nan")

    largest = max(regions, key=lambda region: region.area)
    axis_coords = np.argwhere(components == largest.label)
    all_coords = np.argwhere(binary_mask)
    if axis_coords.shape[0] < 10 or all_coords.shape[0] < 10:
        return float("nan")

    spacing_array = np.asarray(spacing, dtype=float)
    principal_axis = PCA(n_components=1).fit(
        axis_coords * spacing_array
    ).components_[0]
    projections = (all_coords * spacing_array) @ principal_axis

    bin_width = float(np.max(spacing_array))
    if not np.isfinite(bin_width) or bin_width <= 0:
        return float("nan")
    bins = np.arange(projections.min(), projections.max() + bin_width, bin_width)
    if bins.size < 2:
        return float("nan")

    counts, _ = np.histogram(projections, bins=bins)
    voxel_volume = float(np.prod(spacing_array))
    areas = counts * voxel_volume / bin_width
    return float(np.max(areas)) if areas.size else float("nan")


def measure_label(
    label_data: np.ndarray,
    label_id: int,
    spacing: tuple[float, float, float],
    volume_threshold_mm3: float = DEFAULT_VOLUME_THRESHOLD_MM3,
) -> tuple[float, float]:
    """Return labelled volume (mm3) and MCSA (mm2), or two NaNs on failure."""
    binary_mask = label_data == label_id
    volume = float(binary_mask.sum() * np.prod(spacing))
    if volume < volume_threshold_mm3:
        return float("nan"), float("nan")

    mcsa = compute_principal_axis_mcsa(binary_mask, spacing)
    if not np.isfinite(mcsa):
        return float("nan"), float("nan")
    return volume, mcsa


def extract_mask_morphometry(
    mask_path: str | Path,
    file_id: str,
    volume_threshold_mm3: float = DEFAULT_VOLUME_THRESHOLD_MM3,
) -> dict[str, Any]:
    """Extract the eight rectus-muscle measurements from one label volume."""
    image = nib.load(str(mask_path))
    label_data = image.get_fdata().astype(np.uint8)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing in {mask_path}: {spacing}")

    result: dict[str, Any] = {"file_id": str(file_id).strip()}
    for muscle, label_id in MUSCLE_LABELS.items():
        volume, mcsa = measure_label(
            label_data,
            label_id,
            spacing,
            volume_threshold_mm3=volume_threshold_mm3,
        )
        result[f"{muscle}_vol"] = volume
        result[f"{muscle}_mcsa"] = mcsa
    return result
