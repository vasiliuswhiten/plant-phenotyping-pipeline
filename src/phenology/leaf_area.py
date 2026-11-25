"""Leaf area estimation from instance segmentation masks.

Computes per-plant and per-leaf area metrics from pixel-level
instance masks produced by the segmentation module.
"""

from __future__ import annotations

import numpy as np


def compute_leaf_areas(
    leaf_mask: np.ndarray,
    pixels_per_cm2: float | None = None,
) -> dict[int, float]:
    """Compute area of each leaf instance.

    Args:
        leaf_mask: Instance segmentation mask where each unique non-zero
            value represents a different leaf instance.
        pixels_per_cm2: If provided, converts pixel area to cm².
            Requires camera calibration data.

    Returns:
        Dict mapping instance_id -> area (in pixels or cm²).
    """
    instance_ids = np.unique(leaf_mask)
    instance_ids = instance_ids[instance_ids != 0]  # Remove background

    areas: dict[int, float] = {}
    for inst_id in instance_ids:
        pixel_count = np.sum(leaf_mask == inst_id)
        if pixels_per_cm2 is not None:
            areas[int(inst_id)] = float(pixel_count / pixels_per_cm2)
        else:
            areas[int(inst_id)] = float(pixel_count)

    return areas


def compute_total_leaf_area(
    leaf_mask: np.ndarray,
    pixels_per_cm2: float | None = None,
) -> float:
    """Compute total leaf area across all instances.

    Args:
        leaf_mask: Instance segmentation mask.
        pixels_per_cm2: Optional calibration factor.

    Returns:
        Total leaf area.
    """
    areas = compute_leaf_areas(leaf_mask, pixels_per_cm2)
    return sum(areas.values())


def compute_leaf_count(leaf_mask: np.ndarray) -> int:
    """Count number of leaf instances.

    Args:
        leaf_mask: Instance segmentation mask.

    Returns:
        Number of unique leaf instances.
    """
    instance_ids = np.unique(leaf_mask)
    return int(np.sum(instance_ids != 0))


def compute_plant_leaf_distribution(
    plant_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Compute leaf statistics per plant.

    Args:
        plant_mask: Plant instance mask.
        leaf_mask: Leaf instance mask.

    Returns:
        Dict mapping plant_id -> {leaf_count, total_area, mean_leaf_area}.
    """
    plant_ids = np.unique(plant_mask)
    plant_ids = plant_ids[plant_ids != 0]

    distribution: dict[int, dict[str, float]] = {}
    for plant_id in plant_ids:
        plant_region = plant_mask == plant_id
        leaves_in_plant = leaf_mask[plant_region]
        leaf_ids = np.unique(leaves_in_plant)
        leaf_ids = leaf_ids[leaf_ids != 0]

        leaf_areas = []
        for leaf_id in leaf_ids:
            leaf_areas.append(float(np.sum(leaf_mask == leaf_id)))

        total_area = sum(leaf_areas) if leaf_areas else 0.0
        count = len(leaf_areas)

        distribution[int(plant_id)] = {
            "leaf_count": float(count),
            "total_area": total_area,
            "mean_leaf_area": total_area / count if count > 0 else 0.0,
        }

    return distribution
