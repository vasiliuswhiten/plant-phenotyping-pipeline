"""Growth stage estimation from phenological features.

Estimates plant growth stage (BBCH scale) from segmentation-derived
metrics like leaf count, leaf area, and plant size.

TODO: Full regression model implementation in Phase 2.
"""

from __future__ import annotations


def estimate_growth_stage(
    leaf_count: int,
    total_leaf_area: float,
    plant_area: float,
) -> dict[str, float]:
    """Estimate growth stage from phenological features.

    Uses a simple heuristic mapping. Will be replaced with
    a learned regression model in Phase 2.

    Args:
        leaf_count: Number of leaves.
        total_leaf_area: Total leaf area in pixels.
        plant_area: Total plant area in pixels.

    Returns:
        Dict with 'stage' (BBCH code estimate) and 'confidence'.
    """
    # Simple heuristic — to be replaced with trained model
    leaf_area_ratio = total_leaf_area / max(plant_area, 1.0)

    if leaf_count <= 2:
        stage = 12.0  # 2-leaf stage
    elif leaf_count <= 4:
        stage = 14.0  # 4-leaf stage
    elif leaf_count <= 6:
        stage = 16.0  # 6-leaf stage
    elif leaf_count <= 9:
        stage = 19.0  # 9+ leaves
    else:
        stage = 20.0 + min(leaf_count - 9, 10)  # Side shoot formation

    return {
        "stage": stage,
        "leaf_count": float(leaf_count),
        "leaf_area_ratio": leaf_area_ratio,
        "confidence": 0.5,  # Low confidence for heuristic
    }
