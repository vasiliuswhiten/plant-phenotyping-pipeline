"""Plant stress indicators from spectral and morphological data.

Computes vegetation indices (NDVI, etc.) and morphological stress
markers from multi-modal imagery.

TODO: Spectral indices in Phase 3. Currently supports morphological indicators only.
"""

from __future__ import annotations

import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-infrared band array.
        red: Red band array.

    Returns:
        NDVI array with values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
    return np.clip(ndvi, -1.0, 1.0)


def compute_leaf_wilting_score(
    leaf_mask: np.ndarray,
    leaf_id: int,
) -> float:
    """Estimate leaf wilting from shape analysis.

    Uses circularity (4π × area / perimeter²) as a proxy for wilting.
    Healthy leaves tend to be more circular; wilted leaves are elongated.

    Args:
        leaf_mask: Leaf instance mask.
        leaf_id: Instance ID of the target leaf.

    Returns:
        Wilting score in [0, 1] where 0 = healthy, 1 = severe wilting.
    """
    binary = (leaf_mask == leaf_id).astype(np.uint8)
    area = np.sum(binary)

    if area == 0:
        return 0.0

    # Approximate perimeter by counting boundary pixels
    kernel = np.ones((3, 3), np.uint8)
    # Erode and subtract to get boundary
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(binary, kernel)
    boundary = binary.astype(int) - eroded.astype(int)
    perimeter = np.sum(boundary)

    if perimeter == 0:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # Clamp circularity to [0, 1] and invert: low circularity = high wilting
    wilting = 1.0 - min(circularity, 1.0)

    return float(wilting)
