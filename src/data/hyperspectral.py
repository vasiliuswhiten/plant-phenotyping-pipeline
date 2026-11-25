"""Hyperspectral dataset loader for WHISPERS object detection challenge.

Reference: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.data.registry import DatasetRegistry


@DatasetRegistry.register("hyperspectral")
class HyperspectralDataset(Dataset):
    """WHISPERS Hyperspectral Object Detection dataset.

    TODO: Implement in Phase 4 when working with hyperspectral data.

    Args:
        root: Path to dataset root.
        split: One of 'train', 'val', 'test'.
        transforms: Transforms pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        raise NotImplementedError("Hyperspectral dataset loader will be implemented in Phase 4")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError
