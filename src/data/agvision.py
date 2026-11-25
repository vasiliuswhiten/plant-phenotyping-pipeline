"""AgVision ICPR dataset loader for multi-modal crop disease diagnosis.

Supports RGB, multispectral (Green, Red, RedEdge, NIR), and hyperspectral modalities
for crop disease classification.

Reference: https://han-research.gitlab.io/Agvision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.data.registry import DatasetRegistry


@DatasetRegistry.register("agvision")
class AgVisionDataset(Dataset):
    """AgVision ICPR 2026 dataset for multi-modal crop disease diagnosis.

    TODO: Implement in Phase 3 when working with AgVision data.

    Args:
        root: Path to AgVision dataset root.
        split: One of 'train', 'val', 'test'.
        modalities: List of modalities to load ('rgb', 'multispectral', 'hyperspectral').
        transforms: Albumentations transforms pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        modalities: list[str] | None = None,
        transforms: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.modalities = modalities or ["rgb"]
        self.transforms = transforms
        raise NotImplementedError("AgVision dataset loader will be implemented in Phase 3")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError
