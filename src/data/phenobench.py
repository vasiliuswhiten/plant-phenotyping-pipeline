"""PhenoBench dataset loader for plant/leaf segmentation tasks.

PhenoBench provides high-resolution RGB images of sugar beet fields with annotations
for semantic segmentation, plant detection, leaf instance segmentation, and panoptic
segmentation.

Reference: https://www.phenobench.org/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from torch.utils.data import Dataset

from src.data.registry import DatasetRegistry
from src.data.transforms import get_train_transforms, get_val_transforms


@DatasetRegistry.register("phenobench")
class PhenoBenchDataset(Dataset):
    """PhenoBench dataset for plant/leaf detection and segmentation.

    Expected directory structure:
        root/
        ├── train/
        │   ├── images/        # RGB images (1024x1024)
        │   ├── semantics/     # Semantic segmentation masks
        │   ├── plant_instances/  # Plant instance masks
        │   └── leaf_instances/   # Leaf instance masks
        ├── val/
        │   └── ...
        └── test/
            └── images/

    Args:
        root: Path to PhenoBench dataset root.
        split: One of 'train', 'val', 'test'.
        task: One of 'detection', 'semantic', 'plant_instance', 'leaf_instance', 'panoptic'.
        transforms: Albumentations transforms pipeline. If None, uses defaults.
    """

    TASKS = ("detection", "semantic", "plant_instance", "leaf_instance", "panoptic")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        task: str = "plant_instance",
        transforms: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.task = task

        if task not in self.TASKS:
            raise ValueError(f"Task must be one of {self.TASKS}, got '{task}'")

        self.image_dir = self.root / split / "images"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_val_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result: dict[str, Any] = {
            "image": image,
            "image_id": image_path.stem,
        }

        if self.split != "test":
            result.update(self._load_annotations(image_path.stem))

        if self.transforms is not None:
            masks = []
            mask_keys = []
            for key in ("semantic_mask", "plant_mask", "leaf_mask"):
                if key in result:
                    masks.append(result[key])
                    mask_keys.append(key)

            if masks:
                transformed = self.transforms(image=result["image"], masks=masks)
                result["image"] = transformed["image"]
                for key, mask in zip(mask_keys, transformed["masks"], strict=True):
                    result[key] = mask
            else:
                transformed = self.transforms(image=result["image"])
                result["image"] = transformed["image"]

        return result

    def _load_annotations(self, stem: str) -> dict[str, Any]:
        """Load annotation masks based on the task."""
        annotations: dict[str, Any] = {}

        if self.task in ("semantic", "panoptic"):
            sem_path = self.root / self.split / "semantics" / f"{stem}.png"
            if sem_path.exists():
                annotations["semantic_mask"] = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)

        if self.task in ("detection", "plant_instance", "panoptic"):
            plant_path = self.root / self.split / "plant_instances" / f"{stem}.png"
            if plant_path.exists():
                annotations["plant_mask"] = cv2.imread(str(plant_path), cv2.IMREAD_UNCHANGED)

        if self.task in ("leaf_instance", "panoptic"):
            leaf_path = self.root / self.split / "leaf_instances" / f"{stem}.png"
            if leaf_path.exists():
                annotations["leaf_mask"] = cv2.imread(str(leaf_path), cv2.IMREAD_UNCHANGED)

        return annotations

    def get_class_names(self) -> list[str]:
        """Return class names for the semantic segmentation task."""
        return ["background", "crop", "weed"]
