"""Albumentations transform pipelines for training and validation."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 1024) -> A.Compose:
    """Standard training augmentation pipeline.

    Includes geometric and color augmentations suitable for
    outdoor agricultural imagery with variable lighting.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 20,
                max_width=image_size // 20,
                p=0.2,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 1024) -> A.Compose:
    """Validation / test transforms (resize + normalize only)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(image_size: int = 1024) -> A.Compose:
    """Minimal transforms for inference without augmentation."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                value=0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
