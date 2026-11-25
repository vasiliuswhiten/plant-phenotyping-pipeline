"""DINOv2 feature extractor for downstream tasks.

Provides frozen or fine-tunable DINOv2 backbone features
for classification and segmentation heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """DINOv2 feature extractor.

    Args:
        model_name: DINOv2 variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14').
        frozen: Whether to freeze backbone weights.
        output_dim: If set, adds a projection layer to this dimension.
    """

    VARIANTS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        frozen: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        if model_name not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{model_name}'. Choose from {list(self.VARIANTS)}")

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.feature_dim = self.VARIANTS[model_name]

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.projection = None
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )
            self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        features = self.model(x)
        if self.projection is not None:
            features = self.projection(features)
        return features
