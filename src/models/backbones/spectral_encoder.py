"""Spectral encoder for multispectral / hyperspectral data.

Encodes spectral band information using 1D convolutions or
Transformer-based architectures for fusion with RGB features.

TODO: Full implementation in Phase 3-4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    """1D-Conv encoder for spectral band features.

    Processes per-pixel spectral signatures and produces
    feature vectors for fusion with spatial (RGB) features.

    Args:
        num_bands: Number of spectral bands in input.
        hidden_dim: Hidden dimension for conv layers.
        output_dim: Output feature dimension.
    """

    def __init__(
        self,
        num_bands: int = 224,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectral signatures.

        Args:
            x: Spectral input of shape (B, num_bands) or (B, 1, num_bands).

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, num_bands)
        features = self.encoder(x).squeeze(-1)  # (B, hidden_dim * 2)
        return self.head(features)
