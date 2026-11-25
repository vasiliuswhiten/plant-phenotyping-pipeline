"""Tests for phenology metrics extraction."""

import numpy as np
import pytest

from src.phenology.leaf_area import (
    compute_leaf_areas,
    compute_leaf_count,
    compute_plant_leaf_distribution,
    compute_total_leaf_area,
)


class TestLeafArea:
    def test_compute_leaf_areas_basic(self):
        """Test basic leaf area computation."""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:30, 10:30] = 1  # leaf 1: 20x20 = 400 pixels
        mask[50:70, 50:80] = 2  # leaf 2: 20x30 = 600 pixels

        areas = compute_leaf_areas(mask)
        assert areas[1] == 400.0
        assert areas[2] == 600.0

    def test_compute_leaf_areas_with_calibration(self):
        """Test area computation with pixels_per_cm2 calibration."""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:30, 10:30] = 1  # 400 pixels

        areas = compute_leaf_areas(mask, pixels_per_cm2=100.0)
        assert areas[1] == pytest.approx(4.0)  # 400 / 100 = 4 cm²

    def test_compute_leaf_areas_empty(self):
        """Test with empty mask."""
        mask = np.zeros((100, 100), dtype=np.int32)
        areas = compute_leaf_areas(mask)
        assert len(areas) == 0

    def test_compute_total_leaf_area(self):
        """Test total leaf area computation."""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:30, 10:30] = 1  # 400
        mask[50:70, 50:80] = 2  # 600

        total = compute_total_leaf_area(mask)
        assert total == 1000.0

    def test_compute_leaf_count(self):
        """Test leaf counting."""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        mask[50:60, 50:60] = 3

        assert compute_leaf_count(mask) == 3

    def test_compute_leaf_count_empty(self):
        """Test leaf counting with empty mask."""
        mask = np.zeros((100, 100), dtype=np.int32)
        assert compute_leaf_count(mask) == 0


class TestPlantLeafDistribution:
    def test_basic_distribution(self):
        """Test per-plant leaf distribution."""
        plant_mask = np.zeros((100, 100), dtype=np.int32)
        plant_mask[0:50, 0:50] = 1  # plant 1

        leaf_mask = np.zeros((100, 100), dtype=np.int32)
        leaf_mask[10:20, 10:20] = 1  # leaf 1 in plant 1: 100 px
        leaf_mask[30:40, 30:40] = 2  # leaf 2 in plant 1: 100 px

        dist = compute_plant_leaf_distribution(plant_mask, leaf_mask)
        assert 1 in dist
        assert dist[1]["leaf_count"] == 2.0
        assert dist[1]["total_area"] == 200.0
        assert dist[1]["mean_leaf_area"] == 100.0
