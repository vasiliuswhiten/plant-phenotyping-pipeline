"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.segmentation import compute_dice, compute_iou, compute_panoptic_quality


class TestDice:
    def test_perfect_match(self):
        pred = np.ones((10, 10), dtype=bool)
        target = np.ones((10, 10), dtype=bool)
        assert compute_dice(pred, target) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5] = True
        target = np.zeros((10, 10), dtype=bool)
        target[5:] = True
        assert compute_dice(pred, target) == pytest.approx(0.0)

    def test_both_empty(self):
        pred = np.zeros((10, 10), dtype=bool)
        target = np.zeros((10, 10), dtype=bool)
        assert compute_dice(pred, target) == pytest.approx(1.0)

    def test_half_overlap(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:7] = True  # 70 pixels
        target = np.zeros((10, 10), dtype=bool)
        target[3:] = True  # 70 pixels, overlap = 40 pixels
        # dice = 2*40 / (70+70) = 80/140 ≈ 0.571
        assert compute_dice(pred, target) == pytest.approx(80.0 / 140.0, abs=0.01)


class TestIoU:
    def test_perfect_match(self):
        pred = np.zeros((10, 10), dtype=int)
        pred[:5] = 1
        target = pred.copy()
        result = compute_iou(pred, target, num_classes=2)
        assert result["mIoU"] == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.zeros((10, 10), dtype=int)
        pred[:5] = 1
        target = np.zeros((10, 10), dtype=int)
        target[5:] = 1
        result = compute_iou(pred, target, num_classes=2)
        assert result["mIoU"] < 0.5


class TestPanopticQuality:
    def test_perfect_match(self):
        pred = np.zeros((10, 10), dtype=int)
        pred[2:5, 2:5] = 1
        pred[6:9, 6:9] = 2
        target = pred.copy()
        result = compute_panoptic_quality(pred, target)
        assert result["PQ"] == pytest.approx(1.0)
        assert result["TP"] == 2.0
        assert result["FP"] == 0.0
        assert result["FN"] == 0.0

    def test_no_match(self):
        pred = np.zeros((10, 10), dtype=int)
        pred[0:3, 0:3] = 1
        target = np.zeros((10, 10), dtype=int)
        target[7:10, 7:10] = 1
        result = compute_panoptic_quality(pred, target)
        assert result["PQ"] == pytest.approx(0.0)
