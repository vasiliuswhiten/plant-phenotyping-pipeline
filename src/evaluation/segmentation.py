"""Segmentation evaluation metrics: Dice, IoU, Panoptic Quality.

Implements standard metrics for PhenoBench benchmark evaluation.
"""

from __future__ import annotations

import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict[str, float]:
    """Compute per-class and mean IoU.

    Args:
        pred: Predicted segmentation mask (H, W).
        target: Ground truth segmentation mask (H, W).
        num_classes: Number of classes.

    Returns:
        Dict with per-class IoU and mIoU.
    """
    ious: dict[str, float] = {}

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union > 0:
            iou = float(intersection / union)
            ious[f"iou_class_{cls}"] = iou
        else:
            ious[f"iou_class_{cls}"] = float("nan")

    iou_values = [v for v in ious.values() if not np.isnan(v)]
    ious["mIoU"] = float(np.mean(iou_values)) if iou_values else 0.0
    return ious


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient (F1 for segmentation).

    Args:
        pred: Binary predicted mask.
        target: Binary ground truth mask.

    Returns:
        Dice score in [0, 1].
    """
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return 1.0  # Both empty = perfect match

    return float(2 * intersection / total)


def compute_panoptic_quality(
    pred_instances: np.ndarray,
    target_instances: np.ndarray,
    pred_semantics: np.ndarray | None = None,
    target_semantics: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Panoptic Quality (PQ) = SQ × RQ.

    PQ decomposes into:
    - SQ (Segmentation Quality): average IoU of matched segments
    - RQ (Recognition Quality): F1 of segment matching

    Args:
        pred_instances: Predicted instance mask.
        target_instances: Ground truth instance mask.
        pred_semantics: Optional predicted semantic mask.
        target_semantics: Optional ground truth semantic mask.
        iou_threshold: IoU threshold for matching (default 0.5).

    Returns:
        Dict with PQ, SQ, RQ, TP, FP, FN.
    """
    pred_ids = set(np.unique(pred_instances)) - {0}
    target_ids = set(np.unique(target_instances)) - {0}

    matched_pred: set[int] = set()
    matched_target: set[int] = set()
    total_iou = 0.0

    # Match predictions to targets by IoU
    for target_id in target_ids:
        target_mask = target_instances == target_id
        best_iou = 0.0
        best_pred_id = -1

        for pred_id in pred_ids:
            if pred_id in matched_pred:
                continue
            pred_mask = pred_instances == pred_id

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id

        if best_iou >= iou_threshold and best_pred_id >= 0:
            matched_pred.add(best_pred_id)
            matched_target.add(int(target_id))
            total_iou += best_iou

    tp = len(matched_target)
    fp = len(pred_ids - matched_pred)
    fn = len(target_ids - matched_target)

    sq = total_iou / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return {
        "PQ": float(pq),
        "SQ": float(sq),
        "RQ": float(rq),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
    }
