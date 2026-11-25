"""Detection evaluation metrics: mAP, precision, recall, F1."""

from __future__ import annotations

import numpy as np


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """Compute Average Precision using 11-point interpolation.

    Args:
        precisions: Array of precision values.
        recalls: Array of recall values (must be sorted ascending).

    Returns:
        Average Precision score.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    return float(ap / 11.0)


def compute_detection_metrics(
    pred_boxes: list[np.ndarray],
    pred_scores: list[float],
    pred_classes: list[int],
    gt_boxes: list[np.ndarray],
    gt_classes: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute detection metrics (precision, recall, F1).

    Args:
        pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2].
        pred_scores: Confidence scores for each prediction.
        pred_classes: Predicted class labels.
        gt_boxes: Ground truth bounding boxes.
        gt_classes: Ground truth class labels.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with precision, recall, F1.
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(gt_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-np.array(pred_scores))

    tp = 0
    fp = 0
    matched_gt: set[int] = set()

    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        pred_cls = pred_classes[idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_idx in matched_gt or gt_cls != pred_cls:
                continue

            iou = _box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0
