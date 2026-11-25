"""Plant and leaf detection models based on YOLO.

Wraps Ultralytics YOLO for plant/weed detection on PhenoBench data,
with configurable backbone and head variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


class PlantDetector:
    """YOLO-based plant and weed detector.

    Provides a high-level interface for training, evaluation, and inference
    with Ultralytics YOLO models on PhenoBench-style detection tasks.

    Args:
        model_name: YOLO model variant (e.g., 'yolov8n', 'yolov8s', 'yolov8m').
        num_classes: Number of object classes (default: 2 for crop/weed).
        pretrained: Whether to use COCO-pretrained weights.
        checkpoint: Path to a custom checkpoint to resume from.
    """

    CLASS_NAMES = ["crop", "weed"]

    def __init__(
        self,
        model_name: str = "yolov8s",
        num_classes: int = 2,
        pretrained: bool = True,
        checkpoint: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes

        if checkpoint is not None:
            self.model = YOLO(str(checkpoint))
        elif pretrained:
            self.model = YOLO(f"{model_name}.pt")
        else:
            self.model = YOLO(f"{model_name}.yaml")

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 100,
        imgsz: int = 1024,
        batch: int = 8,
        project: str = "results/phenobench",
        name: str = "plant_det",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the detector on PhenoBench data.

        Args:
            data_yaml: Path to YOLO-format data.yaml file.
            epochs: Number of training epochs.
            imgsz: Input image size.
            batch: Batch size.
            project: Output project directory.
            name: Run name.
            **kwargs: Additional Ultralytics training arguments.

        Returns:
            Training results dict.
        """
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            exist_ok=True,
            **kwargs,
        )
        return results

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
        **kwargs: Any,
    ) -> list[Any]:
        """Run detection on images.

        Args:
            source: Image path, directory, or glob pattern.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input image size.

        Returns:
            List of Ultralytics Results objects.
        """
        return self.model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            **kwargs,
        )

    def evaluate(self, data_yaml: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Evaluate on validation set.

        Args:
            data_yaml: Path to YOLO-format data.yaml file.

        Returns:
            Validation metrics dict.
        """
        return self.model.val(data=str(data_yaml), **kwargs)

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export model to deployment format.

        Args:
            format: Export format ('onnx', 'tensorrt', 'torchscript').

        Returns:
            Path to exported model.
        """
        return self.model.export(format=format, **kwargs)
