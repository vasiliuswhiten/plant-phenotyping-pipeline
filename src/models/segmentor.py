"""Instance segmentation models for plant and leaf segmentation.

Supports YOLO-seg and Mask2Former architectures for PhenoBench
instance segmentation tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


class PlantSegmentor:
    """YOLO-seg based plant/leaf instance segmentor.

    Uses Ultralytics YOLO segmentation models for simultaneous
    detection and instance mask prediction.

    Args:
        model_name: YOLO-seg variant (e.g., 'yolov8s-seg', 'yolov8m-seg').
        num_classes: Number of classes.
        pretrained: Whether to use pretrained weights.
        checkpoint: Path to custom checkpoint.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-seg",
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
        batch: int = 4,
        project: str = "results/phenobench",
        name: str = "leaf_seg",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the segmentor."""
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            task="segment",
            exist_ok=True,
            **kwargs,
        )

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
        retina_masks: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Run segmentation inference."""
        return self.model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            retina_masks=retina_masks,
            **kwargs,
        )

    def evaluate(self, data_yaml: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Evaluate on validation set."""
        return self.model.val(data=str(data_yaml), **kwargs)

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export to deployment format."""
        return self.model.export(format=format, **kwargs)
