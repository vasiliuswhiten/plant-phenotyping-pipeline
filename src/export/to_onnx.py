"""Export PyTorch / Ultralytics models to ONNX format.

Supports both custom PyTorch models and Ultralytics YOLO models.
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_ultralytics_to_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 1024,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
) -> Path:
    """Export Ultralytics YOLO model to ONNX.

    Args:
        model_path: Path to .pt model checkpoint.
        output_path: Output ONNX path. If None, uses same dir as model.
        imgsz: Input image size.
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX graph.
        dynamic: Whether to use dynamic input shapes.
        half: Whether to export in FP16.

    Returns:
        Path to exported ONNX model.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
    )
    return Path(export_path)


def export_torch_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 1024, 1024),
    opset: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
    """Export a generic PyTorch model to ONNX.

    Args:
        model: PyTorch model in eval mode.
        output_path: Output ONNX path.
        input_shape: Dummy input shape for tracing.
        opset: ONNX opset version.
        dynamic_axes: Dynamic axis specification.
        input_names: Input tensor names.
        output_names: Output tensor names.

    Returns:
        Path to exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=dynamic_axes,
    )

    return output_path
