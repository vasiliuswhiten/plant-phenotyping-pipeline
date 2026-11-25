#!/usr/bin/env python3
"""Model export entry point.

Usage:
    python scripts/export.py --format onnx --checkpoint results/phenobench/best.pt
    python scripts/export.py --format onnx --checkpoint results/phenobench/best.pt --benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to deployment formats")
    parser.add_argument("--format", choices=["onnx", "tensorrt", "torchscript"], default="onnx")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--half", action="store_true", help="Export in FP16")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    from src.export.to_onnx import export_ultralytics_to_onnx

    print(f"Exporting {args.checkpoint} to {args.format}...")
    output_path = export_ultralytics_to_onnx(
        model_path=args.checkpoint,
        imgsz=args.imgsz,
        half=args.half,
    )
    print(f"Exported to: {output_path}")

    if args.benchmark:
        from src.export.benchmark import benchmark_onnx

        print("\nRunning inference benchmark...")
        results = benchmark_onnx(str(output_path), input_shape=(1, 3, args.imgsz, args.imgsz))
        for k, v in results.items():
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
