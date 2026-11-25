#!/usr/bin/env python3
"""Inference entry point for running predictions on new images.

Usage:
    python scripts/predict.py --model results/phenobench/best.pt --source data/images/ --task detection
    python scripts/predict.py --model results/phenobench/best.pt --source image.png --task segmentation
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--source", type=str, required=True, help="Image or directory")
    parser.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--save", action="store_true", help="Save visualized results")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output dir")
    args = parser.parse_args()

    if args.task == "detection":
        from src.models.detector import PlantDetector

        model = PlantDetector(checkpoint=args.model)
    else:
        from src.models.segmentor import PlantSegmentor

        model = PlantSegmentor(checkpoint=args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        project=args.output,
    )

    for r in results:
        print(f"Image: {r.path}")
        if r.boxes is not None:
            print(f"  Detections: {len(r.boxes)}")
        if hasattr(r, "masks") and r.masks is not None:
            print(f"  Masks: {len(r.masks)}")


if __name__ == "__main__":
    main()
