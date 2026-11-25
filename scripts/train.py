#!/usr/bin/env python3
"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiment/phenobench_plant_det.yaml
    python scripts/train.py --config configs/experiment/phenobench_leaf_seg.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train plant phenotyping models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format",
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    if args.override:
        override_conf = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, override_conf)

    print(f"Training with config:\n{OmegaConf.to_yaml(config)}")

    task = config.get("task", "detection")

    if task == "detection":
        from src.models.detector import PlantDetector

        model = PlantDetector(
            model_name=config.model.name,
            pretrained=config.model.get("pretrained", True),
        )
        model.train(
            data_yaml=config.data.yaml_path,
            epochs=config.training.epochs,
            imgsz=config.training.imgsz,
            batch=config.training.batch_size,
            project=config.training.get("project", "results/phenobench"),
            name=config.training.get("name", "plant_det"),
        )

    elif task == "segmentation":
        from src.models.segmentor import PlantSegmentor

        model = PlantSegmentor(
            model_name=config.model.name,
            pretrained=config.model.get("pretrained", True),
        )
        model.train(
            data_yaml=config.data.yaml_path,
            epochs=config.training.epochs,
            imgsz=config.training.imgsz,
            batch=config.training.batch_size,
            project=config.training.get("project", "results/phenobench"),
            name=config.training.get("name", "leaf_seg"),
        )

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
