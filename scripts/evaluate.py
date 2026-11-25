#!/usr/bin/env python3
"""Evaluation entry point with benchmark submission generation.

Usage:
    python scripts/evaluate.py --config configs/experiment/phenobench_plant_det.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models and generate submissions")
    parser.add_argument("--config", type=str, required=True, help="Experiment config")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--submit", action="store_true", help="Generate Codabench submission")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print(f"Evaluating with config:\n{OmegaConf.to_yaml(config)}")

    # TODO: Implement evaluation pipeline
    print("Evaluation pipeline will be implemented with training in Phase 1")


if __name__ == "__main__":
    main()
