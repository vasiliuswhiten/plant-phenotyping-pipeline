# 🌿 Plant Phenotyping & Disease Detection Pipeline

Modular computer vision pipeline for plant analysis across multiple imaging modalities (RGB, multispectral, hyperspectral).

## 🎯 What This Does

| Module | Task | Data Source |
|--------|------|-------------|
| **Plant & Leaf Segmentation** | Instance segmentation of individual plants and leaves | [PhenoBench](https://www.phenobench.org/) |
| **Phenology Estimation** | Leaf area, growth stage, stress indicators from masks | Derived from segmentation |
| **Disease Classification** | Multi-modal disease detection (RGB + spectral fusion) | [AgVision ICPR 2026](https://han-research.gitlab.io/Agvision) |
| **Anomaly / OOD Detection** | Flag unknown pathologies unseen during training | [WHISPERS HSI Challenge](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026) |

## 📊 Benchmark Results

| Benchmark | Task | Metric | Score |
|-----------|------|--------|-------|
| PhenoBench (Codabench) | Leaf Instance Segmentation | PQ | _TBD_ |
| PhenoBench (Codabench) | Plant Detection | mAP | _TBD_ |
| AgVision ICPR | Disease Classification | F1 | _TBD_ |

## 🏗️ Architecture

```
Input (RGB / Multispectral / Hyperspectral)
    │
    ├─► Module 1: Plant & Leaf Instance Segmentation (YOLOv8/v11 + SAM2)
    │       │
    │       └─► Module 2: Phenology Metrics (leaf area, growth stage, stress)
    │
    ├─► Module 3: Disease Classification (DINOv2 + Spectral Encoder → Fusion)
    │
    └─► Module 4: OOD / Anomaly Detection (Energy-based / Mahalanobis)
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/vasiliuswhiten/plant-phenotyping-pipeline.git
cd plant-phenotyping-pipeline

# Install dependencies
pip install -e ".[dev]"

# Download data (via DVC)
dvc pull

# Train plant detection (PhenoBench)
python scripts/train.py --config configs/experiment/phenobench_plant_det.yaml

# Evaluate
python scripts/evaluate.py --config configs/experiment/phenobench_plant_det.yaml

# Export to ONNX
python scripts/export.py --format onnx --checkpoint results/phenobench/best.ckpt
```

## 🛠️ Tech Stack

- **Framework:** PyTorch, PyTorch Lightning
- **Models:** Ultralytics YOLOv8/v11, SAM2, DINOv2, Mask2Former
- **Config:** Hydra
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Deployment:** Docker, ONNX, TensorRT
- **CV/Image:** OpenCV, Albumentations, NumPy, SciPy
- **CI:** GitHub Actions (lint, test, smoke-train)

## 📁 Project Structure

```
├── configs/              # Hydra experiment & model configs
│   ├── experiment/       # Per-task training configs
│   └── model/            # Model architecture configs
├── src/                  # Main source code
│   ├── data/             # Dataloaders (PhenoBench, AgVision, HSI)
│   ├── models/           # Detection, segmentation, classification, OOD
│   ├── phenology/        # Phenological metrics extraction
│   ├── training/         # Trainer, losses, callbacks
│   ├── evaluation/       # Metrics & benchmark submission
│   └── export/           # ONNX / TensorRT export
├── notebooks/            # EDA & visualization
├── scripts/              # CLI entry points
├── tests/                # Unit & integration tests
├── docker/               # Dockerfiles
├── dvc.yaml              # DVC pipeline definition
└── results/              # Outputs (DVC-tracked)
```

## 📄 License

MIT
