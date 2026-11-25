.PHONY: install train eval export lint test clean docker-build docker-run

# ─── Setup ───────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-export:
	pip install -e ".[dev,export]"

# ─── Training ────────────────────────────────────────────
train:
	python scripts/train.py --config $(CONFIG)

train-phenobench-det:
	python scripts/train.py --config configs/experiment/phenobench_plant_det.yaml

train-phenobench-seg:
	python scripts/train.py --config configs/experiment/phenobench_leaf_seg.yaml

# ─── Evaluation ──────────────────────────────────────────
eval:
	python scripts/evaluate.py --config $(CONFIG)

# ─── Export ──────────────────────────────────────────────
export-onnx:
	python scripts/export.py --format onnx --checkpoint $(CKPT)

export-tensorrt:
	python scripts/export.py --format tensorrt --checkpoint $(CKPT)

# ─── Code Quality ────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

test:
	pytest tests/ -v --tb=short

typecheck:
	mypy src/

# ─── Docker ──────────────────────────────────────────────
docker-build:
	docker build -t plant-pipeline -f docker/Dockerfile .

docker-run:
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results plant-pipeline

# ─── DVC ─────────────────────────────────────────────────
dvc-pull:
	dvc pull

dvc-push:
	dvc push

# ─── Cleanup ─────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
