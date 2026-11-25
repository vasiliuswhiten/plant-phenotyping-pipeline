"""Inference benchmarking for latency and throughput measurement."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def benchmark_onnx(
    model_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 1024, 1024),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict[str, float]:
    """Benchmark ONNX model inference latency.

    Args:
        model_path: Path to ONNX model.
        input_shape: Input tensor shape.
        num_warmup: Number of warmup runs.
        num_runs: Number of benchmark runs.

    Returns:
        Dict with mean, std, min, max latency in ms and throughput (FPS).
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))

    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(latencies_arr)),
        "min_ms": float(np.min(latencies_arr)),
        "max_ms": float(np.max(latencies_arr)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "throughput_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }
