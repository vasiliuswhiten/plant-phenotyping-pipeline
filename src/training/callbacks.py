"""Training callbacks for MLflow logging and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MLflowCallback:
    """Callback for logging training metrics to MLflow.

    Args:
        experiment_name: MLflow experiment name.
        run_name: Optional run name.
        tracking_uri: MLflow tracking server URI.
    """

    def __init__(
        self,
        experiment_name: str = "plant-phenotyping",
        run_name: str | None = None,
        tracking_uri: str = "mlruns",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_name = run_name
        self._run = None

    def on_train_start(self, config: dict[str, Any]) -> None:
        """Log hyperparameters at training start."""
        self._run = mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(_flatten_dict(config))

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics at end of each epoch."""
        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, best_model_path: str | Path | None = None) -> None:
        """Log final model artifact and end run."""
        if best_model_path is not None:
            mlflow.log_artifact(str(best_model_path))
        if self._run is not None:
            mlflow.end_run()


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dict for MLflow param logging."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
