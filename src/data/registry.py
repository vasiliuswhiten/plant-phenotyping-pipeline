"""Unified dataset registry for all data sources."""

from __future__ import annotations

from typing import Any


class DatasetRegistry:
    """Central registry for dataset classes.

    Usage:
        @DatasetRegistry.register("phenobench")
        class PhenoBenchDataset:
            ...

        dataset = DatasetRegistry.create("phenobench", root="/data/phenobench", split="train")
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class."""

        def decorator(dataset_cls: type) -> type:
            cls._registry[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """Create a dataset instance by name."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered dataset names."""
        return list(cls._registry.keys())
