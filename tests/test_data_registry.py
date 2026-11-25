"""Tests for the dataset registry."""

from src.data.registry import DatasetRegistry


def test_registry_available():
    """Check that known datasets are registered."""
    available = DatasetRegistry.available()
    assert "phenobench" in available
    assert "agvision" in available
    assert "hyperspectral" in available


def test_registry_unknown_raises():
    """Check that unknown dataset name raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown dataset"):
        DatasetRegistry.create("nonexistent_dataset")
