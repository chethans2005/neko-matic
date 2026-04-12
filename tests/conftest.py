"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """Generate dataset with missing values."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.choice(["A", "B", "C", np.nan], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }
    df = pd.DataFrame(data)
    # Inject some NaN values
    df.loc[::10, "feature_1"] = np.nan
    df.loc[::15, "feature_2"] = np.nan
    return df


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory for tests."""
    models_dir = tmp_path / "models"
    (models_dir / "saved_models").mkdir(parents=True)
    (models_dir / "runs").mkdir(parents=True)
    return models_dir
