from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_settings": {
        "target_column": None,
        "problem_type_override": None,
        "train_test_split": 0.2,
        "cross_validation_folds": 5,
    },
    "data_cleaning": {
        "missing_value_strategy": "median",
        "categorical_encoding": "onehot",
        "feature_scaling": "standard",
    },
    "outlier_removal": {
        "method": "none",
        "threshold_parameters": {
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "isolation_forest_contamination": 0.05,
        },
    },
    "feature_engineering": {
        "log_transform": False,
        "polynomial_features": False,
        "feature_interactions": False,
        "feature_selection": {
            "enabled": False,
            "method": "variance_threshold",
            "k_features": 20,
        },
    },
    "model_selection": {
        "list_of_models_to_train": None,
    },
    "hyperparameter_optimization": {
        "optimization_method": "optuna",
        "number_of_trials": 20,
        "timeout": None,
        "early_stopping": True,
    },
    "training_strategy": {
        "parallel_training": False,
        "gpu_usage": "auto",
        "time_budget": None,
    },
    "evaluation_metrics": {
        "primary_metric": "accuracy",
    },
    "explainability": {
        "enable_shap": True,
    },
}


class BackendConfigLoader:
    """Loads and merges backend configuration dictionaries."""

    def __init__(self, config_path: str | None = None) -> None:
        self._config = deepcopy(DEFAULT_CONFIG)
        if config_path:
            self.merge_file(config_path)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def merge_file(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            return self._config
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        self.merge_dict(payload)
        return self._config

    def merge_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._config = self._deep_merge(self._config, payload)
        return self._config

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
