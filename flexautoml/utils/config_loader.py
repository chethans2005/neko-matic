"""
Configuration Loader
=====================
Loads FlexAutoML settings from a YAML file, merging user values on top of
sensible defaults. Provides dot-path accessors (``get`` / ``set``) so
callers never need to navigate nested dictionaries manually.

Features:
  - Schema validation with type checking
  - Default configuration for all subsystems
  - Deep merge of user overrides
  - Configuration persistence
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import yaml

from flexautoml.utils.exceptions import ConfigurationError
from flexautoml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration Schema
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Schema for preprocessing configuration."""
    scaler: Optional[Literal["standard", "minmax"]] = "standard"
    num_impute_strategy: Literal["mean", "median", "most_frequent"] = "median"
    cat_impute_strategy: Literal["most_frequent", "constant"] = "most_frequent"
    use_feature_selection: bool = False
    k_best_features: int = 10


@dataclass
class TrainingConfig:
    """Schema for training configuration."""
    selected_models: Optional[List[str]] = None
    metric: str = "accuracy"
    n_trials: int = 30
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    early_stopping_rounds: Optional[int] = None
    timeout_per_trial: Optional[int] = None


@dataclass
class GpuConfig:
    """Schema for GPU configuration."""
    use_gpu: Union[bool, Literal["auto"]] = "auto"


@dataclass
class OutputConfig:
    """Schema for output configuration."""
    save_dir: str = "models"
    leaderboard_path: str = "leaderboard.csv"


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

VALID_CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc", "precision", "recall"}
VALID_REGRESSION_METRICS = {"rmse", "mae", "r2", "mse"}
VALID_SCALERS = {"standard", "minmax", None}
VALID_IMPUTE_STRATEGIES = {"mean", "median", "most_frequent", "constant"}


class ConfigValidator:
    """Validates configuration dictionaries against schema constraints."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> List[str]:
        """
        Validates the configuration and returns a list of errors.
        
        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Validate problem_type
        problem_type = config.get("problem_type", "auto")
        if problem_type not in ("auto", "classification", "regression"):
            errors.append(
                f"Invalid problem_type: '{problem_type}'. "
                f"Must be 'auto', 'classification', or 'regression'."
            )

        # Validate preprocessing
        prep = config.get("preprocessing", {})
        errors.extend(ConfigValidator._validate_preprocessing(prep))

        # Validate training
        training = config.get("training", {})
        errors.extend(ConfigValidator._validate_training(training, problem_type))

        # Validate gpu
        gpu = config.get("gpu", {})
        errors.extend(ConfigValidator._validate_gpu(gpu))

        # Validate output
        output = config.get("output", {})
        errors.extend(ConfigValidator._validate_output(output))

        return errors

    @staticmethod
    def _validate_preprocessing(prep: Dict[str, Any]) -> List[str]:
        """Validates preprocessing configuration."""
        errors = []

        scaler = prep.get("scaler")
        if scaler is not None and scaler not in ("standard", "minmax"):
            errors.append(
                f"Invalid scaler: '{scaler}'. Must be 'standard', 'minmax', or null."
            )

        num_impute = prep.get("num_impute_strategy", "median")
        if num_impute not in VALID_IMPUTE_STRATEGIES:
            errors.append(
                f"Invalid num_impute_strategy: '{num_impute}'. "
                f"Valid options: {VALID_IMPUTE_STRATEGIES}"
            )

        cat_impute = prep.get("cat_impute_strategy", "most_frequent")
        if cat_impute not in ("most_frequent", "constant"):
            errors.append(
                f"Invalid cat_impute_strategy: '{cat_impute}'. "
                f"Must be 'most_frequent' or 'constant'."
            )

        k_best = prep.get("k_best_features", 10)
        if not isinstance(k_best, int) or k_best < 1:
            errors.append(
                f"Invalid k_best_features: '{k_best}'. Must be a positive integer."
            )

        return errors

    @staticmethod
    def _validate_training(training: Dict[str, Any], problem_type: str) -> List[str]:
        """Validates training configuration."""
        errors = []

        metric = training.get("metric", "accuracy")
        if problem_type == "classification":
            if metric not in VALID_CLASSIFICATION_METRICS:
                errors.append(
                    f"Invalid metric for classification: '{metric}'. "
                    f"Valid options: {VALID_CLASSIFICATION_METRICS}"
                )
        elif problem_type == "regression":
            if metric not in VALID_REGRESSION_METRICS:
                errors.append(
                    f"Invalid metric for regression: '{metric}'. "
                    f"Valid options: {VALID_REGRESSION_METRICS}"
                )

        n_trials = training.get("n_trials", 30)
        if not isinstance(n_trials, int) or n_trials < 1:
            errors.append(
                f"Invalid n_trials: '{n_trials}'. Must be a positive integer."
            )

        cv_folds = training.get("cv_folds", 5)
        if not isinstance(cv_folds, int) or cv_folds < 2:
            errors.append(
                f"Invalid cv_folds: '{cv_folds}'. Must be >= 2."
            )

        test_size = training.get("test_size", 0.2)
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            errors.append(
                f"Invalid test_size: '{test_size}'. Must be between 0 and 1."
            )

        return errors

    @staticmethod
    def _validate_gpu(gpu: Dict[str, Any]) -> List[str]:
        """Validates GPU configuration."""
        errors = []

        use_gpu = gpu.get("use_gpu", "auto")
        if use_gpu not in (True, False, "auto"):
            errors.append(
                f"Invalid use_gpu: '{use_gpu}'. Must be true, false, or 'auto'."
            )

        return errors

    @staticmethod
    def _validate_output(output: Dict[str, Any]) -> List[str]:
        """Validates output configuration."""
        errors = []

        save_dir = output.get("save_dir", "models")
        if not isinstance(save_dir, str) or len(save_dir) == 0:
            errors.append("save_dir must be a non-empty string.")

        return errors


# ---------------------------------------------------------------------------
# Built-in defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # 'auto', 'classification', or 'regression'
    "problem_type": "auto",
    "target_column": None,

    "preprocessing": {
        # 'standard' | 'minmax' | null (no scaling)
        "scaler": "standard",
        "num_impute_strategy": "median",       # mean | median | most_frequent
        "cat_impute_strategy": "most_frequent",
        "use_feature_selection": False,
        "k_best_features": 10,
    },

    "training": {
        # null → all models for the detected problem type
        "selected_models": None,
        # classification: accuracy | f1 | roc_auc | precision | recall
        # regression:     rmse | mae | r2
        "metric": "accuracy",
        "n_trials": 30,
        "cv_folds": 5,
        "test_size": 0.2,
        "random_state": 42,
        "early_stopping_rounds": None,
        "timeout_per_trial": None,
    },

    "gpu": {
        # 'auto' → detect at runtime | True | False
        "use_gpu": "auto",
    },

    "output": {
        "save_dir": "models",
        "leaderboard_path": "leaderboard.csv",
    },
}


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """
    Loads and manages FlexAutoML configuration.

    The constructor merges ``DEFAULT_CONFIG`` with the values from
    ``config_path`` (if provided). Any key not present in the user file
    retains its default value.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML configuration file.
    validate : bool
        Whether to validate configuration on load (default: True).
    strict : bool
        If True, raise ConfigurationError on validation errors.
        If False, log warnings but continue (default: False).

    Examples
    --------
    >>> cfg = ConfigLoader("flexautoml/configs/default.yaml")
    >>> cfg.get("training.n_trials")
    30
    >>> cfg.set("training.n_trials", 50)
    >>> cfg.get("training.n_trials")
    50
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        validate: bool = True,
        strict: bool = False,
    ) -> None:
        self.config_path = config_path
        self.validate_on_load = validate
        self.strict = strict
        self.config: Dict[str, Any] = self._load()
        
        if validate:
            self._validate()

    # ------------------------------------------------------------------
    # Loading & merging
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Returns the merged configuration dictionary."""
        base = copy.deepcopy(DEFAULT_CONFIG)
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"Loading configuration from {self.config_path}")
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    user = yaml.safe_load(fh) or {}
                return self._deep_merge(base, user)
            except yaml.YAMLError as e:
                msg = f"Failed to parse YAML config: {e}"
                logger.error(msg)
                if self.strict:
                    raise ConfigurationError(msg) from e
                return base
        elif self.config_path:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
        return base

    def _validate(self) -> None:
        """Validates the configuration against the schema."""
        errors = ConfigValidator.validate(self.config)
        
        if errors:
            for error in errors:
                logger.warning(f"Config validation: {error}")
            
            if self.strict:
                raise ConfigurationError(
                    f"Configuration validation failed with {len(errors)} error(s): "
                    + "; ".join(errors)
                )

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merges ``override`` into ``base`` (non-destructive)."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Dot-path accessors
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a value using a dot-separated key path.

        Example
        -------
        ``cfg.get("training.n_trials")``  → 30
        ``cfg.get("training.missing_key", 0)``  → 0
        """
        parts = key_path.split(".")
        node: Any = self.config
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return default
            if node is None:
                return default
        return node

    def set(self, key_path: str, value: Any) -> None:
        """
        Sets a value using a dot-separated key path, creating intermediate
        dicts as needed.

        Example
        -------
        ``cfg.set("training.n_trials", 50)``
        """
        parts = key_path.split(".")
        node = self.config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Writes the current configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.config, fh, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Convenience section properties
    # ------------------------------------------------------------------

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """The full ``preprocessing`` section as a dict."""
        return self.config.get("preprocessing", {})

    @property
    def training(self) -> Dict[str, Any]:
        """The full ``training`` section as a dict."""
        return self.config.get("training", {})

    @property
    def gpu(self) -> Dict[str, Any]:
        """The full ``gpu`` section as a dict."""
        return self.config.get("gpu", {})

    @property
    def output(self) -> Dict[str, Any]:
        """The full ``output`` section as a dict."""
        return self.config.get("output", {})
