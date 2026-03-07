"""
Custom Exceptions for FlexAutoML
=================================
Centralized exception hierarchy for consistent error handling across the framework.
"""

from __future__ import annotations

from typing import Any, List, Optional


class FlexAutoMLError(Exception):
    """Base exception for all FlexAutoML errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ---------------------------------------------------------------------------
# Dataset Errors
# ---------------------------------------------------------------------------


class DatasetError(FlexAutoMLError):
    """Base exception for dataset-related errors."""
    pass


class InvalidDatasetError(DatasetError):
    """Raised when the dataset is invalid or cannot be processed."""

    def __init__(
        self,
        message: str = "Invalid dataset provided",
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        details = {}
        if n_rows is not None:
            details["n_rows"] = n_rows
        if n_cols is not None:
            details["n_cols"] = n_cols
        super().__init__(message, details)


class TargetColumnError(DatasetError):
    """Raised when the target column is missing or invalid."""

    def __init__(
        self,
        column: str,
        available_columns: Optional[List[str]] = None,
    ) -> None:
        message = f"Target column '{column}' not found in dataset"
        details = {"target_column": column}
        if available_columns:
            details["available_columns"] = available_columns[:20]  # Limit output
        super().__init__(message, details)


class IncompatibleFeatureError(DatasetError):
    """Raised when feature types are incompatible with the operation."""

    def __init__(
        self,
        feature: str,
        expected_type: str,
        actual_type: str,
    ) -> None:
        message = f"Feature '{feature}' has incompatible type"
        details = {
            "feature": feature,
            "expected_type": expected_type,
            "actual_type": actual_type,
        }
        super().__init__(message, details)


class EmptyDatasetError(DatasetError):
    """Raised when the dataset is empty."""

    def __init__(self) -> None:
        super().__init__("Dataset is empty (0 rows or 0 columns)")


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------


class ConfigurationError(FlexAutoMLError):
    """Base exception for configuration-related errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, key: str, value: Any, reason: str) -> None:
        message = f"Invalid configuration for '{key}': {reason}"
        details = {"key": key, "value": value, "reason": reason}
        super().__init__(message, details)


class MissingConfigError(ConfigurationError):
    """Raised when a required configuration key is missing."""

    def __init__(self, key: str) -> None:
        message = f"Required configuration key '{key}' is missing"
        super().__init__(message, {"key": key})


# ---------------------------------------------------------------------------
# Model Errors
# ---------------------------------------------------------------------------


class ModelError(FlexAutoMLError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not in the registry."""

    def __init__(self, model_name: str, available_models: List[str]) -> None:
        message = f"Model '{model_name}' not found in registry"
        details = {
            "requested_model": model_name,
            "available_models": available_models,
        }
        super().__init__(message, details)


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        original_exception: Optional[Exception] = None,
    ) -> None:
        message = f"Training failed for model '{model_name}': {reason}"
        details = {"model_name": model_name, "reason": reason}
        if original_exception:
            details["original_error"] = str(original_exception)
        super().__init__(message, details)


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(self, model_name: str, reason: str) -> None:
        message = f"Prediction failed for model '{model_name}': {reason}"
        super().__init__(message, {"model_name": model_name, "reason": reason})


# ---------------------------------------------------------------------------
# Preprocessing Errors
# ---------------------------------------------------------------------------


class PreprocessingError(FlexAutoMLError):
    """Base exception for preprocessing-related errors."""
    pass


class PipelineNotFittedError(PreprocessingError):
    """Raised when a pipeline is used before being fitted."""

    def __init__(self, component: str = "Pipeline") -> None:
        message = f"{component} must be fitted before transform/predict"
        super().__init__(message, {"component": component})


# ---------------------------------------------------------------------------
# Optimization Errors
# ---------------------------------------------------------------------------


class OptimizationError(FlexAutoMLError):
    """Base exception for optimization-related errors."""
    pass


class OptimizationTimeoutError(OptimizationError):
    """Raised when optimization times out."""

    def __init__(self, model_name: str, timeout_seconds: int) -> None:
        message = f"Optimization timed out for '{model_name}' after {timeout_seconds}s"
        super().__init__(
            message,
            {"model_name": model_name, "timeout_seconds": timeout_seconds},
        )


class NoValidTrialsError(OptimizationError):
    """Raised when all optimization trials fail."""

    def __init__(self, model_name: str, n_trials: int) -> None:
        message = f"All {n_trials} trials failed for model '{model_name}'"
        super().__init__(
            message,
            {"model_name": model_name, "n_trials": n_trials},
        )
