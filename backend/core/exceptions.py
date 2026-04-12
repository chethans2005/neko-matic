"""Custom exceptions and error handling for neko-matic."""

from typing import Optional


class AutoMLException(Exception):
    """Base exception for all AutoML errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class DatasetError(AutoMLException):
    """Raised when there's an issue with dataset loading or validation."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, error_code="DATASET_ERROR", details=details)


class DatasetFormatError(DatasetError):
    """Raised when dataset format is not supported."""

    def __init__(self, format: str):
        super().__init__(
            f"Unsupported dataset format: {format}. Supported formats: .csv, .xlsx, .xls",
            details={"format": format},
        )


class DatasetLoadError(DatasetError):
    """Raised when dataset cannot be loaded."""

    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Failed to load dataset from {path}: {reason}",
            details={"path": path, "reason": reason},
        )


class InvalidDatasetError(DatasetError):
    """Raised when dataset is invalid or malformed."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid dataset: {reason}", details={"reason": reason})


class ColumnNotFoundError(InvalidDatasetError):
    """Raised when a required column is not found in dataset."""

    def __init__(self, column: str, available_columns: list[str]):
        message = f"Column '{column}' not found in dataset. Available columns: {', '.join(available_columns)}"
        super().__init__(message)


class ConfigError(AutoMLException):
    """Raised when there's an issue with configuration."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, error_code="CONFIG_ERROR", details=details)


class ModelError(AutoMLException):
    """Raised when there's an issue with model training or selection."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, error_code="MODEL_ERROR", details=details)


class UnknownModelError(ModelError):
    """Raised when model is not found in registry."""

    def __init__(self, model_name: str, problem_type: str, available_models: list[str]):
        message = f"Unknown model '{model_name}' for {problem_type}. Available: {', '.join(available_models[:5])}"
        super().__init__(message, details={"model": model_name, "problem_type": problem_type})


class TrainingError(AutoMLException):
    """Raised when training fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, error_code="TRAINING_ERROR", details=details)


class NoDatasetError(AutoMLException):
    """Raised when no dataset is available for training."""

    def __init__(self):
        super().__init__(
            "No dataset set for training. Please upload a dataset first.",
            error_code="NO_DATASET",
        )


class NoActiveRunError(AutoMLException):
    """Raised when no active run is available."""

    def __init__(self):
        super().__init__(
            "No active training run. Start a training run first.",
            error_code="NO_ACTIVE_RUN",
        )


class RunNotFoundError(AutoMLException):
    """Raised when a specific run is not found."""

    def __init__(self, run_id: str):
        super().__init__(
            f"Training run '{run_id}' not found.",
            error_code="RUN_NOT_FOUND",
            details={"run_id": run_id},
        )


class ValidationError(AutoMLException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)


class InsufficientDataError(ValidationError):
    """Raised when dataset is too small."""

    def __init__(self, current_size: int, required_size: int):
        super().__init__(
            f"Dataset too small: {current_size} rows, minimum required: {required_size}",
            details={"current_size": current_size, "required_size": required_size},
        )


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid."""

    def __init__(self, parameter: str, value: str, reason: str):
        super().__init__(
            f"Invalid value for parameter '{parameter}': {value}. {reason}",
            details={"parameter": parameter, "value": value},
        )
