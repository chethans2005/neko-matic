"""
Base Model Interface
=====================
Abstract base class defining the unified interface for all ML models
in the FlexAutoML registry.

Every model wrapper must implement:
  - fit() / predict() for training and inference
  - get_search_space() for Optuna hyperparameter definitions
  - supports_gpu property for device selection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import optuna


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers in FlexAutoML.

    Provides a unified interface for training, prediction, and
    hyperparameter search space definition.

    Parameters
    ----------
    params : dict
        Hyperparameters to configure the underlying estimator.
    random_state : int
        Random seed for reproducibility.
    use_gpu : bool
        Whether to enable GPU acceleration (if supported).

    Attributes
    ----------
    model : Any
        The underlying sklearn-compatible estimator.
    params : dict
        Current hyperparameter configuration.
    is_fitted : bool
        Whether the model has been fitted.
    """

    # Subclasses should override these class attributes
    name: str = "BaseModel"
    problem_type: str = "classification"  # or "regression"
    supports_gpu: bool = False

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_gpu: bool = False,
    ) -> None:
        self.params = params or {}
        self.random_state = random_state
        self.use_gpu = use_gpu and self.supports_gpu
        self.model: Optional[Any] = None
        self.is_fitted: bool = False

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Creates and returns the underlying estimator instance.

        Must be implemented by subclasses.

        Returns
        -------
        Any
            An sklearn-compatible estimator.
        """
        pass

    @classmethod
    @abstractmethod
    def get_search_space(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Defines the Optuna hyperparameter search space.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial.

        Returns
        -------
        dict
            Hyperparameter name → sampled value.
        """
        pass

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Returns default hyperparameters for the model.

        Can be overridden by subclasses.

        Returns
        -------
        dict
            Default hyperparameter configuration.
        """
        return {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModelWrapper":
        """
        Fits the model to training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target vector (n_samples,).

        Returns
        -------
        self
            Fitted model wrapper.
        """
        if self.model is None:
            self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for input data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predictions.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability estimates (classification only).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class probabilities (n_samples, n_classes).

        Raises
        ------
        AttributeError
            If the underlying model does not support predict_proba.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before predict_proba()")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.name} does not support predict_proba()")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Returns the current hyperparameter configuration."""
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseModelWrapper":
        """
        Updates hyperparameters.

        Parameters
        ----------
        **params
            Hyperparameter key-value pairs.

        Returns
        -------
        self
        """
        self.params.update(params)
        if self.model is not None:
            self.model = self._create_model()
            self.is_fitted = False
        return self

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Returns feature importances if available.

        Returns
        -------
        np.ndarray | None
            Feature importance array or None if not supported.
        """
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).ravel()
        return None

    @property
    def estimator(self) -> Any:
        """Returns the underlying sklearn estimator."""
        if self.model is None:
            self.model = self._create_model()
        return self.model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params}, fitted={self.is_fitted})"


# ---------------------------------------------------------------------------
# Model Factory Protocol
# ---------------------------------------------------------------------------


class ModelFactory:
    """
    Factory for creating model wrapper instances.

    Allows registration and retrieval of model wrapper classes.
    """

    _registry: Dict[str, Type[BaseModelWrapper]] = {}

    @classmethod
    def register(cls, model_class: Type[BaseModelWrapper]) -> Type[BaseModelWrapper]:
        """
        Decorator to register a model wrapper class.

        Parameters
        ----------
        model_class : Type[BaseModelWrapper]
            The model wrapper class to register.

        Returns
        -------
        Type[BaseModelWrapper]
            The same class (allows use as decorator).

        Examples
        --------
        >>> @ModelFactory.register
        ... class MyModel(BaseModelWrapper):
        ...     name = "MyModel"
        ...     ...
        """
        cls._registry[model_class.name] = model_class
        return model_class

    @classmethod
    def create(
        cls,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_gpu: bool = False,
    ) -> BaseModelWrapper:
        """
        Creates a model wrapper instance by name.

        Parameters
        ----------
        name : str
            Registered model name.
        params : dict | None
            Hyperparameters.
        random_state : int
            Random seed.
        use_gpu : bool
            Enable GPU acceleration.

        Returns
        -------
        BaseModelWrapper
            Instantiated model wrapper.

        Raises
        ------
        KeyError
            If the model name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](
            params=params,
            random_state=random_state,
            use_gpu=use_gpu,
        )

    @classmethod
    def get_available_models(
        cls, problem_type: Optional[str] = None
    ) -> List[str]:
        """
        Returns names of all registered models.

        Parameters
        ----------
        problem_type : str | None
            Filter by 'classification' or 'regression'.

        Returns
        -------
        list[str]
            Model names.
        """
        if problem_type is None:
            return list(cls._registry.keys())
        return [
            name
            for name, model_cls in cls._registry.items()
            if model_cls.problem_type == problem_type
        ]

    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModelWrapper]:
        """Returns the model wrapper class by name."""
        return cls._registry[name]
