"""
Model Registry
==============
Central catalogue of all available classification and regression models,
with unified interface, Optuna search spaces, and GPU support.

Architecture:
  - ``ModelSpec``            – Dataclass holding model metadata
  - ``SearchSpaces``         – Static class with all Optuna search spaces
  - ``ModelRegistry``        – Main registry class for model management
  - ``get_hyperparameter_space`` – Legacy function for backward compatibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from flexautoml.utils.exceptions import ModelNotFoundError
from flexautoml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Specification
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """
    Specification for a registered model.

    Attributes
    ----------
    name : str
        Unique identifier for the model.
    estimator_class : Type[BaseEstimator]
        The sklearn-compatible estimator class.
    problem_type : str
        "classification" or "regression".
    default_params : dict
        Default hyperparameters.
    supports_gpu : bool
        Whether the model supports GPU acceleration.
    search_space_fn : Callable | None
        Function that returns Optuna search space.
    tags : list[str]
        Tags for categorization (e.g., "tree", "linear", "ensemble").
    """

    name: str
    estimator_class: Type[BaseEstimator]
    problem_type: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    supports_gpu: bool = False
    search_space_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None
    tags: List[str] = field(default_factory=list)

    def create_instance(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> BaseEstimator:
        """
        Creates an instance of the model with given parameters.

        Parameters
        ----------
        params : dict | None
            Override parameters.
        use_gpu : bool
            Enable GPU if supported.
        random_state : int
            Random seed.

        Returns
        -------
        BaseEstimator
            Configured estimator instance.
        """
        config = self.default_params.copy()

        # Apply GPU settings
        if use_gpu and self.supports_gpu:
            if "XGB" in self.name:
                config["device"] = "cuda"
            elif "LGBM" in self.name:
                config["device"] = "gpu"

        # Set random state if supported
        if "random_state" in self.default_params or hasattr(
            self.estimator_class(), "random_state"
        ):
            try:
                config.setdefault("random_state", random_state)
            except Exception:
                pass

        # Override with user params
        if params:
            config.update(params)

        return self.estimator_class(**config)

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Returns the Optuna search space for this model."""
        if self.search_space_fn is None:
            return {}
        return self.search_space_fn(trial)


# ---------------------------------------------------------------------------
# Search Spaces
# ---------------------------------------------------------------------------


class SearchSpaces:
    """Static class containing all Optuna search space definitions."""

    @staticmethod
    def random_forest(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    @staticmethod
    def gradient_boosting(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    @staticmethod
    def xgboost(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    @staticmethod
    def lightgbm(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    @staticmethod
    def svc(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    @staticmethod
    def svr(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
        }

    @staticmethod
    def knn(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }

    @staticmethod
    def logistic_regression(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }

    @staticmethod
    def ridge(trial: optuna.Trial) -> Dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}

    @staticmethod
    def lasso(trial: optuna.Trial) -> Dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}


# ---------------------------------------------------------------------------
# Model Catalogues (Backward Compatibility)
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "supports_gpu": False,
    },
    "RandomForestClassifier": {
        "class": RandomForestClassifier,
        "default_params": {"random_state": 42, "n_jobs": -1},
        "supports_gpu": False,
    },
    "GradientBoostingClassifier": {
        "class": GradientBoostingClassifier,
        "default_params": {"random_state": 42},
        "supports_gpu": False,
    },
    "XGBClassifier": {
        "class": XGBClassifier,
        "default_params": {"random_state": 42, "eval_metric": "logloss", "verbosity": 0},
        "supports_gpu": True,
    },
    "LGBMClassifier": {
        "class": LGBMClassifier,
        "default_params": {"random_state": 42, "verbosity": -1, "n_jobs": -1},
        "supports_gpu": True,
    },
    "SVC": {
        "class": SVC,
        "default_params": {"probability": True, "random_state": 42},
        "supports_gpu": False,
    },
    "KNeighborsClassifier": {
        "class": KNeighborsClassifier,
        "default_params": {"n_jobs": -1},
        "supports_gpu": False,
    },
    "GaussianNB": {
        "class": GaussianNB,
        "default_params": {},
        "supports_gpu": False,
    },
}

REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "LinearRegression": {
        "class": LinearRegression,
        "default_params": {"n_jobs": -1},
        "supports_gpu": False,
    },
    "Ridge": {
        "class": Ridge,
        "default_params": {"random_state": 42},
        "supports_gpu": False,
    },
    "Lasso": {
        "class": Lasso,
        "default_params": {"random_state": 42, "max_iter": 2000},
        "supports_gpu": False,
    },
    "RandomForestRegressor": {
        "class": RandomForestRegressor,
        "default_params": {"random_state": 42, "n_jobs": -1},
        "supports_gpu": False,
    },
    "GradientBoostingRegressor": {
        "class": GradientBoostingRegressor,
        "default_params": {"random_state": 42},
        "supports_gpu": False,
    },
    "XGBRegressor": {
        "class": XGBRegressor,
        "default_params": {"random_state": 42, "verbosity": 0},
        "supports_gpu": True,
    },
    "LGBMRegressor": {
        "class": LGBMRegressor,
        "default_params": {"random_state": 42, "verbosity": -1, "n_jobs": -1},
        "supports_gpu": True,
    },
    "SVR": {
        "class": SVR,
        "default_params": {},
        "supports_gpu": False,
    },
}


# ---------------------------------------------------------------------------
# Search Space Mapping
# ---------------------------------------------------------------------------

_SEARCH_SPACE_MAP: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {
    "RandomForestClassifier": SearchSpaces.random_forest,
    "RandomForestRegressor": SearchSpaces.random_forest,
    "GradientBoostingClassifier": SearchSpaces.gradient_boosting,
    "GradientBoostingRegressor": SearchSpaces.gradient_boosting,
    "XGBClassifier": SearchSpaces.xgboost,
    "XGBRegressor": SearchSpaces.xgboost,
    "LGBMClassifier": SearchSpaces.lightgbm,
    "LGBMRegressor": SearchSpaces.lightgbm,
    "SVC": SearchSpaces.svc,
    "SVR": SearchSpaces.svr,
    "KNeighborsClassifier": SearchSpaces.knn,
    "LogisticRegression": SearchSpaces.logistic_regression,
    "Ridge": SearchSpaces.ridge,
    "Lasso": SearchSpaces.lasso,
}


def get_hyperparameter_space(
    trial: optuna.Trial, model_name: str, problem_type: str
) -> Dict[str, Any]:
    """
    Returns an Optuna-sampled hyperparameter dictionary for ``model_name``.

    Uses the centralized ``_SEARCH_SPACE_MAP`` which delegates to
    ``SearchSpaces`` static methods.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial.
    model_name : str
        Registry key (e.g. ``"XGBClassifier"``).
    problem_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    dict
        Hyperparameter key→value pairs for this trial.
    """
    if model_name in _SEARCH_SPACE_MAP:
        return _SEARCH_SPACE_MAP[model_name](trial)
    
    logger.debug(f"No search space defined for {model_name}, using defaults")
    return {}


# ---------------------------------------------------------------------------
# ModelRegistry class
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Manages model instantiation for a given problem type.

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    use_gpu : bool
        When ``True``, GPU-capable models receive the appropriate device flag.
    
    Examples
    --------
    >>> registry = ModelRegistry("classification", use_gpu=False)
    >>> model = registry.get_model("XGBClassifier", {"n_estimators": 100})
    """

    def __init__(self, problem_type: str, use_gpu: bool = False) -> None:
        if problem_type not in ("classification", "regression"):
            raise ValueError(
                f"problem_type must be 'classification' or 'regression', "
                f"got '{problem_type}'"
            )
        
        self.problem_type = problem_type
        self.use_gpu = use_gpu
        self._registry: Dict[str, Dict[str, Any]] = (
            CLASSIFICATION_MODELS
            if problem_type == "classification"
            else REGRESSION_MODELS
        )
        logger.debug(
            f"ModelRegistry initialized for {problem_type} "
            f"(GPU={'enabled' if use_gpu else 'disabled'})"
        )

    def get_available_models(self) -> List[str]:
        """Returns the list of model names for the current problem type."""
        return list(self._registry.keys())

    def get_model(
        self, model_name: str, params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """
        Instantiates a model with the given hyperparameters, merging over
        the stored defaults. Applies GPU device flags where supported.

        Parameters
        ----------
        model_name : str
            Key in the registry (e.g. ``"XGBClassifier"``).
        params : dict, optional
            Hyperparameters to override or extend the defaults.

        Returns
        -------
        BaseEstimator
            Sklearn-compatible estimator instance.

        Raises
        ------
        ModelNotFoundError
            If ``model_name`` is not found in the registry.
        """
        if model_name not in self._registry:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. "
                f"Available: {self.get_available_models()}"
            )
        
        info = self._registry[model_name]
        config: Dict[str, Any] = info["default_params"].copy()

        # GPU acceleration for supported models
        if self.use_gpu and info["supports_gpu"]:
            if "XGB" in model_name:
                config["device"] = "cuda"
                logger.debug(f"Enabled CUDA for {model_name}")
            elif "LGBM" in model_name:
                config["device"] = "gpu"
                logger.debug(f"Enabled GPU for {model_name}")

        if params:
            config.update(params)

        logger.debug(f"Creating {model_name} with config: {config}")
        return info["class"](**config)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Returns the registry metadata dict for ``model_name``."""
        if model_name not in self._registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found in registry.")
        return self._registry[model_name]

    def get_search_space(
        self, trial: optuna.Trial, model_name: str
    ) -> Dict[str, Any]:
        """
        Returns Optuna search space for a model.
        
        Parameters
        ----------
        trial : optuna.Trial
            Active Optuna trial.
        model_name : str
            Model name from the registry.
            
        Returns
        -------
        dict
            Hyperparameter search space.
        """
        return get_hyperparameter_space(trial, model_name, self.problem_type)

    def supports_gpu(self, model_name: str) -> bool:
        """Check if a model supports GPU acceleration."""
        if model_name not in self._registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found.")
        return self._registry[model_name]["supports_gpu"]

    def list_gpu_models(self) -> List[str]:
        """Returns list of models that support GPU acceleration."""
        return [
            name for name, info in self._registry.items()
            if info["supports_gpu"]
        ]
