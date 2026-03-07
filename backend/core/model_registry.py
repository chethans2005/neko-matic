from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelSpec:
    name: str
    constructor: Callable[..., Any]
    default_params: Dict[str, Any]
    supports_gpu: bool


class ModelRegistry:
    """Model registry with task-specific model sets and search spaces."""

    CLASSIFICATION_MODELS: Dict[str, ModelSpec] = {
        "LogisticRegression": ModelSpec(
            "LogisticRegression",
            LogisticRegression,
            {"max_iter": 1000, "random_state": 42},
            False,
        ),
        "RandomForestClassifier": ModelSpec(
            "RandomForestClassifier",
            RandomForestClassifier,
            {"random_state": 42, "n_jobs": -1},
            False,
        ),
        "GradientBoostingClassifier": ModelSpec(
            "GradientBoostingClassifier",
            GradientBoostingClassifier,
            {"random_state": 42},
            False,
        ),
        "XGBClassifier": ModelSpec(
            "XGBClassifier",
            XGBClassifier,
            {"random_state": 42, "eval_metric": "logloss", "verbosity": 0},
            True,
        ),
        "LGBMClassifier": ModelSpec(
            "LGBMClassifier",
            LGBMClassifier,
            {"random_state": 42, "verbosity": -1, "n_jobs": -1},
            True,
        ),
        "SVC": ModelSpec("SVC", SVC, {"probability": True}, False),
        "KNeighborsClassifier": ModelSpec(
            "KNeighborsClassifier",
            KNeighborsClassifier,
            {"n_neighbors": 5},
            False,
        ),
        "GaussianNB": ModelSpec("GaussianNB", GaussianNB, {}, False),
    }

    REGRESSION_MODELS: Dict[str, ModelSpec] = {
        "LinearRegression": ModelSpec("LinearRegression", LinearRegression, {"n_jobs": -1}, False),
        "Ridge": ModelSpec("Ridge", Ridge, {"random_state": 42}, False),
        "Lasso": ModelSpec("Lasso", Lasso, {"random_state": 42, "max_iter": 3000}, False),
        "RandomForestRegressor": ModelSpec(
            "RandomForestRegressor",
            RandomForestRegressor,
            {"random_state": 42, "n_jobs": -1},
            False,
        ),
        "GradientBoostingRegressor": ModelSpec(
            "GradientBoostingRegressor",
            GradientBoostingRegressor,
            {"random_state": 42},
            False,
        ),
        "XGBRegressor": ModelSpec("XGBRegressor", XGBRegressor, {"random_state": 42, "verbosity": 0}, True),
        "LGBMRegressor": ModelSpec(
            "LGBMRegressor",
            LGBMRegressor,
            {"random_state": 42, "verbosity": -1, "n_jobs": -1},
            True,
        ),
        "SVR": ModelSpec("SVR", SVR, {}, False),
    }

    @classmethod
    def list_models(cls, problem_type: str) -> List[str]:
        if problem_type == "regression":
            return sorted(cls.REGRESSION_MODELS.keys())
        return sorted(cls.CLASSIFICATION_MODELS.keys())

    @classmethod
    def as_dict(cls) -> Dict[str, List[str]]:
        return {
            "classification": sorted(cls.CLASSIFICATION_MODELS.keys()),
            "regression": sorted(cls.REGRESSION_MODELS.keys()),
        }

    @classmethod
    def get_model(
        cls,
        model_name: str,
        problem_type: str,
        params: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
    ) -> Any:
        catalog = cls.REGRESSION_MODELS if problem_type == "regression" else cls.CLASSIFICATION_MODELS
        if model_name not in catalog:
            raise ValueError(f"Unknown model '{model_name}' for {problem_type}")

        spec = catalog[model_name]
        model_params = dict(spec.default_params)
        if params:
            model_params.update(params)

        if use_gpu and spec.supports_gpu:
            if model_name.startswith("XGB"):
                model_params["device"] = "cuda"
            if model_name.startswith("LGBM"):
                model_params["device"] = "gpu"

        return spec.constructor(**model_params)

    @staticmethod
    def search_space(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        if "RandomForest" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 18),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        if "GradientBoosting" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        if model_name.startswith("XGB"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 450),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        if model_name.startswith("LGBM"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 450),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 180),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        if model_name == "SVC":
            return {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
        if model_name == "SVR":
            return {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            }
        if model_name == "LogisticRegression":
            return {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            }
        if model_name == "KNeighborsClassifier":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
        if model_name == "Ridge":
            return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
        if model_name == "Lasso":
            return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
        return {}
