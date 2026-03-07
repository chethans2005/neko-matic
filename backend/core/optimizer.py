from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from backend.core.model_registry import ModelRegistry


@dataclass
class OptimizationResult:
    model_name: str
    best_params: Dict[str, Any]
    best_score: float


class HyperparameterOptimizer:
    """Optimize a model over an Optuna search space using CV."""

    SKLEARN_SCORERS = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "f1 score": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    LOWER_BETTER = {"rmse", "mae", "mse"}

    def __init__(
        self,
        model_name: str,
        problem_type: str,
        metric: str,
        n_trials: int,
        cv_folds: int,
        use_gpu: bool,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.problem_type = problem_type
        self.metric = metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu
        self.timeout = timeout
        self.random_state = random_state

    def optimize(self, X, y, preprocessing_pipeline: Pipeline) -> OptimizationResult:
        scorer = self.SKLEARN_SCORERS.get(self.metric.lower(), self.metric)
        minimize = self.metric.lower() in self.LOWER_BETTER

        if self.problem_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            params = ModelRegistry.search_space(trial, self.model_name)
            estimator = ModelRegistry.get_model(
                model_name=self.model_name,
                problem_type=self.problem_type,
                params=params,
                use_gpu=self.use_gpu,
            )
            pipeline = Pipeline([("preprocessor", preprocessing_pipeline), ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            score = float(np.mean(scores))
            # Optuna minimizes objective; invert metrics where larger is better.
            return score if minimize else -score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        raw = float(study.best_value)
        best_score = raw if minimize else -raw
        return OptimizationResult(
            model_name=self.model_name,
            best_params=study.best_params,
            best_score=best_score,
        )
