from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.core.evaluator import EvaluationEngine
from backend.core.model_registry import ModelRegistry
from backend.core.optimizer import HyperparameterOptimizer, OptimizationResult
from backend.core.preprocessing import PreprocessingPipeline


@dataclass
class TrainingResult:
    model_name: str
    model: Any
    pipeline: Pipeline
    metrics: Dict[str, Any]
    best_params: Dict[str, Any]
    training_time: float
    optimization_score: Optional[float] = None


class AutoMLTrainer:
    """Train selected models, run Optuna tuning, and keep ranked results."""

    def __init__(
        self,
        problem_type: str,
        preprocessing_pipeline: PreprocessingPipeline,
        selected_models: List[str],
        metric: str = "accuracy",
        n_trials: int = 20,
        cv_folds: int = 5,
        test_size: float = 0.2,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.problem_type = problem_type
        self.preprocessing_pipeline = preprocessing_pipeline
        self.selected_models = selected_models
        self.metric = metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.results: List[TrainingResult] = []

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.fitted_preprocessor: Optional[Pipeline] = None

    def train_all(self, X: pd.DataFrame, y: pd.Series) -> List[TrainingResult]:
        if len(X) < self.cv_folds * 2:
            raise ValueError("Dataset too small for configured cross validation folds")

        stratify = y if self.problem_type == "classification" else None
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        y_train = self.preprocessing_pipeline.fit_transform_target(y_train_raw)
        y_test = self.preprocessing_pipeline.transform_target(y_test_raw)

        preprocessor = self.preprocessing_pipeline.build_pipeline()
        preprocessor.fit(X_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.fitted_preprocessor = preprocessor

        evaluator = EvaluationEngine(self.problem_type)

        for model_name in self.selected_models:
            start = time.time()

            optimizer = HyperparameterOptimizer(
                model_name=model_name,
                problem_type=self.problem_type,
                metric=self.metric,
                n_trials=self.n_trials,
                cv_folds=self.cv_folds,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
            )

            opt_result: Optional[OptimizationResult] = None
            best_params: Dict[str, Any] = {}
            try:
                opt_result = optimizer.optimize(X_train, y_train, preprocessor)
                best_params = opt_result.best_params
            except Exception:
                best_params = {}

            model = ModelRegistry.get_model(
                model_name=model_name,
                problem_type=self.problem_type,
                params=best_params,
                use_gpu=self.use_gpu,
            )

            full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            full_pipeline.fit(X_train, y_train)
            metrics = evaluator.evaluate(full_pipeline, X_test, y_test)

            self.results.append(
                TrainingResult(
                    model_name=model_name,
                    model=model,
                    pipeline=full_pipeline,
                    metrics=metrics,
                    best_params=best_params,
                    training_time=time.time() - start,
                    optimization_score=(opt_result.best_score if opt_result else None),
                )
            )

        return self.results

    def get_best_model(self) -> Optional[TrainingResult]:
        if not self.results:
            return None

        metric_map = {
            "f1": "f1_weighted",
            "f1 score": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        key = metric_map.get(self.metric.lower(), self.metric)

        lower_better = self.problem_type == "regression" and self.metric.lower() in {"rmse", "mae", "mse"}
        if lower_better:
            return min(self.results, key=lambda item: item.metrics.get(key, float("inf")))
        return max(self.results, key=lambda item: item.metrics.get(key, float("-inf")))
