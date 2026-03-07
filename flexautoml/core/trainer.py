"""
Training Orchestrator
======================
Coordinates the full AutoML training loop:

  1. Splits data into train / test sets.
  2. Fits the preprocessing pipeline on the training split.
  3. For every selected model:
       a. Runs ``HyperparameterOptimizer`` to find the best params.
       b. Re-trains a fresh estimator with those params on the full
          training set (preprocessing embedded in the pipeline).
       c. Evaluates on the held-out test set.
  4. Exposes helpers to retrieve the best model and persist it to disk.

Architecture:
  - ``TrainingResult`` – Container for single model training artefacts
  - ``AutoMLTrainer`` – Main orchestrator class
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from flexautoml.core.evaluator import ModelEvaluator
from flexautoml.core.model_registry import ModelRegistry
from flexautoml.core.optimizer import HyperparameterOptimizer, OptimizationResult
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.utils.exceptions import (
    ModelTrainingError,
    DatasetError,
    FlexAutoMLError,
)
from flexautoml.utils.logging import get_logger, ProgressLogger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """
    Stores all artefacts produced by training a single model.

    Attributes
    ----------
    model_name : str
        Name of the model from registry.
    model : Any
        The bare (non-pipeline) model with the best hyperparameters.
    pipeline : Pipeline
        Full pipeline (preprocessor + model), ready for ``.predict()``.
    metrics : dict
        Evaluation metrics on the held-out test set.
    best_params : dict
        Optimal hyperparameters found by Optuna.
    training_time : float
        Wall-clock seconds taken to optimise + retrain.
    optimization_score : float | None
        Best CV score from Optuna.
    optimization_result : OptimizationResult | None
        Full optimization result with trial history.
    """

    model_name: str
    model: Any
    pipeline: Pipeline
    metrics: Dict[str, Any]
    best_params: Dict[str, Any]
    training_time: float
    optimization_score: Optional[float] = None
    optimization_result: Optional[OptimizationResult] = None

    def __repr__(self) -> str:
        return (
            f"TrainingResult(model={self.model_name}, "
            f"time={self.training_time:.1f}s, "
            f"metrics={self.metrics})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "training_time": self.training_time,
            "optimization_score": self.optimization_score,
        }


# ---------------------------------------------------------------------------
# AutoML Trainer
# ---------------------------------------------------------------------------


class AutoMLTrainer:
    """
    Orchestrates the full AutoML training pipeline for all selected models.

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    preprocessing_pipeline : PreprocessingPipeline
        Configured (but not yet fitted) preprocessing pipeline.
    selected_models : list[str]
        Registry keys of the models to train.
    metric : str
        Primary evaluation metric used for optimisation and model ranking.
    n_trials : int
        Optuna trials per model.
    cv_folds : int
        Cross-validation folds during optimisation.
    test_size : float
        Fraction of data held out as the final test set.
    use_gpu : bool
        Enable GPU acceleration for supported models.
    random_state : int
        Global seed for reproducibility.
    progress_callback : callable | None
        Optional ``callback(model_name: str, progress_pct: float)`` called
        before each model and once at completion (with ``"Done"``, 100).
    pruner_type : str
        Type of Optuna pruner: "median", "percentile", "hyperband", or "none".
    """

    def __init__(
        self,
        problem_type: str,
        preprocessing_pipeline: PreprocessingPipeline,
        selected_models: List[str],
        metric: str = "accuracy",
        n_trials: int = 30,
        cv_folds: int = 5,
        test_size: float = 0.2,
        use_gpu: bool = False,
        random_state: int = 42,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        pruner_type: str = "median",
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
        self.progress_callback = progress_callback
        self.pruner_type = pruner_type

        self.registry = ModelRegistry(problem_type, use_gpu)
        self.results: List[TrainingResult] = []
        self._progress_logger = ProgressLogger(
            total=len(selected_models),
            description="Training models",
        )
        
        logger.info(
            f"AutoMLTrainer initialized: {len(selected_models)} models, "
            f"metric={metric}, n_trials={n_trials}"
        )

        # Populated after _prepare_data()
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.fitted_preprocessor: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ):
        """Splits data, encodes the target, and fits the preprocessor."""
        stratify = y if self.problem_type == "classification" else None

        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # Encode target (LabelEncoder for classification, passthrough for regression)
        y_train = self.preprocessing_pipeline.fit_transform_target(y_train_raw)
        y_test = self.preprocessing_pipeline.fit_transform_target(y_test_raw)

        # Build & fit preprocessing pipeline on training data only
        preproc = self.preprocessing_pipeline.build_pipeline()
        preproc.fit(X_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.fitted_preprocessor = preproc

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train_all(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[TrainingResult]:
        """
        Trains all selected models end-to-end.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (target column excluded).
        y : pd.Series
            Target column.

        Returns
        -------
        list[TrainingResult]

        Raises
        ------
        DatasetError
            If the dataset is invalid or too small.
        """
        logger.info(f"Starting training with {len(X)} samples, {len(X.columns)} features")
        
        if len(X) < self.cv_folds * 2:
            raise DatasetError(
                f"Dataset too small ({len(X)} samples) for {self.cv_folds}-fold CV"
            )
        
        X_train, X_test, y_train, y_test = self._prepare_data(X, y)
        total = len(self.selected_models)
        failed_models = []

        for idx, model_name in enumerate(self.selected_models):
            if self.progress_callback:
                self.progress_callback(model_name, (idx / total) * 100)
            
            self._progress_logger.update(idx + 1, status=f"Training {model_name}")
            logger.info(f"Training model {idx + 1}/{total}: {model_name}")

            try:
                result = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                self.results.append(result)
                logger.info(
                    f"Completed {model_name}: {self._primary_metric_key()}="
                    f"{result.metrics.get(self._primary_metric_key(), 'N/A'):.4f}"
                )
            except Exception as exc:
                logger.warning(f"Skipping {model_name}: {exc}")
                failed_models.append((model_name, str(exc)))

        if self.progress_callback:
            self.progress_callback("Done", 100.0)
        
        self._progress_logger.finish()
        
        logger.info(
            f"Training complete: {len(self.results)} succeeded, "
            f"{len(failed_models)} failed"
        )
        
        if failed_models and not self.results:
            raise ModelTrainingError(
                f"All models failed to train: {failed_models}"
            )

        return self.results

    # ------------------------------------------------------------------
    # Single-model training
    # ------------------------------------------------------------------

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingResult:
        """
        Runs HPO then trains the final model. Returns a ``TrainingResult``.

        Raises
        ------
        ModelTrainingError
            If model training fails.
        """
        start = time.time()
        logger.debug(f"Starting optimization for {model_name}")

        # ----- Optuna hyperparameter optimisation ----------------------
        optimizer = HyperparameterOptimizer(
            model_name=model_name,
            problem_type=self.problem_type,
            metric=self.metric,
            n_trials=self.n_trials,
            cv_folds=self.cv_folds,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            pruner_type=self.pruner_type,
        )
        
        try:
            opt_result = optimizer.optimize(
                X_train, y_train, self.fitted_preprocessor
            )
            best_params = opt_result.best_params
        except Exception as e:
            raise ModelTrainingError(
                f"Optimization failed for {model_name}: {e}"
            ) from e

        # ----- Final model trained on the full training set ------------
        logger.debug(f"Training final model {model_name} with best params")
        final_model = self.registry.get_model(model_name, best_params)
        full_pipeline = Pipeline([
            ("preprocessor", self.fitted_preprocessor),
            ("model", final_model),
        ])
        
        try:
            full_pipeline.fit(X_train, y_train)
        except Exception as e:
            raise ModelTrainingError(
                f"Final training failed for {model_name}: {e}"
            ) from e

        elapsed = time.time() - start

        # ----- Evaluation ----------------------------------------------
        evaluator = ModelEvaluator(self.problem_type)
        metrics = evaluator.evaluate(full_pipeline, X_test, y_test)

        return TrainingResult(
            model_name=model_name,
            model=final_model,
            pipeline=full_pipeline,
            metrics=metrics,
            best_params=best_params,
            training_time=elapsed,
            optimization_score=opt_result.best_score,
            optimization_result=opt_result,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _primary_metric_key(self) -> str:
        """Maps the user-facing metric name to the key in ``metrics`` dicts."""
        mapping = {
            "accuracy":  "accuracy",
            "f1":        "f1_weighted",
            "f1_score":  "f1_weighted",
            "roc_auc":   "roc_auc",
            "precision": "precision_weighted",
            "recall":    "recall_weighted",
            "rmse":      "rmse",
            "mae":       "mae",
            "r2":        "r2",
        }
        return mapping.get(self.metric.lower(), self.metric)

    def get_best_model(self) -> Optional[TrainingResult]:
        """
        Returns the best ``TrainingResult`` ranked by the primary metric.
        Lower is better for RMSE / MAE; higher is better for everything else.
        """
        if not self.results:
            return None
        key = self._primary_metric_key()
        lower_better = (
            self.problem_type == "regression"
            and self.metric.lower() in ("rmse", "mae", "mse")
        )
        if lower_better:
            return min(
                self.results,
                key=lambda r: r.metrics.get(key, float("inf")),
            )
        return max(
            self.results,
            key=lambda r: r.metrics.get(key, float("-inf")),
        )

    def save_model(self, result: TrainingResult, save_dir: str = "models") -> str:
        """
        Persists a trained pipeline to disk with joblib.

        Parameters
        ----------
        result : TrainingResult
        save_dir : str
            Directory to write the ``.pkl`` file into.

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{result.model_name}_{self.problem_type}.pkl"
        filepath = os.path.join(save_dir, filename)
        joblib.dump(result.pipeline, filepath)
        return filepath

    def save_best_model(self, save_dir: str = "models") -> Optional[str]:
        """Saves the best model to ``save_dir`` and returns the file path."""
        best = self.get_best_model()
        if best:
            return self.save_model(best, save_dir)
        return None
