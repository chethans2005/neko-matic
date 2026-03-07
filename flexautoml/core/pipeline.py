"""
AutoML Pipeline
================
High-level orchestrator that provides a simple API for end-to-end AutoML.

Usage
-----
>>> from flexautoml import AutoMLPipeline
>>> pipeline = AutoMLPipeline(config_path="config.yaml")
>>> pipeline.fit(df, target_column="target")
>>> predictions = pipeline.predict(new_data)
>>> pipeline.save("my_model.pkl")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from flexautoml.core.evaluator import ModelEvaluator
from flexautoml.core.leaderboard import Leaderboard
from flexautoml.core.model_registry import ModelRegistry
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.core.profiler import DatasetProfiler
from flexautoml.core.trainer import AutoMLTrainer, TrainingResult
from flexautoml.explainability.shap_explainer import ShapExplainer
from flexautoml.meta_learning.dataset_difficulty import DatasetDifficultyEstimator
from flexautoml.meta_learning.model_recommender import ModelRecommender
from flexautoml.utils.config_loader import ConfigLoader
from flexautoml.utils.exceptions import (
    DatasetError,
    FlexAutoMLError,
    ModelNotFoundError,
    TargetColumnError,
)
from flexautoml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Container for AutoML pipeline results."""

    best_model_name: str
    best_pipeline: Pipeline
    best_metrics: Dict[str, Any]
    leaderboard: pd.DataFrame
    all_results: List[TrainingResult]
    total_time: float
    problem_type: str
    target_column: str
    dataset_profile: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

    def summary(self) -> str:
        """Returns a summary string of the results."""
        lines = [
            "=" * 60,
            "FlexAutoML Pipeline Results",
            "=" * 60,
            f"Problem Type: {self.problem_type}",
            f"Target Column: {self.target_column}",
            f"Total Training Time: {self.total_time:.1f}s",
            "",
            f"Best Model: {self.best_model_name}",
            f"Best Metrics: {self.best_metrics}",
            "",
            "Leaderboard:",
            self.leaderboard.to_string(),
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoML Pipeline
# ---------------------------------------------------------------------------


class AutoMLPipeline:
    """
    High-level API for end-to-end Automated Machine Learning.

    Orchestrates the full AutoML workflow:
    1. Data profiling and problem type detection
    2. Preprocessing pipeline construction
    3. Model recommendation based on dataset characteristics
    4. Hyperparameter optimization for all selected models
    5. Training and evaluation
    6. Leaderboard generation
    7. Model persistence and explainability

    Parameters
    ----------
    config_path : str | None
        Path to YAML configuration file.
    config : dict | None
        Configuration dictionary (overrides config_path).
    problem_type : str | None
        Force problem type ("classification" or "regression").
        If None, auto-detected from target column.
    selected_models : list[str] | None
        List of model names to train. If None, uses config or all models.
    metric : str | None
        Evaluation metric. If None, uses config default.
    use_recommendations : bool
        Whether to use meta-learning for model recommendations.
    verbose : bool
        Enable verbose logging output.

    Examples
    --------
    >>> pipeline = AutoMLPipeline(config_path="config.yaml")
    >>> result = pipeline.fit(df, target_column="price")
    >>> print(result.summary())
    >>> pipeline.save("best_model.pkl")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        selected_models: Optional[List[str]] = None,
        metric: Optional[str] = None,
        use_recommendations: bool = True,
        verbose: bool = True,
    ) -> None:
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        if config:
            for key, value in config.items():
                self.config_loader.set(key, value)

        # Override with explicit parameters
        self._problem_type = problem_type
        self._selected_models = selected_models
        self._metric = metric
        self.use_recommendations = use_recommendations
        self.verbose = verbose

        # State variables
        self.profiler: Optional[DatasetProfiler] = None
        self.trainer: Optional[AutoMLTrainer] = None
        self.best_result: Optional[TrainingResult] = None
        self.leaderboard: Optional[Leaderboard] = None
        self.results: List[TrainingResult] = []
        self._is_fitted = False

        logger.info("AutoMLPipeline initialized")

    @property
    def problem_type(self) -> str:
        """Returns the problem type (auto-detected or configured)."""
        if self._problem_type:
            return self._problem_type
        return self.config_loader.get("problem_type", "auto")

    @property
    def metric(self) -> str:
        """Returns the evaluation metric."""
        if self._metric:
            return self._metric
        return self.config_loader.get("training.metric", "accuracy")

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str,
        progress_callback: Optional[Any] = None,
    ) -> PipelineResult:
        """
        Fit the AutoML pipeline on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Training data including target column.
        target_column : str
            Name of the target column.
        progress_callback : callable | None
            Optional callback for progress updates.

        Returns
        -------
        PipelineResult
            Container with best model, leaderboard, and all results.

        Raises
        ------
        TargetColumnError
            If target column is not found in data.
        DatasetError
            If dataset is invalid.
        """
        start_time = time.time()
        logger.info(f"Starting AutoML pipeline fit on {len(data)} samples")

        # Validate target column
        if target_column not in data.columns:
            raise TargetColumnError(
                f"Target column '{target_column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Profile the dataset
        self.profiler = DatasetProfiler(data, target_column)
        profile = self.profiler.profile_dataset()

        # Detect problem type
        problem_type = self._detect_problem_type(y, profile)
        logger.info(f"Problem type: {problem_type}")

        # Get model recommendations if enabled
        recommendations = None
        if self.use_recommendations:
            recommender = ModelRecommender()
            recommendations = recommender.recommend(
                X, y, problem_type, top_k=5
            )
            logger.info(f"Recommended models: {[r.model_name for r in recommendations]}")

        # Determine models to train
        selected_models = self._get_selected_models(problem_type, recommendations)
        logger.info(f"Training {len(selected_models)} models: {selected_models}")

        # Build preprocessing pipeline
        preprocessing_config = self.config_loader.preprocessing
        preprocessing_pipeline = PreprocessingPipeline(
            scaler=preprocessing_config.get("scaler", "standard"),
            problem_type=problem_type,
        )

        # Create trainer
        training_config = self.config_loader.training
        self.trainer = AutoMLTrainer(
            problem_type=problem_type,
            preprocessing_pipeline=preprocessing_pipeline,
            selected_models=selected_models,
            metric=self._metric or training_config.get("metric", "accuracy"),
            n_trials=training_config.get("n_trials", 30),
            cv_folds=training_config.get("cv_folds", 5),
            test_size=training_config.get("test_size", 0.2),
            use_gpu=self._should_use_gpu(),
            random_state=training_config.get("random_state", 42),
            progress_callback=progress_callback,
        )

        # Train all models
        self.results = self.trainer.train_all(X, y)

        # Build leaderboard
        self.leaderboard = Leaderboard(problem_type)
        for result in self.results:
            self.leaderboard.add_entry(
                model_name=result.model_name,
                metrics=result.metrics,
                training_time=result.training_time,
            )

        # Get best model
        self.best_result = self.trainer.get_best_model()

        elapsed = time.time() - start_time
        self._is_fitted = True

        logger.info(
            f"Pipeline fit complete in {elapsed:.1f}s. "
            f"Best model: {self.best_result.model_name if self.best_result else 'None'}"
        )

        return PipelineResult(
            best_model_name=self.best_result.model_name if self.best_result else "",
            best_pipeline=self.best_result.pipeline if self.best_result else None,
            best_metrics=self.best_result.metrics if self.best_result else {},
            leaderboard=self.leaderboard.to_dataframe(),
            all_results=self.results,
            total_time=elapsed,
            problem_type=problem_type,
            target_column=target_column,
            dataset_profile=profile,
            recommendations=[r.model_name for r in recommendations] if recommendations else None,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predictions.

        Raises
        ------
        FlexAutoMLError
            If pipeline is not fitted.
        """
        if not self._is_fitted or self.best_result is None:
            raise FlexAutoMLError("Pipeline not fitted. Call fit() first.")
        
        return self.best_result.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        if not self._is_fitted or self.best_result is None:
            raise FlexAutoMLError("Pipeline not fitted. Call fit() first.")
        
        if not hasattr(self.best_result.pipeline, "predict_proba"):
            raise FlexAutoMLError("Model does not support predict_proba")
        
        return self.best_result.pipeline.predict_proba(X)

    def explain(
        self,
        X: pd.DataFrame,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to explain.
        max_samples : int
            Maximum samples to explain.

        Returns
        -------
        dict
            SHAP values and feature importance.
        """
        if not self._is_fitted or self.best_result is None:
            raise FlexAutoMLError("Pipeline not fitted. Call fit() first.")
        
        explainer = ShapExplainer(
            self.best_result.pipeline,
            self.trainer.problem_type,
        )
        
        return explainer.explain(X.head(max_samples))

    def save(self, path: Union[str, Path]) -> str:
        """
        Save the best model to disk.

        Parameters
        ----------
        path : str | Path
            Output file path.

        Returns
        -------
        str
            Absolute path to saved file.
        """
        if not self._is_fitted or self.best_result is None:
            raise FlexAutoMLError("Pipeline not fitted. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_result.pipeline, path)
        
        logger.info(f"Model saved to {path}")
        return str(path.absolute())

    @staticmethod
    def load(path: Union[str, Path]) -> Pipeline:
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str | Path
            Path to saved model file.

        Returns
        -------
        Pipeline
            Loaded sklearn pipeline.
        """
        return joblib.load(path)

    def _detect_problem_type(
        self, y: pd.Series, profile: Dict[str, Any]
    ) -> str:
        """Detect whether this is classification or regression."""
        if self._problem_type and self._problem_type != "auto":
            return self._problem_type
        
        config_type = self.config_loader.get("problem_type", "auto")
        if config_type != "auto":
            return config_type

        # Auto-detect based on target column
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification"
        
        n_unique = y.nunique()
        if n_unique <= 20 and n_unique / len(y) < 0.05:
            return "classification"
        
        return "regression"

    def _get_selected_models(
        self,
        problem_type: str,
        recommendations: Optional[List] = None,
    ) -> List[str]:
        """Get list of models to train."""
        if self._selected_models:
            return self._selected_models
        
        config_models = self.config_loader.get("training.selected_models")
        if config_models:
            return config_models
        
        # Use recommendations if available
        if recommendations and self.use_recommendations:
            return [r.model_name for r in recommendations[:5]]
        
        # Default: all models for problem type
        registry = ModelRegistry(problem_type)
        return registry.get_available_models()

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used."""
        gpu_config = self.config_loader.get("gpu.use_gpu", "auto")
        
        if gpu_config == "auto":
            # Try to detect GPU availability
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                pass
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices("GPU")) > 0
            except ImportError:
                pass
            return False
        
        return bool(gpu_config)
