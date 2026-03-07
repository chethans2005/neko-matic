"""
Model Evaluation Engine
========================
Computes a comprehensive set of metrics for both classification and
regression tasks.

Classification metrics
----------------------
  accuracy, precision (weighted), recall (weighted), F1 (weighted),
  ROC-AUC (binary: standard; multiclass: OvR weighted)

Regression metrics
------------------
  RMSE, MSE, MAE, R²
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


class ModelEvaluator:
    """
    Evaluates a fitted sklearn ``Pipeline`` against a held-out test set.

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    """

    def __init__(self, problem_type: str) -> None:
        self.problem_type = problem_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pipeline: Pipeline,
        X_test: Any,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generates all relevant metrics for the given pipeline on the test set.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Fully fitted pipeline (preprocessor + model).
        X_test : array-like
            Raw (un-preprocessed) test features.
        y_test : np.ndarray
            True labels / values.

        Returns
        -------
        dict
            Metric name → float (or ``None`` when not computable).
        """
        y_pred = pipeline.predict(X_test)

        if self.problem_type == "classification":
            return self._classification_metrics(pipeline, X_test, y_test, y_pred)
        return self._regression_metrics(y_test, y_pred)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classification_metrics(
        self,
        pipeline: Pipeline,
        X_test: Any,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Computes classification metrics including optional ROC-AUC."""
        metrics: Dict[str, Any] = {
            "accuracy":           float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted":    float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted":        float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "roc_auc":            None,
        }

        # ROC-AUC requires probability estimates
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_test, y_prob[:, 1])
                    )
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_test,
                            y_prob,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
        except Exception:
            pass  # Leave roc_auc as None

        return metrics

    def _regression_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Computes RMSE, MSE, MAE, and R²."""
        mse = float(mean_squared_error(y_test, y_pred))
        return {
            "rmse": float(np.sqrt(mse)),
            "mse":  mse,
            "mae":  float(mean_absolute_error(y_test, y_pred)),
            "r2":   float(r2_score(y_test, y_pred)),
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Returns a display-friendly version of a metrics dictionary.

        ``None`` values become ``"N/A"``; floats are rounded to 4 d.p.
        """
        formatted: Dict[str, str] = {}
        for key, value in metrics.items():
            if value is None:
                formatted[key] = "N/A"
            elif isinstance(value, float):
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = str(value)
        return formatted

    def get_primary_metric_value(
        self, metrics: Dict[str, Any], primary_metric: str
    ) -> Optional[float]:
        """
        Extracts the value for ``primary_metric`` from a metrics dict,
        handling common aliases (e.g. ``"f1"`` → ``"f1_weighted"``).
        """
        aliases: Dict[str, str] = {
            "f1":        "f1_weighted",
            "f1_score":  "f1_weighted",
            "precision": "precision_weighted",
            "recall":    "recall_weighted",
        }
        key = aliases.get(primary_metric.lower(), primary_metric.lower())
        return metrics.get(key)
