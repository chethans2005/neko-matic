from __future__ import annotations

from typing import Any, Dict

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


class EvaluationEngine:
    """Compute classification or regression metrics for a trained pipeline."""

    def __init__(self, problem_type: str) -> None:
        self.problem_type = problem_type

    def evaluate(self, pipeline, X_test: Any, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = pipeline.predict(X_test)

        if self.problem_type == "classification":
            metrics: Dict[str, Any] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "roc_auc": None,
            }
            try:
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                        )
            except Exception:
                metrics["roc_auc"] = None
            return metrics

        mse = float(mean_squared_error(y_test, y_pred))
        return {
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
