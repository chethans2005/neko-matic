from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SHAPExplainer:
    """Compute feature importance using SHAP when available, with robust fallback."""

    def __init__(self, pipeline: Pipeline, original_features: pd.DataFrame, problem_type: str) -> None:
        self.pipeline = pipeline
        self.original_features = original_features
        self.problem_type = problem_type

    def _fallback_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_).ravel())
        else:
            values = np.ones(len(feature_names), dtype=float)

        values = values[: len(feature_names)]
        payload = sorted(
            [{"feature": name, "importance": float(val)} for name, val in zip(feature_names, values)],
            key=lambda row: row["importance"],
            reverse=True,
        )
        return {"feature_importance": payload}

    def feature_importance(self, top_k: int = 20) -> Dict[str, Any]:
        preprocessor = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["model"]

        transformed = preprocessor.transform(self.original_features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]

        try:
            import shap

            tree_markers = ("RandomForest", "GradientBoosting", "XGB", "LGBM")
            model_name = type(model).__name__
            if any(marker in model_name for marker in tree_markers):
                explainer = shap.TreeExplainer(model)
            else:
                background = shap.sample(transformed, min(100, len(transformed)))
                predict_fn = model.predict_proba if self.problem_type == "classification" and hasattr(model, "predict_proba") else model.predict
                explainer = shap.KernelExplainer(predict_fn, background)

            shap_values = explainer.shap_values(transformed[: min(200, len(transformed))])
            if isinstance(shap_values, list):
                shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = np.mean(np.abs(shap_values), axis=2)

            importance = np.mean(np.abs(shap_values), axis=0)
            rows = sorted(
                [
                    {"feature": feature_names[idx], "importance": float(importance[idx])}
                    for idx in range(min(len(feature_names), len(importance)))
                ],
                key=lambda row: row["importance"],
                reverse=True,
            )
            return {"feature_importance": rows[:top_k]}
        except Exception:
            return {"feature_importance": self._fallback_importance(model, feature_names)["feature_importance"][:top_k]}
