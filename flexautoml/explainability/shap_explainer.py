"""
SHAP-based Model Explainability
================================
Provides feature importance analysis and SHAP summary plots for any
sklearn-compatible model.

Explainer selection logic
--------------------------
* ``shap.TreeExplainer``    – RandomForest, GradientBoosting, XGB, LGBM
* ``shap.KernelExplainer``  – everything else (uses a background sample)

Both classification and regression are supported.  For multiclass models
the per-class SHAP matrices are averaged element-wise so a single
importance ranking is produced.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPExplainer:
    """
    Wrapper around SHAP that auto-selects the right explainer type.

    Parameters
    ----------
    model : sklearn estimator
        The *bare* model (not the full pipeline) with its best parameters
        already fitted on the transformed training data.
    X_train_transformed : np.ndarray
        Preprocessed training data (output of ``pipeline[:-1].transform``).
    feature_names : list[str] | None
        Column names after preprocessing.  Auto-generated if not provided.
    problem_type : str
        ``"classification"`` or ``"regression"``.
    """

    # Model class-name fragments that trigger TreeExplainer
    _TREE_MODELS = (
        "RandomForest",
        "ExtraTree",
        "GradientBoosting",
        "XGB",
        "LGBM",
        "DecisionTree",
        "AdaBoost",
        "BaggingClassifier",
        "BaggingRegressor",
    )

    def __init__(
        self,
        model: object,
        X_train_transformed: np.ndarray,
        feature_names: Optional[List[str]] = None,
        problem_type: str = "classification",
    ) -> None:
        self.model = model
        self.X_train_transformed = X_train_transformed
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X_train_transformed.shape[1])
        ]
        self.problem_type = problem_type

        self._explainer = None
        self._shap_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_explainer(self) -> None:
        """Lazily initialises the SHAP explainer."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            ) from exc

        model_type = type(self.model).__name__
        is_tree = any(name in model_type for name in self._TREE_MODELS)

        if is_tree:
            self._explainer = shap.TreeExplainer(self.model)
        else:
            # KernelExplainer with a capped background dataset
            n_bg = min(100, len(self.X_train_transformed))
            background = shap.sample(self.X_train_transformed, n_bg)

            # Use predict_proba for classifiers that support it, else predict
            if (
                self.problem_type == "classification"
                and hasattr(self.model, "predict_proba")
            ):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            self._explainer = shap.KernelExplainer(predict_fn, background)

    def _normalise_shap_values(self, raw) -> np.ndarray:
        """
        Collapses multi-class SHAP output to a single 2-D matrix by
        averaging absolute values across classes.
        """
        if isinstance(raw, list):
            # list of 2-D arrays (one per class)
            return np.mean(np.abs(np.array(raw)), axis=0)
        arr = np.array(raw)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            return np.mean(np.abs(arr), axis=2)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap_values(
        self, X: np.ndarray, max_samples: int = 200
    ) -> np.ndarray:
        """
        Computes SHAP values for up to ``max_samples`` rows of ``X``.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix.
        max_samples : int
            Cap to keep computation tractable.

        Returns
        -------
        np.ndarray
            2-D SHAP value matrix [n_samples, n_features].
        """
        if self._explainer is None:
            self._init_explainer()

        X_sample = X[:max_samples]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._explainer.shap_values(X_sample)

        self._shap_values = self._normalise_shap_values(raw)
        return self._shap_values

    def get_feature_importance(
        self, X: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of feature importances (mean |SHAP value|).

        Will compute SHAP values if not already done.

        Parameters
        ----------
        X : np.ndarray | None
            Data to run SHAP on.  Falls back to ``X_train_transformed``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance`` (descending).
        """
        if self._shap_values is None:
            data = X if X is not None else self.X_train_transformed
            self.compute_shap_values(data)

        arr = np.array(self._shap_values)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        importances = np.mean(np.abs(arr), axis=0)
        n = min(len(self.feature_names), len(importances))

        return (
            pd.DataFrame(
                {"feature": self.feature_names[:n], "importance": importances[:n]}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def plot_feature_importance_bar(
        self,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of the top-``top_n`` features by SHAP importance.

        Parameters
        ----------
        top_n : int
            Number of features to display.
        save_path : str | None
            If provided, the figure is saved to this path (PNG).

        Returns
        -------
        matplotlib.figure.Figure
        """
        importance_df = self.get_feature_importance()
        plot_df = importance_df.head(top_n).sort_values("importance")

        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
        bars = ax.barh(plot_df["feature"], plot_df["importance"], color="steelblue")
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances (SHAP)", fontsize=14)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig

    def plot_summary(
        self,
        X: Optional[np.ndarray] = None,
        max_display: int = 20,
        plot_type: str = "bar",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates a SHAP summary plot using the SHAP library's own renderer.

        Parameters
        ----------
        X : np.ndarray | None
            Data for SHAP computation.
        max_display : int
            Maximum number of features shown.
        plot_type : str
            ``"bar"`` (default) or ``"dot"`` (beeswarm).
        save_path : str | None
            Optional path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            ) from exc

        if self._shap_values is None:
            data = X if X is not None else self.X_train_transformed
            self.compute_shap_values(data)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.sca(ax)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(
                self._shap_values,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False,
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig
