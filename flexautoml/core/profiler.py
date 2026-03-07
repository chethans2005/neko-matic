"""
Dataset Profiler
================
Analyzes uploaded datasets to generate a comprehensive profile:
  - Feature types (numerical vs categorical)
  - Missing value statistics
  - Correlation matrix
  - Class distribution / imbalance detection
  - Automatic problem-type detection (classification vs regression)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List


class DatasetProfiler:
    """
    Generates a comprehensive profile of a tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including the target column.
    target_col : str
        Name of the column to predict.
    """

    def __init__(self, df: pd.DataFrame, target_col: str) -> None:
        self.df = df
        self.target_col = target_col
        self.profile: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_problem_type(self) -> str:
        """
        Infers whether the task is classification or regression.

        Rules
        -----
        * Object / category dtype  → classification
        * ≤ 20 unique values       → classification
        * Unique-value ratio < 5 % → classification
        * Else                     → regression
        """
        target = self.df[self.target_col]
        if target.dtype == object or hasattr(target.dtype, "categories"):
            return "classification"
        n_unique = target.nunique()
        if n_unique <= 20 or (n_unique / len(target)) < 0.05:
            return "classification"
        return "regression"

    def _get_feature_types(self) -> Dict[str, List[str]]:
        """Splits feature columns into numerical and categorical lists."""
        numerical, categorical = [], []
        for col in self.df.columns:
            if col == self.target_col:
                continue
            if self.df[col].dtype in (
                "int64", "float64", "int32", "float32", "int16", "float16"
            ):
                numerical.append(col)
            else:
                categorical.append(col)
        return {"numerical": numerical, "categorical": categorical}

    def _get_missing_info(self) -> pd.DataFrame:
        """Returns a DataFrame summarising columns with missing values."""
        counts = self.df.isnull().sum()
        pct = counts / len(self.df) * 100
        return (
            pd.DataFrame({"missing_count": counts, "missing_pct": pct})
            .query("missing_count > 0")
            .sort_values("missing_pct", ascending=False)
        )

    def _get_class_distribution(self) -> pd.Series:
        """Normalised value-counts of the target (classification only)."""
        return self.df[self.target_col].value_counts(normalize=True)

    def _get_correlation_matrix(self) -> pd.DataFrame | None:
        """Returns pearson correlation for all numeric columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return None
        return self.df[num_cols].corr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_dataset(self) -> Dict[str, Any]:
        """
        Runs the complete profiling pipeline.

        Returns
        -------
        dict
            Keys: n_rows, n_cols, n_features, target_col, problem_type,
            feature_types, n_numerical, n_categorical, missing_info,
            total_missing, missing_pct, duplicate_rows, memory_usage_mb,
            dtypes, describe, correlation_matrix.
            (+ class_distribution, n_classes for classification problems)
        """
        feature_types = self._get_feature_types()
        problem_type = self._detect_problem_type()

        self.profile = {
            "n_rows": len(self.df),
            "n_cols": len(self.df.columns),
            "n_features": len(self.df.columns) - 1,
            "target_col": self.target_col,
            "problem_type": problem_type,
            "feature_types": feature_types,
            "n_numerical": len(feature_types["numerical"]),
            "n_categorical": len(feature_types["categorical"]),
            "missing_info": self._get_missing_info(),
            "total_missing": int(self.df.isnull().sum().sum()),
            "missing_pct": self.df.isnull().sum().sum() / self.df.size * 100,
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": self.df.dtypes.to_dict(),
            "describe": self.df.describe(include="all"),
            "correlation_matrix": self._get_correlation_matrix(),
        }

        if problem_type == "classification":
            self.profile["class_distribution"] = self._get_class_distribution()
            self.profile["n_classes"] = int(self.df[self.target_col].nunique())

        return self.profile

    def get_summary(self) -> str:
        """Returns a compact human-readable summary string."""
        if not self.profile:
            self.profile_dataset()
        p = self.profile
        lines = [
            "Dataset Summary",
            "=" * 42,
            f"Rows          : {p['n_rows']:,}",
            f"Columns       : {p['n_cols']}",
            f"Problem type  : {p['problem_type'].upper()}",
            f"Numerical     : {p['n_numerical']}",
            f"Categorical   : {p['n_categorical']}",
            f"Missing vals  : {p['total_missing']:,} ({p['missing_pct']:.2f} %)",
            f"Duplicate rows: {p['duplicate_rows']:,}",
            f"Memory        : {p['memory_usage_mb']:.2f} MB",
        ]
        if p["problem_type"] == "classification":
            lines.append(f"Classes       : {p['n_classes']}")
        return "\n".join(lines)
