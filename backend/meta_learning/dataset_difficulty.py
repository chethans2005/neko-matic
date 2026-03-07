from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class DatasetDifficultyAnalyzer:
    """Compute simple meta-features and an overall dataset difficulty score."""

    def analyze(self, dataframe: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        X = dataframe.drop(columns=[target_column], errors="ignore")
        y = dataframe[target_column]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        missing_ratio = float(X.isnull().sum().sum() / max(X.size, 1))
        class_imbalance_ratio = 1.0
        if y.nunique(dropna=True) > 1:
            counts = y.value_counts(dropna=False)
            class_imbalance_ratio = float(counts.iloc[0] / max(counts.iloc[-1], 1))

        mean_corr = 0.0
        if len(numeric_cols) > 1:
            corr = X[numeric_cols].corr().abs().fillna(0).values
            np.fill_diagonal(corr, 0)
            mean_corr = float(corr.mean())

        outlier_ratio = 0.0
        if numeric_cols:
            subset = X[numeric_cols].dropna()
            if len(subset) > 0:
                q1 = subset.quantile(0.25)
                q3 = subset.quantile(0.75)
                iqr = (q3 - q1).replace(0, np.nan)
                mask = ((subset < (q1 - 1.5 * iqr)) | (subset > (q3 + 1.5 * iqr))).any(axis=1)
                outlier_ratio = float(mask.mean())

        n_samples = len(X)
        n_features = len(X.columns)
        dim_ratio = n_features / max(n_samples, 1)

        difficulty = (
            min(1 - min(n_samples / 10000, 1), 1) * 0.2
            + min(dim_ratio * 10, 1) * 0.2
            + min(missing_ratio * 4, 1) * 0.2
            + min((class_imbalance_ratio - 1) / 10, 1) * 0.2
            + min(outlier_ratio * 3, 1) * 0.2
        )

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numerical": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "missing_ratio": missing_ratio,
            "class_imbalance_ratio": class_imbalance_ratio,
            "mean_correlation": mean_corr,
            "outlier_ratio": outlier_ratio,
            "difficulty_score": float(np.clip(difficulty, 0, 1)),
        }
