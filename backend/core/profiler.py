from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class DataProfiler:
    """Analyze dataset schema, data quality, and inferred task type."""

    PROFILE_SAMPLE_ROWS = 10000
    DESCRIBE_SAMPLE_ROWS = 5000

    @staticmethod
    def _detect_problem_type(dataframe: pd.DataFrame, target_column: str) -> str:
        target = dataframe[target_column]
        if target.dtype == object or hasattr(target.dtype, "categories"):
            return "classification"
        unique_values = target.nunique(dropna=True)
        if unique_values <= 20 or (unique_values / max(len(target), 1)) < 0.05:
            return "classification"
        return "regression"

    @staticmethod
    def _feature_types(dataframe: pd.DataFrame, target_column: str) -> Dict[str, list[str]]:
        numerical = []
        categorical = []
        for column in dataframe.columns:
            if column == target_column:
                continue
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                numerical.append(column)
            else:
                categorical.append(column)
        return {"numerical": numerical, "categorical": categorical}

    def analyze(self, dataframe: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' does not exist")

        # Use bounded samples for expensive statistics so uploads finish quickly.
        profile_df = dataframe
        if len(dataframe) > self.PROFILE_SAMPLE_ROWS:
            profile_df = dataframe.sample(self.PROFILE_SAMPLE_ROWS, random_state=42)

        describe_df = profile_df
        if len(profile_df) > self.DESCRIBE_SAMPLE_ROWS:
            describe_df = profile_df.sample(self.DESCRIBE_SAMPLE_ROWS, random_state=42)

        feature_types = self._feature_types(profile_df, target_column)
        problem_type = self._detect_problem_type(profile_df, target_column)

        missing = profile_df.isnull().sum()
        missing_info = (
            pd.DataFrame(
                {
                    "column": missing.index,
                    "missing_count": missing.values,
                    "missing_pct": ((missing.values / max(len(profile_df), 1)) * 100).round(4),
                }
            )
            .query("missing_count > 0")
            .sort_values("missing_pct", ascending=False)
        )

        numeric_cols = profile_df.select_dtypes(include=[np.number]).columns.tolist()
        correlation = None
        if len(numeric_cols) > 1:
            correlation = profile_df[numeric_cols].corr().fillna(0).round(4).to_dict()

        profile: Dict[str, Any] = {
            "n_rows": int(len(dataframe)),
            "n_cols": int(dataframe.shape[1]),
            "n_features": int(max(dataframe.shape[1] - 1, 0)),
            "target_col": target_column,
            "problem_type": problem_type,
            "feature_types": feature_types,
            "n_numerical": len(feature_types["numerical"]),
            "n_categorical": len(feature_types["categorical"]),
            "missing_info": missing_info.to_dict(orient="records"),
            "total_missing": int(dataframe.isnull().sum().sum()),
            "missing_pct": float((dataframe.isnull().sum().sum() / max(dataframe.size, 1)) * 100),
            "duplicate_rows": int(profile_df.duplicated().sum()),
            "memory_usage_mb": float(dataframe.memory_usage(deep=True).sum() / 1024 / 1024),
            "dtypes": {k: str(v) for k, v in dataframe.dtypes.to_dict().items()},
            "describe": describe_df.describe(include="all").fillna(0).round(4).to_dict(),
            "correlation_matrix": correlation,
            "profile_sample_rows": int(len(profile_df)),
        }

        if problem_type == "classification":
            class_distribution = dataframe[target_column].value_counts(normalize=True)
            profile["class_distribution"] = class_distribution.to_dict()
            profile["n_classes"] = int(dataframe[target_column].nunique())

        return profile
