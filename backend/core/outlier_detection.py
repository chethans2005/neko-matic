from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetectionEngine:
    """Applies configurable outlier filtering before model training."""

    def apply(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        method = str(config.get("method", "none")).lower()
        if method == "none":
            return dataframe

        params = config.get("threshold_parameters", {})
        numeric_cols = dataframe.drop(columns=[target_column], errors="ignore").select_dtypes(
            include=["number"]
        ).columns
        if len(numeric_cols) == 0:
            return dataframe

        work = dataframe.copy()
        features = work[numeric_cols].fillna(work[numeric_cols].median())

        if method == "zscore":
            threshold = float(params.get("zscore_threshold", 3.0))
            z_scores = ((features - features.mean()) / features.std(ddof=0)).abs()
            mask = (z_scores < threshold).all(axis=1)
            return work[mask]

        if method == "iqr":
            multiplier = float(params.get("iqr_multiplier", 1.5))
            q1 = features.quantile(0.25)
            q3 = features.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            mask = ((features >= lower) & (features <= upper)).all(axis=1)
            return work[mask]

        if method == "isolation_forest":
            contamination = float(params.get("isolation_forest_contamination", 0.05))
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
            )
            labels = detector.fit_predict(features)
            return work[labels == 1]

        return dataframe
