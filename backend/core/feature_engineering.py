from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineeringEngine:
    """Applies optional transformations configured by the user."""

    def apply(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        config: Dict[str, Any],
        problem_type: str,
    ) -> pd.DataFrame:
        df = dataframe.copy()
        numeric_cols = df.drop(columns=[target_column], errors="ignore").select_dtypes(include=["number"]).columns

        if config.get("log_transform"):
            for col in numeric_cols:
                min_value = df[col].min()
                shifted = df[col] - min_value + 1 if min_value <= 0 else df[col]
                df[col] = np.log1p(shifted)

        if config.get("polynomial_features") and len(numeric_cols) > 0:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            transformed = poly.fit_transform(df[numeric_cols])
            feature_names = poly.get_feature_names_out(numeric_cols)
            poly_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
            non_numeric = df.drop(columns=list(numeric_cols))
            df = pd.concat([non_numeric, poly_df], axis=1)

        if config.get("feature_interactions") and len(numeric_cols) >= 2:
            cols = list(numeric_cols)[:8]
            for idx, col_a in enumerate(cols):
                for col_b in cols[idx + 1 :]:
                    interaction_name = f"{col_a}__x__{col_b}"
                    if interaction_name not in df.columns:
                        df[interaction_name] = df[col_a] * df[col_b]

        selection_cfg = config.get("feature_selection", {})
        if not selection_cfg.get("enabled"):
            return df

        feature_cols = [col for col in df.columns if col != target_column]
        if len(feature_cols) < 2:
            return df

        method = selection_cfg.get("method", "variance_threshold")
        max_features = int(selection_cfg.get("k_features", min(20, len(feature_cols))))
        max_features = max(1, min(max_features, len(feature_cols)))

        X = df[feature_cols].copy()
        X = X.apply(lambda c: c.astype("category").cat.codes if c.dtype == "object" else c)
        y = df[target_column]

        if method == "variance_threshold":
            selector = VarianceThreshold()
            selector.fit(X)
            selected = X.columns[selector.get_support()].tolist()
        elif method == "mutual_information":
            score_func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
            selector = SelectKBest(score_func=score_func, k=max_features)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        elif method == "recursive_feature_elimination":
            estimator = (
                RandomForestClassifier(n_estimators=100, random_state=42)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=100, random_state=42)
            )
            selector = RFE(estimator=estimator, n_features_to_select=max_features)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        else:
            selected = feature_cols[:max_features]

        selected_cols = selected + [target_column]
        return df[selected_cols]
