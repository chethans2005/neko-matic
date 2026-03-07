"""
Feature Engineering Utilities
===============================
Optional feature-creation and transformation helpers that can be layered
on top of the base preprocessing pipeline:

  - DatetimeFeatureExtractor  – expands datetime columns into numeric parts
  - HighCardinalityEncoder    – target / frequency encodes high-cardinality
                                categorical columns
  - FeatureEngineer           – orchestrates outlier-flagging and optional
                                polynomial / interaction features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# DatetimeFeatureExtractor
# ---------------------------------------------------------------------------


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Expands datetime columns into useful numeric components.

    Extracted features: year, month, day, hour, dayofweek, quarter.
    The original datetime column is dropped after expansion.

    Parameters
    ----------
    datetime_columns : list[str], optional
        Names of columns to parse as datetimes.  If not provided, the
        transformer is a no-op pass-through.
    """

    def __init__(self, datetime_columns: Optional[List[str]] = None) -> None:
        self.datetime_columns = datetime_columns or []

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeFeatureExtractor":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for col in self.datetime_columns:
            if col not in X.columns:
                continue
            try:
                dt = pd.to_datetime(X[col])
                X[f"{col}_year"] = dt.dt.year
                X[f"{col}_month"] = dt.dt.month
                X[f"{col}_day"] = dt.dt.day
                X[f"{col}_hour"] = dt.dt.hour
                X[f"{col}_dayofweek"] = dt.dt.dayofweek
                X[f"{col}_quarter"] = dt.dt.quarter
                X.drop(columns=[col], inplace=True)
            except Exception:
                pass  # Leave the column untouched if parsing fails
        return X


# ---------------------------------------------------------------------------
# HighCardinalityEncoder
# ---------------------------------------------------------------------------


class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes high-cardinality categorical columns via target mean or frequency.

    Columns whose cardinality exceeds ``threshold`` are selected.
    * When a target vector ``y`` is available during ``fit`` → target encoding.
    * Otherwise → frequency encoding (proportion of each category).

    Parameters
    ----------
    threshold : int
        Minimum number of unique values to trigger encoding (default 20).
    """

    def __init__(self, threshold: int = 20) -> None:
        self.threshold = threshold
        self.encoding_map: dict = {}
        self.high_card_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "HighCardinalityEncoder":  # noqa: N803
        self.high_card_cols = []
        self.encoding_map = {}

        for col in X.columns:
            if X[col].dtype != object or X[col].nunique() <= self.threshold:
                continue
            self.high_card_cols.append(col)
            if y is not None:
                tmp = pd.DataFrame({"feature": X[col], "target": y})
                self.encoding_map[col] = (
                    tmp.groupby("feature")["target"].mean().to_dict()
                )
            else:
                self.encoding_map[col] = (
                    X[col].value_counts(normalize=True).to_dict()
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for col in self.high_card_cols:
            if col in X.columns and col in self.encoding_map:
                X[col] = X[col].map(self.encoding_map[col]).fillna(0)
        return X


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """
    Orchestrates optional feature engineering steps.

    Currently provided helpers:
      * ``add_outlier_flags``           – Z-score outlier indicator columns
      * ``generate_polynomial_features`` – Polynomial / interaction terms

    Parameters
    ----------
    add_polynomial_features : bool
        Whether to expand features with polynomial / interaction terms.
    polynomial_degree : int
        Degree of the polynomial expansion.
    flag_outliers : bool
        Whether to call ``add_outlier_flags`` automatically.
    outlier_threshold : float
        Z-score threshold above which a value is flagged as an outlier.
    """

    def __init__(
        self,
        add_polynomial_features: bool = False,
        polynomial_degree: int = 2,
        flag_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ) -> None:
        self.add_polynomial_features = add_polynomial_features
        self.polynomial_degree = polynomial_degree
        self.flag_outliers = flag_outliers
        self.outlier_threshold = outlier_threshold

    def add_outlier_flags(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Adds a binary ``{col}_is_outlier`` column for each numerical column
        using the Z-score method.

        Parameters
        ----------
        df : pd.DataFrame
        numerical_cols : list[str]
            Columns to inspect.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with additional outlier flag columns.
        """
        df = df.copy()
        for col in numerical_cols:
            if col not in df.columns:
                continue
            std = df[col].std()
            if std == 0:
                df[f"{col}_is_outlier"] = 0
                continue
            z_scores = np.abs((df[col] - df[col].mean()) / std)
            df[f"{col}_is_outlier"] = (z_scores > self.outlier_threshold).astype(int)
        return df

    def generate_polynomial_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generates polynomial and interaction features.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        feature_names : list[str]
            Original feature names.

        Returns
        -------
        Tuple[np.ndarray, list[str]]
            Transformed matrix and the corresponding new feature names.
        """
        poly = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=False,
            interaction_only=False,
        )
        X_poly = poly.fit_transform(X)
        poly_names = poly.get_feature_names_out(feature_names).tolist()
        return X_poly, poly_names
