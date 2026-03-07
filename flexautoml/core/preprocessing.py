"""
Automated Preprocessing Pipeline
==================================
Builds a fully automated sklearn preprocessing pipeline that:
  - Imputes missing values (numerical + categorical)
  - Scales numerical features (StandardScaler or MinMaxScaler)
  - One-hot encodes categorical features
  - Optionally selects the K best features using univariate statistics
  - Encodes the target column for classification tasks

Uses ColumnTransformer + Pipeline for clean, reusable composition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from typing import Any, Dict, List, Optional


class PreprocessingPipeline:
    """
    Builds and manages an automated preprocessing pipeline.

    Parameters
    ----------
    numerical_features : list[str]
        Columns treated as continuous / numerical.
    categorical_features : list[str]
        Columns treated as categorical (will be one-hot encoded).
    scaler : {'standard', 'minmax', None}
        Scaling strategy applied to numerical columns.
    num_impute_strategy : str
        Strategy passed to SimpleImputer for numerical columns.
    cat_impute_strategy : str
        Strategy passed to SimpleImputer for categorical columns.
    use_feature_selection : bool
        Whether to append a SelectKBest step after preprocessing.
    k_best_features : int
        Number of features to keep when ``use_feature_selection`` is True.
    problem_type : {'classification', 'regression'}
        Determines the scoring function used for feature selection and
        whether the target is label-encoded.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        scaler: Optional[str] = "standard",
        num_impute_strategy: str = "median",
        cat_impute_strategy: str = "most_frequent",
        use_feature_selection: bool = False,
        k_best_features: int = 10,
        problem_type: str = "classification",
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = scaler
        self.num_impute_strategy = num_impute_strategy
        self.cat_impute_strategy = cat_impute_strategy
        self.use_feature_selection = use_feature_selection
        self.k_best_features = k_best_features
        self.problem_type = problem_type

        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Sub-pipeline builders
    # ------------------------------------------------------------------

    def _build_numerical_pipeline(self) -> Pipeline:
        """Imputation → optional scaling for numerical features."""
        steps = [
            ("imputer", SimpleImputer(strategy=self.num_impute_strategy))
        ]
        if self.scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        return Pipeline(steps)

    def _build_categorical_pipeline(self) -> Pipeline:
        """Imputation → one-hot encoding for categorical features."""
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.cat_impute_strategy,
                        fill_value="missing",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pipeline(self) -> Pipeline:
        """
        Assembles the full preprocessing pipeline using ColumnTransformer.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Ready for ``.fit()`` / ``.transform()`` calls.
        """
        transformers: list = []

        if self.numerical_features:
            transformers.append(
                ("num", self._build_numerical_pipeline(), self.numerical_features)
            )
        if self.categorical_features:
            transformers.append(
                ("cat", self._build_categorical_pipeline(), self.categorical_features)
            )

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        pipeline_steps: list = [("preprocessor", column_transformer)]

        # Append optional feature selection step
        total_features = len(self.numerical_features) + len(self.categorical_features)
        if self.use_feature_selection and total_features > self.k_best_features:
            score_func = (
                f_classif if self.problem_type == "classification" else f_regression
            )
            pipeline_steps.append(
                (
                    "feature_selection",
                    SelectKBest(score_func=score_func, k=self.k_best_features),
                )
            )

        self.pipeline = Pipeline(pipeline_steps)
        return self.pipeline

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        """
        Encodes the target for classification; returns raw values for regression.

        Parameters
        ----------
        y : pd.Series
            Target column.

        Returns
        -------
        np.ndarray
            Encoded (classification) or raw (regression) target array.
        """
        if self.problem_type == "classification":
            return self.label_encoder.fit_transform(y)
        return y.values

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """
        Applies previously fitted label encoding to new target values.
        For regression tasks this simply returns the raw array.
        """
        if self.problem_type == "classification":
            return self.label_encoder.transform(y)
        return y.values

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Decodes label-encoded targets back to the original class labels."""
        if self.problem_type == "classification":
            return self.label_encoder.inverse_transform(y)
        return y

    def get_feature_names(self) -> List[str]:
        """
        Returns the output feature names after the pipeline has been fitted.

        Raises
        ------
        ValueError
            If the pipeline has not been built and fitted yet.
        """
        if self.pipeline is None:
            raise ValueError(
                "Pipeline not built. Call build_pipeline() and fit it first."
            )
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            return []

    def get_config(self) -> Dict[str, Any]:
        """Returns a JSON-serialisable snapshot of this pipeline's settings."""
        return {
            "n_numerical_features": len(self.numerical_features),
            "n_categorical_features": len(self.categorical_features),
            "scaler": self.scaler,
            "num_impute_strategy": self.num_impute_strategy,
            "cat_impute_strategy": self.cat_impute_strategy,
            "use_feature_selection": self.use_feature_selection,
            "k_best_features": self.k_best_features,
            "problem_type": self.problem_type,
        }
