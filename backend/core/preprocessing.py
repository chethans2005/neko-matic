from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


class PreprocessingPipeline:
    """Sklearn preprocessing pipeline with target encoding utilities."""

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
        self.label_encoder = LabelEncoder()

    def build_pipeline(self) -> Pipeline:
        """Build a ColumnTransformer-based pipeline for tabular data."""
        transformers = []

        if self.numerical_features:
            num_steps = [("imputer", SimpleImputer(strategy=self.num_impute_strategy))]
            if self.scaler == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif self.scaler == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            transformers.append(("num", Pipeline(num_steps), self.numerical_features))

        if self.categorical_features:
            cat_pipeline = Pipeline(
                [
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
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        return Pipeline([("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop"))])

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode classification labels and pass through regression targets."""
        if self.problem_type == "classification":
            return self.label_encoder.fit_transform(y)
        return y.values

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transform labels using fitted target encoder."""
        if self.problem_type == "classification":
            return self.label_encoder.transform(y)
        return y.values

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original classes."""
        if self.problem_type == "classification":
            return self.label_encoder.inverse_transform(y)
        return y

class PreprocessingEngine:
    """Builds preprocessing pipeline from backend configuration."""

    def build(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        config: Dict[str, Any],
        problem_type: str,
    ) -> PreprocessingPipeline:
        missing_strategy = config.get("missing_value_strategy", "median")
        encoding = config.get("categorical_encoding", "onehot")
        scaling = config.get("feature_scaling", "standard")

        cat_strategy = "most_frequent"
        if missing_strategy == "drop":
            # Keep compatibility with sklearn SimpleImputer choices.
            missing_strategy = "most_frequent"
        if encoding == "label":
            cat_strategy = "most_frequent"

        pipeline = PreprocessingPipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler=None if scaling == "none" else scaling,
            num_impute_strategy=missing_strategy,
            cat_impute_strategy=cat_strategy,
            use_feature_selection=False,
            k_best_features=10,
            problem_type=problem_type,
        )
        return pipeline
