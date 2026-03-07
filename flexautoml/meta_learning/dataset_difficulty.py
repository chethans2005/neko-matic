"""
Dataset Difficulty Estimation
==============================
Computes meta-features and estimates the difficulty of a dataset
to guide model selection and resource allocation.

Meta-features include:
  - Statistical (mean, std, skewness, kurtosis)
  - Information-theoretic (entropy, mutual information)
  - Complexity (class imbalance, feature correlations)
  - Landmarking (simple model performance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from flexautoml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetMetaFeatures:
    """
    Container for dataset meta-features.

    Attributes
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    n_numerical : int
        Number of numerical features.
    n_categorical : int
        Number of categorical features.
    missing_ratio : float
        Proportion of missing values.
    class_imbalance_ratio : float
        Ratio of majority to minority class (classification only).
    n_classes : int
        Number of classes (classification only).
    mean_correlation : float
        Mean absolute correlation between features.
    max_correlation : float
        Maximum absolute correlation between features.
    mean_skewness : float
        Mean skewness of numerical features.
    mean_kurtosis : float
        Mean kurtosis of numerical features.
    outlier_ratio : float
        Proportion of outliers (IQR method).
    landmark_score : float
        Performance of a simple decision tree (landmarking).
    difficulty_score : float
        Estimated difficulty (0 = easy, 1 = hard).
    """

    n_samples: int = 0
    n_features: int = 0
    n_numerical: int = 0
    n_categorical: int = 0
    missing_ratio: float = 0.0
    class_imbalance_ratio: float = 1.0
    n_classes: int = 2
    mean_correlation: float = 0.0
    max_correlation: float = 0.0
    mean_skewness: float = 0.0
    mean_kurtosis: float = 0.0
    outlier_ratio: float = 0.0
    landmark_score: float = 0.0
    difficulty_score: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert meta-features to a dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_numerical": self.n_numerical,
            "n_categorical": self.n_categorical,
            "missing_ratio": self.missing_ratio,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "n_classes": self.n_classes,
            "mean_correlation": self.mean_correlation,
            "max_correlation": self.max_correlation,
            "mean_skewness": self.mean_skewness,
            "mean_kurtosis": self.mean_kurtosis,
            "outlier_ratio": self.outlier_ratio,
            "landmark_score": self.landmark_score,
            "difficulty_score": self.difficulty_score,
            **self.extra,
        }


class DatasetDifficultyEstimator:
    """
    Estimates dataset difficulty using meta-features.

    The difficulty score helps in:
      - Allocating more optimization trials to harder datasets
      - Recommending ensemble models for complex problems
      - Setting appropriate regularization levels

    Parameters
    ----------
    problem_type : str
        "classification" or "regression".
    compute_landmarks : bool
        Whether to compute landmarking features (slower but informative).
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> estimator = DatasetDifficultyEstimator("classification")
    >>> meta_features = estimator.compute(X, y)
    >>> print(f"Difficulty: {meta_features.difficulty_score:.2f}")
    """

    def __init__(
        self,
        problem_type: str = "classification",
        compute_landmarks: bool = True,
        random_state: int = 42,
    ) -> None:
        self.problem_type = problem_type
        self.compute_landmarks = compute_landmarks
        self.random_state = random_state
        self._meta_features: Optional[DatasetMetaFeatures] = None

    def compute(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DatasetMetaFeatures:
        """
        Computes meta-features for the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        numerical_cols : list[str] | None
            Numerical column names. Auto-detected if None.
        categorical_cols : list[str] | None
            Categorical column names. Auto-detected if None.

        Returns
        -------
        DatasetMetaFeatures
            Computed meta-features with difficulty score.
        """
        logger.debug(f"Computing meta-features for {len(X)} samples")

        # Auto-detect feature types
        if numerical_cols is None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        meta = DatasetMetaFeatures(
            n_samples=len(X),
            n_features=len(X.columns),
            n_numerical=len(numerical_cols),
            n_categorical=len(categorical_cols),
        )

        # Missing value ratio
        meta.missing_ratio = X.isnull().sum().sum() / X.size

        # Class imbalance (classification)
        if self.problem_type == "classification":
            class_counts = y.value_counts()
            meta.n_classes = len(class_counts)
            if len(class_counts) >= 2:
                meta.class_imbalance_ratio = (
                    class_counts.iloc[0] / class_counts.iloc[-1]
                )

        # Correlation features
        if len(numerical_cols) >= 2:
            corr_matrix = X[numerical_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            meta.mean_correlation = corr_matrix.values.mean()
            meta.max_correlation = corr_matrix.values.max()

        # Statistical features
        if numerical_cols:
            skewness = X[numerical_cols].skew().abs()
            kurtosis = X[numerical_cols].kurtosis().abs()
            meta.mean_skewness = skewness.mean() if not skewness.empty else 0.0
            meta.mean_kurtosis = kurtosis.mean() if not kurtosis.empty else 0.0

        # Outlier ratio (IQR method)
        meta.outlier_ratio = self._compute_outlier_ratio(X, numerical_cols)

        # Landmarking
        if self.compute_landmarks:
            meta.landmark_score = self._compute_landmark_score(X, y)

        # Compute difficulty score
        meta.difficulty_score = self._compute_difficulty_score(meta)

        self._meta_features = meta
        logger.debug(f"Difficulty score: {meta.difficulty_score:.3f}")

        return meta

    def _compute_outlier_ratio(
        self, X: pd.DataFrame, numerical_cols: List[str]
    ) -> float:
        """Computes the ratio of outliers using the IQR method."""
        if not numerical_cols:
            return 0.0

        outlier_count = 0
        total_count = 0

        for col in numerical_cols:
            data = X[col].dropna()
            if len(data) < 4:
                continue

            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((data < lower) | (data > upper)).sum()
                outlier_count += outliers
                total_count += len(data)

        return outlier_count / total_count if total_count > 0 else 0.0

    def _compute_landmark_score(
        self, X: pd.DataFrame, y: pd.Series
    ) -> float:
        """
        Computes landmarking score using a simple decision tree.

        Lower score = harder dataset.
        """
        try:
            # Prepare data
            X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
            if X_numeric.empty:
                return 0.5

            if self.problem_type == "classification":
                y_encoded = LabelEncoder().fit_transform(y.fillna("missing"))
                model = DecisionTreeClassifier(
                    max_depth=3,
                    random_state=self.random_state,
                )
                scoring = "accuracy"
            else:
                y_encoded = y.fillna(y.median())
                model = DecisionTreeRegressor(
                    max_depth=3,
                    random_state=self.random_state,
                )
                scoring = "r2"

            scores = cross_val_score(
                model,
                X_numeric,
                y_encoded,
                cv=3,
                scoring=scoring,
                n_jobs=-1,
            )

            return float(np.clip(np.mean(scores), 0, 1))

        except Exception as e:
            logger.warning(f"Landmarking failed: {e}")
            return 0.5

    def _compute_difficulty_score(self, meta: DatasetMetaFeatures) -> float:
        """
        Computes overall difficulty score from meta-features.

        Higher score = harder dataset.
        """
        factors = []

        # Sample size factor (fewer samples = harder)
        size_factor = 1 - min(meta.n_samples / 10000, 1)
        factors.append(size_factor * 0.15)

        # Dimensionality factor (more features relative to samples = harder)
        dim_ratio = meta.n_features / max(meta.n_samples, 1)
        dim_factor = min(dim_ratio * 10, 1)
        factors.append(dim_factor * 0.15)

        # Missing value factor
        factors.append(min(meta.missing_ratio * 5, 1) * 0.1)

        # Class imbalance factor
        imbalance_factor = min((meta.class_imbalance_ratio - 1) / 10, 1)
        factors.append(imbalance_factor * 0.15)

        # Correlation factor (high correlation = easier, paradoxically)
        factors.append((1 - meta.mean_correlation) * 0.1)

        # Outlier factor
        factors.append(min(meta.outlier_ratio * 5, 1) * 0.1)

        # Landmark factor (low landmark score = harder)
        factors.append((1 - meta.landmark_score) * 0.25)

        return float(np.clip(sum(factors), 0, 1))

    def get_recommended_trials(
        self,
        base_trials: int = 30,
        max_trials: int = 100,
    ) -> int:
        """
        Recommends number of optimization trials based on difficulty.

        Parameters
        ----------
        base_trials : int
            Minimum number of trials.
        max_trials : int
            Maximum number of trials.

        Returns
        -------
        int
            Recommended number of trials.
        """
        if self._meta_features is None:
            return base_trials

        difficulty = self._meta_features.difficulty_score
        recommended = int(base_trials + (max_trials - base_trials) * difficulty)
        return min(max(recommended, base_trials), max_trials)

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Returns recommendations based on meta-features.

        Returns
        -------
        dict
            Recommendations for model selection and preprocessing.
        """
        if self._meta_features is None:
            return {}

        meta = self._meta_features
        recs: Dict[str, Any] = {
            "difficulty": meta.difficulty_score,
            "recommended_trials": self.get_recommended_trials(),
            "suggestions": [],
        }

        if meta.class_imbalance_ratio > 5:
            recs["suggestions"].append(
                "High class imbalance detected. Consider using F1 or ROC-AUC metric."
            )

        if meta.missing_ratio > 0.1:
            recs["suggestions"].append(
                f"High missing ratio ({meta.missing_ratio:.1%}). "
                "Consider advanced imputation strategies."
            )

        if meta.outlier_ratio > 0.05:
            recs["suggestions"].append(
                f"High outlier ratio ({meta.outlier_ratio:.1%}). "
                "Consider using robust models like tree ensembles."
            )

        if meta.n_samples < 1000:
            recs["suggestions"].append(
                "Small dataset. Consider simpler models to avoid overfitting."
            )

        if meta.n_features > meta.n_samples * 0.5:
            recs["suggestions"].append(
                "High dimensional data. Consider feature selection."
            )

        return recs
