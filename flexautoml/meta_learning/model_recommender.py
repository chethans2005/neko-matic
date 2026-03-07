"""
Model Recommender
==================
Uses dataset meta-features to recommend which models are likely
to perform well, reducing the search space for hyperparameter optimization.

The recommender uses heuristics based on:
  - Dataset size and dimensionality
  - Feature types (numerical vs categorical)
  - Class imbalance
  - Dataset difficulty score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from flexautoml.meta_learning.dataset_difficulty import (
    DatasetDifficultyEstimator,
    DatasetMetaFeatures,
)
from flexautoml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRecommendation:
    """
    A single model recommendation with priority and reasoning.

    Attributes
    ----------
    model_name : str
        Name of the recommended model.
    priority : int
        Priority ranking (1 = highest).
    score : float
        Suitability score (0-1).
    reasons : list[str]
        Reasons for the recommendation.
    """

    model_name: str
    priority: int
    score: float
    reasons: List[str]


class ModelRecommender:
    """
    Recommends models based on dataset characteristics.

    Uses meta-features to prioritize models that are likely to perform well,
    enabling more efficient hyperparameter search.

    Parameters
    ----------
    problem_type : str
        "classification" or "regression".

    Examples
    --------
    >>> recommender = ModelRecommender("classification")
    >>> recommendations = recommender.recommend(meta_features, top_k=5)
    >>> for rec in recommendations:
    ...     print(f"{rec.model_name}: {rec.score:.2f}")
    """

    # Model characteristics for ranking
    _CLASSIFICATION_MODELS: Dict[str, Dict] = {
        "LGBMClassifier": {
            "handles_categorical": True,
            "handles_missing": True,
            "scales_well": True,
            "handles_imbalance": True,
            "fast": True,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "XGBClassifier": {
            "handles_categorical": False,
            "handles_missing": True,
            "scales_well": True,
            "handles_imbalance": True,
            "fast": True,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "RandomForestClassifier": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "handles_imbalance": True,
            "fast": True,
            "interpretable": True,
            "min_samples": 50,
            "complexity": "medium",
        },
        "GradientBoostingClassifier": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": False,
            "handles_imbalance": True,
            "fast": False,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "LogisticRegression": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "handles_imbalance": False,
            "fast": True,
            "interpretable": True,
            "min_samples": 20,
            "complexity": "low",
        },
        "SVC": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": False,
            "handles_imbalance": False,
            "fast": False,
            "interpretable": False,
            "min_samples": 50,
            "complexity": "high",
        },
        "KNeighborsClassifier": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": False,
            "handles_imbalance": False,
            "fast": False,
            "interpretable": True,
            "min_samples": 30,
            "complexity": "low",
        },
        "GaussianNB": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "handles_imbalance": False,
            "fast": True,
            "interpretable": True,
            "min_samples": 10,
            "complexity": "low",
        },
    }

    _REGRESSION_MODELS: Dict[str, Dict] = {
        "LGBMRegressor": {
            "handles_categorical": True,
            "handles_missing": True,
            "scales_well": True,
            "fast": True,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "XGBRegressor": {
            "handles_categorical": False,
            "handles_missing": True,
            "scales_well": True,
            "fast": True,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "RandomForestRegressor": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "fast": True,
            "interpretable": True,
            "min_samples": 50,
            "complexity": "medium",
        },
        "GradientBoostingRegressor": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": False,
            "fast": False,
            "interpretable": False,
            "min_samples": 100,
            "complexity": "high",
        },
        "Ridge": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "fast": True,
            "interpretable": True,
            "min_samples": 20,
            "complexity": "low",
        },
        "Lasso": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "fast": True,
            "interpretable": True,
            "min_samples": 20,
            "complexity": "low",
        },
        "LinearRegression": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": True,
            "fast": True,
            "interpretable": True,
            "min_samples": 10,
            "complexity": "low",
        },
        "SVR": {
            "handles_categorical": False,
            "handles_missing": False,
            "scales_well": False,
            "fast": False,
            "interpretable": False,
            "min_samples": 50,
            "complexity": "high",
        },
    }

    def __init__(self, problem_type: str = "classification") -> None:
        self.problem_type = problem_type
        self._models = (
            self._CLASSIFICATION_MODELS
            if problem_type == "classification"
            else self._REGRESSION_MODELS
        )

    def recommend(
        self,
        meta_features: DatasetMetaFeatures,
        top_k: Optional[int] = None,
        exclude_models: Optional[List[str]] = None,
    ) -> List[ModelRecommendation]:
        """
        Generates model recommendations based on meta-features.

        Parameters
        ----------
        meta_features : DatasetMetaFeatures
            Computed meta-features for the dataset.
        top_k : int | None
            Number of top models to return. None = all models.
        exclude_models : list[str] | None
            Models to exclude from recommendations.

        Returns
        -------
        list[ModelRecommendation]
            Sorted list of recommendations (highest priority first).
        """
        exclude = set(exclude_models or [])
        recommendations: List[ModelRecommendation] = []

        for model_name, traits in self._models.items():
            if model_name in exclude:
                continue

            score, reasons = self._compute_suitability(
                model_name, traits, meta_features
            )

            recommendations.append(
                ModelRecommendation(
                    model_name=model_name,
                    priority=0,  # Assigned after sorting
                    score=score,
                    reasons=reasons,
                )
            )

        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Assign priorities
        for i, rec in enumerate(recommendations):
            rec.priority = i + 1

        if top_k is not None:
            recommendations = recommendations[:top_k]

        logger.info(
            f"Top recommendations: "
            f"{[r.model_name for r in recommendations[:3]]}"
        )

        return recommendations

    def _compute_suitability(
        self,
        model_name: str,
        traits: Dict,
        meta: DatasetMetaFeatures,
    ) -> Tuple[float, List[str]]:
        """
        Computes suitability score and reasons for a model.

        Returns
        -------
        tuple[float, list[str]]
            Score (0-1) and list of reasons.
        """
        score = 0.5  # Base score
        reasons: List[str] = []

        # Sample size compatibility
        if meta.n_samples < traits.get("min_samples", 10):
            score -= 0.2
            reasons.append(f"Dataset may be too small ({meta.n_samples} samples)")
        elif meta.n_samples > 10000 and traits.get("scales_well"):
            score += 0.1
            reasons.append("Scales well with large datasets")

        # Categorical feature handling
        if meta.n_categorical > meta.n_numerical:
            if traits.get("handles_categorical"):
                score += 0.15
                reasons.append("Good categorical feature handling")
            else:
                score -= 0.1

        # Missing value handling
        if meta.missing_ratio > 0.05:
            if traits.get("handles_missing"):
                score += 0.1
                reasons.append("Native missing value support")
            else:
                score -= 0.05

        # Class imbalance (classification only)
        if self.problem_type == "classification":
            if meta.class_imbalance_ratio > 3:
                if traits.get("handles_imbalance"):
                    score += 0.15
                    reasons.append("Handles class imbalance well")
                else:
                    score -= 0.1

        # Dataset complexity
        difficulty = meta.difficulty_score
        complexity = traits.get("complexity", "medium")

        if difficulty > 0.6 and complexity == "high":
            score += 0.15
            reasons.append("Complex model for difficult dataset")
        elif difficulty < 0.3 and complexity == "low":
            score += 0.1
            reasons.append("Simple model for easy dataset")
        elif difficulty < 0.3 and complexity == "high":
            score -= 0.1
            reasons.append("May overfit on simple dataset")

        # Speed bonus for fast models
        if traits.get("fast"):
            score += 0.05
            reasons.append("Fast training")

        # Interpretability bonus
        if traits.get("interpretable"):
            score += 0.03

        # Boost tree-based ensembles slightly (empirically strong)
        if "Forest" in model_name or "GB" in model_name or "GBM" in model_name:
            score += 0.08

        # Clip score to [0, 1]
        score = max(0.0, min(1.0, score))

        return score, reasons

    def get_quick_recommendations(
        self, meta_features: DatasetMetaFeatures
    ) -> List[str]:
        """
        Returns a quick list of recommended model names.

        Parameters
        ----------
        meta_features : DatasetMetaFeatures
            Dataset meta-features.

        Returns
        -------
        list[str]
            Top 5 recommended model names.
        """
        recs = self.recommend(meta_features, top_k=5)
        return [r.model_name for r in recs]

    def explain_recommendation(
        self, meta_features: DatasetMetaFeatures
    ) -> str:
        """
        Generates a human-readable explanation of recommendations.

        Parameters
        ----------
        meta_features : DatasetMetaFeatures
            Dataset meta-features.

        Returns
        -------
        str
            Formatted explanation text.
        """
        recs = self.recommend(meta_features, top_k=5)

        lines = [
            f"Dataset: {meta_features.n_samples} samples × {meta_features.n_features} features",
            f"Difficulty: {meta_features.difficulty_score:.2f}",
            "",
            "Top Model Recommendations:",
            "-" * 40,
        ]

        for rec in recs:
            lines.append(f"{rec.priority}. {rec.model_name} (score: {rec.score:.2f})")
            for reason in rec.reasons[:2]:  # Show top 2 reasons
                lines.append(f"   • {reason}")

        return "\n".join(lines)
