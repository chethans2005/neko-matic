"""Tests for SHAPExplainer and Meta-Learning modules."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from backend.explainability.shap_explainer import SHAPExplainer
from backend.meta_learning.dataset_difficulty import DatasetDifficultyAnalyzer
from backend.meta_learning.model_recommender import ModelRecommender


class TestSHAPExplainer:
    def test_shap_explainer_feature_importance(self, sample_classification_data):
        X = sample_classification_data.drop(columns=["target"])
        y = sample_classification_data["target"]

        num_features = ["feature_1", "feature_2"]
        preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_features)], remainder="drop")
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X, y)

        explainer = SHAPExplainer(pipeline, X, problem_type="classification")
        result = explainer.feature_importance(top_k=5)

        assert "feature_importance" in result
        assert isinstance(result["feature_importance"], list)
        assert len(result["feature_importance"]) > 0
        assert "feature" in result["feature_importance"][0]
        assert "importance" in result["feature_importance"][0]

    def test_shap_explainer_fallback_on_unsupported_model(self, sample_classification_data):
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))

        X = sample_classification_data.drop(columns=["target"])
        num_features = ["feature_1", "feature_2"]
        preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_features)], remainder="drop")
        preprocessor.fit(X)

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", DummyModel())])
        explainer = SHAPExplainer(pipeline, X, problem_type="classification")
        result = explainer.feature_importance(top_k=5)

        assert "feature_importance" in result
        assert len(result["feature_importance"]) > 0



class TestMetaLearning:
    def test_dataset_difficulty_score(self, sample_classification_data):
        meta_engine = DatasetDifficultyAnalyzer()
        score_info = meta_engine.analyze(sample_classification_data, target_column="target")

        assert "difficulty_score" in score_info
        assert 0.0 <= score_info["difficulty_score"] <= 1.0


    def test_model_recommender(self, sample_classification_data):
        recommender = ModelRecommender()
        recommendations = recommender.recommend(sample_classification_data, target_column="target", problem_type="classification", top_k=3)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert len(recommendations) > 0
