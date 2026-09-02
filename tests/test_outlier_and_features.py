"""Tests for OutlierDetectionEngine and FeatureEngineeringEngine modules."""

import pytest
import pandas as pd
import numpy as np

from backend.core.outlier_detection import OutlierDetectionEngine
from backend.core.feature_engineering import FeatureEngineeringEngine


class TestOutlierDetectionEngine:
    def test_outlier_method_none(self, sample_classification_data):
        engine = OutlierDetectionEngine()
        result = engine.apply(sample_classification_data, target_column="target", config={"method": "none"})
        assert len(result) == len(sample_classification_data)

    def test_outlier_method_zscore(self, sample_classification_data):
        df = sample_classification_data.copy()
        # Add extreme outlier
        df.loc[0, "feature_1"] = 100.0

        engine = OutlierDetectionEngine()
        result = engine.apply(
            df,
            target_column="target",
            config={"method": "zscore", "threshold_parameters": {"zscore_threshold": 3.0}},
        )
        assert len(result) < len(df)

    def test_outlier_method_iqr(self, sample_classification_data):
        df = sample_classification_data.copy()
        df.loc[0, "feature_1"] = 100.0

        engine = OutlierDetectionEngine()
        result = engine.apply(
            df,
            target_column="target",
            config={"method": "iqr", "threshold_parameters": {"iqr_multiplier": 1.5}},
        )
        assert len(result) < len(df)

    def test_outlier_method_isolation_forest(self, sample_classification_data):
        engine = OutlierDetectionEngine()
        result = engine.apply(
            sample_classification_data,
            target_column="target",
            config={"method": "isolation_forest", "threshold_parameters": {"isolation_forest_contamination": 0.05}},
        )
        assert len(result) <= len(sample_classification_data)


class TestFeatureEngineeringEngine:
    def test_log_transform(self, sample_classification_data):
        df = sample_classification_data.copy()
        df["pos_feature"] = np.abs(df["feature_1"]) + 1.0
        original_val = df.loc[0, "pos_feature"]

        engine = FeatureEngineeringEngine()
        config = {"log_transform": True, "polynomial_features": False, "feature_interactions": False, "feature_selection": {"enabled": False}}
        result = engine.apply(df, target_column="target", config=config, problem_type="classification")

        assert result.loc[0, "pos_feature"] < original_val
        assert np.isclose(result.loc[0, "pos_feature"], np.log1p(original_val))


    def test_feature_interactions(self, sample_classification_data):
        engine = FeatureEngineeringEngine()
        config = {"log_transform": False, "polynomial_features": False, "feature_interactions": True, "feature_selection": {"enabled": False}}
        result = engine.apply(sample_classification_data, target_column="target", config=config, problem_type="classification")

        assert any("x" in col for col in result.columns if col != "target")

    def test_feature_selection(self, sample_classification_data):
        engine = FeatureEngineeringEngine()
        config = {
            "log_transform": False,
            "polynomial_features": False,
            "feature_interactions": False,
            "feature_selection": {"enabled": True, "method": "variance_threshold", "k_features": 2},
        }
        result = engine.apply(sample_classification_data, target_column="target", config=config, problem_type="classification")

        assert result.shape[1] <= sample_classification_data.shape[1]
