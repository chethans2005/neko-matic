"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from backend.core.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline."""

    def test_pipeline_creation_with_numeric_features(self):
        """Test that pipeline can be created with numeric features."""
        pipeline = PreprocessingPipeline(
            numerical_features=["feature_1", "feature_2"],
            categorical_features=[],
            scaler="standard",
        )
        p = pipeline.build_pipeline()
        assert p is not None

    def test_pipeline_creation_with_categorical_features(self):
        """Test that pipeline can be created with categorical features."""
        pipeline = PreprocessingPipeline(
            numerical_features=[],
            categorical_features=["feature_1"],
            scaler="standard",
        )
        p = pipeline.build_pipeline()
        assert p is not None

    def test_pipeline_creation_mixed_features(self):
        """Test that pipeline works with mixed feature types."""
        pipeline = PreprocessingPipeline(
            numerical_features=["num_1", "num_2"],
            categorical_features=["cat_1"],
            scaler="minmax",
        )
        p = pipeline.build_pipeline()
        assert p is not None

    def test_target_encoding_classification(self, sample_classification_data):
        """Test target encoding for classification tasks."""
        pipeline = PreprocessingPipeline(
            numerical_features=["feature_1", "feature_2"],
            categorical_features=["feature_3"],
            problem_type="classification",
        )
        y = sample_classification_data["target"]
        encoded = pipeline.fit_transform_target(y)
        assert len(encoded) == len(y)
        assert encoded.dtype in [np.int32, np.int64, int]

    def test_target_encoding_regression(self, sample_regression_data):
        """Test target encoding for regression tasks."""
        pipeline = PreprocessingPipeline(
            numerical_features=["feature_1", "feature_2", "feature_3"],
            categorical_features=[],
            problem_type="regression",
        )
        y = sample_regression_data["target"]
        encoded = pipeline.fit_transform_target(y)
        assert len(encoded) == len(y)
        np.testing.assert_array_almost_equal(encoded, y.values)

    def test_transform_target_after_fit(self, sample_classification_data):
        """Test that transform_target works after fit_transform_target."""
        pipeline = PreprocessingPipeline(
            numerical_features=["feature_1", "feature_2"],
            categorical_features=["feature_3"],
            problem_type="classification",
        )
        y = sample_classification_data["target"]
        _ = pipeline.fit_transform_target(y)
        transformed = pipeline.transform_target(y)
        assert len(transformed) == len(y)

    def test_pipeline_with_missing_values(self, sample_data_with_missing):
        """Test pipeline handles missing values."""
        pipeline = PreprocessingPipeline(
            numerical_features=["feature_1", "feature_2"],
            categorical_features=["feature_3"],
            num_impute_strategy="median",
            cat_impute_strategy="most_frequent",
        )
        p = pipeline.build_pipeline()
        X = sample_data_with_missing.drop("target", axis=1)
        assert p is not None

    def test_different_scalers(self, sample_classification_data):
        """Test that different scalers can be specified."""
        for scaler in ["standard", "minmax"]:
            pipeline = PreprocessingPipeline(
                numerical_features=["feature_1", "feature_2"],
                categorical_features=["feature_3"],
                scaler=scaler,
            )
            p = pipeline.build_pipeline()
            assert p is not None

    def test_empty_feature_lists(self):
        """Test pipeline creation with empty feature lists."""
        pipeline = PreprocessingPipeline(
            numerical_features=[],
            categorical_features=[],
        )
        # Should handle gracefully (with remainder='drop')
        p = pipeline.build_pipeline()
        assert p is not None
