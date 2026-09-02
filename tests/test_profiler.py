"""Tests for DataProfiler core module."""

import pytest
import pandas as pd
import numpy as np
from backend.core.profiler import DataProfiler


class TestDataProfiler:
    def test_profiler_classification_analysis(self, sample_classification_data):
        profiler = DataProfiler()
        profile = profiler.analyze(sample_classification_data, target_column="target")

        assert profile["problem_type"] == "classification"
        assert profile["target_col"] == "target"
        assert "feature_types" in profile
        assert "numerical" in profile["feature_types"]
        assert "categorical" in profile["feature_types"]
        assert "missing_info" in profile
        assert "class_distribution" in profile
        assert "correlation_matrix" in profile

    def test_profiler_regression_analysis(self, sample_regression_data):
        profiler = DataProfiler()
        profile = profiler.analyze(sample_regression_data, target_column="target")

        assert profile["problem_type"] == "regression"
        assert profile["target_col"] == "target"
        assert "class_distribution" not in profile


    def test_profiler_missing_values_calculation(self, sample_data_with_missing):
        profiler = DataProfiler()
        profile = profiler.analyze(sample_data_with_missing, target_column="target")

        missing_dict = {item["column"]: item["missing_count"] for item in profile["missing_info"]}
        assert missing_dict["feature_1"] > 0
        assert missing_dict["feature_2"] > 0
