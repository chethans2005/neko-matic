"""Tests for model registry module."""

import pytest
from backend.core.model_registry import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry."""

    def test_registry_has_classification_models(self):
        """Test that registry contains classification models."""
        registry = ModelRegistry()
        assert len(registry.CLASSIFICATION_MODELS) > 0

    def test_registry_has_regression_models(self):
        """Test that registry contains regression models."""
        registry = ModelRegistry()
        assert len(registry.REGRESSION_MODELS) > 0

    def test_get_classification_models(self):
        """Test retrieving classification models."""
        registry = ModelRegistry()
        models = registry.get_models("classification")
        assert len(models) > 0
        assert all(isinstance(name, str) for name in models.keys())

    def test_get_regression_models(self):
        """Test retrieving regression models."""
        registry = ModelRegistry()
        models = registry.get_models("regression")
        assert len(models) > 0
        assert all(isinstance(name, str) for name in models.keys())

    def test_get_model_constructor_classification(self):
        """Test getting constructor for classification model."""
        registry = ModelRegistry()
        model = registry.get_model("RandomForestClassifier", "classification")
        assert model is not None
        assert callable(model)

    def test_get_model_constructor_regression(self):
        """Test getting constructor for regression model."""
        registry = ModelRegistry()
        model = registry.get_model("RandomForestRegressor", "regression")
        assert model is not None
        assert callable(model)

    def test_get_model_invalid_raises_error(self):
        """Test that getting invalid model raises ValueError."""
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown model"):
            registry.get_model("NonExistentModel", "classification")

    def test_get_hyperparameter_space_classification(self):
        """Test getting hyperparameter search space for classification."""
        registry = ModelRegistry()
        space = registry.get_optuna_space("RandomForestClassifier", "classification")
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_get_hyperparameter_space_regression(self):
        """Test getting hyperparameter search space for regression."""
        registry = ModelRegistry()
        space = registry.get_optuna_space("RandomForestRegressor", "regression")
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_model_spec_has_required_fields(self):
        """Test that model specs have all required fields."""
        registry = ModelRegistry()
        for name, spec in registry.CLASSIFICATION_MODELS.items():
            assert hasattr(spec, "name")
            assert hasattr(spec, "constructor")
            assert hasattr(spec, "default_params")
            assert hasattr(spec, "supports_gpu")

    def test_default_params_are_dicts(self):
        """Test that default params are dictionaries."""
        registry = ModelRegistry()
        all_models = {**registry.CLASSIFICATION_MODELS, **registry.REGRESSION_MODELS}
        for name, spec in all_models.items():
            assert isinstance(spec.default_params, dict)

    def test_model_constructor_callable(self):
        """Test that model constructors are callable."""
        registry = ModelRegistry()
        all_models = {**registry.CLASSIFICATION_MODELS, **registry.REGRESSION_MODELS}
        for name, spec in all_models.items():
            assert callable(spec.constructor)

    def test_gpu_support_is_bool(self):
        """Test that gpu support flag is boolean."""
        registry = ModelRegistry()
        all_models = {**registry.CLASSIFICATION_MODELS, **registry.REGRESSION_MODELS}
        for name, spec in all_models.items():
            assert isinstance(spec.supports_gpu, bool)
