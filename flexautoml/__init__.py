"""
FlexAutoML
==========
A configurable Automated Machine Learning framework for tabular datasets.

Supports classification and regression with automatic preprocessing,
hyperparameter optimization via Optuna, and an interactive Streamlit UI.

Quick Start
-----------
>>> from flexautoml import AutoMLPipeline
>>> pipeline = AutoMLPipeline()
>>> result = pipeline.fit(df, target_column="target")
>>> predictions = pipeline.predict(new_data)
"""

__version__ = "1.0.0"
__author__ = "FlexAutoML"

# Main API
from flexautoml.core.pipeline import AutoMLPipeline, PipelineResult
from flexautoml.core.trainer import AutoMLTrainer, TrainingResult
from flexautoml.core.model_registry import ModelRegistry
from flexautoml.core.optimizer import HyperparameterOptimizer
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.core.profiler import DatasetProfiler
from flexautoml.core.evaluator import ModelEvaluator
from flexautoml.core.leaderboard import Leaderboard
from flexautoml.utils.config_loader import ConfigLoader

# Exceptions
from flexautoml.utils.exceptions import (
    FlexAutoMLError,
    DatasetError,
    ModelError,
    ConfigurationError,
)

__all__ = [
    # Main API
    "AutoMLPipeline",
    "PipelineResult",
    "AutoMLTrainer",
    "TrainingResult",
    "HyperparameterOptimizer",
    "ModelRegistry",
    "PreprocessingPipeline",
    "DatasetProfiler",
    "ModelEvaluator",
    "Leaderboard",
    "ConfigLoader",
    # Exceptions
    "FlexAutoMLError",
    "DatasetError",
    "ModelError", 
    "ConfigurationError",
]
