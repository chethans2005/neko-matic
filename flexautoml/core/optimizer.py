"""
Hyperparameter Optimisation Engine
=====================================
Uses Optuna (TPE sampler + Median Pruner) to search each model's hyperparameter
space via k-fold cross-validation. The full preprocessing pipeline is embedded
inside every CV fold to prevent data leakage.

Key design choices
------------------
* Optuna is always set to *minimise* internally. Maximisation metrics are
  negated before being returned to Optuna and un-negated when stored.
* MedianPruner is used for early stopping of unpromising trials.
* A ``StratifiedKFold`` is used for classification; plain ``KFold`` for
  regression.
* Structured logging replaces verbose output.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from flexautoml.core.model_registry import ModelRegistry, get_hyperparameter_space
from flexautoml.utils.exceptions import OptimizationError
from flexautoml.utils.logging import get_logger, OptimizationLogger

logger = get_logger(__name__)

# Silence Optuna's per-trial output globally
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Optimization Result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Container for hyperparameter optimization results."""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    n_pruned: int
    optimization_time: float
    all_trials: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "optimization_time": self.optimization_time,
        }


# ---------------------------------------------------------------------------
# Pruner Factory
# ---------------------------------------------------------------------------


class PrunerFactory:
    """Factory for creating Optuna pruners."""

    @staticmethod
    def create(
        pruner_type: str = "median",
        n_startup_trials: int = 5,
        n_warmup_steps: int = 3,
        **kwargs,
    ) -> optuna.pruners.BasePruner:
        """
        Creates an Optuna pruner.

        Parameters
        ----------
        pruner_type : str
            Type of pruner: "median", "percentile", "hyperband", or "none".
        n_startup_trials : int
            Number of trials before pruning starts.
        n_warmup_steps : int
            Number of steps before pruning a trial.

        Returns
        -------
        optuna.pruners.BasePruner
            Configured pruner instance.
        """
        if pruner_type == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )
        elif pruner_type == "percentile":
            percentile = kwargs.get("percentile", 25.0)
            return optuna.pruners.PercentilePruner(
                percentile=percentile,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )
        elif pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=kwargs.get("max_resource", 10),
            )
        elif pruner_type == "none" or pruner_type is None:
            return optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner type '{pruner_type}', using MedianPruner")
            return optuna.pruners.MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )


class HyperparameterOptimizer:
    """
    Wraps an Optuna study to optimise a single model's hyperparameters.

    Parameters
    ----------
    model_name : str
        Registry key of the model to optimise.
    problem_type : str
        ``"classification"`` or ``"regression"``.
    metric : str
        Primary evaluation metric (``"accuracy"``, ``"f1"``, ``"roc_auc"``,
        ``"rmse"``, ``"mae"``, ``"r2"``, …).
    n_trials : int
        Number of Optuna trials to execute.
    cv_folds : int
        Number of cross-validation folds.
    timeout : int | None
        Wall-clock time limit in seconds (``None`` = unlimited).
    use_gpu : bool
        Passed to ``ModelRegistry`` for GPU-aware instantiation.
    random_state : int
        Seed for reproducibility (Optuna sampler + CV splits).
    pruner_type : str
        Type of Optuna pruner: "median", "percentile", "hyperband", or "none".
    """

    # Maps our metric names → sklearn scorer strings
    _SKLEARN_METRIC_MAP: Dict[str, str] = {
        "accuracy":   "accuracy",
        "f1":         "f1_weighted",
        "f1_score":   "f1_weighted",
        "roc_auc":    "roc_auc",
        "precision":  "precision_weighted",
        "recall":     "recall_weighted",
        "rmse":       "neg_root_mean_squared_error",
        "mae":        "neg_mean_absolute_error",
        "r2":         "r2",
    }

    # Metrics where a lower raw value is better
    _MINIMISE_METRICS = {"rmse", "mae", "mse", "neg_mean_squared_error",
                         "neg_mean_absolute_error"}

    def __init__(
        self,
        model_name: str,
        problem_type: str,
        metric: str = "accuracy",
        n_trials: int = 30,
        cv_folds: int = 5,
        timeout: Optional[int] = None,
        use_gpu: bool = False,
        random_state: int = 42,
        pruner_type: str = "median",
    ) -> None:
        self.model_name = model_name
        self.problem_type = problem_type
        self.metric = metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.pruner_type = pruner_type

        self.registry = ModelRegistry(problem_type, use_gpu)
        self.best_params: Dict[str, Any] = {}
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.Study] = None
        self._opt_logger = OptimizationLogger(model_name)
        
        logger.info(
            f"Initialized optimizer for {model_name} "
            f"(metric={metric}, n_trials={n_trials}, pruner={pruner_type})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cv(self, y: np.ndarray):
        """Returns a fitted CV splitter appropriate for the problem type."""
        if self.problem_type == "classification":
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        return KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

    def _is_minimise(self) -> bool:
        """True when lower metric values are better."""
        return self.metric.lower() in self._MINIMISE_METRICS

    def _build_objective(
        self,
        X: Any,
        y: np.ndarray,
        preprocessing_pipeline: Pipeline,
    ) -> Callable[[optuna.Trial], float]:
        """
        Returns an Optuna objective function that embeds the preprocessing
        pipeline inside each CV fold to prevent data leakage.
        """
        cv = self._get_cv(y)
        scoring = self._SKLEARN_METRIC_MAP.get(
            self.metric.lower(), self.metric
        )
        minimise = self._is_minimise()

        def objective(trial: optuna.Trial) -> float:
            params = get_hyperparameter_space(
                trial, self.model_name, self.problem_type
            )
            model = self.registry.get_model(self.model_name, params)

            full_pipeline = Pipeline([
                ("preprocessor", preprocessing_pipeline),
                ("model", model),
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    full_pipeline, X, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )

            mean_score = float(np.mean(scores))
            # Optuna always minimises; negate maximisation metrics
            return mean_score if minimise else -mean_score

        return objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        X: Any,
        y: np.ndarray,
        preprocessing_pipeline: Pipeline,
    ) -> OptimizationResult:
        """
        Runs Optuna hyperparameter search.

        Parameters
        ----------
        X : array-like
            Raw (un-preprocessed) feature matrix.
        y : np.ndarray
            Target vector (already encoded for classification).
        preprocessing_pipeline : sklearn.pipeline.Pipeline
            A *fitted* preprocessing pipeline that will be prepended to
            the model in each CV fold.

        Returns
        -------
        OptimizationResult
            Container with best params, score, and trial history.

        Raises
        ------
        OptimizationError
            If optimization fails due to all trials being pruned or errors.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting optimization for {self.model_name}")
        self._opt_logger.start()

        try:
            objective = self._build_objective(X, y, preprocessing_pipeline)

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            pruner = PrunerFactory.create(
                pruner_type=self.pruner_type,
                n_startup_trials=5,
                n_warmup_steps=min(3, self.cv_folds - 1),
            )
            
            self.study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )

            # Add callback for logging
            def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    # Un-negate score for display
                    display_score = (
                        trial.value if self._is_minimise() else -trial.value
                    )
                    self._opt_logger.log_trial(
                        trial.number + 1,
                        self.n_trials,
                        display_score,
                        is_best=(trial == study.best_trial),
                    )
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    logger.debug(f"Trial {trial.number} pruned")

            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
                callbacks=[log_callback],
            )

            # Check if we have any completed trials
            completed_trials = [
                t for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                raise OptimizationError(
                    f"All {len(self.study.trials)} trials failed or were pruned "
                    f"for {self.model_name}"
                )

            self.best_params = self.study.best_params
            raw = self.study.best_value
            # Un-negate if the metric was flipped for Optuna
            self.best_score = raw if self._is_minimise() else -raw

            elapsed = time.time() - start_time
            n_pruned = len([
                t for t in self.study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ])
            
            self._opt_logger.finish(self.best_score, elapsed)
            
            logger.info(
                f"Optimization complete for {self.model_name}: "
                f"best_score={self.best_score:.4f}, "
                f"trials={len(self.study.trials)}, pruned={n_pruned}"
            )

            return OptimizationResult(
                model_name=self.model_name,
                best_params=self.best_params,
                best_score=self.best_score,
                n_trials=len(self.study.trials),
                n_pruned=n_pruned,
                optimization_time=elapsed,
                all_trials=self.get_optimization_history(),
            )

        except Exception as e:
            logger.error(f"Optimization failed for {self.model_name}: {e}")
            raise OptimizationError(
                f"Optimization failed for {self.model_name}: {e}"
            ) from e

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Returns a list of completed trial records for plotting / inspection.

        Each entry contains ``{"trial": int, "value": float, "params": dict}``.
        """
        if self.study is None:
            return []
        return [
            {"trial": t.number, "value": t.value, "params": t.params}
            for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
