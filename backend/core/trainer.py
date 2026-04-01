from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from backend.core.feature_engineering import FeatureEngineeringEngine
from backend.core.leaderboard import LeaderboardManager
from backend.core.model_registry import ModelRegistry
from backend.core.outlier_detection import OutlierDetectionEngine
from backend.core.preprocessing import PreprocessingEngine
from backend.core.profiler import DataProfiler
from backend.explainability.shap_explainer import SHAPExplainer
from backend.meta_learning.model_recommender import ModelRecommender
from backend.utils.config_loader import BackendConfigLoader
from backend.utils.logging import get_logger
from backend.core.automl_trainer import AutoMLTrainer

logger = get_logger(__name__)


@dataclass
class RunRecord:
    run_id: str
    dataset_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    problem_type: Optional[str] = None
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    best_model_name: Optional[str] = None
    best_score: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    pipeline_path: Optional[str] = None
    training_report_path: Optional[str] = None
    feature_importance_path: Optional[str] = None


class TrainingEngine:
    """Owns uploaded assets and executes AutoML runs in background threads.
    
    V2 Design: Single active dataset + config per session, single active run.
    Removes UUID-based dataset_id/run_id; uses session-scoped active state instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # Legacy support (for backward compatibility if needed)
        self.datasets: Dict[str, str] = {}
        self.dataset_profiles: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.runs: Dict[str, RunRecord] = {}
        
        # V2: Active dataset/config per session
        self.active_dataset_path: Optional[str] = None
        self.active_dataset_profile: Optional[Dict[str, Any]] = None
        self.active_config: Optional[Dict[str, Any]] = None
        self.default_config: Optional[Dict[str, Any]] = None
        self.active_run: Optional[RunRecord] = None

        self.registry = ModelRegistry()
        self.profiler = DataProfiler()
        self.preprocessing_engine = PreprocessingEngine()
        self.outlier_engine = OutlierDetectionEngine()
        self.feature_engineering_engine = FeatureEngineeringEngine()
        self.recommender = ModelRecommender()

        self.models_dir = Path("models") / "saved_models"
        self.runs_dir = Path("models") / "runs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # V2 API: Active dataset/config management

    def set_active_dataset(self, dataset_path: str, profile: Dict[str, Any]) -> None:
        """Set the active dataset for this session."""
        with self._lock:
            self.active_dataset_path = dataset_path
            self.active_dataset_profile = profile

    def get_active_dataset_info(self) -> Dict[str, Any]:
        """Get info about the currently active dataset."""
        with self._lock:
            if not self.active_dataset_path:
                return {}
            return {
                "path": self.active_dataset_path,
                "profile": self.active_dataset_profile or {},
            }

    def set_active_config(self, config_payload: Dict[str, Any]) -> None:
        """Set the active config (for current run)."""
        loader = BackendConfigLoader("backend/configs/default.yaml")
        merged = loader.merge_dict(config_payload)
        with self._lock:
            self.active_config = merged

    def set_default_config(self, config_payload: Dict[str, Any]) -> None:
        """Set the default config (persisted for future runs)."""
        loader = BackendConfigLoader("backend/configs/default.yaml")
        merged = loader.merge_dict(config_payload)
        with self._lock:
            self.default_config = merged

    def get_default_config(self) -> Dict[str, Any]:
        """Get the current default config, or load system default if not set."""
        with self._lock:
            if self.default_config:
                return self.default_config.copy()
        loader = BackendConfigLoader("backend/configs/default.yaml")
        return loader.config

    def start_active_run(self, config_payload: Optional[Dict[str, Any]] = None) -> str:
        """Start AutoML run using active dataset and provided (or default) config.
        
        Returns run_id for backward compatibility.
        """
        if not self.active_dataset_path:
            raise ValueError("No active dataset set")

        # Use provided config or fall back to default
        if config_payload:
            self.set_active_config(config_payload)
        elif not self.active_config:
            self.set_active_config(self.get_default_config())

        run_id = str(uuid.uuid4())
        record = RunRecord(
            run_id=run_id,
            dataset_id="",  # Empty for V2 (uses active_dataset)
        )
        with self._lock:
            self.active_run = record
            self.runs[run_id] = record  # Keep for backward compat

        thread = threading.Thread(
            target=self._run_pipeline_active,
            args=(run_id,),
            daemon=True,
        )
        thread.start()
        return run_id

    def get_active_run_status(self) -> Dict[str, Any]:
        """Get status of active run (or empty dict if none active)."""
        with self._lock:
            if not self.active_run:
                return {"status": "none"}
            record = self.active_run
            return {
                "run_id": record.run_id,
                "status": record.status,
                "progress": record.progress,
                "message": record.message,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "problem_type": record.problem_type,
                "best_model_name": record.best_model_name,
                "best_score": record.best_score,
                "metrics": record.metrics,
                "leaderboard": record.leaderboard,
            }

    def register_dataset(self, dataset_path: str, profile: Dict[str, Any]) -> str:
        dataset_id = str(uuid.uuid4())
        with self._lock:
            self.datasets[dataset_id] = dataset_path
            self.dataset_profiles[dataset_id] = profile
        return dataset_id

    def register_config(self, config_payload: Dict[str, Any]) -> str:
        config_id = str(uuid.uuid4())
        loader = BackendConfigLoader("backend/configs/default.yaml")
        merged = loader.merge_dict(config_payload)
        with self._lock:
            self.configs[config_id] = merged
        return config_id

    def get_dataset_profile(self, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.dataset_profiles.get(dataset_id, {})

    def start_run(self, dataset_id: str, config_id: Optional[str]) -> str:
        run_id = str(uuid.uuid4())
        record = RunRecord(run_id=run_id, dataset_id=dataset_id)
        with self._lock:
            self.runs[run_id] = record

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(run_id, dataset_id, config_id),
            daemon=True,
        )
        thread.start()
        return run_id

    def get_status(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self.runs.get(run_id)
            if not record:
                return {"error": "run_not_found"}
            return {
                "run_id": record.run_id,
                "dataset_id": record.dataset_id,
                "status": record.status,
                "progress": record.progress,
                "message": record.message,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "problem_type": record.problem_type,
                "best_model_name": record.best_model_name,
                "best_score": record.best_score,
                "metrics": record.metrics,
                "leaderboard": record.leaderboard,
            }

    def get_leaderboard(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            record = self.runs.get(run_id)
            return record.leaderboard if record else []

    def get_feature_importance(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self.runs.get(run_id)
            if not record or not record.feature_importance_path:
                return {}
            path = Path(record.feature_importance_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def get_model_path(self, run_id: str) -> Optional[Path]:
        with self._lock:
            record = self.runs.get(run_id)
            if not record or not record.model_path:
                return None
            return Path(record.model_path)

    def get_export_paths(self, run_id: str) -> Dict[str, str]:
        with self._lock:
            record = self.runs.get(run_id)
            if not record:
                return {}
            payload = {}
            if record.model_path:
                payload["best_model.pkl"] = record.model_path
            if record.pipeline_path:
                payload["pipeline.pkl"] = record.pipeline_path
            if record.training_report_path:
                payload["training_report.json"] = record.training_report_path
            return payload

    def _update_run(self, run_id: str, **kwargs: Any) -> None:
        with self._lock:
            record = self.runs.get(run_id)
            if not record:
                return
            for key, value in kwargs.items():
                setattr(record, key, value)

    def _read_dataset(self, dataset_path: str) -> pd.DataFrame:
        path = Path(dataset_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    def _resolve_problem_type(self, profile: Dict[str, Any], config: Dict[str, Any]) -> str:
        override = config.get("dataset_settings", {}).get("problem_type_override")
        if override in {"classification", "regression"}:
            return override
        return str(profile.get("problem_type", "classification"))

    def _resolve_models(self, problem_type: str, config: Dict[str, Any], dataframe: pd.DataFrame, target: str) -> list[str]:
        selected = config.get("model_selection", {}).get("list_of_models_to_train")
        if selected:
            return selected
        recommendations = self.recommender.recommend(dataframe, target, problem_type, top_k=5)
        if recommendations:
            return recommendations
        return self.registry.list_models(problem_type)

    def _primary_metric_key(self, metric: str) -> str:
        mapping = {
            "f1 score": "f1_weighted",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        return mapping.get(metric.lower(), metric)

    def _apply_categorical_encoding(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        encoding: str,
    ) -> pd.DataFrame:
        if encoding != "label":
            return dataframe
        df = dataframe.copy()
        for column in df.columns:
            if column == target_column:
                continue
            if df[column].dtype == "object":
                df[column] = df[column].astype("category").cat.codes
        return df

    def _run_pipeline_active(self, run_id: str) -> None:
        """V2: Runs AutoML using active dataset and active config."""
        try:
            # Get active dataset/config
            with self._lock:
                dataset_path = self.active_dataset_path
                config = (self.active_config or self.get_default_config()).copy()
                dataset_id = ""  # V2: no dataset_id
            
            if not dataset_path:
                raise ValueError("No active dataset")

            self._update_run(
                run_id,
                status="running",
                progress=5,
                message="Loading dataset",
                started_at=datetime.utcnow().isoformat(),
            )

            dataframe = self._read_dataset(dataset_path)

            target_column = config.get("dataset_settings", {}).get("target_column")
            if not target_column:
                target_column = dataframe.columns[-1]
                config["dataset_settings"]["target_column"] = target_column
            if target_column not in dataframe.columns:
                raise ValueError(f"Target column '{target_column}' not in dataset")

            profile = self.profiler.analyze(dataframe, target_column)
            problem_type = self._resolve_problem_type(profile, config)
            self._update_run(run_id, problem_type=problem_type, progress=15, message="Profiling complete")

            dataframe = self.outlier_engine.apply(dataframe, target_column, config.get("outlier_removal", {}))
            dataframe = self.feature_engineering_engine.apply(
                dataframe,
                target_column,
                config.get("feature_engineering", {}),
                problem_type,
            )
            encoding = config.get("data_cleaning", {}).get("categorical_encoding", "onehot")
            dataframe = self._apply_categorical_encoding(dataframe, target_column, encoding)
            self._update_run(run_id, progress=30, message="Feature processing complete")

            feature_types = self.profiler.analyze(dataframe, target_column).get("feature_types", {})
            preprocessing = self.preprocessing_engine.build(
                numerical_features=feature_types.get("numerical", []),
                categorical_features=feature_types.get("categorical", []),
                config=config.get("data_cleaning", {}),
                problem_type=problem_type,
            )

            primary_metric = config.get("evaluation_metrics", {}).get("primary_metric", "accuracy")
            train_cfg = config.get("dataset_settings", {})
            hpo_cfg = config.get("hyperparameter_optimization", {})
            strategy_cfg = config.get("training_strategy", {})

            models = self._resolve_models(problem_type, config, dataframe, target_column)

            X = dataframe.drop(columns=[target_column])
            y = dataframe[target_column]

            use_gpu = strategy_cfg.get("gpu_usage", "auto")
            use_gpu = False if use_gpu in {"auto", False} else bool(use_gpu)

            trainer = AutoMLTrainer(
                problem_type=problem_type,
                preprocessing_pipeline=preprocessing,
                selected_models=models,
                metric=primary_metric,
                n_trials=int(hpo_cfg.get("number_of_trials", 20)),
                cv_folds=int(train_cfg.get("cross_validation_folds", 5)),
                test_size=float(train_cfg.get("train_test_split", 0.2)),
                use_gpu=use_gpu,
                random_state=42,
            )

            self._update_run(run_id, progress=45, message="Training models")
            results = trainer.train_all(X, y)

            leaderboard = LeaderboardManager(primary_metric=primary_metric, problem_type=problem_type)
            for result in results:
                leaderboard.add(result.model_name, result.metrics, result.training_time)
            leaderboard_records = leaderboard.as_records()

            best_result = trainer.get_best_model()
            metric_key = self._primary_metric_key(primary_metric)
            best_score = None
            if best_result is not None:
                best_score = best_result.metrics.get(metric_key)

            run_artifact_dir = self.runs_dir / run_id
            run_artifact_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = run_artifact_dir / "best_model.pkl"
            pipeline_path = run_artifact_dir / "pipeline.pkl"
            report_path = run_artifact_dir / "training_report.json"
            feature_importance_path = run_artifact_dir / "feature_importance.json"

            if best_result is not None:
                joblib.dump(best_result.model, best_model_path)
                joblib.dump(best_result.pipeline, pipeline_path)

            report_payload = {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "problem_type": problem_type,
                "target_column": target_column,
                "primary_metric": primary_metric,
                "models": models,
                "best_model": best_result.model_name if best_result else None,
                "best_metrics": best_result.metrics if best_result else {},
                "leaderboard": leaderboard_records,
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

            feature_payload: Dict[str, Any] = {"feature_importance": []}
            if config.get("explainability", {}).get("enable_shap", True) and best_result is not None:
                try:
                    explainer = SHAPExplainer(best_result.pipeline, X, problem_type)
                    feature_payload = explainer.feature_importance(top_k=30)
                except Exception as exc:
                    logger.warning("SHAP generation failed: %s", exc)
            feature_importance_path.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

            self._update_run(
                run_id,
                status="completed",
                progress=100,
                message="Training completed",
                ended_at=datetime.utcnow().isoformat(),
                leaderboard=leaderboard_records,
                best_model_name=best_result.model_name if best_result else None,
                best_score=best_score,
                metrics=best_result.metrics if best_result else {},
                model_path=str(best_model_path),
                pipeline_path=str(pipeline_path),
                training_report_path=str(report_path),
                feature_importance_path=str(feature_importance_path),
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            self._update_run(
                run_id,
                status="failed",
                progress=100,
                message=str(exc),
                ended_at=datetime.utcnow().isoformat(),
            )

    def _run_pipeline(self, run_id: str, dataset_id: str, config_id: Optional[str]) -> None:
        try:
            self._update_run(
                run_id,
                status="running",
                progress=5,
                message="Loading dataset",
                started_at=datetime.utcnow().isoformat(),
            )

            dataset_path = self.datasets[dataset_id]
            dataframe = self._read_dataset(dataset_path)

            if config_id and config_id in self.configs:
                config = self.configs[config_id]
            else:
                config = BackendConfigLoader("backend/configs/default.yaml").config

            target_column = config.get("dataset_settings", {}).get("target_column")
            if not target_column:
                target_column = dataframe.columns[-1]
                config["dataset_settings"]["target_column"] = target_column
            if target_column not in dataframe.columns:
                raise ValueError(f"Target column '{target_column}' not in dataset")

            profile = self.profiler.analyze(dataframe, target_column)
            problem_type = self._resolve_problem_type(profile, config)
            self._update_run(run_id, problem_type=problem_type, progress=15, message="Profiling complete")

            dataframe = self.outlier_engine.apply(dataframe, target_column, config.get("outlier_removal", {}))
            dataframe = self.feature_engineering_engine.apply(
                dataframe,
                target_column,
                config.get("feature_engineering", {}),
                problem_type,
            )
            encoding = config.get("data_cleaning", {}).get("categorical_encoding", "onehot")
            dataframe = self._apply_categorical_encoding(dataframe, target_column, encoding)
            self._update_run(run_id, progress=30, message="Feature processing complete")

            feature_types = self.profiler.analyze(dataframe, target_column).get("feature_types", {})
            preprocessing = self.preprocessing_engine.build(
                numerical_features=feature_types.get("numerical", []),
                categorical_features=feature_types.get("categorical", []),
                config=config.get("data_cleaning", {}),
                problem_type=problem_type,
            )

            primary_metric = config.get("evaluation_metrics", {}).get("primary_metric", "accuracy")
            train_cfg = config.get("dataset_settings", {})
            hpo_cfg = config.get("hyperparameter_optimization", {})
            strategy_cfg = config.get("training_strategy", {})

            models = self._resolve_models(problem_type, config, dataframe, target_column)

            X = dataframe.drop(columns=[target_column])
            y = dataframe[target_column]

            use_gpu = strategy_cfg.get("gpu_usage", "auto")
            use_gpu = False if use_gpu in {"auto", False} else bool(use_gpu)

            trainer = AutoMLTrainer(
                problem_type=problem_type,
                preprocessing_pipeline=preprocessing,
                selected_models=models,
                metric=primary_metric,
                n_trials=int(hpo_cfg.get("number_of_trials", 20)),
                cv_folds=int(train_cfg.get("cross_validation_folds", 5)),
                test_size=float(train_cfg.get("train_test_split", 0.2)),
                use_gpu=use_gpu,
                random_state=42,
            )

            self._update_run(run_id, progress=45, message="Training models")
            results = trainer.train_all(X, y)

            leaderboard = LeaderboardManager(primary_metric=primary_metric, problem_type=problem_type)
            for result in results:
                leaderboard.add(result.model_name, result.metrics, result.training_time)
            leaderboard_records = leaderboard.as_records()

            best_result = trainer.get_best_model()
            metric_key = self._primary_metric_key(primary_metric)
            best_score = None
            if best_result is not None:
                best_score = best_result.metrics.get(metric_key)

            run_artifact_dir = self.runs_dir / run_id
            run_artifact_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = run_artifact_dir / "best_model.pkl"
            pipeline_path = run_artifact_dir / "pipeline.pkl"
            report_path = run_artifact_dir / "training_report.json"
            feature_importance_path = run_artifact_dir / "feature_importance.json"

            if best_result is not None:
                joblib.dump(best_result.model, best_model_path)
                joblib.dump(best_result.pipeline, pipeline_path)

            report_payload = {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "problem_type": problem_type,
                "target_column": target_column,
                "primary_metric": primary_metric,
                "models": models,
                "best_model": best_result.model_name if best_result else None,
                "best_metrics": best_result.metrics if best_result else {},
                "leaderboard": leaderboard_records,
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

            feature_payload: Dict[str, Any] = {"feature_importance": []}
            if config.get("explainability", {}).get("enable_shap", True) and best_result is not None:
                try:
                    explainer = SHAPExplainer(best_result.pipeline, X, problem_type)
                    feature_payload = explainer.feature_importance(top_k=30)
                except Exception as exc:
                    logger.warning("SHAP generation failed: %s", exc)
            feature_importance_path.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

            self._update_run(
                run_id,
                status="completed",
                progress=100,
                message="Training completed",
                ended_at=datetime.utcnow().isoformat(),
                leaderboard=leaderboard_records,
                best_model_name=best_result.model_name if best_result else None,
                best_score=best_score,
                metrics=best_result.metrics if best_result else {},
                model_path=str(best_model_path),
                pipeline_path=str(pipeline_path),
                training_report_path=str(report_path),
                feature_importance_path=str(feature_importance_path),
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            self._update_run(
                run_id,
                status="failed",
                progress=100,
                message=str(exc),
                ended_at=datetime.utcnow().isoformat(),
            )


TRAINING_ENGINE = TrainingEngine()
