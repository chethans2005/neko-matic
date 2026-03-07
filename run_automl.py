"""
FlexAutoML – CLI Entry Point
==============================
Run the complete AutoML pipeline from the command line without the UI.

Usage
-----
    python run_automl.py <data_path> <target_column> [options]

Examples
--------
    # Basic classification run
    python run_automl.py data/titanic.csv Survived

    # Regression with custom config
    python run_automl.py data/housing.csv price --config my_config.yaml

    # Select specific models
    python run_automl.py data/iris.csv species --models RandomForestClassifier XGBClassifier

    # Specify metric and number of trials
    python run_automl.py data/iris.csv species --metric f1 --trials 50
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the project root is importable when run from any working directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from flexautoml.core.leaderboard import Leaderboard
from flexautoml.core.model_registry import CLASSIFICATION_MODELS, REGRESSION_MODELS
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.core.profiler import DatasetProfiler
from flexautoml.core.trainer import AutoMLTrainer
from flexautoml.utils.config_loader import ConfigLoader


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpu() -> bool:
    """Returns True if an NVIDIA GPU is available on the current machine."""
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------


def run_automl(
    data_path: str,
    target_col: str,
    config_path: str | None = None,
    output_dir: str = "output",
    models: list[str] | None = None,
    metric: str | None = None,
    n_trials: int | None = None,
) -> tuple[AutoMLTrainer, pd.DataFrame]:
    """
    Executes the full FlexAutoML pipeline from data loading to model export.

    Parameters
    ----------
    data_path : str
        Path to the input CSV / Excel file.
    target_col : str
        Name of the column to predict.
    config_path : str | None
        Optional path to a YAML configuration file.
    output_dir : str
        Directory for writing the leaderboard CSV and best model `.pkl`.
    models : list[str] | None
        Override ``selected_models`` in the config.
    metric : str | None
        Override ``training.metric`` in the config.
    n_trials : int | None
        Override ``training.n_trials`` in the config.

    Returns
    -------
    tuple[AutoMLTrainer, pd.DataFrame]
        The fitted trainer object and the leaderboard DataFrame.
    """
    print("=" * 62)
    print("  FlexAutoML – Automated Machine Learning Pipeline")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    cfg = ConfigLoader(config_path)
    if models:
        cfg.set("training.selected_models", models)
    if metric:
        cfg.set("training.metric", metric)
    if n_trials is not None:
        cfg.set("training.n_trials", n_trials)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\n📂 Loading dataset: {data_path}")
    path_lower = data_path.lower()
    if path_lower.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif path_lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    print(f"   Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Profile dataset
    # ------------------------------------------------------------------
    print("\n📊 Profiling dataset…")
    profiler = DatasetProfiler(df, target_col)
    profile = profiler.profile_dataset()
    problem_type: str = profile["problem_type"]
    print(profiler.get_summary())

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------
    gpu_cfg = cfg.get("gpu.use_gpu", "auto")
    if gpu_cfg == "auto":
        use_gpu = detect_gpu()
    else:
        use_gpu = bool(gpu_cfg)
    print(f"\n🎮 GPU acceleration: {'Enabled' if use_gpu else 'Disabled (CPU)'}")

    # ------------------------------------------------------------------
    # Build preprocessing pipeline
    # ------------------------------------------------------------------
    print("\n🔧 Configuring preprocessing pipeline…")
    feat = profile["feature_types"]
    numerical_feats = [c for c in feat.get("numerical", []) if c != target_col]
    categorical_feats = [c for c in feat.get("categorical", []) if c != target_col]
    p_cfg = cfg.preprocessing

    preprocessing = PreprocessingPipeline(
        numerical_features=numerical_feats,
        categorical_features=categorical_feats,
        scaler=p_cfg.get("scaler", "standard"),
        num_impute_strategy=p_cfg.get("num_impute_strategy", "median"),
        cat_impute_strategy=p_cfg.get("cat_impute_strategy", "most_frequent"),
        use_feature_selection=p_cfg.get("use_feature_selection", False),
        k_best_features=p_cfg.get("k_best_features", 10),
        problem_type=problem_type,
    )
    print(f"   Preprocessing config: {preprocessing.get_config()}")

    # ------------------------------------------------------------------
    # Resolve models
    # ------------------------------------------------------------------
    t_cfg = cfg.training
    selected_models: list[str] = cfg.get("training.selected_models") or list(
        CLASSIFICATION_MODELS.keys()
        if problem_type == "classification"
        else REGRESSION_MODELS.keys()
    )
    eff_metric: str = cfg.get("training.metric", "accuracy" if problem_type == "classification" else "rmse")
    eff_trials: int = cfg.get("training.n_trials", 30)
    cv_folds: int = t_cfg.get("cv_folds", 5)
    test_size: float = t_cfg.get("test_size", 0.2)
    random_state: int = t_cfg.get("random_state", 42)

    print(f"\n🚀 AutoML Training")
    print(f"   Models  : {', '.join(selected_models)}")
    print(f"   Metric  : {eff_metric}")
    print(f"   Trials  : {eff_trials}")
    print(f"   CV folds: {cv_folds}")
    print()

    # ------------------------------------------------------------------
    # Progress callback
    # ------------------------------------------------------------------
    def _progress(model_name: str, pct: float) -> None:
        if model_name != "Done":
            print(f"   [{pct:5.1f}%] ▶  Training {model_name}…")
        else:
            print("   [100.0%] ✓  All models complete\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    trainer = AutoMLTrainer(
        problem_type=problem_type,
        preprocessing_pipeline=preprocessing,
        selected_models=selected_models,
        metric=eff_metric,
        n_trials=eff_trials,
        cv_folds=cv_folds,
        test_size=test_size,
        use_gpu=use_gpu,
        random_state=random_state,
        progress_callback=_progress,
    )
    results = trainer.train_all(X, y)

    # ------------------------------------------------------------------
    # Build leaderboard
    # ------------------------------------------------------------------
    lb = Leaderboard(problem_type=problem_type, primary_metric=eff_metric)
    leaderboard_df = lb.build(results)

    print("🏆 Leaderboard")
    print("-" * 62)
    print(leaderboard_df.to_string())
    print()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    lb_path = os.path.join(output_dir, "leaderboard.csv")
    leaderboard_df.to_csv(lb_path)
    print(f"💾 Leaderboard saved → {lb_path}")

    best_path = trainer.save_best_model(output_dir)
    if best_path:
        print(f"🤖 Best model saved → {best_path}")

    best = trainer.get_best_model()
    if best:
        print(f"\n✅ Best Model : {best.model_name}")
        print(f"   {eff_metric:10s}: {best.metrics.get(eff_metric, 'N/A')}")

    print("\n" + "=" * 62)
    print("  FlexAutoML run complete!")
    print("=" * 62)

    return trainer, leaderboard_df


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_automl.py",
        description="FlexAutoML – run the full AutoML pipeline from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_automl.py data/titanic.csv Survived
  python run_automl.py data/housing.csv price --metric rmse --trials 50
  python run_automl.py data/iris.csv species --models RandomForestClassifier XGBClassifier
        """,
    )
    parser.add_argument("data_path", help="Path to the input CSV / Excel file.")
    parser.add_argument("target", help="Name of the target column.")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML configuration file (overrides defaults).",
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Directory for leaderboard CSV and model .pkl files (default: output/).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", default=None,
        help="One or more model names to train (overrides config).",
    )
    parser.add_argument(
        "--metric", default=None,
        help="Primary evaluation metric, e.g. accuracy, f1, rmse (overrides config).",
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=None,
        help="Number of Optuna trials per model (overrides config).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_automl(
        data_path=args.data_path,
        target_col=args.target,
        config_path=args.config,
        output_dir=args.output,
        models=args.models,
        metric=args.metric,
        n_trials=args.trials,
    )


if __name__ == "__main__":
    main()
