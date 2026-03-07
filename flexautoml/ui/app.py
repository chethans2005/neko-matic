"""
FlexAutoML – Streamlit Dashboard
==================================
Five-page interactive web UI:

  📊 Dataset Explorer    – upload, inspect, and profile your dataset
  ⚙️  AutoML Configuration – choose models, metrics, and preprocessing options
  🚀 Training Monitor    – launch and track the AutoML run
  🏆 Results             – leaderboard, metric charts, SHAP explainability
  💾 Model Export        – download any trained pipeline as a .pkl file

Run with:
    streamlit run flexautoml/ui/app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# Make sure the project root is on the path when running directly
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flexautoml.core.leaderboard import Leaderboard
from flexautoml.core.model_registry import CLASSIFICATION_MODELS, REGRESSION_MODELS
from flexautoml.core.pipeline import AutoMLPipeline
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.core.profiler import DatasetProfiler
from flexautoml.core.trainer import AutoMLTrainer
from flexautoml.utils.config_loader import ConfigLoader
from flexautoml.utils.exceptions import FlexAutoMLError
from flexautoml.utils.logging import get_logger

# Create UI-specific logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FlexAutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "df": None,
    "target_col": None,
    "profile": None,
    "problem_type": None,
    "training_results": None,
    "leaderboard_df": None,
    "trainer": None,
    "config": ConfigLoader(),
    "training_complete": False,
    "gpu_available": False,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def detect_gpu() -> bool:
    """Returns True when an NVIDIA GPU is accessible."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 FlexAutoML")
    st.caption("Automated Machine Learning for Tabular Data")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dataset Explorer",
            "⚙️ AutoML Configuration",
            "🚀 Training Monitor",
            "🏆 Results",
            "💾 Model Export",
        ],
        key="nav_page",
    )

    st.divider()

    # GPU status badge
    gpu_ok = detect_gpu()
    st.session_state["gpu_available"] = gpu_ok
    if gpu_ok:
        st.success("🎮 GPU Detected")
    else:
        st.info("💻 CPU mode")

    st.divider()
    st.caption("FlexAutoML v1.0.0")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown(
        "Upload a CSV / Excel file, pick your target column, "
        "and explore the automatic dataset profile."
    )

    # ── Upload / sample data ──────────────────────────────────────────
    col_up, col_sample = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx", "xls"],
            help="Supported: CSV, Excel (.xlsx / .xls)",
        )

    with col_sample:
        st.markdown("**Or load a sample dataset**")
        sample = st.selectbox(
            "Sample datasets",
            ["None", "Iris (Classification)", "California Housing (Regression)", "Titanic (Classification)"],
        )

    # Load data from upload
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state["df"] = df
            st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    # Load sample data
    elif sample != "None":
        from sklearn import datasets as _sk_ds  # local import to keep global ns clean

        if "Iris" in sample:
            _d = _sk_ds.load_iris(as_frame=True)
            df = pd.concat([_d.data, _d.target.rename("target")], axis=1)
        elif "California" in sample:
            _d = _sk_ds.fetch_california_housing(as_frame=True)
            df = _d.frame
        else:  # Titanic
            try:
                import seaborn as _sns  # type: ignore
                df = _sns.load_dataset("titanic")
            except Exception:
                st.error("Could not load Titanic; please upload your own CSV.")
                st.stop()

        st.session_state["df"] = df
        st.info(f"📦 Sample loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Main exploration ──────────────────────────────────────────────
    if st.session_state["df"] is None:
        st.info("👆 Upload a dataset or choose a sample above to get started.")
        st.stop()

    df = st.session_state["df"]

    st.divider()

    # Target & problem-type selectors
    sel_col, prob_col = st.columns([2, 1])
    with sel_col:
        target_col = st.selectbox(
            "🎯 Target column",
            df.columns.tolist(),
            index=len(df.columns) - 1,
        )
        st.session_state["target_col"] = target_col

    with prob_col:
        prob_override = st.selectbox(
            "Problem type override",
            ["Auto Detect", "Classification", "Regression"],
        )

    # Profile
    profiler = DatasetProfiler(df, target_col)
    profile = profiler.profile_dataset()
    if prob_override != "Auto Detect":
        profile["problem_type"] = prob_override.lower()

    st.session_state["profile"] = profile
    st.session_state["problem_type"] = profile["problem_type"]

    # ── Overview cards ────────────────────────────────────────────────
    st.subheader("📋 Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{profile['n_rows']:,}")
    c2.metric("Features", profile["n_features"])
    c3.metric("Numerical", profile["n_numerical"])
    c4.metric("Categorical", profile["n_categorical"])
    c5.metric("Problem", profile["problem_type"].upper())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Values", f"{profile['total_missing']:,}")
    c2.metric("Missing %", f"{profile['missing_pct']:.2f}%")
    c3.metric("Duplicates", f"{profile['duplicate_rows']:,}")
    c4.metric("Memory (MB)", f"{profile['memory_usage_mb']:.2f}")

    st.divider()

    # ── Tabbed views ──────────────────────────────────────────────────
    tab_prev, tab_stat, tab_miss, tab_dist, tab_corr = st.tabs(
        ["📄 Preview", "📈 Statistics", "🔍 Missing Values", "📊 Distributions", "🌡️ Correlation"]
    )

    with tab_prev:
        st.dataframe(df.head(200), use_container_width=True)
        st.caption(f"Showing first 200 of {len(df):,} rows")

    with tab_stat:
        st.dataframe(profile["describe"], use_container_width=True)

    with tab_miss:
        missing_df = profile["missing_info"]
        if missing_df.empty:
            st.success("✅ No missing values detected.")
        else:
            fig = px.bar(
                missing_df.reset_index().rename(columns={"index": "feature"}),
                x="feature",
                y="missing_pct",
                title="Missing Value % by Feature",
                color="missing_pct",
                color_continuous_scale="Reds",
                labels={"missing_pct": "Missing %"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(missing_df, use_container_width=True)

    with tab_dist:
        num_features = profile["feature_types"]["numerical"]
        if num_features:
            chosen = st.selectbox("Numerical feature to plot", num_features)
            l_col, r_col = st.columns(2)
            with l_col:
                fig = px.histogram(
                    df, x=chosen, nbins=50,
                    title=f"Histogram: {chosen}",
                    color_discrete_sequence=["steelblue"],
                )
                st.plotly_chart(fig, use_container_width=True)
            with r_col:
                fig = px.box(
                    df, y=chosen,
                    title=f"Box Plot: {chosen}",
                    color_discrete_sequence=["steelblue"],
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Target Distribution — {target_col}")
        if profile["problem_type"] == "classification":
            counts = df[target_col].value_counts()
            fig = px.bar(
                x=counts.index.astype(str), y=counts.values,
                title="Class Counts",
                labels={"x": "Class", "y": "Count"},
                color=counts.values,
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)
            dist = profile.get("class_distribution")
            if dist is not None and len(dist) >= 2:
                ratio = dist.iloc[0] / dist.iloc[-1]
                if ratio > 3:
                    st.warning(
                        f"⚠️ Class imbalance detected (majority/minority ratio ≈ {ratio:.1f}×). "
                        "Consider using F1 or ROC-AUC as the primary metric."
                    )
        else:
            fig = px.histogram(
                df, x=target_col, nbins=50,
                title=f"Distribution: {target_col}",
                color_discrete_sequence=["coral"],
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        corr = profile.get("correlation_matrix")
        if corr is not None and not corr.empty:
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Pearson Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns to build a correlation matrix.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – AutoML Configuration
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚙️ AutoML Configuration":
    st.title("⚙️ AutoML Configuration")
    st.markdown("Select which models to train and tune every aspect of the pipeline.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload a dataset on the **Dataset Explorer** page first.")
        st.stop()

    problem_type: str = st.session_state.get("problem_type", "classification")
    st.info(f"Detected problem type: **{problem_type.upper()}**")

    # ── Model selection + training params ───────────────────────────
    col_models, col_params = st.columns(2)

    with col_models:
        st.subheader("🤖 Model Selection")
        available = list(
            CLASSIFICATION_MODELS.keys()
            if problem_type == "classification"
            else REGRESSION_MODELS.keys()
        )
        selected_models = st.multiselect(
            "Models to train",
            available,
            default=available[:5],
            help="All selected models will be tuned independently via Optuna.",
        )
        if not selected_models:
            st.error("Select at least one model.")

    with col_params:
        st.subheader("🎛️ Training Parameters")
        metric_opts = (
            ["accuracy", "f1", "roc_auc", "precision", "recall"]
            if problem_type == "classification"
            else ["rmse", "mae", "r2"]
        )
        metric = st.selectbox("Primary metric", metric_opts)
        n_trials = st.slider("Optuna trials per model", 5, 200, 30, 5)
        cv_folds = st.slider("CV folds", 2, 10, 5)
        test_size = st.slider("Test set fraction", 0.10, 0.40, 0.20, 0.05)

    # ── Preprocessing + advanced ─────────────────────────────────────
    col_prep, col_adv = st.columns(2)

    with col_prep:
        st.subheader("🔧 Preprocessing")
        scaler = st.selectbox("Feature scaler", ["standard", "minmax", "none"])
        num_impute = st.selectbox("Numerical imputation", ["median", "mean", "most_frequent"])
        cat_impute = st.selectbox("Categorical imputation", ["most_frequent", "constant"])
        use_fs = st.checkbox("Enable feature selection (SelectKBest)")
        k_best = st.number_input("K best features", 3, 200, 10) if use_fs else 10

    with col_adv:
        st.subheader("🌐 Advanced")
        use_gpu = st.checkbox(
            "Enable GPU acceleration",
            value=st.session_state["gpu_available"],
            disabled=not st.session_state["gpu_available"],
            help="Activates device=cuda/gpu for XGBoost and LightGBM.",
        )
        random_state = st.number_input("Random state", 0, 99999, 42)

        if selected_models:
            est_lo = len(selected_models) * n_trials * 0.3
            est_hi = est_lo * 3
            st.info(f"⏱️ Estimated: {est_lo:.0f} – {est_hi:.0f} s (dataset-dependent)")

    st.divider()

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        cfg = st.session_state["config"]
        cfg.set("training.selected_models", selected_models)
        cfg.set("training.metric", metric)
        cfg.set("training.n_trials", n_trials)
        cfg.set("training.cv_folds", cv_folds)
        cfg.set("training.test_size", test_size)
        cfg.set("training.random_state", int(random_state))
        cfg.set("preprocessing.scaler", scaler if scaler != "none" else None)
        cfg.set("preprocessing.num_impute_strategy", num_impute)
        cfg.set("preprocessing.cat_impute_strategy", cat_impute)
        cfg.set("preprocessing.use_feature_selection", use_fs)
        cfg.set("preprocessing.k_best_features", int(k_best))
        cfg.set("gpu.use_gpu", use_gpu)
        st.success("✅ Configuration saved — navigate to **Training Monitor** to start.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Training Monitor
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🚀 Training Monitor":
    st.title("🚀 Training Monitor")

    if st.session_state["df"] is None or st.session_state["target_col"] is None:
        st.warning("⚠️ Complete the **Dataset Explorer** and **Configuration** steps first.")
        st.stop()

    df = st.session_state["df"]
    target_col: str = st.session_state["target_col"]
    profile: dict = st.session_state.get("profile", {})
    problem_type: str = st.session_state.get("problem_type", "classification")
    cfg: ConfigLoader = st.session_state["config"]

    # ── Current-config summary ────────────────────────────────────────
    st.subheader("Current Configuration")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Problem", problem_type.upper())
    m2.metric("Metric", cfg.get("training.metric", "accuracy"))
    m3.metric("Trials / model", cfg.get("training.n_trials", 30))
    m4.metric("CV folds", cfg.get("training.cv_folds", 5))

    sel_models = cfg.get("training.selected_models") or list(
        CLASSIFICATION_MODELS.keys()
        if problem_type == "classification"
        else REGRESSION_MODELS.keys()
    )
    st.info(f"Models queued: **{', '.join(sel_models)}**")
    st.divider()

    # ── Already trained? ──────────────────────────────────────────────
    if st.session_state["training_complete"]:
        st.success("✅ Training complete — check the **Results** page.")
        if st.button("🔄 Reset & Retrain", use_container_width=True):
            st.session_state["training_complete"] = False
            st.session_state["training_results"] = None
            st.session_state["leaderboard_df"] = None
            st.session_state["trainer"] = None
            st.rerun()
        st.stop()

    # ── Start training ────────────────────────────────────────────────
    if st.button("▶️ Start AutoML Training", type="primary", use_container_width=True):
        feature_types = profile.get("feature_types", {})
        numerical_feats = [c for c in feature_types.get("numerical", []) if c != target_col]
        categorical_feats = [c for c in feature_types.get("categorical", []) if c != target_col]

        X = df.drop(columns=[target_col])
        y = df[target_col]

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

        progress_bar = st.progress(0)
        status_text = st.empty()

        def _update(model_name: str, pct: float) -> None:
            progress_bar.progress(min(int(pct), 100))
            if model_name != "Done":
                status_text.markdown(f"⏳ Training **{model_name}** ({pct:.0f}% done)…")
            else:
                status_text.markdown("✅ All models trained!")

        trainer = AutoMLTrainer(
            problem_type=problem_type,
            preprocessing_pipeline=preprocessing,
            selected_models=sel_models,
            metric=cfg.get("training.metric", "accuracy"),
            n_trials=cfg.get("training.n_trials", 30),
            cv_folds=cfg.get("training.cv_folds", 5),
            test_size=cfg.get("training.test_size", 0.2),
            use_gpu=cfg.get("gpu.use_gpu", False),
            random_state=cfg.get("training.random_state", 42),
            progress_callback=_update,
            pruner_type="median",  # Enable early stopping of unpromising trials
        )

        with st.spinner("Running AutoML pipeline — this may take a few minutes…"):
            results = trainer.train_all(X, y)

        lb = Leaderboard(
            problem_type=problem_type,
            primary_metric=cfg.get("training.metric", "accuracy"),
        )
        leaderboard_df = lb.build(results)

        st.session_state["training_results"] = results
        st.session_state["leaderboard_df"] = leaderboard_df
        st.session_state["trainer"] = trainer
        st.session_state["training_complete"] = True

        st.success("🎉 AutoML complete! Navigate to **Results** to view the leaderboard.")
        st.balloons()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🏆 Results":
    st.title("🏆 Results & Leaderboard")

    if not st.session_state["training_complete"]:
        st.warning("⚠️ No results yet — run AutoML training first.")
        st.stop()

    leaderboard_df: pd.DataFrame = st.session_state["leaderboard_df"]
    results = st.session_state["training_results"]
    problem_type: str = st.session_state.get("problem_type", "classification")
    trainer: AutoMLTrainer = st.session_state["trainer"]

    # ── Leaderboard table ─────────────────────────────────────────────
    st.subheader("🏅 Model Leaderboard")

    def _highlight_rank1(row):
        color = "background-color: #FFD700; font-weight: bold;" if row.name == 1 else ""
        return [color] * len(row)

    float_cols = {
        c: "{:.4f}" for c in leaderboard_df.select_dtypes(include="float").columns
    }
    st.dataframe(
        leaderboard_df.style.apply(_highlight_rank1, axis=1).format(float_cols),
        use_container_width=True,
    )

    best_name = leaderboard_df.iloc[0]["model"]
    st.success(f"🥇 Best Model: **{best_name}**")
    st.divider()

    # ── Metric comparison bar charts ──────────────────────────────────
    st.subheader("📊 Metric Comparison")
    if problem_type == "classification":
        metric_cols = [
            c for c in ["accuracy", "f1_weighted", "roc_auc"] if c in leaderboard_df.columns
        ]
    else:
        metric_cols = [
            c for c in ["rmse", "mae", "r2"] if c in leaderboard_df.columns
        ]

    for mc in metric_cols:
        plot_df = leaderboard_df[["model", mc]].dropna()
        fig = px.bar(
            plot_df, x="model", y=mc,
            title=f"{mc.upper()} by Model",
            color=mc,
            color_continuous_scale="Blues",
            text_auto=".3f",
        )
        fig.update_layout(showlegend=False, xaxis_title="Model", yaxis_title=mc.upper())
        st.plotly_chart(fig, use_container_width=True)

    # ── Training time ─────────────────────────────────────────────────
    st.divider()
    st.subheader("⏱️ Training Time")
    if "training_time_s" in leaderboard_df.columns:
        fig = px.bar(
            leaderboard_df, x="model", y="training_time_s",
            title="Training Time per Model (seconds)",
            color="training_time_s",
            color_continuous_scale="Oranges",
            text_auto=".1f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-model details ─────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Model Details")
    chosen_model = st.selectbox("Inspect model", [r.model_name for r in results])
    sel_result = next(r for r in results if r.model_name == chosen_model)

    det_l, det_r = st.columns(2)
    with det_l:
        st.markdown("**Best Hyperparameters**")
        if sel_result.best_params:
            st.dataframe(
                pd.DataFrame(
                    list(sel_result.best_params.items()),
                    columns=["Parameter", "Value"],
                ),
                use_container_width=True,
            )
        else:
            st.info("Default parameters used (no search space defined).")

    with det_r:
        st.markdown("**Evaluation Metrics**")
        rows = [
            (k, f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v)))
            for k, v in sel_result.metrics.items()
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            use_container_width=True,
        )

    # ── SHAP explainability ───────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Feature Importance (SHAP)")

    shap_model_name = st.selectbox(
        "Model for SHAP analysis", [r.model_name for r in results], key="shap_sel"
    )

    if st.button("🔍 Generate SHAP Analysis", type="secondary"):
        if trainer is None or trainer.fitted_preprocessor is None:
            st.error("Trainer state is unavailable — please retrain.")
        else:
            with st.spinner("Computing SHAP values…"):
                try:
                    from flexautoml.explainability.shap_explainer import SHAPExplainer

                    shap_result = next(r for r in results if r.model_name == shap_model_name)
                    X_tr_t = trainer.fitted_preprocessor.transform(trainer.X_train)
                    X_te_t = trainer.fitted_preprocessor.transform(trainer.X_test)

                    try:
                        feat_names = trainer.fitted_preprocessor.get_feature_names_out().tolist()
                    except Exception:
                        feat_names = [f"feature_{i}" for i in range(X_tr_t.shape[1])]

                    explainer = SHAPExplainer(
                        model=shap_result.model,
                        X_train_transformed=X_tr_t,
                        feature_names=feat_names,
                        problem_type=problem_type,
                    )

                    fig = explainer.plot_feature_importance_bar(top_n=15)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    importance_df = explainer.get_feature_importance()
                    st.dataframe(importance_df.head(20), use_container_width=True)

                except ImportError:
                    st.warning("SHAP is not installed.  Run `pip install shap` then restart.")
                except Exception as exc:
                    st.error(f"SHAP analysis failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – Model Export
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💾 Model Export":
    st.title("💾 Model Export")
    st.markdown(
        "Download any trained pipeline as a `.pkl` file.  "
        "Load it in production with `joblib.load()`."
    )

    if not st.session_state["training_complete"]:
        st.warning("⚠️ No trained models yet — complete the AutoML run first.")
        st.stop()

    results = st.session_state["training_results"]
    leaderboard_df: pd.DataFrame = st.session_state["leaderboard_df"]

    if not results or leaderboard_df is None:
        st.error("Training results are unavailable.")
        st.stop()

    best_name: str = leaderboard_df.iloc[0]["model"]
    best_result = next((r for r in results if r.model_name == best_name), None)

    # ── Best model + leaderboard ──────────────────────────────────────
    st.success(f"🥇 Best Model: **{best_name}**")
    dl_left, dl_right = st.columns(2)

    with dl_left:
        st.markdown("**Download Best Model**")
        if best_result is not None:
            buf = io.BytesIO()
            joblib.dump(best_result.pipeline, buf)
            buf.seek(0)
            st.download_button(
                label="⬇️ Best Model (.pkl)",
                data=buf,
                file_name=f"{best_name}_pipeline.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                type="primary",
            )

    with dl_right:
        st.markdown("**Download Leaderboard**")
        st.download_button(
            label="⬇️ Leaderboard (.csv)",
            data=leaderboard_df.to_csv(index=True).encode("utf-8"),
            file_name="flexautoml_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # ── Download any model ────────────────────────────────────────────
    st.subheader("📦 Download Any Model")
    chosen = st.selectbox("Select model", [r.model_name for r in results])
    chosen_result = next(r for r in results if r.model_name == chosen)

    buf2 = io.BytesIO()
    joblib.dump(chosen_result.pipeline, buf2)
    buf2.seek(0)

    st.download_button(
        label=f"⬇️ {chosen} pipeline (.pkl)",
        data=buf2,
        file_name=f"{chosen}_pipeline.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.divider()

    # ── Usage snippet ─────────────────────────────────────────────────
    st.subheader("📖 How to Use the Downloaded Model")
    st.code(
        """\
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("ModelName_pipeline.pkl")

# Pass raw (un-preprocessed) data — same columns as training input
new_data = pd.read_csv("new_data.csv")
predictions = pipeline.predict(new_data)
print(predictions)
""",
        language="python",
    )
