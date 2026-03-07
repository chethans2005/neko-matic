# FlexAutoML

A configurable, production-style **Automated Machine Learning** framework for tabular datasets. FlexAutoML automatically profiles your data, builds preprocessing pipelines, trains multiple ML models, tunes hyperparameters with Optuna, evaluates results across multiple metrics, and presents everything through an interactive Streamlit dashboard.

---

## Features

| Feature | Details |
|---|---|
| **Dataset Profiling** | Row/column counts, feature types, missing values, correlation matrix, class imbalance |
| **Auto Preprocessing** | Imputation, scaling, one-hot encoding, optional SelectKBest feature selection |
| **Model Registry** | 8 classifiers + 8 regressors (sklearn, XGBoost, LightGBM) |
| **HPO** | Optuna TPE sampler with cross-validation; configurable trial budget |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC (classification) · RMSE, MAE, R² (regression) |
| **Leaderboard** | Auto-ranked table with the best model highlighted |
| **Explainability** | SHAP feature importance via TreeExplainer / KernelExplainer |
| **Streamlit UI** | Upload → Configure → Train → Inspect → Download, all in-browser |
| **CLI** | `run_automl.py` for headless / scripted execution |
| **GPU Support** | Auto-detected; XGBoost (`device=cuda`) and LightGBM (`device=gpu`) |
| **YAML Config** | Fully configurable via `flexautoml/configs/default.yaml` |

---

## Project Structure

```
auto-ml/
├── flexautoml/
│   ├── __init__.py
│   ├── core/
│   │   ├── profiler.py           # Dataset analysis & problem-type detection
│   │   ├── preprocessing.py      # ColumnTransformer pipeline builder
│   │   ├── feature_engineering.py# Datetime extraction, high-cardinality enc.
│   │   ├── model_registry.py     # Model catalogue + Optuna search spaces
│   │   ├── optimizer.py          # Optuna HPO wrapper
│   │   ├── trainer.py            # Training orchestrator
│   │   ├── evaluator.py          # Multi-metric evaluation
│   │   └── leaderboard.py        # Ranked results table
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP feature importance & plots
│   ├── ui/
│   │   └── app.py                # Streamlit dashboard (5 pages)
│   ├── configs/
│   │   └── default.yaml          # Example configuration file
│   └── utils/
│       └── config_loader.py      # YAML loader with dot-path accessors
├── run_automl.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

> **GPU (RTX 3050 / CUDA):** XGBoost and LightGBM will automatically use the
> GPU when `nvidia-smi` is accessible.  No extra steps needed.

---

### 2 — Launch the Streamlit UI

```bash
streamlit run flexautoml/ui/app.py
```

Then open `http://localhost:8501` in your browser and follow the five pages:

| Page | Action |
|---|---|
| 📊 Dataset Explorer | Upload a CSV, pick the target column, view the profile |
| ⚙️ AutoML Configuration | Select models, set metric, trials, preprocessing |
| 🚀 Training Monitor | Press **Start** and watch progress |
| 🏆 Results | Leaderboard, metric charts, SHAP analysis |
| 💾 Model Export | Download any pipeline as `.pkl` |

---

### 3 — CLI (headless)

```bash
# Basic run
python run_automl.py data/titanic.csv Survived

# With custom config
python run_automl.py data/housing.csv price --config flexautoml/configs/default.yaml

# Specific models, metric, and trial budget
python run_automl.py data/iris.csv species \
    --models RandomForestClassifier XGBClassifier LGBMClassifier \
    --metric f1 \
    --trials 50 \
    --output results/
```

Outputs written to `--output` directory:
- `leaderboard.csv` — ranked model comparison
- `<BestModel>_classification.pkl` (or `_regression.pkl`) — best pipeline

---

### 4 — Programmatic API

```python
import pandas as pd
from flexautoml.core.profiler import DatasetProfiler
from flexautoml.core.preprocessing import PreprocessingPipeline
from flexautoml.core.trainer import AutoMLTrainer
from flexautoml.core.leaderboard import Leaderboard
from flexautoml.utils.config_loader import ConfigLoader

df = pd.read_csv("data/titanic.csv")
target = "Survived"

# 1. Profile
profiler = DatasetProfiler(df, target)
profile = profiler.profile_dataset()
print(profiler.get_summary())

# 2. Preprocess
feat = profile["feature_types"]
preproc = PreprocessingPipeline(
    numerical_features=[c for c in feat["numerical"] if c != target],
    categorical_features=[c for c in feat["categorical"] if c != target],
    problem_type=profile["problem_type"],
)

# 3. Train
trainer = AutoMLTrainer(
    problem_type=profile["problem_type"],
    preprocessing_pipeline=preproc,
    selected_models=["RandomForestClassifier", "XGBClassifier", "LGBMClassifier"],
    metric="roc_auc",
    n_trials=30,
)
results = trainer.train_all(df.drop(columns=[target]), df[target])

# 4. Leaderboard
lb = Leaderboard(profile["problem_type"], "roc_auc")
print(lb.build(results))

# 5. Save best model
path = trainer.save_best_model("models/")
print(f"Best model saved to {path}")
```

---

## Configuration Reference

Edit `flexautoml/configs/default.yaml` to customise a run:

```yaml
problem_type: auto                # auto | classification | regression
target_column: null               # set here or in the UI

preprocessing:
  scaler: standard                # standard | minmax | null
  num_impute_strategy: median     # mean | median | most_frequent
  cat_impute_strategy: most_frequent
  use_feature_selection: false
  k_best_features: 10

training:
  selected_models: null           # null = all models
  metric: accuracy                # accuracy | f1 | roc_auc | rmse | mae | r2
  n_trials: 30
  cv_folds: 5
  test_size: 0.20
  random_state: 42

gpu:
  use_gpu: auto                   # auto | true | false

output:
  save_dir: models
  leaderboard_path: leaderboard.csv
```

---

## Available Models

### Classification
`LogisticRegression` · `RandomForestClassifier` · `GradientBoostingClassifier` · `XGBClassifier` · `LGBMClassifier` · `SVC` · `KNeighborsClassifier` · `GaussianNB`

### Regression
`LinearRegression` · `Ridge` · `Lasso` · `RandomForestRegressor` · `GradientBoostingRegressor` · `XGBRegressor` · `LGBMRegressor` · `SVR`

---

## Loading a Saved Model

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/XGBClassifier_classification.pkl")
new_data = pd.read_csv("new_data.csv")
predictions = pipeline.predict(new_data)
```

The saved object is the complete `sklearn.pipeline.Pipeline` (preprocessing + model), so no separate preprocessing step is needed at inference time.

---

## Extending FlexAutoML

**Add a new model:** Add an entry to `CLASSIFICATION_MODELS` or `REGRESSION_MODELS` in `flexautoml/core/model_registry.py`, then optionally add a search-space entry in `get_hyperparameter_space()`.

**Add a new metric:** Update `_SKLEARN_METRIC_MAP` in `optimizer.py` and `_regression_metrics` / `_classification_metrics` in `evaluator.py`.

**Custom preprocessing:** Subclass `sklearn.base.TransformerMixin` and add it to the pipeline steps in `preprocessing.py`.

---

## License

MIT
