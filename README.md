
# neko-matic: AutoML Platform

**neko-matic** is a modular AutoML platform featuring:

- **Backend**: FastAPI service with a pluggable ML engine (`backend/`)
- **Frontend**: Next.js dashboard (`frontend/`)

Features include dataset upload, guided data exploration, pipeline configuration, async model training, leaderboard, explainability, and artifact export.

The current UX is optimized around a **single active dataset and a single active AutoML run**. Users upload once, explore once, configure defaults, and then launch training from one unified page.

---

## Architecture

```
frontend (Next.js) → backend API (FastAPI) → backend/core (ML engine) → models/runs/<run_id>/ (artifacts)
```

The UI no longer asks users to manually manage dataset IDs or run IDs. Those identifiers remain internal implementation details only.

---

## Project Structure

```
auto-ml/
├── backend/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── meta_learning/
│   ├── explainability/
│   ├── utils/
│   └── configs/
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── package.json
├── models/
│   ├── runs/
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

---

## Backend Setup

```powershell
# From repo root
python -m venv auto-ml-env
& "./auto-ml-env/Scripts/Activate.ps1"
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Health check:

```
GET http://localhost:8000/
```

Expected response:

```json
{"service": "neko-matic API", "status": "ok"}
```

---

## Frontend Setup

```powershell
# In new terminal
Set-Location frontend
npm install
$env:NEXT_PUBLIC_API_BASE = "http://localhost:8000"
npm run dev
```

Open: [http://localhost:3000](http://localhost:3000)

The recommended workflow is the unified **Training** page at `/training`, which combines upload, guided exploration, configuration, live monitoring, results, and downloads in one place.

---

## API Endpoints (Backend)

### Dataset & Config

- `POST /upload_dataset` (multipart file, also sets the active dataset)
- `GET /active_dataset`
- `POST /upload_config`
- `POST /set_default_config`

Example config payload:

```json
{
	"config": {
		"dataset_settings": {
			"target_column": "target",
			"problem_type_override": null,
			"train_test_split": 0.2,
			"cross_validation_folds": 5
		},
		"data_cleaning": {
			"missing_value_strategy": "median",
			"categorical_encoding": "onehot",
			"feature_scaling": "standard"
		},
		"outlier_removal": {
			"method": "none",
			"threshold_parameters": {
				"zscore_threshold": 3.0,
				"iqr_multiplier": 1.5,
				"isolation_forest_contamination": 0.05
			}
		},
		"feature_engineering": {
			"log_transform": false,
			"polynomial_features": false,
			"feature_interactions": false,
			"feature_selection": {
				"enabled": false,
				"method": "variance_threshold",
				"k_features": 20
			}
		},
		"model_selection": {
			"list_of_models_to_train": null
		},
		"hyperparameter_optimization": {
			"optimization_method": "optuna",
			"number_of_trials": 20,
			"timeout": null,
			"early_stopping": true
		},
		"training_strategy": {
			"parallel_training": false,
			"gpu_usage": "auto",
			"time_budget": null
		},
		"evaluation_metrics": {
			"primary_metric": "accuracy"
		},
		"explainability": {
			"enable_shap": true
		}
	}
}
```

### Training & Status

- `POST /start_automl_run`
- `GET /training_status` (returns the active run when no `run_id` is provided)
- `GET /active_run_status`

Example payload for the active workflow:

```json
{
	"config": {
		"dataset_settings": {
			"target_column": "target"
		}
	}
}
```

### Results & Artifacts

- `GET /leaderboard`
- `GET /feature_importance`
- `GET /download_model`
- `GET /download_artifact?artifact=pipeline.pkl|training_report.json`
- `GET /active_leaderboard`
- `GET /active_feature_importance`
- `GET /download_active_model`
- `GET /download_active_artifact?artifact=training_report.json`

---

## Dashboard Workflow

1. Open **Training** for the unified workflow.
2. Upload a CSV/XLSX dataset once. The backend marks it as the active dataset.
3. Review guided exploration tips and suggested visualizations.
4. Configure the pipeline and optionally click **Save as Default** for non-technical users.
5. Start AutoML and monitor progress, leaderboard, and explainability on the same page.
6. Download the best model and training report when the run completes.

Legacy pages such as `dataset_upload`, `dataset_explorer`, `automl_configuration`, and `training_monitor` are still available, but the main workflow now lives in `/training`.

---

## Artifacts

Run artifacts are saved under:

```
models/runs/<run_id>/
	best_model.pkl
	pipeline.pkl
	training_report.json
	feature_importance.json
```

The run directory name is still backed by an internal identifier, but the UI no longer exposes it or requires users to enter it.

---

## Validation Commands

```powershell
# Backend syntax check
python -m compileall backend

# Frontend production build
Set-Location frontend
npm run build
```

---

## Notes

- Training runs execute asynchronously in background threads.
- The backend keeps the active dataset and active run in memory for the current process.
- If the backend restarts, active state resets, but artifacts on disk remain.
- The UI is intentionally single-run: users should not launch multiple AutoML jobs from the same session.
