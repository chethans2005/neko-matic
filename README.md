# neko-matic

neko-matic is a configurable AutoML platform with:

- `backend/`: FastAPI service + modular ML engine
- `frontend/`: Next.js dashboard

The platform supports dataset upload, pipeline configuration, asynchronous model training, leaderboard tracking, explainability output, and artifact export.

## Architecture

`frontend` -> `backend API` -> `backend/core` training engine -> `models/runs/<run_id>/` artifacts

## Repository Layout

```text
auto-ml/
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── routes_datasets.py
│   │   ├── routes_training.py
│   │   └── routes_results.py
│   ├── core/
│   │   ├── automl_trainer.py
│   │   ├── profiler.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── outlier_detection.py
│   │   ├── model_registry.py
│   │   ├── optimizer.py
│   │   ├── evaluator.py
│   │   ├── leaderboard.py
│   │   └── trainer.py
│   ├── meta_learning/
│   ├── explainability/
│   ├── utils/
│   └── configs/default.yaml
├── frontend/
│   ├── app/
│   │   ├── dataset_upload/
│   │   ├── dataset_explorer/
│   │   ├── automl_configuration/
│   │   ├── training_monitor/
│   │   ├── leaderboard/
│   │   ├── explainability/
│   │   └── model_export/
│   └── package.json
├── models/
│   └── runs/
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Backend Setup

```powershell
# from repo root
python -m venv auto-ml-env
& "./auto-ml-env/Scripts/Activate.ps1"
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Backend health check:

```text
GET http://localhost:8000/
```

Expected response:

```json
{"service": "neko-matic API", "status": "ok"}
```

## Frontend Setup

```powershell
# in new terminal
Set-Location frontend
npm install
$env:NEXT_PUBLIC_API_BASE = "http://localhost:8000"
npm run dev
```

Open:

```text
http://localhost:3000
```

## API Endpoints

### Dataset and config

- `POST /upload_dataset` (multipart file)
- `POST /upload_config`

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

### Training and status

- `POST /start_automl_run`
- `GET /training_status?run_id=<id>`

Example training start payload:

```json
{
	"dataset_id": "<dataset_id>",
	"config_id": "<config_id>"
}
```

### Results and artifacts

- `GET /leaderboard?run_id=<id>`
- `GET /feature_importance?run_id=<id>`
- `GET /download_model?run_id=<id>`
- `GET /download_artifact?run_id=<id>&artifact=pipeline.pkl|training_report.json`

## Dashboard Workflow

1. Upload a CSV/XLSX dataset in `dataset_upload`.
2. Inspect distribution and missingness in `dataset_explorer`.
3. Upload or edit JSON config in `automl_configuration`.
4. Start run and monitor progress in `training_monitor`.
5. Inspect ranking in `leaderboard`.
6. View explainability in `explainability`.
7. Download artifacts in `model_export`.

## Artifacts

Run artifacts are saved under:

```text
models/runs/<run_id>/
	best_model.pkl
	pipeline.pkl
	training_report.json
	feature_importance.json
```

## Validation Commands

```powershell
# backend syntax check
python -m compileall backend

# frontend production build
Set-Location frontend
npm run build
```

## Notes

- Training runs execute asynchronously in background threads.
- Run state is in-memory for the current backend process.
- If backend restarts, in-memory run state resets, while artifacts on disk remain.
