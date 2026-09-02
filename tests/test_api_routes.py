"""Tests for FastAPI backend API routes."""

import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from backend.main import app
from backend.core.trainer import TRAINING_ENGINE

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Reset active dataset and active run state before each test."""
    with TRAINING_ENGINE._lock:
        TRAINING_ENGINE.active_dataset_path = None
        TRAINING_ENGINE.active_dataset_profile = None
        TRAINING_ENGINE.active_config = None
        TRAINING_ENGINE.active_run = None
    yield


class TestRootEndpoint:
    def test_root_returns_ok(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "neko-matic API" in data["service"]


class TestDatasetRoutes:
    def test_upload_dataset_csv_success(self, sample_classification_data, tmp_path):
        csv_bytes = sample_classification_data.to_csv(index=False).encode("utf-8")
        files = {"file": ("test_dataset.csv", io.BytesIO(csv_bytes), "text/csv")}

        response = client.post("/upload_dataset", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test_dataset.csv"
        assert data["shape"] == [100, 4]
        assert len(data["columns"]) == 4
        assert "profile" in data
        assert "preview" in data

    def test_upload_dataset_invalid_format(self):
        files = {"file": ("document.txt", io.BytesIO(b"hello world"), "text/plain")}
        response = client.post("/upload_dataset", files=files)
        assert response.status_code == 400
        assert "Only CSV and Excel" in response.json()["detail"]

    def test_upload_dataset_empty_file(self):
        files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        response = client.post("/upload_dataset", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_dataset_too_small(self):
        small_df = pd.DataFrame({"a": range(5), "target": [0, 1, 0, 1, 0]})
        csv_bytes = small_df.to_csv(index=False).encode("utf-8")
        files = {"file": ("small.csv", io.BytesIO(csv_bytes), "text/csv")}

        response = client.post("/upload_dataset", files=files)
        assert response.status_code == 400
        assert "Dataset too small" in response.json()["detail"]

    def test_get_active_dataset_no_dataset_loaded(self):
        response = client.get("/active_dataset")
        assert response.status_code == 404
        assert "No active dataset" in response.json()["detail"]

    def test_get_active_dataset_success(self, sample_classification_data):
        csv_bytes = sample_classification_data.to_csv(index=False).encode("utf-8")
        files = {"file": ("active_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/upload_dataset", files=files)

        response = client.get("/active_dataset")
        assert response.status_code == 200
        data = response.json()
        assert data["filename"].endswith(".csv")
        assert data["shape"] == [100, 4]

    def test_upload_and_set_default_config(self):
        config_payload = {
            "config": {
                "dataset_settings": {"train_test_split": 0.25},
                "data_cleaning": {"missing_value_strategy": "mean"},
            }
        }
        res_upload = client.post("/upload_config", json=config_payload)
        assert res_upload.status_code == 200
        assert "config_id" in res_upload.json()

        res_default = client.post("/set_default_config", json=config_payload)
        assert res_default.status_code == 200
        assert res_default.json()["status"] == "default_config_saved"


class TestTrainingRoutes:
    def test_start_automl_run_without_dataset_fails(self):
        response = client.post("/start_automl_run", json={})
        assert response.status_code == 400
        assert "No dataset set for training" in response.json()["detail"]


    def test_start_and_monitor_automl_run(self, sample_classification_data):
        csv_bytes = sample_classification_data.to_csv(index=False).encode("utf-8")
        files = {"file": ("train_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/upload_dataset", files=files)

        start_res = client.post("/start_automl_run", json={"config": {"hyperparameter_optimization": {"number_of_trials": 2}}})
        assert start_res.status_code == 200
        run_id = start_res.json()["run_id"]
        assert run_id is not None

        # Check status
        status_res = client.get("/active_run_status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["status"] in ["queued", "running", "completed"]


class TestResultsRoutes:
    def test_results_endpoints_no_active_run(self):
        assert client.get("/active_leaderboard").status_code == 200
        assert client.get("/active_leaderboard").json()["leaderboard"] == []

        assert client.get("/download_active_model").status_code == 404
        assert client.get("/download_active_artifact?artifact=training_report.json").status_code == 404

    def test_invalid_artifact_download_name(self):
        res = client.get("/download_active_artifact?artifact=malicious_file.sh")
        assert res.status_code == 400
        assert "Invalid artifact" in res.json()["detail"]
