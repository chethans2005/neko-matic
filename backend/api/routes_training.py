from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.trainer import TRAINING_ENGINE
from backend.core.exceptions import (
    NoDatasetError,
    TrainingError,
    AutoMLException,
)

router = APIRouter(tags=["training"])


class StartRunRequest(BaseModel):
    dataset_id: Optional[str] = None  # V1 compat, ignored in V2
    config_id: Optional[str] = None  # V1 compat, ignored in V2
    config: Optional[dict] = Field(None, description="Optional config override for V2")


@router.post("/start_automl_run")
async def start_automl_run(payload: StartRunRequest) -> dict[str, str]:
    """V2: Start AutoML run using active dataset and provided/default config.
    
    Raises:
        HTTPException 400: No active dataset or invalid config
        HTTPException 500: Training engine error
    """
    try:
        run_id = TRAINING_ENGINE.start_active_run(payload.config)
        return {"run_id": run_id, "status": "queued"}
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


@router.post("/start_automl_run_legacy")
async def start_automl_run_legacy(payload: StartRunRequest) -> dict[str, str]:
    """V1 backward compat: Start using dataset_id and config_id."""
    if payload.dataset_id not in TRAINING_ENGINE.datasets:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    if payload.config_id and payload.config_id not in TRAINING_ENGINE.configs:
        raise HTTPException(status_code=404, detail="config_id not found")

    try:
        run_id = TRAINING_ENGINE.start_run(payload.dataset_id, payload.config_id)
        return {"run_id": run_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


@router.get("/training_status")
async def training_status(run_id: Optional[str] = None) -> dict:
    """Get training status. If run_id omitted, returns active run status (V2).
    
    Query Parameters:
        run_id: Optional run ID for V1 backward compatibility
    """
    if run_id:
        # V1 compat: look up by run_id
        status = TRAINING_ENGINE.get_status(run_id)
        if status.get("error"):
            raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")
        return status
    else:
        # V2: return active run status
        return TRAINING_ENGINE.get_active_run_status()


@router.get("/active_run_status")
async def get_active_run_status() -> dict:
    """Get active run status (V2 endpoint)."""
    return TRAINING_ENGINE.get_active_run_status()
