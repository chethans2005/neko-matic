from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.trainer import TRAINING_ENGINE

router = APIRouter(tags=["training"])


class StartRunRequest(BaseModel):
    dataset_id: str
    config_id: Optional[str] = None


@router.post("/start_automl_run")
async def start_automl_run(payload: StartRunRequest) -> dict[str, str]:
    if payload.dataset_id not in TRAINING_ENGINE.datasets:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    if payload.config_id and payload.config_id not in TRAINING_ENGINE.configs:
        raise HTTPException(status_code=404, detail="config_id not found")

    run_id = TRAINING_ENGINE.start_run(payload.dataset_id, payload.config_id)
    return {"run_id": run_id, "status": "queued"}


@router.get("/training_status")
async def training_status(run_id: str) -> dict:
    status = TRAINING_ENGINE.get_status(run_id)
    if status.get("error"):
        raise HTTPException(status_code=404, detail="run_id not found")
    return status
