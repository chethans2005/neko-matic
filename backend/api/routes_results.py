from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.trainer import TRAINING_ENGINE

router = APIRouter(tags=["results"])


@router.get("/leaderboard")
async def get_leaderboard(run_id: str) -> dict:
    rows = TRAINING_ENGINE.get_leaderboard(run_id)
    return {"run_id": run_id, "leaderboard": rows}


@router.get("/feature_importance")
async def get_feature_importance(run_id: str) -> dict:
    payload = TRAINING_ENGINE.get_feature_importance(run_id)
    return {"run_id": run_id, **payload}


@router.get("/download_model")
async def download_model(run_id: str):
    model_path = TRAINING_ENGINE.get_model_path(run_id)
    if model_path is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found")

    return FileResponse(
        path=model_path,
        filename="best_model.pkl",
        media_type="application/octet-stream",
    )


@router.get("/download_artifact")
async def download_artifact(run_id: str, artifact: str):
    exports = TRAINING_ENGINE.get_export_paths(run_id)
    if artifact not in exports:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(exports[artifact])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact path missing")

    media = "application/json" if artifact.endswith(".json") else "application/octet-stream"
    return FileResponse(path=path, filename=artifact, media_type=media)
