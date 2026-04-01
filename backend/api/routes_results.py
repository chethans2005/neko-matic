from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.trainer import TRAINING_ENGINE

router = APIRouter(tags=["results"])


@router.get("/leaderboard")
async def get_leaderboard(run_id: Optional[str] = None) -> dict:
    """Get leaderboard. If run_id omitted, returns active run leaderboard (V2)."""
    if run_id:
        # V1 compat
        rows = TRAINING_ENGINE.get_leaderboard(run_id)
        return {"run_id": run_id, "leaderboard": rows}
    else:
        # V2: return active run leaderboard
        status = TRAINING_ENGINE.get_active_run_status()
        return {"leaderboard": status.get("leaderboard", [])}


@router.get("/active_leaderboard")
async def get_active_leaderboard() -> dict:
    """Get leaderboard for active run (V2 endpoint)."""
    status = TRAINING_ENGINE.get_active_run_status()
    return {"leaderboard": status.get("leaderboard", [])}


@router.get("/feature_importance")
async def get_feature_importance(run_id: Optional[str] = None) -> dict:
    """Get feature importance. If run_id omitted, returns active run (V2)."""
    if run_id:
        # V1 compat
        payload = TRAINING_ENGINE.get_feature_importance(run_id)
        return {"run_id": run_id, **payload}
    else:
        # V2: return active run feature importance
        if not TRAINING_ENGINE.active_run:
            raise HTTPException(status_code=404, detail="No active run")
        payload = TRAINING_ENGINE.get_feature_importance(TRAINING_ENGINE.active_run.run_id)
        return payload


@router.get("/active_feature_importance")
async def get_active_feature_importance() -> dict:
    """Get feature importance for active run (V2 endpoint)."""
    if not TRAINING_ENGINE.active_run:
        raise HTTPException(status_code=404, detail="No active run")
    payload = TRAINING_ENGINE.get_feature_importance(TRAINING_ENGINE.active_run.run_id)
    return payload


@router.get("/download_model")
async def download_model(run_id: Optional[str] = None):
    """Download model. If run_id omitted, downloads from active run (V2)."""
    if not run_id and TRAINING_ENGINE.active_run:
        run_id = TRAINING_ENGINE.active_run.run_id
    
    if not run_id:
        raise HTTPException(status_code=404, detail="No run_id or active run")
    
    model_path = TRAINING_ENGINE.get_model_path(run_id)
    if model_path is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found")

    return FileResponse(
        path=model_path,
        filename="best_model.pkl",
        media_type="application/octet-stream",
    )


@router.get("/download_active_model")
async def download_active_model():
    """Download model from active run (V2 endpoint)."""
    if not TRAINING_ENGINE.active_run:
        raise HTTPException(status_code=404, detail="No active run")
    
    model_path = TRAINING_ENGINE.get_model_path(TRAINING_ENGINE.active_run.run_id)
    if model_path is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found")

    return FileResponse(
        path=model_path,
        filename="best_model.pkl",
        media_type="application/octet-stream",
    )


@router.get("/download_artifact")
async def download_artifact(run_id: Optional[str] = None, artifact: str = "training_report.json"):
    """Download artifact. If run_id omitted, uses active run (V2)."""
    if not run_id and TRAINING_ENGINE.active_run:
        run_id = TRAINING_ENGINE.active_run.run_id
    
    if not run_id:
        raise HTTPException(status_code=404, detail="No run_id or active run")
    
    exports = TRAINING_ENGINE.get_export_paths(run_id)
    if artifact not in exports:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(exports[artifact])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact path missing")

    media = "application/json" if artifact.endswith(".json") else "application/octet-stream"
    return FileResponse(path=path, filename=artifact, media_type=media)


@router.get("/download_active_artifact")
async def download_active_artifact(artifact: str = "training_report.json"):
    """Download artifact from active run (V2 endpoint)."""
    if not TRAINING_ENGINE.active_run:
        raise HTTPException(status_code=404, detail="No active run")
    
    exports = TRAINING_ENGINE.get_export_paths(TRAINING_ENGINE.active_run.run_id)
    if artifact not in exports:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(exports[artifact])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact path missing")

    media = "application/json" if artifact.endswith(".json") else "application/octet-stream"
    return FileResponse(path=path, filename=artifact, media_type=media)
