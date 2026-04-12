from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.core.trainer import TRAINING_ENGINE

router = APIRouter(tags=["results"])

ALLOWED_ARTIFACTS = {"best_model.pkl", "pipeline.pkl", "training_report.json", "feature_importance.json"}


def _active_run_id() -> str:
    """Get active run ID or raise 404."""
    if not TRAINING_ENGINE.active_run:
        raise HTTPException(
            status_code=404, 
            detail="No active training run. Start a training run first."
        )
    return TRAINING_ENGINE.active_run.run_id


def _active_leaderboard_payload() -> dict:
    """Get leaderboard for active run."""
    status = TRAINING_ENGINE.get_active_run_status()
    return {"leaderboard": status.get("leaderboard", [])}


def _active_feature_importance_payload() -> dict:
    """Get feature importance for active run."""
    return TRAINING_ENGINE.get_feature_importance(_active_run_id())


@router.get("/leaderboard")
async def get_leaderboard(run_id: Optional[str] = Query(None, description="Run ID for V1 backward compatibility")) -> dict:
    """Get leaderboard. If run_id omitted, returns active run leaderboard (V2).
    
    Query Parameters:
        run_id: Optional run ID for V1 backward compatibility
    """
    if run_id:
        # V1 compat
        rows = TRAINING_ENGINE.get_leaderboard(run_id)
        if not rows:
            raise HTTPException(status_code=404, detail=f"No leaderboard found for run '{run_id}'")
        return {"run_id": run_id, "leaderboard": rows}
    return _active_leaderboard_payload()


@router.get("/active_leaderboard")
async def get_active_leaderboard() -> dict:
    """Get leaderboard for active run (V2 endpoint)."""
    return _active_leaderboard_payload()


@router.get("/feature_importance")
async def get_feature_importance(run_id: Optional[str] = Query(None, description="Run ID for V1 backward compatibility")) -> dict:
    """Get feature importance. If run_id omitted, returns active run (V2).
    
    Query Parameters:
        run_id: Optional run ID for V1 backward compatibility
    """
    if run_id:
        # V1 compat
        try:
            payload = TRAINING_ENGINE.get_feature_importance(run_id)
            if not payload:
                raise HTTPException(status_code=404, detail=f"No feature importance found for run '{run_id}'")
            return {"run_id": run_id, **payload}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving feature importance: {str(e)}")
    return _active_feature_importance_payload()


@router.get("/active_feature_importance")
async def get_active_feature_importance() -> dict:
    """Get feature importance for active run (V2 endpoint)."""
    return _active_feature_importance_payload()


@router.get("/download_model")
async def download_model(run_id: Optional[str] = Query(None, description="Run ID for V1 backward compatibility")):
    """Download model. If run_id omitted, downloads from active run (V2).
    
    Query Parameters:
        run_id: Optional run ID for V1 backward compatibility
    
    Raises:
        HTTPException 404: Model not found or no active run
        HTTPException 500: Error retrieving model
    """
    if not run_id:
        try:
            run_id = _active_run_id()
        except HTTPException:
            raise HTTPException(status_code=404, detail="No run_id provided and no active run")
    
    try:
        model_path = TRAINING_ENGINE.get_model_path(run_id)
        if model_path is None or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model artifact not found for run '{run_id}'")

        return FileResponse(
            path=model_path,
            filename="best_model.pkl",
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")


@router.get("/download_active_model")
async def download_active_model():
    """Download model from active run (V2 endpoint).
    
    Raises:
        HTTPException 404: No active run or model not found
        HTTPException 500: Error retrieving model
    """
    try:
        model_path = TRAINING_ENGINE.get_model_path(_active_run_id())
        if model_path is None or not model_path.exists():
            raise HTTPException(status_code=404, detail="Model artifact not found for active run")

        return FileResponse(
            path=model_path,
            filename="best_model.pkl",
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")


@router.get("/download_artifact")
async def download_artifact(
    run_id: Optional[str] = Query(None, description="Run ID for V1 backward compatibility"),
    artifact: str = Query("training_report.json", description="Artifact filename")
):
    """Download artifact. If run_id omitted, uses active run (V2).
    
    Query Parameters:
        run_id: Optional run ID for V1 backward compatibility
        artifact: Filename of artifact to download (default: training_report.json)
    
    Raises:
        HTTPException 400: Invalid artifact name
        HTTPException 404: Artifact or run not found
        HTTPException 500: Error retrieving artifact
    """
    if artifact not in ALLOWED_ARTIFACTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid artifact '{artifact}'. Allowed: {', '.join(sorted(ALLOWED_ARTIFACTS))}"
        )
    
    if not run_id:
        try:
            run_id = _active_run_id()
        except HTTPException:
            raise HTTPException(status_code=404, detail="No run_id provided and no active run")
    
    try:
        exports = TRAINING_ENGINE.get_export_paths(run_id)
        if artifact not in exports:
            raise HTTPException(status_code=404, detail=f"Artifact '{artifact}' not found for run '{run_id}'")

        path = Path(exports[artifact])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact path missing for '{artifact}'")

        return FileResponse(
            path=path,
            filename=artifact,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading artifact: {str(e)}")


@router.get("/download_active_artifact")
async def download_active_artifact(artifact: str = Query("training_report.json", description="Artifact filename")):
    """Download artifact from active run (V2 endpoint).
    
    Query Parameters:
        artifact: Filename of artifact to download (default: training_report.json)
    
    Raises:
        HTTPException 400: Invalid artifact name
        HTTPException 404: Artifact or no active run
        HTTPException 500: Error retrieving artifact
    """
    if artifact not in ALLOWED_ARTIFACTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid artifact '{artifact}'. Allowed: {', '.join(sorted(ALLOWED_ARTIFACTS))}"
        )
    
    try:
        run_id = _active_run_id()
        exports = TRAINING_ENGINE.get_export_paths(run_id)
        if artifact not in exports:
            raise HTTPException(status_code=404, detail=f"Artifact '{artifact}' not found in active run")

        path = Path(exports[artifact])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact path missing for '{artifact}'")

        return FileResponse(
            path=path,
            filename=artifact,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading artifact: {str(e)}")
