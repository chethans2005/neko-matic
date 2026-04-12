from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.profiler import DataProfiler
from backend.core.trainer import TRAINING_ENGINE
from backend.core.exceptions import DatasetLoadError, DatasetFormatError

router = APIRouter(tags=["datasets"])

UPLOAD_DIR = Path("backend") / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MIN_ROWS = 10


class ConfigUploadRequest(BaseModel):
    config: dict[str, Any]


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload and set as active dataset. V2: no dataset_id in response.
    
    Validates:
        - File type (CSV, XLSX, XLS only)
        - File size (max 50 MB)
        - Dataset is not empty
    
    Raises:
        HTTPException 400: Invalid file type, file too large, or empty dataset
        HTTPException 500: Processing error
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    dataset_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum supported upload size is 50 MB.",
            )
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        dataset_path.write_bytes(data)

        # Load dataset
        try:
            if suffix == ".csv":
                dataframe = pd.read_csv(dataset_path)
            else:
                dataframe = pd.read_excel(dataset_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        
        if dataframe.empty:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty")
        
        if len(dataframe) < MIN_ROWS:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset too small: {len(dataframe)} rows, minimum required: {MIN_ROWS}"
            )

        profiler = DataProfiler()
        target_column = dataframe.columns[-1]
        
        try:
            profile = profiler.analyze(dataframe, target_column)
            profile = json.loads(json.dumps(profile, default=str))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")
        
        # V2: Set as active dataset
        TRAINING_ENGINE.set_active_dataset(str(dataset_path), profile)
        
        # V1 backward compat: also register
        dataset_id = TRAINING_ENGINE.register_dataset(str(dataset_path), profile)

        return {
            "dataset_id": dataset_id,  # V1 compat
            "filename": file.filename,
            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "preview": dataframe.head(15).fillna("").to_dict(orient="records"),
            "columns": dataframe.columns.tolist(),
            "target_column_guess": target_column,
            "profile": profile,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")
    finally:
        # Clean up on error
        if dataset_path.exists() and not dataset_path.stat().st_size > 0:
            try:
                dataset_path.unlink()
            except Exception:
                pass


@router.get("/active_dataset")
async def get_active_dataset() -> dict[str, Any]:
    """Get info about the currently active dataset (V2 endpoint).
    
    Raises:
        HTTPException 404: No active dataset
    """
    info = TRAINING_ENGINE.get_active_dataset_info()
    if not info:
        raise HTTPException(status_code=404, detail="No active dataset. Please upload a dataset first.")
    
    try:
        # Re-parse the dataset to return shape and preview
        dataset_path = info["path"]
        suffix = Path(dataset_path).suffix.lower()
        if suffix == ".csv":
            dataframe = pd.read_csv(dataset_path)
        else:
            dataframe = pd.read_excel(dataset_path)
        
        profile = info["profile"]
        target_column = profile.get("target_column", dataframe.columns[-1])
        
        return {
            "path": info["path"],
            "filename": Path(info["path"]).name,
            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "preview": dataframe.head(15).fillna("").to_dict(orient="records"),
            "columns": dataframe.columns.tolist(),
            "target_column_guess": target_column,
            "profile": profile,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset: {str(e)}")


@router.post("/upload_config")
async def upload_config(payload: ConfigUploadRequest) -> dict[str, str]:
    """Upload config. Can optionally set as default via save_as_default flag."""
    config_id = TRAINING_ENGINE.register_config(payload.config)
    config_path = UPLOAD_DIR / f"{config_id}.json"
    config_path.write_text(json.dumps(payload.config, indent=2), encoding="utf-8")
    return {"config_id": config_id}


@router.post("/set_default_config")
async def set_default_config(payload: ConfigUploadRequest) -> dict[str, str]:
    """Set default config for non-tech users (V2 endpoint)."""
    TRAINING_ENGINE.set_default_config(payload.config)
    return {"status": "default_config_saved"}
