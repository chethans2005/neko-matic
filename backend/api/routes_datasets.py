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

router = APIRouter(tags=["datasets"])

UPLOAD_DIR = Path("backend") / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ConfigUploadRequest(BaseModel):
    config: dict[str, Any]


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    dataset_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum supported upload size is 50 MB.",
        )
    dataset_path.write_bytes(data)

    if suffix == ".csv":
        dataframe = pd.read_csv(dataset_path)
    else:
        dataframe = pd.read_excel(dataset_path)

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty")

    profiler = DataProfiler()
    target_column = dataframe.columns[-1]
    profile = profiler.analyze(dataframe, target_column)
    profile = json.loads(json.dumps(profile, default=str))
    dataset_id = TRAINING_ENGINE.register_dataset(str(dataset_path), profile)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "preview": dataframe.head(15).fillna("").to_dict(orient="records"),
        "columns": dataframe.columns.tolist(),
        "target_column_guess": target_column,
        "profile": profile,
    }


@router.post("/upload_config")
async def upload_config(payload: ConfigUploadRequest) -> dict[str, str]:
    config_id = TRAINING_ENGINE.register_config(payload.config)
    config_path = UPLOAD_DIR / f"{config_id}.json"
    config_path.write_text(json.dumps(payload.config, indent=2), encoding="utf-8")
    return {"config_id": config_id}
