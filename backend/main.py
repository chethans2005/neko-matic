from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_datasets import router as datasets_router
from backend.api.routes_results import router as results_router
from backend.api.routes_training import router as training_router

app = FastAPI(
    title="neko-matic API",
    version="1.0.0",
    description="Production-oriented AutoML backend service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(results_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "neko-matic API", "status": "ok"}
