"""Ayojna dashboard API (FastAPI). Thin routes over service.py.

Run:  uvicorn ayojna.api.app:app --port 8000      then open http://localhost:8000
API docs (Swagger) at http://localhost:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ayojna.api.service import Service
from ayojna.settings import DATA_DIR, REPO_ROOT

WEB = REPO_ROOT / "web" / "index.html"


def create_app(state_dir: str | Path | None = None, lake_dir: str | Path | None = None) -> FastAPI:
    svc = Service(
        state_dir or os.getenv("AYOJNA_STATE", DATA_DIR / "state"),
        lake_dir or os.getenv("AYOJNA_LAKE", DATA_DIR / "lake"),
    )
    api = FastAPI(title="Ayojna", description="AI that plans where every byte should live")

    @api.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(WEB)

    @api.get("/api/kpis")
    def kpis():
        return svc.kpis()

    @api.get("/api/status")
    def status():
        return svc.status()

    @api.get("/api/scoreboard")
    def scoreboard():
        return svc.scoreboard()

    @api.get("/api/placement")
    def placement():
        return svc.placement()

    @api.get("/api/plan")
    def plan(limit: int = 50):
        return svc.plan(limit)

    @api.get("/api/execution")
    def execution():
        return svc.execution()

    @api.get("/api/audit")
    def audit(limit: int = 50):
        return svc.audit(limit)

    @api.get("/api/explain/{volume}/{extent_id}")
    def explain(volume: str, extent_id: int):
        return svc.explain(volume, extent_id)

    return api


app = create_app()