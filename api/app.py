"""FastAPI application factory for the Adaptive Duelist API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import matches, models, system, training

UI_DIR = Path(__file__).parent.parent / "ui"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Adaptive Duelist AI",
        description="Local API for the Adaptive Duelist AI combat sandbox.",
        version="12.0.0",
    )

    app.include_router(system.router)
    app.include_router(matches.router)
    app.include_router(training.router)
    app.include_router(models.router)

    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    return app
