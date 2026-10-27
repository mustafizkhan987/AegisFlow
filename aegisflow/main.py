"""
AegisFlow Main FastAPI Application
==================================
Primary application entrypoint. Configures routers, WebSockets, CORS,
static frontend serving, and server lifespan.

Usage:
    uvicorn aegisflow.main:app --host 127.0.0.1 --port 8000 --reload
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from aegisflow.config import settings
from aegisflow.api import (
    triage_router,
    remediation_router,
    memory_router,
    benchmarks_router,
    webhooks_router,
)
from aegisflow.observability.websocket_hub import ws_hub


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server startup and shutdown lifecycle management."""
    print(f"============================================================")
    print(f"  {settings.app_name} v{settings.app_version} Starting Up")
    print(f"  Classical ML + MCP Tools + Ephemeral Sandbox Engine Active")
    print(f"============================================================")
    yield
    print(f"AegisFlow shutting down cleanly.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Autonomous CI/CD Failure Triage, Root-Cause Analysis, and Sandboxed Repair Agent",
    lifespan=lifespan
)

# Enable CORS for local Vite and dashboard clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(triage_router)
app.include_router(remediation_router)
app.include_router(memory_router)
app.include_router(benchmarks_router)
app.include_router(webhooks_router)


@app.get("/health")
async def health_check():
    """Health check endpoint confirming API service readiness."""
    return {
        "status": "HEALTHY",
        "service": settings.app_name,
        "version": settings.app_version,
        "sandbox_backend": settings.sandbox_backend,
        "active_websockets": len(ws_hub.active_connections)
    }


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """Live telemetry WebSocket stream for dashboard graphs, terminal logs, and cost stats."""
    await ws_hub.connect(websocket)
    try:
        while True:
            # Keep connection alive & accept incoming ping/commands
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket)
    except Exception:
        ws_hub.disconnect(websocket)


# Mount static frontend if built
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
