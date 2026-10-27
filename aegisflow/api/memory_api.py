"""
AegisFlow Memory API Endpoints
==============================
Exposes endpoints for querying episodic memory trajectories and hybrid AST code search.
"""

from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from aegisflow.memory.store import HybridMemoryStore

router = APIRouter(prefix="/api/memory", tags=["Episodic & Semantic Memory"])

_store = HybridMemoryStore()


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Query phrase or error signature")
    target: str = Field(default="both", description="'code', 'incidents', or 'both'")
    top_k: int = Field(default=5, description="Number of results to return")


@router.get("/incidents")
async def list_incident_trajectories():
    """Returns all verified incident resolution trajectories stored in episodic memory."""
    return {
        "total_trajectories": len(_store.trajectories),
        "trajectories": [
            {
                "incident_id": t.incident_id,
                "failure_signature": t.failure_signature,
                "root_cause": t.root_cause,
                "failing_file": t.failing_file,
                "resolution_status": t.resolution_status,
                "tags": t.tags
            }
            for t in _store.trajectories
        ]
    }


@router.post("/search")
async def search_memory(req: MemorySearchRequest):
    """Executes hybrid (Dense + BM25 + RRF) search on code chunks or incident trajectories."""
    results: Dict[str, Any] = {}

    if req.target in ["incidents", "both"]:
        inc_res = _store.search_incidents(req.query, top_k=req.top_k)
        results["incidents"] = [
            {
                "incident_id": r.item.incident_id,
                "failure_signature": r.item.failure_signature,
                "root_cause": r.item.root_cause,
                "failing_file": r.item.failing_file,
                "patch_diff": r.item.patch_diff,
                "score": r.score
            }
            for r in inc_res
        ]

    if req.target in ["code", "both"]:
        code_res = _store.search_code(req.query, top_k=req.top_k)
        results["code_chunks"] = [
            {
                "file_path": r.item.file_path,
                "name": r.item.name,
                "start_line": r.item.start_line,
                "end_line": r.item.end_line,
                "signature": r.item.signature,
                "score": r.score
            }
            for r in code_res
        ]

    return results
