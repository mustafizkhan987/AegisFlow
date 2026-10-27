"""
AegisFlow API Package
=====================
Exposes all subrouters for FastAPI inclusion.
"""

from aegisflow.api.triage import router as triage_router
from aegisflow.api.remediation import router as remediation_router
from aegisflow.api.memory_api import router as memory_router
from aegisflow.api.benchmarks_api import router as benchmarks_router
from aegisflow.api.webhooks import router as webhooks_router

__all__ = [
    "triage_router",
    "remediation_router",
    "memory_router",
    "benchmarks_router",
    "webhooks_router",
]
