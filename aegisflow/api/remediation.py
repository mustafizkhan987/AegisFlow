"""
AegisFlow Remediation API Endpoints
==================================
Provides endpoints to trigger the autonomous multi-step remediation loop,
executing AST retrieval, patch synthesis, and sandboxed test verification.
"""

from fastapi import APIRouter
from aegisflow.agent.orchestrator import (
    RemediationOrchestrator,
    RemediationRequest,
    RemediationResponse,
)

router = APIRouter(prefix="/api/agent", tags=["Autonomous Remediation Agent"])

# Shared orchestrator instance
_orchestrator = RemediationOrchestrator()


@router.post("/remediate", response_model=RemediationResponse)
async def trigger_remediation(req: RemediationRequest):
    """Triggers the full multi-step remediation loop."""
    return await _orchestrator.remediate(req)
