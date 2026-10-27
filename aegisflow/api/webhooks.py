"""
AegisFlow CI/CD Webhook Ingestion API
=====================================
Ingests webhook notifications from GitHub Actions, GitLab CI, or Jenkins.
Automatically runs XGBoost triage and routes genuine code regressions to the
autonomous remediation agent loop.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Header, BackgroundTasks
from pydantic import BaseModel, Field

from aegisflow.api.triage import get_classifier
from aegisflow.agent.orchestrator import RemediationOrchestrator, RemediationRequest
from aegisflow.observability.websocket_hub import ws_hub

router = APIRouter(prefix="/api/webhooks", tags=["Webhook Ingestion"])

_orchestrator = RemediationOrchestrator()


class GenericWebhookPayload(BaseModel):
    event_id: str = Field(..., description="Unique event or run ID")
    repository: str = Field(default="org/repo", description="Repository name")
    branch: str = Field(default="main", description="Git branch name")
    commit_sha: str = Field(default="HEAD", description="Commit SHA")
    log_text: str = Field(..., description="Raw build or test log")
    exit_code: int = Field(default=1, description="Exit code")
    diff_churn: int = Field(default=0, description="Lines modified in commit")


@router.post("/generic")
async def ingest_generic_webhook(payload: GenericWebhookPayload, background_tasks: BackgroundTasks):
    """Ingests generic CI failure logs, triages with XGBoost, and conditionally launches repair."""
    classifier = get_classifier()
    triage_result = classifier.predict(
        log_text=payload.log_text,
        exit_code=payload.exit_code,
        diff_churn_lines=payload.diff_churn,
        historical_failure_rate=0.15
    )

    incident_id = f"INC-{payload.event_id[:8]}"

    # Broadcast triage completion
    await ws_hub.broadcast("TRIAGE_COMPLETED", {
        "incident_id": incident_id,
        "category": triage_result.category,
        "confidence": triage_result.confidence,
        "routing_decision": triage_result.routing_decision,
        "should_invoke_llm": triage_result.should_invoke_llm,
        "estimated_token_savings": triage_result.estimated_token_savings,
        "estimated_cost_saved_usd": triage_result.estimated_cost_saved_usd,
        "key_indicators": triage_result.key_indicators,
        "latency_ms": triage_result.latency_ms,
        "log_preview": payload.log_text[:350]
    })

    # If code regression, launch remediation in background
    if triage_result.should_invoke_llm:
        background_tasks.add_task(
            _orchestrator.remediate,
            RemediationRequest(
                incident_id=incident_id,
                log_text=payload.log_text
            )
        )

    return {
        "status": "ACCEPTED",
        "incident_id": incident_id,
        "triage_result": triage_result,
        "remediation_initiated": triage_result.should_invoke_llm
    }


@router.post("/github")
async def ingest_github_actions_webhook(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """Parses standard GitHub Actions webhook event payloads."""
    workflow_run = payload.get("workflow_run", {})
    run_id = str(workflow_run.get("id", "gh-run-001"))
    conclusion = workflow_run.get("conclusion", "failure")

    # Sample failure text if webhook payload doesn't contain full log
    sample_log = payload.get("log_text") or f"GitHub Actions run {run_id} failed with conclusion: {conclusion}"

    return await ingest_generic_webhook(
        GenericWebhookPayload(
            event_id=run_id,
            repository=payload.get("repository", {}).get("full_name", "org/repo"),
            branch=workflow_run.get("head_branch", "main"),
            commit_sha=workflow_run.get("head_sha", "HEAD"),
            log_text=sample_log,
            exit_code=1,
            diff_churn=15
        ),
        background_tasks
    )
