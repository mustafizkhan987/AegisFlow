"""
AegisFlow Triage API Endpoints
==============================
Provides endpoints for classical ML failure triage, sub-50ms log classification,
and incident simulation triggers.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from aegisflow.classifier.model import TriageClassifier, TriageResult
from aegisflow.classifier.dataset import RAW_LOG_EXAMPLES
from aegisflow.observability.websocket_hub import ws_hub
from aegisflow.observability.tracer import trace_span

router = APIRouter(prefix="/api/triage", tags=["ML Triage & Routing"])

# Shared classifier instance
_classifier: Optional[TriageClassifier] = None


def get_classifier() -> TriageClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TriageClassifier()
        # Initialize or train
        from aegisflow.classifier.dataset import generate_augmented_dataset
        _classifier.fit(generate_augmented_dataset(multiplier=10))
    return _classifier


class TriageRequest(BaseModel):
    incident_id: Optional[str] = Field(default=None, description="Optional incident identifier")
    log_text: str = Field(..., description="Raw CI/CD build or test failure log")
    exit_code: int = Field(default=1, description="Process exit code (e.g. 1, 137, 143)")
    diff_churn_lines: int = Field(default=0, description="Number of modified lines in commit")
    historical_failure_rate: float = Field(default=0.15, description="Flakiness score of test (0.0 - 1.0)")


class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="Simulation scenario: 'flake', 'dependency', 'code_bug', or 'lint'")


@router.post("/classify", response_model=TriageResult)
async def classify_log(req: TriageRequest):
    """Classifies a CI/CD failure log using the XGBoost Gatekeeper."""
    with trace_span("triage_classify_endpoint", {"exit_code": req.exit_code}):
        classifier = get_classifier()
        result = classifier.predict(
            log_text=req.log_text,
            exit_code=req.exit_code,
            diff_churn_lines=req.diff_churn_lines,
            historical_failure_rate=req.historical_failure_rate
        )

        # Broadcast triage event to live dashboard
        await ws_hub.broadcast("TRIAGE_COMPLETED", {
            "incident_id": req.incident_id or f"INC-{result.category[:4]}",
            "category": result.category,
            "confidence": result.confidence,
            "routing_decision": result.routing_decision,
            "should_invoke_llm": result.should_invoke_llm,
            "estimated_token_savings": result.estimated_token_savings,
            "estimated_cost_saved_usd": result.estimated_cost_saved_usd,
            "key_indicators": result.key_indicators,
            "latency_ms": result.latency_ms,
            "log_preview": req.log_text[:350]
        })

        return result


@router.post("/simulate")
async def simulate_incident(req: SimulationRequest):
    """Simulates a realistic CI/CD incident for interactive live testing."""
    scenario_map = {
        "flake": "TRANSIENT_INFRA_FLAKE",
        "dependency": "DEPENDENCY_CONFLICT",
        "code_bug": "CODE_REGRESSION",
        "lint": "SYNTAX_LINT_ERROR"
    }
    target_category = scenario_map.get(req.scenario)
    if not target_category:
        raise HTTPException(status_code=400, detail=f"Invalid scenario '{req.scenario}'. Choose: flake, dependency, code_bug, lint")

    candidates = [item for item in RAW_LOG_EXAMPLES if item["category"] == target_category]
    selected = candidates[0]

    incident_id = f"SIM-{req.scenario.upper()}-01"
    classifier = get_classifier()

    result = classifier.predict(
        log_text=selected["log"],
        exit_code=selected["exit_code"],
        diff_churn_lines=selected["churn"],
        historical_failure_rate=selected["history"]
    )

    await ws_hub.broadcast("TRIAGE_COMPLETED", {
        "incident_id": incident_id,
        "category": result.category,
        "confidence": result.confidence,
        "routing_decision": result.routing_decision,
        "should_invoke_llm": result.should_invoke_llm,
        "estimated_token_savings": result.estimated_token_savings,
        "estimated_cost_saved_usd": result.estimated_cost_saved_usd,
        "key_indicators": result.key_indicators,
        "latency_ms": result.latency_ms,
        "log_preview": selected["log"][:350]
    })

    return {
        "incident_id": incident_id,
        "scenario": req.scenario,
        "log_text": selected["log"],
        "triage_result": result
    }
