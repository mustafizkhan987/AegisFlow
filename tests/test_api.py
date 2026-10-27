"""
Integration Tests for AegisFlow API Endpoints
==============================================
Verifies health check, triage classification, simulation, memory search,
and remediation routes using FastAPI TestClient.
"""

from fastapi.testclient import TestClient
from aegisflow.main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["service"] == "AegisFlow"


def test_triage_classification_endpoint():
    payload = {
        "log_text": "Connection timed out after 30000ms\n504 Gateway Time-out",
        "exit_code": 143,
        "diff_churn_lines": 5
    }
    res = client.post("/api/triage/classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "TRANSIENT_INFRA_FLAKE"
    assert data["should_invoke_llm"] is False
    assert data["routing_decision"] == "AUTO_RETRY"


def test_simulate_endpoint():
    res = client.post("/api/triage/simulate", json={"scenario": "code_bug"})
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "code_bug"
    assert data["triage_result"]["category"] == "CODE_REGRESSION"
    assert data["triage_result"]["should_invoke_llm"] is True


def test_benchmarks_summary():
    res = client.get("/api/benchmarks/summary")
    assert res.status_code == 200
    data = res.json()
    assert "cost_filtering" in data
    assert "swe_bench_subset" in data
    assert "retrieval_precision" in data
    assert data["cost_filtering"]["savings_percentage"] > 50.0


def test_memory_incidents_list():
    res = client.get("/api/memory/incidents")
    assert res.status_code == 200
    data = res.json()
    assert data["total_trajectories"] >= 3
