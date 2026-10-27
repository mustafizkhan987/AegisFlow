"""
Integration Test for Autonomous Remediation Loop
================================================
Verifies that the RemediationOrchestrator takes a broken CI test log,
retrieves AST context & episodic memory, synthesizes a patch,
verifies it in the sandbox, and generates a draft Pull Request.
"""

import pytest
from aegisflow.agent.orchestrator import RemediationOrchestrator, RemediationRequest


@pytest.mark.anyio
async def test_autonomous_remediation_pipeline():
    orchestrator = RemediationOrchestrator()

    broken_log = """
=================================== FAILURES ===================================
billing/calculator.py:88: in calculate_final_invoice
    discount = user.discount_multiplier * cart.subtotal
E   AttributeError: 'User' object has no attribute 'discount_multiplier'

File "billing/calculator.py", line 88, in calculate_final_invoice
    discount = user.discount_multiplier * cart.subtotal
AttributeError: 'User' object has no attribute 'discount_multiplier'
FAILED tests/unit/test_billing.py::test_discount_application - AttributeError
"""

    req = RemediationRequest(
        incident_id="TEST-INC-001",
        log_text=broken_log,
        failing_file="billing/calculator.py"
    )

    response = await orchestrator.remediate(req)

    assert response.incident_id == "TEST-INC-001"
    assert response.status == "RESOLVED"
    assert "discount_multiplier" in response.patch_diff or "discount_rate" in response.patch_diff
    assert response.draft_pr_title is not None
    assert "calculator" in response.draft_pr_title.lower()
    assert response.draft_pr_body is not None
    assert response.trajectory_persisted is True
