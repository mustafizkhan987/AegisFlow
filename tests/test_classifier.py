"""
Unit Tests for AegisFlow Classical ML Triage Classifier
======================================================
Verifies feature extraction, XGBoost inference, and routing decisions.
"""

import pytest
from aegisflow.classifier.features import LogFeatureExtractor
from aegisflow.classifier.model import TriageClassifier
from aegisflow.classifier.dataset import RAW_LOG_EXAMPLES


def test_feature_extraction():
    extractor = LogFeatureExtractor()
    sample_log = """
File "app/main.py", line 42, in calculate
    result = x / 0
ZeroDivisionError: division by zero
FAILED tests/test_math.py::test_division
"""
    extracted = extractor.extract_features(sample_log, exit_code=1, diff_churn_lines=25)
    f_dict = extracted["features_dict"]

    assert f_dict["stack_depth"] >= 1
    assert f_dict["exceptions_count"] >= 1
    assert f_dict["pytest_failures"] >= 1
    assert f_dict["has_assertion"] == 1.0
    assert len(extracted["feature_vector"]) == 16


def test_triage_classification():
    classifier = TriageClassifier()
    # Test transient flake classification
    flake_log = "Error: Failed to establish a new connection: [Errno 110] Connection timed out\n503 Service Unavailable"
    res_flake = classifier.predict(flake_log, exit_code=143)

    assert res_flake.category == "TRANSIENT_INFRA_FLAKE"
    assert res_flake.should_invoke_llm is False
    assert res_flake.routing_decision == "AUTO_RETRY"
    assert res_flake.estimated_token_savings > 0

    # Test code regression classification
    code_log = """
billing/calculator.py:88: AttributeError: 'User' object has no attribute 'discount_multiplier'
FAILED tests/unit/test_billing.py::test_discount_application - AttributeError
"""
    res_code = classifier.predict(code_log, exit_code=1, diff_churn_lines=50)

    assert res_code.category == "CODE_REGRESSION"
    assert res_code.should_invoke_llm is True
    assert res_code.routing_decision == "LLM_AGENT_REPAIR"
