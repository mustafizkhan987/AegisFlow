"""
AegisFlow ML Triage Feature Extraction
======================================
This module transforms unstructured, multi-thousand-line CI/CD logs, exit codes,
and metadata into structured numerical and linguistic feature vectors for the
XGBoost Triage Classifier.

Key Extracted Features:
1. Log Entropy: Character and token Shannon entropy (distinguishes structured traces from random crashes/dumps).
2. Stack Trace Depth: Counts nested 'File "...", line X' frames.
3. Exception Count & Categorization: Matches common Python/Node/Go runtime exceptions.
4. Exit Code & Execution Status: Normalizes process exit codes (e.g. 1, 137 [SIGKILL], 143 [SIGTERM]).
5. Domain Keyword Flag Vectors: Fast boolean indicator flags for timeouts, network drops, dependency conflicts, assertions, and lint failures.
6. Diff & Historical Metrics: Churn lines and historical failure priors.
"""

import math
import re
from typing import Dict, List, Any
import numpy as np


class LogFeatureExtractor:
    """Extracts domain-specific features from raw CI/CD build and test logs."""

    # Pre-compiled regex patterns for maximum extraction speed
    STACK_TRACE_PATTERN = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)', re.IGNORECASE)
    EXCEPTION_PATTERN = re.compile(r'([A-Za-z0-9_]+(?:Error|Exception|Failure)):', re.MULTILINE)
    PYTEST_FAIL_PATTERN = re.compile(r'FAILED\s+([^\s]+::[^\s]+)', re.MULTILINE)

    # Domain indicator keyword sets
    TIMEOUT_KEYWORDS = [
        "timed out", "timeout", "deadline exceeded", "operation timed out",
        "job timed out", "step timed out", "connection reset by peer", "readtimeout"
    ]
    NETWORK_KEYWORDS = [
        "502 bad gateway", "503 service unavailable", "504 gateway time-out",
        "econnrefused", "connection refused", "temporary failure in name resolution",
        "failed to establish a new connection", "max retries exceeded", "tls handshake error",
        "network is unreachable"
    ]
    INFRA_KEYWORDS = [
        "runner lost communication", "no space left on device", "disk full",
        "docker daemon is not running", "oomkilled", "exit code 137", "out of memory",
        "segmentation fault", "killed", "runner disconnected"
    ]
    DEPENDENCY_KEYWORDS = [
        "could not find a version that satisfies the requirement",
        "resolutionimpossible", "no matching distribution found",
        "unable to resolve dependency tree", "modulenotfounderror",
        "importerror: cannot import name", "package not found",
        "conflicting peer dependency", "failed building wheel", "pip failed"
    ]
    ASSERTION_KEYWORDS = [
        "assertionerror", "assert ", "failed (failures=", "expected:",
        "received:", "differs from", "failed tests/", "test failed"
    ]
    LINT_KEYWORDS = [
        "syntaxerror", "indentationerror", "taberror", "flake8", "eslint",
        "ruff failed", "formatting violation", "trailing whitespace", "black would reformat"
    ]

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculates Shannon entropy of the log snippet to measure randomness/entropy."""
        if not text:
            return 0.0
        # Sample first 4000 characters to ensure sub-millisecond calculation
        sample = text[:4000]
        length = len(sample)
        counts: Dict[str, int] = {}
        for char in sample:
            counts[char] = counts.get(char, 0) + 1

        entropy = 0.0
        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)
        return float(entropy)

    def extract_features(
        self,
        log_text: str,
        exit_code: int = 1,
        diff_churn_lines: int = 0,
        historical_failure_rate: float = 0.15
    ) -> Dict[str, Any]:
        """
        Parses a log text string and extracts a structured feature dictionary.

        Args:
            log_text: Raw CI/CD log output.
            exit_code: The exit status code returned by the CI runner (e.g. 1, 137).
            diff_churn_lines: Number of modified code lines in the failing commit.
            historical_failure_rate: Historical flakiness score of this test/job (0.0 - 1.0).

        Returns:
            Dictionary containing numeric scalar features and clean text snippet.
        """
        text_lower = log_text.lower()
        lines = log_text.splitlines()
        total_lines = len(lines)
        total_chars = len(log_text)

        # 1. Structural features
        entropy = self.calculate_shannon_entropy(log_text)
        stack_frames = len(self.STACK_TRACE_PATTERN.findall(log_text))
        exceptions = len(self.EXCEPTION_PATTERN.findall(log_text))
        pytest_failures = len(self.PYTEST_FAIL_PATTERN.findall(log_text))

        # 2. Flag keyword counters (normalized)
        has_timeout = 1.0 if any(k in text_lower for k in self.TIMEOUT_KEYWORDS) else 0.0
        has_network = 1.0 if any(k in text_lower for k in self.NETWORK_KEYWORDS) else 0.0
        has_infra = 1.0 if any(k in text_lower for k in self.INFRA_KEYWORDS) else 0.0
        has_dep = 1.0 if any(k in text_lower for k in self.DEPENDENCY_KEYWORDS) else 0.0
        has_assertion = 1.0 if any(k in text_lower for k in self.ASSERTION_KEYWORDS) else 0.0
        has_lint = 1.0 if any(k in text_lower for k in self.LINT_KEYWORDS) else 0.0

        # Normalization helpers
        log_len_log1p = float(math.log1p(total_chars))
        lines_log1p = float(math.log1p(total_lines))
        stack_depth_scaled = float(min(stack_frames, 50) / 50.0)

        # Exit code encoding (137 = SIGKILL/OOM, 143 = SIGTERM, 1 = error, 2 = usage error)
        is_sigkill = 1.0 if exit_code == 137 else 0.0
        is_sigterm = 1.0 if exit_code == 143 else 0.0

        feature_vector = [
            entropy,
            log_len_log1p,
            lines_log1p,
            stack_depth_scaled,
            float(min(exceptions, 20)),
            float(min(pytest_failures, 20)),
            has_timeout,
            has_network,
            has_infra,
            has_dep,
            has_assertion,
            has_lint,
            is_sigkill,
            is_sigterm,
            float(min(diff_churn_lines, 500) / 500.0),
            float(historical_failure_rate)
        ]

        return {
            "feature_vector": np.array(feature_vector, dtype=np.float32),
            "features_dict": {
                "entropy": entropy,
                "log_len_log1p": log_len_log1p,
                "lines_log1p": lines_log1p,
                "stack_depth": stack_frames,
                "exceptions_count": exceptions,
                "pytest_failures": pytest_failures,
                "has_timeout": has_timeout,
                "has_network": has_network,
                "has_infra": has_infra,
                "has_dep": has_dep,
                "has_assertion": has_assertion,
                "has_lint": has_lint,
                "is_sigkill": is_sigkill,
                "diff_churn_lines": diff_churn_lines,
                "historical_failure_rate": historical_failure_rate
            },
            "clean_text": " ".join(lines[-150:]) if len(lines) > 150 else log_text
        }
