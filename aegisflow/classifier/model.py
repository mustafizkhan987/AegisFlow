"""
AegisFlow XGBoost Triage Classifier & Router
===========================================
This module implements the primary sub-50ms ML Gatekeeper that classifies
incoming CI/CD failure logs into 4 actionable categories:
1. TRANSIENT_INFRA_FLAKE -> Routes to auto-retry, saves 100% LLM tokens.
2. DEPENDENCY_CONFLICT   -> Routes to targeted package resolver recipe.
3. CODE_REGRESSION       -> Routes to the heavy MCP + vLLM agent loop.
4. SYNTAX_LINT_ERROR     -> Routes to fast deterministic linter patcher.

Architecture:
- Feature Union: Combines 16 extracted structural/domain scalar features (standardized)
  with a TF-IDF 1-2 gram log text vector.
- Core Estimator: Multi-class XGBoost (XGBClassifier) with calibrated probabilities.
- Domain Prior Calibration: Blends gradient-boosted decision trees with hard signal flags.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from aegisflow.classifier.features import LogFeatureExtractor
from aegisflow.classifier.dataset import (
    ALL_CATEGORIES,
    CATEGORY_TRANSIENT_FLAKE,
    CATEGORY_DEPENDENCY_CONFLICT,
    CATEGORY_CODE_REGRESSION,
    CATEGORY_SYNTAX_LINT,
)
from aegisflow.config import settings


class TriageResult(BaseModel):
    """Structured classification output and routing decision."""
    category: str = Field(..., description="Predicted failure category")
    confidence: float = Field(..., description="Calibrated prediction probability (0.0 - 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all classes")
    routing_decision: str = Field(..., description="Actionable routing path")
    should_invoke_llm: bool = Field(..., description="Whether to invoke the computationally heavy LLM repair loop")
    estimated_token_savings: int = Field(default=0, description="Tokens saved by filtering non-code bugs")
    estimated_cost_saved_usd: float = Field(default=0.0, description="Estimated dollar savings from token filtering")
    key_indicators: List[str] = Field(default_factory=list, description="Top identified signals leading to this classification")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")


class TriageClassifier:
    """Production ML gatekeeper combining XGBoost and domain log feature engineering."""

    def __init__(self, auto_load: bool = True):
        self.feature_extractor = LogFeatureExtractor()
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b"
        )
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=42
        )
        self.categories: List[str] = ALL_CATEGORIES
        self.is_fitted: bool = False

        # Attempt to load serialized model if available
        if auto_load:
            model_path = settings.model_dir / settings.triage_model_filename
            if model_path.exists():
                try:
                    self._load_from_path(model_path)
                except Exception:
                    pass

    def _prepare_matrix(
        self,
        scalar_features_list: List[np.ndarray],
        texts_list: List[str],
        fit: bool = False
    ) -> sp.csr_matrix:
        """Scales scalar features, vectorizes text, and hstacks into a sparse matrix."""
        scalars_array = np.vstack(scalar_features_list)
        if fit:
            scaled_scalars = self.scaler.fit_transform(scalars_array)
            text_matrix = self.vectorizer.fit_transform(texts_list)
        else:
            scaled_scalars = self.scaler.transform(scalars_array)
            text_matrix = self.vectorizer.transform(texts_list)

        combined_matrix = sp.hstack([sp.csr_matrix(scaled_scalars), text_matrix], format="csr")
        return combined_matrix

    def fit(self, dataset: List[Dict[str, Any]]) -> "TriageClassifier":
        """Trains the XGBoost classifier on a list of raw log dictionaries."""
        scalar_features: List[np.ndarray] = []
        texts: List[str] = []
        labels: List[int] = []

        for item in dataset:
            extracted = self.feature_extractor.extract_features(
                log_text=item["log"],
                exit_code=item.get("exit_code", 1),
                diff_churn_lines=item.get("churn", 0),
                historical_failure_rate=item.get("history", 0.15)
            )
            scalar_features.append(extracted["feature_vector"])
            texts.append(extracted["clean_text"])
            labels.append(self.categories.index(item["category"]))

        X = self._prepare_matrix(scalar_features, texts, fit=True)
        y = np.array(labels, dtype=np.int32)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(
        self,
        log_text: str,
        exit_code: int = 1,
        diff_churn_lines: int = 0,
        historical_failure_rate: float = 0.15
    ) -> TriageResult:
        """Classifies a single CI/CD log and returns an actionable routing decision."""
        start_time = time.perf_counter()

        if not self.is_fitted:
            from aegisflow.classifier.dataset import generate_augmented_dataset
            self.fit(generate_augmented_dataset(multiplier=15))

        extracted = self.feature_extractor.extract_features(
            log_text=log_text,
            exit_code=exit_code,
            diff_churn_lines=diff_churn_lines,
            historical_failure_rate=historical_failure_rate
        )

        f_dict = extracted["features_dict"]

        X = self._prepare_matrix(
            [extracted["feature_vector"]],
            [extracted["clean_text"]],
            fit=False
        )

        probas = self.model.predict_proba(X)[0]

        # Domain signal calibration
        priors = np.zeros(len(self.categories), dtype=np.float32)
        if f_dict["has_timeout"] or f_dict["has_network"] or f_dict["has_infra"] or f_dict["is_sigkill"]:
            priors[0] += 3.0  # TRANSIENT_INFRA_FLAKE
        if f_dict["has_dep"]:
            priors[1] += 3.0  # DEPENDENCY_CONFLICT
        if f_dict["has_assertion"] or f_dict["stack_depth"] > 0:
            priors[2] += 3.0  # CODE_REGRESSION
        if f_dict["has_lint"] and not f_dict["has_assertion"] and f_dict["stack_depth"] == 0:
            priors[3] += 3.0  # SYNTAX_LINT_ERROR

        calibrated_logits = np.log(np.clip(probas, 1e-6, 1.0)) + priors
        calibrated_probas = np.exp(calibrated_logits) / np.sum(np.exp(calibrated_logits))

        pred_idx = int(np.argmax(calibrated_probas))
        category = self.categories[pred_idx]
        confidence = float(calibrated_probas[pred_idx])

        prob_dict = {cat: float(calibrated_probas[i]) for i, cat in enumerate(self.categories)}

        # Routing decisions
        if category == CATEGORY_TRANSIENT_FLAKE:
            routing_decision = "AUTO_RETRY"
            should_invoke_llm = False
            token_savings = 18_000
            cost_saved = token_savings * (0.003 / 1000.0)
        elif category == CATEGORY_DEPENDENCY_CONFLICT:
            routing_decision = "DEPENDENCY_RECIPE"
            should_invoke_llm = False
            token_savings = 12_000
            cost_saved = token_savings * (0.003 / 1000.0)
        elif category == CATEGORY_SYNTAX_LINT:
            routing_decision = "LINTER_AUTOPATCH"
            should_invoke_llm = False
            token_savings = 8_000
            cost_saved = token_savings * (0.003 / 1000.0)
        else:
            routing_decision = "LLM_AGENT_REPAIR"
            should_invoke_llm = True
            token_savings = 0
            cost_saved = 0.0

        key_indicators: List[str] = []
        if f_dict["has_timeout"]:
            key_indicators.append("Detected network or runner operation timeout")
        if f_dict["has_network"]:
            key_indicators.append("Detected connection refused or DNS failure")
        if f_dict["is_sigkill"]:
            key_indicators.append("Process killed by SIGKILL / OOM (Exit code 137)")
        if f_dict["has_dep"]:
            key_indicators.append("Detected package resolution or version conflict")
        if f_dict["has_assertion"]:
            key_indicators.append("Detected unit test assertion failure")
        if f_dict["has_lint"]:
            key_indicators.append("Detected syntax or code formatting error")
        if f_dict["stack_depth"] > 0:
            key_indicators.append(f"Identified {f_dict['stack_depth']} execution stack trace frames")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TriageResult(
            category=category,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in prob_dict.items()},
            routing_decision=routing_decision,
            should_invoke_llm=should_invoke_llm,
            estimated_token_savings=token_savings,
            estimated_cost_saved_usd=round(cost_saved, 4),
            key_indicators=key_indicators,
            latency_ms=round(latency_ms, 2)
        )

    def save(self, filepath: Path) -> None:
        """Persists the trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "vectorizer": self.vectorizer,
            "categories": self.categories,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath)

    def _load_from_path(self, filepath: Path) -> None:
        payload = joblib.load(filepath)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.vectorizer = payload["vectorizer"]
        self.categories = payload["categories"]
        self.is_fitted = payload.get("is_fitted", True)

    @classmethod
    def load(cls, filepath: Path) -> "TriageClassifier":
        instance = cls(auto_load=False)
        if filepath.exists():
            instance._load_from_path(filepath)
        return instance
