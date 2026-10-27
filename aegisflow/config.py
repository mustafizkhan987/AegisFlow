"""
AegisFlow Configuration Module
===============================
This module defines the global configuration settings for AegisFlow using Pydantic Settings.
It reads configuration values from environment variables or a local .env file with sensible,
production-ready defaults.

Key Configuration Areas:
1. Server settings (host, port, debug mode)
2. Classical ML Triage classifier parameters & model artifact paths
3. LLM / vLLM endpoints and inference parameters
4. Episodic & Semantic Memory parameters (hybrid search weights, pgvector/in-memory toggle)
5. Sandbox isolation settings (Docker vs. local subprocess jail quotas and timeouts)
6. OpenTelemetry tracing and WebSocket observability settings
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Determine project root directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"


class Settings(BaseSettings):
    """Global configuration settings for AegisFlow."""

    # --------------------------------------------------------------------------
    # Application & Server Settings
    # --------------------------------------------------------------------------
    app_name: str = Field(default="AegisFlow", description="Application service name")
    app_version: str = Field(default="1.0.0", description="Semantic version string")
    debug: bool = Field(default=False, description="Enable verbose debug logging")
    host: str = Field(default="127.0.0.1", description="FastAPI host bind address")
    port: int = Field(default=8000, description="FastAPI port bind number")

    # --------------------------------------------------------------------------
    # Classical ML Triage & Routing Settings
    # --------------------------------------------------------------------------
    model_dir: Path = Field(
        default=MODELS_DIR,
        description="Path to serialized ML models and vector artifacts"
    )
    triage_model_filename: str = Field(
        default="triage_xgboost.pkl",
        description="Serialized XGBoost classifier model file name"
    )
    triage_vectorizer_filename: str = Field(
        default="triage_tfidf.pkl",
        description="Serialized TF-IDF log vectorizer file name"
    )
    flake_confidence_threshold: float = Field(
        default=0.75,
        description="Confidence threshold above which an incident is classified as transient flake without LLM invocation"
    )

    # --------------------------------------------------------------------------
    # LLM / vLLM Inference Settings
    # --------------------------------------------------------------------------
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for local vLLM or OpenAI-compatible inference server"
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM / OpenAI server (default 'EMPTY' for local vLLM)"
    )
    vllm_model_name: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="Target coding LLM model name loaded in vLLM"
    )
    llm_temperature: float = Field(
        default=0.2,
        description="Temperature for deterministic code repair generation"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens generated per remediation step"
    )
    enable_mock_llm_fallback: bool = Field(
        default=True,
        description="When vLLM is unreachable, use the AST-guided deterministic repair engine for seamless local testing"
    )

    # --------------------------------------------------------------------------
    # Episodic & Semantic Memory Settings
    # --------------------------------------------------------------------------
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string with pgvector (e.g. postgresql://user:pass@localhost:5432/aegisflow). If None, uses high-speed in-memory vector store."
    )
    hybrid_dense_weight: float = Field(
        default=0.6,
        description="Weight alpha for dense vector similarity in Reciprocal Rank Fusion (0.0 to 1.0)"
    )
    hybrid_bm25_weight: float = Field(
        default=0.4,
        description="Weight beta for sparse BM25 symbol matching in Reciprocal Rank Fusion"
    )
    memory_top_k: int = Field(
        default=5,
        description="Number of relevant code chunks or past incident trajectories to retrieve"
    )

    # --------------------------------------------------------------------------
    # Ephemeral Sandbox Isolation Settings
    # --------------------------------------------------------------------------
    sandbox_backend: str = Field(
        default="auto",
        description="Sandbox execution backend: 'docker', 'local', or 'auto' (detects Docker availability)"
    )
    sandbox_docker_image: str = Field(
        default="python:3.11-slim",
        description="Base Docker image for sandboxed verification containers"
    )
    sandbox_timeout_seconds: int = Field(
        default=30,
        description="Maximum wall-clock execution time allowed for running tests in the sandbox"
    )
    sandbox_memory_limit: str = Field(
        default="512m",
        description="Strict memory limit quota for execution containers"
    )
    sandbox_cpu_limit: float = Field(
        default=1.0,
        description="Maximum CPU cores allocated to execution containers (via nano_cpus)"
    )

    # --------------------------------------------------------------------------
    # Observability & Cost Tracking
    # --------------------------------------------------------------------------
    enable_opentelemetry: bool = Field(
        default=True,
        description="Enable OpenTelemetry distributed tracing spans"
    )
    estimated_cost_per_1k_tokens_usd: float = Field(
        default=0.003,
        description="Estimated token cost rate to quantify money saved by XGBoost filtering"
    )

    model_config = SettingsConfigDict(
        env_prefix="AEGISFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton configuration instance
settings = Settings()

# Ensure data directories exist
settings.model_dir.mkdir(parents=True, exist_ok=True)
