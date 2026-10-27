# AegisFlow: Autonomous CI/CD Failure Triage, Root-Cause Analysis & Sandboxed Repair Agent

## 1. Executive Overview

**AegisFlow** is an enterprise-grade autonomous CI/CD failure triage, root-cause diagnosis, and sandboxed patch-verification platform. Engineering organizations lose thousands of engineering hours dealing with flaky test suites, infrastructure glitches, transient network hiccups, dependency conflicts, and production regressions.

Unlike naive LLM wrappers that dump 50,000 lines of raw CI/CD logs directly into an expensive frontier model (costing dollars per run and hallucinating root causes), AegisFlow uses a **two-tier hybrid architecture**:

1. **Sub-50ms Classical ML Gatekeeper (XGBoost + scikit-learn)**: Extracts structured log features (exit codes, entropy, stack depth, error patterns, historical flake rates) to classify incoming failures into:
   - `TRANSIENT_INFRA_FLAKE` -> Auto-retry or discard without invoking LLM (saving 70%+ compute & token costs).
   - `DEPENDENCY_CONFLICT` -> Targeted lockfile/version resolver recipe.
   - `CODE_REGRESSION` -> High-confidence genuine software regression routed to the heavy agent loop.
   - `SYNTAX_LINT_ERROR` -> Fast deterministic auto-fixer.

2. **Autonomous MCP-Driven Remediation Loop**:
   - Standardized tool integration via **Model Context Protocol (MCP)**: Code AST Search, Git Blame/Diff, Sandboxed Runner, File Inspection.
   - **Episodic & Semantic Memory** (Hybrid BM25 + Dense Vectors) storing AST code embeddings and historical incident trajectories.
   - **Ephemeral Isolated Execution Sandbox** (Docker SDK for Python with process-isolated fallback) executing untrusted code with CPU/memory limits and zero network exfiltration.
   - **Full OpenTelemetry Tracing & WebSocket Control Plane** streaming live agent thought graphs, sandbox terminals, and token metrics to a modern dashboard.

---

## 2. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            INCOMING CI/CD TELEMETRY                               |
|        (GitHub Actions Webhooks, GitLab CI, Jenkins, Raw Log Ingestion)           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        FASTAPI INGESTION & NORMALIZER                             |
|               - Log sanitation & ANSI escape stripping                            |
|               - Stack trace segmentation & exit-code parsing                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               TIER 1: HYBRID ML CLASSIFIER & COMPLEXITY ROUTER                   |
|                   (XGBoost + TF-IDF Log Feature Extraction)                       |
+-----------------------------------------+-----------------------------------------+
       |                                  |                                  |
       v (Flake / Timeout)                v (Dep Error)                      v (Code Bug)
+--------------+                   +--------------+                   +-------------+
| AUTO-RETRY / |                   | DEPENDENCY   |                   | TIER 2:     |
| DISCARD NOISE|                   | RESOLVER     |                   | MCP AGENT   |
| (0 LLM Cost) |                   | RECIPE       |                   | LOOP        |
+--------------+                   +--------------+                   +------+------+
                                                                             |
      +----------------------------------------------------------------------+
      |
      v
+-----------------------------------------------------------------------------------+
|                         TIER 2: MCP REPAIR ORCHESTRATOR                           |
|       - Reasoning Loop (Local vLLM / Pluggable Frontier Models)                   |
|       - State Machine: Diagnose -> Hypothesize -> Search AST -> Patch -> Verify   |
+-------------------------------------+---------------------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+-----------------------------+                           +-------------------------+
|   EPISODIC & SEMANTIC       |                           |  MCP TOOL SERVER        |
|   MEMORY STORE              |                           |  - ast_code_search      |
|   - AST Chunks (Dense + BM25|                           |  - git_diff_blame       |
|   - Historical Trajectories |                           |  - read_file_context    |
|   - Prior Successful Patches|                           |  - apply_candidate_patch|
+-----------------------------+                           +------------+------------+
                                                                       |
                                                                       v
+-----------------------------------------------------------------------------------+
|                    SECURE EPHEMERAL EXECUTION SANDBOX                             |
|    - Primary: Docker SDK (Strict nano_cpus, mem_limit="512m", network_mode="none")|
|    - Fallback: Local Jailed Subprocess with strict timeout & environment stripping |
|    - Runs candidate tests in isolation to verify fix without host contamination   |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                    VERIFICATION & DRAFT PR GENERATOR                              |
|   - Patch Diff validated by sandboxed pytest/unit tests                           |
|   - Formats reproduction steps, root cause, and PR description                   |
|   - Emits OpenTelemetry spans & broadcasts via WebSockets to React UI             |
+-----------------------------------------------------------------------------------+
```

---

## 3. Directory Layout & Module Responsibilities

```text
AegisFlow/
├── aegisflow/                       # Core Python Backend Package
│   ├── __init__.py                  # Version & package metadata
│   ├── config.py                    # Pydantic Settings & environment configuration
│   ├── main.py                      # FastAPI application entrypoint & lifespan
│   │
│   ├── classifier/                  # Tier 1: Classical ML Triage & Routing
│   │   ├── __init__.py
│   │   ├── features.py              # Log feature extraction (entropy, stack depth, flags)
│   │   ├── model.py                 # XGBoost + scikit-learn classifier pipeline
│   │   ├── dataset.py               # Synthetic & curated CI failure log datasets
│   │   └── trainer.py               # Model training, calibration, and persistence
│   │
│   ├── memory/                      # Episodic & Semantic Memory Store
│   │   ├── __init__.py
│   │   ├── store.py                 # Hybrid vector + BM25 memory interface
│   │   ├── chunker.py               # AST-aware code chunking for Python & JS/TS
│   │   └── schema.py                # Pydantic models for code embeddings & trajectories
│   │
│   ├── mcp/                         # Model Context Protocol Tools & Server
│   │   ├── __init__.py
│   │   ├── server.py                # Standalone MCP server exposing code & test tools
│   │   ├── tools.py                 # Tool definitions: AST grep, Git blame, test runner
│   │   └── client.py                # MCP client interfacing with LLM agent
│   │
│   ├── sandbox/                     # Secure Ephemeral Execution Environment
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract Sandbox base class & result schemas
│   │   ├── docker_sandbox.py        # Docker SDK isolated container runner
│   │   ├── local_sandbox.py         # Subprocess jail fallback (when Docker is offline)
│   │   └── manager.py               # Auto-detecting Sandbox Manager
│   │
│   ├── agent/                       # Multi-Step Autonomous Repair Loop
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # State machine agent loop (Diagnose -> Fix -> Verify)
│   │   ├── prompts.py               # Structured system prompts & few-shot examples
│   │   └── vllm_client.py           # vLLM / OpenAI-compatible inference client
│   │
│   ├── observability/               # OpenTelemetry Instrumentation & Telemetry
│   │   ├── __init__.py
│   │   ├── tracer.py                # OTel tracer provider, spans, and metrics
│   │   └── websocket_hub.py         # Live WebSocket connection manager & event broadcaster
│   │
│   ├── api/                         # FastAPI Routers & Endpoints
│   │   ├── __init__.py
│   │   ├── webhooks.py              # GitHub / CI log ingestion endpoints
│   │   ├── triage.py                # Triage & classification endpoints
│   │   ├── remediation.py           # Agent repair execution endpoints
│   │   ├── memory_api.py            # Memory search & index management
│   │   └── benchmarks_api.py        # SWE-bench & cost-saving benchmark endpoints
│   │
│   └── benchmarks/                  # Evaluation Suite & Resume Metrics
│       ├── __init__.py
│       ├── cost_filter_eval.py      # XGBoost noise filtering vs token cost reduction
│       ├── swe_bench_subset.py      # Mini SWE-bench verification harness
│       └── retrieval_eval.py        # Hybrid vs naive chunking precision metrics
│
├── frontend/                        # Mission Control Dashboard (React + Vite + TS + Tailwind)
│   ├── src/
│   │   ├── components/              # Live graph, terminal, diff viewer, stats
│   │   ├── App.tsx                  # Main cockpit interface
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── tests/                           # Comprehensive Pytest Suite
│   ├── test_classifier.py           # Unit tests for ML feature extraction & XGBoost
│   ├── test_memory.py               # Unit tests for AST chunker & hybrid memory
│   ├── test_sandbox.py              # Unit tests for local & docker execution sandboxes
│   ├── test_mcp_tools.py            # Unit tests for MCP tool calls
│   └── test_api.py                  # End-to-end API integration tests
│
├── data/                            # Training data, sample logs, and serialized models
├── ARCHITECTURE.md                  # Comprehensive architectural specification
├── QUICKSTART.md                    # Setup, running, and testing guide
└── README.md                        # Flagship presentation & benchmark metrics
```

---

## 4. Key Engineering Trade-Offs & Decisions

| Decision | Selected Approach | Trade-Off & Rationale |
| :--- | :--- | :--- |
| **Triage Classification** | XGBoost + TF-IDF on engineered log features | Sub-15ms inference latency on CPU; filters 70%+ of transient CI flakes without touching costly GPU/vLLM tokens. |
| **Tool Calling Standard** | Model Context Protocol (MCP) | Eliminates brittle custom regex parsing; allows plug-and-play model swapping (vLLM, Claude, GPT-4, Llama-3). |
| **Sandbox Execution** | Dual-mode: Docker SDK + Jailed Local Subprocess | Docker provides network isolation (`network_mode="none"`) and strict memory quotas. Local fallback ensures the project runs on any dev machine even if Docker Desktop is uninstalled. |
| **Vector & Code Memory** | Hybrid Dense + BM25 with Reciprocal Rank Fusion | Dense embeddings capture conceptual similarity; BM25 guarantees exact symbol/identifier matches (crucial for stack traces). |
| **Observability** | OpenTelemetry + WebSockets | Full distributed trace graph with token/latency telemetry streamed live to the React dashboard. |
