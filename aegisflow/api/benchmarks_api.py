"""
AegisFlow Benchmarks API Endpoints
==================================
Exposes quantitative benchmark evaluation metrics for resume and portfolio reporting:
1. Cost & Compute Filtering (Token reduction vs. naive LLM invocation)
2. Autonomous Bug Resolution Rate (Pass@1 comparison on SWE-bench subset)
3. AST Hybrid Retrieval Precision & Groundedness vs. Naive chunking
"""

from fastapi import APIRouter
from aegisflow.benchmarks.cost_filter_eval import run_cost_filter_benchmark
from aegisflow.benchmarks.swe_bench_subset import run_swe_bench_evaluation
from aegisflow.benchmarks.retrieval_eval import run_retrieval_benchmark

router = APIRouter(prefix="/api/benchmarks", tags=["Evaluation & Benchmarks"])

# Cache benchmark results in-memory
_cached_summary = None


@router.get("/summary")
async def get_benchmark_summary():
    """Returns the latest benchmark metrics across all 3 key evaluation pillars."""
    global _cached_summary
    if _cached_summary is None:
        _cached_summary = {
            "cost_filtering": run_cost_filter_benchmark(num_samples=500),
            "swe_bench_subset": run_swe_bench_evaluation(),
            "retrieval_precision": run_retrieval_benchmark()
        }
    return _cached_summary


@router.post("/run")
async def run_full_benchmark():
    """Re-executes the comprehensive benchmark suite."""
    global _cached_summary
    cost_res = run_cost_filter_benchmark(num_samples=1000)
    swe_res = run_swe_bench_evaluation()
    retrieval_res = run_retrieval_benchmark()

    _cached_summary = {
        "cost_filtering": cost_res,
        "swe_bench_subset": swe_res,
        "retrieval_precision": retrieval_res
    }
    return _cached_summary
