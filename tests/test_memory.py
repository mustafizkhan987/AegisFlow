"""
Unit Tests for Episodic & Semantic Memory Store
===============================================
Verifies AST code chunking, BM25 scoring, and hybrid Reciprocal Rank Fusion.
"""

from aegisflow.memory.chunker import ASTCodeChunker
from aegisflow.memory.store import HybridMemoryStore
from aegisflow.memory.schema import CodeChunk


def test_ast_chunker():
    chunker = ASTCodeChunker()
    python_code = """
def compute_sum(a: int, b: int) -> int:
    \"\"\"Calculates sum of two integers.\"\"\"
    return a + b

class MathEngine:
    def multiply(self, x: float, y: float) -> float:
        return x * y
"""
    chunks = chunker.chunk_python_source("math/engine.py", python_code)
    names = [c.name for c in chunks]

    assert "compute_sum" in names
    assert "MathEngine" in names
    assert "MathEngine.multiply" in names

    sum_chunk = next(c for c in chunks if c.name == "compute_sum")
    assert sum_chunk.start_line == 2
    assert "Calculates sum of two integers." in (sum_chunk.docstring or "")


def test_hybrid_memory_retrieval():
    store = HybridMemoryStore()

    # Search pre-seeded episodic incident trajectories
    results = store.search_incidents("AttributeError discount_multiplier", top_k=2)
    assert len(results) > 0
    top_hit = results[0].item

    assert "discount_multiplier" in top_hit.failure_signature or "discount_multiplier" in top_hit.patch_diff
    assert top_hit.resolution_status == "VERIFIED_SUCCESS"
