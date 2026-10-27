"""
AegisFlow Hybrid Memory Store (Dense Vector + BM25 + Reciprocal Rank Fusion)
=============================================================================
Provides unified semantic and lexical retrieval across:
1. AST Code Chunks: Finds exact functions, classes, and logic blocks.
2. Episodic Incident Trajectories: Retrieves past incident root-cause diagnoses,
   unified diff patches, and verification runs to guide the agent.

Why Hybrid RRF (Dense + BM25) is Essential for Code & Bug Repair:
- Dense embeddings capture conceptual meaning (e.g. 'expired session validation').
- BM25 captures exact symbol identifiers and error messages (e.g. 'discount_multiplier').
- Reciprocal Rank Fusion merges both ranked lists with zero hyperparameter instability.
"""

import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aegisflow.memory.schema import CodeChunk, IncidentTrajectory, MemorySearchResult
from aegisflow.memory.chunker import ASTCodeChunker
from aegisflow.config import settings


class BM25Engine:
    """Lightweight, zero-external-dependency Okapi BM25 implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into words, symbols, and camelCase sub-words."""
        # Split on non-alphanumeric and split camelCase
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        tokens = []
        for t in raw_tokens:
            tokens.append(t)
            # CamelCase decomposition
            sub_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", t)
            if len(sub_tokens) > 1:
                tokens.extend([st.lower() for st in sub_tokens])
        return tokens

    def fit(self, documents: List[str]):
        """Indexes documents and computes IDF weights."""
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_term_freqs = []
        self.doc_lens = []
        doc_frequencies: Dict[str, int] = {}

        total_length = 0
        for doc in documents:
            tokens = self.tokenize(doc)
            length = len(tokens)
            self.doc_lens.append(length)
            total_length += length

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in tf.keys():
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

        self.avg_doc_len = total_length / max(1, self.corpus_size)

        # Compute IDF
        self.idf = {}
        for token, df in doc_frequencies.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        """Returns BM25 relevance scores for all indexed documents."""
        if self.corpus_size == 0:
            return np.array([], dtype=np.float32)

        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for q in query_tokens:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]
            for doc_idx, tf_dict in enumerate(self.doc_term_freqs):
                tf = tf_dict.get(q, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf_val * (numerator / denominator)

        return scores


class HybridMemoryStore:
    """
    Episodic & Semantic Hybrid Memory Store for AegisFlow.
    Indexes AST code chunks and past incident resolution trajectories.
    """

    def __init__(
        self,
        dense_weight: float = settings.hybrid_dense_weight,
        bm25_weight: float = settings.hybrid_bm25_weight
    ):
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # Code chunks repository
        self.code_chunks: List[CodeChunk] = []
        self.code_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.code_dense_matrix: Optional[np.ndarray] = None
        self.code_bm25 = BM25Engine()

        # Episodic incident memory repository
        self.trajectories: List[IncidentTrajectory] = []
        self.incident_vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2))
        self.incident_dense_matrix: Optional[np.ndarray] = None
        self.incident_bm25 = BM25Engine()

        # Seed with initial curated episodic knowledge
        self._seed_curated_trajectories()

    def _seed_curated_trajectories(self):
        """Pre-seeds episodic memory with realistic historical bug fixes."""
        seeds = [
            IncidentTrajectory(
                incident_id="INC-2026-101",
                failure_signature="AttributeError: 'User' object has no attribute 'discount_multiplier'",
                log_summary="billing/calculator.py:88: AttributeError: 'User' object has no attribute 'discount_multiplier'",
                root_cause="User domain entity missing discount_multiplier field when tier defaults or discounts are applied.",
                failing_file="billing/calculator.py",
                patch_diff="""--- a/billing/calculator.py
+++ b/billing/calculator.py
@@ -87,2 +87,3 @@
-    discount = user.discount_multiplier * cart.subtotal
+    discount_rate = getattr(user, "discount_multiplier", 0.0)
+    discount = discount_rate * cart.subtotal
""",
                test_command="pytest tests/unit/test_billing.py::test_discount_application",
                resolution_status="VERIFIED_SUCCESS",
                tags=["python", "billing", "attribute_error"]
            ),
            IncidentTrajectory(
                incident_id="INC-2026-102",
                failure_signature="AssertionError: assert True is False in test_user_session_expiry",
                log_summary="tests/auth/test_session.py:65: AssertionError: assert True is False",
                root_cause="Session expiry logic used <= rather than < when checking current timestamp against expires_at.",
                failing_file="auth/session.py",
                patch_diff="""--- a/auth/session.py
+++ b/auth/session.py
@@ -111,2 +111,2 @@
-    return current_timestamp <= self.expires_at
+    return current_timestamp < self.expires_at
""",
                test_command="pytest tests/auth/test_session.py",
                resolution_status="VERIFIED_SUCCESS",
                tags=["python", "auth", "boundary_condition"]
            ),
            IncidentTrajectory(
                incident_id="INC-2026-103",
                failure_signature="TypeError: 'NoneType' object has no attribute 'strip' in batch_encode",
                log_summary="ml/tokenizer.py:42: TypeError: 'NoneType' object has no attribute 'strip'",
                root_cause="batch_encode does not guard against None elements in input text lists.",
                failing_file="ml/tokenizer.py",
                patch_diff="""--- a/ml/tokenizer.py
+++ b/ml/tokenizer.py
@@ -41,3 +41,5 @@
     def encode(self, text):
+        if text is None:
+            return []
         words = text.strip().split()
""",
                test_command="pytest tests/ml/test_tokenizer.py",
                resolution_status="VERIFIED_SUCCESS",
                tags=["python", "ml", "null_check"]
            )
        ]
        self.add_incident_trajectories(seeds)

    # --------------------------------------------------------------------------
    # Code Chunks Management
    # --------------------------------------------------------------------------
    def index_repository(self, repo_dir: Path, extensions: List[str] = [".py", ".ts", ".js", ".json"]) -> int:
        """Indexes all code files in the given directory using the AST chunker."""
        chunker = ASTCodeChunker()
        new_chunks: List[CodeChunk] = []

        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                if any(ignored in file_path.parts for ignored in [".git", ".venv", "node_modules", "__pycache__", "dist"]):
                    continue
                chunks = chunker.chunk_file(file_path, repo_root=repo_dir)
                new_chunks.extend(chunks)

        self.code_chunks.extend(new_chunks)
        self._rebuild_code_index()
        return len(new_chunks)

    def add_code_chunks(self, chunks: List[CodeChunk]):
        """Adds a list of code chunks to the memory store."""
        self.code_chunks.extend(chunks)
        self._rebuild_code_index()

    def _rebuild_code_index(self):
        """Rebuilds TF-IDF dense matrix and BM25 index for code chunks."""
        if not self.code_chunks:
            return
        texts = [f"{c.file_path} {c.name} {c.content} {c.docstring or ''}" for c in self.code_chunks]
        self.code_dense_matrix = self.code_vectorizer.fit_transform(texts).toarray()
        self.code_bm25.fit(texts)

    # --------------------------------------------------------------------------
    # Episodic Incidents Management
    # --------------------------------------------------------------------------
    def add_incident_trajectories(self, trajectories: List[IncidentTrajectory]):
        """Adds verified incident trajectories to episodic memory."""
        self.trajectories.extend(trajectories)
        self._rebuild_incident_index()

    def _rebuild_incident_index(self):
        """Rebuilds TF-IDF dense matrix and BM25 index for incident trajectories."""
        if not self.trajectories:
            return
        texts = [
            f"{t.failure_signature} {t.log_summary} {t.root_cause} {t.failing_file} {' '.join(t.tags)}"
            for t in self.trajectories
        ]
        self.incident_dense_matrix = self.incident_vectorizer.fit_transform(texts).toarray()
        self.incident_bm25.fit(texts)

    # --------------------------------------------------------------------------
    # Hybrid Search via Reciprocal Rank Fusion (RRF)
    # --------------------------------------------------------------------------
    def _reciprocal_rank_fusion(
        self,
        dense_scores: np.ndarray,
        bm25_scores: np.ndarray,
        k_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """Combines dense and BM25 score arrays into a single RRF-ranked index list."""
        n = len(dense_scores)
        if n == 0:
            return []

        # Sort indices by score descending
        dense_ranking = np.argsort(-dense_scores)
        bm25_ranking = np.argsort(-bm25_scores)

        # Build rank maps (0-indexed)
        rank_dense = {idx: rank for rank, idx in enumerate(dense_ranking)}
        rank_bm25 = {idx: rank for rank, idx in enumerate(bm25_ranking)}

        fused_results = []
        for i in range(n):
            r_d = rank_dense[i]
            r_b = rank_bm25[i]
            # Standard RRF formula weighted by configured weights
            rrf_score = (self.dense_weight / (k_constant + r_d)) + (self.bm25_weight / (k_constant + r_b))
            fused_results.append({
                "index": i,
                "score": float(rrf_score),
                "dense_score": float(dense_scores[i]),
                "bm25_score": float(bm25_scores[i]),
                "rank_dense": r_d + 1,
                "rank_bm25": r_b + 1
            })

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results

    def search_code(self, query: str, top_k: int = 5) -> List[MemorySearchResult]:
        """Performs hybrid code retrieval (Dense + BM25 + RRF) for code chunks."""
        if not self.code_chunks or self.code_dense_matrix is None:
            return []

        # 1. Dense cosine similarity
        query_dense = self.code_vectorizer.transform([query]).toarray()
        dense_scores = cosine_similarity(query_dense, self.code_dense_matrix)[0]

        # 2. BM25 scoring
        bm25_scores = self.code_bm25.score(query)

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(dense_scores, bm25_scores)

        results: List[MemorySearchResult] = []
        for item in fused[:top_k]:
            idx = item["index"]
            results.append(MemorySearchResult(
                item=self.code_chunks[idx],
                score=round(item["score"], 5),
                dense_score=round(item["dense_score"], 4),
                bm25_score=round(item["bm25_score"], 4),
                rank_dense=item["rank_dense"],
                rank_bm25=item["rank_bm25"]
            ))

        return results

    def search_incidents(self, query: str, top_k: int = 3) -> List[MemorySearchResult]:
        """Retrieves similar prior incident trajectories from episodic memory."""
        if not self.trajectories or self.incident_dense_matrix is None:
            return []

        query_dense = self.incident_vectorizer.transform([query]).toarray()
        dense_scores = cosine_similarity(query_dense, self.incident_dense_matrix)[0]
        bm25_scores = self.incident_bm25.score(query)

        fused = self._reciprocal_rank_fusion(dense_scores, bm25_scores)

        results: List[MemorySearchResult] = []
        for item in fused[:top_k]:
            idx = item["index"]
            results.append(MemorySearchResult(
                item=self.trajectories[idx],
                score=round(item["score"], 5),
                dense_score=round(item["dense_score"], 4),
                bm25_score=round(item["bm25_score"], 4),
                rank_dense=item["rank_dense"],
                rank_bm25=item["rank_bm25"]
            ))

        return results
