"""
AegisFlow Episodic & Semantic Memory Schema
===========================================
Defines Pydantic data models for:
1. CodeChunk: AST-partitioned code blocks with line markers and syntax metadata.
2. IncidentTrajectory: Episodic memory records capturing past failure traces,
   root cause analyses, unified diff patches, and verification outcomes.
3. MemorySearchResult: Ranked hybrid retrieval container with RRF scoring metadata.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """Represents a syntactically coherent AST chunk of source code."""
    chunk_id: str = Field(..., description="Unique deterministic identifier (hash of file + lines)")
    file_path: str = Field(..., description="Repository-relative file path")
    chunk_type: str = Field(..., description="AST node type: 'function', 'class', 'method', or 'module'")
    name: str = Field(..., description="Identifier name (function name, class name)")
    start_line: int = Field(..., description="1-indexed starting line number")
    end_line: int = Field(..., description="1-indexed ending line number")
    content: str = Field(..., description="Exact raw source code of this chunk")
    docstring: Optional[str] = Field(default=None, description="Docstring if available")
    signature: Optional[str] = Field(default=None, description="Function/class signature definition")
    embedding: Optional[List[float]] = Field(default=None, description="Normalized dense embedding vector")


class IncidentTrajectory(BaseModel):
    """Records an episodic trajectory of a past CI failure and its verified resolution."""
    incident_id: str = Field(..., description="Unique incident identifier")
    failure_signature: str = Field(..., description="Concise error signature (e.g. 'AttributeError: User.discount')")
    log_summary: str = Field(..., description="Key lines from the failure log")
    root_cause: str = Field(..., description="Diagnosed root cause explanation")
    failing_file: str = Field(..., description="Primary file requiring the fix")
    patch_diff: str = Field(..., description="Unified diff patch that successfully resolved the issue")
    test_command: str = Field(..., description="Reproduction test command used to verify the patch")
    resolution_status: str = Field(
        default="VERIFIED_SUCCESS",
        description="Resolution status: 'VERIFIED_SUCCESS' or 'UNRESOLVED'"
    )
    tags: List[str] = Field(default_factory=list, description="Descriptive tags (e.g. ['python', 'billing', 'typo'])")
    embedding: Optional[List[float]] = Field(default=None, description="Dense embedding vector of the failure signature")


class MemorySearchResult(BaseModel):
    """Container for search results returned by the Hybrid Memory Store."""
    item: Union[CodeChunk, IncidentTrajectory]
    score: float = Field(..., description="Fused hybrid score (Reciprocal Rank Fusion)")
    dense_score: float = Field(default=0.0, description="Dense cosine similarity score")
    bm25_score: float = Field(default=0.0, description="Sparse BM25 lexical match score")
    rank_dense: int = Field(default=999, description="Rank in dense retrieval")
    rank_bm25: int = Field(default=999, description="Rank in BM25 retrieval")
