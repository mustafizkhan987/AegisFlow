"""
AegisFlow AST-Aware Code Chunker
================================
Partitions source code repositories into syntactically aware chunks (functions,
classes, methods) rather than naive fixed-size token splits.

Why AST-Aware Chunking Outperforms Naive Splits:
1. Context Integrity: Functions and classes are never split halfway across logic boundaries.
2. Symbol Isolation: Enables exact identifier searches (e.g., finding the exact definition
   of `calculate_final_invoice`).
3. Breadcrumb Line Mapping: Maps exact 1-indexed line numbers for unified diff patch generation.
"""

import ast
import hashlib
from pathlib import Path
from typing import List, Optional

from aegisflow.memory.schema import CodeChunk


class ASTCodeChunker:
    """Parses code files into semantic AST chunks."""

    @staticmethod
    def _generate_chunk_id(file_path: str, start_line: int, end_line: int, name: str) -> str:
        unique_str = f"{file_path}::{name}::{start_line}-{end_line}"
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()[:16]

    def chunk_python_source(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Parses Python source using the native `ast` module into structured chunks."""
        chunks: List[CodeChunk] = []
        lines = source_code.splitlines()

        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError:
            # Fallback to naive block chunker if code has syntax errors
            return self.chunk_generic_source(file_path, source_code)

        for node in tree.body:
            # 1. Top-level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start + len(lines[start - 1:]))
                content = "\n".join(lines[start - 1:end])
                docstring = ast.get_docstring(node)
                chunks.append(CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, start, end, node.name),
                    file_path=file_path,
                    chunk_type="function",
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    content=content,
                    docstring=docstring,
                    signature=f"def {node.name}(...)"
                ))

            # 2. Classes and their internal methods
            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = getattr(node, "end_lineno", start + len(lines[start - 1:]))
                content = "\n".join(lines[start - 1:end])
                docstring = ast.get_docstring(node)

                # Add class header / whole class chunk
                chunks.append(CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, start, end, node.name),
                    file_path=file_path,
                    chunk_type="class",
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    content=content,
                    docstring=docstring,
                    signature=f"class {node.name}"
                ))

                # Also index individual member methods for granular retrieval
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = item.lineno
                        m_end = getattr(item, "end_lineno", m_start + 10)
                        m_content = "\n".join(lines[m_start - 1:m_end])
                        chunks.append(CodeChunk(
                            chunk_id=self._generate_chunk_id(file_path, m_start, m_end, f"{node.name}.{item.name}"),
                            file_path=file_path,
                            chunk_type="method",
                            name=f"{node.name}.{item.name}",
                            start_line=m_start,
                            end_line=m_end,
                            content=m_content,
                            docstring=ast.get_docstring(item),
                            signature=f"def {node.name}.{item.name}(...)"
                        ))

        # If no AST functions/classes were extracted (e.g. flat script/config file)
        if not chunks:
            chunks = self.chunk_generic_source(file_path, source_code)

        return chunks

    def chunk_generic_source(
        self,
        file_path: str,
        source_code: str,
        max_lines_per_chunk: int = 50,
        overlap: int = 10
    ) -> List[CodeChunk]:
        """Sliding-window semantic block chunker for non-Python or unparseable files."""
        chunks: List[CodeChunk] = []
        lines = source_code.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return chunks

        step = max(1, max_lines_per_chunk - overlap)
        for i in range(0, total_lines, step):
            start = i + 1
            end = min(total_lines, i + max_lines_per_chunk)
            chunk_content = "\n".join(lines[start - 1:end])
            name = f"{Path(file_path).stem}_block_{start}_{end}"
            chunks.append(CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, start, end, name),
                file_path=file_path,
                chunk_type="block",
                name=name,
                start_line=start,
                end_line=end,
                content=chunk_content,
                docstring=None,
                signature=None
            ))
            if end >= total_lines:
                break

        return chunks

    def chunk_file(self, file_path: Path, repo_root: Optional[Path] = None) -> List[CodeChunk]:
        """Reads and chunks a physical file on disk."""
        if not file_path.is_file():
            return []

        rel_path = str(file_path.relative_to(repo_root)) if repo_root else str(file_path)
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if file_path.suffix == ".py":
            return self.chunk_python_source(rel_path, content)
        else:
            return self.chunk_generic_source(rel_path, content)
