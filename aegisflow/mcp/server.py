"""
AegisFlow MCP Server
====================
Standalone Model Context Protocol server exposing strictly typed tools for
repository AST inspection, git diffing, sandboxed test execution, and episodic
memory queries. Compatible with any MCP client (vLLM, Claude, Cursor, Goose).

Usage:
    python -m aegisflow.mcp.server
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.mcpserver import MCPServer

from aegisflow.mcp.tools import (
    execute_ast_search,
    execute_git_blame,
    execute_read_file,
    execute_sandbox_test,
    execute_apply_patch,
    execute_query_memory,
)
from aegisflow.memory.store import HybridMemoryStore
from aegisflow.sandbox.manager import SandboxManager

# Initialize the MCP Server
mcp_server = MCPServer("AegisFlow-MCP-Server")

# Global instances for server runtime
_memory_store = HybridMemoryStore()
_active_sandbox = None
_default_repo_dir = Path.cwd()


@mcp_server.tool()
def ast_code_search(query: str, top_k: int = 5) -> str:
    """
    Search repository definitions, functions, and classes using AST analysis.

    Args:
        query: Symbol, function name, class, or semantic query to locate.
        top_k: Maximum number of AST code chunks to return (default 5).
    """
    res = execute_ast_search(query, repo_dir=_default_repo_dir, memory_store=_memory_store, top_k=top_k)
    return str(res["results"])


@mcp_server.tool()
def git_diff_blame(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Inspect git author blame lines and recent diffs for a failing source file.

    Args:
        file_path: Relative path to the file.
        start_line: Optional starting line number for blame slice.
        end_line: Optional ending line number for blame slice.
    """
    res = execute_git_blame(file_path, repo_dir=_default_repo_dir, start_line=start_line, end_line=end_line)
    return f"Blame:\n" + "\n".join(res["blame_lines"]) + f"\n\nRecent Diff:\n{res['recent_diff']}"


@mcp_server.tool()
def read_file_context(file_path: str, start_line: int = 1, end_line: int = 100) -> str:
    """
    Read a slice of source code with 1-indexed line numbers.

    Args:
        file_path: Relative path to the file.
        start_line: 1-indexed starting line number.
        end_line: 1-indexed ending line number.
    """
    res = execute_read_file(file_path, repo_dir=_default_repo_dir, start_line=start_line, end_line=end_line)
    return res["content_slice"]


@mcp_server.tool()
def run_test_in_sandbox(command: str, timeout_seconds: int = 30) -> str:
    """
    Execute a test command inside the secure ephemeral sandbox and return stdout/stderr.

    Args:
        command: Test command to execute (e.g. 'pytest tests/test_billing.py').
        timeout_seconds: Wall-clock timeout limit in seconds.
    """
    global _active_sandbox
    if _active_sandbox is None:
        _active_sandbox = SandboxManager.create_sandbox(repo_dir=_default_repo_dir)

    res = execute_sandbox_test(command, sandbox=_active_sandbox, timeout_seconds=timeout_seconds)
    status = "PASSED" if res["passed"] else "FAILED"
    return f"Status: {status} (Exit Code: {res['exit_code']}, Time: {res['duration_ms']}ms)\nStdout:\n{res['stdout']}\nStderr:\n{res['stderr']}"


@mcp_server.tool()
def apply_candidate_patch(patch_diff: str) -> str:
    """
    Apply a unified diff patch to the sandboxed codebase before running verification tests.

    Args:
        patch_diff: Complete unified diff text (with --- a/... and +++ b/...).
    """
    global _active_sandbox
    if _active_sandbox is None:
        _active_sandbox = SandboxManager.create_sandbox(repo_dir=_default_repo_dir)

    res = execute_apply_patch(patch_diff, sandbox=_active_sandbox)
    return res["message"]


@mcp_server.tool()
def query_episodic_memory(error_signature: str, top_k: int = 3) -> str:
    """
    Query historical incident trajectories to retrieve proven root causes and verified patches.

    Args:
        error_signature: The error message or exception trace to match.
        top_k: Number of historical incident records to retrieve.
    """
    res = execute_query_memory(error_signature, memory_store=_memory_store, top_k=top_k)
    return str(res["trajectories"])


if __name__ == "__main__":
    print("Starting AegisFlow MCP Server via standard stdio transport...")
    mcp_server.run()
