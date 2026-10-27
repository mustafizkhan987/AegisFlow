"""
AegisFlow MCP Tool Dispatcher & Schema Registry
===============================================
Bridges the LLM reasoning agent with MCP tools:
1. Provides standard OpenAI/vLLM-compatible function tool definitions.
2. Dispatches tool calls directly with strict argument validation.
3. Formats results with telemetry metadata.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from aegisflow.mcp.tools import (
    execute_ast_search,
    execute_git_blame,
    execute_read_file,
    execute_sandbox_test,
    execute_apply_patch,
    execute_query_memory,
)
from aegisflow.memory.store import HybridMemoryStore
from aegisflow.sandbox.base import AbstractSandbox


# OpenAI / vLLM function declarations for model tool calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "ast_code_search",
            "description": "Searches repository AST definitions, functions, and classes matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symbol, function, class, or semantic phrase to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff_blame",
            "description": "Inspects git author annotations and recent diffs for a failing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of file to inspect."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_context",
            "description": "Reads line-numbered source code slice from a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of the target file."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed starting line number.",
                        "default": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed ending line number.",
                        "default": 100
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_test_in_sandbox",
            "description": "Executes a test or repro command inside the isolated ephemeral sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Test command to execute (e.g. 'pytest tests/test_billing.py')."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30).",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_candidate_patch",
            "description": "Applies a unified diff patch to the sandboxed codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_diff": {
                        "type": "string",
                        "description": "Unified diff text starting with '--- a/...' and '+++ b/...'"
                    }
                },
                "required": ["patch_diff"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_episodic_memory",
            "description": "Queries historical incident trajectories for proven root causes and verified patches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_signature": {
                        "type": "string",
                        "description": "Error trace, exception, or failure message to look up."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar incident trajectories to retrieve (default 3).",
                        "default": 3
                    }
                },
                "required": ["error_signature"]
            }
        }
    }
]


class MCPToolDispatcher:
    """Dispatches tool execution requests from the Agent Loop to the appropriate MCP tool."""

    def __init__(
        self,
        repo_dir: Path,
        memory_store: HybridMemoryStore,
        sandbox: Optional[AbstractSandbox] = None
    ):
        self.repo_dir = repo_dir
        self.memory_store = memory_store
        self.sandbox = sandbox

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard OpenAI / vLLM function tool declarations."""
        return TOOL_DEFINITIONS

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the specified tool with arguments and returns structured dictionary output."""
        try:
            if tool_name == "ast_code_search":
                return execute_ast_search(
                    query=arguments.get("query", ""),
                    repo_dir=self.repo_dir,
                    memory_store=self.memory_store,
                    top_k=arguments.get("top_k", 5)
                )

            elif tool_name == "git_diff_blame":
                return execute_git_blame(
                    file_path=arguments.get("file_path", ""),
                    repo_dir=self.repo_dir,
                    start_line=arguments.get("start_line"),
                    end_line=arguments.get("end_line")
                )

            elif tool_name == "read_file_context":
                return execute_read_file(
                    file_path=arguments.get("file_path", ""),
                    repo_dir=self.repo_dir,
                    start_line=arguments.get("start_line", 1),
                    end_line=arguments.get("end_line", 100)
                )

            elif tool_name == "run_test_in_sandbox":
                if not self.sandbox:
                    return {"passed": False, "exit_code": 1, "stderr": "No active sandbox provisioned."}
                return execute_sandbox_test(
                    command=arguments.get("command", ""),
                    sandbox=self.sandbox,
                    timeout_seconds=arguments.get("timeout_seconds", 30)
                )

            elif tool_name == "apply_candidate_patch":
                if not self.sandbox:
                    return {"applied": False, "message": "No active sandbox provisioned."}
                return execute_apply_patch(
                    patch_diff=arguments.get("patch_diff", ""),
                    sandbox=self.sandbox
                )

            elif tool_name == "query_episodic_memory":
                return execute_query_memory(
                    error_signature=arguments.get("error_signature", ""),
                    memory_store=self.memory_store,
                    top_k=arguments.get("top_k", 3)
                )

            else:
                return {"error": f"Unknown tool name: {tool_name}"}

        except Exception as e:
            return {"error": f"Exception executing tool '{tool_name}': {str(e)}"}
