"""
AegisFlow Sandbox Base Architecture
===================================
Defines the abstract interface and execution result schemas for ephemeral,
isolated test verification sandboxes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Encapsulates the execution output of a sandboxed test command."""
    exit_code: int = Field(..., description="Process exit code (0 indicates test pass)")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error")
    duration_ms: float = Field(default=0.0, description="Total execution duration in milliseconds")
    timed_out: bool = Field(default=False, description="Whether execution was terminated due to timeout")
    memory_peak_mb: Optional[float] = Field(default=None, description="Peak memory consumption during run")
    backend_used: str = Field(..., description="Sandbox backend used: 'docker' or 'local_jailed'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional environment telemetry")

    @property
    def passed(self) -> bool:
        """Returns True if the test execution succeeded with code 0 and no timeout."""
        return self.exit_code == 0 and not self.timed_out


class AbstractSandbox(ABC):
    """Abstract base class for isolated sandbox environments."""

    @abstractmethod
    def setup(self, repo_dir: Optional[Path] = None) -> "AbstractSandbox":
        """Prepares the isolated sandbox workspace and environment."""
        pass

    @abstractmethod
    def run_command(self, command: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Executes a command within the sandboxed jail and captures stdout/stderr."""
        pass

    @abstractmethod
    def write_file(self, rel_path: str, content: str) -> None:
        """Writes or updates a file within the isolated sandbox."""
        pass

    @abstractmethod
    def read_file(self, rel_path: str) -> str:
        """Reads content of a file from within the sandbox."""
        pass

    @abstractmethod
    def apply_patch(self, patch_diff: str) -> bool:
        """Applies a unified diff patch to the sandboxed codebase."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Tears down containers, scratchpads, and frees system resources."""
        pass

    def __enter__(self):
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
