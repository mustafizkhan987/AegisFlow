"""
AegisFlow Sandbox Package
=========================
Exposes the AbstractSandbox base class, ExecutionResult, DockerSandbox,
LocalProcessSandbox, and the auto-detecting SandboxManager.
"""

from aegisflow.sandbox.base import AbstractSandbox, ExecutionResult
from aegisflow.sandbox.docker_sandbox import DockerSandbox
from aegisflow.sandbox.local_sandbox import LocalProcessSandbox
from aegisflow.sandbox.manager import SandboxManager

__all__ = [
    "AbstractSandbox",
    "ExecutionResult",
    "DockerSandbox",
    "LocalProcessSandbox",
    "SandboxManager"
]
