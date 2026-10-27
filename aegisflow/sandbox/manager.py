"""
AegisFlow Sandbox Manager
=========================
Intelligently creates and provisions sandboxes based on host environment
capabilities, automatically prioritizing Docker SDK container sandboxing
when Docker daemon is responsive and falling back seamlessly to credential-sanitized
subprocess jails.
"""

from pathlib import Path
from typing import Optional
import docker
from docker.errors import DockerException

from aegisflow.sandbox.base import AbstractSandbox
from aegisflow.sandbox.docker_sandbox import DockerSandbox
from aegisflow.sandbox.local_sandbox import LocalProcessSandbox
from aegisflow.config import settings


class SandboxManager:
    """Manages creation, health-checks, and lifecycle of execution sandboxes."""

    @staticmethod
    def is_docker_available() -> bool:
        """Checks if Docker daemon is running and responsive."""
        try:
            client = docker.from_env()
            return bool(client.ping())
        except (DockerException, Exception):
            return False

    @classmethod
    def create_sandbox(
        cls,
        repo_dir: Optional[Path] = None,
        force_backend: Optional[str] = None
    ) -> AbstractSandbox:
        """
        Factory method instantiating the optimal sandbox backend.

        Args:
            repo_dir: Path to source repository to mount/copy into the sandbox.
            force_backend: Override backend: 'docker', 'local', or None (auto-detect).

        Returns:
            Initialized and ready-to-run AbstractSandbox instance.
        """
        backend = force_backend or settings.sandbox_backend

        if backend == "docker":
            sandbox = DockerSandbox()
        elif backend == "local":
            sandbox = LocalProcessSandbox()
        else:
            # Auto-detect mode
            if cls.is_docker_available():
                sandbox = DockerSandbox()
            else:
                sandbox = LocalProcessSandbox()

        sandbox.setup(repo_dir=repo_dir)
        return sandbox
