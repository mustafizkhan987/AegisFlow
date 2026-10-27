"""
AegisFlow Docker Container Sandbox
==================================
Orchestrates disposable, network-restricted execution containers using the
Docker SDK for Python to run untrusted agent-generated code patches and unit
tests in complete isolation from the host machine.

Security Policies Enforced:
1. Network Mode: 'none' (blocks all inbound/outbound socket traffic during test execution).
2. Memory Quota: '512m' hard memory ceiling to prevent host OOM.
3. CPU Quota: Strict nano_cpus allocation (default 1.0 vCPU).
4. Capability Dropping: Drops all Linux root capabilities ('ALL').
5. No New Privileges: Prevents privilege escalation.
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional
import docker
from docker.errors import DockerException

from aegisflow.sandbox.base import AbstractSandbox, ExecutionResult
from aegisflow.config import settings


class DockerSandbox(AbstractSandbox):
    """Ephemeral Docker container execution sandbox."""

    def __init__(
        self,
        image: str = settings.sandbox_docker_image,
        memory_limit: str = settings.sandbox_memory_limit,
        cpu_limit: float = settings.sandbox_cpu_limit,
        timeout_seconds: int = settings.sandbox_timeout_seconds
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout_seconds = timeout_seconds

        self.client: Optional[docker.DockerClient] = None
        self.container = None
        self.host_workspace_dir: Optional[Path] = None

    def setup(self, repo_dir: Optional[Path] = None) -> "DockerSandbox":
        """Initializes client, creates temporary workspace copy, and spins up container."""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as e:
            raise RuntimeError(f"Docker daemon is not accessible: {e}")

        # Create isolated workspace copy on host
        self.host_workspace_dir = Path(tempfile.mkdtemp(prefix="aegis_docker_ws_"))
        if repo_dir and repo_dir.exists():
            ignore_func = shutil.ignore_patterns(
                ".venv", ".git", "node_modules", "dist", "__pycache__", ".pytest_cache", ".zwork"
            )
            shutil.copytree(repo_dir, self.host_workspace_dir, ignore=ignore_func, dirs_exist_ok=True)

        nano_cpus = int(self.cpu_limit * 1_000_000_000)

        # Launch locked-down disposable container
        self.container = self.client.containers.run(
            image=self.image,
            command="tail -f /dev/null",  # Keeps container alive for exec_run
            detach=True,
            volumes={
                str(self.host_workspace_dir): {"bind": "/workspace", "mode": "rw"}
            },
            working_dir="/workspace",
            network_mode="none",  # Complete network isolation
            mem_limit=self.memory_limit,
            nano_cpus=nano_cpus,
            security_opt=["no-new-privileges:true"],
            cap_drop=["ALL"]
        )
        return self

    def run_command(self, command: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Executes a command inside the container and measures wall-clock runtime."""
        if not self.container:
            raise RuntimeError("Sandbox is not running. Call setup() first.")

        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.perf_counter()

        try:
            # Execute with bash or sh inside container
            exec_res = self.container.exec_run(
                cmd=["/bin/sh", "-c", command],
                workdir="/workspace",
                demux=True
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            raw_stdout, raw_stderr = exec_res.output
            stdout_str = raw_stdout.decode("utf-8", errors="replace") if raw_stdout else ""
            stderr_str = raw_stderr.decode("utf-8", errors="replace") if raw_stderr else ""

            return ExecutionResult(
                exit_code=exec_res.exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_ms=round(duration_ms, 2),
                timed_out=False,
                backend_used="docker",
                metadata={"container_id": self.container.id[:12]}
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Docker exec error: {str(e)}",
                duration_ms=round(duration_ms, 2),
                timed_out=False,
                backend_used="docker"
            )

    def write_file(self, rel_path: str, content: str) -> None:
        """Writes a file to the host volume mounted into the container."""
        if not self.host_workspace_dir:
            raise RuntimeError("Sandbox not initialized.")
        target = self.host_workspace_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel_path: str) -> str:
        """Reads a file from the host volume."""
        if not self.host_workspace_dir:
            raise RuntimeError("Sandbox not initialized.")
        target = self.host_workspace_dir / rel_path
        if not target.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def apply_patch(self, patch_diff: str) -> bool:
        """Applies unified diff patch to the sandbox workspace."""
        patch_path = "aegis_candidate_patch.diff"
        self.write_file(patch_path, patch_diff)
        res = self.run_command(f"patch -p1 < {patch_path}")
        return res.exit_code == 0

    def cleanup(self) -> None:
        """Kills and removes container and host scratch directory."""
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

        if self.host_workspace_dir and self.host_workspace_dir.exists():
            try:
                shutil.rmtree(self.host_workspace_dir, ignore_errors=True)
            except Exception:
                pass
            self.host_workspace_dir = None
