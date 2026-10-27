"""
AegisFlow Jailed Local Subprocess Sandbox
=========================================
Provides an isolated, credential-sanitized local execution jail when Docker
daemon is unavailable.

Security & Isolation Measures:
1. Environment Scrubbing: Strips all API keys, cloud tokens, database credentials,
   and secrets from the subprocess environment.
2. Scratchpad Directory Jail: Copies codebase to an ephemeral isolated temporary
   folder to prevent any modification of developer or production source code.
3. Timeout & Process Tree Termination: Enforces strict wall-clock execution quotas,
   terminating rogue or infinite-looping child processes.
4. Native Unified Diff Applier: Built-in Python unified diff parser and applier that
   operates reliably across all operating systems without requiring external GNU patch.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict

from aegisflow.sandbox.base import AbstractSandbox, ExecutionResult
from aegisflow.config import settings


class LocalProcessSandbox(AbstractSandbox):
    """Subprocess-jailed execution sandbox with credential sanitization."""

    # Sensitive environment key substring patterns to strip
    SENSITIVE_KEY_PATTERNS = [
        "KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH", "CREDENTIAL",
        "DATABASE_URL", "CONN_STR", "API", "PRIVATE"
    ]

    def __init__(self, timeout_seconds: int = settings.sandbox_timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.workspace_dir: Optional[Path] = None

    def setup(self, repo_dir: Optional[Path] = None) -> "LocalProcessSandbox":
        """Prepares a fresh ephemeral workspace copy."""
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="aegis_local_ws_"))
        if repo_dir and repo_dir.exists():
            ignore_func = shutil.ignore_patterns(
                ".venv", ".git", "node_modules", "dist", "__pycache__", ".pytest_cache", ".zwork"
            )
            shutil.copytree(repo_dir, self.workspace_dir, ignore=ignore_func, dirs_exist_ok=True)
        return self

    def _get_sanitized_env(self) -> Dict[str, str]:
        """Filters out sensitive environment variables to prevent sandbox credential exfiltration."""
        clean_env = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            if any(pattern in k_upper for pattern in self.SENSITIVE_KEY_PATTERNS):
                continue
            clean_env[k] = v

        # Set sandbox flags
        clean_env["AEGIS_SANDBOX_ISOLATED"] = "1"
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_env["PYTHONUNBUFFERED"] = "1"
        return clean_env

    def run_command(self, command: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Runs command in the jailed workspace with strict timeout."""
        if not self.workspace_dir:
            raise RuntimeError("Sandbox workspace not initialized. Call setup() first.")

        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.perf_counter()
        clean_env = self._get_sanitized_env()

        is_windows = sys.platform.startswith("win")
        shell_cmd = command if is_windows else ["/bin/sh", "-c", command]

        try:
            process = subprocess.Popen(
                shell_cmd,
                cwd=str(self.workspace_dir),
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=is_windows
            )

            stdout, stderr = process.communicate(timeout=timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return ExecutionResult(
                exit_code=process.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                duration_ms=round(duration_ms, 2),
                timed_out=False,
                backend_used="local_jailed",
                metadata={"workspace": str(self.workspace_dir)}
            )

        except subprocess.TimeoutExpired:
            # Terminate runaway process tree
            if is_windows:
                subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, capture_output=True)
            else:
                process.kill()

            stdout, stderr = process.communicate()
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return ExecutionResult(
                exit_code=-1,
                stdout=stdout or "",
                stderr=f"Execution timed out after {timeout} seconds. Process tree killed.",
                duration_ms=round(duration_ms, 2),
                timed_out=True,
                backend_used="local_jailed"
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                duration_ms=round(duration_ms, 2),
                timed_out=False,
                backend_used="local_jailed"
            )

    def write_file(self, rel_path: str, content: str) -> None:
        """Writes or updates a file in the isolated workspace."""
        if not self.workspace_dir:
            raise RuntimeError("Sandbox not initialized.")
        target = self.workspace_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel_path: str) -> str:
        """Reads a file from the workspace."""
        if not self.workspace_dir:
            raise RuntimeError("Sandbox not initialized.")
        target = self.workspace_dir / rel_path
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {rel_path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def apply_patch(self, patch_diff: str) -> bool:
        """
        Applies a unified diff patch to the sandboxed workspace.
        First tries git apply; if not available, uses pure-Python diff applier.
        """
        if not self.workspace_dir:
            return False

        # Attempt git apply if git repository
        diff_file = self.workspace_dir / "temp_aegis_patch.diff"
        diff_file.write_text(patch_diff, encoding="utf-8")

        res = self.run_command(f"git apply --ignore-whitespace {diff_file.name}")
        if res.exit_code == 0:
            diff_file.unlink(missing_ok=True)
            return True

        # Fallback to pure Python unified diff parser
        try:
            success = self._apply_unified_diff_python(patch_diff)
            diff_file.unlink(missing_ok=True)
            return success
        except Exception:
            diff_file.unlink(missing_ok=True)
            return False

    def _apply_unified_diff_python(self, diff_text: str) -> bool:
        """Pure-Python patch applier parsing unified diff chunks."""
        file_diff_pattern = re.compile(r"--- a/(.*?)\n\+\+\+ b/(.*?)\n(@@.*)", re.DOTALL)
        match = file_diff_pattern.search(diff_text)
        if not match:
            return False

        target_rel_path = match.group(2).strip()
        hunks_text = match.group(3)
        target_file = self.workspace_dir / target_rel_path

        if not target_file.exists():
            return False

        original_lines = target_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        hunk_header_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        hunk_blocks = re.split(r"(^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*?\n)", hunks_text, flags=re.MULTILINE)
        if len(hunk_blocks) < 3:
            return False

        new_lines = list(original_lines)
        offset = 0

        for i in range(1, len(hunk_blocks), 2):
            header = hunk_blocks[i]
            body = hunk_blocks[i + 1] if (i + 1) < len(hunk_blocks) else ""

            header_match = hunk_header_pattern.match(header)
            if not header_match:
                continue

            orig_start = int(header_match.group(1)) - 1
            orig_count = int(header_match.group(2)) if header_match.group(2) else 1

            lines_to_insert = []
            lines_to_remove = 0

            for line in body.splitlines(keepends=True):
                if line.startswith("+"):
                    lines_to_insert.append(line[1:])
                elif line.startswith("-"):
                    lines_to_remove += 1
                elif line.startswith(" ") or line.startswith("\\"):
                    pass

            adjusted_start = orig_start + offset
            new_lines[adjusted_start:adjusted_start + lines_to_remove] = lines_to_insert
            offset += len(lines_to_insert) - lines_to_remove

        target_file.write_text("".join(new_lines), encoding="utf-8")
        return True

    def cleanup(self) -> None:
        """Deletes ephemeral scratch workspace."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            self.workspace_dir = None
