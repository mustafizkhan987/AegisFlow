"""
Unit Tests for Execution Sandbox
================================
Verifies isolated subprocess execution, environment variable scrubbing,
and patch application.
"""

import os
from aegisflow.sandbox.local_sandbox import LocalProcessSandbox
from aegisflow.sandbox.manager import SandboxManager


def test_local_sandbox_execution():
    with LocalProcessSandbox() as sandbox:
        # Write test file
        sandbox.write_file("test_script.py", "print('AegisFlow Sandbox Active')")
        read_back = sandbox.read_file("test_script.py")
        assert "AegisFlow Sandbox Active" in read_back

        # Execute
        res = sandbox.run_command("python test_script.py")
        assert res.exit_code == 0
        assert "AegisFlow Sandbox Active" in res.stdout
        assert res.backend_used == "local_jailed"


def test_environment_sanitization():
    # Set simulated sensitive environment variable
    os.environ["SECRET_AWS_ACCESS_KEY"] = "fake_secret_key_12345"
    os.environ["MY_SERVICE_TOKEN"] = "fake_token_abcdef"

    sandbox = LocalProcessSandbox()
    clean_env = sandbox._get_sanitized_env()

    assert "SECRET_AWS_ACCESS_KEY" not in clean_env
    assert "MY_SERVICE_TOKEN" not in clean_env
    assert clean_env.get("AEGIS_SANDBOX_ISOLATED") == "1"


def test_sandbox_manager():
    sandbox = SandboxManager.create_sandbox()
    assert sandbox is not None
    res = sandbox.run_command("python -c \"print('manager ok')\"")
    assert res.exit_code == 0
    assert "manager ok" in res.stdout
    sandbox.cleanup()
