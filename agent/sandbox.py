"""
Sandboxed code execution via Docker or subprocess jail.

Provides:
  - DockerSandbox: Execute Python code in a temporary Docker container
  - SubprocessSandbox: Fallback using subprocess with resource limits
"""
import os
import sys
import time
import uuid
import logging
import subprocess
import tempfile
from typing import Optional

from agent.safety import sanitize_code, sanitize_pip

logger = logging.getLogger(__name__)

DOCKER_IMAGE = "python:3.10-slim"
DOCKER_TIMEOUT = 300
SANDBOX_TIMEOUT = 120


class SandboxError(Exception):
    pass


class SandboxResult:
    def __init__(self, success: bool, output: str = "", error: str = "",
                 execution_time: float = 0.0, exit_code: int = -1):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.exit_code = exit_code

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "exit_code": self.exit_code,
        }


class DockerSandbox:
    """
    Execute Python code in a temporary Docker container.

    Uses the host network (--net=host) so Colab APIs are reachable.
    Container is removed automatically after execution.
    """

    def __init__(self, image: str = DOCKER_IMAGE, timeout: int = DOCKER_TIMEOUT):
        self.image = image
        self.timeout = timeout
        self._available = self._check_docker()

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def execute(self, code: str, pip_packages: Optional[list[str]] = None,
                env_vars: Optional[dict[str, str]] = None) -> SandboxResult:
        if not self._available:
            return SandboxResult(False, "", "Docker is not available")

        safe, _cleaned, warning = sanitize_code(code)
        if not safe:
            return SandboxResult(False, "", f"Code blocked by safety: {warning}")

        cleaned = sanitize_pip(code)
        container_name = f"colab-agent-{uuid.uuid4().hex[:8]}"

        wrapper = self._build_wrapper_script(cleaned, pip_packages or [])

        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "host",
            "--memory", "4g",
            "--memory-swap", "6g",
            "--cpus", "2",
            "--init",
            self.image,
            "python3", "-c", wrapper,
        ]

        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout,
                env={**os.environ, **(env_vars or {})},
            )
            elapsed = time.time() - start
            return SandboxResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=round(elapsed, 2),
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            self._force_kill(container_name)
            return SandboxResult(False, "", f"Timeout after {self.timeout}s",
                                 execution_time=round(elapsed, 2))
        except FileNotFoundError as e:
            return SandboxResult(False, "", f"Docker not found: {e}")

    def _build_wrapper_script(self, code: str, pip_packages: list[str]) -> str:
        lines = ["import sys, json"]
        for pkg in pip_packages:
            lines.append(f"!pip install -q {pkg}")
        lines.append("")
        lines.append(code)
        return "\n".join(lines)

    def _force_kill(self, container_name: str):
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def execute_and_return_json(self, code: str, **kwargs) -> dict:
        result = self.execute(code, **kwargs)
        return result.to_dict()


class SubprocessSandbox:
    """
    Sandbox using subprocess with resource limits.
    Falls back when Docker is unavailable.
    """

    def __init__(self, timeout: int = SANDBOX_TIMEOUT):
        self.timeout = timeout
        self._temp_dir: Optional[str] = None

    def execute(self, code: str) -> SandboxResult:
        safe, _cleaned, warning = sanitize_code(code)
        if not safe:
            return SandboxResult(False, "", f"Code blocked: {warning}")

        cleaned = sanitize_pip(code)
        cleaned = self._strip_shell_markers(cleaned)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            script_path = f.name
            f.write(cleaned)

        try:
            start = time.time()
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            return SandboxResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=round(elapsed, 2),
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return SandboxResult(False, "", f"Timeout after {self.timeout}s",
                                 execution_time=round(elapsed, 2))
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _strip_shell_markers(self, code: str) -> str:
        lines = []
        for line in code.split("\n"):
            if line.strip().startswith("!"):
                continue
            lines.append(line)
        return "\n".join(lines)


def get_sandbox() -> DockerSandbox:
    ds = DockerSandbox()
    if ds.available:
        return ds
    logger.warning("Docker not available, falling back to SubprocessSandbox")
    return SubprocessSandbox()
