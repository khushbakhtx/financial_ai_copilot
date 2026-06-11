"""E2B Sandbox integration for isolated financial ML code execution.

Persistent sandbox strategy:
  - One E2B sandbox per LangGraph thread, stored in _SANDBOX_REGISTRY keyed by thread_id.
  - Packages are installed once on first use; subsequent calls reconnect via Sandbox.connect().
  - Between calls the sandbox is kept alive (timeout extended); it is paused only when
    explicitly torn down or if E2B auto-pauses on timeout (on_timeout="pause").
  - Datasets are uploaded once per sandbox lifetime; a set tracks what's already there.
  - Falls back to local subprocess when E2B_API_KEY is not set.
"""

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FINANCIAL_PACKAGES = (
    "pandas numpy scikit-learn xgboost lightgbm catboost "
    "optuna shap matplotlib seaborn networkx pyod polars plotly"
)

_PYTHON = sys.executable
_MAX_OUTPUT = 60_000
_TERMINAL_URL = os.getenv("TERMINAL_SERVER_URL", "http://localhost:8001")

# Sandbox keep-alive: how long (seconds) the sandbox stays alive between calls.
# E2B Pro supports up to 24h; Base up to 1h. We use 1800s (30 min) as a safe default.
_SANDBOX_TIMEOUT = int(os.getenv("E2B_SANDBOX_TIMEOUT", "1800"))


# ─── Terminal emit ─────────────────────────────────────────────────────────────

def _emit(kind: str, text: str, agent: str = "") -> None:
    """Best-effort emit to terminal_server via HTTP — works across processes."""
    try:
        import urllib.request, json as _json
        payload = _json.dumps({"kind": kind, "text": text, "agent": agent}).encode()
        req = urllib.request.Request(
            f"{_TERMINAL_URL}/terminal/emit",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass


# ─── Persistent sandbox registry ───────────────────────────────────────────────

class _SandboxEntry:
    """Tracks a live E2B sandbox plus metadata for the current thread."""

    def __init__(self, sandbox_id: str, packages_installed: bool):
        self.sandbox_id = sandbox_id
        self.packages_installed = packages_installed
        self.uploaded_datasets: set[str] = set()  # filenames already in sandbox


# thread_id (str) → _SandboxEntry
_SANDBOX_REGISTRY: dict[str, _SandboxEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def _get_thread_id() -> Optional[str]:
    """Extract the LangGraph thread_id from the current execution context."""
    try:
        from langchain_core.runnables import RunnableConfig
        import inspect
        frame = inspect.currentframe()
        for _ in range(20):
            if frame is None:
                break
            locs = frame.f_locals
            if "config" in locs and isinstance(locs["config"], dict):
                cfg = locs["config"]
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
            frame = frame.f_back
    except Exception:
        pass
    return None


def _get_or_create_sandbox(agent: str = "") -> tuple["object", _SandboxEntry]:
    """Return a connected E2B sandbox for the current thread, creating if needed."""
    from e2b_code_interpreter import Sandbox  # type: ignore

    thread_id = _get_thread_id() or "__default__"

    with _REGISTRY_LOCK:
        entry = _SANDBOX_REGISTRY.get(thread_id)

    if entry is not None:
        # Try to reconnect to the existing sandbox
        try:
            _emit("system", f"[E2B] Reconnecting to sandbox {entry.sandbox_id[:8]}...", agent)
            sbx = Sandbox.connect(entry.sandbox_id, timeout=_SANDBOX_TIMEOUT)
            _emit("system", f"[E2B] Sandbox ready.", agent)
            return sbx, entry
        except Exception as e:
            _emit("system", f"[E2B] Sandbox expired, creating new one... ({e})", agent)
            with _REGISTRY_LOCK:
                _SANDBOX_REGISTRY.pop(thread_id, None)
            entry = None

    # Create a fresh sandbox — packages not yet installed
    _emit("system", f"[E2B] Creating new sandbox...", agent)
    sbx = Sandbox.create(timeout=_SANDBOX_TIMEOUT)
    entry = _SandboxEntry(sbx.sandbox_id, packages_installed=False)
    with _REGISTRY_LOCK:
        _SANDBOX_REGISTRY[thread_id] = entry

    _emit("system", f"[E2B] Sandbox created (id={sbx.sandbox_id[:8]}). Installing packages...", agent)
    sbx.commands.run(f"pip install -q {FINANCIAL_PACKAGES}", timeout=180)
    entry.packages_installed = True
    _emit("system", f"[E2B] Packages installed and cached for this thread.", agent)

    return sbx, entry


def release_sandbox(thread_id: Optional[str] = None) -> None:
    """Pause and remove the sandbox for a thread (call at end of investigation)."""
    tid = thread_id or _get_thread_id() or "__default__"
    with _REGISTRY_LOCK:
        entry = _SANDBOX_REGISTRY.pop(tid, None)
    if entry is None:
        return
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore
        sbx = Sandbox.connect(entry.sandbox_id)
        sbx.pause()
        _emit("system", f"[E2B] Sandbox {entry.sandbox_id[:8]} paused.", "")
    except Exception:
        pass


# ─── Dataset upload (once per sandbox lifetime) ────────────────────────────────

def _upload_datasets_to_e2b(sbx, code: str, entry: _SandboxEntry, agent: str = "") -> None:
    """Upload dataset files referenced in code, skipping already-uploaded ones."""
    import re
    data_dir = Path(os.getenv("AGENT_WORKDIR", "/tmp/financial_ai")) / "data"
    if not data_dir.exists():
        return

    candidates = re.findall(r'["\']([^"\']+\.(?:csv|parquet|json|xlsx))["\']', code)
    for candidate in candidates:
        local = Path(candidate)
        if not local.exists():
            local = data_dir / local.name
        if not local.exists():
            continue
        if local.name in entry.uploaded_datasets:
            continue  # already in sandbox from a prior call
        try:
            sbx.files.write(str(candidate), local.read_bytes())
            entry.uploaded_datasets.add(local.name)
            _emit("system", f"[E2B] Uploaded dataset: {local.name}", agent)
        except Exception as e:
            _emit("stderr", f"[E2B] Failed to upload {local.name}: {e}", agent)


# ─── E2B execution ─────────────────────────────────────────────────────────────

def _run_e2b(code: str, timeout: int, agent: str = "") -> "SandboxResult":
    sbx, entry = _get_or_create_sandbox(agent)
    start = time.time()

    try:
        # Upload any new datasets referenced in this code block
        _upload_datasets_to_e2b(sbx, code, entry, agent)

        _emit("system", f"[E2B] Running analysis...", agent)

        try:
            from e2b.sandbox.commands.command_handle import CommandExitException  # type: ignore
        except ImportError:
            CommandExitException = Exception  # type: ignore

        sbx.files.write("/home/user/analysis.py", code)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            result = sbx.commands.run(
                "python3 /home/user/analysis.py",
                timeout=timeout,
                on_stdout=lambda line: (
                    stdout_lines.append(line),
                    _emit("stdout", line, agent),
                ),
                on_stderr=lambda line: (
                    stderr_lines.append(line),
                    _emit("stderr", line, agent),
                ),
            )
            stdout = "\n".join(stdout_lines) or result.stdout or ""
            stderr = "\n".join(stderr_lines) or result.stderr or ""
            exit_code = 0
        except CommandExitException as exc:
            stdout = "\n".join(stdout_lines)
            stderr = "\n".join(stderr_lines) or str(exc)
            exit_code = exc.exit_code if hasattr(exc, "exit_code") else 1

        artifacts = []
        try:
            files = sbx.files.list("/home/user/")
            artifacts = [
                f.name for f in files
                if f.name.endswith((".png", ".csv", ".json", ".html", ".md"))
                and f.name != "analysis.py"
            ]
        except Exception:
            pass

        elapsed_ms = int((time.time() - start) * 1000)
        status = "✓ Done" if exit_code == 0 else "✗ Failed"
        _emit("system", f"[E2B] {status} in {elapsed_ms}ms", agent)

        return SandboxResult(
            stdout[:_MAX_OUTPUT],
            stderr[:2000],
            exit_code,
            artifacts,
            elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        _emit("stderr", f"[E2B] Error: {e}", agent)
        # Invalidate registry entry so next call gets a fresh sandbox
        thread_id = _get_thread_id() or "__default__"
        with _REGISTRY_LOCK:
            _SANDBOX_REGISTRY.pop(thread_id, None)
        return SandboxResult("", str(e), 1, [], elapsed_ms)


# ─── Local fallback ────────────────────────────────────────────────────────────

def _run_local(code: str, timeout: int, agent: str = "") -> "SandboxResult":
    """Run code locally via Popen, streaming each line to the terminal server."""
    working_dir = "/tmp/financial_ai"
    os.makedirs(working_dir, exist_ok=True)

    _emit("system", f"[local] Running analysis...", agent)
    start = time.time()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            [_PYTHON, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
        )

        def _read_stdout():
            for line in proc.stdout:  # type: ignore
                line = line.rstrip("\n")
                stdout_lines.append(line)
                _emit("stdout", line, agent)

        def _read_stderr():
            for line in proc.stderr:  # type: ignore
                line = line.rstrip("\n")
                stderr_lines.append(line)
                _emit("stderr", line, agent)

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            elapsed_ms = int((time.time() - start) * 1000)
            _emit("stderr", f"[local] Timed out after {timeout}s", agent)
            return SandboxResult("", f"Execution timed out after {timeout}s", 1, [], elapsed_ms)

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        elapsed_ms = int((time.time() - start) * 1000)
        exit_code = proc.returncode

        artifacts = []
        try:
            artifacts = [
                fn for fn in os.listdir(working_dir)
                if fn.endswith((".png", ".csv", ".json", ".html", ".md"))
            ]
        except Exception:
            pass

        status = "✓ Done" if exit_code == 0 else "✗ Failed"
        _emit("system", f"[local] {status} in {elapsed_ms}ms (exit={exit_code})", agent)

        return SandboxResult(
            "\n".join(stdout_lines)[:_MAX_OUTPUT],
            "\n".join(stderr_lines)[:2000],
            exit_code,
            artifacts,
            elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        _emit("stderr", f"[local] Error: {e}", agent)
        return SandboxResult("", str(e), 1, [], elapsed_ms)
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


# ─── Public interface ──────────────────────────────────────────────────────────

class SandboxResult:
    def __init__(self, stdout: str, stderr: str, exit_code: int, artifacts: list[str], execution_time_ms: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.artifacts = artifacts
        self.execution_time_ms = execution_time_ms
        self.success = exit_code == 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:5000],
            "stderr": self.stderr[:2000] if self.stderr else "",
            "artifacts": self.artifacts,
            "execution_time_ms": self.execution_time_ms,
        }


def run_financial_analysis(code: str, timeout: int = 300, agent: str = "") -> dict:
    """Run Python code in a persistent E2B sandbox (or local fallback).

    The sandbox is reused across calls within the same LangGraph thread —
    packages install once, datasets upload once, filesystem state persists.
    """
    use_e2b = bool(os.getenv("E2B_API_KEY"))
    _emit("start", f"$ python analysis.py", agent)

    if use_e2b:
        try:
            result = _run_e2b(code, timeout, agent)
        except Exception as e:
            logger.warning("E2B failed, falling back to local: %s", e)
            result = _run_local(code, timeout, agent)
    else:
        result = _run_local(code, timeout, agent)

    _emit("end", f"─── execution complete ───", agent)
    return result.to_dict()


class FinancialSandbox:
    """Thin shim — use run_financial_analysis() directly."""

    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def run_code(self, code: str, timeout: int | None = None) -> SandboxResult:
        result_dict = run_financial_analysis(code, timeout or self.timeout_seconds)
        return SandboxResult(
            result_dict["stdout"],
            result_dict["stderr"],
            result_dict["exit_code"],
            result_dict["artifacts"],
            result_dict["execution_time_ms"],
        )
