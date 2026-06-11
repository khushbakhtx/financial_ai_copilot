from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Protocol

_AGENT_ROOT = Path(__file__).parent.parent          # ai_agent/
_SCRIPTS_ROOT = _AGENT_ROOT / ".deepagents" / "skills"


def resolve_script(skill_script: str) -> Path:
    """
    Resolve a short script name like "data-analysis/discover"
    to an absolute path:  .deepagents/skills/data-analysis/scripts/discover.py
    """
    parts = skill_script.split("/", 1)
    if len(parts) == 2:
        skill, name = parts
    else:
        raise ValueError(
            f"script_name must be '<skill>/<script>', got: {skill_script!r}"
        )
    return _SCRIPTS_ROOT / skill / "scripts" / f"{name}.py"


class SandboxBackend(Protocol):
    def run(self, script_path: Path, file_url: str, extra_args: str = "") -> str:
        ...

    def run_and_read(
        self, script_path: Path, file_url: str, output_path: str, extra_args: str = ""
    ) -> tuple[str, bytes | None, str | None]:
        """Run script then read output_path. Returns (stdout, file_bytes, error)."""
        ...


class LocalSandbox:
    """
    Runs scripts on the local machine.
    Downloads the file from file_url into a temp directory, then calls:
        python <script_path> --file <local_path> [extra_args]
    """

    def run(self, script_path: Path, file_url: str, extra_args: str = "") -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "data.csv")

            try:
                urllib.request.urlretrieve(file_url, local_path)
            except Exception as exc:
                return json.dumps({"error": f"Failed to download file: {exc}"})

            cmd = [sys.executable, str(script_path), "--file", local_path]
            if extra_args.strip():
                cmd += extra_args.strip().split()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                return json.dumps({"error": "Script timed out after 120 seconds."})
            except Exception as exc:
                return json.dumps({"error": f"Failed to run script: {exc}"})

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr[:2000]}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            if len(output) > 60_000:
                output = output[:60_000] + "\n[output truncated]"
            return output or "[no output]"

    def run_and_read(
        self, script_path: Path, file_url: str, output_path: str, extra_args: str = ""
    ) -> tuple[str, bytes | None, str | None]:
        """Run script and read the output file it saves. File persists on local machine."""
        stdout = self.run(script_path, file_url, extra_args)
        try:
            with open(output_path, "rb") as f:
                return stdout, f.read(), None
        except Exception as exc:
            return stdout, None, f"Output file not found at {output_path!r}: {exc}"


class E2BSandbox:
    """
    E2B cloud sandbox for production deployments.

    Install: pip install e2b-code-interpreter
    Set env:  E2B_API_KEY=<your-key>

    Uses the E2B v1 SDK: Sandbox.create() + sbx.run_code().
    """

    _DEPS = "pandas numpy scipy scikit-learn pyarrow"

    def run(self, script_path: Path, file_url: str, extra_args: str = "") -> str:
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
        except ImportError:
            return json.dumps({
                "error": "e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter"
            })

        if not os.environ.get("E2B_API_KEY"):
            return json.dumps({"error": "E2B_API_KEY env var not set."})

        with Sandbox.create() as sbx:
            sbx.commands.run(f"pip install -q {self._DEPS}")

            sbx.files.write("/home/user/script.py", script_path.read_text())

            sbx.run_code(
                f"import urllib.request; "
                f"urllib.request.urlretrieve({file_url!r}, '/home/user/data.csv')"
            )

            cmd = "python /home/user/script.py --file /home/user/data.csv"
            if extra_args.strip():
                cmd += " " + extra_args.strip()

            try:
                from e2b.sandbox.commands.command_handle import CommandExitException  # type: ignore
            except ImportError:
                CommandExitException = Exception  # type: ignore

            try:
                result = sbx.commands.run(cmd)
                output = result.stdout or ""
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr[:2000]}"
            except CommandExitException as exc:
                output = json.dumps({
                    "error": f"Script exited with code {exc.exit_code}",
                    "stderr": exc.stderr[:2000] if exc.stderr else "",
                    "stdout": exc.stdout[:2000] if exc.stdout else "",
                })
            return output[:60_000] or "[no output]"

    def run_and_read(
        self, script_path: Path, file_url: str, output_path: str, extra_args: str = ""
    ) -> tuple[str, bytes | None, str | None]:
        """Run script and read the output file before the sandbox closes."""
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
        except ImportError:
            return "", None, "e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter"

        if not os.environ.get("E2B_API_KEY"):
            return "", None, "E2B_API_KEY env var not set."

        try:
            from e2b.sandbox.commands.command_handle import CommandExitException  # type: ignore
        except ImportError:
            CommandExitException = Exception  # type: ignore

        with Sandbox.create() as sbx:
            sbx.commands.run(f"pip install -q {self._DEPS}")
            sbx.files.write("/home/user/script.py", script_path.read_text())
            sbx.run_code(
                f"import urllib.request; "
                f"urllib.request.urlretrieve({file_url!r}, '/home/user/data.csv')"
            )

            cmd = "python /home/user/script.py --file /home/user/data.csv"
            if extra_args.strip():
                cmd += " " + extra_args.strip()

            try:
                result = sbx.commands.run(cmd)
                output = result.stdout or ""
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr[:2000]}"
            except CommandExitException as exc:
                return "", None, f"Script exited with code {exc.exit_code}: {exc.stderr[:500] if exc.stderr else ''}"

            try:
                raw = sbx.files.read(output_path)
                file_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
            except Exception as exc:
                return output, None, f"Output file not found in sandbox at {output_path!r}: {exc}"

            return output[:60_000], file_bytes, None


def get_sandbox() -> SandboxBackend:
    """Return the appropriate sandbox based on SANDBOX env var."""
    mode = os.environ.get("SANDBOX", "local").lower()
    if mode == "e2b":
        return E2BSandbox()
    return LocalSandbox()
