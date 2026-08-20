# file: agentmesh/tools/run_python.py
"""Sandboxed Python code execution tool."""

import logging
import subprocess
import tempfile
from pathlib import Path

from agentmesh.config import settings
from agentmesh.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="run_python",
    description="Execute Python code in a sandboxed environment and return the output. "
    "The code runs in an isolated subprocess with a 30-second timeout. "
    "Use print() to produce output. The result includes both stdout and stderr.",
    parameters={
        "code": {
            "type": "str",
            "description": "Python code to execute. Use print() for output.",
        }
    },
)
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    if not code or not code.strip():
        return "Error: Empty code string."

    # Create a temporary file for the code
    sandbox_dir = settings.workspace_root / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=sandbox_dir,
            delete=False,
        ) as f:
            f.write(code)
            script_path = f.name

        result = subprocess.run(
            ["python", "-u", script_path],
            capture_output=True,
            text=True,
            timeout=settings.code_execution_timeout,
            cwd=str(sandbox_dir),
        )

        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"Exit code: {result.returncode}")

        output = "\n".join(output_parts) if output_parts else "(No output produced)"
        # Truncate to prevent context window overflow
        if len(output) > settings.file_read_max_chars:
            output = output[: settings.file_read_max_chars] + "\n[TRUNCATED]"
        return output

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {settings.code_execution_timeout} seconds."

    except Exception as e:
        error_msg = f"Error: Code execution failed — {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return error_msg

    finally:
        # Clean up the temp script
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass