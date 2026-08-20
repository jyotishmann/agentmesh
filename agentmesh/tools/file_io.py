# file: agentmesh/tools/file_io.py
"""File read/write tools with path traversal protection."""

import logging
from pathlib import Path

from agentmesh.config import settings
from agentmesh.tools.registry import tool

logger = logging.getLogger(__name__)


def _validate_path(path: str) -> tuple[bool, str, Path | None]:
    """Validate a file path for sandbox access.

    Returns (is_valid, error_message, resolved_path).
    """
    if not path or not path.strip():
        return False, "Error: Empty file path.", None

    # Reject absolute paths
    if Path(path).is_absolute():
        return False, "Error: Absolute paths are not allowed. Use relative paths.", None

    # Reject path traversal
    if ".." in path:
        return False, "Error: Path traversal (..) is not allowed.", None

    # Resolve within sandbox
    sandbox = settings.workspace_root / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    resolved = (sandbox / path).resolve()

    # Verify resolved path is still within sandbox
    if not str(resolved).startswith(str(sandbox.resolve())):
        return False, "Error: Path resolves outside the sandbox directory.", None

    return True, "", resolved


@tool(
    name="read_file",
    description="Read the contents of a file from the sandbox directory. "
    "Only relative paths within the sandbox are allowed. "
    "Large files are truncated to 10,000 characters.",
    parameters={
        "path": {
            "type": "str",
            "description": "Relative path to the file (e.g., 'data.csv', 'reports/output.txt')",
        }
    },
)
def read_file(path: str) -> str:
    """Read a file from the sandbox directory."""
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return error

    if not resolved.exists():
        return f"Error: File not found: {path}"

    if not resolved.is_file():
        return f"Error: Path is not a file: {path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if len(content) > settings.file_read_max_chars:
            content = content[: settings.file_read_max_chars] + "\n[TRUNCATED]"
        return content

    except Exception as e:
        return f"Error: Failed to read file — {type(e).__name__}: {str(e)}"


@tool(
    name="write_file",
    description="Write content to a file in the sandbox directory. "
    "Creates the file if it doesn't exist. Creates intermediate directories. "
    "Only relative paths within the sandbox are allowed.",
    parameters={
        "path": {
            "type": "str",
            "description": "Relative path for the file (e.g., 'output.txt', 'reports/summary.md')",
        },
        "content": {
            "type": "str",
            "description": "Content to write to the file",
        },
    },
)
def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox directory."""
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return error

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        return f"Successfully wrote {byte_count} bytes to {path}"

    except Exception as e:
        return f"Error: Failed to write file — {type(e).__name__}: {str(e)}"