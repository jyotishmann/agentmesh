# file: tests/test_tools.py
"""Tests for tool registry and individual tools."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentmesh.tools.registry import ToolDef, ToolRegistry, tool


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_decorator_attaches_metadata(self):
        """@tool should attach name, description, params to the function."""

        @tool(
            name="test_tool",
            description="A test tool",
            parameters={"query": "The query string"},
        )
        def my_tool(query: str) -> str:
            return f"result: {query}"

        assert hasattr(my_tool, "_tool_def")
        assert my_tool._tool_def.name == "test_tool"
        assert my_tool._tool_def.description == "A test tool"

    def test_decorated_function_still_callable(self):
        """The decorated function should still work normally."""

        @tool(name="echo", description="Echo", parameters={"x": "input"})
        def echo(x: str) -> str:
            return x

        assert echo("hello") == "hello"


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_call(self):
        """Register a tool and call it by name."""
        registry = ToolRegistry()

        @tool(name="add", description="Add two numbers", parameters={"a": "first", "b": "second"})
        def add(a: int, b: int) -> str:
            return str(a + b)

        registry.register(add)
        result = registry.call("add", {"a": 3, "b": 4})
        assert result == "7"

    def test_call_unknown_tool(self):
        """Calling an unregistered tool should return an error string."""
        registry = ToolRegistry()
        result = registry.call("nonexistent", {})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_list_tools(self):
        """list_tools should return registered tool definitions."""
        registry = ToolRegistry()

        @tool(name="t1", description="Tool 1", parameters={})
        def t1() -> str:
            return "ok"

        registry.register(t1)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "t1"

    def test_get_tool_descriptions(self):
        """get_tool_descriptions should return a formatted string."""
        registry = ToolRegistry()

        @tool(name="search", description="Search the web", parameters={"query": "search query"})
        def search(query: str) -> str:
            return "results"

        registry.register(search)
        desc = registry.get_tool_descriptions()
        assert "search" in desc
        assert "query" in desc

    def test_tool_error_handling(self):
        """Tools that raise should return error strings, not crash."""
        registry = ToolRegistry()

        @tool(name="crasher", description="Always crashes", parameters={})
        def crasher() -> str:
            raise ValueError("boom")

        registry.register(crasher)
        result = registry.call("crasher", {})
        assert "error" in result.lower()


class TestRunPython:
    """Test the run_python tool."""

    def test_simple_execution(self):
        """Should execute Python code and return stdout."""
        from agentmesh.tools.run_python import run_python
        result = run_python(code="print(2 + 2)")
        assert "4" in result

    def test_syntax_error(self):
        """Should capture syntax errors."""
        from agentmesh.tools.run_python import run_python
        result = run_python(code="def f(")
        assert "error" in result.lower() or "SyntaxError" in result

    def test_timeout(self):
        """Long-running code should be killed."""
        from agentmesh.tools.run_python import run_python
        result = run_python(code="import time; time.sleep(60)")
        assert "timeout" in result.lower() or "error" in result.lower()


class TestFileIO:
    """Test file I/O tools."""

    def test_write_and_read(self, temp_sandbox):
        """Should write and read a file within the sandbox."""
        from agentmesh.tools.file_io import write_file, read_file

        with patch("agentmesh.tools.file_io.SANDBOX_DIR", temp_sandbox):
            write_result = write_file(
                path="test.txt", content="hello world"
            )
            assert "success" in write_result.lower() or "wrote" in write_result.lower()

            read_result = read_file(path="test.txt")
            assert "hello world" in read_result

    def test_path_traversal_rejected(self, temp_sandbox):
        """Paths with '..' should be rejected."""
        from agentmesh.tools.file_io import write_file

        with patch("agentmesh.tools.file_io.SANDBOX_DIR", temp_sandbox):
            result = write_file(path="../../../etc/passwd", content="bad")
            assert "error" in result.lower() or "invalid" in result.lower()

    def test_absolute_path_rejected(self, temp_sandbox):
        """Absolute paths should be rejected."""
        from agentmesh.tools.file_io import read_file

        with patch("agentmesh.tools.file_io.SANDBOX_DIR", temp_sandbox):
            result = read_file(path="/etc/passwd")
            assert "error" in result.lower() or "invalid" in result.lower()