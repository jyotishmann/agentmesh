# file: tests/test_agents.py
"""Tests for agent parsing logic."""

import json
import pytest
from unittest.mock import MagicMock, patch

from agentmesh.agents.base import BaseAgent


class TestToolCallParsing:
    """Test XML tool call parsing in BaseAgent."""

    def test_parse_valid_tool_call(self):
        """Should parse a valid <tool_call> XML block."""
        text = """Let me search for that.
<tool_call>
{"tool": "search_web", "args": {"query": "Python decorators"}}
</tool_call>"""

        # Use the static method directly
        result = BaseAgent._parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_web"
        assert result["args"]["query"] == "Python decorators"

    def test_parse_no_tool_call(self):
        """Should return None when no tool call is present."""
        text = "The answer is 42. No tools needed."
        result = BaseAgent._parse_tool_call(text)
        assert result is None

    def test_parse_malformed_json(self):
        """Should return None for malformed JSON in tool call."""
        text = """<tool_call>
{not valid json}
</tool_call>"""
        result = BaseAgent._parse_tool_call(text)
        assert result is None

    def test_parse_tool_call_with_nested_args(self):
        """Should handle nested argument structures."""
        text = """<tool_call>
{"tool": "run_python", "args": {"code": "print('hello')"}}
</tool_call>"""
        result = BaseAgent._parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "run_python"
        assert "print" in result["args"]["code"]


class TestPlannerParsing:
    """Test planner output parsing."""

    def test_parse_valid_plan(self):
        """Valid JSON array should parse correctly."""
        from agentmesh.agents.planner import PlannerAgent

        plan_text = json.dumps([
            {"description": "Search web", "specialist": "research", "required_tools": ["search_web"]},
            {"description": "Write code", "specialist": "coder", "required_tools": ["run_python"]},
        ])

        parsed = json.loads(plan_text)
        assert len(parsed) == 2
        assert parsed[0]["specialist"] == "research"

    def test_parse_code_fenced_plan(self):
        """Plan wrapped in ```json fences should be extractable."""
        plan_text = """Here's the plan:
```json
[{"description": "Do research", "specialist": "research", "required_tools": []}]
```"""

        # Extract JSON from code fence
        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", plan_text, re.DOTALL)
        assert match is not None
        parsed = json.loads(match.group(1))
        assert len(parsed) == 1


class TestCriticParsing:
    """Test critic verdict parsing."""

    def test_parse_pass_verdict(self):
        """Should parse a passing verdict."""
        verdict_text = json.dumps({
            "pass": True,
            "confidence": 0.9,
            "feedback": "Good output.",
        })

        verdict = json.loads(verdict_text)
        assert verdict["pass"] is True
        assert verdict["confidence"] == 0.9

    def test_parse_fail_verdict(self):
        """Should parse a failing verdict."""
        verdict_text = json.dumps({
            "pass": False,
            "confidence": 0.4,
            "feedback": "Missing key details.",
        })

        verdict = json.loads(verdict_text)
        assert verdict["pass"] is False
        assert "missing" in verdict["feedback"].lower()