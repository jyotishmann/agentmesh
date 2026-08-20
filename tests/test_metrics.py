# file: tests/test_metrics.py
"""Tests for eval metrics."""

import json
import pytest

from agentmesh.eval.metrics import (
    task_completion,
    tool_call_efficiency,
    loop_detected,
    latency_ms,
    token_efficiency,
    critic_pass,
    compute_all_metrics,
)


def _make_trajectory(
    completed=True,
    tool_calls=None,
    has_loop=False,
    total_latency=1000.0,
    tokens_in=100,
    tokens_out=50,
    critic_verdict=None,
):
    """Build a synthetic trajectory dict for testing."""
    events = []

    for tc in (tool_calls or []):
        events.append({
            "action_type": "tool_call",
            "tool_name": tc,
        })

    if has_loop:
        events.append({"action_type": "loop_detected"})

    if critic_verdict is not None:
        events.append({
            "action_type": "critique",
            "tool_output": json.dumps(critic_verdict),
        })

    return {
        "completed": completed,
        "total_latency_ms": total_latency,
        "token_summary": {
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
        },
        "events": events,
    }


class TestTaskCompletion:
    def test_completed(self):
        t = _make_trajectory(completed=True)
        assert task_completion(t, {}) == 1.0

    def test_not_completed(self):
        t = _make_trajectory(completed=False)
        assert task_completion(t, {}) == 0.0


class TestToolCallEfficiency:
    def test_perfect_match(self):
        """Expected and actual tools are identical."""
        t = _make_trajectory(tool_calls=["search_web", "run_python"])
        task_def = {"expected_tools": ["search_web", "run_python"]}
        assert tool_call_efficiency(t, task_def) == 1.0

    def test_partial_match(self):
        """Some expected tools missing."""
        t = _make_trajectory(tool_calls=["search_web"])
        task_def = {"expected_tools": ["search_web", "run_python"]}
        assert tool_call_efficiency(t, task_def) == 0.5

    def test_extra_tools(self):
        """Agent used tools not expected."""
        t = _make_trajectory(tool_calls=["search_web", "run_python", "write_file"])
        task_def = {"expected_tools": ["search_web"]}
        # Jaccard: {search_web} / {search_web, run_python, write_file} = 1/3
        assert abs(tool_call_efficiency(t, task_def) - 0.333) < 0.01

    def test_both_empty(self):
        """No tools expected, none used."""
        t = _make_trajectory(tool_calls=[])
        task_def = {"expected_tools": []}
        assert tool_call_efficiency(t, task_def) == 1.0

    def test_expected_empty_actual_nonempty(self):
        """No tools expected but agent used some."""
        t = _make_trajectory(tool_calls=["search_web"])
        task_def = {"expected_tools": []}
        assert tool_call_efficiency(t, task_def) == 0.0


class TestLoopDetected:
    def test_no_loop(self):
        t = _make_trajectory(has_loop=False)
        assert loop_detected(t, {}) == 1.0

    def test_loop_found(self):
        t = _make_trajectory(has_loop=True)
        assert loop_detected(t, {}) == 0.0


class TestLatency:
    def test_latency_value(self):
        t = _make_trajectory(total_latency=5432.1)
        assert latency_ms(t, {}) == 5432.1


class TestTokenEfficiency:
    def test_ratio(self):
        t = _make_trajectory(tokens_in=100, tokens_out=100)
        # ratio = 100/100 = 1.0, capped: 1.0/2.0 = 0.5
        assert token_efficiency(t, {}) == 0.5

    def test_high_ratio_capped(self):
        t = _make_trajectory(tokens_in=50, tokens_out=200)
        # ratio = 4.0, capped: 4.0/2.0 = 2.0 → min(2.0, 1.0) = 1.0
        assert token_efficiency(t, {}) == 1.0


class TestCriticPass:
    def test_pass(self):
        t = _make_trajectory(critic_verdict={"pass": True})
        assert critic_pass(t, {}) == 1.0

    def test_fail(self):
        t = _make_trajectory(critic_verdict={"pass": False})
        assert critic_pass(t, {}) == 0.0

    def test_no_critic(self):
        t = _make_trajectory()
        assert critic_pass(t, {}) == 1.0


class TestComputeAll:
    def test_returns_all_metrics(self):
        t = _make_trajectory(
            completed=True,
            tool_calls=["search_web"],
            critic_verdict={"pass": True},
        )
        task_def = {"expected_tools": ["search_web"]}
        results = compute_all_metrics(t, task_def)

        assert "task_completion" in results
        assert "tool_call_efficiency" in results
        assert "loop_detected" in results
        assert "latency_ms" in results
        assert "token_efficiency" in results
        assert "critic_pass" in results
        assert len(results) == 6