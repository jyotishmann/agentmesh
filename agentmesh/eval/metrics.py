# file: agentmesh/eval/metrics.py
"""Eval metrics — computed from trajectories and task definitions."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def task_completion(trajectory: dict, task_def: dict) -> float:
    """Binary: did the orchestrator mark the task as completed?

    Returns:
        1.0 if completed, 0.0 otherwise.
    """
    return 1.0 if trajectory.get("completed", False) else 0.0


def tool_call_efficiency(trajectory: dict, task_def: dict) -> float:
    """Jaccard similarity between expected and actual tool sets.

    expected_tools from task definition vs tools actually called.
    Score = |intersection| / |union|.
    If both sets are empty (no tools expected, none used), returns 1.0.

    Returns:
        Float in [0, 1].
    """
    expected = set(task_def.get("expected_tools", []))
    actual = set()

    for event in trajectory.get("events", []):
        if event.get("action_type") == "tool_call" and event.get("tool_name"):
            actual.add(event["tool_name"])

    if not expected and not actual:
        return 1.0  # no tools expected, none used — perfect

    if not expected or not actual:
        # One set empty, one not — poor efficiency
        return 0.0

    intersection = expected & actual
    union = expected | actual
    return round(len(intersection) / len(union), 3)


def loop_detected(trajectory: dict, task_def: dict) -> float:
    """Binary: was a loop detected during execution?

    Returns:
        0.0 if a loop was detected (bad), 1.0 if no loop (good).
    """
    for event in trajectory.get("events", []):
        if event.get("action_type") == "loop_detected":
            return 0.0
    return 1.0

def latency_ms(trajectory: dict, task_def: dict) -> float:
    """Total wall-clock latency in milliseconds.

    Returns:
        Total latency as a positive float. Not normalised.
    """
    return float(trajectory.get("total_latency_ms", 0.0))


def token_efficiency(trajectory: dict, task_def: dict) -> float:
    """Ratio of output tokens to input tokens, capped at 1.0.

    A ratio near 1.0 means the model generates roughly as much as it reads.
    Above 2.0 → capped to 1.0 (generous output is fine).

    Returns:
        Float in [0, 1].
    """
    token_summary = trajectory.get("token_summary", {})
    tokens_in = token_summary.get("total_tokens_in", 1)
    tokens_out = token_summary.get("total_tokens_out", 0)

    if tokens_in == 0:
        return 0.0

    ratio = tokens_out / tokens_in
    return min(round(ratio / 2.0, 3), 1.0)


def critic_pass(trajectory: dict, task_def: dict) -> float:
    """Did the critic pass the final output?

    Looks for the last critique event and checks its verdict.

    Returns:
        1.0 if passed (or no critic event), 0.0 if rejected.
    """
    last_critique = None
    for event in trajectory.get("events", []):
        if event.get("action_type") == "critique":
            last_critique = event

    if last_critique is None:
        return 1.0  # no critic ran — assume pass

    try:
        verdict = json.loads(last_critique.get("tool_output", "{}"))
        return 1.0 if verdict.get("pass", True) else 0.0
    except json.JSONDecodeError:
        return 0.5  # parse error — uncertain


# ── Aggregator ──────────────────────────────────────────────────────

ALL_METRICS = {
    "task_completion": task_completion,
    "tool_call_efficiency": tool_call_efficiency,
    "loop_detected": loop_detected,
    "latency_ms": latency_ms,
    "token_efficiency": token_efficiency,
    "critic_pass": critic_pass,
}


def compute_all_metrics(
    trajectory: dict, task_def: dict
) -> dict[str, float]:
    """Compute all metrics for a single trajectory.

    Returns:
        Dict mapping metric name to value.
    """
    results = {}
    for name, fn in ALL_METRICS.items():
        try:
            results[name] = fn(trajectory, task_def)
        except Exception as e:
            logger.error(f"Metric '{name}' failed: {e}")
            results[name] = 0.0
    return results