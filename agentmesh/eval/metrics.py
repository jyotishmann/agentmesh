# file: agentmesh/eval/metrics.py
"""Eval metrics — computed from trajectories and task definitions.

Every metric takes (trajectory, task_def) and returns a float, or None when
the metric does not apply to that task (for example, answer correctness on a
task whose answer is random). Aggregation skips None, so an ungradable task
never counts as a pass or a fail.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Execution metrics (did the system run well?) ────────────────────

def task_completion(trajectory: dict, task_def: dict) -> float:
    """1.0 if the orchestrator marked the run completed, else 0.0.

    Note: this measures that the run finished without hitting limits and
    the critic did not reject it. It says nothing about correctness —
    see answer_correct for that.
    """
    return 1.0 if trajectory.get("completed", False) else 0.0


def tool_call_efficiency(trajectory: dict, task_def: dict) -> float:
    """Jaccard similarity between expected and actually-used tool sets."""
    expected = set(task_def.get("expected_tools", []))
    actual = {
        e["tool_name"]
        for e in trajectory.get("events", [])
        if e.get("action_type") == "tool_call" and e.get("tool_name")
    }

    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return round(len(expected & actual) / len(expected | actual), 3)


def loop_detected(trajectory: dict, task_def: dict) -> float:
    """1.0 if no loop was detected (good), 0.0 if one was."""
    for event in trajectory.get("events", []):
        if event.get("action_type") == "loop_detected":
            return 0.0
    return 1.0


def latency_ms(trajectory: dict, task_def: dict) -> float:
    """Total wall-clock latency in milliseconds (not normalised)."""
    return float(trajectory.get("total_latency_ms", 0.0))


def token_efficiency(trajectory: dict, task_def: dict) -> float:
    """Output/input token ratio, halved and capped at 1.0."""
    summary = trajectory.get("token_summary", {})
    tokens_in = summary.get("total_tokens_in", 0)
    tokens_out = summary.get("total_tokens_out", 0)
    if tokens_in == 0:
        return 0.0
    return min(round((tokens_out / tokens_in) / 2.0, 3), 1.0)


def _last_critic_verdict(trajectory: dict) -> Optional[bool]:
    """Pass/fail of the last critique event, or None if the critic never ran."""
    verdict = None
    for event in trajectory.get("events", []):
        if event.get("action_type") != "critique":
            continue
        try:
            verdict = bool(json.loads(event.get("tool_output", "{}")).get("pass", True))
        except (json.JSONDecodeError, AttributeError):
            verdict = None
    return verdict


def critic_pass(trajectory: dict, task_def: dict) -> float:
    """1.0 if the critic's last verdict was pass (or the critic never ran)."""
    verdict = _last_critic_verdict(trajectory)
    return 1.0 if verdict is None or verdict else 0.0


# ── Correctness metrics (was the answer right?) ─────────────────────

# "1,896.20" -> "1896.20", but leave "[1,2,3]" alone (no 3-digit group)
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalise_answer(text: str) -> str:
    """Lower-case and strip thousands separators so numbers compare cleanly."""
    return _THOUSANDS_SEPARATOR.sub("", text.lower())


# Inflections a word may carry and still count as the same word:
# "edit" -> "edits", "edited", "editing"; "pass" -> "passes", "passed"
_WORD_FORMS = r"(?:s|es|d|ed|ing)?"


def contains_term(normalised_text: str, term: str) -> bool:
    """Whole-token, case-insensitive match of term inside normalised_text.

    "41" matches "index 41." but not "141" or "410"; "26.1" does not match
    "26.12" (list both as alternatives when either is acceptable). An
    integer still matches its decimal form: "16470" matches "16470.09".
    A term ending in a letter also matches its common inflections, so
    "edit" matches "gene-editing" (but "Titan" does not match "Titania").
    A whitespace-insensitive second pass lets "[1,2,3]" match "[1, 2, 3]".
    """
    needle = normalise_answer(term).strip()
    if not needle:
        return False

    def _search(haystack: str, pattern_text: str) -> bool:
        # Not preceded or followed by a letter/digit -> whole-token match
        forms = _WORD_FORMS if pattern_text[-1].isalpha() else ""
        pattern = rf"(?<![a-z0-9]){re.escape(pattern_text)}{forms}(?![a-z0-9])"
        return re.search(pattern, haystack) is not None

    if _search(normalised_text, needle):
        return True

    squeezed_needle = re.sub(r"\s+", "", needle)
    squeezed_text = re.sub(r"\s+", "", normalised_text)
    return _search(squeezed_text, squeezed_needle)


def answer_correct(trajectory: dict, task_def: dict) -> Optional[float]:
    """Grade the final answer against the task's expected keywords.

    task_def["expected"] is a list of groups. Every group must match, and a
    group matches if ANY of its alternatives appears in the answer:

        [["Canberra"]]                       -> must mention Canberra
        [["4181", "6765"], ["34"]]           -> (4181 OR 6765) AND 34

    task_def["forbidden"] (optional) lists phrases that make an answer
    wrong even if every group matches, e.g. ["not prime"] so that
    "97 is not a prime number" can't pass on the word "prime".

    Returns None for tasks marked "gradable": false.
    """
    if not task_def.get("gradable", False):
        return None

    groups = task_def.get("expected") or []
    if not groups:
        logger.warning(f"{task_def.get('task_id')}: gradable but no expected answers")
        return None

    answer = normalise_answer(trajectory.get("final_output", "") or "")
    if any(contains_term(answer, phrase) for phrase in task_def.get("forbidden", [])):
        return 0.0
    for alternatives in groups:
        if not any(contains_term(answer, alt) for alt in alternatives):
            return 0.0
    return 1.0


def critic_agreement(trajectory: dict, task_def: dict) -> Optional[float]:
    """1.0 if the critic's verdict matches ground truth, else 0.0.

    Measures how trustworthy the LLM judge is. None when the task is
    ungradable or the critic never ran.
    """
    correct = answer_correct(trajectory, task_def)
    verdict = _last_critic_verdict(trajectory)
    if correct is None or verdict is None:
        return None
    return 1.0 if verdict == bool(correct) else 0.0


# ── Aggregator ──────────────────────────────────────────────────────

ALL_METRICS = {
    "task_completion": task_completion,
    "answer_correct": answer_correct,
    "critic_agreement": critic_agreement,
    "tool_call_efficiency": tool_call_efficiency,
    "loop_detected": loop_detected,
    "latency_ms": latency_ms,
    "token_efficiency": token_efficiency,
    "critic_pass": critic_pass,
}


def compute_all_metrics(trajectory: dict, task_def: dict) -> dict[str, Optional[float]]:
    """Compute every metric for one trajectory. Failing metrics become None."""
    results: dict[str, Optional[float]] = {}
    for name, fn in ALL_METRICS.items():
        try:
            results[name] = fn(trajectory, task_def)
        except Exception as e:
            logger.error(f"Metric '{name}' failed: {e}")
            results[name] = None
    return results
