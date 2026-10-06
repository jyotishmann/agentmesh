# file: tests/test_correctness.py
"""Tests for correctness-grounded evals (PR 13). No GPU needed."""

import json
from pathlib import Path

import pytest

from agentmesh.agents.base import BaseAgent
from agentmesh.eval.metrics import (
    answer_correct,
    compute_all_metrics,
    contains_term,
    critic_agreement,
    normalise_answer,
)
from agentmesh.eval.runner import EvalRunner

TASKS_PATH = Path(__file__).parent.parent / "agentmesh" / "eval" / "tasks.json"


def _trajectory(final_output: str, critic_pass: bool | None = None) -> dict:
    """Minimal trajectory dict, as TrajectoryLogger.get() would return it."""
    events = []
    if critic_pass is not None:
        events.append({
            "action_type": "critique",
            "tool_output": json.dumps({"pass": critic_pass}),
        })
    return {"final_output": final_output, "completed": True, "events": events}


def _task(expected, gradable: bool = True) -> dict:
    return {"task_id": "t", "gradable": gradable, "expected": expected, "expected_tools": []}


# ── Matching rules ──────────────────────────────────────────────────

class TestMatching:
    def test_case_insensitive(self):
        assert contains_term(normalise_answer("The answer is CANBERRA."), "Canberra")

    def test_whole_token_only(self):
        text = normalise_answer("Found at index 141")
        assert not contains_term(text, "41")
        assert contains_term(normalise_answer("Found at index 41."), "41")

    def test_decimal_precision_is_distinct(self):
        # "26.1" must not match "26.12"; list both when either is acceptable
        assert not contains_term(normalise_answer("BMI is 26.12"), "26.1")

    def test_integer_matches_its_decimal_form(self):
        assert contains_term(normalise_answer("Final value: $16,470.09"), "16470")

    def test_thousands_separator_removed(self):
        assert contains_term(normalise_answer("Payment: $1,896.20"), "1896")

    def test_list_spacing_ignored(self):
        assert contains_term(normalise_answer("Result: [1,2,3,4,5,6]"), "[1, 2, 3, 4, 5, 6]")

    def test_negated_phrase_does_not_match(self):
        assert not contains_term(normalise_answer("97 is not prime"), "is prime")


# ── answer_correct ──────────────────────────────────────────────────

class TestAnswerCorrect:
    def test_all_groups_must_match(self):
        task = _task([["Titan"], ["Rhea"]])
        assert answer_correct(_trajectory("Titan and Rhea"), task) == 1.0
        assert answer_correct(_trajectory("Only Titan"), task) == 0.0

    def test_any_alternative_in_group(self):
        task = _task([["4181", "6765"]])
        assert answer_correct(_trajectory("... 2584, 4181]"), task) == 1.0
        assert answer_correct(_trajectory("... 4181, 6765]"), task) == 1.0

    def test_ungradable_returns_none(self):
        assert answer_correct(_trajectory("anything"), _task([], gradable=False)) is None

    def test_real_failure_from_first_eval(self):
        # code_013 on 2026-10-05: the code ran, but the answer showed only the code
        code_only = 'def reverse_string(s):\n    return s[::-1]\nprint(reverse_string("Hello World"))'
        assert answer_correct(_trajectory(code_only), _task([["dlroW olleH"]])) == 0.0


# ── critic_agreement ────────────────────────────────────────────────

class TestCriticAgreement:
    def test_agrees_on_correct_answer(self):
        assert critic_agreement(_trajectory("Canberra", critic_pass=True), _task([["Canberra"]])) == 1.0

    def test_false_pass_is_disagreement(self):
        assert critic_agreement(_trajectory("Sydney", critic_pass=True), _task([["Canberra"]])) == 0.0

    def test_none_when_critic_never_ran(self):
        assert critic_agreement(_trajectory("Canberra"), _task([["Canberra"]])) is None

    def test_none_when_ungradable(self):
        assert critic_agreement(_trajectory("x", critic_pass=True), _task([], gradable=False)) is None

    def test_all_metrics_present(self):
        metrics = compute_all_metrics(_trajectory("Canberra", True), _task([["Canberra"]]))
        assert set(metrics) >= {"answer_correct", "critic_agreement", "task_completion"}


# ── Summary aggregation ─────────────────────────────────────────────

class TestSummary:
    def test_none_values_are_skipped(self):
        results = [
            {"category": "a", "difficulty": 1, "wall_time_s": 1.0,
             "metrics": {"answer_correct": 1.0, "critic_agreement": 1.0, "critic_pass": 1.0}},
            {"category": "a", "difficulty": 1, "wall_time_s": 1.0,
             "metrics": {"answer_correct": None, "critic_agreement": None, "critic_pass": 1.0}},
        ]
        overall = EvalRunner._compute_summary(results)["overall"]
        assert overall["answer_correct"] == 1.0   # not 0.5
        assert overall["graded_tasks"] == 1

    def test_critic_false_pass_rate(self):
        results = [
            {"category": "a", "difficulty": 1, "wall_time_s": 1.0,
             "metrics": {"answer_correct": 0.0, "critic_agreement": 0.0, "critic_pass": 1.0}},
            {"category": "a", "difficulty": 1, "wall_time_s": 1.0,
             "metrics": {"answer_correct": 0.0, "critic_agreement": 1.0, "critic_pass": 0.0}},
        ]
        overall = EvalRunner._compute_summary(results)["overall"]
        assert overall["critic_false_pass_rate"] == 0.5


# ── Execution grounding ─────────────────────────────────────────────

class TestExecutionGrounding:
    def test_appends_output_when_missing(self):
        grounded = BaseAgent._ground_in_execution("def f(): ...", "STDOUT:\ndlroW olleH\n")
        assert "dlroW olleH" in grounded
        assert "**Execution output:**" in grounded

    def test_leaves_answer_alone_when_reported(self):
        answer = "The reversed string is dlroW olleH."
        assert BaseAgent._ground_in_execution(answer, "STDOUT:\ndlroW olleH\n") == answer

    def test_short_lines_are_not_trusted_as_evidence(self):
        # "2" appears in almost any text, so the real output is still appended
        grounded = BaseAgent._ground_in_execution("Primes: see code 2", "STDOUT:\n2\n3\n5\n")
        assert "**Execution output:**" in grounded

    def test_no_execution_means_no_change(self):
        assert BaseAgent._ground_in_execution("4", "") == "4"


# ── Task file integrity ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def tasks():
    return json.loads(TASKS_PATH.read_text())


class TestTaskFile:
    def test_fifty_unique_tasks(self, tasks):
        assert len(tasks) == 50
        assert len({t["task_id"] for t in tasks}) == 50

    def test_gradable_tasks_have_expected_answers(self, tasks):
        for t in tasks:
            if t["gradable"]:
                assert t["expected"], t["task_id"]
                assert all(isinstance(group, list) and group for group in t["expected"]), t["task_id"]
            else:
                assert t["expected"] == [], t["task_id"]
