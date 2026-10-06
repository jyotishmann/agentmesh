# file: tests/test_reliability.py
"""Tests for reliable evals (PR 14). No GPU needed: no model is ever loaded."""

import json
from pathlib import Path

import pytest

import agentmesh.eval.runner as runner_module
from agentmesh.config import settings
from agentmesh.eval.cli import collect_runs
from agentmesh.eval.runner import EvalRunner
from agentmesh.models.base import ModelResponse
from agentmesh.models.manager import ModelManager
from agentmesh.orchestrator import (
    OrchestratorResult,
    _required_specialist,
    _revision_specialist,
    _validate_plan,
)
from agentmesh.trajectory import TrajectoryLogger

TASKS = json.loads((Path(__file__).parent.parent / "agentmesh" / "eval" / "tasks.json").read_text())


# ── Task intent detection ───────────────────────────────────────────

class TestRequiredSpecialist:
    @pytest.mark.parametrize("task", [t["task"] for t in TASKS if t["category"] == "code_generation"])
    def test_every_coding_task_needs_code(self, task):
        assert _required_specialist(task) == "coder"

    @pytest.mark.parametrize(
        "task_def",
        [t for t in TASKS if "run_python" not in t["expected_tools"]],
        ids=lambda t: t["task_id"],
    )
    def test_no_false_positives(self, task_def):
        # A task that shouldn't run code must never get a forced coder/analyst step
        assert _required_specialist(task_def["task"]) is None

    def test_programming_language_is_not_a_coding_request(self):
        assert _required_specialist("What programming language is Linux written in?") is None
        assert _required_specialist(
            "Search the web for the history of the Python programming language. Write a timeline."
        ) is None

    def test_classify_is_not_class(self):
        assert _required_specialist("Classify the result according to WHO categories.") is None

    def test_compute_only_tasks_go_to_analyst(self):
        assert _required_specialist("Calculate the monthly payment on a mortgage.") == "analyst"


# ── Plan validation ─────────────────────────────────────────────────

class TestValidatePlan:
    def test_empty_plan_falls_back(self):
        plan, notes = _validate_plan("Write a Python function to reverse a string.", [])
        assert plan == [{"description": "Write a Python function to reverse a string.", "specialist": "coder"}]
        assert notes

    def test_good_plan_is_untouched(self):
        raw = [{"description": "find c", "specialist": "research"},
               {"description": "compute time", "specialist": "coder"}]
        plan, notes = _validate_plan("Search for c, then write Python code to compute it.", raw)
        assert [s["specialist"] for s in plan] == ["research", "coder"]
        assert notes == []

    def test_research_only_plan_for_code_task_gets_a_coder(self):
        # The real code_012 failure: the planner sent a coding task to research only
        raw = [{"description": "Determine if Python is required tool", "specialist": "research"}]
        plan, notes = _validate_plan("Write a Python script that prints Fibonacci numbers.", raw)
        assert plan[-1]["specialist"] == "coder"
        assert any("added coder" in n for n in notes)

    def test_consecutive_same_specialist_steps_merge(self):
        raw = [{"description": "Generate the numbers", "specialist": "coder"},
               {"description": "Print them", "specialist": "coder"}]
        plan, notes = _validate_plan("Write a Python script for Fibonacci.", raw)
        assert len(plan) == 1
        assert "Generate the numbers" in plan[0]["description"]
        assert "Print them" in plan[0]["description"]

    def test_unknown_specialist_and_malformed_steps(self):
        raw = ["not a step", {"description": "x", "specialist": "wizard"}]
        plan, notes = _validate_plan("Who won the 2023 Nobel Prize?", raw)
        assert plan == [{"description": "x", "specialist": "research"}]
        assert len(notes) == 2

    def test_plan_is_capped(self):
        raw = [{"description": f"s{i}", "specialist": s}
               for i, s in enumerate(["research", "coder", "analyst", "research", "coder"])]
        plan, _ = _validate_plan("Explain something.", raw)
        assert len(plan) == 4


# ── Revision routing ────────────────────────────────────────────────

class TestRevisionSpecialist:
    def test_code_task_revised_by_code_capable_step(self):
        plan = [{"description": "a", "specialist": "coder"},
                {"description": "b", "specialist": "research"}]
        assert _revision_specialist("Write a Python function to reverse a string.", plan) == "coder"

    def test_non_code_task_revised_by_last_step(self):
        plan = [{"description": "a", "specialist": "analyst"},
                {"description": "b", "specialist": "research"}]
        assert _revision_specialist("Who discovered penicillin?", plan) == "research"


# ── Deterministic decoding ──────────────────────────────────────────

class TestDeterministic:
    @staticmethod
    def _recording_manager(deterministic: bool):
        manager = ModelManager(deterministic=deterministic)  # models load lazily: nothing loads here
        seen = []

        def fake_generate(messages, temperature=None, max_new_tokens=None, **kwargs):
            seen.append(temperature)
            return ModelResponse(text="ok", tokens_in=1, tokens_out=1, latency_ms=0.0)

        manager._main_provider.generate = fake_generate
        manager._specialist_provider.generate = fake_generate
        return manager, seen

    def test_forces_greedy_even_with_explicit_temperature(self):
        manager, seen = self._recording_manager(deterministic=True)
        manager.generate([{"role": "user", "content": "hi"}], temperature=0.3)
        manager.generate([{"role": "user", "content": "hi"}], use_specialist=True)
        assert seen == [0.0, 0.0]

    def test_off_by_default(self):
        manager, seen = self._recording_manager(deterministic=False)
        manager.generate([{"role": "user", "content": "hi"}], temperature=0.3)
        assert seen == [0.3]


# ── Runner: eval conditions, saving, resume ─────────────────────────

class _Interrupted(BaseException):
    """Stands in for a Colab disconnect: not caught by run_single."""


class _FakeOrchestrator:
    """Answers every task correctly-ish and records how it was called."""

    def __init__(self, fail_on_call: int | None = None):
        self.model_manager = type("MM", (), {"deterministic": False})()
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call

    def run(self, task: str, session_id: str = "", use_memory: bool = True):
        self.calls.append({"task": task, "use_memory": use_memory,
                           "deterministic": self.model_manager.deterministic})
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise _Interrupted()
        return OrchestratorResult(
            response=f"Answer: Canberra. C. energy entropy absolute zero ({task[:10]})",
            task_id=f"task_{len(self.calls)}_{abs(hash(task)) % 10**8}",
            completed=True,
            token_summary={"total_tokens_in": 10, "total_tokens_out": 5, "total_tokens": 15},
            trajectory_events=[],
        )


@pytest.fixture
def isolated_results(tmp_path, monkeypatch):
    """Point results and the mirror at temporary folders."""
    results = tmp_path / "eval_results"
    mirror = tmp_path / "mirror"
    monkeypatch.setattr(runner_module, "RESULTS_DIR", results)
    monkeypatch.setattr(settings, "eval_mirror_dir", mirror)
    return results, mirror


class TestRunner:
    FILTERS = {"categories": ["factual_qa"], "max_difficulty": 1}  # factual_001, 003, 005

    def test_eval_conditions_applied_and_restored(self, tmp_path, isolated_results):
        orchestrator = _FakeOrchestrator()
        runner = EvalRunner(orchestrator, TrajectoryLogger(db_path=str(tmp_path / "t.db")))
        runner.run_all(**self.FILTERS)

        assert all(c["deterministic"] and c["use_memory"] is False for c in orchestrator.calls)
        assert orchestrator.model_manager.deterministic is False  # restored afterwards

    def test_interrupted_run_keeps_partial_results_and_resumes(self, tmp_path, isolated_results):
        results_dir, mirror_dir = isolated_results
        db = str(tmp_path / "t.db")

        # Disconnect during the 3rd of 3 tasks
        with pytest.raises(_Interrupted):
            EvalRunner(_FakeOrchestrator(fail_on_call=3), TrajectoryLogger(db_path=db)).run_all(**self.FILTERS)

        saved = list(results_dir.glob("eval_*.json"))
        assert len(saved) == 1
        partial = json.loads(saved[0].read_text())
        assert partial["finished"] is False
        assert [r["task_id"] for r in partial["task_results"]] == ["factual_001", "factual_003"]
        assert (mirror_dir / saved[0].name).exists()  # the Drive copy survived too

        # Resume from the mirror copy, as you would after a Colab reset
        orchestrator = _FakeOrchestrator()
        final = EvalRunner(orchestrator, TrajectoryLogger(db_path=db)).run_all(
            resume=str(mirror_dir / saved[0].name)
        )
        assert len(orchestrator.calls) == 1  # only the missing task ran
        assert final["finished"] is True
        assert [r["task_id"] for r in final["task_results"]] == ["factual_001", "factual_003", "factual_005"]
        assert final["filters"] == self.FILTERS


# ── History ─────────────────────────────────────────────────────────

class TestCollectRuns:
    def test_same_run_in_two_folders_counts_once(self, tmp_path):
        local, mirror = tmp_path / "local", tmp_path / "mirror"
        local.mkdir()
        mirror.mkdir()
        (local / "eval_1.json").write_text(json.dumps({"task_results": [1]}))
        (mirror / "eval_1.json").write_text(json.dumps({"task_results": [1, 2]}))
        (mirror / "eval_2.json").write_text(json.dumps({"task_results": []}))

        runs = collect_runs([local, mirror])
        assert [p.name for p, _ in runs] == ["eval_1.json", "eval_2.json"]
        assert len(runs[0][1]["task_results"]) == 2  # the more complete copy wins
