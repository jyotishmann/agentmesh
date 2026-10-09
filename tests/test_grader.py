# file: tests/test_grader.py
"""Tests for the grader fixes and regrading (PR 15). No GPU needed.

Most answers below are real outputs from the October 2026 eval runs, so
each test pins a grading decision that was once wrong.
"""

import json
from pathlib import Path

import pytest

import agentmesh.eval.runner as runner_module
from agentmesh.config import settings
from agentmesh.eval.metrics import answer_correct, contains_term, normalise_answer
from agentmesh.eval.runner import environment_info, regrade
from agentmesh.orchestrator import OrchestratorResult
from agentmesh.trajectory import TrajectoryLogger

TASKS = {t["task_id"]: t for t in json.loads(
    (Path(__file__).parent.parent / "agentmesh" / "eval" / "tasks.json").read_text()
)}


def _grade(task_id: str, answer: str) -> float | None:
    return answer_correct({"final_output": answer}, TASKS[task_id])


# ── Word forms ──────────────────────────────────────────────────────

class TestWordForms:
    @pytest.mark.parametrize("text", ["gene-editing", "edits", "edited", "edit"])
    def test_inflections_match(self, text):
        assert contains_term(normalise_answer(text), "edit")

    @pytest.mark.parametrize("text, term", [
        ("Titania", "Titan"),      # a different moon, not an inflection
        ("index 410", "41"),       # numbers never take word endings
        ("edition", "edit"),       # not in the inflection list
    ])
    def test_non_inflections_do_not_match(self, text, term):
        assert not contains_term(normalise_answer(text), term)


# ── Real answers, regraded ──────────────────────────────────────────

class TestRealAnswers:
    def test_code_011_correct_phrasing_now_passes(self):
        # Oct 9: graded WRONG because "indeed" broke the phrase "is a prime"
        assert _grade("code_011", "The number 97 is indeed a prime number. Its only divisors are 1 and 97.") == 1.0

    @pytest.mark.parametrize("answer", [
        "Is 97 a prime number? False",       # Oct 6: the buggy function's real output
        "97 is not a prime number.",
        "The function says 97 is not prime.",
    ])
    def test_code_011_wrong_answers_still_fail(self, answer):
        # "a prime number" alone would match these; the forbidden list stops it
        assert _grade("code_011", answer) == 0.0

    def test_factual_008_gene_editing_counts_as_edit(self):
        answer = ("CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists "
                  "to precisely alter DNA. It is used to treat sickle cell disease.")
        assert _grade("factual_008", answer) == 1.0

    def test_factual_010_no_longer_requires_a_date(self):
        # Oct 9 answer: correct, but omitted "1950", which the question never asked for
        answer = ("The Turing Test is a method used to assess whether a machine exhibits intelligent "
                  "behavior indistinguishable from that of a human. Proposed by Alan Turing ... If the "
                  "evaluator cannot reliably identify which is which, the AI is considered to have passed the test.")
        assert _grade("factual_010", answer) == 1.0

    def test_creative_047_accepts_conception_date(self):
        answer = "1. Conception: In the late 1980s, Guido van Rossum ... started implementing Python in December 1989."
        assert _grade("creative_047", answer) == 1.0

    def test_genuinely_wrong_answers_stay_wrong(self):
        # Fixing the grader must not let real mistakes through
        assert _grade("factual_002", "Andrea Ghez won the Nobel Prize in Physics in 2023.") == 0.0
        assert _grade("analysis_033", "Total Price = $106.59. Average = $21.32.") == 0.0
        assert _grade("factual_009", "Titan, Rhea, Dione, Tethys and Enceladus.") == 0.0


# ── Task file rules ─────────────────────────────────────────────────

def test_forbidden_only_on_gradable_tasks():
    for task_def in TASKS.values():
        if "forbidden" in task_def:
            assert task_def["gradable"], task_def["task_id"]
            assert all(isinstance(p, str) and p for p in task_def["forbidden"]), task_def["task_id"]


# ── Environment metadata ────────────────────────────────────────────

def test_environment_info_never_raises():
    info = environment_info()
    assert info["python"]
    assert "torch" in info and "gpu" in info


# ── Regrading ───────────────────────────────────────────────────────

@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """A results file graded by the OLD grader, plus its trajectory database."""
    results_dir = tmp_path / "eval_results"
    results_dir.mkdir()
    monkeypatch.setattr(runner_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(settings, "eval_mirror_dir", None)

    db = str(tmp_path / "trajectories.db")
    logger = TrajectoryLogger(db_path=db)
    answers = {
        "code_011": ("The number 97 is indeed a prime number.", 0.0),  # old grader: wrong
        "factual_001": ("The capital of Australia is Canberra.", 1.0),
        "factual_002": ("Andrea Ghez won the 2023 prize.", 0.0),
    }
    task_results = []
    for i, (task_id, (answer, old_verdict)) in enumerate(answers.items()):
        result = OrchestratorResult(response=answer, task_id=f"task_{i}", completed=True,
                                    token_summary={}, trajectory_events=[])
        logger.save(TASKS[task_id]["task"], result)
        task_results.append({
            "task_id": task_id, "category": TASKS[task_id]["category"],
            "difficulty": TASKS[task_id]["difficulty"], "trajectory_id": f"task_{i}",
            "completed": True, "wall_time_s": 1.0, "error": None,
            "metrics": {"answer_correct": old_verdict, "latency_ms": 123.0},
        })
    # A crashed task has no trajectory
    task_results.append({
        "task_id": "creative_050", "category": "creative", "difficulty": 5,
        "trajectory_id": None, "completed": False, "wall_time_s": 1.0, "error": "CUDA out of memory",
        "metrics": {"answer_correct": None},
    })

    path = results_dir / "eval_20261009_154140.json"
    path.write_text(json.dumps({"run_timestamp": "x", "finished": True, "total_tasks": 4,
                                "summary": {}, "task_results": task_results}))
    return path, db, results_dir


class TestRegrade:
    def test_reports_changed_verdicts_and_keeps_original(self, saved_run):
        path, db, results_dir = saved_run
        original_text = path.read_text()

        results, changes = regrade(str(path), db_path=db)

        assert changes == [{"task_id": "code_011", "before": 0.0, "after": 1.0}]
        assert path.read_text() == original_text                    # original untouched
        assert (results_dir / "eval_20261009_154140_regraded.json").exists()
        assert results["regraded_from"] == path.name
        assert results["regrade_missing"] == ["creative_050"]        # crashed task kept as-is
        assert results["summary"]["overall"]["answer_correct"] == round(2 / 3, 3)

    def test_timing_is_carried_over_not_recomputed(self, saved_run):
        path, db, _ = saved_run
        results, _ = regrade(str(path), db_path=db)
        code_011 = next(r for r in results["task_results"] if r["task_id"] == "code_011")
        assert code_011["metrics"]["latency_ms"] == 123.0

    def test_regrading_twice_is_stable(self, saved_run):
        path, db, results_dir = saved_run
        regrade(str(path), db_path=db)
        _, changes = regrade(str(results_dir / "eval_20261009_154140_regraded.json"), db_path=db)
        assert changes == []                                         # same grader, same verdicts
        assert not (results_dir / "eval_20261009_154140_regraded_regraded.json").exists()
