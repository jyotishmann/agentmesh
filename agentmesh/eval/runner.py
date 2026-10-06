# file: agentmesh/eval/runner.py
"""Eval Runner — executes tasks through the orchestrator and computes metrics."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from agentmesh.config import settings
from agentmesh.eval.metrics import ALL_METRICS, compute_all_metrics
from agentmesh.orchestrator import Orchestrator
from agentmesh.trajectory import TrajectoryLogger

logger = logging.getLogger(__name__)

TASKS_PATH = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(settings.db_path).parent / "eval_results"

# Metrics averaged in the summary (latency/token metrics are reported separately)
SUMMARY_METRICS = [
    "task_completion",
    "answer_correct",
    "critic_agreement",
    "tool_call_efficiency",
    "loop_detected",
    "critic_pass",
]


def _load_tasks() -> list[dict]:
    """Load eval tasks from JSON file."""
    with open(TASKS_PATH) as f:
        return json.load(f)


def _mean(values: list[Optional[float]]) -> Optional[float]:
    """Average of the non-None values, or None if there are none."""
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 3) if present else None


class EvalRunner:
    """Runs eval tasks through the orchestrator and computes metrics."""

    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
    ):
        self.orchestrator = orchestrator or Orchestrator()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        self.tasks = _load_tasks()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _find_task(self, task_id: str) -> dict:
        for task_def in self.tasks:
            if task_def["task_id"] == task_id:
                return task_def
        raise ValueError(f"Task '{task_id}' not found in tasks.json")

    def run_single(self, task_id: str) -> dict:
        """Run one eval task and return its metrics."""
        task_def = self._find_task(task_id)
        logger.info(f"[EVAL] Running {task_id}: {task_def['task'][:60]}...")
        start = time.time()

        try:
            result = self.orchestrator.run(task_def["task"])
            self.trajectory_logger.save(task_def["task"], result)
            trajectory = self.trajectory_logger.get(result.task_id)
            metrics = compute_all_metrics(trajectory, task_def)

            return {
                "task_id": task_id,
                "category": task_def["category"],
                "difficulty": task_def["difficulty"],
                "gradable": task_def.get("gradable", False),
                "trajectory_id": result.task_id,
                "completed": result.completed,
                "metrics": metrics,
                "wall_time_s": round(time.time() - start, 2),
                "error": None,
            }

        except Exception as e:
            logger.error(f"[EVAL] Task {task_id} failed: {e}")
            metrics = {name: 0.0 for name in ALL_METRICS}
            # A crash is a wrong answer if the task is gradable; the critic never ran
            metrics["answer_correct"] = 0.0 if task_def.get("gradable") else None
            metrics["critic_agreement"] = None
            return {
                "task_id": task_id,
                "category": task_def["category"],
                "difficulty": task_def["difficulty"],
                "gradable": task_def.get("gradable", False),
                "trajectory_id": None,
                "completed": False,
                "metrics": metrics,
                "wall_time_s": round(time.time() - start, 2),
                "error": str(e),
            }

    def run_all(
        self,
        categories: Optional[list[str]] = None,
        max_difficulty: int = 5,
    ) -> dict:
        """Run all (optionally filtered) tasks and save results to JSON."""
        filtered = [
            t for t in self.tasks
            if (categories is None or t["category"] in categories)
            and t["difficulty"] <= max_difficulty
        ]
        logger.info(
            f"[EVAL] Running {len(filtered)} tasks "
            f"(categories={categories}, max_difficulty={max_difficulty})"
        )

        task_results = []
        for i, task_def in enumerate(filtered, 1):
            logger.info(f"[EVAL] [{i}/{len(filtered)}] {task_def['task_id']}")
            task_results.append(self.run_single(task_def["task_id"]))

        output = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tasks": len(task_results),
            "summary": self._compute_summary(task_results),
            "task_results": task_results,
        }

        output_path = RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"[EVAL] Results saved to {output_path}")
        return output

    @staticmethod
    def _compute_summary(task_results: list[dict]) -> dict:
        """Aggregate metrics overall, by category, and by difficulty.

        None values (ungradable tasks, critic not run) are skipped, so each
        average is over the tasks where that metric actually applies.
        """
        if not task_results:
            return {"overall": {}, "by_category": {}, "by_difficulty": {}}

        def averages(results: list[dict]) -> dict:
            return {
                metric: _mean([r["metrics"].get(metric) for r in results])
                for metric in SUMMARY_METRICS
            }

        overall = averages(task_results)
        overall["graded_tasks"] = sum(
            r["metrics"].get("answer_correct") is not None for r in task_results
        )
        overall["avg_wall_time_s"] = round(
            sum(r["wall_time_s"] for r in task_results) / len(task_results), 2
        )

        # Of the wrong answers the critic saw, how many did it wave through?
        wrong_and_judged = [
            r for r in task_results
            if r["metrics"].get("answer_correct") == 0.0
            and r["metrics"].get("critic_agreement") is not None
        ]
        overall["critic_false_pass_rate"] = _mean(
            [r["metrics"].get("critic_pass") for r in wrong_and_judged]
        )

        by_category = {}
        for category in sorted({r["category"] for r in task_results}):
            subset = [r for r in task_results if r["category"] == category]
            by_category[category] = {**averages(subset), "count": len(subset)}

        by_difficulty = {}
        for level in sorted({r["difficulty"] for r in task_results}):
            subset = [r for r in task_results if r["difficulty"] == level]
            by_difficulty[str(level)] = {
                "count": len(subset),
                "task_completion": _mean([r["metrics"].get("task_completion") for r in subset]),
                "answer_correct": _mean([r["metrics"].get("answer_correct") for r in subset]),
            }

        return {
            "overall": overall,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }
