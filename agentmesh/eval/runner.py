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


def _load_tasks() -> list[dict]:
    """Load eval tasks from JSON file."""
    with open(TASKS_PATH) as f:
        return json.load(f)


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

    def run_single(self, task_id: str) -> dict:
        """Run a single eval task and return its metrics.

        Args:
            task_id: The task_id to run (e.g. 'factual_001').

        Returns:
            Dict with task_id, metrics, and trajectory_id.

        Raises:
            ValueError: If task_id not found.
        """
        task_def = None
        for t in self.tasks:
            if t["task_id"] == task_id:
                task_def = t
                break

        if task_def is None:
            raise ValueError(f"Task '{task_id}' not found in tasks.json")

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
                "trajectory_id": result.task_id,
                "completed": result.completed,
                "metrics": metrics,
                "wall_time_s": round(time.time() - start, 2),
                "error": None,
            }

        except Exception as e:
            logger.error(f"[EVAL] Task {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "category": task_def["category"],
                "difficulty": task_def["difficulty"],
                "trajectory_id": None,
                "completed": False,
                "metrics": {name: 0.0 for name in ALL_METRICS},
                "wall_time_s": round(time.time() - start, 2),
                "error": str(e),
            }

    def run_all(
        self,
        categories: Optional[list[str]] = None,
        max_difficulty: int = 5,
    ) -> dict:
        """Run all eval tasks (optionally filtered) and save results.

        Args:
            categories: Optional list of categories to include.
            max_difficulty: Only run tasks at or below this difficulty.

        Returns:
            Dict with per-task results and aggregate summaries.
        """
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
        for i, task_def in enumerate(filtered):
            logger.info(f"[EVAL] [{i+1}/{len(filtered)}] {task_def['task_id']}")
            result = self.run_single(task_def["task_id"])
            task_results.append(result)

        # Compute aggregates
        summary = self._compute_summary(task_results)

        output = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tasks": len(task_results),
            "summary": summary,
            "task_results": task_results,
        }

        # Save to file
        filename = f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"[EVAL] Results saved to {output_path}")
        return output

    @staticmethod
    def _compute_summary(task_results: list[dict]) -> dict:
        """Compute aggregate metrics across task results.

        Returns:
            Dict with overall and per-category averages.
        """
        if not task_results:
            return {"overall": {}, "by_category": {}, "by_difficulty": {}}

        # Overall averages
        metric_names = [
            "task_completion", "tool_call_efficiency",
            "loop_detected", "critic_pass",
        ]
        overall = {}
        for metric in metric_names:
            values = [
                r["metrics"][metric]
                for r in task_results
                if metric in r["metrics"]
            ]
            overall[metric] = round(sum(values) / len(values), 3) if values else 0.0

        # Average wall time
        wall_times = [r["wall_time_s"] for r in task_results]
        overall["avg_wall_time_s"] = round(
            sum(wall_times) / len(wall_times), 2
        ) if wall_times else 0.0

        # By category
        by_category = {}
        categories = set(r["category"] for r in task_results)
        for cat in categories:
            cat_results = [r for r in task_results if r["category"] == cat]
            by_category[cat] = {}
            for metric in metric_names:
                values = [
                    r["metrics"][metric]
                    for r in cat_results
                    if metric in r["metrics"]
                ]
                by_category[cat][metric] = (
                    round(sum(values) / len(values), 3) if values else 0.0
                )
            by_category[cat]["count"] = len(cat_results)

        # By difficulty
        by_difficulty = {}
        for diff in sorted(set(r["difficulty"] for r in task_results)):
            diff_results = [r for r in task_results if r["difficulty"] == diff]
            by_difficulty[str(diff)] = {
                "count": len(diff_results),
                "task_completion": round(
                    sum(r["metrics"].get("task_completion", 0) for r in diff_results)
                    / len(diff_results), 3
                ),
            }

        return {
            "overall": overall,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }