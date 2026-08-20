# file: agentmesh/eval/cli.py
"""CLI for the eval framework.

Usage:
    python -m agentmesh.eval.cli run [--task-id ID] [--category CAT] [--max-difficulty N]
    python -m agentmesh.eval.cli report [--file PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from agentmesh.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(settings.db_path).parent / "eval_results"


def cmd_run(args: argparse.Namespace) -> None:
    """Execute eval tasks."""
    # Import here to avoid loading models when just viewing reports
    from agentmesh.eval.runner import EvalRunner

    runner = EvalRunner()

    if args.task_id:
        result = runner.run_single(args.task_id)
        print(json.dumps(result, indent=2))
    else:
        categories = [args.category] if args.category else None
        results = runner.run_all(
            categories=categories,
            max_difficulty=args.max_difficulty,
        )
        _print_summary(results["summary"])


def cmd_report(args: argparse.Namespace) -> None:
    """Print a report from saved results."""
    if args.file:
        results_path = Path(args.file)
    else:
        # Find the most recent results file
        results_files = sorted(RESULTS_DIR.glob("eval_*.json"))
        if not results_files:
            print("No eval results found. Run 'python -m agentmesh.eval.cli run' first.")
            sys.exit(1)
        results_path = results_files[-1]

    with open(results_path) as f:
        results = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Eval Report: {results_path.name}")
    print(f"  Timestamp: {results.get('run_timestamp', 'unknown')}")
    print(f"  Total tasks: {results.get('total_tasks', 0)}")
    print(f"{'='*60}\n")

    _print_summary(results.get("summary", {}))

    # Print per-task details
    print(f"\n{'─'*60}")
    print("  Per-Task Results")
    print(f"{'─'*60}")

    for r in results.get("task_results", []):
        status = "PASS" if r.get("completed") else "FAIL"
        error = f" ERROR: {r['error']}" if r.get("error") else ""
        metrics_str = ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.get("metrics", {}).items()
            if k != "latency_ms"
        )
        print(f"  [{status}] {r['task_id']:<20} {metrics_str}{error}")


def _print_summary(summary: dict) -> None:
    """Print formatted summary tables."""
    overall = summary.get("overall", {})
    if overall:
        print("  Overall Metrics:")
        for k, v in overall.items():
            print(f"    {k:<25} {v}")
        print()

    by_cat = summary.get("by_category", {})
    if by_cat:
        print("  By Category:")
        header = f"    {'Category':<20} {'Completion':>10} {'Tool Eff':>10} {'No Loops':>10} {'Critic':>10} {'Count':>6}"
        print(header)
        print(f"    {'─'*66}")
        for cat, metrics in sorted(by_cat.items()):
            print(
                f"    {cat:<20} "
                f"{metrics.get('task_completion', 0):>10.3f} "
                f"{metrics.get('tool_call_efficiency', 0):>10.3f} "
                f"{metrics.get('loop_detected', 0):>10.3f} "
                f"{metrics.get('critic_pass', 0):>10.3f} "
                f"{metrics.get('count', 0):>6}"
            )
        print()

    by_diff = summary.get("by_difficulty", {})
    if by_diff:
        print("  By Difficulty:")
        for diff, metrics in sorted(by_diff.items()):
            print(
                f"    Level {diff}: "
                f"completion={metrics.get('task_completion', 0):.3f} "
                f"(n={metrics.get('count', 0)})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentMesh Eval Framework CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run eval tasks")
    run_parser.add_argument("--task-id", type=str, help="Run a single task by ID")
    run_parser.add_argument("--category", type=str, help="Filter by category")
    run_parser.add_argument(
        "--max-difficulty", type=int, default=5,
        help="Max difficulty level (1-5, default=5)"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Print eval report")
    report_parser.add_argument("--file", type=str, help="Path to results JSON")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()