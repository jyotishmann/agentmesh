# file: agentmesh/eval/cli.py
"""CLI for the eval framework.

Usage:
    python -m agentmesh.eval.cli run [--task-id ID] [--category CAT] [--max-difficulty N]
                                     [--resume PATH] [--sampled]
    python -m agentmesh.eval.cli report [--file PATH]
    python -m agentmesh.eval.cli history
    python -m agentmesh.eval.cli regrade FILE [--db PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from agentmesh.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(settings.db_path).parent / "eval_results"


def _pct(value: Optional[float]) -> str:
    """Format a 0-1 metric as a percentage; None (not applicable) as n/a."""
    return "  n/a" if value is None else f"{value * 100:4.0f}%"


def collect_runs(dirs: list[Path]) -> list[tuple[Path, dict]]:
    """Load every eval results file found in dirs, one entry per run.

    The same run can exist in two places (local results and the mirror).
    Runs are keyed by file name, and the copy with more finished tasks wins.
    """
    best: dict[str, tuple[Path, dict]] = {}
    for directory in dirs:
        if not directory or not Path(directory).is_dir():
            continue
        for path in sorted(Path(directory).glob("eval_*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Skipping unreadable results file: {path}")
                continue
            current = best.get(path.name)
            if current is None or len(data.get("task_results", [])) > len(current[1].get("task_results", [])):
                best[path.name] = (path, data)
    return [best[name] for name in sorted(best)]


def cmd_run(args: argparse.Namespace) -> None:
    """Execute eval tasks."""
    # Import here so `report` and `history` don't load any models
    from agentmesh.eval.runner import EvalRunner

    runner = EvalRunner(deterministic=not args.sampled)

    if args.task_id:
        print(json.dumps(runner.run_single(args.task_id), indent=2))
        return

    results = runner.run_all(
        categories=[args.category] if args.category else None,
        max_difficulty=args.max_difficulty,
        resume=args.resume,
    )
    _print_summary(results["summary"])


def cmd_report(args: argparse.Namespace) -> None:
    """Print a report for one saved run (the latest by default)."""
    if args.file:
        path = Path(args.file)
        results = json.loads(path.read_text())
    else:
        runs = collect_runs([RESULTS_DIR, settings.eval_mirror_dir])
        if not runs:
            print("No eval results found. Run 'python -m agentmesh.eval.cli run' first.")
            sys.exit(1)
        path, results = runs[-1]

    status = "finished" if results.get("finished", True) else "INCOMPLETE (resume with --resume)"
    print(f"\n{'=' * 64}")
    print(f"  Eval report: {path.name}  [{status}]")
    print(f"  Tasks: {results.get('total_tasks', 0)} of {results.get('planned_tasks', '?')}"
          f" | deterministic: {results.get('deterministic', 'unknown')}")
    env = results.get("environment")
    if env:
        print(f"  Env: torch {env.get('torch')} | transformers {env.get('transformers')} | GPU {env.get('gpu')}")
    if results.get("regraded_from"):
        print(f"  Regraded from {results['regraded_from']} at {results.get('regraded_at')}")
    print(f"{'=' * 64}\n")

    _print_summary(results.get("summary", {}))

    print(f"\n  {'Task':<16} {'Answer':<9} {'Critic':<7} {'Tool eff':>8} {'Time':>8}")
    print(f"  {'-' * 52}")
    for r in results.get("task_results", []):
        m = r.get("metrics", {})
        answer = {1.0: "correct", 0.0: "WRONG"}.get(m.get("answer_correct"), "-")
        critic = {1.0: "pass", 0.0: "reject"}.get(m.get("critic_pass"), "-")
        tool_eff = m.get("tool_call_efficiency")
        tool_str = "-" if tool_eff is None else f"{tool_eff:.2f}"
        error = f"  ERROR: {r['error']}" if r.get("error") else ""
        print(f"  {r['task_id']:<16} {answer:<9} {critic:<7} {tool_str:>8} "
              f"{r.get('wall_time_s', 0):>7.1f}s{error}")


def cmd_regrade(args: argparse.Namespace) -> None:
    """Re-score a saved run with the current grader. No GPU, no model calls."""
    from agentmesh.eval.runner import regrade

    results, changes = regrade(args.file, db_path=args.db)
    marks = {1.0: "correct", 0.0: "WRONG", None: "-"}

    print(f"\n  Regraded {Path(args.file).name}: {len(changes)} verdict(s) changed")
    for change in changes:
        print(f"    {change['task_id']:<16} {marks[change['before']]:>7} -> {marks[change['after']]}")
    missing = results.get("regrade_missing", [])
    if missing:
        print(f"  Kept original metrics for {len(missing)} task(s) with no trajectory: {', '.join(missing)}")
    print()
    _print_summary(results["summary"])


def cmd_history(args: argparse.Namespace) -> None:
    """Print one line per saved run, oldest first, for regression tracking."""
    runs = collect_runs([RESULTS_DIR, settings.eval_mirror_dir])
    if not runs:
        print("No eval results found.")
        return

    print(f"\n  {'Run':<36} {'Tasks':>7} {'Det':>4} {'Complete':>9} {'Correct':>8} "
          f"{'Graded':>7} {'Critic agr':>11} {'False pass':>11} {'Avg time':>9}")
    print(f"  {'-' * 109}")
    for path, data in runs:
        o = data.get("summary", {}).get("overall", {})
        tasks = f"{data.get('total_tasks', 0)}/{data.get('planned_tasks', data.get('total_tasks', 0))}"
        det = {True: "yes", False: "no"}.get(data.get("deterministic"), "?")
        flag = "" if data.get("finished", True) else "  (incomplete)"
        print(f"  {path.name:<36} {tasks:>7} {det:>4} {_pct(o.get('task_completion')):>9} "
              f"{_pct(o.get('answer_correct')):>8} {o.get('graded_tasks', '-'):>7} "
              f"{_pct(o.get('critic_agreement')):>11} {_pct(o.get('critic_false_pass_rate')):>11} "
              f"{o.get('avg_wall_time_s', 0):>8.1f}s{flag}")


def _print_summary(summary: dict) -> None:
    """Print the headline metrics, then per-category accuracy."""
    o = summary.get("overall", {})
    if o:
        print("  Overall")
        print(f"    Answer correct     {_pct(o.get('answer_correct'))}  (over {o.get('graded_tasks', 0)} gradable tasks)")
        print(f"    Task completion    {_pct(o.get('task_completion'))}")
        print(f"    Critic agreement   {_pct(o.get('critic_agreement'))}")
        print(f"    Critic false-pass  {_pct(o.get('critic_false_pass_rate'))}")
        print(f"    Tool efficiency    {_pct(o.get('tool_call_efficiency'))}")
        print(f"    No loops           {_pct(o.get('loop_detected'))}")
        print(f"    Avg time / task    {o.get('avg_wall_time_s', 0):.1f}s\n")

    by_cat = summary.get("by_category", {})
    if by_cat:
        print(f"  {'Category':<18} {'Correct':>8} {'Complete':>9} {'Critic agr':>11} {'Count':>6}")
        for category, m in sorted(by_cat.items()):
            print(f"  {category:<18} {_pct(m.get('answer_correct')):>8} "
                  f"{_pct(m.get('task_completion')):>9} {_pct(m.get('critic_agreement')):>11} "
                  f"{m.get('count', 0):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMesh Eval Framework CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run eval tasks")
    run_parser.add_argument("--task-id", type=str, help="Run a single task by ID")
    run_parser.add_argument("--category", type=str, help="Filter by category")
    run_parser.add_argument("--max-difficulty", type=int, default=5,
                            help="Max difficulty level (1-5, default=5)")
    run_parser.add_argument("--resume", type=str,
                            help="Continue an interrupted run from its results file")
    run_parser.add_argument("--sampled", action="store_true",
                            help="Sample instead of greedy decoding (results will vary)")

    report_parser = subparsers.add_parser("report", help="Report on one saved run")
    report_parser.add_argument("--file", type=str, help="Path to a results JSON file")

    subparsers.add_parser("history", help="One line per saved run")

    regrade_parser = subparsers.add_parser("regrade", help="Re-score a saved run with the current grader")
    regrade_parser.add_argument("file", type=str, help="Path to a results JSON file")
    regrade_parser.add_argument("--db", type=str,
                                help="Trajectory database to read answers from (default: data/trajectories.db)")

    args = parser.parse_args()
    commands = {"run": cmd_run, "report": cmd_report, "history": cmd_history, "regrade": cmd_regrade}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
