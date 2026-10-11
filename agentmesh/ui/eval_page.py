# file: agentmesh/ui/eval_page.py
"""Eval dashboard — browse saved runs, compare them, and start new ones.

Correctness comes first: answer accuracy, critic agreement and the critic's
false-pass rate lead every view. Completion and tool metrics are secondary,
because a run can "complete" with a wrong answer.
"""

import pandas as pd
import requests
import streamlit as st

CATEGORIES = ["factual_qa", "code_generation", "multi_step", "analysis", "creative"]


# ── Helpers ─────────────────────────────────────────────────────────


def _pct(value: float | None) -> str:
    """0-1 metric as a percentage; None (metric doesn't apply) as n/a."""
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _delta_pp(before: float | None, after: float | None) -> str | None:
    """Change in percentage points, or None when either side is missing."""
    if before is None or after is None:
        return None
    return f"{(after - before) * 100:+.0f} pp"


def _get(api_url: str, path: str, params: dict | None = None) -> dict | None:
    """GET a JSON endpoint, showing a readable error instead of a traceback."""
    try:
        resp = requests.get(f"{api_url}{path}", params=params, timeout=15)
        if resp.status_code != 200:
            st.error(f"API {path} returned {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {api_url}: {e}")
        return None


def _verdict(value: float | None, yes: str, no: str) -> str:
    return {1.0: yes, 0.0: no}.get(value, "–")


def _category_table(task_results: list[dict]) -> pd.DataFrame:
    """Per-category accuracy computed from task results, with graded counts."""
    rows = []
    for category in CATEGORIES:
        subset = [r for r in task_results if r["category"] == category]
        if not subset:
            continue
        graded = [
            r["metrics"].get("answer_correct")
            for r in subset
            if r["metrics"].get("answer_correct") is not None
        ]
        agreement = [
            r["metrics"].get("critic_agreement")
            for r in subset
            if r["metrics"].get("critic_agreement") is not None
        ]
        rows.append(
            {
                "Category": category,
                "Correct %": round(100 * sum(graded) / len(graded)) if graded else None,
                "Completion %": round(
                    100
                    * sum(r["metrics"].get("task_completion", 0) for r in subset)
                    / len(subset)
                ),
                "Critic agreement %": round(100 * sum(agreement) / len(agreement))
                if agreement
                else None,
                "Graded": f"{int(sum(graded))}/{len(graded)}",
                "Tasks": len(subset),
            }
        )
    return pd.DataFrame(rows).set_index("Category")


def _task_table(task_results: list[dict]) -> pd.DataFrame:
    """One row per task, verdicts spelled out."""
    rows = []
    for r in task_results:
        m = r.get("metrics", {})
        tool_eff = m.get("tool_call_efficiency")
        rows.append(
            {
                "Task": r["task_id"],
                "Category": r["category"],
                "Diff": r.get("difficulty"),
                "Answer": _verdict(m.get("answer_correct"), "✓ correct", "✗ wrong"),
                "Critic": _verdict(m.get("critic_pass"), "pass", "reject"),
                "Critic right?": _verdict(m.get("critic_agreement"), "✓", "✗"),
                "Completed": "yes" if r.get("completed") else "no",
                "Tool eff": None if tool_eff is None else round(tool_eff, 2),
                "Time (s)": r.get("wall_time_s"),
                "Error": r.get("error") or "",
            }
        )
    return pd.DataFrame(rows)


# ── Views ───────────────────────────────────────────────────────────


def _render_results(results: dict, name: str = "") -> None:
    """Render one run: headline correctness, secondary metrics, categories, tasks."""
    overall = results.get("summary", {}).get("overall", {})
    task_results = results.get("task_results", [])
    graded = overall.get("graded_tasks", 0)

    # Run metadata: what was measured, how, and on what
    total = results.get("total_tasks", len(task_results))
    planned = results.get("planned_tasks", len(task_results))
    meta = [f"{total} of {planned} tasks"]
    if results.get("deterministic") is not None:
        meta.append("deterministic" if results["deterministic"] else "sampled")
    if not results.get("finished", True):
        meta.append("⚠️ incomplete run")
    if results.get("regraded_from"):
        meta.append(f"regraded from {results['regraded_from']}")
    env = results.get("environment") or {}
    if env:
        meta.append(
            f"torch {env.get('torch')} · transformers {env.get('transformers')} · {env.get('gpu') or 'CPU'}"
        )
    st.caption(" · ".join(meta))

    # Headline: correctness and how far the critic can be trusted
    st.markdown("### Correctness")
    c = st.columns(4)
    c[0].metric(
        "Answer accuracy",
        _pct(overall.get("answer_correct")),
        help=f"Share of the {graded} gradable tasks whose final answer matches the expected answer.",
    )
    c[1].metric(
        "Critic agreement",
        _pct(overall.get("critic_agreement")),
        help="How often the critic's pass/reject verdict matches ground truth.",
    )
    c[2].metric(
        "Critic false-pass",
        _pct(overall.get("critic_false_pass_rate")),
        help="Of the wrong answers the critic judged, the share it approved. Lower is better.",
    )
    c[3].metric(
        "Task completion",
        _pct(overall.get("task_completion")),
        help="Runs that finished without hitting limits or a final critic rejection. "
        "Not the same as correct.",
    )
    st.caption(
        f"Accuracy is measured over {graded} tasks with a fixed answer; "
        f"the other {len(task_results) - graded} (random, live or creative output) are excluded."
    )

    st.markdown("### Execution")
    e = st.columns(3)
    e[0].metric(
        "Tool efficiency",
        _pct(overall.get("tool_call_efficiency")),
        help="Overlap between the tools a task should use and the tools actually used.",
    )
    e[1].metric(
        "No loops",
        _pct(overall.get("loop_detected")),
        help="Runs where no identical tool call was repeated.",
    )
    e[2].metric("Avg time / task", f"{overall.get('avg_wall_time_s', 0):.1f}s")

    if not task_results:
        return

    st.markdown("### By category")
    categories = _category_table(task_results)
    # Horizontal bars: category names sit on the y-axis, unrotated and untruncated
    st.bar_chart(
        categories[["Correct %", "Completion %", "Critic agreement %"]],
        horizontal=True,
        stack=False,
        x_label="",
        y_label="%",
        height=360,
    )
    st.dataframe(categories, use_container_width=True)
    st.caption(
        "Categories have few graded tasks each, so a single task moves a category by "
        "10–50 points. Compare categories with care; the overall accuracy is the robust number."
    )

    st.markdown("### Tasks")
    view = st.radio(
        "Show",
        ["All tasks", "Wrong answers", "Critic was wrong"],
        horizontal=True,
        key=f"task_filter_{name}",
    )
    tasks = _task_table(task_results)
    if view == "Wrong answers":
        tasks = tasks[tasks["Answer"] == "✗ wrong"]
    elif view == "Critic was wrong":
        tasks = tasks[tasks["Critic right?"] == "✗"]
    st.dataframe(tasks, hide_index=True, use_container_width=True)


def _render_comparison(name_a: str, data_a: dict, name_b: str, data_b: dict) -> None:
    """Headline deltas between two runs, then every task whose verdict changed."""
    a = data_a.get("summary", {}).get("overall", {})
    b = data_b.get("summary", {}).get("overall", {})

    st.markdown("### What changed")
    cols = st.columns(4)
    for col, (key, label, colour) in zip(
        cols,
        [
            ("answer_correct", "Answer accuracy", "normal"),
            ("critic_agreement", "Critic agreement", "normal"),
            # Lower is better, so a rise shows red
            ("critic_false_pass_rate", "Critic false-pass", "inverse"),
            ("task_completion", "Task completion", "normal"),
        ],
    ):
        col.metric(
            label,
            _pct(b.get(key)),
            delta=_delta_pp(a.get(key), b.get(key)),
            delta_color=colour,
        )
    st.caption(f"Run A: {name_a} → Run B: {name_b}")

    before = {
        r["task_id"]: r.get("metrics", {}).get("answer_correct")
        for r in data_a.get("task_results", [])
    }
    changes = [
        {
            "Task": r["task_id"],
            "Category": r["category"],
            "Run A": _verdict(before.get(r["task_id"]), "✓ correct", "✗ wrong"),
            "Run B": _verdict(
                r.get("metrics", {}).get("answer_correct"), "✓ correct", "✗ wrong"
            ),
        }
        for r in data_b.get("task_results", [])
        if r["task_id"] in before
        and before[r["task_id"]] != r.get("metrics", {}).get("answer_correct")
    ]

    st.markdown(f"### Answer verdicts that changed ({len(changes)})")
    if changes:
        st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)
    else:
        st.success("No answer verdicts changed between these runs.")


def _render_run_form(api_url: str) -> None:
    """Start an eval through the API (best for single tasks or small batches)."""
    st.info(
        "For the full 50-task suite, use the CLI (`python -m agentmesh.eval.cli run`): "
        "it saves after every task and can resume. Runs started here block until done."
    )

    cols = st.columns(3)
    categories = cols[0].multiselect(
        "Categories",
        CATEGORIES,
        default=None,
        help="Leave empty to run all categories.",
    )
    max_difficulty = cols[1].slider("Max difficulty", min_value=1, max_value=5, value=5)
    single_task = cols[2].text_input(
        "Single task ID",
        placeholder="e.g. factual_001",
        help="Run one task. Overrides the filters.",
    )

    if st.button("▶️ Run eval", type="primary"):
        payload = {"max_difficulty": max_difficulty}
        if single_task:
            payload["task_id"] = single_task
        elif categories:
            payload["categories"] = categories

        with st.spinner("Running evaluation..."):
            try:
                resp = requests.post(f"{api_url}/eval/run", json=payload, timeout=3600)
            except requests.exceptions.Timeout:
                st.error("Eval run timed out. Check the server log.")
                return
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                return

        if resp.status_code != 200:
            st.error(f"Eval failed: {resp.text[:300]}")
            return
        data = resp.json()
        st.success("Eval run completed.")
        if "result" in data:
            st.json(data["result"])
        elif "results" in data:
            _render_results(data["results"], name="new_run")


# ── Page ────────────────────────────────────────────────────────────


def render_eval_page():
    """Render the eval dashboard."""
    st.title("📊 Eval Dashboard")
    api_url = st.session_state.get("api_url", "http://localhost:8000")

    tab_results, tab_compare, tab_run = st.tabs(
        ["📋 Results", "⚖️ Compare", "🚀 Run Eval"]
    )

    listing = _get(api_url, "/eval/results", params={"latest": False})
    files = (listing or {}).get("files", [])

    with tab_results:
        if not files:
            st.info(
                "No saved eval runs yet. Run the suite, or copy saved runs into data/eval_results/."
            )
        else:
            name = st.selectbox(
                "Run",
                files,
                index=0,
                help="Newest first. *_regraded files are re-scored with the current grader.",
            )
            data = _get(api_url, f"/eval/results/{name}")
            if data:
                _render_results(data, name=name)

    with tab_compare:
        if len(files) < 2:
            st.info("Save at least two runs to compare them.")
        else:
            cols = st.columns(2)
            name_a = cols[0].selectbox(
                "Run A (baseline)", files, index=1, key="compare_a"
            )
            name_b = cols[1].selectbox(
                "Run B (compare)", files, index=0, key="compare_b"
            )
            if name_a == name_b:
                st.warning("Pick two different runs.")
            else:
                comparison = _get(
                    api_url, "/eval/compare", params={"run_a": name_a, "run_b": name_b}
                )
                if comparison:
                    _render_comparison(
                        name_a,
                        comparison["run_a"]["data"],
                        name_b,
                        comparison["run_b"]["data"],
                    )

    with tab_run:
        _render_run_form(api_url)
