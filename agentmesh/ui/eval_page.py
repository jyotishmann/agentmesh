# file: agentmesh/ui/eval_page.py
"""Eval dashboard — run evals, view results, compare runs."""

import json

import requests
import streamlit as st


def render_eval_page():
    """Render the eval dashboard."""

    st.title("📊 Eval Dashboard")

    api_url = st.session_state.get("api_url", "http://localhost:8000")

    tab_run, tab_results, tab_compare = st.tabs([
        "🚀 Run Eval", "📋 Results", "⚖️ Compare"
    ])

    # ── Tab 1: Run Eval ─────────────────────────────────────────
    with tab_run:
        st.subheader("Run Evaluation")
        st.caption(
            "Execute eval tasks through the orchestrator and compute metrics. "
            "This may take 10-30 minutes for the full suite."
        )

        run_cols = st.columns(3)

        with run_cols[0]:
            categories = st.multiselect(
                "Categories",
                ["factual_qa", "code_generation", "multi_step", "analysis", "creative"],
                default=None,
                help="Leave empty to run all categories.",
            )

        with run_cols[1]:
            max_difficulty = st.slider(
                "Max Difficulty", min_value=1, max_value=5, value=5
            )

        with run_cols[2]:
            single_task = st.text_input(
                "Single Task ID",
                placeholder="e.g. factual_001",
                help="Run a single task. Overrides filters above.",
            )

        if st.button("▶️ Run Eval", type="primary"):
            with st.spinner("Running evaluation... This may take a while."):
                try:
                    payload = {
                        "max_difficulty": max_difficulty,
                    }
                    if single_task:
                        payload["task_id"] = single_task
                    elif categories:
                        payload["categories"] = categories

                    resp = requests.post(
                        f"{api_url}/eval/run",
                        json=payload,
                        timeout=3600,  # 1 hour max
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("Eval run completed!")

                        if "result" in data:
                            # Single task result
                            st.json(data["result"])
                        elif "results" in data:
                            # Full run results
                            _render_results(data["results"])
                    else:
                        st.error(f"Eval failed: {resp.text}")

                except requests.exceptions.Timeout:
                    st.error("Eval run timed out. Check server logs.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Results ──────────────────────────────────────────
    with tab_results:
        st.subheader("Latest Results")

        try:
            resp = requests.get(
                f"{api_url}/eval/results",
                params={"latest": True},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            st.error(f"Could not load results: {e}")
            return

        files = data.get("files", [])
        latest = data.get("latest")

        if not files:
            st.info("No eval results yet. Run an evaluation first.")
            return

        st.caption(f"{len(files)} eval run(s) saved")

        if latest:
            _render_results(latest)
        else:
            st.info("Select an eval run to view.")

        # File list
        st.divider()
        st.caption("Saved runs:")
        for f in files[:10]:
            st.code(f, language="text")

    # ── Tab 3: Compare ──────────────────────────────────────────
    with tab_compare:
        st.subheader("Compare Eval Runs")

        try:
            resp = requests.get(
                f"{api_url}/eval/results",
                params={"latest": False},
                timeout=10,
            )
            data = resp.json()
            files = data.get("files", [])
        except Exception as e:
            st.error(f"Could not load result files: {e}")
            return

        if len(files) < 2:
            st.info("Need at least 2 eval runs to compare.")
            return

        compare_cols = st.columns(2)

        with compare_cols[0]:
            run_a = st.selectbox("Run A (baseline)", files, index=1)
        with compare_cols[1]:
            run_b = st.selectbox("Run B (current)", files, index=0)

        if run_a == run_b:
            st.warning("Select two different runs to compare.")
            return

        if st.button("⚖️ Compare"):
            try:
                resp = requests.get(
                    f"{api_url}/eval/compare",
                    params={"run_a": run_a, "run_b": run_b},
                    timeout=10,
                )
                comparison = resp.json()
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                return

            data_a = comparison["run_a"]["data"]
            data_b = comparison["run_b"]["data"]

            summary_a = data_a.get("summary", {}).get("overall", {})
            summary_b = data_b.get("summary", {}).get("overall", {})

            # Side-by-side metrics
            st.markdown("### Overall Comparison")
            metric_names = [
                ("task_completion", "Completion"),
                ("tool_call_efficiency", "Tool Efficiency"),
                ("loop_detected", "No Loops"),
                ("critic_pass", "Critic Pass"),
            ]

            m_cols = st.columns(len(metric_names))
            for i, (key, label) in enumerate(metric_names):
                val_a = summary_a.get(key, 0)
                val_b = summary_b.get(key, 0)
                delta = val_b - val_a

                with m_cols[i]:
                    st.metric(
                        label,
                        f"{val_b * 100:.1f}%",
                        delta=f"{delta * 100:+.1f}%",
                        delta_color="normal",
                    )

            # Detailed comparison
            st.divider()
            detail_cols = st.columns(2)

            with detail_cols[0]:
                st.markdown(f"**Run A: {run_a}**")
                st.caption(f"Tasks: {data_a.get('total_tasks', 0)}")
                st.caption(f"Time: {data_a.get('run_timestamp', '?')}")

            with detail_cols[1]:
                st.markdown(f"**Run B: {run_b}**")
                st.caption(f"Tasks: {data_b.get('total_tasks', 0)}")
                st.caption(f"Time: {data_b.get('run_timestamp', '?')}")


def _render_results(results: dict):
    """Render eval results with charts and tables."""

    summary = results.get("summary", {})
    overall = summary.get("overall", {})
    by_category = summary.get("by_category", {})
    by_difficulty = summary.get("by_difficulty", {})
    task_results = results.get("task_results", [])

    # ── Overall metrics ─────────────────────────────────────────
    st.markdown("### Overall")
    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Completion",
        f"{overall.get('task_completion', 0) * 100:.1f}%",
    )
    metric_cols[1].metric(
        "Tool Efficiency",
        f"{overall.get('tool_call_efficiency', 0) * 100:.1f}%",
    )
    metric_cols[2].metric(
        "No Loops",
        f"{overall.get('loop_detected', 0) * 100:.1f}%",
    )
    metric_cols[3].metric(
        "Critic Pass",
        f"{overall.get('critic_pass', 0) * 100:.1f}%",
    )
    metric_cols[4].metric(
        "Avg Time",
        f"{overall.get('avg_wall_time_s', 0):.1f}s",
    )

    # ── By Category ─────────────────────────────────────────────
    if by_category:
        st.markdown("### By Category")

        chart_data = {
            "Category": [],
            "Completion %": [],
            "Tool Efficiency %": [],
            "Critic Pass %": [],
        }

        for cat, metrics in sorted(by_category.items()):
            chart_data["Category"].append(cat)
            chart_data["Completion %"].append(
                metrics.get("task_completion", 0) * 100
            )
            chart_data["Tool Efficiency %"].append(
                metrics.get("tool_call_efficiency", 0) * 100
            )
            chart_data["Critic Pass %"].append(
                metrics.get("critic_pass", 0) * 100
            )

        st.bar_chart(
            data={
                k: v for k, v in chart_data.items() if k != "Category"
            },
        )

        # Category table
        cat_cols = st.columns(len(by_category))
        for i, (cat, metrics) in enumerate(sorted(by_category.items())):
            with cat_cols[i]:
                st.caption(cat)
                st.write(f"n={metrics.get('count', 0)}")
                st.write(
                    f"✅ {metrics.get('task_completion', 0)*100:.0f}%"
                )

    # ── By Difficulty ───────────────────────────────────────────
    if by_difficulty:
        st.markdown("### By Difficulty")

        diff_data = {"Difficulty": [], "Completion %": [], "Count": []}
        for diff, metrics in sorted(by_difficulty.items()):
            diff_data["Difficulty"].append(int(diff))
            diff_data["Completion %"].append(
                metrics.get("task_completion", 0) * 100
            )
            diff_data["Count"].append(metrics.get("count", 0))

        st.line_chart(
            data={"Completion %": diff_data["Completion %"]},
        )

    # ── Per-Task Table ──────────────────────────────────────────
    if task_results:
        st.markdown("### Per-Task Results")

        for r in task_results:
            status = "✅" if r.get("completed") else "❌"
            error = f" — {r['error']}" if r.get("error") else ""

            metrics = r.get("metrics", {})
            tool_eff = metrics.get("tool_call_efficiency", 0)
            critic = metrics.get("critic_pass", 0)

            st.markdown(
                f"{status} **{r['task_id']}** "
                f"(d={r.get('difficulty', '?')}) "
                f"— tool_eff={tool_eff:.2f}, critic={'✅' if critic else '❌'} "
                f"— {r.get('wall_time_s', 0):.1f}s"
                f"{error}"
            )
