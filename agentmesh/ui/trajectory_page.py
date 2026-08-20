# file: agentmesh/ui/trajectory_page.py
"""Trajectory viewer — browse and inspect task execution timelines."""

import json

import requests
import streamlit as st


def render_trajectory_page():
    """Render the trajectory browser and detail viewer."""

    st.title("🔍 Trajectory Viewer")

    api_url = st.session_state.get("api_url", "http://localhost:8000")

    if "selected_trajectory_id" not in st.session_state:
        st.session_state["selected_trajectory_id"] = None
    if "traj_offset" not in st.session_state:
        st.session_state["traj_offset"] = 0

    # ── Mode: Detail view ───────────────────────────────────────
    selected_id = st.session_state["selected_trajectory_id"]
    if selected_id:
        _render_detail_view(api_url, selected_id)
        return

    # ── Mode: Browse ────────────────────────────────────────────
    st.caption("Click any trajectory to inspect its full event timeline.")

    try:
        resp = requests.get(
            f"{api_url}/trajectory/list",
            params={
                "limit": 15,
                "offset": st.session_state["traj_offset"],
            },
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        st.error(f"Could not load trajectories: {e}")
        return

    trajectories = data.get("trajectories", [])
    total = data.get("total", 0)

    if not trajectories:
        st.info("No trajectories yet. Run a chat or eval to generate them.")
        return

    st.caption(f"Showing {len(trajectories)} of {total} trajectories")

    # Render as a table with view buttons
    for traj in trajectories:
        cols = st.columns([1, 3, 1, 1, 1, 1])

        status = "✅" if traj.get("completed") else "❌"
        cols[0].write(status)
        cols[1].write(traj.get("user_task", "")[:60])
        cols[2].write(f"🔧 {traj.get('total_tool_calls', 0)}")
        cols[3].write(f"🪙 {traj.get('total_tokens', 0)}")
        cols[4].write(traj.get("created_at", "")[:10])

        if cols[5].button("View", key=f"view_{traj['task_id']}"):
            st.session_state["selected_trajectory_id"] = traj["task_id"]
            st.rerun()

    # Pagination
    st.divider()
    nav_cols = st.columns(3)
    offset = st.session_state["traj_offset"]

    if nav_cols[0].button("⬅️ Previous", disabled=(offset == 0)):
        st.session_state["traj_offset"] = max(0, offset - 15)
        st.rerun()

    nav_cols[1].caption(f"Page {offset // 15 + 1}")

    if nav_cols[2].button("Next ➡️", disabled=(offset + 15 >= total)):
        st.session_state["traj_offset"] = offset + 15
        st.rerun()

def _render_detail_view(api_url: str, task_id: str):
    """Render the detail view for a single trajectory."""

    # Back button
    if st.button("← Back to list"):
        st.session_state["selected_trajectory_id"] = None
        st.rerun()

    try:
        resp = requests.get(f"{api_url}/trajectory/{task_id}", timeout=10)
        if resp.status_code == 404:
            st.error(f"Trajectory '{task_id}' not found.")
            return
        traj = resp.json()
    except Exception as e:
        st.error(f"Could not load trajectory: {e}")
        return

    # ── Header ──────────────────────────────────────────────────
    status = "✅ Completed" if traj.get("completed") else "❌ Partial"
    st.subheader(f"{status} — {task_id}")
    st.markdown(f"**Task:** {traj.get('user_task', '')}")

    header_cols = st.columns(4)
    header_cols[0].metric("Tool Calls", traj.get("total_tool_calls", 0))
    header_cols[1].metric("Tokens", traj.get("total_tokens", 0))
    header_cols[2].metric("Latency", f"{traj.get('total_latency_ms', 0):.0f}ms")
    header_cols[3].metric("Events", len(traj.get("events", [])))

    # ── Mermaid flowchart ───────────────────────────────────────
    events = traj.get("events", [])
    if events:
        st.divider()
        st.subheader("🔀 Pipeline Flow")

        mermaid_lines = ["graph TD"]
        for i, event in enumerate(events):
            agent = event.get("agent_name", "Unknown").replace(" ", "_")
            action = event.get("action_type", "")
            tool = event.get("tool_name", "")

            # Node label
            if action == "tool_call" and tool:
                label = f"{agent}: {tool}"
            elif action == "plan":
                label = f"{agent}: Plan"
            elif action == "critique":
                label = f"{agent}: Critique"
            elif action == "loop_detected":
                label = f"{agent}: LOOP"
            elif action == "revision":
                label = f"{agent}: Revise"
            else:
                label = f"{agent}: {action}"

            node_id = f"N{i}"
            mermaid_lines.append(f'    {node_id}["{label}"]')

            # Arrow from previous node
            if i > 0:
                mermaid_lines.append(f"    N{i-1} --> {node_id}")

            # Style by action type
            style_map = {
                "plan": "fill:#4CAF50,color:#fff",
                "tool_call": "fill:#2196F3,color:#fff",
                "agent_output": "fill:#8BC34A,color:#000",
                "critique": "fill:#FF9800,color:#fff",
                "loop_detected": "fill:#f44336,color:#fff",
                "revision": "fill:#9C27B0,color:#fff",
                "final": "fill:#607D8B,color:#fff",
            }
            style = style_map.get(action, "fill:#9E9E9E,color:#fff")
            mermaid_lines.append(f"    style {node_id} {style}")

        mermaid_code = "\n".join(mermaid_lines)

        try:
            st.markdown(f"```mermaid\n{mermaid_code}\n```")
        except Exception:
            # Fallback if mermaid rendering fails
            st.code(mermaid_code, language="text")

    # ── Event timeline ──────────────────────────────────────────
    st.divider()
    st.subheader("📜 Event Timeline")

    for i, event in enumerate(events):
        action = event.get("action_type", "")
        agent = event.get("agent_name", "")
        tool = event.get("tool_name", "")
        latency = event.get("latency_ms", 0)

        icons = {
            "plan": "📋",
            "tool_call": "🔧",
            "agent_output": "💡",
            "critique": "🔍",
            "revision": "✏️",
            "loop_detected": "⚠️",
            "final": "✅",
        }
        icon = icons.get(action, "📌")

        label = f"Step {event.get('step_number', i+1)}: {icon} {agent}"
        if tool:
            label += f" → {tool}"
        if latency:
            label += f" ({latency:.0f}ms)"

        with st.expander(label, expanded=(i == 0)):
            detail_cols = st.columns(2)

            with detail_cols[0]:
                st.caption("Input")
                tool_input = event.get("tool_input", "")
                if tool_input:
                    st.code(tool_input[:800], language="json")
                else:
                    st.caption("—")

            with detail_cols[1]:
                st.caption("Output")
                tool_output = event.get("tool_output", "")
                if tool_output:
                    st.code(tool_output[:800], language="text")
                else:
                    st.caption("—")

            if event.get("tokens_in") or event.get("tokens_out"):
                st.caption(
                    f"Tokens: {event.get('tokens_in', 0)} in / "
                    f"{event.get('tokens_out', 0)} out"
                )

    # ── Final output ────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Final Output")
    st.markdown(traj.get("final_output", "*No output*"))