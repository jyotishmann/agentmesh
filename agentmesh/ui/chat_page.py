# file: agentmesh/ui/chat_page.py
"""Chat page — conversational interface with SSE streaming."""

import json
import time
from uuid import uuid4

import requests
import streamlit as st


def render_chat_page():
    """Render the chat interface."""

    st.title("💬 Chat")

    # ── Session state init ──────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{uuid4().hex[:8]}"
    if "last_trajectory" not in st.session_state:
        st.session_state["last_trajectory"] = []
    if "last_token_summary" not in st.session_state:
        st.session_state["last_token_summary"] = {}

    api_url = st.session_state.get("api_url", "http://localhost:8000")

    # ── Layout: chat + trajectory panel ─────────────────────────
    chat_col, traj_col = st.columns([3, 1])

    with chat_col:
        # Display existing messages
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Chat input ──────────────────────────────────────────
        user_input = st.chat_input("Ask AgentMesh anything...")

        if user_input:
            # Add user message
            st.session_state["messages"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Stream response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                status_placeholder = st.empty()
                trajectory_events = []

                try:
                    status_placeholder.caption("🔄 Planning...")

                    # Try SSE streaming first
                    resp = requests.post(
                        f"{api_url}/chat/stream",
                        json={
                            "task": user_input,
                            "session_id": st.session_state["session_id"],
                        },
                        stream=True,
                        timeout=120,
                    )

                    if resp.status_code != 200:
                        raise ConnectionError(f"API returned {resp.status_code}")

                    final_response = ""
                    task_id = ""
                    token_summary = {}

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # strip "data: "
                        if data_str == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "event":
                            agent = event.get("agent", "")
                            action = event.get("action", "")
                            tool = event.get("tool", "")

                            # Update status
                            if action == "plan":
                                status_placeholder.caption("📋 Plan received. Executing...")
                            elif action == "tool_call":
                                status_placeholder.caption(
                                    f"🔧 {agent} → {tool}..."
                                )
                            elif action == "critique":
                                status_placeholder.caption("🔍 Critic evaluating...")
                            elif action == "revision":
                                status_placeholder.caption("✏️ Revising...")
                            elif action == "loop_detected":
                                status_placeholder.caption("⚠️ Loop detected!")

                            trajectory_events.append(event)

                        elif event_type == "final":
                            final_response = event.get("response", "")
                            task_id = event.get("task_id", "")
                            token_summary = event.get("token_summary", {})

                        elif event_type == "error":
                            final_response = f"❌ Error: {event.get('message', 'Unknown')}"

                    status_placeholder.empty()
                    response_placeholder.markdown(final_response)

                    # Store results
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": final_response}
                    )
                    st.session_state["last_trajectory"] = trajectory_events
                    st.session_state["last_token_summary"] = token_summary

                except Exception as e:
                    # Fallback to sync endpoint
                    status_placeholder.caption("⚡ Using sync mode...")

                    try:
                        resp = requests.post(
                            f"{api_url}/chat",
                            json={
                                "task": user_input,
                                "session_id": st.session_state["session_id"],
                            },
                            timeout=120,
                        )
                        data = resp.json()
                        final_response = data.get("response", "Error")
                        status_placeholder.empty()
                        response_placeholder.markdown(final_response)

                        st.session_state["messages"].append(
                            {"role": "assistant", "content": final_response}
                        )
                        st.session_state["last_token_summary"] = data.get(
                            "token_summary", {}
                        )

                    except Exception as e2:
                        status_placeholder.empty()
                        response_placeholder.error(
                            f"Could not reach API at {api_url}. "
                            f"Is the server running?\n\nError: {e2}"
                        )
                        
    # ── Trajectory panel (right column) ─────────────────────────
    with traj_col:
        st.subheader("📜 Trajectory")

        trajectory = st.session_state.get("last_trajectory", [])
        if trajectory:
            for event in trajectory:
                action = event.get("action", "")
                agent = event.get("agent", "")
                tool = event.get("tool", "")
                content = event.get("content", "")
                latency = event.get("latency_ms", 0)

                # Icon by action type
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

                label = f"{icon} {agent}"
                if tool:
                    label += f" → {tool}"
                if latency:
                    label += f" ({latency:.0f}ms)"

                with st.expander(label, expanded=False):
                    if content:
                        st.code(content[:500], language="text")
                    else:
                        st.caption("No output")

        else:
            st.caption("Send a message to see the trajectory.")

        # Token summary
        st.divider()
        st.subheader("📊 Tokens")
        token_summary = st.session_state.get("last_token_summary", {})
        if token_summary:
            cols = st.columns(2)
            with cols[0]:
                st.metric("In", token_summary.get("total_tokens_in", 0))
            with cols[1]:
                st.metric("Out", token_summary.get("total_tokens_out", 0))
            st.metric(
                "Tool Calls",
                token_summary.get("total_tool_calls", 0),
            )
        else:
            st.caption("No data yet.")

        # Session controls
        st.divider()
        if st.button("🔄 Reset Session"):
            st.session_state["messages"] = []
            st.session_state["session_id"] = f"session_{uuid4().hex[:8]}"
            st.session_state["last_trajectory"] = []
            st.session_state["last_token_summary"] = {}
            st.rerun()