# file: agentmesh/ui/app.py
"""AgentMesh Streamlit UI — multi-page app shell."""

import streamlit as st


def main():
    st.set_page_config(
        page_title="AgentMesh",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.title("🤖 AgentMesh")
        st.caption("Multi-Agent System with Evals")
        st.divider()

        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Trajectories", "📊 Eval Dashboard"],
            label_visibility="collapsed",
        )

        st.divider()

        # API configuration
        api_url = st.text_input(
            "API URL",
            value=st.session_state.get("api_url", "http://localhost:8000"),
            help="Base URL of the AgentMesh API server.",
        )
        st.session_state["api_url"] = api_url

        # Health check
        if st.button("🩺 Check API"):
            import requests
            try:
                resp = requests.get(f"{api_url}/health", timeout=5)
                data = resp.json()
                if data.get("model_loaded"):
                    st.success("API is running. Models loaded.")
                else:
                    st.warning("API is running but models not loaded.")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")

        st.divider()
        st.caption("Built from first principles. No LangChain.")

    # ── Page Router ─────────────────────────────────────────────
    if page == "💬 Chat":
        from agentmesh.ui.chat_page import render_chat_page
        render_chat_page()
    elif page == "🔍 Trajectories":
        from agentmesh.ui.trajectory_page import render_trajectory_page 
        render_trajectory_page()
    elif page == "📊 Eval Dashboard":
        from agentmesh.ui.eval_page import render_eval_page
        render_eval_page()


if __name__ == "__main__":
    main()