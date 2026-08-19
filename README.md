
# AgentMesh

A production multi-agent system built from first principles — no LangChain, no CrewAI, no framework abstractions.

## What This Is

AgentMesh takes a natural-language task, breaks it down through a **Planner Agent**, delegates sub-tasks to **Specialist Agents** (Research, Coder, Analyst), runs them against a **Tool Registry** (web search, code execution, file I/O, vector knowledge base), passes the output through a **Critic Agent** for quality gating, and returns the result — all while logging every tool call, token cost, and latency into a trajectory evaluation framework.

## Why It's Different

- **No framework lock-in**: The agent loop is ~200 lines of Python. Every piece is explainable.
- **Loop detection**: Catches infinite tool-call cycles — the #1 failure mode in deployed agents.
- **Trajectory eval suite**: 50 labelled tasks with known answers, run on every prompt change.
- **Token cost tracking**: Every task produces a cost breakdown.

## Architecture

```text
User Task → Planner Agent → Sub-tasks
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ResearchAgent   CoderAgent     AnalystAgent
        (search_web,    (run_python,   (read_file,
         query_kb)       read/write)    run_python)
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                        Critic Agent
                        (pass/fail)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Final Output        Revision Loop
                                  (max 2 cycles)

```
All calls logged → Trajectory DB → Eval Framework (50 tasks)

## Quick Start

The "30 seconds to running" section. A reviewer should be able to clone, install, and see the app working with exactly these commands. The `make` commands abstract away the underlying `python` and `uvicorn` calls.

### Quick Start
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/agentmesh.git
cd agentmesh

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure (defaults work out of the box)
cp .env.example .env

# Start the API server
make api

# In a second terminal, start the Streamlit UI
make ui

# Open http://localhost:8501 in your browser
```

## Usage and Eval Sections

The eval section is highlighted because it's the differentiator. Showing that the project has a test suite with quantitative metrics is what separates it from demo-quality agent projects. The models section tells the reader exactly what hardware they need.

### Running the Eval Suite
```bash
# Run all 50 evaluation tasks
make eval

# Run a specific category
python -m agentmesh.eval run --category research

# Compare two eval runs
python -m agentmesh.eval compare run_A run_B

