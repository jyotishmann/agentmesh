# AgentMesh

A multi-agent system built from scratch (no LangChain, no CrewAI), with an evaluation framework that measures whether its answers are actually right.

A planner splits a task into sub-tasks, specialist agents solve them with tools, and a critic checks the result. All of it runs on small open models (Qwen2.5-3B and 1.5B) on a single T4 GPU. Every tool call, token and verdict is logged, and a 50-task eval suite scores each run against known answers.

![Eval dashboard](docs/images/eval_results.png)

## How it works

```text
User Task → Planner Agent → Sub-tasks (validated, max 4)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ResearchAgent    CoderAgent     AnalystAgent
        (search_web,    (run_python,   (read_file,
         query_kb)       read/write)    run_python)
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                        Critic Agent
                        (pass / reject)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Final Output        Revision Loop
                                  (max 2 cycles)

All steps → Trajectory DB (SQLite) → Eval framework (50 tasks) → Dashboard
```

## Design choices

- **No agent framework.** The agent loop, tool-call parsing and orchestration are plain Python, so every behaviour can be read, tested and changed.
- **Least-privilege tools.** Each specialist sees and can call only its own tools; a research agent can't run code.
- **Loop detection and budgets.** Repeated identical tool calls are flagged in the trajectory and scored by the eval, and per-agent and per-task tool-call limits stop runaway runs.
- **Correctness over completion.** Answers are graded against expected keyword groups with word-form matching and forbidden phrases (e.g. "not prime"). Tasks with random, live or creative output are excluded from accuracy rather than guessed at.
- **Reproducible evals.** Greedy decoding, no memory carried between tasks, a save after every task, resumable runs, and library versions recorded in every results file.
- **Cheap re-scoring.** `regrade` re-scores a saved run from its stored trajectories. A grader fix is measured in seconds, without a GPU.

## Dashboard

A Streamlit UI over a FastAPI backend:

- **Eval Dashboard**: accuracy and critic metrics for any saved run, per-category breakdown, filters for wrong answers and critic mistakes, and a comparison view that lists every verdict that changed between two runs.
- **Trajectory Viewer**: every step of a run (plan, tool calls, critique, revisions) with inputs, outputs, tokens and latency.
- **Chat**: run a task and watch the agents work.

| Compare two runs | Per-category results |
|---|---|
| ![Compare](docs/images/eval_compare.png) | ![Categories](docs/images/eval_categories.png) |

![Trajectory viewer](docs/images/trajectories.png)

## Quick start

### Colab (recommended: free T4)

Open [`notebooks/agentmesh_colab.ipynb`](notebooks/agentmesh_colab.ipynb) in Colab. It has sections for setup, tests, evals, results and the dashboard. Browsing results and the dashboard works on a CPU runtime; only running evals needs the GPU.

### Local (needs a CUDA GPU with 12 GB+, e.g. a T4)

```bash
git clone https://github.com/jyotishmann/agentmesh.git
cd agentmesh
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # defaults work as-is

make api   # terminal 1: API on http://localhost:8000
make ui    # terminal 2: UI on http://localhost:8501
```

Both Qwen models stay loaded together (~9.5 GB in fp16). They download from Hugging Face on first use: Qwen2.5-3B-Instruct (planner and critic), Qwen2.5-1.5B-Instruct (specialists), and bge-small-en-v1.5 (embeddings).

## Running evals

```bash
python -m agentmesh.eval.cli run                         # all 50 tasks (~55 min on a T4)
python -m agentmesh.eval.cli run --category factual_qa   # one category
python -m agentmesh.eval.cli run --max-difficulty 1      # quick check (~5 min)
python -m agentmesh.eval.cli run --resume FILE           # finish an interrupted run
python -m agentmesh.eval.cli history                     # one line per saved run
python -m agentmesh.eval.cli report --file FILE          # per-task detail
python -m agentmesh.eval.cli regrade FILE                # re-score with the current grader (no GPU)
```

Categories: `factual_qa`, `code_generation`, `multi_step`, `analysis`, `creative` (10 tasks each, difficulty 1–5).

## Tests

```bash
pytest tests/test_api.py tests/test_api_routes.py tests/test_correctness.py tests/test_reliability.py tests/test_grader.py -q
```

102 tests, no GPU and no model downloads: API routing and file access, grading rules pinned to real answers, metric aggregation, plan validation, and resumable runs.

## Results run

Baseline from Oct 9, 2026: all 50 tasks, deterministic decoding, one T4.

| Metric | Value | Notes |
|---|---|---|
| **Answer accuracy** | **74%** (23 of 31) | Final answer matches the expected answer, on the 31 tasks with a fixed answer |
| Critic agreement | 71% | The critic's pass/reject matches the grader |
| Critic false-pass | 88% (7 of 8) | Wrong answers the critic approved |
| Task completion | 88% | Runs that finished within limits (finished is not the same as correct) |
| Avg time per task | 65 s | Plan, execute, critique, and up to 2 revisions |


| Category | Correct | Graded tasks |
|---|---|---|
| code_generation | 89% | 8 / 9 |
| factual_qa | 75% | 6 / 8 |
| multi_step | 75% | 3 / 4 |
| analysis | 50% | 4 / 8 |
| creative | 100% | 2 / 2 |
