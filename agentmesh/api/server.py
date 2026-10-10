# file: agentmesh/api/server.py
"""FastAPI server — HTTP interface to the AgentMesh system."""

import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agentmesh.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalResultResponse,
    EvalRunRequest,
    HealthResponse,
    TrajectoryListResponse,
    TrajectoryResponse,
)
from agentmesh.config import settings
from agentmesh.orchestrator import Orchestrator
from agentmesh.trajectory import TrajectoryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Results files are named eval_<timestamp>[_regraded].json. Anything else,
# including "../" paths, is rejected before touching the filesystem.
_RESULTS_FILE = re.compile(r"^eval_[A-Za-z0-9_]+\.json$")


def _results_dir() -> Path:
    results_dir = Path(settings.db_path).parent / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load_results_file(filename: str) -> dict:
    """Load one results file by name, refusing anything outside the results folder."""
    if not _RESULTS_FILE.match(filename):
        raise HTTPException(status_code=400, detail=f"Invalid results file name: {filename!r}")
    path = _results_dir() / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Results file '{filename}' not found.")
    return json.loads(path.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload models. Shutdown: cleanup."""
    logger.info("Starting AgentMesh API server...")

    # Initialise shared instances
    app.state.orchestrator = Orchestrator()
    app.state.trajectory_logger = TrajectoryLogger()
    app.state.model_loaded = True

    logger.info("Models loaded. Server ready.")
    yield

    # Shutdown
    logger.info("Shutting down AgentMesh API server.")


app = FastAPI(
    title="AgentMesh API",
    description="Multi-agent system with evaluation framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model_loaded", False),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Execute a task synchronously and return the result.

    The orchestrator runs the full pipeline: plan → execute → critique.
    Trajectory is saved automatically.
    """
    try:
        result = app.state.orchestrator.run(
            task=request.task,
            session_id=request.session_id or "",
        )

        # Save trajectory
        app.state.trajectory_logger.save(request.task, result)

        return ChatResponse(
            status="completed" if result.completed else "partial",
            response=result.response,
            task_id=result.task_id,
            token_summary=result.token_summary,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Execute a task and stream trajectory events via SSE.

    Events are sent as they are replayed from the completed trajectory.
    Each event is a JSON object with agent_name, action_type, and content.
    The stream ends with data: [DONE].
    """

    def event_generator():
        try:
            result = app.state.orchestrator.run(
                task=request.task,
                session_id=request.session_id or "",
            )

            # Save trajectory
            app.state.trajectory_logger.save(request.task, result)

            # Stream each trajectory event
            for event in result.trajectory_events:
                payload = json.dumps({
                    "type": "event",
                    "step": event.get("step_number", 0),
                    "agent": event.get("agent_name", ""),
                    "action": event.get("action_type", ""),
                    "tool": event.get("tool_name", ""),
                    "content": event.get("tool_output", "")[:500],
                    "latency_ms": event.get("latency_ms", 0),
                })
                yield f"data: {payload}\n\n"

            # Final result
            final_payload = json.dumps({
                "type": "final",
                "response": result.response,
                "task_id": result.task_id,
                "completed": result.completed,
                "token_summary": result.token_summary,
            })
            yield f"data: {final_payload}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_payload = json.dumps({
                "type": "error",
                "message": str(e),
            })
            yield f"data: {error_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )

# Order matters: FastAPI matches routes in registration order, so
# "/trajectory/{task_id}" must come after "/trajectory/list" or it
# answers "list" as a task ID (404).
@app.get("/trajectory/list", response_model=TrajectoryListResponse)
def list_trajectories(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List recent trajectories with pagination."""
    trajectories = app.state.trajectory_logger.list_recent(
        limit=limit, offset=offset
    )

    # Get total count for pagination
    stats = app.state.trajectory_logger.get_stats()
    total = stats.get("total_runs", 0)

    return TrajectoryListResponse(
        trajectories=trajectories,
        total=total,
    )

@app.get("/trajectory/{task_id}", response_model=TrajectoryResponse)
def get_trajectory(task_id: str):
    """Load a full trajectory by task_id."""
    trajectory = app.state.trajectory_logger.get(task_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"Trajectory '{task_id}' not found.")
    return TrajectoryResponse(**trajectory)


@app.post("/eval/run")
def run_eval(request: EvalRunRequest):
    """Trigger an eval run.

    WARNING: This is a long-running request. For 50 tasks on a T4 GPU,
    expect 10-30 minutes depending on task complexity.
    """
    from agentmesh.eval.runner import EvalRunner

    try:
        runner = EvalRunner(
            orchestrator=app.state.orchestrator,
            trajectory_logger=app.state.trajectory_logger,
        )

        if request.task_id:
            result = runner.run_single(request.task_id)
            return {"status": "completed", "result": result}

        results = runner.run_all(
            categories=request.categories,
            max_difficulty=request.max_difficulty,
        )
        return {"status": "completed", "results": results}

    except Exception as e:
        logger.error(f"Eval run error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/results", response_model=EvalResultResponse)
def get_eval_results(latest: bool = Query(default=True)):
    """List saved eval results. Optionally include the latest result."""
    files = sorted(
        [f.name for f in _results_dir().glob("eval_*.json")],
        reverse=True,
    )
    latest_data = _load_results_file(files[0]) if latest and files else None
    return EvalResultResponse(files=files, latest=latest_data)


@app.get("/eval/results/{filename}")
def get_eval_result(filename: str):
    """Load one saved results file by name (used by the dashboard's run picker)."""
    return _load_results_file(filename)


@app.get("/eval/compare")
def compare_eval_runs(
    run_a: str = Query(..., description="Filename of first eval run"),
    run_b: str = Query(..., description="Filename of second eval run"),
):
    """Compare two eval runs side by side."""
    # Both names go through the same check, so "../" can't escape the folder
    return {
        "run_a": {"filename": run_a, "data": _load_results_file(run_a)},
        "run_b": {"filename": run_b, "data": _load_results_file(run_b)},
    }
