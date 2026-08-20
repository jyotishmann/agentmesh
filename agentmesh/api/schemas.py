# file: agentmesh/api/schemas.py
"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for /chat and /chat/stream."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural-language task.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "Search the web for the latest Python release and write a script to print its version.",
                    "session_id": "session_abc123",
                }
            ]
        }
    }


class EvalRunRequest(BaseModel):
    """Request body for /eval/run."""

    categories: Optional[list[str]] = Field(
        default=None,
        description="Filter tasks by categories. None = all.",
    )
    max_difficulty: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Only run tasks at or below this difficulty.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Run a single task by ID. Overrides category/difficulty filters.",
    )


class TrajectoryListRequest(BaseModel):
    """Query params for /trajectory/list."""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

# ── Response Schemas ────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Response body for /chat."""

    status: str = Field(description="'completed' or 'partial'")
    response: str = Field(description="The agent's final response text.")
    task_id: str = Field(description="Trajectory ID for this task.")
    token_summary: dict = Field(
        default_factory=dict,
        description="Token usage breakdown.",
    )


class TrajectoryResponse(BaseModel):
    """Response body for /trajectory/{id}."""

    task_id: str
    user_task: str
    final_output: str
    completed: bool
    total_tool_calls: int
    total_tokens: int
    total_latency_ms: float
    events: list[dict]
    created_at: str
    finished_at: str


class TrajectoryListItem(BaseModel):
    """Single item in trajectory list response."""

    task_id: str
    user_task: str
    completed: bool
    total_tool_calls: int
    total_tokens: int
    created_at: str


class TrajectoryListResponse(BaseModel):
    """Response body for /trajectory/list."""

    trajectories: list[TrajectoryListItem]
    total: int


class EvalResultResponse(BaseModel):
    """Response body for /eval/results."""

    files: list[str]
    latest: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str = "ok"
    model_loaded: bool = False
    version: str = "0.1.0"