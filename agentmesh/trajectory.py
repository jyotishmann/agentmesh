# file: agentmesh/trajectory.py
"""Trajectory Logger — records every event in a task execution for replay and eval."""

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agentmesh.config import settings

logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trajectories (
    task_id         TEXT PRIMARY KEY,
    user_task       TEXT NOT NULL,
    final_output    TEXT NOT NULL DEFAULT '',
    completed       INTEGER NOT NULL DEFAULT 0,
    total_tool_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    token_summary   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    finished_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL,
    step_number     INTEGER NOT NULL,
    agent_name      TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    tool_name       TEXT NOT NULL DEFAULT '',
    tool_input      TEXT NOT NULL DEFAULT '',
    tool_output     TEXT NOT NULL DEFAULT '',
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    latency_ms      REAL NOT NULL DEFAULT 0.0,
    timestamp       TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (task_id) REFERENCES trajectories(task_id)
);

CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_created ON trajectories(created_at);
"""


class TrajectoryLogger:
    """Persists full task trajectories to SQLite for debugging, eval, and replay."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path(settings.db_path).parent / "trajectories.db"
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA_SQL)

    def save(self, user_task: str, result) -> str:
        """Save an OrchestratorResult as a trajectory.

        Args:
            user_task: The original user task string.
            result: An OrchestratorResult from the orchestrator.

        Returns:
            The task_id of the saved trajectory.
        """
        token_summary = result.token_summary or {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN")

            try:
                # Insert trajectory header
                conn.execute(
                    """
                    INSERT INTO trajectories
                        (task_id, user_task, final_output, completed,
                         total_tool_calls, total_tokens, total_latency_ms,
                         token_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.task_id,
                        user_task,
                        result.response[:5000],  # cap storage
                        int(result.completed),
                        token_summary.get("total_tool_calls", 0),
                        token_summary.get("total_tokens", 0),
                        sum(
                            e.get("latency_ms", 0)
                            for e in result.trajectory_events
                        ),
                        json.dumps(token_summary),
                    ),
                )

                # Insert all events
                event_rows = [
                    (
                        result.task_id,
                        e.get("step_number", 0),
                        e.get("agent_name", ""),
                        e.get("action_type", ""),
                        e.get("tool_name", ""),
                        e.get("tool_input", ""),
                        e.get("tool_output", ""),
                        e.get("tokens_in", 0),
                        e.get("tokens_out", 0),
                        e.get("latency_ms", 0.0),
                        e.get("timestamp", ""),
                        e.get("metadata", "{}"),
                    )
                    for e in result.trajectory_events
                ]

                conn.executemany(
                    """
                    INSERT INTO events
                        (task_id, step_number, agent_name, action_type,
                         tool_name, tool_input, tool_output,
                         tokens_in, tokens_out, latency_ms,
                         timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )

                conn.execute("COMMIT")
                logger.info(
                    f"Saved trajectory {result.task_id} "
                    f"({len(event_rows)} events)"
                )

            except Exception:
                conn.execute("ROLLBACK")
                raise

        return result.task_id

    def get(self, task_id: str) -> Optional[dict]:
        """Load a full trajectory by task_id.

        Returns:
            Dict with trajectory header and events list, or None.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM trajectories WHERE task_id = ?",
                (task_id,),
            ).fetchone()

            if row is None:
                return None

            trajectory = dict(row)
            trajectory["completed"] = bool(trajectory["completed"])

            # Parse JSON fields
            try:
                trajectory["token_summary"] = json.loads(
                    trajectory.get("token_summary", "{}")
                )
            except json.JSONDecodeError:
                trajectory["token_summary"] = {}

            # Load events
            event_rows = conn.execute(
                """
                SELECT * FROM events
                WHERE task_id = ?
                ORDER BY step_number ASC
                """,
                (task_id,),
            ).fetchall()

            trajectory["events"] = []
            for e in event_rows:
                event = dict(e)
                try:
                    event["metadata"] = json.loads(event.get("metadata", "{}"))
                except json.JSONDecodeError:
                    event["metadata"] = {}
                trajectory["events"].append(event)

            return trajectory

    def list_recent(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """List recent trajectories (headers only, no events).

        Args:
            limit: Max number of results.
            offset: Pagination offset.

        Returns:
            List of trajectory header dicts.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT task_id, user_task, completed,
                       total_tool_calls, total_tokens, total_latency_ms,
                       created_at, finished_at
                FROM trajectories
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            return [
                {**dict(r), "completed": bool(r["completed"])}
                for r in rows
            ]

    def get_stats(self) -> dict:
        """Compute aggregate statistics across all trajectories.

        Returns:
            Dict with count, completion_rate, avg_tool_calls,
            avg_tokens, avg_latency_ms.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_runs,
                    COALESCE(SUM(completed), 0) as completed_runs,
                    COALESCE(AVG(total_tool_calls), 0) as avg_tool_calls,
                    COALESCE(AVG(total_tokens), 0) as avg_tokens,
                    COALESCE(AVG(total_latency_ms), 0) as avg_latency_ms
                FROM trajectories
                """
            ).fetchone()

            total = row[0]
            return {
                "total_runs": total,
                "completed_runs": row[1],
                "completion_rate": round(row[1] / total, 3) if total > 0 else 0.0,
                "avg_tool_calls": round(row[2], 2),
                "avg_tokens": round(row[3], 0),
                "avg_latency_ms": round(row[4], 1),
            }

    def delete(self, task_id: str) -> bool:
        """Delete a trajectory and its events.

        Returns:
            True if a trajectory was deleted, False if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM events WHERE task_id = ?", (task_id,))
            cursor = conn.execute(
                "DELETE FROM trajectories WHERE task_id = ?", (task_id,)
            )
            return cursor.rowcount > 0