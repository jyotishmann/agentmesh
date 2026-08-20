# file: tests/test_api.py
"""Tests for FastAPI API endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _create_test_app():
    """Create a test app with mocked dependencies."""
    from agentmesh.api.server import app

    # Mock the orchestrator
    mock_result = MagicMock()
    mock_result.response = "Test response"
    mock_result.task_id = "task_test"
    mock_result.completed = True
    mock_result.token_summary = {"total_tokens": 100}
    mock_result.trajectory_events = []

    mock_orchestrator = MagicMock()
    mock_orchestrator.run.return_value = mock_result

    # Mock the trajectory logger
    mock_traj_logger = MagicMock()
    mock_traj_logger.save.return_value = "task_test"
    mock_traj_logger.get.return_value = {
        "task_id": "task_test",
        "user_task": "test",
        "final_output": "output",
        "completed": True,
        "total_tool_calls": 1,
        "total_tokens": 100,
        "total_latency_ms": 500.0,
        "events": [],
        "created_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T00:01:00Z",
        "token_summary": {},
    }
    mock_traj_logger.list_recent.return_value = []
    mock_traj_logger.get_stats.return_value = {"total_runs": 0}

    app.state.orchestrator = mock_orchestrator
    app.state.trajectory_logger = mock_traj_logger
    app.state.model_loaded = True

    return app


class TestHealthEndpoint:
    def test_health(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestChatEndpoint:
    def test_chat_success(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/chat", json={"task": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["task_id"] == "task_test"

    def test_chat_empty_task(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/chat", json={"task": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_chat_with_session(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/chat",
            json={"task": "Hello", "session_id": "my_session"},
        )
        assert resp.status_code == 200


class TestTrajectoryEndpoints:
    def test_get_trajectory(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/trajectory/task_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task_test"

    def test_get_trajectory_not_found(self):
        app = _create_test_app()
        app.state.trajectory_logger.get.return_value = None
        client = TestClient(app)
        resp = client.get("/trajectory/nonexistent")
        assert resp.status_code == 404

    def test_list_trajectories(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.get("/trajectory/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "trajectories" in data
        assert "total" in data


class TestStreamEndpoint:
    def test_stream_response(self):
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"task": "Test streaming"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")