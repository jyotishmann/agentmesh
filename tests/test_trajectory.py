# file: tests/test_trajectory.py
"""Tests for trajectory logger."""

import pytest

from agentmesh.trajectory import TrajectoryLogger


def _make_mock_result(task_id="task_test123", completed=True, n_events=3):
    """Create a mock OrchestratorResult-like object."""

    class MockResult:
        pass

    result = MockResult()
    result.task_id = task_id
    result.response = "Test response"
    result.completed = completed
    result.token_summary = {
        "total_tokens_in": 100,
        "total_tokens_out": 50,
        "total_tokens": 150,
        "total_tool_calls": n_events,
    }
    result.trajectory_events = [
        {
            "step_number": i + 1,
            "agent_name": f"Agent{i}",
            "action_type": "tool_call",
            "tool_name": f"tool_{i}",
            "tool_input": f"input_{i}",
            "tool_output": f"output_{i}",
            "tokens_in": 10,
            "tokens_out": 5,
            "latency_ms": 100.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "metadata": "{}",
        }
        for i in range(n_events)
    ]
    return result


class TestTrajectoryLogger:
    """Test suite for TrajectoryLogger."""

    def test_save_and_get(self, temp_db):
        """Should save a trajectory and retrieve it."""
        logger = TrajectoryLogger(db_path=temp_db)
        result = _make_mock_result()

        saved_id = logger.save("Test task", result)
        assert saved_id == "task_test123"

        loaded = logger.get("task_test123")
        assert loaded is not None
        assert loaded["task_id"] == "task_test123"
        assert loaded["user_task"] == "Test task"
        assert loaded["completed"] is True
        assert len(loaded["events"]) == 3

    def test_get_nonexistent(self, temp_db):
        """Should return None for unknown task_id."""
        logger = TrajectoryLogger(db_path=temp_db)
        assert logger.get("nonexistent") is None

    def test_list_recent(self, temp_db):
        """Should list trajectories in reverse chronological order."""
        logger = TrajectoryLogger(db_path=temp_db)

        for i in range(5):
            result = _make_mock_result(task_id=f"task_{i}")
            logger.save(f"Task {i}", result)

        recent = logger.list_recent(limit=3)
        assert len(recent) == 3

    def test_get_stats(self, temp_db):
        """Should compute aggregate statistics."""
        logger = TrajectoryLogger(db_path=temp_db)

        logger.save("Task A", _make_mock_result("t1", completed=True))
        logger.save("Task B", _make_mock_result("t2", completed=False))
        logger.save("Task C", _make_mock_result("t3", completed=True))

        stats = logger.get_stats()
        assert stats["total_runs"] == 3
        assert stats["completed_runs"] == 2
        assert abs(stats["completion_rate"] - 0.667) < 0.01

    def test_delete(self, temp_db):
        """Should delete a trajectory and its events."""
        logger = TrajectoryLogger(db_path=temp_db)
        logger.save("Task X", _make_mock_result("task_del"))

        assert logger.delete("task_del") is True
        assert logger.get("task_del") is None

    def test_delete_nonexistent(self, temp_db):
        """Deleting a nonexistent trajectory should return False."""
        logger = TrajectoryLogger(db_path=temp_db)
        assert logger.delete("nope") is False

    def test_events_ordered(self, temp_db):
        """Events should be ordered by step_number."""
        logger = TrajectoryLogger(db_path=temp_db)
        logger.save("Task", _make_mock_result("task_order", n_events=5))

        loaded = logger.get("task_order")
        steps = [e["step_number"] for e in loaded["events"]]
        assert steps == sorted(steps)