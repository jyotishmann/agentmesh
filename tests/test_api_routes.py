# file: tests/test_api_routes.py
"""Tests for API routing and results-file access (PR 16). No GPU, no models.

The app's lifespan isn't run (TestClient is used without a `with` block),
so the orchestrator and trajectory logger are replaced with mocks.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agentmesh.api.server import app
from agentmesh.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose results folder is a temp dir holding two saved runs."""
    monkeypatch.setattr(settings, "db_path", tmp_path / "agentmesh.db")
    results = tmp_path / "eval_results"
    results.mkdir()
    (results / "eval_20261009_154140.json").write_text(
        json.dumps({"summary": {"overall": {"answer_correct": 0.613}}})
    )
    (results / "eval_20261009_154140_regraded.json").write_text(
        json.dumps({"summary": {"overall": {"answer_correct": 0.742}}})
    )
    (tmp_path / "secret.json").write_text('{"token": "do-not-leak"}')

    trajectories = MagicMock()
    trajectories.list_recent.return_value = [
        {
            "task_id": "task_1",
            "user_task": "What is 2 + 2?",
            "completed": True,
            "total_tool_calls": 0,
            "total_tokens": 100,
            "created_at": "2026-10-09T15:41:40Z",
        }
    ]
    trajectories.get_stats.return_value = {"total_runs": 1}
    trajectories.get.return_value = {
        "task_id": "task_1",
        "user_task": "What is 2 + 2?",
        "final_output": "4",
        "completed": True,
        "total_tool_calls": 0,
        "total_tokens": 100,
        "total_latency_ms": 50.0,
        "events": [],
        "created_at": "2026-10-09T15:41:40Z",
        "finished_at": "2026-10-09T15:41:41Z",
    }
    app.state.trajectory_logger = trajectories
    app.state.orchestrator = MagicMock()
    app.state.model_loaded = True
    return TestClient(app)


class TestTrajectoryRoutes:
    def test_list_is_not_swallowed_by_the_id_route(self, client):
        # Before PR 16, "/trajectory/{task_id}" was registered first and
        # answered this request as "trajectory with ID 'list'": a 404.
        resp = client.get("/trajectory/list")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        app.state.trajectory_logger.get.assert_not_called()

    def test_single_trajectory_still_works(self, client):
        resp = client.get("/trajectory/task_1")
        assert resp.status_code == 200
        assert resp.json()["final_output"] == "4"


class TestResultsFiles:
    def test_listing_is_newest_first(self, client):
        files = client.get("/eval/results", params={"latest": False}).json()["files"]
        assert files == [
            "eval_20261009_154140_regraded.json",
            "eval_20261009_154140.json",
        ]

    def test_load_one_run_by_name(self, client):
        resp = client.get("/eval/results/eval_20261009_154140_regraded.json")
        assert resp.status_code == 200
        assert resp.json()["summary"]["overall"]["answer_correct"] == 0.742

    def test_unknown_run_is_404(self, client):
        assert client.get("/eval/results/eval_19990101_000000.json").status_code == 404

    @pytest.mark.parametrize(
        "name", ["secret.json", "..%2Fsecret.json", "eval_x.json.bak", "eval_../x.json"]
    )
    def test_names_outside_the_pattern_are_rejected(self, client, name):
        resp = client.get(f"/eval/results/{name}")
        assert resp.status_code in (400, 404)
        assert "do-not-leak" not in resp.text

    def test_compare_rejects_path_traversal(self, client):
        # Before PR 16, run_a was joined straight onto the results folder
        resp = client.get(
            "/eval/compare",
            params={"run_a": "../secret.json", "run_b": "eval_20261009_154140.json"},
        )
        assert resp.status_code == 400
        assert "do-not-leak" not in resp.text

    def test_compare_two_runs(self, client):
        resp = client.get(
            "/eval/compare",
            params={
                "run_a": "eval_20261009_154140.json",
                "run_b": "eval_20261009_154140_regraded.json",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_a"]["data"]["summary"]["overall"]["answer_correct"] == 0.613
        assert body["run_b"]["data"]["summary"]["overall"]["answer_correct"] == 0.742
