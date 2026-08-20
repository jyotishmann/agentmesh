# file: tests/test_config.py
"""Tests for configuration loading."""

import os
import pytest
from agentmesh.config import Settings


class TestConfig:
    """Test suite for Settings."""

    def test_defaults_load(self):
        """Default settings should load without any env vars."""
        s = Settings()
        assert s.main_model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert s.specialist_model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert s.embedding_model_name == "BAAI/bge-small-en-v1.5"
        assert s.max_tool_calls > 0
        assert s.max_total_tool_calls > 0

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("AGENTMESH_MAX_TOOL_CALLS", "10")
        monkeypatch.setenv("AGENTMESH_SERVER_PORT", "9999")

        s = Settings()
        assert s.max_tool_calls == 10
        assert s.server_port == 9999

    def test_model_paths_are_strings(self):
        """Model names should be non-empty strings."""
        s = Settings()
        assert isinstance(s.main_model_name, str)
        assert len(s.main_model_name) > 0
        assert isinstance(s.specialist_model_name, str)
        assert len(s.specialist_model_name) > 0

    def test_db_path_default(self):
        """DB path should have a sensible default."""
        s = Settings()
        assert "data" in s.db_path
        assert s.db_path.endswith(".db")

    def test_generation_params(self):
        """Generation params should be within valid ranges."""
        s = Settings()
        assert 0.0 <= s.temperature <= 2.0
        assert s.max_new_tokens > 0
        assert s.max_new_tokens <= 4096