# file: tests/conftest.py
"""Shared fixtures and mock providers for testing."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentmesh.config import settings
from agentmesh.models.base import ModelResponse


class MockModelProvider:
    """Fake model provider for GPU-free testing.

    Returns canned responses based on prompt keywords.
    Implements the same interface as QwenModelProvider.
    """

    def __init__(self):
        self.call_count = 0
        self.last_prompt = ""

    def generate(self, messages: list[dict], **kwargs) -> ModelResponse:
        """Return a canned response based on the prompt content."""
        self.call_count += 1

        # Extract the last user message
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        self.last_prompt = prompt

        # Route to canned responses
        if any(kw in prompt.lower() for kw in ["break down", "plan", "sub-task"]):
            text = json.dumps([
                {
                    "description": "Research the topic",
                    "specialist": "research",
                    "required_tools": ["search_web"],
                },
                {
                    "description": "Summarise findings",
                    "specialist": "research",
                    "required_tools": [],
                },
            ])
        elif any(kw in prompt.lower() for kw in ["critic", "evaluate", "quality"]):
            text = json.dumps({
                "pass": True,
                "confidence": 0.85,
                "feedback": "Output meets quality standards.",
            })
        elif "<tool_call>" in prompt.lower() or "tool" in prompt.lower():
            text = "Based on my analysis, the answer is 42."
        else:
            text = "This is a mock response for testing purposes."

        return ModelResponse(
            text=text,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            latency_ms=10.0,
            model_name="mock-model",
        )


class MockEmbeddingProvider:
    """Fake embedding provider for testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic fake embeddings (384-dim)."""
        import hashlib
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(c, 16) / 15.0 for c in h] + [0.0] * (384 - 32)
            # Normalise
            norm = sum(x**2 for x in vec) ** 0.5
            results.append([x / (norm or 1.0) for x in vec])
        return results


@pytest.fixture
def mock_model_provider():
    """Fixture: a MockModelProvider instance."""
    return MockModelProvider()


@pytest.fixture
def mock_embedding_provider():
    """Fixture: a MockEmbeddingProvider instance."""
    return MockEmbeddingProvider()


@pytest.fixture
def temp_db(tmp_path):
    """Fixture: temporary database path for isolated tests."""
    db_path = str(tmp_path / "test.db")
    return db_path


@pytest.fixture
def tool_registry():
    """Fixture: a real tool registry with all tools registered."""
    from agentmesh.tools import create_default_registry
    return create_default_registry()


@pytest.fixture
def temp_sandbox(tmp_path):
    """Fixture: temporary sandbox directory for file I/O tests."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return str(sandbox)