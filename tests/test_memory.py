# file: tests/test_memory.py
"""Tests for memory layer — buffer and persistent store."""

import pytest
from unittest.mock import patch, MagicMock

from agentmesh.memory.buffer import ConversationBuffer


class TestConversationBuffer:
    """Test the in-memory conversation buffer."""

    def test_add_and_get(self):
        """Should store and retrieve messages."""
        buf = ConversationBuffer(max_messages=10)
        buf.add("user", "hello")
        buf.add("assistant", "hi there")

        msgs = buf.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_overflow(self):
        """Buffer should drop oldest messages when full."""
        buf = ConversationBuffer(max_messages=3)
        buf.add("user", "msg1")
        buf.add("assistant", "msg2")
        buf.add("user", "msg3")
        buf.add("assistant", "msg4")  # should push out msg1

        msgs = buf.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg2"

    def test_get_context_string(self):
        """Context string should format messages for the model."""
        buf = ConversationBuffer(max_messages=10)
        buf.add("user", "What is Python?")
        buf.add("assistant", "A programming language.")

        ctx = buf.get_context_string()
        assert "user:" in ctx.lower() or "User:" in ctx
        assert "Python" in ctx

    def test_empty_buffer(self):
        """Empty buffer should return empty list and empty string."""
        buf = ConversationBuffer()
        assert buf.get_messages() == []
        assert buf.get_context_string() == ""


class TestPersistentMemory:
    """Test the persistent memory store."""

    def test_store_and_get_recent(self, temp_db, mock_embedding_provider):
        """Should store a memory and retrieve it."""
        from agentmesh.memory.persistent import PersistentMemory

        mem = PersistentMemory(db_path=temp_db)
        mem._embedding_provider = mock_embedding_provider

        mem.store("Built a calculator app", "Successfully created calculator with +,-,*,/")
        recent = mem.get_recent(limit=5)

        assert len(recent) >= 1
        assert "calculator" in recent[0]["task_summary"].lower()

    def test_multiple_stores(self, temp_db, mock_embedding_provider):
        """Should handle multiple memory entries."""
        from agentmesh.memory.persistent import PersistentMemory

        mem = PersistentMemory(db_path=temp_db)
        mem._embedding_provider = mock_embedding_provider

        mem.store("Task A", "Result A")
        mem.store("Task B", "Result B")
        mem.store("Task C", "Result C")

        recent = mem.get_recent(limit=10)
        assert len(recent) == 3

    def test_memory_context(self, temp_db, mock_embedding_provider):
        """get_memory_context should return a formatted string."""
        from agentmesh.memory.persistent import PersistentMemory

        mem = PersistentMemory(db_path=temp_db)
        mem._embedding_provider = mock_embedding_provider

        mem.store("Previous task about Python", "Learned about decorators")

        ctx = mem.get_memory_context("Tell me about Python decorators")
        assert isinstance(ctx, str)