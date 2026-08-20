# file: agentmesh/memory/persistent.py
"""Long-term persistent memory — SQLite + FAISS semantic search."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import faiss # type: ignore
import numpy as np

from agentmesh.config import settings
from agentmesh.models.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

# SQLite schema for the memories table
MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id             TEXT PRIMARY KEY,
    task_summary   TEXT NOT NULL,
    result_summary TEXT NOT NULL,
    tags           TEXT DEFAULT '[]',
    embedding_id   INTEGER,
    created_at     TEXT NOT NULL
);
"""

class PersistentMemory:
    """Long-term memory with structured + semantic search.

    Structured store: SQLite database with task/result summaries and tags.
    Semantic store: FAISS index over task summaries for similarity search.
    """

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self._embedder = embedding_provider or EmbeddingProvider()
        self._db_path = settings.database_path
        self._memory_index_path = self._db_path.parent / "memory_index.bin"

        # Ensure data directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialise SQLite
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(MEMORY_SCHEMA)
        self._conn.commit()

        # Initialise FAISS index for semantic search
        self._index: Optional[faiss.IndexFlatL2] = None
        self._embedding_ids: list[str] = []  # parallel to FAISS vectors
        self._load_index()

    def _load_index(self) -> None:
        """Load the memory FAISS index from disk, or create empty."""
        if self._memory_index_path.exists():
            self._index = faiss.read_index(str(self._memory_index_path))
            # Reload the ID mapping from SQLite
            cursor = self._conn.execute(
                "SELECT id FROM memories WHERE embedding_id IS NOT NULL "
                "ORDER BY embedding_id ASC"
            )
            self._embedding_ids = [row["id"] for row in cursor.fetchall()]
            logger.info(
                f"Loaded memory index with {self._index.ntotal} vectors"
            )
        else:
            dim = self._embedder.embedding_dim
            self._index = faiss.IndexFlatL2(dim)
            self._embedding_ids = []

    def store(
        self,
        task_summary: str,
        result_summary: str,
        tags: list[str] | None = None,
    ) -> str:
        """Store a completed task in long-term memory.

        Args:
            task_summary: One-line summary of the task.
            result_summary: One-line summary of the result.
            tags: Optional categorical tags for filtering.

        Returns:
            The generated memory ID.
        """
        memory_id = str(uuid4())[:8]
        tags_json = json.dumps(tags or [])
        now = datetime.now(timezone.utc).isoformat()

        # Embed the task summary for semantic search
        embedding = self._embedder.encode(task_summary)
        embedding_id = self._index.ntotal
        self._index.add(embedding)
        self._embedding_ids.append(memory_id)

        # Store in SQLite
        self._conn.execute(
            "INSERT INTO memories (id, task_summary, result_summary, tags, "
            "embedding_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, task_summary, result_summary, tags_json, embedding_id, now),
        )
        self._conn.commit()

        # Persist FAISS index
        faiss.write_index(self._index, str(self._memory_index_path))

        logger.info(f"Stored memory {memory_id}: {task_summary[:50]}...")
        return memory_id

    def search_similar(self, query: str, top_k: int = 3) -> list[dict]:
        """Find past tasks semantically similar to the query.

        Args:
            query: Natural language description of a task.
            top_k: Number of results to return.

        Returns:
            List of memory dicts with task_summary, result_summary, etc.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query_vector = self._embedder.encode(query)
        distances, indices = self._index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._embedding_ids):
                continue
            memory_id = self._embedding_ids[idx]
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if row:
                results.append({
                    "id": row["id"],
                    "task_summary": row["task_summary"],
                    "result_summary": row["result_summary"],
                    "tags": json.loads(row["tags"]),
                    "created_at": row["created_at"],
                    "similarity_distance": float(dist),
                })

        return results

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get the most recent memories, ordered by creation time."""
        cursor = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": row["id"],
                "task_summary": row["task_summary"],
                "result_summary": row["result_summary"],
                "tags": json.loads(row["tags"]),
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def get_memory_context(self, current_task: str, top_k: int = 3) -> str:
        """Build a memory context string for the planner agent.

        Combines semantic search (similar past tasks) with recent
        memories to give the planner awareness of past work.

        Args:
            current_task: The current task description.
            top_k: Number of similar memories to retrieve.

        Returns:
            Formatted string for injection into the planner's prompt.
        """
        similar = self.search_similar(current_task, top_k=top_k)
        recent = self.get_recent(limit=3)

        if not similar and not recent:
            return "No relevant past tasks found."

        lines = ["Relevant past tasks:"]
        seen_ids = set()

        for mem in similar:
            if mem["id"] not in seen_ids:
                lines.append(
                    f"- Task: {mem['task_summary']} → Result: {mem['result_summary']}"
                )
                seen_ids.add(mem["id"])

        # Add recent memories not already included
        for mem in recent:
            if mem["id"] not in seen_ids:
                lines.append(
                    f"- (Recent) Task: {mem['task_summary']} → Result: {mem['result_summary']}"
                )
                seen_ids.add(mem["id"])

        return "\n".join(lines)

    def clear(self) -> None:
        """Delete all memories (used in tests and resets)."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        dim = self._embedder.embedding_dim
        self._index = faiss.IndexFlatL2(dim)
        self._embedding_ids = []
        if self._memory_index_path.exists():
            self._memory_index_path.unlink()
        logger.info("Cleared all memories.")

    def __del__(self):
        """Close the SQLite connection on cleanup."""
        try:
            self._conn.close()
        except Exception:
            pass