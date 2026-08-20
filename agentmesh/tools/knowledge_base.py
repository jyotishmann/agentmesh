# file: agentmesh/tools/knowledge_base.py
"""Knowledge base tool using FAISS vector search."""

import logging
from pathlib import Path
from typing import Optional

import faiss # type: ignore
import numpy as np

from agentmesh.config import settings
from agentmesh.models.embeddings import EmbeddingProvider
from agentmesh.tools.registry import tool

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """FAISS-backed knowledge base with document chunking.

    Lazily loads or builds the index on first query.
    """

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self._embedder = embedding_provider or EmbeddingProvider()
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks_metadata: list[dict] = []  # parallel to FAISS vectors
        self._is_built = False

    def _chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return [c for c in chunks if c.strip()]

    def build_index(self) -> None:
        """Build the FAISS index from documents in the knowledge base directory."""
        kb_dir = settings.knowledge_base_dir
        if not kb_dir.exists():
            kb_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created empty knowledge base directory: {kb_dir}")
            self._is_built = True
            return

        # Collect all text files
        all_chunks = []
        for file_path in sorted(kb_dir.glob("**/*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": str(file_path.relative_to(kb_dir)),
                    "chunk_index": i,
                })

        # Also index .md files
        for file_path in sorted(kb_dir.glob("**/*.md")):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": str(file_path.relative_to(kb_dir)),
                    "chunk_index": i,
                })

        if not all_chunks:
            logger.info("No documents found in knowledge base directory.")
            self._is_built = True
            return

        # Embed all chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = self._embedder.encode(texts)

        # Build FAISS index (brute-force L2 — correct for small collections)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        self._chunks_metadata = all_chunks

        # Save index to disk
        faiss.write_index(self._index, str(settings.faiss_index_path))

        logger.info(
            f"Built knowledge base index: {len(all_chunks)} chunks "
            f"from {kb_dir}"
        )
        self._is_built = True

    def _ensure_loaded(self) -> None:
        """Load or build the index."""
        if self._is_built:
            return

        index_path = settings.faiss_index_path
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")
            self._is_built = True
        else:
            self.build_index()

    def query(self, query_text: str, top_k: Optional[int] = None) -> str:
        """Search the knowledge base and return matching chunks.

        Args:
            query_text: Natural language query.
            top_k: Number of results to return (default from settings).

        Returns:
            Formatted string with matching chunks and sources.
        """
        self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return "Knowledge base is empty. No documents have been indexed."

        k = top_k or settings.knowledge_base_top_k
        k = min(k, self._index.ntotal)

        # Embed the query
        query_vector = self._embedder.encode(query_text)

        # Search
        distances, indices = self._index.search(query_vector, k)

        results = []
        for i, (dist, idx) in enumerate(
            zip(distances[0], indices[0]), 1
        ):
            if idx == -1:
                continue
            meta = self._chunks_metadata[idx]
            results.append(
                f"{i}. [Source: {meta['source']}, Chunk {meta['chunk_index']}] "
                f"(distance: {dist:.4f})\n{meta['text']}"
            )

        return "\n\n".join(results) if results else "No relevant results found."


# Module-level instance (shared across tool calls)
_kb = KnowledgeBase()


@tool(
    name="query_knowledge_base",
    description="Search the local knowledge base for relevant information. "
    "The knowledge base contains indexed documents. Returns the most "
    "semantically similar text chunks to your query.",
    parameters={
        "query": {
            "type": "str",
            "description": "Natural language query to search for",
        }
    },
)
def query_knowledge_base(query: str) -> str:
    """Search the knowledge base and return relevant chunks."""
    if not query or not query.strip():
        return "Error: Empty query."

    try:
        return _kb.query(query)
    except Exception as e:
        error_msg = f"Error: Knowledge base query failed — {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return error_msg