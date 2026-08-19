# file: agentmesh/models/embeddings.py
"""Embedding provider using sentence-transformers."""

from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from agentmesh.config import settings


class EmbeddingProvider:
    """Generates dense vector embeddings for text.

    Uses sentence-transformers with BAAI/bge-small-en-v1.5 by default.
    Lazy-loads the model on first encode() call.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.embedding_model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure_loaded(self) -> None:
        """Load the embedding model if not already loaded."""
        if self._model is not None:
            return
        # Force CPU — embedding model is tiny, no need for GPU
        self._model = SentenceTransformer(self._model_name, device="cpu")

    def encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) into dense vectors.

        Args:
            texts: Single string or list of strings to embed.
            normalize: L2-normalize vectors (enables cosine similarity
                       via L2 distance in FAISS).

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        return self._model is not None