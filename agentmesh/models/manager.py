# file: agentmesh/models/manager.py
"""Unified model access — single entry point for all model operations."""

from typing import Optional

import numpy as np

from agentmesh.config import settings
from agentmesh.models.base import ModelResponse
from agentmesh.models.embeddings import EmbeddingProvider
from agentmesh.models.qwen import QwenModelProvider


class ModelManager:
    """Manages all model providers and tracks cumulative token usage.

    Usage:
        mm = ModelManager()
        resp = mm.generate([{"role": "user", "content": "Hello"}])
        embeddings = mm.embed("some text to embed")
    """

    def __init__(self):
        self._main_provider = QwenModelProvider(settings.model_name)
        self._specialist_provider = QwenModelProvider(settings.specialist_model_name)
        self._embedding_provider = EmbeddingProvider(settings.embedding_model_name)

        # Cumulative token counters for cost tracking
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def generate(
        self,
        messages: list[dict],
        use_specialist: bool = False,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response using the main or specialist model.

        Args:
            messages: Chat messages in OpenAI format.
            use_specialist: If True, use the smaller specialist model.
            temperature: Override sampling temperature.
            max_new_tokens: Override max output tokens.

        Returns:
            ModelResponse with text, token counts, and latency.
        """
        provider = (
            self._specialist_provider if use_specialist else self._main_provider
        )
        response = provider.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        # Track cumulative tokens
        self._total_tokens_in += response.tokens_in
        self._total_tokens_out += response.tokens_out

        return response

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embedding vectors."""
        return self._embedding_provider.encode(texts)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._embedding_provider.embedding_dim

    def get_token_stats(self) -> dict:
        """Return cumulative token usage across all calls."""
        return {
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "total_tokens": self._total_tokens_in + self._total_tokens_out,
        }

    def reset_token_stats(self) -> None:
        """Reset cumulative token counters (used between eval tasks)."""
        self._total_tokens_in = 0
        self._total_tokens_out = 0