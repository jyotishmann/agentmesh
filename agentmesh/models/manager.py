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

    deterministic=True forces greedy decoding (temperature 0) on every call,
    overriding any per-agent temperature. The eval runner switches it on so
    the same code on the same tasks gives the same answers; interactive chat
    keeps it off and samples normally.
    """

    def __init__(self, deterministic: bool = False):
        self._main_provider = QwenModelProvider(settings.main_model_name)
        self._specialist_provider = QwenModelProvider(settings.specialist_model_name)
        self._embedding_provider = EmbeddingProvider(settings.embedding_model_name)

        self.deterministic = deterministic

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
        """Generate a response using the main or specialist model."""
        if self.deterministic:
            temperature = 0.0  # greedy decoding: same input, same output

        provider = self._specialist_provider if use_specialist else self._main_provider
        response = provider.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

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
        """Return cumulative token usage since the last reset."""
        return {
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "total_tokens": self._total_tokens_in + self._total_tokens_out,
        }

    def reset_token_stats(self) -> None:
        """Reset cumulative token counters (called at the start of each task)."""
        self._total_tokens_in = 0
        self._total_tokens_out = 0
