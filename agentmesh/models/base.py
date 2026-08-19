# file: agentmesh/models/base.py
"""Base classes for model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelResponse:
    """Immutable response from a model call.

    Every model call in the system returns this object. The trajectory
    logger and eval framework read tokens_in/out and latency_ms directly.
    """

    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model_name: str = ""
    raw_output: Optional[str] = field(default=None, repr=False)

class BaseModelProvider(ABC):
    """Abstract interface for language model providers.

    All agents interact with models exclusively through this interface.
    Implementations handle model-specific loading, tokenisation, and
    generation details.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of {"role": str, "content": str} dicts.
                      Roles: "system", "user", "assistant".
            temperature: Override default sampling temperature.
            max_new_tokens: Override default max output length.

        Returns:
            ModelResponse with text, token counts, and latency.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded into memory."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        ...