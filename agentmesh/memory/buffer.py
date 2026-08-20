# file: agentmesh/memory/buffer.py
"""Short-term conversational memory — bounded message buffer."""

from collections import deque
from typing import Optional


class ConversationBuffer:
    """Ring buffer of recent conversation messages.

    Maintains the last N messages for short-term conversational context.
    When the buffer is full, the oldest message is evicted on each add.

    Messages are stored in the OpenAI message format:
    {"role": "user"|"assistant"|"system", "content": "..."}
    """

    def __init__(self, max_size: Optional[int] = None):
        from agentmesh.config import settings

        self._max_size = max_size or settings.conversation_buffer_size
        self._messages: deque[dict] = deque(maxlen=self._max_size)

    def add(self, role: str, content: str) -> None:
        """Add a message to the buffer.

        Args:
            role: "user", "assistant", or "system".
            content: The message text.
        """
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return a copy of all messages in the buffer (oldest first)."""
        return list(self._messages)

    def get_context_string(self) -> str:
        """Format the buffer as a single string for prompt injection.

        Returns a human-readable conversation history string.
        """
        if not self._messages:
            return "No previous conversation."

        lines = []
        for msg in self._messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_last_n(self, n: int) -> list[dict]:
        """Return the last N messages."""
        messages = list(self._messages)
        return messages[-n:] if n < len(messages) else messages

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self._messages.clear()

    @property
    def size(self) -> int:
        """Current number of messages in the buffer."""
        return len(self._messages)

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    def __repr__(self) -> str:
        return f"ConversationBuffer(size={self.size}, max_size={self.max_size})"