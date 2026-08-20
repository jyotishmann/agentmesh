# file: agentmesh/agents/base.py
"""Base agent class with shared tool-call parsing logic."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agentmesh.models.base import ModelResponse
from agentmesh.models.manager import ModelManager
from agentmesh.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent run.

    Captures the output text, tool call history, token costs,
    and completion status.
    """

    output: str
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    total_latency_ms: float = 0.0
    completed: bool = True
    error: Optional[str] = None

class BaseAgent:
    """Abstract base class for all agents.

    Subclasses must define:
    - system_prompt: The agent's identity and instructions
    - run(): The main execution method
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: Optional[ToolRegistry] = None,
        use_specialist_model: bool = False,
    ):
        self.model_manager = model_manager
        self.tool_registry = tool_registry
        self.use_specialist = use_specialist_model
        self._max_tool_calls: int = 5

        from agentmesh.config import settings
        self._max_tool_calls = settings.max_tool_calls_per_agent

    @property
    def name(self) -> str:
        """Agent class name for logging and trajectory."""
        return self.__class__.__name__

    @property
    def system_prompt(self) -> str:
        """System prompt defining this agent's identity. Must override."""
        raise NotImplementedError

    def _build_messages(
        self,
        instruction: str,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Build the full message list for a model call.

        Structure: [system prompt] + [context messages] + [user instruction]
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[dict]:
        """Extract a tool call from the model's output.

        Looks for: <tool_call>{"tool": "name", "args": {...}}</tool_call>

        Returns:
            Dict with "tool" and "args" keys, or None if no valid
            tool call found.
        """
        # Primary pattern: full XML tags
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            # Fallback: opening tag without closing (model truncation)
            pattern_open = r"<tool_call>\s*(.*)"
            match = re.search(pattern_open, text, re.DOTALL)

        if not match:
            return None

        json_str = match.group(1).strip()

        try:
            parsed = json.loads(json_str)
            if "tool" in parsed and "args" in parsed:
                return parsed
            logger.warning(f"Tool call JSON missing 'tool' or 'args': {parsed}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}\nRaw: {json_str}")
            return None


    def _run_with_tools(
        self,
        messages: list[dict],
        context_for_log: str = "",
    ) -> AgentResponse:
        """Execute the agent's tool-use loop.

        Repeatedly: generate → parse tool call → execute tool → inject
        result → generate again. Stops when the model produces output
        without a tool call, or the tool call limit is reached.

        Args:
            messages: Initial message list (including system prompt).
            context_for_log: Description for logging.

        Returns:
            AgentResponse with output, tool call history, and token stats.
        """
        tool_calls = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency = 0.0
        conversation = list(messages)

        for step in range(self._max_tool_calls + 1):
            # Generate model response
            response: ModelResponse = self.model_manager.generate(
                conversation,
                use_specialist=self.use_specialist,
            )
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_latency += response.latency_ms

            # Check for tool call
            tool_call = self._parse_tool_call(response.text)

            if tool_call is None:
                # No tool call — model is done reasoning
                return AgentResponse(
                    output=response.text,
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                )

            # Check tool call limit
            if step >= self._max_tool_calls:
                logger.warning(
                    f"{self.name}: Hit tool call limit ({self._max_tool_calls})"
                )
                return AgentResponse(
                    output=response.text,
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                    completed=False,
                    error=f"Tool call limit ({self._max_tool_calls}) reached.",
                )

            # Dispatch the tool call
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            if self.tool_registry is None:
                tool_result = f"Error: No tool registry available."
            else:
                tool_result = self.tool_registry.call(tool_name, tool_args)

            # Record the tool call
            tool_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result[:500],  # truncate for logging
            })

            logger.info(
                f"{self.name}: Called {tool_name}({tool_args}) → "
                f"{tool_result[:100]}..."
            )

            # Inject model output and tool result into the conversation
            conversation.append({
                "role": "assistant",
                "content": response.text,
            })
            conversation.append({
                "role": "user",
                "content": (
                    f"[Tool Result: {tool_name}]\n{tool_result}"
                ),
            })

        # Should not reach here, but safety fallback
        return AgentResponse(
            output="Agent loop ended without producing a final response.",
            tool_calls=tool_calls,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            total_latency_ms=round(total_latency, 2),
            completed=False,
        )