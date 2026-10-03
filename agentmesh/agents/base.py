# file: agentmesh/agents/base.py
"""Base agent class with shared tool-call parsing and the tool-use loop."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agentmesh.config import settings
from agentmesh.models.base import ModelResponse
from agentmesh.models.manager import ModelManager
from agentmesh.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent run: output, tool history, tokens, status."""

    output: str
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    total_latency_ms: float = 0.0
    completed: bool = True
    error: Optional[str] = None


class BaseAgent:
    """Base class for all agents. Subclasses define system_prompt and run()."""

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: Optional[ToolRegistry] = None,
        use_specialist_model: bool = False,
    ):
        self.model_manager = model_manager
        self.tool_registry = tool_registry
        self.use_specialist = use_specialist_model
        self._max_tool_calls = settings.max_tool_calls_per_agent

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def _build_messages(
        self,
        instruction: str,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """[system prompt] + [context messages] + [user instruction]."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[dict]:
        """Extract the first tool call from model output.

        Accepts tagged (<tool_call>...</tool_call>) or bare JSON, and both
        key conventions: {"tool", "args"} or Qwen-native {"name", "arguments"}.
        Always returns the normalised form {"tool": str, "args": dict}.
        """
        tagged = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.DOTALL)
        search_space = tagged.group(1) if tagged else text

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", search_space):
            try:
                obj, _ = decoder.raw_decode(search_space, match.start())
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            name = obj.get("tool") or obj.get("name")
            args = obj.get("args", obj.get("arguments"))
            if isinstance(args, str):  # some models emit arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(name, str) and isinstance(args, dict):
                return {"tool": name, "args": args}

        return None

    def _run_with_tools(
        self,
        messages: list[dict],
        context_for_log: str = "",
    ) -> AgentResponse:
        """generate -> parse tool call -> execute -> inject result -> repeat."""
        tool_calls: list[dict] = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency = 0.0
        conversation = list(messages)
        nudged = False  # one-time reminder when code is written but not executed

        for step in range(self._max_tool_calls + 1):
            response: ModelResponse = self.model_manager.generate(
                conversation,
                use_specialist=self.use_specialist,
            )
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_latency += response.latency_ms

            tool_call = self._parse_tool_call(response.text)

            if tool_call is None:
                if (
                    not nudged
                    and not tool_calls
                    and "```python" in response.text
                    and self.tool_registry is not None
                    and "run_python" in self.tool_registry.list_tools()
                ):
                    nudged = True
                    conversation.append({"role": "assistant", "content": response.text})
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You wrote code but did not execute it. Call run_python "
                            "now with that code, then report the actual output."
                        ),
                    })
                    continue

                return AgentResponse(
                    output=response.text,
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                )

            if step >= self._max_tool_calls:
                logger.warning(f"{self.name}: Hit tool call limit ({self._max_tool_calls})")
                return AgentResponse(
                    output=response.text,
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                    completed=False,
                    error=f"Tool call limit ({self._max_tool_calls}) reached.",
                )

            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            if self.tool_registry is None:
                tool_result = "Error: No tool registry available."
            else:
                tool_result = self.tool_registry.call(tool_name, tool_args)

            tool_calls.append({"tool": tool_name, "args": tool_args, "result": tool_result[:500]})
            logger.info(f"{self.name}: Called {tool_name}({tool_args}) → {tool_result[:100]}...")

            conversation.append({"role": "assistant", "content": response.text})
            conversation.append({
                "role": "user",
                "content": f"[Tool Result: {tool_name}]\n{tool_result}",
            })

        return AgentResponse(
            output="Agent loop ended without producing a final response.",
            tool_calls=tool_calls,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            total_latency_ms=round(total_latency, 2),
            completed=False,
        )
