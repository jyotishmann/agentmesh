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

# Matches ```python ... ``` or ```py ... ``` fenced blocks
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)


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

    @staticmethod
    def _extract_code_block(text: str) -> Optional[str]:
        """Return the last ```python fenced block in the text, if any."""
        blocks = _CODE_BLOCK_RE.findall(text)
        return blocks[-1].strip() if blocks else None

    @staticmethod
    def _ground_in_execution(text: str, exec_output: str) -> str:
        """Append real sandbox output to an answer that doesn't report it.

        Small models often finish by repeating their code instead of stating
        what it printed. If the most informative of the first few output
        lines isn't already in the answer, the genuine output is appended.
        The appended text comes from the sandbox, so it cannot be invented.
        """
        if not exec_output or not exec_output.strip():
            return text

        body = exec_output.strip()
        if body.startswith("STDOUT:"):
            body = body[len("STDOUT:"):].strip()

        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return text

        # Short lines like "2" appear in almost any text, so only trust a
        # line of 3+ characters as evidence that the answer reports output.
        key_line = max(lines[:5], key=len)
        if len(key_line) >= 3 and key_line in text:
            return text

        return f"{text.rstrip()}\n\n**Execution output:**\n```\n{body[:1500]}\n```"

    def _can_run_python(self) -> bool:
        return (
            self.tool_registry is not None
            and "run_python" in self.tool_registry.list_tools()
        )

    def _run_with_tools(
        self,
        messages: list[dict],
        context_for_log: str = "",
    ) -> AgentResponse:
        """generate -> parse tool call -> execute -> inject result -> repeat.

        If the model writes a ```python block instead of a formal tool call,
        and this agent may use run_python, the block is executed implicitly.
        Identical code is never executed twice, which ends the loop when the
        model's final answer repeats the code alongside its output.
        """
        tool_calls: list[dict] = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency = 0.0
        conversation = list(messages)
        executed_code: set[str] = set()
        last_exec_output = ""

        for step in range(self._max_tool_calls + 1):
            response: ModelResponse = self.model_manager.generate(
                conversation,
                use_specialist=self.use_specialist,
            )
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_latency += response.latency_ms

            tool_call = self._parse_tool_call(response.text)
            implicit = False

            # Code-as-action fallback: run a fenced code block the model didn't wrap
            if tool_call is None and self._can_run_python():
                code = self._extract_code_block(response.text)
                if code and code not in executed_code:
                    tool_call = {"tool": "run_python", "args": {"code": code}}
                    implicit = True

            if tool_call is None:
                # No tool call — model is done reasoning
                return AgentResponse(
                    output=self._ground_in_execution(response.text, last_exec_output),
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                )

            if step >= self._max_tool_calls:
                logger.warning(f"{self.name}: Hit tool call limit ({self._max_tool_calls})")
                return AgentResponse(
                    output=self._ground_in_execution(response.text, last_exec_output),
                    tool_calls=tool_calls,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_latency_ms=round(total_latency, 2),
                    completed=False,
                    error=f"Tool call limit ({self._max_tool_calls}) reached.",
                )

            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            if tool_name == "run_python":
                executed_code.add(str(tool_args.get("code", "")).strip())

            if self.tool_registry is None:
                tool_result = "Error: No tool registry available."
            else:
                tool_result = self.tool_registry.call(tool_name, tool_args)

            if tool_name == "run_python":
                last_exec_output = tool_result

            tool_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result[:500],
                "implicit": implicit,
            })
            verb = "Implicitly ran" if implicit else "Called"
            logger.info(f"{self.name}: {verb} {tool_name} → {tool_result[:100]}...")

            conversation.append({"role": "assistant", "content": response.text})
            conversation.append({
                "role": "user",
                "content": f"[Tool Result: {tool_name}]\n{tool_result}",
            })

        return AgentResponse(
            output=self._ground_in_execution(
                "Agent loop ended without producing a final response.", last_exec_output
            ),
            tool_calls=tool_calls,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            total_latency_ms=round(total_latency, 2),
            completed=False,
        )
