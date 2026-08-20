# file: agentmesh/tools/registry.py
"""Tool registration system — decorator, registry, and dispatch."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Definition of a registered tool.

    Stored in the registry and used to generate the tool description
    block that goes into agent system prompts.
    """

    name: str
    description: str
    parameters: dict[str, dict[str, str]]  # param_name -> {type, description}
    func: Callable[..., str] = field(repr=False)

def tool(
    name: str,
    description: str,
    parameters: dict[str, dict[str, str]],
) -> Callable:
    """Decorator that marks a function as a tool.

    Usage:
        @tool(
            name="search_web",
            description="Search the web for information",
            parameters={"query": {"type": "str", "description": "Search query"}}
        )
        def search_web(query: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        func._tool_name = name
        func._tool_description = description
        func._tool_parameters = parameters
        return func

    return decorator

class ToolRegistry:
    """Registry that collects, stores, and dispatches tool calls.

    Tools are registered via the @tool decorator. The registry provides
    dispatch (name → function call) and prompt formatting (generating
    the tool description block for system prompts).
    """

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, func: Callable) -> None:
        """Register a decorated tool function."""
        if not hasattr(func, "_tool_name"):
            raise ValueError(f"{func.__name__} is not decorated with @tool")

        tool_def = ToolDef(
            name=func._tool_name,
            description=func._tool_description,
            parameters=func._tool_parameters,
            func=func,
        )
        self._tools[tool_def.name] = tool_def
        logger.info(f"Registered tool: {tool_def.name}")

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Call a tool by name with the given arguments.

        Returns the tool's string output. Never raises — errors are
        returned as descriptive error strings.
        """
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'. Available tools: {list(self._tools.keys())}"

        tool_def = self._tools[name]

        try:
            result = tool_def.func(**args)
            return str(result)
        except Exception as e:
            error_msg = f"Error calling {name}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def list_tools(self) -> list[str]:
        """Return a list of registered tool names."""
        return list(self._tools.keys())

    def get_tool_def(self, name: str) -> ToolDef | None:
        """Return the ToolDef for a given tool name, or None."""
        return self._tools.get(name)

    def get_tool_descriptions(self) -> str:
        """Generate a formatted tool description block for system prompts.

        Returns a string describing all registered tools and the
        expected format for tool calls.
        """
        if not self._tools:
            return "No tools available."

        lines = ["You have access to the following tools:\n"]

        for tool_def in self._tools.values():
            lines.append(f"### {tool_def.name}")
            lines.append(f"Description: {tool_def.description}")
            lines.append("Parameters:")
            for param_name, param_info in tool_def.parameters.items():
                ptype = param_info.get("type", "str")
                pdesc = param_info.get("description", "")
                lines.append(f"  - {param_name} ({ptype}): {pdesc}")
            lines.append("")

        lines.append("To use a tool, output a tool call in this exact format:")
        lines.append("<tool_call>")
        lines.append('{"tool": "tool_name", "args": {"param1": "value1"}}')
        lines.append("</tool_call>")
        lines.append("")
        lines.append("You may include reasoning before the tool call.")
        lines.append("After receiving the tool result, continue your reasoning.")
        lines.append("Only call one tool at a time. Wait for the result before calling another.")

        return "\n".join(lines)    