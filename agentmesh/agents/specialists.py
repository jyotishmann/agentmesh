# file: agentmesh/agents/specialists.py
"""Specialist sub-agents — Research, Coder, Analyst.

Each specialist receives its own restricted ToolRegistry from the
orchestrator, so the tool list in its prompt (and what it can actually
call) is limited to its role.
"""

import logging

from agentmesh.agents.base import AgentResponse, BaseAgent
from agentmesh.models.manager import ModelManager
from agentmesh.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class _Specialist(BaseAgent):
    """Shared plumbing: specialist model + role prompt + its own tool list."""

    ROLE = ""
    INSTRUCTIONS: list[str] = []

    def __init__(self, model_manager: ModelManager, tool_registry: ToolRegistry):
        super().__init__(model_manager, tool_registry=tool_registry, use_specialist_model=True)

    @property
    def system_prompt(self) -> str:
        tool_desc = self.tool_registry.get_tool_descriptions() if self.tool_registry else ""
        rules = "\n".join(f"- {line}" for line in self.INSTRUCTIONS)
        return f"{self.ROLE}\n\nInstructions:\n{rules}\n\n{tool_desc}"

    def run(self, task: str, context: list[dict] | None = None) -> AgentResponse:
        messages = self._build_messages(task, context)
        return self._run_with_tools(messages, context_for_log=task[:50])


class ResearchAgent(_Specialist):
    """Information gathering and synthesis. Tools: search_web, query_knowledge_base."""

    ROLE = (
        "You are a research specialist agent. Your job is to find information "
        "and synthesise it into clear, accurate answers."
    )
    INSTRUCTIONS = [
        "For current figures (populations, prices, rankings, recent events, "
        "statistics), call search_web before answering, even if you think you know.",
        "Use query_knowledge_base to search local documents.",
        "Answer the question directly first, then add supporting detail.",
        "If you cannot find the answer, say so clearly.",
        "Only use the tools listed below.",
    ]


class CoderAgent(_Specialist):
    """Code writing and execution. Tools: run_python, read_file, write_file."""

    ROLE = (
        "You are a coding specialist agent. Your job is to write, execute, "
        "and debug Python code."
    )
    INSTRUCTIONS = [
        "Write clean Python code and execute it with run_python. Always run it.",
        "Use print() so the result is visible in the output.",
        "If the code fails, read the error and fix it.",
        "Use read_file and write_file for files in the sandbox.",
        "Your final answer must state the actual output returned by run_python.",
        "Only use the tools listed below.",
    ]


class AnalystAgent(_Specialist):
    """Data analysis and summarisation. Tools: read_file, query_knowledge_base, run_python."""

    ROLE = (
        "You are a data analysis specialist agent. Your job is to analyse "
        "data, compute statistics, and produce clear insights."
    )
    INSTRUCTIONS = [
        "Compute every number with run_python instead of estimating it.",
        "Use read_file to load data files and query_knowledge_base for context.",
        "State the key numbers explicitly in your final answer.",
        "Only use the tools listed below.",
    ]
