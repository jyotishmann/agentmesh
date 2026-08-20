# file: agentmesh/agents/specialists.py
"""Specialist sub-agents — Research, Coder, Analyst."""

import logging

from agentmesh.agents.base import AgentResponse, BaseAgent
from agentmesh.models.manager import ModelManager
from agentmesh.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Specialist agent for information gathering and synthesis.

    Tools: search_web, query_knowledge_base
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry,
    ):
        super().__init__(
            model_manager,
            tool_registry=tool_registry,
            use_specialist_model=True,
        )

    @property
    def system_prompt(self) -> str:
        tool_desc = ""
        if self.tool_registry:
            tool_desc = self.tool_registry.get_tool_descriptions()

        return (
            "You are a research specialist agent. Your job is to find "
            "information and synthesise it into clear, accurate summaries.\n\n"
            "Instructions:\n"
            "- Use search_web to find current information.\n"
            "- Use query_knowledge_base to search local documents.\n"
            "- Synthesise findings into a clear answer.\n"
            "- Cite your sources when possible.\n"
            "- If you cannot find the answer, say so clearly.\n"
            "- Only use the tools listed below.\n\n"
            f"{tool_desc}"
        )

    def run(self, task: str, context: list[dict] | None = None) -> AgentResponse:
        """Research a topic using web search and knowledge base."""
        messages = self._build_messages(task, context)
        return self._run_with_tools(messages, context_for_log=task[:50])


class CoderAgent(BaseAgent):
    """Specialist agent for code writing and execution.

    Tools: run_python, read_file, write_file
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry,
    ):
        super().__init__(
            model_manager,
            tool_registry=tool_registry,
            use_specialist_model=True,
        )

    @property
    def system_prompt(self) -> str:
        tool_desc = ""
        if self.tool_registry:
            tool_desc = self.tool_registry.get_tool_descriptions()

        return (
            "You are a coding specialist agent. Your job is to write, "
            "execute, and debug Python code.\n\n"
            "Instructions:\n"
            "- Write clean, well-commented Python code.\n"
            "- Use run_python to test your code. Always test before submitting.\n"
            "- Use read_file to examine existing files.\n"
            "- Use write_file to save code or results.\n"
            "- If code fails, read the error and fix it.\n"
            "- Use print() in your code to produce visible output.\n"
            "- Only use the tools listed below.\n\n"
            f"{tool_desc}"
        )

    def run(self, task: str, context: list[dict] | None = None) -> AgentResponse:
        """Write and execute code for the given task."""
        messages = self._build_messages(task, context)
        return self._run_with_tools(messages, context_for_log=task[:50])

class AnalystAgent(BaseAgent):
    """Specialist agent for data analysis and summarisation.

    Tools: read_file, query_knowledge_base, run_python
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry,
    ):
        super().__init__(
            model_manager,
            tool_registry=tool_registry,
            use_specialist_model=True,
        )

    @property
    def system_prompt(self) -> str:
        tool_desc = ""
        if self.tool_registry:
            tool_desc = self.tool_registry.get_tool_descriptions()

        return (
            "You are a data analysis specialist agent. Your job is to "
            "analyse data, compute statistics, and produce clear insights.\n\n"
            "Instructions:\n"
            "- Use read_file to load data files.\n"
            "- Use run_python to compute statistics and generate analysis.\n"
            "- Use query_knowledge_base for domain context.\n"
            "- Present results clearly with key findings highlighted.\n"
            "- Include numbers and specific data points in your analysis.\n"
            "- Only use the tools listed below.\n\n"
            f"{tool_desc}"
        )

    def run(self, task: str, context: list[dict] | None = None) -> AgentResponse:
        """Analyse data and produce insights."""
        messages = self._build_messages(task, context)
        return self._run_with_tools(messages, context_for_log=task[:50])