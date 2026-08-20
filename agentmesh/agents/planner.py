# file: agentmesh/agents/planner.py
"""Planner agent — breaks tasks into sub-task plans."""

import json
import logging
import re
from typing import Optional

from agentmesh.agents.base import AgentResponse, BaseAgent
from agentmesh.models.manager import ModelManager

logger = logging.getLogger(__name__)

# The expected output format for the planner
PLAN_FORMAT_EXAMPLE = """
[
    {
        "description": "Search for recent advances in quantum computing",
        "specialist": "research",
        "required_tools": ["search_web"]
    },
    {
        "description": "Write a summary report of the findings",
        "specialist": "analyst",
        "required_tools": ["write_file"]
    }
]
"""


class PlannerAgent(BaseAgent):
    """Breaks user tasks into structured sub-task plans.

    Outputs a JSON list of sub-tasks, each assigned to a specialist
    agent with specified tool requirements.
    """

    def __init__(self, model_manager: ModelManager):
        # Planner uses the main (larger) model, no tools
        super().__init__(model_manager, tool_registry=None, use_specialist_model=False)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a task planning agent. Your job is to break down a user's "
            "task into a structured plan of sub-tasks.\n\n"
            "For each sub-task, specify:\n"
            '- "description": What needs to be done\n'
            '- "specialist": Which agent handles it — one of: "research", "coder", "analyst"\n'
            '- "required_tools": List of tools needed\n\n'
            "Available specialists and their tools:\n"
            "- research: search_web, query_knowledge_base\n"
            "- coder: run_python, read_file, write_file\n"
            "- analyst: read_file, query_knowledge_base, run_python\n\n"
            "Output ONLY a valid JSON array. No markdown, no explanation.\n\n"
            f"Example output:\n{PLAN_FORMAT_EXAMPLE}\n"
            "Rules:\n"
            "- Keep plans concise (1-4 sub-tasks).\n"
            "- Each sub-task should be self-contained.\n"
            "- Simple tasks need only 1 sub-task.\n"
            "- Never create more sub-tasks than necessary.\n"
        )

    def run(
        self,
        task: str,
        memory_context: str = "",
    ) -> AgentResponse:
        """Generate a plan for the given task.

        Args:
            task: The user's task description.
            memory_context: Formatted string of relevant past tasks.

        Returns:
            AgentResponse where output is a JSON string of the plan.
        """
        instruction = task
        if memory_context and memory_context != "No relevant past tasks found.":
            instruction = (
                f"Context from past tasks:\n{memory_context}\n\n"
                f"Current task: {task}"
            )

        messages = self._build_messages(instruction)
        response = self.model_manager.generate(
            messages,
            use_specialist=False,
            temperature=0.3,  # Low temp for structured output
        )

        # Try to parse the plan
        plan = self._parse_plan(response.text)

        if plan is None:
            # Retry once with a correction prompt
            logger.warning("Planner output invalid JSON. Retrying with correction.")
            messages.append({"role": "assistant", "content": response.text})
            messages.append({
                "role": "user",
                "content": (
                    "Your output was not valid JSON. Please output ONLY a "
                    "JSON array of sub-tasks. No explanation, no markdown."
                ),
            })
            retry_response = self.model_manager.generate(
                messages,
                use_specialist=False,
                temperature=0.1,
            )
            plan = self._parse_plan(retry_response.text)

            response_tokens_in = response.tokens_in + retry_response.tokens_in
            response_tokens_out = response.tokens_out + retry_response.tokens_out
            response_latency = response.latency_ms + retry_response.latency_ms
        else:
            response_tokens_in = response.tokens_in
            response_tokens_out = response.tokens_out
            response_latency = response.latency_ms

        if plan is None:
            # Final fallback: single research sub-task
            logger.warning("Planner failed to produce valid JSON. Using fallback plan.")
            plan = [
                {
                    "description": task,
                    "specialist": "research",
                    "required_tools": ["search_web"],
                }
            ]

        return AgentResponse(
            output=json.dumps(plan),
            tokens_in=response_tokens_in,
            tokens_out=response_tokens_out,
            total_latency_ms=round(response_latency, 2),
        )

    @staticmethod
    def _parse_plan(text: str) -> Optional[list[dict]]:
        """Parse the planner's output into a list of sub-tasks.

        Handles raw JSON, JSON inside markdown code blocks, and
        partial JSON extraction.
        """
        text = text.strip()

        # Try direct JSON parse
        try:
            plan = json.loads(text)
            if isinstance(plan, list) and all(isinstance(t, dict) for t in plan):
                return plan
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if code_block:
            try:
                plan = json.loads(code_block.group(1))
                if isinstance(plan, list):
                    return plan
            except json.JSONDecodeError:
                pass

        # Try extracting array from text
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            try:
                plan = json.loads(array_match.group(0))
                if isinstance(plan, list):
                    return plan
            except json.JSONDecodeError:
                pass

        return None