# file: agentmesh/agents/critic.py
"""Critic agent — evaluates output quality and requests revisions."""

import json
import logging
import re
from typing import Optional

from agentmesh.agents.base import AgentResponse, BaseAgent
from agentmesh.models.manager import ModelManager

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Evaluates agent output and issues pass/fail verdicts.

    No tool access — the critic only reads and judges.
    Uses the main (larger) model for stronger evaluation.
    """

    def __init__(self, model_manager: ModelManager):
        # Critic uses the main model, no tools
        super().__init__(model_manager, tool_registry=None, use_specialist_model=False)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a strict quality critic. Your job is to evaluate "
            "whether an agent's output correctly and completely addresses "
            "the original task.\n\n"
            "Evaluate for:\n"
            "1. Correctness: Are facts accurate? Does code work?\n"
            "2. Completeness: Is the full task addressed?\n"
            "3. Quality: Is the output clear and well-structured?\n"
            "4. Relevance: Does the output answer what was asked?\n\n"
            "Output ONLY a JSON object with these fields:\n"
            '- "pass": true if the output is acceptable, false otherwise\n'
            '- "confidence": float from 0.0 to 1.0\n'
            '- "feedback": string explaining your assessment\n\n'
            "Example:\n"
            '{"pass": true, "confidence": 0.85, "feedback": "The summary '
            'is accurate and covers the main points."}\n\n'
            "Be strict. Reject vague, incomplete, or incorrect output. "
            "If unsure, err on the side of rejection."
        )

    def run(self, task: str, output: str) -> AgentResponse:
        """Evaluate the output against the original task.

        Args:
            task: The original user task.
            output: The assembled output from specialist agents.

        Returns:
            AgentResponse where output is a JSON verdict string.
        """
        instruction = (
            f"Original task:\n{task}\n\n"
            f"Agent output:\n{output}\n\n"
            "Evaluate this output. Respond with ONLY a JSON verdict."
        )

        messages = self._build_messages(instruction)
        response = self.model_manager.generate(
            messages,
            use_specialist=False,
            temperature=0.2,  # Low temp for consistent evaluation
        )

        # Parse the verdict
        verdict = self._parse_verdict(response.text)

        if verdict is None:
            # Retry with correction
            messages.append({"role": "assistant", "content": response.text})
            messages.append({
                "role": "user",
                "content": (
                    "Your output was not valid JSON. Respond with ONLY:\n"
                    '{"pass": true/false, "confidence": 0.0-1.0, "feedback": "..."}'
                ),
            })
            retry = self.model_manager.generate(
                messages, use_specialist=False, temperature=0.1
            )
            verdict = self._parse_verdict(retry.text)

            total_in = response.tokens_in + retry.tokens_in
            total_out = response.tokens_out + retry.tokens_out
            total_latency = response.latency_ms + retry.latency_ms
        else:
            total_in = response.tokens_in
            total_out = response.tokens_out
            total_latency = response.latency_ms

        if verdict is None:
            # Final fallback: pass with warning
            logger.warning("Critic failed to produce valid JSON. Defaulting to pass.")
            verdict = {
                "pass": True,
                "confidence": 0.5,
                "feedback": "Critic could not produce structured evaluation.",
            }

        return AgentResponse(
            output=json.dumps(verdict),
            tokens_in=total_in,
            tokens_out=total_out,
            total_latency_ms=round(total_latency, 2),
        )

    @staticmethod
    def _parse_verdict(text: str) -> Optional[dict]:
        """Parse the critic's JSON verdict from output text."""
        text = text.strip()

        # Try direct parse
        try:
            v = json.loads(text)
            if "pass" in v:
                v.setdefault("confidence", 0.5)
                v.setdefault("feedback", "")
                return v
        except json.JSONDecodeError:
            pass

        # Try code block extraction
        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if code_block:
            try:
                v = json.loads(code_block.group(1))
                if "pass" in v:
                    v.setdefault("confidence", 0.5)
                    v.setdefault("feedback", "")
                    return v
            except json.JSONDecodeError:
                pass

        # Try extracting object from text
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                v = json.loads(obj_match.group(0))
                if "pass" in v:
                    v.setdefault("confidence", 0.5)
                    v.setdefault("feedback", "")
                    return v
            except json.JSONDecodeError:
                pass

        return None