# file: agentmesh/orchestrator.py
"""Orchestrator — the central agent loop coordinating all components."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from agentmesh.agents import (
    AgentResponse,
    AnalystAgent,
    CoderAgent,
    CriticAgent,
    PlannerAgent,
    ResearchAgent,
)
from agentmesh.config import settings
from agentmesh.memory import ConversationBuffer, PersistentMemory
from agentmesh.models import ModelManager
from agentmesh.tools import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Final result from a task execution."""

    response: str
    task_id: str
    completed: bool = True
    token_summary: dict = field(default_factory=dict)
    trajectory_events: list[dict] = field(default_factory=list)


class _LoopDetector:
    """Detects repeated tool calls with identical arguments.

    Tracks (tool_name, hash(args)) pairs. Returns True if a pair
    has been seen before, indicating a potential infinite loop.
    """

    def __init__(self):
        self._seen: dict[str, str] = {}  # key -> previous result

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Check if this tool+args combination has been called before.

        Returns:
            (is_duplicate, previous_result_or_empty)
        """
        key = f"{tool_name}:{self._hash_args(args)}"
        if key in self._seen:
            return True, self._seen[key]
        return False, ""

    def record(self, tool_name: str, args: dict, result: str) -> None:
        """Record a tool call for future duplicate detection."""
        key = f"{tool_name}:{self._hash_args(args)}"
        self._seen[key] = result[:500]

    @staticmethod
    def _hash_args(args: dict) -> str:
        """Produce a stable hash of the arguments dict."""
        serialised = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

class Orchestrator:
    """Central agent loop — coordinates planning, execution, and evaluation.

    The run() method executes a full task lifecycle:
    plan → specialist execution → critic evaluation → revision if needed.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[PersistentMemory] = None,
    ):
        self.model_manager = model_manager or ModelManager()
        self.tool_registry = tool_registry or create_default_registry()
        self.memory = memory or PersistentMemory()

        # Create agents
        self.planner = PlannerAgent(self.model_manager)
        self.critic = CriticAgent(self.model_manager)

        self.specialists = {
            "research": ResearchAgent(self.model_manager, self.tool_registry),
            "coder": CoderAgent(self.model_manager, self.tool_registry),
            "analyst": AnalystAgent(self.model_manager, self.tool_registry),
        }

        # Session management
        self._sessions: dict[str, ConversationBuffer] = {}

    def run(
        self,
        task: str,
        session_id: str = "",
    ) -> OrchestratorResult:
        """Execute a full task lifecycle.

        Args:
            task: The user's natural-language task.
            session_id: Optional session ID for conversation continuity.

        Returns:
            OrchestratorResult with response, task_id, and trajectory.
        """
        task_id = f"task_{uuid4().hex[:12]}"
        session_id = session_id or f"session_{uuid4().hex[:8]}"
        events: list[dict] = []
        step = 0
        total_tool_calls = 0

        # Reset per-task token tracking
        self.model_manager.reset_token_stats()

        # Get conversation context
        buffer = self._get_session(session_id)
        buffer.add("user", task)

        # Load long-term memory context
        memory_context = self.memory.get_memory_context(task)

        logger.info(f"[{task_id}] Starting task: {task[:80]}...")

        # ── Phase 1: Planning ──────────────────────────────────────
        plan_response = self.planner.run(task, memory_context)
        step += 1
        self._log_event(
            events, step, "PlannerAgent", "plan",
            tool_output=plan_response.output,
            tokens_in=plan_response.tokens_in,
            tokens_out=plan_response.tokens_out,
            latency_ms=plan_response.total_latency_ms,
        )

        try:
            sub_tasks = json.loads(plan_response.output)
        except json.JSONDecodeError:
            sub_tasks = [{"description": task, "specialist": "research", "required_tools": []}]

        logger.info(f"[{task_id}] Plan: {len(sub_tasks)} sub-tasks")

        # ── Phase 2: Specialist Execution ──────────────────────────
        loop_detector = _LoopDetector()
        sub_task_outputs: list[str] = []
        hit_limit = False

        for i, sub_task in enumerate(sub_tasks):
            if hit_limit:
                break

            specialist_name = sub_task.get("specialist", "research")
            description = sub_task.get("description", task)

            agent = self.specialists.get(specialist_name)
            if agent is None:
                logger.warning(f"Unknown specialist '{specialist_name}', falling back to research")
                agent = self.specialists["research"]

            logger.info(f"[{task_id}] Sub-task {i+1}: {specialist_name} — {description[:60]}")

            # Run the specialist
            agent_response = agent.run(description)
            step += 1

            # Log each tool call with loop detection
            for tc in agent_response.tool_calls:
                total_tool_calls += 1

                # Loop detection
                is_dup, prev_result = loop_detector.check(tc["tool"], tc["args"])
                if is_dup:
                    logger.warning(
                        f"[{task_id}] Loop detected: {tc['tool']}({tc['args']}) "
                        f"called with same args before."
                    )
                    step += 1
                    self._log_event(
                        events, step, agent.name, "loop_detected",
                        tool_name=tc["tool"],
                        tool_input=json.dumps(tc["args"]),
                        metadata={"previous_result": prev_result},
                    )
                else:
                    loop_detector.record(tc["tool"], tc["args"], tc.get("result", ""))

                step += 1
                self._log_event(
                    events, step, agent.name, "tool_call",
                    tool_name=tc["tool"],
                    tool_input=json.dumps(tc["args"]),
                    tool_output=tc.get("result", ""),
                )

                # Check total tool call limit
                if total_tool_calls >= settings.max_total_tool_calls:
                    logger.warning(f"[{task_id}] Total tool call limit reached.")
                    hit_limit = True
                    break

            # Log the specialist's output
            step += 1
            self._log_event(
                events, step, agent.name, "agent_output",
                tool_output=agent_response.output[:500],
                tokens_in=agent_response.tokens_in,
                tokens_out=agent_response.tokens_out,
                latency_ms=agent_response.total_latency_ms,
            )

            sub_task_outputs.append(
                f"[{specialist_name.upper()}: {description}]\n{agent_response.output}"
            )

        assembled_output = "\n\n---\n\n".join(sub_task_outputs)

        # ── Phase 3: Critic Evaluation ─────────────────────────────
        critic_verdict = {"pass": True, "confidence": 1.0, "feedback": ""}

        if not hit_limit and assembled_output.strip():
            for revision_cycle in range(settings.max_revision_cycles + 1):
                critic_response = self.critic.run(task, assembled_output)
                step += 1

                try:
                    critic_verdict = json.loads(critic_response.output)
                except json.JSONDecodeError:
                    critic_verdict = {"pass": True, "confidence": 0.5, "feedback": "Parse error"}

                self._log_event(
                    events, step, "CriticAgent", "critique",
                    tool_output=json.dumps(critic_verdict),
                    tokens_in=critic_response.tokens_in,
                    tokens_out=critic_response.tokens_out,
                    latency_ms=critic_response.total_latency_ms,
                )

                if critic_verdict.get("pass", True):
                    logger.info(
                        f"[{task_id}] Critic passed (confidence: "
                        f"{critic_verdict.get('confidence', '?')})"
                    )
                    break

                # Critic rejected — attempt revision
                if revision_cycle < settings.max_revision_cycles:
                    feedback = critic_verdict.get("feedback", "Please improve the output.")
                    logger.info(
                        f"[{task_id}] Critic rejected (cycle {revision_cycle + 1}). "
                        f"Feedback: {feedback[:80]}"
                    )

                    # Re-run the last specialist with critic feedback
                    last_specialist = sub_tasks[-1].get("specialist", "research")
                    agent = self.specialists.get(last_specialist, self.specialists["research"])

                    revision_task = (
                        f"The previous output was rejected by the quality critic.\n"
                        f"Original task: {task}\n"
                        f"Critic feedback: {feedback}\n"
                        f"Previous output:\n{assembled_output}\n\n"
                        f"Please revise and improve the output."
                    )

                    revision_response = agent.run(revision_task)
                    step += 1
                    self._log_event(
                        events, step, agent.name, "revision",
                        tool_output=revision_response.output[:500],
                        tokens_in=revision_response.tokens_in,
                        tokens_out=revision_response.tokens_out,
                        latency_ms=revision_response.total_latency_ms,
                    )

                    # Replace the assembled output with the revised version
                    assembled_output = revision_response.output

                    for tc in revision_response.tool_calls:
                        total_tool_calls += 1
                        step += 1
                        self._log_event(
                            events, step, agent.name, "tool_call",
                            tool_name=tc["tool"],
                            tool_input=json.dumps(tc["args"]),
                            tool_output=tc.get("result", ""),
                        )
                else:
                    logger.warning(f"[{task_id}] Max revision cycles reached. Returning current output.")

        # ── Phase 4: Finalisation ──────────────────────────────────
        step += 1
        self._log_event(
            events, step, "Orchestrator", "final",
            tool_output=assembled_output[:500],
        )

        # Store in long-term memory
        try:
            task_summary = task[:200]
            result_summary = assembled_output[:200]
            self.memory.store(task_summary, result_summary)
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

        # Update conversation buffer
        buffer.add("assistant", assembled_output)

        # Gather token stats
        token_stats = self.model_manager.get_token_stats()

        completed = not hit_limit and critic_verdict.get("pass", True)

        logger.info(
            f"[{task_id}] Task complete. Completed: {completed}. "
            f"Tool calls: {total_tool_calls}. Tokens: {token_stats['total_tokens']}"
        )

        return OrchestratorResult(
            response=assembled_output,
            task_id=task_id,
            completed=completed,
            token_summary={
                **token_stats,
                "total_tool_calls": total_tool_calls,
            },
            trajectory_events=events,
        )