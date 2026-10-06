# file: agentmesh/orchestrator.py
"""Orchestrator — the central agent loop coordinating all components."""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from agentmesh.agents import (
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
    """Detects repeated tool calls with identical arguments."""

    def __init__(self):
        self._seen: dict[str, str] = {}

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        key = f"{tool_name}:{self._hash_args(args)}"
        if key in self._seen:
            return True, self._seen[key]
        return False, ""

    def record(self, tool_name: str, args: dict, result: str) -> None:
        key = f"{tool_name}:{self._hash_args(args)}"
        self._seen[key] = result[:500]

    @staticmethod
    def _hash_args(args: dict) -> str:
        serialised = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]


# ── Plan routing rules ──────────────────────────────────────────────

SPECIALISTS = ("research", "coder", "analyst")
CODE_CAPABLE = ("coder", "analyst")  # specialists that can call run_python
MAX_SUB_TASKS = 4

# "Write a Python function", "Implement merge sort in Python", "using Python".
# A verb is required, and [^.?!]* keeps the match inside one sentence, so
# "the history of the Python programming language" is NOT a coding request.
_CODE_INTENT = re.compile(
    r"\b(?:write|implement|create|build|generate)\b[^.?!]*"
    r"\b(?:python|code|script|function|class|program)\b"
    r"|\b(?:in|using|with) python\b",
    re.IGNORECASE,
)
_COMPUTE_INTENT = re.compile(
    r"\b(?:calculate|compute|simulate|count|mean|median|average|statistics?)\b",
    re.IGNORECASE,
)


def _required_specialist(task: str) -> str | None:
    """Return 'coder' or 'analyst' if the task needs code execution, else None."""
    if _CODE_INTENT.search(task):
        return "coder"
    if _COMPUTE_INTENT.search(task):
        return "analyst"
    return None


def _fallback_specialist(task: str) -> str:
    """Specialist to use when the planner gives no usable plan."""
    return _required_specialist(task) or "research"


def _validate_plan(task: str, raw_plan) -> tuple[list[dict], list[str]]:
    """Turn the planner's raw output into a plan the orchestrator can trust.

    Returns (plan, notes). Each note describes one correction, so the
    trajectory shows exactly how the planner's plan was changed.

    Rules, in order:
      1. Drop malformed steps; map unknown specialists to research.
      2. Empty plan -> one step for the fallback specialist.
      3. Merge consecutive steps for the same specialist (one call, not two).
      4. Cap the plan at MAX_SUB_TASKS steps.
      5. If the task needs code execution but no step can run code,
         append a step for the specialist that can.
    """
    notes: list[str] = []
    steps: list[dict] = []

    if isinstance(raw_plan, list):
        for item in raw_plan:
            if not isinstance(item, dict):
                notes.append("dropped a malformed step")
                continue
            specialist = str(item.get("specialist", "")).strip().lower()
            if specialist not in SPECIALISTS:
                notes.append(f"unknown specialist '{specialist}' -> research")
                specialist = "research"
            description = str(item.get("description") or "").strip() or task
            steps.append({"description": description, "specialist": specialist})

    if not steps:
        specialist = _fallback_specialist(task)
        notes.append(f"empty plan -> single {specialist} step")
        return [{"description": task, "specialist": specialist}], notes

    merged = [steps[0]]
    for step in steps[1:]:
        if step["specialist"] == merged[-1]["specialist"]:
            merged[-1] = {
                "description": f"{merged[-1]['description']} Then: {step['description']}",
                "specialist": step["specialist"],
            }
            notes.append(f"merged consecutive {step['specialist']} steps")
        else:
            merged.append(step)

    if len(merged) > MAX_SUB_TASKS:
        notes.append(f"plan capped at {MAX_SUB_TASKS} steps (was {len(merged)})")
        merged = merged[:MAX_SUB_TASKS]

    required = _required_specialist(task)
    if required and not any(s["specialist"] in CODE_CAPABLE for s in merged):
        merged.append({"description": task, "specialist": required})
        notes.append(f"task needs code execution -> added {required} step")

    return merged, notes


def _revision_specialist(task: str, plan: list[dict]) -> str:
    """Pick who revises a rejected answer.

    The revision replaces the whole answer, so it should go to a specialist
    that can redo the work: for tasks that need code, the last code-capable
    step in the plan; otherwise the last step's specialist.
    """
    if _required_specialist(task):
        capable = [s["specialist"] for s in plan if s["specialist"] in CODE_CAPABLE]
        if capable:
            return capable[-1]
    return plan[-1]["specialist"]


class Orchestrator:
    """Central agent loop: plan -> specialists -> critic -> revision."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[PersistentMemory] = None,
    ):
        self.model_manager = model_manager or ModelManager()
        self.tool_registry = tool_registry or create_default_registry()
        self.memory = memory or PersistentMemory()

        self.planner = PlannerAgent(self.model_manager)
        self.critic = CriticAgent(self.model_manager)

        # Least privilege: each specialist sees and can call only its own tools
        reg = self.tool_registry
        self.specialists = {
            "research": ResearchAgent(
                self.model_manager, reg.subset(["search_web", "query_knowledge_base"])
            ),
            "coder": CoderAgent(
                self.model_manager, reg.subset(["run_python", "read_file", "write_file"])
            ),
            "analyst": AnalystAgent(
                self.model_manager, reg.subset(["read_file", "query_knowledge_base", "run_python"])
            ),
        }

        self._sessions: dict[str, ConversationBuffer] = {}

    # ── Helpers ─────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> ConversationBuffer:
        """Get or create a conversation buffer for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBuffer()
        return self._sessions[session_id]

    @staticmethod
    def _log_event(
        events: list[dict],
        step: int,
        agent_name: str,
        action_type: str,
        tool_name: str = "",
        tool_input: str = "",
        tool_output: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Append a structured event to the trajectory list."""
        events.append({
            "step_number": step,
            "agent_name": agent_name,
            "action_type": action_type,
            "tool_name": tool_name,
            "tool_input": tool_input[:1000],
            "tool_output": tool_output[:1000],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": json.dumps(metadata or {}),
        })

    # ── Main loop ───────────────────────────────────────────────

    def run(self, task: str, session_id: str = "", use_memory: bool = True) -> OrchestratorResult:
        """Execute a full task lifecycle.

        use_memory=False skips reading and writing long-term memory. The eval
        runner uses it so one eval run can't see answers from a previous one.
        """
        task_id = f"task_{uuid4().hex[:12]}"
        session_id = session_id or f"session_{uuid4().hex[:8]}"
        events: list[dict] = []
        step = 0
        total_tool_calls = 0

        self.model_manager.reset_token_stats()

        buffer = self._get_session(session_id)
        buffer.add("user", task)

        memory_context = self.memory.get_memory_context(task) if use_memory else ""
        logger.info(f"[{task_id}] Starting task: {task[:80]}...")

        # ── Phase 1: Planning ───────────────────────────────────
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
            raw_plan = json.loads(plan_response.output)
        except json.JSONDecodeError:
            raw_plan = []

        sub_tasks, plan_notes = _validate_plan(task, raw_plan)
        if plan_notes:
            logger.warning(f"[{task_id}] Plan adjusted: {'; '.join(plan_notes)}")
            step += 1
            self._log_event(
                events, step, "Orchestrator", "plan_adjusted",
                tool_output=json.dumps(sub_tasks),
                metadata={"notes": plan_notes},
            )

        logger.info(f"[{task_id}] Plan: {len(sub_tasks)} sub-tasks")

        # ── Phase 2: Specialist execution ───────────────────────
        loop_detector = _LoopDetector()
        labelled_outputs: list[str] = []  # "[ROLE: step]\noutput" — context for later steps
        raw_outputs: list[str] = []       # bare outputs — used for single-step answers
        hit_limit = False

        for i, sub_task in enumerate(sub_tasks):
            if hit_limit:
                break

            specialist_name = sub_task["specialist"]  # validated by _validate_plan
            description = sub_task["description"]
            agent = self.specialists[specialist_name]

            logger.info(f"[{task_id}] Sub-task {i+1}: {specialist_name} — {description[:60]}")

            # Give each specialist the overall goal and what earlier steps found
            instruction = f"Overall task: {task}\nYour step: {description}"
            if labelled_outputs:
                earlier = "\n\n".join(labelled_outputs)[-3000:]  # cap context cost
                instruction += f"\n\nResults from earlier steps:\n{earlier}"

            agent_response = agent.run(instruction)

            for tc in agent_response.tool_calls:
                total_tool_calls += 1

                is_dup, prev_result = loop_detector.check(tc["tool"], tc["args"])
                if is_dup:
                    logger.warning(f"[{task_id}] Loop detected: {tc['tool']}({tc['args']})")
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
                    metadata={"implicit": tc.get("implicit", False)},
                )

                if total_tool_calls >= settings.max_total_tool_calls:
                    logger.warning(f"[{task_id}] Total tool call limit reached.")
                    hit_limit = True
                    break

            step += 1
            self._log_event(
                events, step, agent.name, "agent_output",
                tool_output=agent_response.output[:500],
                tokens_in=agent_response.tokens_in,
                tokens_out=agent_response.tokens_out,
                latency_ms=agent_response.total_latency_ms,
            )

            raw_outputs.append(agent_response.output)
            labelled_outputs.append(
                f"[{specialist_name.upper()}: {description}]\n{agent_response.output}"
            )

        # Section headers only help when several specialists contributed
        if len(raw_outputs) == 1:
            assembled_output = raw_outputs[0]
        else:
            assembled_output = "\n\n---\n\n".join(labelled_outputs)

        # ── Phase 3: Critic + revision ──────────────────────────
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
                    logger.info(f"[{task_id}] Critic passed ({critic_verdict.get('confidence', '?')})")
                    break

                if revision_cycle < settings.max_revision_cycles:
                    feedback = critic_verdict.get("feedback", "Please improve the output.")
                    logger.info(f"[{task_id}] Critic rejected (cycle {revision_cycle + 1}): {feedback[:80]}")

                    agent = self.specialists[_revision_specialist(task, sub_tasks)]

                    revision_task = (
                        f"Task: {task}\n"
                        f"Reviewer feedback: {feedback}\n"
                        f"Previous answer:\n{assembled_output}\n\n"
                        f"Write an improved final answer to the task. "
                        f"Respond with only the improved answer. "
                        f"Do not mention the reviewer, the revision, or apologise."
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

                    assembled_output = revision_response.output

                    for tc in revision_response.tool_calls:
                        total_tool_calls += 1
                        step += 1
                        self._log_event(
                            events, step, agent.name, "tool_call",
                            tool_name=tc["tool"],
                            tool_input=json.dumps(tc["args"]),
                            tool_output=tc.get("result", ""),
                            metadata={"implicit": tc.get("implicit", False)},
                        )
                else:
                    logger.warning(f"[{task_id}] Max revision cycles reached.")

        # ── Phase 4: Finalisation ───────────────────────────────
        step += 1
        self._log_event(events, step, "Orchestrator", "final",
                        tool_output=assembled_output[:500])

        if use_memory:
            try:
                self.memory.store(task[:200], assembled_output[:200])
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")

        buffer.add("assistant", assembled_output)

        token_stats = self.model_manager.get_token_stats()
        completed = not hit_limit and critic_verdict.get("pass", True)

        logger.info(
            f"[{task_id}] Done. Completed: {completed}. "
            f"Tool calls: {total_tool_calls}. Tokens: {token_stats['total_tokens']}"
        )

        return OrchestratorResult(
            response=assembled_output,
            task_id=task_id,
            completed=completed,
            token_summary={**token_stats, "total_tool_calls": total_tool_calls},
            trajectory_events=events,
        )
