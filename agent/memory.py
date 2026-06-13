"""
Phase 2 — LLM Agent Core items 16-25.

Provides:
  - PromptBuilder            consistent agent instructions  (item 17)
  - MemoryStore              short-term conversation context (item 22)
  - GoalRefiner              clarify vague goals            (item 23)
  - CostEstimator            LLM tokens + Colab compute      (item 24)
  - AgentStateMachine        idle→planning→executing→done    (item 25)
"""
import json
import re
import time
import logging
from typing import Optional
from enum import Enum
from datetime import datetime, timezone



# ==================================================================== #
#  17 — PromptBuilder
# ==================================================================== #

class PromptBuilder:
    """Build consistent, structured prompts for the LLM."""

    @staticmethod
    def system() -> str:
        return """You are an autonomous AI agent for fine-tuning HuggingFace models in Google Colab.
You follow a strict Plan → Execute → Parse loop.
Always respond with structured JSON when taking actions."""

    @staticmethod
    def with_context(base: str, context: dict) -> str:
        parts = [base]
        for key, val in context.items():
            if val:
                parts.append(f"\n## {key}\n{val}")
        return "\n".join(parts)

    @staticmethod
    def few_shot(task: str, examples: list[tuple[str, str]]) -> str:
        parts = [f"Task: {task}\n"]
        for inp, out in examples:
            parts.append(f"Input: {inp}\nOutput: {out}\n")
        return "\n".join(parts)

    @staticmethod
    def goal_refinement() -> str:
        return """The user's goal is vague. Ask clarifying questions to determine:
1. What base model to use (or what size parameters?)
2. What dataset (or what kind of data?)
3. What task (chat, code, classification, etc.)
4. Any specific constraints (Colab free tier, time limit, etc.)

Return a JSON with "questions" as a list of 3-5 questions."""

    @staticmethod
    def plan_prompt(goal: str) -> str:
        return f"""Analyze this goal and produce a step-by-step Colab execution plan.

Goal: {goal}

Return ONLY valid JSON with schema:
{{
  "goal": "...",
  "analysis": {{ "model": {{"name":"...","params_b":N}}, "method": {{"type":"lora|qlora|full"}}, "dataset": {{"name":"..."}}, "runtime": {{"required":"...","vram_needed_gb":N}} }},
  "steps": [ {{"id":1, "action":"...", "description":"..."}} ]
}}"""


# ==================================================================== #
#  22 — MemoryStore
# ==================================================================== #

class MemoryStore:
    """Short-term conversational context with sliding window."""

    def __init__(self, max_turns: int = 20, max_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def add(self, role: str, content: str, metadata: Optional[dict] = None):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })
        self._trim()

    def _trim(self):
        while len(self.history) > self.max_turns:
            self.history.pop(0)
        total = sum(len(m.get("content", "")) for m in self.history)
        while total > self.max_tokens and len(self.history) > 1:
            self.history.pop(0)
            total = sum(len(m.get("content", "")) for m in self.history)

    def get_context(self, n: int = 5) -> list[dict]:
        return self.history[-n:]

    def get_recent(self, role: Optional[str] = None, n: int = 5) -> list[dict]:
        filtered = [m for m in self.history if not role or m["role"] == role]
        return filtered[-n:]

    def summarize_recent(self, n: int = 5) -> str:
        recent = self.get_context(n)
        lines = [f"[{m['role']}] {m['content'][:100]}" for m in recent]
        return "\n".join(lines)

    def clear(self):
        self.history = []


# ==================================================================== #
#  23 — GoalRefiner
# ==================================================================== #

class GoalRefiner:
    """Ask LLM to clarify vague goals before planning."""

    def __init__(self, llm_client=None):
        if llm_client is None:
            from agent.llm_client import LLMClient
            try:
                self.llm = LLMClient()
            except Exception:
                self.llm = None
        else:
            self.llm = llm_client

    def is_vague(self, goal: str) -> bool:
        word_count = len(goal.split())
        if word_count < 5:
            return True
        has_model = bool(re.search(
            r"(phi|llama|mistral|gemma|qwen|falcon|bert|gpt|t5|distil)", goal.lower()))
        has_dataset = bool(re.search(
            r"(dataset|data|dolly|alpaca|sharegpt|oasst|imdb|squad)", goal.lower()))
        return not (has_model or has_dataset) and word_count < 10

    def generate_questions(self, goal: str) -> list[str]:
        if self.llm is None:
            return [
                "What base model do you want to fine-tune (e.g., microsoft/phi-2)?",
                "What dataset should I use (HuggingFace dataset name or custom path)?",
                "What task is this for (chat, code generation, classification, etc.)?",
            ]
        prompt = PromptBuilder.goal_refinement() + f"\n\nUser goal: {goal}"
        response = self.llm.safe_json_chat([
            {"role": "system", "content": PromptBuilder.system()},
            {"role": "user", "content": prompt},
        ], parser="json_only")
        if response and "questions" in response:
            return response["questions"]
        return [
            "What base model do you want to fine-tune (e.g., microsoft/phi-2)?",
            "What dataset should I use (HuggingFace dataset name or custom path)?",
            "What task is this for (chat, code generation, classification, etc.)?",
        ]


# ==================================================================== #
#  24 — CostEstimator
# ==================================================================== #

# Cost per 1K tokens (USD approximate)
LLM_COST_PER_1K = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "gemini-pro": {"input": 0.001, "output": 0.002},
}

# Colab compute units per hour
COLAB_UNITS_PER_HOUR = {
    "None": 0, "T4": 1.0, "V100": 1.5, "A100": 2.5, "A100-80GB": 3.0, "TPU": 4.0,
}


class CostEstimator:
    """Estimate total cost in USD and compute units."""

    @staticmethod
    def estimate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = LLM_COST_PER_1K.get(model, {"input": 0.01, "output": 0.02})
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])

    @staticmethod
    def estimate_colab_cost(runtime: str, hours: float) -> float:
        units_per_hour = COLAB_UNITS_PER_HOUR.get(runtime, 1.0)
        return hours * units_per_hour

    @staticmethod
    def estimate_training_cost(model_params_b: float,
                               runtime: str = "T4",
                               method: str = "qlora",
                               epochs: int = 3,
                               dataset_size: int = 10000) -> dict:
        from colab.runtime import estimate_needed_vram
        vram_needed = estimate_needed_vram(model_params_b, method)
        tokens_per_second = {
            "T4": {"qlora": 500, "lora": 300, "full": 100},
            "V100": {"qlora": 800, "lora": 500, "full": 200},
            "A100": {"qlora": 1500, "lora": 1000, "full": 400},
        }
        tps = tokens_per_second.get(runtime, {}).get(method, 500)
        total_tokens = dataset_size * model_params_b * 1e9 * 0.1
        hours = total_tokens / (tps * 3600) * epochs

        return {
            "colab_hours": round(hours, 2),
            "colab_units": round(hours * COLAB_UNITS_PER_HOUR.get(runtime, 1.0), 2),
            "vram_needed_gb": vram_needed,
            "recommended_runtime": "V100" if vram_needed > 16 else "T4",
        }

    @staticmethod
    def format_cost(goal: str) -> str:
        """Generate Colab code that prints cost estimate."""
        return f"""
import json
# Cost estimate for: {goal}
estimate = {json.dumps(CostEstimator.estimate_training_cost(7.0))}
print(json.dumps(estimate, indent=2))
"""


# ==================================================================== #
#  25 — AgentStateMachine
# ==================================================================== #

class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    GENERATING_CODE = "generating_code"
    EXECUTING = "executing"
    PARSING_RESULT = "parsing_result"
    RECOVERING = "recovering"
    SWITCHING_RUNTIME = "switching_runtime"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


TRANSITIONS = {
    AgentState.IDLE: [AgentState.PLANNING, AgentState.AWAITING_CLARIFICATION],
    AgentState.PLANNING: [AgentState.GENERATING_CODE, AgentState.AWAITING_CLARIFICATION],
    AgentState.AWAITING_CLARIFICATION: [AgentState.PLANNING, AgentState.IDLE],
    AgentState.GENERATING_CODE: [AgentState.EXECUTING, AgentState.PLANNING],
    AgentState.EXECUTING: [AgentState.PARSING_RESULT, AgentState.RECOVERING],
    AgentState.PARSING_RESULT: [AgentState.GENERATING_CODE, AgentState.COMPLETED,
                                AgentState.RECOVERING, AgentState.SWITCHING_RUNTIME],
    AgentState.RECOVERING: [AgentState.GENERATING_CODE, AgentState.SWITCHING_RUNTIME,
                            AgentState.FAILED],
    AgentState.SWITCHING_RUNTIME: [AgentState.GENERATING_CODE, AgentState.FAILED],
    AgentState.COMPLETED: [AgentState.IDLE],
    AgentState.FAILED: [AgentState.IDLE],
    AgentState.ABORTED: [],
}


class AgentStateMachine:
    """Track and validate agent state transitions."""

    def __init__(self):
        self.state = AgentState.IDLE
        self.history: list[dict] = []
        self.start_time = time.time()

    def transition(self, new_state: AgentState) -> bool:
        allowed = TRANSITIONS.get(self.state, [])
        if new_state in allowed:
            old = self.state
            self.state = new_state
            self.history.append({
                "from": old.value,
                "to": new_state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed": time.time() - self.start_time,
            })
            return True
        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid transition: {self.state.value} -> {new_state.value}")
        return False

    def can_transition(self, new_state: AgentState) -> bool:
        return new_state in TRANSITIONS.get(self.state, [])

    def reset(self):
        self.state = AgentState.IDLE
        self.history = []
        self.start_time = time.time()

    def get_stats(self) -> dict:
        total_time = time.time() - self.start_time
        return {
            "current_state": self.state.value,
            "transitions": len(self.history),
            "elapsed_seconds": round(total_time, 1),
            "elapsed_hours": round(total_time / 3600, 3),
        }
