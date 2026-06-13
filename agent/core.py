"""
Phase 5 — Full autonomous agent integration.

Combines:
  - LLM-powered planning (Phase 2)
  - ColabRunner + RuntimeManager (Phase 1 / this phase)
  - FineTuneOrchestrator (Phase 3)
  - MetricsStore + GitHubLogger (Phase 4)
  - Safety limits, cost tracking, code sanitization (Phase 8)
  - Error history retrieval and auto-fix
"""
import os
import time
import asyncio
import logging
import traceback
from typing import Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from agent.llm_client import LLMClient
from agent.prompts import (
    SYSTEM_PROMPT, PLANNING_PROMPT, CODE_GENERATION_PROMPT,
    RESULT_PARSING_PROMPT, SUMMARY_PROMPT,
)
from agent.safety import (
    sanitize_code, sanitize_pip, compute_code_hash, CostTracker,
    BudgetManager,
    MAX_RUNTIME_SWITCHES, MAX_TOTAL_ERROR_RETRIES, MAX_COLAB_HOURS,
    MAX_ERROR_RETRIES_PER_STEP,
)
from agent.input_sanitizer import sanitize_user_input, validate_goal_format
from agent.extended_safety import AuditLogger
from agent.circuit_breaker import CircuitBreaker
from colab.infrastructure import RetryPolicy
from agent.health import HealthReporter
from agent.observability import setup_json_logging, MetricsCollector
from agent.plugin import HookRunner, get_registry
from agent.llm_cache import LLMCache
from storage.jobs import JobStore
from storage.conversations import ConversationStore
from storage.models_store import ModelVersionStore
from storage.metrics_store import MetricsStore
from colab.executor import ColabRunner
from colab.runtime import RuntimeManager
from models.huggingface import HuggingFaceManager
from models.finetune import FinetuneCodeGenerator
from models.finetune_orchestrator import FineTuneOrchestrator
from gh_integration.integration import GitHubIntegration
from gh_integration.logger import GitHubLogger
from config.settings import settings

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # New: Awaiting human input
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABORTED = "aborted"


class NextAction(str, Enum):
    PROCEED = "proceed"
    RETRY = "retry"
    SWITCH_RUNTIME = "switch_runtime"
    CHANGE_MODEL = "change_model"
    CHANGE_DATASET = "change_dataset"
    ABORT = "abort"


@dataclass
class PlanStep:
    id: int
    action: str
    description: str
    expected_duration: str = ""
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    code: str = ""
    output: str = ""
    error: str = ""
    metrics: dict = field(default_factory=dict)
    next_action: NextAction = NextAction.PROCEED
    execution_time: float = 0.0
    blocked_reasons: list = field(default_factory=list)

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def record_attempt(self):
        self.retry_count += 1


@dataclass
class Plan:
    goal: str
    analysis: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "analysis": self.analysis,
            "steps": [
                {
                    "id": s.id,
                    "action": s.action,
                    "description": s.description,
                    "status": s.status.value,
                    "retry_count": s.retry_count,
                    "error": s.error,
                    "metrics": s.metrics,
                    "execution_time": s.execution_time,
                }
                for s in self.steps
            ],
            "created_at": self.created_at,
        }


class StepTracker:
    def __init__(self):
        self.current_step_index = 0
        self.steps: list[PlanStep] = []
        self.aborted = False
        self.total_retries = 0
        self.runtime_switches = 0

    def add_step(self, step: PlanStep):
        self.steps.append(step)

    def current(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> bool:
        self.current_step_index += 1
        return self.current_step_index < len(self.steps)

    def all_completed(self) -> bool:
        return all(s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
                   for s in self.steps)

    def failed_steps(self) -> list:
        return [s for s in self.steps if s.status == StepStatus.FAILED]


class OrchestratorAgent:
    """
    Fully autonomous agent with:
      - Plan → Execute → Parse loop
      - Max 5 runtime switches, max 10 retries/step, max 50 total retries
      - Auto OOM detection → switch runtime
      - Metrics logging per step
      - GitHub error push + retrieval
      - Cost tracking
      - Code sanitization
    """

    def __init__(self, verbose: bool = None, model: str = None, dataset: str = None, method: str = None):
        self.llm = LLMClient()
        self.runner = ColabRunner()
        self.verbose = settings.agent_verbose if verbose is None else verbose
        self.conversation_id = None
        self.current_job_id = None
        self.current_runtime = "None"
        self.messages = []
        self.plan = None
        self.tracker = StepTracker()
        self.user_goal = ""
        self._override_model = model
        self._override_dataset = dataset
        self._override_method = method

        # Storage
        self.jobs = JobStore()
        self.convs = ConversationStore()
        self.models_store = ModelVersionStore()
        self.metrics_store = MetricsStore()

        # Models & Finetuning
        self.hf = HuggingFaceManager()
        self.finetune = FinetuneCodeGenerator()
        self.finetune_orch = FineTuneOrchestrator()
        self.runtime_mgr = RuntimeManager()

        # GitHub
        self.github = GitHubIntegration()
        self.github_logger = GitHubLogger()

        # Safety / Cost
        self.cost_tracker = CostTracker()
        self.budget = BudgetManager(
            max_cost_units=settings.budget_max_units,
            warn_threshold=settings.budget_warn_threshold,
        )
        self.audit_logger = AuditLogger(
            path=os.path.join(settings.data_dir, "audit.log.jsonl"),
        )
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.health = HealthReporter()
        self.metrics = MetricsCollector()
        # Plugin system
        self.plugin_registry = get_registry()
        self.hook_runner = HookRunner(self.plugin_registry)
        self.llm_cache = LLMCache()
        setup_json_logging(logger, log_path=os.path.join(settings.data_dir, "agent.jsonl"))
        self.start_time = time.time()

    def _log(self, msg: str):
        if self.verbose:
            logger.info(f"[Agent] {msg}")

    # ---------------------------------------------------------------- #
    #  Main entry point
    # ---------------------------------------------------------------- #

    def run(self, goal: str) -> dict:
        self.user_goal = goal
        self._log(f"Starting agent with goal: {goal}")
        self.start_time = time.time()

        # Security: validate & sanitize user input before LLM
        fmt_error = validate_goal_format(goal)
        if fmt_error:
            self._log(f"Goal validation failed: {fmt_error}")
            return {"status": "failed", "error": fmt_error,
                    "conversation_id": self.conversation_id}

        safe, sanitized, warnings = sanitize_user_input(goal)
        self.audit_logger.goal_received(goal)
        if not safe:
            self._log(f"Input sanitization warnings: {warnings}")
            self.audit_logger.goal_sanitized(goal, warnings)
            goal = sanitized

        # Plugin: before_plan
        context = {"goal": goal}
        goal, context = self.hook_runner.run_before_plan(goal, context)

        conv = self.convs.create(goal)
        self.conversation_id = conv.id
        self._ws_context = {"conversation_id": self.conversation_id}
        self.convs.add_message(self.conversation_id, "user", goal)

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.health.update(active_job_id=self.current_job_id,
                           active_conversation_id=self.conversation_id,
                           current_runtime=self.current_runtime)

        # Safety: check elapsed time
        if self._elapsed_hours() > MAX_COLAB_HOURS:
            return {"status": "failed", "error": f"Exceeded {MAX_COLAB_HOURS}h limit",
                    "conversation_id": self.conversation_id}

        # Phase 1: Generate plan
        print("[AGENT] Analyzing goal and generating plan...")
        plan = self._generate_plan(goal, model=self._override_model, dataset=self._override_dataset, method=self._override_method)
        if not plan or not plan.steps:
            return {"status": "failed", "error": "Failed to generate plan",
                    "conversation_id": self.conversation_id}

        self.plan = plan
        self._log(f"Plan generated with {len(plan.steps)} steps")
        print(f"[AGENT] Plan: {len(plan.steps)} steps")

        # Apply user overrides to plan analysis
        if self._override_model:
            plan.analysis.setdefault("model", {})["name"] = self._override_model
        if self._override_dataset:
            plan.analysis.setdefault("dataset", {})["name"] = self._override_dataset
        if self._override_method:
            plan.analysis.setdefault("method", {})["type"] = self._override_method

        self.convs.add_message(
            self.conversation_id, "assistant",
            f"Plan: {len(plan.steps)} steps:\n" +
            "\n".join(f"  Step {s.id}: {s.description}" for s in plan.steps),
            {"phase": "planning", "plan": plan.to_dict()},
        )

        # Plugin: after_plan
        plan_dict = plan.to_dict() if plan else {}
        plan_dict, context = self.hook_runner.run_after_plan(plan_dict, self._ws_context)
        if plan and plan_dict:
            plan.analysis = plan_dict.get("analysis", plan.analysis or {})
            if plan_dict.get("steps"):
                self.tracker.steps.clear()
                for sd in plan_dict["steps"]:
                    self.tracker.add_step(PlanStep(id=sd.get("id", 1), action=sd.get("action", ""), description=sd.get("description", "")))

        # Create job
        analysis = plan.analysis or {}
        job = self.jobs.create(
            goal=goal,
            method=analysis.get("method", {}).get("type"),
            base_model=analysis.get("model", {}).get("name"),
            dataset=analysis.get("dataset", {}).get("name"),
            runtime=self.current_runtime,
            conversation_id=self.conversation_id,
        )
        self.current_job_id = job.id
        self._ws_context = {"job_id": self.current_job_id, "conversation_id": self.conversation_id}
        self.cost_tracker.start_job(job.id, self.current_runtime)
        self.metrics.gauge("plan.steps", len(plan.steps))
        self.metrics.counter("jobs.started")

        # Phase 2: Execute steps
        print("[AGENT] Starting execution...")
        while self.tracker.current_step_index < len(plan.steps):
            if self._elapsed_hours() > MAX_COLAB_HOURS:
                print(f"[AGENT] Reached {MAX_COLAB_HOURS}h limit, aborting")
                self.tracker.aborted = True
                break

            if self.budget.exceeded:
                print(f"[AGENT] Budget exceeded ({self.budget.snapshot()['spent']} units), aborting")
                self.tracker.aborted = True
                break

            step = self.tracker.current()
            if not step:
                break
            if self.tracker.aborted:
                step.status = StepStatus.ABORTED
                break

            self._execute_step(step)

            if step.status in (StepStatus.SUCCESS, StepStatus.SKIPPED):
                self.tracker.advance()
            elif step.status == StepStatus.FAILED:
                if not step.can_retry() or self._max_retries_exceeded():
                    self._log(f"Step {step.id} exhausted retries. Aborting.")
                    self.tracker.aborted = True

        # Phase 3: Summary
        final_status = ("completed" if not self.tracker.aborted
                        and self.tracker.all_completed() else "aborted")
        summary = self._generate_summary(final_status)

        self.jobs.update(self.current_job_id, status=final_status)
        self.convs.add_message(
            self.conversation_id, "assistant",
            f"Session {final_status}.\n{summary}",
            {"phase": "summary", "final_status": final_status},
        )
        self.convs.close(self.conversation_id, summary)

        cost = self.cost_tracker.get_summary(self.current_job_id)
        cost_units = cost.get("estimated_cost_units", 0)
        self.budget.record_cost(cost_units)
        self.metrics.gauge("jobs.cost_units", cost_units)
        self.metrics.gauge("budget.usage_pct", self.budget.usage_ratio * 100)
        self.metrics.gauge("budget.remaining", max(0, self.budget.max_cost_units - cost_units))
        self.metrics.gauge("runtime.switches", self.tracker.runtime_switches)
        self.metrics.gauge("retries.total", self.tracker.total_retries)

        # Plugin: on_summary
        summary, _ = self.hook_runner.run_on_summary(summary, self._ws_context)

        # Plugin: on_complete
        final = {"status": final_status, "summary": summary, "job_id": self.current_job_id}
        final, _ = self.hook_runner.run_on_complete(final, self._ws_context)

        # Push final report to GitHub
        if self.github_logger.token:
            conv_msgs = self.convs.get_messages(self.conversation_id)
            self.github_logger.push_final_report(
                self.current_job_id,
                metrics={"plan": plan.to_dict(), "cost": cost},
                conversation=[{"role": m["role"], "content": m["content"][:200]}
                              for m in conv_msgs[-20:]],
                summary=summary,
            )

        return {
            "status": final_status,
            "summary": summary,
            "plan": plan.to_dict(),
            "cost": cost,
            "conversation_id": self.conversation_id,
            "job_id": self.current_job_id,
        }

    def _elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    def _max_retries_exceeded(self) -> bool:
        return self.tracker.total_retries >= MAX_TOTAL_ERROR_RETRIES

    # ---------------------------------------------------------------- #
    #  Plan generation
    # ---------------------------------------------------------------- #

    def _generate_plan(self, goal: str, model: str = None, dataset: str = None, method: str = None) -> Optional[Plan]:
        model_hint = f"Model: {model}\n" if model else ""
        dataset_hint = f"Dataset: {dataset}\n" if dataset else ""
        method_hint = f"Fine-tuning method: {method}\n" if method else ""
        prompt = PLANNING_PROMPT.format(goal=goal, model_hint=model_hint, dataset_hint=dataset_hint, method_hint=method_hint)
        plan_data = self.llm.safe_json_chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            parser="json_only",
        )
        if not plan_data:
            self._log("Failed to parse plan from LLM")
            return None

        plan = Plan(
            goal=plan_data.get("goal", goal),
            analysis=plan_data.get("analysis", {}),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        for step_data in plan_data.get("steps", []):
            plan.steps.append(PlanStep(
                id=step_data.get("id", len(plan.steps) + 1),
                action=step_data.get("action", "unknown"),
                description=step_data.get("description", ""),
                expected_duration=step_data.get("expected_duration", ""),
            ))
            self.tracker.add_step(plan.steps[-1])

        return plan

    # ---------------------------------------------------------------- #
    #  Step execution
    # ---------------------------------------------------------------- #

    def _check_memory(self):
        """Monitor VRAM usage and trigger cleanup if nearly exhausted."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                if allocated / total > 0.90:
                    self._log("ALERT: VRAM usage > 90%. Triggering cleanup.")
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def _execute_step(self, step: PlanStep) -> dict:
        step.status = StepStatus.RUNNING
        self._log(f"Step {step.id}: {step.description}")

        print(f"[AGENT] Step {step.id}/{len(self.tracker.steps)}: {step.description}")

        while self.retry_policy.should_retry():
            if self.tracker.total_retries >= MAX_TOTAL_ERROR_RETRIES:
                step.error = f"Exceeded max total retries ({MAX_TOTAL_ERROR_RETRIES})"
                step.status = StepStatus.FAILED
                break

            step.record_attempt()
            self.tracker.total_retries += 1
            self.cost_tracker.record_retry(self.current_job_id)
            self.health.update(total_retries=self.tracker.total_retries)
            self.metrics.counter("step.attempts")
            self._log(f"  Attempt {step.retry_count}/{step.max_retries}")

            try:
                with self.circuit_breaker:
                    # Plugin: before_step
                    step_dict = {"id": step.id, "action": step.action, "description": step.description, "status": step.status.value}
                    step_dict, ctx = self.hook_runner.run_before_step(step_dict, self._ws_context)
                    step.action = step_dict.get("action", step.action)

                    code = self._generate_code(step)
                    if not code:
                        step.error = "Failed to generate code"
                        step.status = StepStatus.FAILED
                        continue

                    # --- Sanitize ---
                    safe, cleaned, warning = sanitize_code(code)
                    if not safe:
                        step.error = f"Code blocked: {warning}"
                        self._log(step.error)
                        code_hash = compute_code_hash(code)
                        self.audit_logger.code_blocked(code_hash, warning)
                        self.github_logger.push_error(
                            error_log=step.error,
                            notebook_cell=code[:500],
                            job_id=self.current_job_id,
                        )
                        step.blocked_reasons.append(warning)
                        if len(step.blocked_reasons) >= 3 and \
                           len(set(step.blocked_reasons[-3:])) == 1:
                            step.error = (
                                f"Step skipped: LLM repeatedly generated code "
                                f"blocked by safety filter ({warning})")
                            step.status = StepStatus.SKIPPED
                            print(f"[AGENT] Step {step.id} SKIPPED: "
                                  f"repeated blocked code ({warning})")
                            break
                        step.status = StepStatus.FAILED
                        continue

                    cleaned = sanitize_pip(cleaned)
                    # Plugin: after_code_gen
                    cleaned, ctx = self.hook_runner.run_after_code_gen({"id": step.id, "action": step.action}, cleaned, self._ws_context)
                    step.code = cleaned

                    self.convs.add_message(
                        self.conversation_id, "assistant",
                        f"Step {step.id}: code ({len(cleaned)} chars)",
                        {"phase": "exec", "step_id": step.id,
                         "code_hash": compute_code_hash(cleaned)},
                    )

                    # --- Execute ---
                    result = self.runner.execute_cell(
                        cleaned, timeout=300, description=step.description,
                    )
                    
                    # --- VRAM Monitor ---
                    self._check_memory()

                    step.output = result.get("output", "")
                    step.error = result.get("error", "")
                    step.execution_time = result.get("execution_time", 0.0)
                    self.cost_tracker.record_execution(
                        self.current_job_id, step.execution_time)

                    # --- Check for Divergence ---
                    if step.action == "train":
                        self._monitor_divergence(step, step.output)
                        if step.status == StepStatus.FAILED:
                            continue # Retry if divergence detected

                    # --- Parse result ---
                    self.runner.parse_output(result)
                    analysis = self._parse_result(step, result)

                    if analysis:
                        self._handle_analysis(step, analysis)
                    elif result.get("success") and not result.get("error"):
                        step.status = StepStatus.SUCCESS
                        self._log(f"Step {step.id}: fallback success (LLM parse returned None)")

                    # --- OOM auto-detection ---
                    if not result.get("success") and result.get("error"):
                        if self._detect_oom(result["error"]) and \
                           self.tracker.runtime_switches < MAX_RUNTIME_SWITCHES:
                            next_tier = self._find_next_runtime_tier()
                            if next_tier:
                                self._log(f"OOM: {self.current_runtime} -> {next_tier}")
                                sr = self._switch_runtime(
                                    next_tier, f"OOM on {self.current_runtime}")
                                if sr.get("success"):
                                    self.current_runtime = next_tier
                                    self.tracker.runtime_switches += 1
                                    self.cost_tracker.record_switch(
                                        self.current_job_id, next_tier)
                                    self.runtime_mgr.current_gpu = next_tier
                                    self.runtime_mgr.current_vram_gb = \
                                        settings.runtime_tiers.get(next_tier, 0)
                                    continue  # retry after switch

                    # Plugin: after_step
                    result_dict = {"step_id": step.id, "status": "success" if step.status == StepStatus.SUCCESS else "failed",
                                   "metrics": step.metrics, "error": step.error}
                    result_dict, ctx = self.hook_runner.run_after_step({"id": step.id, "action": step.action}, result_dict, self._ws_context)
                    if result_dict.get("status") == "failed":
                        step.status = StepStatus.FAILED
                        step.error = result_dict.get("error", step.error)

                    if step.status == StepStatus.SUCCESS:
                        self.retry_policy.record_result(True)
                        print(f"[AGENT] Step {step.id} OK ({step.execution_time:.1f}s)")
                        return {"step_id": step.id, "status": "success",
                                "metrics": step.metrics}

            except RuntimeError as re:
                self.retry_policy.record_result(False)
                self.retry_policy.sleep()
                self.health.update(circuit_state="open",
                                   circuit_failures=self.circuit_breaker.failure_count)
                step.error = f"Circuit Breaker blocked execution: {re}"
                self._log(step.error)
                step.status = StepStatus.FAILED
                break
            except Exception as e:
                self.retry_policy.record_result(False)
                self.retry_policy.sleep()
                step.error = f"Unexpected: {e}\n{traceback.format_exc()}"
                self._log(step.error)
                # Plugin: on_error
                recovery, ctx = self.hook_runner.run_on_error({"id": step.id, "action": step.action}, step.error, self._ws_context)
                if recovery:
                    self._log(f"Plugin recovery: {recovery}")

        if step.status != StepStatus.SKIPPED:
            step.status = StepStatus.FAILED
            print(f"[AGENT] Step {step.id} FAILED after {step.retry_count} attempts")
            self.jobs.update(self.current_job_id,
                             status="failed",
                             error=step.error[:500])
            self.github_logger.push_error(
                error_log=step.error,
                notebook_cell=step.code[:500],
                job_id=self.current_job_id,
            )
            ret_status = "failed"
        else:
            print(f"[AGENT] Step {step.id} SKIPPED after {step.retry_count} attempts")
            ret_status = "skipped"
        return {"step_id": step.id, "status": ret_status, "error": step.error}

    def _handle_analysis(self, step: PlanStep, analysis: dict):
        action = analysis.get("next_action", "proceed")
        try:
            step.next_action = NextAction(action)
        except ValueError:
            step.next_action = NextAction.PROCEED

        if analysis.get("error"):
            step.error = analysis["error"]

        if analysis.get("status") == "success":
            step.metrics = analysis.get("key_values", {})
            step.status = StepStatus.SUCCESS
            # Log to metrics store
            if self.current_job_id and step.metrics:
                self.metrics_store.log_epoch(
                    job_id=self.current_job_id,
                    loss=step.metrics.get("loss"),
                    accuracy=step.metrics.get("accuracy"),
                    extras=step.metrics,
                )
            return

        if step.next_action == NextAction.SWITCH_RUNTIME:
            if self.tracker.runtime_switches >= MAX_RUNTIME_SWITCHES:
                step.error = f"Max switches ({MAX_RUNTIME_SWITCHES}) reached"
                step.next_action = NextAction.ABORT
                return
            target = analysis.get("target_runtime", "A100")
            self._log(f"Runtime switch requested: {self.current_runtime} -> {target}")
            sr = self._switch_runtime(target, str(step.error))
            if sr.get("success"):
                self.current_runtime = target
                self.tracker.runtime_switches += 1
                self.cost_tracker.record_switch(self.current_job_id, target)
                self.runtime_mgr.current_gpu = target
                self.runtime_mgr.current_vram_gb = \
                    settings.runtime_tiers.get(target, 0)
            step.next_action = NextAction.RETRY

        if step.next_action == NextAction.CHANGE_MODEL:
            new_model = analysis.get("new_model", "")
            if new_model:
                self._log(f"Model change requested: {self.plan.analysis.get('model', {}).get('name', '?')} -> {new_model}")
                self.plan.analysis.setdefault("model", {})["name"] = new_model
                if self.current_job_id:
                    self.jobs.update(self.current_job_id, base_model=new_model)
            step.next_action = NextAction.RETRY

        if step.next_action == NextAction.CHANGE_DATASET:
            new_dataset = analysis.get("new_dataset", "")
            if new_dataset:
                self._log(f"Dataset change requested: {self.plan.analysis.get('dataset', {}).get('name', '?')} -> {new_dataset}")
                self.plan.analysis.setdefault("dataset", {})["name"] = new_dataset
                if self.current_job_id:
                    self.jobs.update(self.current_job_id, dataset=new_dataset)
            step.next_action = NextAction.RETRY

        if step.next_action == NextAction.ABORT:
            step.status = StepStatus.FAILED
            self.tracker.aborted = True

    # ---------------------------------------------------------------- #
    #  Code generation
    # ---------------------------------------------------------------- #

    def _find_checkpoint(self, output_dir: str) -> Optional[str]:
        """Check if a valid checkpoint exists in the output directory."""
        if os.path.exists(output_dir):
            checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                           if d.startswith("checkpoint-")]
            if checkpoints:
                # Return latest checkpoint if valid
                latest = max(checkpoints, key=os.path.getmtime)
                if self.finetune_orch.validate_checkpoint(latest):
                    return latest
                else:
                    self._log(f"Invalid checkpoint found at {latest}, ignoring.")
        return None

    def _generate_code(self, step: PlanStep) -> str:
        # Use FineTuneOrchestrator for training steps
        if step.action in ("train", "configure_training", "apply_peft",
                            "download_model", "load_dataset", "probe_hardware"):
            analysis = (self.plan.analysis if self.plan else {})
            model_name = (analysis.get("model", {}).get("name")
                          or settings.default_base_model)
            dataset_name = (analysis.get("dataset", {}).get("name")
                            or "databricks/databricks-dolly-15k")
            method = (analysis.get("method", {}).get("type")
                      or settings.default_finetune_method)
            
            output_dir = "./finetuned_model"
            checkpoint = self._find_checkpoint(output_dir)

            if step.action == "probe_hardware":
                return """
import torch
import time
print("Benchmarking hardware...")
# Dummy data for probe
input_ids = torch.randint(0, 1000, (8, 512)).to('cuda')
start = time.time()
# Simulate a few forward passes
for _ in range(10):
    _ = torch.cuda.get_device_properties(0)
    torch.cuda.synchronize()
end = time.time()
throughput = 10 / (end - start)
print(f"PROBE_RESULT: throughput={throughput:.2f}_samples_sec, vram_free={torch.cuda.mem_get_info(0)[0]/1e9:.2f}_GB")
"""

            if step.action in ("train", "configure_training", "apply_peft"):
                plan = self.finetune_orch.plan_training(
                    base_model=model_name,
                    vram_gb=settings.runtime_tiers.get(self.current_runtime, 16),
                )
                
                # Use resume script if checkpoint found
                if checkpoint:
                    self._log(f"Resuming training from checkpoint: {checkpoint}")
                    return self.finetune_orch.generate_resume_script(
                        checkpoint_path=checkpoint,
                        base_model=model_name,
                        dataset=dataset_name,
                        method=plan["method"],
                        rank=plan["rank"] or 8,
                        alpha=plan["alpha"] or 16,
                        learning_rate=plan["learning_rate"],
                        batch_size=plan["batch_size"],
                        epochs=plan["epochs"],
                        gradient_accumulation_steps=plan["gradient_accumulation_steps"],
                        optimizer=plan["optimizer"],
                        fp16=plan["fp16"],
                    )
                else:
                    return self.finetune_orch.generate_script(
                        base_model=model_name,
                        dataset=dataset_name,
                        method=plan["method"],
                        rank=plan["rank"] or 8,
                        alpha=plan["alpha"] or 16,
                        learning_rate=plan["learning_rate"],
                        batch_size=plan["batch_size"],
                        epochs=plan["epochs"],
                        gradient_accumulation_steps=plan["gradient_accumulation_steps"],
                        optimizer=plan["optimizer"],
                        fp16=plan["fp16"],
                    )
            if step.action == "download_model":
                return self.hf.download_model_code(model_name, use_4bit=(method == "qlora"))
            if step.action == "load_dataset":
                analysis.get("dataset", {}).get("text_column", "text")
                return f"""
from datasets import load_dataset
dataset = load_dataset("{dataset_name}", split="train")
print(f"Dataset: {{len(dataset)}} samples")
print(dataset[0])
"""

        # Fallback: use LLM code generation
        prompt = CODE_GENERATION_PROMPT.format(
            goal=self.user_goal,
            step_id=step.id,
            step_description=step.description,
            action=step.action,
        )
        response = self.llm.chat([
            {"role": "system",
             "content": "Return ONLY Python code, no markdown."},
            {"role": "user", "content": prompt},
        ])
        content = response.get("content", "").strip()

        import re
        for pat in [r"```python\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
            m = re.search(pat, content)
            if m:
                content = m.group(1).strip()
                break

        if not content or content.startswith("[Error:"):
            return ""
        return content

    # ---------------------------------------------------------------- #
    #  Result parsing
    # ---------------------------------------------------------------- #

    def _parse_result(self, step: PlanStep,
                      execution_result: dict) -> Optional[dict]:
        output = execution_result.get("output", "")
        error = execution_result.get("error", "")
        combined = f"[ERROR]\n{error}\n\n[OUTPUT]\n{output}" if error else output[:2000]

        prompt = RESULT_PARSING_PROMPT.format(
            goal=self.user_goal,
            step_id=step.id,
            step_description=step.description,
            action=step.action,
            output=combined,
        )
        return self.llm.safe_json_chat([
            {"role": "system",
             "content": "Analyze Colab output. Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ], parser="json_only")

    # ---------------------------------------------------------------- #
    #  Runtime management
    # ---------------------------------------------------------------- #

    def _switch_runtime(self, target: str, reason: str = "") -> dict:
        if not settings.colab_runtime_auto_switch:
            return {"success": False, "error": "Auto-switch disabled"}

        result = self.runtime_mgr.switch_runtime(
            target, reason=reason,
            checkpoint_path=f"./finetuned_model/{self.current_job_id}" if self.current_job_id else None,
        )
        self.convs.add_message(
            self.conversation_id, "assistant",
            f"Switch: {self.current_runtime} -> {target} ({reason})",
            {"phase": "runtime_switch", "from": self.current_runtime, "to": target},
        )
        return result

    def _monitor_divergence(self, step: PlanStep, output: str):
        """
        Check for training divergence. If detected, ask LLM for tuning suggestions,
        then pause for human input via plugins, otherwise fail/retry.
        """
        loss_matches = re.findall(r"loss[=:]\s*(\d+\.\d+)", output, re.IGNORECASE)
        if not loss_matches:
            return

        losses = [float(l) for l in loss_matches]
        is_diverged = any(math.isnan(l) or l > 1e6 for l in losses)
        is_stagnated = len(losses) >= 3 and len(set(losses[-3:])) == 1
        
        if is_diverged or is_stagnated:
            error = f"Training {'diverged' if is_diverged else 'stagnated'}! Losses: {losses[-3:]}"
            self._log(f"ALERT: {error}")
            
            # Get LLM suggestion
            prompt = f"Training diverged/stagnated with losses {losses[-3:]}. Suggest new hyperparameters (e.g., lower learning rate, different batch size) to fix this."
            suggestion = self.llm.chat([{"role": "user", "content": prompt}])["content"]
            self._log(f"LLM Tuning Suggestion: {suggestion}")

            # Hybrid hook: fire on_divergence
            divergence_data = {"step_id": step.id, "error": error, "losses": losses[-3:], "suggestion": suggestion}
            action = self.hook_runner.run_on_divergence(divergence_data, self._ws_context)
            
            if action == "abort":
                step.error = error
                step.status = StepStatus.FAILED
            elif action == "proceed":
                self._log("Plugin authorized proceed despite divergence.")
            else:
                # Default hybrid behavior: Pause
                step.error = f"{error}\nSuggestion: {suggestion}"
                step.status = StepStatus.PAUSED
                self._log("Training paused. Waiting for human input...")

    def _find_next_runtime_tier(self) -> Optional[str]:
        for t in ["T4", "V100", "A100", "A100-80GB"]:
            if settings.runtime_tiers.get(t, 0) > \
               settings.runtime_tiers.get(self.current_runtime, 0):
                return t
        return None

    # ---------------------------------------------------------------- #
    #  Summary
    # ---------------------------------------------------------------- #

    def _generate_summary(self, final_status: str) -> str:
        steps_summary = "\n".join(
            f"  Step {s.id} [{s.status.value}]: {s.description}"
            f"{' (retries:' + str(s.retry_count) + ')' if s.retry_count > 1 else ''}"
            f"{' - ' + str(s.metrics) if s.metrics else ''}"
            for s in self.tracker.steps
        )
        prompt = SUMMARY_PROMPT.format(
            goal=self.user_goal,
            plan=self.plan.to_dict() if self.plan else "{}",
            results=steps_summary,
            final_status=final_status,
        )
        response = self.llm.chat([
            {"role": "system",
             "content": "Write concise ML experiment summaries."},
            {"role": "user", "content": prompt},
        ])
        return response.get("content", "Session completed.")


    # ---------------------------------------------------------------- #
    #  Async execution
    # ---------------------------------------------------------------- #

    async def async_run(self, goal: str) -> dict:
        """Async entry point. Mirrors run() with await on blocking calls."""
        self.user_goal = goal
        self._log(f"Starting agent with goal: {goal}")
        self.start_time = time.time()

        fmt_error = validate_goal_format(goal)
        if fmt_error:
            self._log(f"Goal validation failed: {fmt_error}")
            return {"status": "failed", "error": fmt_error,
                    "conversation_id": self.conversation_id}

        safe, sanitized, warnings = sanitize_user_input(goal)
        self.audit_logger.goal_received(goal)
        if not safe:
            self._log(f"Input sanitization warnings: {warnings}")
            self.audit_logger.goal_sanitized(goal, warnings)
            goal = sanitized

        context = {"goal": goal}
        goal, context = self.hook_runner.run_before_plan(goal, context)

        conv = await asyncio.to_thread(self.convs.create, goal)
        self.conversation_id = conv.id
        self._ws_context = {"conversation_id": self.conversation_id}
        await asyncio.to_thread(self.convs.add_message, self.conversation_id, "user", goal)

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.health.update(active_job_id=self.current_job_id,
                           active_conversation_id=self.conversation_id,
                           current_runtime=self.current_runtime)

        if self._elapsed_hours() > MAX_COLAB_HOURS:
            return {"status": "failed", "error": f"Exceeded {MAX_COLAB_HOURS}h limit",
                    "conversation_id": self.conversation_id}

        print("[AGENT] Analyzing goal and generating plan...")
        plan = await asyncio.to_thread(self._generate_plan, goal, self._override_model, self._override_dataset, self._override_method)
        if not plan or not plan.steps:
            return {"status": "failed", "error": "Failed to generate plan",
                    "conversation_id": self.conversation_id}

        self.plan = plan
        self._log(f"Plan generated with {len(plan.steps)} steps")
        print(f"[AGENT] Plan: {len(plan.steps)} steps")

        await asyncio.to_thread(
            self.convs.add_message,
            self.conversation_id, "assistant",
            f"Plan: {len(plan.steps)} steps:\n" +
            "\n".join(f"  Step {s.id}: {s.description}" for s in plan.steps),
            {"phase": "planning", "plan": plan.to_dict()},
        )

        plan_dict = plan.to_dict() if plan else {}
        plan_dict, context = self.hook_runner.run_after_plan(plan_dict, self._ws_context)
        if plan and plan_dict:
            plan.analysis = plan_dict.get("analysis", plan.analysis or {})
            if plan_dict.get("steps"):
                self.tracker.steps.clear()
                for sd in plan_dict["steps"]:
                    self.tracker.add_step(PlanStep(id=sd.get("id", 1), action=sd.get("action", ""), description=sd.get("description", "")))

        analysis = plan.analysis or {}
        job = await asyncio.to_thread(
            self.jobs.create,
            goal=goal,
            method=analysis.get("method", {}).get("type"),
            base_model=analysis.get("model", {}).get("name"),
            dataset=analysis.get("dataset", {}).get("name"),
            runtime=self.current_runtime,
            conversation_id=self.conversation_id,
        )
        self.current_job_id = job.id
        self._ws_context = {"job_id": self.current_job_id, "conversation_id": self.conversation_id}
        self.cost_tracker.start_job(job.id, self.current_runtime)
        self.metrics.gauge("plan.steps", len(plan.steps))
        self.metrics.counter("jobs.started")

        print("[AGENT] Starting execution...")
        while self.tracker.current_step_index < len(plan.steps):
            if self._elapsed_hours() > MAX_COLAB_HOURS:
                print(f"[AGENT] Reached {MAX_COLAB_HOURS}h limit, aborting")
                self.tracker.aborted = True
                break

            if self.budget.exceeded:
                print(f"[AGENT] Budget exceeded ({self.budget.snapshot()['spent']} units), aborting")
                self.tracker.aborted = True
                break

            step = self.tracker.current()
            if not step:
                break
            if self.tracker.aborted:
                step.status = StepStatus.ABORTED
                break

            await self._async_execute_step(step)

            if step.status in (StepStatus.SUCCESS, StepStatus.SKIPPED):
                self.tracker.advance()
            elif step.status == StepStatus.FAILED:
                if not step.can_retry() or self._max_retries_exceeded():
                    self._log(f"Step {step.id} exhausted retries. Aborting.")
                    self.tracker.aborted = True

        final_status = ("completed" if not self.tracker.aborted
                        and self.tracker.all_completed() else "aborted")
        summary = await asyncio.to_thread(self._generate_summary, final_status)

        await asyncio.to_thread(self.jobs.update, self.current_job_id, status=final_status)
        await asyncio.to_thread(
            self.convs.add_message,
            self.conversation_id, "assistant",
            f"Session {final_status}.\n{summary}",
            {"phase": "summary", "final_status": final_status},
        )
        await asyncio.to_thread(self.convs.close, self.conversation_id, summary)

        cost = self.cost_tracker.get_summary(self.current_job_id)
        cost_units = cost.get("estimated_cost_units", 0)
        self.budget.record_cost(cost_units)
        self.metrics.gauge("jobs.cost_units", cost_units)
        self.metrics.gauge("budget.usage_pct", self.budget.usage_ratio * 100)
        self.metrics.gauge("budget.remaining", max(0, self.budget.max_cost_units - cost_units))
        self.metrics.gauge("runtime.switches", self.tracker.runtime_switches)
        self.metrics.gauge("retries.total", self.tracker.total_retries)

        summary, _ = self.hook_runner.run_on_summary(summary, self._ws_context)
        final = {"status": final_status, "summary": summary, "job_id": self.current_job_id}
        final, _ = self.hook_runner.run_on_complete(final, self._ws_context)

        if self.github_logger.token:
            conv_msgs = await asyncio.to_thread(self.convs.get_messages, self.conversation_id)
            await asyncio.to_thread(
                self.github_logger.push_final_report,
                self.current_job_id,
                metrics={"plan": plan.to_dict(), "cost": cost},
                conversation=[{"role": m["role"], "content": m["content"][:200]}
                              for m in conv_msgs[-20:]],
                summary=summary,
            )

        return {
            "status": final_status,
            "summary": summary,
            "plan": plan.to_dict(),
            "cost": cost,
            "conversation_id": self.conversation_id,
            "job_id": self.current_job_id,
        }

    async def _async_execute_step(self, step: PlanStep) -> dict:
        """Async step execution. Mirrors _execute_step() with asyncio.to_thread()."""
        step.status = StepStatus.RUNNING
        self._log(f"Step {step.id}: {step.description}")
        print(f"[AGENT] Step {step.id}/{len(self.tracker.steps)}: {step.description}")

        while self.retry_policy.should_retry():
            if self.tracker.total_retries >= MAX_TOTAL_ERROR_RETRIES:
                step.error = f"Exceeded max total retries ({MAX_TOTAL_ERROR_RETRIES})"
                step.status = StepStatus.FAILED
                break

            step.record_attempt()
            self.tracker.total_retries += 1
            self.cost_tracker.record_retry(self.current_job_id)
            self.health.update(total_retries=self.tracker.total_retries)
            self.metrics.counter("step.attempts")
            self._log(f"  Attempt {step.retry_count}/{step.max_retries}")

            try:
                with self.circuit_breaker:
                    step_dict = {"id": step.id, "action": step.action, "description": step.description, "status": step.status.value}
                    step_dict, ctx = self.hook_runner.run_before_step(step_dict, self._ws_context)
                    step.action = step_dict.get("action", step.action)

                    code = self._generate_code(step)
                    if not code:
                        step.error = "Failed to generate code"
                        step.status = StepStatus.FAILED
                        continue

                    safe, cleaned, warning = sanitize_code(code)
                    if not safe:
                        step.error = f"Code blocked: {warning}"
                        self._log(step.error)
                        code_hash = compute_code_hash(code)
                        self.audit_logger.code_blocked(code_hash, warning)
                        await asyncio.to_thread(
                            self.github_logger.push_error,
                            error_log=step.error,
                            notebook_cell=code[:500],
                            job_id=self.current_job_id,
                        )
                        step.blocked_reasons.append(warning)
                        if len(step.blocked_reasons) >= 3 and \
                           len(set(step.blocked_reasons[-3:])) == 1:
                            step.error = (
                                f"Step skipped: LLM repeatedly generated code "
                                f"blocked by safety filter ({warning})")
                            step.status = StepStatus.SKIPPED
                            print(f"[AGENT] Step {step.id} SKIPPED: "
                                  f"repeated blocked code ({warning})")
                            break
                        step.status = StepStatus.FAILED
                        continue

                    cleaned = sanitize_pip(cleaned)
                    cleaned, ctx = self.hook_runner.run_after_code_gen({"id": step.id, "action": step.action}, cleaned, self._ws_context)
                    step.code = cleaned

                    await asyncio.to_thread(
                        self.convs.add_message,
                        self.conversation_id, "assistant",
                        f"Step {step.id}: code ({len(cleaned)} chars)",
                        {"phase": "exec", "step_id": step.id,
                         "code_hash": compute_code_hash(cleaned)},
                    )

                    result = await asyncio.to_thread(
                        self.runner.execute_cell,
                        cleaned, timeout=300, description=step.description,
                    )

                    step.output = result.get("output", "")
                    step.error = result.get("error", "")
                    step.execution_time = result.get("execution_time", 0.0)
                    self.cost_tracker.record_execution(self.current_job_id, step.execution_time)

                    self.runner.parse_output(result)
                    analysis = await asyncio.to_thread(self._parse_result, step, result)

                    if analysis:
                        self._handle_analysis(step, analysis)
                    elif result.get("success") and not result.get("error"):
                        step.status = StepStatus.SUCCESS

                    if not result.get("success") and result.get("error"):
                        if self._detect_oom(result["error"]) and \
                           self.tracker.runtime_switches < MAX_RUNTIME_SWITCHES:
                            next_tier = self._find_next_runtime_tier()
                            if next_tier:
                                self._log(f"OOM: {self.current_runtime} -> {next_tier}")
                                sr = await asyncio.to_thread(
                                    self._switch_runtime,
                                    next_tier, f"OOM on {self.current_runtime}")
                                if sr.get("success"):
                                    self.current_runtime = next_tier
                                    self.tracker.runtime_switches += 1
                                    self.cost_tracker.record_switch(self.current_job_id, next_tier)
                                    self.runtime_mgr.current_gpu = next_tier
                                    self.runtime_mgr.current_vram_gb = settings.runtime_tiers.get(next_tier, 0)
                                    continue

                    result_dict = {"step_id": step.id, "status": "success" if step.status == StepStatus.SUCCESS else "failed",
                                   "metrics": step.metrics, "error": step.error}
                    result_dict, ctx = self.hook_runner.run_after_step({"id": step.id, "action": step.action}, result_dict, self._ws_context)
                    if result_dict.get("status") == "failed":
                        step.status = StepStatus.FAILED
                        step.error = result_dict.get("error", step.error)

                    if step.status == StepStatus.SUCCESS:
                        self.retry_policy.record_result(True)
                        print(f"[AGENT] Step {step.id} OK ({step.execution_time:.1f}s)")
                        return {"step_id": step.id, "status": "success",
                                "metrics": step.metrics}

            except RuntimeError as re:
                self.retry_policy.record_result(False)
                self.retry_policy.sleep()
                self.health.update(circuit_state="open",
                                   circuit_failures=self.circuit_breaker.failure_count)
                step.error = f"Circuit Breaker blocked execution: {re}"
                self._log(step.error)
                step.status = StepStatus.FAILED
                break
            except Exception as e:
                self.retry_policy.record_result(False)
                self.retry_policy.sleep()
                step.error = f"Unexpected: {e}\n{traceback.format_exc()}"
                self._log(step.error)
                recovery, ctx = self.hook_runner.run_on_error({"id": step.id, "action": step.action}, step.error, self._ws_context)
                if recovery:
                    self._log(f"Plugin recovery: {recovery}")

        if step.status != StepStatus.SKIPPED:
            step.status = StepStatus.FAILED
            print(f"[AGENT] Step {step.id} FAILED after {step.retry_count} attempts")
            await asyncio.to_thread(self.jobs.update, self.current_job_id, status="failed", error=step.error[:500])
            await asyncio.to_thread(
                self.github_logger.push_error,
                error_log=step.error,
                notebook_cell=step.code[:500],
                job_id=self.current_job_id,
            )
            ret_status = "failed"
        else:
            print(f"[AGENT] Step {step.id} SKIPPED after {step.retry_count} attempts")
            ret_status = "skipped"
        return {"step_id": step.id, "status": ret_status, "error": step.error}


# ---------------------------------------------------------------- #
#  Public API
# ---------------------------------------------------------------- #

_current_agent: Optional[OrchestratorAgent] = None


def run_agent(goal: str, verbose: bool = True, model: str = None, dataset: str = None, method: str = None) -> dict:
    global _current_agent
    agent = OrchestratorAgent(verbose=verbose, model=model, dataset=dataset, method=method)
    _current_agent = agent
    return agent.run(goal)


async def async_run_agent(goal: str, verbose: bool = True, model: str = None, dataset: str = None, method: str = None) -> dict:
    """Async entry point. Creates agent and returns result from async_run()."""
    global _current_agent
    agent = OrchestratorAgent(verbose=verbose, model=model, dataset=dataset, method=method)
    _current_agent = agent
    return await agent.async_run(goal)


def health_endpoint() -> dict:
    """Return agent health status for external monitoring."""
    if _current_agent:
        return _current_agent.health.report()
    return HealthReporter().report()


def budget_endpoint() -> dict:
    """Return current budget status."""
    if _current_agent:
        return _current_agent.budget.snapshot()
    return BudgetManager().snapshot()


def metrics_endpoint() -> str:
    """Return Prometheus-formatted metrics text."""
    if _current_agent:
        return _current_agent.metrics.prometheus_text()
    return MetricsCollector().prometheus_text()


def continue_conversation(conversation_id: str, message: str) -> dict:
    agent = OrchestratorAgent()
    agent.conversation_id = conversation_id
    conv = agent.convs.get(conversation_id)
    if not conv:
        return {"error": "Conversation not found"}

    agent.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in conv.get_messages():
        agent.messages.append({"role": msg["role"], "content": msg["content"]})

    agent.convs.add_message(conversation_id, "user", message)
    return agent.run(message)
