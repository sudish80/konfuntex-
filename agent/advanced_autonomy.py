"""
Phase 10 — Advanced Autonomy (items 96-100).

Provides:
  - AutoResumeManager         auto-resume on reconnect      (98)
  - MultiGoalPlanner          queue + prioritize goals      (99)
  - MetaAgentOrchestrator     high-level agent manager      (100)
"""
import json
import os
import time
import uuid
import logging
from typing import Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# ==================================================================== #
#  98 — AutoResumeManager
# ==================================================================== #

class AutoResumeManager:
    """
    Detect Colab disconnect and auto-resume training from last checkpoint.
    Saves state periodically and generates resume code.
    """

    def __init__(self, state_path: str = "agent_state.json"):
        self.state_path = state_path

    def save_state(self, job_id: str, step_id: int, step_index: int,
                   runtime: str, checkpoint_path: str,
                   metadata: dict = None):
        state = {
            "job_id": job_id,
            "step_id": step_id,
            "step_index": step_index,
            "runtime": runtime,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Agent state saved: {json.dumps(state)}")

    def load_state(self) -> Optional[dict]:
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load state: {e}")
            return None

    def has_state(self) -> bool:
        return os.path.exists(self.state_path)

    def clear_state(self):
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def generate_resume_code(self, agent_class_path: str = "agent.core.OrchestratorAgent") -> str:
        state = self.load_state()
        if not state:
            return "print('No saved state to resume from')"
        return f"""
import json, time
from {agent_class_path} import OrchestratorAgent

# Auto-resume from saved state
state = {json.dumps(state)}
print(f"Resuming job {{state['job_id']}} from step {{state['step_index']}}")
print(f"Checkpoint: {{state['checkpoint_path']}}")

agent = OrchestratorAgent(verbose=True)
agent.current_job_id = state['job_id']
agent.current_runtime = state['runtime']

# Restore from checkpoint
from models.finetune_orchestrator import FineTuneOrchestrator
orch = FineTuneOrchestrator()
resume_script = orch.generate_resume_script(
    checkpoint_path=state['checkpoint_path'],
    base_model=state.get('metadata', {{}}).get('base_model', ''),
    dataset=state.get('metadata', {{}}).get('dataset', ''),
)
print("Resume script generated: {{len(resume_script)}} chars")
"""

    def heartbeat(self, interval_sec: int = 60) -> Callable:
        """Return a no-op; in Colab, a cell that prints keeps the session alive."""
        import threading

        def _heartbeat(job_id: str, step: int):
            while True:
                time.sleep(interval_sec)
                self.save_state(job_id, step, 0, "", "", {"heartbeat": True})
                print(f"[heartbeat] Job {job_id} alive at {datetime.now(timezone.utc).isoformat()}")

        def _start(job_id: str, step: int):
            t = threading.Thread(target=_heartbeat, args=(job_id, step), daemon=True)
            t.start()
            return t

        return _start


# ==================================================================== #
#  99 — MultiGoalPlanner
# ==================================================================== #

@dataclass
class GoalItem:
    id: str = ""
    description: str = ""
    priority: int = 0  # higher = more important
    dependencies: list[str] = field(default_factory=list)
    estimated_steps: int = 1
    status: str = "pending"  # pending | queued | running | done | failed
    created_at: str = ""
    result: Optional[dict] = None


class MultiGoalPlanner:
    """
    Queue multiple goals, detect dependencies, and schedule execution order.
    """

    def __init__(self):
        self.goals: dict[str, GoalItem] = {}
        self.queue: list[str] = []
        self.max_consecutive_failures = 3
        self._fail_count = 0

    def add_goal(self, description: str, priority: int = 0,
                 dependencies: list[str] = None) -> str:
        gid = str(uuid.uuid4())[:8]
        self.goals[gid] = GoalItem(
            id=gid,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._reschedule()
        return gid

    def _reschedule(self):
        """Topological sort by priority then dependencies."""
        pending = {k: v for k, v in self.goals.items() if v.status == "pending"}
        ready = [
            k for k, v in pending.items()
            if all(dep not in pending or self.goals[dep].status == "done"
                   for dep in v.dependencies)
        ]
        ready.sort(key=lambda k: (-pending[k].priority, pending[k].created_at))
        self.queue = ready

    def next_goal(self) -> Optional[GoalItem]:
        self._reschedule()
        while self.queue:
            gid = self.queue[0]
            goal = self.goals.get(gid)
            if goal and goal.status == "pending":
                goal.status = "queued"
                return goal
            self.queue.pop(0)
        return None

    def mark_running(self, gid: str):
        if gid in self.goals:
            self.goals[gid].status = "running"

    def mark_done(self, gid: str, result: dict = None):
        if gid in self.goals:
            self.goals[gid].status = "done"
            self.goals[gid].result = result
            self._fail_count = 0
            self._reschedule()

    def mark_failed(self, gid: str):
        if gid in self.goals:
            self.goals[gid].status = "failed"
            self._fail_count += 1
            self._reschedule()

    def should_abort(self) -> bool:
        return self._fail_count >= self.max_consecutive_failures

    def get_summary(self) -> dict:
        return {
            "total": len(self.goals),
            "pending": sum(1 for g in self.goals.values() if g.status == "pending"),
            "queued": len(self.queue),
            "running": sum(1 for g in self.goals.values() if g.status == "running"),
            "done": sum(1 for g in self.goals.values() if g.status == "done"),
            "failed": sum(1 for g in self.goals.values() if g.status == "failed"),
            "consecutive_failures": self._fail_count,
        }

    def list_goals(self) -> list[dict]:
        return [
            {"id": g.id, "description": g.description, "priority": g.priority,
             "status": g.status, "dependencies": g.dependencies}
            for g in sorted(self.goals.values(),
                            key=lambda x: (-x.priority, x.created_at))
        ]


# ==================================================================== #
#  100 — MetaAgentOrchestrator
# ==================================================================== #

class MetaAgentOrchestrator:
    """
    High-level manager that:
      - accepts multiple goals
      - delegates to MultiGoalPlanner for scheduling
      - runs each goal via OrchestratorAgent
      - handles auto-resume across goals
      - stops on consecutive failures or user abort
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.planner = MultiGoalPlanner()
        self.resume_mgr = AutoResumeManager()
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.results: list[dict] = []
        self._aborted = False

    def submit_goal(self, description: str, priority: int = 0,
                    dependencies: list[str] = None) -> str:
        return self.planner.add_goal(description, priority, dependencies)

    def run_all(self, max_goals: int = 10) -> list[dict]:
        """Execute all queued goals in priority order."""
        from agent.core import OrchestratorAgent

        completed = 0
        while completed < max_goals:
            if self._aborted:
                logger.info("MetaAgent aborted by user")
                break

            if self.planner.should_abort():
                logger.warning("Too many consecutive failures, aborting")
                break

            goal = self.planner.next_goal()
            if not goal:
                logger.info("No more goals to execute")
                break

            self.planner.mark_running(goal.id)
            logger.info(f"MetaAgent running: {goal.description}")

            try:
                agent = OrchestratorAgent(verbose=self.verbose)
                result = agent.run(goal.description)

                status = result.get("status", "failed")
                if status == "completed":
                    self.planner.mark_done(goal.id, result)
                    self.results.append(result)
                else:
                    self.planner.mark_failed(goal.id)
                    self.results.append({"goal_id": goal.id, "status": status,
                                         "error": result.get("summary", "")})

            except Exception as e:
                logger.error(f"Goal failed with exception: {e}")
                self.planner.mark_failed(goal.id)
                self.results.append({"goal_id": goal.id, "status": "exception",
                                     "error": str(e)})

            completed += 1

        return self.results

    def abort(self):
        self._aborted = True

    def get_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "elapsed_hours": round((time.time() - self.start_time) / 3600, 2),
            "goals": self.planner.get_summary(),
            "results_count": len(self.results),
        }
