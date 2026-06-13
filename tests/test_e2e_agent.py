"""End-to-end tests for the OrchestratorAgent — uses mock_llm_client fixture."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_plan_generation(mock_llm_client):
    from agent.core import OrchestratorAgent
    from storage.database import init_db
    init_db()
    agent = OrchestratorAgent()
    plan = agent._generate_plan("List 3 Python ML libraries")
    assert plan is not None
    assert len(plan.steps) >= 1


def test_agent_full_run(mock_llm_client):
    from agent.core import OrchestratorAgent
    from storage.database import init_db
    init_db()
    agent = OrchestratorAgent()
    result = agent.run("List 3 Python ML libraries")
    status = result.get("status", "unknown")
    assert status in ("completed", "partial"), f"Agent failed: {status}"


def test_code_generation(mock_llm_client):
    from agent.core import OrchestratorAgent, PlanStep
    from storage.database import init_db
    import ast
    init_db()
    agent = OrchestratorAgent()
    step = PlanStep(id=1, action="setup_environment",
                    description="Install Python ML libraries")
    code = agent._generate_code(step)
    assert code, "No code generated"
    python_lines = [ln for ln in code.split("\n") if not ln.strip().startswith("!")]
    python_code = "\n".join(python_lines)
    ast.parse(python_code if python_code.strip() else "pass")
