import optuna
from agent.plugin import Plugin, plugin

@plugin(name="OptunaTuner", description="Hyperparameter optimization using Optuna")
class OptunaTuner(Plugin):
    def before_plan(self, goal: str, context: dict) -> tuple[str, dict]:
        # Inject Optuna search space into the planning prompt
        return f"{goal}\n\n[HYPERPARAMETER_SEARCH_REQUIRED]", context

    def after_step(self, step: dict, result: dict, context: dict) -> tuple[dict, dict]:
        if step.get("action") == "train" and "metrics" in result:
            # Example: Report loss to Optuna
            # study.tell(trial, result["metrics"]["loss"])
            pass
        return result, context
