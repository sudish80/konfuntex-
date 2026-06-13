from agent.plugin import Plugin, plugin

@plugin(name="HumanApprovalPlugin", description="Pauses training for human approval", priority=0)
class HumanApprovalPlugin(Plugin):
    def before_step(self, step: dict, context: dict) -> tuple[dict, dict]:
        # Only pause for critical/expensive training steps
        if step.get("action") in ["train", "push_to_hub"]:
            input(f"\n[!] Human approval required for step {step['id']}: {step['description']}. Press Enter to continue...")
        return step, context
