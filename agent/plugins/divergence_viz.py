import json
from agent.plugin import Plugin, plugin
from agent.events import get_event_bus

@plugin(name="DivergenceVisualizer", description="Pushes loss data to WebSocket on divergence")
class DivergenceVisualizer(Plugin):
    def on_divergence(self, divergence_data: dict, context: dict) -> dict:
        bus = get_event_bus()
        job_id = context.get("job_id")
        if job_id:
            bus.publish(job_id, "divergence_detected", {
                "step_id": divergence_data.get("step_id"),
                "losses": divergence_data.get("losses"),
                "error": divergence_data.get("error")
            })
        return {"action": "pause"}
