"""
Phase 6 — Gradio interface for the Colab Agent.

Runs inside Colab or locally. Shows:
  - Agent's step-by-step thinking
  - Current runtime and resource usage
  - Manual override (pause, switch runtime, edit code)
  - Streaming logs and metrics
  - Download buttons for model, conversation, metrics
"""
import json
import time

import gradio as gr

from agent.core import OrchestratorAgent
from colab.executor import ColabRunner
from colab.runtime import RuntimeManager
from agent.safety import CostTracker

# ------------------------------------------------------------------ #
#  Global state
# ------------------------------------------------------------------ #

agent_state = {
    "running": False,
    "paused": False,
    "agent": None,
    "current_step": "",
    "logs": [],
    "runtime_info": {},
    "job_id": None,
    "cost_tracker": CostTracker(),
}

runner = ColabRunner()
rt_mgr = RuntimeManager()


def format_log(msg: str, level: str = "INFO") -> str:
    ts = time.strftime("%H:%M:%S")
    return f"[{ts}] [{level}] {msg}"


def get_runtime_display() -> str:
    info = runner.detect_current_runtime()
    agent_state["runtime_info"] = info.to_dict()
    return (
        f"**GPU:** {info.gpu_name}  |  "
        f"**VRAM:** {info.vram_total_gb:.1f} GB / {info.vram_free_gb:.1f} GB free  |  "
        f"**RAM:** {info.ram_total_gb:.1f} GB / {info.ram_available_gb:.1f} GB free  |  "
        f"**Runtime:** {info.runtime_label}"
    )


def run_goal(goal: str, progress=gr.Progress()):
    """Main entry point — runs the agent in a background thread."""
    if agent_state["running"]:
        yield format_log("Agent already running", "WARN")
        return

    agent_state["running"] = True
    agent_state["paused"] = False
    agent_state["logs"] = []
    agent = OrchestratorAgent(verbose=False)
    agent_state["agent"] = agent

    yield from _stream_agent(agent, goal, progress)

    agent_state["running"] = False


def _stream_agent(agent: OrchestratorAgent, goal: str, progress):
    """Stream agent decisions and execution results."""
    yield format_log(f"Starting goal: {goal}", "GOAL")

    # Phase 1: Plan
    yield format_log("Generating plan...", "PLAN")
    plan = agent._generate_plan(goal)
    if not plan or not plan.steps:
        yield format_log("Failed to generate plan", "ERROR")
        return

    agent.plan = plan
    agent_state["logs"].append(json.dumps(plan.to_dict(), indent=2))

    step_count = len(plan.steps)
    yield format_log(f"Plan created: {step_count} steps", "PLAN")
    for s in plan.steps:
        yield format_log(f"  Step {s.id}: {s.description}", "PLAN")

    # Phase 2: Execute steps
    yield format_log("Starting execution...", "EXEC")
    progress(0.0)

    while agent.tracker.current_step_index < step_count:
        if agent_state["paused"]:
            yield format_log("PAUSED — waiting for resume...", "WARN")
            while agent_state["paused"]:
                time.sleep(0.5)
            yield format_log("Resumed", "INFO")

        if agent.tracker.aborted:
            yield format_log("Agent aborted", "ERROR")
            break

        step = agent.tracker.current()
        if not step:
            break

        # Update UI
        agent_state["current_step"] = f"Step {step.id}: {step.description}"
        yield format_log(f"Executing: {step.description}", "STEP")
        progress(agent.tracker.current_step_index / max(step_count, 1))

        # Execute with streaming
        agent._execute_step(step)

        if step.status.value == "success":
            yield format_log(
                f"Step {step.id} OK ({step.execution_time:.1f}s)",
                "OK")
        else:
            yield format_log(
                f"Step {step.id} FAILED: {step.error[:200]}",
                "ERROR")

        # Log metrics
        if step.metrics:
            yield format_log(f"  Metrics: {step.metrics}", "DATA")

        agent_state["logs"].append(
            json.dumps({"step": step.id, "status": step.status.value,
                        "error": step.error, "metrics": step.metrics,
                        "time": step.execution_time}))

        if step.status.value == "failed" and not step.can_retry():
            agent.tracker.aborted = True
            yield format_log("Max retries exhausted, aborting", "ERROR")
            break

        if step.status.value == "success":
            agent.tracker.advance()

    # Phase 3: Summary
    progress(1.0)
    final = "completed" if not agent.tracker.aborted and agent.tracker.all_completed() else "aborted"
    summary = agent._generate_summary(final)
    yield format_log(f"Session {final}. {summary}", "DONE")

    # Cost
    cost = agent_state["cost_tracker"].get_summary(agent.current_job_id or "?")
    yield format_log(f"Cost estimate: {cost}", "DONE")


def pause_agent():
    agent_state["paused"] = True
    return "PAUSED"


def resume_agent():
    agent_state["paused"] = False
    return "RUNNING"


def get_logs():
    return "\n".join(agent_state["logs"])


def refresh_runtime():
    return get_runtime_display()


# ------------------------------------------------------------------ #
#  Build Gradio interface
# ------------------------------------------------------------------ #

def build_ui():
    with gr.Blocks(title="Colab Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Colab Agent — Fine-Tuning Studio")
        gr.Markdown("Autonomous LLM-powered agent for fine-tuning HuggingFace models in Colab.")

        with gr.Row():
            with gr.Column(scale=2):
                goal_input = gr.Textbox(
                    label="Your Goal",
                    placeholder="e.g., Fine-tune Phi-2 on Dolly for instruction following",
                    lines=3,
                )
                with gr.Row():
                    run_btn = gr.Button("🚀 Run", variant="primary")
                    pause_btn = gr.Button("⏸️ Pause")
                    resume_btn = gr.Button("▶️ Resume")

            with gr.Column(scale=1):
                runtime_display = gr.Markdown(get_runtime_display())
                refresh_btn = gr.Button("🔄 Refresh Runtime")

        with gr.Row():
            with gr.Column(scale=2):
                output = gr.Textbox(label="Agent Output", lines=20, max_lines=40)
            with gr.Column(scale=1):
                override_goal = gr.Textbox(label="Override: New Goal", lines=2)
                override_btn = gr.Button("Apply Override")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Manual Code Override")
                code_override = gr.Textbox(
                    label="Colab Code to Execute",
                    lines=5,
                    placeholder="# Enter Python code to execute directly in Colab",
                )
                exec_btn = gr.Button("⚡ Execute Code")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Runtime Controls")
                with gr.Row():
                    runtime_selector = gr.Dropdown(
                        choices=["T4", "V100", "A100", "A100-80GB", "TPU", "None"],
                        label="Target Runtime",
                        value="V100",
                    )
                    switch_btn = gr.Button("🔄 Switch Runtime")

        gr.Markdown("---")
        with gr.Row():
            gr.Button("📥 Download Model (adapter)")
            download_logs = gr.Button("📥 Download Logs")
            gr.Button("📥 Download Metrics")

        # ---- Event wiring ----

        async def run_wrapper(goal, progress=gr.Progress()):
            logs = []
            for msg in run_goal(goal, progress):
                logs.append(msg)
                yield "\n".join(logs[-50:])
                output.value = "\n".join(logs[-50:])

        run_btn.click(
            fn=run_wrapper,
            inputs=[goal_input],
            outputs=[output],
        )

        pause_btn.click(fn=pause_agent, outputs=[output])
        resume_btn.click(fn=resume_agent, outputs=[output])
        refresh_btn.click(fn=refresh_runtime, outputs=[runtime_display])

        switch_btn.click(
            fn=lambda t: rt_mgr.switch_runtime(t),
            inputs=[runtime_selector],
            outputs=[output],
        )

        exec_btn.click(
            fn=lambda code: runner.execute_cell(code, timeout=300),
            inputs=[code_override],
            outputs=[output],
        )

        override_btn.click(
            fn=lambda g: run_goal(g) if g else "No goal",
            inputs=[override_goal],
            outputs=[output],
        )

        download_logs.click(
            fn=get_logs,
            outputs=[output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(debug=True, share=True)
