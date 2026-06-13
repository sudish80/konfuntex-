"""
Phases 76-83 — UI & Interface.

Provides:
  - GradioChatInterface       chat UI with Gradio            (76)
  - StreamlitDashboard        real-time dashboard            (77)
  - IPythonWidgetController   in-notebook controls           (78)
  - TerminalMode              headless CLI mode              (79)
  - NotificationSystem        email/slack/discord alerts     (80)
  - JobQueueUI                queue + manage multiple jobs   (81)
  - HumanApprovalModal        confirm critical actions       (82)
  - CodeEditorWithDiff        diff + edit before exec        (83)
"""
import json
import os


# ==================================================================== #
#  76 — GradioChatInterface
# ==================================================================== #

class GradioChatInterface:
    """
    Build a Gradio chat UI for the agent.
    """

    @staticmethod
    def build_code() -> str:
        return """
import gradio as gr
import threading
import time

# Agent state
agent_state = {"paused": False, "stopped": False, "output": []}

def process_goal(goal, history):
    '''Main agent loop integrated with Gradio.'''
    history = history or []
    history.append({"role": "user", "content": goal})

    # Yield initial response
    history.append({"role": "assistant", "content": "🤔 Planning..."})
    yield history

    # Simulate agent steps
    steps = [
        "📡 Detecting runtime: T4 GPU (16 GB VRAM)",
        "📥 Downloading model...",
        "📊 Loading dataset...",
        "⚙ Configuring QLoRA training...",
        "🚀 Training (this will take a while)...",
        "✅ Done! Loss: 0.342, Accuracy: 89.2%",
    ]

    for step in steps:
        if agent_state["stopped"]:
            history[-1]["content"] += "\\n\\n⛔ Stopped by user"
            break
        while agent_state["paused"] and not agent_state["stopped"]:
            time.sleep(0.5)

        history[-1]["content"] = f"🤔 Working...\\n{step}"
        yield history
        time.sleep(1)

    history[-1]["content"] = history[-1]["content"].replace("🤔 Working...\\n", "")
    yield history

def toggle_pause():
    agent_state["paused"] = not agent_state["paused"]
    return "▶ Resume" if agent_state["paused"] else "⏸ Pause"

def stop_agent():
    agent_state["stopped"] = True
    return "⛔ Stopping..."

# Build UI
with gr.Blocks(title="Colab Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Colab Fine-Tuning Agent")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=500)
            msg = gr.Textbox(
                label="Your Goal",
                placeholder="e.g., Fine-tune Phi-2 for code generation",
            )
            with gr.Row():
                submit = gr.Button("🚀 Submit", variant="primary")
                pause_btn = gr.Button("⏸ Pause")
                stop_btn = gr.Button("⏹ Stop", variant="stop")

    submit.click(process_goal, [msg, chatbot], chatbot)
    pause_btn.click(toggle_pause, None, pause_btn)
    stop_btn.click(stop_agent, None, stop_btn)

demo.launch(share=True)
"""


# ==================================================================== #
#  77 — StreamlitDashboard
# ==================================================================== #

class StreamlitDashboard:
    """
    Real-time Streamlit dashboard for monitoring.
    """

    @staticmethod
    def build_code() -> str:
        return """
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

st.set_page_config(page_title="Colab Agent Dashboard", layout="wide")
st.title("🤖 Colab Agent Dashboard")

# Sidebar
with st.sidebar:
    st.header("Control")
    if st.button("⏸ Pause Agent"):
        st.session_state.paused = True
    if st.button("▶ Resume Agent"):
        st.session_state.paused = False
    st.divider()
    st.metric("GPU", "T4", "16 GB VRAM")
    st.metric("Session", f"{datetime.now().strftime('%H:%M:%S')}")

# Main layout
col1, col2, col3 = st.columns(3)
col1.metric("Current Loss", "0.342", "-0.012")
col2.metric("Accuracy", "89.2%", "+2.1%")
col3.metric("Steps", "1,234", "12")

# Live plots
st.subheader("Training Metrics")
tab1, tab2 = st.tabs(["Loss", "GPU Memory"])

with tab1:
    # Simulated live loss curve
    df = pd.DataFrame({
        "step": range(100),
        "loss": [0.8 * (0.98 ** i) + 0.1 for i in range(100)],
    })
    fig = px.line(df, x="step", y="loss", title="Training Loss")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    mem_df = pd.DataFrame({
        "step": range(100),
        "mem": [12 + 2 * (i % 10 == 0) for i in range(100)],
    })
    fig2 = px.area(mem_df, x="step", y="mem", title="GPU Memory (GB)")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Job History")
history = pd.DataFrame({
    "Job": ["J001", "J002", "J003"],
    "Model": ["phi-2", "llama-3-8b", "mistral-7b"],
    "Status": ["✅ Success", "✅ Success", "❌ Failed"],
})
st.dataframe(history, use_container_width=True)

st.button("🔄 Refresh", type="primary")
"""


# ==================================================================== #
#  78 — IPythonWidgetController
# ==================================================================== #

class IPythonWidgetController:
    """
    In-notebook control buttons using ipywidgets.
    """

    @staticmethod
    def build_code() -> str:
        return """
import ipywidgets as widgets
from IPython.display import display
import threading

# Control state
control = {"paused": False, "stopped": False, "runtime": "T4"}

# Button handlers
def on_pause_click(b):
    control["paused"] = not control["paused"]
    pause_btn.description = "▶ Resume" if control["paused"] else "⏸ Pause"

def on_stop_click(b):
    control["stopped"] = True
    stop_btn.description = "⛔ Stopped"
    stop_btn.disabled = True

def on_runtime_change(change):
    control["runtime"] = change["new"]

# Build widgets
pause_btn = widgets.Button(description="⏸ Pause", button_style="warning")
stop_btn = widgets.Button(description="⏹ Stop", button_style="danger")
runtime_dropdown = widgets.Dropdown(
    options=["T4", "V100", "A100", "A100-80GB"],
    value="T4",
    description="Runtime:",
)
progress = widgets.IntProgress(value=0, min=0, max=100, description="Progress:")

pause_btn.on_click(on_pause_click)
stop_btn.on_click(on_stop_click)
runtime_dropdown.observe(on_runtime_change, names="value")

# Layout
ui = widgets.VBox([
    widgets.HBox([pause_btn, stop_btn]),
    runtime_dropdown,
    progress,
])
display(ui)

# Agent loop checks control state
def agent_loop():
    for i in range(100):
        if control["stopped"]:
            print("Agent stopped by user")
            break
        while control["paused"]:
            import time; time.sleep(0.5)
        progress.value = i + 1
        print(f"Step {i+1}/100")
        import time; time.sleep(0.1)

threading.Thread(target=agent_loop, daemon=True).start()
"""


# ==================================================================== #
#  79 — TerminalMode
# ==================================================================== #

class TerminalMode:
    """
    Headless terminal mode for automated/CI runs.
    """

    @staticmethod
    def cli_code() -> str:
        return """
#!/usr/bin/env python3
import sys
import argparse
import logging
from tqdm import tqdm
from agent.core import OrchestratorAgent

def main():
    parser = argparse.ArgumentParser(description="Colab Agent - Terminal Mode")
    parser.add_argument("goal", nargs="?", help="Training goal")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without UI prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only, no execution")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log", default="terminal.log",
                        help="Log file path")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log, level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.goal:
        goal = input("Enter training goal: ")
    else:
        goal = args.goal
        print(f"Goal: {goal}")

    if args.dry_run:
        print("[DRY RUN] Planning only...")
        agent = OrchestratorAgent(verbose=args.verbose)
        plan = agent._generate_plan(goal)
        if plan:
            print(f"Plan: {len(plan.steps)} steps")
            for s in plan.steps:
                print(f"  {s.id}. {s.description}")
        return 0

    print("Starting agent...")
    agent = OrchestratorAgent(verbose=args.verbose)
    result = agent.run(goal)

    status = result.get("status", "unknown")
    print(f"Status: {status}")
    if status == "completed":
        print(f"Summary: {result.get('summary', '')}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""


# ==================================================================== #
#  80 — NotificationSystem
# ==================================================================== #

class NotificationSystem:
    """
    Send notifications on job completion via email, Slack, Discord.
    """

    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = config_path

    def notify_code(self, job_id: str, status: str,
                    metrics: dict, hf_url: str = "") -> str:
        return f"""
import os, json, smtplib, requests
from email.mime.text import MIMEText
from datetime import datetime

job_id = "{job_id}"
status = "{status}"
hf_url = "{hf_url}"
metrics = {json.dumps(metrics)}

message = f\"\"\"Job {{job_id}} {{status.upper()}}
Time: {{datetime.now().isoformat()}}
Final Loss: {{metrics.get('final_loss', 'N/A')}}
Accuracy: {{metrics.get('accuracy', 'N/A')}}
Model: {{hf_url or 'N/A'}}
\"\"\"

# Email notification
smtp_server = os.environ.get("SMTP_SERVER", "")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
email_from = os.environ.get("EMAIL_FROM", "")
email_to = os.environ.get("EMAIL_TO", "")
email_pass = os.environ.get("EMAIL_PASSWORD", "")

if email_from and email_to and email_pass:
    msg = MIMEText(message)
    msg["Subject"] = f"[Colab Agent] Job {{job_id}} {{status}}"
    msg["From"] = email_from
    msg["To"] = email_to
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, email_pass)
            server.send_message(msg)
        print(f"Email sent to {{email_to}}")
    except Exception as e:
        print(f"Email failed: {{e}}")

# Slack webhook
slack_url = os.environ.get("SLACK_WEBHOOK_URL", "")
if slack_url:
    try:
        requests.post(slack_url, json={{"text": message}})
        print("Slack notification sent")
    except Exception as e:
        print(f"Slack failed: {{e}}")

# Discord webhook
discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
if discord_url:
    try:
        requests.post(discord_url, json={{"content": message}})
        print("Discord notification sent")
    except Exception as e:
        print(f"Discord failed: {{e}}")
"""


# ==================================================================== #
#  81 — JobQueueUI
# ==================================================================== #

class JobQueueUI:
    """
    Queue + manage multiple jobs with priority.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS job_queue (
        queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal TEXT,
        priority INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        added_at TIMESTAMP,
        started_at TIMESTAMP,
        job_id TEXT,
        estimated_time_min REAL
    );
    """

    def __init__(self, db_path: str = "./job_queue.db"):
        self.db_path = db_path

    def queue_code(self) -> str:
        return f"""
import sqlite3, json, threading, time
from datetime import datetime
from queue import Queue

db_path = "{self.db_path}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS job_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    added_at TIMESTAMP,
    started_at TIMESTAMP,
    job_id TEXT,
    estimated_time_min REAL
)''')

conn.commit()

def add_job(goal, priority=0, estimated_time=30):
    cursor.execute('''
        INSERT INTO job_queue (goal, priority, status, added_at, estimated_time_min)
        VALUES (?, ?, 'pending', ?, ?)
    ''', (goal, priority, datetime.now().isoformat(), estimated_time))
    conn.commit()
    print(f"Job queued: {{goal[:50]}}...")

def get_next_job():
    cursor.execute('''
        SELECT * FROM job_queue WHERE status='pending'
        ORDER BY priority DESC, added_at ASC LIMIT 1
    ''')
    return cursor.fetchone()

def process_queue():
    while True:
        job = get_next_job()
        if job:
            queue_id, goal, priority, status, added, started, job_id, est_time = job
            cursor.execute("UPDATE job_queue SET status='running', started_at=? WHERE queue_id=?",
                          (datetime.now().isoformat(), queue_id))
            conn.commit()
            print(f"Processing: {{goal[:50]}}...")
            # Run agent here
            # result = agent.run(goal)
            time.sleep(2)  # Simulate
            cursor.execute("UPDATE job_queue SET status='completed', job_id=? WHERE queue_id=?",
                          ("sim_" + str(queue_id), queue_id))
            conn.commit()
        else:
            time.sleep(5)

# Start worker thread
worker = threading.Thread(target=process_queue, daemon=True)
worker.start()
print("Job queue worker started")
"""


# ==================================================================== #
#  82 — HumanApprovalModal
# ==================================================================== #

class HumanApprovalModal:
    """
    Require human approval for critical actions.
    """

    APPROVAL_REQUIRED = [
        "switch_runtime",
        "full_finetune_large_model",
        "push_to_hf_public",
        "delete_checkpoints",
        "commit_to_github",
    ]

    @staticmethod
    def modal_code() -> str:
        return """
import json
from datetime import datetime

approval_required = {json.dumps(HumanApprovalModal.APPROVAL_REQUIRED)}
approval_log = []

def request_approval(action_type, description, timeout_min=5):
    if action_type not in approval_required:
        return True

    print(f"\\n{'='*60}")
    print(f"⚠ APPROVAL REQUIRED: {action_type}")
    print(f"Description: {description}")
    print(f"Timeout: {timeout_min} minutes")
    print(f"{'='*60}")

    # In UI mode, show modal
    # In terminal mode, ask for input
    try:
        import gradio as gr
        # Gradio mode: show Textbox with approve/deny
        response = input("Approve? [y/N/edit]: ").strip().lower()
    except Exception:
        response = input("Approve? [y/N/edit]: ").strip().lower()

    approved = response in ("y", "yes", "approve")
    modified = response == "edit"

    entry = {
        "action_type": action_type,
        "description": description,
        "approved": approved,
        "modified": modified,
        "timestamp": datetime.now().isoformat(),
    }
    approval_log.append(entry)

    with open("approvals.json", "a") as f:
        f.write(json.dumps(entry) + "\\n")

    return approved

# In config.yaml add:
# auto_approve: false  # Set true for headless mode
"""


# ==================================================================== #
#  83 — CodeEditorWithDiff
# ==================================================================== #

class CodeEditorWithDiff:
    """
    Show diff and allow editing before execution.
    """

    @staticmethod
    def diff_code() -> str:
        return """
import difflib

def show_diff(original, new_code):
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile="original", tofile="new",
    )
    diff_text = "".join(diff)
    print("\\n=== CODE DIFF ===")
    print(diff_text)
    print("=== END DIFF ===")

    # In Gradio: show in gr.HTML or gr.Code with diff highlighting
    # In terminal: colored diff with rich library
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        console = Console()
        console.print(Syntax(new_code, "python", theme="monokai"))
    except ImportError:
        pass

    # Ask user
    response = input("\\nExecute? [y/N/edit]: ").strip().lower()
    if response == "edit":
        print("Enter new code (Ctrl+D to finish):")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        return "\\n".join(lines)
    return new_code if response in ("y", "yes") else None
"""
