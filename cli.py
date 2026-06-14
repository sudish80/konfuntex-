#!/usr/bin/env python3
"""
Konfuntex CLI — Claude Code-style terminal interface.
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from datetime import timezone
from datetime import datetime

from config.settings import settings
from storage.database import init_db, get_session
from storage.jobs import JobStore, Job
from storage.conversations import ConversationStore, Conversation
from storage.models_store import ModelVersionStore, ModelVersion
from storage.metrics_store import MetricsStore
from agent.core import run_agent, migrate

# Module-level aliases for test compatibility
init_db = init_db
run_agent = run_agent
JobStore = JobStore
ConversationStore = ConversationStore
ModelVersionStore = ModelVersionStore
MetricsStore = MetricsStore

console = Console()

ACCENT = "bright_magenta"
HEADER = "bold bright_magenta"
DIM = "dim"
SUCCESS = "bright_green"
ERROR = "bright_red"
WARN = "yellow"
INFO = "bright_cyan"
MUTED = "grey62"

BANNER = r"""[bright_magenta]  _  ___  _  _ ___ _   _ _  _ _____ _____  __
 | |/ / || \| | __| | | | \| |_   _| __\ \/ /
 | ' <| || .  | _|| |_| | .  | | | | _| >  <
 |_|\_\_||_|\_|_|  \___/|_|\_| |_| |___/_/\_\[/bright_magenta]"""


def _header():
    console.print(BANNER)
    console.print(
        f"[{MUTED}]autonomous local fine-tuning agent  "
        f"[{ACCENT}]/help[/{ACCENT}] for commands[/{MUTED}]\n"
    )


def _sep():
    console.print(Rule(style="grey23"))


def _step(n, total, label, detail=""):
    prefix = f"[{MUTED}]step {n}/{total}[/{MUTED}]"
    body = f"[white]{label}[/white]"
    if detail:
        body += f"  [{MUTED}]{escape(detail)}[/{MUTED}]"
    console.print(f"  {prefix}  {body}")


def _ok(msg):
    console.print(f"  [{SUCCESS}]✔[/{SUCCESS}]  {msg}")


def _fail(msg):
    console.print(f"  [{ERROR}]✗[/{ERROR}]  [{ERROR}]{escape(msg)}[/{ERROR}]")


def _warn(msg):
    console.print(f"  [{WARN}]⚠[/{WARN}]  [{WARN}]{escape(msg)}[/{WARN}]")


def _info(msg):
    console.print(f"  [{INFO}]→[/{INFO}]  {msg}")


def _kv(key, val, key_w=18):
    console.print(f"    [{MUTED}]{key:<{key_w}}[/{MUTED}]  [{INFO}]{escape(str(val))}[/{INFO}]")


def _metrics_row(items: list[tuple]):
    """items = [(label, value, unit?), ...]"""
    cells = []
    for item in items:
        label, val = item[0], item[1]
        unit = item[2] if len(item) > 2 else ""
        cells.append(
            Panel(
                f"[{ACCENT}]{val}[/{ACCENT}][{MUTED}]{unit}[/{MUTED}]",
                title=f"[{MUTED}]{label}[/{MUTED}]",
                border_style="grey23",
                padding=(0, 2),
            )
        )
    console.print(Columns(cells))


def _code_block(code: str, language="python"):
    console.print(Syntax(code, language, theme="monokai", line_numbers=False, background_color="default"))


def cmd_init():
    os.makedirs(settings.data_dir, exist_ok=True)
    init_db()
    _ok(f"database initialised at [white]{settings.data_dir}[/white]")


def cmd_run(goal: str, model: str = None, dataset: str = None, method: str = None, executor: str = "auto",
            browser_path: str = None):
    _sep()
    console.print(f"\n  [{ACCENT}]❯[/{ACCENT}]  [white bold]{escape(goal)}[/white bold]\n")

    overrides = []
    if model:
        overrides.append(f"model=[{INFO}]{model}[/{INFO}]")
    if dataset:
        overrides.append(f"dataset=[{INFO}]{dataset}[/{INFO}]")
    if method:
        overrides.append(f"method=[{INFO}]{method}[/{INFO}]")
    if executor != "auto":
        overrides.append(f"executor=[{INFO}]{executor}[/{INFO}]")
    if overrides:
        console.print(f"  [{MUTED}]" + "  ".join(overrides) + f"[/{MUTED}]\n")

    spinner_cols = [
        SpinnerColumn(style=ACCENT),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ]
    with Progress(*spinner_cols, console=console, transient=True) as prog:
        task = prog.add_task("planning…", total=None)
        result = run_agent(goal, model=model, dataset=dataset, method=method, executor=executor,
                           browser_path=browser_path)
        prog.update(task, description="done")

    _sep()

    status = result.get("status", "unknown")
    if status == "completed":
        _ok(f"[{SUCCESS}]job complete[/{SUCCESS}]")
        summary = result.get("summary", "")
        if summary:
            console.print(f"\n  [{MUTED}]{escape(summary)}[/{MUTED}]")
        metrics = result.get("metrics", {})
        if metrics:
            items = [(k, v) for k, v in metrics.items() if v is not None]
            if items:
                _metrics_row([(k, v) for k, v in items[:4]])
    else:
        _fail(f"job {status}")
        err = result.get("error", "")
        if err:
            console.print(f"\n  [{ERROR}]{escape(err)}[/{ERROR}]")

    console.print()


def cmd_interactive():
    _header()

    history_file = Path(
        __import__("config.settings", fromlist=["settings"]).settings.data_dir
    ) / "history.json"
    history: list[str] = json.loads(history_file.read_text()) if history_file.exists() else []

    console.print(f"  [{MUTED}]type a goal and press enter. [/{MUTED}][{ACCENT}]/help[/{ACCENT}][{MUTED}] for commands.[/{MUTED}]\n")

    while True:
        try:
            goal = Prompt.ask(f"[{ACCENT}]❯[/{ACCENT}]")
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{MUTED}]goodbye[/{MUTED}]")
            break

        if not goal.strip():
            continue

        if goal.lower() in ("exit", "quit", "q"):
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text(json.dumps(history[-200:], indent=2))
            console.print(f"\n  [{MUTED}]goodbye[/{MUTED}]")
            break

        if goal.startswith("/"):
            _handle_slash(goal, history)
            continue

        history.append(goal)
        cmd_run(goal)


def _handle_slash(cmd: str, history: list):
    parts = cmd.strip().split()
    verb = parts[0].lower()

    if verb == "/help":
        _sep()
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(style=ACCENT, width=20)
        table.add_column(style=MUTED)
        rows = [
            ("/run <goal>",          "run the agent with a goal"),
            ("/jobs",                "list recent fine-tuning jobs"),
            ("/job <id>",            "show job details"),
            ("/models",              "list saved model versions"),
            ("/convs",               "list conversations"),
            ("/status",              "show GPU and agent status"),
            ("/config",              "show current configuration"),
            ("/colab",               "print standalone Colab setup script"),
            ("/backup [path]",       "export all data to JSON"),
            ("/restore <path>",      "restore from JSON backup"),
            ("/clear",               "clear the screen"),
            ("/help",                "show this message"),
            ("exit / quit",         "exit interactive mode"),
        ]
        for r in rows:
            table.add_row(*r)
        console.print(table)
        console.print()

    elif verb == "/clear":
        console.clear()
        _header()

    elif verb == "/status":
        _sep()
        _status_display()

    elif verb == "/jobs":
        cmd_list_jobs()

    elif verb == "/models":
        cmd_list_models()

    elif verb == "/convs":
        cmd_list_convs()

    elif verb == "/config":
        cmd_config()

    elif verb == "/colab":
        cmd_colab_code()

    elif verb == "/job":
        if len(parts) < 2:
            _fail("usage: /job <id>")
        else:
            cmd_show_job(parts[1])

    elif verb == "/conv":
        if len(parts) < 2:
            _fail("usage: /conv <id>")
        else:
            cmd_show_conv(parts[1])

    elif verb == "/backup":
        path = parts[1] if len(parts) > 1 else None
        cmd_backup(path)

    elif verb == "/restore":
        if len(parts) < 2:
            _fail("usage: /restore <path>")
        else:
            cmd_restore(parts[1])

    elif verb == "/run":
        if len(parts) < 2:
            _fail("usage: /run <goal>")
        else:
            cmd_run(" ".join(parts[1:]))

    else:
        _fail(f"unknown command: {verb}  (try /help)")


def _status_display():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            used = torch.cuda.memory_allocated(0) / 1024**3
            _ok(f"GPU  [{SUCCESS}]{name}[/{SUCCESS}]  [{MUTED}]{total:.0f} GB total · {used:.1f} GB used[/{MUTED}]")
        else:
            _warn("no CUDA GPU detected — running on CPU")
    except ImportError:
        _warn("torch not installed — cannot query GPU")

    _ok(f"agent  [{SUCCESS}]ready[/{SUCCESS}]")

    try:
        from config.settings import settings
        db = settings.get_db_url()
        _ok(f"db  [{INFO}]{db[:60]}[/{INFO}]")
    except Exception:
        _warn("could not read db config")
    console.print()


def cmd_list_jobs():
    store = JobStore()
    jobs = store.list()
    if not jobs:
        console.print(f"  [{MUTED}]No jobs found[/{MUTED}]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style=MUTED, border_style="grey23")
    table.add_column("id", style=INFO, width=10, no_wrap=True)
    table.add_column("goal", style="white", max_width=44)
    table.add_column("method", style=ACCENT, width=8)
    table.add_column("status", width=10)
    table.add_column("created", style=MUTED, width=16, no_wrap=True)

    status_color = {"completed": SUCCESS, "failed": ERROR, "running": WARN, "pending": MUTED}

    for j in jobs[:25]:
        sc = status_color.get(j.status, MUTED)
        table.add_row(
            j.id[:8],
            j.goal[:44],
            j.method or "—",
            f"[{sc}]{j.status}[/{sc}]",
            j.created_at.strftime("%Y-%m-%d %H:%M") if j.created_at else "—",
        )
    console.print(table)


def cmd_list_models():
    store = ModelVersionStore()
    models = store.list_all()
    if not models:
        console.print(f"  [{MUTED}]no model versions found[/{MUTED}]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style=MUTED, border_style="grey23")
    table.add_column("id", style=INFO, width=10, no_wrap=True)
    table.add_column("base model", style="white", max_width=36)
    table.add_column("method", style=ACCENT, width=8)
    table.add_column("loss", style=ACCENT, width=8)
    table.add_column("created", style=MUTED, width=16, no_wrap=True)

    for m in models[:25]:
        loss = f"{m.final_loss:.4f}" if m.final_loss else "—"
        table.add_row(
            m.id[:8],
            m.base_model[:36],
            m.method or "—",
            loss,
            m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else "—",
        )
    console.print(table)


def cmd_list_convs():
    store = ConversationStore()
    convs = store.list_all()
    if not convs:
        console.print(f"  [{MUTED}]no conversations found[/{MUTED}]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style=MUTED, border_style="grey23")
    table.add_column("id", style=INFO, width=10, no_wrap=True)
    table.add_column("goal", style="white", max_width=44)
    table.add_column("status", width=10)
    table.add_column("msgs", style=MUTED, width=6)
    table.add_column("updated", style=MUTED, width=16, no_wrap=True)

    status_color = {"active": SUCCESS, "completed": INFO, "failed": ERROR}
    for c in convs[:25]:
        sc = status_color.get(c.status, MUTED)
        table.add_row(
            c.id[:8],
            c.goal[:44],
            f"[{sc}]{c.status}[/{sc}]",
            str(len(c.get_messages())),
            c.updated_at.strftime("%Y-%m-%d %H:%M") if c.updated_at else "—",
        )
    console.print(table)


def cmd_show_job(job_id: str):
    store = JobStore()
    job = store.get(job_id)
    if not job:
        _fail(f"job not found: {job_id}")
        return

    _sep()
    console.print(f"  [white bold]{job.id}[/white bold]  [{MUTED}]{job.goal[:60]}[/{MUTED}]\n")
    _kv("status", job.status)
    _kv("method", job.method or "—")
    _kv("base model", job.base_model or "—")
    _kv("dataset", job.dataset or "—")
    _kv("runtime", job.runtime or "—")
    created_str = job.created_at
    if hasattr(created_str, 'strftime'):
        created_str = created_str.strftime("%Y-%m-%d %H:%M")
    elif isinstance(created_str, str):
        pass  # already a string
    else:
        created_str = "—"
    _kv("created", created_str)

    if job.metrics:
        m = job.get_metrics()
        items = [(k, v) for k, v in m.items() if v is not None]
        if items:
            console.print()
            _metrics_row([(k, v) for k, v in items[:4]])

    if job.error:
        console.print()
        _fail(job.error)
    console.print()


def cmd_show_conv(conv_id: str):
    store = ConversationStore()
    conv = store.get(conv_id)
    if not conv:
        _fail(f"conversation not found: {conv_id}")
        return

    _sep()
    console.print(f"  [white bold]{conv.id}[/white bold]  [{MUTED}]{conv.goal[:60]}[/{MUTED}]\n")
    role_style = {"assistant": INFO, "user": ACCENT, "system": MUTED}

    for msg in conv.get_messages()[-12:]:
        rs = role_style.get(msg["role"], MUTED)
        console.print(f"  [{rs}]{msg['role']}[/{rs}]  [{MUTED}]{escape(msg['content'][:280])}[/{MUTED}]")
    console.print()


def cmd_config():
    _sep()
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style=MUTED, width=28)
    table.add_column(style=INFO)
    for key in sorted(dir(settings)):
        if key.startswith("_") or callable(getattr(settings, key)):
            continue
        val = getattr(settings, key)
        if isinstance(val, str) and len(val) > 60:
            val = val[:60] + "…"
        table.add_row(key, str(val))
    console.print(table)
    console.print()


def cmd_colab_code():
    code = '''# ===== Konfuntex — Standalone Colab Setup =====
!pip install -q transformers datasets accelerate peft trl bitsandbytes huggingface_hub torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

MODEL_NAME   = "microsoft/phi-2"
DATASET_NAME = "databricks/databricks-dolly-15k"
METHOD       = "qlora"
OUTPUT_DIR   = "./finetuned_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
) if METHOD == "qlora" else None

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","v_proj","k_proj","o_proj"], task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

dataset = load_dataset(DATASET_NAME, split="train").select(range(1000))
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="instruction",
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=8,
        fp16=True, logging_steps=25, save_steps=500,
        gradient_checkpointing=True, report_to="none",
    ),
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved → {OUTPUT_DIR}")
# TRAINING complete
'''
    _code_block(code)
    return code


def cmd_backup(path: str = None):
    backup_path = path or os.path.join(
        settings.data_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)

    jobs = JobStore().list(limit=99999)
    convs = ConversationStore().list_all(limit=99999)
    models = ModelVersionStore().list_all(limit=99999)

    backup = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "jobs": [
                {"id": j.id, "goal": j.goal, "status": j.status, "method": j.method,
                 "base_model": j.base_model, "dataset": j.dataset, "error": j.error,
                 "metrics_json": j.metrics,
                 "created_at": j.created_at.isoformat() if j.created_at else None}
                for j in jobs
            ],
            "conversations": [
                {"id": c.id, "goal": c.goal, "status": c.status,
                 "messages_json": c.messages_json,
                 "created_at": c.created_at.isoformat() if c.created_at else None}
                for c in convs
            ],
            "models": [
                {"id": m.id, "base_model": m.base_model, "method": m.method,
                 "final_loss": m.final_loss, "finetuned_path": m.finetuned_path,
                 "created_at": m.created_at.isoformat() if m.created_at else None}
                for m in models
            ],
        },
    }

    with open(backup_path, "w") as f:
        json.dump(backup, f, indent=2, default=str)

    _ok(f"backup written  [{INFO}]{backup_path}[/{INFO}]")
    _kv("jobs", len(jobs))
    _kv("conversations", len(convs))
    _kv("models", len(models))
    console.print()


def cmd_restore(path: str):
    if not os.path.exists(path):
        _fail(f"file not found: {path}")
        return
    with open(path) as f:
        backup = json.load(f)
    if backup.get("version") != 1:
        _fail(f"unsupported backup version: {backup.get('version')}")
        return
    _warn(f"restoring from {path}…")
    console.print(f"  [{MUTED}]created: {backup.get('created_at', 'unknown')}[/{MUTED}]")
    
    data = backup.get("data", {})
    jobs_data = data.get("jobs", [])
    convs_data = data.get("conversations", [])
    models_data = data.get("models", [])
    
    # Restore jobs
    for jd in jobs_data:
        job = Job(
            id=jd["id"],
            tenant_id=jd.get("tenant_id"),
            goal=jd["goal"],
            status=jd["status"],
            method=jd.get("method"),
            base_model=jd.get("base_model"),
            dataset=jd.get("dataset"),
            runtime=jd.get("runtime"),
            conversation_id=jd.get("conversation_id"),
            error=jd.get("error"),
        )
        if jd.get("metrics_json"):
            job.set_metrics(json.loads(jd["metrics_json"]) if isinstance(jd["metrics_json"], str) else jd["metrics_json"])
        if jd.get("created_at"):
            job.created_at = datetime.fromisoformat(jd["created_at"])
        if jd.get("finished_at"):
            job.finished_at = datetime.fromisoformat(jd["finished_at"])
        session = get_session()
        session.merge(job)
        session.commit()
    
    # Restore conversations
    for cd in convs_data:
        conv = Conversation(
            id=cd["id"],
            tenant_id=cd.get("tenant_id"),
            goal=cd["goal"],
            status=cd["status"],
        )
        if cd.get("messages_json"):
            conv.set_messages(json.loads(cd["messages_json"]) if isinstance(cd["messages_json"], str) else cd["messages_json"])
        if cd.get("summary"):
            conv.summary = cd["summary"]
        if cd.get("created_at"):
            conv.created_at = datetime.fromisoformat(cd["created_at"])
        if cd.get("updated_at"):
            conv.updated_at = datetime.fromisoformat(cd["updated_at"])
        session = get_session()
        session.merge(conv)
        session.commit()
    
    # Restore models
    for md in models_data:
        model = ModelVersion(
            id=md["id"],
            tenant_id=md.get("tenant_id"),
            job_id=md.get("job_id"),
            base_model=md["base_model"],
            finetuned_path=md.get("finetuned_path"),
            hf_repo_id=md.get("hf_repo_id"),
            method=md.get("method"),
            runtime_used=md.get("runtime_used"),
            training_steps=md.get("training_steps"),
            final_loss=md.get("final_loss"),
        )
        if md.get("created_at"):
            model.created_at = datetime.fromisoformat(md["created_at"])
        if md.get("metrics"):
            model.set_metrics(json.loads(md["metrics"]) if isinstance(md["metrics"], str) else md["metrics"])
        session = get_session()
        session.merge(model)
        session.commit()
    
    _ok("restore complete")
    console.print()


def cmd_serve(host: str = "0.0.0.0", port: int = 8080):
    os.environ["PORT"] = str(port)
    sys.argv.append("--fastapi")
    from agent.service import serve
    _ok(f"starting service on [{INFO}]{host}:{port}[/{INFO}]")
    serve()


def cmd_migrate(revision: str = "head"):
    """Apply database migrations."""
    _ok(f"Running migrations to [{ACCENT}]{revision}[/{ACCENT}]")
    migrate(revision)
    _ok("Migrations complete")


def cmd_colab_automate():
    """Drive Colab browser automation via Playwright."""
    from colab.automation import ColabAutomation
    import json

    console.print(f"[{HEADER}]Colab Automation[/{HEADER}]")
    url = Prompt.ask(f"[{ACCENT}]Notebook URL[/{ACCENT}]",
                     default="https://colab.research.google.com/#create=true")
    headless = Confirm.ask(f"[{ACCENT}]Headless mode?[/{ACCENT}]", default=True)

    with ColabAutomation(headless=headless) as auto:
        if not auto.is_available():
            console.print(f"[{WARN}]Playwright not installed. Install: pip install playwright && playwright install chromium[/{WARN}]")
            return

        result = auto.open_notebook(url)
        console.print(json.dumps(result, indent=2))

        if result.get("success"):
            auto.wait_for_connection(timeout=60)
            runtime = auto.detect_runtime()
            console.print(f"[green]Runtime: {runtime}[/green]")

            target = Prompt.ask(f"[{ACCENT}]Switch to GPU[/{ACCENT}]",
                                choices=["None", "T4", "V100", "A100", "A100-80GB", "skip"], default="skip")
            if target != "skip":
                sr = auto.switch_runtime(target)
                console.print(json.dumps(sr, indent=2))


def cmd_colab_sync():
    """Start Drive sync daemon."""
    from colab.drive_sync import DriveSyncDaemon
    import json
    import time

    console.print(f"[{HEADER}]Drive Sync Daemon[/{HEADER}]")
    job_id = Prompt.ask(f"[{ACCENT}]Job ID[/{ACCENT}]", default="default")
    interval = int(Prompt.ask(f"[{ACCENT}]Sync interval (seconds)[/{ACCENT}]", default="60"))

    d = DriveSyncDaemon(job_id=job_id)
    d.start(interval=interval)
    console.print(f"[green]Sync daemon started (job={job_id}, interval={interval}s)[/green]")

    try:
        while True:
            time.sleep(interval)
            status = d.status
            console.print(f"  Sync #{status['sync_count']}: last={status['last_sync']}, versions={len(d.list_versions())}")
    except KeyboardInterrupt:
        console.print(f"\n[{WARN}]Stopping...[/{WARN}]")
    finally:
        d.stop()
        console.print("[green]Sync daemon stopped[/green]")


def cmd_colab_resume():
    """Detect and generate resume code from Drive checkpoints."""
    from colab.resumer import ColabResumer
    import json

    console.print(f"[{HEADER}]Colab Resume Detector[/{HEADER}]")
    drive_dir = Prompt.ask(f"[{ACCENT}]Drive directory[/{ACCENT}]",
                           default="/content/drive/MyDrive/colab-agent")

    r = ColabResumer(drive_dir=drive_dir)
    state = r.detect_previous_run()

    if state.get("has_checkpoint"):
        console.print(f"[green]Checkpoint found![/green]")
        console.print(f"  Job: {state.get('job_id')}")
        console.print(f"  Version: v{state.get('checkpoint_version')}")
        console.print(f"  Model: {state.get('model_name')}")
        console.print(f"  Dataset: {state.get('dataset_name')}")
        console.print(f"  Epochs completed: {state.get('epochs_completed')}")
        console.print(f"  Last loss: {state.get('last_loss')}")

        if Confirm.ask(f"[{ACCENT}]Generate resume code?[/{ACCENT}]", default=True):
            code = r.build_resume_code(state)
            print("\n" + code)
    else:
        console.print(f"[{WARN}]No checkpoint found: {state.get('error', 'unknown')}[/{WARN}]")


def cmd_colab_remote(extra: list = None):
    """Execute code in Colab via Playwright automation (no manual steps)."""
    from colab.remote_executor import RemoteColabExecutor
    import os

    # Parse extra args for --browser-path
    browser_path = None
    if extra:
        it = iter(extra)
        for arg in it:
            if arg == "--browser-path" or arg == "-b":
                browser_path = next(it, None)
            elif arg == "--login":
                # One-time login mode
                from colab.remote_executor import RemoteColabExecutor
                r = RemoteColabExecutor(browser_path=browser_path)
                print(f"Using browser: {browser_path or 'default Chromium'}")
                r.login_once()
                return

    console.print(f"[{HEADER}]Remote Colab Executor[/{HEADER}]")
    if not RemoteColabExecutor._check_playwright():
        console.print(f"[{WARN}]Playwright not installed. Install: pip install playwright && playwright install chromium[/{WARN}]")
        return

    headless = not Confirm.ask(f"[{ACCENT}]Show browser window?[/{ACCENT}]", default=True)

    with RemoteColabExecutor(headless=headless, browser_path=browser_path) as executor:
        console.print("[green]Connecting to Colab...[/green]")
        conn = executor.connect(timeout=60)
        if not conn.get("success"):
            console.print(f"[{ERR}]Connection failed: {conn.get('error')}[/{ERR}]")
            return

        console.print(f"[green]Connected! Runtime ready: {conn.get('runtime_ready')}[/green]")

        while True:
            code = Prompt.ask(f"[{ACCENT}]Code to execute (or 'quit')[/{ACCENT}]")
            if code.lower() in ("quit", "exit", "q"):
                break

            console.print("[yellow]Executing...[/yellow]")
            result = executor.execute(code, timeout=120)
            if result.get("success"):
                console.print(f"[green]Output:[/green]")
                print(result.get("output", "")[:2000])
            else:
                console.print(f"[{ERR}]Error: {result.get('error')}[/{ERR}]")
                if result.get("output"):
                    print(result["output"][:500])

        console.print("[green]Disconnected.[/green]")


def cmd_colab_enterprise():
    """Manage Colab Enterprise runtimes."""
    from colab.enterprise import ColabEnterprise
    import json

    console.print(f"[{HEADER}]Colab Enterprise[/{HEADER}]")
    project = Prompt.ask(f"[{ACCENT}]Google Cloud project[/{ACCENT}]",
                         default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))

    e = ColabEnterprise(project=project)
    if not e.is_available():
        console.print(f"[{WARN}]SDK not available. Install: pip install google-cloud-aiplatform[/{WARN}]")
        return

    while True:
        action = Prompt.ask(
            f"[{ACCENT}]Action[/{ACCENT}]",
            choices=["list", "create", "delete", "setup-code", "quit"],
            default="list",
        )
        if action == "quit":
            break
        elif action == "list":
            runtimes = e.list_runtimes()
            console.print(json.dumps(runtimes, indent=2))
        elif action == "create":
            spec = Prompt.ask(f"[{ACCENT}]Runtime spec[/{ACCENT}]",
                              choices=["T4", "V100", "A100", "A100-80GB", "TPU"], default="T4")
            result = e.create_runtime(spec)
            console.print(json.dumps(result, indent=2))
        elif action == "delete":
            name = Prompt.ask(f"[{ACCENT}]Runtime name[/{ACCENT}]")
            result = e.delete_runtime(name)
            console.print(json.dumps(result, indent=2))
        elif action == "setup-code":
            print(e.generate_setup_code())

    console.print("[green]Done[/green]")


def cmd_detect():
    """Detect hardware and recommend best model."""
    from models.selector import ModelSelector, print_model_summary
    sel = ModelSelector()
    print_model_summary(sel)


def cmd_list_models_spec():
    """List all supported models with hardware requirements."""
    from models.selector import MODEL_REGISTRY
    import json

    console.print(f"[{HEADER}]Supported Models ({len(MODEL_REGISTRY)} total)[/{HEADER}]")

    by_tier = {}
    for spec in MODEL_REGISTRY:
        by_tier.setdefault(spec.tier.value, []).append(spec)

    tier_order = ["none", "low", "medium", "high", "very_high", "extreme"]
    for tier_val in tier_order:
        specs = by_tier.get(tier_val, [])
        if not specs:
            continue
        console.print(f"\n[bold]{tier_val.upper().replace('_', ' ')} tier[/bold]")
        for s in specs:
            needs = []
            if s.requires_auth:
                needs.append("HF auth")
            print(f"  {s.name}")
            print(f"    {s.params_b:.1f}B params | "
                  f"LoRA {s.vram_gb_lora:.0f}GB | "
                  f"QLoRA {s.vram_gb_qlora:.0f}GB | "
                  f"Full {s.vram_gb_full:.0f}GB VRAM | "
                  f"{s.context_window} ctx")
            if needs:
                print(f"    Needs: {', '.join(needs)}")


def cmd_backup_sqlite(path: str = None):
    """Backup SQLite database file."""
    from config.settings import settings
    db_path = settings.get_db_url().replace("sqlite:///", "")
    if not os.path.exists(db_path):
        _fail(f"Database file not found: {db_path}")
        return
    backup_path = path or f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import shutil
    shutil.copy2(db_path, backup_path)
    _ok(f"SQLite backup written to [{INFO}]{backup_path}[/{INFO}]")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="konfuntex — autonomous local fine-tuning agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands:
  init                  initialise database
  run <goal>            run the agent
  interactive           interactive mode  (default)
  jobs                  list jobs
  job <id>              job details
  models                list saved models
  convs                 list conversations
  config                show configuration
  colab                 print Colab setup script
  detect                detect hardware + auto-select best model
  list-models           list all supported models with VRAM requirements
  serve [host] [port]   start HTTP service
  backup [path]         export data to JSON
  restore <path>        restore from JSON
""",
    )
    parser.add_argument("command", nargs="?", default="interactive")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--model", help="base model id")
    parser.add_argument("--dataset", help="dataset id")
    parser.add_argument("--method", choices=["lora", "qlora", "full"])
    parser.add_argument("--executor", choices=["local", "colab", "auto", "remote"], default="auto",
                        help="Execution environment: local (persistent kernel), colab (upload code), remote (Playwright auto), auto (detect)")
    parser.add_argument("--browser-path", "-b", help="Path to browser executable (e.g. Brave) for remote executor")

    args = parser.parse_args()
    cmd = args.command
    extra = args.args

    dispatch = {
        "init":         lambda: cmd_init(),
        "run":          lambda: cmd_run(
                            " ".join(extra) if extra else Prompt.ask(f"[{ACCENT}]goal[/{ACCENT}]"),
                            model=args.model, dataset=args.dataset, method=args.method, executor=args.executor,
                            browser_path=args.browser_path),
        "interactive":  lambda: cmd_interactive(),
        "jobs":         lambda: cmd_list_jobs(),
        "job":          lambda: cmd_show_job(extra[0]) if extra else _fail("need job id"),
        "models":       lambda: cmd_list_models(),
        "convs":        lambda: cmd_list_convs(),
        "conv":         lambda: cmd_show_conv(extra[0]) if extra else _fail("need conv id"),
        "config":       lambda: cmd_config(),
        "colab":        lambda: cmd_colab_code(),
        "colab-automate":  lambda: cmd_colab_automate(),
        "colab-sync":      lambda: cmd_colab_sync(),
        "colab-resume":    lambda: cmd_colab_resume(),
        "colab-enterprise": lambda: cmd_colab_enterprise(),
        "colab-remote":      lambda: cmd_colab_remote(extra),
        "detect":       lambda: cmd_detect(),
        "list-models":  lambda: cmd_list_models_spec(),
        "backup":       lambda: cmd_backup(" ".join(extra) if extra else None),
        "restore":      lambda: cmd_restore(extra[0]) if extra else _fail("usage: restore <path>"),
        "serve":        lambda: cmd_serve(
                            extra[0] if extra else "0.0.0.0",
                            int(extra[1]) if len(extra) > 1 else 8080),
        "migrate":      lambda: cmd_migrate(extra[0] if extra else "head"),
        "backup-sqlite": lambda: cmd_backup_sqlite(" ".join(extra) if extra else None),
    }

    handler = dispatch.get(cmd)
    if handler:
        handler()
    else:
        _fail(f"Unknown command: {cmd}")
        console.print(f"  [{MUTED}]try: konfuntex --help[/{MUTED}]")


if __name__ == "__main__":
    main()
