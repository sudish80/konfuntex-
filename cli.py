#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from storage.database import init_db
from storage.jobs import JobStore
from storage.conversations import ConversationStore
from storage.models_store import ModelVersionStore
from agent.core import run_agent
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt


console = Console()


def cmd_init():
    """Initialize the database and data directories."""
    os.makedirs(settings.data_dir, exist_ok=True)
    init_db()
    console.print("[green]OK - Database initialized[/green]")


def cmd_run(goal: str, model: str = None, dataset: str = None, method: str = None):
    """Run the agent with a given goal."""
    console.print(Panel(f"[bold cyan]Goal:[/bold cyan] {goal}", title="Colab Agent"))
    if model:
        console.print(f"  Model: [green]{model}[/green]")
    if dataset:
        console.print(f"  Dataset: [green]{dataset}[/green]")
    if method:
        console.print(f"  Method: [green]{method}[/green]")
    result = run_agent(goal, model=model, dataset=dataset, method=method)
    console.print("\n[bold green]=== Result ===[/bold green]")
    console.print(json.dumps(result, indent=2, default=str))


def cmd_interactive():
    """Run in interactive mode."""
    console.print("[bold cyan]Colab Agent - Interactive Mode[/bold cyan]")
    console.print("Type your goals. Type 'exit' to quit.\n")

    history_file = Path(settings.data_dir) / "history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else []

    while True:
        goal = Prompt.ask("\n[bold yellow]What do you want to do?[/bold yellow]")

        if goal.lower() in ("exit", "quit", "q"):
            history_file.write_text(json.dumps(history, indent=2))
            console.print("[green]Goodbye![/green]")
            break

        if not goal.strip():
            continue

        history.append({"role": "user", "content": goal, "timestamp": __import__("datetime").datetime.now().isoformat()})
        result = run_agent(goal)
        history.append({"role": "agent", "content": result.get("summary", str(result)), "timestamp": __import__("datetime").datetime.now().isoformat()})

        console.print(f"\n[bold green]✓[/bold green] {result.get('summary', 'Done')}")


def cmd_list_jobs():
    """List all fine-tuning jobs."""
    store = JobStore()
    jobs = store.list()
    if not jobs:
        console.print("[yellow]No jobs found[/yellow]")
        return

    table = Table(title="Fine-tuning Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Goal", style="white")
    table.add_column("Method", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Created", style="blue")

    for j in jobs[:20]:
        table.add_row(
            j.id[:8],
            j.goal[:50],
            j.method or "-",
            j.status,
            j.created_at.strftime("%Y-%m-%d %H:%M") if j.created_at else "-",
        )
    console.print(table)


def cmd_list_convs():
    """List all conversations."""
    store = ConversationStore()
    convs = store.list_all()
    if not convs:
        console.print("[yellow]No conversations found[/yellow]")
        return

    table = Table(title="Conversations")
    table.add_column("ID", style="cyan")
    table.add_column("Goal", style="white")
    table.add_column("Status", style="green")
    table.add_column("Messages", style="blue")
    table.add_column("Updated", style="yellow")

    for c in convs[:20]:
        msgs = c.get_messages()
        table.add_row(
            c.id[:8],
            c.goal[:50],
            c.status,
            str(len(msgs)),
            c.updated_at.strftime("%Y-%m-%d %H:%M") if c.updated_at else "-",
        )
    console.print(table)


def cmd_show_job(job_id: str):
    """Show details of a specific job."""
    store = JobStore()
    job = store.get(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        return

    console.print(Panel(f"[bold]Job: {job.id}[/bold]"))
    console.print(f"Goal: {job.goal}")
    console.print(f"Status: {job.status}")
    console.print(f"Method: {job.method}")
    console.print(f"Base Model: {job.base_model}")
    console.print(f"Dataset: {job.dataset}")
    console.print(f"Runtime: {job.runtime}")
    console.print(f"Created: {job.created_at}")
    console.print(f"Updated: {job.updated_at}")

    if job.metrics:
        console.print("\n[bold]Metrics:[/bold]")
        console.print(json.dumps(job.get_metrics(), indent=2))

    if job.error:
        console.print(f"\n[bold red]Error:[/bold red] {job.error}")


def cmd_show_conv(conv_id: str):
    """Show details of a specific conversation."""
    store = ConversationStore()
    conv = store.get(conv_id)
    if not conv:
        console.print(f"[red]Conversation not found: {conv_id}[/red]")
        return

    console.print(Panel(f"[bold]Conversation: {conv.id}[/bold]"))
    console.print(f"Goal: {conv.goal}")
    console.print(f"Status: {conv.status}")
    console.print(f"Messages: {len(conv.get_messages())}")

    for msg in conv.get_messages()[-10:]:
        role_color = "cyan" if msg["role"] == "assistant" else "green" if msg["role"] == "user" else "yellow"
        content = msg["content"][:500]
        console.print(f"\n[bold {role_color}]{msg['role']}:[/bold {role_color}]")
        console.print(content)


def cmd_list_models():
    """List all model versions."""
    store = ModelVersionStore()
    models = store.list_all()
    if not models:
        console.print("[yellow]No model versions found[/yellow]")
        return

    table = Table(title="Model Versions")
    table.add_column("ID", style="cyan")
    table.add_column("Base Model", style="white")
    table.add_column("Method", style="magenta")
    table.add_column("Runtime", style="blue")
    table.add_column("Created", style="yellow")

    for m in models[:20]:
        table.add_row(
            m.id[:8],
            m.base_model[:30],
            m.method or "-",
            m.runtime_used or "-",
            m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else "-",
        )
    console.print(table)


def cmd_config():
    """Show current configuration."""
    console.print("[bold]Current Configuration:[/bold]")
    for key in dir(settings):
        if not key.startswith("_") and not callable(getattr(settings, key)):
            val = getattr(settings, key)
            if isinstance(val, str) and len(val) > 50:
                val = val[:50] + "..."
            console.print(f"  {key}: {val}")


def cmd_colab_code():
    """Generate standalone Colab notebook code to run the agent."""
    code = '''# ===== Colab Agent Standalone Setup =====
# Run this in Google Colab to set up the fine-tuning environment

# Step 1: Install dependencies
!pip install -q transformers datasets accelerate peft trl bitsandbytes huggingface_hub torch

# Step 2: Login to HuggingFace (optional - for gated models)
from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")  # Uncomment and add your token

# Step 3: Set up your fine-tuning
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ===== CONFIGURATION =====
MODEL_NAME = "microsoft/phi-2"        # Base model
DATASET_NAME = "databricks/databricks-dolly-15k"  # Dataset
METHOD = "qlora"                      # lora, qlora, or full
OUTPUT_DIR = "./finetuned_model"
HF_TOKEN = None                       # Set if using gated models

# ===== LOAD MODEL =====
print(f"Loading model: {MODEL_NAME}")

bnb_config = None
if METHOD == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
print(f"Model loaded! Params: {model.num_parameters() / 1e9:.2f}B")

# ===== PEFT SETUP =====
if METHOD in ("lora", "qlora"):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# ===== LOAD DATASET =====
dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset loaded: {len(dataset)} samples")

# ===== TRAINING =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=25,
    save_steps=500,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset.select(range(min(1000, len(dataset)))),
)

print("Starting training...")
trainer.train()
print("Training complete!")

# ===== SAVE MODEL =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ===== (Optional) Save to Google Drive =====
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r {OUTPUT_DIR} /content/drive/MyDrive/
'''
    console.print(Syntax(code, "python", theme="monokai"))
    return code


def cmd_backup(path: str = None):
    """Export all data to a portable JSON backup file."""
    import json
    from datetime import datetime, timezone

    backup_path = path or os.path.join(settings.data_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)

    from storage.jobs import JobStore
    from storage.conversations import ConversationStore
    from storage.models_store import ModelVersionStore
    from storage.metrics_store import MetricsStore

    jobs = JobStore()
    convs = ConversationStore()
    models = ModelVersionStore()
    metrics = MetricsStore()

    backup = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_url": settings.get_db_url(),
        "data": {
            "jobs": [
                {
                    "id": j.id, "goal": j.goal, "status": j.status,
                    "method": j.method, "base_model": j.base_model,
                    "dataset": j.dataset, "runtime": j.runtime,
                    "error": j.error, "metrics_json": j.metrics,
                    "metadata_json": j.metadata_json,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "updated_at": j.updated_at.isoformat() if j.updated_at else None,
                    "finished_at": j.finished_at.isoformat() if j.finished_at else None,
                    "conversation_id": j.conversation_id,
                    "model_version_id": j.model_version_id,
                }
                for j in jobs.list(limit=99999)
            ],
            "conversations": [
                {
                    "id": c.id, "goal": c.goal, "status": c.status,
                    "messages_json": c.messages_json,
                    "summary": c.summary,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in convs.list_all(limit=99999)
            ],
            "models": [
                {
                    "id": m.id, "job_id": m.job_id, "base_model": m.base_model,
                    "finetuned_path": m.finetuned_path, "hf_repo_id": m.hf_repo_id,
                    "method": m.method, "metrics": m.metrics,
                    "runtime_used": m.runtime_used, "training_steps": m.training_steps,
                    "final_loss": m.final_loss,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "tags": m.tags, "metadata_json": m.metadata_json,
                }
                for m in models.list_all(limit=99999)
            ],
        },
    }

    # Add metrics (query per job)
    metrics_list = []
    for j in backup["data"]["jobs"]:
        for rec in metrics.get_job_metrics(j["id"]):
            metrics_list.append({
                "id": rec.id, "job_id": rec.job_id,
                "epoch": rec.epoch, "global_step": rec.global_step,
                "loss": rec.loss, "accuracy": rec.accuracy,
                "gpu_mem_gb": rec.gpu_mem_gb,
                "tokens_per_second": rec.tokens_per_second,
                "learning_rate": rec.learning_rate,
                "grad_norm": rec.grad_norm,
                "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
                "extras_json": rec.extras_json,
            })
    backup["data"]["metrics"] = metrics_list

    with open(backup_path, "w") as f:
        json.dump(backup, f, indent=2, default=str)

    stats = {k: len(v) for k, v in backup["data"].items()}
    console.print(f"[green]Backup written to {backup_path}[/green]")
    console.print(f"  Jobs: {stats['jobs']}, Conversations: {stats['conversations']}, "
                  f"Models: {stats['models']}, Metrics: {stats['metrics']}")


def cmd_restore(path: str):
    """Restore data from a JSON backup file."""
    import json

    if not os.path.exists(path):
        console.print(f"[red]Backup file not found: {path}[/red]")
        return

    with open(path) as f:
        backup = json.load(f)

    version = backup.get("version", 0)
    if version != 1:
        console.print(f"[red]Unsupported backup version: {version}[/red]")
        return

    console.print(f"[yellow]Restoring from {path}...[/yellow]")
    console.print(f"  Backup created: {backup.get('created_at', 'unknown')}")

    from storage.jobs import JobStore
    from storage.conversations import ConversationStore
    from storage.models_store import ModelVersionStore
    from storage.metrics_store import MetricsStore

    jobs_store = JobStore()
    convs_store = ConversationStore()
    models_store = ModelVersionStore()
    metrics_store = MetricsStore()
    metrics_store.create_tables()

    data = backup.get("data", {})

    # Restore jobs first (metrics/models reference them)
    job_count = 0
    for jd in data.get("jobs", []):
        existing = jobs_store.get(jd["id"])
        if existing:
            continue
        from storage.database import Job
        job = Job(
            id=jd["id"], goal=jd["goal"], status=jd.get("status", "pending"),
            method=jd.get("method"), base_model=jd.get("base_model"),
            dataset=jd.get("dataset"), runtime=jd.get("runtime"),
            error=jd.get("error"), metrics=jd.get("metrics_json"),
            metadata_json=jd.get("metadata_json"),
            conversation_id=jd.get("conversation_id"),
            model_version_id=jd.get("model_version_id"),
            colab_notebook_id=jd.get("colab_notebook_id"),
            colab_notebook_url=jd.get("colab_notebook_url"),
        )
        jobs_store.session.add(job)
        job_count += 1
    jobs_store.session.commit()

    # Restore conversations
    conv_count = 0
    for cd in data.get("conversations", []):
        existing = convs_store.get(cd["id"])
        if existing:
            continue
        from storage.database import Conversation
        conv = Conversation(
            id=cd["id"], goal=cd["goal"], status=cd.get("status", "active"),
            messages_json=cd.get("messages_json", "[]"),
            summary=cd.get("summary"),
        )
        convs_store.session.add(conv)
        conv_count += 1
    convs_store.session.commit()

    # Restore model versions
    model_count = 0
    for md in data.get("models", []):
        existing = models_store.get(md["id"])
        if existing:
            continue
        from storage.database import ModelVersion
        mv = ModelVersion(
            id=md["id"], job_id=md.get("job_id"),
            base_model=md["base_model"],
            finetuned_path=md.get("finetuned_path"),
            hf_repo_id=md.get("hf_repo_id"), method=md.get("method"),
            metrics=md.get("metrics"), runtime_used=md.get("runtime_used"),
            training_steps=md.get("training_steps"),
            final_loss=md.get("final_loss"),
            tags=md.get("tags"), metadata_json=md.get("metadata_json"),
        )
        models_store.session.add(mv)
        model_count += 1
    models_store.session.commit()

    # Restore metrics
    metric_count = 0
    for rd in data.get("metrics", []):
        from storage.metrics_store import MetricRecord
        existing = metrics_store.session.query(MetricRecord).filter_by(id=rd["id"]).first()
        if existing:
            continue
        rec = MetricRecord(
            id=rd["id"], job_id=rd["job_id"],
            epoch=rd.get("epoch"), global_step=rd.get("global_step"),
            loss=rd.get("loss"), accuracy=rd.get("accuracy"),
            gpu_mem_gb=rd.get("gpu_mem_gb"),
            tokens_per_second=rd.get("tokens_per_second"),
            learning_rate=rd.get("learning_rate"),
            grad_norm=rd.get("grad_norm"),
            extras_json=rd.get("extras_json"),
        )
        metrics_store.session.add(rec)
        metric_count += 1
    metrics_store.session.commit()

    console.print(f"[green]Restored: {job_count} jobs, {conv_count} conversations, "
                  f"{model_count} models, {metric_count} metrics[/green]")


def cmd_backup_sqlite(path: str = None):
    """Fast SQLite binary backup via the sqlite3 CLI (SQLite only)."""
    import subprocess
    from datetime import datetime

    db_url = settings.get_db_url()
    if not db_url.startswith("sqlite"):
        console.print("[red]Backup-sqlite is only available for SQLite databases[/red]")
        return

    db_path = db_url.replace("sqlite:///", "")
    if not db_path:
        console.print("[red]Cannot determine database file path[/red]")
        return

    backup_path = path or os.path.join(settings.data_dir, f"sqlite_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
    os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)

    try:
        result = subprocess.run(
            ["sqlite3", db_path, f".backup '{backup_path}'"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            size = os.path.getsize(backup_path)
            console.print(f"[green]SQLite backup written to {backup_path} ({size / 1024:.1f} KB)[/green]")
        else:
            console.print(f"[red]sqlite3 backup failed: {result.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]sqlite3 CLI not found. Install SQLite or use 'backup' command instead.[/red]")
    except subprocess.TimeoutExpired:
        console.print("[red]sqlite3 backup timed out (60s)[/red]")


def cmd_migrate(revision: str = "head"):
    """Run database migrations via Alembic."""
    from alembic.config import Config as AlembicConfig
    from alembic import command

    ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alembic.ini")
    if not os.path.exists(ini_path):
        console.print("[red]alembic.ini not found. Run 'alembic init alembic' first.[/red]")
        return

    alembic_cfg = AlembicConfig(ini_path)
    try:
        command.upgrade(alembic_cfg, revision)
        console.print(f"[green]Migrations applied up to {revision}[/green]")
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        raise


def cmd_serve(host: str = "0.0.0.0", port: int = 8080):
    """Start the HTTP health/metrics service."""
    os.environ["PORT"] = str(port)
    import sys
    sys.argv.append("--fastapi")
    from agent.service import serve
    console.print(f"[green]Starting agent service on {host}:{port} (FastAPI)...[/green]")
    serve()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Colab Agent - Autonomous LLM-powered ML fine-tuning")
    parser.add_argument("command", nargs="?", default="interactive",
                        help="Command to run")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--model", help="Base model to fine-tune (e.g. distilbert-base-uncased)")
    parser.add_argument("--dataset", help="Dataset to use (e.g. imdb)")
    parser.add_argument("--method", choices=["lora", "qlora", "full"], help="Fine-tuning method")

    args = parser.parse_args()

    cmd = args.command
    extra = args.args

    commands = {
        "init": lambda: cmd_init(),
        "run": lambda: cmd_run(
            " ".join(extra) if extra else Prompt.ask("Goal"),
            model=args.model,
            dataset=args.dataset,
            method=args.method,
        ),
        "interactive": lambda: cmd_interactive(),
        "jobs": lambda: cmd_list_jobs(),
        "convs": lambda: cmd_list_convs(),
        "models": lambda: cmd_list_models(),
        "job": lambda: cmd_show_job(extra[0]) if extra else console.print("[red]Need job ID[/red]"),
        "conv": lambda: cmd_show_conv(extra[0]) if extra else console.print("[red]Need conversation ID[/red]"),
        "config": lambda: cmd_config(),
        "colab": lambda: cmd_colab_code(),
        "backup": lambda: cmd_backup(" ".join(extra) if extra else None),
        "restore": lambda: cmd_restore(extra[0]) if extra else console.print("[red]Usage: restore <backup.json>[/red]"),
        "backup-sqlite": lambda: cmd_backup_sqlite(" ".join(extra) if extra else None),
        "migrate": lambda: cmd_migrate(extra[0] if extra else "head"),
        "serve": lambda: cmd_serve(extra[0] if len(extra) > 0 else "0.0.0.0",
                                   int(extra[1]) if len(extra) > 1 else 8080),
    }

    handler = commands.get(cmd)
    if handler:
        handler()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print("Commands: init, run, interactive, jobs, convs, models, job <id>, conv <id>, config, colab, backup [path], restore <path>, backup-sqlite [path], migrate [revision], serve <host> <port>")
        console.print("Run flags: --model <name> --dataset <name> --method <lora|qlora|full>")


if __name__ == "__main__":
    main()
