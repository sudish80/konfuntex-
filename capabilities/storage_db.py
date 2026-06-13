"""
Phases 69-75 — Storage & Database.

Provides:
  - SQLiteJobStore           jobs table CRUD                (69)
  - MetricsTimeSeriesDB      time-series metrics storage    (70)
  - ModelRegistry            metadata registry per model    (71)
  - ArtifactCompressor       compress + upload artifacts    (72)
  - StorageCleanupPolicy     scheduled cleanup              (73)
  - ExportToHTML             HTML report generation         (74)
  - CrossSessionMemory       persist agent learnings        (75)
"""
import json
import os
from datetime import datetime


# ==================================================================== #
#  69 — SQLiteJobStore
# ==================================================================== #

class SQLiteJobStore:
    """
    SQLite-backed CRUD for job metadata.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        user_goal TEXT,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        status TEXT,
        model_name TEXT,
        dataset_name TEXT,
        fine_tune_method TEXT,
        runtime_used TEXT,
        error_message TEXT,
        github_repo_url TEXT,
        hf_model_url TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_name);
    CREATE INDEX IF NOT EXISTS idx_jobs_start ON jobs(start_time);
    """

    def __init__(self, db_path: str = "./jobs.db"):
        self.db_path = db_path

    def init_code(self) -> str:
        return f"""
import sqlite3, os

db_path = "{self.db_path}"
os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    user_goal TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT,
    model_name TEXT,
    dataset_name TEXT,
    fine_tune_method TEXT,
    runtime_used TEXT,
    error_message TEXT,
    github_repo_url TEXT,
    hf_model_url TEXT
)''')

cursor.execute('''CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_name)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_jobs_start ON jobs(start_time)''')

conn.commit()
print(f"Job store ready: {{db_path}}")
"""

    def crud_code(self) -> str:
        return """
# Create job
cursor.execute('''
    INSERT INTO jobs (job_id, user_goal, start_time, status)
    VALUES (?, ?, ?, 'running')
''', (job_id, user_goal, datetime.now().isoformat()))

# Update job on completion
cursor.execute('''
    UPDATE jobs SET status=?, end_time=?, model_name=?, runtime_used=?,
    error_message=?, github_repo_url=?, hf_model_url=?
    WHERE job_id=?
''', (status, datetime.now().isoformat(), model_name, runtime,
      error_message, gh_url, hf_url, job_id))

# Query recent successful jobs
cursor.execute('''
    SELECT * FROM jobs WHERE status='success'
    ORDER BY end_time DESC LIMIT 10
''')
rows = cursor.fetchall()

conn.commit()
conn.close()
"""


# ==================================================================== #
#  70 — MetricsTimeSeriesDB
# ==================================================================== #

class MetricsTimeSeriesDB:
    """
    Time-series metrics storage with batch insert and downsampling.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metrics_ts (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        step INTEGER,
        epoch INTEGER,
        train_loss REAL,
        val_loss REAL,
        accuracy REAL,
        gpu_memory_used_mb REAL,
        tokens_per_second REAL,
        learning_rate REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(job_id) REFERENCES jobs(job_id)
    );
    CREATE INDEX IF NOT EXISTS idx_metrics_job ON metrics_ts(job_id);
    """

    def __init__(self, db_path: str = "./metrics.db"):
        self.db_path = db_path

    def init_code(self) -> str:
        return f"""
import sqlite3, os

db_path = "{self.db_path}"
os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS metrics_ts (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step INTEGER,
    epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    accuracy REAL,
    gpu_memory_used_mb REAL,
    tokens_per_second REAL,
    learning_rate REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
)''')

cursor.execute('''CREATE INDEX IF NOT EXISTS idx_metrics_job ON metrics_ts(job_id)''')

conn.commit()
print(f"Metrics DB ready: {{db_path}}")
"""

    def insert_and_query_code(self) -> str:
        return '''
import sqlite3, json

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Batch insert every 10 steps
buffer = []
def log_metric(job_id, step, epoch, train_loss, val_loss, accuracy, gpu_mem, lr):
    buffer.append((job_id, step, epoch, train_loss, val_loss,
                   accuracy, gpu_mem, None, lr))
    if len(buffer) >= 10:
        cursor.executemany("""
            INSERT INTO metrics_ts
            (job_id, step, epoch, train_loss, val_loss, accuracy,
             gpu_memory_used_mb, tokens_per_second, learning_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, buffer)
        conn.commit()
        buffer.clear()

# Query for plotting
cursor.execute("""
    SELECT step, train_loss, val_loss
    FROM metrics_ts
    WHERE job_id=?
    ORDER BY step
""", (job_id,))
data = cursor.fetchall()

# Downsampling for long runs
if len(data) > 1000:
    step = 100
    data = [data[i] for i in range(0, len(data), step)]
    print(f"Downsampled to {len(data)} points")
'''


# ==================================================================== #
#  71 — ModelRegistry
# ==================================================================== #

class ModelRegistry:
    """
    Store/query/search model metadata.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id TEXT PRIMARY KEY,
        job_id TEXT,
        base_model_name TEXT,
        adapter_path TEXT,
        hf_repo_id TEXT,
        created_at TIMESTAMP,
        task_type TEXT,
        metrics_json TEXT,
        size_mb REAL,
        is_deleted BOOLEAN DEFAULT FALSE
    );
    """

    def __init__(self, db_path: str = "./model_registry.db"):
        self.db_path = db_path

    def registry_code(self) -> str:
        return f"""
import sqlite3, json, os
from datetime import datetime

db_path = "{self.db_path}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    job_id TEXT,
    base_model_name TEXT,
    adapter_path TEXT,
    hf_repo_id TEXT,
    created_at TIMESTAMP,
    task_type TEXT,
    metrics_json TEXT,
    size_mb REAL,
    is_deleted BOOLEAN DEFAULT FALSE
)''')

conn.commit()

def register_model(model_id, job_id, base_model, adapter_path,
                   hf_repo_id, task_type, metrics):
    cursor.execute('''
        INSERT OR REPLACE INTO model_registry
        (model_id, job_id, base_model_name, adapter_path, hf_repo_id,
         created_at, task_type, metrics_json, size_mb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_id, job_id, base_model, adapter_path, hf_repo_id,
        datetime.now().isoformat(), task_type,
        json.dumps(metrics),
        sum(f.stat().st_size for f in __import__("pathlib").Path(adapter_path).rglob("*")) / 1e6
        if os.path.exists(adapter_path) else 0,
    ))
    conn.commit()
    print(f"Registered model: {{model_id}}")

# Search by task type
cursor.execute('''
    SELECT * FROM model_registry
    WHERE task_type=? AND is_deleted=0
    ORDER BY json_extract(metrics_json, '$.accuracy') DESC
''', (task_type,))
results = cursor.fetchall()
print(f"Found {{len(results)}} models for task {{task_type}}")
"""


# ==================================================================== #
#  72 — ArtifactCompressor
# ==================================================================== #

class ArtifactCompressor:
    """
    Compress job artifacts into a zip file and upload to Drive.
    """

    def __init__(self, artifacts_dir: str = "./artifacts",
                 compression_level: int = 6):
        self.artifacts_dir = artifacts_dir
        self.compression_level = compression_level
        os.makedirs(artifacts_dir, exist_ok=True)

    def compress_code(self, job_id: str, paths_to_include: list[str]) -> str:
        return f"""
import zipfile, os, shutil

job_id = "{job_id}"
paths = {json.dumps(paths_to_include)}
artifacts_dir = "{self.artifacts_dir}"
comp_level = {self.compression_level}
os.makedirs(artifacts_dir, exist_ok=True)

zip_path = os.path.join(artifacts_dir, f"job_{{job_id}}_artifacts.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=comp_level) as zf:
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        zf.write(fp, arcname=os.path.relpath(fp, os.path.dirname(path)))
            else:
                zf.write(path, arcname=os.path.basename(path))

# Show compression stats
original_size = sum(os.path.getsize(p) if os.path.isfile(p)
                    else sum(f.stat().st_size for _, _, fs in os.walk(p) for f in fs
                             if os.path.isfile(os.path.join(_, f)))
                    for p in paths if os.path.exists(p))
compressed_size = os.path.getsize(zip_path)
print(f"Compressed: {{original_size/1e6:.1f}} MB -> {{compressed_size/1e6:.1f}} MB "
      f"({{(1-compressed_size/original_size)*100 if original_size else 0:.0f}}% reduction)")

# Delete originals
for path in paths:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"Deleted original: {{path}}")
"""


# ==================================================================== #
#  73 — StorageCleanupPolicy
# ==================================================================== #

class StorageCleanupPolicy:
    """
    Scheduled cleanup of old artifacts, logs, and checkpoints.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def cleanup_code(self, max_age_days: int = 30,
                     min_keep_jobs: int = 3) -> str:
        return f"""
import os, shutil, time, json
from datetime import datetime, timedelta
from pathlib import Path

max_age_days = {max_age_days}
min_keep_jobs = {min_keep_jobs}
dry_run = {str(self.dry_run).lower()}
cutoff = datetime.now() - timedelta(days=max_age_days)

LOG = []

# Clean artifacts
artifacts_dir = "./artifacts"
if os.path.exists(artifacts_dir):
    all_zips = sorted(Path(artifacts_dir).glob("*.zip"), key=os.path.getctime)
    keep = all_zips[-min_keep_jobs:] if len(all_zips) > min_keep_jobs else all_zips
    for f in all_zips:
        if f not in keep and datetime.fromtimestamp(os.path.getctime(f)) < cutoff:
            if not dry_run:
                os.remove(f)
            LOG.append(f"Deleted artifact: {{f.name}}")

# Clean temp files
for p in Path("/tmp").glob("agent_*"):
    if datetime.fromtimestamp(os.path.getctime(p)) < cutoff:
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True)
        LOG.append(f"Deleted temp: {{p.name}}")

# Clean old checkpoints (keep best only)
ckpt_dir = "./checkpoints"
if os.path.exists(ckpt_dir):
    for d in Path(ckpt_dir).iterdir():
        if d.is_dir() and "best" not in d.name and \
           datetime.fromtimestamp(os.path.getctime(d)) < cutoff:
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
            LOG.append(f"Deleted checkpoint: {{d.name}}")

print(f"Cleanup {{'(DRY RUN)' if dry_run else ''}}: {{len(LOG)}} items")
for l in LOG:
    print(f"  {{l}}")

with open("./cleanup.log", "a") as f:
    for l in LOG:
        f.write(f"{{datetime.now().isoformat()}}: {{l}}\\\\n")
"""


# ==================================================================== #
#  74 — ExportToHTML
# ==================================================================== #

class ExportToHTML:
    """
    Generate HTML reports using Jinja2 templates.
    """

    TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Job Report - {{ job_id }}</title>
<style>
body { font-family: Arial; max-width: 900px; margin: auto; padding: 20px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
img { max-width: 100%; }
</style>
</head>
<body>
<h1>Job Report: {{ job_id }}</h1>
<p><strong>Model:</strong> {{ model_name }}</p>
<p><strong>Dataset:</strong> {{ dataset }}</p>
<p><strong>Runtime:</strong> {{ runtime }}</p>
<p><strong>Status:</strong> {{ status }}</p>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{% for k, v in metrics.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>

<h2>Plots</h2>
<img src="metrics/loss_curve.png" alt="Loss Curve">

<h2>Links</h2>
<ul>
<li><a href="{{ hf_url }}">HF Hub</a></li>
<li><a href="{{ gh_url }}">GitHub</a></li>
</ul>
</body>
</html>
"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_code(self, job_id: str, model_name: str,
                    dataset: str, runtime: str, status: str,
                    metrics: dict, hf_url: str = "", gh_url: str = "") -> str:
        return f"""
import os, json
from jinja2 import Template

job_id = "{job_id}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

template_str = '''{self.TEMPLATE}'''
template = Template(template_str)

html = template.render(
    job_id=job_id,
    model_name="{model_name}",
    dataset="{dataset}",
    runtime="{runtime}",
    status="{status}",
    metrics={json.dumps(metrics)},
    hf_url="{hf_url}",
    gh_url="{gh_url}",
)

with open(os.path.join(output_dir, job_id, "report.html"), "w") as f:
    f.write(html)

print(f"HTML report saved: {{os.path.join(output_dir, job_id, 'report.html')}}")
"""


# ==================================================================== #
#  75 — CrossSessionMemory
# ==================================================================== #

class CrossSessionMemory:
    """
    Store and recall agent learnings across Colab sessions.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_memory (
        memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_type TEXT,
        key TEXT UNIQUE,
        value TEXT,
        created_at TIMESTAMP,
        last_used TIMESTAMP
    );
    """

    def __init__(self, db_path: str = "./agent_memory.db"):
        self.db_path = db_path

    def memory_code(self) -> str:
        return f"""
import sqlite3, json
from datetime import datetime

db_path = "{self.db_path}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT,
    key TEXT UNIQUE,
    value TEXT,
    created_at TIMESTAMP,
    last_used TIMESTAMP
)''')

conn.commit()

def remember(memory_type, key, value):
    cursor.execute('''
        INSERT INTO agent_memory (memory_type, key, value, created_at, last_used)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            last_used=excluded.last_used
    ''', (memory_type, key, json.dumps(value) if isinstance(value, dict) else str(value),
          datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()

def recall(memory_type=None, key=None):
    if key:
        cursor.execute("SELECT * FROM agent_memory WHERE key=?", (key,))
    elif memory_type:
        cursor.execute("SELECT * FROM agent_memory WHERE memory_type=? ORDER BY last_used DESC",
                       (memory_type,))
    else:
        cursor.execute("SELECT * FROM agent_memory ORDER BY last_used DESC LIMIT 20")
    results = cursor.fetchall()
    return [{{"id": r[0], "type": r[1], "key": r[2], "value": r[3],
              "created": r[4], "last_used": r[5]}} for r in results]

# Load at startup
preferences = recall("preference")
learned_fixes = recall("learned_fix")
print(f"Loaded {{len(preferences)}} preferences, {{len(learned_fixes)}} learned fixes")
"""
