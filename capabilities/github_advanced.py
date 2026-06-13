"""
Phases 61-68 — GitHub & Error Management.

Provides:
  - GitHubRepoManager         create/clone/commit/push repos   (61)
  - ErrorCommitAutomation     auto-commit errors to GitHub     (62)
  - SimilarErrorRetriever     find similar past errors         (63)
  - SolutionLearner           learn + recommend fixes          (64)
  - GitHubPullRequester       create PRs from agent results    (65)
  - ConversationArchiver      save full conversations          (66)
  - MetricsVisualizerGitHub   plots + markdown reports         (67)
  - VersionTagOnSuccess       git tag + release on success     (68)
"""
import json
import os
import re
import hashlib
from typing import Optional
from pathlib import Path
from datetime import datetime


# ==================================================================== #
#  61 — GitHubRepoManager
# ==================================================================== #

class GitHubRepoManager:
    """
    Create, clone, commit, and push to GitHub repos.
    """

    def __init__(self, repo_root: str = "./github_workspace"):
        self.repo_root = repo_root
        os.makedirs(repo_root, exist_ok=True)

    def ensure_repo_code(self, repo_name: str, private: bool = True) -> str:
        return f"""
import os, subprocess
from github import Github

repo_name = "{repo_name}"
private = {str(private).lower()}
token = os.environ.get("GITHUB_TOKEN", "")
user_name = os.environ.get("GITHUB_USER", "")

if not token:
    print("GITHUB_TOKEN not set; using local-only mode")
else:
    g = Github(token)
    user = g.get_user()
    try:
        repo = user.get_repo(repo_name)
        print(f"Repo exists: {{repo.html_url}}")
    except Exception:
        repo = user.create_repo(repo_name, private=private)
        print(f"Created repo: {{repo.html_url}}")

    # Clone or pull
    repo_dir = os.path.join("{self.repo_root}", repo_name)
    if os.path.exists(repo_dir):
        subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"],
                       capture_output=True)
        print("Pulled latest")
    else:
        subprocess.run(["git", "clone", repo.clone_url, repo_dir], check=True)
        print(f"Cloned to {{repo_dir}}")
"""

    def commit_code(self, message: str = "Agent update") -> str:
        return f"""
import subprocess, os

repo_dir = "{self.repo_root}"
message = "{message}"

if not os.path.exists(os.path.join(repo_dir, ".git")):
    print("No git repo found. Run ensure_repo first.")
    return

subprocess.run(["git", "-C", repo_dir, "add", "."], check=True)
result = subprocess.run(
    ["git", "-C", repo_dir, "commit", "-m", message],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print(f"Committed: {{result.stdout[:200]}}")
    subprocess.run(["git", "-C", repo_dir, "push", "origin", "main"],
                   capture_output=True)
    print("Pushed to origin/main")
else:
    print(f"Nothing to commit: {{result.stderr[:200]}}")
"""


# ==================================================================== #
#  62 — ErrorCommitAutomation
# ==================================================================== #

class ErrorCommitAutomation:
    """
    On error: capture traceback, cell content, agent state, and commit
    to GitHub automatically.
    """

    def __init__(self, errors_dir: str = "./errors"):
        self.errors_dir = errors_dir
        os.makedirs(errors_dir, exist_ok=True)

    def capture_and_commit_code(self, job_id: str, cell_code: str,
                                traceback_str: str, agent_history: list,
                                gpu_type: str = "T4") -> str:
        return f"""
import os, json, subprocess
from datetime import datetime

job_id = "{job_id}"
errors_dir = "{self.errors_dir}"
gpu_type = "{gpu_type}"
os.makedirs(errors_dir, exist_ok=True)

# Build error report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
error_summary = traceback_str.split("\\\\n")[-2][:50] if traceback_str else "unknown"

report = f\"\"\"# Error Report
**Job ID:** {{job_id}}
**Time:** {{timestamp}}
**Runtime:** {{gpu_type}}

## Cell Code
```python
{cell_code[:1000]}
```

## Traceback
```
{traceback_str[:2000]}
```

## Agent History
{json.dumps(agent_history[-5:], indent=2)}
\"\"\"

# Write file
filepath = os.path.join(errors_dir, f"{{timestamp}}_error.md")
with open(filepath, "w") as f:
    f.write(report)

# Auto-commit
token = os.environ.get("GITHUB_TOKEN", "")
if token and os.path.exists(".git"):
    subprocess.run(["git", "add", filepath], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Error in {{job_id}}: {{error_summary}}"],
        capture_output=True,
    )
    subprocess.run(["git", "push"], capture_output=True)
    print(f"Error committed: {{filepath}}")
else:
    print(f"Error saved locally: {{filepath}}")
"""


# ==================================================================== #
#  63 — SimilarErrorRetriever
# ==================================================================== #

class SimilarErrorRetriever:
    """
    Find similar past errors using sentence embeddings or TF-IDF.
    """

    def __init__(self, embeddings_path: str = "./errors/embeddings.pkl"):
        self.embeddings_path = embeddings_path

    def retrieve_code(self, error_message: str, top_k: int = 3) -> str:
        return f"""
import os, pickle, json
import numpy as np

error_message = '''{error_message}'''
top_k = {top_k}
embeddings_path = "{self.embeddings_path}"

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([error_message])
except ImportError:
    model = None

if os.path.exists(embeddings_path):
    with open(embeddings_path, "rb") as f:
        db = pickle.load(f)
else:
    db = {{"errors": [], "embeddings": [], "solutions": []}}

if model is not None and len(db["embeddings"]) > 0:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(query_emb, np.array(db["embeddings"]))[0]
    indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in indices:
        if sims[i] > 0.5:
            results.append({{
                "error": db["errors"][i][:100],
                "similarity": float(sims[i]),
                "solution": db["solutions"][i] if i < len(db["solutions"]) else "",
            }})
    print(json.dumps(results, indent=2))
elif os.path.exists(embeddings_path):
    # TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(db["errors"] + [error_message])
    query_vec = tfidf_matrix[-1]
    sims = (tfidf_matrix[:-1] @ query_vec.T).toarray().flatten()
    indices = np.argsort(sims)[::-1][:top_k]
    print(f"Fallback (TF-IDF): {{[db['errors'][i][:60] for i in indices if sims[i] > 0.1]}}")
else:
    print("No error history found")
"""


# ==================================================================== #
#  64 — SolutionLearner
# ==================================================================== #

class SolutionLearner:
    """
    Learn from successful fixes and recommend solutions.
    """

    def __init__(self, solutions_dir: str = "./solutions"):
        self.solutions_dir = solutions_dir
        os.makedirs(solutions_dir, exist_ok=True)

    def record_solution_code(self, error_type: str, error_signature: str,
                             solution_actions: list, solution_code: str) -> str:
        return f"""
import json, os, hashlib
from collections import Counter

solutions_dir = "{self.solutions_dir}"
os.makedirs(solutions_dir, exist_ok=True)

error_type = "{error_type}"
error_signature = '''{error_signature}'''
solution_actions = {json.dumps([])}
solution_code = '''{{}}'''

# Create error hash
sig_hash = hashlib.sha256(error_signature.encode()).hexdigest()[:16]
filepath = os.path.join(solutions_dir, f"{{sig_hash}}.json")

existing = {{"success_count": 0, "failure_count": 0}}
if os.path.exists(filepath):
    with open(filepath) as f:
        existing = json.load(f)

entry = {{
    "error_type": error_type,
    "error_signature": error_signature,
    "solution_actions": solution_actions,
    "solution_code": solution_code,
    "success_count": existing.get("success_count", 0) + 1,
    "failure_count": existing.get("failure_count", 0),
}}

with open(filepath, "w") as f:
    json.dump(entry, f, indent=2)

print(f"Solution recorded: {{filepath}} (success #{{entry['success_count']}})")

# Recommend best solution for error
best = None
best_rate = 0
for fname in os.listdir(solutions_dir):
    with open(os.path.join(solutions_dir, fname)) as f:
        sol = json.load(f)
    total = sol["success_count"] + sol["failure_count"]
    rate = sol["success_count"] / total if total > 0 else 0
    if rate > best_rate:
        best_rate = rate
        best = sol

if best:
    print(f"Best solution: {{best['error_type']}} ({{best_rate*100:.0f}}% success)")
    print(f"Actions: {{best['solution_actions']}}")
"""


# ==================================================================== #
#  65 — GitHubPullRequester
# ==================================================================== #

class GitHubPullRequester:
    """
    Create pull requests from agent results.
    """

    def __init__(self, base_branch: str = "main"):
        self.base_branch = base_branch

    def create_pr_code(self, repo_name: str, job_id: str,
                       pr_title: str = "", pr_body: str = "") -> str:
        return f"""
import os, subprocess
from github import Github

repo_name = "{repo_name}"
job_id = "{job_id}"
base_branch = "{self.base_branch}"
branch_name = f"job_{{job_id}}_results"

token = os.environ.get("GITHUB_TOKEN", "")
if not token:
    print("No GITHUB_TOKEN; skipping PR creation")
else:
    g = Github(token)
    repo = g.get_repo(repo_name)

    # Create branch
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Results for job {{job_id}}"], capture_output=True)
    subprocess.run(["git", "push", "origin", branch_name], check=True)
    print(f"Pushed branch: {{branch_name}}")

    # Create PR
    pr = repo.create_pull(
        title="{pr_title}" or f"[Agent] Results for {{job_id}}",
        body="{pr_body}" or f"Auto-generated results for job {{job_id}}.",
        head=branch_name,
        base=base_branch,
    )
    pr.add_to_labels("agent-generated")
    print(f"PR created: {{pr.html_url}}")
"""


# ==================================================================== #
#  66 — ConversationArchiver
# ==================================================================== #

class ConversationArchiver:
    """
    Save full agent conversations to GitHub and Drive.
    """

    def __init__(self, archive_dir: str = "./conversations"):
        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)

    def archive_code(self, job_id: str, conversation: list[dict]) -> str:
        return f"""
import os, json, zipfile
from datetime import datetime

job_id = "{job_id}"
archive_dir = "{self.archive_dir}"
os.makedirs(archive_dir, exist_ok=True)

conversation = {json.dumps(conversation)}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Format as markdown
md_lines = [f"## User Goal: {{conversation[0]['content'] if conversation else 'N/A'}}"]
for msg in conversation:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    ts = msg.get("timestamp", "")
    md_lines.append(f"**[{{ts}}] {{role.capitalize()}}:** {{content[:500]}}")

md_content = "\\\\n".join(md_lines)

# Save as markdown
md_path = os.path.join(archive_dir, f"{{job_id}}_conversation.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# Also save as JSONL
jsonl_path = os.path.join(archive_dir, f"{{job_id}}_conversation.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for msg in conversation:
        f.write(json.dumps(msg) + "\\\\n")

# Create zip
zip_path = os.path.join(archive_dir, f"{{job_id}}_conversation.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(md_path, arcname="conversation.md")
    zf.write(jsonl_path, arcname="conversation.jsonl")

print(f"Archived: {{zip_path}} ({{len(conversation)}} messages)")
"""


# ==================================================================== #
#  67 — MetricsVisualizerGitHub
# ==================================================================== #

class MetricsVisualizerGitHub:
    """
    Generate plots and HTML reports, commit to GitHub.
    """

    def __init__(self, reports_dir: str = "./reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_report_code(self, job_id: str, metrics: dict) -> str:
        return f"""
import os, json, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

job_id = "{job_id}"
metrics = {json.dumps(metrics)}
reports_dir = "{self.reports_dir}"
output_dir = os.path.join(reports_dir, job_id, "metrics")
os.makedirs(output_dir, exist_ok=True)

# Loss curve
if metrics.get("history"):
    history = metrics["history"]
    steps = [h.get("_step", i) for i, h in enumerate(history)]
    losses = [h.get("loss") for h in history if "loss" in h]
    if losses:
        plt.figure(figsize=(8, 4))
        plt.plot(steps[:len(losses)], losses, "b-", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss - Job {{job_id}}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=100, bbox_inches="tight")
        plt.close()

# GPU memory
memories = [h.get("gpu_mem_gb") for h in history if "gpu_mem_gb" in h] if history else []
if memories:
    plt.figure(figsize=(8, 3))
    plt.fill_between(range(len(memories)), memories, alpha=0.3, color="green")
    plt.plot(range(len(memories)), memories, "g-")
    plt.xlabel("Log Step")
    plt.ylabel("GPU Memory (GB)")
    plt.title("GPU Memory Usage")
    plt.savefig(os.path.join(output_dir, "gpu_memory.png"), dpi=100, bbox_inches="tight")
    plt.close()

# Generate markdown report
md = f\"\"\"## Training Metrics for Job {{job_id}}
![Loss Curve](./metrics/loss_curve.png)
![GPU Memory](./metrics/gpu_memory.png)

### Best Metrics
- Loss: {{metrics.get("final_loss", "N/A")}}
- Accuracy: {{metrics.get("accuracy", "N/A")}}
\"\"\"

with open(os.path.join(reports_dir, job_id, "report.md"), "w") as f:
    f.write(md)

# Commit to GitHub
if os.path.exists(".git"):
    subprocess.run(["git", "add", output_dir], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"Metrics report for job {{job_id}}"],
                   capture_output=True)
    subprocess.run(["git", "push"], capture_output=True)
    print(f"Report pushed to GitHub: {{output_dir}}")
else:
    print(f"Report saved locally: {{output_dir}}")
"""


# ==================================================================== #
#  68 — VersionTagOnSuccess
# ==================================================================== #

class VersionTagOnSuccess:
    """
    Create git tags and releases on successful fine-tuning.
    """

    def __init__(self):
        pass

    def tag_code(self, job_id: str, model_name: str,
                 metrics: dict, hf_url: str = "") -> str:
        sanitized_model = model_name.replace("/", "-").replace("_", "-")
        return f"""
import os, json, subprocess
from github import Github
from datetime import datetime

job_id = "{job_id}"
model_name = "{model_name}"
hf_url = "{hf_url}"
metrics = {json.dumps(metrics)}

tag_name = f"v{{job_id}}_{{{model_name.replace('/', '-')}}}_{{datetime.now().strftime('%Y%m%d')}}"
release_notes = f\"\"\"## Release {{tag_name}}

**Model:** {{model_name}}
**Job ID:** {{job_id}}
**Date:** {{datetime.now().isoformat()}}

### Performance
- Loss: {{metrics.get("final_loss", "N/A")}}
- Accuracy: {{metrics.get("accuracy", "N/A")}}

### Links
- HF Hub: {{hf_url or "N/A"}}
\"\"\"

# Local git tag
subprocess.run(["git", "tag", "-a", tag_name, "-m", release_notes[:200]], capture_output=True)
subprocess.run(["git", "push", "origin", tag_name], capture_output=True)
print(f"Git tag created: {{tag_name}}")

# GitHub release
token = os.environ.get("GITHUB_TOKEN", "")
repo_name = os.environ.get("GITHUB_REPO", "")
if token and repo_name:
    g = Github(token)
    repo = g.get_repo(repo_name)
    release = repo.create_release(
        tag=tag_name,
        name=tag_name,
        body=release_notes,
        draft=False,
        prerelease=False,
    )
    print(f"Release created: {{release.html_url}}")
"""
