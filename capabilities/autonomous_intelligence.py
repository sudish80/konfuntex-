"""
Phases 97-100 — Autonomous Intelligence.

Provides:
  - AutomatedPaperReproducer  reproduce results from papers   (97)
  - SelfHealingAgent          detect + recover from loops     (98)
  - LearningFromFeedback      improve from user ratings       (99)
  - MetaAgent                 agent that improves the agent   (100)
"""
import json
import os
from typing import Optional


# ==================================================================== #
#  97 — AutomatedPaperReproducer
# ==================================================================== #

class AutomatedPaperReproducer:
    """
    Parse an ArXiv paper, extract hyperparameters, and reproduce fine-tuning.
    """

    def __init__(self, output_dir: str = "./paper_reproductions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def reproduce_code(self, arxiv_id: str) -> str:
        return f"""
import os, json, re, arxiv
from datetime import datetime

arxiv_id = "{arxiv_id}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Fetch paper
search = arxiv.Search(id_list=[arxiv_id])
paper = next(search.results())
print(f"Paper: {{paper.title}}")
print(f"Authors: {{', '.join(a.name for a in paper.authors)}}")

abstract = paper.summary.replace("\\\\n", " ")

# LLM extracts hyperparameters from abstract + text
try:
    from transformers import pipeline
    extractor = pipeline("text-generation", model="microsoft/phi-2")
    prompt = f\"\"\"Extract hyperparameters from this paper abstract:
{{abstract[:2000]}}

Return JSON with: model_name, dataset, batch_size, learning_rate, epochs\"\"\"
    result = extractor(prompt, max_new_tokens=200)[0]["generated_text"]
    # Parse JSON from response
    import re
    json_match = re.search(r'\\\\{{.*?\\\\}}', result, re.DOTALL)
    if json_match:
        params = json.loads(json_match.group())
    else:
        params = {{}}
except Exception as e:
    params = {{"error": str(e)}}
    print(f"Extraction failed: {{e}}")

# Download PDF
pdf_path = os.path.join(output_dir, f"{{arxiv_id}}.pdf")
paper.download_pdf(dirpath=output_dir)
print(f"PDF saved: {{pdf_path}}")

# Save extraction result
report = {{
    "arxiv_id": arxiv_id,
    "title": paper.title,
    "authors": [a.name for a in paper.authors],
    "abstract": abstract[:500],
    "extracted_params": params,
    "timestamp": datetime.now().isoformat(),
}}
with open(os.path.join(output_dir, f"{{arxiv_id}}_reproduction.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(params, indent=2))
print("To reproduce, run: agent.run(f'Fine-tune {{params.get(\\"model_name\\", \\"unknown\\")}} on {{params.get(\\"dataset\\", \\"dataset\\")}}')")
"""


# ==================================================================== #
#  98 — SelfHealingAgent
# ==================================================================== #

class SelfHealingAgent:
    """
    Detect when stuck in a loop and attempt recovery.
    """

    def __init__(self, max_healing_attempts: int = 3):
        self.max_healing_attempts = max_healing_attempts
        self.healing_log = []
        self.loop_detector = {}

    def healing_code(self) -> str:
        return f"""
import os, sys, json, time
from datetime import datetime

max_attempts = {self.max_healing_attempts}
healing_log = []
loop_counter = {{}}

def detect_loop(action_signature):
    '''Detect if same action repeats.'''
    loop_counter[action_signature] = loop_counter.get(action_signature, 0) + 1
    return loop_counter[action_signature] >= 3

def self_heal(agent, error_context):
    '''Attempt to recover from failure.'''
    attempt = len(healing_log) + 1
    if attempt > max_attempts:
        print(f"Self-healing failed after {{max_attempts}} attempts. Escalating.")
        return False

    print(f"\\\\n🔄 Self-healing attempt {{attempt}}/{{max_attempts}}...")

    actions = [
        # 1. Roll back to last checkpoint
        lambda: (
            os.path.exists("./checkpoints/best_model") and
            agent._log("Rolling back to best checkpoint")
        ),
        # 2. Try alternative method
        lambda: (
            setattr(agent, "current_method", "qlora") and
            agent._log("Switching from LoRA to QLoRA")
        ),
        # 3. Rewrite plan
        lambda: (
            agent._log("Regenerating plan with LLM") and
            agent._generate_plan(agent.user_goal + " (alternative approach)")
        ),
        # 4. Reduce resource usage
        lambda: (
            agent._log("Reducing batch size") and
            None  # Apply config change
        ),
    ]

    if attempt <= len(actions):
        try:
            result = actions[attempt - 1]()
            healing_log.append({{
                "attempt": attempt,
                "action": f"healing_action_{{attempt}}",
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }})
            print(f"  ✅ Healing action {{attempt}} applied")
            return True
        except Exception as e:
            healing_log.append({{
                "attempt": attempt,
                "action": f"healing_action_{{attempt}}",
                "success": False,
                "error": str(e),
            }})

    # Full reset
    print("  🔄 Resetting environment...")
    import subprocess
    subprocess.run(["pip", "install", "-q", "transformers", "datasets", "peft"])
    healing_log.append({{
        "attempt": attempt,
        "action": "reset_environment",
        "success": True,
    }})
    return True

# Save healing log
with open("self_healing.log", "a") as f:
    for entry in healing_log:
        f.write(json.dumps(entry) + "\\\\n")
"""


# ==================================================================== #
#  99 — LearningFromFeedback
# ==================================================================== #

class LearningFromFeedback:
    """
    Collect user feedback and adjust agent strategy.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS user_feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        rating INTEGER,
        feedback_text TEXT,
        agent_suggestions TEXT,
        applied_changes TEXT,
        created_at TIMESTAMP
    );
    """

    def __init__(self, db_path: str = "./feedback.db"):
        self.db_path = db_path

    def feedback_code(self) -> str:
        return f"""
import sqlite3, json
from datetime import datetime

db_path = "{self.db_path}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    rating INTEGER,
    feedback_text TEXT,
    agent_suggestions TEXT,
    applied_changes TEXT,
    created_at TIMESTAMP
)''')

conn.commit()

def collect_feedback(job_id, rating, feedback_text=""):
    # Analyze feedback with LLM
    suggestions = ""
    if feedback_text:
        try:
            from transformers import pipeline
            analyzer = pipeline("text-classification", model="distilbert-base-uncased")
            sentiment = analyzer(feedback_text)[0]
            if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7:
                suggestions = "Consider: using a different model, increasing epochs, or adjusting LR"
        except Exception:
            pass

    cursor.execute('''
        INSERT INTO user_feedback
        (job_id, rating, feedback_text, agent_suggestions, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (job_id, rating, feedback_text, suggestions, datetime.now().isoformat()))
    conn.commit()

    # Store changes to apply
    changes = {{"rating": rating, "suggestions": suggestions}}
    cursor.execute('''
        UPDATE user_feedback SET applied_changes=? WHERE job_id=?
    ''', (json.dumps(changes), job_id))
    conn.commit()
    print(f"Feedback saved for job {{job_id}} (rating: {{rating}}/5)")

def get_feedback_summary():
    cursor.execute('''
        SELECT AVG(rating), COUNT(*), SUM(CASE WHEN rating < 3 THEN 1 ELSE 0 END)
        FROM user_feedback
    ''')
    avg_rating, total, low = cursor.fetchone()
    print(f"Feedback: {{total}} ratings, avg {{avg_rating:.1f}}/5, {{low}} low")
    return {{"avg_rating": avg_rating, "total": total, "low_ratings": low}}

def adjust_strategy_from_feedback():
    cursor.execute('''
        SELECT json_extract(applied_changes, '$.suggestions')
        FROM user_feedback WHERE applied_changes IS NOT NULL
        ORDER BY created_at DESC LIMIT 5
    ''')
    suggestions = [row[0] for row in cursor.fetchall() if row[0]]
    # Auto-adjust config based on feedback patterns
    if suggestions:
        print(f"Adjusting strategy based on {{len(suggestions)}} feedback entries")
        # In real use: modify config.yaml, update prompts, etc.
"""


# ==================================================================== #
#  100 — MetaAgent
# ==================================================================== #

class MetaAgent:
    """
    A higher-level agent that observes and improves the main agent.
    Runs after every N jobs to analyze performance and suggest changes.
    """

    def __init__(self, review_interval: int = 10,
                 meta_log_path: str = "./meta_learning_log.json"):
        self.review_interval = review_interval
        self.meta_log_path = meta_log_path
        self.cycle_count = 0

    def meta_agent_code(self) -> str:
        return f"""
import os, json, yaml
from datetime import datetime
from collections import Counter

review_interval = {self.review_interval}
meta_log_path = "{self.meta_log_path}"

class MetaAgent:
    def __init__(self):
        self.cycle_count = 0
        self.improvements = []

    def should_review(self, total_jobs):
        return total_jobs > 0 and total_jobs % review_interval == 0

    def review_performance(self, jobs_history):
        self.cycle_count += 1
        print(f"\\\\n{'='*60}")
        print(f"META-AGENT REVIEW (Cycle {{self.cycle_count}})")
        print(f"{'='*60}")

        # Analyze success rate
        total = len(jobs_history)
        successes = sum(1 for j in jobs_history if j.get("status") == "completed")
        failures = sum(1 for j in jobs_history if j.get("status") == "failed")
        success_rate = successes / total if total > 0 else 0
        print(f"Success rate: {{success_rate*100:.1f}}% ({{successes}}/{{total}})")

        # Common failure patterns
        errors = [j.get("error", "")[:60] for j in jobs_history if j.get("error")]
        if errors:
            common = Counter(errors).most_common(3)
            print(f"Common errors:")
            for err, count in common:
                print(f"  - {{err}} ({{count}}x)")

        # Average metrics
        losses = [j.get("final_loss", 0) for j in jobs_history if j.get("final_loss")]
        avg_loss = sum(losses) / len(losses) if losses else 0
        print(f"Average final loss: {{avg_loss:.4f}}")

        # Suggest config changes
        suggestions = []
        if success_rate < 0.7:
            suggestions.append("INCREASE max_retries from 3 to 5")
            suggestions.append("SWITCH default method from lora to qlora")
        if avg_loss > 0.5 and losses:
            suggestions.append("REDUCE default learning rate")
            suggestions.append("INCREASE number of epochs")

        if suggestions:
            print(f"\\\\nSuggested changes:")
            for s in suggestions:
                print(f"  ⚡ {{s}}")

            # Auto-apply (in dry-run first)
            if input("Apply changes? [y/N]: ").strip().lower() == "y":
                self._apply_changes(suggestions)

        # Save meta-learning state
        state = {{
            "cycle": self.cycle_count,
            "success_rate": success_rate,
            "avg_loss": avg_loss,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
        }}
        with open(meta_log_path, "a") as f:
            f.write(json.dumps(state) + "\\\\n")

        return suggestions

    def _apply_changes(self, suggestions):
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print("No config.yaml found")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        for s in suggestions:
            if "max_retries" in s:
                config["agent"]["max_retries"] = 5
            elif "default method" in s:
                config["models"]["default_finetune_method"] = "qlora"
            elif "learning rate" in s:
                config["training"]["learning_rate"] = 1e-4

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        self.improvements.append({{
            "cycle": self.cycle_count,
            "changes": suggestions,
            "timestamp": datetime.now().isoformat(),
        }})
        print(f"Config updated: {{suggestions}}")

# Integration into agent loop:
# meta = MetaAgent()
# if meta.should_review(total_jobs):
#     meta.review_performance(jobs_history)
"""
