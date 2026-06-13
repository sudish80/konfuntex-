"""
Self-improvement system for Colab Agent.

Analyzes completed runs to detect failure patterns and suggest
improvements to prompts, code generation, and model selection.

Hooks into on_complete lifecycle event.
"""

import os
import json
import time
import logging
import threading
from collections import Counter

from agent.plugin import Plugin, plugin


logger = logging.getLogger(__name__)


FAILURE_PATTERNS = {
    "OOM": ["out of memory", "cuda out of", "oom", "allocate"],
    "API_ERROR": ["api error", "429", "rate limit", "connection"],
    "IMPORT_ERROR": ["module not found", "no module named", "import error"],
    "SYNTAX_ERROR": ["syntaxerror", "invalid syntax", "unexpected indent"],
    "CUDA_ERROR": ["cuda error", "device-side assert", "cuda driver"],
    "DATASET_ERROR": ["dataset not found", "cannot load dataset"],
    "MODEL_ERROR": ["model not found", "cannot load model"],
    "DISK_FULL": ["no space left", "disk full", "insufficient space"],
    "TIMEOUT": ["timed out", "timeout after"],
}


def classify_error(error_text: str) -> str:
    if not error_text:
        return "UNKNOWN"
    error_lower = error_text.lower()
    for pattern_name, keywords in FAILURE_PATTERNS.items():
        for kw in keywords:
            if kw in error_lower:
                return pattern_name
    return "UNKNOWN"


IMPROVEMENT_SUGGESTIONS = {
    "OOM": {
        "title": "Reduce memory usage",
        "detail": "Use QLoRA instead of LoRA, reduce batch size, or switch to a smaller model",
        "prompt_fix": "When generating training code, prefer QLoRA for memory-constrained runtimes",
    },
    "API_ERROR": {
        "title": "Add API retry with backoff",
        "detail": "Wrap API calls in retry logic with exponential backoff",
        "prompt_fix": "Add retry logic with exponential backoff (max 3 retries, starting at 2s delay)",
    },
    "IMPORT_ERROR": {
        "title": "Add missing pip install",
        "detail": "Ensure all required packages are installed before import",
        "prompt_fix": "Always include pip install commands for required packages before imports",
    },
    "SYNTAX_ERROR": {
        "title": "Fix code syntax",
        "detail": "Review generated code for syntax errors before execution",
        "prompt_fix": "Return ONLY valid Python code. Use proper indentation and syntax.",
    },
    "CUDA_ERROR": {
        "title": "Check CUDA availability",
        "detail": "Add CUDA version checks and fallback to CPU",
        "prompt_fix": "Add torch.cuda.is_available() check with graceful fallback",
    },
    "DATASET_ERROR": {
        "title": "Verify dataset availability",
        "detail": "Check dataset exists and is accessible before loading",
        "prompt_fix": "Verify the dataset name is correct and publicly available on HuggingFace",
    },
    "MODEL_ERROR": {
        "title": "Verify model availability",
        "detail": "Check model exists on HuggingFace and is accessible",
        "prompt_fix": "Verify the model ID is correct and available on HuggingFace Hub",
    },
    "TIMEOUT": {
        "title": "Increase timeout or optimize step",
        "detail": "Operation timed out — consider increasing timeout or optimizing the code",
        "prompt_fix": "Add progress printing and consider splitting long operations into chunks",
    },
    "DISK_FULL": {
        "title": "Free disk space",
        "detail": "Disk is full — clean up temporary files, reduce cache size, or use a smaller dataset",
        "prompt_fix": "Clean up temporary files after download and use streaming datasets to save disk space",
    },
}


@plugin(name="self_improvement", version="1.0.0",
        description="Analyzes runs and suggests improvements to prompts and code generation",
        priority=50)
class SelfImprovementPlugin(Plugin):
    """Analyzes completed runs and suggests improvements.

    Thread-safe. Stores failure history in a JSONL file and generates
    improvement recommendations on each run completion.
    """

    def __init__(self):
        super().__init__()
        from config.settings import settings
        self._lock = threading.RLock()
        self._history_path = os.path.join(settings.data_dir, "improvement_history.jsonl")
        self._run_count = 0
        self._failure_counts: Counter = Counter()
        self._pattern_history: list[dict] = []

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        with self._lock:
            self._run_count += 1
            status = result.get("status", "unknown")

        logger.info(f"Self-improvement: run completed with status '{status}'")

        if status == "completed":
            result["_improvement"] = {"run_count": self._run_count, "suggestions": []}
            return result, context

        job_id = result.get("job_id", "unknown")
        error_patterns = self._analyze_failures(result)

        entry = {
            "timestamp": time.time(),
            "run_count": self._run_count,
            "job_id": job_id,
            "status": status,
            "patterns": error_patterns,
        }
        self._append_history(entry)

        with self._lock:
            self._pattern_history.append(entry)
            for p in error_patterns:
                self._failure_counts[p["type"]] += 1
            suggestions = self._generate_suggestions(error_patterns)
            failure_summary = dict(self._failure_counts.most_common())

        result["_improvement"] = {
            "run_count": self._run_count,
            "suggestions": suggestions,
            "total_runs_analyzed": self._run_count,
            "failure_summary": failure_summary,
        }

        if suggestions:
            logger.info(f"Self-improvement: {len(suggestions)} suggestions generated")
            for s in suggestions:
                logger.info(f"  - [{s['type']}] {s['title']}")

        return result, context

    def _analyze_failures(self, result: dict) -> list[dict]:
        if not isinstance(result, dict):
            return []
        seen_types = set()
        patterns = []
        summary = result.get("summary", "")
        plan = result.get("plan", {})

        if isinstance(summary, str):
            pat = classify_error(summary)
            if pat != "UNKNOWN" and pat not in seen_types:
                seen_types.add(pat)
                patterns.append({"type": pat, "source": "summary"})

        if isinstance(plan, dict):
            for step in plan.get("steps", []):
                if isinstance(step, dict) and step.get("status") == "failed":
                    error = step.get("error", "")
                    if isinstance(error, str):
                        step_pat = classify_error(error)
                        if step_pat != "UNKNOWN" and step_pat not in seen_types:
                            seen_types.add(step_pat)
                            patterns.append({
                                "type": step_pat, "source": "step",
                                "step_id": step.get("id"),
                            })

        return patterns

    def _generate_suggestions(self, patterns: list[dict]) -> list[dict]:
        seen = set()
        suggestions = []
        for p in patterns:
            ptype = p["type"]
            if ptype in seen:
                continue
            seen.add(ptype)
            info = IMPROVEMENT_SUGGESTIONS.get(ptype)
            if info:
                with self._lock:
                    confidence = min(1.0, self._failure_counts.get(ptype, 0) * 0.25)
                suggestions.append({
                    "type": ptype,
                    "title": info["title"],
                    "detail": info["detail"],
                    "prompt_fix": info["prompt_fix"],
                    "confidence": confidence,
                })
        return suggestions

    def _append_history(self, entry: dict):
        try:
            os.makedirs(os.path.dirname(self._history_path), exist_ok=True)
            with open(self._history_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.warning(f"Could not write improvement history: {e}")

    def get_statistics(self) -> dict:
        with self._lock:
            return {
                "run_count": self._run_count,
                "failure_counts": dict(self._failure_counts.most_common()),
                "history_count": len(self._pattern_history),
            }

    def get_best_practices(self) -> list[str]:
        with self._lock:
            practices = []
            for ptype, count in self._failure_counts.most_common(5):
                if count >= 2:
                    info = IMPROVEMENT_SUGGESTIONS.get(ptype)
                    if info:
                        practices.append(f"[{ptype}] {info['prompt_fix']}")
            return list(practices)

    def on_summary(self, summary: str, context: dict) -> tuple[str, dict]:
        best = self.get_best_practices()
        if best:
            practices_text = "\n".join(f"  \u2022 {p}" for p in best)
            summary += f"\n\n[Self-Improvement]\nApplied best practices from previous runs:\n{practices_text}"
        return summary, context
