"""Tests for agent/safety.py — sanitize_code, sanitize_pip, compute_code_hash, CostTracker."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.safety import (
    sanitize_code, sanitize_pip, compute_code_hash, CostTracker,
    ALLOWED_PIP_PACKAGES, COST_PER_GPU_HOUR,
    MAX_RUNTIME_SWITCHES, MAX_ERROR_RETRIES_PER_STEP, MAX_TOTAL_ERROR_RETRIES,
    MAX_COLAB_HOURS, MAX_VRAM_GB, MAX_TRAINING_STEPS,
)


class TestSanitizeCode:
    def test_safe_code(self):
        is_safe, cleaned, warn = sanitize_code("print('hello')")
        assert is_safe
        assert cleaned == "print('hello')"
        assert warn == ""

    def test_safe_training_code(self):
        code = """
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
print("loaded")
"""
        is_safe, cleaned, warn = sanitize_code(code)
        assert is_safe

    def test_blocks_rm_rf(self):
        is_safe, _, warn = sanitize_code("import os; os.system('rm -rf /')")
        assert not is_safe
        assert "rm -rf" in warn

    def test_allows_rmtree(self):
        is_safe, _, warn = sanitize_code("shutil.rmtree('/')")
        assert is_safe

    def test_blocks_dynamic_os_import(self):
        is_safe, _, warn = sanitize_code('__import__("os")')
        assert not is_safe
        assert "dynamic os import" in warn

    def test_blocks_system_file_access(self):
        is_safe, _, warn = sanitize_code('open("/etc/passwd")')
        assert not is_safe

    def test_safe_non_matching(self):
        is_safe, _, warn = sanitize_code('print("hello world")')
        assert is_safe
        assert warn == ""

    def test_blocks_rm_in_string(self):
        is_safe, _, warn = sanitize_code('print("the term rm -rf is mentioned")')
        assert not is_safe
        assert "rm -rf" in warn


class TestSanitizePip:
    def test_allowed_package_passes(self):
        for pkg in ["transformers", "datasets", "torch", "pandas", "numpy"]:
            assert pkg in ALLOWED_PIP_PACKAGES, f"{pkg} should be allowed"
            result = sanitize_pip(f"!pip install {pkg}")
            assert "BLOCKED" not in result, f"{pkg} should not be blocked"

    def test_blocked_package(self):
        result = sanitize_pip("!pip install malicious_package_xyz")
        assert "BLOCKED" in result

    def test_mixed_allowed_and_blocked(self):
        result = sanitize_pip("!pip install transformers malicious_package")
        assert "BLOCKED" in result
        assert "malicious_package" in result

    def test_no_pip_line(self):
        code = "import torch\nprint('hi')"
        assert sanitize_pip(code) == code

    def test_with_flags(self):
        code = "!pip install -q -U transformers"
        result = sanitize_pip(code)
        assert "BLOCKED" not in result

    def test_version_pinned(self):
        result = sanitize_pip("!pip install transformers>=4.36.0")
        assert "BLOCKED" not in result


class TestComputeCodeHash:
    def test_deterministic(self):
        code = "print('hello')"
        assert compute_code_hash(code) == compute_code_hash(code)

    def test_different_inputs_different(self):
        h1 = compute_code_hash("print('hello')")
        h2 = compute_code_hash("print('world')")
        assert h1 != h2

    def test_length(self):
        h = compute_code_hash("any code")
        assert len(h) == 16

    def test_empty_string(self):
        h = compute_code_hash("")
        assert len(h) == 16


class TestCostTracker:
    def test_start_job_creates_entry(self):
        ct = CostTracker()
        ct.start_job("job-1", "T4")
        s = ct.get_summary("job-1")
        assert s["job_id"] == "job-1"
        assert s["current_runtime"] == "T4"

    def test_record_execution_adds_seconds(self):
        ct = CostTracker()
        ct.start_job("job-1")
        ct.record_execution("job-1", 3600)
        s = ct.get_summary("job-1")
        assert s["gpu_hours"] == 1.0

    def test_record_execution_unknown_job(self):
        ct = CostTracker()
        ct.record_execution("nonexistent", 100)

    def test_record_retry(self):
        ct = CostTracker()
        ct.start_job("job-1")
        ct.record_retry("job-1")
        ct.record_retry("job-1")
        s = ct.get_summary("job-1")
        assert s["total_retries"] == 2

    def test_record_switch_updates_runtime(self):
        ct = CostTracker()
        ct.start_job("job-1", "T4")
        ct.record_switch("job-1", "V100")
        s = ct.get_summary("job-1")
        assert s["current_runtime"] == "V100"
        assert s["runtime_switches"] == 1

    def test_get_summary(self):
        ct = CostTracker()
        ct.start_job("job-1", "T4")
        ct.record_execution("job-1", 7200)
        ct.record_retry("job-1")
        summary = ct.get_summary("job-1")
        assert summary["job_id"] == "job-1"
        assert summary["gpu_hours"] == 2.0
        assert summary["estimated_cost_units"] == 2.0
        assert summary["total_retries"] == 1

    def test_get_summary_v100(self):
        ct = CostTracker()
        ct.start_job("job-1", "V100")
        ct.record_execution("job-1", 3600)
        summary = ct.get_summary("job-1")
        assert summary["estimated_cost_units"] == 1.5

    def test_get_summary_nonexistent_job(self):
        ct = CostTracker()
        summary = ct.get_summary("no-such-job")
        assert summary["job_id"] == "no-such-job"


class TestConstants:
    def test_max_runtime_switches(self):
        assert MAX_RUNTIME_SWITCHES == 5

    def test_max_retries_per_step(self):
        assert MAX_ERROR_RETRIES_PER_STEP == 10

    def test_max_total_retries(self):
        assert MAX_TOTAL_ERROR_RETRIES == 50

    def test_max_colab_hours(self):
        assert MAX_COLAB_HOURS == 12

    def test_max_vram_gb(self):
        assert MAX_VRAM_GB == 80

    def test_max_training_steps(self):
        assert MAX_TRAINING_STEPS == 100000

    def test_cost_per_gpu_hour_keys(self):
        for gpu in ["None", "T4", "V100", "A100", "A100-80GB", "TPU"]:
            assert gpu in COST_PER_GPU_HOUR


class TestBudgetManager:
    def test_default_not_exceeded(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=100)
        assert bm.exceeded is False
        assert bm.usage_ratio == 0.0

    def test_exceeded(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=10)
        bm.record_cost(15)
        assert bm.exceeded is True

    def test_usage_ratio(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=100)
        bm.record_cost(25)
        assert bm.usage_ratio == 0.25

    def test_alert_on_warning(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=100, warn_threshold=0.5)
        bm.record_cost(60)
        assert len(bm.alerts) >= 1
        assert "WARNING" in bm.alerts[0]

    def test_alert_on_exceeded(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=100)
        bm.record_cost(100)
        assert len(bm.alerts) >= 1
        assert "EXCEEDED" in bm.alerts[0]

    def test_clear_alerts(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=10)
        bm.record_cost(10)
        bm.clear_alerts()
        assert bm.alerts == []

    def test_snapshot_keys(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=100)
        bm.record_cost(30)
        s = bm.snapshot()
        assert s["max_cost_units"] == 100
        assert s["spent"] == 30
        assert s["remaining"] == 70
        assert s["usage_pct"] == 30.0
        assert s["exceeded"] is False
        assert s["alert_count"] == 0

    def test_snapshot_remaining_floor(self):
        from agent.safety import BudgetManager
        bm = BudgetManager(max_cost_units=10)
        bm.record_cost(20)
        assert bm.snapshot()["remaining"] == 0

    def test_budget_endpoint(self):
        from agent.core import budget_endpoint
        s = budget_endpoint()
        assert "max_cost_units" in s
        assert "spent" in s
        assert "remaining" in s
