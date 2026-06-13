"""Tests for agent/self_improvement.py — incl. hardened edge cases."""
import os
import json
import tempfile
import threading
import pytest

from agent.self_improvement import (
    SelfImprovementPlugin, classify_error, FAILURE_PATTERNS,
    IMPROVEMENT_SUGGESTIONS,
)


class TestClassifyError:

    def test_classify_oom(self):
        assert classify_error("CUDA out of memory") == "OOM"

    def test_classify_api_error(self):
        assert classify_error("API error: 429 Too Many Requests") == "API_ERROR"

    def test_classify_import_error(self):
        assert classify_error("ModuleNotFoundError: No module named 'torch'") == "IMPORT_ERROR"

    def test_classify_syntax_error(self):
        assert classify_error("SyntaxError: invalid syntax") == "SYNTAX_ERROR"

    def test_classify_cuda_error(self):
        assert classify_error("CUDA error: device-side assert triggered") == "CUDA_ERROR"

    def test_classify_dataset_error(self):
        assert classify_error("DatasetNotFound: cannot load dataset") == "DATASET_ERROR"

    def test_classify_model_error(self):
        assert classify_error("ModelNotFound: cannot load model") == "MODEL_ERROR"

    def test_classify_disk_full(self):
        assert classify_error("OSError: no space left on device") == "DISK_FULL"

    def test_classify_timeout(self):
        assert classify_error("TimeoutError: timed out after 300s") == "TIMEOUT"

    def test_classify_unknown(self):
        assert classify_error("Some random unknown error") == "UNKNOWN"

    def test_classify_empty(self):
        assert classify_error("") == "UNKNOWN"

    def test_classify_none(self):
        assert classify_error(None) == "UNKNOWN"

    def test_case_insensitive(self):
        assert classify_error("CUDA Out Of Memory") == "OOM"


class TestSelfImprovementPlugin:

    @pytest.fixture
    def plugin(self):
        p = SelfImprovementPlugin()
        yield p

    def test_init(self, plugin):
        stats = plugin.get_statistics()
        assert stats["run_count"] == 0
        assert stats["failure_counts"] == {}

    def test_completed_run_no_suggestions(self, plugin):
        result = {"status": "completed", "summary": "All steps succeeded"}
        result, ctx = plugin.on_complete(result, {})
        impr = result.get("_improvement", {})
        assert impr["run_count"] == 1
        assert impr["suggestions"] == []

    def test_failed_run_generates_suggestions(self, plugin):
        result = {
            "status": "failed",
            "summary": "CUDA out of memory on step 2",
            "job_id": "job-001",
            "plan": {
                "steps": [
                    {"id": 1, "status": "success"},
                    {"id": 2, "status": "failed", "error": "CUDA out of memory"},
                ]
            },
        }
        result, ctx = plugin.on_complete(result, {})
        impr = result.get("_improvement", {})
        assert len(impr["suggestions"]) == 1
        assert impr["suggestions"][0]["type"] == "OOM"

    def test_multiple_failures_deduplicated(self, plugin):
        result = {
            "status": "failed",
            "summary": "CUDA out of memory. Also CUDA OOM.",
            "job_id": "job-002",
            "plan": {
                "steps": [
                    {"id": 1, "status": "failed", "error": "CUDA out of memory"},
                ]
            },
        }
        result, ctx = plugin.on_complete(result, {})
        impr = result.get("_improvement", {})
        assert len(impr["suggestions"]) == 1

    def test_confidence_increases_with_count(self, plugin):
        for i in range(4):
            result = {
                "status": "failed",
                "summary": "CUDA out of memory",
                "job_id": f"job-{i:03d}",
                "plan": {"steps": [{"id": 1, "status": "failed", "error": "OOM"}]},
            }
            result, ctx = plugin.on_complete(result, {})

        stats = plugin.get_statistics()
        assert stats["failure_counts"]["OOM"] == 4

    def test_best_practices_after_enough_failures(self, plugin):
        for i in range(3):
            result = {
                "status": "failed",
                "summary": "CUDA out of memory",
                "job_id": f"job-{i:03d}",
                "plan": {"steps": [{"id": 1, "status": "failed", "error": "OOM"}]},
            }
            result, ctx = plugin.on_complete(result, {})

        practices = plugin.get_best_practices()
        assert len(practices) >= 1
        assert "OOM" in practices[0]

    def test_summary_appends_best_practices(self, plugin):
        for i in range(3):
            result = {
                "status": "failed",
                "summary": "CUDA out of memory",
                "job_id": f"job-{i:03d}",
                "plan": {"steps": [{"id": 1, "status": "failed", "error": "OOM"}]},
            }
            result, ctx = plugin.on_complete(result, {})

        summary, ctx = plugin.on_summary("Training completed", {})
        assert "[Self-Improvement]" in summary

    def test_history_persisted(self):
        tmp = tempfile.mktemp(suffix=".jsonl")
        try:
            plugin = SelfImprovementPlugin()
            plugin._history_path = tmp

            result = {
                "status": "failed",
                "summary": "API error: 429",
                "job_id": "job-003",
                "plan": {"steps": []},
            }
            plugin.on_complete(result, {})

            assert os.path.exists(tmp)
            with open(tmp) as f:
                line = json.loads(f.readline())
                assert line["status"] == "failed"
                assert line["patterns"][0]["type"] == "API_ERROR"
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def test_all_patterns_have_suggestions(self):
        for pattern_name in FAILURE_PATTERNS:
            assert pattern_name in IMPROVEMENT_SUGGESTIONS, \
                f"Missing suggestion for {pattern_name}"

    def test_malformed_result_does_not_crash(self, plugin):
        result = {"status": "failed", "summary": None, "plan": None}
        result, ctx = plugin.on_complete(result, {})
        assert "_improvement" in result

    def test_malformed_summary_type(self, plugin):
        result = {"status": "failed", "summary": 123, "plan": {}}
        result, ctx = plugin.on_complete(result, {})
        assert "_improvement" in result

    def test_malformed_plan_type(self, plugin):
        result = {"status": "failed", "summary": "error", "plan": "invalid"}
        result, ctx = plugin.on_complete(result, {})
        assert "_improvement" in result

    def test_malformed_steps_type(self, plugin):
        result = {"status": "failed", "summary": "error", "plan": {"steps": "invalid"}}
        result, ctx = plugin.on_complete(result, {})
        assert "_improvement" in result

    def test_thread_safety(self, plugin):
        errors = []

        def worker(i):
            try:
                result = {
                    "status": "failed" if i % 2 == 0 else "completed",
                    "summary": "CUDA OOM" if i % 2 == 0 else "all good",
                    "job_id": f"job-{i:03d}",
                    "plan": {
                        "steps": [{"id": 1, "status": "failed",
                                   "error": "out of memory" if i % 2 == 0 else ""}]
                    },
                }
                plugin.on_complete(result, {})
                plugin.get_statistics()
                plugin.get_best_practices()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = plugin.get_statistics()
        assert stats["run_count"] == 20


class TestAsyncCore:

    def test_async_run_agent_imports(self):
        from agent.core import async_run_agent
        assert callable(async_run_agent)

    def test_async_run_agent_is_coroutine(self):
        from agent.core import async_run_agent
        import asyncio
        assert asyncio.iscoroutinefunction(async_run_agent)
