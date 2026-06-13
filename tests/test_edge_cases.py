"""Edge case and coverage tests — focusing on known-working APIs."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSettings:
    def test_default_provider(self):
        from config.settings import Settings
        s = Settings()
        assert s.llm_provider in ("openai", "anthropic", "gemini", "local")

    def test_get_db_url_default(self):
        from config.settings import Settings
        s = Settings()
        url = s.get_db_url()
        assert url.startswith("sqlite")

    def test_runtime_tiers_has_t4(self):
        from config.settings import Settings
        s = Settings()
        assert "T4" in s.runtime_tiers

    def test_runtime_tiers_vram_values(self):
        from config.settings import Settings
        s = Settings()
        assert s.runtime_tiers["T4"] == 16
        assert s.runtime_tiers["V100"] == 32

    def test_default_finetune_method(self):
        from config.settings import Settings
        s = Settings()
        assert s.default_finetune_method in ("lora", "qlora", "full")

    def test_env_prefix(self):
        from config.settings import Settings
        assert Settings.model_config["env_prefix"] == "COLAB_AGENT_"

    def test_settings_singleton(self):
        from config.settings import settings
        assert hasattr(settings, "llm_provider")

    def test_data_dir(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_DATA_DIR", raising=False)
        import importlib
        import config.settings
        importlib.reload(config.settings)
        from config.settings import settings
        assert settings.data_dir is not None
        assert "colab-agent" in settings.data_dir


class TestMemoryStoreEdgeCases:
    def test_max_turns_enforced(self):
        from agent.memory import MemoryStore
        ms = MemoryStore(max_turns=2, max_tokens=1000)
        for i in range(5):
            ms.add("user", f"msg{i}")
        assert len(ms.history) <= 2

    def test_add_and_get(self):
        from agent.memory import MemoryStore
        ms = MemoryStore()
        ms.add("user", "hello")
        ms.add("assistant", "world")
        assert len(ms.history) == 2

    def test_clear(self):
        from agent.memory import MemoryStore
        ms = MemoryStore()
        ms.add("user", "test")
        ms.clear()
        assert ms.history == []

    def test_sliding_window(self):
        from agent.memory import MemoryStore
        ms = MemoryStore(max_turns=3)
        for i in range(5):
            ms.add("user", f"msg{i}")
        assert len(ms.history) == 3


class TestAgentStateMachine:
    def test_initial_state(self):
        from agent.memory import AgentStateMachine, AgentState
        asm = AgentStateMachine()
        assert asm.state == AgentState.IDLE

    def test_transition_to_planning(self):
        from agent.memory import AgentStateMachine, AgentState
        asm = AgentStateMachine()
        asm.transition(AgentState.PLANNING)
        assert asm.state == AgentState.PLANNING


class TestImmutableModeEdgeCases:
    def test_context_manager(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode(enabled=False)
        with im:
            assert im.active is True
        assert im.active is False

    def test_guard_or_raise(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode(enabled=True)
        import pytest
        with pytest.raises(PermissionError):
            im.guard_or_raise("write")

    def test_disable(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode(enabled=True)
        im.disable()
        assert im.active is False


class TestEmergencyStopEdgeCases:
    def test_not_triggered_initially(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        assert es.is_triggered is False

    def test_trigger_sets_reason(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger("test reason")
        state = es.get_state()
        assert state["reason"] == "test reason"

    def test_check_raises_after_trigger(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger()
        import pytest
        with pytest.raises(RuntimeError, match="Emergency stop"):
            es.check()

    def test_reset_clears(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger()
        es.reset()
        assert es.is_triggered is False


class TestTimeBombEdgeCases:
    def test_not_expired_initially(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        tb.start()
        assert tb.expired is False

    def test_elapsed_increases(self):
        from agent.extended_safety import TimeBomb
        import time
        tb = TimeBomb(max_hours=1)
        tb.start()
        time.sleep(0.01)
        assert tb.elapsed > 0

    def test_remaining(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        tb.start()
        assert tb.remaining > 0
        assert tb.remaining <= 3600

    def test_fraction_used(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        assert tb.fraction_used() == 0.0
        tb.start()
        assert 0 <= tb.fraction_used() <= 1.0

    def test_no_start(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        assert tb.elapsed == 0.0
        assert tb.expired is False


class TestAutoResumeManagerEdgeCases:
    def test_init(self):
        from agent.advanced_autonomy import AutoResumeManager
        arm = AutoResumeManager()
        assert hasattr(arm, "save_state")

    def test_generate_resume_code(self):
        from agent.advanced_autonomy import AutoResumeManager
        arm = AutoResumeManager()
        code = arm.generate_resume_code("test-job")
        assert code is not None
        assert isinstance(code, str)


class TestMultiGoalPlannerEdgeCases:
    def test_init(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        assert hasattr(mgp, "add_goal")

    def test_add_and_list(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        mgp.add_goal("goal1", priority=1)
        mgp.add_goal("goal2", priority=2)
        goals = mgp.list_goals()
        assert len(goals) == 2

    def test_priorities(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        mgp.add_goal("low", priority=1)
        mgp.add_goal("high", priority=10)
        goals = mgp.list_goals()
        assert len(goals) == 2


class TestCostTrackerEdgeCases:
    def test_missing_job_summary(self):
        from agent.safety import CostTracker
        ct = CostTracker()
        summary = ct.get_summary("nonexistent")
        assert summary["job_id"] == "nonexistent"
        assert summary["gpu_hours"] == 0

    def test_start_job_and_record(self):
        from agent.safety import CostTracker
        ct = CostTracker()
        ct.start_job("test_job", "T4")
        ct.record_execution("test_job", 3600)
        summary = ct.get_summary("test_job")
        assert summary["gpu_hours"] == 1.0

    def test_multiple_switches(self):
        from agent.safety import CostTracker
        ct = CostTracker()
        ct.start_job("j1")
        ct.record_switch("j1", "V100")
        ct.record_switch("j1", "A100")
        summary = ct.get_summary("j1")
        assert summary["runtime_switches"] == 2

    def test_cost_units_t4(self):
        from agent.safety import CostTracker, COST_PER_GPU_HOUR
        ct = CostTracker()
        ct.start_job("j", "T4")
        ct.record_execution("j", 3600)
        s = ct.get_summary("j")
        assert s["estimated_cost_units"] == 1.0 * COST_PER_GPU_HOUR["T4"]


class TestJobSealer:
    def test_seal_verify(self):
        from agent.extended_safety import JobSealer
        js = JobSealer(secret="test-secret")
        config = {"method": "qlora", "model": "phi-2"}
        sig = js.seal(config)
        assert len(sig) == 16
        assert js.verify(config, sig) is True

    def test_verify_tampered(self):
        from agent.extended_safety import JobSealer
        js = JobSealer(secret="test-secret")
        config = {"method": "qlora", "model": "phi-2"}
        sig = js.seal(config)
        assert js.verify({"method": "full", "model": "phi-2"}, sig) is False

    def test_different_secrets_produce_different_sigs(self):
        from agent.extended_safety import JobSealer
        js1 = JobSealer(secret="secret1")
        js2 = JobSealer(secret="secret2")
        config = {"method": "qlora"}
        assert js1.seal(config) != js2.seal(config)


class TestRateLimiter:
    def test_consume_sufficient(self):
        from agent.extended_safety import RateLimiter
        rl = RateLimiter(calls_per_minute=1000, tokens_per_minute=100000)
        assert rl.consume(tokens=100) is True

    def test_consume_excessive_tokens(self):
        from agent.extended_safety import RateLimiter
        rl = RateLimiter(calls_per_minute=1000, tokens_per_minute=100)
        assert rl.consume(tokens=1000) is False


class TestTelemetrySink:
    def test_emit_and_read(self):
        from agent.extended_safety import TelemetrySink
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ts = TelemetrySink(path=path)
            ts.emit("test_event", {"key": "value"})
            records = ts.read_all()
            assert len(records) == 1
            assert records[0]["event"] == "test_event"
        finally:
            os.unlink(path)

    def test_emit_event_and_error(self):
        from agent.extended_safety import TelemetrySink
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ts = TelemetrySink(path=path)
            ts.emit_event("custom", {"data": 1})
            ts.emit_error("test_error", "detail")
            assert len(ts.read_all()) == 2
        finally:
            os.unlink(path)

    def test_empty_read_all(self):
        from agent.extended_safety import TelemetrySink
        ts = TelemetrySink(path=os.devnull + ".jsonl")
        assert ts.read_all() == []


class TestIntegrityCheckerEdgeCases:
    def test_not_found_in_verify_all(self):
        from agent.extended_safety import IntegrityChecker
        ic = IntegrityChecker(state_file=os.devnull)
        ic._cache["/nonexistent"] = "abc123"
        results = ic.verify_all()
        assert len(results) == 1
        assert results[0]["ok"] is False


class TestSafetyConstants:
    def test_all_constants_defined(self):
        from agent.safety import (MAX_RUNTIME_SWITCHES, MAX_ERROR_RETRIES_PER_STEP,
                                  MAX_TOTAL_ERROR_RETRIES, MAX_COLAB_HOURS,
                                  MAX_VRAM_GB)
        assert MAX_RUNTIME_SWITCHES == 5
        assert MAX_ERROR_RETRIES_PER_STEP == 10
        assert MAX_TOTAL_ERROR_RETRIES == 50
        assert MAX_COLAB_HOURS == 12
        assert MAX_VRAM_GB == 80

    def test_cost_per_gpu_hour(self):
        from agent.safety import COST_PER_GPU_HOUR
        assert "T4" in COST_PER_GPU_HOUR
        assert "A100-80GB" in COST_PER_GPU_HOUR
        assert "TPU" in COST_PER_GPU_HOUR

    def test_allowed_packages_count(self):
        from agent.safety import ALLOWED_PIP_PACKAGES
        assert len(ALLOWED_PIP_PACKAGES) > 30

    def test_dangerous_patterns_count(self):
        from agent.safety import DANGEROUS_PATTERNS
        assert len(DANGEROUS_PATTERNS) >= 10


class TestPromptEdgeCases:
    def test_planning_prompt_has_goal_placeholder(self):
        from agent.prompts import PLANNING_PROMPT
        assert "{goal}" in PLANNING_PROMPT

    def test_summary_prompt_has_placeholders(self):
        from agent.prompts import SUMMARY_PROMPT
        for ph in ["{goal}", "{results}", "{final_status}"]:
            assert ph in SUMMARY_PROMPT

    def test_code_gen_has_placeholders(self):
        from agent.prompts import CODE_GENERATION_PROMPT
        for ph in ["{goal}", "{step_description}"]:
            assert ph in CODE_GENERATION_PROMPT

    def test_error_analysis_prompt_has_content(self):
        from agent.prompts import ERROR_ANALYSIS_PROMPT
        assert len(ERROR_ANALYSIS_PROMPT) > 100


class TestLLMClientEdgeCases:
    def test_singleton(self):
        from agent.llm_client import LLMClient
        client = LLMClient()
        assert client.provider is not None

    def test_extract_json_invalid(self):
        from agent.llm_client import LLMClient
        client = LLMClient()
        result = client.extract_json_from_response("")
        assert result is None

    def test_extract_json_codeblock(self):
        from agent.llm_client import LLMClient
        client = LLMClient()
        result = client.extract_json_from_response(
            '```json\n{"key": "value"}\n```'
        )
        assert result is not None
        assert result["key"] == "value"

    def test_extract_json_brace(self):
        from agent.llm_client import LLMClient
        client = LLMClient()
        result = client.extract_json_from_response(
            'Some text {"a": 1} trailing'
        )
        assert result is not None
        assert result["a"] == 1
