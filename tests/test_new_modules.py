"""Unit tests for newly added Phase 2-10 modules."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- Phase 2: agent/memory.py ---- #

class TestMemoryStore:
    def test_add_and_trim(self):
        from agent.memory import MemoryStore
        ms = MemoryStore(max_turns=3, max_tokens=500)
        for i in range(5):
            ms.add("user", f"message_{i}")
        context = ms.get_context()
        assert len(context) <= 3
        assert context[-1]["content"] == "message_4"

    def test_clear(self):
        from agent.memory import MemoryStore
        ms = MemoryStore()
        ms.add("user", "hello")
        ms.clear()
        assert len(ms.history) == 0


class TestGoalRefiner:
    def test_vague_short(self):
        from agent.memory import GoalRefiner
        gr = GoalRefiner()
        assert gr.is_vague("train a model")

    def test_not_vague(self):
        from agent.memory import GoalRefiner
        gr = GoalRefiner()
        assert not gr.is_vague("Fine-tune phi3 on dolly dataset for chat")


class TestAgentStateMachine:
    def test_valid_transition(self):
        from agent.memory import AgentState, AgentStateMachine
        sm = AgentStateMachine()
        assert sm.state == AgentState.IDLE
        assert sm.transition(AgentState.PLANNING)
        assert sm.state == AgentState.PLANNING

    def test_invalid_transition(self):
        from agent.memory import AgentState, AgentStateMachine
        sm = AgentStateMachine()
        assert not sm.transition(AgentState.COMPLETED)

    def test_reset(self):
        from agent.memory import AgentState, AgentStateMachine
        sm = AgentStateMachine()
        sm.transition(AgentState.PLANNING)
        sm.reset()
        assert sm.state == AgentState.IDLE


class TestCostEstimator:
    def test_llm_cost(self):
        from agent.memory import CostEstimator
        cost = CostEstimator.estimate_llm_cost("gpt-4", 1000, 500)
        assert cost > 0

    def test_colab_cost(self):
        from agent.memory import CostEstimator
        units = CostEstimator.estimate_colab_cost("T4", 2.0)
        assert units == 2.0


# ---- Phase 3: models/selector.py ---- #

class TestHFModelSelector:
    def test_recommend_returns_list(self):
        from models.selector import HFModelSelector
        results = HFModelSelector.recommend("text-generation", vram_gb=16)
        assert len(results) > 0
        assert "name" in results[0]
        assert "fits_on_current_runtime" in results[0]

    def test_best_fit(self):
        from models.selector import HFModelSelector
        model = HFModelSelector.best_fit("code-generation", vram_gb=80)
        assert model is not None
        assert isinstance(model, str)

    def test_selector_code(self):
        from models.selector import HFModelSelector
        code = HFModelSelector.selector_code("text-generation")
        assert "MODEL_NAME" in code


class TestDatasetLoader:
    def test_detect_format_hf(self):
        from models.selector import DatasetLoader
        assert DatasetLoader.detect_format("databricks/dolly") == "hf"

    def test_detect_format_csv(self):
        from models.selector import DatasetLoader
        assert DatasetLoader.detect_format("data.csv") == "csv"

    def test_load_code_hf(self):
        from models.selector import DatasetLoader
        code = DatasetLoader.load_code("databricks/dolly")
        assert "load_dataset" in code


class TestModelSizeEstimator:
    def test_estimate_vram(self):
        from models.selector import ModelSizeEstimator
        est = ModelSizeEstimator.estimate_vram(7.0, "qlora")
        assert est["model_params_b"] == 7.0
        assert est["vram_total_gb"] > 0


class TestTokenizerManager:
    def test_auto_config_code(self):
        from models.selector import TokenizerManager
        code = TokenizerManager.auto_config_code("microsoft/phi-2")
        assert "AutoTokenizer" in code


class TestCacheManager:
    def test_cache_size(self):
        from models.selector import CacheManager
        cm = CacheManager(cache_dir=tempfile.gettempdir())
        size = cm.cache_size()
        assert isinstance(size, str)
        assert "GB" in size


class TestQuantizationSelector:
    def test_4bit_for_small_vram(self):
        from models.selector import QuantizationSelector
        assert QuantizationSelector.select(6, 7.0) == "4bit"

    def test_none_for_large_vram(self):
        from models.selector import QuantizationSelector
        assert QuantizationSelector.select(80, 1.0) == "none"


class TestModelCardGenerator:
    def test_generate(self):
        from models.selector import ModelCardGenerator
        card = ModelCardGenerator.generate("phi-2", "dolly", "qlora")
        assert "phi-2" in card
        assert "qlora" in card
        assert "dolly" in card


# ---- Phase 4: models/training_engine.py ---- #

class TestHyperparameterTuner:
    def test_grid(self):
        from models.training_engine import HyperparameterTuner
        tuner = HyperparameterTuner()
        grid = {"lr": [1e-4, 2e-4], "bs": [4, 8]}
        combos = tuner.grid(grid)
        assert len(combos) == 4


class TestEarlyStoppingScheduler:
    def test_stops_on_plateau(self):
        from models.training_engine import EarlyStoppingScheduler
        es = EarlyStoppingScheduler(patience=2, min_delta=0.01)
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert es.step(1.0)

    def test_does_not_stop_on_improvement(self):
        from models.training_engine import EarlyStoppingScheduler
        es = EarlyStoppingScheduler(patience=3)
        assert not es.step(1.0)
        assert not es.step(0.5)
        assert not es.step(0.3)


class TestMetricsCallback:
    def test_log_and_final(self):
        from models.training_engine import MetricsCallback
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            cb = MetricsCallback(log_path=f.name)
        class Args: pass
        class State: pass
        args = Args(); state = State(); state.global_step = 10; state.epoch = 1.0
        cb.on_log(args, state, None, logs={"loss": 0.5, "learning_rate": 2e-4})
        fin = cb.get_final_metrics()
        assert fin["final_loss"] == 0.5


class TestBatchSizeTuner:
    def test_find_max_batch_code(self):
        from models.training_engine import BatchSizeTuner
        code = BatchSizeTuner.find_max_batch("microsoft/phi-2")
        assert "for batch in range" in code


class TestBestModelCheckpointer:
    def test_saves_on_improvement(self):
        from models.training_engine import BestModelCheckpointer
        with tempfile.TemporaryDirectory() as tmp:
            ck = BestModelCheckpointer(tmp)
            assert ck.step(1.0)
            assert ck.step(0.5)  # improvement (0.5 < 1.0 is better for min)
            assert ck.best_value == 0.5


class TestValidationSplitter:
    def test_split_code(self):
        from models.training_engine import ValidationSplitter
        code = ValidationSplitter.split_code("dataset")
        assert "train_test_split" in code
        assert "train_dataset" in code


class TestTrainingVisualizer:
    def test_plot_none_on_empty(self):
        from models.training_engine import TrainingVisualizer
        assert TrainingVisualizer.plot_loss_curve([]) is None


# ---- Phase 9: agent/extended_safety.py ---- #

class TestImmutableMode:
    def test_blocks_when_enabled(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode(enabled=True)
        assert not im.guard("write_file")

    def test_allows_when_disabled(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode(enabled=False)
        assert im.guard("write_file")

    def test_context_manager(self):
        from agent.extended_safety import ImmutableMode
        im = ImmutableMode()
        with im:
            assert im.active
        assert not im.active


class TestEmergencyStop:
    def test_default_not_triggered(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        assert not es.is_triggered

    def test_trigger(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger("test")
        assert es.is_triggered
        state = es.get_state()
        assert state["reason"] == "test"

    def test_reset(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger("test")
        es.reset()
        assert not es.is_triggered

    def test_check_raises(self):
        from agent.extended_safety import EmergencyStop
        es = EmergencyStop()
        es.trigger("stop")
        try:
            es.check()
            assert False, "Should have raised"
        except RuntimeError:
            pass


class TestRateLimiter:
    def test_consume_ok(self):
        from agent.extended_safety import RateLimiter
        rl = RateLimiter(calls_per_minute=1000, tokens_per_minute=100000)
        assert rl.consume(100)


class TestJobSealer:
    def test_seal_and_verify(self):
        from agent.extended_safety import JobSealer
        js = JobSealer(secret="test-secret")
        config = {"model": "phi-2", "method": "qlora"}
        sig = js.seal(config)
        assert js.verify(config, sig)
        assert not js.verify({"model": "other"}, sig)


class TestTimeBomb:
    def test_not_expired_initially(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        assert not tb.expired

    def test_elapsed(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        tb.start()
        e = tb.elapsed
        assert e >= 0

    def test_fraction_used(self):
        from agent.extended_safety import TimeBomb
        tb = TimeBomb(max_hours=1)
        tb.start()
        assert 0 <= tb.fraction_used() <= 1.0


class TestTelemetrySink:
    def test_emit_and_read(self):
        from agent.extended_safety import TelemetrySink
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        ts = TelemetrySink(path=path)
        ts.emit("test", {"key": "val"})
        records = ts.read_all()
        assert len(records) == 1
        assert records[0]["event"] == "test"
        os.unlink(path)

    def test_get_summary(self):
        from agent.extended_safety import TelemetrySink
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        ts = TelemetrySink(path=path)
        ts.emit("event_a")
        ts.emit("event_b", level="error")
        summary = ts.get_summary()
        assert summary["total_events"] == 2
        assert summary["errors"] == 1
        os.unlink(path)


# ---- Phase 10: agent/advanced_autonomy.py ---- #

class TestAutoResumeManager:
    def test_save_and_load(self):
        from agent.advanced_autonomy import AutoResumeManager
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            arm = AutoResumeManager(state_path=path)
            arm.save_state("job1", 3, 2, "T4", "/ckpt", {"base": "phi-2"})
            assert arm.has_state()
            state = arm.load_state()
            assert state["job_id"] == "job1"
            assert state["step_id"] == 3
            arm.clear_state()
            assert not arm.has_state()

    def test_generate_resume_code(self):
        from agent.advanced_autonomy import AutoResumeManager
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            arm = AutoResumeManager(state_path=path)
            arm.save_state("job1", 3, 2, "T4", "/ckpt")
            code = arm.generate_resume_code()
            assert "OrchestratorAgent" in code
            assert "job1" in code


class TestMultiGoalPlanner:
    def test_add_and_next(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        _ = mgp.add_goal("goal 1", priority=5)
        g2 = mgp.add_goal("goal 2", priority=10)
        first = mgp.next_goal()
        assert first is not None
        assert first.id == g2  # higher priority

    def test_dependencies(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        g1 = mgp.add_goal("setup")
        _ = mgp.add_goal("train", dependencies=[g1])
        first = mgp.next_goal()
        assert first.id == g1

    def test_summary(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        mgp.add_goal("test")
        s = mgp.get_summary()
        assert s["total"] == 1
        assert s["pending"] == 1

    def test_should_abort(self):
        from agent.advanced_autonomy import MultiGoalPlanner
        mgp = MultiGoalPlanner()
        mgp.max_consecutive_failures = 2
        g1 = mgp.add_goal("g1")
        g2 = mgp.add_goal("g2")
        g3 = mgp.add_goal("g3")
        mgp.mark_failed(g1)
        mgp.mark_failed(g2)
        mgp.mark_failed(g3)
        assert mgp.should_abort()


class TestMetaAgentOrchestrator:
    def test_submit_and_summary(self):
        from agent.advanced_autonomy import MetaAgentOrchestrator
        ma = MetaAgentOrchestrator()
        ma.submit_goal("test goal")
        s = ma.get_summary()
        assert s["goals"]["total"] == 1

    def test_abort(self):
        from agent.advanced_autonomy import MetaAgentOrchestrator
        ma = MetaAgentOrchestrator()
        ma.abort()
        assert ma._aborted
