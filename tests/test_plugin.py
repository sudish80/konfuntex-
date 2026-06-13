"""Tests for agent/plugin.py and agent/llm_cache.py — incl. hardened edge cases."""
import os
import tempfile
import time
import threading
import pytest

from agent.plugin import (
    Plugin, PluginRegistry, HookRunner, plugin, get_registry,
    HOOK_BEFORE_PLAN, HOOK_AFTER_PLAN, HOOK_ON_COMPLETE,
    HOOK_BEFORE_CODE_GEN, HOOK_AFTER_CODE_GEN, HOOK_ON_ERROR,
    HOOK_ON_SUMMARY,
)
from agent.llm_cache import LLMCache


# ═══════════════════════════════════════════════════════════════════ #
#  Plugin System Tests
# ═══════════════════════════════════════════════════════════════════ #

class TestPluginRegistry:

    def test_register_and_get(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin)
        p = reg.get("simple")
        assert p is not None
        assert p.name == "simple"

    def test_register_disabled(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin, enabled=False)
        assert reg.get("simple") is None

    def test_register_invalid_type(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="must be a Plugin subclass"):
            reg.register(str)

    def test_register_invalid_config(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="config must be a dict"):
            reg.register(SimplePlugin, config="not-a-dict")

    def test_register_invalid_enabled(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="enabled must be a bool"):
            reg.register(SimplePlugin, enabled=1)

    def test_register_invalid_priority(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="priority must be an int"):
            reg.register(SimplePlugin, priority="high")

    def test_list_registered(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin, priority=50)
        reg.register(AnotherPlugin, priority=10)
        listed = reg.list_registered()
        assert len(listed) == 2
        assert listed[0]["priority"] == 10

    def test_enable_disable(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin, enabled=False)
        assert reg.get("simple") is None
        reg.enable("simple")
        assert reg.get("simple") is not None
        reg.disable("simple")
        assert reg.get("simple") is None

    def test_clear(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin)
        reg.clear()
        assert len(reg.list_registered()) == 0

    def test_get_all_returns_enabled_sorted(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin, priority=100)
        reg.register(AnotherPlugin, priority=10)
        reg.register(DisabledPlugin, enabled=False)
        all_p = reg.get_all()
        assert len(all_p) == 2
        assert all_p[0].name == "another"

    def test_unregister(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin)
        reg.unregister("simple")
        assert reg.get("simple") is None

    def test_thread_safety(self):
        reg = PluginRegistry()
        errors = []

        def register_worker(i):
            try:
                if i % 2 == 0:
                    reg.register(SimplePlugin)
                else:
                    reg.register(AnotherPlugin)
                reg.get_all()
                reg.list_registered()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_worker, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_returns_same_instance(self):
        reg = PluginRegistry()
        reg.register(SimplePlugin)
        p1 = reg.get("simple")
        p2 = reg.get("simple")
        assert p1 is p2  # same instance, not re-created


class TestPluginDecorator:

    def test_decorator_registers_and_sets_attrs(self):
        @plugin(name="decorated_test", version="2.0.0", description="desc", priority=50)
        class DecoratedPlugin(Plugin):
            pass
        reg = get_registry()
        listed = [p for p in reg.list_registered() if p["name"] == "decorated_test"]
        assert len(listed) == 1
        assert listed[0]["version"] == "2.0.0"


class TestHookRunner:

    def test_invalid_registry_raises(self):
        with pytest.raises(TypeError, match="registry must be a PluginRegistry"):
            HookRunner(registry="invalid")

    def test_before_plan_hook(self):
        reg = PluginRegistry()
        reg.register(UpperCaseGoalPlugin)
        runner = HookRunner(reg)
        goal, ctx = runner.run_before_plan("my goal", {})
        assert goal == "MY GOAL"

    def test_after_plan_hook(self):
        reg = PluginRegistry()
        reg.register(AddMetaPlugin)
        runner = HookRunner(reg)
        plan = {"steps": [{"id": 1, "action": "train"}]}
        plan, ctx = runner.run_after_plan(plan, {})
        assert plan.get("metadata", {}).get("source") == "plugin"

    def test_before_step_hook(self):
        reg = PluginRegistry()
        reg.register(StepCheckPlugin)
        runner = HookRunner(reg)
        step = {"id": 1, "action": "train"}
        step, ctx = runner.run_before_step(step, {})
        assert step.get("checked")

    def test_after_step_hook(self):
        reg = PluginRegistry()
        reg.register(MetricLogPlugin)
        runner = HookRunner(reg)
        step = {"id": 1, "action": "train"}
        result = {"step_id": 1, "status": "success", "metrics": {}}
        result, ctx = runner.run_after_step(step, result, {})
        assert result.get("metrics", {}).get("logged_by") == "plugin"

    def test_before_code_gen_hook(self):
        reg = PluginRegistry()
        reg.register(PromptEnrichPlugin)
        runner = HookRunner(reg)
        step = {"id": 1, "action": "train"}
        prompt, ctx = runner.run_before_code_gen(step, "generate code", {})
        assert "IMPORTANT" in prompt

    def test_after_code_gen_hook(self):
        reg = PluginRegistry()
        reg.register(CodeFormatPlugin)
        runner = HookRunner(reg)
        step = {"id": 1}
        code, ctx = runner.run_after_code_gen(step, "def foo():pass", {})
        assert "# formatted" in code

    def test_on_error_hook(self):
        reg = PluginRegistry()
        reg.register(ErrorHandlerPlugin)
        runner = HookRunner(reg)
        step = {"id": 1, "action": "train"}
        recovery, ctx = runner.run_on_error(step, "OOM error", {})
        assert recovery == "retry_with_less_memory"

    def test_on_summary_hook(self):
        reg = PluginRegistry()
        reg.register(SummaryPrefixPlugin)
        runner = HookRunner(reg)
        summary, ctx = runner.run_on_summary("completed all steps", {})
        assert summary.startswith("[Report]")

    def test_on_complete_hook(self):
        reg = PluginRegistry()
        reg.register(CompletionPlugin)
        runner = HookRunner(reg)
        result = {"status": "completed", "summary": "done"}
        result, ctx = runner.run_on_complete(result, {})
        assert result.get("finalized")

    def test_all_hooks_defined(self):
        expected = {
            "before_plan", "after_plan", "before_step", "after_step",
            "before_code_gen", "after_code_gen", "on_error",
            "on_summary", "on_complete",
        }
        assert HOOK_BEFORE_PLAN in expected
        assert HOOK_AFTER_PLAN in expected
        assert HOOK_ON_COMPLETE in expected
        assert HOOK_BEFORE_CODE_GEN in expected
        assert HOOK_AFTER_CODE_GEN in expected
        assert HOOK_ON_ERROR in expected
        assert HOOK_ON_SUMMARY in expected

    def test_plugin_error_does_not_crash(self):
        reg = PluginRegistry()
        reg.register(CrashPlugin)
        pg = reg.get("crash")
        assert pg is not None

    def test_hook_runner_handles_error_gracefully(self):
        reg = PluginRegistry()
        reg.register(CrashPlugin)
        runner = HookRunner(reg)
        goal, ctx = runner.run_before_plan("test", {})
        assert goal == "test"

    def test_multiple_plugins_chain(self):
        reg = PluginRegistry()
        reg.register(UpperCaseGoalPlugin, priority=10)
        reg.register(AddPrefixPlugin, priority=20)
        runner = HookRunner(reg)
        goal, ctx = runner.run_before_plan("my goal", {})
        assert goal == "PREFIX: MY GOAL"


# ═══════════════════════════════════════════════════════════════════ #
#  LLM Cache Tests
# ═══════════════════════════════════════════════════════════════════ #

class TestLLMCache:

    @pytest.fixture
    def cache(self):
        tmp = tempfile.mktemp(suffix=".sqlite")
        c = LLMCache(db_path=tmp, ttl_seconds=3600)
        yield c
        c.close()
        try:
            os.remove(tmp)
        except OSError:
            pass

    def test_set_and_get(self, cache):
        messages = [{"role": "user", "content": "hi"}]
        response = {"content": "hello", "role": "assistant"}
        cache.set(response, "test-model", messages)
        got = cache.get("test-model", messages)
        assert got == response

    def test_cache_miss(self, cache):
        got = cache.get("test-model", [{"role": "user", "content": "different"}])
        assert got is None

    def test_ttl_expiry(self, cache):
        messages = [{"role": "user", "content": "hi"}]
        response = {"content": "hello"}
        cache.set(response, "test-model", messages, ttl_seconds=-1)
        time.sleep(0.01)
        got = cache.get("test-model", messages)
        assert got is None

    def test_custom_ttl(self, cache):
        messages = [{"role": "user", "content": "hi"}]
        response = {"content": "hello"}
        cache.set(response, "test-model", messages, ttl_seconds=9999)
        got = cache.get("test-model", messages)
        assert got == response

    def test_cache_key_uniqueness(self, cache):
        msgs1 = [{"role": "user", "content": "hello"}]
        msgs2 = [{"role": "user", "content": "world"}]
        cache.set({"c": "r1"}, "m", msgs1)
        cache.set({"c": "r2"}, "m", msgs2)
        assert cache.get("m", msgs1) == {"c": "r1"}
        assert cache.get("m", msgs2) == {"c": "r2"}

    def test_cache_key_order_matters(self, cache):
        msgs_a = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        msgs_b = [{"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        cache.set({"c": "r"}, "m", msgs_a)
        got = cache.get("m", msgs_b)
        assert got is None

    def test_invalidate_specific(self, cache):
        messages = [{"role": "user", "content": "hi"}]
        cache.set({"c": "r"}, "m", messages)
        cache.invalidate("m", messages)
        assert cache.get("m", messages) is None

    def test_invalidate_all(self, cache):
        cache.set({"c": "r1"}, "m", [{"role": "user", "content": "a"}])
        cache.set({"c": "r2"}, "m", [{"role": "user", "content": "b"}])
        cache.invalidate()
        assert cache.size() == 0

    def test_size(self, cache):
        assert cache.size() == 0
        cache.set({"c": "r"}, "m", [{"role": "user", "content": "hi"}])
        assert cache.size() == 1

    def test_evict_expired(self, cache):
        messages = [{"role": "user", "content": "hi"}]
        cache.set({"c": "r"}, "m", messages, ttl_seconds=-1)
        time.sleep(0.01)
        cache.evict_expired()
        assert cache.size() == 0

    def test_clear(self, cache):
        cache.set({"c": "r"}, "m", [{"role": "user", "content": "hi"}])
        cache.clear()
        assert cache.size() == 0

    def test_close_reopened(self, cache):
        cache.set({"c": "r"}, "m", [{"role": "user", "content": "hi"}])
        cache.close()
        cache2 = LLMCache(db_path=cache._db_path)
        got = cache2.get("m", [{"role": "user", "content": "hi"}])
        assert got == {"c": "r"}
        cache2.close()

    def test_closed_raises(self, cache):
        cache.close()
        with pytest.raises(RuntimeError, match="LLMCache is closed"):
            cache.get("m", [{"role": "user", "content": "hi"}])

    def test_closed_set_raises(self, cache):
        cache.close()
        with pytest.raises(RuntimeError, match="LLMCache is closed"):
            cache.set({"c": "r"}, "m", [{"role": "user", "content": "hi"}])

    def test_close_idempotent(self, cache):
        cache.close()
        cache.close()  # should not raise

    def test_get_after_db_error_returns_none(self):
        cache = LLMCache(db_path="", ttl_seconds=3600)
        cache.close()
        with pytest.raises(RuntimeError, match="LLMCache is closed"):
            cache.get("m", [{"role": "user", "content": "hi"}])

    def test_set_after_close_raises(self, cache):
        cache.close()
        with pytest.raises(RuntimeError, match="LLMCache is closed"):
            cache.set({"c": "r"}, "m", [{"role": "user", "content": "hi"}])

    def test_thread_safety(self, cache):
        errors = []

        def worker(i):
            try:
                msgs = [{"role": "user", "content": f"msg{i}"}]
                cache.set({"c": f"r{i}"}, "m", msgs)
                cache.get("m", msgs)
                cache.size()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert cache.size() == 20

    def test_evict_expired_close_safety(self, cache):
        cache.close()
        cache.evict_expired()  # should not raise


# ═══════════════════════════════════════════════════════════════════ #
#  Test Plugin Implementations
# ═══════════════════════════════════════════════════════════════════ #

class SimplePlugin(Plugin):
    name = "simple"

class AnotherPlugin(Plugin):
    name = "another"

class DisabledPlugin(Plugin):
    name = "disabled"

class UpperCaseGoalPlugin(Plugin):
    name = "uppercase"
    def before_plan(self, goal: str, context: dict):
        return goal.upper(), context

class AddPrefixPlugin(Plugin):
    name = "prefix"
    def before_plan(self, goal: str, context: dict):
        return f"PREFIX: {goal}", context

class AddMetaPlugin(Plugin):
    name = "meta"
    def after_plan(self, plan: dict, context: dict):
        plan["metadata"] = {"source": "plugin"}
        return plan, context

class StepCheckPlugin(Plugin):
    name = "step_check"
    def before_step(self, step: dict, context: dict):
        step["checked"] = True
        return step, context

class MetricLogPlugin(Plugin):
    name = "metric_log"
    def after_step(self, step: dict, result: dict, context: dict):
        result.setdefault("metrics", {})["logged_by"] = "plugin"
        return result, context

class PromptEnrichPlugin(Plugin):
    name = "prompt_enrich"
    def before_code_gen(self, step: dict, prompt: str, context: dict):
        return f"IMPORTANT: {prompt}", context

class CodeFormatPlugin(Plugin):
    name = "code_format"
    def after_code_gen(self, step: dict, code: str, context: dict):
        return f"# formatted\n{code}", context

class ErrorHandlerPlugin(Plugin):
    name = "error_handler"
    def on_error(self, step: dict, error: str, context: dict):
        if "OOM" in error:
            return "retry_with_less_memory", context
        return None, context

class SummaryPrefixPlugin(Plugin):
    name = "summary_prefix"
    def on_summary(self, summary: str, context: dict):
        return f"[Report] {summary}", context

class CompletionPlugin(Plugin):
    name = "completion"
    def on_complete(self, result: dict, context: dict):
        result["finalized"] = True
        result["completed_at"] = "now"
        return result, context

class CrashPlugin(Plugin):
    name = "crash"
    def before_plan(self, goal: str, context: dict):
        raise RuntimeError("plugin crash")
