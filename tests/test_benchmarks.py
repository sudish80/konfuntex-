"""Performance benchmarks for LLM cache, plugin dispatch, and retention.

Run with: pytest tests/test_benchmarks.py --benchmark-only -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLLMCacheBenchmarks:
    """Measure LLM cache read/write throughput."""

    def test_cache_write_throughput(self, benchmark):
        from agent.llm_cache import LLMCache
        import tempfile
        cache = LLMCache()
        cache._db_path = tempfile.mktemp(suffix=".db")
        cache._init_db()
        resp = {"choices": [{"text": "x" * 500}]}
        msgs = [{"role": "user", "content": "benchmark-key-" * 10}]

        def write_roundtrip():
            cache.set(resp, model="gpt-4", messages=msgs)
            cache.get(model="gpt-4", messages=msgs)

        benchmark(write_roundtrip)

    def test_cache_read_throughput(self, benchmark):
        from agent.llm_cache import LLMCache
        import tempfile
        cache = LLMCache()
        cache._db_path = tempfile.mktemp(suffix=".db")
        cache._init_db()
        all_msgs = [[{"role": "user", "content": f"k-{i}-{'x' * 50}"}] for i in range(100)]
        for msgs in all_msgs:
            cache.set({"text": "x" * 200}, model="gpt-4", messages=msgs)

        def read_all():
            for msgs in all_msgs:
                cache.get(model="gpt-4", messages=msgs)

        benchmark(read_all)


class TestPluginDispatchBenchmarks:
    """Measure HookRunner dispatch overhead."""

    def test_hook_runner_overhead(self, benchmark):
        from agent.plugin import PluginRegistry, HookRunner, Plugin

        class NoopPlugin(Plugin):
            name = "noop"
            def before_step(self, step, context):
                return step, context

        reg = PluginRegistry()
        for i in range(10):
            cls = type(f"NoopPlugin{i}", (NoopPlugin,), {"name": f"noop{i}"})
            reg.register(cls)

        runner = HookRunner(reg)
        step = {"id": 1}
        context = {"job_id": "j1"}

        def dispatch():
            runner.run_before_step(step, context)

        benchmark(dispatch)

    def test_hook_runner_20_plugins(self, benchmark):
        from agent.plugin import PluginRegistry, HookRunner, Plugin

        reg = PluginRegistry()
        for i in range(20):
            cls = type(f"P{i}", (Plugin,), {
                "name": f"p{i}",
                "before_step": lambda self, s, c: (s, c),
            })
            reg.register(cls)

        runner = HookRunner(reg)
        benchmark(lambda: runner.run_before_step({"id": 1}, {}))


class TestRetentionBenchmarks:
    """Measure retention cleanup speed."""

    def test_cleanup_1000_jobs(self, benchmark):
        from agent.retention import clean_old_jobs
        from storage.database import reset_session
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from storage.database import Base, Job
        from datetime import datetime, timezone, timedelta
        import uuid

        reset_session()
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        sess = Session()
        for i in range(1000):
            sess.add(Job(
                id=str(uuid.uuid4()),
                goal=f"bench goal {i}",
                created_at=datetime.now(timezone.utc) - timedelta(days=400),
            ))
        sess.commit()
        sess.close()

        def cleanup():
            clean_old_jobs(days=365)

        benchmark(cleanup)
