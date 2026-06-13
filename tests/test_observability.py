import json
from agent.observability import JSONFormatter, MetricsCollector, setup_json_logging
import logging


class TestMetricsCollector:
    def test_counter(self):
        mc = MetricsCollector()
        mc.counter("requests", 1)
        assert mc.get_counter("requests") == 1.0

    def test_counter_increment(self):
        mc = MetricsCollector()
        mc.counter("req", 2)
        mc.counter("req", 3)
        assert mc.get_counter("req") == 5.0

    def test_gauge(self):
        mc = MetricsCollector()
        mc.gauge("memory", 512.0)
        assert mc.get_gauge("memory") == 512.0

    def test_gauge_overwrite(self):
        mc = MetricsCollector()
        mc.gauge("mem", 100)
        mc.gauge("mem", 200)
        assert mc.get_gauge("mem") == 200

    def test_snapshot(self):
        mc = MetricsCollector()
        mc.counter("a", 1)
        mc.gauge("b", 2.0)
        s = mc.snapshot()
        assert s["counters"]["a"] == 1.0
        assert s["gauges"]["b"] == 2.0

    def test_prometheus_text_format(self):
        mc = MetricsCollector()
        mc.counter("test_requests_total", 3)
        mc.gauge("test_memory_bytes", 1024)
        text = mc.prometheus_text()
        assert "# TYPE test_requests_total counter" in text
        assert "test_requests_total 3" in text
        assert "# TYPE test_memory_bytes gauge" in text
        assert "test_memory_bytes 1024" in text

    def test_empty_prometheus(self):
        mc = MetricsCollector()
        assert mc.prometheus_text() == "\n"

    def test_metrics_endpoint_live(self):
        from agent.core import metrics_endpoint
        text = metrics_endpoint()
        assert isinstance(text, str)


class TestJSONFormatter:
    def test_formatter_output(self):
        logger = logging.getLogger("test_json")
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        import io
        buf = io.StringIO()
        handler.setStream(buf)
        logger.info("hello world")
        out = buf.getvalue()
        parsed = json.loads(out)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_setup_json_logging(self):
        logger = logging.getLogger("test_setup")
        setup_json_logging(logger)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_setup_json_logging_with_path(self):
        import tempfile, os
        logger = logging.getLogger("test_path")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            setup_json_logging(logger, log_path=path)
            logger.info("file test")
            logger.handlers.clear()
            with open(path) as fh:
                line = fh.read()
            assert "file test" in line
        finally:
            try:
                os.unlink(path)
            except PermissionError:
                pass
