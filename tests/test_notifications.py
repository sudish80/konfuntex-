"""Tests for notification plugins (Slack, email, webhook)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch


class TestSlackNotifier:
    def test_noop_when_no_webhook(self):
        from agent.notifications import SlackNotifier
        p = SlackNotifier()
        assert p._webhook_url == ""
        result = {"status": "completed"}
        r, ctx = p.on_complete(result, {})
        assert r == result
        r, ctx = p.on_error({"step": 1}, "err", {})
        assert r is None

    def test_sends_on_complete(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        from agent.notifications import SlackNotifier
        p = SlackNotifier()
        sent = []
        def fake_send(text):
            sent.append(text)
        p._send = fake_send
        result = {"status": "completed", "summary": "All good", "method": "qlora", "base_model": "phi-2"}
        p.on_complete(result, {})
        assert len(sent) == 1
        assert "Job completed" in sent[0]
        assert "All good" in sent[0]

    def test_sends_on_error(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        from agent.notifications import SlackNotifier
        p = SlackNotifier()
        sent = []
        def fake_send(text):
            sent.append(text)
        p._send = fake_send
        p.on_error({"step": 1}, "GPU OOM error", {"goal": "train model"})
        assert len(sent) == 1
        assert "Job failed" in sent[0]
        assert "GPU OOM error" in sent[0]

    def test_send_http_success(self):
        from agent.notifications import _send_http
        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            _send_http("https://example.com", {"key": "val"})
            mock_post.assert_called_once_with("https://example.com", json={"key": "val"}, timeout=10)

    def test_send_http_empty_url(self):
        from agent.notifications import _send_http
        _send_http("", {"key": "val"})

    def test_send_http_retries_on_failure(self):
        from agent.notifications import _send_http
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("connection error")
            _send_http("https://example.com", {"key": "val"})

    def test_auto_registered(self):
        from agent.plugin import get_registry
        reg = get_registry()
        plugins = reg.list_registered()
        names = [p["name"] for p in plugins]
        assert "slack_notifier" in names


class TestEmailNotifier:
    def test_noop_when_no_config(self):
        from agent.notifications import EmailNotifier
        p = EmailNotifier()
        assert p._smtp_host == ""
        result = {"status": "completed"}
        r, ctx = p.on_complete(result, {})
        assert r == result

    def test_sends_on_complete(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_EMAIL", "me@test.com")
        monkeypatch.setenv("COLAB_AGENT_SMTP_USER", "user")
        monkeypatch.setenv("COLAB_AGENT_SMTP_PASSWORD", "pass")
        from agent.notifications import EmailNotifier
        p = EmailNotifier()
        sent = []
        def fake_send(subject, body):
            sent.append((subject, body))
        p._send_email = fake_send
        result = {"status": "completed", "summary": "Done", "method": "lora", "base_model": "gpt2"}
        p.on_complete(result, {})
        assert len(sent) == 1
        assert sent[0][0] == "Job completed"

    def test_sends_on_error(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_EMAIL", "me@test.com")
        from agent.notifications import EmailNotifier
        p = EmailNotifier()
        sent = []
        def fake_send(subject, body):
            sent.append((subject, body))
        p._send_email = fake_send
        p.on_error({"step": 1}, "OOM", {"goal": "test"})
        assert len(sent) == 1
        assert "Job failed" in sent[0][0]

    def test_send_email_smtp(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_EMAIL", "me@test.com")
        from agent.notifications import EmailNotifier
        p = EmailNotifier()
        with patch("smtplib.SMTP") as mock_smtp:
            p._send_email("Subj", "Body")
            mock_smtp.assert_called_once()

    def test_auto_registered(self):
        from agent.plugin import get_registry
        reg = get_registry()
        names = [p["name"] for p in reg.list_registered()]
        assert "email_notifier" in names


class TestWebhookNotifier:
    def test_noop_when_no_url(self):
        from agent.notifications import WebhookNotifier
        p = WebhookNotifier()
        assert p._webhook_url == ""
        result = {"status": "completed"}
        r, ctx = p.on_complete(result, {})
        assert r == result

    def test_sends_on_complete(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_WEBHOOK_URL", "https://hook.test/ev")
        from agent.notifications import WebhookNotifier
        p = WebhookNotifier()
        sent = []
        original = p._build_payload
        def tracking_build(event, result=None, step=None, error=None, context=None):
            sent.append(event)
            return original(event, result=result, step=step, error=error, context=context)
        p._build_payload = tracking_build
        with patch("agent.notifications._send_http") as mock_send:
            p.on_complete({"status": "ok"}, {"goal": "test"})
            assert len(sent) == 1
            assert sent[0] == "job.completed"
            mock_send.assert_called_once()

    def test_sends_on_error(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_WEBHOOK_URL", "https://hook.test/ev")
        from agent.notifications import WebhookNotifier
        p = WebhookNotifier()
        sent = []
        original = p._build_payload
        def tracking_build(event, result=None, step=None, error=None, context=None):
            sent.append(event)
            return original(event, result=result, step=step, error=error, context=context)
        p._build_payload = tracking_build
        with patch("agent.notifications._send_http") as mock_send:
            p.on_error({"step": 1}, "err", {"goal": "test"})
            assert len(sent) == 1
            assert sent[0] == "job.error"
            mock_send.assert_called_once()

    def test_payload_structure(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_WEBHOOK_URL", "https://hook.test/ev")
        from agent.notifications import WebhookNotifier
        p = WebhookNotifier()
        payload = p._build_payload("test.event", result={"status": "ok"}, context={"goal": "x"})
        assert payload["event"] == "test.event"
        assert payload["source"] == "colab-agent"
        assert payload["result"] == {"status": "ok"}
        assert "llm_client" not in payload.get("context", {})

    def test_context_llm_client_removed(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_NOTIFICATION_WEBHOOK_URL", "https://hook.test/ev")
        from agent.notifications import WebhookNotifier
        p = WebhookNotifier()
        payload = p._build_payload("ev", context={"llm_client": "secret", "goal": "x"})
        assert "llm_client" not in payload.get("context", {})

    def test_auto_registered(self):
        from agent.plugin import get_registry
        reg = get_registry()
        names = [p["name"] for p in reg.list_registered()]
        assert "webhook_notifier" in names


class TestFormatJobResult:
    def test_format_includes_status(self):
        from agent.notifications import _format_job_result
        s = _format_job_result({"status": "completed"})
        assert "Status: completed" in s

    def test_format_with_all_fields(self):
        from agent.notifications import _format_job_result
        s = _format_job_result({"status": "completed", "summary": "Training done", "method": "qlora", "base_model": "phi-2"})
        assert "Training done" in s
        assert "qlora" in s
        assert "phi-2" in s
