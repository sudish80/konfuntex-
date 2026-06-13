"""Notification plugins for Slack, email, and generic webhook.

All three plugins auto-register via ``@plugin()`` decorator and are
no-ops when their required config is missing.
"""

import logging
import smtplib
import os
import threading
from email.mime.text import MIMEText
from typing import Optional

import requests

from agent.plugin import Plugin, plugin

logger = logging.getLogger(__name__)


def _read_env(key: str, default: str = "") -> str:
    return os.environ.get(f"COLAB_AGENT_{key}", default)


def _format_job_result(result: dict) -> str:
    status = result.get("status", "unknown")
    summary = result.get("summary", "")
    method = result.get("method", "?")
    model = result.get("base_model", "?")
    parts = [f"Status: {status}"]
    if summary:
        parts.append(f"Summary: {summary[:200]}")
    if method:
        parts.append(f"Method: {method}")
    if model:
        parts.append(f"Model: {model}")
    return "\n".join(parts)


def _send_http(url: str, payload: dict, timeout: int = 10) -> None:
    if not url:
        return
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        logger.info("HTTP notification sent to %s (status %s)", url, resp.status_code)
    except Exception as e:
        logger.warning("HTTP notification to %s failed: %s", url, e)


@plugin(
    name="slack_notifier",
    version="1.0.0",
    description="Sends job completion/error notifications to Slack via webhook",
    priority=80,
)
class SlackNotifier(Plugin):
    def __init__(self):
        super().__init__()
        self._webhook_url: str = _read_env("SLACK_WEBHOOK_URL")
        if self._webhook_url:
            logger.info("SlackNotifier configured with webhook URL")
        else:
            logger.debug("SlackNotifier disabled: COLAB_AGENT_SLACK_WEBHOOK_URL not set")

    def _send(self, text: str) -> None:
        if not self._webhook_url:
            return
        payload = {"text": text, "mrkdwn": True}
        _send_http(self._webhook_url, payload)

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        if not self._webhook_url:
            return result, context
        text = f"✅ Job completed\n{_format_job_result(result)}"
        threading.Thread(target=self._send, args=(text,), daemon=True).start()
        return result, context

    def on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        if not self._webhook_url:
            return None, context
        goal = context.get("goal", "?")
        text = f"❌ Job failed\nGoal: {goal}\nError: {error[:300]}"
        threading.Thread(target=self._send, args=(text,), daemon=True).start()
        return None, context


@plugin(
    name="email_notifier",
    version="1.0.0",
    description="Sends job completion/error notifications via SMTP email",
    priority=85,
)
class EmailNotifier(Plugin):
    def __init__(self):
        super().__init__()
        self._smtp_host: str = _read_env("SMTP_HOST")
        self._smtp_port: int = int(_read_env("SMTP_PORT", "587"))
        self._smtp_user: str = _read_env("SMTP_USER")
        self._smtp_password: str = _read_env("SMTP_PASSWORD")
        self._from: str = _read_env("SMTP_FROM")
        self._to: str = _read_env("NOTIFICATION_EMAIL")
        if self._smtp_host and self._to:
            logger.info("EmailNotifier configured: %s -> %s", self._from or self._smtp_user, self._to)
        else:
            logger.debug("EmailNotifier disabled: SMTP_HOST or NOTIFICATION_EMAIL not set")

    def _send_email(self, subject: str, body: str) -> None:
        if not self._smtp_host or not self._to:
            return
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = self._from or self._smtp_user or "noreply@colab-agent"
        msg["To"] = self._to
        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=15) as server:
                if self._smtp_user and self._smtp_password:
                    server.starttls()
                    server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)
            logger.info("Email sent to %s", self._to)
        except smtplib.SMTPException as e:
            logger.warning("Email notification to %s failed: %s", self._to, e)

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        if not self._smtp_host or not self._to:
            return result, context
        body = _format_job_result(result)
        threading.Thread(target=self._send_email, args=("Job completed", body), daemon=True).start()
        return result, context

    def on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        if not self._smtp_host or not self._to:
            return None, context
        goal = context.get("goal", "?")
        body = f"Goal: {goal}\n\nError:\n{error[:500]}"
        threading.Thread(target=self._send_email, args=("Job failed", body), daemon=True).start()
        return None, context


@plugin(
    name="webhook_notifier",
    version="1.0.0",
    description="Sends job completion/error notifications to a generic HTTP webhook",
    priority=90,
)
class WebhookNotifier(Plugin):
    def __init__(self):
        super().__init__()
        self._webhook_url: str = _read_env("NOTIFICATION_WEBHOOK_URL")
        if self._webhook_url:
            logger.info("WebhookNotifier configured: %s", self._webhook_url)
        else:
            logger.debug("WebhookNotifier disabled: COLAB_AGENT_NOTIFICATION_WEBHOOK_URL not set")

    def _build_payload(self, event: str, result: dict = None,
                       step: dict = None, error: str = None,
                       context: dict = None) -> dict:
        payload = {
            "event": event,
            "source": "colab-agent",
        }
        if result is not None:
            payload["result"] = result
        if step is not None:
            payload["step"] = step
        if error is not None:
            payload["error"] = error
        if context is not None:
            payload["context"] = {k: v for k, v in context.items()
                                  if k != "llm_client"}
        return payload

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        if not self._webhook_url:
            return result, context
        payload = self._build_payload("job.completed", result=result, context=context)
        threading.Thread(target=_send_http, args=(self._webhook_url, payload), daemon=True).start()
        return result, context

    def on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        if not self._webhook_url:
            return None, context
        payload = self._build_payload("job.error", step=step, error=error, context=context)
        threading.Thread(target=_send_http, args=(self._webhook_url, payload), daemon=True).start()
        return None, context
