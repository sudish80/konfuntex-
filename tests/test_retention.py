"""Tests for agent/retention.py — data retention policy."""
import os
import tempfile
import time
from unittest.mock import patch

from agent.retention import (
    get_retention_days, run_retention_policy, clean_old_logs,
)


class TestRetentionConfig:

    def test_default_retention_days(self):
        assert get_retention_days("jobs") == 90
        assert get_retention_days("logs") == 30
        assert get_retention_days("llm_cache") == 7
        assert get_retention_days("audit_log") == 365

    def test_custom_env_var(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_RETENTION_JOBS_DAYS", "30")
        assert get_retention_days("jobs") == 30

    def test_unknown_category(self):
        assert get_retention_days("unknown") == 90


class TestCleanOldLogs:

    def test_cleans_old_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = os.path.join(tmpdir, "old.jsonl")
            new_path = os.path.join(tmpdir, "new.jsonl")
            with open(old_path, "w") as f:
                f.write("old\n")
            with open(new_path, "w") as f:
                f.write("new\n")
            old_mtime = time.time() - 100 * 86400
            os.utime(old_path, (old_mtime, old_mtime))

            with patch("agent.retention.settings") as mock_settings:
                mock_settings.data_dir = tmpdir
                count = clean_old_logs(days=30)
                assert count == 1
                assert not os.path.exists(old_path)
                assert os.path.exists(new_path)

    def test_no_logs_dir(self, monkeypatch):
        nonexistent = os.path.join(tempfile.gettempdir(), "nonexistent_" + str(time.time()))
        with patch("agent.retention.settings") as mock_settings:
            mock_settings.data_dir = nonexistent
            count = clean_old_logs(days=30)
            assert count == 0

    def test_no_files_to_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.retention.settings") as mock_settings:
                mock_settings.data_dir = tmpdir
                count = clean_old_logs(days=30)
                assert count == 0


class TestRetentionCleanup:

    def test_run_retention_policy_does_not_crash(self):
        with patch("agent.retention.clean_old_jobs") as mock_jobs:
            with patch("agent.retention.clean_old_conversations") as mock_convs:
                with patch("agent.retention.clean_old_metrics") as mock_metrics:
                    with patch("agent.retention.clean_old_logs") as mock_logs:
                        run_retention_policy()
                        mock_jobs.assert_called_once()
                        mock_convs.assert_called_once()
                        mock_metrics.assert_called_once()
                        mock_logs.assert_called_once()
