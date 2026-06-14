"""Tests for cli.py — all CLI commands."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, MagicMock


class TestCLICommands:
    def test_cmd_init_creates_dirs(self, tmp_path, monkeypatch):
        from cli import cmd_init
        from config.settings import settings
        monkeypatch.setattr(settings, "data_dir", str(tmp_path))
        with patch("cli.init_db") as mock_init:
            cmd_init()
            assert os.path.isdir(tmp_path)
            mock_init.assert_called_once()

    def test_cmd_run_calls_agent(self):
        from cli import cmd_run
        with patch("cli.run_agent") as mock_agent:
            mock_agent.return_value = {"status": "completed"}
            from unittest.mock import patch as p
            with p("cli.console.print"):
                cmd_run("test goal")
            mock_agent.assert_called_once_with("test goal", model=None, dataset=None, method=None, executor="auto", browser_path=None)

    def test_cmd_list_jobs_empty(self, monkeypatch):
        from cli import cmd_list_jobs
        mock_store = MagicMock()
        mock_store.list.return_value = []
        monkeypatch.setattr("cli.JobStore", lambda: mock_store)
        with patch("cli.console.print") as mock_print:
            cmd_list_jobs()
            # Should print "No jobs found"
            calls = [c[0][0] for c in mock_print.call_args_list]
            assert any("No jobs found" in str(c) for c in calls)

    def test_cmd_list_jobs_with_data(self, monkeypatch):
        from cli import cmd_list_jobs
        from storage.jobs import Job
        mock_job = MagicMock(spec=Job)
        mock_job.id = "abc12345"
        mock_job.goal = "Fine-tune model"
        mock_job.method = "qlora"
        mock_job.status = "completed"
        mock_job.created_at = MagicMock()
        mock_job.created_at.strftime.return_value = "2024-01-01 12:00"
        mock_store = MagicMock()
        mock_store.list.return_value = [mock_job]
        monkeypatch.setattr("cli.JobStore", lambda: mock_store)
        with patch("cli.console.print") as mock_print:
            cmd_list_jobs()
            assert mock_print.call_count >= 1

    def test_cmd_list_convs_empty(self, monkeypatch):
        from cli import cmd_list_convs
        mock_conv = MagicMock()
        mock_conv.id = "conv123"
        mock_conv.goal = "test conversation"
        mock_conv.status = "active"
        mock_conv.get_messages.return_value = [{"role": "user", "content": "hi"}]
        mock_conv.updated_at = MagicMock()
        mock_conv.updated_at.strftime.return_value = "2024-01-01"
        mock_store = MagicMock()
        mock_store.list_all.return_value = [mock_conv]
        monkeypatch.setattr("cli.ConversationStore", lambda: mock_store)
        with patch("cli.console.print"):
            cmd_list_convs()

    def test_cmd_show_job_found(self, monkeypatch):
        from cli import cmd_show_job
        mock_job = MagicMock()
        mock_job.id = "abc123"
        mock_job.goal = "test"
        mock_job.status = "completed"
        mock_job.method = "lora"
        mock_job.base_model = "phi-2"
        mock_job.dataset = "dolly"
        mock_job.runtime = "T4"
        mock_job.created_at = "2024-01-01"
        mock_job.updated_at = "2024-01-02"
        mock_job.error = None
        mock_job.metrics = None
        mock_job.get_metrics.return_value = {}
        mock_store = MagicMock()
        mock_store.get.return_value = mock_job
        monkeypatch.setattr("cli.JobStore", lambda: mock_store)
        with patch("cli.console.print"):
            cmd_show_job("abc123")

    def test_cmd_show_job_not_found(self, monkeypatch):
        from cli import cmd_show_job
        mock_store = MagicMock()
        mock_store.get.return_value = None
        monkeypatch.setattr("cli.JobStore", lambda: mock_store)
        with patch("cli.console.print") as mock_print:
            cmd_show_job("nonexistent")
            calls = [c[0][0] for c in mock_print.call_args_list]
            assert any("not found" in str(c) for c in calls)

    def test_cmd_show_conv_found(self, monkeypatch):
        from cli import cmd_show_conv
        mock_conv = MagicMock()
        mock_conv.id = "conv123"
        mock_conv.goal = "test"
        mock_conv.status = "active"
        mock_conv.get_messages.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        mock_store = MagicMock()
        mock_store.get.return_value = mock_conv
        monkeypatch.setattr("cli.ConversationStore", lambda: mock_store)
        with patch("cli.console.print"):
            cmd_show_conv("conv123")

    def test_cmd_show_conv_not_found(self, monkeypatch):
        from cli import cmd_show_conv
        mock_store = MagicMock()
        mock_store.get.return_value = None
        monkeypatch.setattr("cli.ConversationStore", lambda: mock_store)
        with patch("cli.console.print") as mock_print:
            cmd_show_conv("nonexistent")
            calls = [c[0][0] for c in mock_print.call_args_list]
            assert any("not found" in str(c) for c in calls)

    def test_cmd_list_models_empty(self, monkeypatch):
        from cli import cmd_list_models
        mock_store = MagicMock()
        mock_store.list_all.return_value = []
        monkeypatch.setattr("cli.ModelVersionStore", lambda: mock_store)
        with patch("cli.console.print"):
            cmd_list_models()

    def test_cmd_config(self, monkeypatch):
        from cli import cmd_config
        with patch("cli.console.print"):
            cmd_config()

    def test_cmd_colab_code(self):
        from cli import cmd_colab_code
        with patch("cli.console.print"):
            code = cmd_colab_code()
            assert "MODEL_NAME" in code
            assert "TRAINING" in code
            assert "trainer.train" in code

    def test_main_help(self):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "--help"]):
            try:
                with patch("cli.console.print"):
                    main()
            except SystemExit:
                pass

    def test_main_init(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "init"]):
            with patch("cli.cmd_init") as mock:
                main()
                mock.assert_called_once()

    def test_main_run(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "run", "test goal"]):
            with patch("cli.cmd_run") as mock:
                main()
                mock.assert_called_once_with("test goal", model=None, dataset=None, method=None, executor="auto", browser_path=None)

    def test_main_interactive(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "interactive"]):
            with patch("cli.cmd_interactive") as mock:
                main()
                mock.assert_called_once()

    def test_main_jobs(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "jobs"]):
            with patch("cli.cmd_list_jobs") as mock:
                main()
                mock.assert_called_once()

    def test_main_convs(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "convs"]):
            with patch("cli.cmd_list_convs") as mock:
                main()
                mock.assert_called_once()

    def test_main_models(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "models"]):
            with patch("cli.cmd_list_models") as mock:
                main()
                mock.assert_called_once()

    def test_main_job(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "job", "abc123"]):
            with patch("cli.cmd_show_job") as mock:
                main()
                mock.assert_called_once_with("abc123")

    def test_main_conv(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "conv", "conv123"]):
            with patch("cli.cmd_show_conv") as mock:
                main()
                mock.assert_called_once_with("conv123")

    def test_main_config(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "config"]):
            with patch("cli.cmd_config") as mock:
                main()
                mock.assert_called_once()

    def test_main_colab(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "colab"]):
            with patch("cli.cmd_colab_code") as mock:
                main()
                mock.assert_called_once()

    def test_main_unknown_command(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "badcommand"]):
            with patch("cli.console.print") as mock:
                main()
                calls = [c[0][0] for c in mock.call_args_list]
                assert any("Unknown command" in str(c) for c in calls)

    def test_main_no_args_defaults_interactive(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py"]):
            with patch("cli.cmd_interactive") as mock:
                main()
                mock.assert_called_once()

    def test_main_serve(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "serve"]):
            with patch("cli.cmd_serve") as mock:
                main()
                mock.assert_called_once()

    def test_cmd_serve_imports(self):
        from cli import cmd_serve
        assert callable(cmd_serve)

    # ── Backup / Restore ────────────────────────────────────────

    def test_cmd_backup_writes_json(self, monkeypatch, tmp_path):
        from cli import cmd_backup
        from storage.database import reset_session, init_db
        from storage.jobs import JobStore
        from storage.conversations import ConversationStore
        reset_session()
        init_db()

        # Seed some data
        js = JobStore()
        js.create(goal="backup test", method="qlora")
        cs = ConversationStore()
        cs.create(goal="backup conv")

        backup_path = str(tmp_path / "test_backup.json")
        with patch("cli.console.print"):
            cmd_backup(backup_path)

        assert os.path.exists(backup_path)
        import json
        with open(backup_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["data"]["jobs"]) >= 1
        assert len(data["data"]["conversations"]) >= 1

    def test_cmd_restore_from_backup(self, monkeypatch, tmp_path):
        from cli import cmd_backup, cmd_restore
        from storage.database import reset_session, init_db
        from storage.jobs import JobStore
        reset_session()
        init_db()

        js = JobStore()
        js.create(goal="original", method="lora")
        backup_path = str(tmp_path / "restore_test.json")
        with patch("cli.console.print"):
            cmd_backup(backup_path)

        # Wipe and restore
        reset_session()
        init_db()
        with patch("cli.console.print"):
            cmd_restore(backup_path)

        js2 = JobStore()
        jobs = js2.list()
        assert any(j.goal == "original" for j in jobs)

    def test_cmd_restore_missing_file(self, monkeypatch):
        from cli import cmd_restore
        with patch("cli.console.print") as mock:
            cmd_restore("/nonexistent/backup.json")
            calls = [c[0][0] for c in mock.call_args_list]
            assert any("not found" in str(c) for c in calls)

    def test_main_backup(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "backup"]):
            with patch("cli.cmd_backup") as mock:
                main()
                mock.assert_called_once_with(None)

    def test_main_restore(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "restore", "backup.json"]):
            with patch("cli.cmd_restore") as mock:
                main()
                mock.assert_called_once_with("backup.json")

    def test_main_backup_sqlite(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "backup-sqlite"]):
            with patch("cli.cmd_backup_sqlite") as mock:
                main()
                mock.assert_called_once_with(None)

    def test_main_migrate_default(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "migrate"]):
            with patch("cli.cmd_migrate") as mock:
                main()
                mock.assert_called_once_with("head")

    def test_main_migrate_revision(self, monkeypatch):
        from cli import main
        with patch.object(sys, "argv", ["cli.py", "migrate", "0001"]):
            with patch("cli.cmd_migrate") as mock:
                main()
                mock.assert_called_once_with("0001")

    def test_cmd_migrate_applies_upgrade(self, tmp_path):
        from cli import cmd_migrate
        from config.settings import settings
        db_path = str(tmp_path / "test.db")
        with patch.object(settings, "db_url", f"sqlite:///{db_path}"):
            import sqlite3
            cmd_migrate()
            conn = sqlite3.connect(db_path)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            conn.close()
            assert "jobs" in tables
            assert "conversations" in tables
            assert "model_versions" in tables
            assert "runtime_logs" in tables
            assert "metrics" in tables
            assert "alembic_version" in tables

    def test_cmd_migrate_idempotent(self, tmp_path):
        from cli import cmd_migrate
        from config.settings import settings
        db_path = str(tmp_path / "test2.db")
        with patch.object(settings, "db_url", f"sqlite:///{db_path}"):
            cmd_migrate()
            cmd_migrate()
            import sqlite3
            conn = sqlite3.connect(db_path)
            version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
            conn.close()
            assert version in ("0001", "0002")
