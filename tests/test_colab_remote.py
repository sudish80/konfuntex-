"""
Tests for RemoteColabExecutor + ColabRunner fallback chain.

Covers:
  - Constructor and property checks
  - Session path, availability detection
  - Fallback when Playwright unavailable
  - Fallback chain in ColabRunner (remote -> local -> simulate)
  - Thread safety (concurrent access)
  - status(), static methods, edge cases
  - Context manager
  - All ColabRunner executor modes
"""

import os
import re
import base64
import threading
import tempfile
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
from pathlib import Path

import colab.executor
from colab.remote_executor import RemoteColabExecutor


# ------------------------------------------------------------------ #
#  Tests: Constructor, properties, static methods
# ------------------------------------------------------------------ #

class TestRemoteColabExecutorInit(unittest.TestCase):

    def test_init_defaults(self):
        e = RemoteColabExecutor()
        self.assertTrue(e.headless)
        self.assertTrue(e.session_path.endswith(".colab-session"))

    def test_init_custom_headless(self):
        e = RemoteColabExecutor(headless=False)
        self.assertFalse(e.headless)

    def test_init_custom_session_path(self):
        with tempfile.TemporaryDirectory() as td:
            e = RemoteColabExecutor(user_data_dir=td)
            self.assertEqual(e.session_path, td)

    def test_available_false_when_no_playwright(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            e = RemoteColabExecutor()
            self.assertFalse(e.available)

    def test_available_true_when_playwright_installed(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=True):
            e = RemoteColabExecutor()
            self.assertTrue(e.available)

    def test_not_connected_after_init(self):
        e = RemoteColabExecutor()
        self.assertFalse(e.connected)

    def test_status_report(self):
        e = RemoteColabExecutor()
        s = e.status()
        self.assertIn("available", s)
        self.assertIn("connected", s)
        self.assertIn("headless", s)
        self.assertIn("session_path", s)
        self.assertIn("has_session", s)


class TestRemoteColabExecutorConnect(unittest.TestCase):
    """Connect without Playwright (falls through immediately)."""

    def test_connect_unavailable(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            e = RemoteColabExecutor()
            result = e.connect()
            self.assertFalse(result["success"])
            self.assertIn("Playwright not available", result["error"])

    def test_disconnect_idempotent(self):
        e = RemoteColabExecutor()
        e.disconnect()
        e.disconnect()
        self.assertFalse(e.connected)


class TestRemoteColabExecutorUtility(unittest.TestCase):
    """Static methods and utility functions."""

    def test_capture_screenshot_not_connected(self):
        e = RemoteColabExecutor()
        self.assertIsNone(e.capture_screenshot())

    def test_get_runtime_info_not_connected(self):
        e = RemoteColabExecutor()
        info = e.get_runtime_info()
        self.assertIsNone(info["gpu"])

    def test_static_has_error(self):
        self.assertTrue(RemoteColabExecutor._has_error("Traceback (most recent call last)"))
        self.assertTrue(RemoteColabExecutor._has_error("Error: something"))
        self.assertTrue(RemoteColabExecutor._has_error("Exception: fail"))
        self.assertTrue(RemoteColabExecutor._has_error("CUDA out of memory"))
        self.assertFalse(RemoteColabExecutor._has_error("success"))
        self.assertFalse(RemoteColabExecutor._has_error(""))

    def test_static_extract_error(self):
        output = "line1\nTraceback\nline3\nError: x\nline5"
        err = RemoteColabExecutor._extract_error(output)
        self.assertIsNotNone(err)
        self.assertIn("Traceback", err)
        self.assertIn("Error:", err)

    def test_static_extract_error_none(self):
        self.assertIsNone(RemoteColabExecutor._extract_error("all good"))
        self.assertIsNone(RemoteColabExecutor._extract_error(""))

    def test_static_extract_error_truncated(self):
        long_err = "\n".join([f"Error: line {i}" for i in range(100)])
        err = RemoteColabExecutor._extract_error(long_err)
        self.assertIsNotNone(err)

    def test_session_subdir_created(self):
        with tempfile.TemporaryDirectory() as td:
            path = RemoteColabExecutor.ensure_session_dir(td)
            self.assertTrue(os.path.isdir(path))

    def test_session_subdir_already_exists(self):
        with tempfile.TemporaryDirectory() as td:
            path = RemoteColabExecutor.ensure_session_dir(td)
            path2 = RemoteColabExecutor.ensure_session_dir(td)
            self.assertEqual(path, path2)


class TestRemoteColabExecutorLogin(unittest.TestCase):

    def test_login_fails_when_unavailable(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            e = RemoteColabExecutor()
            self.assertFalse(e.login_once())

    @patch("builtins.input", return_value="")
    @patch.object(RemoteColabExecutor, "_check_playwright", return_value=True)
    def test_login_creates_session_dir(self, _check, _input):
        with patch("playwright.sync_api.sync_playwright") as mock_pw:
            pw = MagicMock()
            browser = MagicMock()
            pw.chromium.launch_persistent_context.return_value = browser
            mock_pw.return_value.__enter__.return_value = pw
            with tempfile.TemporaryDirectory() as td:
                e = RemoteColabExecutor(user_data_dir=td)
                result = e.login_once()
                self.assertTrue(result)


class TestRemoteColabExecutorContextManager(unittest.TestCase):

    def test_context_manager_disconnects_on_exit(self):
        e = RemoteColabExecutor()
        with e as ex:
            self.assertIs(ex, e)
        self.assertFalse(e.connected)


class TestRemoteColabExecutorThreadSafety(unittest.TestCase):

    def test_concurrent_disconnect(self):
        e = RemoteColabExecutor()
        errors = []

        def disconnect_wrapper():
            try:
                e.disconnect()
            except Exception as ex:
                errors.append(str(ex))

        threads = [threading.Thread(target=disconnect_wrapper) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_status(self):
        e = RemoteColabExecutor()
        errors = []

        def status_wrapper():
            try:
                for _ in range(5):
                    e.status()
            except Exception as ex:
                errors.append(str(ex))

        threads = [threading.Thread(target=status_wrapper) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_mixed_operations(self):
        e = RemoteColabExecutor()
        errors = []

        def mixed(idx):
            try:
                if idx % 3 == 0:
                    e.status()
                elif idx % 3 == 1:
                    e.disconnect()
                else:
                    e.capture_screenshot()
            except Exception as ex:
                errors.append(str(ex))

        threads = [threading.Thread(target=mixed, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


# ------------------------------------------------------------------ #
#  ColabRunner integration tests (no Playwright needed)
# ------------------------------------------------------------------ #

class TestColabRunnerRemote(unittest.TestCase):
    """ColabRunner with executor='remote' -- lifecycle and fallback."""

    def test_init_remote_sets_simulate_false(self):
        r = colab.executor.ColabRunner(executor="remote")
        self.assertEqual(r._mode, "remote")
        self.assertFalse(r.simulate)

    def test_connect_remote_fallback_to_local(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            with patch("colab.executor.LocalIPythonRunner") as mock_local:
                mock_instance = MagicMock()
                mock_local.return_value = mock_instance
                r = colab.executor.ColabRunner(executor="remote")
                result = r.connect()
                self.assertTrue(result["success"])
                self.assertEqual(result["mode"], "local")

    def test_connect_remote_fallback_to_simulate(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            with patch("colab.executor.LocalIPythonRunner",
                       side_effect=Exception("no local jupyter")):
                r = colab.executor.ColabRunner(executor="remote")
                result = r.connect()
                self.assertTrue(result["success"])
                self.assertEqual(result["mode"], "colab")

    def test_execute_remote_fallback_on_failure(self):
        """Remote executor connect fails -> falls back to local kernel."""
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            with patch("colab.executor.LocalIPythonRunner") as mock_local:
                mock_instance = MagicMock()
                mock_instance.execute.return_value = {
                    "success": True, "output": "local result", "error": None,
                }
                mock_local.return_value = mock_instance
                r = colab.executor.ColabRunner(executor="remote")
                r.connect()
                result = r.execute_cell("x=1")
                self.assertTrue(result["success"])
                self.assertEqual(r._mode, "local")

    def test_execute_cell_with_history(self):
        with patch("colab.executor.LocalIPythonRunner") as mock_local:
            mock_instance = MagicMock()
            mock_instance.execute.return_value = {
                "success": True, "output": "ok", "error": None,
            }
            mock_local.return_value = mock_instance
            r = colab.executor.ColabRunner(executor="local")
            r.execute_cell("x=1", description="test step")
            self.assertEqual(len(r.execution_history), 1)
            self.assertEqual(r.execution_history[0]["description"], "test step")

    def test_execute_cell_triggers_on_progress(self):
        callback = MagicMock()
        with patch("colab.executor.LocalIPythonRunner") as mock_local:
            mock_instance = MagicMock()
            mock_instance.execute.return_value = {
                "success": True, "output": "ok", "error": None,
            }
            mock_local.return_value = mock_instance
            r = colab.executor.ColabRunner(executor="local")
            r.execute_cell("x=1", on_progress=callback)
            callback.assert_called_once()
            self.assertEqual(callback.call_args[0][0]["status"], "completed")

    def test_context_manager_disconnects(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            with patch("colab.executor.LocalIPythonRunner") as mock_local:
                mock_instance = MagicMock()
                mock_local.return_value = mock_instance
                r = colab.executor.ColabRunner(executor="remote")
                with r:
                    self.assertTrue(r._connected)
                self.assertFalse(r._connected)

    def test_double_disconnect_safe(self):
        r = colab.executor.ColabRunner(executor="local")
        r.disconnect()
        r.disconnect()
        self.assertFalse(r._connected)


class TestColabRunnerFallbackMetrics(unittest.TestCase):

    def test_fallback_reason_set_on_fallback(self):
        with patch.object(RemoteColabExecutor, "_check_playwright",
                          return_value=False):
            with patch("colab.executor.LocalIPythonRunner") as mock_local:
                mock_instance = MagicMock()
                mock_local.return_value = mock_instance
                r = colab.executor.ColabRunner(executor="remote")
                r.connect()
                self.assertEqual(r._fallback_reason, "remote_unavailable")


class TestColabRunnerExecutorModes(unittest.TestCase):

    def test_auto_mode_defaults_to_simulate(self):
        with patch.dict(os.environ, {}, clear=True):
            r = colab.executor.ColabRunner(executor="auto")
            self.assertTrue(r.simulate)

    def test_auto_mode_env_var(self):
        with patch.dict(os.environ, {"COLAB_AGENT_SIMULATE": "false"},
                        clear=True):
            r = colab.executor.ColabRunner(executor="auto")
            self.assertFalse(r.simulate)

    def test_local_mode(self):
        r = colab.executor.ColabRunner(executor="local")
        self.assertEqual(r._mode, "local")
        self.assertFalse(r.simulate)

    def test_colab_mode(self):
        r = colab.executor.ColabRunner(executor="colab")
        self.assertEqual(r._mode, "colab")
        self.assertTrue(r.simulate)

    def test_remote_mode(self):
        r = colab.executor.ColabRunner(executor="remote")
        self.assertEqual(r._mode, "remote")
        self.assertFalse(r.simulate)

    def test_invalid_mode_defaults_to_simulate(self):
        r = colab.executor.ColabRunner(executor="invalid")
        self.assertTrue(r.simulate)

    def test_get_history_empty(self):
        r = colab.executor.ColabRunner()
        self.assertEqual(r.get_history(), [])

    def test_clear_history(self):
        with patch("colab.executor.LocalIPythonRunner") as mock_local:
            mock_instance = MagicMock()
            mock_instance.execute.return_value = {"success": True,
                                                   "output": "ok"}
            mock_local.return_value = mock_instance
            r = colab.executor.ColabRunner(executor="local")
            r.execute_cell("x=1")
            r.clear_history()
            self.assertEqual(r.get_history(), [])


class TestColabRunnerSimulate(unittest.TestCase):

    def test_simulate_valid_code(self):
        r = colab.executor.ColabRunner(executor="colab")
        result = r.execute_cell("x = 1 + 2")
        self.assertTrue(result["success"])
        self.assertIn("execution_time", result)

    def test_simulate_syntax_error(self):
        r = colab.executor.ColabRunner(executor="colab")
        result = r.execute_cell("x = ")
        self.assertFalse(result["success"])

    def test_simulate_oom_detection(self):
        r = colab.executor.ColabRunner(executor="colab")
        result = r.execute_cell("CUDA_OUT_OF_MEMORY")
        self.assertFalse(result["success"])
        self.assertIn("CUDA out of memory", result["error"])

    def test_simulate_training_code(self):
        r = colab.executor.ColabRunner(executor="colab")
        result = r.execute_cell("trainer.train()")
        self.assertTrue(result["success"])
        self.assertIn("TRAIN", result["output"])


class TestParseOutput(unittest.TestCase):

    def test_parse_output_oom(self):
        r = colab.executor.ColabRunner()
        result = r.parse_output({
            "output": "",
            "error": "CUDA out of memory. Tried to allocate 5.2 GiB",
        })
        self.assertEqual(result["error_type"], "runtime_oom")
        self.assertAlmostEqual(result["vram_needed_gb"], 5.2)

    def test_parse_output_syntax_error(self):
        r = colab.executor.ColabRunner()
        result = r.parse_output({
            "output": "",
            "error": "SyntaxError: invalid syntax",
        })
        self.assertEqual(result["error_type"], "syntax_error")

    def test_parse_output_import_error(self):
        r = colab.executor.ColabRunner()
        result = r.parse_output({
            "output": "",
            "error": "ModuleNotFoundError: No module named 'xyz'",
        })
        self.assertEqual(result["error_type"], "import_error")

    def test_parse_output_loss(self):
        r = colab.executor.ColabRunner()
        result = r.parse_output({
            "output": "loss=0.8234",
            "error": None,
        })
        self.assertEqual(result["error_type"], None)
        self.assertEqual(result["metrics"]["final_loss"], 0.8234)

    def test_parse_output_no_error(self):
        r = colab.executor.ColabRunner()
        result = r.parse_output({
            "output": "all good",
            "error": None,
        })
        self.assertIsNone(result["error_type"])


class TestRuntimeInfo(unittest.TestCase):

    def test_runtime_info_defaults(self):
        info = colab.executor.ColabRuntimeInfo()
        self.assertEqual(info.gpu_name, "")
        self.assertEqual(info.vram_total_gb, 0.0)

    def test_runtime_info_to_dict(self):
        info = colab.executor.ColabRuntimeInfo(
            gpu_name="Tesla T4", vram_total_gb=16.0)
        d = info.to_dict()
        self.assertEqual(d["gpu_name"], "Tesla T4")
        self.assertEqual(d["vram_total_gb"], 16.0)

    def test_runtime_info_repr(self):
        info = colab.executor.ColabRuntimeInfo(
            gpu_name="Tesla T4", vram_total_gb=16.0, ram_total_gb=12.5)
        r = repr(info)
        self.assertIn("T4", r)


if __name__ == "__main__":
    unittest.main()
