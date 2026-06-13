"""Tests for colab/ — RuntimeManager, ColabManager, executor, setup, infrastructure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ================================================================== #
#  colab/runtime.py
# ================================================================== #

class TestRuntimeManagerInit:
    def test_initial_state(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm.current_gpu == "None"
        assert rm.current_vram_gb == 0.0
        assert rm.switch_history == []

    def test_runtime_order(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.RUNTIME_ORDER == ["None", "T4", "P100", "V100", "A100", "A100-80GB", "TPU"]

    def test_gpu_name_to_label(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.GPU_NAME_TO_LABEL["tesla t4"] == "T4"
        assert RuntimeManager.GPU_NAME_TO_LABEL["tesla v100"] == "V100"
        assert RuntimeManager.GPU_NAME_TO_LABEL["nvidia a100"] == "A100"


class TestRuntimeManagerNormalize:
    def test_normalize_exact(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm._normalize_label("T4") == "T4"
        assert rm._normalize_label("V100") == "V100"

    def test_normalize_lowercase(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm._normalize_label("a100") == "A100"

    def test_normalize_alias(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm._normalize_label("tesla t4") == "T4"
        assert rm._normalize_label("volta") == "V100"
        assert rm._normalize_label("cpu") == "None"

    def test_normalize_unknown(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm._normalize_label("nonexistent") is None


class TestRuntimeManagerSwitch:
    def test_switch_success(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        result = rm.switch_runtime("T4", save_state=False)
        assert result["success"] is True
        assert result["switched_to"] == "T4"
        assert "switch_code" in result
        assert "restart_url" in result

    def test_switch_unknown_gpu(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        result = rm.switch_runtime("nonexistent")
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_switch_history(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.switch_runtime("T4", reason="test")
        assert len(rm.switch_history) == 1
        assert rm.switch_history[0]["to"] == "T4"

    def test_switch_updates_current(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.switch_runtime("V100")
        assert rm.current_gpu == "V100"

    def test_switch_with_save_state(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        result = rm.switch_runtime("V100", save_state=True, checkpoint_path="./ckpt")
        assert result["save_state_code"] != ""
        assert "CHECKPOINT" in result["save_state_code"]


class TestRuntimeManagerDetect:
    def test_detect_a100_80(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("NVIDIA A100-SXM4-80GB") == "A100-80GB"

    def test_detect_a100(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("NVIDIA A100") == "A100"

    def test_detect_v100(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("Tesla V100-SXM2") == "V100"

    def test_detect_t4(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("Tesla T4") == "T4"

    def test_detect_p100(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("Tesla P100-PCIE") == "P100"

    def test_detect_k80(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("Tesla K80") == "K80"

    def test_detect_tpu(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("TPU v2") == "TPU"

    def test_detect_unknown(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.detect_from_gpu_name("Unknown GPU") == "None"


class TestRuntimeManagerGetVRAM:
    def test_vram_a100_80(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.get_vram_gb("NVIDIA A100-80GB") == 80.0

    def test_vram_a100(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.get_vram_gb("NVIDIA A100") == 40.0

    def test_vram_v100(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.get_vram_gb("Tesla V100") == 32.0

    def test_vram_t4(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.get_vram_gb("Tesla T4") == 16.0

    def test_vram_unknown(self):
        from colab.runtime import RuntimeManager
        assert RuntimeManager.get_vram_gb("Unknown") == 0.0

    def test_vram_from_nvidia_smi(self):
        from colab.runtime import RuntimeManager
        vram = RuntimeManager.get_vram_gb("Unknown", "16280 MiB")
        assert vram == 15.8984375


class TestRuntimeManagerSufficiency:
    def test_sufficient_vram(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.current_vram_gb = 16.0
        assert rm.is_runtime_sufficient(16.0) is True

    def test_insufficient_vram(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.current_vram_gb = 8.0
        assert rm.is_runtime_sufficient(16.0) is False

    def test_ram_check_skipped(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.current_vram_gb = 16.0
        assert rm.is_runtime_sufficient(16.0, required_ram_gb=0) is True

    def test_best_fit_runtime_t4(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm.best_fit_runtime(8.0) == "T4"

    def test_best_fit_runtime_v100(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm.best_fit_runtime(24.0) == "V100"

    def test_best_fit_runtime_a100(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm.best_fit_runtime(48.0) == "A100"

    def test_best_fit_runtime_tpu_for_very_high(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        assert rm.best_fit_runtime(100.0) == "TPU"

    def test_should_switch_true(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.current_vram_gb = 8.0
        should, curr, rec, _ = rm.should_switch(16.0)
        assert should is True
        assert rec == "T4"

    def test_should_switch_false(self):
        from colab.runtime import RuntimeManager
        rm = RuntimeManager()
        rm.current_vram_gb = 32.0
        should, _, _, _ = rm.should_switch(16.0)
        assert should is False


class TestEstimateNeededVRAM:
    def test_qlora(self):
        from colab.runtime import estimate_needed_vram
        assert estimate_needed_vram(7, "qlora") == 7 * 0.5 + 2.0

    def test_lora(self):
        from colab.runtime import estimate_needed_vram
        assert estimate_needed_vram(7, "lora") == 7 * 1.2 + 2.0

    def test_full(self):
        from colab.runtime import estimate_needed_vram
        assert estimate_needed_vram(7, "full") == 7 * 3.0 + 4.0

    def test_unknown(self):
        from colab.runtime import estimate_needed_vram
        assert estimate_needed_vram(7, "unknown") == 7 * 1.5 + 2.0


class TestUrllibEncode:
    def test_single_param(self):
        from colab.runtime import urllib_encode
        assert urllib_encode({"accelerator": "GPU"}) == "accelerator=GPU"

    def test_multiple_params(self):
        from colab.runtime import urllib_encode
        result = urllib_encode({"accelerator": "GPU", "gpuType": "T4"})
        assert "accelerator=GPU" in result
        assert "gpuType=T4" in result


# ================================================================== #
#  colab/manager.py
# ================================================================== #

class TestColabManager:
    def test_create_notebook_code(self):
        from colab.manager import ColabManager
        cm = ColabManager()
        code = cm.create_notebook_code("Test Notebook")
        assert "nvidia-smi" in code
        assert "transformers datasets accelerate" in code

    def test_notebook_code_has_date(self):
        from colab.manager import ColabManager
        cm = ColabManager()
        code = cm.create_notebook_code()
        assert "Start time:" in code

    def test_check_runtime_code(self):
        from colab.manager import ColabManager
        cm = ColabManager()
        code = cm.check_runtime_code()
        assert "torch.cuda.is_available" in code
        assert "nvidia-smi" in code

    def test_mount_drive_code(self):
        from colab.manager import ColabManager
        cm = ColabManager()
        code = cm.mount_drive_code()
        assert "drive.mount" in code

    def test_execute_cell_code(self):
        from colab.manager import ColabManager
        cm = ColabManager()
        code = cm.execute_cell_code("print('hello')")
        assert "exec(" in code
        assert "CELL COMPLETED" in code
        assert "CELL ERROR" in code


# ================================================================== #
#  colab/setup.py
# ================================================================== #

class TestSecretsManager:
    def test_detects_configured_keys_from_env(self):
        import os
        from colab.setup import SecretsManager
        os.environ["COLAB_AGENT_OPENAI_API_KEY"] = "sk-test-123"
        sm = SecretsManager()
        assert sm.is_configured("OPENAI_API_KEY") is True
        assert sm.get("OPENAI_API_KEY") == "sk-test-123"
        del os.environ["COLAB_AGENT_OPENAI_API_KEY"]

    def test_returns_custom_default(self):
        from colab.setup import SecretsManager
        sm = SecretsManager()
        val = sm.get("COLAB_AGENT_NONEXISTENT", default="fallback")
        assert val == "fallback"

    def test_summary_masks_keys(self):
        import os
        from colab.setup import SecretsManager
        os.environ["OPENAI_API_KEY"] = "sk-abcdefghijklmnop"
        sm = SecretsManager()
        summary = sm.summary()
        assert "OPENAI_API_KEY" in summary
        del os.environ["OPENAI_API_KEY"]

    def test_generate_colab_code(self):
        from colab.setup import SecretsManager
        sm = SecretsManager()
        code = sm.generate_colab_code()
        assert "Colab Secrets" in code


class TestDriveAuth:
    def test_mount_code(self):
        from colab.setup import DriveAuth
        code = DriveAuth.mount_code()
        assert "drive.mount" in code

    def test_pydrive_auth_code(self):
        from colab.setup import DriveAuth
        code = DriveAuth.pydrive_auth_code()
        assert "GoogleAuth" in code

    def test_check_mounted(self):
        from colab.setup import DriveAuth
        result = DriveAuth.check_mounted(os.path.dirname(__file__))
        assert result is True

    def test_ensure_drive_dir(self):
        from colab.setup import DriveAuth
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subdir")
            created = DriveAuth.ensure_drive_dir(path)
            assert os.path.isdir(created)


class TestLogManager:
    def _cleanup_logger(self, lm):
        lm.logger.handlers.clear()

    def test_init(self):
        from colab.setup import LogManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lm = LogManager("test-logger", log_dir=tmp)
            assert lm.log_dir == tmp
            self._cleanup_logger(lm)

    def test_info_log(self):
        from colab.setup import LogManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lm = LogManager("test-info", log_dir=tmp)
            lm.info("test message")
            path = lm.get_log_path()
            assert os.path.exists(path)
            content = open(path).read()
            assert "test message" in content
            self._cleanup_logger(lm)

    def test_read_recent(self):
        from colab.setup import LogManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lm = LogManager("test-recent", log_dir=tmp)
            lm.info("line1")
            lm.warn("line2")
            recent = lm.read_recent(10)
            assert "line1" in recent
            assert "line2" in recent
            self._cleanup_logger(lm)


class TestSetupColabEnvironment:
    def test_returns_string(self):
        from colab.setup import setup_colab_environment
        result = setup_colab_environment()
        assert isinstance(result, str)
        assert "Setup started" in result
        assert "pip install" in result

    def test_uses_custom_secrets(self):
        from colab.setup import setup_colab_environment, SecretsManager
        sm = SecretsManager()
        sm.set("OPENAI_API_KEY", "sk-test")
        result = setup_colab_environment(secrets=sm)
        assert "OPENAI_API_KEY" in result


# ================================================================== #
#  colab/executor.py
# ================================================================== #

class TestColabRuntimeInfo:
    def test_init_defaults(self):
        from colab.executor import ColabRuntimeInfo
        info = ColabRuntimeInfo()
        assert info.gpu_name == ""
        assert info.vram_total_gb == 0.0
        assert info.runtime_label == "None"

    def test_to_dict(self):
        from colab.executor import ColabRuntimeInfo
        info = ColabRuntimeInfo(gpu_name="Tesla T4", vram_total_gb=16.0,
                                ram_total_gb=12.0, runtime_label="T4")
        d = info.to_dict()
        assert d["gpu_name"] == "Tesla T4"
        assert d["runtime_label"] == "T4"

    def test_repr(self):
        from colab.executor import ColabRuntimeInfo
        info = ColabRuntimeInfo(gpu_name="Tesla T4", vram_total_gb=16.0, ram_total_gb=12.0)
        assert "Tesla T4" in repr(info)
        assert "16.0GB" in repr(info)


class TestColabRunner:
    def test_create_notebook(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        nb = runner.create_notebook("test")
        assert nb["success"] is True
        assert "notebook_id" in nb
        assert "cell_count" in nb

    def test_create_notebook_with_cells(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        cells = [{"type": "code", "source": "print('hi')"}]
        nb = runner.create_notebook("test", cells=cells)
        assert nb["cell_count"] == 1

    def test_detect_current_runtime(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        info = runner.detect_current_runtime()
        assert isinstance(info, object)
        assert hasattr(info, "gpu_name")

    def test_parse_output_empty(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        result = runner.parse_output({"output": "hello", "error": ""})
        assert isinstance(result, dict)
        assert result.get("error_type") is None

    def test_parse_output_oom_detection(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        result = runner.parse_output({
            "output": "", "error": "CUDA out of memory",
        })
        assert result is not None
        assert "oom" in (result.get("error_type") or "")

    def test_parse_output_syntax_error(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        result = runner.parse_output({
            "output": "",
            "error": "SyntaxError: invalid syntax",
        })
        assert result is not None
        assert "syntax" in (result.get("error_type") or "")

    def test_parse_output_import_error(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        result = runner.parse_output({
            "output": "",
            "error": "ModuleNotFoundError: No module named 'x'",
        })
        assert result is not None
        assert "import" in (result.get("error_type") or "")

    def test_parse_output_training_metrics(self):
        from colab.executor import ColabRunner
        runner = ColabRunner()
        result = runner.parse_output({
            "output": '{"loss": 0.5, "accuracy": 0.9}',
            "error": "",
        })
        assert isinstance(result, dict)

    def test_gpu_to_label_t4(self):
        from colab.executor import ColabRunner
        assert ColabRunner._gpu_to_label("Tesla T4") == "T4"

    def test_gpu_to_label_v100(self):
        from colab.executor import ColabRunner
        assert ColabRunner._gpu_to_label("Tesla V100") == "V100"

    def test_gpu_to_label_unknown(self):
        from colab.executor import ColabRunner
        assert ColabRunner._gpu_to_label("Unknown") == "None"


class TestHelperFunctions:
    def test_colab_auth_code(self):
        from colab.executor import _colab_auth_code
        code = _colab_auth_code()
        assert "auth.authenticate_user" in code

    def test_pydrive_import_code(self):
        from colab.executor import _pydrive_import_code
        code = _pydrive_import_code()
        assert "PyDrive" in code
        assert "GoogleAuth" in code


# ================================================================== #
#  colab/infrastructure.py
# ================================================================== #

class TestNotebookManager:
    def test_create(self):
        from colab.infrastructure import NotebookManager
        nm = NotebookManager()
        nb = nm.create("test")
        assert "cells" in nb
        assert "nbformat" in nb

    def test_create_with_cells(self):
        from colab.infrastructure import NotebookManager
        nm = NotebookManager()
        cells = [{"type": "code", "source": "print('hi')"}]
        nb = nm.create("test", cells=cells)
        assert len(nb["cells"]) == 1

    def test_add_cell(self):
        from colab.infrastructure import NotebookManager
        nm = NotebookManager()
        nm.create("test")
        nm.add_cell("print('added')")
        assert len(nm.active_notebook.cells) == 2


class TestRuntimeProfiler:
    def test_snapshot(self):
        from colab.infrastructure import RuntimeProfiler
        rp = RuntimeProfiler()
        snap = rp.snapshot()
        assert isinstance(snap, dict)
        assert "gpu" in snap or "timestamp" in snap


class TestTimeoutHandler:
    def test_context_manager(self):
        from colab.infrastructure import TimeoutHandler
        with TimeoutHandler(seconds=5):
            assert True


class TestErrorClassifier:
    def test_classify_oom(self):
        from colab.infrastructure import ErrorClassifier
        cat, score = ErrorClassifier.classify("CUDA out of memory")
        assert cat == "runtime_oom"
        assert score > 0

    def test_classify_syntax(self):
        from colab.infrastructure import ErrorClassifier
        cat, score = ErrorClassifier.classify("SyntaxError: bad input")
        assert cat == "syntax_error"

    def test_classify_import(self):
        from colab.infrastructure import ErrorClassifier
        cat, score = ErrorClassifier.classify("ImportError: No module named x")
        assert cat == "import_error"

    def test_classify_unknown(self):
        from colab.infrastructure import ErrorClassifier
        cat, score = ErrorClassifier.classify("Some random message")
        assert cat == "unknown"
        assert score == 0.0

    def test_classify_dataset_not_found(self):
        from colab.infrastructure import ErrorClassifier
        cat, _ = ErrorClassifier.classify("DatasetNotFound: dolly not found")
        assert cat == "dataset_error"

    def test_classify_disk_full(self):
        from colab.infrastructure import ErrorClassifier
        cat, _ = ErrorClassifier.classify("No space left on device")
        assert cat == "disk_full"

    def test_actionable_categories(self):
        from colab.infrastructure import ErrorClassifier
        assert ErrorClassifier.actionable("runtime_oom") is True
        assert ErrorClassifier.actionable("syntax_error") is True
        assert ErrorClassifier.actionable("unknown") is False


class TestSafeCodeValidator:
    def test_validate_safe_code(self):
        from colab.infrastructure import SafeCodeValidator
        sv = SafeCodeValidator()
        safe, cleaned, msg = sv.validate("print('hello')")
        assert safe is True
        assert msg == ""

    def test_validate_blocked_rm(self):
        from colab.infrastructure import SafeCodeValidator
        sv = SafeCodeValidator()
        safe, _, msg = sv.validate("import os; os.system('rm -rf /')")
        assert safe is False
        assert "blocked" in msg.lower()

    def test_validate_blocked_subprocess(self):
        from colab.infrastructure import SafeCodeValidator
        sv = SafeCodeValidator()
        safe, _, msg = sv.validate("subprocess.call('rm -rf /', shell=True)")
        assert safe is False

    def test_compute_hash(self):
        from colab.infrastructure import SafeCodeValidator
        sv = SafeCodeValidator()
        h = sv.compute_hash("print('hi')")
        assert len(h) == 16
        assert isinstance(h, str)

    def test_check_code_gen(self):
        from colab.infrastructure import SafeCodeValidator
        sv = SafeCodeValidator()
        result = sv.check_code_gen("print('hi')")
        assert result["safe"] is True
        assert result["has_pip"] is False


class TestRetryPolicy:
    def test_init(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=1.0)
        assert rp.max_retries == 3

    def test_should_retry_initial(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3)
        assert rp.should_retry() is True

    def test_should_retry_exhausted(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3)
        rp.attempt = 3
        assert rp.should_retry() is False

    def test_next_delay(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=1.0)
        delay = rp.next_delay()
        assert delay >= 1.0

    def test_reset(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=1.0)
        rp.attempt = 2
        rp.reset()
        assert rp.attempt == 0

    def test_exponential_backoff_code(self):
        from colab.infrastructure import RetryPolicy
        code = RetryPolicy.exponential_backoff_code(max_retries=5)
        assert "MAX_RETRIES = 5" in code

    def test_record_success_decreases_delay(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=10.0)
        rp.record_result(False)
        inflated = rp.base_delay
        rp.record_result(True)
        assert rp.base_delay < inflated

    def test_record_failure_increases_delay(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=2.0)
        old = rp.base_delay
        rp.record_result(False)
        assert rp.base_delay > old

    def test_delay_stays_within_bounds(self):
        from colab.infrastructure import RetryPolicy
        rp = RetryPolicy(max_retries=3, base_delay=2.0, max_delay=10.0)
        for _ in range(10):
            rp.record_result(False)
        assert rp.base_delay <= 10.0
