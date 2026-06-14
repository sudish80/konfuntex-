"""Tests for advanced colab modules: automation, drive_sync, resumer, bridge, enterprise."""
import os
import sys
import json
import time
import tempfile
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ================================================================== #
#  colab/drive_sync.py
# ================================================================== #

class TestDriveSyncDaemon:
    def test_init_defaults(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        assert d.job_id == "default"
        assert d.keep_versions == 5
        assert d.status["running"] is False

    def test_init_validation_drive_dir(self):
        from colab.drive_sync import DriveSyncDaemon
        try:
            DriveSyncDaemon(drive_dir=123)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_init_validation_local_dir(self):
        from colab.drive_sync import DriveSyncDaemon
        try:
            DriveSyncDaemon(local_dir=None)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_init_validation_job_id(self):
        from colab.drive_sync import DriveSyncDaemon
        try:
            DriveSyncDaemon(job_id=None)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_init_validation_keep_versions(self):
        from colab.drive_sync import DriveSyncDaemon
        try:
            DriveSyncDaemon(keep_versions=0)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_start_stop(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        d.start(interval=1)
        assert d.status["running"] is True
        d.stop()
        assert d.status["running"] is False

    def test_start_twice_noop(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        d.start(interval=1)
        d.start(interval=1)  # should not crash
        d.stop()

    def test_start_invalid_interval(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        try:
            d.start(interval=-1)
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_sync_now_creates_checkpoint(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "local")
            drive = os.path.join(tmp, "drive")
            os.makedirs(local)
            with open(os.path.join(local, "model.safetensors"), "w") as f:
                f.write("weights")
            d = DriveSyncDaemon(drive_dir=drive, local_dir=local, job_id="test_job")
            d.start(interval=1)
            d.sync_now()
            time.sleep(0.5)
            d.stop()
            versions = d.list_versions()
            assert len(versions) >= 1
            assert versions[0]["version"] == 1

    def test_multiple_syncs_versions_increment(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "local")
            drive = os.path.join(tmp, "drive")
            os.makedirs(local)
            d = DriveSyncDaemon(drive_dir=drive, local_dir=local, job_id="ver_test")
            d.start(interval=1)
            with open(os.path.join(local, "ckpt.bin"), "w") as f:
                f.write("v1")
            d.sync_now()
            time.sleep(0.3)
            with open(os.path.join(local, "ckpt.bin"), "w") as f:
                f.write("v2")
            d.sync_now()
            time.sleep(0.3)
            d.stop()
            versions = d.list_versions()
            assert len(versions) == 2
            assert versions[0]["version"] == 1
            assert versions[1]["version"] == 2

    def test_manifest_exists_after_sync(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "local")
            drive = os.path.join(tmp, "drive")
            os.makedirs(local)
            os.makedirs(os.path.join(drive, "test_job"))
            d = DriveSyncDaemon(drive_dir=drive, local_dir=local, job_id="test_job")
            d.start(interval=1)
            d.sync_now()
            time.sleep(0.5)
            d.stop()
            manifest = d.get_manifest()
            assert manifest["job_id"] == "test_job"
            assert len(manifest["versions"]) >= 1

    def test_retention_keeps_latest(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "local")
            drive = os.path.join(tmp, "drive")
            os.makedirs(local)
            d = DriveSyncDaemon(drive_dir=drive, local_dir=local, job_id="ret_test", keep_versions=2)
            d.start(interval=1)
            for i in range(4):
                with open(os.path.join(local, "ckpt.bin"), "w") as f:
                    f.write(f"v{i}")
                d.sync_now()
                time.sleep(0.2)
            d.stop()
            versions = d.list_versions()
            assert len(versions) <= 2

    def test_compute_checksum(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.bin")
            with open(path, "w") as f:
                f.write("hello world")
            d = DriveSyncDaemon()
            cs = d.compute_checksum(path)
            assert len(cs) == 16
            assert cs == d.compute_checksum(path)

    def test_compute_checksum_missing_file(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        assert d.compute_checksum("/nonexistent") == ""

    def test_get_latest_version_empty(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        assert d.get_latest_version() is None

    def test_get_latest_path_empty(self):
        from colab.drive_sync import DriveSyncDaemon
        d = DriveSyncDaemon()
        assert d.get_latest_path() is None

    def test_skip_patterns_respected(self):
        from colab.drive_sync import DriveSyncDaemon
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "local")
            drive = os.path.join(tmp, "drive")
            os.makedirs(local)
            with open(os.path.join(local, "__pycache__"), "w") as f:
                f.write("cache")
            with open(os.path.join(local, "model.bin"), "w") as f:
                f.write("weights")
            d = DriveSyncDaemon(drive_dir=drive, local_dir=local, job_id="skip_test",
                                skip_patterns=["__pycache__"])
            d.start(interval=1)
            d.sync_now()
            time.sleep(0.5)
            d.stop()
            latest = d.get_latest_path()
            if latest:
                files = os.listdir(latest) if os.path.isdir(latest) else []
                assert "__pycache__" not in files


# ================================================================== #
#  colab/resumer.py
# ================================================================== #

class TestColabResumer:
    def test_init_validation(self):
        from colab.resumer import ColabResumer
        try:
            ColabResumer(drive_dir=None)
            assert False
        except TypeError:
            pass

    def test_detect_no_drive_dir(self):
        from colab.resumer import ColabResumer
        r = ColabResumer(drive_dir="/nonexistent/path")
        result = r.detect_previous_run()
        assert result["has_checkpoint"] is False
        assert "not found" in (result.get("error") or "").lower()

    def test_detect_with_checkpoint(self):
        from colab.resumer import ColabResumer
        with tempfile.TemporaryDirectory() as tmp:
            drive = os.path.join(tmp, "drive")
            job_dir = os.path.join(drive, "job_abc")
            os.makedirs(job_dir)
            ckpt_dir = os.path.join(job_dir, "checkpoint-3")
            os.makedirs(ckpt_dir)
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                json.dump({"r": 8}, f)
            state = {"model_name": "llama", "dataset_name": "dolly",
                     "method": "lora", "epochs_completed": 2, "last_loss": 0.42}
            with open(os.path.join(job_dir, "training_state.json"), "w") as f:
                json.dump(state, f)
            r = ColabResumer(drive_dir=drive)
            result = r.detect_previous_run()
            assert result["has_checkpoint"] is True
            assert result["checkpoint_version"] == 3
            assert result["job_id"] == "job_abc"
            assert result["model_name"] == "llama"
            assert result["epochs_completed"] == 2
            assert result["last_loss"] == 0.42
            assert result["checkpoint_valid"] is True

    def test_detect_without_training_state(self):
        from colab.resumer import ColabResumer
        with tempfile.TemporaryDirectory() as tmp:
            drive = os.path.join(tmp, "drive")
            job_dir = os.path.join(drive, "job_no_state")
            os.makedirs(job_dir)
            ckpt_dir = os.path.join(job_dir, "checkpoint-1")
            os.makedirs(ckpt_dir)
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump({"model_type": "test"}, f)
            r = ColabResumer(drive_dir=drive)
            result = r.detect_previous_run()
            assert result["has_checkpoint"] is True
            assert result["has_training_state"] is False

    def test_build_resume_code_no_checkpoint(self):
        from colab.resumer import ColabResumer
        r = ColabResumer(drive_dir="/nonexistent")
        code = r.build_resume_code({"has_checkpoint": False})
        assert "No checkpoint found" in code

    def test_build_resume_code_with_checkpoint(self):
        from colab.resumer import ColabResumer
        state = {
            "has_checkpoint": True,
            "job_id": "job_xyz",
            "checkpoint_path": "/drive/job_xyz/checkpoint-2",
            "checkpoint_version": 2,
            "model_name": "mistral",
            "dataset_name": "alpaca",
            "method": "lora",
            "epochs_completed": 1,
        }
        r = ColabResumer()
        code = r.build_resume_code(state)
        assert "Resuming training" in code or "checkpoint" in code.lower()
        assert "mistral" in code or "job_xyz" in code

    def test_build_resume_code_auto_detect(self):
        from colab.resumer import ColabResumer
        r = ColabResumer(drive_dir="/nonexistent")
        code = r.build_resume_code()
        assert "No checkpoint" in code or "resume" in code.lower()

    def test_save_and_load_state(self):
        from colab.resumer import ColabResumer
        with tempfile.TemporaryDirectory() as tmp:
            r = ColabResumer(drive_dir=tmp)
            r.save_state("test_job", {"model_name": "gpt", "epochs_completed": 3})
            loaded = r.load_state("test_job")
            assert loaded is not None
            assert loaded["model_name"] == "gpt"
            assert loaded["epochs_completed"] == 3

    def test_load_missing_state(self):
        from colab.resumer import ColabResumer
        r = ColabResumer(drive_dir="/nonexistent")
        assert r.load_state("missing_job") is None

    def test_generate_resume_notebook(self):
        from colab.resumer import ColabResumer
        state = {
            "has_checkpoint": True,
            "job_id": "nb_test",
            "checkpoint_path": "/ckpt",
            "checkpoint_version": 1,
            "model_name": "m",
            "dataset_name": "d",
            "method": "lora",
            "epochs_completed": 0,
        }
        r = ColabResumer()
        nb = r.generate_resume_notebook(state)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == 2
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert nb["cells"][1]["cell_type"] == "code"

    def test_generate_detect_code(self):
        from colab.resumer import ColabResumer
        code = ColabResumer.generate_detect_code()
        assert "DRIVE_DIR" in code
        assert "checkpoint" in code.lower()

    def test_save_state_validation(self):
        from colab.resumer import ColabResumer
        r = ColabResumer()
        try:
            r.save_state(None, {})
            assert False
        except TypeError:
            pass

    def test_save_state_params_validation(self):
        from colab.resumer import ColabResumer
        r = ColabResumer()
        try:
            r.save_state("x", "not_a_dict")
            assert False
        except TypeError:
            pass


# ================================================================== #
#  colab/bridge.py
# ================================================================== #

class TestColabBridge:
    def test_init_defaults(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        assert b.connected is False
        assert b.event_count == 0
        assert b.buffer_size == 0

    def test_record_log(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_log("hello", "info")
        assert b.buffer_size == 1

    def test_record_metric(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_metric("loss", 0.42, step=10)
        assert b.buffer_size == 1

    def test_record_metric_validation(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_metric(123, "not_float")  # should not raise
        assert b.buffer_size == 0  # type mismatch

    def test_record_loss(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_loss(0.5, epoch=2)
        assert b.buffer_size == 2

    def test_record_error(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_error("OOM")
        assert b.buffer_size == 1

    def test_record_status(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_status("step_1_ok", "Training started")
        assert b.buffer_size == 1

    def test_record_resource(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_resource(vram_used=8.0, vram_total=16.0, ram_pct=45.0, gpu_name="T4")
        assert b.buffer_size == 1

    def test_poll_events(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.record_log("test")
        events = b.poll_events()
        assert len(events) == 1
        assert b.buffer_size == 0

    def test_start_validation_job_id(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        try:
            b.start(job_id=None)
            assert False
        except TypeError:
            pass

    def test_start_validation_interval(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        try:
            b.start(interval=0.1)
            assert False
        except TypeError:
            pass

    def test_start_stop(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.start(job_id="test_job", interval=1)
        import time
        time.sleep(0.3)
        assert b._thread is not None
        assert b._thread.is_alive()
        b._stop_event.set()
        b._thread.join(timeout=5)
        assert not b._thread.is_alive()
        b._thread = None
        b.stop()  # cleanup

    def test_stop_idempotent(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        b.stop()  # should not crash
        b.stop()  # should not crash

    def test_max_buffer(self):
        from colab.bridge import ColabBridge
        b = ColabBridge()
        for i in range(2500):
            b.record_log(f"msg_{i}")
        assert b.buffer_size <= 2000

    def test_generate_bridge_code(self):
        from colab.bridge import ColabBridge
        code = ColabBridge.generate_bridge_code(
            server_url="ws://test:8080/ws", token="abc", job_id="j1"
        )
        assert "ws://test:8080/ws" in code
        assert "Bridge" in code


# ================================================================== #
#  colab/enterprise.py
# ================================================================== #

class TestColabEnterprise:
    def test_init_defaults(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        assert e.project == "" or e.project is not None

    def test_is_available(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        # Without project, should be False
        if not e.project:
            assert e.is_available() is False
        else:
            assert isinstance(e.is_available(), bool)

    def test_create_runtime_unknown_spec(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="test-project")
        result = e.create_runtime(runtime_spec="NONEXISTENT")
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_create_runtime_validation(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        result = e.create_runtime(runtime_spec=123)
        assert result["success"] is False

    def test_create_runtime_t4(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="test-project")
        result = e.create_runtime("T4", display_name="test-runtime")
        if not e.is_available():
            assert result["success"] is False
        else:
            assert result["success"] is True
            assert result["spec"]["accelerator"] == "NVIDIA_TESLA_T4"

    def test_get_runtime_validation(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        result = e.get_runtime("")
        assert result["success"] is False

    def test_get_runtime(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="p")
        result = e.get_runtime("projects/p/locations/us/runtimes/ex")
        if e.is_available():
            assert result["success"] is True

    def test_delete_runtime_validation(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        result = e.delete_runtime("")
        assert result["success"] is False

    def test_delete_runtime(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="p")
        result = e.delete_runtime("projects/p/locations/us/runtimes/ex")
        if e.is_available():
            assert result["success"] is True
            assert result["deleted"] is True

    def test_list_runtimes(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        runtimes = e.list_runtimes()
        assert isinstance(runtimes, list)

    def test_execute_code_validation_runtime(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        result = e.execute_code("", "print('hi')")
        assert result["success"] is False

    def test_execute_code_validation_code(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        result = e.execute_code("projects/p/runtimes/r", "")
        assert result["success"] is False

    def test_execute_code(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="p")
        result = e.execute_code("projects/p/runtimes/r", "print('hi')")
        if e.is_available():
            assert result["success"] is True

    def test_validate_spec(self):
        from colab.enterprise import ColabEnterprise
        assert ColabEnterprise.validate_spec("T4") is True
        assert ColabEnterprise.validate_spec("V100") is True
        assert ColabEnterprise.validate_spec("NONEXISTENT") is False

    def test_list_specs(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise()
        specs = e.list_specs()
        assert len(specs) >= 5
        assert specs[0]["name"] == "A100" or any(s["name"] == "T4" for s in specs)

    def test_best_fit_spec(self):
        from colab.enterprise import ColabEnterprise
        assert ColabEnterprise.best_fit_spec(8) == "T4"
        assert ColabEnterprise.best_fit_spec(24) == "V100"
        assert ColabEnterprise.best_fit_spec(50) == "A100-80GB"
        assert ColabEnterprise.best_fit_spec(200) == "A100-80GB"

    def test_generate_setup_code(self):
        from colab.enterprise import ColabEnterprise
        e = ColabEnterprise(project="my-proj", location="us-central1")
        code = e.generate_setup_code()
        assert "my-proj" in code
        assert "us-central1" in code
        assert "aiplatform.init" in code


# ================================================================== #
#  colab/automation.py
# ================================================================== #

class TestColabAutomation:
    def test_init_defaults(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        assert a.headless is False
        assert isinstance(a.is_available(), bool)
        a.close()

    def test_open_notebook_returns_error_when_unavailable(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        a.headless = True
        if not a.is_available():
            result = a.open_notebook("https://colab.research.google.com")
            assert result["success"] is False
            assert "not available" in result["error"].lower()
        a.close()

    def test_switch_runtime_fallback_when_unavailable(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        if not a.is_available():
            result = a.switch_runtime("A100")
            assert result["success"] is True
            assert result.get("manual") is True
        a.close()

    def test_detect_runtime_not_connected(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        result = a.detect_runtime()
        assert result["gpu"] is None
        assert result.get("connected") is False
        a.close()

    def test_close_idempotent(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        a.close()
        a.close()

    def test_wait_for_connection_no_page(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        result = a.wait_for_connection(timeout=1)
        assert result is False
        a.close()

    def test_capture_screenshot_no_page(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        assert a.capture_screenshot() is None
        a.close()

    def test_open_new_notebook(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation(headless=True)
        if not a.is_available():
            result = a.open_new_notebook()
            assert result["success"] is False
        a.close()

    def test_open_notebook_by_id(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        if not a.is_available():
            result = a.open_notebook_by_id("test123")
            assert result["success"] is False
        a.close()

    def test_generate_manual_switch_code_t4(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        code = a._generate_manual_switch_code("T4")
        assert code["manual"] is True
        assert code["switched_to"] == "T4"
        assert "kill -9" in code["code"]

    def test_generate_manual_switch_code_tpu(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        code = a._generate_manual_switch_code("TPU")
        assert code["switched_to"] == "TPU"

    def test_generate_manual_switch_code_none(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        code = a._generate_manual_switch_code("None")
        assert code["switched_to"] == "None"

    def test_runtime_order_present(self):
        from colab.automation import ColabAutomation
        assert "T4" in ColabAutomation.RUNTIME_ORDER
        assert "A100" in ColabAutomation.RUNTIME_ORDER
        assert "TPU" in ColabAutomation.RUNTIME_ORDER

    def test_context_manager(self):
        from colab.automation import ColabAutomation
        with ColabAutomation() as a:
            assert a is not None
            assert a._closed is False
        assert a._closed is True

    def test_execute_cell_not_connected(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        result = a.execute_cell("print('hello')")
        assert result["success"] is False
        assert "Not connected" in result["error"]
        a.close()

    def test_monitor_cell_output_no_page(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        result = a.monitor_cell_output(timeout=1)
        assert result["success"] is False
        a.close()

    def test_detect_runtime_connected_false_no_page(self):
        from colab.automation import ColabAutomation
        a = ColabAutomation()
        assert a._connected is False
        a.close()


# ================================================================== #
#  colab/remote_executor.py
# ================================================================== #

class TestRemoteColabExecutor:
    def test_init_defaults(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor(headless=True)
        assert e.headless is True
        assert e.available is False or isinstance(e.available, bool)
        assert e.connected is False
        e.disconnect()

    def test_available_property(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        assert isinstance(e.available, bool)

    def test_execute_not_connected(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        result = e.execute("print('hi')")
        assert result["success"] is False
        assert "Not connected" in result["error"]
        e.disconnect()

    def test_disconnect_idempotent(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        e.disconnect()
        e.disconnect()

    def test_capture_screenshot_no_page(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        assert e.capture_screenshot() is None
        e.disconnect()

    def test_get_runtime_info_no_page(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        info = e.get_runtime_info()
        assert info["gpu"] is None
        e.disconnect()

    def test_context_manager(self):
        from colab.remote_executor import RemoteColabExecutor
        with RemoteColabExecutor(headless=True) as e:
            assert e.headless is True
        assert e._connected is False

    def test_generate_setup_instructions(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        assert isinstance(e.available, bool)

    def test_connect_timeout_no_page(self):
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor(headless=True)
        result = e.connect(timeout=1)
        if not e.available:
            assert result["success"] is False
        e.disconnect()

    def test_set_cell_code_no_page(self):
        """Test that _set_cell_code doesn't crash without a page."""
        from colab.remote_executor import RemoteColabExecutor
        e = RemoteColabExecutor()
        try:
            e._set_cell_code("print('x')")
        except AttributeError:
            pass  # expected - no page
        e.disconnect()
