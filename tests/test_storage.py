"""Tests for storage/ modules — JobStore, ConversationStore, MetricsStore, ModelVersionStore."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.database import init_db, reset_session
from storage.jobs import JobStore
from storage.conversations import ConversationStore
from storage.metrics_store import MetricsStore
from storage.models_store import ModelVersionStore


def _fresh_db():
    reset_session()
    init_db()
    return True


class TestJobStore:
    def setup_method(self):
        _fresh_db()
        self.store = JobStore()

    def test_create_job(self):
        job = self.store.create(goal="test fine-tune", method="qlora")
        assert job.id is not None
        assert job.goal == "test fine-tune"
        assert job.status == "pending"
        assert job.method == "qlora"

    def test_create_job_with_all_fields(self):
        job = self.store.create(
            goal="full params",
            method="full",
            base_model="meta/llama-2-7b",
            dataset="dolly",
            runtime="A100",
        )
        assert job.base_model == "meta/llama-2-7b"
        assert job.dataset == "dolly"
        assert job.runtime == "A100"

    def test_get_job(self):
        created = self.store.create(goal="get test")
        fetched = self.store.get(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_nonexistent(self):
        assert self.store.get("no-such-id") is None

    def test_update_job(self):
        job = self.store.create(goal="update test")
        updated = self.store.update(job.id, status="running", runtime="T4")
        assert updated.status == "running"
        assert updated.runtime == "T4"

    def test_update_nonexistent(self):
        result = self.store.update("no-such-id", status="completed")
        assert result is None

    def test_list_jobs(self):
        self.store.create(goal="list test 1")
        self.store.create(goal="list test 2", method="lora")
        jobs = self.store.list()
        assert len(jobs) >= 2

    def test_list_filter_by_status(self):
        self.store.create(goal="status test", method="lora")
        pending = self.store.list(status="pending")
        for j in pending:
            assert j.status == "pending"

    def test_list_with_limit(self):
        for i in range(5):
            self.store.create(goal=f"limit test {i}")
        jobs = self.store.list(limit=3)
        assert len(jobs) <= 3

    def test_delete_job(self):
        job = self.store.create(goal="delete test")
        self.store.delete(job.id)
        assert self.store.get(job.id) is None

    def test_delete_nonexistent(self):
        self.store.delete("no-such-id")

    def test_set_and_get_metrics(self):
        job = self.store.create(goal="metrics test")
        job.set_metrics({"loss": 0.5, "accuracy": 0.9})
        assert job.get_metrics()["loss"] == 0.5

    def test_set_and_get_metadata(self):
        job = self.store.create(goal="metadata test")
        job.set_metadata({"epochs": 3, "lr": 2e-5})
        self.store.session.commit()
        fetched = self.store.get(job.id)
        assert fetched.get_metadata()["epochs"] == 3


class TestConversationStore:
    def setup_method(self):
        _fresh_db()
        self.store = ConversationStore()

    def test_create(self):
        conv = self.store.create(goal="test chat")
        assert conv.id is not None
        assert conv.status == "active"

    def test_get(self):
        created = self.store.create(goal="get conv")
        fetched = self.store.get(created.id)
        assert fetched.id == created.id

    def test_get_nonexistent(self):
        assert self.store.get("no-such-id") is None

    def test_add_and_get_messages(self):
        conv = self.store.create(goal="message test")
        self.store.add_message(conv.id, "user", "Hello")
        self.store.add_message(conv.id, "assistant", "Hi!")
        msgs = self.store.get_messages(conv.id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_add_message_unknown_conv(self):
        result = self.store.add_message("no-such-id", "user", "test")
        assert result is None

    def test_get_messages_empty(self):
        conv = self.store.create(goal="empty msgs")
        assert self.store.get_messages(conv.id) == []

    def test_close(self):
        conv = self.store.create(goal="close test")
        self.store.close(conv.id, summary="done successfully")
        fetched = self.store.get(conv.id)
        assert fetched.status == "closed"
        assert fetched.summary == "done successfully"

    def test_list_active(self):
        self.store.create(goal="active test")
        active = self.store.list_active()
        for c in active:
            assert c.status == "active"

    def test_list_all(self):
        self.store.create(goal="all test 1")
        self.store.create(goal="all test 2")
        all_conv = self.store.list_all()
        assert len(all_conv) >= 2


class TestMetricsStore:
    def setup_method(self):
        _fresh_db()
        self.store = MetricsStore()
        self.store.create_tables()
        self.job_id = "metrics-test-job"

    def test_log_epoch(self):
        rec = self.store.log_epoch(
            job_id=self.job_id, epoch=1.0, global_step=100,
            loss=0.5, accuracy=0.85, gpu_mem_gb=12.5,
        )
        assert rec.id is not None
        assert rec.loss == 0.5

    def test_log_epoch_with_extras(self):
        rec = self.store.log_epoch(
            job_id=self.job_id, extras={"grad_norm": 0.1, "custom": "val"}
        )
        assert rec.get_extras()["grad_norm"] == 0.1

    def test_log_batch(self):
        batch = [
            {"epoch": 1, "loss": 0.6, "accuracy": 0.8},
            {"epoch": 2, "loss": 0.4, "accuracy": 0.9, "custom_field": "x"},
        ]
        self.store.log_batch(self.job_id, batch)
        metrics = self.store.get_job_metrics(self.job_id)
        assert len(metrics) >= 2

    def test_get_job_metrics_ordered(self):
        self.store.log_epoch(job_id=self.job_id, global_step=1, loss=0.9)
        self.store.log_epoch(job_id=self.job_id, global_step=2, loss=0.3)
        rows = self.store.get_job_metrics(self.job_id)
        steps = [r.global_step for r in rows if r.global_step is not None]
        assert steps == sorted(steps)

    def test_get_job_summary(self):
        self.store.log_epoch(job_id="summary-test", global_step=1, loss=0.8, accuracy=0.7)
        self.store.log_epoch(job_id="summary-test", global_step=2, loss=0.3, accuracy=0.95, gpu_mem_gb=10)
        summary = self.store.get_job_summary("summary-test")
        assert summary["final_loss"] == 0.3
        assert summary["min_loss"] == 0.3
        assert summary["best_accuracy"] == 0.95
        assert summary["peak_gpu_mem_gb"] == 10.0

    def test_get_job_summary_empty(self):
        assert self.store.get_job_summary("no-metrics") == {}

    def test_export_to_json(self, tmp_path):
        fpath = str(tmp_path / "metrics.json")
        self.store.log_epoch(job_id="export-test", loss=0.5, accuracy=0.8)
        self.store.export_to_json("export-test", fpath)
        assert os.path.exists(fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert len(data) >= 1
        assert data[0]["loss"] == 0.5


class TestModelVersionStore:
    def setup_method(self):
        _fresh_db()
        self.store = ModelVersionStore()

    def test_create(self):
        mv = self.store.create(
            base_model="microsoft/phi-2",
            job_id="job-1",
            method="qlora",
            runtime_used="T4",
        )
        assert mv.id is not None
        assert mv.base_model == "microsoft/phi-2"
        assert mv.method == "qlora"

    def test_get(self):
        created = self.store.create(base_model="test/model")
        fetched = self.store.get(created.id)
        assert fetched.id == created.id

    def test_get_nonexistent(self):
        assert self.store.get("no-such-id") is None

    def test_update(self):
        mv = self.store.create(base_model="test/model")
        updated = self.store.update(mv.id, finetuned_path="./output", final_loss=0.35)
        assert updated.finetuned_path == "./output"
        assert updated.final_loss == 0.35

    def test_update_nonexistent(self):
        assert self.store.update("no-such-id", status="done") is None

    def test_list_by_job(self):
        self.store.create(base_model="m1", job_id="job-list-a")
        self.store.create(base_model="m2", job_id="job-list-a")
        self.store.create(base_model="m3", job_id="job-list-b")
        versions = self.store.list_by_job("job-list-a")
        assert len(versions) == 2

    def test_list_all(self):
        initial_count = len(self.store.list_all())
        self.store.create(base_model="m1")
        self.store.create(base_model="m2")
        assert len(self.store.list_all()) >= initial_count + 2
