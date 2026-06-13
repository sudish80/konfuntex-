"""Tests for multi-tenant isolation across storage layers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _setup_db():
    from storage.database import reset_session, Base
    from sqlalchemy import create_engine
    reset_session()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def _teardown_db():
    from storage.database import reset_session
    reset_session()


class TestMultiTenantJobs:
    def setup_method(self):
        from storage.database import reset_session
        reset_session()
        import os as os_mod
        self._old_url = os_mod.environ.get("COLAB_AGENT_DB_URL")
        os_mod.environ["COLAB_AGENT_DB_URL"] = "sqlite://"
        self.engine = _setup_db()

    def teardown_method(self):
        import os as os_mod
        if self._old_url is None:
            os_mod.environ.pop("COLAB_AGENT_DB_URL", None)
        else:
            os_mod.environ["COLAB_AGENT_DB_URL"] = self._old_url
        _teardown_db()

    def test_tenant_a_does_not_see_tenant_b_jobs(self):
        from storage.jobs import JobStore
        store_a = JobStore(tenant_id="tenant-a")
        store_b = JobStore(tenant_id="tenant-b")

        store_a.create("goal a")
        store_b.create("goal b")

        assert len(store_a.list()) == 1
        assert len(store_b.list()) == 1
        assert store_a.list()[0].goal == "goal a"
        assert store_b.list()[0].goal == "goal b"

    def test_unscoped_store_sees_all(self):
        from storage.jobs import JobStore
        store_a = JobStore(tenant_id="tenant-a")
        store_all = JobStore(tenant_id=None)

        store_a.create("goal a")
        store_all.create("goal all")

        all_jobs = store_all.list()
        assert len(all_jobs) == 2

    def test_get_respects_tenant(self):
        from storage.jobs import JobStore
        store_a = JobStore(tenant_id="tenant-a")
        store_b = JobStore(tenant_id="tenant-b")

        job = store_a.create("secret a")
        assert store_b.get(job.id) is None
        assert store_a.get(job.id) is not None


class TestMultiTenantConversations:
    def setup_method(self):
        from storage.database import reset_session
        reset_session()
        import os as os_mod
        self._old_url = os_mod.environ.get("COLAB_AGENT_DB_URL")
        os_mod.environ["COLAB_AGENT_DB_URL"] = "sqlite://"
        self.engine = _setup_db()

    def teardown_method(self):
        import os as os_mod
        if self._old_url is None:
            os_mod.environ.pop("COLAB_AGENT_DB_URL", None)
        else:
            os_mod.environ["COLAB_AGENT_DB_URL"] = self._old_url
        _teardown_db()

    def test_tenant_scoped_conversations(self):
        from storage.conversations import ConversationStore
        ca = ConversationStore(tenant_id="a")
        cb = ConversationStore(tenant_id="b")

        ca.create("goal a")
        ca.create("goal a2")
        cb.create("goal b")

        assert len(ca.list_all()) == 2
        assert len(cb.list_all()) == 1


class TestMultiTenantModels:
    def setup_method(self):
        from storage.database import reset_session
        reset_session()
        import os as os_mod
        self._old_url = os_mod.environ.get("COLAB_AGENT_DB_URL")
        os_mod.environ["COLAB_AGENT_DB_URL"] = "sqlite://"
        self.engine = _setup_db()

    def teardown_method(self):
        import os as os_mod
        if self._old_url is None:
            os_mod.environ.pop("COLAB_AGENT_DB_URL", None)
        else:
            os_mod.environ["COLAB_AGENT_DB_URL"] = self._old_url
        _teardown_db()

    def test_tenant_scoped_models(self):
        from storage.models_store import ModelVersionStore
        ma = ModelVersionStore(tenant_id="a")
        mb = ModelVersionStore(tenant_id="b")

        ma.create("phi-2", method="qlora")
        mb.create("llama", method="lora")

        assert len(ma.list_all()) == 1
        assert len(mb.list_all()) == 1


class TestMultiTenantMetrics:
    def setup_method(self):
        from storage.database import reset_session
        reset_session()
        import os as os_mod
        self._old_url = os_mod.environ.get("COLAB_AGENT_DB_URL")
        os_mod.environ["COLAB_AGENT_DB_URL"] = "sqlite://"
        self.engine = _setup_db()
        # Create jobs to satisfy FK constraints
        from storage.database import Job
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=self.engine)
        sess = Session()
        sess.add(Job(id="job-1", goal="test"))
        sess.add(Job(id="job-2", goal="test2"))
        sess.commit()
        sess.close()

    def teardown_method(self):
        import os as os_mod
        if self._old_url is None:
            os_mod.environ.pop("COLAB_AGENT_DB_URL", None)
        else:
            os_mod.environ["COLAB_AGENT_DB_URL"] = self._old_url
        _teardown_db()

    def test_tenant_scoped_metrics(self):
        from storage.metrics_store import MetricsStore
        ma = MetricsStore(tenant_id="a")
        mb = MetricsStore(tenant_id="b")

        ma.log_epoch("job-1", loss=0.5)
        ma.log_epoch("job-1", loss=0.3)
        mb.log_epoch("job-2", loss=0.9)

        assert len(ma.get_job_metrics("job-1")) == 2
        assert len(mb.get_job_metrics("job-2")) == 1
