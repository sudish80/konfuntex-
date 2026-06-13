"""
MetricsStore — Phase 4
Logs per-epoch training metrics (loss, accuracy, GPU_mem, tokens_per_sec, etc.)
to SQLite for later analysis and reporting.
"""
import json
import uuid
from datetime import datetime, timezone
from storage.database import get_session, Base
import sqlalchemy as sa


class MetricRecord(Base):
    __tablename__ = "metrics"

    id = sa.Column(sa.String, primary_key=True)
    tenant_id = sa.Column(sa.String, nullable=True, index=True)
    job_id = sa.Column(sa.String, sa.ForeignKey("jobs.id"), nullable=False)
    epoch = sa.Column(sa.Float, nullable=True)
    global_step = sa.Column(sa.Integer, nullable=True)
    loss = sa.Column(sa.Float, nullable=True)
    accuracy = sa.Column(sa.Float, nullable=True)
    gpu_mem_gb = sa.Column(sa.Float, nullable=True)
    tokens_per_second = sa.Column(sa.Float, nullable=True)
    learning_rate = sa.Column(sa.Float, nullable=True)
    grad_norm = sa.Column(sa.Float, nullable=True)
    timestamp = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc))
    extras_json = sa.Column(sa.Text, nullable=True)

    def get_extras(self) -> dict:
        return json.loads(self.extras_json) if self.extras_json else {}

    def set_extras(self, d: dict):
        self.extras_json = json.dumps(d)


class MetricsStore:
    def __init__(self, tenant_id: str = None):
        self.session = get_session()
        self.tenant_id = tenant_id

    def create_tables(self):
        Base.metadata.create_all(bind=self.session.bind)

    def log_epoch(self, job_id: str, epoch: float = None,
                  global_step: int = None, loss: float = None,
                  accuracy: float = None, gpu_mem_gb: float = None,
                  tokens_per_second: float = None,
                  learning_rate: float = None,
                  grad_norm: float = None,
                  extras: dict = None) -> MetricRecord:
        rec = MetricRecord(
            id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            job_id=job_id,
            epoch=epoch,
            global_step=global_step,
            loss=loss,
            accuracy=accuracy,
            gpu_mem_gb=gpu_mem_gb,
            tokens_per_second=tokens_per_second,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
        )
        if extras:
            rec.set_extras(extras)
        self.session.add(rec)
        self.session.commit()
        return rec

    def log_batch(self, job_id: str, records: list[dict]):
        """Bulk-insert multiple metric records (from callback stream)."""
        for r in records:
            rec = MetricRecord(
                id=str(uuid.uuid4()),
                tenant_id=self.tenant_id,
                job_id=job_id,
                epoch=r.get("epoch"),
                global_step=r.get("global_step") or r.get("_step"),
                loss=r.get("loss"),
                accuracy=r.get("accuracy"),
                gpu_mem_gb=r.get("gpu_mem_gb"),
                tokens_per_second=r.get("tokens_per_second"),
                learning_rate=r.get("learning_rate"),
                grad_norm=r.get("grad_norm"),
                extras_json=json.dumps({k: v for k, v in r.items()
                                        if k not in ("epoch", "_step", "loss",
                                                      "accuracy", "learning_rate")})
                if r else None,
            )
            self.session.add(rec)
        self.session.commit()

    def get_job_metrics(self, job_id: str) -> list[MetricRecord]:
        q = self.session.query(MetricRecord).filter_by(job_id=job_id)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.order_by(MetricRecord.global_step.asc()).all()

    def get_job_summary(self, job_id: str) -> dict:
        rows = self.get_job_metrics(job_id)
        if not rows:
            return {}
        losses = [r.loss for r in rows if r.loss is not None]
        return {
            "job_id": job_id,
            "epochs_logged": len(rows),
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "best_accuracy": max((r.accuracy for r in rows if r.accuracy is not None), default=None),
            "peak_gpu_mem_gb": max((r.gpu_mem_gb for r in rows if r.gpu_mem_gb is not None), default=None),
        }

    def export_to_json(self, job_id: str, filepath: str):
        rows = self.get_job_metrics(job_id)
        data = [
            {
                "epoch": r.epoch,
                "step": r.global_step,
                "loss": r.loss,
                "accuracy": r.accuracy,
                "gpu_mem_gb": r.gpu_mem_gb,
                "tokens_per_sec": r.tokens_per_second,
                "learning_rate": r.learning_rate,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def list_by_tenant(self, tenant_id: str) -> list:
        return self.session.query(MetricRecord).filter_by(tenant_id=tenant_id).order_by(MetricRecord.timestamp.desc()).all()

    def delete_by_tenant(self, tenant_id: str) -> int:
        count = self.session.query(MetricRecord).filter_by(tenant_id=tenant_id).delete()
        self.session.commit()
        return count

    def delete_before(self, cutoff) -> int:
        q = self.session.query(MetricRecord).filter(MetricRecord.timestamp < cutoff)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        count = q.delete()
        self.session.commit()
        return count
