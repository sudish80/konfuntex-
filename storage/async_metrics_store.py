import uuid
from datetime import datetime, timezone
from sqlalchemy import select, delete
from storage.database import get_async_sessionmaker, MetricRecord
import json

class AsyncMetricsStore:
    def __init__(self, tenant_id: str = None):
        self.async_session = get_async_sessionmaker()
        self.tenant_id = tenant_id

    async def log_epoch(self, job_id: str, epoch: float = None,
                        global_step: int = None, loss: float = None,
                        accuracy: float = None, gpu_mem_gb: float = None,
                        tokens_per_second: float = None,
                        learning_rate: float = None,
                        grad_norm: float = None,
                        extras: dict = None) -> MetricRecord:
        async with self.async_session() as session:
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
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return rec

    async def get_job_metrics(self, job_id: str) -> list[MetricRecord]:
        async with self.async_session() as session:
            q = select(MetricRecord).filter_by(job_id=job_id).order_by(MetricRecord.global_step.asc())
            if self.tenant_id:
                q = q.filter_by(tenant_id=self.tenant_id)
            result = await session.execute(q)
            return result.scalars().all()
