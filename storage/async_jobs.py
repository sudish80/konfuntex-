import uuid
from datetime import datetime, timezone
from sqlalchemy import select, delete
from storage.database import get_async_sessionmaker, Job


class AsyncJobStore:
    def __init__(self, tenant_id: str = None):
        self.async_session = get_async_sessionmaker()
        self.tenant_id = tenant_id

    async def create(self, goal: str, method: str = None, base_model: str = None, dataset: str = None,
                     runtime: str = None, conversation_id: str = None) -> Job:
        async with self.async_session() as session:
            job = Job(
                id=str(uuid.uuid4()),
                tenant_id=self.tenant_id,
                goal=goal,
                method=method,
                base_model=base_model,
                dataset=dataset,
                runtime=runtime,
                conversation_id=conversation_id,
                status="pending",
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    async def get(self, job_id: str) -> Job:
        async with self.async_session() as session:
            q = select(Job).filter_by(id=job_id)
            if self.tenant_id:
                q = q.filter_by(tenant_id=self.tenant_id)
            result = await session.execute(q)
            return result.scalar_one_or_none()

    async def update(self, job_id: str, **kwargs) -> Job:
        async with self.async_session() as session:
            q = select(Job).filter_by(id=job_id)
            if self.tenant_id:
                q = q.filter_by(tenant_id=self.tenant_id)
            result = await session.execute(q)
            job = result.scalar_one_or_none()
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                job.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await session.refresh(job)
            return job
