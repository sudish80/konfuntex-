import uuid
from datetime import datetime, timezone
from storage.database import get_session, Job


class JobStore:
    def __init__(self, tenant_id: str = None):
        self.session = get_session()
        self.tenant_id = tenant_id

    def create(self, goal: str, method: str = None, base_model: str = None, dataset: str = None,
               runtime: str = None, conversation_id: str = None) -> Job:
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
        self.session.add(job)
        self.session.commit()
        return job

    def get(self, job_id: str) -> Job:
        q = self.session.query(Job).filter_by(id=job_id)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.first()

    def update(self, job_id: str, **kwargs) -> Job:
        job = self.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = datetime.now(timezone.utc)
            self.session.commit()
        return job

    def list(self, status: str = None, limit: int = 50) -> list:
        q = self.session.query(Job).order_by(Job.created_at.desc())
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        if status:
            q = q.filter_by(status=status)
        return q.limit(limit).all()

    def delete(self, job_id: str):
        job = self.get(job_id)
        if job:
            self.session.delete(job)
            self.session.commit()

    def list_by_tenant(self, tenant_id: str) -> list:
        return self.session.query(Job).filter_by(tenant_id=tenant_id).order_by(Job.created_at.desc()).all()

    def delete_by_tenant(self, tenant_id: str) -> int:
        count = self.session.query(Job).filter_by(tenant_id=tenant_id).delete()
        self.session.commit()
        return count

    def delete_before(self, cutoff) -> int:
        q = self.session.query(Job).filter(Job.created_at < cutoff)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        count = q.delete()
        self.session.commit()
        return count
