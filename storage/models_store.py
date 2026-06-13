import uuid
from storage.database import get_session, ModelVersion


class ModelVersionStore:
    def __init__(self, tenant_id: str = None):
        self.session = get_session()
        self.tenant_id = tenant_id

    def create(self, base_model: str, job_id: str = None, method: str = None,
               finetuned_path: str = None, hf_repo_id: str = None, runtime_used: str = None) -> ModelVersion:
        mv = ModelVersion(
            id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            job_id=job_id,
            base_model=base_model,
            method=method,
            finetuned_path=finetuned_path,
            hf_repo_id=hf_repo_id,
            runtime_used=runtime_used,
        )
        self.session.add(mv)
        self.session.commit()
        return mv

    def get(self, version_id: str) -> ModelVersion:
        q = self.session.query(ModelVersion).filter_by(id=version_id)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.first()

    def update(self, version_id: str, **kwargs) -> ModelVersion:
        mv = self.get(version_id)
        if mv:
            for k, v in kwargs.items():
                setattr(mv, k, v)
            self.session.commit()
        return mv

    def list_by_job(self, job_id: str) -> list:
        q = self.session.query(ModelVersion).filter_by(job_id=job_id)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.order_by(ModelVersion.created_at.desc()).all()

    def list_by_tenant(self, tenant_id: str) -> list:
        return self.session.query(ModelVersion).filter_by(tenant_id=tenant_id).order_by(ModelVersion.created_at.desc()).all()

    def delete_by_tenant(self, tenant_id: str) -> int:
        count = self.session.query(ModelVersion).filter_by(tenant_id=tenant_id).delete()
        self.session.commit()
        return count

    def list_all(self, limit: int = 50) -> list:
        q = self.session.query(ModelVersion).order_by(ModelVersion.created_at.desc())
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.limit(limit).all()
