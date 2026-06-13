from sqlalchemy import select
from storage.database import get_async_sessionmaker, ModelVersion

class AsyncModelVersionStore:
    def __init__(self, tenant_id: str = None):
        self.async_session = get_async_sessionmaker()
        self.tenant_id = tenant_id

    async def get(self, version_id: str) -> ModelVersion:
        async with self.async_session() as session:
            q = select(ModelVersion).filter_by(id=version_id)
            if self.tenant_id:
                q = q.filter_by(tenant_id=self.tenant_id)
            result = await session.execute(q)
            return result.scalar_one_or_none()
