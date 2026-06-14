from sqlalchemy import select
from storage.database import get_async_sessionmaker, RuntimeLog
from datetime import datetime, timezone

class AsyncRuntimeLogStore:
    def __init__(self, tenant_id: str = None):
        self.async_session = get_async_sessionmaker()
        self.tenant_id = tenant_id

    async def log_switch(self, job_id: str, runtime_type: str, switched_from: str, 
                         switched_to: str, reason: str) -> RuntimeLog:
        async with self.async_session() as session:
            log = RuntimeLog(
                id=str(uuid.uuid4()),
                tenant_id=self.tenant_id,
                job_id=job_id,
                runtime_type=runtime_type,
                switched_from=switched_from,
                switched_to=switched_to,
                switch_reason=reason,
                status="active",
                started_at=datetime.now(timezone.utc)
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log
