from sqlalchemy import select
from storage.database import get_async_sessionmaker, Conversation
from datetime import datetime, timezone
import json

class AsyncConversationStore:
    def __init__(self, tenant_id: str = None):
        self.async_session = get_async_sessionmaker()
        self.tenant_id = tenant_id

    async def create(self, goal: str) -> Conversation:
        async with self.async_session() as session:
            conv = Conversation(
                id=str(uuid.uuid4()),
                tenant_id=self.tenant_id,
                goal=goal,
                status="active"
            )
            session.add(conv)
            await session.commit()
            await session.refresh(conv)
            return conv

    async def add_message(self, conv_id: str, role: str, content: str, meta: dict = None):
        async with self.async_session() as session:
            result = await session.execute(select(Conversation).filter_by(id=conv_id))
            conv = result.scalar_one_or_none()
            if conv:
                msgs = conv.get_messages()
                msgs.append({"role": role, "content": content, "meta": meta, "timestamp": datetime.now(timezone.utc).isoformat()})
                conv.set_messages(msgs)
                conv.updated_at = datetime.now(timezone.utc)
                await session.commit()
