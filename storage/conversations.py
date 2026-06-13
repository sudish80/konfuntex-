import uuid
from datetime import datetime, timezone
from storage.database import get_session, Conversation


class ConversationStore:
    def __init__(self, tenant_id: str = None):
        self.session = get_session()
        self.tenant_id = tenant_id

    def create(self, goal: str) -> Conversation:
        conv = Conversation(
            id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            goal=goal,
            status="active",
            messages_json="[]",
        )
        self.session.add(conv)
        self.session.commit()
        return conv

    def get(self, conv_id: str) -> Conversation:
        if not conv_id:
            return None
        q = self.session.query(Conversation).filter_by(id=conv_id)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.first()

    def add_message(self, conv_id: str, role: str, content: str, metadata: dict = None):
        conv = self.get(conv_id)
        if not conv:
            return None
        msgs = conv.get_messages()
        msgs.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })
        conv.set_messages(msgs)
        conv.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return conv

    def get_messages(self, conv_id: str) -> list:
        conv = self.get(conv_id)
        return conv.get_messages() if conv else []

    def close(self, conv_id: str, summary: str = None):
        conv = self.get(conv_id)
        if conv:
            conv.status = "closed"
            conv.summary = summary
            conv.updated_at = datetime.now(timezone.utc)
            self.session.commit()

    def list_active(self) -> list:
        q = self.session.query(Conversation).filter_by(status="active")
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.order_by(Conversation.updated_at.desc()).all()

    def list_all(self, limit: int = 50) -> list:
        q = self.session.query(Conversation).order_by(Conversation.updated_at.desc())
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        return q.limit(limit).all()

    def list_by_tenant(self, tenant_id: str) -> list:
        return self.session.query(Conversation).filter_by(tenant_id=tenant_id).order_by(Conversation.created_at.desc()).all()

    def delete_by_tenant(self, tenant_id: str) -> int:
        count = self.session.query(Conversation).filter_by(tenant_id=tenant_id).delete()
        self.session.commit()
        return count

    def delete_before(self, cutoff) -> int:
        q = self.session.query(Conversation).filter(Conversation.created_at < cutoff)
        if self.tenant_id:
            q = q.filter_by(tenant_id=self.tenant_id)
        count = q.delete()
        self.session.commit()
        return count
