"""FastAPI application for agent health, metrics, and budget endpoints.

Includes:
  - RBAC via multiple API keys (admin / read-only)
  - Rate limiting (in-memory token bucket, per-IP)
  - CORS (configurable origins)
  - Optional Sentry integration
  - WebSocket auth via query parameter
  - API versioning under /v1/
"""

import asyncio
import os
import time
import logging
import threading
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from agent.core import health_endpoint, metrics_endpoint, budget_endpoint
from agent.events import get_event_bus
from agent.observability import setup_json_logging
from config.settings import settings


logger = logging.getLogger(__name__)

# ── Env constants ────────────────────────────────────────────────────

_ENV = settings.env
_API_KEY = os.environ.get("COLAB_AGENT_API_KEY") or settings.api_key or ""
_ADMIN_KEY = os.environ.get("COLAB_AGENT_ADMIN_KEY") or _API_KEY
_READONLY_KEY = os.environ.get("COLAB_AGENT_READONLY_KEY") or ""


class Role(str, Enum):
    ADMIN = "admin"
    READONLY = "readonly"
    NONE = "none"


def _key_role(key: str) -> Role:
    if key and key == _ADMIN_KEY:
        return Role.ADMIN
    if key and key == _READONLY_KEY:
        return Role.READONLY
    if key and key == _API_KEY and not _ADMIN_KEY:
        return Role.ADMIN
    if not key and not _ADMIN_KEY and not _READONLY_KEY:
        return Role.ADMIN  # no auth configured → unrestricted
    return Role.NONE


# ── Auth (HTTP Bearer) ───────────────────────────────────────────────

_SECURITY = HTTPBearer(auto_error=False)


async def verify_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_SECURITY)):
    token = credentials.credentials if credentials else ""
    role = _key_role(token)
    if role == Role.NONE:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return credentials


def require_role(min_role: Role):
    """Dependency factory — rejects requests below *min_role*."""
    async def _check(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_SECURITY)):
        token = credentials.credentials if credentials else ""
        role = _key_role(token)
        if role == Role.NONE:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        if role == Role.READONLY and min_role == Role.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        return role
    return _check


# ── WebSocket auth helper ────────────────────────────────────────────

async def _ws_auth(websocket: WebSocket) -> Role:
    token = websocket.query_params.get("token", "")
    role = _key_role(token)
    if _ENV == "prod" and role == Role.NONE:
        await websocket.close(code=4001, reason="Missing or invalid API key")
    return role


# ── Rate limiter ─────────────────────────────────────────────────────

class RateLimiter:
    """In-memory token bucket rate limiter, per-IP."""

    def __init__(self, rate: int = 60, burst: int = 10):
        self._rate = rate
        self._burst = burst
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            last, tokens = self._buckets.get(ip, (now, self._burst))
            elapsed = now - last
            tokens = min(self._burst, tokens + elapsed * (self._rate / 60))
            if tokens < 1:
                return False
            tokens -= 1
            self._buckets[ip] = (now, tokens)
            return True


_rate_limiter = RateLimiter(
    rate=int(os.environ.get("COLAB_AGENT_RATE_LIMIT", "60")),
    burst=int(os.environ.get("COLAB_AGENT_RATE_LIMIT_BURST", "10")),
)


async def check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.check(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ── Sentry ───────────────────────────────────────────────────────────

def _init_sentry():
    dsn = os.environ.get("COLAB_AGENT_SENTRY_DSN")
    if dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=dsn,
                environment=_ENV,
                traces_sample_rate=float(os.environ.get("COLAB_AGENT_SENTRY_TRACES_RATE", "0.1")),
            )
            logger.info("Sentry SDK initialized")
        except ImportError:
            logger.warning("sentry_sdk not installed, skipping Sentry setup")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")


# ── CORS ─────────────────────────────────────────────────────────────

_cors_origins = os.environ.get("COLAB_AGENT_CORS_ORIGINS", "*")


# ── App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_sentry()
    setup_json_logging(logger)
    logger.info("FastAPI agent service starting (env=%s, cors=%s)", _ENV, _cors_origins)
    yield
    logger.info("FastAPI agent service stopped")


app = FastAPI(
    title="Colab Agent API",
    version="1.0.0",
    description="Autonomous agent for fine-tuning HuggingFace models in Google Colab",
    lifespan=lifespan,
    docs_url="/docs" if _ENV != "prod" else None,
    redoc_url="/redoc" if _ENV != "prod" else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins.split(",") if _cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    circuit_state: str
    circuit_failures: int
    total_retries: int
    runtime_switches: int
    current_runtime: str
    uptime_hours: float
    active_job_id: Optional[str] = None
    active_conversation_id: Optional[str] = None
    error: str = ""
    version: str


class BudgetResponse(BaseModel):
    max_cost_units: float
    spent: float
    remaining: float
    usage_pct: float
    exceeded: bool
    alert_count: int


class RootResponse(BaseModel):
    service: str
    version: str
    env: str


class GdprExportResponse(BaseModel):
    tenant_id: str
    jobs: int
    conversations: int
    model_versions: int
    metric_records: int
    data: dict


class GdprDeleteResponse(BaseModel):
    tenant_id: str
    deleted_jobs: int
    deleted_conversations: int
    deleted_model_versions: int
    deleted_metric_records: int


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/v1/", response_model=RootResponse,
         dependencies=[Depends(check_rate_limit)])
async def root():
    return RootResponse(service="colab-agent", version="1.0.0", env=_ENV)


@app.get("/v1/health", response_model=HealthResponse,
         dependencies=[Depends(check_rate_limit)])
async def health():
    return health_endpoint()


@app.get("/v1/metrics", response_class=PlainTextResponse,
         dependencies=[Depends(check_rate_limit)])
async def metrics():
    return metrics_endpoint()


@app.get("/v1/budget", response_model=BudgetResponse,
         dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def budget():
    return budget_endpoint()


# ── GDPR ─────────────────────────────────────────────────────────────

def _gdpr_data(tenant_id: str) -> dict:
    from storage.jobs import JobStore
    from storage.conversations import ConversationStore
    from storage.models_store import ModelVersionStore
    from storage.metrics_store import MetricsStore
    jobs = JobStore().list_by_tenant(tenant_id)
    convs = ConversationStore().list_by_tenant(tenant_id)
    models = ModelVersionStore().list_by_tenant(tenant_id)
    metrics = MetricsStore().list_by_tenant(tenant_id)
    return {
        "jobs": [{"id": j.id, "goal": j.goal, "status": j.status,
                   "created_at": str(j.created_at)} for j in jobs],
        "conversations": [{"id": c.id, "goal": c.goal, "status": c.status,
                           "messages": c.get_messages()[:50]} for c in convs],
        "model_versions": [{"id": m.id, "base_model": m.base_model,
                            "method": m.method} for m in models],
        "metric_records": [{"id": m.id, "job_id": m.job_id, "loss": m.loss,
                            "epoch": m.epoch} for m in metrics],
    }


@app.post("/v1/gdpr/export/{tenant_id}",
          response_model=GdprExportResponse,
          dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def gdpr_export(tenant_id: str):
    from storage.jobs import JobStore
    from storage.conversations import ConversationStore
    from storage.models_store import ModelVersionStore
    from storage.metrics_store import MetricsStore
    jobs = JobStore().list_by_tenant(tenant_id)
    convs = ConversationStore().list_by_tenant(tenant_id)
    models = ModelVersionStore().list_by_tenant(tenant_id)
    metrics = MetricsStore().list_by_tenant(tenant_id)
    data = _gdpr_data(tenant_id)
    return GdprExportResponse(
        tenant_id=tenant_id,
        jobs=len(jobs),
        conversations=len(convs),
        model_versions=len(models),
        metric_records=len(metrics),
        data=data,
    )


@app.post("/v1/gdpr/delete/{tenant_id}",
          response_model=GdprDeleteResponse,
          dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def gdpr_delete(tenant_id: str):
    from storage.jobs import JobStore
    from storage.conversations import ConversationStore
    from storage.models_store import ModelVersionStore
    from storage.metrics_store import MetricsStore
    d_jobs = JobStore().delete_by_tenant(tenant_id)
    d_convs = ConversationStore().delete_by_tenant(tenant_id)
    d_models = ModelVersionStore().delete_by_tenant(tenant_id)
    d_metrics = MetricsStore().delete_by_tenant(tenant_id)
    return GdprDeleteResponse(
        tenant_id=tenant_id,
        deleted_jobs=d_jobs,
        deleted_conversations=d_convs,
        deleted_model_versions=d_models,
        deleted_metric_records=d_metrics,
    )


# ── Legacy unversioned routes (deprecated) ───────────────────────────

@app.get("/", response_model=RootResponse,
         dependencies=[Depends(check_rate_limit)])
async def root_deprecated():
    return RootResponse(service="colab-agent", version="1.0.0", env=_ENV)


@app.get("/health", response_model=HealthResponse,
         dependencies=[Depends(check_rate_limit)])
async def health_deprecated():
    return health_endpoint()


@app.get("/metrics", response_class=PlainTextResponse,
         dependencies=[Depends(check_rate_limit)])
async def metrics_deprecated():
    return metrics_endpoint()


@app.get("/budget", response_model=BudgetResponse,
         dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def budget_deprecated():
    return budget_endpoint()


@app.post("/gdpr/export/{tenant_id}", response_model=GdprExportResponse,
          dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def gdpr_export_deprecated(tenant_id: str):
    return await gdpr_export(tenant_id)


@app.post("/gdpr/delete/{tenant_id}", response_model=GdprDeleteResponse,
          dependencies=[Depends(check_rate_limit), Depends(require_role(Role.ADMIN))])
async def gdpr_delete_deprecated(tenant_id: str):
    return await gdpr_delete(tenant_id)


# ── WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/v1/ws/{job_id}")
async def ws_job_events(websocket: WebSocket, job_id: str):
    role = await _ws_auth(websocket)
    if role == Role.NONE:
        return

    await websocket.accept()

    bus = get_event_bus()
    last_index = 0

    try:
        while True:
            events, last_index = bus.poll(job_id, last_index)
            for event in events:
                await websocket.send_json(event)

            if not events:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.5,
                    )
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)


@app.websocket("/ws/{job_id}")
async def ws_job_events_deprecated(websocket: WebSocket, job_id: str):
    await ws_job_events(websocket, job_id)
