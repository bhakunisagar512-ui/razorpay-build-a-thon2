"""
FastAPI application.

The product API is the revenue-risk pipeline (services/app/revenue_api.py: events, simulated payment requests, automated dataset runs,
Human Intervention) plus the testing console support routes (services/app/test_api.py: datasets, CSV, the dropdown options, the UI itself).
This module keeps what is shared:

  GET  /health        database reachability
  GET  /audit/verify  hash-chain verification of the append-only audit log (historical rows included)
  POST /shadow/*      Model C shadow mode: score and RECORD a payment attempt, never act on it
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from temporalio.client import Client

from services.app import shadow
from services.app.db import engine
from services.app.domain import ledger
from services.app.domain.taxonomy import Cause
from services.app.settings import settings

app = FastAPI(title="rzp-recovery", version="0.2.0")

_engine = engine()
_temporal: Client | None = None


async def temporal() -> Client:
    global _temporal
    if _temporal is None:
        _temporal = await Client.connect(settings.temporal_address)
    return _temporal


# ------------------------------------------------------------------------------------------------- Model C shadow mode
class ShadowEvent(BaseModel):
    """One payment attempt as the gateway would report it. There is deliberately no cause/label field."""
    model_config = {"extra": "forbid"}
    payment_id: str
    order_id: str | None = None
    customer_id: str | None = None
    issuer_id: str | None = None
    psp_id: str | None = None
    instrument_id: str | None = None
    amount_minor: int = Field(default=0, ge=0)
    method: str = "card"
    occurred_at: datetime
    observed_at: datetime | None = None
    status: str = "failed"
    instrument_hint: str = "na"
    gw_error_code: str = ""
    gw_error_source: str = ""
    gw_error_step: str = ""
    latency_ms: float | None = None
    error_reason: str | None = None
    payment_flow: str | None = None
    attempt_duration_s: float | None = None
    diagnose: bool = True      # False: history only (e.g. warm-up failures)
    observe: bool = True       # False: score without adding to history (telemetry not visible)


class ShadowDowntime(BaseModel):
    model_config = {"extra": "forbid"}
    event_id: str
    kind: Literal["started", "resolved"]
    at: datetime
    method: str | None = None
    severity: str | None = None
    issuer_id: str | None = None
    psp_id: str | None = None


@app.post("/shadow/event")
def shadow_event(ev: ShadowEvent):
    """Shadow mode: score a failed attempt with Model C (features built from prior events only), then add it to
    history. Records the prediction; triggers no recovery action."""
    data = ev.model_dump()
    flags = {k: data.pop(k) for k in ("diagnose", "observe")}
    rec = shadow.process(shadow.make_observation(**data), **flags)
    return rec or {"payment_id": ev.payment_id, "status": "history_updated"}


@app.post("/shadow/downtime")
def shadow_downtime(n: ShadowDowntime):
    from services.app.domain.features_v4 import DowntimeNotice
    return shadow.notify(DowntimeNotice(**n.model_dump()))


@app.put("/shadow/label/{payment_id}")
def shadow_label(payment_id: str, actual_cause: Cause):
    """Evaluation only: attach the synthetic ground truth to the stored record. Never read by the model."""
    return {"payment_id": payment_id, "attached": shadow.attach_label(payment_id, str(actual_cause))}


@app.get("/shadow/diagnoses")
def shadow_diagnoses(limit: int = 100, payment_id: str | None = None):
    if payment_id:
        rec = shadow.get(payment_id)
        if rec is None:
            raise HTTPException(404, "no shadow record")
        return rec
    return shadow.recent(min(limit, 1000))


# ------------------------------------------------------------------------------------------------- health and audit
@app.get("/audit/verify")
def audit_verify():
    with _engine.begin() as conn:
        return ledger.verify_chain(conn)


@app.get("/health")
def health():
    with _engine.begin() as conn:
        conn.execute(text("SELECT 1"))
    return {"ok": True, "demo_mode": settings.demo_mode}


from services.app.test_api import router as _console_router    # noqa: E402  console support: datasets, CSV, dropdown options, the UI itself
from services.app.revenue_api import router as _revenue_router  # noqa: E402  the revenue-risk pipeline
app.include_router(_console_router)
app.include_router(_revenue_router)
