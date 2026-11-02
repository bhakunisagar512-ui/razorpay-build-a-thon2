"""Activities of RevenueRecoveryWorkflow. All I/O lives here so the workflow stays deterministic. None of these diagnoses anything:
the diagnosis was made once, before the workflow started, and is already stored on the event."""
from datetime import datetime

from sqlalchemy import text
from temporalio import activity

from services.app.revenue import catalog as C, engine as E, store
from services.app.db import engine


@activity.defn
async def rv_start(event_id: str, workflow_id: str) -> dict:
    with engine().begin() as conn:
        return store.start(conn, event_id, workflow_id)


@activity.defn
async def rv_execute_step(event_id: str, idx: int, now_iso: str, response_timeout_s: int) -> dict:
    with engine().begin() as conn:
        return store.execute_step(conn, event_id, idx, datetime.fromisoformat(now_iso), response_timeout_s)


@activity.defn
async def rv_simulate_response(event_id: str, request_id: str) -> str:
    """Dataset Testing: the independent simulator answers instead of a person. It may read the hidden cause; nothing else may."""
    with engine().begin() as conn:
        ev = store._lock_event(conn, event_id)
        req = conn.execute(text("SELECT step_idx, action FROM payment_request WHERE request_id=:r"), {'r': request_id}).mappings().first()
        hidden = store.hidden_cause(conn, event_id) or (ev['payload'].get('failure_kind') if ev['event_type'] == C.SUBSCRIPTION else None)
        decision = E.simulate(ev['event_type'], req['action'], ev['payload'].get('sim_key', event_id), req['step_idx'], hidden)
        return store.resolve_request(conn, request_id, 'ACCEPT' if decision == 'ACCEPTED' else 'REJECT', 'simulator')['status']


@activity.defn
async def rv_settle_request(request_id: str) -> str:
    with engine().begin() as conn:
        return store.settle_request(conn, request_id)


@activity.defn
async def rv_finish(event_id: str, mode: str) -> str:
    with engine().begin() as conn:
        return (store.finish_exhausted(conn, event_id) if mode == 'exhausted' else store.finish(conn, event_id, C.RECOVERED, 'accepted'))['status']


ACTIVITIES = [rv_start, rv_execute_step, rv_simulate_response, rv_settle_request, rv_finish]
