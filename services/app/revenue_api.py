"""HTTP routes of the revenue-risk pipeline (four event types, interactive simulated payment requests, automated dataset runs).

Everything is synthetic: there is no real payment gateway, no real message and no real money. DEMO_MODE only.
Custom events accept controlled values only (extra fields are rejected, every value must come from the catalog); ids are generated here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from services.app.revenue import catalog as C, fixtures, routing, runs, service as svc, store
from services.app.settings import settings
from services.app.db import engine

router = APIRouter()
STATIC = Path(__file__).parent / 'static'
PAGE_HEADERS = {'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"}


def _demo():
    if not settings.demo_mode:
        raise HTTPException(403, 'DEMO_MODE is off')


def _422(errors):
    raise HTTPException(422, [{'loc': ['body', e['field']], 'msg': e['message'], 'type': 'validation'} for e in errors])


class CustomEvent(BaseModel):
    """Controlled event inputs. No free text: every value is validated against the catalog or the payment-failure compatibility rules."""
    model_config = {'extra': 'forbid'}
    event_type: str
    customer_id: str | None = None
    amount_rupees: int | None = None
    failure_time: str | None = None
    checkout_stage: str | None = None
    plan_id: str | None = None
    method: str | None = None
    failure_kind: str | None = None
    invoice_id: str | None = None
    issuer_id: str | None = None
    instrument_hint: str | None = None
    gw_error_code: str | None = None
    gw_error_source: str | None = None
    gw_error_step: str | None = None
    error_reason: str | None = None
    diagnosis_scenario: Literal['clear', 'generic', 'contradictory'] | None = Field(default=None, description='which inputs the form could build; never overrides the router')
    evaluation_label: str | None = Field(default=None, description='HIDDEN label for offline comparison; never used by diagnosis or planning')


class DatasetEvent(BaseModel):
    dataset: str
    payment_id: str


class Response_(BaseModel):
    decision: Literal['ACCEPT', 'REJECT']


async def _launch(event_id: str, responder: str = 'manual', timeout_s: int | None = None):
    try:
        await svc.start_workflow(event_id, responder=responder, response_timeout_s=timeout_s)
        return None
    except Exception as exc:                                                    # noqa: BLE001
        return f'{type(exc).__name__}: could not reach Temporal'


def _view(event_id: str, reveal_label: bool = False):
    with engine().begin() as conn:
        return store.event_view(conn, event_id, reveal_label=reveal_label)


async def _create(spec: dict, source: str, reveal: bool, timeout_s: int | None = None):
    out = await asyncio.to_thread(svc.ingest, engine(), spec, source)
    warning = None
    if out['route'] != routing.CONTRADICTORY:
        warning = await _launch(spec['event_id'], 'manual', timeout_s)
    view = _view(spec['event_id'], reveal)
    return {**view, **({'workflow_warning': warning} if warning else {})}


@router.get('/revenue/catalog')
def catalog():
    return {**C.catalog_view(), 'datasets': runs.dataset_options()}


@router.post('/revenue/events', status_code=201)
async def create_event(req: CustomEvent, timeout_s: int | None = Query(None, ge=3, le=3600, description='demo timeout: seconds to wait for ACCEPT / REJECT on each payment request')):
    _demo()
    errors = svc.validate_custom(req.event_type, req.model_dump(exclude={'event_type'}))
    if errors:
        _422(errors)
    return await _create(svc.custom_spec(req.event_type, req.model_dump()), 'custom', True, timeout_s)


@router.post('/revenue/events/from-dataset', status_code=201)
async def create_dataset_event(req: DatasetEvent, timeout_s: int | None = Query(None, ge=3, le=3600)):
    """One payment failure from a prebuilt dataset or the routing fixtures, through the same pipeline (interactive: you answer ACCEPT / REJECT)."""
    _demo()
    try:
        if req.dataset == fixtures.NAME:
            rec = fixtures.by_id().get(req.payment_id)
            if rec is None:
                raise KeyError(req.payment_id)
            return await _create(svc.fixture_spec(rec), 'fixture', False, timeout_s)
        return await _create(svc.dataset_spec(req.dataset, req.payment_id), 'dataset', False, timeout_s)
    except KeyError:
        raise HTTPException(404, f'{req.payment_id!r} is not in dataset {req.dataset!r}')
    except FileNotFoundError:
        raise HTTPException(404, f'dataset {req.dataset!r} has not been built')


@router.get('/revenue/events/{event_id}')
def get_event(event_id: str):
    view = _view(event_id, True)
    if view is None:
        raise HTTPException(404, 'unknown event')
    return view


@router.get('/revenue/requests')
def pending_requests(limit: int = Query(50, ge=1, le=200)):
    with engine().begin() as conn:
        return {'pending': store.pending_requests(conn, limit)}


@router.get('/revenue/requests/{request_id}')
def get_request(request_id: str):
    with engine().begin() as conn:
        view = store.request_view(conn, request_id)
    if view is None:
        raise HTTPException(404, 'unknown payment request')
    return view


@router.post('/revenue/requests/{request_id}/respond')
async def respond(request_id: str, body: Response_):
    """ACCEPT = the simulated customer completed the requested payment. REJECT = they did not. Resolves a request at most once."""
    _demo()
    with engine().begin() as conn:
        res = store.resolve_request(conn, request_id, body.decision, 'tester')
    if not res['found']:
        raise HTTPException(404, 'unknown payment request')
    wanted = 'ACCEPTED' if body.decision == 'ACCEPT' else 'REJECTED'
    if not res['changed']:
        if res['status'] == wanted:                                                    # the same answer twice: harmless, nothing is counted again
            return {'request_id': request_id, 'status': res['status'], 'changed': False, 'duplicate': True}
        raise HTTPException(409, f"this request is already {res['status']}" + ('; the answer came too late' if res.get('late') else ''))
    notified = True
    try:
        from services.app import main
        client = await asyncio.wait_for(main.temporal(), 3)
        from services.app.workflows.revenue import RevenueRecoveryWorkflow
        await client.get_workflow_handle(f"revenue-{res['event_id']}").signal(RevenueRecoveryWorkflow.payment_response, args=[request_id, body.decision])
    except Exception:                                                                  # noqa: BLE001  the database already holds the answer; the workflow settles from it
        notified = False
    return {'request_id': request_id, 'status': res['status'], 'changed': True, 'workflow_notified': notified, 'event_id': res['event_id']}


@router.get('/revenue/fixtures/payments')
def fixture_payments(offset: int = Query(0, ge=0), limit: int = Query(12, ge=1, le=200), method: str | None = None, route: str | None = None):
    """The controlled routing fixtures, listed like a dataset. The hidden cause is never listed; the intended route is a fixture description."""
    rows = [r for r in fixtures.records() if (not method or r['method'] == method) and (not route or r['expected_route'] == route)]
    items = [{'payment_id': r['payment_id'], 'method': r['method'], 'gw_error_code': r['gw_error_code'], 'gw_error_source': r['gw_error_source'], 'gw_error_step': r['gw_error_step'],
              'error_reason': r['error_reason'], 'instrument_hint': r['instrument_hint'], 'amount_minor_raw': r['amount_minor'], 'issuer_id': r['issuer_id'],
              'fixture_route': r['expected_route']} for r in rows[offset:offset + limit]]
    return {'dataset': fixtures.NAME, 'total': len(rows), 'offset': offset, 'limit': limit, 'items': items}


# ------------------------------------------------------------------------------------------------- the separate simulated payment page
@router.get('/pay/assets/{name}', include_in_schema=False)
def pay_asset(name: str):
    kinds = {'pay.js': 'application/javascript; charset=utf-8'}
    if name not in kinds:
        raise HTTPException(404)
    return Response((STATIC / name).read_bytes(), media_type=kinds[name], headers=PAGE_HEADERS)


@router.get('/pay/{request_id}', include_in_schema=False)
def pay_page(request_id: str):
    return HTMLResponse((STATIC / 'pay.html').read_text(encoding='utf-8'), headers=PAGE_HEADERS)


# ------------------------------------------------------------------------------------------------- automated dataset runs (Phase 9)
class RunRequest(BaseModel):
    dataset: str | None = Field(default=None, description='a prebuilt dataset, routing_fixtures or event_type_mix')
    csv_id: str | None = Field(default=None, description='an uploaded and validated CSV (from POST /tests/csv/validate)')
    limit: int | None = Field(default=None, ge=1, le=5000)
    temporal_limit: int = Field(default=0, ge=0, le=50, description='also run the first N cases through Temporal and compare with the in-process result')


@router.post('/revenue/runs', status_code=202)
def start_run(req: RunRequest):
    _demo()
    if bool(req.dataset) == bool(req.csv_id):
        raise HTTPException(422, 'give exactly one of dataset or csv_id')
    try:
        return {'run_id': runs.start(req.dataset or f'csv:{req.csv_id}', req.limit, req.temporal_limit), 'status': 'queued'}
    except runs.Busy as busy:
        raise HTTPException(409, f'another revenue run is in progress ({busy.run_id}); only one at a time')
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except FileNotFoundError:
        raise HTTPException(404, 'dataset has not been built')


@router.get('/revenue/runs')
def list_runs(limit: int = Query(20, ge=1, le=100)):
    return {'runs': runs.list_runs(limit)}


@router.get('/revenue/runs/{run_id}')
def get_run(run_id: str):
    doc = runs.get_run(run_id)
    if doc is None:
        raise HTTPException(404, 'unknown run')
    return doc


@router.get('/revenue/runs/{run_id}/cases')
def run_cases(run_id: str, route: str | None = None, status: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    out = runs.cases(run_id, route, status, offset, limit)
    if out is None:
        raise HTTPException(404, 'unknown or unfinished run')
    return out


# ------------------------------------------------------------------------------------------------- Human Intervention (read-only)
@router.get('/interventions')
def interventions(kind: str | None = None, dataset: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
    """Events a person must look at: contradictory gateway data, risk blocks, no safe action, unresolved diagnoses. There is no route that acts on them."""
    if kind is not None and kind not in store.INTERVENTION_KINDS:
        raise HTTPException(422, f'kind must be one of {", ".join(store.INTERVENTION_KINDS)}')
    with engine().begin() as conn:
        items = store.intervention_items(conn, kind, dataset)
        everything = store.intervention_items(conn, None, dataset)
    return {'read_only': True, 'total': len(items), 'counts_by_kind': {k: sum(1 for i in everything if i['kind'] == k) for k in store.INTERVENTION_KINDS},
            'offset': offset, 'limit': limit, 'items': items[offset:offset + limit]}


@router.get('/interventions/historical')
def historical_interventions(limit: int = Query(100, ge=1, le=500)):
    """Read-only: cases from the retired three-cycle research architecture (table intervention_case). Kept as records; nothing here can be acted on."""
    from sqlalchemy import text
    with engine().begin() as conn:
        rows = conn.execute(text("SELECT payment_id, kind, reason, details, created_at FROM intervention_case ORDER BY created_at DESC LIMIT :n"), {'n': limit}).mappings().all()
        total = conn.execute(text("SELECT count(*) FROM intervention_case")).scalar()
    return {'read_only': True, 'total': total, 'note': 'Records from the earlier research architecture (up to three diagnosis cycles). They are not part of the current pipeline and cannot be acted on.',
            'items': [{**dict(r), 'created_at': r['created_at'].isoformat()} for r in rows]}


@router.get('/interventions/revenue/{event_id}')
def intervention_detail(event_id: str):
    from sqlalchemy import text
    with engine().begin() as conn:
        row = conn.execute(text("SELECT kind, reason, details, created_at FROM revenue_intervention WHERE event_id=:e"), {'e': event_id}).mappings().first()
        view = store.event_view(conn, event_id) if row else None
    if row is None:
        raise HTTPException(404, 'no revenue intervention case for this event')
    return {'read_only': True, 'source': 'revenue', 'id': event_id, 'kind': row['kind'], 'reason': row['reason'], 'reason_text': store.REASON_TEXT.get(row['reason'], row['reason']),
            'created_at': row['created_at'], 'details': row['details'], 'event': view, 'actions_available': []}
