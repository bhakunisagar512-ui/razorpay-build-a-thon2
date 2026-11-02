"""Console support routes: the prebuilt datasets and the CSV template / schema / validation, the Custom Testing dropdown options, and the single-page
console itself (/ui). The recovery pipeline is services/app/revenue_api.py; there is no route here that diagnoses, plans or acts on a payment.
Everything is synthetic. DEMO_MODE only where a route changes state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from services.app import test_datasets as tds
from services.app.domain import custom_input_rules as rules
from services.app.revenue import routing

router = APIRouter()


class OptionsRequest(BaseModel):
    """What the tester has selected so far. A field that is absent is not chosen; error_reason null means 'no reason provided'."""
    method: str | None = None
    instrument_hint: str | None = None
    gw_error_code: str | None = None
    gw_error_source: str | None = None
    gw_error_step: str | None = None
    error_reason: str | None = None
    true_cause: str | None = None
    changed: str | None = Field(default=None, description='the field the tester just changed; its value is kept, the others are reconciled around it')
    scope: Literal['all', 'clear', 'generic', 'conflict'] = Field(default='all', description="which inputs can be built: 'clear' = clear and consistent, 'generic' = missing / generic (Model C route), 'conflict' = any vocabulary value, so contradictions can be built")


@router.post('/tests/custom-options')
def custom_options(req: OptionsRequest):
    """Read-only. The valid options of every custom-payment dropdown for the current selections, the selections that had to be
    cleared (never replaced), and why each withheld option is unavailable. Same rules as the validation of POST /revenue/events."""
    selected = {f: getattr(req, f) for f in req.model_fields_set if f not in ('changed', 'scope') and (f == 'error_reason' or getattr(req, f) is not None)}
    out = rules.build_options(selected, req.changed, req.scope)
    if out['complete']:                                                      # a read-only preview of the router's verdict on these inputs; Model C is NOT called here
        a = routing.assess({k: out['selected'].get(k) for k in routing.INFO_FIELDS})
        out['route_preview'] = a.as_dict()
    return out




@router.get('/tests/datasets')
def datasets():
    return {'datasets': tds.list_datasets(), 'csv': {'schema': '/tests/csv-schema', 'template': '/tests/csv-template',
                                                     'validate': 'POST /tests/csv/validate (text/csv body)'}}


@router.get('/tests/datasets/{name}/payments')
def dataset_payments(name: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), method: str | None = None,
                     error_code: str | None = None, search: str | None = None):
    try:
        return tds.dataset_payments(name, offset, limit, method, error_code, search)
    except KeyError:
        raise HTTPException(404, f'unknown dataset {name!r}')
    except FileNotFoundError:
        raise HTTPException(404, f'dataset {name!r} has not been built')


@router.get('/tests/csv-template')
def csv_template():
    return PlainTextResponse(tds.csv_template(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="payment_failures_template.csv"'})


@router.get('/tests/csv-schema')
def csv_schema():
    return tds.csv_schema()


@router.post('/tests/csv/validate')
async def csv_validate(request: Request):
    raw = await request.body()
    if len(raw) > tds.MAX_BYTES:
        raise HTTPException(413, f'file is larger than {tds.MAX_BYTES // (1024 * 1024)} MB')
    try:
        text_ = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise HTTPException(400, 'the file must be UTF-8 text')
    return tds.validate_csv(text_)


STATIC = Path(__file__).parent / 'static'
ASSETS = {'app.css': 'text/css; charset=utf-8', 'app.js': 'application/javascript; charset=utf-8'}
UI_HEADERS = {'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff',
              'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"}


@router.get('/ui', include_in_schema=False)
def ui_index():
    """The single-page testing console (HTML + CSS + vanilla JS, no build step)."""
    return HTMLResponse((STATIC / 'index.html').read_text(encoding='utf-8'), headers=UI_HEADERS)


@router.get('/ui/', include_in_schema=False)
def ui_slash():
    return RedirectResponse('/ui')


@router.get('/ui/assets/{name}', include_in_schema=False)
def ui_asset(name: str):
    if name not in ASSETS:                                   # whitelist: no path traversal, nothing else is served
        raise HTTPException(404, 'unknown asset')
    return Response((STATIC / name).read_text(encoding='utf-8'), media_type=ASSETS[name], headers=UI_HEADERS)
