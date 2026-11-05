"""Upload format for revenue events: one CSV row = one revenue-risk event, in the same controlled-input fields as `POST /revenue/events`.

This is the format of eval/data/demo_203/entries.csv (all four event types). It is a SEPARATE upload from the payment-failure CSV of Dataset Testing
(`/tests/csv/validate`): nothing is forced into that schema and no field is invented. Each row is validated by the same rules the Custom form uses
(`service.validate_custom`); a payment failure then takes its route from the router like any other event.

Columns: `event_type` (required) plus the fields of that type (see service.REQUIRED). Optional: `entry_id`, `scenario` (a label for the report),
`diagnosis_scenario` (clear / generic / contradictory: which gateway inputs the row may carry; when absent any vocabulary value is accepted and the router
decides), `evaluation_label` (hidden label, simulator only). Columns the file may carry for its own bookkeeping (`expected_*`, `customer_behaviour`, `purpose`)
are ignored: the simulated customer's answers come from the run's acceptance rate, not from the file.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path

from services.app import test_datasets as tds
from services.app.revenue import catalog as C, service as svc

MAX_ROWS, MAX_BYTES, MAX_ERRORS = 5000, 2 * 1024 * 1024, 50
FIELDS = sorted({f for fields in svc.REQUIRED.values() for f in fields} | set(svc.OPTIONAL[C.PAYMENT_FAILURE]) - {'diagnosis_scenario'})
OPTIONAL_COLUMNS = ('entry_id', 'scenario', 'diagnosis_scenario')
IGNORED = ('expected_route', 'expected_cause', 'customer_behaviour', 'purpose')
COLUMNS = ['entry_id', 'scenario', 'event_type', *[f for f in FIELDS if f != 'evaluation_label'], 'diagnosis_scenario', 'evaluation_label']
INTEGERS = ('amount_rupees',)
SCOPE_WHEN_ABSENT = 'contradictory'                  # the widest scope: any vocabulary value may be uploaded; the router, not the file, decides the route
EXAMPLE = ('entry_id,scenario,event_type,customer_id,amount_rupees,failure_time,checkout_stage,plan_id,method,failure_kind,invoice_id,issuer_id,diagnosis_scenario,'
           'instrument_hint,gw_error_code,gw_error_source,gw_error_step,error_reason\n'
           'row_001,CHECKOUT_ABANDONMENT,CHECKOUT_ABANDONMENT,cust_syn_01,999,t_1500,payment_page,,,,,,,,,,,\n'
           'row_002,OVERDUE_RECEIVABLE,OVERDUE_RECEIVABLE,cust_syn_07,,t_1200,,,,,inv_syn_03,,,,,,,\n')


def template() -> str:
    return EXAMPLE


def _clean(value):
    value = (value or '').strip()
    return value or None


def parse(text: str):
    """(entries, errors, warnings, summary). An entry is {entry_id, scenario, event_type, request}; request is the body of POST /revenue/events (no label unless the file has one)."""
    errors, warnings = [], []

    def err(row, column, message):
        if len(errors) < MAX_ERRORS:
            errors.append({'row': row, 'column': column, 'message': message})

    if len(text.encode('utf-8')) > MAX_BYTES:
        err(None, None, f'file is larger than {MAX_BYTES // (1024 * 1024)} MB')
        return [], errors, warnings, None
    try:
        reader = csv.DictReader(io.StringIO(text.lstrip('﻿')))
        rows = list(reader)
        header = reader.fieldnames or []
    except csv.Error as exc:
        err(None, None, f'not a readable CSV: {exc}')
        return [], errors, warnings, None
    if 'event_type' not in header:
        err(None, 'event_type', 'required column is missing (this upload is for revenue events; payment-failure CSVs use the other upload)')
        return [], errors, warnings, None
    if not rows:
        err(None, None, 'the file has a header but no data rows')
        return [], errors, warnings, None
    if len(rows) > MAX_ROWS:
        err(None, None, f'{len(rows)} rows exceeds the {MAX_ROWS}-row limit')
        return [], errors, warnings, None
    known = set(COLUMNS) | set(IGNORED)
    unknown = [c for c in header if c not in known]
    if unknown:
        warnings.append(f'ignored unknown columns: {", ".join(unknown)}')
    ignored = [c for c in header if c in IGNORED]
    if ignored:
        warnings.append(f'ignored columns (the simulated customer follows the acceptance rate you choose): {", ".join(ignored)}')
    entries, seen = [], set()
    for n, raw in enumerate(rows, start=2):
        row = {k: _clean(v) for k, v in raw.items() if k in COLUMNS}
        etype = row.get('event_type')
        entry_id = row.get('entry_id') or f'row_{n - 1:04d}'
        if entry_id in seen:
            err(n, 'entry_id', f'duplicate entry_id {entry_id!r}')
        seen.add(entry_id)
        if etype not in C.EVENT_TYPES:
            err(n, 'event_type', f'must be one of {", ".join(C.EVENT_TYPES)} (got {etype!r})')
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,60}', entry_id):
            err(n, 'entry_id', 'may contain letters, digits and _ . : - only (max 60)')
            continue
        body = {k: v for k, v in row.items() if k not in ('event_type', 'entry_id', 'scenario') and v is not None}
        for col in INTEGERS:
            if col in body:
                try:
                    body[col] = int(float(body[col])) if float(body[col]) == int(float(body[col])) else None
                except ValueError:
                    body[col] = None
                if body[col] is None:
                    err(n, col, 'must be a whole number')
        stated = body.get('diagnosis_scenario')
        checked = {**body, **({'diagnosis_scenario': SCOPE_WHEN_ABSENT} if etype == C.PAYMENT_FAILURE and not stated else {})}
        # error_reason may legitimately be empty (no reason provided): only payment failures carry it, and the rules treat a missing key as "required but null allowed"
        if etype == C.PAYMENT_FAILURE:
            checked.setdefault('error_reason', None)
        problems = svc.validate_custom(etype, checked)
        for p in problems:
            err(n, p['field'], p['message'])
        if not problems:
            entries.append({'entry_id': entry_id, 'scenario': row.get('scenario') or etype, 'event_type': etype, 'request': {'event_type': etype, **body},
                            'diagnosis_scenario_given': bool(stated)})
    summary = None
    if not errors:
        counts = {}
        for e in entries:
            counts[e['scenario']] = counts.get(e['scenario'], 0) + 1
        summary = {'rows': len(entries), 'by_scenario': counts, 'by_event_type': {t: sum(e['event_type'] == t for e in entries) for t in C.EVENT_TYPES if any(e['event_type'] == t for e in entries)},
                   'has_hidden_labels': any('evaluation_label' in e['request'] for e in entries)}
    return entries, errors, warnings, summary


def _folder() -> Path:
    p = tds.root() / 'uploads'
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate(text: str) -> dict:
    entries, errors, warnings, summary = parse(text)
    out = {'schema_version': 'event-csv@1', 'valid': not errors and bool(entries), 'errors': errors, 'warnings': warnings, 'summary': summary, 'events_csv_id': None}
    if out['valid']:
        out['events_csv_id'] = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        (_folder() / f"events_{out['events_csv_id']}.csv").write_text(text, encoding='utf-8')
    return out


@lru_cache(maxsize=8)
def _load(csv_id: str, stamp: int):
    return parse((_folder() / f'events_{csv_id}.csv').read_text(encoding='utf-8'))[0]


def load(csv_id: str) -> list:
    """The validated entries of an uploaded events CSV. Raises KeyError for an unknown id."""
    if not re.fullmatch(r'[0-9a-f]{16}', csv_id or ''):
        raise KeyError('bad events csv id')
    path = _folder() / f'events_{csv_id}.csv'
    if not path.exists():
        raise KeyError(f'unknown events csv id {csv_id}')
    return _load(csv_id, path.stat().st_mtime_ns)
