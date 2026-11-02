"""Phase 4 / Stage 2: dataset registry and custom-CSV handling for the testing API.

Prebuilt datasets (eval/build_ui_datasets.py) are read here: `dataset_inputs` / `input_row` return INPUTS ONLY. The hidden
ground truth is reachable through exactly one function, `hidden_truth`, which only the server-side writer of
the evaluator-read truth table (individual tests) and the offline evaluator (dataset runs) call. Nothing here ever passes truth to Model C.

Custom CSV: raw event rows (failed attempts to test, optional successful attempts as history). Engineered features are NOT
supplied by the user: they are derived by replaying the rows in time order through the existing FeatureBuilderV4 in a fresh,
local builder (the shared in-process history is never touched).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from services.app.settings import settings

NAMES = ('normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime')
MAX_ROWS, MAX_BYTES, MAX_ERRORS = 5000, 2 * 1024 * 1024, 50
METHODS = ('card', 'upi', 'netbanking')
CAUSES = ('BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'RISK_BLOCKED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
SCHEMA_VERSION = 'csv-schema@1'

# name, required, type, description, example
SCHEMA = [
    ('payment_id', True, 'string', 'Unique id of the attempt.', 'pay_0001'),
    ('occurred_at', True, 'ISO-8601 datetime', 'When the attempt happened (UTC assumed when no timezone is given).', '2026-05-04T10:15:00+00:00'),
    ('status', True, 'failed | success', 'failed rows are tested; success rows only build history (customer/instrument/issuer windows).', 'failed'),
    ('customer_id', True, 'string', 'Customer identifier.', 'cust_17'),
    ('issuer_id', True, 'string', 'Issuing bank identifier.', 'HDFC'),
    ('amount_minor', True, 'integer > 0', 'Amount in paise.', '49900'),
    ('method', True, 'card | upi | netbanking', 'Payment method.', 'card'),
    ('gw_error_code', 'failed rows', 'string', 'Razorpay error.code of the failed attempt.', 'BAD_REQUEST_ERROR'),
    ('gw_error_source', 'failed rows', 'string', 'Razorpay error.source.', 'issuer_bank'),
    ('gw_error_step', 'failed rows', 'string', 'Razorpay error.step.', 'payment_authorization'),
    ('observed_at', False, 'ISO-8601 datetime', 'When the event reached us. Defaults to occurred_at.', '2026-05-04T10:15:02+00:00'),
    ('error_reason', False, 'string', 'Razorpay error.reason, e.g. insufficient_funds.', 'insufficient_funds'),
    ('order_id', False, 'string', 'Order id. Defaults to ord_<payment_id>.', 'ord_0001'),
    ('instrument_id', False, 'string', 'Card/VPA token. Enables instrument-history features.', 'card_tok_9'),
    ('psp_id', False, 'string', 'Payment service provider. Defaults to unknown.', 'psp_a'),
    ('instrument_hint', False, 'string', 'na | expired | valid | unknown. Defaults to na.', 'na'),
    ('latency_ms', False, 'number >= 0', 'Gateway latency of the attempt.', '850'),
    ('payment_flow', False, 'string', 'e.g. card_3ds, upi_intent.', 'card_3ds'),
    ('attempt_duration_s', False, 'number >= 0', 'Customer-side duration of the attempt.', '42'),
    ('true_cause', False, 'one of the six causes', 'HIDDEN ground truth for evaluation. Give it for ALL failed rows or for none. '
                                                    'It is never shown to Model C or to recovery decisions.', ''),
]
COLUMNS = [c[0] for c in SCHEMA]
REQUIRED = [c[0] for c in SCHEMA if c[1] is True]
FAILED_REQUIRED = ['gw_error_code', 'gw_error_source', 'gw_error_step']


def _bud():
    from eval import build_ui_datasets          # lazy: keeps API start-up light
    return build_ui_datasets


def root() -> Path:
    return Path(settings.ui_data_dir)


# ----------------------------------------------------------------------------------------------------- prebuilt datasets
def list_datasets() -> list[dict]:
    out = []
    for name in NAMES:
        path = root() / 'datasets' / name / 'manifest.json'
        if not path.exists():
            out.append({'name': name, 'available': False})
            continue
        m = json.loads(path.read_text(encoding='utf-8'))
        out.append({'name': name, 'available': True, 'description': m['description'], 'records': m['records'], 'label': m['label'],
                    'representative_of_natural_performance': m['representative_of_natural_performance'],
                    'composition_by_cause': m['composition_by_cause'], 'source_world_seeds': m['source']['world_seeds'],
                    'has_ground_truth': True})
    return out


def _check_name(name: str) -> str:
    if name not in NAMES:
        raise KeyError(f'unknown dataset {name!r}')
    return name


def dataset_inputs(name: str) -> pd.DataFrame:
    """Inputs only (43 Model C features + serving fields). The manifest hash is verified before the pickle is read."""
    return _bud().load_inputs(_check_name(name), root() / 'datasets')


def hidden_truth(name: str) -> dict:
    """HIDDEN ground truth {payment_id: {...}}. Callers: the server-side evaluator-table writer and the offline evaluator only."""
    return _bud().load_truth(_check_name(name), root() / 'datasets')


LIST_COLUMNS = ['payment_id', 'method', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason', 'amount_minor_raw',
                'issuer_id', 'instrument_hint', 'failed_at']


def dataset_payments(name: str, offset: int = 0, limit: int = 50, method: str | None = None, error_code: str | None = None,
                     search: str | None = None) -> dict:
    frame = dataset_inputs(name)
    if method:
        frame = frame[frame.method == method]
    if error_code:
        frame = frame[frame.gw_error_code == error_code]
    if search:
        frame = frame[frame.payment_id.str.contains(re.escape(search), case=False)]
    page = frame.iloc[offset:offset + limit][LIST_COLUMNS]
    items = [{**{k: (None if pd.isna(v) else v) for k, v in r.items()}, 'failed_at': pd.Timestamp(r['failed_at']).isoformat()}
             for r in page.to_dict('records')]
    return {'dataset': name, 'total': int(len(frame)), 'offset': offset, 'limit': limit, 'items': items}


def input_row(name: str, payment_id: str) -> dict | None:
    frame = dataset_inputs(name)
    hit = frame[frame.payment_id == payment_id]
    return None if hit.empty else hit.iloc[0].to_dict()


# ----------------------------------------------------------------------------------------------------- CSV template / schema
def csv_schema() -> dict:
    return {'schema_version': SCHEMA_VERSION, 'limits': {'max_rows': MAX_ROWS, 'max_bytes': MAX_BYTES},
            'columns': [{'name': n, 'required': r, 'type': t, 'description': d, 'example': e} for n, r, t, d, e in SCHEMA],
            'rules': ['Header row required; columns may be in any order; unknown columns are ignored with a warning.',
                      'Rows are sorted by observed_at (then payment_id) before features are derived; history only looks backwards.',
                      'Only status=failed rows are tested; status=success rows provide history.',
                      'Engineered features are derived for you. A failed row with no earlier events gets cold-start (empty-history) features.',
                      'true_cause is optional: all failed rows or none. Without it results are predictions only, never verified recovery.',
                      'Downtime notices are not supported in CSV uploads (downtime features stay empty).']}


def csv_template() -> str:
    rows = [
        ['pay_h1', '2026-05-04T09:00:00+00:00', 'success', 'cust_1', 'HDFC', '49900', 'card', '', '', '', '', '', 'ord_h1', 'card_tok_1', 'psp_a', 'na', '700', 'card_3ds', '30', ''],
        ['pay_h2', '2026-05-04T09:20:00+00:00', 'success', 'cust_2', 'ICICI', '120000', 'upi', '', '', '', '', '', 'ord_h2', 'vpa_2', 'psp_a', 'na', '400', 'upi_intent', '12', ''],
        ['pay_f1', '2026-05-04T10:15:00+00:00', 'failed', 'cust_1', 'HDFC', '49900', 'card', 'BAD_REQUEST_ERROR', 'issuer_bank', 'payment_authorization',
         '2026-05-04T10:15:02+00:00', 'insufficient_funds', 'ord_f1', 'card_tok_1', 'psp_a', 'na', '850', 'card_3ds', '42', ''],
        ['pay_f2', '2026-05-04T10:40:00+00:00', 'failed', 'cust_2', 'ICICI', '120000', 'upi', 'GATEWAY_ERROR', 'gateway', 'payment_request',
         '', 'payment_timed_out', 'ord_f2', 'vpa_2', 'psp_a', 'na', '9000', 'upi_intent', '95', ''],
    ]
    out = io.StringIO()
    pd.DataFrame(rows, columns=['payment_id', 'occurred_at', 'status', 'customer_id', 'issuer_id', 'amount_minor', 'method', 'gw_error_code',
                                'gw_error_source', 'gw_error_step', 'observed_at', 'error_reason', 'order_id', 'instrument_id', 'psp_id',
                                'instrument_hint', 'latency_ms', 'payment_flow', 'attempt_duration_s', 'true_cause']).to_csv(out, index=False)
    return out.getvalue()


# ----------------------------------------------------------------------------------------------------- CSV validation
def _parse(text: str):
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def validate_csv(text: str) -> dict:
    errors, warnings = [], []

    def err(row, column, message):
        if len(errors) < MAX_ERRORS:
            errors.append({'row': row, 'column': column, 'message': message})

    result = {'schema_version': SCHEMA_VERSION, 'valid': False, 'errors': errors, 'warnings': warnings, 'summary': None, 'csv_id': None}
    if len(text.encode('utf-8')) > MAX_BYTES:
        err(None, None, f'file is larger than {MAX_BYTES // (1024 * 1024)} MB')
        return result
    try:
        df = _parse(text)
    except Exception as exc:                                                  # noqa: BLE001
        err(None, None, f'not a readable CSV: {exc}')
        return result
    if len(df) == 0:
        err(None, None, 'the file has a header but no data rows')
        return result
    if len(df) > MAX_ROWS:
        err(None, None, f'{len(df)} rows exceeds the {MAX_ROWS}-row limit')
        return result
    missing = [c for c in REQUIRED if c not in df.columns]
    for c in missing:
        err(None, c, 'required column is missing')
    unknown = [c for c in df.columns if c not in COLUMNS]
    if unknown:
        warnings.append(f'ignored unknown columns: {", ".join(unknown)}')
    if missing:
        return result
    df = df.assign(rownum=range(2, len(df) + 2))                                  # 2 = first data line in the file
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ''
    failed = df.status == 'failed'
    for r in df[~df.status.isin(['failed', 'success'])].itertuples():
        err(r.rownum, 'status', f'must be failed or success (got {r.status!r})')
    for col in ('payment_id', 'customer_id', 'issuer_id'):
        for r in df[df[col].str.strip() == ''].itertuples():
            err(r.rownum, col, 'must not be empty')
    dup = df[df.payment_id.duplicated(keep=False) & (df.payment_id != '')]
    for r in dup.itertuples():
        err(r.rownum, 'payment_id', f'duplicate payment_id {r.payment_id!r}')
    for col in ('occurred_at', 'observed_at'):
        parsed = pd.to_datetime(df[col].replace('', pd.NA), errors='coerce', utc=True)
        bad = df[(df[col] != '') & parsed.isna()]
        for r in bad.itertuples():
            err(r.rownum, col, f'not an ISO-8601 datetime ({getattr(r, col)!r})')
    for r in df[df.occurred_at == ''].itertuples():
        err(r.rownum, 'occurred_at', 'must not be empty')
    amount = pd.to_numeric(df.amount_minor, errors='coerce')
    for r in df[amount.isna() | (amount <= 0) | (amount % 1 != 0)].itertuples():
        err(r.rownum, 'amount_minor', f'must be a positive integer (got {r.amount_minor!r})')
    for r in df[~df.method.isin(METHODS)].itertuples():
        err(r.rownum, 'method', f'must be one of {", ".join(METHODS)} (got {r.method!r})')
    for col in FAILED_REQUIRED:
        for r in df[failed & (df[col].str.strip() == '')].itertuples():
            err(r.rownum, col, 'required for failed rows')
    for col in ('latency_ms', 'attempt_duration_s'):
        num = pd.to_numeric(df[col].replace('', pd.NA), errors='coerce')
        for r in df[(df[col] != '') & (num.isna() | (num < 0))].itertuples():
            err(r.rownum, col, f'must be a number >= 0 (got {getattr(r, col)!r})')
    for r in df[(df.true_cause != '') & ~df.true_cause.isin(CAUSES)].itertuples():
        err(r.rownum, 'true_cause', f'must be one of the six causes (got {r.true_cause!r})')
    n_failed = int(failed.sum())
    labelled = int((failed & (df.true_cause != '')).sum())
    if n_failed == 0:
        err(None, 'status', 'at least one failed row is required')
    elif 0 < labelled < n_failed:
        err(None, 'true_cause', f'labels must be given for all {n_failed} failed rows or for none (found {labelled})')
    if not errors:
        order = pd.to_datetime(df.observed_at.where(df.observed_at != '', df.occurred_at), utc=True)
        if not order.is_monotonic_increasing:
            warnings.append('rows are not in chronological order; they will be sorted by observed_at before features are derived')
    result['summary'] = {'rows': int(len(df)), 'failed_rows': n_failed, 'history_rows': int(len(df) - n_failed),
                         'has_ground_truth': bool(n_failed and labelled == n_failed),
                         'mode': 'verified' if (n_failed and labelled == n_failed) else 'predictions_only'}
    if not errors:
        result['valid'] = True
        result['csv_id'] = _store(text)
    return result


def _store(text: str) -> str:
    csv_id = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    folder = root() / 'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{csv_id}.csv').write_text(text, encoding='utf-8')
    return csv_id


def load_csv_text(csv_id: str) -> str:
    if not re.fullmatch(r'[0-9a-f]{16}', csv_id):
        raise KeyError('bad csv id')
    path = root() / 'uploads' / f'{csv_id}.csv'
    if not path.exists():
        raise KeyError(f'unknown csv id {csv_id}')
    return path.read_text(encoding='utf-8')


# ----------------------------------------------------------------------------------------------------- feature derivation
def _dt(value, fallback=None):
    ts = pd.to_datetime(value if value not in ('', None) else fallback, utc=True)
    return ts.to_pydatetime()


def _num(value):
    return None if value in ('', None) else float(value)


def observation_from(r: dict):
    from services.app.domain.features_v4 import PaymentObservationV4
    occurred = _dt(r['occurred_at'])
    return PaymentObservationV4(
        payment_id=r['payment_id'], order_id=r.get('order_id') or f"ord_{r['payment_id']}", customer_id=r['customer_id'],
        issuer_id=r['issuer_id'], psp_id=r.get('psp_id') or 'unknown', amount_minor=int(float(r['amount_minor'])), method=r['method'],
        occurred_at=occurred, observed_at=_dt(r.get('observed_at'), r['occurred_at']), status=r['status'],
        instrument_hint=r.get('instrument_hint') or 'na', gw_error_code=r.get('gw_error_code') or '',
        gw_error_source=r.get('gw_error_source') or '', gw_error_step=r.get('gw_error_step') or '', latency_ms=_num(r.get('latency_ms')),
        instrument_id=r.get('instrument_id') or 'unknown', error_reason=r.get('error_reason') or None,
        payment_flow=r.get('payment_flow') or None, attempt_duration_s=_num(r.get('attempt_duration_s')))


def feature_row(builder, obs) -> dict:
    """The engineered features for one failed attempt (built BEFORE it is observed) plus the serving fields."""
    feats = builder.build(obs)
    return {**feats, 'payment_id': obs.payment_id, 'order_id': obs.order_id, 'customer_id': obs.customer_id,
            'instrument_id': obs.instrument_id, 'psp_id': obs.psp_id, 'issuer_id': obs.issuer_id, 'amount_minor_raw': obs.amount_minor,
            'failed_at': obs.occurred_at, 'observed_at': obs.observed_at}


def derive_frame(text: str):
    """(features frame for the failed rows, {payment_id: true_cause} or None, meta). Fresh local builder: no shared state."""
    from services.app.domain.features_v4 import FeatureBuilderV4
    check = validate_csv(text)
    if not check['valid']:
        raise ValueError(f"invalid CSV: {check['errors'][:3]}")
    df = _parse(text)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ''
    df['_obs'] = pd.to_datetime(df.observed_at.where(df.observed_at != '', df.occurred_at), utc=True)
    df = df.sort_values(['_obs', 'payment_id'], kind='stable')
    builder, rows, truth = FeatureBuilderV4(), [], {}
    seen_customer, seen_instrument = Counter(), Counter()
    for r in df.to_dict('records'):
        obs = observation_from(r)
        if obs.status == 'failed':
            row = feature_row(builder, obs)
            row['history_events_before'] = seen_customer[obs.customer_id]
            row['cold_start_features'] = seen_customer[obs.customer_id] == 0 and seen_instrument[obs.instrument_id] == 0
            rows.append(row)
            if r['true_cause']:
                truth[obs.payment_id] = r['true_cause']
        builder.observe(obs)
        seen_customer[obs.customer_id] += 1
        if obs.instrument_id != 'unknown':
            seen_instrument[obs.instrument_id] += 1
    frame = pd.DataFrame(rows)
    meta = {'rows': int(len(df)), 'failed_rows': len(rows), 'cold_start_rows': int(frame.cold_start_features.sum()), 'has_ground_truth': bool(truth)}
    return frame, (truth or None), meta


def derive_single(fields: dict) -> dict:
    """One custom payment with NO history: features are cold-start and the caller must say so."""
    from services.app.domain.features_v4 import FeatureBuilderV4
    row = feature_row(FeatureBuilderV4(), observation_from({**fields, 'status': 'failed'}))
    return {**row, 'history_events_before': 0, 'cold_start_features': True}
