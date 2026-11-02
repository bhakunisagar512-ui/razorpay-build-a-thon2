"""Shadow-mode Model C diagnosis (Phase 2).

Scores each failed payment with the packaged Phase 4 model, using point-in-time features from the existing
FeatureBuilderV4, and RECORDS the result. Nothing here feeds the live recovery plan: main.ingest_failure calls
`on_webhook_*` after it has already decided and never reads the return value, and every entry point swallows errors.

The actual (synthetic) cause is never an input. It can only be attached afterwards with `attach_label`, into a
record field the model never sees.

State is in-process (one FeatureBuilderV4 per API process, bounded record store). A restart loses the history
and the records; persistence and multi-worker sharing are Phase 2 follow-ups.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

from services.app.domain.features_v4 import DowntimeNotice, FeatureBuilderV4, PaymentObservationV4

log = logging.getLogger('shadow')
MAX_RECORDS = 50_000
LOW_CONFIDENCE = 0.60            # the frozen safety-gate confidence threshold

_lock = threading.Lock()
_builder = FeatureBuilderV4()
_records: OrderedDict[str, dict] = OrderedDict()
_labels: dict[str, str] = {}


def reset() -> None:
    global _builder
    with _lock:
        _builder = FeatureBuilderV4()
        _records.clear()
        _labels.clear()


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def make_observation(payment_id: str, *, order_id=None, customer_id=None, issuer_id=None, psp_id=None,
                     instrument_id=None, amount_minor=0, method='card', occurred_at, observed_at=None,
                     status='failed', **rest) -> PaymentObservationV4:
    """Missing identifiers degrade safely: a missing customer becomes a private bucket (never a shared
    'unknown' history), a missing instrument is 'unknown' (instrument features are then null)."""
    occurred_at = _utc(occurred_at)
    return PaymentObservationV4(
        payment_id=payment_id, order_id=order_id or 'unknown',
        customer_id=customer_id if customer_id and customer_id != 'unknown' else f'anon:{payment_id}',
        issuer_id=issuer_id or 'unknown', psp_id=psp_id or 'unknown', amount_minor=int(amount_minor or 0),
        method=method or 'card', occurred_at=occurred_at, observed_at=_utc(observed_at) if observed_at else occurred_at,
        status=status, instrument_id=instrument_id or 'unknown', **rest)


def notify(notice: DowntimeNotice) -> dict:
    with _lock:
        try:
            _builder.notify(notice)
        except ValueError as exc:
            return {'status': 'out_of_order' if 'nondecreasing' in str(exc) else 'invalid_event', 'detail': str(exc)}
    return {'status': 'ok'}


def process(obs: PaymentObservationV4, *, diagnose: bool = True, observe: bool = True, art=None) -> dict | None:
    """Score `obs` (features built BEFORE it is observed), then add it to history. Returns the stored record, or
    None when only history was updated. Out-of-order events are recorded as such and not added to history."""
    from services.app.adapters import diagnosis_v4
    with _lock:
        record = None
        try:
            if diagnose and obs.status == 'failed':
                started = time.perf_counter()
                features = _builder.build(obs)
                d = diagnosis_v4.diagnose(gw_error_code=obs.gw_error_code, gw_error_source=obs.gw_error_source,
                                          gw_error_step=obs.gw_error_step, method=obs.method,
                                          instrument_hint=obs.instrument_hint, features=features, art=art)
                record = {
                    'payment_id': obs.payment_id, 'status': 'scored' if d.stage == diagnosis_v4.STAGE else d.stage,
                    'predicted_cause': d.top_cause or None, 'policy_cause': d.cause, 'confidence': d.confidence,
                    'low_confidence': d.confidence < LOW_CONFIDENCE, 'decision': d.decision, 'route': d.route,
                    'probabilities': d.probabilities, 'model_id': d.model_id,
                    'instrument_prev_reason': features.get('instrument_prev_reason'),
                    'observed_at': obs.observed_at.isoformat(), 'actual_cause': _labels.get(obs.payment_id),
                    'latency_ms': round((time.perf_counter() - started) * 1000, 2)}
            if observe:
                _builder.observe(obs)
        except ValueError as exc:                 # out-of-order observed_at or an invalid event; history is untouched
            status = 'out_of_order' if 'nondecreasing' in str(exc) else 'invalid_event'
            record = {'payment_id': obs.payment_id, 'status': status, 'detail': str(exc), 'predicted_cause': None,
                      'actual_cause': _labels.get(obs.payment_id)} if diagnose and obs.status == 'failed' else None
        if record is not None:
            _records[obs.payment_id] = record
            while len(_records) > MAX_RECORDS:
                _records.popitem(last=False)
            log.info('shadow_diagnosis %s', {k: record[k] for k in ('payment_id', 'status', 'predicted_cause', 'confidence')
                                             if k in record})
        return record


def attach_label(payment_id: str, actual_cause: str) -> bool:
    """Evaluation-only: store the synthetic ground truth next to the record. Never read by the model path."""
    with _lock:
        _labels[payment_id] = actual_cause
        record = _records.get(payment_id)
        if record is not None:
            record['actual_cause'] = actual_cause
        return record is not None


def get(payment_id: str) -> dict | None:
    with _lock:
        return _records.get(payment_id)


def recent(limit: int = 100) -> list[dict]:
    with _lock:
        return list(_records.values())[-limit:]
