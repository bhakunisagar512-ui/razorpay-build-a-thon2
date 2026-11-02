"""
Phase 4 diagnosis adapter (PREPARED, NOT ACTIVATED).

Nothing in services.app imports this module; the live path still calls
services.app.adapters.diagnosis.diagnose(). It exists so the benchmark winner can be swapped in by changing
one import once the owner approves, and so its behaviour on bad input is already tested.

Differences from the live adapter, all deliberate:
  * scores a full feature dict (the deployment-compatible feature set), not the five legacy fields;
  * returns calibrated probabilities, then applies safety-policy@v2 (services.app.domain.safety_policy_v2): the frozen
    confidence/rule/expiry gate, deterministic vetoes on explicit hard reasons, and a two-band hard-decline budget;
  * absent fields are "missing", unseen categories are tolerated, junk numerics become NaN;
  * every failure path returns AMBIGUOUS with confidence 0.0, exactly like the live adapter.

Artifact directory (written by eval/phase4_package.py): model.joblib, schema.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import threading
from dataclasses import dataclass, field

from services.app.adapters.diagnosis import Diagnosis
from services.app.domain.safety_policy_v2 import THRESHOLDS_V2, decide_v2
from services.app.domain.taxonomy import Cause

ARTIFACT_DIR = pathlib.Path('eval/artifacts/phase4')
GATE = THRESHOLDS_V2     # confidence .60 (frozen); hard-decline budget .10 (safety-policy@v2.1)
STAGE = 'ml_v4'

_lock = threading.Lock()
_bundle = None


@dataclass(frozen=True)
class DiagnosisV4(Diagnosis):
    """Diagnosis plus the detail an audit log needs. Still unpacks as (cause, confidence, stage) for callers
    of the live adapter because they read attributes by name."""
    decision: str = ''
    probabilities: dict = field(default_factory=dict)
    top_cause: str = ''
    model_id: str = ''
    route: str = ''               # AUTO_RETRY | NO_RETRY | REVIEW


class ArtifactError(RuntimeError):
    pass


def load_artifact(directory=ARTIFACT_DIR):
    """Load and integrity-check the packaged model. Raises ArtifactError, never half-loads."""
    import joblib
    directory = pathlib.Path(directory)
    try:
        schema = json.loads((directory / 'schema.json').read_text())
        blob = (directory / 'model.joblib').read_bytes()
    except OSError as exc:
        raise ArtifactError(f'artifact missing: {exc}') from exc
    if hashlib.sha256(blob).hexdigest() != schema['model_sha256']:
        raise ArtifactError('model.joblib does not match schema.json (modified after packaging)')
    bundle = joblib.load(directory / 'model.joblib')
    model = bundle['model']
    if list(model.features) != schema['features'] or list(model.classes_) != schema['classes']:
        raise ArtifactError('schema.json disagrees with the pickled model')
    return {'model': model, 'calibrator': bundle['calibrator'], 'schema': schema}


def bundle(directory=ARTIFACT_DIR):
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is None:
            _bundle = load_artifact(directory)
        return _bundle


def reset():
    global _bundle
    _bundle = None


def _clean(value):
    """Request value -> something the preprocessing can coerce. Blank strings and NaN mean "not provided"."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def derive(features: dict) -> dict:
    """Fields that are pure functions of other request fields, filled only when the caller did not send them."""
    from services.app.domain.failure_vocab import GENERIC_REASONS, reason_family, step_ordinal
    out = dict(features)
    reason = _clean(out.get('error_reason'))
    out.setdefault('reason_family', reason_family(reason))
    out.setdefault('is_generic_reason', int(reason is None or reason in GENERIC_REASONS))
    out.setdefault('step_ordinal', step_ordinal(out.get('gw_error_step') or ''))
    return out


def score(features: dict, art=None):
    """Calibrated class probabilities for one request dict. Missing keys and junk values are tolerated."""
    import pandas as pd
    art = art or bundle()
    model = art['model']
    features = derive(features)
    row = {name: _clean(features.get(name)) for name in model.features}
    frame = pd.DataFrame([row], columns=list(model.features))
    return dict(zip(model.classes_, art['calibrator'].transform(model.predict_proba(frame))[0].tolist()))


def diagnose(*, gw_error_code: str | None, gw_error_source: str | None, gw_error_step: str | None, method: str,
             instrument_hint: str = 'na', features: dict | None = None, art=None, **extra) -> DiagnosisV4:
    """Same keyword surface as services.app.adapters.diagnosis.diagnose, plus optional `features`
    (history / reason / timing fields) and any extra keyword treated as a feature. Fails closed."""
    if not gw_error_code:
        return DiagnosisV4(str(Cause.AMBIGUOUS), 0.0, 'no_error_code', decision='no_error_code')
    try:
        art = art or bundle()
        request = {**(features or {}), **extra, 'gw_error_code': gw_error_code,
                   'gw_error_source': gw_error_source or 'gateway', 'gw_error_step': gw_error_step or 'authorization',
                   'method': method, 'instrument_hint': instrument_hint or 'na'}
        probabilities = score(request, art)
        known = (gw_error_code, request['gw_error_source'], request['gw_error_step']) in art['model'].known_tuples
        from eval.classifier import RULES
        rule = RULES.get((gw_error_code, request['gw_error_source'], request['gw_error_step']))
        verdict = decide_v2(probabilities, rule_result=str(rule) if rule is not None else None,
                            instrument_hint=request['instrument_hint'], known_tuple=known,
                            error_reason=_clean(request.get('error_reason')),
                            instrument_prev_reason=_clean(request.get('instrument_prev_reason')))
        top = max(probabilities, key=probabilities.get)
        return DiagnosisV4(verdict['cause'], float(probabilities[top]), STAGE, decision=verdict['decision'],
                           probabilities=probabilities, top_cause=top, model_id=art['schema']['model_id'],
                           route=verdict['route'])
    except Exception as exc:                          # noqa: BLE001
        return DiagnosisV4(str(Cause.AMBIGUOUS), 0.0, f'error:{type(exc).__name__}', decision='error')


def features_from_razorpay_payment(payment: dict, event_created_at: int | None = None) -> dict:
    """Map the fields a Razorpay `payment` entity (fetch API or payment.failed webhook) carries directly onto
    model inputs. History features (instrument / issuer / customer windows) are NOT derivable from one
    entity; they come from the feature store (services.app.domain.features_v4.FeatureBuilderV4 replay).
    """
    from datetime import datetime, timedelta, timezone
    card = payment.get('card') or {}
    ist = timezone(timedelta(hours=5, minutes=30))
    created = payment.get('created_at')
    out = {
        'gw_error_code': payment.get('error_code'), 'gw_error_source': payment.get('error_source'),
        'gw_error_step': payment.get('error_step'), 'error_reason': payment.get('error_reason'),
        'method': payment.get('method'), 'amount_minor': payment.get('amount'),
        'issuer_id': card.get('issuer') or payment.get('bank'),
    }
    if isinstance(created, (int, float)) and created > 0:
        local = datetime.fromtimestamp(created, tz=ist)
        out.update(hour_ist=local.hour, weekday_ist=local.weekday())
    if isinstance(out['amount_minor'], (int, float)) and out['amount_minor'] > 0:
        out['log_amount_minor'] = math.log1p(out['amount_minor'])
    return {k: v for k, v in out.items() if v is not None}
