"""Payment-failure information-quality router and the single diagnosis decision (Phase 3 and 5).

assess() looks only at the structured gateway fields and returns one of three routes:

  CLEAR_AND_CONSISTENT  an explicit reason identifies exactly one cause for this method (or the instrument is known to be expired), and
                        nothing contradicts it  -> deterministic rule, Model C is NOT called.
  MISSING_OR_GENERIC    no reason, a generic catch-all (payment_failed, card_declined, ...), or a reason that several causes share
                        -> the frozen Model C is called exactly once, behind its existing safety gate.
  CONTRADICTORY         the fields disagree with the documented vocabulary or with each other -> Human Intervention, Model C NOT called.

Missing is not contradictory: a missing or generic reason is a lack of information, a contradiction is information that cannot all be true.
Rules and where they come from:
  [vocab]  documented per-method sources and steps, the BAD_REQUEST / GATEWAY reason sets (failure_vocab)
  [world]  which causes produce which specific reason, per method (the emission tables Model C was trained on, via custom_input_rules)
  [owner]  card-only reasons and the instrument-hint values per method (owner decisions recorded in custom_input_rules)

Diagnosis happens ONCE per event. Nothing in this module reads a hidden evaluation label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from services.app.domain import custom_input_rules as R
from services.app.domain.failure_vocab import DOCUMENTED_REASONS, SOURCES, STEPS

CLEAR, GENERIC, CONTRADICTORY = 'CLEAR_AND_CONSISTENT', 'MISSING_OR_GENERIC', 'CONTRADICTORY'
ROUTES = (CLEAR, GENERIC, CONTRADICTORY)
INFO_FIELDS = ('method', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason')


@dataclass
class Assessment:
    route: str
    cause: str | None = None
    rule_id: str | None = None
    explanation: str = ''
    conflicts: list = field(default_factory=list)       # [{'fields': [...], 'message': str}]

    def as_dict(self):
        return {'route': self.route, 'cause': self.cause, 'rule_id': self.rule_id, 'explanation': self.explanation, 'conflicts': self.conflicts}


def normalise(info: dict) -> dict:
    """The six routing fields, with a blank / 'missing' / NaN reason turned into None (not provided)."""
    out = {f: info.get(f) for f in INFO_FIELDS}
    reason = out['error_reason']
    if reason is None or (isinstance(reason, float) and math.isnan(reason)) or str(reason).strip() in ('', 'missing', 'nan', 'None'):
        out['error_reason'] = None
    out['instrument_hint'] = out['instrument_hint'] or 'na'
    return out


def conflicts_of(i: dict) -> list:
    m, hint, code, source, step, reason = (i[f] for f in INFO_FIELDS)
    found = []
    if m not in R.METHODS:
        return [{'fields': ['method'], 'message': f'Unknown payment method {m!r}.'}]
    if source not in SOURCES[m]:
        found.append({'fields': ['method', 'gw_error_source'], 'message': f"Error source '{source}' does not exist for {R.METHOD_INFO[m]} payments."})
    if step not in STEPS[m]:
        found.append({'fields': ['method', 'gw_error_step'], 'message': f"Error step '{step}' does not exist in the {R.METHOD_INFO[m]} payment flow."})
    if (m == 'card') != (hint in R.CARD_HINTS):
        found.append({'fields': ['method', 'instrument_hint'], 'message': f"Instrument hint '{hint}' is not a {R.METHOD_INFO[m]} value."})
    if reason is not None:
        if reason not in DOCUMENTED_REASONS:
            found.append({'fields': ['error_reason'], 'message': f"Error reason '{reason}' is not a documented Razorpay reason."})
        else:
            if reason in R.CARD_ONLY_REASONS and m != 'card':
                found.append({'fields': ['method', 'error_reason'], 'message': f"'{reason}' describes a card instrument and cannot occur on {R.METHOD_INFO[m]}."})
            if code not in R.codes_for_reason(reason):
                found.append({'fields': ['gw_error_code', 'error_reason'], 'message': f"Reason '{reason}' is never reported with {code} (it goes with {' / '.join(R.codes_for_reason(reason))})."})
    elif code == 'SERVER_ERROR':
        pass                                                    # a missing reason is not evidence either way
    cause = R.clear_reason_cause(m, reason)
    if hint == 'expired' and cause and cause != 'CARD_EXPIRED':
        found.append({'fields': ['instrument_hint', 'error_reason'], 'message': f"The card is known to be expired, but reason '{reason}' points to {R.CAUSE_INFO[cause].lower()}."})
    return found


def assess(info: dict) -> Assessment:
    i = normalise(info)
    found = conflicts_of(i)
    if found:
        return Assessment(CONTRADICTORY, rule_id='CONFLICT', conflicts=found,
                          explanation='The gateway fields contradict each other, so no automatic diagnosis or recovery is attempted: ' + '; '.join(c['message'] for c in found))
    m, reason = i['method'], i['error_reason']
    cause = R.clear_reason_cause(m, reason)
    if cause:
        return Assessment(CLEAR, cause, f'REASON:{reason}', f"Reason '{reason}' identifies {R.CAUSE_INFO[cause]} for {R.METHOD_INFO[m]}; the source, step and instrument hint are consistent with it.")
    if i['instrument_hint'] == 'expired':
        return Assessment(CLEAR, 'CARD_EXPIRED', 'HINT:expired', 'The instrument is known to be expired and no other field disagrees.')
    if reason is None:
        why = 'The gateway sent no error reason.'
    elif reason in R.specific_reason_causes(m):
        why = f"Reason '{reason}' is shared by several causes for {R.METHOD_INFO[m]}: " + ', '.join(sorted(R.specific_reason_causes(m)[reason])) + '.'
    else:
        why = f"Reason '{reason}' is a generic catch-all."
    return Assessment(GENERIC, rule_id='NEEDS_MODEL', explanation=why + ' Not enough to decide by rule; the frozen Model C is asked once.')


# ------------------------------------------------------------------------------------------------- Model C (frozen), called at most once
_calls = 0


def model_c_call_count() -> int:
    return _calls


def cold_start_features(info: dict, *, customer_id: str, issuer_id: str, amount_minor: int, occurred_at: str) -> dict:
    """Engineered features from an EMPTY history (labelled cold start). Nothing is invented: absent history stays absent."""
    from services.app import test_datasets as tds
    i = normalise(info)
    return tds.derive_single({'payment_id': 'cold', 'status': 'failed', 'customer_id': customer_id, 'issuer_id': issuer_id, 'amount_minor': amount_minor,
                              'occurred_at': occurred_at, 'method': i['method'], 'instrument_hint': i['instrument_hint'], 'gw_error_code': i['gw_error_code'],
                              'gw_error_source': i['gw_error_source'], 'gw_error_step': i['gw_error_step'], 'error_reason': i['error_reason']})


def model_c_diagnose(features: dict) -> dict:
    """One frozen-Model-C call plus the serving path's independent-risk rule (same as main.apply_independent_risk). Fails closed."""
    global _calls
    _calls += 1
    from services.app.adapters import diagnosis_v4
    from services.app.domain.safety_policy_v2 import detect_independent_risk
    d = diagnosis_v4.diagnose(gw_error_code=features['gw_error_code'], gw_error_source=features['gw_error_source'], gw_error_step=features['gw_error_step'],
                              method=features['method'], instrument_hint=features.get('instrument_hint') or 'na', features=features)
    if d.stage != diagnosis_v4.STAGE:
        return {'status': d.stage, 'cause': 'AMBIGUOUS', 'confidence': 0.0, 'decision': d.decision or d.stage, 'route': 'REVIEW', 'probabilities': {}, 'top_cause': None, 'model_id': ''}
    out = {'status': 'scored', 'cause': d.cause, 'confidence': float(d.confidence), 'decision': d.decision, 'route': d.route,
           'probabilities': {k: float(v) for k, v in d.probabilities.items()}, 'top_cause': d.top_cause, 'model_id': d.model_id}
    if detect_independent_risk(out['cause'], out['probabilities']):
        out.update(cause='RISK_BLOCKED', decision='independent_risk_detection', route='NO_RETRY', confidence=out['probabilities']['RISK_BLOCKED'])
    return out


def diagnose(info: dict, *, features_provider=None) -> dict:
    """The one diagnosis decision for a payment failure. `features_provider()` supplies the Model C feature row, and is called only
    on the MISSING_OR_GENERIC route. Returns a JSON-able record; 'model_c_calls' is 0 or 1."""
    a = assess(info)
    base = {'route': a.route, 'rule_id': a.rule_id, 'explanation': a.explanation, 'conflicts': a.conflicts, 'model_c_calls': 0}
    if a.route == CLEAR:
        return {**base, 'method': 'rule', 'cause': a.cause}
    if a.route == CONTRADICTORY:
        return {**base, 'method': 'none', 'cause': None}
    scored = model_c_diagnose(features_provider())
    top = sorted(scored['probabilities'].items(), key=lambda kv: -kv[1])[:3]
    if scored['cause'] == 'AMBIGUOUS':
        note = f"Model C abstained ({scored['decision']}); the conservative payment-reminder fallback is used."
    else:
        note = f"Model C diagnosed {R.CAUSE_INFO.get(scored['cause'], scored['cause'])} ({scored['decision']})."
    return {**base, 'method': 'model_c', 'cause': scored['cause'], 'explanation': a.explanation + ' ' + note, 'model_c_calls': 1,
            'model_c': {'confidence': scored['confidence'], 'decision': scored['decision'], 'route': scored['route'], 'model_id': scored['model_id'],
                        'top_probabilities': [{'cause': c, 'probability': round(p, 4)} for c, p in top]}}
