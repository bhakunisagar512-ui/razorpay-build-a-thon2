"""Deterministic safety layer v2 (Phase 4 follow-up). PREPARED, NOT ACTIVATED: nothing in the serving path imports it.

It wraps the frozen Phase 2.5 gate (services.app.domain.diagnosis_safety.decide) and, compared with that gate run at
the SAME budget, can only REMOVE automated retries, never add one. (The frozen gate itself uses 0.03; v2.1 runs at 0.10,
so in the 0.03-0.10 band it retries what the frozen 0.03 control would hold.) Three additions, each tied to a finding in
docs/phase4/safety-investigation.md:

  1. Explicit hard reasons. `error_reason` is a documented Razorpay field. card_expired and
     payment_risk_check_failed are authoritative; the ML model must not be able to override them
     (they were already honoured by the models in the benchmark, but nothing enforced it).
  2. Hard-family reasons (risk / instrument_state, e.g. debit_instrument_blocked) and a recent explicit hard
     reason on the SAME instrument in the last 24 h: automated retry is withheld and the case goes to review.
  3. A hard-decline budget. Calibrated hard-decline probability <= AUTO_RETRY_MAX_HARD_MASS allows an automated
     retry; above it the retry is withheld and the case goes to REVIEW (the frozen gate allowed up to 0.03; v2 used
     0.01 with a 0.01-0.03 review band). v2.1 uses 0.10 for both cycle 1 and reassessment: a risk choice made by the
     owner after the Phase 3 offline sweep (see eval/phase3_heldout/budget_ablation_sweep.json), not a physical
     constant. The 3% unsafe-hard-decline-payment limit is an EVALUATION acceptance criterion, never a runtime value.

route: AUTO_RETRY | NO_RETRY (known non-retryable cause) | REVIEW (retry safety cannot be established).
"""
from services.app.domain.diagnosis_safety import HARD, SafetyThresholds, decide
from services.app.domain.failure_vocab import reason_family
from services.app.domain.hard_evidence import EXPLICIT_REASONS

POLICY_VERSION = 'safety-policy@v2.2'                        # v2.1: hard-decline budget .01 -> .10; v2.2: independent RISK_BLOCKED detection at P >= .20
RISK_DETECTION_MIN_PROBABILITY = .20                         # owner decision after the Phase 3 offline sweep (20/30/40%); NOT the retry budget
HARD_REASON_CAUSE = dict(EXPLICIT_REASONS)                   # {'card_expired': ..., 'payment_risk_check_failed': ...}
HARD_REASON_FAMILIES = frozenset({'risk', 'instrument_state'})
MIN_CONFIDENCE = .60                                         # unchanged from the frozen gate
AUTO_RETRY_MAX_HARD_MASS = .10                               # was .01 in v2 (the frozen Phase 2.5 gate allowed .03); one value for cycle 1 AND reassessment
REVIEW_MAX_HARD_MASS = AUTO_RETRY_MAX_HARD_MASS              # no separate review band: above the budget the retry is withheld
THRESHOLDS_V2 = SafetyThresholds(MIN_CONFIDENCE, AUTO_RETRY_MAX_HARD_MASS)


def hard_signal(error_reason, instrument_prev_reason=None):
    """Authoritative evidence against retrying: (cause_or_None, code_or_None). Pure, no model input."""
    if error_reason in HARD_REASON_CAUSE:
        return HARD_REASON_CAUSE[error_reason], 'explicit_hard_reason_veto'
    if reason_family(error_reason) in HARD_REASON_FAMILIES:
        return 'AMBIGUOUS', 'hard_family_reason_hold'
    if instrument_prev_reason in HARD_REASON_CAUSE:
        return 'AMBIGUOUS', 'recent_hard_reason_on_instrument_hold'
    return None, None


def risk_evidence(probabilities, error_reason=None, instrument_prev_reason=None):
    """Genuine evidence of a risk block, independent of the hard-decline budget: an explicit risk reason, or Model C's own
    P(RISK_BLOCKED) at or above RISK_DETECTION_MIN_PROBABILITY. Applied identically in every cycle."""
    cause, _ = hard_signal(error_reason, instrument_prev_reason)
    return cause == 'RISK_BLOCKED' or probabilities.get('RISK_BLOCKED', 0.0) >= RISK_DETECTION_MIN_PROBABILITY


def detect_independent_risk(gate_cause, probabilities):
    """Cycle 1: escalate on P(RISK_BLOCKED) alone when the gate did not already name a hard cause (explicit evidence and
    every gate veto keep precedence). Uses the individual class probability, never the combined hard-decline mass."""
    return str(gate_cause) not in HARD and probabilities.get('RISK_BLOCKED', 0.0) >= RISK_DETECTION_MIN_PROBABILITY


def route_of(verdict):
    if verdict['retry_allowed']:
        return 'AUTO_RETRY'
    return 'NO_RETRY' if verdict['cause'] in HARD else 'REVIEW'


def decide_v2(probabilities, *, error_reason=None, instrument_hint='na', instrument_prev_reason=None,
              rule_result=None, known_tuple=True, auto_max_hard_mass=AUTO_RETRY_MAX_HARD_MASS,
              review_max_hard_mass=REVIEW_MAX_HARD_MASS, min_confidence=MIN_CONFIDENCE):
    """Frozen gate first (same confidence, rule, expiry-hint, unknown-tuple checks), then the v2 vetoes."""
    base = decide(probabilities, rule_result=rule_result, instrument_hint=instrument_hint, known_tuple=known_tuple,
                  thresholds=SafetyThresholds(min_confidence, auto_max_hard_mass))
    if not base['retry_allowed']:
        if base['decision'] == 'hard_decline_mass_veto':
            mass = sum(probabilities.get(label, 0) for label in HARD)
            if mass <= review_max_hard_mass:
                base = {**base, 'decision': 'hard_decline_mass_review'}      # the band the frozen gate used to retry
        return {**base, 'route': route_of(base)}
    cause, code = hard_signal(error_reason, instrument_prev_reason)
    if code:
        verdict = {'cause': cause, 'retry_allowed': False, 'decision': code}
        return {**verdict, 'route': route_of(verdict)}
    return {**base, 'route': 'AUTO_RETRY'}
