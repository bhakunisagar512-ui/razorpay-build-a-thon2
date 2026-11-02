"""safety-policy@v2: deterministic vetoes, the two-band hard-decline budget, and proof that the vectorised
evaluation used in the Phase 4 safety report is the same policy as the pure function."""
import numpy as np
import pandas as pd
import pytest

from eval.classifier import RULES
from eval.phase4_safety import apply_variant
from eval.safety_pipeline import apply_policy, prepare
from services.app.domain.diagnosis_safety import SafetyThresholds, decide
from services.app.domain.safety_policy_v2 import (AUTO_RETRY_MAX_HARD_MASS, REVIEW_MAX_HARD_MASS, THRESHOLDS_V2,
                                                  decide_v2, hard_signal)

CLASSES = ['BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'RISK_BLOCKED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT']
FROZEN = SafetyThresholds(.60, .03)


def probs(**kw):
    p = {c: 0. for c in CLASSES}
    p.update(kw)
    p['BD_INSUFFICIENT_FUNDS'] += 1 - sum(p.values())
    return p


# ---- failure 1: nothing enforced that an explicit authoritative reason beats the model -----------------------
@pytest.mark.parametrize('reason,cause', [('card_expired', 'CARD_EXPIRED'), ('payment_risk_check_failed', 'RISK_BLOCKED')])
def test_explicit_hard_reason_cannot_be_overridden_by_a_confident_model(reason, cause):
    confident_retry = probs(BD_INSUFFICIENT_FUNDS=.995, RISK_BLOCKED=.002, CARD_EXPIRED=.003)
    assert decide(confident_retry, thresholds=FROZEN)['retry_allowed']              # the frozen gate alone would retry
    v = decide_v2(confident_retry, error_reason=reason)
    assert (v['cause'], v['retry_allowed'], v['route'], v['decision']) == (cause, False, 'NO_RETRY', 'explicit_hard_reason_veto')


# ---- failure 2: instrument-state / risk reasons and recent hard reasons on the same instrument ----------------
@pytest.mark.parametrize('reason', ['debit_instrument_blocked', 'transaction_on_vpa_restricted', 'debit_instrument_inactive'])
def test_hard_family_reason_holds_the_retry_for_review(reason):
    v = decide_v2(probs(BD_INSUFFICIENT_FUNDS=.995, RISK_BLOCKED=.003), error_reason=reason)
    assert not v['retry_allowed'] and v['route'] == 'REVIEW' and v['decision'] == 'hard_family_reason_hold'


def test_recent_explicit_hard_reason_on_the_same_instrument_holds_the_retry():
    p = probs(BD_INSUFFICIENT_FUNDS=.995, RISK_BLOCKED=.003)
    assert decide_v2(p, error_reason='insufficient_funds', instrument_prev_reason='payment_risk_check_failed')['route'] == 'REVIEW'
    assert decide_v2(p, error_reason='insufficient_funds', instrument_prev_reason='none')['route'] == 'AUTO_RETRY'
    assert hard_signal('insufficient_funds', 'card_expired') == ('AMBIGUOUS', 'recent_hard_reason_on_instrument_hold')


# ---- failure 3: the hard-decline budget (v2: .01 with a review band; v2.1: .10, one value, no band) ------------
def test_hard_mass_budget_is_ten_percent_everywhere_with_no_separate_review_band():
    assert (AUTO_RETRY_MAX_HARD_MASS, REVIEW_MAX_HARD_MASS) == (.10, .10) and THRESHOLDS_V2.min_confidence == .60
    assert THRESHOLDS_V2.max_hard_mass == AUTO_RETRY_MAX_HARD_MASS
    low = decide_v2(probs(BD_INSUFFICIENT_FUNDS=.97, RISK_BLOCKED=.005, CARD_EXPIRED=.005))
    mid = decide_v2(probs(BD_INSUFFICIENT_FUNDS=.90, RISK_BLOCKED=.06, CARD_EXPIRED=.01))      # mass .07: held under v2 (.01), retried now
    high = decide_v2(probs(RISK_BLOCKED=.11, CARD_EXPIRED=.02))                                  # mass .13 > .10
    assert low['route'] == 'AUTO_RETRY' and mid['route'] == 'AUTO_RETRY'
    assert (high['route'], high['decision']) == ('REVIEW', 'hard_decline_mass_veto')           # never relabelled: no review band


def test_budget_boundary_is_inclusive():
    at = decide_v2(probs(RISK_BLOCKED=.05, CARD_EXPIRED=.05))                                    # mass exactly .10 is allowed
    over = decide_v2(probs(RISK_BLOCKED=.05, CARD_EXPIRED=.0501))
    assert at['route'] == 'AUTO_RETRY' and over['decision'] == 'hard_decline_mass_veto'


def test_frozen_phase4_control_is_still_the_three_percent_gate():
    from eval.phase4_confidence import GATE
    assert (GATE.min_confidence, GATE.max_hard_mass) == (.60, .03)
    p = probs(RISK_BLOCKED=.04, CARD_EXPIRED=.03)                                                # mass .07: only the 10% budget retries it
    assert not decide(p, thresholds=FROZEN)['retry_allowed'] and decide_v2(p)['retry_allowed']


# ---- recorded benchmark cases (HGB, deployment_core, fresh test worlds) ---------------------------------------
def test_recorded_unsafe_case_is_retried_again_at_ten_percent_a_pinned_limitation():
    # payment 6037: RISK_BLOCKED, reason gateway_technical_error, model TD_PSP_TIMEOUT conf .897, hard mass .022.
    # v2 (.01) held it for review; v2.1 (.10) retries it, as the frozen gate did. Accepted trade-off, recorded here.
    p = probs(TD_PSP_TIMEOUT=.897, RISK_BLOCKED=.022, TD_ISSUER_DOWN=.05, BD_INSUFFICIENT_FUNDS=.02, BD_USER_ABANDONED_OTP=.011)
    assert decide(p, thresholds=FROZEN)['retry_allowed']
    assert decide_v2(p, error_reason='gateway_technical_error')['route'] == 'AUTO_RETRY'


def test_known_undetectable_case_is_documented_as_residual_not_hidden():
    # payment 4015: RISK_BLOCKED, reason payment_timed_out, model TD_PSP_TIMEOUT conf .904, hard mass .003.
    # Nothing observable separates it from a real PSP timeout; v2 still retries it. This test pins the limitation.
    p = probs(TD_PSP_TIMEOUT=.904, RISK_BLOCKED=.003, TD_ISSUER_DOWN=.06, BD_INSUFFICIENT_FUNDS=.03, BD_USER_ABANDONED_OTP=.003)
    assert decide_v2(p, error_reason='payment_timed_out', instrument_prev_reason='none')['route'] == 'AUTO_RETRY'


# ---- v2.1 may only remove retries relative to the frozen gate RUN AT THE SAME BUDGET; the frozen vetoes still win ---
def test_v2_never_allows_a_retry_the_frozen_gate_blocks():
    frozen_at_budget = SafetyThresholds(.60, AUTO_RETRY_MAX_HARD_MASS)
    rng = np.random.default_rng(3)
    reasons = [None, 'card_expired', 'payment_risk_check_failed', 'debit_instrument_blocked', 'insufficient_funds', 'payment_failed']
    for _ in range(3000):
        raw = rng.dirichlet(np.full(6, rng.choice([.05, .3, 2.])))
        p = dict(zip(CLASSES, raw / raw.sum()))
        kw = dict(instrument_hint=str(rng.choice(['na', 'valid', 'unknown', 'expired'])), known_tuple=bool(rng.random() < .8),
                  rule_result=str(rng.choice(['', 'CARD_EXPIRED', 'BD_INSUFFICIENT_FUNDS'])) or None)
        frozen = decide(p, thresholds=frozen_at_budget, **kw)
        v2 = decide_v2(p, error_reason=reasons[rng.integers(len(reasons))],
                       instrument_prev_reason=str(rng.choice(['none', 'card_expired'])), **kw)
        assert not (v2['retry_allowed'] and not frozen['retry_allowed'])
        assert (v2['route'] == 'AUTO_RETRY') == v2['retry_allowed']


def test_expiry_hint_rule_veto_and_low_confidence_are_unchanged():
    p = probs(BD_INSUFFICIENT_FUNDS=.998)
    assert decide_v2(p, instrument_hint='expired')['decision'] == 'expired_instrument_veto'
    assert decide_v2(p, rule_result='RISK_BLOCKED')['decision'] == 'hard_rule_veto'
    assert decide_v2(probs(BD_INSUFFICIENT_FUNDS=.5, TD_PSP_TIMEOUT=.5))['decision'] == 'low_confidence'
    assert decide_v2(p, known_tuple=False)['decision'] == 'unknown_error_tuple'


# ---- the benchmark's vectorised layering == the pure function, row by row -------------------------------------
class _Stub:
    classes_ = CLASSES
    known_tuples = {('BAD_REQUEST_ERROR', 'bank', 'payment')}


def test_vectorised_evaluation_matches_the_pure_policy_row_by_row():
    rng = np.random.default_rng(11)
    n = 1500
    reasons = np.array(['missing', 'card_expired', 'payment_risk_check_failed', 'debit_instrument_blocked', 'insufficient_funds',
                        'payment_failed', 'request_timed_out'])
    raw = rng.dirichlet(np.full(6, .25), size=n)
    raw = np.where(rng.random((n, 1)) < .7, np.clip(raw ** 3, 1e-4, None), raw)
    p = raw / raw.sum(axis=1, keepdims=True)
    code = np.where(rng.random(n) < .9, 'BAD_REQUEST_ERROR', 'GATEWAY_ERROR')
    frame = pd.DataFrame({'true_cause': rng.choice(CLASSES, n), 'gw_error_code': code, 'gw_error_source': 'bank',
                          'gw_error_step': 'payment',
                          'instrument_hint': rng.choice(['na', 'valid', 'unknown', 'expired'], n, p=[.5, .25, .2, .05]),
                          'error_reason': rng.choice(reasons, n),
                          'instrument_prev_reason': rng.choice(['none', 'card_expired', 'insufficient_funds'], n),
                          'n_causes_fired': 1})
    data = prepare(frame, _Stub(), p)
    cause, _, retry = apply_policy(data, SafetyThresholds(.60, AUTO_RETRY_MAX_HARD_MASS))   # v2 layers on the gate at the same budget
    cases = frame.assign(hard=frame.true_cause.isin(['CARD_EXPIRED', 'RISK_BLOCKED']), hard_mass=data['hard_mass'],
                         retry_allowed=retry, gate_cause=cause)
    v_retry, v_cause = apply_variant(cases, 'v2')
    retried = 0
    for i in range(n):
        row = frame.iloc[i]
        key = (row.gw_error_code, row.gw_error_source, row.gw_error_step)
        pure = decide_v2(dict(zip(CLASSES, p[i])), error_reason=None if row.error_reason == 'missing' else row.error_reason,
                         instrument_hint=row.instrument_hint, instrument_prev_reason=row.instrument_prev_reason,
                         rule_result=str(RULES.get(key, '')) or None, known_tuple=key in _Stub.known_tuples)
        assert bool(v_retry[i]) == pure['retry_allowed'], (i, pure)
        retried += pure['retry_allowed']
        if retry[i] and not pure['retry_allowed']:          # a retry that v2 withheld: same abstention accounting
            assert (v_cause[i] == 'AMBIGUOUS') == (pure['cause'] == 'AMBIGUOUS'), (i, v_cause[i], pure)
    assert 50 < retried < n - 50                              # the comparison exercised both outcomes
