from collections import Counter
from dataclasses import asdict
from datetime import timedelta
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eval.classifier import RULES
from eval.datasets import safety_split
from eval.generate import generate_dataset
from eval.probability_calibration import ProbabilityCalibrator, calibration_metrics, expected_calibration_error
from eval.safety_pipeline import apply_policy, prepare, trace_failures, tune_policy
from services.app.domain.diagnosis_safety import SafetyThresholds, decide

CLASSES = ['BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'RISK_BLOCKED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT']


@pytest.fixture(scope='module')
def corrected_data():
    return generate_dataset(2000, include_candidates=True)


def test_complete_horizon_late_psp_and_every_episode_covered(corrected_data):
    from eval.generator_checks import verify_generator
    data = corrected_data
    assert verify_generator(data)['passed']
    events = data.transactions[~data.transactions.is_warmup]
    start = pd.Timestamp(data.metadata['measurement_start'])
    end = pd.Timestamp(data.metadata['measurement_end'])
    assert events.occurred_at.max() >= end - timedelta(seconds=1)
    assert events.occurred_at.min() == start
    for episode in data.infrastructure.to_dict('records'):
        field = 'issuer_id' if episode['kind'] == 'issuer' else 'psp_id'
        mask = (events[field].eq(episode['entity']) &
                events.occurred_at.ge(start + timedelta(seconds=episode['start_s'])) &
                events.occurred_at.lt(start + timedelta(seconds=episode['end_s'])))
        assert mask.sum() > 0, episode


def test_warmup_is_seven_days_and_excluded_from_labels(corrected_data):
    data = corrected_data
    start = pd.Timestamp(data.metadata['measurement_start'])
    history = data.transactions[data.transactions.is_warmup]
    assert history.occurred_at.min() < start - timedelta(days=6.9)
    assert set(history.payment_id).isdisjoint(data.failures.payment_id)
    assert set(history.order_id).isdisjoint(data.outcomes.order_id)
    assert data.failures.customer_attempts_7d.min() > 0
    for instrument, group in data.transactions[~data.transactions.is_warmup].groupby('instrument_id'):
        first = group.observed_at.min()
        assert ((history.instrument_id == instrument) & history.telemetry_visible & (history.observed_at < first)).any()


def test_calibration_policy_and_confirmation_have_disjoint_orders(corrected_data):
    split = safety_split(corrected_data)
    previous_max = None
    used = set()
    for name in ('fit', 'calibration', 'policy', 'confirmation'):
        frame = split[name]
        assert not frame.empty
        assert {'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT'} <= set(frame.true_cause)
        assert not used & set(frame.order_id)
        used.update(frame.order_id)
        if previous_max is not None:
            assert previous_max < frame.observed_at.min()
        previous_max = frame.observed_at.max()


def test_later_success_cannot_change_earlier_snapshots(corrected_data):
    from eval.datasets import assemble
    frame = corrected_data.transactions.copy()
    cutoff = frame.observed_at.quantile(.8)
    frame.loc[frame.observed_at > cutoff, 'status'] = 'success'
    frame.loc[frame.observed_at > cutoff, 'true_cause'] = None
    replay = assemble(frame.to_dict('records'), [], name='future-mutation', seed=42, include_candidates=True)
    before = corrected_data.failures[corrected_data.failures.observed_at <= cutoff].reset_index(drop=True)
    after = replay.failures[replay.failures.observed_at <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize('method', ['uncalibrated', 'sigmoid', 'isotonic'])
def test_calibration_fits_only_heldout_labels_and_returns_probabilities(method):
    probabilities = np.array([[.8,.2],[.7,.3],[.3,.7],[.2,.8]])
    calibration_labels = np.array(['a','b','a','b'])
    calibrator = ProbabilityCalibrator(method, ['a','b']).fit(probabilities, calibration_labels)
    result = calibrator.transform(np.array([[.9,.1],[.4,.6]]))
    np.testing.assert_allclose(result.sum(axis=1), 1)
    assert np.isfinite(result).all() and (result >= 0).all()


def test_calibration_metrics_have_known_perfect_values():
    result = calibration_metrics(np.eye(2), ['CARD_EXPIRED','TD_ISSUER_DOWN'], ['CARD_EXPIRED','TD_ISSUER_DOWN'])
    assert all(value == 0 for value in result.values())
    assert expected_calibration_error([1.,1.], [1.,0.]) == .5


def test_retryable_rule_cannot_bypass_low_confidence_or_disagreement():
    probabilities = dict(zip(CLASSES, [.4,.3,.1,.1,.05,.05]))
    assert not decide(probabilities, rule_result='BD_INSUFFICIENT_FUNDS')['retry_allowed']
    probabilities = dict(zip(CLASSES, [.01,.01,.01,.01,.95,.01]))
    result = decide(probabilities, rule_result='BD_INSUFFICIENT_FUNDS')
    assert result['decision'] == 'rule_model_disagreement'


def test_unknown_codes_and_expired_instruments_veto_high_confidence_retry():
    probabilities = dict(zip(CLASSES, [.99,.002,.002,.002,.002,.002]))
    assert decide(probabilities, known_tuple=False)['decision'] == 'unknown_error_tuple'
    assert decide(probabilities, instrument_hint='expired')['decision'] == 'expired_instrument_veto'
    assert decide(probabilities, rule_result='RISK_BLOCKED')['decision'] == 'hard_rule_veto'
    assert not decide({'invented': 1.})['retry_allowed']


def test_vectorized_policy_matches_shared_pure_policy():
    rng = np.random.default_rng(42)
    p = rng.dirichlet(np.ones(6), 100)
    rules = np.asarray(['BD_INSUFFICIENT_FUNDS', 'RISK_BLOCKED', ''] * 34)[:100]
    frame = pd.DataFrame({'instrument_hint': ['expired','valid','unknown','na'] * 25})
    data = {'frame': frame, 'p': p, 'classes': np.asarray(CLASSES), 'rules': rules,
            'known': rng.random(100) > .5, 'top': np.asarray(CLASSES)[p.argmax(axis=1)],
            'confidence': p.max(axis=1), 'hard_mass': p[:,2:4].sum(axis=1)}
    thresholds = SafetyThresholds(.6,.1)
    causes, reasons, retry = apply_policy(data, thresholds)
    for i in range(100):
        result = decide(dict(zip(CLASSES,p[i])), rule_result=rules[i] or None,
                        instrument_hint=frame.iloc[i].instrument_hint, known_tuple=data['known'][i], thresholds=thresholds)
        assert (causes[i],reasons[i],bool(retry[i])) == (result['cause'],result['decision'],result['retry_allowed'])


def test_every_unsafe_rule_override_has_a_complete_trace():
    frame = pd.DataFrame([{'payment_id':'pay_trace','true_cause':'CARD_EXPIRED', 'instrument_hint':'unknown',
                           'gw_error_code':'GATEWAY_ERROR','gw_error_source':'issuer','gw_error_step':'authorization'}])
    model = SimpleNamespace(classes_=CLASSES, known_tuples=set(RULES))
    p = np.array([[.01,.94,.01,.01,.02,.01]])
    data = prepare(frame,model,p)
    stream, counters = StringIO(), Counter()
    result = apply_policy(data,legacy=True)
    trace_failures(stream,counters,data,p,result,group='baseline',method='uncalibrated',partition='validation',policy='legacy',thresholds={})
    record = json.loads(stream.getvalue())
    assert record['true_cause'] == 'CARD_EXPIRED'
    assert record['rule_result'] == 'TD_ISSUER_DOWN'
    assert record['diagnostic_flags']['rule_overrode_model']
    assert len(record['class_probabilities']) == 6
    assert record['final_gate_decision'] == 'legacy_retryable_rule'
    assert sum(counters.values()) == 1
    assert not apply_policy(data)[2].any()


def test_threshold_search_traces_every_grid_candidate_and_enforces_safety():
    frame = pd.DataFrame([{'payment_id':'pay_tune','true_cause':'CARD_EXPIRED','instrument_hint':'unknown',
                           'gw_error_code':'BAD_REQUEST_ERROR','gw_error_source':'bank','gw_error_step':'authorization'}])
    model = SimpleNamespace(classes_=CLASSES, known_tuples=set(RULES))
    data = prepare(frame,model,np.array([[.99,.002,.002,.002,.002,.002]]))
    calls = []
    selected, grid = tune_policy({'validation':data}, on_evaluation=lambda *args: calls.append(args))
    assert len(calls) == len(grid) == 31
    assert selected['partitions']['validation']['unsafe_retry_count'] == 0
    assert any(row['partitions']['validation']['unsafe_retry_count'] > 0 for row in grid)
