"""Phase 3 candidate features: point-in-time behaviour of each of the 31 additions."""
import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from services.app.domain import failure_vocab as vocab
from services.app.domain.features import FEATURES, FeatureBuilder
from services.app.domain.features_v4 import (
    CATEGORICAL_FEATURES_V4, FEATURE_GROUPS_V4, FEATURES_V4, NEW_FEATURES_V4, DowntimeNotice, FeatureBuilderV4,
    PaymentObservationV4,
)

BASE = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)  # 17:30 IST, 31-day month => 21 days to month end


def mk(pid, at_s, status='failed', *, issuer='A', psp='p1', method='upi', inst='i1', cust='c1', amount=10000,
       reason=None, order=None, latency=100., duration=None, flow='upi_intent', step='payment_debit_response',
       hint='na'):
    at = BASE + timedelta(seconds=at_s)
    failed = status == 'failed'
    return PaymentObservationV4(
        payment_id=pid, order_id=order or f'o_{pid}', customer_id=cust, issuer_id=issuer, psp_id=psp,
        amount_minor=amount, method=method, occurred_at=at, observed_at=at + timedelta(seconds=1), status=status,
        instrument_hint=hint, gw_error_code='BAD_REQUEST_ERROR' if failed else '',
        gw_error_source='issuer_bank' if failed else '', gw_error_step=step if failed else '', latency_ms=latency,
        instrument_id=inst, error_reason=reason, payment_flow=flow, attempt_duration_s=duration)


def replay(builder, events):
    for event in events:
        builder.observe(event)
    return builder


def test_schema_has_26_plus_31_unique_features_in_six_groups():
    assert FEATURES_V4[:26] == FEATURES and len(FEATURES_V4) == 57 == len(set(FEATURES_V4))
    assert len(NEW_FEATURES_V4) == 31 and set(FEATURE_GROUPS_V4) == {
        'A_reason', 'B_auth_timing', 'C_instrument', 'D_infrastructure', 'E_customer', 'F_external_calendar'}
    assert [len(g) for g in FEATURE_GROUPS_V4.values()] == [4, 4, 7, 10, 4, 2]
    assert set(CATEGORICAL_FEATURES_V4) <= set(FEATURES_V4)


def test_build_returns_exactly_the_schema_and_never_mutates_state():
    builder = replay(FeatureBuilderV4(), [mk('a', 0, reason='insufficient_funds')])
    current = mk('now', 100, reason='insufficient_funds')
    first, second = builder.build(current), builder.build(current)
    assert list(first) == list(FEATURES_V4) and first == second


def test_reason_group_documented_mapping_generic_and_missing():
    builder = FeatureBuilderV4()
    row = builder.build(mk('x', 10, reason='bank_technical_error', step='payment_authorization', duration=120.))
    assert (row['error_reason'], row['reason_family'], row['is_generic_reason']) == ('bank_technical_error', 'bank_technical', 0)
    assert row['step_ordinal'] == 3 and row['attempt_duration_s'] == 120. and row['duration_vs_timeout_ratio'] == pytest.approx(.2)
    generic = builder.build(mk('y', 20, reason='payment_failed'))
    assert generic['reason_family'] == 'generic' and generic['is_generic_reason'] == 1
    missing = builder.build(mk('z', 30, reason=None, flow=None, duration=None))
    assert missing['error_reason'] == 'missing' and missing['reason_family'] == 'missing'
    assert missing['is_generic_reason'] == 1 and missing['payment_flow'] == 'missing'
    assert missing['duration_vs_timeout_ratio'] is None
    assert builder.build(mk('w', 40, reason='something_new'))['reason_family'] == 'other'


def test_instrument_history_streak_previous_reason_and_distinct_reasons():
    builder = replay(FeatureBuilderV4(), [mk('1', 0, reason='card_expired'), mk('2', 60, reason='card_expired'),
                                          mk('3', 120, reason='payment_failed')])
    row = builder.build(mk('c', 200, reason='card_expired'))
    assert row['instrument_consecutive_failures'] == 3 and row['instrument_attempts_24h'] == 3
    assert row['instrument_failure_rate_24h'] == 1.0 and row['instrument_last_success_age_s'] is None
    assert row['instrument_prev_reason'] == 'payment_failed' and row['same_reason_as_prev'] == 0
    assert row['instrument_distinct_reasons_24h'] == 2
    repeat = builder.build(mk('d', 210, reason='payment_failed'))
    assert repeat['same_reason_as_prev'] == 1


def test_instrument_success_resets_streak_and_sets_last_success_age():
    builder = replay(FeatureBuilderV4(), [mk('1', 0, reason='x'), mk('2', 30, 'success'), mk('3', 60, reason='y')])
    row = builder.build(mk('c', 90, reason='y'))
    assert row['instrument_consecutive_failures'] == 1
    assert row['instrument_last_success_age_s'] == pytest.approx(91 - 31)


def test_instrument_window_is_24h_but_last_success_persists():
    builder = replay(FeatureBuilderV4(), [mk('1', 0, 'success'), mk('2', 100, reason='x')])
    row = builder.build(mk('c', 3 * 86400, reason='x'))
    assert row['instrument_attempts_24h'] == 0 and row['instrument_failure_rate_24h'] is None
    assert row['instrument_prev_reason'] == 'none' and row['instrument_last_success_age_s'] > 2 * 86400
    unknown = FeatureBuilderV4().build(mk('u', 5, inst='unknown', reason='x'))
    assert unknown['instrument_attempts_24h'] is None and unknown['instrument_prev_reason'] == 'none'


def _issuer_stream():
    events = [mk(f'old{i}', 6500 + 100 * i, 'success', inst=f'o{i}', cust=f'oc{i}') for i in range(20)]
    reasons = ['request_timed_out', 'request_timed_out', 'bank_technical_error', 'payment_failed']
    recent = []
    for i in range(10):
        if i < 4:
            recent.append(mk(f'f{i}', 9700 + 30 * i, reason=reasons[i], inst=f'r{i}', cust=f'rc{i}'))
        else:
            recent.append(mk(f'f{i}', 9700 + 30 * i, 'success', inst=f'r{i}', cust=f'rc{i}'))
    return events + recent


def test_issuer_rolling_rates_timeouts_tech_share_and_delta():
    builder = replay(FeatureBuilderV4(), _issuer_stream())
    row = builder.build(mk('now', 10000, reason='x'))
    assert row['issuer_failure_rate_1h'] == pytest.approx(4 / 30)
    assert row['issuer_timeout_rate_5m'] == pytest.approx(2 / 10)
    assert row['issuer_tech_reason_share_5m'] == pytest.approx(1 / 4)
    assert row['issuer_failure_delta'] == pytest.approx(.4 - 4 / 30)
    assert row['issuer_method_failure_rate_5m'] == pytest.approx(.4)
    assert row['global_failure_rate_5m'] == pytest.approx(.4)
    assert row['psp_failure_rate_1h'] == pytest.approx(4 / 30) and row['psp_timeout_rate_5m'] == pytest.approx(.2)
    # Other entities see no such history.
    other = builder.build(mk('o', 10000, issuer='B', psp='p2', method='card', inst='z', cust='zz', reason='x'))
    assert other['issuer_failure_rate_1h'] is None and other['issuer_method_failure_rate_5m'] is None
    assert other['global_failure_rate_5m'] == pytest.approx(.4)


def test_issuer_baseline_zscore_needs_enough_history():
    stream = _issuer_stream()
    assert FeatureBuilderV4 and replay(FeatureBuilderV4(), stream).build(mk('n', 10000, reason='x'))[
        'issuer_failure_vs_baseline_7d'] is None
    history = [mk(f'h{i}', -2 * 86400 + 60 * i, 'success' if i % 10 else 'failed', inst=f'h{i}', cust=f'hc{i}',
                  reason='x') for i in range(60)]
    builder = replay(FeatureBuilderV4(), history + stream)
    row = builder.build(mk('n', 10000, reason='x'))
    base = (6 + 4) / (60 + 30)
    sd = math.sqrt(base * (1 - base) / 30)
    assert row['issuer_failure_vs_baseline_7d'] == pytest.approx((4 / 30 - base) / sd)


def test_latency_relative_to_issuer_median():
    builder = replay(FeatureBuilderV4(), [mk(f'l{i}', 100 * i, 'success', latency=v, inst=f'l{i}', cust=f'lc{i}')
                                          for i, v in enumerate([100., 200., 300.])])
    assert builder.build(mk('c', 500, latency=400., reason='x'))['latency_vs_issuer_p50'] == pytest.approx(2.)
    assert builder.build(mk('d', 510, latency=None, reason='x'))['latency_vs_issuer_p50'] is None


def test_customer_behaviour_window_amounts_and_methods():
    events = [mk('old', -15 * 86400, 'success', cust='c', inst='i0', amount=90000),
              mk('a', 0, 'success', cust='c', inst='i1', amount=20000, method='card'),
              mk('b', 100, 'failed', cust='c', inst='i2', amount=50000, reason='x', method='upi'),
              mk('d', 200, 'success', cust='c', inst='i3', amount=30000, method='netbanking')]
    row = FeatureBuilderV4().__class__().build(mk('q', 0, cust='nobody', reason='x'))
    assert row['secs_since_prev_attempt'] is None and row['amount_vs_max_success_amount'] is None
    assert row['customer_success_rate_14d'] is None and row['customer_methods_tried_24h'] == 0
    builder = replay(FeatureBuilderV4(), events)
    row = builder.build(mk('now', 400, cust='c', amount=60000, reason='x'))
    assert row['secs_since_prev_attempt'] == pytest.approx(400 + 1 - 201)
    assert row['customer_methods_tried_24h'] == 3
    assert row['amount_vs_max_success_amount'] == pytest.approx(60000 / 30000)  # the 15-day-old 90,000 is outside 14d
    assert row['customer_success_rate_14d'] == pytest.approx(2 / 3)


def test_downtime_matching_method_entity_resolution_and_staleness():
    def build(builder, **kw):
        return builder.build(mk('x', kw.pop('at', 100), reason='r', **kw))['matched_downtime_severity']
    builder = FeatureBuilderV4()
    builder.notify(DowntimeNotice('d1', 'started', BASE, method='upi', severity='high', issuer_id='A'))
    assert build(builder) == 2
    assert build(builder, method='card') == 0       # different method
    assert build(builder, issuer='B') == 0          # different issuer
    builder.notify(DowntimeNotice('d2', 'started', BASE + timedelta(seconds=150), method='card', severity='medium',
                                  psp_id='p9'))
    assert build(builder, method='card', psp='p9', issuer='Z', at=200) == 1
    builder.notify(DowntimeNotice('d1', 'resolved', BASE + timedelta(seconds=300)))
    assert build(builder, at=400) == 0
    assert build(builder, method='card', psp='p9', issuer='Z', at=13 * 3600) == 0  # stale: never resolved


def test_downtime_notice_cannot_leak_backwards():
    builder = FeatureBuilderV4()
    builder.notify(DowntimeNotice('d1', 'started', BASE + timedelta(seconds=500), method='upi', severity='high',
                                  issuer_id='A'))
    with pytest.raises(ValueError):
        builder.build(mk('early', 10, reason='r'))  # watermark already at 500s: replay must be ordered


def test_calendar_days_to_month_end_uses_ist_local_date():
    assert FeatureBuilderV4().build(mk('x', 0, reason='r'))['days_to_month_end'] == 21
    near_midnight = datetime(2026, 5, 31, 19, 0, tzinfo=timezone.utc)  # 00:30 IST on 1 June
    obs = PaymentObservationV4(payment_id='m', order_id='o', customer_id='c', issuer_id='A', psp_id='p', amount_minor=100,
                               method='upi', occurred_at=near_midnight, observed_at=near_midnight, status='failed')
    assert FeatureBuilderV4().build(obs)['days_to_month_end'] == 29


def test_duplicates_and_out_of_order_replay():
    builder = FeatureBuilderV4()
    event = mk('dup', 0, reason='x')
    builder.observe(event)
    builder.observe(event)
    assert builder.build(mk('c', 100, reason='x'))['instrument_attempts_24h'] == 1
    with pytest.raises(ValueError):
        builder.build(mk('earlier', -500, reason='x'))


def test_unobserved_attempts_do_not_influence_features():
    with_drop = FeatureBuilderV4()
    without = FeatureBuilderV4()
    replay(with_drop, [mk('a', 0, reason='x'), mk('b', 50, reason='x')])
    replay(without, [mk('a', 0, reason='x')])  # 'b' lost by telemetry: never observed
    assert with_drop.build(mk('c', 90, reason='x')) != without.build(mk('c', 90, reason='x'))
    assert without.build(mk('c', 90, reason='x'))['instrument_attempts_24h'] == 1


def test_original_26_features_match_the_unchanged_builder():
    rng = random.Random(3)
    legacy, extended = FeatureBuilder(include_candidates=True), FeatureBuilderV4()
    events = []
    for i in range(80):
        ok = rng.random() < .6
        events.append(mk(f'p{i}', 20 * i, 'success' if ok else 'failed', issuer=rng.choice('AB'), psp=rng.choice(['p1', 'p2']),
                         method=rng.choice(['upi', 'card']), cust=f'c{rng.randrange(6)}', inst=f'i{rng.randrange(8)}',
                         amount=rng.choice([5000, 20000, 90000]), reason=None if ok else rng.choice(['a', 'insufficient_funds']),
                         latency=rng.choice([None, 80., 300.])))
    checked = 0
    for event in events:
        if event.status == 'failed':
            old, new = legacy.build(event.legacy()), extended.build(event)
            for key in FEATURES:
                assert (old[key] is None and new[key] is None) or old[key] == new[key], key
            checked += 1
        legacy.observe(event.legacy())
        extended.observe(event)
    assert checked > 10


def test_no_feature_name_hints_at_labels_or_latent_state():
    banned = ('true_cause', 'fixture', 'label', 'cause', 'outage', 'episode', 'latent', 'severity_truth')
    assert not [f for f in FEATURES_V4 if any(b in f for b in banned)]


def test_vocabulary_tables_are_internally_consistent():
    assert vocab.GENERIC_REASONS <= vocab.DOCUMENTED_REASONS
    assert set(vocab.REASON_FAMILY) <= vocab.DOCUMENTED_REASONS
    assert vocab.reason_family('card_expired') == 'instrument_state' and vocab.reason_family('payment_risk_check_failed') == 'risk'
    assert vocab.step_ordinal('payment_initiation') == 0 and vocab.step_ordinal('payment_capture') == 4
    assert vocab.step_ordinal('') is None and vocab.step_ordinal(None) is None
