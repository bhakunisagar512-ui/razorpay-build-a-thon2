from datetime import datetime, timedelta, timezone

import pytest

from services.app.domain.features import BASELINE_FEATURES, FEATURES, FeatureBuilder, PaymentObservation

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


def observation(pid, seconds, **kw):
    values = dict(payment_id=pid, order_id='order', customer_id='customer', issuer_id='bank', psp_id='psp',
                  amount_minor=10000, method='card', status='failed',
                  occurred_at=T0 + timedelta(seconds=seconds), observed_at=T0 + timedelta(seconds=seconds))
    values.update(kw)
    return PaymentObservation(**values)


def test_cold_start_schema_and_missing_rates():
    features = FeatureBuilder(include_candidates=True).build(observation('current', 0))
    assert len(FEATURES) == 26 and len(BASELINE_FEATURES) == 5
    assert tuple(features) == FEATURES
    assert features['issuer_attempts_5m'] == 0
    assert features['issuer_failure_rate_5m'] is None
    assert features['customer_last_success_age_s'] is None
    assert not {'true_cause', 'status', 'recovered_minor', 'payment_id'} & set(features)


def test_success_and_failure_denominator_excludes_current_and_duplicates():
    builder = FeatureBuilder(include_candidates=True)
    success = observation('success', 0, status='success', latency_ms=100)
    builder.observe(success)
    builder.observe(success)
    builder.observe(observation('failure', 60, latency_ms=200))
    current = observation('current', 120)
    features = builder.build(current)
    assert features['issuer_attempts_5m'] == 2
    assert features['issuer_failure_rate_5m'] == .5
    assert features['issuer_latency_p95_ms_5m'] == 200
    assert features['customer_attempts_7d'] == 2
    assert features['customer_successes_7d'] == 1
    assert features['order_prior_attempts_7d'] == 2
    builder.observe(current)
    assert builder.build(current) == features


def test_late_arrivals_are_not_available_to_earlier_predictions():
    builder = FeatureBuilder(include_candidates=True)
    builder.observe(observation('prior', 0))
    before = builder.build(observation('before', 100))
    late = observation('late', 10, observed_at=T0 + timedelta(seconds=200), status='success')
    builder.observe(late)
    after = builder.build(observation('after', 210))
    assert before['customer_successes_7d'] == 0
    assert after['customer_successes_7d'] == 1
    with pytest.raises(ValueError, match='nondecreasing'):
        builder.build(observation('past', 150))


def test_simultaneous_observations_do_not_see_each_other():
    builder = FeatureBuilder(include_candidates=True)
    builder.observe(observation('same-time', 10, status='success'))
    assert builder.build(observation('current', 10))['customer_attempts_7d'] == 0


def test_stale_telemetry_is_not_recent_health():
    builder = FeatureBuilder(include_candidates=True)
    builder.observe(observation('old', 0, status='success'))
    features = builder.build(observation('now', 301))
    assert features['issuer_attempts_5m'] == 0
    assert features['issuer_failure_rate_5m'] is None
    assert features['customer_attempts_7d'] == 1
    builder.observe(observation('recent', 302))
    # Exercise global eviction after infrastructure-window pruning.
    features = builder.build(observation('week-later', 8 * 86400))
    assert features['customer_attempts_7d'] == 0


def test_separate_infrastructure_and_customer_histories():
    builder = FeatureBuilder(include_candidates=True)
    builder.observe(observation('other', 0, customer_id='other-customer', issuer_id='other-bank'))
    features = builder.build(observation('current', 30))
    assert features['customer_attempts_7d'] == features['issuer_attempts_5m'] == 0
    assert features['psp_attempts_5m'] == 1


@pytest.mark.parametrize('changes', [
    {'occurred_at': datetime(2026, 8, 1)},
    {'observed_at': T0 - timedelta(seconds=1)},
    {'amount_minor': 0}, {'latency_ms': float('nan')}, {'latency_ms': -1}, {'status': 'unknown'},
])
def test_invalid_observations_rejected(changes):
    with pytest.raises(ValueError):
        observation('bad', 0, **changes)
