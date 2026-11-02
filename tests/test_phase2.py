
import numpy as np
import pandas as pd
import pytest

from eval import legacy_generate
from eval.classifier import CauseClassifier, FEATURES as ORIGINAL_FEATURES, train_test_split_by_hash
from eval.datasets import assemble, temporal_split
from eval.feature_ablation import FEATURE_GROUPS, FeatureAblationClassifier
from eval.generate import generate, generate_dataset
from eval.scenarios import SCENARIOS, generate_scenario
from services.app.domain.features import BASELINE_FEATURES, FEATURES, FeatureBuilder, PaymentObservation
from services.app.domain.taxonomy import Cause
from dataclasses import fields


@pytest.fixture(scope='module')
def dataset():
    return generate_dataset(n=2000, seed=42, include_candidates=True)


def test_legacy_generator_exact_compatibility():
    pd.testing.assert_frame_equal(generate(n=300, seed=7), legacy_generate.generate(n=300, seed=7))
    assert ORIGINAL_FEATURES == list(BASELINE_FEATURES)


def test_sequential_reproducibility_and_seed_variation(dataset):
    pd.testing.assert_frame_equal(dataset.transactions, generate_dataset(2000, 42, include_candidates=True).transactions)
    assert set(dataset.failures.payment_id).isdisjoint(generate_dataset(300, 43).failures.payment_id)
    assert dataset.metadata['n_fresh_orders'] == 2000
    assert dataset.transactions[~dataset.transactions.is_warmup].order_id.nunique() == 2000
    assert set(dataset.transactions.status) == {'success', 'failed'}
    assert set(dataset.failures.true_cause) == {str(c) for c in Cause if c != Cause.AMBIGUOUS}
    assert dataset.transactions.order_id.duplicated().any()
    assert dataset.outcomes.later_success_observed.any()
    assert dataset.deliveries.is_duplicate.any()
    assert not dataset.failures.payment_id.duplicated().any()
    assert not dataset.outcomes.order_id.duplicated().any()
    assert (dataset.outcomes.loc[dataset.outcomes.later_success_observed, 'outcome_available_at'].notna()).all()


def test_batch_features_equal_online_replay(dataset):
    builder = FeatureBuilder(include_candidates=True)
    names = [f.name for f in fields(PaymentObservation)]
    expected = []
    for row in dataset.transactions.to_dict('records'):
        current = PaymentObservation(**{name: row[name] for name in names})
        if row['status'] == 'failed' and not row['is_warmup']:
            expected.append(builder.build(current))
        if row['telemetry_visible']:
            builder.observe(current)
    pd.testing.assert_frame_equal(pd.DataFrame(expected), dataset.failures[list(FEATURES)].reset_index(drop=True))


def test_temporal_split_has_no_shared_orders_and_is_chronological(dataset):
    split = temporal_split(dataset)
    train, validation, test = [split[name] for name in ('train', 'validation', 'test')]
    assert not any(frame.empty for frame in split.values())
    assert set(train.order_id).isdisjoint(validation.order_id)
    assert set(train.order_id).isdisjoint(test.order_id)
    assert set(validation.order_id).isdisjoint(test.order_id)
    assert train.observed_at.max() < validation.observed_at.min() < test.observed_at.min()


def test_shared_issuer_outage_increases_failure_rate():
    dataset = generate_dataset(6000, 42)
    events = dataset.transactions[~dataset.transactions.is_warmup]
    origin = events.occurred_at.min().normalize()
    elapsed = (events.occurred_at - origin).dt.total_seconds()
    outage = (events.issuer_id == 'issuer_A') & elapsed.between(.18 * 259200, .23 * 259200, inclusive='left')
    outside = (events.issuer_id == 'issuer_A') & ~outage
    assert events.loc[outage, 'status'].eq('failed').mean() > events.loc[outside, 'status'].eq('failed').mean() + .15
    assert events.loc[outage, 'customer_id'].nunique() > 1


@pytest.mark.parametrize('name', SCENARIOS)
def test_scenarios_independent_and_well_formed(name, monkeypatch):
    import eval.sequences
    monkeypatch.setattr(eval.sequences, 'generate_dataset', lambda *a, **kw: pytest.fail('Training simulator used by test scenario'))
    scenario = generate_scenario(name, 500, include_candidates=True)
    assert len(scenario.transactions) == 500
    assert scenario.failures[list(FEATURES)].shape[1] == 26
    assert not scenario.failures.true_cause.isna().any()
    assert set(scenario.failures.issuer_id).isdisjoint({'issuer_A', 'issuer_B', 'issuer_C', 'issuer_D'})
    if name == 'unseen_codes':
        assert scenario.failures.gw_error_code.str.startswith('NEW_BANK_REASON_').all()
    if name == 'telemetry_gap':
        assert scenario.failures.current_latency_ms.isna().all()


def test_ablation_preserves_baseline_predictions_and_safety_gate():
    train, test = train_test_split_by_hash(generate(1500).drop_duplicates('payment_id'))
    original = CauseClassifier().fit(train).predict_df(test)
    experiment = FeatureAblationClassifier(BASELINE_FEATURES).fit(train).predict_df(test)
    assert original.pred_cause.tolist() == experiment.pred_cause.tolist()
    assert original.pred_stage.tolist() == experiment.pred_stage.tolist()
    np.testing.assert_allclose(original.pred_conf, experiment.pred_conf)


def test_all_feature_groups_are_allowlisted():
    assert len(FEATURE_GROUPS['all_26']) == 26
    assert len(FEATURE_GROUPS['all_without_identifiers']) == 24
    for group in FEATURE_GROUPS.values():
        assert len(group) == len(set(group))
        assert set(group) <= set(FEATURES)


def test_hard_decline_collision_is_a_deliberate_stress_case():
    scenario = generate_scenario('hard_decline_collision', 500)
    hard = scenario.failures[scenario.failures.true_cause.isin(['CARD_EXPIRED', 'RISK_BLOCKED'])]
    assert len(hard) > 100
    assert hard.gw_error_code.eq('GATEWAY_ERROR').all()
    assert hard.gw_error_source.eq('issuer').all()


def test_psp_degradation_produces_shared_latency_spike(dataset):
    frame = dataset.transactions[~dataset.transactions.is_warmup]
    origin = frame.occurred_at.min().normalize()
    elapsed = (frame.occurred_at - origin).dt.total_seconds()
    same_psp = frame.psp_id.eq('psp_alpha')
    window = elapsed.between(.35 * 259200, .43 * 259200, inclusive='left')
    assert frame.loc[same_psp & window, 'latency_ms'].median() > 2 * frame.loc[same_psp & ~window, 'latency_ms'].median()
    assert frame.loc[same_psp & window, 'customer_id'].nunique() > 1


def test_mixed_missing_latency_survives_dataframe_replay(dataset):
    failure = dataset.transactions[(~dataset.transactions.is_warmup) & dataset.transactions.status.eq('failed')].iloc[0].to_dict()
    success = dataset.transactions[dataset.transactions.status.eq('success')].iloc[0].to_dict()
    failure['latency_ms'] = None
    result = assemble([failure, success], [], name='missing-latency-test', seed=42, include_candidates=True)
    assert result.failures.current_latency_ms.isna().all()
