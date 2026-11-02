"""sequences@v4 data-quality regression tests (small worlds; cheap enough for CI)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.phase3_quality import lookup_ceilings, world_quality
from eval.sequences_v4 import assemble_v4, generate_world, rebuild_features_at
from eval.world_config_v4 import (
    GENERIC_BY_METHOD, SCENARIOS_V4, SPECIFIC_REASONS, STEP_SOURCE, UPI_ANCHORS,
)
from services.app.domain.failure_vocab import DOCUMENTED_REASONS, ERROR_CODES, SOURCES, STEPS
from services.app.domain.features_v4 import FEATURES_V4
from services.app.domain.taxonomy import Cause

SIX = {c for c in map(str, Cause) if c != 'AMBIGUOUS'}


@pytest.fixture(scope='module')
def typical():
    world = generate_world(5000, 21, scenario='typical')
    return world, assemble_v4(world)


def test_emission_tables_use_only_documented_vocabulary():
    for cause, methods in SPECIFIC_REASONS.items():
        for method, table in methods.items():
            assert set(table) <= DOCUMENTED_REASONS, (cause, method)
            assert all(w > 0 for w in table.values())
    for method, table in GENERIC_BY_METHOD.items():
        assert set(table) <= DOCUMENTED_REASONS
    for method, causes in STEP_SOURCE.items():
        for cause, (steps, sources) in causes.items():
            assert set(steps) <= set(STEPS[method]) and set(sources) <= set(SOURCES[method]), (method, cause)


def test_generated_failures_use_documented_vocabulary_and_six_labels(typical):
    _, data = typical
    tx = data.transactions
    failed = tx[(tx.status == 'failed')]
    assert set(failed.true_cause) <= SIX and tx[tx.status == 'success'].true_cause.isna().all()
    assert set(failed.gw_error_code) <= set(ERROR_CODES)
    assert set(failed.error_reason.dropna()) <= DOCUMENTED_REASONS
    assert all(r.gw_error_step in STEPS[r.method] and r.gw_error_source in SOURCES[r.method] for r in failed.itertuples())
    assert tx[tx.status == 'success'].error_reason.isna().all()


def test_population_has_successes_and_realistic_failure_rate(typical):
    _, data = typical
    measured = data.transactions[~data.transactions.is_warmup]
    rate = (measured.status == 'failed').mean()
    assert 0.04 < rate < 0.25            # v3 had ~50% failures; the base rate must be a minority
    assert set(measured.method) == {'card', 'upi', 'netbanking'}
    assert data.transactions.issuer_id.nunique() >= 10 and data.transactions.customer_id.nunique() > 500


def test_same_seed_reproduces_and_new_seed_changes_world():
    a = assemble_v4(generate_world(700, 5, scenario='calm')).failures
    b = assemble_v4(generate_world(700, 5, scenario='calm')).failures
    c = assemble_v4(generate_world(700, 6, scenario='calm')).failures
    pd.testing.assert_frame_equal(a, b)
    assert len(a) != len(c) or not a.payment_id.equals(c.payment_id) or not a.error_reason.equals(c.error_reason)


def test_outage_calendar_is_random_per_world_and_not_shared():
    signatures = set()
    for seed in (31, 32, 33):
        outages = generate_world(500, seed, scenario='typical').outages
        signatures.add(tuple(sorted(zip(outages.kind, outages.entity, outages.start_s.round()))))
    assert len(signatures) == 3
    same = generate_world(500, 31, scenario='typical').outages
    assert tuple(sorted(zip(same.kind, same.entity, same.start_s.round()))) in signatures


def test_instrument_hint_is_uninformative_about_expiry(typical):
    _, data = typical
    failed = data.transactions[(data.transactions.status == 'failed') & (data.transactions.method == 'card')]
    expired = failed[failed.true_cause == 'CARD_EXPIRED']
    assert len(expired) >= 20
    assert (expired.instrument_hint == 'expired').mean() < .10   # v3: 84.5%
    other = failed[failed.true_cause != 'CARD_EXPIRED']
    assert abs((expired.instrument_hint == 'unknown').mean() - (other.instrument_hint == 'unknown').mean()) < .2


def test_risk_declines_are_not_a_permanent_customer_flag(typical):
    _, data = typical
    tx = data.transactions[~data.transactions.is_warmup]
    risky = set(tx[tx.true_cause == 'RISK_BLOCKED'].instrument_id)
    assert len(risky) > 10 and (tx[tx.instrument_id.isin(risky)].status == 'success').mean() > .3   # v3: 0%


def test_overlapping_causes_and_ambiguous_reasons_exist(typical):
    world, data = typical
    assert data.metadata['failure_overlap_fraction'] > .01
    f = data.failures
    timed_out = f[f.error_reason == 'payment_timed_out'].true_cause.value_counts(normalize=True)
    assert len(timed_out) >= 3 and timed_out.iloc[0] < .95                # documented ambiguity is reproduced
    failed = f[f.error_reason == 'payment_failed']
    assert failed.true_cause.nunique() >= 5                                # the catch-all hides the cause


def test_labels_are_not_trivially_predictable_from_one_field(typical):
    world, data = typical
    other = assemble_v4(generate_world(5000, 22, scenario='typical')).failures
    ceilings = lookup_ceilings(data.failures, other)
    assert ceilings['error_tuple'] < .85 and ceilings['error_tuple+method+hint'] < .88
    assert ceilings['error_reason'] < .90 and max(ceilings.values()) < .96
    assert data.failures.true_cause.value_counts(normalize=True).iloc[0] < .6


def test_all_classes_present_in_a_small_world(typical):
    # Per-class adequacy at evaluation scale (>=200 train / >=60 test) is enforced in the data-quality report;
    # a single 5,000-order world only guarantees that no class vanishes.
    _, data = typical
    counts = data.failures.true_cause.value_counts()
    assert set(counts.index) == SIX and counts.min() >= 10


def test_scenarios_are_ordered_by_difficulty_and_documented_as_assumptions():
    rates = {}
    for name in ('calm', 'typical', 'stressed'):
        measured = generate_world(2500, 41, scenario=name)
        tx = pd.DataFrame(measured.rows)
        rates[name] = (tx[~tx.is_warmup].status == 'failed').mean()
    assert rates['calm'] < rates['typical'] < rates['stressed']
    calm, typical_cfg, stressed = (SCENARIOS_V4[n] for n in ('calm', 'typical', 'stressed'))
    assert calm.generic_share < typical_cfg.generic_share < stressed.generic_share
    assert calm.outage_rate_per_week < typical_cfg.outage_rate_per_week < stressed.outage_rate_per_week
    assert calm.telemetry_drop < typical_cfg.telemetry_drop < stressed.telemetry_drop


def test_typical_upi_declines_fall_inside_npci_informed_band():
    # Scenario MEAN over worlds: a single 3-day world can legitimately spike (random outages),
    # like NPCI bank-months do; the band is for the expected level.
    bd, td = [], []
    for seed in (900, 901, 902):
        quality = world_quality(assemble_v4(generate_world(3000, seed, scenario='typical')))
        bd.append(quality['upi_bd_pct'])
        td.append(quality['upi_td_pct'])
    lo, hi = UPI_ANCHORS['bd_pct_range']
    assert lo <= np.mean(bd) <= hi
    lo, hi = UPI_ANCHORS['td_pct_range']
    assert lo <= np.mean(td) <= hi


def test_npci_summary_supports_the_anchor_bands_when_present():
    path = Path('eval/data/npci_upi_summary.json')
    if not path.exists():
        pytest.skip('NPCI summary not generated')
    summary = json.loads(path.read_text())
    for year in summary['by_year'].values():
        assert UPI_ANCHORS['bd_pct_range'][0] <= year['volume_weighted_bd_pct'] <= UPI_ANCHORS['bd_pct_range'][1]
        assert UPI_ANCHORS['td_pct_range'][0] <= year['volume_weighted_td_pct'] <= UPI_ANCHORS['td_pct_range'][1]
    assert 'UPI remitter' in summary['scope']


def test_features_are_point_in_time_future_events_cannot_change_them():
    world = generate_world(800, 61, scenario='stressed')
    data = assemble_v4(world)
    rng = np.random.default_rng(0)
    for payment_id in rng.choice(data.failures.payment_id.to_numpy(), size=6, replace=False):
        stored = data.failures[data.failures.payment_id == payment_id].iloc[0]
        rebuilt = rebuild_features_at(world, payment_id)
        for name, value in rebuilt.items():
            assert (pd.isna(stored[name]) and pd.isna(value)) or stored[name] == value or \
                   abs(float(stored[name]) - float(value)) < 1e-9, name


def test_feature_frame_contains_only_measurement_failures_and_no_label_features(typical):
    world, data = typical
    warmup_ids = set(data.transactions[data.transactions.is_warmup].payment_id)
    assert not set(data.failures.payment_id) & warmup_ids
    assert data.failures.payment_id.is_unique and data.transactions.payment_id.is_unique
    assert not {'true_cause', 'n_causes_fired'} & set(FEATURES_V4)
    assert list(data.failures[list(FEATURES_V4)].columns) == list(FEATURES_V4)


def test_telemetry_noise_and_missingness_are_present(typical):
    _, data = typical
    assert 0.0 < (data.failures.error_reason == 'missing').mean() < .15
    assert data.failures.latency_vs_issuer_p50.isna().mean() > .0
    tx = data.transactions
    assert (~tx.telemetry_visible).mean() > 0 and tx.latency_ms.isna().mean() > 0


def test_generator_rejects_unusable_parameters():
    with pytest.raises(ValueError):
        generate_world(100, 1)
    with pytest.raises(ValueError):
        generate_world(1000, 1, warmup_days=3)
    with pytest.raises(KeyError):
        generate_world(1000, 1, scenario='unknown')
