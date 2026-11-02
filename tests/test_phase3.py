"""Phase 3 evaluation logic: split independence, metric helpers, predeclared selection rules."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from eval import phase3
from eval.phase3 import (GATE, Phase3Model, SCENARIOS, ROLE_COUNT, ROLE_SEED, evaluate, fast_macro_f1,
                         paired_bootstrap, select_features, upper95, world_seed)
from eval.probability_calibration import ProbabilityCalibrator
from eval.sequences_v4 import assemble_v4, generate_world
from services.app.domain.features import BASELINE_FEATURES, FEATURES
from services.app.domain.features_v4 import FEATURE_GROUPS_V4, FEATURES_V4, NEW_FEATURES_V4


def test_world_seeds_are_unique_and_roles_cannot_collide():
    seeds = [world_seed(s, role, i) for s in SCENARIOS for role, n in ROLE_COUNT.items() for i in range(n)]
    assert len(seeds) == len(set(seeds)) == 33
    bands = sorted(ROLE_SEED.values())
    assert all(b - a >= 100000 for a, b in zip(bands, bands[1:]))
    assert phase3.ROLE_MARKET['shifted_test'] != phase3.ROLE_MARKET['test'] == phase3.ROLE_MARKET['train']


def test_frozen_gate_is_the_phase_2_5_control_not_retuned():
    assert (GATE.min_confidence, GATE.max_hard_mass, GATE.retry_enabled) == (.60, .03, True)


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 6, 400), rng.integers(0, 6, 400)
    labels = [str(i) for i in range(6)]
    expected = f1_score(y.astype(str), p.astype(str), labels=labels, average='macro', zero_division=0)
    assert fast_macro_f1(y, p, 6) == pytest.approx(expected)


def test_paired_bootstrap_detects_real_gain_and_ignores_identical_models():
    rng = np.random.default_rng(1)
    labels = ['a', 'b', 'c']
    y = rng.choice(labels, 600)
    good = y.copy()
    bad = np.where(rng.random(600) < .5, rng.choice(labels, 600), y)
    same = paired_bootstrap(y, good, good, labels, draws=50)
    assert same['delta_mean'] == 0 and same['lo95'] == 0 == same['hi95']
    gain = paired_bootstrap(y, good, bad, labels, draws=100)
    assert gain['lo95'] > 0
    other = np.where(np.random.default_rng(9).random(600) < .5, rng.choice(labels, 600), y)  # equally good, independent
    noise = paired_bootstrap(y, bad, other, labels, draws=100)
    assert noise['lo95'] < 0 < noise['hi95']      # two equally noisy models must not look significantly different


def test_upper95_exact_binomial_bounds():
    assert upper95(0, 100) == pytest.approx(1 - .05 ** (1 / 100))
    assert upper95(5, 5) == 1.0 and upper95(0, 0) is None
    assert upper95(3, 100) > .03


def _analysis(group_lo, feature_lofo):
    ablation = {g: {'drop_from_57': {'loss_vs_full_57': {'lo95': group_lo.get(g, -.01)}},
                    'add_to_26': {'gain_vs_existing_26': {'lo95': -.01}}} for g in FEATURE_GROUPS_V4}
    lofo = {f: {'dlogloss': feature_lofo.get(f, 0.), 'dmacro_f1': 0.} for f in NEW_FEATURES_V4}
    perm = {f: {'dlogloss': 0., 'dmacro_f1': 0.} for f in FEATURES_V4}
    return {'ablation': ablation, 'leave_one_feature_out': lofo, 'permutation_importance': perm}


def test_selection_rules_need_evidence_in_two_scenarios_and_never_resurrect_dropped_groups():
    analyses = {
        'calm': _analysis({'A_reason': .02, 'B_auth_timing': .02}, {'error_reason': .01, 'attempt_duration_s': .05}),
        'typical': _analysis({'A_reason': .01}, {'error_reason': .01, 'step_ordinal': .01}),
        'stressed': _analysis({}, {'attempt_duration_s': .05, 'issuer_failure_rate_1h': .5}),
    }
    selection = select_features(analyses)
    assert selection['groups']['A_reason']['retained'] and not selection['groups']['B_auth_timing']['retained']
    assert 'error_reason' in selection['selected_new_features']           # useful in 2 of 3
    assert 'step_ordinal' not in selection['selected_new_features']       # only 1 of 3
    assert 'attempt_duration_s' not in selection['selected_new_features'] # its group was not retained
    assert 'issuer_failure_rate_1h' not in selection['selected_new_features']
    assert selection['selected_feature_set'][:26] == list(FEATURES)
    assert set(selection['selected_new_features']) | set(selection['dropped_new_features']) == set(NEW_FEATURES_V4)


def test_add_to_26_evidence_alone_can_retain_a_group():
    analyses = {}
    for name in SCENARIOS:
        a = _analysis({}, {'matched_downtime_severity': .01})
        a['ablation']['F_external_calendar']['add_to_26']['gain_vs_existing_26']['lo95'] = .02 if name != 'calm' else -.01
        analyses[name] = a
    selection = select_features(analyses)
    assert selection['groups']['F_external_calendar']['retained']
    assert 'matched_downtime_severity' in selection['selected_new_features']


@pytest.fixture(scope='module')
def tiny_frames():
    frames = []
    for i, seed in enumerate((71, 72, 73)):
        frame = assemble_v4(generate_world(1500, seed, scenario='typical')).failures
        frame['world_index'] = i
        frames.append(frame)
    return frames


def test_model_handles_unseen_categories_missing_columns_values_and_every_feature_set(tiny_frames):
    train = pd.concat(tiny_frames[:2], ignore_index=True)
    for features in (BASELINE_FEATURES, FEATURES, FEATURES_V4):
        model = Phase3Model(features).fit(train)
        probe = tiny_frames[2].head(40).copy()
        probe['error_reason'] = 'never_seen_reason'
        probe['issuer_id'] = 'issuer_zz'
        p = model.predict_proba(probe)
        assert p.shape == (40, 6) and np.allclose(p.sum(axis=1), 1)
    assert Phase3Model(FEATURES_V4).categorical == [
        'gw_error_code', 'gw_error_source', 'gw_error_step', 'method', 'instrument_hint', 'issuer_id', 'psp_id',
        'error_reason', 'reason_family', 'payment_flow', 'instrument_prev_reason']


def test_evaluate_reports_consistent_counts_and_safety_denominators(tiny_frames):
    train, validation, test = pd.concat(tiny_frames[:1]), tiny_frames[1], tiny_frames[2:]
    model = Phase3Model(FEATURES_V4).fit(train)
    calibrator = ProbabilityCalibrator('sigmoid', model.classes_).fit(model.predict_proba(validation),
                                                                    validation.true_cause.to_numpy())
    result = evaluate(model, calibrator, test)
    matrix = np.array(result['confusion_matrix']['rows_true_cols_predicted'])
    assert matrix.sum() == result['n_failures'] == len(test[0])
    assert sum(v['support'] for v in result['per_class'].values()) == result['n_failures']
    assert result['unsafe_retry_gated']['hard_declines'] == int(test[0].true_cause.isin(['CARD_EXPIRED', 'RISK_BLOCKED']).sum())
    assert result['unsafe_retry_gated']['count'] <= result['unsafe_retry_gated']['hard_declines']
    assert 0 <= result['gate_abstention_rate'] <= 1 and 0 <= result['accuracy']['pooled'] <= 1
    assert result['calibration_sigmoid']['multiclass_brier'] >= 0
