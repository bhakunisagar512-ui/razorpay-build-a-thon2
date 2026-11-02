"""Phase 4: availability audit, model wrappers, confidence metrics, frozen selection rule, inactive adapter."""
import hashlib
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from eval import phase4
from eval.phase4_confidence import (GATE, brier, calibration_block, classification_block, confidence_profile, reliability_bins,
                                    safety_block, subsets, upper95)
from eval.phase4_models import FAMILIES, SIMPLICITY_RANK, BenchModel, trial_params
from eval.probability_calibration import ProbabilityCalibrator
from eval.sequences_v4 import assemble_v4, generate_world
from services.app.adapters import diagnosis, diagnosis_v4
from services.app.domain.deployment_features import (
    AVAILABILITY, DEPLOYMENT_CORE, DEPLOYMENT_WITH_TIMING, DERIVED_FROM_HEADLINE, HEADLINE_FEATURES, SIMULATED, TIMING, of_class)
from services.app.domain.features_v4 import FEATURES_V4

CLASSES = ['BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'RISK_BLOCKED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT']


# ------------------------------------------------------------------ availability audit
def test_every_candidate_feature_is_classified_and_deployment_sets_are_nested():
    assert set(AVAILABILITY) == set(FEATURES_V4)
    assert set(DEPLOYMENT_CORE) < set(DEPLOYMENT_WITH_TIMING) <= set(FEATURES_V4)
    assert not set(DEPLOYMENT_CORE) & (set(of_class(SIMULATED)) | set(of_class(TIMING)))
    assert not set(DEPLOYMENT_WITH_TIMING) & set(of_class(SIMULATED))


def test_deployment_core_keeps_the_reason_but_not_the_unverifiable_duration_or_gateway_fields():
    assert 'error_reason' in DEPLOYMENT_CORE and 'reason_family' in DEPLOYMENT_CORE
    for name in ('attempt_duration_s', 'psp_id', 'psp_failure_rate_1h', 'payment_flow', 'current_latency_ms'):
        assert name not in DEPLOYMENT_CORE
    assert 'attempt_duration_s' in DEPLOYMENT_WITH_TIMING and 'psp_id' not in DEPLOYMENT_WITH_TIMING


def test_ablation_configs_remove_exactly_the_requested_fields():
    configs = phase4.feature_configs()
    assert len(configs['full_57']) == 57 and len(configs['selected_42']) == 42
    for base in ('full_57', 'selected_42', 'deployment_with_timing'):
        assert not set(HEADLINE_FEATURES) & set(configs[f'{base}-no_headline3'])
        assert 'attempt_duration_s' not in configs[f'{base}-no_duration'] and 'error_reason' in configs[f'{base}-no_duration']
        assert 'error_reason' not in configs[f'{base}-no_reason'] and 'reason_family' not in configs[f'{base}-no_reason']
        closure = set(configs[f'{base}-no_headline3_or_derived'])
        assert not closure & (set(HEADLINE_FEATURES) | set(DERIVED_FROM_HEADLINE))
    assert 'attempt_duration_s' not in configs['deployment_core']
    assert set(configs['deployment_core-no_reason']) == set(DEPLOYMENT_CORE) - {'error_reason', 'reason_family'}


# ------------------------------------------------------------------ model families
@pytest.fixture(scope='module')
def tiny():
    frames = [assemble_v4(generate_world(1500, seed, scenario='typical')).failures for seed in (81, 82, 83)]
    for i, f in enumerate(frames):
        f['world_key'] = f'w{i}'
    return frames


def _available(family):
    if family == 'xgboost':
        pytest.importorskip('xgboost')
    if family == 'catboost':
        pytest.importorskip('catboost')


@pytest.mark.parametrize('family', FAMILIES)
def test_every_family_gives_valid_probabilities_and_survives_hostile_input(family, tiny):
    _available(family)
    params = {'hgb': {'max_iter': 20}, 'xgboost': {'n_estimators': 20}, 'catboost': {'iterations': 30},
              'random_forest': {'n_estimators': 30}}[family]
    model = BenchModel(family, DEPLOYMENT_CORE, params, threads=2).fit(pd.concat(tiny[:2], ignore_index=True))
    assert model.classes_ == sorted(model.classes_) and len(model.classes_) == 6
    probe = tiny[2].head(30).copy()
    p = model.predict_proba(probe)
    assert p.shape == (30, 6) and np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    hostile = probe.copy()
    hostile['error_reason'] = 'a_reason_never_seen'
    hostile['issuer_id'] = None
    hostile['instrument_failure_rate_24h'] = 'garbage'
    hostile['hour_ist'] = np.inf
    hostile = hostile.drop(columns=['method_failure_rate_5m', 'reason_family'])      # absent columns
    q = model.predict_proba(hostile)
    assert q.shape == (30, 6) and np.isfinite(q).all() and np.allclose(q.sum(axis=1), 1)
    again = BenchModel(family, DEPLOYMENT_CORE, params, threads=2).fit(pd.concat(tiny[:2], ignore_index=True))
    assert np.allclose(p, again.predict_proba(probe), atol=1e-6)        # seeded: reproducible


def test_search_budget_is_equal_deterministic_and_trial_zero_is_the_existing_recipe():
    for family in FAMILIES:
        trials = trial_params(family)
        assert len(trials) == 10 and trials == trial_params(family)
        assert len({json.dumps(t, sort_keys=True) for t in trials}) == 10
    assert trial_params('hgb')[0] == {'max_iter': 200}          # eval.classifier / Phase 3 recipe


# ------------------------------------------------------------------ confidence metrics
def test_confidence_bands_partition_the_predictions_and_error_rate_is_consistent():
    p = np.array([[.9, .1], [.7, .3], [.55, .45], [.5, .5], [.2, .8], [.65, .35]])
    y = np.array(['a', 'b', 'a', 'b', 'b', 'a'])
    c = confidence_profile(p, y, ['a', 'b'])
    assert c['share_below_60'] + c['share_60_to_80'] + c['share_above_80'] == pytest.approx(1)
    assert c['share_below_60'] == pytest.approx(2 / 6)
    assert c['coverage_at_60'] == pytest.approx(4 / 6)
    assert c['accuracy_when_confident_ge_60'] + c['error_rate_when_confident_ge_60'] == pytest.approx(1)
    assert c['confidence_correct']['n'] + c['confidence_incorrect']['n'] == 6
    assert sum(c['histogram_10_bins']['correct']) + sum(c['histogram_10_bins']['incorrect']) == 6
    assert sum(b['n'] for b in c['reliability_bins']) == 6


def test_perfect_probabilities_have_zero_brier_and_overconfidence_is_penalised():
    y = np.array(['a', 'b', 'a', 'b'])
    perfect = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    assert brier(perfect, y, ['a', 'b']) == 0
    wrong = np.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    assert brier(wrong, y, ['a', 'b']) == pytest.approx(2)
    assert calibration_block(perfect * .99 + .005, y, ['a', 'b'])['ece_top_label_10_bins'] < .02


def test_well_calibrated_sample_has_small_ece_and_overconfident_sample_does_not():
    rng = np.random.default_rng(0)
    conf = rng.uniform(.5, 1, 20000)
    correct = rng.random(20000) < conf
    ece_good = sum(b['n'] * abs(b['mean_confidence'] - b['accuracy']) for b in reliability_bins(conf, correct) if b['n']) / 20000
    ece_bad = sum(b['n'] * abs(b['mean_confidence'] - b['accuracy'])
                  for b in reliability_bins(np.minimum(conf + .15, .999), correct) if b['n']) / 20000
    assert ece_good < .02 < ece_bad


class _Stub:
    classes_ = CLASSES
    known_tuples = {('X', 'bank', 'payment')}


def test_safety_block_counts_unsafe_retries_with_denominators_under_the_frozen_gate():
    assert (GATE.min_confidence, GATE.max_hard_mass, GATE.retry_enabled) == (.60, .03, True)
    frame = pd.DataFrame({'true_cause': ['CARD_EXPIRED', 'RISK_BLOCKED', 'BD_INSUFFICIENT_FUNDS', 'TD_ISSUER_DOWN'],
                          'gw_error_code': 'X', 'gw_error_source': 'bank', 'gw_error_step': 'payment',
                          'instrument_hint': 'unknown'})
    # row0: argmax is retryable funds but hard-decline mass .07 > .03 -> the gate vetoes; row1: low confidence -> abstains.
    # Both would be unsafe retries without the gate. Rows 2,3: correct and allowed.
    p = np.array([[.9, 0, .04, .03, .02, .01], [.4, .1, .1, .3, .05, .05],
                  [.97, 0, .005, .005, .01, .01], [.01, 0, .005, .005, .97, .01]])
    s = safety_block(frame, _Stub(), p)
    assert s['hard_declines'] == 2 and s['retryable_payments'] == 2
    assert s['unsafe_retry_gated']['count'] == 0 and s['unsafe_retry_gated']['upper95'] > 0     # zero observed != zero risk
    assert s['unsafe_retry_ungated_argmax']['count'] == 2
    assert s['abstention_rate'] == pytest.approx(.5) and s['retryable_recall_gated'] == 1.0
    assert s['recovery_eligibility']['retry_allowed_count'] == 2


def test_upper95_exact_bound():
    assert upper95(0, 100) == pytest.approx(1 - .05 ** (1 / 100)) and upper95(0, 0) is None


def test_subset_masks_flag_generic_missing_and_overlap():
    frame = pd.DataFrame({'error_reason': ['payment_failed', 'missing', 'bank_unavailable'],
                          'attempt_duration_s': [10., None, 5.], 'n_causes_fired': [1, 2, 3]})
    m = subsets(frame)
    assert m['generic_error_reason'].tolist() == [True, False, False]
    assert m['missing_error_reason'].tolist() == [False, True, False]
    assert m['missing_timing_telemetry'].tolist() == [False, True, False]
    assert m['ambiguous_overlapping_causes'].tolist() == [False, True, True]


def test_classification_block_confusion_matrix_sums_to_n():
    y = np.array(['a', 'b', 'a', 'b']); p = np.array([[.9, .1], [.2, .8], [.3, .7], [.6, .4]])
    block = classification_block(p, y, ['a', 'b'])
    assert np.array(block['confusion_matrix']['rows_true_cols_predicted']).sum() == 4
    assert sum(v['support'] for v in block['per_class'].values()) == 4


# ------------------------------------------------------------------ frozen selection rule
def _s(f1, ll, ece=.02, unsafe=.005):
    return {'macro_f1': f1, 'logloss': ll, 'ece': ece, 'unsafe_rate': unsafe}


def test_selection_prefers_the_simpler_family_when_scores_tie():
    winner, why = phase4.select_model({'hgb': _s(.880, .30), 'xgboost': _s(.883, .295), 'catboost': _s(.884, .290),
                                       'random_forest': _s(.85, .35)})
    assert winner == 'hgb' and set(why['tied_on_macro_f1']) == {'hgb', 'xgboost', 'catboost'}


def test_selection_requires_a_real_gain_to_leave_the_simplest_model():
    winner, _ = phase4.select_model({'hgb': _s(.85, .33), 'xgboost': _s(.88, .28), 'catboost': _s(.87, .30),
                                     'random_forest': _s(.84, .34)})
    assert winner == 'xgboost'


def test_selection_excludes_unsafe_or_miscalibrated_models_unless_none_qualify():
    winner, why = phase4.select_model({'hgb': _s(.85, .30), 'xgboost': _s(.90, .25, unsafe=.03),
                                       'catboost': _s(.89, .26, ece=.09), 'random_forest': _s(.80, .4)})
    assert winner == 'hgb' and why['eligible'] == ['hgb', 'random_forest']
    winner, why = phase4.select_model({f: _s(.8 + .01 * i, .3, unsafe=.05) for i, f in enumerate(FAMILIES)})
    assert len(why['eligible']) == 4


def test_simplicity_order_puts_the_existing_dependency_first():
    assert min(SIMPLICITY_RANK, key=SIMPLICITY_RANK.get) == 'hgb'


def test_final_test_seeds_never_overlap_phase3_worlds():
    from eval import phase3
    used = {phase3.world_seed(s, r, i) for s in phase3.SCENARIOS for r, n in phase3.ROLE_COUNT.items() for i in range(n)}
    fresh = {phase4.FINAL_SEED[r] + 1000 * k + i for r, n in phase4.FINAL_COUNT.items() for k in range(3) for i in range(n)}
    assert not used & fresh and len(fresh) == 18


# ------------------------------------------------------------------ inactive adapter
@pytest.fixture(scope='module')
def artifact(tiny, tmp_path_factory):
    model = BenchModel('hgb', DEPLOYMENT_CORE, {'max_iter': 40}).fit(pd.concat(tiny[:2], ignore_index=True))
    calibration = tiny[2]
    cal = ProbabilityCalibrator('sigmoid', model.classes_).fit(model.predict_proba(calibration), calibration.true_cause.to_numpy())
    directory = tmp_path_factory.mktemp('artifact')
    joblib.dump({'model': model, 'calibrator': cal}, directory / 'model.joblib')
    schema = {'model_id': 'test-model', 'features': list(model.features), 'classes': list(model.classes_),
              'model_sha256': hashlib.sha256((directory / 'model.joblib').read_bytes()).hexdigest()}
    (directory / 'schema.json').write_text(json.dumps(schema))
    return directory, tiny[2]


def test_adapter_returns_calibrated_probabilities_and_handles_missing_fields_and_unseen_categories(artifact):
    directory, frame = artifact
    art = diagnosis_v4.load_artifact(directory)
    row = frame.iloc[0].to_dict()
    full = diagnosis_v4.diagnose(**{k: row[k] for k in ('gw_error_code', 'gw_error_source', 'gw_error_step', 'method', 'instrument_hint')},
                                 features=row, art=art)
    assert full.stage == 'ml_v4' and abs(sum(full.probabilities.values()) - 1) < 1e-6 and 0 <= full.confidence <= 1
    assert full.cause in CLASSES + ['AMBIGUOUS']
    bare = diagnosis_v4.diagnose(gw_error_code='BAD_REQUEST_ERROR', gw_error_source='bank', gw_error_step='payment',
                                 method='card', art=art)
    assert bare.stage == 'ml_v4' and bare.cause in CLASSES + ['AMBIGUOUS']          # every history field absent
    odd = diagnosis_v4.diagnose(gw_error_code='NEVER_SEEN', gw_error_source='mars', gw_error_step='teleport', method='hologram',
                                features={'error_reason': 'zzz', 'issuer_id': float('nan'), 'amount_minor': 'abc',
                                          'hour_ist': float('inf')}, art=art)
    assert odd.stage == 'ml_v4'
    assert abs(sum(odd.probabilities.values()) - 1) < 1e-6
    assert odd.cause in ('AMBIGUOUS', *CLASSES)


def test_adapter_fails_closed_and_keeps_the_hard_decline_veto(artifact, monkeypatch):
    directory, frame = artifact
    art = diagnosis_v4.load_artifact(directory)
    none = diagnosis_v4.diagnose(gw_error_code='', gw_error_source=None, gw_error_step=None, method='card', art=art)
    assert (none.cause, none.confidence, none.stage) == ('AMBIGUOUS', 0.0, 'no_error_code')
    diagnosis_v4.reset()
    monkeypatch.setattr(diagnosis_v4, 'load_artifact', lambda *a, **k: (_ for _ in ()).throw(diagnosis_v4.ArtifactError('gone')))
    broken = diagnosis_v4.diagnose(gw_error_code='X', gw_error_source='bank', gw_error_step='payment', method='card')
    assert (broken.cause, broken.confidence, broken.stage) == ('AMBIGUOUS', 0.0, 'error:ArtifactError')
    monkeypatch.undo()
    diagnosis_v4.reset()
    expired = diagnosis_v4.diagnose(gw_error_code='X', gw_error_source='bank', gw_error_step='payment', method='card',
                                    instrument_hint='expired', art=art)
    assert expired.cause == 'CARD_EXPIRED' and expired.decision == 'expired_instrument_veto'
    from services.app.domain.taxonomy import RETRYABLE, Cause
    assert not RETRYABLE[Cause(expired.cause)]


def test_tampered_artifact_is_rejected(artifact, tmp_path):
    directory, _ = artifact
    (tmp_path / 'schema.json').write_text((directory / 'schema.json').read_text())
    (tmp_path / 'model.joblib').write_bytes((directory / 'model.joblib').read_bytes() + b'x')
    with pytest.raises(diagnosis_v4.ArtifactError):
        diagnosis_v4.load_artifact(tmp_path)
    with pytest.raises(diagnosis_v4.ArtifactError):
        diagnosis_v4.load_artifact(tmp_path / 'nowhere')


def test_only_the_router_and_the_shadow_recorder_call_the_adapter():
    """The product calls Model C in ONE place per payment failure (revenue/routing.py, once, on the missing-or-generic route); shadow mode records it."""
    live, new = inspect.signature(diagnosis.diagnose), inspect.signature(diagnosis_v4.diagnose)
    for name, parameter in live.parameters.items():
        assert name in new.parameters and new.parameters[name].default == parameter.default
    assert issubclass(diagnosis_v4.DiagnosisV4, diagnosis.Diagnosis)
    for source in Path('services/app').rglob('*.py'):
        if source.name not in ('diagnosis_v4.py', 'shadow.py', 'routing.py'):
            assert 'diagnosis_v4' not in source.read_text(encoding='utf-8'), source
    assert diagnosis.MODEL_PATH.name == 'rzp_cause_clf.joblib'


def test_razorpay_payment_entity_mapping_uses_only_documented_fields():
    payment = {'id': 'pay_1', 'amount': 129900, 'method': 'card', 'status': 'failed', 'created_at': 1_700_000_000,
               'error_code': 'BAD_REQUEST_ERROR', 'error_source': 'bank', 'error_step': 'payment_authorization',
               'error_reason': 'payment_failed', 'card': {'issuer': 'HDFC'}}
    f = diagnosis_v4.features_from_razorpay_payment(payment)
    assert f['gw_error_code'] == 'BAD_REQUEST_ERROR' and f['error_reason'] == 'payment_failed' and f['issuer_id'] == 'HDFC'
    assert 0 <= f['hour_ist'] < 24 and f['amount_minor'] == 129900
    assert 'attempt_duration_s' not in f            # no documented source: never invented
    assert diagnosis_v4.features_from_razorpay_payment({}) == {}
    assert set(f) <= set(FEATURES_V4)


# ------------------------------------------------------------------ resumable runner
def test_ledger_skips_finished_units_isolates_failures_and_retries_them(tmp_path):
    from eval.phase4_ledger import Ledger
    calls = {'a': 0, 'b': 0}

    def good():
        calls['a'] += 1
        return {'value': 1.5, 'nan': float('nan')}

    def flaky():
        calls['b'] += 1
        if calls['b'] == 1:
            raise MemoryError('simulated')
        return {'ok': True}

    L = Ledger(tmp_path)
    L.expected = {'unit': 2}
    assert L.run('unit', 'a', good) == {'value': 1.5, 'nan': None}
    assert L.run('unit', 'b', flaky) is None                       # failure is contained, run continues
    assert L.failed() == ['unit__b'] and 'MemoryError' in (tmp_path / 'logs' / 'errors.log').read_text()
    assert L.run('unit', 'a', good)['value'] == 1.5 and calls['a'] == 1      # finished unit not recomputed
    L2 = Ledger(tmp_path)                                          # a new process (resume)
    L2.expected = {'unit': 2}
    assert L2.run('unit', 'b', flaky) == {'ok': True} and L2.failed() == []   # failed unit retried and cleared
    assert not list(tmp_path.rglob('*.tmp'))
    L2.write_progress()
    assert 'overall: 2/2' in (tmp_path / 'PROGRESS.txt').read_text()


def test_unit_with_missing_artifact_is_recomputed_even_if_its_json_exists(tmp_path):
    from eval.phase4_ledger import Ledger
    artifact, runs = tmp_path / 'model.bin', []

    def fn():
        runs.append(1)
        artifact.write_bytes(b'x')
        return {'n': len(runs)}
    L = Ledger(tmp_path)
    L.run('fit', 'm', fn, required=[artifact])
    L.run('fit', 'm', fn, required=[artifact])
    assert len(runs) == 1
    artifact.unlink()
    L.run('fit', 'm', fn, required=[artifact])
    assert len(runs) == 2


def test_atomic_json_never_leaves_partial_files(tmp_path):
    from eval.phase4_ledger import atomic_write_json
    atomic_write_json(tmp_path / 'x.json', {'a': [1, float('inf') if False else 2]})
    assert json.loads((tmp_path / 'x.json').read_text()) == {'a': [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ['x.json']


@pytest.mark.skipif(not Path('/proc').exists(), reason='liveness check is /proc based (container)')
def test_second_run_is_refused_while_the_first_is_alive_but_a_stale_lock_is_ignored(tmp_path):
    import subprocess, sys
    from eval.phase4_ledger import RunLocked, acquire_lock
    sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)  # eval.phase4'])
    try:
        (tmp_path / 'run.lock').write_text(str(sleeper.pid))
        with pytest.raises(RunLocked):
            acquire_lock(tmp_path)
    finally:
        sleeper.kill()
        sleeper.wait()
    (tmp_path / 'run.lock').write_text(str(sleeper.pid))           # now dead
    acquire_lock(tmp_path)


def test_frozen_selection_is_reused_not_recomputed(tmp_path):
    from eval.phase4_ledger import Ledger
    (tmp_path / 'selection.json').write_text(json.dumps({'winner': 'hgb', 'frozen': True}))
    assert phase4.freeze_selection(tmp_path, Ledger(tmp_path), ['xgboost'], {}, {}) == {'winner': 'hgb', 'frozen': True}


def test_smoke_config_is_small_but_covers_main_paths():
    assert 'deployment_core' in phase4.SMOKE.configs and len(phase4.feature_configs(phase4.SMOKE.configs)) == 3
    assert phase4.SMOKE.trials < phase4.FULL.trials and phase4.SMOKE.orders < phase4.FULL.orders
    assert dict(phase4.FULL.final_count) == phase4.FINAL_COUNT


def test_adapter_applies_safety_policy_v2_even_when_the_model_is_certain(artifact, monkeypatch):
    directory, _ = artifact
    art = diagnosis_v4.load_artifact(directory)
    certain = {c: 0. for c in CLASSES}
    certain['BD_INSUFFICIENT_FUNDS'] = 1.0
    monkeypatch.setattr(diagnosis_v4, 'score', lambda *a, **k: dict(certain))
    from eval.classifier import RULES
    code, source, step = next(t for t in sorted(art['model'].known_tuples) if t not in RULES)   # known tuple, no legacy rule
    base = dict(gw_error_code=code, gw_error_source=source, gw_error_step=step, method='card', art=art)
    ok = diagnosis_v4.diagnose(features={'error_reason': 'insufficient_funds'}, **base)
    assert (ok.route, ok.decision) == ('AUTO_RETRY', 'reliable_diagnosis')
    for reason, cause, route in (('card_expired', 'CARD_EXPIRED', 'NO_RETRY'), ('payment_risk_check_failed', 'RISK_BLOCKED', 'NO_RETRY'),
                                 ('debit_instrument_blocked', 'AMBIGUOUS', 'REVIEW')):
        d = diagnosis_v4.diagnose(features={'error_reason': reason}, **base)
        assert (d.cause, d.route) == (cause, route), reason
    prev = diagnosis_v4.diagnose(features={'error_reason': 'insufficient_funds', 'instrument_prev_reason': 'payment_risk_check_failed'}, **base)
    assert prev.route == 'REVIEW'
