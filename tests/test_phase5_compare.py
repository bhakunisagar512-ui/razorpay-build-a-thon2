"""Phase 5A.2 comparison script: record alignment, metric consistency, artifact loading, read-only behaviour.
Small worlds only; the real comparison is run by scripts/run_phase5_comparison.ps1."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval import phase4, phase5_compare as p5
from eval.classifier import CauseClassifier, FEATURES as ORIGINAL_FEATURES
from eval.generate import generate
from eval.phase4_models import BenchModel
from eval.probability_calibration import ProbabilityCalibrator
from eval.sequences_v4 import assemble_v4, generate_world
from services.app.domain.deployment_features import DEPLOYMENT_CORE
from services.app.domain.features import BASELINE_FEATURES

ARTIFACT = Path('eval/artifacts/phase4')


@pytest.fixture(scope='module')
def tiny():
    frames = [phase4._tag(assemble_v4(generate_world(1500, seed, scenario='typical')).failures, 'typical', role, i)
              for i, (seed, role) in enumerate([(91, 'train'), (92, 'train'), (93, 'final_test')])]
    return pd.concat(frames[:2], ignore_index=True), frames[2]


@pytest.fixture(scope='module')
def comparison(tiny, tmp_path_factory):
    train, test = tiny
    legacy = generate(n=1500, seed=5).drop_duplicates(subset=['payment_id'])
    clf_a = CauseClassifier().fit(legacy)
    model_c = BenchModel('hgb', DEPLOYMENT_CORE, {'max_iter': 25}).fit(train)
    calibrator = ProbabilityCalibrator('uncalibrated', model_c.classes_).fit(model_c.predict_proba(test), test.true_cause.to_numpy())
    art_c = {'model': model_c, 'calibrator': calibrator,
             'schema': {'model_id': 'test-c', 'model_sha256': 'x', 'calibration': 'uncalibrated', 'hyperparameters': {'max_iter': 25}}}
    metrics = p5.compare(test, train, clf_a, art_c, train_a=legacy, seeds_b={'a': 1}, seeds_test={'b': 2}, bootstrap=20, latency_repeats=10)
    out = tmp_path_factory.mktemp('p5')
    p5.write_outputs(metrics, out, out / 'docs')
    return metrics, out, test


def test_check_fresh_cache_refuses_to_regenerate_or_write_anything(tmp_path):
    with pytest.raises(FileNotFoundError):
        p5.check_fresh_cache(tmp_path)
    assert list(tmp_path.iterdir()) == []                    # nothing was created in the (Phase 4) cache directory


def test_integrity_reports_overlap_and_disjointness(tiny):
    train, test = tiny
    clean = p5.integrity(train, test, seeds_b={'x': 100000}, seeds_test={'y': 500000})
    assert clean['b_train_vs_test_payment_id_overlap'] == 0 and clean['b_train_vs_test_world_overlap'] == []
    assert clean['seed_ranges_disjoint'] is True
    leaky = p5.integrity(pd.concat([train, test.head(5)]), test)
    assert leaky['b_train_vs_test_payment_id_overlap'] == 5 and leaky['b_train_vs_test_world_overlap']


def test_alignment_requires_every_column_of_every_model(tiny):
    _, test = tiny
    p5.assert_aligned(test, ORIGINAL_FEATURES, BASELINE_FEATURES, DEPLOYMENT_CORE)
    with pytest.raises(AssertionError):
        p5.assert_aligned(test.drop(columns=['error_reason']), DEPLOYMENT_CORE)
    with pytest.raises(AssertionError):
        p5.assert_aligned(pd.concat([test, test.head(3)]), ORIGINAL_FEATURES)          # duplicate ids


def test_baseline_b_uses_the_same_five_columns_and_recipe_as_the_original():
    assert tuple(BASELINE_FEATURES) == tuple(ORIGINAL_FEATURES)
    original = CauseClassifier().model.get_params()
    retrained = __import__('eval.phase3', fromlist=['x']).Phase3Model(BASELINE_FEATURES).model.get_params()
    for key in ('max_iter', 'random_state', 'learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization'):
        assert original[key] == retrained[key], key


def test_to_labels_pads_a_class_the_model_never_saw():
    p = np.array([[.7, .3]])
    out = p5.to_labels(p, ['b', 'd'], ['a', 'b', 'c', 'd'])
    assert out.tolist() == [[0., .7, 0., .3]]


def test_all_three_models_score_exactly_the_same_records(comparison):
    metrics, _, test = comparison
    pooled = metrics['results']['pooled']
    assert pooled['rows'] == len(test) == metrics['meta']['test_rows']
    for key in 'ABC':
        r = pooled['raw'][key]
        assert sum(v['support'] for v in r['per_class'].values()) == len(test)
        assert np.array(r['confusion_matrix']['rows_true_cols_predicted']).sum() == len(test)
        assert r['confidence']['n'] == len(test)
    supports = [{k: v['support'] for k, v in pooled['raw'][m]['per_class'].items()} for m in 'ABC']
    assert supports[0] == supports[1] == supports[2]


def test_metrics_are_internally_consistent(comparison):
    metrics, _, test = comparison
    pooled = metrics['results']['pooled']
    for key in 'ABC':
        c = pooled['raw'][key]['confidence']
        assert c['share_below_60'] + c['share_60_to_80'] + c['share_above_80'] == pytest.approx(1)
        assert 0 <= pooled['raw'][key]['accuracy'] <= 1
    b, c = pooled['raw']['B'], pooled['raw']['C']
    assert pooled['B_vs_C_macro_f1']['delta_C_minus_B'] == pytest.approx(c['macro_f1'] - b['macro_f1'])
    boot = pooled['B_vs_C_macro_f1']['paired_bootstrap_C_minus_B']
    assert boot['lo95'] <= boot['hi95']


def test_native_ambiguity_for_a_and_the_frozen_gate_for_b_and_c_are_labelled_separately(comparison):
    metrics, _, _ = comparison
    pol = metrics['results']['pooled']['policy_adjusted']
    a = pol['A_native']
    assert 0 <= a['ambiguous_rate'] <= 1 and abs(sum(a['stage_mix'].values()) - 1) < 1e-9
    # with the LLM tier off, AMBIGUOUS is produced only by the gated stages
    assert a['ambiguous_rate'] == pytest.approx(sum(v for k, v in a['stage_mix'].items() if k.startswith('gated')))
    for key in ('B_frozen_gate', 'C_frozen_gate'):
        assert 'confidence 0.60' in pol[key]['policy'] and 'mass 0.03' in pol[key]['policy']
        assert 0 <= pol[key]['abstention_rate'] <= 1 and pol[key]['unsafe_retries'] <= pol[key]['hard_declines']
    assert 'native' in a['policy']
    assert set(metrics['results']['pooled']['raw']) == {'A', 'B', 'C'}               # raw never mixed with policy blocks


def test_outputs_exist_and_the_report_flags_the_differences(comparison):
    metrics, out, _ = comparison
    for name in ('metrics.json', 'model_comparison.md', 'confusion_A_raw.csv', 'confusion_B_raw.csv', 'confusion_C_raw.csv',
                 'confusion_matrices.png', 'confidence_distribution.png'):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert (out / 'docs' / 'model_comparison.md').exists()
    assert json.loads((out / 'metrics.json').read_text())['meta']['models']['C']['n_features'] == 43
    md = (out / 'model_comparison.md').read_text()
    for heading in ('## 2. Headline: B vs C', '## 5. Policy-adjusted', '## 6. A and the legacy vocabulary mismatch', '## 8. Limitations'):
        assert heading in md
    assert 'B is not the serving model' in md and 'not** applied' in md
    csv = pd.read_csv(out / 'confusion_B_raw.csv', index_col=0)
    assert csv.values.sum() == metrics['meta']['test_rows']


@pytest.mark.skipif(not (ARTIFACT / 'model.joblib').exists(), reason='Phase 4 artifact not present')
def test_real_phase4_artifact_loads_without_retraining_and_is_not_modified(tiny):
    from services.app.adapters import diagnosis_v4
    before = hashlib.sha256((ARTIFACT / 'model.joblib').read_bytes()).hexdigest()
    art = diagnosis_v4.load_artifact(ARTIFACT)
    assert len(art['model'].features) == 43 and tuple(art['model'].features) == tuple(DEPLOYMENT_CORE)
    assert before == art['schema']['model_sha256']
    _, test = tiny
    p = art['calibrator'].transform(art['model'].predict_proba(test.head(50)))
    assert p.shape == (50, 6) and np.allclose(p.sum(axis=1), 1)
    assert hashlib.sha256((ARTIFACT / 'model.joblib').read_bytes()).hexdigest() == before


@pytest.mark.skipif(not Path('scripts/run_phase5_comparison.ps1').exists(), reason='scripts/ is not mounted in the container; runs on the host')
def test_launcher_is_scoped_to_the_comparison():
    text = Path('scripts/run_phase5_comparison.ps1').read_text(encoding='utf-8')
    assert 'eval.phase5_compare' in text and 'docker update --cpus=6' in text
    for forbidden in ('eval.phase4 ', 'git commit', 'git add', 'make phase4', 'uvicorn'):
        assert forbidden not in text
