"""Phase 4 / Stage 1: prebuilt UI datasets. Input/truth separation, no training-data leakage, disjointness, determinism,
integrity, Model C compatibility, expected safe outcomes, and the shared CPU cap in docker-compose.yml.
Needs the locally built datasets (python -m eval.build_ui_datasets) and the Phase 3/4 caches; skips otherwise."""
import json
import re
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval import build_ui_datasets as b

NEEDS = pytest.mark.skipif(not (b.OUT / 'normal_recoverable' / 'manifest.json').exists() or not b.CACHE.exists()
                           or not b.TRAIN_CACHE.exists(), reason='datasets or caches not present; run eval.build_ui_datasets')


@pytest.fixture(scope='module')
def data():
    return {n: (b.load_inputs(n), b.load_truth(n), b.load_manifest(n)) for n in b.NAMES}


@NEEDS
def test_all_four_datasets_load_with_1000_records(data):
    assert set(data) == set(b.NAMES) == {'normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime'}
    for name, (inputs, truth, manifest) in data.items():
        assert len(inputs) == len(truth) == manifest['records'] == 1000, name
        assert manifest['builder_version'] == b.BUILDER_VERSION and manifest['rng_seed'] == b.SEED == 20261031


@NEEDS
def test_inputs_hold_only_permitted_fields_and_never_ground_truth(data):
    for name, (inputs, truth, manifest) in data.items():
        assert list(inputs.columns) == list(b.INPUT_COLUMNS), name
        assert not set(inputs.columns) & set(b.FORBIDDEN_INPUT_COLUMNS), name
        assert set(inputs.payment_id) == set(truth) and inputs.payment_id.is_unique
        text = json.dumps(manifest)
        assert not any(pid in text for pid in inputs.payment_id.head(50)), 'manifest must not carry per-payment data'
    assert set(b.DEPLOYMENT_CORE) <= set(b.INPUT_COLUMNS) and len(b.DEPLOYMENT_CORE) == 43


def test_only_the_dataset_module_reads_the_truth_files():
    readers = set()
    for root in ('services', 'eval'):
        for path in Path(root).rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            if 'truth.json' in text or 'load_truth(' in text:
                readers.add(path.as_posix())
    assert readers == {'eval/build_ui_datasets.py', 'services/app/test_datasets.py'}    # the builder and the API's single hidden_truth() accessor


@NEEDS
def test_no_overlap_with_training_validation_or_calibration_data_and_datasets_are_disjoint(data):
    ids, seeds = b.earlier_data_footprint()
    all_ids = [i for inputs, _, _ in data.values() for i in inputs.payment_id]
    assert not ids & set(all_ids) and len(all_ids) == len(set(all_ids)) == 4000
    world_seeds = {s for _, _, m in data.values() for s in m['source']['world_seeds']}
    assert not seeds & world_seeds
    assert all(500000 <= s <= 502003 or 600000 <= s <= 602001 for s in world_seeds) and max(seeds) < 500000
    assert data['shifted_regime'][2]['source']['cache_role'] == 'final_shift' and data['shifted_regime'][2]['source']['market_seed'] == 12
    for name in ('normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult'):
        assert data[name][2]['source']['cache_role'] == 'final_test' and data[name][2]['source']['market_seed'] == 11


@NEEDS
def test_rebuild_is_deterministic_byte_for_byte(tmp_path, data):
    rebuilt = b.build(tmp_path)
    for name in b.NAMES:
        assert rebuilt[name]['files'] == data[name][2]['files'], name
        assert (tmp_path / name / 'manifest.json').read_bytes() == (b.OUT / name / 'manifest.json').read_bytes()


@NEEDS
def test_manifest_hashes_match_and_tampering_is_detected(tmp_path, data):
    for name, (_, _, manifest) in data.items():
        assert manifest['files'] == {'inputs.pkl': b.sha256(b.OUT / name / 'inputs.pkl'), 'truth.json': b.sha256(b.OUT / name / 'truth.json')}
    shutil.copytree(b.OUT / 'normal_recoverable', tmp_path / 'normal_recoverable')
    blob = bytearray((tmp_path / 'normal_recoverable' / 'inputs.pkl').read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    (tmp_path / 'normal_recoverable' / 'inputs.pkl').write_bytes(bytes(blob))
    with pytest.raises(ValueError, match='SHA-256'):
        b.load_inputs('normal_recoverable', tmp_path)
    (tmp_path / 'normal_recoverable' / 'truth.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        b.load_truth('normal_recoverable', tmp_path)


@NEEDS
def test_model_c_accepts_every_dataset_with_the_packaged_feature_names_and_class_order(data):
    from services.app.adapters import diagnosis_v4
    art = diagnosis_v4.bundle()
    model = art['model']
    assert list(model.features) == list(b.DEPLOYMENT_CORE) and list(model.classes_) == art['schema']['classes']
    for name, (inputs, _, _) in data.items():
        p = art['calibrator'].transform(model.predict_proba(inputs.head(100)))
        assert p.shape == (100, 6) and np.allclose(p.sum(axis=1), 1), name
    row = data['shifted_regime'][0].iloc[0].to_dict()                      # the adapter works from the stored features alone
    out = diagnosis_v4.diagnose(gw_error_code=row['gw_error_code'], gw_error_source=row['gw_error_source'], gw_error_step=row['gw_error_step'],
                                method=row['method'], instrument_hint=row['instrument_hint'], features=row)
    assert out.stage == diagnosis_v4.STAGE and set(out.probabilities) == set(model.classes_)


def test_expected_outcomes_follow_the_recovery_rules_and_never_assume_recovery():
    risk = b.expected_outcome('RISK_BLOCKED', ['human_escalation'], True)
    assert risk['ideal_outcome'] == 'SAFETY_ESCALATION' and 'SIMULATED_RECOVERY_SUCCESS' not in risk['safe_outcomes']
    assert set(risk['forbidden_actions']) == {'silent_retry', 'rail_switch'} and risk['recovery_guaranteed'] is False
    card = b.expected_outcome('CARD_EXPIRED', ['card_update_link', 'card_update_link'], True)
    assert set(card['forbidden_actions']) == {'silent_retry', 'rail_switch'} and card['recovery_guaranteed'] is False
    funds = b.expected_outcome('BD_INSUFFICIENT_FUNDS', ['silent_retry', 'notify', 'silent_retry'], True)
    assert funds['ideal_is_conditional'] and funds['recovery_guaranteed'] is False and funds['forbidden_actions'] == []
    assert 'SAFETY_ESCALATION' in funds['unnecessary_but_safe_outcomes']
    assert b.expected_outcome('TD_PSP_TIMEOUT', [], False)['ideal_outcome'] == 'PLAN_INTERVENTION'      # an unplannable case is not "expected to recover"


@NEEDS
def test_truth_entries_are_consistent_with_the_planner(data):
    from eval.oracle_plan import Ctx, build_plan
    from services.app.domain.taxonomy import Cause
    for name, (inputs, truth, _) in data.items():
        for pid in list(inputs.payment_id.head(15)):
            t = truth[pid]
            assert t['recovery_guaranteed'] is False and t['ideal_outcome'] in {'SAFETY_ESCALATION', 'SIMULATED_RECOVERY_SUCCESS', 'PLAN_INTERVENTION'}
            assert (t['true_cause'] == 'RISK_BLOCKED') == (t['ideal_outcome'] == 'SAFETY_ESCALATION')
            if t['true_cause'] in b.HARD:
                assert not set(t['oracle_plan_actions']) & set(b.RETRY_ACTIONS)         # the policy never plans a retry for a hard decline
                assert set(t['forbidden_actions']) == set(b.RETRY_ACTIONS)
            row = inputs[inputs.payment_id == pid].iloc[0]
            ctx = Ctx(int(row.amount_minor_raw), row.method, pd.Timestamp(row.failed_at).date(), frozenset(b.h.RAILS.split(',')), True)
            planned = [s.action for s in build_plan(Cause(t['true_cause']), ctx).steps]
            assert len(t['oracle_plan_actions']) <= len(planned)


@NEEDS
def test_dataset_compositions_and_honest_labels(data):
    assert data['normal_recoverable'][2]['composition_by_cause'] == {c: 250 for c in sorted(b.RETRYABLE)}
    assert data['risk_and_hard_declines'][2]['composition_by_cause'] == {'CARD_EXPIRED': 500, 'RISK_BLOCKED': 500}
    mixed = data['mixed_difficult'][2]
    assert mixed['representative_of_natural_performance'] is False and 'not representative' in mixed['label'].lower()
    assert 'not representative' in mixed['description'].lower()
    assert all(t['selection_criteria'] for t in data['mixed_difficult'][1].values())
    shifted = data['shifted_regime'][2]
    assert shifted['representative_of_natural_performance'] is True
    source = b.load_role('final_shift').true_cause.value_counts()
    assert shifted['composition_by_cause'] == b.proportional_counts(source, 1000)           # natural frequencies, not balanced
    for cause, n in shifted['composition_by_cause'].items():
        assert abs(n - 1000 * source[cause] / source.sum()) <= 1


@NEEDS
def test_mixed_difficult_criteria_claims_are_true(data):
    inputs, truth, _ = data['mixed_difficult']
    from eval import phase3_heldout as h
    art, classes, probs = h.probabilities(inputs)
    classes = list(classes)
    risk, card = classes.index('RISK_BLOCKED'), classes.index('CARD_EXPIRED')
    actual = {'low_confidence': probs.max(axis=1) < .60,
              'incorrect_top1': np.asarray(classes)[probs.argmax(axis=1)] != np.array([truth[p]['true_cause'] for p in inputs.payment_id]),
              'ambiguous_risk_probability': (probs[:, risk] >= .10) & (probs[:, risk] < .40),
              'elevated_hard_mass': ((probs[:, risk] + probs[:, card]) > .05) & ((probs[:, risk] + probs[:, card]) <= .30)}
    for i, pid in enumerate(inputs.payment_id):
        for claimed in truth[pid]['selection_criteria']:
            assert actual[claimed][i], (pid, claimed)
        assert any(actual[f][i] for f in actual)                     # every selected row really is "difficult" by at least one rule


def test_compose_applies_one_shared_six_core_cap_to_all_six_services_and_keeps_port_5433():
    path = Path('docker-compose.yml')
    if not path.exists():
        pytest.skip('docker-compose.yml is not mounted in this container')
    text = path.read_text(encoding='utf-8')
    anchor = re.search(r'x-cpu-cap: &cpu-cap\n((?:  .+\n)+)', text).group(1)
    assert 'cpus: 6' in anchor and 'cpuset: "0-5"' in anchor
    services = re.findall(r'^  ([a-z-]+):\n    <<: (?:\[[^\]]*\*cpu-cap\]|\*cpu-cap)', text, flags=re.M)
    assert sorted(services) == sorted(['api', 'worker', 'postgres', 'temporal', 'temporal-ui'])
    base = re.search(r'x-app-base: &app-base\n((?:  .+\n|\n)+)', text).group(1)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert re.search(rf'{var}: "3"', base), var
    assert '"5433:5432"' in text
