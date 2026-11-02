"""Phase 1 freeze of Model C. These tests pin the contract described in docs/model_c_freeze.md.

They fail when the artifact, the 43-feature schema and its order, the class order, the gate thresholds, the gate's decisions or the
feature builder's output change. A failure here is not a bug to be patched around: it means Model C is being changed, which needs an
explicit decision, an updated docs/model_c_freeze.md and regenerated golden values (tests/fixtures/model_c_golden.json).

Read-only: nothing here trains, writes, starts a workflow or touches a database.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'eval' / 'artifacts' / 'phase4'
GOLDEN = json.loads((Path(__file__).parent / 'fixtures' / 'model_c_golden.json').read_text(encoding='utf-8'))

pytestmark = pytest.mark.skipif(not (ARTIFACT / 'model.joblib').exists(),
                                reason='the Model C artifact is not present (eval/artifacts/phase4 is git-ignored; restore it before relying on the freeze)')

MODEL_SHA256 = '71bf19e4430f1b9532a5b4dd21a9d4076ea044de4655b8c4a2a5ca13b45594e6'
SCHEMA_SHA256 = '9b2ad0136fe650fc9e83d1ddf97be19c3057794bf0da592af57afeb6adbb0acc'
MODEL_ID = 'phase4-hgb-deployment_core-uncalibrated-71bf19e4'
CLASSES = ['BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'RISK_BLOCKED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT']
FEATURES = [
    'gw_error_code', 'gw_error_source', 'gw_error_step', 'method', 'instrument_hint', 'issuer_id', 'amount_minor', 'log_amount_minor', 'hour_ist',
    'weekday_ist', 'customer_attempts_7d', 'customer_successes_7d', 'customer_failure_rate_7d', 'customer_last_success_age_s',
    'amount_to_customer_mean_7d', 'order_prior_attempts_7d', 'issuer_attempts_5m', 'issuer_failure_rate_5m', 'method_attempts_5m',
    'method_failure_rate_5m', 'error_reason', 'reason_family', 'is_generic_reason', 'step_ordinal', 'instrument_consecutive_failures',
    'instrument_attempts_24h', 'instrument_failure_rate_24h', 'instrument_last_success_age_s', 'instrument_prev_reason', 'same_reason_as_prev',
    'instrument_distinct_reasons_24h', 'issuer_failure_rate_1h', 'issuer_failure_delta', 'issuer_failure_vs_baseline_7d', 'issuer_timeout_rate_5m',
    'issuer_tech_reason_share_5m', 'issuer_method_failure_rate_5m', 'global_failure_rate_5m', 'secs_since_prev_attempt', 'customer_methods_tried_24h',
    'amount_vs_max_success_amount', 'customer_success_rate_14d', 'days_to_month_end']
CATEGORICAL = ['gw_error_code', 'gw_error_source', 'gw_error_step', 'method', 'instrument_hint', 'issuer_id', 'error_reason', 'reason_family', 'instrument_prev_reason']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


@pytest.fixture(scope='module')
def art():
    from services.app.adapters import diagnosis_v4
    return diagnosis_v4.load_artifact()                                                  # also runs the loader's own integrity check


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def diagnose(features):
    from services.app.adapters import diagnosis_v4
    return diagnosis_v4.diagnose(gw_error_code=features['gw_error_code'], gw_error_source=features['gw_error_source'], gw_error_step=features['gw_error_step'],
                                 method=features['method'], instrument_hint=features.get('instrument_hint') or 'na', features=features)


# ------------------------------------------------------------------------------------------------- the artifact is not modified
def test_artifact_files_are_byte_identical_to_the_frozen_release():
    assert sorted(p.name for p in ARTIFACT.iterdir()) == ['model.joblib', 'schema.json']
    assert hashlib.sha256((ARTIFACT / 'model.joblib').read_bytes()).hexdigest() == MODEL_SHA256
    assert sha(ARTIFACT / 'schema.json') == SCHEMA_SHA256
    assert json.loads((ARTIFACT / 'schema.json').read_text())['model_sha256'] == MODEL_SHA256


def test_the_loader_accepts_the_artifact_and_reports_the_frozen_identity(art):
    schema, model, cal = art['schema'], art['model'], art['calibrator']
    assert (schema['model_id'], schema['family'], schema['calibration'], schema['activated'], schema['synthetic_only']) == (MODEL_ID, 'hgb', 'uncalibrated', False, True)
    assert (model.family, type(model).__name__) == ('hgb', 'BenchModel')
    assert model.params == {'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'l2_regularization': 0.1}
    assert (cal.method, list(cal.models)) == ('uncalibrated', []) and list(cal.classes) == CLASSES                    # the identity calibrator: raw HGB probabilities
    assert len(model.known_tuples) == 262


def test_runtime_libraries_match_the_ones_the_artifact_was_trained_with(art):
    import sklearn
    import numpy
    libs = art['schema']['libraries']
    assert sklearn.__version__ == libs['scikit-learn'] and numpy.__version__ == libs['numpy']                          # a pickle is only safe on the same major stack


# ------------------------------------------------------------------------------------------------- feature schema and ordering
def test_the_43_feature_schema_and_its_order_are_unchanged(art):
    from services.app.domain.deployment_features import DEPLOYMENT_CORE
    from services.app.domain.features import FEATURES as ORIGINAL
    from services.app.domain.features_v4 import FEATURES_V4
    assert len(FEATURES) == 43 == len(set(FEATURES))
    assert list(art['model'].features) == FEATURES == art['schema']['features'] == list(DEPLOYMENT_CORE) == GOLDEN['features_order']
    assert list(art['model'].categorical) == CATEGORICAL == art['schema']['categorical'] and len(art['model'].numeric) == 34
    assert list(art['model'].classes_) == CLASSES == art['schema']['classes']
    assert len(FEATURES_V4) == 57 and list(FEATURES_V4[:26]) == list(ORIGINAL)
    assert [f for f in FEATURES_V4 if f in FEATURES] == FEATURES                                                       # the model set keeps its V4 relative order


def test_the_feature_builder_output_is_unchanged_for_cold_start_and_history(art, monkeypatch):
    from services.app import test_datasets as tds
    monkeypatch.setattr(tds, '_store', lambda text: 'golden')                           # validate_csv would otherwise save the upload under eval/data
    for case in GOLDEN['predictions']:
        row = tds.derive_single({**case['input'], 'payment_id': 'golden_' + case['name'], 'status': 'failed'})
        for name in FEATURES:
            assert close(_clean(row.get(name)), case['features'][name]), (case['name'], name)
    frame, _, meta = tds.derive_frame(GOLDEN['history_case']['csv'])
    hist = frame.iloc[-1].to_dict()
    assert len(frame) == GOLDEN['history_case']['failed_rows']
    for name in FEATURES:
        assert close(_clean(hist.get(name)), GOLDEN['history_case']['features'][name]), name


def _clean(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return v.item() if hasattr(v, 'item') else v


# ------------------------------------------------------------------------------------------------- same input, same prediction
@pytest.mark.parametrize('case', GOLDEN['predictions'], ids=lambda c: c['name'])
def test_golden_predictions_are_reproduced(case):
    from services.app import test_datasets as tds
    row = tds.derive_single({**case['input'], 'payment_id': 'golden_' + case['name'], 'status': 'failed'})
    d = diagnose(row)
    exp = case['expected']
    assert list(d.probabilities) == CLASSES == list(exp['probabilities'])
    for c in CLASSES:
        assert close(d.probabilities[c], exp['probabilities'][c]), c
    assert abs(sum(d.probabilities.values()) - 1) < 1e-9 and all(0 <= p <= 1 for p in d.probabilities.values())
    assert (d.cause, d.stage, d.decision, d.route, d.top_cause, d.model_id) == (exp['cause'], exp['stage'], exp['decision'], exp['route'], exp['top_cause'], exp['model_id'])
    assert close(d.confidence, exp['confidence'])


def test_the_same_input_gives_bit_identical_probabilities_every_time():
    from services.app import test_datasets as tds
    case = GOLDEN['predictions'][1]
    row = tds.derive_single({**case['input'], 'payment_id': 'repeat', 'status': 'failed'})
    first = diagnose(row).probabilities
    for other in GOLDEN['predictions'][:4]:                                              # interleave other inputs: no hidden state
        diagnose(tds.derive_single({**other['input'], 'payment_id': 'x', 'status': 'failed'}))
        assert diagnose(row).probabilities == first


def test_the_investigated_out_of_distribution_input_still_abstains():
    case = next(c for c in GOLDEN['predictions'] if c['name'] == 'upi_investigated_out_of_distribution')
    assert round(case['expected']['probabilities']['BD_USER_ABANDONED_OTP'], 4) == 0.5259 and case['expected']['decision'] == 'low_confidence'


def test_missing_or_unusable_input_fails_closed():
    from services.app.adapters import diagnosis_v4
    d = diagnosis_v4.diagnose(gw_error_code=None, gw_error_source=None, gw_error_step=None, method='upi')
    assert (d.cause, d.confidence, d.stage, d.decision) == ('AMBIGUOUS', 0.0, 'no_error_code', 'no_error_code')


def test_the_router_does_not_alter_the_model_output():
    """The ONE place the product calls Model C (revenue/routing.py) returns exactly what the frozen adapter returns, for every golden input."""
    from services.app import test_datasets as tds
    from services.app.revenue import routing
    for case in GOLDEN['predictions']:
        row = tds.derive_single({**case['input'], 'payment_id': 'wrap', 'status': 'failed'})
        rec = routing.model_c_diagnose(row)
        direct = diagnose(row)
        assert rec['status'] == 'scored' and rec['probabilities'] == direct.probabilities == case['expected']['probabilities']
        if case['expected']['decision'] != 'reliable_diagnosis' or rec['decision'] != 'independent_risk_detection':
            assert rec['decision'] in (case['expected']['decision'], 'independent_risk_detection')


# ------------------------------------------------------------------------------------------------- confidence thresholds and safety gate
def test_confidence_thresholds_and_policy_constants_are_unchanged():
    from services.app.adapters import diagnosis_v4
    from services.app.domain import diagnosis_safety as ds, safety_policy_v2 as sp
    assert (sp.POLICY_VERSION, sp.MIN_CONFIDENCE, sp.AUTO_RETRY_MAX_HARD_MASS, sp.REVIEW_MAX_HARD_MASS, sp.RISK_DETECTION_MIN_PROBABILITY) == ('safety-policy@v2.2', .60, .10, .10, .20)
    assert sp.THRESHOLDS_V2 == ds.SafetyThresholds(.60, .10, True) == diagnosis_v4.GATE
    assert ds.SafetyThresholds() == ds.SafetyThresholds(.95, .03, True)                                                  # the original Phase 2.5 control values
    params = inspect.signature(sp.decide_v2).parameters
    assert (params['auto_max_hard_mass'].default, params['review_max_hard_mass'].default, params['min_confidence'].default) == (.10, .10, .60)
    assert ds.HARD == ('CARD_EXPIRED', 'RISK_BLOCKED') and sp.HARD_REASON_CAUSE == {'card_expired': 'CARD_EXPIRED', 'payment_risk_check_failed': 'RISK_BLOCKED'}
    assert sp.HARD_REASON_FAMILIES == frozenset({'risk', 'instrument_state'})
    assert (diagnosis_v4.STAGE, str(diagnosis_v4.ARTIFACT_DIR).replace('\\', '/')) == ('ml_v4', 'eval/artifacts/phase4')


def test_the_artifact_metadata_still_records_the_original_gate_not_the_runtime_gate():
    """Known, documented discrepancy (docs/model_c_freeze.md): schema.json says max_hard_mass 0.03; the runtime budget is 0.10."""
    gate = json.loads((ARTIFACT / 'schema.json').read_text())['safety_gate']
    assert (gate['min_confidence'], gate['max_hard_mass']) == (0.6, 0.03)


def test_retryability_of_each_cause_is_unchanged():
    from services.app.domain.taxonomy import RETRYABLE
    assert {str(k): v for k, v in RETRYABLE.items()} == {'BD_INSUFFICIENT_FUNDS': True, 'BD_USER_ABANDONED_OTP': False, 'TD_ISSUER_DOWN': True,
                                                         'TD_PSP_TIMEOUT': True, 'CARD_EXPIRED': False, 'RISK_BLOCKED': False}


@pytest.mark.parametrize('case', GOLDEN['gate_cases'], ids=lambda c: c['name'])
def test_gate_decisions_are_preserved(case):
    from services.app.domain.safety_policy_v2 import decide_v2
    out = decide_v2(case['probabilities'], **case['kwargs'])
    assert {k: out[k] for k in ('cause', 'retry_allowed', 'decision', 'route')} == case['expected']


def test_independent_risk_detection_boundaries_are_preserved():
    from services.app.domain.safety_policy_v2 import detect_independent_risk
    base = dict.fromkeys(CLASSES, 0.0)
    for case in GOLDEN['risk_cases']:
        p = {**base, 'RISK_BLOCKED': case['p_risk'], 'BD_INSUFFICIENT_FUNDS': 1 - case['p_risk']}
        assert detect_independent_risk(case['gate_cause'], p) is case['expected'], case


def test_independent_risk_escalation_is_applied_exactly_as_before(monkeypatch):
    from services.app.adapters import diagnosis_v4
    from services.app.revenue import routing

    def fake(probs, cause='AMBIGUOUS'):
        d = diagnosis_v4.DiagnosisV4(cause, max(probs.values()), diagnosis_v4.STAGE, decision='low_confidence', probabilities=probs, top_cause=max(probs, key=probs.get),
                                     model_id='fake', route='REVIEW')
        return lambda **kw: d
    p = {**dict.fromkeys(CLASSES, 0.0), 'BD_USER_ABANDONED_OTP': .6, 'RISK_BLOCKED': .25, 'BD_INSUFFICIENT_FUNDS': .15}
    monkeypatch.setattr(diagnosis_v4, 'diagnose', fake(p))
    out = routing.model_c_diagnose({'gw_error_code': 'x', 'gw_error_source': 'y', 'gw_error_step': 'z', 'method': 'upi'})
    assert (out['cause'], out['decision'], out['route'], out['confidence']) == ('RISK_BLOCKED', 'independent_risk_detection', 'NO_RETRY', .25)
    quiet = {**p, 'RISK_BLOCKED': .1999, 'BD_USER_ABANDONED_OTP': .6501}
    monkeypatch.setattr(diagnosis_v4, 'diagnose', fake(quiet))
    out = routing.model_c_diagnose({'gw_error_code': 'x', 'gw_error_source': 'y', 'gw_error_step': 'z', 'method': 'upi'})
    assert (out['cause'], out['decision']) == ('AMBIGUOUS', 'low_confidence')


def test_the_deterministic_rules_the_gate_consults_are_unchanged():
    from eval.classifier import RULES
    digest = hashlib.sha256(json.dumps(sorted((list(k), str(v)) for k, v in RULES.items()), sort_keys=True).encode()).hexdigest()
    assert (len(RULES), digest) == (GOLDEN['classifier_rules_count'], GOLDEN['classifier_rules_sha256'])


# ------------------------------------------------------------------------------------------------- the frozen modules themselves
@pytest.mark.parametrize('path', sorted(GOLDEN['source_sha256']))
def test_frozen_source_modules_are_unchanged(path):
    """Line endings are ignored. If this fails you are changing Model C's feature, gate or loading code: follow docs/model_c_freeze.md."""
    assert sha(ROOT / path) == GOLDEN['source_sha256'][path], f'{path} changed; Model C is frozen (see docs/model_c_freeze.md)'
