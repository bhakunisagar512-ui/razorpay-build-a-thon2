import pandas as pd

from eval.investigate_features import audit_generator, conditional_shuffle, inspect, fit
from eval.generate import generate_dataset
from eval.datasets import temporal_split
from services.app.domain.features import BASELINE_FEATURES


def test_conditional_shuffle_keeps_baseline_and_joint_feature_distribution():
    frame = pd.DataFrame([{**dict(zip(BASELINE_FEATURES, ('code', 'bank', 'step', 'card', 'valid'))),
                           'x': i, 'y': i * 10, 'true_cause': f'label_{i}'} for i in range(20)])
    shuffled, moved = conditional_shuffle(frame, ('x', 'y'), 42)
    pd.testing.assert_frame_equal(shuffled[list(BASELINE_FEATURES)], frame[list(BASELINE_FEATURES)])
    assert shuffled.true_cause.tolist() == frame.true_cause.tolist()
    assert sorted(zip(shuffled.x, shuffled.y)) == sorted(zip(frame.x, frame.y))
    assert (shuffled.y == shuffled.x * 10).all()
    assert moved > 0


def test_generator_audit_reports_missing_episode_coverage():
    dataset = generate_dataset(1000, include_candidates=True)
    # Deliberately truncate a fixture; do not require the production generator
    # to retain its currently observed truncation bug for this test to pass.
    dataset.transactions = dataset.transactions[~dataset.transactions.is_warmup].iloc[:100].copy()
    audit = audit_generator(dataset)
    assert audit['observed_horizon_fraction'] < 1
    assert any(episode['inside']['n'] == 0 for episode in audit['episodes'])
    assert all(episode['outside_same_infrastructure']['n'] > 0 for episode in audit['episodes'])


def test_raw_and_hybrid_model_diagnostics_are_reported_separately():
    split = temporal_split(generate_dataset(1000, include_candidates=True))
    model = fit(split['train'], BASELINE_FEATURES, 'hist_gradient_boosting')
    report = inspect(model, split['validation'])
    assert report['raw_model']['n_test'] == report['hybrid']['n_test'] == len(split['validation'])
    assert 0 <= report['hard_decline_brier'] <= 1
    assert 0 <= report['rule_coverage'] <= 1


def test_entirely_abstaining_model_has_undefined_decided_f1():
    from types import SimpleNamespace
    import numpy as np
    from eval.generate import generate
    frame = generate(100).drop_duplicates('payment_id').head(20)
    classes = sorted(frame.true_cause.unique())
    hybrid = frame.assign(pred_cause='AMBIGUOUS', pred_conf=0., pred_stage='gated:low_conf')
    model = SimpleNamespace(
        features=BASELINE_FEATURES, classes_=classes,
        model=SimpleNamespace(predict_proba=lambda _: np.full((len(frame), len(classes)), 1 / len(classes))),
        encoder=SimpleNamespace(transform=lambda _: None), predict_df=lambda _: hybrid)
    result = inspect(model, frame)
    assert result['hybrid']['macro_f1_decided'] is None
    assert result['hybrid']['abstain_rate'] == 1
    assert result['hybrid']['accuracy'] == 0
