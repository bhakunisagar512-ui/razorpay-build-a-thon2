"""Safety must remain a hard constraint during feature selection."""
from eval.select_features import admissible


def metrics(f1=.7, accuracy=.75, unsafe=2):
    return {'temporal': {'macro_f1_all': f1, 'accuracy': accuracy, 'unsafe_retry_count': unsafe},
            'stress': {'macro_f1_all': f1, 'accuracy': accuracy, 'unsafe_retry_count': unsafe}}


def test_f1_and_accuracy_cannot_compensate_for_a_safety_regression():
    baseline = metrics()
    candidate = metrics(.99, .99, 2)
    candidate['stress']['unsafe_retry_count'] = 3
    accepted, contribution = admissible(candidate, baseline, baseline)
    assert not accepted
    assert contribution['safety_regressions'] == ['stress']


def test_improved_f1_with_worse_accuracy_is_not_selected():
    baseline = metrics()
    assert not admissible(metrics(.8, .7, 1), baseline, baseline)[0]


def test_safe_improvement_is_eligible_but_redundant_feature_is_not():
    baseline = metrics()
    improved = metrics(.8, .8, 1)
    assert admissible(improved, baseline, baseline)[0]
    assert not admissible(improved, baseline, improved)[0]


def test_default_feature_builder_and_model_exclude_candidates():
    from services.app.domain.features import FeatureBuilder, BASELINE_FEATURES
    from services.app.domain.selected_features import MODEL_FEATURES
    from eval.feature_ablation import FeatureAblationClassifier
    assert FeatureBuilder().feature_names == MODEL_FEATURES
    assert FeatureAblationClassifier().features == MODEL_FEATURES
    assert MODEL_FEATURES == BASELINE_FEATURES


def test_default_dataset_contains_only_selected_engineered_inputs():
    from eval.generate import generate_dataset
    from services.app.domain.selected_features import MODEL_FEATURES
    dataset = generate_dataset(300)
    assert dataset.metadata['model_features'] == list(MODEL_FEATURES)
    assert 'issuer_failure_rate_5m' not in dataset.failures
    assert 'customer_failure_rate_7d' not in dataset.failures
    assert 'amount_minor' in dataset.failures  # Policy/accounting metadata, not a selected model input.
