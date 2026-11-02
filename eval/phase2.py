"""Reproducible baseline capture and validation-led feature-group comparison.

No LLM calls, serving artifact writes, database changes, or production promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version

from sklearn.metrics import f1_score

from eval.classifier import CauseClassifier, evaluate, train_test_split_by_hash
from eval.datasets import temporal_split
from eval.feature_ablation import FEATURE_GROUPS, FeatureAblationClassifier
from eval.generate import generate, generate_dataset
from eval.scenarios import SCENARIOS, generate_scenario
from services.app.domain.features import FEATURE_SCHEMA_VERSION


def fingerprint(frame):
    return hashlib.sha256(frame.to_json(orient='split', date_format='iso').encode()).hexdigest()


def score(model, frame):
    if frame.empty:
        raise ValueError('Empty evaluation partition; increase dataset size')
    prediction = model.predict_df(frame)
    return {**evaluate(prediction), 'macro_f1_all': float(f1_score(
        prediction.true_cause, prediction.pred_cause, labels=sorted(set(prediction.true_cause)),
        average='macro', zero_division=0))}


def run(n=20000, scenario_n=4000, seed=42, export_dir=None, legacy_n=40000, all_candidates=False):
    from services.app.domain.selected_features import MODEL_FEATURES
    groups = FEATURE_GROUPS if all_candidates else {'baseline_5': FEATURE_GROUPS['baseline_5']}
    if not all_candidates and tuple(MODEL_FEATURES) != tuple(FEATURE_GROUPS['baseline_5']):
        groups['selected'] = MODEL_FEATURES
    print('Capturing unchanged legacy classifier baseline', flush=True)
    legacy = generate(n=legacy_n, seed=seed).drop_duplicates('payment_id').reset_index(drop=True)
    legacy_train, legacy_test = train_test_split_by_hash(legacy)
    frozen = CauseClassifier().fit(legacy_train)
    report = {'feature_schema': FEATURE_SCHEMA_VERSION, 'seed': seed, 'synthetic_only': True,
              'environment': {'python': platform.python_version(),
                              **{pkg: version(pkg) for pkg in ('numpy', 'pandas', 'scikit-learn')}},
              'legacy_baseline': {'n_generated': legacy_n, 'train_rows': len(legacy_train),
                                  'dataset_sha256': fingerprint(legacy), 'metrics': score(frozen, legacy_test)},
              'feature_groups': {name: list(features) for name, features in groups.items()}}
    report['legacy_baseline']['generator_source_sha256'] = hashlib.sha256(Path('eval/legacy_generate.py').read_bytes()).hexdigest()
    report['legacy_baseline']['classifier_source_sha256'] = hashlib.sha256(Path('eval/classifier.py').read_bytes()).hexdigest()
    report['legacy_baseline']['published_reference'] = {
        'path': 'eval/data/step4.txt',
        'sha256': hashlib.sha256(Path('eval/data/step4.txt').read_bytes()).hexdigest(),
        'note': 'Original 40,000-row published metrics are retained unchanged; this report re-evaluates classification only.'}
    print('Generating sequential training world and temporal partitions', flush=True)
    dataset = generate_dataset(n=n, seed=seed, include_candidates=all_candidates)
    split = temporal_split(dataset)
    if any(part.empty for part in split.values()):
        raise ValueError('Temporal partitions need more data')
    report['sequences'] = {**dataset.metadata, 'dataset_sha256': fingerprint(dataset.transactions),
                           'partition_rows': {name: len(frame) for name, frame in split.items()},
                           'excluded_failures': len(dataset.failures) - sum(len(f) for f in split.values()),
                           'later_success_orders': int(dataset.outcomes.later_success_observed.sum()),
                           'split_sha256': {name: fingerprint(frame[['payment_id']]) for name, frame in split.items()},
                           'missing_rates': dataset.failures[list(dict.fromkeys(f for group in groups.values() for f in group))].isna().mean().to_dict()}
    models, validation = {}, {}
    for name, features in groups.items():
        print(f'Fitting {name} ({len(features)} features)', flush=True)
        model = CauseClassifier() if name == 'baseline_5' else FeatureAblationClassifier(features)
        models[name] = model.fit(split['train'])
        validation[name] = score(model, split['validation'])
    # This rule is fixed before test/scenario scoring. No threshold tuning.
    baseline = validation['baseline_5']
    eligible = [name for name, metrics in validation.items()
                if metrics['unsafe_retry_rate'] <= baseline['unsafe_retry_rate']
                and metrics['abstain_rate'] <= baseline['abstain_rate'] + .10]
    selected = max(eligible, key=lambda name: validation[name]['macro_f1_all'])
    report['validation'] = validation
    report['provisional_selection'] = {'name': selected,
        'rule': 'Highest validation macro_f1_all, no higher unsafe_retry_rate than baseline, abstention <= baseline + 0.10',
        'production_promoted': False}
    report['temporal_test'] = {name: score(model, split['test']) for name, model in models.items()}
    report['frozen_legacy_on_sequences'] = score(frozen, split['test'])
    report['scenarios'] = {}
    for i, name in enumerate(SCENARIOS):
        print(f'Evaluating independent scenario: {name}', flush=True)
        scenario = generate_scenario(name, n=scenario_n, seed=901 + i, include_candidates=all_candidates)
        assert not set(scenario.failures.payment_id) & set(dataset.failures.payment_id)
        report['scenarios'][name] = {**scenario.metadata, 'dataset_sha256': fingerprint(scenario.transactions),
                                   'frozen_legacy_baseline': score(frozen, scenario.failures),
                                   'feature_groups': {key: score(model, scenario.failures) for key, model in models.items()}}
        if export_dir:
            export(scenario, Path(export_dir) / name)
    regressions = [name for name, result in report['scenarios'].items()
                   if result['feature_groups'][selected]['unsafe_retry_rate'] > result['feature_groups']['baseline_5']['unsafe_retry_rate']]
    report['robustness_review'] = {
        'validation_selected_group': selected,
        'scenarios_with_higher_unsafe_retry_rate': regressions,
        'recommendation': 'Keep serving baseline; feature candidates require further safety validation' if regressions
                          else 'Candidate requires replication and real-data validation before serving',
        'scenario_results_used_to_retune': False,
    }
    if export_dir:
        export(dataset, Path(export_dir) / 'sequences')
    return report


def export(dataset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for field in ('transactions', 'failures', 'deliveries', 'outcomes', 'infrastructure'):
        getattr(dataset, field).to_csv(directory / f'{field}.csv', index=False)
    (directory / 'metadata.json').write_text(json.dumps(dataset.metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--scenario-n', type=int, default=4000)
    parser.add_argument('--legacy-n', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='/tmp/phase2-features.json')
    parser.add_argument('--export-dir')
    parser.add_argument('--all-candidates', action='store_true', help='Opt into rejected/unselected experimental feature sets')
    args = parser.parse_args()
    report = run(args.n, args.scenario_n, args.seed, args.export_dir, args.legacy_n, args.all_candidates)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Report: {destination}; provisional selection: {report["provisional_selection"]["name"]}', flush=True)


if __name__ == '__main__':
    main()
