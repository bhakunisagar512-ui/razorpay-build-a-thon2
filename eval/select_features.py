"""Sparse, safety-constrained forward selection using development data only.

Previously published temporal test rows and scenario seeds 901..905 are excluded.
Accuracy counts abstentions as incorrect. Safety is a hard constraint, not a
weighted penalty that an F1 gain can compensate for.
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from eval.classifier import CauseClassifier
from eval.datasets import temporal_split
from eval.feature_ablation import FeatureAblationClassifier
from eval.generate import generate_dataset
from eval.phase2 import score
from eval.scenarios import SCENARIOS, generate_scenario
from services.app.domain.features import BASELINE_FEATURES, FEATURES


def measure(model, frame):
    metrics = score(model, frame)
    prediction = model.predict_df(frame)
    metrics['accuracy'] = float(accuracy_score(prediction.true_cause, prediction.pred_cause))
    return metrics


def admissible(metrics, baseline, current):
    """Never trade more unsafe retries for better average diagnosis."""
    safe = all(value['unsafe_retry_count'] <= baseline[key]['unsafe_retry_count']
               for key, value in metrics.items())
    mean = lambda results, field: sum(m[field] for m in results.values()) / len(results)
    df1 = mean(metrics, 'macro_f1_all') - mean(current, 'macro_f1_all')
    daccuracy = mean(metrics, 'accuracy') - mean(current, 'accuracy')
    return safe and df1 > 1e-6 and daccuracy > 1e-6, {
        'mean_f1_gain': df1, 'mean_accuracy_gain': daccuracy,
        'safety_regressions': [key for key in metrics if metrics[key]['unsafe_retry_count'] > baseline[key]['unsafe_retry_count']],
        'score_gain': (df1 + daccuracy) / 2,
    }


def run(n=20000, scenario_n=2000):
    seeds = (42, 43)
    worlds = {}
    for seed in seeds:
        print(f'Preparing development data seed {seed}', flush=True)
        split = temporal_split(generate_dataset(n=n, seed=seed, include_candidates=True))
        worlds[seed] = (split['train'], {'temporal_validation': split['validation']})
        for i, name in enumerate(SCENARIOS):
            worlds[seed][1][name] = generate_scenario(name, n=scenario_n, seed=3101 + 100 * seed + i,
                                                   include_candidates=True).failures

    def evaluate_features(features):
        metrics = {}
        for seed, (train, validation) in worlds.items():
            model = CauseClassifier() if tuple(features) == BASELINE_FEATURES else FeatureAblationClassifier(features)
            model.fit(train)
            for name, frame in validation.items():
                metrics[f'{seed}:{name}'] = measure(model, frame)
        return metrics

    selected = list(BASELINE_FEATURES)
    remaining = [feature for feature in FEATURES if feature not in BASELINE_FEATURES]
    baseline = evaluate_features(selected)
    current = baseline
    rounds = []
    while remaining:
        candidates = []
        for feature in remaining:
            metrics = evaluate_features([*selected, feature])
            keep, contribution = admissible(metrics, baseline, current)
            candidates.append({'feature': feature, 'eligible': keep, **contribution, 'metrics': metrics})
            print(f'{len(selected)} + {feature}: eligible={keep} F1={contribution["mean_f1_gain"]:+.5f} accuracy={contribution["mean_accuracy_gain"]:+.5f} safety_regressions={len(contribution["safety_regressions"])}', flush=True)
        eligible = [row for row in candidates if row['eligible']]
        best = max(eligible, key=lambda row: row['score_gain']) if eligible else None
        rounds.append({'selected_before': selected.copy(), 'candidates': candidates,
                       'added': best['feature'] if best else None})
        if not best:
            break
        selected.append(best['feature'])
        remaining.remove(best['feature'])
        current = best['metrics']

    # Final joint-set checks: do not describe individual gains as additive.
    removals = {}
    for feature in selected[len(BASELINE_FEATURES):]:
        subset = [f for f in selected if f != feature]
        metrics = evaluate_features(subset)
        _, delta = admissible(current, baseline, metrics)
        removals[feature] = {'contribution_when_removed': delta, 'without_feature_metrics': metrics}
    return {
        'method': 'Greedy forward selection with per-partition zero unsafe-retry-count regression; both mean macro-F1 and accuracy must improve',
        'development_seeds': list(seeds), 'scenario_seed_rule': '3101 + 100 * train_seed + scenario_index',
        'n_attempts_per_training_world': n, 'n_attempts_per_development_scenario': scenario_n,
        'accuracy_definition': 'All-row exact-match accuracy; abstention is incorrect',
        'aggregation': 'Equal weight per development partition; safety checked separately on every partition',
        'baseline_metrics': baseline, 'rounds': rounds, 'leave_one_out': removals,
        'selected_features': selected, 'removed_features': [f for f in FEATURES if f not in selected],
        'selected_metrics': current, 'production_promoted': False,
        'limitations': 'Greedy search does not exhaustively test interactions. Selection on finite synthetic development data does not prove population safety. New untouched confirmation data is required before promotion.',
    }


def add_contribution_ranking(report):
    """Readable one-feature effects, not additive or causal importance scores."""
    ranking = []
    for row in report['rounds'][0]['candidates']:
        ranking.append({
            'feature': row['feature'], 'retained': row['feature'] in report['selected_features'],
            'mean_f1_gain': row['mean_f1_gain'], 'mean_accuracy_gain': row['mean_accuracy_gain'],
            'safety_regression_partitions': len(row['safety_regressions']),
            'max_unsafe_retry_rate_increase': max(
                metrics['unsafe_retry_rate'] - report['baseline_metrics'][key]['unsafe_retry_rate']
                for key, metrics in row['metrics'].items()),
        })
    report['ranked_single_feature_contributions'] = sorted(
        ranking, key=lambda row: row['mean_f1_gain'] + row['mean_accuracy_gain'], reverse=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--scenario-n', type=int, default=2000)
    parser.add_argument('--output', default='/tmp/phase2-feature-selection.json')
    args = parser.parse_args()
    report = add_contribution_ranking(run(args.n, args.scenario_n))
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print('SELECTED', report['selected_features'], flush=True)
