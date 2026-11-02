"""Diagnose generator coverage, conditional information, and model/rule behavior.

Development-only investigation, with no feature promotion or serving changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder

from eval.classifier import HARD_CAUSES, HARD_DECLINE_MASS_LIMIT, RULES, evaluate
from eval.datasets import temporal_split
from eval.feature_ablation import FeatureAblationClassifier, INFRASTRUCTURE, HISTORY, CONTEXT
from eval.generate import generate_dataset
from eval.scenarios import SCENARIOS, generate_scenario
from eval.sequences import DURATION_S
from services.app.domain.features import BASELINE_FEATURES, CATEGORICAL_FEATURES, FEATURES

SETS = {
    'gateway_only': BASELINE_FEATURES[:3],
    'baseline_5': BASELINE_FEATURES,
    'context_without_gateway': tuple(f for f in FEATURES if f not in (*BASELINE_FEATURES[:3], 'issuer_id', 'psp_id')),
    'all_26': FEATURES,
}


def audit_generator(dataset):
    frame = dataset.transactions[~dataset.transactions.is_warmup]
    origin = pd.Timestamp(dataset.metadata['measurement_start']) if 'measurement_start' in dataset.metadata else frame.occurred_at.min().normalize()
    elapsed = (frame.occurred_at - origin).dt.total_seconds()
    episodes = []
    for event in dataset.infrastructure.to_dict('records'):
        field = 'issuer_id' if event['kind'] == 'issuer' else 'psp_id'
        same = frame[field].eq(event['entity'])
        inside = same & elapsed.ge(event['start_s']) & elapsed.lt(event['end_s'])
        outside = same & ~inside
        stats = lambda mask: {'n': int(mask.sum()),
                              'failure_rate': float(frame.loc[mask, 'status'].eq('failed').mean()) if mask.any() else None,
                              'technical_cause_rate': float(frame.loc[mask, 'true_cause'].isin(['TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT']).mean()) if mask.any() else None,
                              'median_latency_ms': float(frame.loc[mask, 'latency_ms'].median()) if mask.any() else None}
        episodes.append({**event, 'inside': stats(inside), 'outside_same_infrastructure': stats(outside)})
    failures = dataset.failures
    counts = {name: {'zero_fraction': float(failures[name].eq(0).mean()),
                     'median': float(failures[name].median()), 'p90': float(failures[name].quantile(.9))}
              for name in ('issuer_attempts_5m', 'psp_attempts_5m', 'method_attempts_5m', 'customer_attempts_7d')}
    return {'configured_duration_s': DURATION_S, 'last_event_s': float(elapsed.max()),
            'observed_horizon_fraction': float(elapsed.max() / DURATION_S), 'episodes': episodes,
            'feature_support': counts, 'class_mix': failures.true_cause.value_counts(normalize=True).to_dict(),
            'no_prior_success_fraction': float(failures.customer_last_success_age_s.isna().mean()),
            'repeated_order_fraction': float(frame.order_id.duplicated().mean())}


def fit(train, features, family):
    model = FeatureAblationClassifier(features)
    if family == 'random_forest':
        categorical = [f for f in features if f in CATEGORICAL_FEATURES]
        numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
        model.encoder = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('numeric', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), numeric),
        ])
        model.model = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_depth=18,
                                              random_state=0, n_jobs=2)
    return model.fit(train)


def inspect(model, frame):
    probability = model.model.predict_proba(model.encoder.transform(frame[list(model.features)]))
    top = np.asarray(model.classes_)[probability.argmax(axis=1)]
    raw = frame.copy()
    raw['pred_cause'] = top
    raw['pred_conf'] = probability.max(axis=1)
    raw['pred_stage'] = 'raw_model'
    hybrid = model.predict_df(frame)
    def metrics(prediction):
        if prediction.pred_cause.eq('AMBIGUOUS').all():
            # Decided-only F1 is undefined, not a crash or evidence of accuracy.
            summary = {'n_test': len(prediction), 'abstain_rate': 1.,
                       'stage_mix': prediction.pred_stage.value_counts(normalize=True).to_dict(),
                       'macro_f1_decided': None,
                       'per_class_recall': {str(label): 0. for label in sorted(set(frame.true_cause))},
                       'per_class_precision': {str(label): 0. for label in sorted(set(frame.true_cause))},
                       'unsafe_retry_rate': 0., 'unsafe_retry_count': 0,
                       'hard_decline_count': int(frame.true_cause.isin(HARD_CAUSES).sum())}
        else:
            summary = evaluate(prediction)
        return {**summary, 'accuracy': float(accuracy_score(prediction.true_cause, prediction.pred_cause)),
                'macro_f1_all': float(f1_score(prediction.true_cause, prediction.pred_cause,
                                             labels=sorted(set(frame.true_cause)), average='macro', zero_division=0))}
    hard_probability = probability[:, [i for i, cause in enumerate(model.classes_) if cause in HARD_CAUSES]].sum(axis=1)
    hard = frame.true_cause.isin(HARD_CAUSES).to_numpy()
    rules = np.array([str(RULES.get((row.gw_error_code, row.gw_error_source, row.gw_error_step), ''))
                      for row in frame.itertuples(index=False)])
    rule_rows = rules != ''
    raw_confident = probability.max(axis=1) >= .60
    return {'raw_model': metrics(raw), 'hybrid': metrics(hybrid),
            'raw_multiclass_log_loss': float(log_loss(frame.true_cause, probability, labels=model.classes_)),
            'hard_decline_brier': float(np.mean((hard_probability - hard.astype(float)) ** 2)),
            'true_hard_below_safety_mass_limit_fraction': float((hard_probability[hard] <= HARD_DECLINE_MASS_LIMIT).mean()) if hard.any() else None,
            'rule_coverage': float(rule_rows.mean()),
            'rule_model_disagreement_fraction': float((rules[rule_rows] != top[rule_rows]).mean()) if rule_rows.any() else None,
            'raw_hybrid_disagreement_fraction': float((top != hybrid.pred_cause).mean()),
            'raw_confident_wrong_fraction': float((raw_confident & (top != frame.true_cause.to_numpy())).mean())}


def conditional_shuffle(frame, columns, seed):
    """Shuffle a group jointly within identical baseline tuples (labels unchanged)."""
    result = frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    moved = 0
    for indices in result.groupby(list(BASELINE_FEATURES), dropna=False).indices.values():
        order = rng.permutation(indices)
        moved += int((order != indices).sum())
        result.loc[indices, list(columns)] = frame.reset_index(drop=True).loc[order, list(columns)].to_numpy()
    return result, moved / len(frame)


def run(n=20000, scenario_n=2000):
    report = {'purpose': 'Development investigation, not production selection', 'seeds': [42, 43],
              'n': n, 'scenario_n': scenario_n, 'worlds': {}, 'serving_changed': False,
              'limits': 'Two synthetic seeds, fixed untuned model configurations. Conditional shuffles are model-usage probes, not proof of redundancy or causality. Previously inspected scenario families remain development data.'}
    for seed in (42, 43):
        print(f'Generating seed {seed}', flush=True)
        dataset = generate_dataset(n=n, seed=seed, include_candidates=True)
        split = temporal_split(dataset)
        validation = {'temporal_validation': split['validation']}
        for i, name in enumerate(SCENARIOS):
            validation[name] = generate_scenario(name, n=scenario_n, seed=11001 + seed * 100 + i,
                                                 include_candidates=True).failures
        world = {'generator_audit': audit_generator(dataset), 'models': {}, 'conditional_permutation': {}}
        for family in ('hist_gradient_boosting', 'random_forest'):
            for name, features in SETS.items():
                print(f'Seed {seed}: {family} / {name}', flush=True)
                model = fit(split['train'], features, family)
                world['models'][f'{family}:{name}'] = {key: inspect(model, frame) for key, frame in validation.items()}
                if name == 'all_26':
                    permutation = {}
                    for group, columns in (('infrastructure', INFRASTRUCTURE), ('history', HISTORY),
                                           ('context', CONTEXT), ('identifiers', ('issuer_id', 'psp_id'))):
                        permutation[group] = []
                        for permutation_seed in (91, 92, 93):
                            shuffled, moved = conditional_shuffle(split['validation'], columns, permutation_seed)
                            permutation[group].append({'seed': permutation_seed, 'row_position_changed_fraction': moved,
                                                       'metrics': inspect(model, shuffled)})
                    world['conditional_permutation'][family] = permutation
        report['worlds'][str(seed)] = world
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--scenario-n', type=int, default=2000)
    parser.add_argument('--output', default='/tmp/phase2-investigation.json')
    args = parser.parse_args()
    report = run(args.n, args.scenario_n)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Report: {args.output}', flush=True)
