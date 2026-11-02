"""Corrected-data calibration, traceable safety tuning, then feature-group ablation.

All artifacts are offline. No live DB writes or serving model replacement.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from eval.classifier import HARD_CAUSES, RULES, RETRYABLE, CONF_THRESHOLD, HARD_DECLINE_MASS_LIMIT
from eval.datasets import safety_split
from eval.feature_ablation import FEATURE_GROUPS, FeatureAblationClassifier
from eval.generate import generate_dataset
from eval.generator_checks import verify_generator
from eval.probability_calibration import ProbabilityCalibrator, calibration_metrics
from eval.scenarios import SCENARIOS, generate_scenario
from services.app.domain.diagnosis_safety import SafetyThresholds


def prepare(frame, model, probabilities):
    classes = np.asarray(model.classes_)
    rules = np.asarray([str(RULES.get(tuple(row), '')) for row in
                        frame[['gw_error_code', 'gw_error_source', 'gw_error_step']].itertuples(index=False, name=None)])
    known = np.asarray([tuple(row) in model.known_tuples for row in
                        frame[['gw_error_code', 'gw_error_source', 'gw_error_step']].itertuples(index=False, name=None)])
    return {'frame': frame.reset_index(drop=True), 'p': probabilities, 'classes': classes,
            'rules': rules, 'known': known, 'top': classes[probabilities.argmax(axis=1)],
            'confidence': probabilities.max(axis=1),
            'hard_mass': probabilities[:, [i for i, c in enumerate(classes) if c in HARD_CAUSES]].sum(axis=1)}


def apply_policy(data, thresholds=SafetyThresholds(), legacy=False, raw=False):
    top, confidence, mass, rules = (data[k] for k in ('top', 'confidence', 'hard_mass', 'rules'))
    cause = top.copy().astype(object)
    reason = np.full(len(top), 'reliable_diagnosis', dtype=object)
    if raw:
        reason[:] = 'raw_model'
    elif legacy:
        for i in range(len(top)):
            if rules[i] in HARD_CAUSES:
                cause[i], reason[i] = rules[i], 'legacy_hard_rule'
            elif rules[i]:
                cause[i], reason[i] = ((rules[i], 'legacy_retryable_rule') if mass[i] <= HARD_DECLINE_MASS_LIMIT
                                       else ('AMBIGUOUS', 'legacy_rules_hard_risk'))
            elif confidence[i] < CONF_THRESHOLD:
                cause[i], reason[i] = 'AMBIGUOUS', 'legacy_low_confidence'
            elif top[i] not in HARD_CAUSES and mass[i] > HARD_DECLINE_MASS_LIMIT:
                cause[i], reason[i] = 'AMBIGUOUS', 'legacy_hard_risk'
    else:
        retry_top = np.isin(top, [str(c) for c, allowed in RETRYABLE.items() if allowed])
        # First matching condition wins, matching the shared pure policy exactly.
        expired = data['frame'].instrument_hint.eq('expired').to_numpy()
        hard_rule = np.isin(rules, HARD_CAUSES)
        conditions = [expired, hard_rule, confidence < thresholds.min_confidence,
                      retry_top & (not thresholds.retry_enabled),
                      retry_top & ~data['known'], retry_top & (mass > thresholds.max_hard_mass),
                      retry_top & (rules != '') & (rules != top)]
        reasons = ['expired_instrument_veto', 'hard_rule_veto', 'low_confidence',
                   'retry_disabled_by_validation', 'unknown_error_tuple', 'hard_decline_mass_veto',
                   'rule_model_disagreement']
        reason = np.select(conditions, reasons, default='reliable_diagnosis')
        cause[reason != 'reliable_diagnosis'] = 'AMBIGUOUS'
        cause[expired] = 'CARD_EXPIRED'
        cause[hard_rule & ~expired] = rules[hard_rule & ~expired]
    retry = np.isin(cause, [str(c) for c, allowed in RETRYABLE.items() if allowed])
    return cause, reason, retry


def metrics(data, result):
    cause, reason, retry = result
    truth = data['frame'].true_cause.to_numpy()
    hard = np.isin(truth, HARD_CAUSES)
    retryable = np.isin(truth, [str(c) for c, allowed in RETRYABLE.items() if allowed])
    unsafe = hard & retry
    decided = cause != 'AMBIGUOUS'
    return {'n': len(cause), 'macro_f1_all': float(f1_score(truth, cause, labels=sorted(set(truth)), average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(truth, cause)), 'abstain_rate': float((~decided).mean()),
            'unsafe_retry_count': int(unsafe.sum()), 'hard_decline_count': int(hard.sum()),
            'unsafe_retry_rate': float(unsafe.sum()/hard.sum()) if hard.any() else None,
            'retry_coverage': float(retry.mean()),
            'true_retryable_recall': float((retryable & retry).sum()/retryable.sum()) if retryable.any() else None,
            'false_retry_fraction_of_allowed': float(unsafe.sum()/retry.sum()) if retry.any() else None,
            'decisions': dict(Counter(reason.tolist()))}


def tune_policy(partitions, max_unsafe_rate=0., on_evaluation=None):
    candidates = [SafetyThresholds(conf, mass) for conf in (.60, .75, .90, .95, .99)
                  for mass in (0., .01, .03, .06, .10, .25)]
    candidates.append(SafetyThresholds(.60, 0., retry_enabled=False))
    table = []
    for candidate_id, thresholds in enumerate(candidates):
        values = {}
        for name, data in partitions.items():
            result = apply_policy(data, thresholds)
            values[name] = metrics(data, result)
            if on_evaluation is not None:
                on_evaluation(candidate_id, thresholds, name, data, result)
        feasible = all(v['unsafe_retry_rate'] is not None and v['unsafe_retry_rate'] <= max_unsafe_rate
                       for v in values.values())
        table.append({'candidate_id': candidate_id, 'thresholds': asdict(thresholds), 'feasible': feasible,
                      'mean_f1': float(np.mean([v['macro_f1_all'] for v in values.values()])),
                      'mean_accuracy': float(np.mean([v['accuracy'] for v in values.values()])),
                      'mean_retry_coverage': float(np.mean([v['retry_coverage'] for v in values.values()])),
                      'partitions': values})
    feasible = [row for row in table if row['feasible']]
    if not feasible:
        raise ValueError('No policy meets the validation safety constraint')
    selected = max(feasible, key=lambda row: (row['mean_f1'], row['mean_accuracy'], row['mean_retry_coverage']))
    return selected, table


def trace_failures(stream, counters, data, raw_probabilities, result, *, group, method, partition, policy, thresholds):
    cause, reason, retry = result
    hard = data['frame'].true_cause.isin(HARD_CAUSES).to_numpy()
    for i in np.flatnonzero(hard & retry):
        row = data['frame'].iloc[i]
        model_wrong = data['top'][i] != row.true_cause
        record = {'payment_id': row.payment_id, 'partition': partition, 'feature_group': group,
                  'calibration': method, 'policy': policy, 'true_cause': row.true_cause,
                  'predicted_cause': cause[i], 'model_prediction': data['top'][i],
                  'class_probabilities': dict(zip(data['classes'].tolist(), data['p'][i].tolist())),
                  'uncalibrated_class_probabilities': dict(zip(data['classes'].tolist(), raw_probabilities[i].tolist())),
                  'rule_result': data['rules'][i] or None, 'known_error_tuple': bool(data['known'][i]),
                  'instrument_hint': row.instrument_hint, 'hard_decline_mass': float(data['hard_mass'][i]),
                  'confidence': float(data['confidence'][i]), 'thresholds': thresholds,
                  'final_gate_decision': str(reason[i]), 'retry_allowed': True,
                  'diagnostic_flags': {'incorrect_model_classification': bool(model_wrong),
                                       'confident_wrong_prediction': bool(model_wrong and data['confidence'][i] >= .60),
                                       'rule_overrode_model': bool(cause[i] == data['rules'][i] and cause[i] != data['top'][i])}}
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        counters[f'{partition}:{policy}'] += 1


def probabilities(model, frame):
    return model.model.predict_proba(model.encoder.transform(frame[list(model.features)]))


def write_summary(report, directory):
    groups = {}
    for name, result in report['feature_groups'].items():
        method = result['selected_calibration']
        confirmation = result['confirmation'][method]
        groups[name] = {
            'features': result['features'], 'calibration': method,
            'thresholds': result['methods'][method]['policy_selection']['thresholds'],
            'mean_confirmation_f1': float(np.mean([v['revised_tuned']['macro_f1_all'] for v in confirmation.values()])),
            'mean_confirmation_accuracy': float(np.mean([v['revised_tuned']['accuracy'] for v in confirmation.values()])),
            'unsafe_confirmation_retries': sum(v['revised_tuned']['unsafe_retry_count'] for v in confirmation.values()),
            'retry_coverage': {key: value['revised_tuned']['retry_coverage'] for key,value in confirmation.items()}}
    summary = {'generator': report['generator']['generator'], 'partitions': report['partitions'],
               'generator_verification': report['generator_verification'],
               'validation_selected_group': report['validation_selected_group'],
               'confirmation_passed_safety_constraint': report['confirmation_passed_safety_constraint'],
               'feature_groups': groups, 'trace_records': report['trace_records'],
               'trace_file': report['trace_file'], 'serving_changed': False}
    (Path(directory)/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


def run(n=20000, scenario_n=2000, seed=42, output_dir='/tmp/phase2-safety'):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(n=n, seed=seed, include_candidates=True)
    generator_verification = verify_generator(dataset)
    split = safety_split(dataset)
    print('Corrected generator and four partitions ready', flush=True)
    if any(frame.empty for frame in split.values()):
        raise ValueError('Insufficient data for four-way evaluation')
    calibration_parts = {'temporal_calibration': split['calibration']}
    policy_parts = {'temporal_policy': split['policy']}
    for i, name in enumerate(SCENARIOS):
        calibration_parts[name] = generate_scenario(name, n=scenario_n, seed=41000+i, include_candidates=True).failures
        policy_parts[name] = generate_scenario(name, n=scenario_n, seed=51000+i, include_candidates=True).failures
    # The source groups, calibration methods, grid and zero-observed-unsafe budget
    # are fixed before inspecting confirmation data.
    report = {'generator': dataset.metadata, 'generator_verification': generator_verification,
              'partitions': {k: len(v) for k, v in split.items()},
              'partition_ids_sha256': {k: hashlib.sha256('|'.join(sorted(v.payment_id)).encode()).hexdigest() for k, v in split.items()},
              'calibration_scenario_seeds': list(range(41000,41005)),
              'policy_scenario_seeds': list(range(51000,51005)), 'confirmation_scenario_seeds': list(range(61000,61005)),
              'unsafe_rate_budget': 0., 'feature_groups': {}, 'serving_changed': False,
              'limits': 'Synthetic data only. Zero observed unsafe retries is not zero population risk. Scenario families are known development families; held-out seeds are not real-world validation. No production promotion.'}
    artifacts, trace_counts = {}, Counter()
    with gzip.open(destination/'unsafe-retries.jsonl.gz', 'wt', encoding='utf-8') as trace:
        for group, features in FEATURE_GROUPS.items():
            print(f'Fit, calibrate, and validate {group}', flush=True)
            model = FeatureAblationClassifier(features).fit(split['fit'])
            model.known_tuples = set(split['fit'][['gw_error_code','gw_error_source','gw_error_step']].itertuples(index=False, name=None))
            calibration_frame = pd.concat(calibration_parts.values(), ignore_index=True)
            calibration_raw = probabilities(model, calibration_frame)
            policy_raw = {name: probabilities(model, frame) for name, frame in policy_parts.items()}
            methods = {}
            for method in ('uncalibrated', 'sigmoid', 'isotonic'):
                calibrator = ProbabilityCalibrator(method, model.classes_).fit(calibration_raw, calibration_frame.true_cause)
                prepared = {name: prepare(frame, model, calibrator.transform(policy_raw[name]))
                            for name, frame in policy_parts.items()}
                def trace_grid(candidate_id, thresholds, name, data, result):
                    trace_failures(trace, trace_counts, data, policy_raw[name], result,
                                   group=group, method=method, partition=f'policy_validation/{name}',
                                   policy=f'grid_candidate_{candidate_id:02d}', thresholds=asdict(thresholds))
                chosen, grid = tune_policy(prepared, on_evaluation=trace_grid)
                thresholds = SafetyThresholds(**chosen['thresholds'])
                values = {}
                for name, data in prepared.items():
                    variants = {'raw_model': apply_policy(data, raw=True), 'legacy_gate': apply_policy(data, legacy=True),
                                'revised_default': apply_policy(data), 'revised_tuned': apply_policy(data, thresholds)}
                    values[name] = {'probability_quality': calibration_metrics(data['p'], data['frame'].true_cause, model.classes_),
                                    **{policy: metrics(data, result) for policy, result in variants.items()}}
                    for policy, result in variants.items():
                        trace_failures(trace, trace_counts, data, policy_raw[name], result,
                                       group=group, method=method, partition=f'policy_validation/{name}', policy=policy,
                                       thresholds=asdict(thresholds) if policy == 'revised_tuned' else
                                                  asdict(SafetyThresholds()) if policy == 'revised_default' else
                                                  {'min_confidence': CONF_THRESHOLD, 'max_hard_mass': HARD_DECLINE_MASS_LIMIT} if policy == 'legacy_gate' else None)
                methods[method] = {'policy_selection': chosen, 'threshold_grid': grid, 'policy_validation': values}
                artifacts[(group,method)] = (model, calibrator, thresholds)
            selected_method = max(methods, key=lambda name: (methods[name]['policy_selection']['mean_f1'], methods[name]['policy_selection']['mean_accuracy']))
            report['feature_groups'][group] = {'features': list(features), 'methods': methods, 'selected_calibration': selected_method}
        # Freeze feature/model/calibration/policy choice before confirmation generation.
        baseline = report['feature_groups']['baseline_5']
        baseline_choice = baseline['methods'][baseline['selected_calibration']]['policy_selection']
        eligible = []
        for group, result in report['feature_groups'].items():
            choice = result['methods'][result['selected_calibration']]['policy_selection']
            if choice['mean_f1'] >= baseline_choice['mean_f1'] and choice['mean_accuracy'] >= baseline_choice['mean_accuracy']:
                eligible.append(group)
        selected_group = max(eligible, key=lambda group: (
            report['feature_groups'][group]['methods'][report['feature_groups'][group]['selected_calibration']]['policy_selection']['mean_f1'],
            -len(report['feature_groups'][group]['features'])))
        report['validation_selected_group'] = selected_group
        report['selection_frozen_before_confirmation'] = True
        confirmation_parts = {'temporal_confirmation': split['confirmation']}
        for i, name in enumerate(SCENARIOS):
            confirmation_parts[name] = generate_scenario(name, n=scenario_n, seed=61000+i, include_candidates=True).failures
        for group, result in report['feature_groups'].items():
            print(f'Confirm frozen policies for {group}', flush=True)
            result['confirmation'] = {}
            for method in ('uncalibrated', 'sigmoid', 'isotonic'):
                model, calibrator, thresholds = artifacts[(group,method)]
                values = {}
                for name, frame in confirmation_parts.items():
                    raw = probabilities(model, frame)
                    data = prepare(frame, model, calibrator.transform(raw))
                    variants = {'raw_model': apply_policy(data, raw=True), 'legacy_gate': apply_policy(data, legacy=True),
                                'revised_default': apply_policy(data), 'revised_tuned': apply_policy(data, thresholds)}
                    values[name] = {'probability_quality': calibration_metrics(data['p'], frame.true_cause, model.classes_),
                                    **{policy: metrics(data, decision) for policy, decision in variants.items()}}
                    for policy, decision in variants.items():
                        trace_failures(trace, trace_counts, data, raw, decision, group=group, method=method,
                                       partition=f'confirmation/{name}', policy=policy, thresholds=asdict(thresholds) if policy == 'revised_tuned' else
                                       asdict(SafetyThresholds()) if policy == 'revised_default' else
                                       {'min_confidence': CONF_THRESHOLD, 'max_hard_mass': HARD_DECLINE_MASS_LIMIT} if policy == 'legacy_gate' else None)
                result['confirmation'][method] = values
        selected = report['feature_groups'][selected_group]
        confirmed = selected['confirmation'][selected['selected_calibration']]
        report['confirmation_passed_safety_constraint'] = all(v['revised_tuned']['unsafe_retry_count'] == 0 for v in confirmed.values())
        report['selected_retry_coverage_by_partition'] = {name: v['revised_tuned']['retry_coverage'] for name,v in confirmed.items()}
    report['trace_counts'] = dict(trace_counts)
    report['trace_records'] = sum(trace_counts.values())
    expected_trace_records = 0
    for result in report['feature_groups'].values():
        for method in result['methods']:
            expected_trace_records += sum(value['unsafe_retry_count']
                                          for candidate in result['methods'][method]['threshold_grid']
                                          for value in candidate['partitions'].values())
            for partitions in (result['methods'][method]['policy_validation'], result['confirmation'][method]):
                for values in partitions.values():
                    expected_trace_records += sum(values[policy]['unsafe_retry_count']
                                                  for policy in ('raw_model','legacy_gate','revised_default','revised_tuned'))
    assert expected_trace_records == report['trace_records'], 'Unsafe-retry trace must match every evaluated failure'
    report['trace_file'] = 'unsafe-retries.jsonl.gz'
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    write_summary(report, destination)
    print(f'Report: {destination}/report.json; selected group: {selected_group}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--scenario-n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='/tmp/phase2-safety')
    args = parser.parse_args()
    run(args.n, args.scenario_n, args.seed, args.output_dir)
