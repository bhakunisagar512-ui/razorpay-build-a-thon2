"""Binary safety experiment against a frozen sigmoid-calibrated multiclass model.

No serving changes. Model/calibration fit, threshold selection, retrospective
failure analysis and fresh confirmation have explicitly separate roles.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from importlib.metadata import version
import platform

import numpy as np
import pandas as pd
from scipy.stats import beta

from eval.classifier import HARD_CAUSES
from eval.datasets import safety_split
from eval.feature_ablation import FeatureAblationClassifier
from eval.generate import generate_dataset
from eval.generator_checks import verify_generator
from eval.probability_calibration import ProbabilityCalibrator, expected_calibration_error
from eval.safety_pipeline import apply_policy, metrics, prepare, probabilities
from eval.scenarios import SCENARIOS, generate_scenario
from services.app.domain.diagnosis_safety import SafetyThresholds
from services.app.domain.features import BASELINE_FEATURES, FEATURES

FAMILIES = ('sequential', *SCENARIOS)
MULTICLASS_POLICY = SafetyThresholds(.60, .03)
REFERENCE_POLICY = SafetyThresholds(.60, .01)
MIN_GAIN_RETAINED = .80
MAX_LEGITIMATE_BLOCKED = .05
MAX_ABSTENTION_INCREASE = .05


def one_sided_upper(failures, total, alpha=.05):
    """Exact one-sided Clopper-Pearson upper bound for independent Bernoulli units."""
    if not 0 < alpha < 1 or total < 0 or not 0 <= failures <= total:
        raise ValueError('Invalid bound inputs')
    if total == 0:
        return None
    if failures == total:
        return 1.
    if failures == 0:
        return float(-math.expm1(math.log(alpha)/total))
    return float(beta.ppf(1-alpha, failures+1, total-failures))


def required_zero_failure_worlds(max_risk=.05, alpha=.05, families=len(FAMILIES)):
    if not 0 < max_risk < 1 or not 0 < alpha < 1 or families < 1:
        raise ValueError('Invalid sample-planning inputs')
    return math.ceil(math.log(alpha/families)/math.log1p(-max_risk))


class BinarySafetyClassifier:
    def __init__(self):
        self.features = tuple(FEATURES)
        self.estimator = FeatureAblationClassifier(self.features)

    @staticmethod
    def target(frame):
        return np.where(frame.true_cause.isin(HARD_CAUSES), 'HARD_DECLINE', 'OTHER')

    def fit(self, fit_frame, calibration_frame):
        training = fit_frame.copy()
        training['true_cause'] = self.target(fit_frame)
        self.estimator.fit(training)
        self.calibrator = ProbabilityCalibrator('sigmoid', self.estimator.classes_).fit(
            probabilities(self.estimator, calibration_frame), self.target(calibration_frame))
        self.hard_index = self.estimator.classes_.index('HARD_DECLINE')
        return self

    def score(self, frame):
        return self.predict_scores(frame)[1]

    def predict_scores(self, frame):
        raw = probabilities(self.estimator, frame)
        return raw[:,self.hard_index], self.calibrator.transform(raw)[:,self.hard_index]


def apply_binary(result, hard_probability, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Binary safety threshold must be a probability')
    cause, reason, retry = [value.copy() for value in result]
    valid = np.isfinite(hard_probability) & (hard_probability >= 0) & (hard_probability <= 1)
    blocked = retry & (~valid | (hard_probability > threshold))
    cause[blocked] = 'AMBIGUOUS'
    reason[blocked] = np.where(valid[blocked], 'binary_hard_decline_veto', 'invalid_binary_safety_score')
    retry[blocked] = False
    return cause, reason, retry


def decision_metrics(data, result, control):
    values = metrics(data, result)
    truth = data['frame'].true_cause.to_numpy()
    legitimate = np.isin(truth, ['BD_INSUFFICIENT_FUNDS','TD_ISSUER_DOWN','TD_PSP_TIMEOUT'])
    previously_allowed = control[2] & legitimate
    blocked = previously_allowed & ~result[2]
    hard = np.isin(truth, HARD_CAUSES)
    values.update(
        previously_allowed_legitimate=int(previously_allowed.sum()),
        additional_legitimate_blocked=int(blocked.sum()),
        additional_legitimate_blocked_fraction=float(blocked.sum()/previously_allowed.sum()) if previously_allowed.any() else 0.,
        additional_abstention=float(np.mean(result[0]=='AMBIGUOUS')-np.mean(control[0]=='AMBIGUOUS')),
        previously_unsafe_caught=int((control[2] & hard & ~result[2]).sum()),
        common_unsafe_count=int((control[2] & hard & result[2]).sum()),
        by_hard_class={label: {'n': int((truth==label).sum()),
                              'unsafe': int(((truth==label) & result[2]).sum())} for label in HARD_CAUSES})
    return values


def evaluate_candidate(partitions, family, threshold):
    values = {}
    for name, data in partitions.items():
        result = (data['control'] if family=='control' else
                  apply_binary(data['control'],data['binary_probability'],threshold) if family=='binary' else
                  apply_policy(data, SafetyThresholds(.60,threshold)))
        values[name] = decision_metrics(data,result,data['control'])
    return values


def assess_candidate(values, reference, control):
    mean = lambda rows,key: float(np.mean([value[key] for value in rows.values()]))
    expanded_gain = mean(control,'macro_f1_all')-mean(reference,'macro_f1_all')
    gain_retained = (mean(values,'macro_f1_all')-mean(reference,'macro_f1_all'))/expanded_gain if expanded_gain>0 else None
    safe = all(value['unsafe_retry_count']==0 for value in values.values())
    utility = (gain_retained is not None and gain_retained>=MIN_GAIN_RETAINED
               and all(value['additional_legitimate_blocked_fraction']<=MAX_LEGITIMATE_BLOCKED
                       and value['additional_abstention']<=MAX_ABSTENTION_INCREASE for value in values.values()))
    return {'safe': safe, 'utility_passed': utility, 'eligible': safe and utility,
            'gain_retained': gain_retained, 'mean_f1': mean(values,'macro_f1_all'),
            'mean_accuracy': mean(values,'accuracy'),
            'mean_legitimate_retry_recall': mean(values,'true_retryable_recall'),
            'unsafe_retries': sum(value['unsafe_retry_count'] for value in values.values())}


def choose_candidate(candidates):
    eligible = [row for row in candidates if row['assessment']['eligible']]
    if not eligible:
        return None
    # At equal validated safety, retain legitimate retries; prefer the simpler
    # multiclass-only policy on ties. No confirmation results enter this choice.
    return max(eligible,key=lambda row:(row['assessment']['mean_legitimate_retry_recall'],
                                        row['assessment']['mean_f1'],row['family']!='binary'))


def confirmation_frame(family, n, seed):
    if family=='sequential':
        return generate_dataset(n=n,seed=seed,include_candidates=True).failures
    return generate_scenario(family,n=n,seed=seed,include_candidates=True).failures


def enrich_report(report, directory):
    """Paired unsafe-error overlap and calibration summaries; no policy changes."""
    directory = Path(directory)
    unsafe = {name:set() for name in ('expanded_control','multiclass_comparator','binary_comparator','selected')}
    with gzip.open(directory/'unsafe-retries.jsonl.gz','rt') as stream:
        count = 0
        for line in stream:
            row = json.loads(line)
            assert row['true_cause'] in HARD_CAUSES
            assert abs(sum(row['multiclass_probabilities'].values())-1)<1e-6
            unsafe[row['policy']].add((row['family'],row['world_seed'],row['payment_id']))
            count += 1
    assert count==report['unsafe_trace_records']
    b, m, c = (unsafe[name] for name in ('binary_comparator','multiclass_comparator','expanded_control'))
    assert b<=c and m<=c, 'A safety veto cannot introduce new allowed retries'
    report['paired_error_overlap'] = {
        'unsafe_control_payments':len(c), 'unsafe_binary_payments':len(b), 'unsafe_multiclass_payments':len(m),
        'unsafe_under_both_binary_and_multiclass':len(b&m),
        'binary_blocks_unsafe_missed_by_multiclass':len(m-b),
        'multiclass_blocks_unsafe_missed_by_binary':len(b-m),
        'binary_catches_control_unsafe':len(c-b), 'multiclass_catches_control_unsafe':len(c-m)}
    quality = {}
    world_counts, failed_counts = Counter(), Counter()
    with gzip.open(directory/'confirmation-worlds.jsonl.gz','rt') as stream:
        for line in stream:
            row = json.loads(line)
            quality.setdefault(row['family'],[]).append(row['probability_quality'])
            world_counts[row['family']] += 1
            failed_counts[row['family']] += row['policies']['expanded_control']['n']
    for family,rows in quality.items():
        assert world_counts[family]==report['confirmation'][family]['policies']['expanded_control']['worlds']
        for value in report['confirmation'][family]['policies'].values():
            value['failed_payments'] = int(failed_counts[family])
        report['confirmation'][family]['mean_probability_quality'] = {
            key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
    assert hashlib.sha256((directory/'frozen-selection.json').read_text().encode()).hexdigest()==report['frozen_selection_sha256']
    report['artifact_verification'] = {'frozen_selection_hash_matches':True,
        'independent_worlds':sum(world_counts.values()),'failed_payments':sum(failed_counts.values()),
        'worlds_by_family':dict(world_counts)}
    (directory/'artifact-verification.json').write_text(json.dumps(report['artifact_verification'],indent=2))
    summary = {'selected_family':None if report['frozen_selection']['selected'] is None else report['frozen_selection']['selected']['family'],
        'selected_threshold':None if report['frozen_selection']['selected'] is None else report['frozen_selection']['selected']['threshold'],
        'release_checks':report['release_checks'],'release_criterion_passed':report['release_criterion_passed'],
        'raw_multiclass_model_unchanged':True,'serving_changed':False,'f1_gain_retained':report['f1_gain_retained'],
        'paired_error_overlap':report['paired_error_overlap'],
        'confirmation':{family:{'hard_decline_worlds':data['hard_decline_worlds'],
            'policies':{name:{key:values[key] for key in ('hard_declines','unsafe_retries','mean_f1','mean_abstention',
                'legitimate_blocked_fraction','mean_additional_abstention','simultaneous_world_risk_upper','by_hard_class')}
                        for name,values in data['policies'].items()}} for family,data in report['confirmation'].items()}}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (directory/'retrospective-failures.json').write_text(json.dumps(report['retrospective_failures'],indent=2,allow_nan=False))
    return report


def run(n=20000, world_n=1000, max_world_risk=.05, output_dir='/tmp/binary-safety', smoke_worlds=None,
        prior_failures='docs/phase2-safety/selected-policy-failures.json', confirmation_seed_base=1000000):
    destination = Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    print('Preparing corrected fit/calibration data',flush=True)
    dataset = generate_dataset(n=n,seed=42,include_candidates=True)
    verification = verify_generator(dataset)
    split = safety_split(dataset)
    calibration = pd.concat([split['calibration'], *[
        generate_scenario(name,n=2000,seed=41000+i,include_candidates=True).failures for i,name in enumerate(SCENARIOS)]],ignore_index=True)
    model = FeatureAblationClassifier(FEATURES).fit(split['fit'])
    model.known_tuples = set(split['fit'][['gw_error_code','gw_error_source','gw_error_step']].itertuples(index=False,name=None))
    calibrator = ProbabilityCalibrator('sigmoid',model.classes_).fit(probabilities(model,calibration),calibration.true_cause)
    print('Fitting binary hard-decline objective on the same 26 inputs',flush=True)
    binary = BinarySafetyClassifier().fit(split['fit'],calibration)
    reference_model = FeatureAblationClassifier(BASELINE_FEATURES).fit(split['fit'])
    reference_model.known_tuples = model.known_tuples

    def prepare_frame(frame):
        data = prepare(frame,model,calibrator.transform(probabilities(model,frame)))
        data['control'] = apply_policy(data,MULTICLASS_POLICY)
        binary_raw, data['binary_probability'] = binary.predict_scores(frame)
        target = frame.true_cause.isin(HARD_CAUSES).to_numpy().astype(float)
        data['score_quality'] = {'multiclass_hard_brier':float(np.mean((data['hard_mass']-target)**2)),
            'binary_raw_brier':float(np.mean((binary_raw-target)**2)),
            'binary_sigmoid_brier':float(np.mean((data['binary_probability']-target)**2)),
            'multiclass_hard_ece':expected_calibration_error(data['hard_mass'],target),
            'binary_sigmoid_ece':expected_calibration_error(data['binary_probability'],target)}
        reference_data = prepare(frame,reference_model,probabilities(reference_model,frame))
        data['reference_metrics'] = metrics(reference_data,apply_policy(reference_data,REFERENCE_POLICY))
        return data

    print('Building fresh development partitions; old confirmation failures are retrospective only',flush=True)
    development = {'temporal_policy': prepare_frame(split['policy'])}
    for size in (1000,3000):
        for i,name in enumerate(SCENARIOS):
            development[f'{name}_{size}'] = prepare_frame(generate_scenario(name,n=size,seed=71000+size+i,include_candidates=True).failures)
    fit_ids, calibration_ids = set(split['fit'].payment_id), set(calibration.payment_id)
    development_ids = set().union(*(set(data['frame'].payment_id) for data in development.values()))
    assert fit_ids.isdisjoint(calibration_ids) and fit_ids.isdisjoint(development_ids) and calibration_ids.isdisjoint(development_ids)
    development_seen = fit_ids | calibration_ids | development_ids
    control = evaluate_candidate(development,'control',None)
    reference = {name:data['reference_metrics'] for name,data in development.items()}
    candidates = []
    settings = [('control',None), *[('multiclass',t) for t in (0.,.001,.003,.005,.01,.02,.03)],
                *[('binary',t) for t in (0.,.001,.003,.005,.01,.02,.03,.05,.10,.15,.25,.50,.75,1.)]]
    for family,threshold in settings:
        values = evaluate_candidate(development,family,threshold)
        candidates.append({'family':family,'threshold':threshold,'assessment':assess_candidate(values,reference,control),'partitions':values})
    chosen = choose_candidate(candidates)
    best = {}
    for family in ('multiclass','binary'):
        rows = [row for row in candidates if row['family']==family]
        best[family] = choose_candidate(rows) or max(rows,key=lambda row:(-row['assessment']['unsafe_retries'],row['assessment']['mean_legitimate_retry_recall'],row['assessment']['mean_f1']))

    # Old failures are inspected AFTER selection, never passed to the selector.
    prior = json.loads(Path(prior_failures).read_text())
    old_frame = generate_scenario('telemetry_gap',n=2000,seed=61004,include_candidates=True).failures
    old_frame = old_frame[old_frame.payment_id.isin([row['payment_id'] for row in prior])].reset_index(drop=True)
    old_data = prepare_frame(old_frame)
    if n==20000:
        previous = {row['payment_id']:row for row in prior}
        for i,row in old_frame.iterrows():
            expected = np.asarray([previous[row.payment_id]['class_probabilities'][cause] for cause in model.classes_])
            np.testing.assert_allclose(old_data['p'][i],expected,rtol=1e-8,atol=1e-10,
                                       err_msg='The expanded multiclass model must remain fixed')
    retrospective = []
    for i,row in old_frame.iterrows():
        retrospective.append({'payment_id':row.payment_id,'true_cause':row.true_cause,
            'predicted_cause':old_data['top'][i],'multiclass_hard_probability':float(old_data['hard_mass'][i]),
            'binary_hard_probability':float(old_data['binary_probability'][i]),
            'features':{key:(None if pd.isna(row[key]) else row[key]) for key in FEATURES},
            'blocked_by_validation_selected_binary_threshold':bool(old_data['binary_probability'][i]>best['binary']['threshold'])})

    required = required_zero_failure_worlds(max_risk=max_world_risk)
    target = smoke_worlds if smoke_worlds is not None else required
    seed_base = 2000000 if smoke_worlds is not None else confirmation_seed_base
    lock = {'selected':chosen,'best_multiclass_comparator':best['multiclass'],'best_binary_comparator':best['binary'],
            'required_hard_decline_worlds_per_family':required,'actual_target':target,
            'world_n':world_n,'families':list(FAMILIES),'seed_rule':f'{seed_base} + 10000 * family_index + world_index',
            'utility_limits':{'minimum_f1_gain_retained':MIN_GAIN_RETAINED,'maximum_additional_legitimate_blocked':MAX_LEGITIMATE_BLOCKED,
                              'maximum_abstention_increase':MAX_ABSTENTION_INCREASE},
            'confidence':.95,'familywise_alpha':.05,'max_world_any_unsafe_risk':max_world_risk,
            'statistical_unit':'Independent simulated world with at least one hard decline; conditionally independent given frozen models',
            'no_eligible_candidate':chosen is None}
    lock_text = json.dumps(lock,indent=2,allow_nan=False)
    (destination/'frozen-selection.json').write_text(lock_text)
    lock_hash = hashlib.sha256(lock_text.encode()).hexdigest()
    print(f'Selection frozen: {None if chosen is None else (chosen["family"],chosen["threshold"])}; target={target} hard-decline worlds per family',flush=True)
    policies = {'expanded_control':('control',None),'multiclass_comparator':('multiclass',best['multiclass']['threshold']),
                'binary_comparator':('binary',best['binary']['threshold'])}
    # An ineligible fallback is diagnostic and can never pass release.
    policies['selected'] = (chosen['family'],chosen['threshold']) if chosen else ('control',None)
    family_reports = {}
    unsafe_traces = 0
    with gzip.open(destination/'confirmation-worlds.jsonl.gz','wt') as worlds_stream, gzip.open(destination/'unsafe-retries.jsonl.gz','wt') as trace:
        for family_index,family in enumerate(FAMILIES):
            hard_worlds = 0
            summaries = {policy:[] for policy in policies}
            unsafe_worlds = Counter()
            reference_worlds = []
            for world_index in range(max(target*2,1)):
                seed = seed_base + 10000*family_index + world_index
                frame = confirmation_frame(family,world_n,seed)
                assert set(frame.payment_id).isdisjoint(development_seen), 'Confirmation overlaps model development data'
                data = prepare_frame(frame)
                hard = frame.true_cause.isin(HARD_CAUSES).to_numpy()
                hard_worlds += int(hard.any())
                reference_worlds.append(data['reference_metrics'])
                world_report = {'family':family,'seed':seed,'has_hard_declines':bool(hard.any()),
                                'probability_quality':data['score_quality'],'policies':{}}
                for name,(kind,threshold) in policies.items():
                    result = (data['control'] if kind=='control' else apply_binary(data['control'],data['binary_probability'],threshold)
                              if kind=='binary' else apply_policy(data,SafetyThresholds(.60,threshold)))
                    values = decision_metrics(data,result,data['control'])
                    summaries[name].append(values)
                    unsafe_worlds[name] += int(values['unsafe_retry_count']>0)
                    world_report['policies'][name] = values
                    for i in np.flatnonzero(hard & result[2]):
                        row = frame.iloc[i]
                        trace.write(json.dumps({'family':family,'world_seed':seed,'policy':name,'threshold':threshold,
                            'payment_id':row.payment_id,'true_cause':row.true_cause,'predicted_cause':str(result[0][i]),
                            'multiclass_probabilities':dict(zip(model.classes_,data['p'][i].tolist())),
                            'multiclass_hard_mass':float(data['hard_mass'][i]),'binary_hard_probability':float(data['binary_probability'][i]),
                            'rule_result':data['rules'][i] or None,'gate_decision':str(result[1][i])})+'\n')
                        unsafe_traces += 1
                worlds_stream.write(json.dumps(world_report,allow_nan=False)+'\n')
                if (world_index+1)%10==0 or hard_worlds>=target:
                    print(f'{family}: {hard_worlds}/{target} hard-decline worlds, selected unsafe={sum(v["unsafe_retry_count"] for v in summaries["selected"])}',flush=True)
                if hard_worlds>=target:
                    break
            def aggregate(values):
                hard_count = sum(v['hard_decline_count'] for v in values)
                unsafe_count = sum(v['unsafe_retry_count'] for v in values)
                allowed_legitimate = sum(v['previously_allowed_legitimate'] for v in values)
                blocked_legitimate = sum(v['additional_legitimate_blocked'] for v in values)
                return {'worlds':len(values),'failed_payments':sum(v['n'] for v in values),'hard_declines':hard_count,'unsafe_retries':unsafe_count,
                    'mean_f1':float(np.mean([v['macro_f1_all'] for v in values])),
                    'mean_accuracy':float(np.mean([v['accuracy'] for v in values])),
                    'mean_abstention':float(np.mean([v['abstain_rate'] for v in values])),
                    'mean_additional_abstention':float(np.mean([v['additional_abstention'] for v in values])),
                    'legitimate_blocked_fraction':blocked_legitimate/allowed_legitimate if allowed_legitimate else 0.,
                    'previously_allowed_legitimate':allowed_legitimate,'additional_legitimate_blocked':blocked_legitimate,
                    'previously_unsafe_caught':sum(v['previously_unsafe_caught'] for v in values),
                    'mean_retryable_recall':float(np.mean([v['true_retryable_recall'] for v in values])),
                    'event_iid_upper_95_descriptive_only':one_sided_upper(unsafe_count,hard_count),
                    'by_hard_class':{label:{key:sum(v['by_hard_class'][label][key] for v in values) for key in ('n','unsafe')} for label in HARD_CAUSES}}
            family_reports[family] = {'hard_decline_worlds':hard_worlds,'reference_5_mean_f1':float(np.mean([v['macro_f1_all'] for v in reference_worlds])),
                                      'policies':{name:aggregate(values) for name,values in summaries.items()}}
            for name,summary in family_reports[family]['policies'].items():
                summary.update(unsafe_worlds=int(unsafe_worlds[name]),
                    simultaneous_world_risk_upper=one_sided_upper(unsafe_worlds[name],hard_worlds,alpha=.05/len(FAMILIES)))
    mean_family = lambda name,key: float(np.mean([f['policies'][name][key] for f in family_reports.values()]))
    ref_f1 = float(np.mean([f['reference_5_mean_f1'] for f in family_reports.values()]))
    control_f1 = mean_family('expanded_control','mean_f1')
    selected_f1 = mean_family('selected','mean_f1')
    gain = (selected_f1-ref_f1)/(control_f1-ref_f1) if control_f1>ref_f1 else None
    enough = all(f['hard_decline_worlds']>=required for f in family_reports.values())
    safety = all(f['policies']['selected']['unsafe_retries']==0 for f in family_reports.values())
    bound_ok = all(f['policies']['selected']['simultaneous_world_risk_upper'] is not None and
                   f['policies']['selected']['simultaneous_world_risk_upper']<=max_world_risk for f in family_reports.values())
    utility = gain is not None and gain>=MIN_GAIN_RETAINED and all(
        f['policies']['selected']['legitimate_blocked_fraction']<=MAX_LEGITIMATE_BLOCKED and
        f['policies']['selected']['mean_additional_abstention']<=MAX_ABSTENTION_INCREASE for f in family_reports.values())
    report = {'generator_verification':verification,'training_generator':dataset.metadata,
        'environment':{'python':platform.python_version(),**{package:version(package) for package in ('numpy','pandas','scikit-learn','scipy')}},
        'development_ids_sha256':{name:hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest() for name,ids in
                                 (('fit',fit_ids),('calibration',calibration_ids),('policy',development_ids))},
        'binary_features':list(FEATURES),'hard_labels':list(HARD_CAUSES),
        'retrospective_failures':retrospective,'development_candidates':candidates,'frozen_selection_sha256':lock_hash,
        'frozen_selection':lock,'confirmation':family_reports,'confirmation_mean_f1':selected_f1,'control_mean_f1':control_f1,
        'reference_5_mean_f1':ref_f1,'f1_gain_retained':gain,'release_checks':{'eligible_on_validation':chosen is not None,
            'zero_observed_unsafe':safety,'sufficient_independent_worlds':enough,'world_risk_bounds_pass':bound_ok,'utility_pass':utility},
        'release_criterion_passed':bool(chosen is not None and safety and enough and bound_ok and utility),
        'unsafe_trace_records':unsafe_traces,'serving_changed':False,
        'limits':'All evidence is synthetic. Independent-world bounds condition on the frozen models and specified simulation family. Event-level IID bounds are descriptive only because payment attempts within a world are correlated. Failed confirmation never changes thresholds.'}
    assert hashlib.sha256((destination/'frozen-selection.json').read_text().encode()).hexdigest()==lock_hash
    expected = sum(f['policies'][name]['unsafe_retries'] for f in family_reports.values() for name in policies)
    assert unsafe_traces==expected
    report = enrich_report(report,destination)
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'Report: {destination}/report.json; release passed={report["release_criterion_passed"]}',flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=20000)
    parser.add_argument('--world-n',type=int,default=1000)
    parser.add_argument('--max-world-risk',type=float,default=.05)
    parser.add_argument('--output-dir',default='/tmp/binary-safety')
    parser.add_argument('--smoke-worlds',type=int,help='Reduced confirmation for smoke tests; cannot satisfy the default sample-size criterion')
    parser.add_argument('--prior-failures',default='docs/phase2-safety/selected-policy-failures.json')
    parser.add_argument('--confirmation-seed-base',type=int,default=1000000,help='Reserve a new seed range before any new policy confirmation')
    args = parser.parse_args()
    run(args.n,args.world_n,args.max_world_risk,args.output_dir,args.smoke_worlds,args.prior_failures,args.confirmation_seed_base)
