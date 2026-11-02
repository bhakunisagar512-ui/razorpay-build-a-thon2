"""Phase 2.6: fixed classifiers, evidence vetoes and ambiguity abstention.

Evidence channels are explicitly hypothetical; no new classifier is trained.
Existing frozen weights are reconstructed once when a compatible cache is absent.
"""
import argparse
from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eval.binary_safety import BinarySafetyClassifier,FAMILIES,apply_binary,confirmation_frame,decision_metrics,one_sided_upper
from eval.datasets import safety_split
from eval.evidence_fixture import PROFILES,FIXTURE_CODES,make_evidence
from eval.feature_ablation import FeatureAblationClassifier
from eval.generate import generate_dataset
from eval.probability_calibration import ProbabilityCalibrator
from eval.safety_pipeline import prepare,probabilities,apply_policy
from eval.scenarios import SCENARIOS,generate_scenario
from services.app.domain.diagnosis_safety import SafetyThresholds
from services.app.domain.features import FEATURES
from services.app.domain.hard_evidence import evidence_verdict,evidence_veto,ambiguity_veto

MAX_WORLD_RISK = .05
MIN_RETAINED_LEGITIMATE = .90
MIN_RETRYABLE_RECALL = .30
POLICIES = ('binary','evidence','ambiguity')


def write_summary(report,path):
    summary = {'mode':report['frozen_policy']['mode'],'operational_release_eligible':False,
        'serving_changed':False,'eligibility_by_profile':report['eligibility_by_profile'],'profiles':{}}
    for profile in PROFILES:
        summary['profiles'][profile.name] = {}
        for policy in POLICIES:
            rows = [family['profiles'][profile.name]['policies'][policy] for family in report['confirmation'].values()]
            allowed = sum(row['previously_allowed_legitimate'] for row in rows)
            blocked = sum(row['additional_legitimate_blocked'] for row in rows)
            summary['profiles'][profile.name][policy] = {
                'unsafe_retries':sum(row['unsafe_retries'] for row in rows),'hard_declines':sum(row['hard_declines'] for row in rows),
                'mean_f1':float(np.mean([row['mean_f1'] for row in rows])),
                'mean_abstention':float(np.mean([row['mean_abstention'] for row in rows])),
                'family_balanced_true_retryable_recall':float(np.mean([row['mean_true_retryable_recall'] for row in rows])),
                'additional_legitimate_blocked_fraction':blocked/allowed if allowed else None,
                'maximum_family_world_risk_upper':max(row['world_risk_upper'] for row in rows)}
    (Path(path)/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


class FrozenModels:
    def __init__(self,cache='/tmp/phase26-models.joblib',prior_dir='docs/binary-safety'):
        files = ('eval/sequences.py','eval/scenarios.py','eval/datasets.py','eval/feature_ablation.py',
                 'eval/probability_calibration.py','services/app/domain/features.py','eval/binary_safety.py','eval/phase26.py')
        signature = hashlib.sha256(b''.join(Path(name).read_bytes() for name in files)+version('scikit-learn').encode()).hexdigest()
        path = Path(cache)
        stored = joblib.load(path) if path.exists() else None
        if stored is None or stored['signature']!=signature:
            self.reconstructed = True
            print('Reconstructing the existing frozen models; no new target, features or hyperparameters',flush=True)
            data = generate_dataset(20000,42,include_candidates=True)
            split = safety_split(data)
            calibration = pd.concat([split['calibration'], *[
                generate_scenario(name,2000,41000+i,include_candidates=True).failures for i,name in enumerate(SCENARIOS)]],ignore_index=True)
            model = FeatureAblationClassifier(FEATURES).fit(split['fit'])
            model.known_tuples = set(split['fit'][['gw_error_code','gw_error_source','gw_error_step']].itertuples(index=False,name=None))
            calibrator = ProbabilityCalibrator('sigmoid',model.classes_).fit(probabilities(model,calibration),calibration.true_cause)
            binary = BinarySafetyClassifier().fit(split['fit'],calibration)
            stored = {'signature':signature,'model':model,'calibrator':calibrator,'binary':binary}
            joblib.dump(stored,path)
        else:
            self.reconstructed = False
        self.model,self.calibrator,self.binary = (stored[name] for name in ('model','calibrator','binary'))
        self.signature = signature
        prior = json.loads((Path(prior_dir)/'retrospective-failures.json').read_text())
        old_frame = generate_scenario('telemetry_gap',2000,61004,include_candidates=True).failures
        old_frame = old_frame[old_frame.payment_id.isin([row['payment_id'] for row in prior])].reset_index(drop=True)
        prepared = self.prepare(old_frame)
        lookup = {row['payment_id']:row for row in prior}
        assert len(old_frame)==len(prior)==2
        for i,row in old_frame.iterrows():
            np.testing.assert_allclose(prepared['hard_mass'][i],lookup[row.payment_id]['multiclass_hard_probability'],rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(prepared['binary_probability'][i],lookup[row.payment_id]['binary_hard_probability'],rtol=1e-8,atol=1e-10)

    def prepare(self,frame):
        data = prepare(frame,self.model,self.calibrator.transform(probabilities(self.model,frame)))
        data['expanded'] = apply_policy(data,SafetyThresholds(.60,.03))
        data['binary_probability'] = self.binary.score(frame)
        data['baseline'] = apply_binary(data['expanded'],data['binary_probability'],.15)
        return data


def world_dataset(family,n,seed):
    return generate_dataset(n,seed,include_candidates=True) if family=='sequential' else generate_scenario(family,n,seed,include_candidates=True)


def evidence_policies(data,raw,profile,mode):
    baseline = data['baseline']
    hard_result = tuple(values.copy() for values in baseline)
    ambiguity_result = tuple(values.copy() for values in baseline)
    coverage = Counter()
    metadata = raw.set_index('payment_id',drop=False)
    for i in np.flatnonzero(baseline[2]):
        row = data['frame'].iloc[i]
        observed = metadata.loc[row.payment_id]
        items = make_evidence(observed,profile)
        verdict = evidence_verdict(items,payment_id=row.payment_id,instrument_id=observed.instrument_id,
            as_of=row.observed_at,method=row.method,issuer_id=row.issuer_id,psp_id=row.psp_id,
            issuer_code_registry=FIXTURE_CODES)
        coverage.update(considered=1,hard_evidence=int(bool(verdict['hard_cause'])),
                        valid_expiry=int(verdict['expiry_verified_valid']),matched_downtime=int(verdict['temporary_hold']))
        original = {'cause':baseline[0][i],'decision':baseline[1][i],'retry_allowed':True}
        evidence = evidence_veto(original,verdict)
        abstention = ambiguity_veto(original,method=row.method,instrument_hint=row.instrument_hint,
            error_tuple=(row.gw_error_code,row.gw_error_source,row.gw_error_step),verdict=verdict,mode=mode)
        for target,result in ((hard_result,evidence),(ambiguity_result,abstention)):
            target[0][i],target[1][i],target[2][i] = result['cause'],result['decision'],result['retry_allowed']
    assert not (hard_result[2] & ~baseline[2]).any() and not (ambiguity_result[2] & ~baseline[2]).any()
    return {'binary':baseline,'evidence':hard_result,'ambiguity':ambiguity_result},dict(coverage)


def summarize_world(data,results):
    return {name:decision_metrics(data,result,data['baseline']) for name,result in results.items()}


def run(output_dir='/tmp/phase26',world_n=1000,worlds=150,seed_base=5000000,smoke=False,
        model_cache='/tmp/phase26-models.joblib',prior_dir='docs/binary-safety'):
    path = Path(output_dir)
    path.mkdir(parents=True,exist_ok=True)
    models = FrozenModels(model_cache,prior_dir)
    print('Validating ambiguity modes on development data',flush=True)
    development = []
    # These worlds are development-only, distinct from both previous and new
    # confirmation seeds. No thresholds or modes use new confirmation labels.
    for size in (1000,3000):
        for i,family in enumerate(FAMILIES):
            dataset = world_dataset(family,size,81000+size+i)
            development.append((models.prepare(dataset.failures),dataset.transactions,f'{family}:{size}'))
    mode_results = {}
    for mode in ('collision_only','unknown_expiry'):
        values = {}
        for data,raw,name in development:
            results,_ = evidence_policies(data,raw,PROFILES[0],mode)
            values[name] = summarize_world(data,results)['ambiguity']
        mode_results[mode] = {'partitions':values,'unsafe_retries':sum(v['unsafe_retry_count'] for v in values.values()),
            'mean_legitimate_recall':float(np.mean([v['true_retryable_recall'] for v in values.values()])),
            'coverage_pass':all(v['additional_legitimate_blocked_fraction']<=1-MIN_RETAINED_LEGITIMATE for v in values.values())}
    # Safety is primary; no-evidence coverage shortfall is recorded, not hidden.
    mode = min(mode_results,key=lambda name:(mode_results[name]['unsafe_retries'],-mode_results[name]['mean_legitimate_recall']))
    seed_base = 6000000 if smoke else seed_base
    lock = {'mode':mode,'validation':mode_results,'model_signature':models.signature,
        'model_inputs':list(FEATURES),'binary_threshold':.15,'multiclass_thresholds':{'confidence':.60,'hard_mass':.03},
        'worlds_per_family':worlds,'world_n':world_n,'seed_base':seed_base,
        'max_world_any_unsafe_risk':MAX_WORLD_RISK,'familywise_confidence':.95,
        'min_retained_legitimate_retries':MIN_RETAINED_LEGITIMATE,'min_family_balanced_true_retryable_recall':MIN_RETRYABLE_RECALL,
        'primary_policy':'ambiguity','primary_evidence_profile':'no_additional_evidence',
        'evidence_profiles':[vars(profile) for profile in PROFILES],
        'minimum_planned_worlds':150,'hypothetical_evidence_only':True,'new_model_family_or_objective':False,
        'existing_models_reconstructed':models.reconstructed,'prior_failure_probabilities_reproduced':True,
        'policy_source_sha256':hashlib.sha256(Path('services/app/domain/hard_evidence.py').read_bytes()).hexdigest(),
        'fixture_source_sha256':hashlib.sha256(Path('eval/evidence_fixture.py').read_bytes()).hexdigest()}
    lock_text = json.dumps(lock,indent=2,allow_nan=False)
    (path/'frozen-policy.json').write_text(lock_text)
    lock_hash = hashlib.sha256(lock_text.encode()).hexdigest()
    print(f'Frozen ambiguity mode: {mode}; confirmation {worlds} worlds/family',flush=True)
    family_reports = {}
    trace_count = 0
    with gzip.open(path/'worlds.jsonl.gz','wt') as stream,gzip.open(path/'unsafe-retries.jsonl.gz','wt') as trace:
        for family_index,family in enumerate(FAMILIES):
            collection = {profile.name:{name:[] for name in POLICIES} for profile in PROFILES}
            unsafe_worlds = Counter()
            coverage_totals = {profile.name:Counter() for profile in PROFILES}
            hard_worlds = 0
            for index in range(worlds*2):
                seed = seed_base+10000*family_index+index
                dataset = world_dataset(family,world_n,seed)
                data = models.prepare(dataset.failures)
                hard = data['frame'].true_cause.isin(('CARD_EXPIRED','RISK_BLOCKED')).to_numpy()
                hard_worlds += int(hard.any())
                world = {'family':family,'seed':seed,'has_hard_declines':bool(hard.any()),'profiles':{}}
                for profile in PROFILES:
                    results,coverage = evidence_policies(data,dataset.transactions,profile,mode)
                    values = summarize_world(data,results)
                    world['profiles'][profile.name] = values
                    coverage_totals[profile.name].update(coverage)
                    for name,value in values.items():
                        collection[profile.name][name].append(value)
                        unsafe_worlds[(profile.name,name)] += int(value['unsafe_retry_count']>0)
                        for row_index in np.flatnonzero(hard & results[name][2]):
                            row = data['frame'].iloc[row_index]
                            trace.write(json.dumps({'family':family,'seed':seed,'profile':profile.name,'policy':name,
                                'payment_id':row.payment_id,'true_cause':row.true_cause,'predicted_cause':results[name][0][row_index],
                                'decision':str(results[name][1][row_index]),'binary_score':float(data['binary_probability'][row_index])})+'\n')
                            trace_count += 1
                stream.write(json.dumps(world,allow_nan=False)+'\n')
                if (index+1)%25==0 or hard_worlds>=worlds:
                    print(f'{family}: {hard_worlds}/{worlds} worlds',flush=True)
                if hard_worlds>=worlds:
                    break
            profiles = {}
            for profile,policies in collection.items():
                profiles[profile] = {'observed_evidence_counts':dict(coverage_totals[profile]),'policies':{}}
                for name,values in policies.items():
                    total_hard = sum(v['hard_decline_count'] for v in values)
                    unsafe = sum(v['unsafe_retry_count'] for v in values)
                    before = sum(v['previously_allowed_legitimate'] for v in values)
                    blocked = sum(v['additional_legitimate_blocked'] for v in values)
                    profiles[profile]['policies'][name] = {
                        'worlds':len(values),'hard_declines':total_hard,'unsafe_retries':unsafe,
                        'unsafe_worlds':int(unsafe_worlds[(profile,name)]),
                        'world_risk_upper':one_sided_upper(unsafe_worlds[(profile,name)],hard_worlds,alpha=.05/len(FAMILIES)),
                        'mean_f1':float(np.mean([v['macro_f1_all'] for v in values])),
                        'mean_accuracy':float(np.mean([v['accuracy'] for v in values])),
                        'mean_abstention':float(np.mean([v['abstain_rate'] for v in values])),
                        'mean_true_retryable_recall':float(np.mean([v['true_retryable_recall'] for v in values])),
                        'previously_allowed_legitimate':before,'additional_legitimate_blocked':blocked,
                        'retained_legitimate_fraction':1-blocked/before if before else None,
                        'by_hard_class':{label:{key:sum(v['by_hard_class'][label][key] for v in values) for key in ('n','unsafe')}
                                         for label in ('CARD_EXPIRED','RISK_BLOCKED')}}
            family_reports[family] = {'hard_decline_worlds':hard_worlds,'profiles':profiles}
    eligibility = {}
    for profile in PROFILES:
        eligibility[profile.name] = {}
        for policy in POLICIES:
            risks,coverage,recalls = [],[],[]
            for family,value in family_reports.items():
                result = value['profiles'][profile.name]['policies'][policy]
                risks.append(result['world_risk_upper'] is not None and result['world_risk_upper']<=MAX_WORLD_RISK)
                relative = result['retained_legitimate_fraction']
                coverage.append(relative is None or relative>=MIN_RETAINED_LEGITIMATE)
                recalls.append(result['mean_true_retryable_recall'])
            utility = all(coverage) and float(np.mean(recalls))>=MIN_RETRYABLE_RECALL
            eligibility[profile.name][policy] = {'risk_bound_pass':all(risks),'legitimate_coverage_pass':utility,
                'family_balanced_true_retryable_recall':float(np.mean(recalls)),
                'sample_size_pass':all(value['hard_decline_worlds']>=150 for value in family_reports.values()),
                'synthetic_scope_eligible':all(risks) and utility and worlds>=150}
    expected = sum(p['unsafe_retries'] for family in family_reports.values() for profile in family['profiles'].values() for p in profile['policies'].values())
    assert expected==trace_count
    assert hashlib.sha256((path/'frozen-policy.json').read_text().encode()).hexdigest()==lock_hash
    report = {'frozen_policy':lock,'frozen_policy_sha256':lock_hash,'confirmation':family_reports,
        'eligibility_by_profile':eligibility,'unsafe_trace_records':trace_count,'serving_changed':False,
        'real_provider_evidence_quality_validated':False,'operational_release_eligible':False,
        'limits':'Evidence coverage/precision are assumed what-if fixtures derived from detector state, never real provider observations or copied true_cause labels. All classifiers are the prior frozen models. Near matches do not prove irreducibility. Statistical bounds apply to fixed simulated worlds, not real payments. Auxiliary profile intervals are not a joint guarantee across all profiles/policies.'}
    (path/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    write_summary(report,path)
    print(f'Report: {path}/report.json; operational release remains blocked pending real evidence',flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='/tmp/phase26')
    parser.add_argument('--world-n',type=int,default=1000)
    parser.add_argument('--worlds',type=int,default=150)
    parser.add_argument('--seed-base',type=int,default=5000000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model-cache',default='/tmp/phase26-models.joblib')
    parser.add_argument('--prior-dir',default='docs/binary-safety')
    args = parser.parse_args()
    run(args.output_dir,args.world_n,args.worlds,args.seed_base,args.smoke,args.model_cache,args.prior_dir)
