"""Observable-information audit. Exact contradictions and near matches are distinct.

No model is fitted. Replays the previously observed confirmation worlds to recover
the 26-feature snapshots, then searches for hard/retryable label conflicts.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.binary_safety import confirmation_frame
from services.app.domain.features import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

RETRYABLE = ('BD_INSUFFICIENT_FUNDS','TD_ISSUER_DOWN','TD_PSP_TIMEOUT')
HARD = ('CARD_EXPIRED','RISK_BLOCKED')


def json_value(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value,np.generic) else value


def snapshot(row):
    return {name:json_value(row[name]) for name in FEATURES}


def observable_equal(a,b):
    return all((pd.isna(a[key]) and pd.isna(b[key])) or a[key]==b[key] for key in FEATURES)


def near_tolerances(row, *, sensitivity=False):
    """Predeclared operating resolution, not a claim of intrinsic equivalence."""
    values = {}
    for name in NUMERIC_FEATURES:
        if name=='amount_minor' and sensitivity:
            values[name] = max(100.,abs(float(row[name]))*.01)
        elif name=='log_amount_minor' and sensitivity:
            values[name] = .011
        elif name in ('amount_minor','log_amount_minor','weekday_ist'):
            values[name] = 0.
        elif name.endswith('_attempts_5m') or name.endswith('_7d') and ('attempts' in name or 'successes' in name):
            values[name] = 1.
        elif 'failure_rate' in name or name=='amount_to_customer_mean_7d':
            values[name] = .05
        elif name=='customer_last_success_age_s':
            values[name] = max(300.,abs(float(row[name]))*.05) if sensitivity else 60.
        elif 'latency' in name:
            values[name] = max(10.,abs(float(row[name]))*.05) if not pd.isna(row[name]) else 10.
        elif name=='hour_ist':
            values[name] = 1.
        else:
            raise ValueError(f'No documented near-match tolerance: {name}')
    return values


def neighborhood_key(row):
    return (*[json_value(row[name]) for name in CATEGORICAL_FEATURES],
            tuple(pd.isna(row[name]) for name in NUMERIC_FEATURES))


def nearest_retryable(case, candidates, *, sensitivity=False):
    if candidates.empty:
        return None
    tolerance = near_tolerances(case,sensitivity=sensitivity)
    fixed = [name for name,limit in tolerance.items() if limit==0]
    for name in fixed:
        candidates = candidates[candidates[name].eq(case[name])]
    if candidates.empty:
        return None
    variable = [name for name,limit in tolerance.items() if limit>0 and not pd.isna(case[name])]
    differences = abs(candidates[variable].to_numpy(dtype=float)-np.array([case[name] for name in variable],dtype=float))
    distances = (differences/np.array([tolerance[name] for name in variable])).max(axis=1)
    position = int(np.argmin(distances))
    row = candidates.iloc[position]
    return {'payment_id':row.payment_id,'true_cause':row.true_cause,'family':row.world_family,
            'world_seed':int(row.world_seed),'distance':float(distances[position]),
            'within_predeclared_resolution':bool(distances[position]<=1.),'features':snapshot(row),
            'differences':{name:float(abs(row[name]-case[name])) for name in variable if row[name]!=case[name]}}


def analyze_pool(pool, failures):
    assert pool.payment_id.is_unique
    fingerprints = pd.util.hash_pandas_object(pool[list(FEATURES)],index=False)
    pool = pool.copy()
    pool['fingerprint'] = fingerprints.to_numpy()
    hard, retry = pool.true_cause.isin(HARD), pool.true_cause.isin(RETRYABLE)
    groups = pd.DataFrame({'fingerprint':pool.fingerprint,'hard':hard,'retryable':retry}).groupby('fingerprint').sum()
    possible = groups[(groups.hard>0)&(groups.retryable>0)]
    exact_conflicts = []
    for fingerprint,counts in possible.iterrows():
        rows = pool[pool.fingerprint.eq(fingerprint)]
        h = rows[rows.true_cause.isin(HARD)].iloc[0]
        r = rows[rows.true_cause.isin(RETRYABLE)].iloc[0]
        if observable_equal(h,r):  # Hash collisions never constitute evidence.
            exact_conflicts.append({'hard_payment_id':h.payment_id,'retryable_payment_id':r.payment_id,
                'hard_cause':h.true_cause,'retryable_cause':r.true_cause,
                'hard_count':int(counts.hard),'retryable_count':int(counts.retryable),
                'features':snapshot(h)})
    lookup = pool.set_index('payment_id',drop=False)
    candidate_groups = {}
    for indices in pool[retry].groupby(list(CATEGORICAL_FEATURES),dropna=False).groups.values():
        rows = pool.loc[indices]
        for mask,sub in rows.groupby([rows[name].isna() for name in NUMERIC_FEATURES],dropna=False):
            first = sub.iloc[0]
            candidate_groups[neighborhood_key(first)] = sub
    hard_groups = {}
    for indices in pool[hard].groupby(list(CATEGORICAL_FEATURES),dropna=False).groups.values():
        rows = pool.loc[indices]
        for _,sub in rows.groupby([rows[name].isna() for name in NUMERIC_FEATURES],dropna=False):
            hard_groups[neighborhood_key(sub.iloc[0])] = sub
    records = []
    for failure in failures:
        case = lookup.loc[failure['payment_id']]
        assert case.true_cause==failure['true_cause']
        same = pool[retry & pool.fingerprint.eq(case.fingerprint)]
        same = [row.payment_id for _,row in same.iterrows() if observable_equal(case,row)]
        candidates = candidate_groups.get(neighborhood_key(case),pool.iloc[:0])
        hard_candidates = hard_groups.get(neighborhood_key(case),pool.iloc[:0])
        hard_candidates = hard_candidates[hard_candidates.payment_id.ne(case.payment_id)]
        records.append({'payment_id':case.payment_id,'true_cause':case.true_cause,
            'predicted_cause':failure['predicted_cause'],'scenario':case.world_family,'world_seed':int(case.world_seed),
            'gateway_error':{name:case[name] for name in ('gw_error_code','gw_error_source','gw_error_step')},
            'features':snapshot(case),'exact_retryable_matches':same[:10],'exact_retryable_match_count':len(same),
            'nearest_retryable':nearest_retryable(case,candidates),
            'nearest_retryable_sensitivity':nearest_retryable(case,candidates,sensitivity=True),
            'nearest_other_hard_sensitivity':nearest_retryable(case,hard_candidates,sensitivity=True),
            'baseline_5_retryable_match_count':int(pool[retry][list(FEATURES[:5])].eq(case[list(FEATURES[:5])]).all(axis=1).sum())})
    grouping = group_failure_records(records)
    return {'pool_failed_payments':len(pool),'unsafe_cases':len(records),
        'exact_conflicting_cells':len(exact_conflicts),'unsafe_cases_with_exact_retryable_match':sum(bool(row['exact_retryable_match_count']) for row in records),
        'unsafe_cases_with_near_retryable_match':sum(bool(row['nearest_retryable'] and row['nearest_retryable']['within_predeclared_resolution']) for row in records),
        'unsafe_cases_with_sensitivity_near_match':sum(bool(row['nearest_retryable_sensitivity'] and row['nearest_retryable_sensitivity']['within_predeclared_resolution']) for row in records),
        'unsafe_cases_with_other_near_hard_match':sum(bool(row['nearest_other_hard_sensitivity'] and row['nearest_other_hard_sensitivity']['within_predeclared_resolution']) for row in records),
        'sensitivity_note':'Exploratory amount-aware sensitivity after the initial strict analysis: amount +/-1% or INR 1, log amount +/-0.011, success age +/-5% or 5 minutes. Other tolerances unchanged. Not used to choose a policy and not proof of irreducibility.',
        'groups':grouping,'cases':records,'exact_conflict_examples':exact_conflicts[:100],
        'limits':'Exact observed contradictions establish non-determinism of labels for this feature mapping in this synthetic sample; label noise can also cause them. Near matches alone do not prove irreducibility. No population Bayes-risk claim is made.'}


def group_failure_records(records):
    from collections import Counter
    counts = Counter((row['true_cause'],row['predicted_cause'],row['scenario'],*row['gateway_error'].values()) for row in records)
    return [{'true_cause':key[0],'predicted_cause':key[1],'scenario':key[2],
             'gw_error_code':key[3],'gw_error_source':key[4],'gw_error_step':key[5],'n':n} for key,n in sorted(counts.items())]


def run(prior_dir='/tmp/binary-safety-final',output_dir='/tmp/ambiguity-analysis',world_n=1000):
    source,destination = Path(prior_dir),Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    failures = []
    with gzip.open(source/'unsafe-retries.jsonl.gz','rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['policy']=='selected':
                failures.append(row)
    frames = []
    with gzip.open(source/'confirmation-worlds.jsonl.gz','rt') as stream:
        for i,line in enumerate(stream):
            world = json.loads(line)
            frame = confirmation_frame(world['family'],world_n,world['seed'])
            frame['world_family'],frame['world_seed'] = world['family'],world['seed']
            frames.append(frame)
            if (i+1)%50==0:
                print(f'Replayed {i+1} observed worlds for ambiguity analysis',flush=True)
    pool = pd.concat(frames,ignore_index=True)
    pool.to_pickle(destination/'observations.pkl.gz',compression='gzip')
    report = analyze_pool(pool,failures)
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:report[key] for key in ('pool_failed_payments','unsafe_cases','exact_conflicting_cells',
         'unsafe_cases_with_exact_retryable_match','unsafe_cases_with_near_retryable_match')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-dir',default='/tmp/binary-safety-final')
    parser.add_argument('--output-dir',default='/tmp/ambiguity-analysis')
    args = parser.parse_args()
    run(args.prior_dir,args.output_dir)
