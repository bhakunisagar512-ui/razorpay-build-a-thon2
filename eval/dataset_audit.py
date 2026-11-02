"""Phase 3 dataset-realism audit: measures how determined labels are by observables.

Read-only. Run: docker compose exec -T api python - < eval/dataset_audit.py > docs/phase3-audit-metrics.json
"""
import numpy as np, pandas as pd, json
from eval.sequences import generate_dataset
from eval.scenarios import generate_scenario, SCENARIOS
def purity(df, cols):
    g=df.groupby(cols,dropna=False).true_cause
    return float(g.agg(lambda s:s.value_counts().iloc[0]).sum()/len(df)), int(g.ngroups)
out={}
d=generate_dataset(20000,42,include_candidates=True)
f=d.failures; t=d.transactions[~d.transactions.is_warmup]
out['seq']={'attempts':len(t),'failures':len(f),'fail_rate':len(f)/len(t),
 'class_dist':f.true_cause.value_counts(normalize=True).round(4).to_dict(),
 'method_dist_attempts':t.method.value_counts(normalize=True).round(3).to_dict(),
 'n_customers':int(t.customer_id.nunique()),'n_issuers':int(t.issuer_id.nunique()),'n_psp':int(t.psp_id.nunique()),
 'duration_days':3,'orders_w_multi_attempt':float((t.groupby('order_id').size()>1).mean())}
tup=['gw_error_code','gw_error_source','gw_error_step']
out['seq']['majority_ceiling']={'tuple':purity(f,tup),'tuple+method':purity(f,tup+['method']),
  'tuple+method+hint':purity(f,tup+['method','instrument_hint']),
  'baseline5':purity(f,tup+['method','instrument_hint']),
  'method+hint':purity(f,['method','instrument_hint'])}
c=f[f.method=='card'];
out['seq']['hint_vs_expired']=pd.crosstab(c.instrument_hint,c.true_cause,normalize='index').round(3).to_dict()
out['seq']['expired_hint_given_expired']=float((c[c.true_cause=='CARD_EXPIRED'].instrument_hint=='expired').mean())
cust=t.groupby('customer_id').apply(lambda x:(x.status=='failed').mean())
out['seq']['customer_fail_rate_quantiles']=cust.quantile([.1,.5,.9,.99]).round(3).to_dict()
risk=f[f.true_cause=='RISK_BLOCKED']; out['seq']['risk_methods']=risk.method.value_counts(normalize=True).round(3).to_dict()
# per-customer: does a RISK customer ever succeed?
rc=set(risk.customer_id); out['seq']['risk_customers_success_rate']=float((t[t.customer_id.isin(rc)].status=='success').mean())
out['seq']['latency_by_cause']=f.groupby('true_cause').current_latency_ms.median().round(0).to_dict()
out['seq']['issuer_fail_by_cause_corr']=f.groupby('true_cause').issuer_failure_rate_5m.mean().round(3).to_dict()
out['seq']['psp_fail_by_cause']=f.groupby('true_cause').psp_failure_rate_5m.mean().round(3).to_dict()
sc={}
for n in SCENARIOS:
    s=generate_scenario(n,4000,901,include_candidates=True).failures
    sc[n]={'n':len(s),'dist':s.true_cause.value_counts(normalize=True).round(3).to_dict(),'tuple_purity':purity(s,tup+['method','instrument_hint'])}
out['scenarios']=sc
print(json.dumps(out,indent=1,default=str))
