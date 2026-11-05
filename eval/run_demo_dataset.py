"""Run the 203-entry demo dataset through the live system, using ONLY its public API (standard library, no extra install):

    1. POST /revenue/events          one entry = one event (validated, diagnosed once, planned, persisted, Temporal workflow started)
    2. POST /revenue/requests/{id}/respond   the simulated customer answers each payment request ACCEPT or REJECT, as the entry's `customer_behaviour` says
    3. GET  /revenue/events/{id}     until the workflow completes

Nothing here touches the model, the gate, the planner or the database directly. ACCEPT is a SIMULATED payment, never real money. Needs the stack up
(`docker compose ps`: api, worker, postgres, temporal) and the worker serving queue revenue-q.

    python eval/run_demo_dataset.py                       # all 203 entries, 8 at a time
    python eval/run_demo_dataset.py --limit 20            # the first 20
    python eval/run_demo_dataset.py --scenario PAYMENT_FAILURE
    python eval/run_demo_dataset.py --api http://localhost:8000 --workers 8
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data' / 'demo_203'
BEHAVIOUR = {'accept_first': ['ACCEPT'], 'reject_then_accept': ['REJECT', 'ACCEPT'], 'reject_twice_then_accept': ['REJECT', 'REJECT', 'ACCEPT'], 'reject_all': []}
API = 'http://localhost:8000'


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method, headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'null')


def run_entry(e, patience_s=150):
    started = time.time()
    status, ev = call('POST', '/revenue/events?timeout_s=120', e['request'])
    if status != 201:
        return {'entry_id': e['entry_id'], 'scenario': e['scenario'], 'error': f'HTTP {status}: {ev}'}
    eid, decisions, answered = ev['event_id'], BEHAVIOUR[e['customer_behaviour']], []
    view = ev
    while time.time() - started < patience_s:
        if view['route'] == 'CONTRADICTORY' or view['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED'):
            break
        for r in view['requests']:
            if r['status'] == 'PENDING' and r['request_id'] not in {a[0] for a in answered}:
                decision = decisions[len(answered)] if len(answered) < len(decisions) else 'REJECT'
                s, _ = call('POST', f"/revenue/requests/{r['request_id']}/respond", {'decision': decision})
                answered.append((r['request_id'], decision, s))
        time.sleep(0.4)
        view = call('GET', f'/revenue/events/{eid}')[1]
    d = view.get('diagnosis') or {}
    mc = d.get('model_c') or {}
    return {'entry_id': e['entry_id'], 'scenario': e['scenario'], 'event_type': view['event_type'], 'event_id': eid, 'route': view['route'], 'decided_by': d.get('method'),
            'cause': d.get('cause'), 'model_c_calls': d.get('model_c_calls', 0), 'confidence': mc.get('confidence'), 'gate': mc.get('decision'),
            'plan': [s['action'] for s in view['plan']['steps']], 'workflow_id': view.get('workflow_id'), 'workflow_state': view['workflow_state'], 'status': view['status'],
            'status_reason': view['status_reason'], 'recovered_minor': view['recovered_amount_minor'], 'amount_minor': view['amount_minor'], 'requests': [r['status'] for r in view['requests']],
            'answers': [a[1] for a in answered], 'seconds': round(time.time() - started, 1), 'expected': e['expected'],
            'finished': view['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED') or view['route'] == 'CONTRADICTORY'}


def histogram(values):
    bins = [('< 0.50', 0, .5), ('0.50 - 0.60', .5, .6), ('0.60 - 0.70', .6, .7), ('0.70 - 0.80', .7, .8), ('0.80 - 0.90', .8, .9), ('>= 0.90', .9, 1.01)]
    return {name: sum(lo <= v < hi for v in values) for name, lo, hi in bins}


def summarise(rows):
    ok = [r for r in rows if 'error' not in r]
    ml = [r for r in ok if r['model_c_calls'] == 1]
    conf = [r['confidence'] for r in ml if r['confidence'] is not None]
    started = [r for r in ok if r['workflow_id']]
    out = {'entries_run': len(rows), 'api_errors': [r for r in rows if 'error' in r], 'unfinished': [r['entry_id'] for r in ok if not r['finished']],
           'workflows_started': len(started), 'workflows_completed': sum(r['workflow_state'] == 'WORKFLOW_COMPLETED' for r in ok),
           'no_workflow_by_design': sum(not r['workflow_id'] for r in ok),
           'by_status': dict(collections.Counter(r['status'] for r in ok)),
           'by_scenario_status': {s: dict(collections.Counter(r['status'] for r in ok if r['scenario'] == s)) for s in sorted({r['scenario'] for r in ok})},
           'escalation_reasons': dict(collections.Counter(r['status_reason'] for r in ok if r['status'] == 'HUMAN_INTERVENTION')),
           'failure_reasons': dict(collections.Counter(r['status_reason'] for r in ok if r['status'] == 'SIMULATED_FAILED')),
           'recovery_reasons': dict(collections.Counter(('original_payment_succeeded' if r['status_reason'] == 'original_payment_succeeded' else 'simulated payment accepted') for r in ok if r['status'] == 'SIMULATED_RECOVERED')),
           'simulated_recovered_rupees': sum(r['recovered_minor'] for r in ok) / 100, 'amount_at_risk_rupees': sum(r['amount_minor'] for r in ok) / 100,
           'routes_payment_failure': dict(collections.Counter(r['route'] for r in ok if r['event_type'] == 'PAYMENT_FAILURE')),
           'model_c': {'entries': len(ml), 'calls_total': sum(r['model_c_calls'] for r in ok), 'max_calls_per_entry': max([r['model_c_calls'] for r in ok] or [0]),
                       'confidence_histogram': histogram(conf), 'above_0.6': sum(c > .6 for c in conf), 'share_above_0.6': round(sum(c > .6 for c in conf) / len(conf), 3) if conf else None,
                       'min': round(min(conf), 3) if conf else None, 'median': round(statistics.median(conf), 3) if conf else None, 'max': round(max(conf), 3) if conf else None,
                       'gate_decisions': dict(collections.Counter(r['gate'] for r in ml)), 'predicted_causes': dict(collections.Counter(r['cause'] for r in ml))},
           'deterministic_entries_with_a_confidence': sum(1 for r in ok if r['model_c_calls'] == 0 and r['confidence'] is not None),
           'route_mismatches': [r['entry_id'] for r in ok if r['expected'].get('route') and r['route'] != r['expected']['route']]}
    return out


def main():
    global API
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--api', default=API)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--scenario')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    API = a.api.rstrip('/')
    if call('GET', '/health')[0] != 200:
        sys.exit('the API is not answering; start the stack first (docker compose up -d)')
    entries = [json.loads(line) for line in (DATA / 'entries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if a.scenario:
        entries = [e for e in entries if e['scenario'] == a.scenario]
    entries = entries[:a.limit] if a.limit else entries
    # a missing worker would leave every event PENDING / NOT_STARTED: find out on the first entry before launching the rest
    probe = next((e for e in entries if e['expected']['workflow']), None)
    first = run_entry(probe, patience_s=45) if probe else None
    if first is not None and not first.get('finished'):
        sys.exit(f"the first workflow did not progress ({first.get('workflow_state')}). Is the worker running? Check: docker compose logs --tail 3 worker (expect 'worker up on queue revenue-q')")
    rows = [first] if first else []
    todo = [e for e in entries if e is not probe]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(run_entry, todo), 1):
            rows.append(row)
            if i % 25 == 0:
                print(f'  {i + len(rows[:1])}/{len(entries)} entries done', flush=True)
    summary = summarise(rows)
    summary['wall_seconds'] = round(time.time() - t0, 1)
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    (DATA / f'run_{stamp}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=str))
    print(f'\nper-entry results: {DATA / f"run_{stamp}.json"}')


if __name__ == '__main__':
    main()
