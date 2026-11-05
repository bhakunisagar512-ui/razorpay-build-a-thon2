"""A 200-row CSV for Dataset Testing's "Upload a CSV" (schema csv-schema@1, see GET /tests/csv-schema): synthetic payment failures only.

Dataset Testing's CSV accepts payment-failure records (payment_id, occurred_at, status, customer, issuer, amount_minor, method, gateway error fields, optional true_cause).
It is NOT the format of eval/data/demo_203/entries.csv, which holds the four event types as `POST /revenue/events` bodies.

Mix: 80 clear and consistent (rule), 100 missing or generic (frozen Model C, once each), 20 contradictory (Human Intervention). Every row has a `true_cause` (the hidden
label: only the independent simulator and the after-the-fact comparison read it; diagnosis and planning never do). For the Model C rows the label is drawn at random from the
causes that are consistent with the row, NOT from Model C's answer.

The Model C rows were SELECTED: tuples the model finds hard are swapped out (and predicted risk blocks capped) until most Model C answers are above 0.6, keeping 8 hard ones on purpose. That makes
the set useful for demonstrating the pipeline; it says nothing about Model C on real traffic. Nothing about the model, gate or planner is changed.

    docker run --rm -e PYTHONPATH=/code -v "$PWD:/code" -w /code rzp-recovery-api:latest python -m eval.csv_demo_dataset build
    docker run --rm -e PYTHONPATH=/code -v "$PWD:/code:ro" -w /code rzp-recovery-api:latest python -m eval.csv_demo_dataset verify
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.app import test_datasets as tds
from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, routing, service as svc

SEED = 200
OUT = Path('eval/data/csv_demo_200')
FILE = OUT / 'dataset_200.csv'
N = 200
MIX = {'clear': 80, 'generic': 100, 'conflict': 20}
CLEAR_QUOTA = [('BD_INSUFFICIENT_FUNDS', 22), ('BD_USER_ABANDONED_OTP', 16), ('CARD_EXPIRED', 12), ('RISK_BLOCKED', 12), ('TD_ISSUER_DOWN', 9), ('TD_PSP_TIMEOUT', 9)]
CONFIDENT_MIN, HARD_ROWS, RISK_CAP = 0.66, 8, 12
COLUMNS = ['payment_id', 'occurred_at', 'status', 'customer_id', 'issuer_id', 'amount_minor', 'method', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'observed_at',
           'error_reason', 'order_id', 'instrument_id', 'instrument_hint', 'true_cause']
AMOUNTS = [19900, 49900, 99900, 249900, 499900]
CUSTOMERS = [f'cust_csv_{i:02d}' for i in range(1, 61)]
START = datetime(2026, 10, 5, 3, 45, tzinfo=timezone.utc)


def instrument(customer, method):
    return {'card': f'card_tok_{customer}', 'upi': f'vpa_{customer}', 'netbanking': f'nb_{customer}'}[method]


def make_row(i, customer, issuer, amount, method, tuple_, cause):
    hint, code, source, step, reason = tuple_
    day, slot = divmod(i, 14)                                                      # 14 failures a day between 09:15 and 20:15 IST
    at = START + timedelta(days=day, minutes=slot * 47 + (i * 7) % 11)
    pid = f'csv200_{i + 1:03d}'
    return {'payment_id': pid, 'occurred_at': at.isoformat(), 'status': 'failed', 'customer_id': customer, 'issuer_id': issuer, 'amount_minor': amount, 'method': method,
            'gw_error_code': code, 'gw_error_source': source, 'gw_error_step': step, 'observed_at': (at + timedelta(seconds=2)).isoformat(), 'error_reason': reason or '',
            'order_id': f'ord_{pid}', 'instrument_id': instrument(customer, method), 'instrument_hint': hint, 'true_cause': cause}


def consistent_causes(method, t):
    hint, _code, source, step, reason = t
    out = [c for c, p in R.cause_profiles(method).items() if hint in p['hints'] and source in p['sources'] and step in p['steps'] and (reason is None or reason in p['reasons'])]
    return out or list(R.labels_for(method))


def draw_generic(rng):
    method = rng.choices(list(R.METHODS), [55, 30, 15])[0]
    t = rng.choice(sorted(R.scenarios(method, 'generic'), key=str))
    return method, t, rng.choice(consistent_causes(method, t))


def draw_conflict(rng):
    while True:
        method = rng.choice(list(R.METHODS))
        info = {'method': method, 'instrument_hint': rng.choice(R.ALL_HINTS), 'gw_error_code': rng.choice(R.universe('gw_error_code')), 'gw_error_source': rng.choice(R.ALL_SOURCES),
                'gw_error_step': rng.choice(R.ALL_STEPS), 'error_reason': rng.choice(R.ALL_REASONS)}
        if routing.assess(info).route == routing.CONTRADICTORY:
            return method, (info['instrument_hint'], info['gw_error_code'], info['gw_error_source'], info['gw_error_step'], info['error_reason']), rng.choice(R.labels_for(method))


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def evaluate(rows):
    """Score the rows exactly the way a Dataset Testing run does: features derived from the CSV history, router, then the frozen model for the generic route."""
    stored, tds._store = tds._store, (lambda text: 'x')                            # building must not write to the UI's uploads folder
    try:
        frame, truth, _ = tds.derive_frame(to_csv(rows))
    finally:
        tds._store = stored
    out = {}
    for rec in frame.to_dict('records'):
        info = svc._info(rec)
        a = routing.assess(info)
        got = {'route': a.route, 'cause': a.cause}
        if a.route == routing.GENERIC:
            m = routing.model_c_diagnose(rec)
            got.update(cause=m['cause'], confidence=m['confidence'], gate=m['decision'])
        out[rec['payment_id']] = got
    return out


def build():
    rng = random.Random(SEED)
    routes = ['clear'] * MIX['clear'] + ['generic'] * MIX['generic'] + ['conflict'] * MIX['conflict']
    rng.shuffle(routes)
    clear_causes = [c for c, n in CLEAR_QUOTA for _ in range(n)]
    rng.shuffle(clear_causes)
    plan, ci = [], 0
    for route in routes:
        if route == 'clear':
            cause = clear_causes[ci]
            ci += 1
            pool = [(m, r) for m in R.METHODS for r in sorted(R.scenarios(m, 'clear'), key=str) if R.clear_reason_cause(m, r[4]) == cause]
            method, t = rng.choice(pool)
            plan.append({'route': route, 'method': method, 'tuple': t, 'cause': cause})
        elif route == 'generic':
            method, t, cause = draw_generic(rng)
            plan.append({'route': route, 'method': method, 'tuple': t, 'cause': cause})
        else:
            method, t, cause = draw_conflict(rng)
            plan.append({'route': route, 'method': method, 'tuple': t, 'cause': cause})
    people = [(rng.choice(CUSTOMERS), rng.choice(C.ISSUERS), rng.choices(AMOUNTS, [20, 30, 25, 15, 10])[0]) for _ in plan]
    issuer_of = {}
    hard = set()
    for rnd in range(14):
        rows = []
        for i, (p, (cust, issuer, amount)) in enumerate(zip(plan, people)):
            rows.append(make_row(i, cust, issuer_of.setdefault(cust, issuer), amount, p['method'], p['tuple'], p['cause']))
        res = evaluate(rows)
        generic = [(i, res[rows[i]['payment_id']]) for i, p in enumerate(plan) if p['route'] == 'generic']
        conf = {i: g['confidence'] for i, g in generic}
        hard = {i for i in hard if conf[i] < 0.6}                                    # a hard row that stopped being hard is released
        for c, i in sorted((c, i) for i, c in conf.items() if c < 0.6 and i not in hard):
            if len(hard) < HARD_ROWS:
                hard.add(i)
        risk = [i for i, g in generic if i not in hard and g['cause'] == 'RISK_BLOCKED']
        swap = [i for i, g in generic if i not in hard and g['confidence'] < CONFIDENT_MIN] + risk[RISK_CAP:]      # also cap risk blocks so the set is not mostly escalations
        if not swap:
            break
        for i in swap:
            method, t, cause = draw_generic(rng)
            plan[i] = {'route': 'generic', 'method': method, 'tuple': t, 'cause': cause}
    else:
        raise RuntimeError('selection did not converge')
    OUT.mkdir(parents=True, exist_ok=True)
    FILE.write_text(to_csv(rows), encoding='utf-8')
    return verify()


def verify():
    text = FILE.read_text(encoding='utf-8')
    stored, tds._store = tds._store, (lambda t: 'x')
    try:
        check = tds.validate_csv(text)
    finally:
        tds._store = stored
    rows = list(csv.DictReader(io.StringIO(text)))
    res = evaluate(rows)
    by_route = collections.Counter(r['route'] for r in res.values())
    ml = [r for r in res.values() if 'confidence' in r]
    conf = [r['confidence'] for r in ml]
    bins = [('< 0.50', 0, .5), ('0.50 - 0.60', .5, .6), ('0.60 - 0.70', .6, .7), ('0.70 - 0.80', .7, .8), ('0.80 - 0.90', .8, .9), ('>= 0.90', .9, 1.01)]
    truth = {r['payment_id']: r['true_cause'] for r in rows}
    doc = {'file': str(FILE), 'valid': check['valid'], 'errors': check['errors'], 'warnings': check['warnings'], 'summary': check['summary'], 'rows': len(rows),
           'routes': dict(by_route), 'model_c_rows': len(ml), 'above_0.6': sum(c > .6 for c in conf), 'share_above_0.6': round(sum(c > .6 for c in conf) / len(conf), 3),
           'confidence_histogram': {n: sum(lo <= c < hi for c in conf) for n, lo, hi in bins}, 'confidence_min_median_max': [round(min(conf), 3), round(statistics.median(conf), 3), round(max(conf), 3)],
           'model_c_gate': dict(collections.Counter(r['gate'] for r in ml)), 'model_c_predicted_causes': dict(collections.Counter(r['cause'] for r in ml)),
           'model_c_matches_hidden_label': round(sum(r['cause'] == truth[pid] for pid, r in res.items() if 'confidence' in r) / len(ml), 3),
           'rule_matches_hidden_label': round(sum(r['cause'] == truth[pid] for pid, r in res.items() if r['route'] == routing.CLEAR) / by_route[routing.CLEAR], 3),
           'hidden_label_counts': dict(collections.Counter(truth.values())), 'customers': len({r['customer_id'] for r in rows}), 'issuers': len({r['issuer_id'] for r in rows}),
           'first_occurred_at': rows[0]['occurred_at'], 'last_occurred_at': rows[-1]['occurred_at']}
    (OUT / 'manifest.json').write_text(json.dumps(doc, indent=2), encoding='utf-8')
    print(json.dumps(doc, indent=2))
    return doc


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['build', 'verify'])
    a = ap.parse_args()
    build() if a.command == 'build' else verify()
