"""Synthetic demo dataset: 203 revenue-risk events for the live demonstration (50 checkout abandonment, 50 subscription failure, 50 overdue receivable,
50 payment failure, plus 3 edge cases).

Every entry is the exact body of `POST /revenue/events` (the controlled-input contract: dropdown values only, validated by `svc.validate_custom`) plus a
design note and the simulated customer's behaviour. Nothing here changes the model, the gate, the planner or any policy; entries are only chosen so that the
existing pipeline has something to recover, to refuse and to escalate. Model C's confidence is measured, never assigned: for every entry that will reach the
model, `build` scores it through the real decision path (`svc.decide`) under several event ids, and keeps the entries whose frozen-model answer is stable.
Deterministic entries (checkout, overdue, subscription, clear and contradictory payment failures) have no confidence.

    docker run ... python -m eval.demo_dataset build            # needs the frozen model (run in the api image); writes eval/data/demo_203/
    python -m eval.demo_dataset validate [--with-model]         # schema and business rules; --with-model re-scores the Model C entries

Run it through the system with eval/run_demo_dataset.py (API only, stdlib only). The hidden evaluation label is not used anywhere.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, routing, service as svc

SEED = 203
OUT = Path('eval/data/demo_203')
VERSION = 'demo-203@1'
PRECHECK_IDS = [f'pay_demo_precheck_{i}' for i in range(8)]          # event ids the pre-check scores under (Model C sees a calendar day derived from the id)
CONFIDENT_MIN = 0.66                                                   # an entry is kept as "expected confident" only if every pre-check score is at least this
CONSENTING = [c for c in C.CUSTOMER_IDS if c != 'cust_syn_05']        # cust_syn_05 has no contact consent: used only for the legitimate-escalation edge case
TIMES = ['t_0930', 't_1200', 't_1500', 't_1830']                      # outside quiet hours (21:00-09:00 IST); the edge case uses t_2230
SCENARIO_FIELDS = ('method', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason')
BEHAVIOUR = {'accept_first': ['ACCEPT'], 'reject_then_accept': ['REJECT', 'ACCEPT'], 'reject_twice_then_accept': ['REJECT', 'REJECT', 'ACCEPT'], 'reject_all': []}
# what the simulated customer answers to each payment request, in order; any request beyond the list is REJECTED. ACCEPT = simulated payment, never real money.


def pick(rng, weights):
    names = list(weights)
    return rng.choices(names, [weights[n] for n in names])[0]


def entry(rng, idx, scenario, body, purpose, *, route=None, cause=None, behaviour='reject_all', workflow=True, ml=False):
    return {'entry_id': f'demo203-{idx:03d}', 'scenario': scenario, 'event_type': body['event_type'], 'purpose': purpose,
            'expected': {'route': route, 'cause': cause, 'workflow': workflow, 'model_c': ml}, 'customer_behaviour': behaviour, 'request': body}


# ------------------------------------------------------------------------------------------------- deterministic scenarios
def checkout_entries(rng, start):
    out = []
    behaviours = ['accept_first'] * 28 + ['reject_then_accept'] * 12 + ['reject_all'] * 10
    rng.shuffle(behaviours)
    for i, b in enumerate(behaviours):
        body = {'event_type': C.CHECKOUT, 'customer_id': CONSENTING[i % len(CONSENTING)], 'amount_rupees': C.AMOUNTS_RUPEES[(i * 3) % len(C.AMOUNTS_RUPEES)],
                'failure_time': rng.choice(TIMES), 'checkout_stage': C.CHECKOUT_STAGES[i % 3]['id']}
        out.append(entry(rng, start + i, 'CHECKOUT_ABANDONMENT', body, f'checkout left at {body["checkout_stage"]}; the customer behaves as {b}', behaviour=b))
    return out


def subscription_entries(rng, start):
    out, i = [], 0
    quota = [('BD_INSUFFICIENT_FUNDS', 14), ('BD_USER_ABANDONED_OTP', 8), ('CARD_EXPIRED', 8), ('TD_ISSUER_DOWN', 8), ('TD_PSP_TIMEOUT', 6), ('RISK_BLOCKED', 6)]
    for kind, n in quota:
        for j in range(n):
            method = 'card' if kind == 'CARD_EXPIRED' else rng.choice(list(C.METHODS))
            body = {'event_type': C.SUBSCRIPTION, 'customer_id': CONSENTING[i % len(CONSENTING)], 'plan_id': C.SUBSCRIPTION_PLANS[i % 3]['id'], 'method': method,
                    'failure_kind': kind, 'failure_time': rng.choice(TIMES)}
            b = 'reject_all' if kind == 'RISK_BLOCKED' else pick(rng, {'accept_first': 55, 'reject_then_accept': 25, 'reject_all': 20})
            note = {'RISK_BLOCKED': 'a risk block is escalated to a person, never retried', 'TD_PSP_TIMEOUT': 'the payment state is verified before anyone is asked to pay again'}.get(kind, f'renewal failed ({kind})')
            out.append(entry(rng, start + i, 'SUBSCRIPTION_FAILURE', body, note + f'; customer: {b}', cause=kind, behaviour=b))
            i += 1
    rng.shuffle(out)
    for k, e in enumerate(out):
        e['entry_id'] = f'demo203-{start + k:03d}'
    return out


def overdue_entries(rng, start):
    out = []
    for i in range(50):
        inv = C.INVOICES[i % len(C.INVOICES)]
        pool = {'accept_first': 40, 'reject_then_accept': 24, 'reject_all': 16}
        if inv['days_overdue'] >= 30:
            pool['reject_twice_then_accept'] = 20                     # only the 30+ day plan has a third reminder
        b = pick(rng, pool)
        body = {'event_type': C.OVERDUE, 'customer_id': CONSENTING[(i * 5) % len(CONSENTING)], 'invoice_id': inv['id'], 'failure_time': rng.choice(TIMES)}
        out.append(entry(rng, start + i, 'OVERDUE_RECEIVABLE', body, f'{inv["days_overdue"]}-day overdue invoice; customer: {b}', behaviour=b))
    return out


# ------------------------------------------------------------------------------------------------- payment failure
def pf_body(rng, scenario, method, row, customer=None, issuer=None, amount=None, time=None):
    hint, code, source, step, reason = row
    return {'event_type': C.PAYMENT_FAILURE, 'customer_id': customer or rng.choice(CONSENTING), 'issuer_id': issuer or rng.choice(C.ISSUERS), 'amount_rupees': amount or rng.choice(C.AMOUNTS_RUPEES),
            'failure_time': time or rng.choice(TIMES), 'diagnosis_scenario': scenario, 'method': method, 'instrument_hint': hint, 'gw_error_code': code, 'gw_error_source': source,
            'gw_error_step': step, 'error_reason': reason}


def clear_entries(rng, start):
    out, i = [], 0
    quota = [('BD_INSUFFICIENT_FUNDS', 7), ('BD_USER_ABANDONED_OTP', 5), ('CARD_EXPIRED', 3), ('RISK_BLOCKED', 3), ('TD_ISSUER_DOWN', 2), ('TD_PSP_TIMEOUT', 1)]
    quota = [(c, n) for c, n in quota]
    for cause, n in quota:
        pool = [(m, r) for m in R.METHODS for r in sorted(R.scenarios(m, 'clear'), key=str) if R.clear_reason_cause(m, r[4]) == cause]
        for _ in range(n):
            method, row = rng.choice(pool)
            b = 'reject_all' if cause == 'RISK_BLOCKED' else pick(rng, {'accept_first': 55, 'reject_then_accept': 25, 'reject_all': 20})
            out.append(entry(rng, start + i, 'PAYMENT_FAILURE', pf_body(rng, 'clear', method, row), f'clear gateway evidence ({row[4]}): decided by rule, Model C not called; customer: {b}',
                             route=routing.CLEAR, cause=cause, behaviour=b))
            i += 1
    return out


def contradictory_entries(rng, start, n=4):
    out = []
    while len(out) < n:
        method = rng.choice(list(R.METHODS))
        info = {'method': method, 'instrument_hint': rng.choice(R.ALL_HINTS), 'gw_error_code': rng.choice(R.universe('gw_error_code')), 'gw_error_source': rng.choice(R.ALL_SOURCES),
                'gw_error_step': rng.choice(R.ALL_STEPS), 'error_reason': rng.choice(R.ALL_REASONS)}
        a = routing.assess(info)
        if a.route != routing.CONTRADICTORY:
            continue
        row = (info['instrument_hint'], info['gw_error_code'], info['gw_error_source'], info['gw_error_step'], info['error_reason'])
        out.append(entry(rng, start + len(out), 'PAYMENT_FAILURE', pf_body(rng, 'contradictory', method, row),
                         'contradictory gateway evidence: Human Intervention, Model C not called, no workflow', route=routing.CONTRADICTORY, workflow=False))
    return out


def score(body, ids=PRECHECK_IDS):
    """Frozen Model C through the real decision path (svc.decide) under several event ids. Returns [(cause, gate decision, top confidence)]."""
    got = []
    for eid in ids:
        d, _ = svc.decide(svc.custom_spec(C.PAYMENT_FAILURE, body, eid))
        mc = d['model_c']
        got.append((d['cause'], mc['decision'], mc['confidence']))
    return got


def generic_entries(rng, start, quotas, n_borderline):
    """Entries that need Model C. `quotas` = {cause: n} of entries whose answer is stable and confident under every pre-check id (gate: reliable_diagnosis);
    `n_borderline` entries that the model finds hard, kept on purpose so the set also shows abstention (they are expected to be low confidence)."""
    confident, borderline, tries = [], [], 0
    left = dict(quotas)
    while (sum(left.values()) > 0 or len(borderline) < n_borderline) and tries < 4000:
        tries += 1
        method = pick(rng, {'upi': 6, 'card': 3, 'netbanking': 1.5})
        row = rng.choice(sorted(R.scenarios(method, 'generic'), key=str))
        body = pf_body(rng, 'generic', method, row)
        quick = score(body, PRECHECK_IDS[:2])
        reliable = all(g == 'reliable_diagnosis' for _, g, _ in quick)
        if reliable and min(c for _, _, c in quick) >= CONFIDENT_MIN + 0.04 and len({c for c, _, _ in quick}) == 1 and left.get(quick[0][0], 0) > 0:
            full = score(body)
            if all(g == 'reliable_diagnosis' for _, g, _ in full) and len({c for c, _, _ in full}) == 1 and min(c for _, _, c in full) >= CONFIDENT_MIN:
                cause = full[0][0]
                confident.append((body, cause, full))
                left[cause] -= 1
        elif not reliable and len(borderline) < n_borderline:
            full = score(body)
            if sum(g != 'reliable_diagnosis' for _, g, _ in full) >= 4:
                borderline.append((body, 'AMBIGUOUS', full))
    if sum(left.values()) or len(borderline) < n_borderline:
        raise RuntimeError(f'could not fill the quotas: {left}, borderline {len(borderline)}')
    out = []
    for k, (body, cause, full) in enumerate(confident + borderline):
        sure = (body, cause, full) in confident
        e = entry(rng, start + k, 'PAYMENT_FAILURE', body, ('missing or generic evidence: the frozen Model C is asked once; expected confident' if sure
                                                         else 'missing or generic evidence the model finds hard: expected to abstain (low confidence, conservative plan)'),
                  route=routing.GENERIC, cause=cause if sure else None, behaviour=('reject_all' if cause == 'RISK_BLOCKED' else pick(rng, {'accept_first': 60, 'reject_then_accept': 25, 'reject_all': 15})) if sure else 'accept_first', ml=True)
        e['precheck'] = {'expected_confident': sure, 'scores': [[c, g, round(p, 4)] for c, g, p in full]}
        out.append(e)
    return out


def edge_entries(rng, start, borderline_body):
    funds = next(r for r in sorted(R.scenarios('upi', 'clear'), key=str) if r[4] == 'insufficient_funds' and r[3] == 'payment_debit_response' and r[2] == 'issuer_bank')
    return [
        entry(rng, start, 'EDGE_CASE', pf_body(rng, 'clear', 'upi', funds, customer='cust_syn_05', time='t_1200'),
              'no contact consent: a legitimate escalation, never a customer contact (cust_syn_05)', route=routing.CLEAR, cause='BD_INSUFFICIENT_FUNDS', behaviour='reject_all'),
        entry(rng, start + 1, 'EDGE_CASE', {'event_type': C.CHECKOUT, 'customer_id': 'cust_syn_01', 'amount_rupees': 999, 'failure_time': 't_2230', 'checkout_stage': 'payment_page'},
              'quiet hours: the 22:30 reminder waits on the simulated clock until 09:00, then the customer pays', behaviour='accept_first'),
        entry(rng, start + 2, 'EDGE_CASE', borderline_body, 'low-confidence Model C: the gate abstains and the plan verifies the payment state first', route=routing.GENERIC, behaviour='accept_first', ml=True),
    ]


# ------------------------------------------------------------------------------------------------- build / validate
def build():
    rng = random.Random(SEED)
    entries = checkout_entries(rng, 1)
    entries += subscription_entries(rng, 51)
    entries += overdue_entries(rng, 101)
    pf = clear_entries(rng, 151)
    generic = generic_entries(rng, 151 + len(pf), {'BD_INSUFFICIENT_FUNDS': 6, 'TD_ISSUER_DOWN': 4, 'TD_PSP_TIMEOUT': 3, 'BD_USER_ABANDONED_OTP': 4, 'CARD_EXPIRED': 2, 'RISK_BLOCKED': 4}, 3)
    edge_borderline = generic.pop()                                    # the third borderline entry becomes the low-confidence edge case
    pf += generic
    pf += contradictory_entries(rng, 151 + len(pf))
    for k, e in enumerate(pf):
        e['entry_id'] = f'demo203-{151 + k:03d}'
    entries += pf
    edge = edge_entries(rng, 201, edge_borderline['request'])
    edge[2]['precheck'] = edge_borderline['precheck']
    entries += edge
    assert len(entries) == 203 and len({e['entry_id'] for e in entries}) == 203
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'entries.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in entries), encoding='utf-8')
    write_csv(entries)
    manifest(entries)
    return entries


def write_csv(entries):
    cols = ['entry_id', 'scenario', 'event_type', 'customer_id', 'amount_rupees', 'failure_time', 'checkout_stage', 'plan_id', 'method', 'failure_kind', 'invoice_id', 'issuer_id',
            'diagnosis_scenario', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason', 'expected_route', 'expected_cause', 'customer_behaviour', 'purpose']
    with (OUT / 'entries.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for e in entries:
            row = {**e['request'], 'entry_id': e['entry_id'], 'scenario': e['scenario'], 'expected_route': e['expected']['route'], 'expected_cause': e['expected']['cause'],
                   'customer_behaviour': e['customer_behaviour'], 'purpose': e['purpose']}
            w.writerow({c: ('' if row.get(c) is None else row.get(c)) for c in cols})


def ml_scores(entries):
    return [e['precheck']['scores'] for e in entries if e['expected']['model_c']]


def manifest(entries):
    ml = [e for e in entries if e['expected']['model_c']]
    confs = [min(s[2] for s in e['precheck']['scores']) for e in ml]
    doc = {'version': VERSION, 'seed': SEED, 'entries': len(entries), 'by_scenario': dict(collections.Counter(e['scenario'] for e in entries)),
           'by_event_type': dict(collections.Counter(e['event_type'] for e in entries)),
           'by_expected_route_payment_failure': dict(collections.Counter(e['expected']['route'] for e in entries if e['event_type'] == C.PAYMENT_FAILURE)),
           'workflows_expected': sum(e['expected']['workflow'] for e in entries), 'model_c_entries': len(ml),
           'precheck': {'event_ids_scored_per_entry': len(PRECHECK_IDS), 'confident_threshold': CONFIDENT_MIN, 'expected_confident': sum(e['precheck']['expected_confident'] for e in ml),
                        'entries_whose_lowest_precheck_confidence_exceeds_0.6': sum(c > 0.6 for c in confs), 'share': round(sum(c > 0.6 for c in confs) / len(ml), 3)},
           'selection_note': 'Entries were SELECTED so that most Model C answers are stable and above 0.6, plus a few hard ones kept on purpose. The confidence of those entries is therefore not '
                             'representative of Model C on real traffic. Deterministic entries have no confidence. The hidden evaluation label is not used.',
           'sha256_entries': hashlib.sha256((OUT / 'entries.jsonl').read_bytes()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(doc, indent=2), encoding='utf-8')
    return doc


def load():
    return [json.loads(line) for line in (OUT / 'entries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def validate(with_model=False):
    entries = load()
    problems = []
    if len(entries) != 203 or len({e['entry_id'] for e in entries}) != 203:
        problems.append('expected 203 unique entries')
    counts = collections.Counter(e['scenario'] for e in entries)
    for want, n in (('CHECKOUT_ABANDONMENT', 50), ('SUBSCRIPTION_FAILURE', 50), ('OVERDUE_RECEIVABLE', 50), ('PAYMENT_FAILURE', 50), ('EDGE_CASE', 3)):
        if counts[want] != n:
            problems.append(f'{want}: {counts[want]} (expected {n})')
    for e in entries:
        body = {k: v for k, v in e['request'].items() if k != 'event_type'}
        errs = svc.validate_custom(e['request']['event_type'], body)
        if errs:
            problems.append(f"{e['entry_id']}: {errs}")
        if e['event_type'] == C.PAYMENT_FAILURE:
            got = routing.assess({k: e['request'].get(k) for k in routing.INFO_FIELDS}).route
            if got != e['expected']['route']:
                problems.append(f"{e['entry_id']}: router says {got}, expected {e['expected']['route']}")
        if e['customer_behaviour'] not in BEHAVIOUR:
            problems.append(f"{e['entry_id']}: unknown behaviour")
        if 'evaluation_label' in e['request']:
            problems.append(f"{e['entry_id']}: the hidden label must not be in the dataset")
        if e['expected']['model_c'] != (e['expected']['route'] == routing.GENERIC):
            problems.append(f"{e['entry_id']}: model_c flag does not match the route")
    ml = [e for e in entries if e['expected']['model_c']]
    result = {'entries': len(entries), 'model_c_entries': len(ml), 'problems': problems}
    if with_model:
        mins = []
        for e in ml:
            body = {k: v for k, v in e['request'].items() if k != 'event_type'}
            mins.append(min(c for _, _, c in score({**e['request']})) if body else 0)
        result['model_c_lowest_confidence_over_8_event_ids'] = {'above_0.6': sum(m > 0.6 for m in mins), 'of': len(mins), 'share': round(sum(m > 0.6 for m in mins) / len(mins), 3),
                                                                'median_of_lowest': round(statistics.median(mins), 3)}
        if result['model_c_lowest_confidence_over_8_event_ids']['share'] < 0.8:
            problems.append('fewer than 80% of the Model C entries stay above 0.6 under every pre-check id')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['build', 'validate'])
    ap.add_argument('--with-model', action='store_true')
    a = ap.parse_args()
    if a.command == 'build':
        built = build()
        print(json.dumps(json.loads((OUT / 'manifest.json').read_text(encoding='utf-8')), indent=2))
    else:
        res = validate(a.with_model)
        print(json.dumps(res, indent=2))
        raise SystemExit(1 if res['problems'] else 0)
