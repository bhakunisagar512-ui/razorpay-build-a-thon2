"""Synthetic revenue-events dataset of exactly 240 entries (60 checkout abandonment, 60 subscription failure, 60 overdue receivable, 60 payment failure) in the
format of eval/data/demo_203/entries.csv, ready for Dataset Testing -> "Uploaded revenue-events CSV".

Payment failures: 20 clear and consistent (rule), 35 missing or generic (frozen Model C, once each), 5 contradictory (human, Model C not called).

Model C confidence is MEASURED, never assigned: every candidate generic row is scored by the real decision path (`svc.decide`: router, cold-start feature extraction, the frozen
model, its safety gate) under the id the Dataset Testing run will give it (`entry_id`, which fixes the simulated failure day Model C sees) and under a few run clocks, and only
rows whose answer stays above 0.60 are kept. A few rows the model finds hard are kept on purpose so abstention is visible. Because the rows are selected, the confidence of this set
says nothing about Model C on real traffic, and a confident answer is not necessarily a correct one (the file carries no hidden label and the model never sees one).

    docker run --rm -e PYTHONPATH=/code -v "$PWD:/code" -w /code rzp-recovery-api:latest python -m eval.demo_240 build
    docker run --rm -e PYTHONPATH=/code -v "$PWD:/code" -w /code rzp-recovery-api:latest python -m eval.demo_240 verify
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import statistics
from pathlib import Path

from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, event_csv, routing, runs, service as svc

SEED = 240
OUT = Path('eval/data/demo_240')
CSV_FILE, REPORT = OUT / 'entries.csv', OUT / 'validation_report.md'
N_EACH = 60
PF_MIX = {routing.CLEAR: 20, routing.GENERIC: 35, routing.CONTRADICTORY: 5}
CONF_MIN = 0.60                      # the requirement: strictly greater
SAFETY_MARGIN = 0.64                 # a kept row must clear this under every pre-check clock, so a different run clock cannot push it under 0.60
HARD_ROWS = 2                        # generic rows kept although the model is unsure, so the set also shows abstention
CONSENTING = [c for c in C.CUSTOMER_IDS if c != 'cust_syn_05']        # cust_syn_05 has no contact consent; this set has no edge cases
TIMES = ['t_0930', 't_1200', 't_1500', 't_1830']
COLUMNS = ['entry_id', 'scenario', 'event_type', 'customer_id', 'amount_rupees', 'failure_time', 'checkout_stage', 'plan_id', 'method', 'failure_kind', 'invoice_id', 'issuer_id',
           'diagnosis_scenario', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason', 'expected_route', 'expected_cause', 'customer_behaviour', 'purpose']
BEHAVIOURS = {'accept_first': 55, 'reject_then_accept': 25, 'reject_all': 20}        # bookkeeping column only; Dataset Testing ignores it and uses the acceptance rate


def pick(rng, weights):
    return rng.choices(list(weights), list(weights.values()))[0]


def row(scenario, etype, body, purpose, *, route=None, cause=None, behaviour='accept_first'):
    return {'scenario': scenario, 'event_type': etype, 'request': {'event_type': etype, **body}, 'expected_route': route, 'expected_cause': cause, 'customer_behaviour': behaviour, 'purpose': purpose}


def checkout(rng):
    out = []
    for i in range(N_EACH):
        stage = C.CHECKOUT_STAGES[i % 3]['id']
        body = {'customer_id': CONSENTING[(i * 3) % len(CONSENTING)], 'amount_rupees': C.AMOUNTS_RUPEES[(i * 2 + i // 5) % len(C.AMOUNTS_RUPEES)], 'failure_time': TIMES[(i // 2) % len(TIMES)], 'checkout_stage': stage}
        out.append(row('CHECKOUT_ABANDONMENT', C.CHECKOUT, body, f'checkout left at {stage}', behaviour=pick(rng, BEHAVIOURS)))
    return out


def subscription(rng):
    quota = [('BD_INSUFFICIENT_FUNDS', 16), ('BD_USER_ABANDONED_OTP', 10), ('CARD_EXPIRED', 9), ('TD_ISSUER_DOWN', 9), ('TD_PSP_TIMEOUT', 8), ('RISK_BLOCKED', 8)]
    out, i = [], 0
    for kind, n in quota:
        for _ in range(n):
            method = 'card' if kind == 'CARD_EXPIRED' else rng.choice(list(C.METHODS))
            body = {'customer_id': CONSENTING[(i * 5) % len(CONSENTING)], 'plan_id': C.SUBSCRIPTION_PLANS[i % 3]['id'], 'method': method, 'failure_kind': kind, 'failure_time': TIMES[(i // 3) % len(TIMES)]}
            note = {'RISK_BLOCKED': 'a risk block is escalated to a person, never retried', 'TD_PSP_TIMEOUT': 'the payment state is verified before anyone is asked to pay again'}.get(kind, f'renewal failed ({kind})')
            out.append(row('SUBSCRIPTION_FAILURE', C.SUBSCRIPTION, body, note, cause=kind, behaviour='reject_all' if kind == 'RISK_BLOCKED' else pick(rng, BEHAVIOURS)))
            i += 1
    rng.shuffle(out)
    return out


def overdue(rng):
    out, seen = [], set()
    for i in range(N_EACH):
        inv = C.INVOICES[i % len(C.INVOICES)]
        k = 0
        while True:                                                      # every (customer, invoice, time) combination appears once
            customer, when = CONSENTING[(i * 7 + i // 7 + k) % len(CONSENTING)], TIMES[(i // 3 + k // len(CONSENTING)) % len(TIMES)]
            if (customer, inv['id'], when) not in seen:
                seen.add((customer, inv['id'], when))
                break
            k += 1
        body = {'customer_id': customer, 'invoice_id': inv['id'], 'failure_time': when}
        out.append(row('OVERDUE_RECEIVABLE', C.OVERDUE, body, f'{inv["days_overdue"]}-day overdue invoice', behaviour=pick(rng, {**BEHAVIOURS, 'reject_twice_then_accept': 15 if inv['days_overdue'] >= 30 else 0})))
    return out


# ------------------------------------------------------------------------------------------------- payment failures
def pf_body(rng, scenario, method, t, customer=None):
    hint, code, source, step, reason = t
    return {'customer_id': customer or rng.choice(CONSENTING), 'issuer_id': rng.choice(C.ISSUERS), 'amount_rupees': rng.choice(C.AMOUNTS_RUPEES), 'failure_time': rng.choice(TIMES),
            'diagnosis_scenario': scenario, 'method': method, 'instrument_hint': hint, 'gw_error_code': code, 'gw_error_source': source, 'gw_error_step': step, 'error_reason': reason}


def clear_rows(rng, used):
    quota = [('BD_INSUFFICIENT_FUNDS', 5), ('BD_USER_ABANDONED_OTP', 4), ('CARD_EXPIRED', 3), ('RISK_BLOCKED', 3), ('TD_ISSUER_DOWN', 3), ('TD_PSP_TIMEOUT', 2)]
    out = []
    for cause, n in quota:
        pool = [(m, t) for m in R.METHODS for t in sorted(R.scenarios(m, 'clear'), key=str) if R.clear_reason_cause(m, t[4]) == cause]
        methods = sorted({m for m, _ in pool})
        for k in range(n):
            while True:
                m = methods[k % len(methods)] if rng.random() < .6 else rng.choice(methods)
                method, t = rng.choice([p for p in pool if p[0] == m])
                if (method, t) not in used:
                    used.add((method, t))
                    break
            out.append(row('PAYMENT_FAILURE', C.PAYMENT_FAILURE, pf_body(rng, 'clear', method, t), f'clear gateway evidence ({t[4]}): decided by rule, Model C not called', route=routing.CLEAR, cause=cause,
                           behaviour='reject_all' if cause == 'RISK_BLOCKED' else pick(rng, BEHAVIOURS)))
    return out


def conflicts(rng, used):
    """Five genuinely contradictory records, one kind of conflict each (the router names the conflicting fields)."""
    kinds = [('a card-only reason on a UPI payment', lambda: ('upi', ('na', 'BAD_REQUEST_ERROR', 'customer', 'payment_authentication', 'card_expired'))),
             ('a card instrument hint on a UPI payment', lambda: ('upi', ('valid', 'BAD_REQUEST_ERROR', 'customer', 'payment_authentication', 'payment_cancelled'))),
             ('an error source that does not exist for netbanking', lambda: ('netbanking', ('na', 'BAD_REQUEST_ERROR', 'customer_psp', 'payment_authentication', 'authentication_failed'))),
             ('a reason that is never reported with GATEWAY_ERROR', lambda: ('card', ('valid', 'GATEWAY_ERROR', 'issuer_bank', 'payment_authorization', 'insufficient_funds'))),
             ('an expired card hint with an insufficient-funds reason', lambda: ('card', ('expired', 'BAD_REQUEST_ERROR', 'issuer_bank', 'payment_authorization', 'insufficient_funds')))]
    out = []
    for why, make in kinds:
        method, t = make()
        info = dict(method=method, instrument_hint=t[0], gw_error_code=t[1], gw_error_source=t[2], gw_error_step=t[3], error_reason=t[4])
        a = routing.assess(info)
        if a.route != routing.CONTRADICTORY:
            raise RuntimeError(f'{why}: the router says {a.route}')
        out.append(row('PAYMENT_FAILURE', C.PAYMENT_FAILURE, pf_body(rng, 'contradictory', method, t), f'contradictory gateway evidence ({why}): Human Intervention, Model C not called, no workflow',
                       route=routing.CONTRADICTORY))
    return out


def clocks():
    """Run clocks a Dataset Testing run may use (whole calendar cycles, see runs.sim_offset_days): Model C must give the same confidence under each."""
    return [runs.sim_offset_days(f'rv20261105T0000{i:02d}-{i:04x}') for i in (1, 2, 3)]


def score(request, entry_id, offset=0):
    """(cause, gate decision, confidence) from the real decision path for one row; Model C is asked exactly once."""
    spec = svc.custom_spec(C.PAYMENT_FAILURE, {k: v for k, v in request.items() if k != 'event_type'}, event_id=entry_id)
    if offset:
        from datetime import timedelta
        spec['sim_start'] = spec['sim_start'] + timedelta(days=offset)
    d, _ = svc.decide(spec)
    mc = d['model_c']
    return d['cause'], mc['decision'], mc['confidence'], d['model_c_calls']


def generic_rows(rng, used, first_id):
    """35 generic rows, in their FINAL positions: 33 confident under every clock (distinct input combinations, a spread of predicted causes) and HARD_ROWS the model finds hard.
    A candidate is scored under the id it will carry in the file, because that id fixes the failure day Model C sees in a Dataset Testing run."""
    want = {'BD_INSUFFICIENT_FUNDS': 8, 'BD_USER_ABANDONED_OTP': 7, 'TD_ISSUER_DOWN': 5, 'TD_PSP_TIMEOUT': 5, 'CARD_EXPIRED': 4, 'RISK_BLOCKED': 4}      # predicted causes (measured, not assumed)
    n = PF_MIX[routing.GENERIC]
    assert sum(want.values()) + HARD_ROWS == n
    hard_pos = [11, 27][:HARD_ROWS]
    free = [i for i in range(n) if i not in hard_pos]
    seen_tuples, slots, tries = set(), {}, 0
    offs = clocks()
    while len(slots) < n and tries < 8000:
        tries += 1
        method = pick(rng, {'upi': 5, 'card': 3.5, 'netbanking': 1.5})
        t = rng.choice(sorted(R.scenarios(method, 'generic'), key=str))
        if (method, t) in used or (method, t) in seen_tuples:
            continue
        body = pf_body(rng, 'generic', method, t)
        pos_c = next((i for i in free if i not in slots), None)
        if pos_c is not None and sum(want.values()) > 0:
            eid = f'demo240-{first_id + pos_c:03d}'
            cause, gate, conf, calls = score(body, eid)
            assert calls == 1
            if gate == 'reliable_diagnosis' and conf >= SAFETY_MARGIN and want.get(cause, 0) > 0:
                if all(g == 'reliable_diagnosis' and c == cause and p > CONF_MIN for c, g, p, _ in [score(body, eid, o) for o in offs]):
                    seen_tuples.add((method, t))
                    want[cause] -= 1
                    slots[pos_c] = (method, t, body)
                    continue
        pos_h = next((p for p in hard_pos if p not in slots), None)
        if pos_h is not None:
            cause, gate, conf, calls = score(body, f'demo240-{first_id + pos_h:03d}')
            if conf <= CONF_MIN:
                seen_tuples.add((method, t))
                slots[pos_h] = (method, t, body)
    if sum(want.values()) or len(slots) < n:
        raise RuntimeError(f'could not fill the generic rows: {want}, filled {len(slots)}')
    used |= seen_tuples
    return [row('PAYMENT_FAILURE', C.PAYMENT_FAILURE, slots[i][2], 'missing or generic gateway evidence: the frozen Model C is asked once', route=routing.GENERIC, behaviour=pick(rng, BEHAVIOURS)) for i in range(n)]


def build():
    rng = random.Random(SEED)
    entries = checkout(rng) + subscription(rng) + overdue(rng)
    used = set()
    pf = clear_rows(rng, used)
    pf_first = len(entries) + 1
    for i, e in enumerate(entries):
        e['entry_id'] = f'demo240-{i + 1:03d}'
    # payment-failure ids are fixed BEFORE scoring: Model C sees a failure day derived from the id, so the id decides what the Dataset Testing run will measure
    layout = pf + [None] * PF_MIX[routing.GENERIC] + conflicts(rng, used)
    generic_slots = [i for i, x in enumerate(layout) if x is None]
    gen = generic_rows(rng, used, pf_first + len(pf))
    for slot, g in zip(generic_slots, gen):
        layout[slot] = g
    for i, e in enumerate(layout):
        e['entry_id'] = f'demo240-{pf_first + i:03d}'
    entries += layout
    assert len(entries) == 240 and len({e['entry_id'] for e in entries}) == 240
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(entries)
    return entries


def write_csv(entries):
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, lineterminator='\n')
    w.writeheader()
    for e in entries:
        r = {**e['request'], 'entry_id': e['entry_id'], 'scenario': e['scenario'], 'expected_route': e['expected_route'], 'expected_cause': e['expected_cause'],
             'customer_behaviour': e['customer_behaviour'], 'purpose': e['purpose']}
        w.writerow({c: ('' if r.get(c) is None else r.get(c)) for c in COLUMNS})
    CSV_FILE.write_text(buf.getvalue(), encoding='utf-8')


def load_rows():
    return list(csv.DictReader(io.StringIO(CSV_FILE.read_text(encoding='utf-8'))))


def verify():
    text = CSV_FILE.read_text(encoding='utf-8')
    entries, errors, warnings, summary = event_csv.parse(text)           # the upload validator, without storing the file
    rows = load_rows()
    problems = list(map(str, errors))
    by_scenario = collections.Counter(r['scenario'] for r in rows)
    if len(rows) != 240 or len({r['entry_id'] for r in rows}) != 240 or by_scenario != {s: 60 for s in ('CHECKOUT_ABANDONMENT', 'SUBSCRIPTION_FAILURE', 'OVERDUE_RECEIVABLE', 'PAYMENT_FAILURE')}:
        problems.append(f'distribution: {len(rows)} rows, {dict(by_scenario)}')
    keys = [tuple(r[c] for c in COLUMNS[2:18]) for r in rows]
    dup = [k for k, n in collections.Counter(keys).items() if n > 1]
    generic_tuples = [tuple(r[c] for c in ('method', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason')) for r in rows if r['expected_route'] == routing.GENERIC]
    route_of = {}
    scored = []
    calls_before = routing.model_c_call_count()
    for r in rows:
        if r['event_type'] != C.PAYMENT_FAILURE:
            continue
        info = {k: (r[k] or None) for k in routing.INFO_FIELDS}
        route_of[r['entry_id']] = routing.assess(info).route
    for e in entries:
        if e['event_type'] != C.PAYMENT_FAILURE or route_of[e['entry_id']] != routing.GENERIC:
            continue
        per = [score(e['request'], e['entry_id'], o) for o in [0, *clocks()]]
        cause, gate, conf, calls = per[0]
        scored.append({'entry_id': e['entry_id'], 'cause': cause, 'gate': gate, 'confidence': conf, 'calls_per_decision': calls, 'min_over_clocks': min(p[2] for p in per),
                       'same_over_clocks': len({(p[0], p[1]) for p in per}) == 1, 'request': e['request']})
    pf_routes = collections.Counter(route_of.values())
    if pf_routes != PF_MIX:
        problems.append(f'payment failure routes: {dict(pf_routes)}')
    above = sum(s['confidence'] > CONF_MIN for s in scored)
    if len(scored) != 35 or above < 32:
        problems.append(f'Model C rows {len(scored)}, above 0.60: {above}')
    if any(s['calls_per_decision'] != 1 for s in scored):
        problems.append('Model C was called more than once for a row')
    if dup:
        problems.append(f'{len(dup)} duplicated input combinations')
    if len(set(generic_tuples)) != len(generic_tuples):
        problems.append('duplicate generic evidence combinations')
    report(rows, scored, pf_routes, above, problems, warnings, summary, len(dup), routing.model_c_call_count() - calls_before)
    print(json.dumps({'valid': not errors, 'rows': len(rows), 'by_scenario': dict(by_scenario), 'pf_routes': dict(pf_routes), 'model_c_rows': len(scored), 'above_0.6': above, 'problems': problems}, indent=1))
    return problems


def report(rows, scored, pf_routes, above, problems, warnings, summary, dup, model_calls):
    confs = [s['confidence'] for s in scored]
    pf = [r for r in rows if r['event_type'] == C.PAYMENT_FAILURE]
    bins = [('below 0.50', 0, .5), ('0.50 - 0.60', .5, .6), ('0.60 - 0.70', .6, .7), ('0.70 - 0.80', .7, .8), ('0.80 - 0.90', .8, .9), ('0.90 and above', .9, 1.01)]
    L = ['# demo_240 validation report', '',
         'Synthetic data only. `entries.csv` is a revenue-events CSV for Dataset Testing -> "Uploaded revenue-events CSV". Generated by `eval/demo_240.py` (seed 240); re-verify with `python -m eval.demo_240 verify`.', '',
         '## Result', '', f"**{'PASSED' if not problems else 'ISSUES FOUND'}**" + ('' if not problems else ': ' + '; '.join(problems)), '',
         '## Scenario distribution', '', '| Scenario | Entries |', '|---|---:|']
    L += [f'| {s.replace("_", " ").title()} | {n} |' for s, n in collections.Counter(r['scenario'] for r in rows).items()] + [f'| **Total** | **{len(rows)}** |', '']
    L += ['## Payment-failure routing (the existing router, applied to the file)', '', '| Route | Entries | Required |', '|---|---:|---:|']
    L += [f'| {k} | {pf_routes.get(k, 0)} | {v} |' for k, v in PF_MIX.items()] + ['']
    L += [f'Model C was invoked exactly once per Missing/Generic row ({model_calls} inferences for {len(scored)} rows in this verification pass; each row was also re-scored under 3 run clocks for the stability check, which is why the counter is higher than 35 in the verifier). The other {len(pf) - len(scored)} payment failures never reach the model.', '']
    L += ['## Model C confidence (35 missing / generic rows)', '', f'Rows with confidence strictly above 0.60: **{above} of {len(scored)} ({100 * above / len(scored):.1f}%)** (requirement: at least 32 of 35, i.e. 90%).',
          f'Median {statistics.median(confs):.3f}, minimum {min(confs):.3f}, maximum {max(confs):.3f}.', '', '| Confidence | Rows |', '|---|---:|']
    L += [f'| {n} | {sum(lo <= c < hi for c in confs)} |' for n, lo, hi in bins] + ['']
    L += ['Confidence is the frozen model\'s top class probability for that row (uncalibrated), measured through `svc.decide` with the id Dataset Testing uses for the row; it is not accuracy. '
          'The "stable" column says the cause and gate decision were identical under three other run clocks and "min" is the lowest confidence over those clocks.', '',
          '| Entry | Method | Evidence (code / source / step / reason) | Predicted cause | Gate decision | Confidence | > 0.60 | Stable | Min over clocks |', '|---|---|---|---|---|---:|:---:|:---:|---:|']
    for s in sorted(scored, key=lambda s: s['entry_id']):
        q = s['request']
        L.append(f"| {s['entry_id']} | {q['method']} | {q['gw_error_code']} / {q['gw_error_source']} / {q['gw_error_step']} / {q.get('error_reason') or 'none'} | {s['cause']} | {s['gate']} | {s['confidence']:.3f} | "
                 f"{'yes' if s['confidence'] > CONF_MIN else 'no'} | {'yes' if s['same_over_clocks'] else 'no'} | {s['min_over_clocks']:.3f} |")
    causes = collections.Counter(s['cause'] for s in scored)
    L += ['', f'Predicted causes: ' + ', '.join(f'{k} {v}' for k, v in causes.most_common()) + '.', '']
    L += ['## Variety', '', f"* Payment failures use {len({r['method'] for r in pf})} methods, {len({r['issuer_id'] for r in pf})} issuers and {len({r['gw_error_code'] for r in pf})} error codes; "
          f"{len({r['gw_error_source'] for r in pf})} error sources and {len({r['gw_error_step'] for r in pf})} steps; {len({(r['method'], r['instrument_hint'], r['gw_error_code'], r['gw_error_source'], r['gw_error_step'], r['error_reason']) for r in pf})} distinct evidence combinations among 60 rows.",
          f"* Subscription failures cover {len({r['failure_kind'] for r in rows if r['event_type'] == 'SUBSCRIPTION_FAILURE'})} causes, checkout covers 3 stages, overdue covers all {len({r['invoice_id'] for r in rows if r['event_type'] == 'OVERDUE_RECEIVABLE'})} invoices.",
          f'* Duplicated input combinations in the whole file: {dup}.', '* No edge-case scenario; no customer without contact consent (`cust_syn_05` is not used).', '']
    L += ['## Validator', '', f'`event_csv.parse` (the code behind `POST /revenue/events-csv/validate`): {"valid, " + str(summary["rows"]) + " events" if summary else "NOT valid"}.']
    L += [f'* Warning: {w}' for w in warnings]
    L += ['', '## Notes and limits', '',
          '* The rows were generated and then SELECTED: candidates the model was unsure about were dropped (two are kept on purpose to show abstention). This set is therefore not representative of Model C on real traffic.',
          '* A high confidence is not a claim of correctness: no hidden label exists in the file and none was used.',
          '* `expected_route`, `expected_cause`, `customer_behaviour` and `purpose` are bookkeeping columns (as in demo_203); Dataset Testing ignores them and uses the acceptance rate you choose.',
          '* Model C, its 43 features, thresholds, routing, the planner, safety policies, Temporal workflows and the other datasets were not changed.', '']
    REPORT.write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['build', 'verify'])
    a = ap.parse_args()
    if a.command == 'build':
        build()
        raise SystemExit(1 if verify() else 0)
    raise SystemExit(1 if verify() else 0)
