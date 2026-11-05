"""Automated Dataset Testing for the revenue pipeline (Phase 9). No manual Accept/Reject: the independent simulator answers.

Each case: receive the synthetic event -> classify its type -> (payment failures) assess information quality -> rules, Model C once, or
Human Intervention -> ONE diagnosis -> plan -> run the plan (stopping rules, simulated customer outcomes) -> record.

Rules this module keeps:
  * diagnosis and planning never see the hidden cause; only the simulator (engine.simulate) does;
  * recovery is counted ONLY when a simulated payment succeeds: a correct diagnosis is reported separately and is not revenue;
  * results are reproducible: stable case keys, deterministic simulator, no wall-clock input;
  * an optional sample (temporal_limit) is also executed through the real Temporal workflow and compared with the in-process result.
Old evaluation files (eval/data/phase4_ui/runs) are untouched; these runs are stored apart in revenue_runs/.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import Counter
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, engine as E, event_csv, fixtures, routing, service as svc
from services.app.settings import settings

MIX = 'event_type_mix'
DEMO = 'revenue_events_203'                                                              # the original 203-entry demo dataset (revenue-event format, eval/data/demo_203)
DEMO_FILE = Path(__file__).resolve().parents[3] / 'eval' / 'data' / 'demo_203' / 'entries.jsonl'
SCENARIO_ORDER = ('CHECKOUT_ABANDONMENT', 'SUBSCRIPTION_FAILURE', 'OVERDUE_RECEIVABLE', 'PAYMENT_FAILURE', 'EDGE_CASE')
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
BATCH = 25                                                                                 # workflows started per batch in 'All eligible cases' mode
_lock = threading.Lock()
_current = {'run_id': None}
_state: dict = {}


class Busy(Exception):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.run_id = run_id


def runs_dir() -> Path:
    p = Path(settings.ui_data_dir) / 'revenue_runs'
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_options() -> list:
    from services.app import test_datasets as tds
    out = [{'name': d['name'], 'available': d['available'], 'records': d.get('records'), 'description': d.get('description', ''), 'kind': 'prebuilt'} for d in tds.list_datasets()]
    out.append({'name': fixtures.NAME, 'available': True, 'records': len(fixtures.records()), 'kind': 'fixtures',
                'description': 'Controlled records for all three payment-failure routes: clear, missing or generic, and contradictory.'})
    out.append({'name': MIX, 'available': True, 'records': len(_mix_plan()), 'kind': 'fixtures',
                'description': 'A controlled mix of all four event types: checkout abandonment, subscription failure, overdue receivable and payment failure.'})
    if DEMO_FILE.exists():
        out.append({'name': DEMO, 'available': True, 'records': len(demo_entries()), 'kind': 'events',
                    'description': 'The original 203 revenue-risk events (50 checkout abandonment, 50 subscription failure, 50 overdue receivable, 50 payment failure, 3 edge cases) in their own event format, not CSV.'})
    return out


def demo_entries() -> list:
    return _load_demo(DEMO_FILE.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load_demo(_stamp) -> list:
    return [json.loads(line) for line in DEMO_FILE.read_text(encoding='utf-8').splitlines() if line.strip()]


def event_entries(dataset: str) -> list:
    """The entries of a revenue-event dataset: the built-in 203-entry dataset or an uploaded events CSV (`events:<id>`)."""
    if dataset == DEMO:
        return demo_entries()
    return event_csv.load(dataset[len('events:'):])


def entry_page(dataset: str, offset: int = 0, limit: int = 50, scenario: str | None = None) -> dict:
    """The entries of a revenue-event dataset for the console: controlled event inputs only (the built-in file holds no hidden label)."""
    entries = event_entries(dataset)
    rows = [e for e in entries if not scenario or e['scenario'] == scenario]
    items = [{'entry_id': e['entry_id'], 'scenario': e['scenario'], 'event_type': e['event_type'],
              'inputs': {k: v for k, v in e['request'].items() if v is not None and k not in ('event_type', 'evaluation_label')},
              'expected_route': (e.get('expected') or {}).get('route'),
              'purpose': re.split(r';\s*(?:the )?customer(?: behaves as|:)', e.get('purpose') or '')[0]} for e in rows[offset:offset + limit]]
    return {'dataset': dataset, 'total': len(rows), 'offset': offset, 'limit': limit, 'scenarios': dict(Counter(e['scenario'] for e in entries)), 'items': items}


def demo_entry_page(offset: int = 0, limit: int = 50, scenario: str | None = None) -> dict:
    return entry_page(DEMO, offset, limit, scenario)


def _spread(n: int, k: int | None) -> list:
    """k indexes spread evenly over n entries, so a Temporal sample of a dataset ordered by scenario still covers every scenario."""
    if not k or k >= n:
        return list(range(n))
    return [0] if k == 1 else sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


# ------------------------------------------------------------------------------------------------- the cases of a dataset
def _mix_plan() -> list:
    """(stable_key, event_type, fields) for the event-type mix, built only from catalog values."""
    out, plans, methods = [], C.SUBSCRIPTION_PLANS, C.METHODS
    cust, times = C.CUSTOMER_IDS, C.FAILURE_TIME_IDS
    for i in range(12):
        c, t = cust[i % len(cust)], times[i % len(times)]
        out.append((f'mix|chk|{i:02d}', C.CHECKOUT, {'customer_id': c, 'amount_rupees': C.AMOUNTS_RUPEES[i % 5], 'failure_time': t, 'checkout_stage': C.CHECKOUT_STAGES[i % 3]['id']}))
        method = methods[i % 3]
        kinds = [k for k in C.FAILURE_KIND_IDS if method == 'card' or k != 'CARD_EXPIRED']
        out.append((f'mix|sub|{i:02d}', C.SUBSCRIPTION, {'customer_id': c, 'plan_id': plans[i % 3]['id'], 'method': method, 'failure_kind': kinds[i % len(kinds)], 'failure_time': t}))
        out.append((f'mix|inv|{i:02d}', C.OVERDUE, {'customer_id': c, 'invoice_id': C.INVOICES[i % 5]['id'], 'failure_time': t}))
    clear = [r for r in fixtures.records() if r['expected_route'] == routing.CLEAR][:12]
    out += [(f"mix|pay|{r['payment_id']}", C.PAYMENT_FAILURE, r) for r in clear]
    return out


def stamp_acceptance(spec: dict, accept: dict, assigned: dict | None) -> None:
    spec['payload']['sim_acceptance_pct'] = accept['pct']
    spec['payload']['sim_seed'] = accept['seed']
    if assigned is not None:
        spec['payload']['sim_accepts'] = assigned['will']
        spec['payload']['sim_accept_at'] = assigned['ordinal']


def assign_acceptance(decided: list, accept: dict) -> dict:
    """{case_key: {'will': bool, 'ordinal': int}} for the ELIGIBLE cases (a payment opportunity exists after diagnosis, planning and safety validation).
    Exactly round(pct% x eligible) customers will pay, spread over the scenarios in proportion (largest remainder), chosen with random.Random(seed): the same dataset, rate and seed
    always give the same assignment. Each paying customer also gets the payment opportunity at which they pay. This depends only on the plans, never on the hidden label."""
    import random
    rng = random.Random(accept['seed'])
    groups = {}
    for key, spec, diagnosis, plan in decided:
        if diagnosis.get('route') != routing.CONTRADICTORY and any(s['payment_opportunity'] for s in plan['steps']):
            groups.setdefault(spec.get('scenario') or spec['event_type'], []).append((key, sum(s['payment_opportunity'] for s in plan['steps'])))
    total = sum(len(v) for v in groups.values())
    quota = round(accept['pct'] / 100 * total)
    exact = {g: accept['pct'] / 100 * len(v) for g, v in groups.items()}
    take = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: (exact[g] - take[g], g), reverse=True)[:max(quota - sum(take.values()), 0)]:
        take[g] += 1
    out = {}
    for g in sorted(groups):
        members = groups[g]
        chosen = set(rng.sample(range(len(members)), min(take[g], len(members))))
        for i, (key, n_opp) in enumerate(members):
            out[key] = {'will': i in chosen, 'ordinal': rng.randrange(n_opp)}
    return out


def sim_offset_days(run_id: str | None) -> int:
    """Every run lives on its own stretch of the SIMULATED calendar. Events of a dataset get stable simulated dates (from their case key), and the weekly contact cap counts a
    customer's contacts in the database across events, so re-running the same dataset would stack contacts on the same simulated week and start suppressing steps. A run therefore
    shifts its simulated clock by a whole number of 28-year (10,227 days = 1,461 weeks) and 400-year (146,097 days) Gregorian cycles: the weekday, the date, the hour and the days
    to month end are all unchanged (so Model C sees exactly the same features and the quiet-hours / weekly-cap rules see the same local times), only the absolute year differs.
    The same offset is applied in the in-process run and in Temporal, so the two stay comparable. 57 slots, chosen from the run id."""
    if not run_id:
        return 0
    slot = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) % 57
    return (slot % 3) * 10227 + (slot // 3) * 146097


def specs(dataset: str, limit: int | None, event_id=lambda i, key: None, sample: bool = False, accept: dict | None = None, offset_days: int = 0, assignment: dict | None = None):
    """(case_key, spec) for every case. `accept` ({'pct', 'seed'}) is the configured simulated-customer acceptance rate: it is stored on each event's payload, which is how the
    Temporal activity (which only knows the event) answers with the same rule as the in-process run."""
    for key, spec in _specs(dataset, limit, event_id, sample):
        if offset_days:
            spec['sim_start'] = spec['sim_start'] + timedelta(days=offset_days)
        if accept is not None:
            stamp_acceptance(spec, accept, (assignment or {}).get(key))
        yield key, spec


def _specs(dataset: str, limit: int | None, event_id=lambda i, key: None, sample: bool = False):
    """Yield (case_key, spec) for a dataset. `event_id(i, key)` may give each case its own id (Temporal sample); the simulator key stays stable."""
    from services.app import test_datasets as tds
    if dataset == fixtures.NAME:
        for i, rec in enumerate(fixtures.records()[:limit]):
            yield rec['payment_id'], svc.fixture_spec(rec, event_id(i, rec['payment_id']))
    elif dataset == MIX:
        for i, (key, etype, f) in enumerate(_mix_plan()[:limit]):
            if etype == C.PAYMENT_FAILURE:
                yield key, svc.fixture_spec(f, event_id(i, key))
                continue
            spec = svc.custom_spec(etype, f, event_id=key.replace('|', '_'))              # stable id => stable simulated clock
            spec['payload']['sim_key'] = key
            spec['event_id'] = event_id(i, key) or spec['event_id']
            yield key, spec
    elif dataset == DEMO or dataset.startswith('events:'):
        entries = event_entries(dataset)
        for j, idx in enumerate(_spread(len(entries), limit) if sample else range(min(limit or len(entries), len(entries)))):
            e = entries[idx]
            key = e['entry_id']
            spec = svc.custom_spec(e['event_type'], {k: v for k, v in e['request'].items() if k != 'event_type'}, event_id=key)       # stable id => stable simulated clock
            spec['payload']['sim_key'] = key
            spec['scenario'] = e['scenario']
            if e['event_type'] == C.PAYMENT_FAILURE and e.get('diagnosis_scenario_given') is False:
                spec['payload'].pop('diagnosis_scenario', None)               # the file did not state one: do not record a scenario it never had
            if e['event_type'] == C.PAYMENT_FAILURE and (e.get('expected') or {}).get('route'):
                spec['expected_route'] = e['expected']['route']
            spec['event_id'] = event_id(j, key) or spec['event_id']
            yield key, spec
    elif dataset.startswith('csv:'):                                              # an uploaded CSV: history rows build the features, true_cause (optional) is the hidden label
        frame, truth, _meta = tds.derive_frame(tds.load_csv_text(dataset[4:]))
        if limit:
            frame = frame.head(limit)
        for i, row in enumerate(frame.to_dict('records')):
            pid = row['payment_id']
            yield pid, svc.dataset_spec(dataset, pid, event_id(i, pid), row=row, truth=(truth or {}).get(pid))
    else:
        tds._check_name(dataset)
        frame = tds.dataset_inputs(dataset)
        truth = tds.hidden_truth(dataset)
        if limit:
            frame = frame.head(limit)
        for i, row in enumerate(frame.to_dict('records')):
            pid = row['payment_id']
            yield pid, svc.dataset_spec(dataset, pid, event_id(i, pid), row=row, truth=truth[pid]['true_cause'])


def case_result(spec: dict, diagnosis: dict, plan: dict, outcome: dict) -> dict:
    """One row of results. `hidden_cause` is reported for the evaluation only; it was never an input to diagnosis or planning."""
    hidden = spec.get('hidden_cause') or spec.get('failure_kind')
    steps = {s['idx']: s for s in plan['steps']}
    violations, verified_failed = [], False
    for log in outcome['steps']:
        step = steps[log['idx']]
        if log['status'] != 'EXECUTED':
            continue
        verified_failed = verified_failed or log.get('verified') == 'FAILED'
        if step['payment_opportunity'] and hidden == E.UNCERTAIN and not verified_failed:
            violations.append('payment_requested_while_state_uncertain')       # a misdiagnosed timeout reached a payment request without a verification
        if step['system_retry'] and hidden in HARD:
            violations.append('hard_decline_retried')
        if step['channel'] and not spec['consent'].get(step['channel']):
            violations.append('contacted_without_consent')
    if diagnosis.get('route') == routing.CONTRADICTORY and outcome['executed']:
        violations.append('automatic_action_on_contradictory_input')
    decisions = [s['decision'] for s in outcome['steps'] if s.get('decision')]
    eligible = diagnosis.get('route') != routing.CONTRADICTORY and any(s['payment_opportunity'] for s in plan['steps'])         # a payment opportunity exists after diagnosis, planning and safety validation
    return {'case': spec['payload'].get('sim_key') or spec['event_id'], 'event_type': spec['event_type'], 'route': diagnosis.get('route'), 'diagnosis_method': diagnosis['method'],
            'customer_id': spec['customer_id'], 'eligible': eligible,
            'opportunities': len(decisions), 'accepts': decisions.count('ACCEPTED'), 'rejects': decisions.count('REJECTED'),
            'inputs': {k: v for k, v in spec['payload'].items() if not k.startswith('sim_')}, 'consent': spec['consent'],
            'diagnosis': {k: diagnosis.get(k) for k in ('route', 'method', 'cause', 'rule_id', 'explanation', 'conflicts', 'model_c_calls', 'model_c')},
            'plan': {k: plan.get(k) for k in ('version', 'cause', 'rationale', 'escalate_on_exhaustion')} | {'steps': [{k: s.get(k) for k in ('idx', 'action', 'label', 'delay_s', 'channel', 'payment_opportunity', 'system_retry')} for s in plan['steps']]},
            'rule_id': diagnosis.get('rule_id'), 'cause': diagnosis.get('cause'), 'model_c_calls': diagnosis['model_c_calls'],
            'diagnosis_matches_hidden_cause': (diagnosis.get('cause') == hidden) if diagnosis.get('cause') and hidden and spec['event_type'] != C.SUBSCRIPTION else None,
            'plan_actions': [s['action'] for s in plan['steps']], 'status': outcome['status'], 'reason': outcome['reason'], 'executed': outcome['executed'],
            'suppressed': outcome['suppressed'], 'amount_minor': spec['amount_minor'], 'recovered_amount_minor': outcome['recovered_amount_minor'],
            'violations': violations, 'steps': outcome['steps'], 'expected_route': spec.get('expected_route'), 'scenario': spec.get('scenario'),
            'model_c_confidence': round(diagnosis['model_c']['confidence'], 4) if diagnosis.get('model_c') else None}     # a confidence exists only where Model C ran


def run_case(spec: dict, decided: tuple | None = None) -> dict:
    diagnosis, plan = decided if decided is not None else svc.decide(spec)
    if diagnosis.get('route') == routing.CONTRADICTORY:
        outcome = {'status': C.HUMAN, 'reason': 'contradictory_gateway_information', 'recovered_amount_minor': 0, 'steps': [], 'executed': 0, 'suppressed': 0}
    else:
        outcome = E.execute(event_id=spec['event_id'], event_type=spec['event_type'], customer_id=spec['customer_id'], amount_minor=spec['amount_minor'], plan=plan,
                            consent=spec['consent'], sim_start=spec['sim_start'], hidden_cause=spec.get('hidden_cause') or spec.get('failure_kind'),
                            week={}, sim_key=spec['payload'].get('sim_key'), accept=E.accept_params(spec['payload']))
    return case_result(spec, diagnosis, plan, outcome)


# ------------------------------------------------------------------------------------------------- metrics
def confidence_stats(values: list):
    """Distribution of Model C's reported confidence over the cases that used it (None when no case did: nothing is invented for rule-based cases)."""
    if not values:
        return None
    import statistics
    bins = (('< 0.50', 0, .5), ('0.50 - 0.60', .5, .6), ('0.60 - 0.70', .6, .7), ('0.70 - 0.80', .7, .8), ('0.80 - 0.90', .8, .9), ('>= 0.90', .9, 1.01))
    return {'cases': len(values), 'above_0.6': sum(v > .6 for v in values), 'share_above_0.6': round(sum(v > .6 for v in values) / len(values), 3), 'min': round(min(values), 3),
            'median': round(statistics.median(values), 3), 'max': round(max(values), 3), 'histogram': {name: sum(lo <= v < hi for v in values) for name, lo, hi in bins}}


def summarise(results: list, accept: dict | None = None, assigned: int | None = None) -> dict:
    n = len(results)
    recovered = [r for r in results if r['status'] == C.RECOVERED]
    by_route = {}
    for route in (*routing.ROUTES, None):
        rows = [r for r in results if r['route'] == route]
        if rows:
            by_route[route or 'NOT_A_PAYMENT_FAILURE'] = {'cases': len(rows), C.RECOVERED: sum(r['status'] == C.RECOVERED for r in rows), C.FAILED: sum(r['status'] == C.FAILED for r in rows),
                                                          C.HUMAN: sum(r['status'] == C.HUMAN for r in rows), 'model_c_calls': sum(r['model_c_calls'] for r in rows)}
    judged = [r for r in results if r['diagnosis_matches_hidden_cause'] is not None]
    return {
        'total_cases': n, 'by_event_type': dict(Counter(r['event_type'] for r in results)),
        'route_distribution': dict(Counter(r['route'] for r in results if r['route'])), 'by_route': by_route,
        'diagnosis_methods': dict(Counter(r['diagnosis_method'] for r in results)),
        'model_c_calls': sum(r['model_c_calls'] for r in results), 'max_model_c_calls_per_case': max((r['model_c_calls'] for r in results), default=0),
        'human_interventions': sum(r['status'] == C.HUMAN for r in results), 'human_intervention_reasons': dict(Counter(r['reason'] for r in results if r['status'] == C.HUMAN)),
        'safety_violations': sum(len(r['violations']) for r in results), 'safety_violation_types': dict(Counter(v for r in results for v in r['violations'])),
        C.RECOVERED: len(recovered), C.FAILED: sum(r['status'] == C.FAILED for r in results),
        'simulated_recovered_amount_minor': sum(r['recovered_amount_minor'] for r in recovered), 'total_amount_minor': sum(r['amount_minor'] for r in results),
        'simulated_recovery_percentage': round(100 * len(recovered) / n, 1) if n else 0.0,
        'diagnosis_match': {'judged': len(judged), 'matched': sum(r['diagnosis_matches_hidden_cause'] for r in judged),
                            'note': 'Diagnosis matches the hidden cause. This is NOT recovered revenue: recovery is counted only from simulated successful payments.'},
        'model_c_confidence': confidence_stats([r['model_c_confidence'] for r in results if r.get('model_c_confidence') is not None]),
        'by_scenario': _by_scenario(results), 'simulation': _simulation(results, accept, assigned),
        'note': 'All outcomes are simulated. No real payment, message or revenue was involved.'}


def _simulation(results: list, accept: dict | None, assigned: int | None = None) -> dict:
    """What the simulated customer was asked and answered, in the words Dataset Testing reports. These are different numbers: eligible cases (a payment opportunity exists),
    recovery opportunities (payment requests actually made), simulated ACCEPT / REJECT answers, and successful recoveries (an ACCEPT, or a verified original payment)."""
    eligible = [r for r in results if r['eligible']]
    accepts, rejects = sum(r['accepts'] for r in results), sum(r['rejects'] for r in results)
    recovered = [r for r in results if r['status'] == C.RECOVERED]
    return {'acceptance_rate_selected_pct': None if accept is None else accept['pct'], 'seed': None if accept is None else accept['seed'],
            'customer_behaviour': 'built-in cause-based simulator' if accept is None else 'configured acceptance rate',
            'eligible_cases': len(eligible), 'contradictory_cases': sum(r['route'] == routing.CONTRADICTORY for r in results),
            'directly_escalated_cases': sum(r['route'] != routing.CONTRADICTORY and not r['eligible'] for r in results),
            'recovery_opportunities': accepts + rejects, 'simulated_accepts': accepts, 'simulated_rejects': rejects,
            'cases_with_an_accept': sum(r['accepts'] > 0 for r in results), 'successful_recoveries': len(recovered),
            'recoveries_by_accept': sum(r['accepts'] > 0 for r in recovered), 'recoveries_by_verified_original_payment': sum(r['reason'] == 'original_payment_succeeded' for r in recovered),
            'cases_assigned_to_accept': assigned}


def _by_scenario(results: list):
    """Per-scenario results for datasets that name a scenario per case (the 203-entry dataset); None for every other dataset."""
    if not any(r.get('scenario') for r in results):
        return None
    out = {}
    for sc in sorted({r['scenario'] for r in results if r.get('scenario')}, key=lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99):
        rows = [r for r in results if r.get('scenario') == sc]
        out[sc] = {'cases': len(rows), C.RECOVERED: sum(r['status'] == C.RECOVERED for r in rows), C.FAILED: sum(r['status'] == C.FAILED for r in rows), C.HUMAN: sum(r['status'] == C.HUMAN for r in rows),
                   'workflows_expected': sum(r['route'] != routing.CONTRADICTORY for r in rows), 'model_c_calls': sum(r['model_c_calls'] for r in rows),
                   'model_c_confidence': confidence_stats([r['model_c_confidence'] for r in rows if r.get('model_c_confidence') is not None]),
                   'human_intervention_reasons': dict(Counter(r['reason'] for r in rows if r['status'] == C.HUMAN)),
                   'simulated_recovered_amount_minor': sum(r['recovered_amount_minor'] for r in rows if r['status'] == C.RECOVERED), 'total_amount_minor': sum(r['amount_minor'] for r in rows)}
    return out


# ------------------------------------------------------------------------------------------------- Temporal sample
def _terminal(state) -> bool:
    return state in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED')


async def _temporal_run(run_id: str, dataset: str, limit: int | None, count: int, all_eligible: bool, in_process: dict, accept: dict | None, tick, offset_days: int = 0, assignment: dict | None = None) -> dict:
    """Execute cases through the real workflow (the simulator answers) and compare each with its in-process result.
    count > 0 : the first/evenly-spread `count` cases, as before (contradictory ones are persisted but start no workflow).
    all_eligible : EVERY case with a payment opportunity in the evaluated set, in batches of BATCH. Contradictory and directly escalated cases are never sent to Temporal.
    The run is only complete when every started workflow has reached a terminal state; anything else is reported (timed out / failed to start), never counted as done."""
    import asyncio
    from temporalio.client import Client
    from sqlalchemy import text
    from services.app.db import engine
    out = {'mode': 'all_eligible' if all_eligible else 'sample', 'requested': 0, 'started': 0, 'completed': 0, 'running': 0, 'failed_to_start': 0, 'timed_out': 0, 'matched_in_process': 0,
           'mismatches': [], 'error': None, 'task_queue': settings.revenue_task_queue, 'details': {}, 'responses': {'ACCEPTED': 0, 'REJECTED': 0, 'EXPIRED': 0}, 'batches': 0}
    try:
        client = await Client.connect(settings.temporal_address)
        picked = []
        for i, (key, spec) in enumerate(specs(dataset, limit if all_eligible else count, event_id=lambda i, k: f'{run_id}_{i:04d}', sample=not all_eligible, accept=accept, offset_days=offset_days, assignment=assignment)):
            if all_eligible and not in_process[key]['eligible']:
                continue
            picked.append((key, spec))
        out['requested'] = sum(1 for k, _ in picked if in_process[k]['route'] != routing.CONTRADICTORY)         # workflows to run; a contradictory case is saved but has no workflow
        out['without_workflow'] = len(picked) - out['requested']
        tick(requested=out['requested'])
        source = 'fixture' if dataset in (fixtures.NAME, MIX, DEMO) else 'dataset'
        done = {}
        for b in range(0, len(picked), BATCH):
            batch, ids = picked[b:b + BATCH], {}
            out['batches'] += 1
            for key, spec in batch:
                try:
                    await asyncio.to_thread(svc.ingest, engine(), spec, source, run_id)
                    ids[key] = spec['event_id']
                    if in_process[key]['route'] != routing.CONTRADICTORY:
                        await svc.start_workflow(spec['event_id'], responder='simulator', client=client)
                        out['started'] += 1
                        tick(started=out['started'])
                except Exception as exc:                                                            # noqa: BLE001
                    out['failed_to_start'] += 1
                    tick(failed=out['failed_to_start'])
                    out['details'][key] = {'started': False, 'error': f'{type(exc).__name__}: {exc}'[:200]}
                    ids.pop(key, None)
            deadline, finished = time.time() + 60 + 3 * len(batch), {}
            while time.time() < deadline and len(finished) < len(ids):
                with engine().begin() as conn:
                    for key, eid in ids.items():
                        if key in finished:
                            continue
                        r = conn.execute(text("SELECT workflow_state FROM revenue_event WHERE event_id=:e"), {'e': eid}).scalar()
                        if _terminal(r):
                            finished[key] = eid
                out['completed'] = len(done) + len(finished)
                tick(completed=out['completed'])
                if len(finished) < len(ids):
                    await asyncio.sleep(1)
            from services.app.revenue import store
            with engine().begin() as conn:
                for key, eid in ids.items():
                    view = store.event_view(conn, eid)
                    ran = in_process[key]['route'] != routing.CONTRADICTORY
                    d = {'started': ran, 'event_id': eid, 'workflow_id': view['workflow_id'] if ran else None, 'workflow_state': view['workflow_state'], 'status': view['status'],
                         'status_reason': view['status_reason'], 'recovered_amount_minor': view['recovered_amount_minor'],
                         'finished': key in finished or not ran,
                         'requests': [{k: r[k] for k in ('step_idx', 'title', 'status', 'decided_by')} for r in view['requests']],
                         'steps': [{k: s.get(k) for k in ('idx', 'action', 'label', 'status', 'guard', 'channel', 'sim_at')} for s in view['plan']['steps']]}
                    out['details'][key] = d
                    if d['finished']:
                        done[key] = d
                        for r in d['requests']:
                            out['responses'][r['status']] = out['responses'].get(r['status'], 0) + 1
                    else:
                        out['timed_out'] += 1
            if ids and not finished and any(in_process[k]['route'] != routing.CONTRADICTORY for k in ids):
                out['error'] = f'no workflow finished in batch {out["batches"]}; is a worker running on queue {settings.revenue_task_queue}? The remaining cases were not started.'
                break
        out['completed'] = sum(1 for k, d in done.items() if in_process[k]['route'] != routing.CONTRADICTORY)
        out['compared'] = len(done)
        out['running'] = out['timed_out']
        for key, d in done.items():
            mine = in_process[key]
            if (d['status'], d['recovered_amount_minor']) == (mine['status'], mine['recovered_amount_minor']):
                out['matched_in_process'] += 1
            else:
                out['mismatches'].append({'case': key, 'temporal': [d['status'], d['recovered_amount_minor']], 'in_process': [mine['status'], mine['recovered_amount_minor']]})
        if out['timed_out'] and not out['error']:
            out['error'] = f"{out['timed_out']} workflow(s) did not finish in time; they may still be running (check Temporal on queue {settings.revenue_task_queue})."
        if out['failed_to_start'] and not out['error']:
            out['error'] = f"{out['failed_to_start']} workflow(s) could not be started."
    except Exception as exc:                                                                            # noqa: BLE001
        out['error'] = f'{type(exc).__name__}: {exc}'[:300]
    return out


# ------------------------------------------------------------------------------------------------- lifecycle
def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, default=str), encoding='utf-8')
    tmp.replace(path)


def start(dataset: str, limit: int | None = None, temporal_limit: int = 0, accept_pct: float | None = None, accept_seed: int = 1, temporal_all: bool = False) -> str:
    if dataset.startswith('events:'):
        try:
            event_csv.load(dataset[len('events:'):])
        except (OSError, ValueError, KeyError):
            raise KeyError('unknown events CSV upload')
    elif dataset.startswith('csv:'):
        from services.app import test_datasets as tds
        try:
            tds.load_csv_text(dataset[4:])
        except (OSError, ValueError, KeyError):
            raise KeyError('unknown CSV upload')
    else:
        names = {d['name']: d for d in dataset_options()}
        if dataset not in names:
            raise KeyError(f'unknown dataset {dataset!r}')
        if not names[dataset]['available']:
            raise FileNotFoundError(dataset)
    if not _lock.acquire(blocking=False):
        raise Busy(_current['run_id'])
    run_id = 'rv' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:4]
    _current['run_id'] = run_id
    accept = None if accept_pct is None else {'pct': float(accept_pct), 'seed': int(accept_seed)}
    _state[run_id] = {'run_id': run_id, 'dataset': dataset, 'status': 'queued', 'limit': limit, 'temporal_limit': temporal_limit, 'temporal_all': temporal_all, 'acceptance': accept, 'started_at': time.time(),
                      'progress': {'phase': 'queued', 'total': 0, 'evaluated': 0, 'eligible': 0, 'workflows_requested': 0, 'workflows_started': 0, 'workflows_completed': 0}}
    _write(runs_dir() / f'{run_id}.status.json', _state[run_id])
    threading.Thread(target=_guarded, args=(run_id, dataset, limit, temporal_limit, accept, temporal_all), daemon=True, name=f'rv-{run_id}').start()
    return run_id


def _guarded(run_id, dataset, limit, temporal_limit, accept=None, temporal_all=False):
    try:
        execute(run_id, dataset, limit, temporal_limit, accept, temporal_all)
    except Exception as exc:                                                            # noqa: BLE001
        _state[run_id].update(status='failed', error=f'{type(exc).__name__}: {exc}'[:300], finished_at=time.time())
        _write(runs_dir() / f'{run_id}.status.json', _state[run_id])
    finally:
        _current['run_id'] = None
        _lock.release()


_PROGRESS_KEYS = {'requested': 'workflows_requested', 'started': 'workflows_started', 'completed': 'workflows_completed', 'failed': 'workflows_failed'}


def _total(dataset: str, limit: int | None) -> int:
    try:
        n = {fixtures.NAME: len(fixtures.records()), MIX: len(_mix_plan())}.get(dataset)
        if n is None and (dataset == DEMO or dataset.startswith('events:')):
            n = len(event_entries(dataset))
        return min(n, limit) if n and limit else (n or 0)
    except Exception:                                                                       # noqa: BLE001  progress is informative only
        return 0


def execute(run_id: str, dataset: str, limit: int | None, temporal_limit: int = 0, accept: dict | None = None, temporal_all: bool = False) -> dict:
    st = _state.setdefault(run_id, {'run_id': run_id, 'dataset': dataset, 'started_at': time.time(), 'progress': {}})
    prog = st.setdefault('progress', {})
    st.update(status='running')
    last = [0.0]

    def tick(force=False, **kw):
        prog.update(kw)
        if force or time.time() - last[0] > 1:
            last[0] = time.time()
            _write(runs_dir() / f'{run_id}.status.json', st)

    tick(True, phase='evaluating', total=_total(dataset, limit), evaluated=0, eligible=0)
    results = []
    offset = sim_offset_days(run_id)
    decided = []
    for key, spec in specs(dataset, limit, offset_days=offset):
        diagnosis, plan = svc.decide(spec)                                                 # ONE diagnosis and plan per case; the run below reuses them (Model C is not asked again)
        decided.append((key, spec, diagnosis, plan))
        tick(evaluated=len(decided), total=max(prog.get('total', 0), len(decided)))
    assignment = assign_acceptance(decided, accept) if accept is not None else None
    for key, spec, diagnosis, plan in decided:
        if accept is not None:
            stamp_acceptance(spec, accept, (assignment or {}).get(key))
        row = run_case(spec, (diagnosis, plan))
        row['case'] = key                                                                  # the case key every consumer (and the Temporal run) looks results up by
        results.append(row)
    tick(eligible=sum(r['eligible'] for r in results))
    by_key = {r['case']: r for r in results}
    tick(True, phase='evaluated', evaluated=len(results), total=len(results))
    doc = {'run_id': run_id, 'dataset': dataset, 'limit': limit, 'mode': 'automated', 'started_at': st['started_at'], 'acceptance': accept, 'sim_clock_offset_days': offset,
           'label': 'Automated simulation: an independent simulator answers every payment opportunity. Nothing here is real revenue.', 'metrics': summarise(results, accept, sum(a['will'] for a in (assignment or {}).values()) if accept is not None else None)}
    status = 'completed'
    if temporal_all or temporal_limit:
        import asyncio
        tick(True, phase='temporal')
        doc['temporal'] = asyncio.run(_temporal_run(run_id, dataset, limit, min(temporal_limit, len(results)), bool(temporal_all), by_key, accept,
                                                    lambda **kw: tick(**{_PROGRESS_KEYS[k]: v for k, v in kw.items()}), offset, assignment))
        t = doc['temporal']
        details = t.pop('details')
        for r in results:
            if r['case'] in details:
                r['temporal'] = details[r['case']]
        if t['error'] or t['timed_out'] or t['failed_to_start']:
            status = 'partial'                                                              # never reported as complete while a required workflow is unfinished or failed
    doc.update(status=status, finished_at=time.time())
    tick(True, phase='done', workflows_completed=(doc.get('temporal') or {}).get('completed', 0))
    _write(runs_dir() / f'{run_id}.json', {**doc, 'cases': results})
    st.update(status=status, finished_at=doc['finished_at'])
    _write(runs_dir() / f'{run_id}.status.json', {**st, **{k: doc[k] for k in ('metrics',)}})
    return doc


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def get_run(run_id: str):
    import re
    if not re.fullmatch(r'rv[0-9T]+-[0-9a-f]{4}', run_id):
        return None
    full = _read(runs_dir() / f'{run_id}.json')
    if full:
        return {k: v for k, v in full.items() if k != 'cases'}
    live = _state.get(run_id) or _read(runs_dir() / f'{run_id}.status.json')
    if live and live.get('status') in ('queued', 'running') and run_id not in _state:
        live = {**live, 'status': 'interrupted'}                                           # the API restarted mid-run
    return live


def list_runs(limit: int = 20) -> list:
    files = sorted(runs_dir().glob('rv*.status.json'), reverse=True)[:limit]
    return [{k: d.get(k) for k in ('run_id', 'dataset', 'status', 'started_at', 'finished_at', 'limit')} for d in (_read(f) for f in files) if d]


def cases(run_id: str, route=None, status=None, offset=0, limit=50, scenario=None):
    full = _read(runs_dir() / f'{run_id}.json') if get_run(run_id) else None
    if not full:
        return None
    rows = [r for r in full['cases'] if (not route or r['route'] == route) and (not status or r['status'] == status) and (not scenario or r.get('scenario') == scenario)]
    return {'run_id': run_id, 'total': len(rows), 'offset': offset, 'limit': limit, 'items': rows[offset:offset + limit]}
