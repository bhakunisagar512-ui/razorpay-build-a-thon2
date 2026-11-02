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

import json
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, engine as E, fixtures, routing, service as svc
from services.app.settings import settings

MIX = 'event_type_mix'
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
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
    return out


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


def specs(dataset: str, limit: int | None, event_id=lambda i, key: None):
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
    return {'case': spec['payload'].get('sim_key') or spec['event_id'], 'event_type': spec['event_type'], 'route': diagnosis.get('route'), 'diagnosis_method': diagnosis['method'],
            'rule_id': diagnosis.get('rule_id'), 'cause': diagnosis.get('cause'), 'model_c_calls': diagnosis['model_c_calls'],
            'diagnosis_matches_hidden_cause': (diagnosis.get('cause') == hidden) if diagnosis.get('cause') and hidden and spec['event_type'] != C.SUBSCRIPTION else None,
            'plan_actions': [s['action'] for s in plan['steps']], 'status': outcome['status'], 'reason': outcome['reason'], 'executed': outcome['executed'],
            'suppressed': outcome['suppressed'], 'amount_minor': spec['amount_minor'], 'recovered_amount_minor': outcome['recovered_amount_minor'],
            'violations': violations, 'steps': outcome['steps'], 'expected_route': spec.get('expected_route')}


def run_case(spec: dict) -> dict:
    diagnosis, plan = svc.decide(spec)
    if diagnosis.get('route') == routing.CONTRADICTORY:
        outcome = {'status': C.HUMAN, 'reason': 'contradictory_gateway_information', 'recovered_amount_minor': 0, 'steps': [], 'executed': 0, 'suppressed': 0}
    else:
        outcome = E.execute(event_id=spec['event_id'], event_type=spec['event_type'], customer_id=spec['customer_id'], amount_minor=spec['amount_minor'], plan=plan,
                            consent=spec['consent'], sim_start=spec['sim_start'], hidden_cause=spec.get('hidden_cause') or spec.get('failure_kind'),
                            week={}, sim_key=spec['payload'].get('sim_key'))
    return case_result(spec, diagnosis, plan, outcome)


# ------------------------------------------------------------------------------------------------- metrics
def summarise(results: list) -> dict:
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
        'note': 'All outcomes are simulated. No real payment, message or revenue was involved.'}


# ------------------------------------------------------------------------------------------------- Temporal sample
async def _temporal_sample(run_id: str, dataset: str, count: int, in_process: dict) -> dict:
    """Execute the first `count` cases through the real workflow (simulator answers) and compare with the in-process results."""
    import asyncio
    from temporalio.client import Client
    from sqlalchemy import text
    from services.app.db import engine
    out = {'requested': count, 'completed': 0, 'matched_in_process': 0, 'mismatches': [], 'error': None, 'task_queue': settings.revenue_task_queue}
    try:
        client = await Client.connect(settings.temporal_address)
        ids = {}
        for i, (key, spec) in enumerate(specs(dataset, count, event_id=lambda i, k: f'{run_id}_{i:04d}')):
            await asyncio.to_thread(svc.ingest, engine(), spec, 'fixture' if dataset in (fixtures.NAME, MIX) else 'dataset', run_id)
            ids[key] = spec['event_id']
            if in_process[key]['route'] != routing.CONTRADICTORY:
                await svc.start_workflow(spec['event_id'], responder='simulator', client=client)
        deadline = time.time() + 120
        done = {}
        while time.time() < deadline and len(done) < len(ids):
            with engine().begin() as conn:
                for key, eid in ids.items():
                    if key in done:
                        continue
                    r = conn.execute(text("SELECT status, status_reason, workflow_state, recovered_amount_minor FROM revenue_event WHERE event_id=:e"), {'e': eid}).mappings().first()
                    if r['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED'):
                        done[key] = dict(r)
            if len(done) < len(ids):
                await asyncio.sleep(1)
        out['completed'] = len(done)
        for key, r in done.items():
            mine = in_process[key]
            if (r['status'], r['recovered_amount_minor']) == (mine['status'], mine['recovered_amount_minor']):
                out['matched_in_process'] += 1
            else:
                out['mismatches'].append({'case': key, 'temporal': [r['status'], r['recovered_amount_minor']], 'in_process': [mine['status'], mine['recovered_amount_minor']]})
        if len(done) < len(ids):
            out['error'] = f'{len(ids) - len(done)} workflow(s) did not finish in time; is a worker running on queue {settings.revenue_task_queue}?'
    except Exception as exc:                                                            # noqa: BLE001
        out['error'] = f'{type(exc).__name__}: {exc}'[:300]
    return out


# ------------------------------------------------------------------------------------------------- lifecycle
def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, default=str), encoding='utf-8')
    tmp.replace(path)


def start(dataset: str, limit: int | None = None, temporal_limit: int = 0) -> str:
    if dataset.startswith('csv:'):
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
    _state[run_id] = {'run_id': run_id, 'dataset': dataset, 'status': 'queued', 'limit': limit, 'temporal_limit': temporal_limit, 'started_at': time.time()}
    _write(runs_dir() / f'{run_id}.status.json', _state[run_id])
    threading.Thread(target=_guarded, args=(run_id, dataset, limit, temporal_limit), daemon=True, name=f'rv-{run_id}').start()
    return run_id


def _guarded(run_id, dataset, limit, temporal_limit):
    try:
        execute(run_id, dataset, limit, temporal_limit)
    except Exception as exc:                                                            # noqa: BLE001
        _state[run_id].update(status='failed', error=f'{type(exc).__name__}: {exc}'[:300], finished_at=time.time())
        _write(runs_dir() / f'{run_id}.status.json', _state[run_id])
    finally:
        _current['run_id'] = None
        _lock.release()


def execute(run_id: str, dataset: str, limit: int | None, temporal_limit: int = 0) -> dict:
    st = _state.setdefault(run_id, {'run_id': run_id, 'dataset': dataset, 'started_at': time.time()})
    st.update(status='running')
    results = []
    for key, spec in specs(dataset, limit):
        row = run_case(spec)
        row['case'] = key                                                                  # the case key every consumer (and the Temporal sample) looks results up by
        results.append(row)
    by_key = {r['case']: r for r in results}
    st.update(processed=len(results), total=len(results))
    doc = {'run_id': run_id, 'dataset': dataset, 'limit': limit, 'mode': 'automated', 'status': 'completed', 'started_at': st['started_at'], 'finished_at': time.time(),
           'label': 'Automated simulation: an independent simulator answers every payment opportunity. Nothing here is real revenue.',
           'metrics': summarise(results)}
    if temporal_limit:
        import asyncio
        doc['temporal'] = asyncio.run(_temporal_sample(run_id, dataset, min(temporal_limit, len(results)), by_key))
    _write(runs_dir() / f'{run_id}.json', {**doc, 'cases': results})
    st.update(status='completed', finished_at=doc['finished_at'])
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


def cases(run_id: str, route=None, status=None, offset=0, limit=50):
    full = _read(runs_dir() / f'{run_id}.json') if get_run(run_id) else None
    if not full:
        return None
    rows = [r for r in full['cases'] if (not route or r['route'] == route) and (not status or r['status'] == status)]
    return {'run_id': run_id, 'total': len(rows), 'offset': offset, 'limit': limit, 'items': rows[offset:offset + limit]}
