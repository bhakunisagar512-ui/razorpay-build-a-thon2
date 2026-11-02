"""From a controlled event to a diagnosed, planned, persisted revenue event (Phase 2, 3, 5, 6).

An *event spec* is built from one of three controlled sources (a custom form submission, a prebuilt dataset record, or a routing fixture),
then decide() makes THE ONE diagnosis and builds THE plan. Nothing downstream (workflow, executor, Accept/Reject) diagnoses again, and
no hidden evaluation label is read here: the label rides along in the spec only so the independent simulator can use it later.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from zlib import crc32

from services.app.domain import custom_input_rules as R, guards as G
from services.app.revenue import catalog as C, planner, routing, store
from services.app.settings import settings

BASE_DAY = date(2026, 1, 5)
SPREAD_DAYS = 3650                      # simulated days an event can start on; wide, so repeated tests on the same customer rarely share a week (the weekly contact cap)
SCENARIO = ('method', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason')


def new_event_id(event_type: str) -> str:
    return f'{C.EVENT_PREFIX[event_type]}_{uuid.uuid4().hex[:10]}'


def sim_start_for(event_id: str, time_id: str) -> datetime:
    """The simulated clock origin: a day spread over ~10 years by the event id (so one customer's weekly contact cap is not hit by repeat tests)
    at the chosen time of day in IST. Quiet hours and the weekly cap run on this clock, never on the wall clock."""
    day = BASE_DAY + timedelta(days=crc32(event_id.encode()) % SPREAD_DAYS)
    t = C.lookup('time', time_id)
    return datetime(day.year, day.month, day.day, t['hour'], t['minute'], tzinfo=G.IST)


def derived_consent(customer_id: str) -> dict:
    """Deterministic consent for dataset customers (the prebuilt datasets carry none)."""
    u = lambda tag: crc32(f'{tag}|{customer_id}'.encode()) / 2 ** 32
    return {'whatsapp': u('w') < .90, 'sms': u('s') < .50, 'email': u('e') < .20}


# ------------------------------------------------------------------------------------------------- custom (controlled form)
REQUIRED = {
    C.CHECKOUT: ('customer_id', 'amount_rupees', 'failure_time', 'checkout_stage'),
    C.SUBSCRIPTION: ('customer_id', 'plan_id', 'method', 'failure_kind', 'failure_time'),
    C.OVERDUE: ('customer_id', 'invoice_id', 'failure_time'),
    C.PAYMENT_FAILURE: ('customer_id', 'amount_rupees', 'issuer_id', 'failure_time', *SCENARIO),
}
OPTIONAL = {C.PAYMENT_FAILURE: ('evaluation_label', 'diagnosis_scenario')}
# The selector only chooses WHICH INPUTS CAN BE BUILT (the scope of the dependent-dropdown rules). It is never read by routing, diagnosis or the planner.
SCENARIO_SCOPE = {'clear': 'clear', 'generic': 'generic', 'contradictory': 'conflict'}
DIAGNOSIS_SCENARIOS = tuple(SCENARIO_SCOPE)


def validate_custom(event_type: str, f: dict) -> list:
    """[{'field', 'message'}]; empty = valid. Every value must come from the catalog; fields that do not belong to the type are rejected."""
    if event_type not in C.EVENT_TYPES:
        return [{'field': 'event_type', 'message': f'event_type must be one of {", ".join(C.EVENT_TYPES)}.'}]
    errors, allowed = [], set(REQUIRED[event_type]) | set(OPTIONAL.get(event_type, ()))
    errors += [{'field': k, 'message': f'{k} is not used for {C.EVENT_LABEL[event_type]} events.'} for k, v in f.items() if v is not None and k not in allowed and k != 'event_type']
    for k in REQUIRED[event_type]:
        if k == 'error_reason' or f.get(k) is not None:
            continue
        errors.append({'field': k, 'message': f'{k} is required.'})
    if errors:
        return errors
    checks = (('customer_id', C.CUSTOMER_IDS), ('failure_time', C.FAILURE_TIME_IDS), ('amount_rupees', C.AMOUNTS_RUPEES), ('issuer_id', C.ISSUERS),
              ('checkout_stage', [x['id'] for x in C.CHECKOUT_STAGES]), ('plan_id', [x['id'] for x in C.SUBSCRIPTION_PLANS]), ('method', C.METHODS),
              ('failure_kind', C.FAILURE_KIND_IDS), ('invoice_id', [x['id'] for x in C.INVOICES]))
    for k, options in checks:
        if f.get(k) is not None and f[k] not in options:
            errors.append({'field': k, 'message': f'{k} must be one of the predefined options.'})
    if event_type == C.SUBSCRIPTION and f.get('failure_kind') == 'CARD_EXPIRED' and f.get('method') not in (None, 'card'):
        errors.append({'field': 'failure_kind', 'message': 'An expired card can only fail a card subscription.'})
    if event_type == C.PAYMENT_FAILURE and not errors:
        scenario = {k: f.get(k) for k in SCENARIO} | {'true_cause': f.get('evaluation_label')}
        chosen = f.get('diagnosis_scenario') or 'clear'
        if chosen not in SCENARIO_SCOPE:
            return [{'field': 'diagnosis_scenario', 'message': f'diagnosis_scenario must be one of {", ".join(DIAGNOSIS_SCENARIOS)}.'}]
        errors += R.validate(scenario, scope=SCENARIO_SCOPE[chosen])      # which inputs may be built; the router still decides the route from the inputs alone
    return errors


def custom_spec(event_type: str, f: dict, event_id: str | None = None) -> dict:
    eid = event_id or new_event_id(event_type)
    customer = C.customer(f['customer_id'])
    spec = {'event_id': eid, 'event_type': event_type, 'customer_id': customer['id'], 'consent': customer['consent'], 'sim_start': sim_start_for(eid, f['failure_time']),
            'hidden_cause': None, 'params': {}}
    base = {'customer_id': customer['id'], 'failure_time': f['failure_time']}
    if event_type == C.CHECKOUT:
        spec.update(amount_minor=f['amount_rupees'] * 100, payload={**base, 'amount_rupees': f['amount_rupees'], 'checkout_stage': f['checkout_stage']})
    elif event_type == C.SUBSCRIPTION:
        plan = C.lookup('plan', f['plan_id'])
        spec.update(amount_minor=plan['amount_minor'], payload={**base, 'plan_id': plan['id'], 'method': f['method'], 'failure_kind': f['failure_kind']}, failure_kind=f['failure_kind'])
    elif event_type == C.OVERDUE:
        inv = C.lookup('invoice', f['invoice_id'])
        spec.update(amount_minor=inv['amount_minor'], payload={**base, 'invoice_id': inv['id'], 'days_overdue': inv['days_overdue']}, params={'days_overdue': inv['days_overdue']})
    else:
        info = {k: f.get(k) for k in SCENARIO}
        amount = f['amount_rupees'] * 100
        spec.update(amount_minor=amount, info=info, hidden_cause=f.get('evaluation_label'),
                    payload={**base, 'amount_rupees': f['amount_rupees'], 'issuer_id': f['issuer_id'], **info, 'diagnosis_scenario': f.get('diagnosis_scenario') or 'clear'},
                    features_provider=lambda: routing.cold_start_features(info, customer_id=customer['id'], issuer_id=f['issuer_id'], amount_minor=amount,
                                                                          occurred_at=spec['sim_start'].isoformat()))
    return spec


# ------------------------------------------------------------------------------------------------- dataset records and fixtures
def _info(row) -> dict:
    return {k: (None if row.get(k) != row.get(k) else row.get(k)) for k in SCENARIO}          # NaN -> None


def dataset_spec(dataset: str, payment_id: str, event_id: str | None = None, *, row: dict | None = None, truth: str | None = None) -> dict:
    """A payment failure from a prebuilt dataset. The row carries the engineered features Model C needs (history included), unchanged.
    `row` can be passed pre-loaded (Dataset Testing reads each dataset once). `truth` is the hidden cause for the independent simulator: only the automated
    dataset evaluator (runs.py) supplies it. Interactive events get none, because a person answers their payment requests; this module never reads hidden truth."""
    from services.app import test_datasets as tds
    row = row if row is not None else tds.input_row(dataset, payment_id)
    if row is None:
        raise KeyError(payment_id)
    eid = event_id or new_event_id(C.PAYMENT_FAILURE)
    started = row['failed_at']
    started = started.to_pydatetime() if hasattr(started, 'to_pydatetime') else started
    started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
    info = _info(row)
    cid = row.get('customer_id') or f'cust_{payment_id}'
    return {'event_id': eid, 'event_type': C.PAYMENT_FAILURE, 'customer_id': cid, 'consent': derived_consent(cid), 'sim_start': started, 'amount_minor': int(row['amount_minor_raw']),
            'info': info, 'params': {}, 'hidden_cause': truth, 'features_provider': lambda: row,
            'payload': {'dataset': dataset, 'source_payment_id': payment_id, 'sim_key': f'{dataset}|{payment_id}', 'issuer_id': row.get('issuer_id'), **info,
                        'amount_rupees': round(int(row['amount_minor_raw']) / 100, 2)}}


def fixture_spec(record: dict, event_id: str | None = None) -> dict:
    eid = event_id or new_event_id(C.PAYMENT_FAILURE)
    info = {k: record.get(k) for k in SCENARIO}
    customer = C.customer(record['customer_id'])
    started = sim_start_for(record['payment_id'], record['time_id'])          # keyed by the fixture, so every run of it starts at the same simulated instant
    return {'event_id': eid, 'event_type': C.PAYMENT_FAILURE, 'customer_id': customer['id'], 'consent': customer['consent'], 'sim_start': started, 'amount_minor': record['amount_minor'],
            'info': info, 'params': {}, 'hidden_cause': record['hidden_cause'], 'expected_route': record['expected_route'],
            'payload': {'dataset': 'routing_fixtures', 'source_payment_id': record['payment_id'], 'sim_key': 'routing_fixtures|' + record['payment_id'], 'issuer_id': record['issuer_id'], **info, 'amount_rupees': record['amount_minor'] / 100},
            'features_provider': lambda: routing.cold_start_features(info, customer_id=customer['id'], issuer_id=record['issuer_id'], amount_minor=record['amount_minor'], occurred_at=started.isoformat())}


# ------------------------------------------------------------------------------------------------- the one diagnosis, the plan
def decide(spec: dict):
    """(diagnosis, plan). Payment failures go through the three-route router (Model C at most once); subscription failures carry the cause the
    subscription system reported; checkout and overdue events need no diagnosis."""
    t = spec['event_type']
    if t == C.PAYMENT_FAILURE:
        diagnosis = routing.diagnose(spec['info'], features_provider=spec['features_provider'])
        cause = diagnosis['cause']
    elif t == C.SUBSCRIPTION:
        cause = spec['failure_kind']
        diagnosis = {'route': None, 'method': 'reported_by_subscription_system', 'cause': cause, 'rule_id': f'KIND:{cause}', 'conflicts': [], 'model_c_calls': 0,
                     'explanation': f'The subscription system reported {R.CAUSE_INFO[cause].lower()}; no gateway diagnosis is needed.'}
    else:
        cause = None
        diagnosis = {'route': None, 'method': 'event_policy', 'cause': None, 'rule_id': f'EVENT:{t}', 'conflicts': [], 'model_c_calls': 0,
                     'explanation': f'{C.EVENT_LABEL[t]} needs no failure diagnosis: the plan follows from the event.'}
    if diagnosis['route'] == routing.CONTRADICTORY:
        plan = {'version': planner.VERSION, 'event_type': t, 'cause': None, 'steps': [], 'escalate_on_exhaustion': False,
                'rationale': 'Contradictory gateway information: no automatic recovery action is taken; a human reviews the record.'}
    else:
        plan = planner.build(t, cause=cause, consent=spec['consent'], params=spec.get('params'))
        problems = planner.validate(plan)
        if problems:
            raise ValueError('unsafe plan: ' + '; '.join(problems))
    return diagnosis, plan


def ingest(engine, spec: dict, source: str, run_id: str | None = None) -> dict:
    diagnosis, plan = decide(spec)
    with engine.begin() as conn:
        store.create_event(conn, spec=spec, diagnosis=diagnosis, plan=plan, source=source, run_id=run_id)
    return {'event_id': spec['event_id'], 'route': diagnosis.get('route'), 'diagnosis': diagnosis, 'plan': plan}


# ------------------------------------------------------------------------------------------------- Temporal
def workflow_options(responder: str = 'manual', response_timeout_s: int | None = None) -> dict:
    return {'responder': responder, 'response_timeout_s': response_timeout_s or settings.revenue_response_timeout_s, 'time_scale': settings.revenue_time_scale}


async def start_workflow(event_id: str, *, responder: str = 'manual', response_timeout_s: int | None = None, client=None):
    from temporalio.exceptions import WorkflowAlreadyStartedError
    from services.app import main
    from services.app.workflows.revenue import RevenueRecoveryWorkflow
    client = client or await main.temporal()
    try:
        await client.start_workflow(RevenueRecoveryWorkflow.run, args=[event_id, workflow_options(responder, response_timeout_s)], id=f'revenue-{event_id}',
                                    task_queue=settings.revenue_task_queue)
    except WorkflowAlreadyStartedError:
        pass
    return f'revenue-{event_id}'
