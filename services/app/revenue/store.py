"""Database operations for the revenue pipeline (Phase 7, 8). All writes are idempotent or guarded, so Temporal activity retries,
duplicate ACCEPT/REJECT clicks and late responses can never double-count a recovery or resolve a request twice.

Tables (migration 005): revenue_event, revenue_truth (hidden labels, evaluator side only), revenue_step, payment_request,
revenue_intervention. Every state change also goes to the hash-chained audit_log with entity_type 'revenue_event'.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from services.app.domain import guards as G, ledger
from services.app.revenue import catalog as C, engine as E

ENTITY = 'revenue_event'
REASON_TEXT = {
    'contradictory_gateway_information': 'The gateway fields contradict each other; no automatic diagnosis or recovery was attempted.',
    'no_safe_recovery_action': 'No customer-facing recovery action was permitted (consent, budgets or safety rules).',
    'plan_exhausted_unresolved': 'Every planned action was used and the diagnosis was still unresolved.',
    'risk_blocked': 'The failure was identified as a risk block; automatic recovery is not allowed.',
    'payment_state_unverified': 'The payment state could not be verified (the original payment may have succeeded), so no new payment was requested; a person must check it.',
}
KIND_OF = {'contradictory_gateway_information': 'DIAGNOSIS_INTERVENTION', 'no_safe_recovery_action': 'PLAN_INTERVENTION',
           'plan_exhausted_unresolved': 'DIAGNOSIS_INTERVENTION', 'risk_blocked': 'SAFETY_ESCALATION', 'payment_state_unverified': 'SAFETY_ESCALATION'}


def _j(v):
    return json.dumps(v, default=str)


def audit(conn, event_id, actor, action, payload):
    ledger.append(conn, entity_type=ENTITY, entity_id=event_id, actor=actor, action=action, payload=payload)


def upsert_consent(conn, customer_id: str, consent: dict) -> None:
    """Missing consent stays denied and existing consent is never changed by an incoming event (same rule as payment ingestion)."""
    conn.execute(text("""INSERT INTO customer_consent(customer_id, whatsapp, sms, email) VALUES (:c,:w,:s,:e) ON CONFLICT (customer_id) DO NOTHING"""),
                 {'c': customer_id, 'w': bool(consent.get('whatsapp')), 's': bool(consent.get('sms')), 'e': bool(consent.get('email'))})


def consent_of(conn, customer_id: str) -> dict:
    row = conn.execute(text("SELECT whatsapp, sms, email FROM customer_consent WHERE customer_id=:c"), {'c': customer_id}).mappings().first()
    return {k: bool(row[k]) for k in C.CHANNELS} if row else {k: False for k in C.CHANNELS}


# ------------------------------------------------------------------------------------------------- create
def create_event(conn, *, spec: dict, diagnosis: dict, plan: dict, source: str, run_id: str | None = None) -> dict:
    """Persist a diagnosed, planned event. Contradictory payment failures are closed at once as HUMAN_INTERVENTION and never get a workflow."""
    eid = spec['event_id']
    contradictory = diagnosis.get('route') == 'CONTRADICTORY'
    risk = bool(plan['steps']) and plan['steps'][0]['action'] == 'human_escalation' and plan.get('cause') == 'RISK_BLOCKED'
    status, reason = (C.HUMAN, 'contradictory_gateway_information') if contradictory else (C.PENDING, None)
    upsert_consent(conn, spec['customer_id'], spec['consent'])
    conn.execute(text("""INSERT INTO revenue_event (event_id, event_type, source, run_id, customer_id, amount_minor, payload, route, diagnosis, plan, status,
            status_reason, workflow_state, sim_started_at) VALUES (:e,:t,:s,:r,:c,:a,CAST(:p AS jsonb),:rt,CAST(:d AS jsonb),CAST(:pl AS jsonb),:st,:sr,:ws,:sim)"""),
                 {'e': eid, 't': spec['event_type'], 's': source, 'r': run_id, 'c': spec['customer_id'], 'a': spec['amount_minor'], 'p': _j(spec['payload']),
                  'rt': diagnosis.get('route'), 'd': _j(diagnosis), 'pl': _j(plan), 'st': status, 'sr': reason,
                  'ws': 'NOT_REQUIRED' if contradictory else 'NOT_STARTED', 'sim': spec['sim_start']})
    if spec.get('hidden_cause'):
        conn.execute(text("INSERT INTO revenue_truth(event_id, actual_cause) VALUES (:e,:c)"), {'e': eid, 'c': spec['hidden_cause']})
    for s in plan['steps']:
        conn.execute(text("""INSERT INTO revenue_step(event_id, step_idx, action, label, channel, status) VALUES (:e,:i,:a,:l,:c,:st)"""),
                     {'e': eid, 'i': s['idx'], 'a': s['action'], 'l': s['label'], 'c': s['channel'], 'st': 'CANCELLED' if contradictory else 'PLANNED'})
    audit(conn, eid, 'ingest@v1', 'event_received', {'event_type': spec['event_type'], 'source': source, 'customer_id': spec['customer_id'], 'amount_minor': spec['amount_minor']})
    audit(conn, eid, 'router@v1', 'diagnosis', {k: diagnosis.get(k) for k in ('route', 'method', 'cause', 'rule_id', 'explanation', 'model_c_calls', 'conflicts')})
    audit(conn, eid, 'planner@v1', 'plan_created', {'version': plan['version'], 'steps': [(s['idx'], s['action'], s['channel']) for s in plan['steps']], 'rationale': plan['rationale']})
    if contradictory:
        create_intervention(conn, eid, reason, {'conflicts': diagnosis.get('conflicts')})
        audit(conn, eid, 'router@v1', 'human_intervention', {'reason': reason})
    return {'event_id': eid, 'status': status}


def create_intervention(conn, event_id: str, reason: str, details: dict | None = None) -> None:
    conn.execute(text("""INSERT INTO revenue_intervention(event_id, kind, reason, details) VALUES (:e,:k,:r,CAST(:d AS jsonb)) ON CONFLICT (event_id) DO NOTHING"""),
                 {'e': event_id, 'k': KIND_OF.get(reason, 'DIAGNOSIS_INTERVENTION'), 'r': reason, 'd': _j(details or {})})


# ------------------------------------------------------------------------------------------------- execution (activities)
def _lock_event(conn, event_id):
    row = conn.execute(text("SELECT * FROM revenue_event WHERE event_id=:e FOR UPDATE"), {'e': event_id}).mappings().first()
    if row is None:
        raise KeyError(event_id)
    return row


def guard_state(conn, ev, now: datetime) -> dict:
    steps = conn.execute(text("SELECT channel, action FROM revenue_step WHERE event_id=:e AND status='EXECUTED'"), {'e': ev['event_id']}).mappings().all()
    week = [r[0] for r in conn.execute(text("""SELECT s.sim_at FROM revenue_step s JOIN revenue_event v ON v.event_id=s.event_id
        WHERE v.customer_id=:c AND s.status='EXECUTED' AND s.channel IS NOT NULL AND s.sim_at > CAST(:since AS timestamptz) AND s.sim_at <= CAST(:now AS timestamptz)"""),
                                   {'c': ev['customer_id'], 'since': now - G.WEEK, 'now': now})]
    state = E.new_state(consent_of(conn, ev['customer_id']), week)
    state['contacts'] = sum(1 for s in steps if s['channel'])
    state['retries'] = sum(1 for s in steps if s['action'] == 'delayed_retry')
    state['paid'] = ev['status'] == C.RECOVERED
    return state


def start(conn, event_id: str, workflow_id: str) -> dict:
    ev = _lock_event(conn, event_id)
    conn.execute(text("UPDATE revenue_event SET workflow_state='RUNNING', workflow_id=:w, updated_at=now() WHERE event_id=:e AND workflow_state IN ('NOT_STARTED','RUNNING')"),
                 {'e': event_id, 'w': workflow_id})
    if ev['workflow_state'] == 'NOT_STARTED':
        audit(conn, event_id, 'workflow@v1', 'workflow_started', {'workflow_id': workflow_id})
    return {'plan': ev['plan'], 'sim_now': ev['sim_started_at'].isoformat(), 'status': ev['status'], 'event_type': ev['event_type']}


def execute_step(conn, event_id: str, idx: int, now: datetime, response_timeout_s: int) -> dict:
    ev = _lock_event(conn, event_id)
    if ev['status'] != C.PENDING:
        return {'outcome': 'closed'}
    step = ev['plan']['steps'][idx]
    done = conn.execute(text("SELECT status, guard, request_id FROM revenue_step WHERE event_id=:e AND step_idx=:i"), {'e': event_id, 'i': idx}).mappings().first()
    if done and done['status'] in ('EXECUTED', 'SUPPRESSED'):                         # an activity retry: report what already happened
        return {'outcome': 'executed' if done['status'] == 'EXECUTED' else 'suppressed', 'guard': done['guard'], 'request_id': done['request_id']}
    ok, guard = E.check(guard_state(conn, ev, now), step, now)
    if not ok:
        if guard in G.TEMPORARY_GUARDS:
            resume = E.next_permitted(guard, now, guard_state(conn, ev, now))
            audit(conn, event_id, 'workflow@v1', 'step_waiting', {'step': idx, 'guard': guard, 'resume_at': resume.isoformat()})
            return {'outcome': 'waiting', 'guard': guard, 'resume_at': resume.isoformat()}
        if guard == 'order_paid_elsewhere':
            return {'outcome': 'closed'}
        conn.execute(text("UPDATE revenue_step SET status='SUPPRESSED', guard=:g, sim_at=:t WHERE event_id=:e AND step_idx=:i"), {'g': guard, 'e': event_id, 'i': idx, 't': now})
        audit(conn, event_id, 'guards@v1', 'step_suppressed', {'step': idx, 'action': step['action'], 'guard': guard})
        return {'outcome': 'suppressed', 'guard': guard}
    if step['action'] == 'human_escalation':
        conn.execute(text("UPDATE revenue_step SET status='EXECUTED', sim_at=:t WHERE event_id=:e AND step_idx=:i"), {'e': event_id, 'i': idx, 't': now})
        audit(conn, event_id, 'executor@v1', 'step_executed', {'step': idx, 'action': 'human_escalation', 'mode': 'mock'})
        finish(conn, event_id, C.HUMAN, step.get('escalation_reason', 'risk_blocked'))
        return {'outcome': 'escalated'}
    conn.execute(text("UPDATE revenue_step SET status='EXECUTED', sim_at=:t WHERE event_id=:e AND step_idx=:i"), {'e': event_id, 'i': idx, 't': now})
    audit(conn, event_id, 'executor@v1', 'step_executed', {'step': idx, 'action': step['action'], 'channel': step['channel'], 'mode': 'mock', 'sim_at': now.isoformat()})
    if step['action'] == 'verify_payment_state':
        hidden = hidden_cause(conn, event_id) or (ev['payload'].get('failure_kind') if ev['event_type'] == C.SUBSCRIPTION else None)
        state = E.verify_state(ev['payload'].get('sim_key', event_id), hidden)
        audit(conn, event_id, 'psp-lookup@v1', 'payment_state_verified', {'step': idx, 'state': state, 'simulated': True})
        if state == 'SUCCEEDED':
            conn.execute(text("""UPDATE revenue_event SET status=:s, status_reason='original_payment_succeeded', recovered_amount_minor=:a, updated_at=now() WHERE event_id=:e AND status='PENDING'"""),
                         {'s': C.RECOVERED, 'a': ev['amount_minor'], 'e': event_id})
            conn.execute(text("UPDATE revenue_step SET status='CANCELLED' WHERE event_id=:e AND status='PLANNED'"), {'e': event_id})
            audit(conn, event_id, 'outcome@v1', 'simulated_recovered', {'reason': 'original_payment_succeeded', 'amount_minor': ev['amount_minor'], 'simulated': True})
            return {'outcome': 'resolved_paid'}
        if state == 'UNRESOLVED':
            finish(conn, event_id, C.HUMAN, 'payment_state_unverified')
            return {'outcome': 'escalated'}
        return {'outcome': 'executed', 'request_id': None, 'verified': state}
    request_id = None
    if step['payment_opportunity']:
        request_id = 'rq_' + uuid.uuid4().hex[:12]
        conn.execute(text("""INSERT INTO payment_request(request_id, event_id, step_idx, action, title, amount_minor, method, workflow_id, expires_at)
            VALUES (:r,:e,:i,:a,:t,:m,:me,:w, now() + make_interval(secs => :to)) ON CONFLICT (event_id, step_idx) DO NOTHING"""),
                     {'r': request_id, 'e': event_id, 'i': idx, 'a': step['action'], 't': step['label'], 'm': ev['amount_minor'],
                      'me': ev['payload'].get('method'), 'w': ev['workflow_id'], 'to': response_timeout_s})
        request_id = conn.execute(text("SELECT request_id FROM payment_request WHERE event_id=:e AND step_idx=:i"), {'e': event_id, 'i': idx}).scalar()
        conn.execute(text("UPDATE revenue_step SET request_id=:r WHERE event_id=:e AND step_idx=:i"), {'r': request_id, 'e': event_id, 'i': idx})
        audit(conn, event_id, 'executor@v1', 'payment_request_created', {'request_id': request_id, 'step': idx, 'action': step['action']})
    return {'outcome': 'executed', 'request_id': request_id}


def resolve_request(conn, request_id: str, decision: str, by: str) -> dict:
    """ACCEPT / REJECT, at most once. The first valid response wins; later ones change nothing and double-count nothing."""
    req = conn.execute(text("SELECT * FROM payment_request WHERE request_id=:r FOR UPDATE"), {'r': request_id}).mappings().first()
    if req is None:
        return {'found': False}
    if req['status'] != 'PENDING':
        return {'found': True, 'changed': False, 'status': req['status'], 'event_id': req['event_id']}
    ev = _lock_event(conn, req['event_id'])
    if ev['status'] != C.PENDING:                                                    # the event is already closed: this response is late
        conn.execute(text("UPDATE payment_request SET status='EXPIRED', decided_by='event_closed', resolved_at=now() WHERE request_id=:r"), {'r': request_id})
        return {'found': True, 'changed': False, 'status': 'EXPIRED', 'event_id': ev['event_id'], 'late': True}
    new = 'ACCEPTED' if decision == 'ACCEPT' else 'REJECTED'
    conn.execute(text("UPDATE payment_request SET status=:s, decided_by=:b, resolved_at=now() WHERE request_id=:r"), {'s': new, 'b': by, 'r': request_id})
    audit(conn, ev['event_id'], 'payment-page@v1' if by == 'tester' else 'simulator@v1', f'payment_request_{new.lower()}',
          {'request_id': request_id, 'step': req['step_idx'], 'simulated': True})
    if new == 'ACCEPTED':
        conn.execute(text("""UPDATE revenue_event SET status=:s, status_reason=:r, recovered_amount_minor=:a, updated_at=now() WHERE event_id=:e AND status='PENDING'"""),
                     {'s': C.RECOVERED, 'r': f"accepted_step_{req['step_idx']}", 'a': req['amount_minor'], 'e': ev['event_id']})
        conn.execute(text("UPDATE payment_request SET status='EXPIRED', decided_by='superseded', resolved_at=now() WHERE event_id=:e AND status='PENDING'"), {'e': ev['event_id']})
        conn.execute(text("UPDATE revenue_step SET status='CANCELLED' WHERE event_id=:e AND status='PLANNED'"), {'e': ev['event_id']})
        audit(conn, ev['event_id'], 'outcome@v1', 'simulated_recovered', {'request_id': request_id, 'amount_minor': req['amount_minor'], 'simulated': True})
    return {'found': True, 'changed': True, 'status': new, 'event_id': ev['event_id']}


def settle_request(conn, request_id: str) -> str:
    """What the workflow acts on after it wakes: the database is the source of truth. A request still PENDING at timeout expires."""
    req = conn.execute(text("SELECT status, event_id FROM payment_request WHERE request_id=:r FOR UPDATE"), {'r': request_id}).mappings().first()
    if req is None:
        return 'EXPIRED'
    if req['status'] == 'PENDING':
        conn.execute(text("UPDATE payment_request SET status='EXPIRED', decided_by='timeout', resolved_at=now() WHERE request_id=:r"), {'r': request_id})
        audit(conn, req['event_id'], 'workflow@v1', 'payment_request_expired', {'request_id': request_id})
        return 'EXPIRED'
    return req['status']


def finish(conn, event_id: str, status: str, reason: str) -> dict:
    ev = _lock_event(conn, event_id)
    if ev['status'] == C.PENDING:
        conn.execute(text("UPDATE revenue_event SET status=:s, status_reason=:r, updated_at=now() WHERE event_id=:e"), {'s': status, 'r': reason, 'e': event_id})
        conn.execute(text("UPDATE revenue_step SET status='CANCELLED' WHERE event_id=:e AND status='PLANNED'"), {'e': event_id})
        conn.execute(text("UPDATE payment_request SET status='EXPIRED', decided_by='event_closed', resolved_at=now() WHERE event_id=:e AND status='PENDING'"), {'e': event_id})
        if status == C.HUMAN:
            create_intervention(conn, event_id, reason, {'diagnosis': ev['diagnosis'].get('cause') if ev['diagnosis'] else None})
        audit(conn, event_id, 'workflow@v1', 'terminal_classified', {'status': status, 'reason': reason, 'simulated': True})
    conn.execute(text("UPDATE revenue_event SET workflow_state='WORKFLOW_COMPLETED', updated_at=now() WHERE event_id=:e"), {'e': event_id})
    audit(conn, event_id, 'workflow@v1', 'workflow_completed', {})
    return {'status': conn.execute(text("SELECT status FROM revenue_event WHERE event_id=:e"), {'e': event_id}).scalar()}


def finish_exhausted(conn, event_id: str) -> dict:
    ev = _lock_event(conn, event_id)
    if ev['status'] != C.PENDING:
        return finish(conn, event_id, ev['status'], ev['status_reason'] or '')
    steps = conn.execute(text("SELECT status FROM revenue_step WHERE event_id=:e"), {'e': event_id}).scalars().all()
    status, reason = E.final_status(ev['plan'], sum(s == 'EXECUTED' for s in steps), sum(s == 'SUPPRESSED' for s in steps))
    return finish(conn, event_id, status, reason)


def hidden_cause(conn, event_id: str):
    return conn.execute(text("SELECT actual_cause FROM revenue_truth WHERE event_id=:e"), {'e': event_id}).scalar()


# ------------------------------------------------------------------------------------------------- reads
def customer_context(event_type: str, payload: dict) -> dict:
    """What the simulated customer page may show about the event: catalog labels of the stage / plan / invoice and the method that was attempted. Read-only, derived
    from the stored controlled inputs; it never includes the diagnosis, the cause or any hidden evaluation label."""
    ctx = {'original_method': payload.get('method')}
    stage = C.lookup('stage', payload.get('checkout_stage')) if payload.get('checkout_stage') else None
    plan = C.lookup('plan', payload.get('plan_id')) if payload.get('plan_id') else None
    invoice = C.lookup('invoice', payload.get('invoice_id')) if payload.get('invoice_id') else None
    if stage:
        ctx['checkout_stage'] = stage['label']
    if plan:
        ctx['subscription_plan'] = plan['label']
    if invoice:
        ctx.update(invoice_number=invoice['label'].split(' · ')[0], days_overdue=invoice['days_overdue'])
    return ctx


def request_view(conn, request_id: str) -> dict | None:
    r = conn.execute(text("""SELECT r.*, e.event_type, e.status AS event_status, e.customer_id, e.payload FROM payment_request r JOIN revenue_event e ON e.event_id=r.event_id
        WHERE r.request_id=:r"""), {'r': request_id}).mappings().first()
    if r is None:
        return None
    return {'context': customer_context(r['event_type'], r['payload'] or {}),
            'request_id': r['request_id'], 'event_id': r['event_id'], 'workflow_id': r['workflow_id'], 'event_type': r['event_type'], 'event_label': C.EVENT_LABEL[r['event_type']],
            'action': r['action'], 'recovery_action': r['title'], 'amount_minor': r['amount_minor'], 'currency': 'INR', 'method': r['method'], 'status': r['status'],
            'event_status': r['event_status'], 'created_at': r['created_at'], 'resolved_at': r['resolved_at'], 'decided_by': r['decided_by'],
            'message': 'Your payment is pending. Complete payment using the available option.', 'step': r['step_idx'], 'simulated': True}


def event_view(conn, event_id: str, *, reveal_label: bool = False) -> dict | None:
    ev = conn.execute(text("SELECT * FROM revenue_event WHERE event_id=:e"), {'e': event_id}).mappings().first()
    if ev is None:
        return None
    steps = {r['step_idx']: dict(r) for r in conn.execute(text("SELECT * FROM revenue_step WHERE event_id=:e"), {'e': event_id}).mappings()}
    reqs = [dict(r) for r in conn.execute(text("SELECT * FROM payment_request WHERE event_id=:e ORDER BY step_idx"), {'e': event_id}).mappings()]
    timeline = [{'seq': r['seq'], 'actor': r['actor'], 'action': r['action'], 'payload': r['payload'], 'at': r['created_at']}
                for r in conn.execute(text("SELECT seq, actor, action, payload, created_at FROM audit_log WHERE entity_type=:t AND entity_id=:e ORDER BY seq"), {'t': ENTITY, 'e': event_id}).mappings()]
    plan_steps = [{**s, 'status': steps.get(s['idx'], {}).get('status', 'PLANNED'), 'guard': steps.get(s['idx'], {}).get('guard'),
                   'sim_at': steps.get(s['idx'], {}).get('sim_at'), 'request_id': steps.get(s['idx'], {}).get('request_id')} for s in ev['plan']['steps']]
    out = {'event_id': ev['event_id'], 'event_type': ev['event_type'], 'event_label': C.EVENT_LABEL[ev['event_type']], 'source': ev['source'], 'run_id': ev['run_id'],
           'customer_id': ev['customer_id'], 'amount_minor': ev['amount_minor'], 'currency': ev['currency'], 'inputs': ev['payload'], 'route': ev['route'],
           'diagnosis': ev['diagnosis'], 'plan': {**ev['plan'], 'steps': plan_steps}, 'status': ev['status'], 'status_reason': ev['status_reason'],
           'workflow_state': ev['workflow_state'], 'workflow_id': ev['workflow_id'], 'recovered_amount_minor': ev['recovered_amount_minor'],
           'requests': [{k: r[k] for k in ('request_id', 'step_idx', 'action', 'title', 'amount_minor', 'status', 'decided_by', 'created_at', 'resolved_at')} for r in reqs],
           'timeline': timeline, 'created_at': ev['created_at'], 'simulated': True,
           'note': 'Everything here is simulated. No real message was sent and no real payment or revenue was involved.'}
    if reveal_label and ev['source'] == 'custom':                                    # only a label the tester chose themselves is ever compared
        label = hidden_cause(conn, event_id)
        if label:
            out['evaluation'] = {'label': label, 'diagnosis_matches_label': (ev['diagnosis'] or {}).get('cause') == label}
    return out


def pending_requests(conn, limit: int = 50) -> list:
    return [dict(r) for r in conn.execute(text("""SELECT r.request_id, r.event_id, r.action, r.title, r.amount_minor, r.created_at, e.event_type FROM payment_request r
        JOIN revenue_event e ON e.event_id=r.event_id WHERE r.status='PENDING' ORDER BY r.created_at DESC LIMIT :n"""), {'n': limit}).mappings()]


def list_interventions(conn, kind=None, limit: int = 500) -> list:
    rows = conn.execute(text("""SELECT i.event_id, i.kind, i.reason, i.created_at, e.event_type, e.payload, e.amount_minor, e.diagnosis, e.source,
            (SELECT count(*) FROM revenue_step s WHERE s.event_id=e.event_id AND s.status='EXECUTED') AS executed
        FROM revenue_intervention i JOIN revenue_event e ON e.event_id=i.event_id WHERE (CAST(:k AS text) IS NULL OR i.kind=:k) ORDER BY i.created_at DESC LIMIT :n"""),
                        {'k': kind, 'n': limit}).mappings().all()
    return [dict(r) for r in rows]


INTERVENTION_LABEL = 'Revenue pipeline (Temporal run; simulated)'
INTERVENTION_KINDS = ('SAFETY_ESCALATION', 'DIAGNOSIS_INTERVENTION', 'PLAN_INTERVENTION')


def intervention_items(conn, kind=None, dataset=None) -> list:
    """The Human Intervention queue as the console lists it: one item per event a person must look at, newest first."""
    items = []
    for r in list_interventions(conn, kind, 5000):
        payload = r['payload'] or {}
        if dataset and (payload.get('dataset') or r['event_type']) != dataset:
            continue
        cause = (r['diagnosis'] or {}).get('cause')
        items.append({'source': 'revenue', 'source_label': INTERVENTION_LABEL, 'id': r['event_id'], 'kind': r['kind'], 'reason': r['reason'],
                      'reason_text': REASON_TEXT.get(r['reason'], r['reason']), 'created_at': r['created_at'], 'dataset': payload.get('dataset') or r['event_type'],
                      'source_payment_id': payload.get('source_payment_id'), 'executed_steps': r['executed'], 'cause': cause, 'method': payload.get('method'),
                      'amount_minor': r['amount_minor'], 'event_type': r['event_type']})
    return items
