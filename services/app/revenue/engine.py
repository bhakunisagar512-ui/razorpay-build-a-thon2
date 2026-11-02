"""Execution rules shared by the Temporal workflow activities and the in-process Dataset Testing executor (Phase 6, 8, 9).

check()          the stopping rules, with the SAME constants and order as services.app.domain.guards (consent, contact and retry budgets,
                 weekly contact cap, quiet hours), applied to in-memory state so the same code serves the database-backed workflow and
                 the in-process run.
simulate()       the independent outcome simulator. It is the ONLY code that may use a hidden cause. It never feeds planning or diagnosis.
final_status()   how a run ends when nothing was accepted.
execute()        a deterministic in-process run of a plan (no database, no Temporal), mirroring the workflow's loop.

Everything is a SIMULATION: no real message is sent and no real payment is made.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from services.app.domain import guards as G, outcome_sim
from services.app.revenue import catalog as C

WEEK = G.WEEK
MAX_WAIT_LOOPS = 6

# [assumed] how often a simulated customer completes a payment, for events that have no hidden cause. Directionally sensible, not measured.
P_ACCEPT = {'checkout_reminder': .30, 'invoice_reminder': .40, 'payment_reminder': .35, 'alternative_payment_method': .30, 'card_update_request': .35, 'delayed_retry': .45}
OUTCOME_ACTION = {'payment_reminder': 'payment_link', 'checkout_reminder': 'payment_link', 'invoice_reminder': 'payment_link',
                  'alternative_payment_method': 'payment_link', 'card_update_request': 'card_update_link', 'delayed_retry': 'silent_retry'}


def new_state(consent: dict, week: list | None = None) -> dict:
    return {'paid': False, 'consent': dict(consent), 'contacts': 0, 'retries': 0, 'week': list(week or [])}


def check(state: dict, step: dict, now: datetime):
    """(allowed, guard). Same order as guards._check: paid elsewhere, budgets, consent, weekly cap, quiet hours."""
    is_contact, is_retry = bool(step.get('channel')), bool(step.get('system_retry'))
    if state['paid']:
        return False, 'order_paid_elsewhere'
    if is_contact and state['contacts'] >= G.MAX_CONTACTS_PER_PAYMENT:
        return False, 'max_contact_attempts'
    if is_retry and state['retries'] >= G.MAX_RETRIES_PER_PAYMENT:
        return False, 'max_retries'
    if is_contact:
        if not state['consent'].get(step['channel'], False):
            return False, 'consent_required'
        if len([t for t in state['week'] if now - WEEK < t <= now]) >= G.MAX_CUSTOMER_CONTACTS_PER_WEEK:
            return False, 'max_customer_contacts_per_week'
        h = now.astimezone(G.IST).hour
        if h >= G.QUIET_START or h < G.QUIET_END:
            return False, 'quiet_hours'
    return True, None


def next_permitted(guard: str, now: datetime, state: dict):
    if guard == 'quiet_hours':
        local = now.astimezone(G.IST)
        reset = local.replace(hour=G.QUIET_END, minute=0, second=0, microsecond=0)
        return reset if local.hour < G.QUIET_END else reset + timedelta(days=1)
    if guard == 'max_customer_contacts_per_week':
        times = sorted(t for t in state['week'] if now - WEEK < t <= now)
        return now if len(times) < G.MAX_CUSTOMER_CONTACTS_PER_WEEK else times[len(times) - G.MAX_CUSTOMER_CONTACTS_PER_WEEK] + WEEK
    return None


def record_execution(state: dict, step: dict, now: datetime) -> None:
    if step.get('channel'):
        state['contacts'] += 1
        state['week'].append(now)
    if step.get('system_retry'):
        state['retries'] += 1


def _u(*parts) -> float:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def simulate(event_type: str, action: str, event_id: str, step_idx: int, hidden_cause: str | None = None) -> str:
    """ACCEPTED only when the simulated customer completes the payment. A correct diagnosis is not a payment."""
    if hidden_cause and action in OUTCOME_ACTION:
        return 'ACCEPTED' if outcome_sim.simulate(OUTCOME_ACTION[action], hidden_cause, event_id, 1, step_idx) == outcome_sim.SUCCESS else 'REJECTED'
    return 'ACCEPTED' if _u('revenue', event_type, action, event_id, step_idx) < P_ACCEPT.get(action, .3) else 'REJECTED'


# [assumed] what a synthetic payment-status lookup finds when a payment's state is genuinely uncertain (a timeout, or no usable evidence).
VERIFY_SUCCEEDED, VERIFY_FAILED = .35, .45          # the rest (.20) stays unresolved
UNCERTAIN = 'TD_PSP_TIMEOUT'


def verify_state(sim_key: str, hidden_cause: str | None = None) -> str:
    """The synthetic PSP status lookup behind the verify_payment_state step: SUCCEEDED (the original payment was captured), FAILED (confirmed) or
    UNRESOLVED (still unknown). A failure the gateway DECLINED is a confirmed failure; only a timeout, or an unknown cause, is genuinely uncertain."""
    if hidden_cause is not None and hidden_cause != UNCERTAIN:
        return 'FAILED'
    x = _u('verify', sim_key)
    return 'SUCCEEDED' if x < VERIFY_SUCCEEDED else 'FAILED' if x < VERIFY_SUCCEEDED + VERIFY_FAILED else 'UNRESOLVED'


def final_status(plan: dict, executed: int, suppressed: int):
    """(status, reason) when no payment was accepted and the plan is used up."""
    if executed == 0 and (suppressed > 0 or not plan['steps']):
        return C.HUMAN, 'no_safe_recovery_action'
    if plan.get('escalate_on_exhaustion'):
        return C.HUMAN, 'plan_exhausted_unresolved'
    return C.FAILED, 'plan_exhausted'


def execute(*, event_id: str, event_type: str, customer_id: str, amount_minor: int, plan: dict, consent: dict, sim_start: datetime,
            hidden_cause: str | None = None, week: dict | None = None, sim_key: str | None = None) -> dict:
    """Run a plan to its end, deterministically. `hidden_cause` is for the simulator only. `week` is shared per-customer contact history."""
    week = week if week is not None else {}
    state = new_state(consent, week.setdefault(customer_id, []))
    state['week'] = week[customer_id]                                   # shared list: contacts by earlier events count
    now, executed, suppressed, log = sim_start, 0, 0, []
    for step in plan['steps']:
        now += timedelta(seconds=step['delay_s'])
        for _ in range(MAX_WAIT_LOOPS):
            ok, guard = check(state, step, now)
            if ok or guard not in G.TEMPORARY_GUARDS:
                break
            now = next_permitted(guard, now, state)
        if not ok:
            suppressed += 1
            log.append({'idx': step['idx'], 'action': step['action'], 'status': 'SUPPRESSED', 'guard': guard})
            continue
        if step['action'] == 'human_escalation':
            log.append({'idx': step['idx'], 'action': step['action'], 'status': 'EXECUTED', 'sim_at': now.isoformat()})
            return {'status': C.HUMAN, 'reason': step.get('escalation_reason', 'risk_blocked'), 'recovered_amount_minor': 0, 'steps': log, 'executed': executed + 1, 'suppressed': suppressed}
        record_execution(state, step, now)
        executed += 1
        entry = {'idx': step['idx'], 'action': step['action'], 'status': 'EXECUTED', 'sim_at': now.isoformat()}
        if step['action'] == 'verify_payment_state':
            entry['verified'] = verify_state(sim_key or event_id, hidden_cause)
            log.append(entry)
            if entry['verified'] == 'SUCCEEDED':        # the original payment went through: nothing to recover, and no second payment is requested
                return {'status': C.RECOVERED, 'reason': 'original_payment_succeeded', 'recovered_amount_minor': amount_minor, 'steps': log, 'executed': executed, 'suppressed': suppressed}
            if entry['verified'] == 'UNRESOLVED':       # cannot tell: do not ask the customer to pay again, a person checks
                return {'status': C.HUMAN, 'reason': 'payment_state_unverified', 'recovered_amount_minor': 0, 'steps': log, 'executed': executed, 'suppressed': suppressed}
            continue
        if step['payment_opportunity']:
            entry['decision'] = simulate(event_type, step['action'], sim_key or event_id, step['idx'], hidden_cause)
        log.append(entry)
        if entry.get('decision') == 'ACCEPTED':
            return {'status': C.RECOVERED, 'reason': f"accepted_step_{step['idx']}", 'recovered_amount_minor': amount_minor, 'steps': log, 'executed': executed, 'suppressed': suppressed}
    status, reason = final_status(plan, executed, suppressed)
    return {'status': status, 'reason': reason, 'recovered_amount_minor': 0, 'steps': log, 'executed': executed, 'suppressed': suppressed}
