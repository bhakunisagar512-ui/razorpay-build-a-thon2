"""Deterministic recovery plans for the four event types (Phase 6). Pure functions: no I/O, no model, no hidden label.

A plan is built from the event type and, for payment and subscription failures, the ONE diagnosed cause. It is then run, step by step,
by the Temporal workflow (or the in-process executor for Dataset Testing), with the stopping rules in engine.py applied at execution time.

Every step that tries to collect money (a reminder, a payment link, an alternative method, a card update, an automatic retry) creates a
simulated payment opportunity. ACCEPT on that opportunity means the simulated customer completed the payment; REJECT means they did not.
Non-collecting steps (verify_payment_state, human_escalation) create none.
"""
from __future__ import annotations


from services.app.revenue import catalog as C

VERSION = 'revenue-plan@2'
DAY = 86400
MIN_FUNDS_RETRY_GAP_S = 3 * DAY      # never retry an instrument known to have had insufficient funds sooner than this after the failure
MAX_FUNDS_RETRY_GAP_S = 7 * DAY      # and not later than this: a recovery chance that old is mostly gone
ISSUER_RETRY_DELAY_S = 2 * 3600      # a bank outage lasts about an hour or more (median 70 min in the training world): do not retry into it
UNCERTAIN_CAUSES = ('TD_PSP_TIMEOUT', 'AMBIGUOUS')      # the payment may have succeeded: its state must be verified before anyone is asked to pay again

# action -> (label, customer_facing, system_retry, creates_payment_opportunity)
ACTIONS = {
    'checkout_reminder': ('Checkout reminder', True, False, True),
    'payment_reminder': ('Payment reminder', True, False, True),
    'invoice_reminder': ('Invoice reminder', True, False, True),
    'alternative_payment_method': ('Pay with another method', True, False, True),
    'card_update_request': ('Update card and pay', True, False, True),
    'delayed_retry': ('Automatic retry', False, True, True),
    'verify_payment_state': ('Verify the payment state', False, False, False),
    'human_escalation': ('Escalate to a human', False, False, False),
}
REMINDER_BY_EVENT = {C.CHECKOUT: 'checkout_reminder', C.OVERDUE: 'invoice_reminder', C.SUBSCRIPTION: 'payment_reminder', C.PAYMENT_FAILURE: 'payment_reminder'}
PAYMENT_MESSAGE = 'Your payment is pending. Complete payment using the available option.'
RETRY_CAUSES = ('BD_INSUFFICIENT_FUNDS', 'TD_ISSUER_DOWN')   # the causes whose plan includes ONE automatic retry, always delayed, and only when the failure was a confirmed decline
HARD_CAUSES = ('CARD_EXPIRED', 'RISK_BLOCKED')


def channels_for(consent: dict) -> list:
    return [c for c in C.CHANNELS if consent.get(c)]


def _step(idx, action, delay_s, channels, k=0, **extra):
    label, facing, retry, opportunity = ACTIONS[action]
    channel = channels[k % len(channels)] if facing and channels else None
    return {'idx': idx, 'action': action, 'label': extra.get('label', label), 'delay_s': delay_s, 'channel': channel, 'customer_facing': facing,
            'system_retry': retry, 'payment_opportunity': opportunity}


def _sequence(specs, channels):
    """specs: [(action, delay_s)] -> numbered steps, rotating the contact channel on each customer-facing step."""
    out, k = [], 0
    for idx, (action, delay) in enumerate(specs):
        out.append(_step(idx, action, delay, channels, k))
        k += int(ACTIONS[action][1])
    return out


def _cause_specs(event_type, cause):
    reminder = REMINDER_BY_EVENT[event_type]
    # delays are relative to the PREVIOUS step. The synthetic data carries no salary date or pay cycle for a customer, so none is assumed: the single
    # retry uses the fixed minimum gap (MIN_FUNDS_RETRY_GAP_S after the failure), which is documented, bounded by MAX_FUNDS_RETRY_GAP_S and never a claim of salary alignment.
    retry_at = MIN_FUNDS_RETRY_GAP_S
    return {
        'BD_INSUFFICIENT_FUNDS': ([('alternative_payment_method', 0), (reminder, DAY), ('delayed_retry', retry_at - DAY)],
                                  'Insufficient funds: offer another payment method now, remind the next day, and retry the original instrument once after a fixed 3-day delay (no salary date is known, so none is assumed).'),
        'BD_USER_ABANDONED_OTP': ([(reminder, 300), (reminder, DAY)], 'The customer did not finish authentication: send a fresh payment link, then a reminder.'),
        'CARD_EXPIRED': ([('card_update_request', 0), ('alternative_payment_method', 2 * DAY)], 'Expired card: never retried; ask for another method.'),
        'TD_ISSUER_DOWN': ([('delayed_retry', ISSUER_RETRY_DELAY_S), (reminder, 6 * 3600)], 'Issuer downtime: one retry after the outage has had time to clear (2 hours), then a reminder.'),
        'TD_PSP_TIMEOUT': ([('verify_payment_state', 60), (reminder, 3600)],
                           'Timeout: the payment may have succeeded. Its state is verified first; only a confirmed failure leads to a payment request, an unresolved state goes to a person, and nothing is retried.'),
        'RISK_BLOCKED': ([('human_escalation', 0)], 'Blocked by a risk check: no recovery attempt, escalate to a human.'),
        'AMBIGUOUS': ([('verify_payment_state', 60), (reminder, 6 * 3600)],
                      'No confident diagnosis (it could be a timeout): verify the payment state first, then send only the conservative payment reminder.'),
    }[cause]


def build(event_type: str, *, cause: str | None = None, consent: dict, params: dict | None = None) -> dict:
    """The recovery plan. `cause` is the single diagnosed cause (payment / subscription failures). Returns
    {'version', 'event_type', 'cause', 'steps', 'escalate_on_exhaustion', 'rationale'}."""
    params = params or {}
    channels = channels_for(consent)
    escalate = False
    if event_type == C.CHECKOUT:
        specs, why = [('checkout_reminder', 3600), ('checkout_reminder', DAY)], 'Cart left at checkout: a reminder with a payment opportunity, then one more on another channel.'
    elif event_type == C.OVERDUE:
        days = int(params.get('days_overdue', 0))
        specs = [('invoice_reminder', 0), ('invoice_reminder', 7 * DAY)] + ([('invoice_reminder', 14 * DAY)] if days >= 30 else [])
        escalate = days >= 30
        why = 'Overdue invoice: payment reminders with a payment opportunity' + ('; a long-overdue invoice goes to a human if unpaid.' if escalate else '.')
    else:
        cause = cause if cause in {'BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'CARD_EXPIRED', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT', 'RISK_BLOCKED'} else 'AMBIGUOUS'
        specs, why = _cause_specs(event_type, cause)
        escalate = cause == 'AMBIGUOUS'
    facing = [a for a, _ in specs if ACTIONS[a][1]]
    reason = 'risk_blocked'
    if facing and not channels:                                   # nobody to contact with consent: no safe customer-facing action exists
        specs, why, escalate, reason = [('human_escalation', 0)], 'The customer has given no contact consent, so no customer-facing recovery action is permitted: escalate.', False, 'no_safe_recovery_action'
    steps = _sequence(specs, channels)
    for s in steps:
        if s['action'] == 'human_escalation':
            s['escalation_reason'] = reason
    return {'version': VERSION, 'event_type': event_type, 'cause': cause if event_type in (C.SUBSCRIPTION, C.PAYMENT_FAILURE) else None, 'steps': steps,
            'escalate_on_exhaustion': escalate, 'rationale': why}


def validate(plan: dict) -> list:
    """Structural safety checks on a plan (used by tests and before a plan is persisted). Returns a list of problems."""
    problems = []
    for s in plan['steps']:
        if s['system_retry'] and plan.get('cause') in HARD_CAUSES:
            problems.append(f"step {s['idx']}: a hard-decline cause must never be retried")
        if s['customer_facing'] and not s['payment_opportunity']:
            problems.append(f"step {s['idx']}: customer-facing action without a payment opportunity")
        if s['customer_facing'] and not s['channel']:
            problems.append(f"step {s['idx']}: customer-facing action without a consented channel")
        if s['action'] == 'human_escalation' and s['idx'] != len(plan['steps']) - 1:
            problems.append('human_escalation must be the last step')
    if plan.get('cause') in UNCERTAIN_CAUSES:
        if any(s['system_retry'] for s in plan['steps']):
            problems.append('no retry while the payment state is uncertain')
        verified = False
        for s in plan['steps']:
            verified = verified or s['action'] == 'verify_payment_state'
            if s['payment_opportunity'] and not verified:
                problems.append(f"step {s['idx']}: a payment is requested before the payment state is verified")
    elapsed = 0
    for s in plan['steps']:
        elapsed += s['delay_s']
        if s['system_retry'] and plan.get('cause') == 'BD_INSUFFICIENT_FUNDS' and elapsed < MIN_FUNDS_RETRY_GAP_S:
            problems.append(f"step {s['idx']}: insufficient funds must not be retried within 3 days")
        if s['system_retry'] and plan.get('cause') == 'TD_ISSUER_DOWN' and elapsed < ISSUER_RETRY_DELAY_S:
            problems.append(f"step {s['idx']}: do not retry into a bank outage within 2 hours")
    return problems
