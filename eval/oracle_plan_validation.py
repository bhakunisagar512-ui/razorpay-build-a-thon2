"""Plan-time validation and at most ONE deterministic correction, before anything is executed.

Inputs are the planned steps, the PREDICTED cause and operational facts (consent, rails, budgets). The hidden synthetic
cause is never an input. Quiet hours and the weekly contact cap are NOT validation errors: they are temporary and the
workflow waits for them. `correct_plan` only uses permitted alternatives (another consented channel, dropping a step that
cannot be made valid); it never creates consent and never returns the original plan unchanged.

facts: {'consent': {'whatsapp': bool, 'sms': bool, 'email': bool}, 'available_rails': [...], 'method': str,
        'prior_retries': int, 'prior_contacts': int, 'max_retries': int, 'max_contacts': int}
"""
RETRY_ACTIONS = ('silent_retry', 'rail_switch')
HARD_CAUSES = ('CARD_EXPIRED', 'RISK_BLOCKED')
CHANNEL_ORDER = ('whatsapp', 'sms', 'email')
ALLOWED_ACTIONS = {                                    # what the deterministic policy may do for each cause
    'BD_INSUFFICIENT_FUNDS': {'silent_retry', 'notify', 'payment_link'},
    'BD_USER_ABANDONED_OTP': {'payment_link'},
    'TD_ISSUER_DOWN': {'rail_switch', 'silent_retry'},
    'TD_PSP_TIMEOUT': {'verify_no_silent_success', 'silent_retry', 'payment_link'},
    'CARD_EXPIRED': {'card_update_link'},
    'RISK_BLOCKED': {'human_escalation'},
    'AMBIGUOUS': {'payment_link'},
}


def _err(code, step=None, **detail):
    return {'code': code, 'step': step, **detail}


def validate_plan(steps, cause, facts) -> list:
    cause = str(cause)
    errors = []
    if not steps:
        return [_err('EMPTY_PLAN')]
    allowed = ALLOWED_ACTIONS.get(cause, set())
    retries = contacts = 0
    for i, s in enumerate(steps):
        action, channel = s.get('action'), s.get('channel')
        if action not in allowed:
            errors.append(_err('ACTION_NOT_PERMITTED', i, action=action))
        if action in RETRY_ACTIONS:
            retries += 1
            if cause in HARD_CAUSES:
                errors.append(_err('HARD_DECLINE_RETRY', i, action=action))
            rail = s.get('rail')
            if action == 'rail_switch' and (not rail or rail not in facts['available_rails']):
                errors.append(_err('NO_RAIL', i, rail=rail))
        if channel is not None:
            contacts += 1
            if not facts['consent'].get(channel, False):
                errors.append(_err('NO_CONSENT', i, channel=channel))
    if facts['prior_retries'] + retries > facts['max_retries']:
        errors.append(_err('RETRY_BUDGET', over=facts['prior_retries'] + retries - facts['max_retries']))
    if facts['prior_contacts'] + contacts > facts['max_contacts']:
        errors.append(_err('CONTACT_BUDGET', over=facts['prior_contacts'] + contacts - facts['max_contacts']))
    return errors


def correct_plan(steps, errors, cause, facts):
    """One deterministic correction from permitted alternatives, or None when there is none."""
    cause = str(cause)
    by_step = {}
    for e in errors:
        by_step.setdefault(e['step'], set()).add(e['code'])
    consented = [c for c in CHANNEL_ORDER if facts['consent'].get(c, False)]
    out = []
    for i, s in enumerate(steps):
        codes = by_step.get(i, set())
        if codes & {'ACTION_NOT_PERMITTED', 'HARD_DECLINE_RETRY'}:
            continue
        s = dict(s)
        if 'NO_RAIL' in codes:
            if 'silent_retry' not in ALLOWED_ACTIONS.get(cause, set()):
                continue
            s.update(action='silent_retry', rail=facts['method'])
        if 'NO_CONSENT' in codes:
            if not consented:
                continue
            s['channel'] = consented[0]
        out.append(s)
    retries_left = max(0, facts['max_retries'] - facts['prior_retries'])
    contacts_left = max(0, facts['max_contacts'] - facts['prior_contacts'])
    kept, r, c = [], 0, 0
    for s in out:
        if s.get('action') in RETRY_ACTIONS:
            r += 1
            if r > retries_left:
                continue
        if s.get('channel') is not None:
            c += 1
            if c > contacts_left:
                continue
        kept.append(s)
    return kept if kept and kept != [dict(s) for s in steps] else None


def validate_and_correct(steps, cause, facts) -> dict:
    """{'valid', 'steps' (the plan to run when valid, else the original), 'original_steps', 'errors', 'corrected_steps',
    'final_errors', 'corrected'}. At most one correction; the correction is validated again."""
    steps = [dict(s) for s in steps]
    errors = validate_plan(steps, cause, facts)
    record = {'valid': not errors, 'steps': steps, 'original_steps': steps, 'errors': errors, 'corrected_steps': None,
              'final_errors': [], 'corrected': False}
    if not errors:
        return record
    corrected = correct_plan(steps, errors, cause, facts)
    if corrected is None:
        record['final_errors'] = errors
        return record
    final_errors = validate_plan(corrected, cause, facts)
    record.update(corrected_steps=corrected, final_errors=final_errors, corrected=True, valid=not final_errors)
    if not final_errors:
        record['steps'] = corrected
    return record
