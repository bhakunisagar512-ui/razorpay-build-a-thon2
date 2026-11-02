"""Stress fixtures independent of training priors, emissions, and simulator.

Only taxonomy and point-in-time replay are shared. These are synthetic robustness
checks, not an independent real-world validation dataset. Do not tune on them.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
from eval.datasets import assemble
from services.app.domain.taxonomy import Cause

SCENARIOS = ('issuer_outage', 'psp_degradation', 'unseen_codes', 'hard_decline_collision', 'telemetry_gap')
SCENARIO_VERSION = 'stress-world@v1'
ERRORS = {
    Cause.TD_ISSUER_DOWN: ('GATEWAY_ERROR', 'issuer', 'authorization'),
    Cause.TD_PSP_TIMEOUT: ('SERVER_ERROR', 'internal', 'payment'),
    Cause.BD_INSUFFICIENT_FUNDS: ('BAD_REQUEST_ERROR', 'bank', 'authorization'),
    Cause.BD_USER_ABANDONED_OTP: ('BAD_REQUEST_ERROR', 'customer', 'authentication'),
    Cause.CARD_EXPIRED: ('BAD_REQUEST_ERROR', 'customer', 'payment'),
    Cause.RISK_BLOCKED: ('BAD_REQUEST_ERROR', 'razorpay', 'payment'),
}


def generate_scenario(name, n=4000, seed=901, *, include_candidates=False):
    if name not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {name}')
    if n < 100:
        raise ValueError('Stress generation requires at least 100 attempts')
    rng = np.random.default_rng(seed)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    duration = 86400
    prefix = f'stress_{name}_{seed}'
    rows, episodes = [], []
    if name == 'issuer_outage':
        episodes = [dict(kind='issuer', entity='unseen_bank_0', start_s=duration * .25,
                         end_s=duration * .65, severity=1.0)]
    elif name == 'psp_degradation':
        episodes = [dict(kind='psp_ramp', entity='unseen_psp_0', start_s=duration * .15,
                         end_s=duration * .85, severity=1.0)]
    previous = None
    for i in range(n):
        seconds = i * duration / n
        retry = bool(previous and previous['status'] == 'failed' and i % 4 == 1)
        customer_number = (i // 4) % max(15, n // 30)
        customer = f'cust_{prefix}_{customer_number}'
        issuer = f'unseen_bank_{customer_number % 3}'
        psp = f'unseen_psp_{i % 2}'
        method = str(rng.choice(['card', 'upi', 'netbanking'], p=[.60, .25, .15]))
        order_id = f'ord_{prefix}_{i}'
        amount = int(rng.choice([9900, 49900, 199900, 599900]))
        if retry:
            customer, issuer, psp, method, order_id, amount = (
                previous[k] for k in ('customer_id', 'issuer_id', 'psp_id', 'method', 'order_id', 'amount_minor'))
        at = start + timedelta(seconds=seconds)
        cause, shock = None, 0.
        hard_cohort = customer_number % (4 if name == 'hard_decline_collision' else 13)
        if method == 'card' and hard_cohort == 0:
            cause = Cause.CARD_EXPIRED
        elif hard_cohort == 1:
            cause = Cause.RISK_BLOCKED
        else:
            outage = name == 'issuer_outage' and issuer == 'unseen_bank_0' and .25 <= seconds / duration < .65
            ramp = name == 'psp_degradation' and psp == 'unseen_psp_0' and .15 <= seconds / duration < .85
            shock = float(np.sin((seconds / duration - .15) / .70 * np.pi) ** 2) if ramp else 0.
            if outage and rng.random() < .92:
                cause = Cause.TD_ISSUER_DOWN
            elif ramp and rng.random() < .12 + .75 * shock:
                cause = Cause.TD_PSP_TIMEOUT
            elif amount >= 199900 and rng.random() < (.12 if retry else .55):
                cause = Cause.BD_INSUFFICIENT_FUNDS
            elif method != 'upi' and rng.random() < .16:
                cause = Cause.BD_USER_ABANDONED_OTP
            elif rng.random() < .045:
                cause = Cause.TD_ISSUER_DOWN if i % 2 else Cause.TD_PSP_TIMEOUT
        code, source, step = ERRORS[cause] if cause else ('', '', '')
        if cause and name == 'unseen_codes':
            code = f'NEW_BANK_REASON_{str(cause)}'
            source, step = 'external_processor', 'confirmation'
        elif cause and name == 'hard_decline_collision' and cause in (Cause.CARD_EXPIRED, Cause.RISK_BLOCKED):
            code, source, step = 'GATEWAY_ERROR', 'issuer', 'authorization'
        elif cause and i % 5 == 0:
            code, source, step = 'GATEWAY_ERROR', 'gateway', 'authorization'
        hint = ('expired' if cause == Cause.CARD_EXPIRED else 'valid') if method == 'card' else 'na'
        if method == 'card' and (name == 'hard_decline_collision' or rng.random() < .35):
            hint = 'unknown'
        row = dict(payment_id=f'pay_{prefix}_{i:07d}', order_id=order_id, customer_id=customer,
                   instrument_id=f'inst_{customer}_{method}',
                   issuer_id=issuer, psp_id=psp, amount_minor=amount, method=method,
                   occurred_at=at, observed_at=at + timedelta(seconds=90 if i % 11 == 0 else 1),
                   status='failed' if cause else 'success', instrument_hint=hint,
                   gw_error_code=code, gw_error_source=source, gw_error_step=step,
                   latency_ms=None if name == 'telemetry_gap' else float(90 + 2500 * shock + rng.uniform(0, 180)),
                   true_cause=str(cause) if cause else None, available_rails='upi_intent,card,netbanking',
                   consent_whatsapp=bool(i % 5), telemetry_visible=bool(name != 'telemetry_gap' or i % 7 == 0))
        rows.append(row)
        row.update(fixture_expired=bool(method=='card' and hard_cohort==0),
                   fixture_risk_blocked=bool(hard_cohort==1),
                   fixture_downtime=bool((name=='issuer_outage' and issuer=='unseen_bank_0' and .25<=seconds/duration<.65)
                                        or (name=='psp_degradation' and psp=='unseen_psp_0' and .15<=seconds/duration<.85)))
        row.update(fixture_issuer_down=bool(name=='issuer_outage' and issuer=='unseen_bank_0' and .25<=seconds/duration<.65),
                   fixture_psp_down=bool(name=='psp_degradation' and psp=='unseen_psp_0' and .15<=seconds/duration<.85))
        previous = row
    return assemble(rows, episodes, name=f'{SCENARIO_VERSION}:{name}', seed=seed, duplicate_rate=.08,
                    include_candidates=include_candidates)
