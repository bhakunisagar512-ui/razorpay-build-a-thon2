"""Training-world simulation: persistent customers, shared outages, organic retries.

All parameters are explicit synthetic assumptions, not estimates from gateway logs.
"""
from datetime import datetime, timedelta, timezone
import heapq

import numpy as np

from eval.datasets import assemble
from eval.legacy_generate import CODE_EMISSION, _pick
from services.app.domain.taxonomy import Cause

GENERATOR_VERSION = 'sequences@v3'
DURATION_S = 3 * 86400


def generate_dataset(n=20000, seed=42, *, include_candidates=False, warmup_days=7):
    if n < 100:
        raise ValueError('Sequential generation requires at least 100 attempts')
    if warmup_days < 7:
        raise ValueError('At least seven days of history warm-up are required')
    rng = np.random.default_rng(seed)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    namespace = f'seq_v3_{seed}'
    issuers = ['issuer_A', 'issuer_B', 'issuer_C', 'issuer_D']
    psps = ['psp_alpha', 'psp_beta']
    # An episode is shared by every attempt on that infrastructure/time window.
    episodes = [
        dict(kind='issuer', entity='issuer_A', start_s=.18 * DURATION_S, end_s=.23 * DURATION_S, severity=.97),
        dict(kind='issuer', entity='issuer_C', start_s=.68 * DURATION_S, end_s=.73 * DURATION_S, severity=.88),
        dict(kind='psp', entity='psp_alpha', start_s=.35 * DURATION_S, end_s=.43 * DURATION_S, severity=.72),
        dict(kind='psp', entity='psp_beta', start_s=.85 * DURATION_S, end_s=.90 * DURATION_S, severity=.62),
    ]
    # Additional episodes ensure calibration and policy windows contain incidents.
    episodes.extend([
        dict(kind='issuer', entity='issuer_B', start_s=.50 * DURATION_S, end_s=.53 * DURATION_S, severity=.85),
        dict(kind='psp', entity='psp_beta', start_s=.55 * DURATION_S, end_s=.58 * DURATION_S, severity=.70),
        dict(kind='psp', entity='psp_alpha', start_s=.63 * DURATION_S, end_s=.66 * DURATION_S, severity=.65),
        dict(kind='issuer', entity='issuer_D', start_s=.82 * DURATION_S, end_s=.84 * DURATION_S, severity=.90),
    ])
    customers = []
    for i in range(max(20, n // 12)):
        customers.append(dict(id=f'cust_{namespace}_{i}', issuer=str(rng.choice(issuers)),
                              balance=int(rng.lognormal(10.5, 1.1)), expired=rng.random() < .10,
                              risk=rng.random() < .055, otp_friction=rng.uniform(.02, .20),
                              consent=bool(rng.random() < .78)))
    # n is fresh measurement orders, NOT a cap on retries or history rows.
    # Stratified arrivals guarantee coverage through the last second of the horizon.
    raw_times = np.linspace(0, DURATION_S - 1, n)
    warmup_s = warmup_days * 86400
    history_times = rng.uniform(-warmup_s, -1, max(n, int(n * warmup_s / DURATION_S)))
    queue = []
    serial = 0
    def enqueue(seconds, order=None, retry=0):
        nonlocal serial
        heapq.heappush(queue, (float(seconds), serial, order, retry))
        serial += 1
    for seconds in np.concatenate([history_times, raw_times]):
        enqueue(seconds)
    # Every instrument has an observed attempt shortly before measurement starts.
    # This supplies history, not an artificial success or a copied ground-truth label.
    for customer in customers:
        for method in ('card', 'upi', 'netbanking'):
            enqueue(-float(rng.uniform(60, 3600)), dict(
                id=f'ord_{namespace}_bootstrap_{serial}', customer=customer, method=method, bootstrap=True,
                psp=str(rng.choice(psps)), amount=int(np.clip(rng.lognormal(10.2, .9), 100, 1500000))))
    rows = []
    censored_retries = 0
    while queue:
        seconds, event_serial, order, retry = heapq.heappop(queue)
        if seconds >= DURATION_S:
            censored_retries += 1
            continue
        if order is None:
            customer = customers[int(rng.integers(len(customers)))]
            method = str(rng.choice(['card', 'upi', 'netbanking'], p=[.48, .42, .10]))
            order = dict(id=f'ord_{namespace}_{event_serial}', customer=customer, method=method,
                         psp=str(rng.choice(psps)), amount=int(np.clip(rng.lognormal(10.2, .9), 100, 1500000)))
        # History orders may retry during measurement; retain their warm-up origin
        # and exclude them from supervised partitions to prevent shared-order leakage.
        order.setdefault('is_warmup', seconds < 0)
        customer = order['customer']
        at = start + timedelta(seconds=seconds)
        issuer_severity = max([e['severity'] for e in episodes if e['kind'] == 'issuer' and e['entity'] == customer['issuer']
                               and e['start_s'] <= seconds < e['end_s']] or [0])
        psp_severity = max([e['severity'] for e in episodes if e['kind'] == 'psp' and e['entity'] == order['psp']
                            and e['start_s'] <= seconds < e['end_s']] or [0])
        # Persistent instrument/customer state generates correlated business declines.
        cause = None
        if customer['risk']:
            cause = Cause.RISK_BLOCKED
        elif order['method'] == 'card' and customer['expired']:
            cause = Cause.CARD_EXPIRED
        elif rng.random() < .015 + .88 * psp_severity:
            cause = Cause.TD_PSP_TIMEOUT
        elif rng.random() < .012 + .95 * issuer_severity:
            cause = Cause.TD_ISSUER_DOWN
        elif order['amount'] > customer['balance'] * (1 + .12 * retry):
            cause = Cause.BD_INSUFFICIENT_FUNDS
        elif order['method'] != 'upi' and rng.random() < customer['otp_friction']:
            cause = Cause.BD_USER_ABANDONED_OTP
        status = 'failed' if cause else 'success'
        code, source, step = _pick(rng, CODE_EMISSION[cause]) if cause else ('', '', '')
        if cause and rng.random() < .06:
            code, source, step = 'GATEWAY_ERROR', 'gateway', 'authorization'
        hint = ('expired' if customer['expired'] else 'valid') if order['method'] == 'card' else 'na'
        if order['method'] == 'card' and rng.random() < .18:
            hint = 'unknown'
        latency = float(rng.lognormal(5.4 + 2.0 * psp_severity + .8 * issuer_severity, .35))
        rows.append(dict(payment_id=f'pay_{namespace}_{len(rows):07d}', order_id=order['id'], customer_id=customer['id'],
                         issuer_id=customer['issuer'], psp_id=order['psp'], method=order['method'], amount_minor=order['amount'],
                         occurred_at=at, observed_at=at + timedelta(seconds=float(rng.exponential(3))),
                         status=status, instrument_hint=hint, gw_error_code=code, gw_error_source=source, gw_error_step=step,
                         latency_ms=latency, true_cause=str(cause) if cause else None,
                         available_rails='upi_intent,card,netbanking', consent_whatsapp=customer['consent'],
                         telemetry_visible=bool(order.get('bootstrap') or rng.random() > .02), is_warmup=order['is_warmup'],
                         instrument_id=f"inst_{customer['id']}_{order['method']}"))
        # Simulation-only detector state for evidence-channel experiments. These
        # are NOT feature inputs and never substitute for actual provider data.
        # No RNG draws: original observations/model inputs remain reproducible.
        rows[-1].update(fixture_expired=bool(order['method']=='card' and customer['expired']),
                        fixture_risk_blocked=bool(customer['risk']),
                        fixture_downtime=bool(issuer_severity or psp_severity))
        rows[-1].update(fixture_issuer_down=bool(issuer_severity),fixture_psp_down=bool(psp_severity))
        if cause and retry < 2 and rng.random() < .58:
            enqueue(seconds + float(rng.uniform(120, 1800)), order, retry + 1)
        # Payday-like balance drift is latent and not exported as a feature.
        if retry and rng.random() < .15:
            customer['balance'] = int(customer['balance'] * 1.5)
    dataset = assemble(rows, episodes, name=GENERATOR_VERSION, seed=seed, include_candidates=include_candidates)
    dataset.metadata.update(n_fresh_orders=n, warmup_days=warmup_days,
                            measurement_start=start.isoformat(), measurement_end=(start + timedelta(seconds=DURATION_S)).isoformat(),
                            censored_retries=censored_retries)
    return dataset
