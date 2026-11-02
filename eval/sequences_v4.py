"""sequences@v4: a more realistic, scenario-configurable payment world. v1-v3 are untouched.

What changed versus v3 (see docs/phase3-dataset-audit.md for why):
  * Razorpay vocabulary: error_reason / error_source / error_step per method, a catch-all
    reason share, missing and noisy reasons, documented section -> error_code.
  * Causes are independent hazards that can overlap; a stage race decides which one is
    reported (the label). There is no fixed priority chain and no 100%-persistent flag.
  * Expired cards come from reissues the merchant does not know about, so the stored
    hint is uninformative; risk is a transaction-level hazard with a cooldown, not a
    customer property; funds failures depend on amount vs. a noisy balance and a salary cycle.
  * Many issuers with heavy-tailed volume and persistent baselines (the 'market'); a
    randomised outage calendar per world (partial, ramped, method-scoped, plus
    network-wide incidents); downtime notices with detection delay, misses and false alarms.
  * Successes and failures; ~10% overall failure rate instead of 50%.

All parameters are assumptions unless tagged [documented]/[NPCI] in world_config_v4.py.
Nothing here is a measurement of real traffic.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import heapq
import math
import time

import numpy as np
import pandas as pd

from eval.world_config_v4 import (
    CAUSES, DURATION, GENERIC_BY_METHOD, GENERIC_MULTIPLIER, REASON_POOL, SCENARIOS_V4, SPECIFIC_REASONS,
    STAGE_SCALE, STEP_SOURCE, SUCCESS_DURATION,
)
from services.app.domain.failure_vocab import BAD_REQUEST_REASONS, GATEWAY_REASONS, SOURCES, STEPS
from services.app.domain.features_v4 import (
    FEATURES_V4, DowntimeNotice, FeatureBuilderV4, PaymentObservationV4,
)

GENERATOR_VERSION = 'sequences@v4'
METHODS = ('card', 'upi', 'netbanking')
DIURNAL = np.array([.3, .2, .15, .12, .12, .2, .4, .7, 1, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.3, 1.5, 1.8, 2.0,
                    1.9, 1.5, 1., .6])  # [assumed] relative order intensity by IST-like hour
DIURNAL = DIURNAL / DIURNAL.sum()
METHOD_BASE_PREF = {'card': .38, 'upi': .50, 'netbanking': .12}  # [assumed]
BASE_AMOUNT_INR = {'card': 1400., 'upi': 450., 'netbanking': 1500.}  # [assumed]
NETBANKING_LEVEL_MULT = 2.5  # [assumed] netbanking users hold/route larger balances for larger payments
LEVEL_MEDIAN_INR = {'account': 8000., 'card': 25000.}  # [assumed] spendable amount
AUTH_BASE = {'card': .055, 'upi': .018, 'netbanking': .05}  # [assumed; UPI total BD checked vs NPCI range]
RISK_BASE = {'card': .006, 'upi': .004, 'netbanking': .004}  # [assumed]
RETRY_P = {'BD_INSUFFICIENT_FUNDS': .30, 'BD_USER_ABANDONED_OTP': .60, 'TD_ISSUER_DOWN': .50,
           'TD_PSP_TIMEOUT': .65, 'CARD_EXPIRED': .30, 'RISK_BLOCKED': .22}  # [assumed]
RETRY_DELAY_MEDIAN_S = {'BD_INSUFFICIENT_FUNDS': 1800., 'BD_USER_ABANDONED_OTP': 200., 'TD_ISSUER_DOWN': 300.,
                        'TD_PSP_TIMEOUT': 120., 'CARD_EXPIRED': 600., 'RISK_BLOCKED': 900.}  # [assumed]
SWITCH_P = {'BD_INSUFFICIENT_FUNDS': .35, 'BD_USER_ABANDONED_OTP': .15, 'TD_ISSUER_DOWN': .15,
            'TD_PSP_TIMEOUT': .15, 'CARD_EXPIRED': .55, 'RISK_BLOCKED': .40}  # [assumed]
TIMEOUT_TYPE = {'payment_timed_out', 'payment_collect_request_expired', 'payment_session_expired'}


def _weighted(rng, table):
    keys = list(table)
    p = np.array([table[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


class Market:
    """Persistent institution characteristics, shared by worlds that model the same period."""

    def __init__(self, seed, n_issuers=14):
        rng = np.random.default_rng(seed)
        rank = np.arange(1, n_issuers + 1)
        share = 1.0 / rank ** 1.05 * rng.lognormal(0, .25, n_issuers)  # heavy-tailed volumes [assumed]
        self.issuer_ids = [f'issuer_{i:02d}' for i in range(n_issuers)]
        self.issuer_share = share / share.sum()
        self.issuer = {
            i: dict(td_base=float(rng.lognormal(math.log(.003), .7)),  # [NPCI] order of magnitude (UPI TD 0.2-0.8%)
                    bd_mult=float(rng.lognormal(0, .30)), friction=float(rng.lognormal(0, .30)),
                    risk=float(rng.lognormal(0, .40)), latency=float(rng.lognormal(math.log(220), .30)))
            for i in self.issuer_ids}
        self.gateway_ids = ['gw_a', 'gw_b', 'gw_c']
        self.gateway_share = np.array([.55, .30, .15])
        self.upi_app_ids = ['upi_app_a', 'upi_app_b', 'upi_app_c', 'upi_app_d', 'upi_app_e']
        self.upi_share = np.array([.42, .28, .15, .10, .05])
        self.psp = {i: dict(td_base=float(rng.lognormal(math.log(.003), .6)), latency=float(rng.lognormal(math.log(160), .25)))
                    for i in [*self.gateway_ids, *self.upi_app_ids]}

    @property
    def psp_ids(self):
        return [*self.gateway_ids, *self.upi_app_ids]


class _Instrument:
    __slots__ = ('id', 'method', 'issuer', 'account', 'level', 'upi_psp', 'expiry_s', 'reissue_s', 'replaced',
                 'risk_until', 'attempts', 'created_s')

    def __init__(self, **kw):
        self.replaced, self.risk_until, self.attempts = False, -1e18, 0
        self.expiry_s = self.reissue_s = math.inf
        self.upi_psp = None
        for key, value in kw.items():
            setattr(self, key, value)


def _cycle(local_day):
    """[assumed] balances are higher just after salary and thinner at month end."""
    return 1.5 if local_day <= 7 else 1.0 if local_day <= 20 else .6


def _make_outages(rng, cfg, market, t0, t1):
    """Randomised incident calendar in seconds since world start. Independent per world seed."""
    weeks = (t1 - t0) / (7 * 86400.)
    episodes, notices = [], []

    def episode(kind, entity, methods):
        start = rng.uniform(t0, t1)
        duration = float(np.clip(rng.lognormal(math.log(cfg.outage_median_minutes * 60), .8), 300, 8 * 3600))
        peak = float(np.clip(rng.beta(*cfg.outage_severity), .05, .98))
        return dict(kind=kind, entity=entity, methods=methods, start_s=start, end_s=start + duration, peak=peak,
                    ramp=.15, decay=.25)

    for issuer in market.issuer_ids:
        for _ in range(rng.poisson(cfg.outage_rate_per_week * weeks)):
            methods = METHODS if rng.random() < .5 else (str(rng.choice(METHODS)),)
            episodes.append(episode('issuer', issuer, methods))
    for psp in market.psp_ids:
        served = ('upi',) if psp.startswith('upi_app') else ('card', 'netbanking')
        for _ in range(rng.poisson(cfg.outage_rate_per_week * weeks)):
            episodes.append(episode('psp', psp, served))
    for _ in range(rng.poisson(cfg.network_incidents_per_week * weeks)):
        e = episode('network', '*', ('upi',))
        e['peak'] = float(np.clip(rng.beta(1.5, 6.), .03, .5))
        episodes.append(e)
    # Razorpay-style downtime notices: delayed, sometimes missed, sometimes false.
    for e in episodes:
        if e['kind'] == 'network':
            continue
        for method in e['methods']:
            if rng.random() >= cfg.downtime_detect * (.4 + .6 * e['peak']):
                continue
            start = e['start_s'] + float(rng.exponential(cfg.downtime_delay_minutes * 60))
            if start >= e['end_s']:
                continue
            end = e['end_s'] + float(rng.exponential(300))
            notices.append((f"dt_{len(notices)}", method, 'high' if e['peak'] >= .6 else 'medium',
                            e['entity'] if e['kind'] == 'issuer' else None,
                            e['entity'] if e['kind'] == 'psp' else None, start, end))
    for _ in range(rng.poisson(cfg.false_alarms_per_week * weeks)):
        kind = 'issuer' if rng.random() < .6 else 'psp'
        entity = str(rng.choice(market.issuer_ids if kind == 'issuer' else market.psp_ids))
        served = METHODS if kind == 'issuer' else (('upi',) if entity.startswith('upi_app') else ('card', 'netbanking'))
        start = rng.uniform(t0, t1)
        notices.append((f"dt_{len(notices)}", str(rng.choice(served)), str(rng.choice(['medium', 'high'])),
                        entity if kind == 'issuer' else None, entity if kind == 'psp' else None,
                        start, start + float(rng.uniform(20, 120)) * 60))
    return episodes, notices


@dataclass
class WorldV4:
    rows: list
    notices: list
    outages: pd.DataFrame
    metadata: dict
    transactions: pd.DataFrame = field(default=None)


def generate_world(n=24000, seed=1, *, scenario='typical', market_seed=11, warmup_days=14, days=3, config=None):
    cfg = config or SCENARIOS_V4[scenario]
    if n < 500:
        raise ValueError('World generation needs at least 500 measurement orders')
    if warmup_days < 14:
        raise ValueError('Feature windows need at least 14 days of warm-up history')
    rng = np.random.default_rng(seed)
    market = Market(market_seed)
    day0 = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 200)))
    t_start = -warmup_days * 86400.  # seconds relative to the first measured second
    t_end = days * 86400.
    ns = f'v4_{scenario}_{seed}'
    episodes, raw_notices = _make_outages(rng, cfg, market, t_start, t_end)
    by_entity = {}
    for e in episodes:
        by_entity.setdefault((e['kind'], e['entity']), []).append(e)

    def level(kind, entity, method, t):
        out = 0.
        for e in by_entity.get((kind, entity), ()):
            if e['start_s'] <= t < e['end_s'] and method in e['methods']:
                u = (t - e['start_s']) / (e['end_s'] - e['start_s'])
                shape = u / e['ramp'] if u < e['ramp'] else (1 - u) / e['decay'] if u > 1 - e['decay'] else 1.
                out = max(out, e['peak'] * min(1., shape))
        return out

    n_orders_warm = int(round(n * warmup_days / days))
    total_orders = n_orders_warm + n
    n_customers = max(300, total_orders // 4)
    activity = rng.gamma(.6, 1., n_customers)
    activity /= activity.sum()

    def new_card(cust, issuer, now_s, fresh=False):
        horizon = (t_start, t_end)
        card = _Instrument(id=f"inst_{cust['id']}_card{len(cust['cards'])}", method='card', issuer=issuer,
                           level=float(rng.lognormal(math.log(LEVEL_MEDIAN_INR['card']), 1.0)),
                           created_s=now_s if fresh else -float(rng.uniform(30, 900)) * 86400.)
        if not fresh:
            if rng.random() < .04:  # stored expiry date passes inside the horizon
                card.expiry_s = float(rng.uniform(*horizon))
            if rng.random() < cfg.reissue_probability:  # issuer reissues; merchant still holds old data
                card.reissue_s = float(rng.uniform(*horizon))
        cust['cards'].append(card)
        return card

    customers = []
    for i in range(n_customers):
        cid = f'cust_{ns}_{i}'
        accounts = [dict(issuer=str(rng.choice(market.issuer_ids, p=market.issuer_share)),
                         level=float(rng.lognormal(math.log(LEVEL_MEDIAN_INR['account']), 1.2)))]
        if rng.random() < .25:
            accounts.append(dict(issuer=str(rng.choice(market.issuer_ids, p=market.issuer_share)),
                                 level=float(rng.lognormal(math.log(LEVEL_MEDIAN_INR['account']), 1.2))))
        cust = dict(id=cid, accounts=accounts, cards=[], upi=None, nb=None, recent=[],
                    amount_scale=float(rng.lognormal(0, .5)), friction=float(rng.lognormal(0, .4)),
                    risk_prop=float(rng.lognormal(0, .8)))
        if rng.random() < .65:
            for _ in range(1 + int(rng.random() < .3)):
                issuer = accounts[0]['issuer'] if rng.random() < .6 else str(rng.choice(market.issuer_ids, p=market.issuer_share))
                new_card(cust, issuer, 0.)
        if rng.random() < .85:
            account = accounts[0]
            cust['upi'] = _Instrument(id=f'inst_{cid}_upi', method='upi', issuer=account['issuer'], account=account,
                                      level=account['level'], upi_psp=str(rng.choice(market.upi_app_ids, p=market.upi_share)),
                                      created_s=-float(rng.uniform(30, 900)) * 86400.)
        if rng.random() < .35:
            account = accounts[-1]
            cust['nb'] = _Instrument(id=f'inst_{cid}_nb', method='netbanking', issuer=account['issuer'], account=account,
                                     level=account['level'], created_s=-float(rng.uniform(30, 900)) * 86400.)
        available = [m for m, ok in (('card', cust['cards']), ('upi', cust['upi']), ('netbanking', cust['nb'])) if ok]
        if not available:
            cust['upi'] = _Instrument(id=f'inst_{cid}_upi', method='upi', issuer=accounts[0]['issuer'],
                                      account=accounts[0], level=accounts[0]['level'],
                                      upi_psp=str(rng.choice(market.upi_app_ids, p=market.upi_share)), created_s=-1e7)
            available = ['upi']
        weights = rng.dirichlet(np.array([METHOD_BASE_PREF[m] for m in available]) * 8)
        cust['pref'] = dict(zip(available, weights))
        customers.append(cust)

    def hours_to_seconds(count, lo_day, hi_day):
        day = rng.integers(lo_day, hi_day, count)
        hour = rng.choice(24, size=count, p=DIURNAL)
        return (day * 24 + hour) * 3600. + rng.uniform(0, 3600, count)

    arrivals = np.concatenate([hours_to_seconds(n_orders_warm, 0, warmup_days) + t_start,
                               hours_to_seconds(n, 0, days)])
    chosen = rng.choice(n_customers, size=len(arrivals), p=activity)
    queue, serial = [], 0
    for seconds, c in zip(arrivals, chosen):
        heapq.heappush(queue, (float(seconds), serial, int(c), None, 0))
        serial += 1

    rows, counts = [], dict.fromkeys(CAUSES, 0)
    secondary_total = overlap = 0
    censored = 0
    while queue:
        seconds, _, cidx, order, retry = heapq.heappop(queue)
        if seconds >= t_end:
            censored += 1
            continue
        cust = customers[cidx]
        at = day0 + timedelta(seconds=seconds)
        local = at + timedelta(hours=5.5)
        if order is None:
            method = _weighted(rng, cust['pref'])
            instr = (cust['cards'][int(rng.integers(len(cust['cards'])))] if method == 'card'
                     else cust['upi'] if method == 'upi' else cust['nb'])
            draw = float(rng.normal(0, .8))
            amount_inr = float(np.clip(BASE_AMOUNT_INR[method] * cust['amount_scale'] * math.exp(draw), 10, 150000))
            order = dict(id=f'ord_{ns}_{serial}', instr=instr, amount_inr=amount_inr, draw=draw)
            serial += 1
        instr, amount_inr = order['instr'], order['amount_inr']
        if instr.method == 'card' and seconds >= instr.expiry_s and rng.random() < .97:
            instr = new_card(cust, instr.issuer, seconds, fresh=True)  # checkout blocks a known-expired card
            order['instr'] = instr
        method = instr.method
        issuer_params = market.issuer[instr.issuer]
        psp = instr.upi_psp if method == 'upi' else str(rng.choice(market.gateway_ids, p=market.gateway_share))
        sev_issuer = level('issuer', instr.issuer, method, seconds)
        sev_psp = level('psp', psp, method, seconds)
        sev_net = level('network', '*', method, seconds)
        method_factor = {'upi': 1., 'card': .8, 'netbanking': 1.5}[method]
        fired = {}
        if rng.random() < min(.98, market.psp[psp]['td_base'] + sev_psp):
            fired['TD_PSP_TIMEOUT'] = 1
        if rng.random() < min(.98, issuer_params['td_base'] * method_factor + sev_issuer):
            fired['TD_ISSUER_DOWN'] = 1
        if sev_net and rng.random() < sev_net:
            fired['TD_PSP_TIMEOUT' if rng.random() < .6 else 'TD_ISSUER_DOWN'] = 1
        if method == 'card':
            available_inr = instr.level * math.exp(rng.normal(0, .5))
        else:
            available_inr = (instr.account['level'] * _cycle(local.day) * math.exp(rng.normal(0, .6))
                             * (NETBANKING_LEVEL_MULT if method == 'netbanking' else 1.))
        if amount_inr > available_inr:
            fired['BD_INSUFFICIENT_FUNDS'] = 1
        night_boost = 1.2 if 19 <= local.hour <= 22 else 1.
        if rng.random() < min(.5, AUTH_BASE[method] * cfg.bd_scale * cust['friction'] * issuer_params['friction'] * night_boost):
            fired['BD_USER_ABANDONED_OTP'] = 1
        cust['recent'] = [t for t in cust['recent'] if t > seconds - 3600]
        risk = (RISK_BASE[method] * cfg.bd_scale * cust['risk_prop'] * issuer_params['risk'] * math.exp(.45 * order['draw'])
                * (3. if instr.attempts < 3 else 1.) * (2.2 if len(cust['recent']) >= 3 else 1.)
                * (1.8 if local.hour < 6 else 1.) * (6. if seconds < instr.risk_until else 1.))
        if rng.random() < min(.9, risk):
            fired['RISK_BLOCKED'] = 1
        hint, slipped = 'na', False
        if method == 'card':
            hint = 'valid' if rng.random() < .8 else 'unknown'  # uninformative about expiry by construction
            if seconds >= instr.expiry_s:  # slipped past the checkout block (~3% of such attempts)
                hint, slipped = 'expired', True
                if rng.random() < .97:
                    fired['CARD_EXPIRED'] = 1
            elif seconds >= instr.reissue_s and not instr.replaced and rng.random() < .92:
                fired['CARD_EXPIRED'] = 1
        cust['recent'].append(seconds)
        instr.attempts += 1
        cause = None
        if fired:
            draws = {c: rng.exponential(STAGE_SCALE[c]) for c in fired}  # earliest failing stage wins
            cause = min(draws, key=draws.get)
            secondary_total += len(fired) - 1
            overlap += int(len(fired) > 1)
            counts[cause] += 1
        flow = {'card': lambda: 'card_3ds' if rng.random() < .7 else 'card_native_otp',
                'upi': lambda: 'upi_intent' if rng.random() < .8 else 'upi_collect',
                'netbanking': lambda: 'netbanking'}[method]()
        code = source = step = ''
        reason = None
        latency = float(issuer_params['latency'] * {'upi': 1., 'card': 1.3, 'netbanking': 2.}[method]
                        * math.exp(rng.normal(0, .35)) * (1 + 3 * sev_psp + 1.5 * sev_issuer + 2 * sev_net))
        if cause is None:
            duration = float(np.clip(rng.lognormal(math.log(SUCCESS_DURATION[0]), SUCCESS_DURATION[1]), 3, 600))
        else:
            g = min(.95, cfg.generic_share * GENERIC_MULTIPLIER[cause])
            reason = (_weighted(rng, GENERIC_BY_METHOD[method]) if rng.random() < g
                      else _weighted(rng, SPECIFIC_REASONS[cause][method]))
            if rng.random() < cfg.reason_noise:
                reason = str(rng.choice(REASON_POOL))
            both = reason in BAD_REQUEST_REASONS and reason in GATEWAY_REASONS
            code = ('SERVER_ERROR' if reason == 'server_error' else
                    ('BAD_REQUEST_ERROR' if rng.random() < .5 else 'GATEWAY_ERROR') if both else
                    'GATEWAY_ERROR' if reason in GATEWAY_REASONS else 'BAD_REQUEST_ERROR')
            steps, sources = STEP_SOURCE[method][cause]
            step, source = _weighted(rng, steps), _weighted(rng, sources)
            if rng.random() < .05:
                source = str(rng.choice(SOURCES[method]))
            if rng.random() < .05:
                step = str(rng.choice(STEPS[method]))
            if rng.random() < cfg.reason_missing:
                reason = None
            median, sigma = DURATION[cause]
            duration = float(np.clip(rng.lognormal(math.log(median), sigma), 1, 900))
            if reason in TIMEOUT_TYPE:
                duration = float(rng.uniform(420, 600))
            elif reason == 'request_timed_out':
                duration = float(rng.uniform(25, 90))
            if reason in TIMEOUT_TYPE or reason == 'request_timed_out':
                latency = None if rng.random() < cfg.latency_missing_on_timeout else float(30000 * rng.uniform(.8, 1.2))
        if latency is not None and rng.random() < cfg.telemetry_drop / 2:
            latency = None
        failed_at = at + timedelta(seconds=duration)
        delay = float(rng.exponential(120. if rng.random() < .02 else 4.))
        row = dict(
            payment_id=f'pay_{ns}_{len(rows):07d}', order_id=order['id'], customer_id=cust['id'],
            instrument_id=instr.id, issuer_id=instr.issuer, psp_id=psp, method=method,
            amount_minor=int(round(amount_inr * 100)), occurred_at=at, observed_at=failed_at + timedelta(seconds=delay),
            status='failed' if cause else 'success', instrument_hint=hint, gw_error_code=code, gw_error_source=source,
            gw_error_step=step, error_reason=reason, payment_flow=flow, attempt_duration_s=float(duration),
            latency_ms=latency, true_cause=cause, n_causes_fired=len(fired), retry_index=retry,
            telemetry_visible=bool(rng.random() >= cfg.telemetry_drop), is_warmup=bool(seconds < 0),
            world_id=ns)
        rows.append(row)
        if cause is None:
            continue
        if cause == 'RISK_BLOCKED':
            instr.risk_until = seconds + 12 * 3600
        if retry < 3 and rng.random() < RETRY_P[cause] * cfg.retry_scale * (.65 ** retry):
            next_instr = instr
            if rng.random() < SWITCH_P[cause]:
                if cause == 'CARD_EXPIRED' and method == 'card':
                    instr.replaced = True
                    others = [c for c in cust['cards'] if c is not instr and not (c.reissue_s <= seconds)]
                    next_instr = (others[0] if others and rng.random() < .5
                                  else new_card(cust, instr.issuer, seconds, fresh=True))
                else:
                    options = [c for c in [*cust['cards'], cust['upi'], cust['nb']] if c is not None and c is not instr]
                    if options:
                        next_instr = options[int(rng.integers(len(options)))]
            delay_s = float(np.clip(rng.lognormal(math.log(RETRY_DELAY_MEDIAN_S[cause]), 1.0), 30, 7200))
            retry_order = dict(order, instr=next_instr)
            heapq.heappush(queue, (seconds + duration + delay_s, serial, cidx, retry_order, retry + 1))
            serial += 1

    notices = []
    for event_id, method, severity, issuer, psp, start, end in raw_notices:
        notices.append(DowntimeNotice(f'{ns}_{event_id}', 'started', day0 + timedelta(seconds=start), method,
                                      severity, issuer, psp))
        notices.append(DowntimeNotice(f'{ns}_{event_id}', 'resolved', day0 + timedelta(seconds=end)))
    outages = pd.DataFrame([{**{k: v for k, v in e.items() if k != 'methods'}, 'methods': ','.join(e['methods']),
                             'start_at': day0 + timedelta(seconds=e['start_s']),
                             'end_at': day0 + timedelta(seconds=e['end_s'])} for e in episodes])
    metadata = dict(generator=GENERATOR_VERSION, scenario=cfg.name, seed=seed, market_seed=market_seed,
                    measurement_start=day0.isoformat(), measurement_end=(day0 + timedelta(seconds=t_end)).isoformat(),
                    n_fresh_orders=n, warmup_days=warmup_days, days=days, censored_retries=censored,
                    n_customers=n_customers, n_episodes=len(episodes), n_notices=len(raw_notices),
                    failure_overlap_fraction=overlap / max(1, sum(counts.values())), synthetic_only=True)
    return WorldV4(rows, notices, outages, metadata)


@dataclass
class DatasetV4:
    failures: pd.DataFrame
    transactions: pd.DataFrame
    outages: pd.DataFrame
    notices: pd.DataFrame
    metadata: dict


_LABEL_COLUMNS = ('true_cause', 'n_causes_fired')


def assemble_v4(world: WorldV4) -> DatasetV4:
    """Replay the world in arrival order, building features before each failure is observed."""
    builder = FeatureBuilderV4()
    rows = sorted(world.rows, key=lambda r: (r['observed_at'], r['payment_id']))
    notices = sorted(world.notices, key=lambda n: n.at)
    failures, cursor, build_seconds = [], 0, 0.
    for row in rows:
        while cursor < len(notices) and notices[cursor].at <= row['observed_at']:
            builder.notify(notices[cursor])
            cursor += 1
        obs = _observation(row)
        if row['status'] == 'failed' and not row['is_warmup']:
            started = time.perf_counter()
            features = builder.build(obs)
            build_seconds += time.perf_counter() - started
            failures.append({**features, 'payment_id': row['payment_id'], 'order_id': row['order_id'],
                             'customer_id': row['customer_id'], 'instrument_id': row['instrument_id'],
                             'amount_minor_raw': row['amount_minor'], 'observed_at': row['observed_at'],
                             'failed_at': row['occurred_at'], 'true_cause': row['true_cause'],
                             'n_causes_fired': row['n_causes_fired'], 'world_id': row['world_id'],
                             'scenario': world.metadata['scenario']})
        if row['telemetry_visible']:
            builder.observe(obs)
    frame = pd.DataFrame(failures)
    if frame.empty:
        raise ValueError('No failures generated')
    transactions = pd.DataFrame(rows)
    metadata = dict(world.metadata, n_attempts=len(rows), n_failures=len(frame),
                    n_measurement_attempts=int((~transactions.is_warmup).sum()),
                    feature_build_us_per_failure=build_seconds / len(frame) * 1e6,
                    model_features=list(FEATURES_V4))
    notice_frame = pd.DataFrame([vars(n) for n in notices])
    return DatasetV4(frame, transactions, world.outages, notice_frame, metadata)


def generate_dataset_v4(n=24000, seed=1, *, scenario='typical', market_seed=11, **kwargs):
    return assemble_v4(generate_world(n, seed, scenario=scenario, market_seed=market_seed, **kwargs))


def _observation(row):
    return PaymentObservationV4(**{k: row[k] for k in (
        'payment_id', 'order_id', 'customer_id', 'issuer_id', 'psp_id', 'amount_minor', 'method', 'occurred_at',
        'observed_at', 'status', 'instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step',
        'latency_ms', 'instrument_id', 'error_reason', 'payment_flow', 'attempt_duration_s')})


def rebuild_features_at(world: WorldV4, payment_id: str) -> dict:
    """Features for one failure from a replay that physically excludes every later event.

    Equality with the full-replay features proves nothing after the failure influenced them.
    """
    rows = sorted(world.rows, key=lambda r: (r['observed_at'], r['payment_id']))
    target_index = next(i for i, r in enumerate(rows) if r['payment_id'] == payment_id)
    target = rows[target_index]
    builder = FeatureBuilderV4()
    notices = sorted(world.notices, key=lambda n: n.at)
    cursor = 0
    for row in rows[:target_index]:
        while cursor < len(notices) and notices[cursor].at <= row['observed_at']:
            builder.notify(notices[cursor])
            cursor += 1
        if row['telemetry_visible']:
            builder.observe(_observation(row))
    while cursor < len(notices) and notices[cursor].at <= target['observed_at']:
        builder.notify(notices[cursor])
        cursor += 1
    return builder.build(_observation(target))
