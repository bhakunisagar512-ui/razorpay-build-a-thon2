"""Phase 3 candidate features: the existing 26 plus 31 point-in-time additions (57).

Offline only; the serving classifier and features.py are unchanged. The 26 original
features are produced by the unchanged FeatureBuilder, so "existing 26-feature model"
means exactly those definitions.

Point-in-time contract: build(current) uses only attempts and downtime notices whose
observed/announced time is not later than current.observed_at, never the current row's
status, never any label or latent simulator state. Replay in nondecreasing time order,
call build() BEFORE observe() for the same attempt.

Deliberate deviation from the proposal: customer_success_rate uses a 14-day window
(LONG_WINDOW), not 30, to bound the simulation warm-up cost. The window is a constant.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import calendar
import math

from services.app.domain.failure_vocab import (
    GENERIC_REASONS, SESSION_TIMEOUT_S, TIMEOUT_REASONS, reason_family, step_ordinal,
)
from services.app.domain.features import (
    CATEGORICAL_FEATURES, FEATURES, IST, FeatureBuilder, PaymentObservation,
)

FEATURE_SCHEMA_VERSION_V4 = 'payment-features@v4-candidates'
LONG_WINDOW = timedelta(days=14)
DAY = timedelta(hours=24)
HOUR = timedelta(hours=1)
FIVE_MIN = timedelta(minutes=5)
BASELINE_WINDOW = timedelta(days=7)
DOWNTIME_STALE = timedelta(hours=12)
SEVERITY = {'high': 2, 'medium': 1}

FEATURE_GROUPS_V4 = {
    'A_reason': ('error_reason', 'reason_family', 'is_generic_reason', 'step_ordinal'),
    'B_auth_timing': ('attempt_duration_s', 'duration_vs_timeout_ratio', 'payment_flow', 'latency_vs_issuer_p50'),
    'C_instrument': ('instrument_consecutive_failures', 'instrument_attempts_24h', 'instrument_failure_rate_24h',
                     'instrument_last_success_age_s', 'instrument_prev_reason', 'same_reason_as_prev',
                     'instrument_distinct_reasons_24h'),
    'D_infrastructure': ('issuer_failure_rate_1h', 'issuer_failure_delta', 'issuer_failure_vs_baseline_7d',
                         'issuer_timeout_rate_5m', 'issuer_tech_reason_share_5m', 'psp_failure_rate_1h',
                         'psp_failure_delta', 'psp_timeout_rate_5m', 'issuer_method_failure_rate_5m',
                         'global_failure_rate_5m'),
    'E_customer': ('secs_since_prev_attempt', 'customer_methods_tried_24h', 'amount_vs_max_success_amount',
                   'customer_success_rate_14d'),
    'F_external_calendar': ('matched_downtime_severity', 'days_to_month_end'),
}
NEW_FEATURES_V4 = tuple(name for group in FEATURE_GROUPS_V4.values() for name in group)
FEATURES_V4 = (*FEATURES, *NEW_FEATURES_V4)
CATEGORICAL_FEATURES_V4 = (*CATEGORICAL_FEATURES, 'error_reason', 'reason_family', 'payment_flow',
                           'instrument_prev_reason')
NUMERIC_FEATURES_V4 = tuple(f for f in FEATURES_V4 if f not in CATEGORICAL_FEATURES_V4)
assert len(NEW_FEATURES_V4) == 31 and len(FEATURES_V4) == 57 and len(set(FEATURES_V4)) == 57


@dataclass(frozen=True)
class PaymentObservationV4:
    payment_id: str
    order_id: str
    customer_id: str
    issuer_id: str
    psp_id: str
    amount_minor: int
    method: str
    occurred_at: datetime
    observed_at: datetime
    status: str
    instrument_hint: str = 'na'
    gw_error_code: str = ''
    gw_error_source: str = ''
    gw_error_step: str = ''
    latency_ms: float | None = None
    instrument_id: str = 'unknown'
    error_reason: str | None = None
    payment_flow: str | None = None
    attempt_duration_s: float | None = None

    def legacy(self) -> PaymentObservation:
        return PaymentObservation(
            payment_id=self.payment_id, order_id=self.order_id, customer_id=self.customer_id,
            issuer_id=self.issuer_id, psp_id=self.psp_id, amount_minor=self.amount_minor, method=self.method,
            occurred_at=self.occurred_at, observed_at=self.observed_at, status=self.status,
            instrument_hint=self.instrument_hint, gw_error_code=self.gw_error_code,
            gw_error_source=self.gw_error_source, gw_error_step=self.gw_error_step,
            latency_ms=self.latency_ms, instrument_id=self.instrument_id)


@dataclass(frozen=True)
class DowntimeNotice:
    """A payment.downtime started/resolved notification as it ARRIVES (announcement time)."""
    event_id: str
    kind: str  # 'started' | 'resolved'
    at: datetime
    method: str | None = None
    severity: str | None = None
    issuer_id: str | None = None
    psp_id: str | None = None


class _Window:
    """Sliding window with running sums over (t, n=1, failed, timeout, tech, latency)."""

    __slots__ = ('span', 'items', 'attempts', 'failed', 'timeouts', 'tech')

    def __init__(self, span):
        self.span, self.items = span, deque()
        self.attempts = self.failed = self.timeouts = self.tech = 0

    def add(self, t, failed, timeout, tech, latency):
        self.items.append((t, failed, timeout, tech, latency))
        self.attempts += 1
        self.failed += failed
        self.timeouts += timeout
        self.tech += tech

    def trim(self, now):
        cutoff = now - self.span
        items = self.items
        while items and items[0][0] < cutoff:
            _, failed, timeout, tech, _ = items.popleft()
            self.attempts -= 1
            self.failed -= failed
            self.timeouts -= timeout
            self.tech -= tech

    def rate(self):
        return self.failed / self.attempts if self.attempts else None

    def timeout_rate(self):
        return self.timeouts / self.attempts if self.attempts else None

    def tech_share(self):
        return self.tech / self.failed if self.failed else None

    def latencies(self):
        return [item[4] for item in self.items if item[4] is not None]


def _median(values):
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


class FeatureBuilderV4:
    def __init__(self):
        self.legacy = FeatureBuilder(include_candidates=True)
        self._windows = defaultdict(dict)  # (scope, key) -> {span_name: _Window}
        self._instrument = defaultdict(deque)  # instrument -> deque[(t, failed, reason, method)]
        self._streak = defaultdict(int)
        self._last_success = {}
        self._customer = defaultdict(deque)  # customer -> deque[(t, method, success, amount)]
        self._seen = {}
        self._arrivals = deque()
        self._active_downtime = {}
        self._watermark = None

    # ---- stream handling -------------------------------------------------
    def _advance(self, now):
        if self._watermark is not None and now < self._watermark:
            raise ValueError('Replay observations in nondecreasing observed_at order')
        self._watermark = now
        cutoff = now - LONG_WINDOW
        while self._arrivals and self._arrivals[0][0] < cutoff:
            self._seen.pop(self._arrivals.popleft()[1], None)

    def notify(self, notice: DowntimeNotice) -> None:
        self._advance(notice.at)
        if notice.kind == 'started':
            self._active_downtime[notice.event_id] = notice
        elif notice.kind == 'resolved':
            self._active_downtime.pop(notice.event_id, None)
        else:
            raise ValueError('Unknown downtime notice kind')

    def _window(self, scope, key, name, span):
        windows = self._windows[(scope, key)]
        window = windows.get(name)
        if window is None:
            window = windows[name] = _Window(span)
        return window

    def _view(self, scope, key, now):
        """Trimmed 5m / 1h / 7d windows for an entity (created lazily)."""
        spans = (('5m', FIVE_MIN), ('1h', HOUR), ('7d', BASELINE_WINDOW))
        out = {}
        for name, span in spans:
            window = self._window(scope, key, name, span)
            window.trim(now)
            out[name] = window
        return out

    def observe(self, obs: PaymentObservationV4) -> None:
        self._advance(obs.observed_at)
        if obs.payment_id in self._seen:
            return
        self.legacy.observe(obs.legacy())
        self._seen[obs.payment_id] = obs.observed_at
        self._arrivals.append((obs.observed_at, obs.payment_id))
        now, failed = obs.observed_at, int(obs.status == 'failed')
        family = reason_family(obs.error_reason) if failed else None
        timeout = int(bool(failed and obs.error_reason in TIMEOUT_REASONS))
        tech = int(family == 'bank_technical')
        for scope, key in (('issuer', obs.issuer_id), ('psp', obs.psp_id),
                           ('issuer_method', (obs.issuer_id, obs.method)), ('global', '*')):
            for window in self._view(scope, key, now).values():
                window.add(now, failed, timeout, tech, obs.latency_ms)
        if obs.instrument_id not in ('unknown', ''):
            queue = self._instrument[obs.instrument_id]
            queue.append((now, failed, obs.error_reason, obs.method))
            if failed:
                self._streak[obs.instrument_id] = min(20, self._streak[obs.instrument_id] + 1)
            else:
                self._streak[obs.instrument_id] = 0
                self._last_success[obs.instrument_id] = now
        history = self._customer[obs.customer_id]
        history.append((now, obs.method, not failed, obs.amount_minor))

    # ---- feature construction -------------------------------------------
    def build(self, cur: PaymentObservationV4) -> dict:
        self._advance(cur.observed_at)
        now = cur.observed_at
        out = dict(self.legacy.build(cur.legacy()))
        reason = cur.error_reason or None
        out['error_reason'] = reason or 'missing'
        out['reason_family'] = reason_family(reason)
        out['is_generic_reason'] = int(reason is None or reason in GENERIC_REASONS)
        out['step_ordinal'] = step_ordinal(cur.gw_error_step)
        duration = cur.attempt_duration_s
        out['attempt_duration_s'] = duration
        out['duration_vs_timeout_ratio'] = duration / SESSION_TIMEOUT_S if duration is not None else None
        out['payment_flow'] = cur.payment_flow or 'missing'

        issuer = self._view('issuer', cur.issuer_id, now)
        psp = self._view('psp', cur.psp_id, now)
        issuer_method = self._view('issuer_method', (cur.issuer_id, cur.method), now)
        network = self._view('global', '*', now)
        median = _median(issuer['1h'].latencies())
        out['latency_vs_issuer_p50'] = (cur.latency_ms / median
                                       if cur.latency_ms is not None and median else None)

        # instrument history (24h window; streak and last success persist)
        out.update(self._instrument_features(cur, now, reason))

        r5, r1h = issuer['5m'].rate(), issuer['1h'].rate()
        out['issuer_failure_rate_1h'] = r1h
        out['issuer_failure_delta'] = r5 - r1h if r5 is not None and r1h is not None else None
        base, n1h = issuer['7d'].rate(), issuer['1h'].attempts
        if base is not None and issuer['7d'].attempts >= 50 and n1h >= 5 and r1h is not None:
            sd = math.sqrt(max(base * (1 - base), 1e-3) / n1h)
            out['issuer_failure_vs_baseline_7d'] = (r1h - base) / sd
        else:
            out['issuer_failure_vs_baseline_7d'] = None
        out['issuer_timeout_rate_5m'] = issuer['5m'].timeout_rate()
        out['issuer_tech_reason_share_5m'] = issuer['5m'].tech_share()
        p5, p1h = psp['5m'].rate(), psp['1h'].rate()
        out['psp_failure_rate_1h'] = p1h
        out['psp_failure_delta'] = p5 - p1h if p5 is not None and p1h is not None else None
        out['psp_timeout_rate_5m'] = psp['5m'].timeout_rate()
        out['issuer_method_failure_rate_5m'] = issuer_method['5m'].rate()
        out['global_failure_rate_5m'] = network['5m'].rate()

        out.update(self._customer_features(cur, now))
        out['matched_downtime_severity'] = self._downtime(cur, now)
        local = cur.occurred_at.astimezone(IST)
        out['days_to_month_end'] = calendar.monthrange(local.year, local.month)[1] - local.day
        return {name: out[name] for name in FEATURES_V4}

    def _instrument_features(self, cur, now, reason):
        keys = ('instrument_consecutive_failures', 'instrument_attempts_24h', 'instrument_failure_rate_24h',
                'instrument_last_success_age_s', 'instrument_prev_reason', 'same_reason_as_prev',
                'instrument_distinct_reasons_24h')
        if cur.instrument_id in ('unknown', ''):
            return dict.fromkeys(keys) | {'instrument_prev_reason': 'none', 'same_reason_as_prev': 0}
        queue = self._instrument[cur.instrument_id]
        while queue and queue[0][0] < now - DAY:
            queue.popleft()
        attempts = len(queue)
        failures = [item for item in queue if item[1]]
        previous = failures[-1][2] if failures else 'none'
        previous = 'missing' if previous is None else previous
        last = self._last_success.get(cur.instrument_id)
        return {
            'instrument_consecutive_failures': self._streak.get(cur.instrument_id, 0),
            'instrument_attempts_24h': attempts,
            'instrument_failure_rate_24h': len(failures) / attempts if attempts else None,
            'instrument_last_success_age_s': (now - last).total_seconds() if last else None,
            'instrument_prev_reason': previous,
            'same_reason_as_prev': int(reason is not None and previous == reason),
            'instrument_distinct_reasons_24h': len({item[2] for item in failures if item[2]}),
        }

    def _customer_features(self, cur, now):
        history = self._customer[cur.customer_id]
        while history and history[0][0] < now - LONG_WINDOW:
            history.popleft()
        recent = [item for item in history if item[0] >= now - DAY]
        wins = [item[3] for item in history if item[2]]
        return {
            'secs_since_prev_attempt': (now - history[-1][0]).total_seconds() if history else None,
            'customer_methods_tried_24h': len({item[1] for item in recent}),
            'amount_vs_max_success_amount': cur.amount_minor / max(wins) if wins else None,
            'customer_success_rate_14d': sum(item[2] for item in history) / len(history) if history else None,
        }

    def _downtime(self, cur, now):
        best = 0
        for notice in list(self._active_downtime.values()):
            if now - notice.at > DOWNTIME_STALE:
                continue  # missing 'resolved' notice must not keep an alert alive forever
            if notice.method is not None and notice.method != cur.method:
                continue
            if notice.issuer_id is None and notice.psp_id is None:
                continue
            if ((notice.issuer_id is not None and notice.issuer_id == cur.issuer_id)
                    or (notice.psp_id is not None and notice.psp_id == cur.psp_id)):
                best = max(best, SEVERITY.get(notice.severity, 0))
        return best
