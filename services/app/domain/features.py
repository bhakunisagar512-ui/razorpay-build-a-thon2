"""Versioned, point-in-time features shared by offline replay and inference.

Only observable attempts enter this module. Labels, latent infrastructure state,
future successes, and simulated recovery outcomes are never feature inputs.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import mean

FEATURE_SCHEMA_VERSION = "payment-features@v2"
BASELINE_FEATURES = (
    "gw_error_code", "gw_error_source", "gw_error_step", "method", "instrument_hint",
)
CATEGORICAL_FEATURES = (*BASELINE_FEATURES, "issuer_id", "psp_id")
NUMERIC_FEATURES = (
    "amount_minor", "log_amount_minor", "hour_ist", "weekday_ist",
    "customer_attempts_7d", "customer_successes_7d", "customer_failure_rate_7d",
    "customer_last_success_age_s", "amount_to_customer_mean_7d", "order_prior_attempts_7d",
    "issuer_attempts_5m", "issuer_failure_rate_5m", "issuer_latency_p95_ms_5m",
    "psp_attempts_5m", "psp_failure_rate_5m", "psp_latency_p95_ms_5m",
    "method_attempts_5m", "method_failure_rate_5m", "current_latency_ms",
)
FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
IST = timezone(timedelta(hours=5, minutes=30))
LOOKBACK = timedelta(days=7)


@dataclass(frozen=True)
class PaymentObservation:
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
    instrument_hint: str = "na"
    gw_error_code: str = ""
    gw_error_source: str = ""
    gw_error_step: str = ""
    latency_ms: float | None = None
    instrument_id: str = 'unknown'

    def __post_init__(self):
        for value in (self.occurred_at, self.observed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Feature timestamps must be timezone-aware")
        if self.observed_at < self.occurred_at:
            raise ValueError("An observation cannot precede its event")
        if self.status not in {"success", "failed"} or self.amount_minor <= 0:
            raise ValueError("Expected a positive amount and success/failed status")
        if self.latency_ms is not None and (not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("Latency must be finite and nonnegative")


def _stats(events):
    values = sorted(e.latency_ms for e in events if e.latency_ms is not None)
    return {
        "attempts": len(events),
        "failure_rate": sum(e.status == "failed" for e in events) / len(events) if events else None,
        "latency_p95_ms": values[max(0, math.ceil(len(values) * .95) - 1)] if values else None,
    }


class FeatureBuilder:
    """Replay observations in arrival order; build before observing the current row.

    Counts describe known attempts, not raw webhook deliveries. Duplicates are
    ignored. Missing rates/latencies remain None rather than imply healthy rails.
    An online consumer must persist/replay this state from its telemetry source.
    """

    def __init__(self, *, include_candidates=False):
        from services.app.domain.selected_features import MODEL_FEATURES
        self.feature_names = FEATURES if include_candidates else MODEL_FEATURES
        self._indexes = {name: defaultdict(deque) for name in ("customer_id", "order_id", "issuer_id", "psp_id", "method")}
        self._seen = {}
        self._arrivals = deque()
        self._watermark = None

    def _advance(self, as_of):
        if self._watermark is not None and as_of < self._watermark:
            raise ValueError("Replay observations in nondecreasing observed_at order")
        self._watermark = as_of
        while self._arrivals and self._arrivals[0].observed_at < as_of - LOOKBACK:
            old = self._arrivals.popleft()
            self._seen.pop(old.payment_id, None)
            for field, index in self._indexes.items():
                key = getattr(old, field)
                queue = index.get(key)
                if not queue or queue[0].payment_id != old.payment_id:
                    continue  # A shorter infrastructure window already evicted it.
                queue.popleft()
                if not queue:
                    del index[key]

    def _prior(self, field, current, window):
        queue = self._indexes[field].get(getattr(current, field), ())
        if field in ('issuer_id', 'psp_id', 'method'):
            while queue and queue[0].observed_at < current.observed_at - window:
                queue.popleft()
        return [e for e in queue
                if current.observed_at - window <= e.occurred_at
                and e.observed_at < current.observed_at and e.payment_id != current.payment_id]

    def build(self, current: PaymentObservation) -> dict:
        self._advance(current.observed_at)
        out = {name: getattr(current, name) for name in CATEGORICAL_FEATURES}
        if set(self.feature_names) <= set(CATEGORICAL_FEATURES):
            return {name: out[name] for name in self.feature_names}
        local = current.occurred_at.astimezone(IST)
        customer = self._prior("customer_id", current, LOOKBACK)
        successes = [e for e in customer if e.status == "success"]
        out.update(amount_minor=current.amount_minor, log_amount_minor=math.log1p(current.amount_minor),
                   hour_ist=local.hour, weekday_ist=local.weekday(),
                   customer_attempts_7d=len(customer), customer_successes_7d=len(successes),
                   customer_failure_rate_7d=_stats(customer)["failure_rate"],
                   customer_last_success_age_s=(current.observed_at - max(e.occurred_at for e in successes)).total_seconds() if successes else None,
                   amount_to_customer_mean_7d=current.amount_minor / mean(e.amount_minor for e in customer) if customer else None,
                   order_prior_attempts_7d=len(self._prior("order_id", current, LOOKBACK)),
                   current_latency_ms=current.latency_ms)
        for prefix, field in (("issuer", "issuer_id"), ("psp", "psp_id"), ("method", "method")):
            stats = _stats(self._prior(field, current, timedelta(minutes=5)))
            out[f"{prefix}_attempts_5m"] = stats["attempts"]
            out[f"{prefix}_failure_rate_5m"] = stats["failure_rate"]
            if prefix != "method":
                out[f"{prefix}_latency_p95_ms_5m"] = stats["latency_p95_ms"]
        return {name: out[name] for name in self.feature_names}

    def observe(self, observation: PaymentObservation) -> None:
        self._advance(observation.observed_at)
        if observation.payment_id in self._seen:
            return
        self._seen[observation.payment_id] = observation.observed_at
        self._arrivals.append(observation)
        for field, index in self._indexes.items():
            index[getattr(observation, field)].append(observation)
