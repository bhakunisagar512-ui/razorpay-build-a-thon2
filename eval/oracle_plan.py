"""
ORACLE PLAN (reference, offline only). The original per-cause plan builder: given the TRUE cause of a synthetic failure it returns the plan an
ideal planner would have produced. It exists solely so the prebuilt Dataset Testing datasets (eval/build_ui_datasets.py) and their truth files
stay reproducible, and so evaluation scripts can state what a perfect diagnosis would have allowed. It is NOT the runtime planner: the product
plans with services/app/revenue/planner.py (revenue-plan@2). Nothing under services/ imports this module.

build_plan() is a pure function: no I/O, no LLM, no clock reads beyond ctx.
"""
from __future__ import annotations
import calendar
from dataclasses import dataclass
from datetime import date
from services.app.domain.taxonomy import Cause, DAY

POLICY_VERSION = "policy@v1"
BASELINE_VERSION = "baseline@v1"
STRONG_BASELINE_VERSION = "baseline@v2-strong"


@dataclass(frozen=True)
class Ctx:
    amount_minor: int
    method: str
    today: date
    available_rails: frozenset[str]
    consent_whatsapp: bool = True


@dataclass(frozen=True)
class Step:
    action: str
    delay_s: int
    rail: str | None = None
    channel: str | None = None


@dataclass(frozen=True)
class Plan:
    cause: Cause
    steps: tuple[Step, ...]
    policy_ver: str = POLICY_VERSION

    @property
    def retry_count(self) -> int:
        return sum(1 for s in self.steps if s.action in ("silent_retry", "rail_switch"))


def _secs_to_next_salary_window(d: date) -> int:
    """Balance failures are time-dependent. Retrying in 1h is pure waste;
    the money arrives on the 1st or the 7th."""
    dim = calendar.monthrange(d.year, d.month)[1]
    for target in (1, 7):
        if d.day < target:
            return (target - d.day) * DAY
    return (dim - d.day + 1) * DAY


def _alt_rail(ctx: Ctx) -> str | None:
    for r in ("upi_intent", "upi_collect", "netbanking", "card"):
        if r in ctx.available_rails and not r.startswith(ctx.method):
            return r
    return None


def _channel(ctx: Ctx) -> str:
    return "whatsapp" if ctx.consent_whatsapp else "sms"


def build_plan(cause: Cause, ctx: Ctx) -> Plan:
    match cause:
        case Cause.BD_INSUFFICIENT_FUNDS:
            d = _secs_to_next_salary_window(ctx.today)
            return Plan(cause, (
                Step("silent_retry", d, rail=ctx.method),
                Step("notify", d + 3600, channel=_channel(ctx)),
                Step("silent_retry", d + 2 * DAY, rail=ctx.method),
            ))
        case Cause.BD_USER_ABANDONED_OTP:
            return Plan(cause, (
                Step("payment_link", 300, rail="upi_intent", channel=_channel(ctx)),
                Step("payment_link", DAY, rail="upi_intent", channel="sms"),
            ))
        case Cause.TD_ISSUER_DOWN:
            alt = _alt_rail(ctx)
            return Plan(cause, (Step("rail_switch", 0, rail=alt),) if alt
                        else (Step("silent_retry", 1800, rail=ctx.method),))
        case Cause.TD_PSP_TIMEOUT:
            # Found by eval: v1 shipped ONE retry here against the baseline's three,
            # on a highly retryable cause, and lost. Timeouts are transient — the
            # correct plan is a verify (never double-charge) then exponential backoff.
            return Plan(cause, (
                Step("verify_no_silent_success", 60),
                Step("silent_retry", 300, rail=ctx.method),
                Step("silent_retry", 3600, rail=ctx.method),
                Step("silent_retry", 6 * 3600, rail=ctx.method),
                Step("payment_link", 36 * 3600, rail="upi_intent", channel=_channel(ctx)),
            ))
        case Cause.CARD_EXPIRED:
            return Plan(cause, (
                Step("card_update_link", 0, channel=_channel(ctx)),
                Step("card_update_link", 3 * DAY, channel="email"),
            ))
        case Cause.RISK_BLOCKED:
            return Plan(cause, (Step("human_escalation", 0),))
        case _:
            # AMBIGUOUS conservative path. NOT a silent retry: if the true cause
            # is a hard decline, a retry hammers a dead instrument. A payment
            # link is the one action safe under every cause — it requests a
            # fresh attempt instead of replaying the failed one.
            return Plan(cause, (
                Step("payment_link", 6 * 3600, rail="upi_intent", channel=_channel(ctx)),
            ))


def baseline_plan(cause: Cause, ctx: Ctx) -> Plan:
    """What every naive dunning system does: three fixed retries, cause-blind.
    This is what the holdout arm receives."""
    return Plan(cause, (
        Step("silent_retry", 3600, rail=ctx.method),
        Step("silent_retry", 6 * 3600, rail=ctx.method),
        Step("silent_retry", 24 * 3600, rail=ctx.method),
    ), policy_ver=BASELINE_VERSION)


def strong_baseline_plan(cause: Cause, ctx: Ctx) -> Plan:
    """Cause-blind retries PLUS a generic payment link — what production dunning
    actually does.

    baseline_plan() never contacts the customer, which means part of the uplift
    measured against it is messaging asymmetry rather than diagnosis. This arm
    removes that advantage, and the number against it is the one worth
    defending."""
    return Plan(cause, (
        Step("silent_retry", 3600, rail=ctx.method),
        Step("silent_retry", 6 * 3600, rail=ctx.method),
        Step("silent_retry", 24 * 3600, rail=ctx.method),
        Step("payment_link", 36 * 3600, rail="upi_intent", channel=_channel(ctx)),
    ), policy_ver=STRONG_BASELINE_VERSION)
