"""Stopping rules. Evaluated at EXECUTION time, never plan time — the world
moves during a 72-hour timer. Every suppression is logged with the guard name;
the suppression log is evidence, not a gap."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

IST = timezone(timedelta(hours=5, minutes=30))
MAX_CONTACTS_PER_PAYMENT = 4
MAX_RETRIES_PER_PAYMENT = 6
MAX_CUSTOMER_CONTACTS_PER_WEEK = 6
QUIET_START, QUIET_END = 21, 9   # IST, outbound messages only
TEMPORARY_GUARDS = ("quiet_hours", "max_customer_contacts_per_week")   # clear by themselves: wait, do not fail
WEEK = timedelta(days=7)

@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    guard: str | None = None

def check(conn, payment_id: str, step: dict, now: datetime | None = None) -> GuardResult:
    return _check(conn, payment_id, step, now, simulated=False)


def check_at(conn, payment_id: str, step: dict, now: datetime) -> GuardResult:
    """Same rules on a simulated clock: the weekly window is measured around `now` and ignores attempts after it."""
    return _check(conn, payment_id, step, now, simulated=True)


def _check(conn, payment_id: str, step: dict, now, simulated: bool) -> GuardResult:
    now = now or datetime.now(tz=IST)
    is_contact = step.get("channel") is not None
    is_retry = step.get("action") in ("silent_retry", "rail_switch")

    row = conn.execute(text("SELECT customer_id, order_id FROM failed_payment WHERE id=:p"),
                       {"p": payment_id}).first()
    if row is None:
        return GuardResult(False, "unknown_payment")
    customer_id, order_id = row

    # the one everybody forgets: order already paid through some other path
    paid = conn.execute(text("SELECT 1 FROM order_settlement WHERE order_id=:o"), {"o": order_id}).first()
    if paid:
        return GuardResult(False, "order_paid_elsewhere")

    counts = conn.execute(text("""SELECT
          COUNT(*) FILTER (WHERE a.channel IS NOT NULL AND a.outcome <> 'suppressed'),
          COUNT(*) FILTER (WHERE a.action IN ('silent_retry','rail_switch') AND a.outcome <> 'suppressed')
        FROM recovery_attempt a JOIN recovery_plan p ON p.id = a.plan_id
        WHERE p.payment_id = :p"""), {"p": payment_id}).first()
    contacts, retries = counts or (0, 0)
    if is_contact and contacts >= MAX_CONTACTS_PER_PAYMENT:
        return GuardResult(False, "max_contact_attempts")
    if is_retry and retries >= MAX_RETRIES_PER_PAYMENT:
        return GuardResult(False, "max_retries")

    if is_contact:
        consent = conn.execute(text("SELECT whatsapp, sms, email FROM customer_consent WHERE customer_id=:c"),
                               {"c": customer_id}).mappings().first()
        if not consent or not consent.get(step["channel"], False):
            return GuardResult(False, "consent_required")
        wk = conn.execute(text("""SELECT COUNT(*) FROM recovery_attempt a
            JOIN recovery_plan p ON p.id = a.plan_id
            JOIN failed_payment f ON f.id = p.payment_id
            WHERE f.customer_id = :c AND a.channel IS NOT NULL
              AND a.outcome <> 'suppressed'
              AND a.executed_at > CAST(:since AS timestamptz)
              AND (NOT :sim OR a.executed_at <= CAST(:upper AS timestamptz))"""),
            {"c": customer_id, "sim": simulated, "upper": now,
             "since": (now - WEEK) if simulated else datetime.now(tz=timezone.utc) - WEEK}).scalar()
        if (wk or 0) >= MAX_CUSTOMER_CONTACTS_PER_WEEK:
            return GuardResult(False, "max_customer_contacts_per_week")
        h = now.astimezone(IST).hour
        if h >= QUIET_START or h < QUIET_END:
            return GuardResult(False, "quiet_hours")
    return GuardResult(True)


def next_permitted(conn, payment_id: str, guard: str, now: datetime, simulated: bool):
    """When a temporary restriction ends, or None if the reset time is not known."""
    if guard == "quiet_hours":
        local = now.astimezone(IST)
        reset = local.replace(hour=QUIET_END, minute=0, second=0, microsecond=0)
        return reset if local.hour < QUIET_END else reset + timedelta(days=1)
    if guard == "max_customer_contacts_per_week":
        customer_id = conn.execute(text("SELECT customer_id FROM failed_payment WHERE id=:p"), {"p": payment_id}).scalar()
        since = (now - WEEK) if simulated else datetime.now(tz=timezone.utc) - WEEK
        times = [r[0] for r in conn.execute(text("""SELECT a.executed_at FROM recovery_attempt a
            JOIN recovery_plan p ON p.id=a.plan_id JOIN failed_payment f ON f.id=p.payment_id
            WHERE f.customer_id=:c AND a.channel IS NOT NULL AND a.outcome <> 'suppressed'
              AND a.executed_at > CAST(:since AS timestamptz) AND (NOT :sim OR a.executed_at <= CAST(:upper AS timestamptz))
            ORDER BY a.executed_at"""), {"c": customer_id, "since": since, "sim": simulated, "upper": now})]
        if len(times) < MAX_CUSTOMER_CONTACTS_PER_WEEK:
            return now
        return times[len(times) - MAX_CUSTOMER_CONTACTS_PER_WEEK] + WEEK
    return None
