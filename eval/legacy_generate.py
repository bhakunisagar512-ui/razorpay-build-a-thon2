"""
Causal synthetic failure generator.

Design rule: sample the HIDDEN CAUSE first, then emit an OBSERVABLE error code
from a distribution conditioned on it. Real gateways return misleading codes;
if the generator emits a clean 1:1 mapping, the classifier grades its own
homework and the whole eval is worthless.

Priors are placeholders grounded in the shape of NPCI's published business-decline
vs technical-decline split. Replace with cited monthly figures before submission.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from services.app.domain.taxonomy import Cause

# ---- hidden cause prior -------------------------------------------------
CAUSE_PRIOR: dict[Cause, float] = {
    Cause.BD_INSUFFICIENT_FUNDS: 0.31,
    Cause.BD_USER_ABANDONED_OTP: 0.22,
    Cause.TD_ISSUER_DOWN: 0.14,
    Cause.TD_PSP_TIMEOUT: 0.11,
    Cause.CARD_EXPIRED: 0.13,
    Cause.RISK_BLOCKED: 0.09,
}

# ---- observable emission: cause -> {(code, source, step): p} ------------
# NOTE the deliberate overlap on GATEWAY_ERROR:payment_failed. That collision is
# what the classifier has to disambiguate, and it is why macro-F1 (not accuracy)
# is the reported metric.
CODE_EMISSION: dict[Cause, dict[tuple[str, str, str], float]] = {
    Cause.BD_INSUFFICIENT_FUNDS: {
        ("BAD_REQUEST_ERROR", "bank", "authorization"): 0.82,
        ("GATEWAY_ERROR", "gateway", "authorization"): 0.13,
        ("BAD_REQUEST_ERROR", "customer", "authentication"): 0.05,
    },
    Cause.BD_USER_ABANDONED_OTP: {
        ("BAD_REQUEST_ERROR", "customer", "authentication"): 0.74,
        ("GATEWAY_ERROR", "gateway", "authentication"): 0.18,
        ("BAD_REQUEST_ERROR", "customer", "authorization"): 0.08,
    },
    Cause.TD_ISSUER_DOWN: {
        ("GATEWAY_ERROR", "issuer", "authorization"): 0.61,
        ("GATEWAY_ERROR", "gateway", "authorization"): 0.29,
        ("BAD_REQUEST_ERROR", "bank", "authorization"): 0.10,
    },
    Cause.TD_PSP_TIMEOUT: {
        ("GATEWAY_ERROR", "gateway", "payment"): 0.68,
        ("GATEWAY_ERROR", "gateway", "authorization"): 0.24,
        ("SERVER_ERROR", "internal", "payment"): 0.08,
    },
    Cause.CARD_EXPIRED: {
        ("BAD_REQUEST_ERROR", "customer", "payment"): 0.79,
        ("BAD_REQUEST_ERROR", "bank", "authorization"): 0.14,
        ("GATEWAY_ERROR", "gateway", "authorization"): 0.07,
    },
    Cause.RISK_BLOCKED: {
        ("BAD_REQUEST_ERROR", "razorpay", "payment"): 0.71,
        ("GATEWAY_ERROR", "issuer", "authorization"): 0.21,
        ("BAD_REQUEST_ERROR", "bank", "authorization"): 0.08,
    },
}

# method must be consistent with cause: an expired card cannot fail over UPI
METHOD_BY_CAUSE: dict[Cause, dict[str, float]] = {
    Cause.BD_INSUFFICIENT_FUNDS: {"upi": 0.62, "card": 0.26, "netbanking": 0.12},
    Cause.BD_USER_ABANDONED_OTP: {"card": 0.71, "netbanking": 0.29},
    Cause.TD_ISSUER_DOWN: {"upi": 0.48, "card": 0.34, "netbanking": 0.18},
    Cause.TD_PSP_TIMEOUT: {"upi": 0.57, "card": 0.31, "netbanking": 0.12},
    Cause.CARD_EXPIRED: {"card": 1.0},
    Cause.RISK_BLOCKED: {"card": 0.63, "upi": 0.27, "netbanking": 0.10},
}

RAILS = ("upi_intent", "upi_collect", "card", "netbanking")

# adversarial noise rates
NOISE = {"duplicate": 0.04, "out_of_order": 0.03, "wrong_code": 0.05, "paisa_drift": 0.02}


def _pick(rng: np.random.Generator, dist: dict):
    keys = list(dist)
    probs = np.array([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=probs / probs.sum())]


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-08-01T00:00:00+05:30")

    for i in range(n):
        cause = _pick(rng, CAUSE_PRIOR)
        method = _pick(rng, METHOD_BY_CAUSE[cause])
        code, source, step = _pick(rng, CODE_EMISSION[cause])

        # amount: lognormal, floored, in paise. Indian D2C skews low with a long tail.
        amount_minor = int(np.clip(rng.lognormal(mean=7.6, sigma=0.95), 4900, 15_00_000)) * 100

        # failure time spread over 30 days, weighted toward evening commerce peak
        hour = int(np.clip(rng.normal(19, 4.2), 0, 23))
        failed_at = t0 + pd.Timedelta(days=int(rng.integers(0, 30)), hours=hour,
                                      minutes=int(rng.integers(0, 60)))

        avail = set(rng.choice(RAILS, size=rng.integers(2, 5), replace=False).tolist())
        avail.add(method if method != "upi" else "upi_intent")

        # Instrument metadata: card expiry is on the saved token, so the merchant
        # usually KNOWS it independently of the error code. 'unknown' models guest
        # checkouts / missing tokens. This is the feature that carries the signal
        # the error code cannot.
        if method == "card":
            if cause == Cause.CARD_EXPIRED:
                hint = "expired" if rng.random() < 0.85 else "unknown"
            else:
                hint = "valid" if rng.random() < 0.85 else "unknown"
        else:
            hint = "na"

        rows.append({
            "payment_id": f"pay_{i:07d}",
            "order_id": f"ord_{i:07d}",
            "customer_id": f"cust_{rng.integers(0, max(2, n // 3)):06d}",
            "amount_minor": amount_minor,
            "method": method,
            "instrument_hint": hint,
            "gw_error_code": code,
            "gw_error_source": source,
            "gw_error_step": step,
            "true_cause": str(cause),
            "available_rails": ",".join(sorted(avail)),
            "consent_whatsapp": bool(rng.random() < 0.78),
            "failed_at": failed_at,
        })

    df = pd.DataFrame(rows)

    # ---- adversarial noise ------------------------------------------------
    # wrong code: swap the observable to another cause's typical code, keep true label
    k = int(len(df) * NOISE["wrong_code"])
    idx = rng.choice(len(df), size=k, replace=False)
    for j in idx:
        other = _pick(rng, {c: p for c, p in CAUSE_PRIOR.items()
                            if str(c) != df.at[j, "true_cause"]})
        c, s, st = _pick(rng, CODE_EMISSION[other])
        df.loc[j, ["gw_error_code", "gw_error_source", "gw_error_step"]] = [c, s, st]

    # paisa drift: off-by-one amounts that break naive exact-match reconciliation
    k = int(len(df) * NOISE["paisa_drift"])
    idx = rng.choice(len(df), size=k, replace=False)
    df.loc[idx, "amount_minor"] += rng.choice([-1, 1], size=k)

    # duplicate webhooks: same payment_id delivered twice
    k = int(len(df) * NOISE["duplicate"])
    dupes = df.iloc[rng.choice(len(df), size=k, replace=False)].copy()
    dupes["is_duplicate"] = True
    df["is_duplicate"] = False
    df = pd.concat([df, dupes], ignore_index=True)

    # out-of-order delivery: shuffle a slice of the arrival sequence
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def assign_arm(payment_id: str, holdout_pct: int = 10) -> str:
    """Deterministic, reproducible, independent of row order or run time."""
    h = int(hashlib.sha256(payment_id.encode()).hexdigest()[:8], 16)
    return "holdout" if h % 100 < holdout_pct else "treatment"


def split_hash(df: pd.DataFrame) -> str:
    """Commit this to git BEFORE tuning anything. Proves you never peeked."""
    ids = "|".join(sorted(df["payment_id"].unique()))
    return hashlib.sha256(ids.encode()).hexdigest()
