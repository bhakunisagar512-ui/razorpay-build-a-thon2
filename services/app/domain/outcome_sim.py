"""Synthetic payment-outcome model for the demo (no real payment is ever made).

Given the action a recovery step took and the HIDDEN actual cause, return SUCCESS / FAILURE / UNKNOWN.
Deterministic per (payment, cycle, step): the same replay gives the same outcome, and a correct action is never a
guaranteed success. The probabilities are assumptions chosen to be directionally sensible, not measured rates.

Only the outcome activity calls this. Diagnosis, the planner and reassessment never see the actual cause.
"""
import hashlib

SUCCESS, FAILURE, UNKNOWN = 'SUCCESS', 'FAILURE', 'UNKNOWN'
RETRY_ACTIONS = ('silent_retry', 'rail_switch')
OUTCOME_ACTIONS = (*RETRY_ACTIONS, 'payment_link', 'card_update_link')      # others (notify, verify, escalate) make no attempt
TRANSIENT = ('TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
VERIFY_SUCCESS_RATE = 0.40                                                   # an UNKNOWN that turns out to have been paid

P_SUCCESS = {
    'silent_retry': {'BD_INSUFFICIENT_FUNDS': .50, 'BD_USER_ABANDONED_OTP': .15, 'TD_ISSUER_DOWN': .60, 'TD_PSP_TIMEOUT': .65,
                     'CARD_EXPIRED': 0., 'RISK_BLOCKED': 0.},
    'rail_switch': {'BD_INSUFFICIENT_FUNDS': .15, 'BD_USER_ABANDONED_OTP': .25, 'TD_ISSUER_DOWN': .65, 'TD_PSP_TIMEOUT': .55,
                    'CARD_EXPIRED': 0., 'RISK_BLOCKED': 0.},
    'payment_link': {'BD_INSUFFICIENT_FUNDS': .35, 'BD_USER_ABANDONED_OTP': .55, 'TD_ISSUER_DOWN': .40, 'TD_PSP_TIMEOUT': .40,
                     'CARD_EXPIRED': .10, 'RISK_BLOCKED': .05},
    'card_update_link': {'BD_INSUFFICIENT_FUNDS': .03, 'BD_USER_ABANDONED_OTP': .03, 'TD_ISSUER_DOWN': .03, 'TD_PSP_TIMEOUT': .03,
                         'CARD_EXPIRED': .60, 'RISK_BLOCKED': .02},
}


def _u(*parts) -> float:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def simulate(action: str, actual_cause: str, payment_id: str, cycle: int, step_idx: int) -> str | None:
    """None when the action makes no payment attempt (notify, verify, human escalation)."""
    if action not in OUTCOME_ACTIONS:
        return None
    p = P_SUCCESS[action].get(actual_cause, 0.)
    p_unknown = .10 if action in RETRY_ACTIONS and actual_cause in TRANSIENT else .02 if action in RETRY_ACTIONS else 0.
    u = _u('outcome', payment_id, cycle, step_idx, action)
    return SUCCESS if u < p else UNKNOWN if u < p + p_unknown else FAILURE


def verify(payment_id: str, cycle: int, step_idx: int) -> str:
    """Resolve an UNKNOWN by checking the payment status (the existing verify-before-retry rule)."""
    return SUCCESS if _u('verify', payment_id, cycle, step_idx) < VERIFY_SUCCESS_RATE else FAILURE
