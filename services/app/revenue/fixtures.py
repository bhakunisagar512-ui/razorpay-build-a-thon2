"""Controlled synthetic records that exercise ALL THREE payment-failure routes (Dataset Testing and backend tests only).

Custom Testing offers clear inputs only; the missing/generic and contradictory cases live here and in the prebuilt datasets. Every record is
generated deterministically from the same compatibility tables the dropdowns use, and carries a hidden cause for the independent simulator
(never for diagnosis or planning). Nothing here is a real payment.
"""
from __future__ import annotations

from datetime import datetime

from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C

NAME = 'routing_fixtures'
PER_ROUTE = {'CLEAR_AND_CONSISTENT': 8, 'MISSING_OR_GENERIC': 8, 'CONTRADICTORY': 6}


def _pick(rows, n):
    rows = sorted(rows, key=str)
    step = max(1, len(rows) // n)
    return [rows[i * step] for i in range(n)]


def _cause_of(method, row):
    """The hidden cause used by the simulator: the first cause whose declared profile contains the scenario."""
    hint, _code, source, step, reason = row
    for cause, p in R.cause_profiles(method).items():
        if hint in p['hints'] and source in p['sources'] and step in p['steps'] and (reason is None or reason in p['reasons']):
            return cause
    return 'BD_INSUFFICIENT_FUNDS'


def _contradictions(method):
    base = {'method': method, 'instrument_hint': R.default_hint(method), 'gw_error_code': 'BAD_REQUEST_ERROR', 'gw_error_source': 'issuer_bank',
            'gw_error_step': 'payment_authentication', 'error_reason': 'insufficient_funds'}
    wrong_step = {'upi': 'payment_authorization', 'card': 'payment_response', 'netbanking': 'card_enrollment_check'}[method]
    out = [
        {**base, 'gw_error_step': wrong_step},                                                                    # a step from another method's flow
        {**base, 'gw_error_code': 'GATEWAY_ERROR'},                                                                # insufficient_funds is never a GATEWAY_ERROR
        {**base, 'gw_error_code': 'SERVER_ERROR'},                                                                 # SERVER_ERROR goes only with server_error
        {**base, 'error_reason': 'Customer abandoned authentication without completing the payment.'},          # not a documented reason
        {**base, 'instrument_hint': 'na' if method == 'card' else 'valid'},                                      # an instrument hint from the other family
    ]
    out.append({**base, 'instrument_hint': 'expired', 'method': 'card'} if method == 'card' else {**base, 'error_reason': 'card_expired', 'gw_error_code': 'BAD_REQUEST_ERROR'})
    return out


def records() -> list:
    out = []
    k = 0
    for method in R.METHODS:
        groups = {
            'CLEAR_AND_CONSISTENT': [(dict(zip(R.OBSERVABLE, r)) | {'method': method}, R.clear_reason_cause(method, r[4])) for r in _pick(R.scenarios(method, 'clear'), PER_ROUTE['CLEAR_AND_CONSISTENT'])],
            'MISSING_OR_GENERIC': [(dict(zip(R.OBSERVABLE, r)) | {'method': method}, _cause_of(method, r))
                                   for r in _pick([r for r in R.scenarios(method) if R.clear_reason_cause(method, r[4]) is None and r[0] != 'expired'], PER_ROUTE['MISSING_OR_GENERIC'])],
            'CONTRADICTORY': [(c, 'BD_INSUFFICIENT_FUNDS') for c in _contradictions(method)[:PER_ROUTE['CONTRADICTORY']]],
        }
        for route, items in groups.items():
            for info, cause in items:
                k += 1
                fid = f'fx_{method[:3]}_{route[:3].lower()}_{k:03d}'
                out.append({'payment_id': fid, 'expected_route': route, 'hidden_cause': cause, 'customer_id': C.CUSTOMER_IDS[k % len(C.CUSTOMER_IDS)],
                            'amount_minor': C.AMOUNTS_RUPEES[k % len(C.AMOUNTS_RUPEES)] * 100, 'issuer_id': C.ISSUERS[k % len(C.ISSUERS)],
                            'time_id': C.FAILURE_TIME_IDS[k % len(C.FAILURE_TIME_IDS)], **info})
    return out


def by_id() -> dict:
    return {r['payment_id']: r for r in records()}
