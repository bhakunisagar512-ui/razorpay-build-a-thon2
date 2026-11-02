"""Razorpay failure vocabulary and our grouping of it. No simulation parameters here.

Strings come from Razorpay's public documentation (List of Errors, Cards Error Codes,
Payment-method error parameters, payment.downtime webhooks), read in Oct 2026 through
a summarising fetch tool; re-verify before relying on an exact string in production.

Phase 1 re-check against razorpay.com/docs/errors (Oct 2026): SOURCES and STEPS are confirmed per
method. The reason strings and the GATEWAY_ERROR / SERVER_ERROR codes were NOT found on the pages
reachable (the only reason shown there is `invalid_otp`, not `incorrect_otp`); treat them as
unverified. NOT documented by Razorpay: REASON_FAMILY. It is our own
reading of the descriptions, and several reasons are genuinely ambiguous (for example
payment_timed_out means "customer did not finish" OR "no gateway response"). The
mapping is a coarse feature, never a ground-truth label.
"""

BAD_REQUEST_REASONS = frozenset({
    'authentication_failed', 'bank_technical_error', 'card_expired', 'debit_instrument_blocked',
    'incorrect_card_expiry_date', 'incorrect_cvv', 'incorrect_otp', 'incorrect_pin', 'insufficient_funds',
    'otp_attempts_exceeded', 'otp_expired', 'payment_cancelled', 'payment_failed', 'payment_risk_check_failed',
    'payment_timed_out', 'pin_attempts_exceeded', 'server_error', 'transaction_daily_limit_exceeded',
    'transaction_frequency_limit_exceeded', 'transaction_limit_exceeded', 'transaction_on_vpa_restricted',
    'upi_app_technical_error', 'card_not_enrolled',
})
GATEWAY_REASONS = frozenset({
    'authentication_failed', 'bank_cutoff_in_progress', 'bank_not_available', 'bank_technical_error',
    'card_declined', 'credit_limit_exceeded', 'debit_declined', 'debit_instrument_blocked',
    'debit_instrument_inactive', 'gateway_technical_error', 'invalid_response_from_gateway',
    'issuer_technical_error', 'payment_cancelled', 'payment_collect_request_expired', 'payment_declined',
    'payment_declined_due_to_high_traffic', 'payment_failed', 'payment_risk_check_failed',
    'payment_session_expired', 'payment_timed_out', 'psp_app_not_available', 'psp_not_available',
    'request_timed_out', 'server_error', 'transaction_daily_count_exceeded', 'transaction_daily_limit_exceeded',
    'transaction_frequency_limit_exceeded',
})
DOCUMENTED_REASONS = BAD_REQUEST_REASONS | GATEWAY_REASONS
# Documented codes: BAD_REQUEST_ERROR (Bad Request section), GATEWAY_ERROR (Gateway section), SERVER_ERROR.
ERROR_CODES = ('BAD_REQUEST_ERROR', 'GATEWAY_ERROR', 'SERVER_ERROR')

SOURCES = {
    'card': ('customer', 'business', 'internal', 'gateway', 'issuer_bank'),
    'upi': ('customer', 'business', 'internal', 'customer_psp', 'gateway', 'network', 'issuer_bank', 'beneficiary_bank'),
    'netbanking': ('customer', 'business', 'internal', 'issuer_bank'),
}
# Documented step order for the flow; used for the step_ordinal feature.
STEPS = {
    'card': ('payment_initiation', 'card_enrollment_check', 'payment_authentication', 'payment_authorization',
             'payment_capture'),
    'netbanking': ('payment_initiation', 'payment_authentication', 'payment_authorization'),
    'upi': ('payment_initiation', 'payment_creation', 'payment_authentication', 'payment_request',
            'payment_authentication_request', 'payment_authentication_response',
            'payment_request_beneficiary_details', 'payment_response_beneficiary_details',
            'payment_debit_request', 'payment_debit_response', 'payment_credit_request',
            'payment_credit_response', 'payment_status_request', 'payment_status_response', 'payment_response'),
}
# Coarse, method-comparable progression through the flow (our grouping of the documented steps).
_STAGE = {'payment_initiation': 0, 'payment_creation': 0, 'card_enrollment_check': 1,
          'payment_authentication': 2, 'payment_authentication_request': 2, 'payment_authentication_response': 2,
          'payment_authorization': 3, 'payment_request': 3, 'payment_request_beneficiary_details': 3,
          'payment_response_beneficiary_details': 3, 'payment_debit_request': 3, 'payment_debit_response': 3,
          'payment_credit_request': 4, 'payment_credit_response': 4, 'payment_capture': 4,
          'payment_status_request': 4, 'payment_status_response': 4, 'payment_response': 4}

FLOWS = ('card_3ds', 'card_native_otp', 'upi_intent', 'upi_collect', 'netbanking')
GENERIC_REASONS = frozenset({'payment_failed', 'card_declined', 'payment_declined', 'debit_declined'})
TIMEOUT_REASONS = frozenset({'payment_timed_out', 'request_timed_out', 'mandate_creation_timeout'})
SESSION_TIMEOUT_S = 600  # Cards Error Codes: customer exceeded the payment time limit (typically 10 minutes).

REASON_FAMILY = {}
for _family, _reasons in {
    'customer_auth': ('authentication_failed', 'incorrect_otp', 'otp_expired', 'otp_attempts_exceeded',
                      'incorrect_pin', 'pin_attempts_exceeded', 'incorrect_cvv'),
    'customer_session': ('payment_cancelled', 'payment_timed_out', 'payment_collect_request_expired',
                         'payment_session_expired'),
    'funds_limits': ('insufficient_funds', 'transaction_limit_exceeded', 'transaction_daily_limit_exceeded',
                     'transaction_frequency_limit_exceeded', 'credit_limit_exceeded',
                     'transaction_daily_count_exceeded'),
    'bank_technical': ('bank_technical_error', 'issuer_technical_error', 'bank_not_available',
                       'bank_cutoff_in_progress'),
    'gateway_technical': ('gateway_technical_error', 'request_timed_out', 'invalid_response_from_gateway',
                          'payment_declined_due_to_high_traffic', 'psp_not_available', 'psp_app_not_available',
                          'upi_app_technical_error', 'server_error'),
    'instrument_state': ('card_expired', 'debit_instrument_blocked', 'debit_instrument_inactive', 'card_not_enrolled',
                         'transaction_on_vpa_restricted', 'incorrect_card_expiry_date'),
    'risk': ('payment_risk_check_failed',),
    'generic': tuple(sorted(GENERIC_REASONS)),
}.items():
    for _reason in _reasons:
        REASON_FAMILY[_reason] = _family
REASON_FAMILIES = ('customer_auth', 'customer_session', 'funds_limits', 'bank_technical', 'gateway_technical',
                   'instrument_state', 'risk', 'generic', 'other', 'missing')


def reason_family(reason):
    if reason is None or reason == '':
        return 'missing'
    return REASON_FAMILY.get(reason, 'other')


def step_ordinal(step):
    """Coarse progress through the payment flow, 0 (initiation) .. 4 (capture/status). None if unknown."""
    return _STAGE.get(step)
