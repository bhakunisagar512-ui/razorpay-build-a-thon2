"""Scenario configuration and emission tables for the sequences@v4 generator.

PROVENANCE TAGS used in comments:
  [documented] taken from Razorpay's public documentation (strings, sections, flow steps).
  [NPCI]       order of magnitude informed by NPCI UPI remitter-bank statistics
               (eval/data/npci_upi_summary.json). UPI payer banks ONLY; never applied to
               cards or netbanking.
  [assumed]    our own modelling choice. No public source gives the real value. Anything
               tagged this way must not be quoted as a real-world statistic.
"""
from dataclasses import dataclass

from services.app.domain.failure_vocab import BAD_REQUEST_REASONS, GATEWAY_REASONS

CAUSES = ('BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT',
          'CARD_EXPIRED', 'RISK_BLOCKED')


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    # --- business-decline intensity [NPCI for UPI level; assumed for card / netbanking]
    bd_scale: float            # multiplies auth-friction and risk hazards
    # --- how often the provider surfaces only a generic reason [assumed; docs only say it exists]
    generic_share: float
    reason_missing: float      # error_reason absent from the payload [assumed]
    reason_noise: float        # reason replaced by an unrelated documented reason [assumed]
    # --- infrastructure incidents [assumed shapes; NPCI motivates heavy-tailed bank-level spikes]
    outage_rate_per_week: float   # expected episodes per issuer / PSP per week
    outage_severity: tuple        # Beta(a, b) for peak failure probability
    outage_median_minutes: float
    network_incidents_per_week: float
    # --- telemetry quality [assumed]
    telemetry_drop: float         # attempt never reaches the feature store
    latency_missing_on_timeout: float
    downtime_detect: float        # P(Razorpay downtime notice is issued | real outage) [assumed]
    downtime_delay_minutes: float
    false_alarms_per_week: float
    # --- instrument events [assumed]
    reissue_probability: float    # P(card is reissued during the simulated horizon)
    # --- retry behaviour [assumed]
    retry_scale: float


SCENARIOS_V4 = {
    'calm': ScenarioConfig(
        'calm', bd_scale=.8, generic_share=.15, reason_missing=.01, reason_noise=.01,
        outage_rate_per_week=.35, outage_severity=(2., 3.), outage_median_minutes=25.,
        network_incidents_per_week=.15, telemetry_drop=.01, latency_missing_on_timeout=.2,
        downtime_detect=.90, downtime_delay_minutes=6., false_alarms_per_week=.5,
        reissue_probability=.03, retry_scale=1.0),
    'typical': ScenarioConfig(
        'typical', bd_scale=1.0, generic_share=.30, reason_missing=.03, reason_noise=.03,
        outage_rate_per_week=1.0, outage_severity=(2., 2.), outage_median_minutes=40.,
        network_incidents_per_week=.4, telemetry_drop=.03, latency_missing_on_timeout=.4,
        downtime_detect=.75, downtime_delay_minutes=8., false_alarms_per_week=1.,
        reissue_probability=.05, retry_scale=1.0),
    'stressed': ScenarioConfig(
        'stressed', bd_scale=1.35, generic_share=.50, reason_missing=.08, reason_noise=.06,
        outage_rate_per_week=2.5, outage_severity=(2.5, 1.5), outage_median_minutes=70.,
        network_incidents_per_week=1.0, telemetry_drop=.08, latency_missing_on_timeout=.6,
        downtime_detect=.55, downtime_delay_minutes=12., false_alarms_per_week=2.5,
        reissue_probability=.08, retry_scale=1.15),
}

# [assumed] fraction of generic-reason substitution per cause: issuers rarely explain risk declines.
GENERIC_MULTIPLIER = {
    'BD_INSUFFICIENT_FUNDS': .8, 'BD_USER_ABANDONED_OTP': .25, 'TD_ISSUER_DOWN': .8,
    'TD_PSP_TIMEOUT': .5, 'CARD_EXPIRED': .7, 'RISK_BLOCKED': 1.3,
}
# [documented] which strings are generic; [assumed] how they split.
GENERIC_BY_METHOD = {
    'card': {'payment_failed': .45, 'card_declined': .30, 'payment_declined': .15, 'debit_declined': .10},
    'upi': {'payment_failed': .60, 'payment_declined': .30, 'debit_declined': .10},
    'netbanking': {'payment_failed': .70, 'payment_declined': .20, 'debit_declined': .10},
}

# [documented] reason strings and their sections; [assumed] every weight.
# payment_timed_out / payment_cancelled deliberately appear under several causes: the docs
# describe them as both "customer did not finish" and "bank/gateway did not respond".
SPECIFIC_REASONS = {
    'BD_INSUFFICIENT_FUNDS': {
        'card': {'insufficient_funds': .70, 'transaction_limit_exceeded': .17, 'transaction_daily_limit_exceeded': .13},
        'upi': {'insufficient_funds': .68, 'transaction_limit_exceeded': .12, 'transaction_frequency_limit_exceeded': .12,
                'transaction_daily_limit_exceeded': .08},
        'netbanking': {'insufficient_funds': .75, 'transaction_limit_exceeded': .25},
    },
    'BD_USER_ABANDONED_OTP': {
        'card': {'authentication_failed': .30, 'incorrect_otp': .14, 'otp_expired': .12, 'payment_cancelled': .17,
                 'payment_timed_out': .15, 'otp_attempts_exceeded': .06, 'incorrect_cvv': .06},
        'upi': {'payment_cancelled': .25, 'payment_timed_out': .25, 'incorrect_pin': .15, 'pin_attempts_exceeded': .05,
                'authentication_failed': .15, 'payment_collect_request_expired': .15},
        'netbanking': {'authentication_failed': .35, 'payment_cancelled': .30, 'payment_timed_out': .20,
                       'incorrect_otp': .10, 'otp_expired': .05},
    },
    'TD_ISSUER_DOWN': {
        'card': {'bank_technical_error': .33, 'issuer_technical_error': .24, 'bank_not_available': .10,
                 'payment_cancelled': .07, 'authentication_failed': .08, 'payment_timed_out': .12,
                 'request_timed_out': .06},
        'upi': {'bank_technical_error': .28, 'bank_not_available': .24, 'issuer_technical_error': .10,
                'bank_cutoff_in_progress': .05, 'payment_timed_out': .17, 'request_timed_out': .16},
        'netbanking': {'bank_not_available': .33, 'bank_technical_error': .28, 'bank_cutoff_in_progress': .09,
                       'payment_timed_out': .16, 'payment_cancelled': .14},
    },
    'TD_PSP_TIMEOUT': {
        'card': {'gateway_technical_error': .33, 'request_timed_out': .25, 'invalid_response_from_gateway': .12,
                 'payment_declined_due_to_high_traffic': .10, 'payment_timed_out': .14, 'server_error': .06},
        'upi': {'upi_app_technical_error': .28, 'psp_app_not_available': .14, 'psp_not_available': .10,
                'request_timed_out': .20, 'payment_timed_out': .17, 'gateway_technical_error': .11},
        'netbanking': {'gateway_technical_error': .32, 'request_timed_out': .26, 'invalid_response_from_gateway': .12,
                       'payment_timed_out': .18, 'server_error': .12},
    },
    'CARD_EXPIRED': {'card': {'card_expired': .80, 'card_declined': .12, 'debit_instrument_blocked': .08}},
    'RISK_BLOCKED': {
        'card': {'payment_risk_check_failed': .55, 'debit_instrument_blocked': .17, 'card_declined': .14,
                 'payment_declined': .14},
        'upi': {'payment_risk_check_failed': .38, 'transaction_on_vpa_restricted': .26, 'payment_declined': .21,
                'debit_declined': .15},
        'netbanking': {'payment_risk_check_failed': .52, 'payment_declined': .28, 'debit_declined': .20},
    },
}

# [documented] vocab; [assumed] weights. Deliberately overlapping across causes.
_CARD = {
    'BD_INSUFFICIENT_FUNDS': ({'payment_authorization': .84, 'payment_authentication': .10, 'payment_capture': .06},
                              {'issuer_bank': .55, 'customer': .35, 'gateway': .10}),
    'BD_USER_ABANDONED_OTP': ({'payment_authentication': .84, 'payment_authorization': .10, 'payment_initiation': .06},
                              {'customer': .60, 'issuer_bank': .28, 'gateway': .12}),
    'TD_ISSUER_DOWN': ({'payment_authorization': .55, 'payment_authentication': .33, 'card_enrollment_check': .12},
                       {'issuer_bank': .58, 'gateway': .37, 'internal': .05}),
    'TD_PSP_TIMEOUT': ({'payment_initiation': .30, 'payment_authorization': .28, 'payment_authentication': .12,
                        'payment_capture': .12, 'card_enrollment_check': .18},
                       {'gateway': .58, 'internal': .24, 'issuer_bank': .18}),
    'CARD_EXPIRED': ({'card_enrollment_check': .30, 'payment_authorization': .58, 'payment_initiation': .12},
                     {'issuer_bank': .50, 'customer': .28, 'gateway': .22}),
    'RISK_BLOCKED': ({'payment_authorization': .48, 'payment_authentication': .20, 'payment_initiation': .16,
                      'card_enrollment_check': .16},
                     {'issuer_bank': .44, 'gateway': .30, 'internal': .26}),
}
_UPI = {
    'BD_INSUFFICIENT_FUNDS': ({'payment_debit_response': .72, 'payment_debit_request': .12, 'payment_response': .16},
                              {'issuer_bank': .70, 'customer': .14, 'network': .16}),
    'BD_USER_ABANDONED_OTP': ({'payment_authentication': .70, 'payment_creation': .15, 'payment_response': .15},
                              {'customer': .46, 'customer_psp': .40, 'network': .14}),
    'TD_ISSUER_DOWN': ({'payment_debit_request': .40, 'payment_debit_response': .40, 'payment_status_response': .20},
                       {'issuer_bank': .50, 'network': .28, 'gateway': .22}),
    'TD_PSP_TIMEOUT': ({'payment_request': .30, 'payment_creation': .22, 'payment_response': .28,
                        'payment_initiation': .20},
                       {'customer_psp': .48, 'gateway': .28, 'network': .24}),
    'RISK_BLOCKED': ({'payment_debit_response': .50, 'payment_request': .20, 'payment_authentication': .15,
                      'payment_initiation': .15},
                     {'issuer_bank': .45, 'network': .28, 'internal': .15, 'beneficiary_bank': .12}),
}
_NB = {
    'BD_INSUFFICIENT_FUNDS': ({'payment_authorization': .86, 'payment_authentication': .14},
                              {'issuer_bank': .62, 'customer': .38}),
    'BD_USER_ABANDONED_OTP': ({'payment_authentication': .88, 'payment_initiation': .12},
                              {'customer': .62, 'issuer_bank': .38}),
    'TD_ISSUER_DOWN': ({'payment_authentication': .46, 'payment_authorization': .40, 'payment_initiation': .14},
                       {'issuer_bank': .70, 'internal': .30}),
    'TD_PSP_TIMEOUT': ({'payment_initiation': .46, 'payment_authentication': .30, 'payment_authorization': .24},
                       {'internal': .52, 'issuer_bank': .48}),
    'RISK_BLOCKED': ({'payment_authorization': .55, 'payment_authentication': .30, 'payment_initiation': .15},
                     {'issuer_bank': .55, 'internal': .30, 'business': .15}),
}
STEP_SOURCE = {'card': _CARD, 'upi': _UPI, 'netbanking': _NB}

# [assumed] seconds from payment creation until the failure is reported, lognormal (median, sigma).
DURATION = {
    'BD_INSUFFICIENT_FUNDS': (35., .6), 'BD_USER_ABANDONED_OTP': (150., .9), 'TD_ISSUER_DOWN': (18., .8),
    'TD_PSP_TIMEOUT': (30., .9), 'CARD_EXPIRED': (9., .5), 'RISK_BLOCKED': (22., .7),
}
SUCCESS_DURATION = (40., .6)

# [assumed] stage-race scale: smaller = fails earlier in the flow, so wins when causes overlap.
STAGE_SCALE = {
    'TD_PSP_TIMEOUT': 1.0, 'CARD_EXPIRED': 1.6, 'RISK_BLOCKED': 3.0, 'TD_ISSUER_DOWN': 4.0,
    'BD_USER_ABANDONED_OTP': 5.0, 'BD_INSUFFICIENT_FUNDS': 6.0,
}

# [NPCI] UPI ranges the 'typical' scenario should land inside (checked by tests; see npci_upi_summary.json).
UPI_ANCHORS = {'bd_pct_range': (3.5, 16.0), 'td_pct_range': (0.1, 2.5)}

REASON_POOL = tuple(sorted(BAD_REQUEST_REASONS | GATEWAY_REASONS))
