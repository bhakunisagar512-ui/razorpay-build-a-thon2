"""Phase 4 feature-availability audit and the deployment-compatible feature sets.

Each of the 57 Phase 3 candidates is classified by where its value could come from in a
real Razorpay integration. Sources were checked against Razorpay's public docs
(Payment entity, payment.failed and payment.downtime.* webhook payloads) on 2026-10-30.
This module is data only: it does not touch the serving classifier.

Availability classes
  direct            a documented Razorpay payment / webhook field (or a pure function of one)
  merchant_history  computed from the merchant's own stored payments (needs retention and a
                    stable customer / instrument identifier)
  derived_timing    needs a duration. The Payment entity has no failure timestamp or duration
                    field; it could be approximated as webhook `created_at` minus payment
                    `created_at`, which also contains webhook delivery lag. UNVERIFIED.
  simulated_only    no documented Razorpay source (gateway/acquirer identity, card checkout
                    sub-flow, internal latency)
"""
from services.app.domain.features import FEATURES
from services.app.domain.features_v4 import FEATURES_V4

DIRECT, HISTORY, TIMING, SIMULATED = 'direct', 'merchant_history', 'derived_timing', 'simulated_only'


def _a(klass, source, note=''):
    return {'class': klass, 'source': source, 'note': note}


AVAILABILITY = {
    # ---- the 5 baseline fields
    'gw_error_code': _a(DIRECT, 'payment.error_code'),
    'gw_error_source': _a(DIRECT, 'payment.error_source'),
    'gw_error_step': _a(DIRECT, 'payment.error_step'),
    'method': _a(DIRECT, 'payment.method'),
    'instrument_hint': _a(HISTORY, 'merchant-held token/card expiry',
                          'The Payment entity returns no expiry; only a merchant storing token expiry can set "expired". '
                          'Missing is encoded as "na".'),
    'issuer_id': _a(DIRECT, 'payment.card.issuer / payment.bank / UPI VPA handle', 'Coverage differs by method.'),
    'psp_id': _a(SIMULATED, 'none', 'Razorpay does not expose the acquirer/gateway that handled a card or netbanking payment.'),
    # ---- original numeric 19
    'amount_minor': _a(DIRECT, 'payment.amount'),
    'log_amount_minor': _a(DIRECT, 'payment.amount'),
    'hour_ist': _a(DIRECT, 'payment.created_at'),
    'weekday_ist': _a(DIRECT, 'payment.created_at'),
    'customer_attempts_7d': _a(HISTORY, 'own payments by customer_id / hashed contact'),
    'customer_successes_7d': _a(HISTORY, 'own payments by customer_id / hashed contact'),
    'customer_failure_rate_7d': _a(HISTORY, 'own payments by customer_id / hashed contact'),
    'customer_last_success_age_s': _a(HISTORY, 'own payments by customer_id / hashed contact'),
    'amount_to_customer_mean_7d': _a(HISTORY, 'own payments by customer_id / hashed contact'),
    'order_prior_attempts_7d': _a(HISTORY, 'own payments by order_id'),
    'issuer_attempts_5m': _a(HISTORY, 'own traffic per issuer', 'Thin traffic makes short windows noisy for small merchants.'),
    'issuer_failure_rate_5m': _a(HISTORY, 'own traffic per issuer'),
    'issuer_latency_p95_ms_5m': _a(TIMING, 'webhook created_at - payment created_at'),
    'psp_attempts_5m': _a(SIMULATED, 'none'),
    'psp_failure_rate_5m': _a(SIMULATED, 'none'),
    'psp_latency_p95_ms_5m': _a(SIMULATED, 'none'),
    'method_attempts_5m': _a(HISTORY, 'own traffic per method'),
    'method_failure_rate_5m': _a(HISTORY, 'own traffic per method'),
    'current_latency_ms': _a(TIMING, 'webhook created_at - payment created_at'),
    # ---- Phase 3 additions
    'error_reason': _a(DIRECT, 'payment.error_reason', 'Documented values include catch-alls (payment_failed, card_declined, ...). '
                       'Our label simulator derives it from the cause, so its predictive value here is assumption-driven.'),
    'reason_family': _a(DIRECT, 'function of error_reason', 'The family mapping is our own reading, not a Razorpay field.'),
    'is_generic_reason': _a(DIRECT, 'function of error_reason'),
    'step_ordinal': _a(DIRECT, 'function of error_step', 'The ordering is our own reading of the documented steps.'),
    'attempt_duration_s': _a(TIMING, 'webhook created_at - payment created_at',
                             'No documented failure timestamp or duration. Includes webhook delivery lag; the meaning of the '
                             'webhook created_at is not defined in the docs. Simulated from cause-specific distributions.'),
    'duration_vs_timeout_ratio': _a(TIMING, 'function of attempt_duration_s'),
    'payment_flow': _a(SIMULATED, 'upi.flow for UPI only', 'Card 3DS vs native-OTP sub-flow is not exposed; the simulator emits it.'),
    'latency_vs_issuer_p50': _a(TIMING, 'function of latency_ms'),
    'instrument_consecutive_failures': _a(HISTORY, 'own payments by card_id / token_id / vpa'),
    'instrument_attempts_24h': _a(HISTORY, 'own payments by card_id / token_id / vpa'),
    'instrument_failure_rate_24h': _a(HISTORY, 'own payments by card_id / token_id / vpa'),
    'instrument_last_success_age_s': _a(HISTORY, 'own payments by card_id / token_id / vpa'),
    'instrument_prev_reason': _a(HISTORY, 'previous error_reason on the same instrument'),
    'same_reason_as_prev': _a(HISTORY, 'previous error_reason on the same instrument'),
    'instrument_distinct_reasons_24h': _a(HISTORY, 'previous error_reason on the same instrument'),
    'issuer_failure_rate_1h': _a(HISTORY, 'own traffic per issuer'),
    'issuer_failure_delta': _a(HISTORY, 'own traffic per issuer'),
    'issuer_failure_vs_baseline_7d': _a(HISTORY, 'own traffic per issuer'),
    'issuer_timeout_rate_5m': _a(HISTORY, 'own traffic per issuer; error_reason timeouts'),
    'issuer_tech_reason_share_5m': _a(HISTORY, 'own traffic per issuer; error_reason family'),
    'psp_failure_rate_1h': _a(SIMULATED, 'none'),
    'psp_failure_delta': _a(SIMULATED, 'none'),
    'psp_timeout_rate_5m': _a(SIMULATED, 'none'),
    'issuer_method_failure_rate_5m': _a(HISTORY, 'own traffic per issuer and method'),
    'global_failure_rate_5m': _a(HISTORY, 'own traffic'),
    'secs_since_prev_attempt': _a(HISTORY, 'own payments by customer'),
    'customer_methods_tried_24h': _a(HISTORY, 'own payments by customer'),
    'amount_vs_max_success_amount': _a(HISTORY, 'own payments by customer'),
    'customer_success_rate_14d': _a(HISTORY, 'own payments by customer'),
    'matched_downtime_severity': _a(DIRECT, 'payment.downtime.* webhooks (method, severity, instrument)',
                                    'Requires webhook subscription; Phase 3 dropped this group.'),
    'days_to_month_end': _a(DIRECT, 'payment.created_at'),
}
assert set(AVAILABILITY) == set(FEATURES_V4), set(FEATURES_V4) ^ set(AVAILABILITY)

# The three fields that carried most of the Phase 3 gain, and everything that is computed from them.
HEADLINE_FEATURES = ('error_reason', 'reason_family', 'attempt_duration_s')
REASON_PAIR = ('error_reason', 'reason_family')
DURATION_FEATURES = ('attempt_duration_s',)
# Features that are functions of (or history of) the headline fields; dropped together in the "closure" ablation.
DERIVED_FROM_HEADLINE = ('is_generic_reason', 'instrument_prev_reason', 'same_reason_as_prev',
                         'instrument_distinct_reasons_24h', 'issuer_timeout_rate_5m', 'issuer_tech_reason_share_5m',
                         'psp_timeout_rate_5m', 'duration_vs_timeout_ratio', 'latency_vs_issuer_p50')


def of_class(*classes):
    return tuple(f for f in FEATURES_V4 if AVAILABILITY[f]['class'] in classes)


# Conservative deployment set: nothing simulated-only and nothing that needs an unverified duration.
DEPLOYMENT_CORE = tuple(f for f in FEATURES_V4 if AVAILABILITY[f]['class'] in (DIRECT, HISTORY)
                        and f != 'matched_downtime_severity')
# Conditional upgrade: also use webhook-timestamp-derived timing once it has been validated on real traffic.
# psp-scoped and flow features stay out because they have no real source at all.
DEPLOYMENT_WITH_TIMING = tuple(f for f in FEATURES_V4 if AVAILABILITY[f]['class'] in (DIRECT, HISTORY, TIMING)
                               and f != 'matched_downtime_severity')
assert not set(DEPLOYMENT_CORE) & set(of_class(SIMULATED, TIMING))
assert set(DEPLOYMENT_CORE) <= set(DEPLOYMENT_WITH_TIMING)
assert set(FEATURES) - set(DEPLOYMENT_WITH_TIMING) == {f for f in FEATURES if AVAILABILITY[f]['class'] == SIMULATED}


def without(features, *drops):
    drop = {d for group in drops for d in group}
    return tuple(f for f in features if f not in drop)
