# Phase 4, Step 1: feature-availability audit

Written for: the project owner, ahead of any decision to move a model toward real Razorpay traffic.

**Bottom line.** Of the 57 Phase 3 candidate features, 15 are direct Razorpay fields (or pure functions of them), 29 can be computed from the merchant's own payment history, 5 depend on a failure duration that Razorpay does not provide as a field, and 8 have no documented Razorpay source at all. The two features that carried most of the Phase 3 gain are not equally trustworthy: `error_reason` is a real, documented field, but `attempt_duration_s` is not. The deployment-compatible set therefore **excludes durations and gateway-scoped features** (43 features, `deployment_core`). A 48-feature `deployment_with_timing` variant is benchmarked as a conditional upgrade.

## How this was checked

Sources: Razorpay's public docs for the Payment entity (fetch payment), the `payment.failed` webhook payload and the `payment.downtime.*` payloads, read on 2026-10-30. I could not test against a live Razorpay account, so "available" means "documented", not "observed in production traffic".

| Question | Finding |
|---|---|
| Does the Payment entity carry `error_code`, `error_source`, `error_step`, `error_reason`? | Yes, all four strings. Samples show them as `null` on successful payments. |
| Is there a failure timestamp or attempt-duration field? | **No.** `created_at` is the payment's creation time. |
| What does the webhook add? | A top-level `created_at` for the event. The docs do not define it precisely; in samples it is slightly later than the entity's own `created_at`. |
| Card issuer / bank / UPI handle? | `card.issuer`, `bank`, `vpa` and `upi.flow` are documented. Coverage depends on the method. |
| Card expiry? | Not on the Payment entity (only `last4`, `network`, `type`, `issuer`). Only a merchant that stores token expiry can know a card is expired. |
| Which gateway/acquirer handled the payment? | Not exposed. |
| Downtime signals? | `payment.downtime.started/updated/resolved` webhooks with method, severity and instrument. Phase 3 dropped this group, so no deployed feature depends on it. |

## The three headline features

| Feature | Real source | Verdict |
|---|---|---|
| `error_reason` | `payment.error_reason` | **Available.** Many documented values are catch-alls (`payment_failed`, `card_declined`, ...). In our simulator the reason is emitted from the true cause through tables that we wrote, so its power here is partly a property of the simulator. Real reason strings may be less (or differently) informative. |
| `reason_family` | function of `error_reason` | Available, but the family grouping is **our** reading of the vocabulary, not a Razorpay field. |
| `attempt_duration_s` | webhook `created_at` minus payment `created_at` (approximation) | **Not verified.** Includes webhook delivery delay and queueing, and the semantics of the webhook timestamp are undocumented. In the simulator, duration is drawn from cause-specific distributions (`DURATION` in `eval/world_config_v4.py`, tagged `[assumed]`). Treat any gain from it as an upper bound until the timestamp difference is measured on real data. |

Features computed from these inherit their status: `duration_vs_timeout_ratio`, `latency_vs_issuer_p50`, `current_latency_ms` and `issuer_latency_p95_ms_5m` are timing-derived; `is_generic_reason`, `instrument_prev_reason`, `same_reason_as_prev`, `instrument_distinct_reasons_24h`, `issuer_timeout_rate_5m` and `issuer_tech_reason_share_5m` are functions of `error_reason`.

## Leakage review

| Check | Result |
|---|---|
| Label or latent simulator state in the feature columns (`true_cause`, `n_causes_fired`, `retry_index`, outage calendar) | None are in `FEATURES_V4`; asserted by `tests/test_sequences_v4.py`. |
| Future information in rolling features | Point-in-time replay; a truncated-replay check recomputes features from the past only and found 0 mismatches in Phase 3 and in the fresh Phase 4 test worlds. |
| Current row's outcome in its own window | `build()` runs before `observe()`; windows use `observed_at` strictly earlier. |
| Identifiers that could memorise a world | `payment_id`, `order_id`, `customer_id`, `instrument_id`, `world_*` are not model inputs. Worlds differ by seed, so a model cannot memorise customers or outage calendars across splits. |
| **Label-construction leakage** (the real risk) | In this project the "true cause" is *generated first* and `error_reason` is then emitted from it. In production no such label exists: root cause would have to be assigned by an analyst or a rule, and if that rule reads `error_reason`, a model will appear to "predict" a label that is a function of its input. Phase 3/4 accuracy says how well the model recovers the simulator's causal story, not how well it names real causes. |

No row-level leakage was found. The material risk is the label-construction one, and it cannot be removed inside a simulation.

## Deployment-compatible sets

* `deployment_core` (**43 features, the Phase 4 default**): direct fields plus merchant-history features. Excludes every `simulated_only` feature, every timing feature and the downtime group.
* `deployment_with_timing` (48): adds the five timing features. Conditional: use only after `webhook.created_at - payment.created_at` has been compared with a trusted duration on real traffic.
* Excluded in both: `psp_id` and all PSP-scoped windows (Razorpay does not expose the acquirer), `payment_flow` (the card 3DS/native-OTP sub-flow is not exposed; only UPI `flow` is), `matched_downtime_severity` (dropped in Phase 3).

Merchant-history features additionally need: stable customer identity (customer_id or a hashed contact), instrument identity (`card_id`/`token_id`/`vpa`), retention of at least 14 days of payments, and enough per-issuer traffic for 5-minute windows to be meaningful. A small merchant will see these windows empty or noisy.

## Full classification

| Feature | Class | Source | Core | +timing | Note |
|---|---|---|---|---|---|
| `gw_error_code` | direct | payment.error_code | yes | yes |  |
| `gw_error_source` | direct | payment.error_source | yes | yes |  |
| `gw_error_step` | direct | payment.error_step | yes | yes |  |
| `method` | direct | payment.method | yes | yes |  |
| `instrument_hint` | merchant_history | merchant-held token/card expiry | yes | yes | The Payment entity returns no expiry; only a merchant storing token expiry can set "expired". Missing is encoded as "na". |
| `issuer_id` | direct | payment.card.issuer / payment.bank / UPI VPA handle | yes | yes | Coverage differs by method. |
| `psp_id` | simulated_only | none | - | - | Razorpay does not expose the acquirer/gateway that handled a card or netbanking payment. |
| `amount_minor` | direct | payment.amount | yes | yes |  |
| `log_amount_minor` | direct | payment.amount | yes | yes |  |
| `hour_ist` | direct | payment.created_at | yes | yes |  |
| `weekday_ist` | direct | payment.created_at | yes | yes |  |
| `customer_attempts_7d` | merchant_history | own payments by customer_id / hashed contact | yes | yes |  |
| `customer_successes_7d` | merchant_history | own payments by customer_id / hashed contact | yes | yes |  |
| `customer_failure_rate_7d` | merchant_history | own payments by customer_id / hashed contact | yes | yes |  |
| `customer_last_success_age_s` | merchant_history | own payments by customer_id / hashed contact | yes | yes |  |
| `amount_to_customer_mean_7d` | merchant_history | own payments by customer_id / hashed contact | yes | yes |  |
| `order_prior_attempts_7d` | merchant_history | own payments by order_id | yes | yes |  |
| `issuer_attempts_5m` | merchant_history | own traffic per issuer | yes | yes | Thin traffic makes short windows noisy for small merchants. |
| `issuer_failure_rate_5m` | merchant_history | own traffic per issuer | yes | yes |  |
| `issuer_latency_p95_ms_5m` | derived_timing | webhook created_at - payment created_at | - | yes |  |
| `psp_attempts_5m` | simulated_only | none | - | - |  |
| `psp_failure_rate_5m` | simulated_only | none | - | - |  |
| `psp_latency_p95_ms_5m` | simulated_only | none | - | - |  |
| `method_attempts_5m` | merchant_history | own traffic per method | yes | yes |  |
| `method_failure_rate_5m` | merchant_history | own traffic per method | yes | yes |  |
| `current_latency_ms` | derived_timing | webhook created_at - payment created_at | - | yes |  |
| `error_reason` | direct | payment.error_reason | yes | yes | Documented values include catch-alls (payment_failed, card_declined, ...). Our label simulator derives it from the cause, so its predictive value here is assumption-driven. |
| `reason_family` | direct | function of error_reason | yes | yes | The family mapping is our own reading, not a Razorpay field. |
| `is_generic_reason` | direct | function of error_reason | yes | yes |  |
| `step_ordinal` | direct | function of error_step | yes | yes | The ordering is our own reading of the documented steps. |
| `attempt_duration_s` | derived_timing | webhook created_at - payment created_at | - | yes | No documented failure timestamp or duration. Includes webhook delivery lag; the meaning of the webhook created_at is not defined in the docs. Simulated from cause-specific distributions. |
| `duration_vs_timeout_ratio` | derived_timing | function of attempt_duration_s | - | yes |  |
| `payment_flow` | simulated_only | upi.flow for UPI only | - | - | Card 3DS vs native-OTP sub-flow is not exposed; the simulator emits it. |
| `latency_vs_issuer_p50` | derived_timing | function of latency_ms | - | yes |  |
| `instrument_consecutive_failures` | merchant_history | own payments by card_id / token_id / vpa | yes | yes |  |
| `instrument_attempts_24h` | merchant_history | own payments by card_id / token_id / vpa | yes | yes |  |
| `instrument_failure_rate_24h` | merchant_history | own payments by card_id / token_id / vpa | yes | yes |  |
| `instrument_last_success_age_s` | merchant_history | own payments by card_id / token_id / vpa | yes | yes |  |
| `instrument_prev_reason` | merchant_history | previous error_reason on the same instrument | yes | yes |  |
| `same_reason_as_prev` | merchant_history | previous error_reason on the same instrument | yes | yes |  |
| `instrument_distinct_reasons_24h` | merchant_history | previous error_reason on the same instrument | yes | yes |  |
| `issuer_failure_rate_1h` | merchant_history | own traffic per issuer | yes | yes |  |
| `issuer_failure_delta` | merchant_history | own traffic per issuer | yes | yes |  |
| `issuer_failure_vs_baseline_7d` | merchant_history | own traffic per issuer | yes | yes |  |
| `issuer_timeout_rate_5m` | merchant_history | own traffic per issuer; error_reason timeouts | yes | yes |  |
| `issuer_tech_reason_share_5m` | merchant_history | own traffic per issuer; error_reason family | yes | yes |  |
| `psp_failure_rate_1h` | simulated_only | none | - | - |  |
| `psp_failure_delta` | simulated_only | none | - | - |  |
| `psp_timeout_rate_5m` | simulated_only | none | - | - |  |
| `issuer_method_failure_rate_5m` | merchant_history | own traffic per issuer and method | yes | yes |  |
| `global_failure_rate_5m` | merchant_history | own traffic | yes | yes |  |
| `secs_since_prev_attempt` | merchant_history | own payments by customer | yes | yes |  |
| `customer_methods_tried_24h` | merchant_history | own payments by customer | yes | yes |  |
| `amount_vs_max_success_amount` | merchant_history | own payments by customer | yes | yes |  |
| `customer_success_rate_14d` | merchant_history | own payments by customer | yes | yes |  |
| `matched_downtime_severity` | direct | payment.downtime.* webhooks (method, severity, instrument) | - | - | Requires webhook subscription; Phase 3 dropped this group. |
| `days_to_month_end` | direct | payment.created_at | yes | yes |  |
