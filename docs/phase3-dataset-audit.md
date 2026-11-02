# Phase 3, Steps 1–2: Dataset realism audit and payment-failure research

Status: **findings and proposal only. No generator, feature, model or serving code was changed.**
Measurements: [phase3-audit-metrics.json](phase3-audit-metrics.json), produced by
[eval/dataset_audit.py](../eval/dataset_audit.py) (sequences@v3, n=20,000, seed 42; stress scenarios n=4,000, seed 901).

---

## Part 1 — Audit of the existing synthetic data

### 1.1 How a failure is generated today

| Generator | Mechanism | Used by |
|---|---|---|
| `legacy_generate` | Sample the **cause first** from a fixed prior, then emit `(code, source, step)` from a fixed per-cause table; 5% of codes swapped at random. Customers are random IDs with no history. | Original 5-feature baseline |
| `sequences@v3` (`sequences.py`) | 3-day world. Customers carry persistent latent state (issuer, balance, `expired`, `risk`, `otp_friction`). Each attempt walks a **strict priority chain**: risk → expired (card only) → PSP outage → issuer outage → amount > balance → OTP friction. First hit is the one cause. Code is then drawn from the same per-cause table; 6% overwritten with the generic `GATEWAY_ERROR/gateway/authorization`. | 26-feature model, safety and binary experiments |
| `scenarios.py` (5 stress worlds) | Deterministic cohorts (`customer_number % 13`), one outage window, fixed 1:1 `ERRORS` table, 1-in-5 overwritten with the generic code. Written to be independent of the training generator. | Robustness and "independent world" confirmation |
| `sequences_v2` | Superseded version of v3 (kept for reproducibility). | — |

`features.py` builds the 26 features point-in-time. Reading the code, features are computed **before** the current row is observed, duplicates are ignored, and nothing from labels or latent state enters. I found **no direct target leakage in the feature builder**. The problems below are in the generator.

### 1.2 Findings

Severity: **High** = invalidates a claim we make; **Medium** = limits what the results can mean; **Low** = cosmetic or minor.

#### H1 — The model's most informative real-world field is missing, and the vocabulary is not Razorpay's
Razorpay's `payment.failed` payload carries `error_code`, `error_description`, `error_source`, `error_step` and **`error_reason`**. We model the first four of these only, with a simplified vocabulary:

| Field | Our generator emits | Razorpay documents |
|---|---|---|
| `error_step` | `authorization`, `authentication`, `payment` | `payment_initiation`, `card_enrollment_check`, `payment_authentication`, `payment_authorization`, `payment_capture` (plus UPI steps such as `payment_debit_response`) |
| `error_source` | `bank`, `gateway`, `customer`, `internal`, `razorpay`, `issuer` | per method: `customer`, `business`, `internal`, `gateway`, `issuer_bank`, `customer_psp`, `network`, `beneficiary_bank`, … |
| `error_reason` | **absent** | ~120 documented values (see Part 2) |

Consequence: the model is trained to infer from a crude error tuple something that production data would state directly (`card_expired`, `payment_risk_check_failed`, `bank_not_available`, …). It also never learns the realistic hard case, which is the catch-all `payment_failed` ("bank or wallet failed with no specific code").

#### H2 — `instrument_hint` is close to a label leak, and it is unrealistic
`P(hint = expired | CARD_EXPIRED) = 84.5%`, and hint `expired` appears only for CARD_EXPIRED (plus RISK_BLOCKED customers who also hold an expired card). The hint is copied from the latent `customer.expired` flag, which is also what causes the label.
It is also implausible: if the merchant knew a saved card was expired, the attempt would normally be blocked before the gateway (Razorpay has `incorrect_card_expiry_date` for typed expiry). Real `card_expired` declines more likely arise when stored data says "valid" but the issuer disagrees (reissued card). The generator should make `valid`/`unknown` the common hint on expired-card failures.

#### H3 — Labels are mostly determined by a handful of fields
Best-possible accuracy of a **majority-vote lookup table** (no model) on the sequential failures:

| Lookup key | Cells | Accuracy |
|---|---:|---:|
| error tuple only | 10 | **81.1%** |
| tuple + method | 26 | 81.8% |
| tuple + method + hint (= the 5 baseline features) | 40 | **84.6%** |
| method + hint (no error code) | 5 | 50.5% |

So a 40-row dictionary gets 85% on this world, and the 26-feature model's gain is measured above an easy floor. In the stress worlds the picture is similar (purity 72–92%), and **`unseen_codes` is 100% pure: the injected code string is literally `NEW_BANK_REASON_<CAUSE>`**, so the label is embedded in the (unseen) category value.

#### H4 — Outage schedules are identical in every seed, so "independent worlds" share structure
`episodes` in `sequences.py` are fixed constants (e.g. issuer_A down at 18–23% of the window, severity 0.97). Seeds change customers and noise, not the outage calendar. Similarly the stress scenarios use fixed windows (issuer outage 25–65%, PSP ramp 15–85%). The Phase 2.5 "564 independent worlds" are independent in customer draws only; incident timing, severity, and the entity set are common. Confirmation bounds therefore do not cover outage-structure variation.

#### M1 — Failure base rate is unrealistic, which distorts the health features
Attempt-level failure rate is **49.9%** (13,240 of 26,558). Published figures put UPI technical declines near 1% or below and business declines around 5–10% (secondary sources; see Part 2). Our `*_failure_rate_5m` baseline is ≈0.47–0.50 in healthy conditions, so rate features sit on the wrong scale and outage "lift" (0.47 → 0.84) is far larger than a real brownout would produce.

#### M2 — Persistent-customer logic makes RISK_BLOCKED trivially identifiable from history
A risk-flagged customer fails **100% of attempts on every method** (measured success rate: 0.0). 5.5% of customers produce **16.4%** of failures. `customer_failure_rate_7d`, `customer_successes_7d = 0` therefore identify them regardless of the error code. Real risk declines are issuer/transaction-specific and a flagged customer commonly succeeds on another instrument.

#### M3 — Single-cause, strict-priority labeling
Exactly one cause per failure, resolved by a fixed order. Real failures can have several co-existing conditions (low balance during an issuer brownout). Priority order also guarantees that RISK/EXPIRED mask everything else.

#### M4 — Infrastructure is tiny, clean and mechanically coupled to the label
4 issuers and 2 PSPs (stress: 3 unseen banks, 2 PSPs). Latency is `lognormal(5.4 + 2.0·psp_severity + 0.8·issuer_severity, 0.35)`, a deterministic function of the outage flag. Median latency by cause: PSP 807 ms, issuer 416 ms, all business/hard causes ≈225–238 ms. Real timeouts often have *missing* or very long latency, and issuers have different baselines.

#### M5 — Little missingness or noise; none of it is realistic
Only 2% of attempts are invisible to telemetry (sequential) and six more error-code swaps. There is no catch-all-reason rate, no delayed or lost webhooks beyond a 90 s delay on 1 in 11 rows (stress), no partial outages, no issuer-specific baselines.

#### M6 — Class distribution is emergent, not calibrated
Sequential failure mix: INSUFFICIENT_FUNDS 40.3%, RISK_BLOCKED 16.4%, PSP_TIMEOUT 14.6%, CARD_EXPIRED 12.1%, ISSUER_DOWN 9.7%, OTP 6.8%. It is not artificially balanced (the earlier concern), but it is a by-product of parameters such as `balance ~ lognormal(10.5, 1.1)` vs `amount ~ lognormal(10.2, 0.9)` (comparable magnitudes, so ~40% of orders exceed balance). Stress worlds range from 39.5% RISK (collision) to 2% PSP. The legacy generator's own header says its priors are "placeholders … replace with cited figures before submission"; that was never done.

#### L1 — Other
- Amount units are mixed: legacy multiplies by 100 (rupees→paise), v3 treats lognormal(10.2) directly as paise (median ≈ ₹270, balance ≈ ₹365).
- Method choice is independent of customer; real customers have stable method preferences.
- 3-day horizon; no weekly or salary-cycle effects; `hour_ist` has no daily load pattern in v3.
- History features use at most 7 days, and only one longer-term customer signal exists.

### 1.3 Realism vs arbitrary — summary

| Aspect | Verdict |
|---|---|
| Top-level `error_code` names (`BAD_REQUEST_ERROR`, `GATEWAY_ERROR`, `SERVER_ERROR`) | Real Razorpay values |
| `error_source` / `error_step` | Simplified; partly real words, not real vocabulary |
| `error_reason` | Missing |
| Point-in-time feature construction | Sound (no leakage found in builder) |
| Cause → code emission probabilities | Arbitrary, hand-set |
| Customer/issuer/PSP persistence | Present but shallow; risk/expiry are fixed-for-life flags |
| Outages | Present, correlated by entity and time, but fixed calendar, near-total, clean |
| Class mix, base failure rate | Arbitrary; failure rate ~5–50× too high |
| Missing/noisy observations | Minimal |

### 1.4 What this means for earlier results
The 0.693 → 0.803 macro-F1 gain, the safety experiments and the "independent world" bounds are all **properties of this generator**. They show relative behaviour of models on a controlled simulator, not real-world performance. The remaining 41 unsafe retries share a specific generator pattern (expired card, hint `unknown`, generic gateway tuple), i.e. a region where our generator makes expired-card and insufficient-funds failures look alike; whether that region exists in real data is unknown.

---

## Part 2 — What public sources actually say

Provenance key: **[doc]** official documentation, **[agg]** aggregate statistics, **[blog]** vendor/third-party commentary, **[sim]** simulated data.
Page contents were read through a summarising fetch tool; **the exact strings below should be re-verified against the live pages before they are hard-coded into a mapping.**

### 2.1 Razorpay failure fields and reasons
- **[doc]** Payment errors expose `code`, `description`, `source`, `step`, `reason` (and `error_`-prefixed on payment entities / `payment.failed` webhooks). `source` is "the point of failure", `step` "the stage where the failure occurred (varies by payment method)", and `reason` "the exact error reason, meant to be handled programmatically".
  — [About Errors](https://razorpay.com/docs/us/errors/), [Payment method error parameters](https://razorpay.com/docs/errors/payments/payment-methods-error-parameters.md)
- **[doc]** Steps/sources per method (cards, UPI intent/collect, netbanking, wallet, emandate) as in the table in H1. UPI has the richest chain (`payment_request` → `payment_debit_request/response` → `payment_credit_*` → `payment_status_*`), with sources such as `customer_psp`, `network`, `issuer_bank`, `beneficiary_bank`. UPI Collect is documented as deprecated from 28 Feb 2026.
- **[doc]** [List of Errors](https://razorpay.com/docs/errors/payments/list/?preferred-country=US) enumerates ~70 "Bad Request" and ~53 "Gateway" reasons. The ones relevant to our taxonomy:

| Our cause | Documented `error_reason` values (candidates) |
|---|---|
| BD_INSUFFICIENT_FUNDS | `insufficient_funds`; limit-type (`transaction_limit_exceeded`, `transaction_daily_limit_exceeded`, `transaction_frequency_limit_exceeded`, `credit_limit_exceeded`) overlap and are retryable only later/elsewhere |
| BD_USER_ABANDONED_OTP | `authentication_failed`, `incorrect_otp`, `otp_expired`, `otp_attempts_exceeded`, `payment_cancelled`, `payment_timed_out` ("customer didn't finish in time, or no gateway response") |
| TD_ISSUER_DOWN | `bank_not_available`, `bank_technical_error`, `issuer_technical_error`, `bank_cutoff_in_progress` |
| TD_PSP_TIMEOUT | `gateway_technical_error`, `request_timed_out`, `invalid_response_from_gateway`, `payment_declined_due_to_high_traffic`, `psp_not_available`, `psp_app_not_available`, `upi_app_technical_error` |
| CARD_EXPIRED | `card_expired` ("card's validity period has ended") |
| RISK_BLOCKED | `payment_risk_check_failed` ("declined by risk checks (Razorpay, gateway, or issuer)"); `debit_instrument_blocked` is a neighbouring hard state |
| **Not attributable from reason** | `payment_failed` ("bank or wallet failed with no specific code"), `card_declined`, `payment_declined`, `debit_declined` |

  Note: `payment_timed_out` and `payment_cancelled` appear with different meanings in different sections (customer didn't finish vs bank downtime), so the mapping is **many-to-many, not clean**. The mapping above is my reading of the descriptions, not a Razorpay-published taxonomy.
- **[doc]** The card-errors page ([Cards Error Codes](https://razorpay.com/docs/us/errors/payments/cards/?preferred-country=US)) states for `payment_failed` that the bank declined with the specific reason unavailable; for `payment_risk_check_failed`, "suspected fraud". It lists `gateway_technical_error` as partner-bank downtime and recommends the Downtime API.
- **Implication:** the existing `EXPLICIT_REASONS` in [hard_evidence.py](../services/app/domain/hard_evidence.py) (`card_expired`, `payment_risk_check_failed`) are **real documented reason strings**. This resolves one open Phase 2.6 question (existence), not the other (how often providers return a specific reason rather than the catch-all).
- **Product consequence:** where `error_reason` is specific, the cause is largely *stated*. The ML model's real job is the catch-all and ambiguous cases, attributing infrastructure causes from context, and (Phase 5) predicting whether a retry will succeed. Our current setup does not reflect that split.

### 2.2 Outage / downtime signals
- **[doc]** Razorpay emits `payment.downtime.started / updated / resolved` webhooks ([Payments webhooks](https://razorpay.com/docs/webhooks/payments/?preferred-country=IN)). Fields: `method`, `begin`, `end` (null while ongoing), `status`, `scheduled`, `severity` (e.g. high/medium), `instrument` (netbanking: `bank`; card: `issuer` + `type`, or `network` + `type`; UPI: `vpa_handle` or `psp` + `flow`). Downtimes are attributed to partner/issuing-bank technical issues.
  Matches our hard-evidence fixture design, with the caveat that detection delay and coverage are unknown.
- **[blog]** Razorpay's UPI-failure article ([blog](https://razorpay.com/blog/tackling-upi-payment-failures-with-razorpay/)) names NPCI codes (e.g. U69 collect expired, Z9 insufficient funds, U28 bank down, Z7/Z8 limits, U30 debit failed). It gives **no failure-rate statistics**, and its "ML routing detects degradation in seconds" claims are unquantified marketing.

### 2.3 Real failure-level datasets
**None found that suit this task.** Searches turned up:
- **[sim]** Kaggle "UPI Transactions 2024" describes itself as simulated.
- **[agg]** NPCI publishes aggregate business-decline (BD) and technical-decline (TD) statistics per bank and product (monthly), e.g. via the [India Data Portal](https://ckandev.indiadataportal.com/dataset/national-payments-corporation-of-india-npci/resource/f8c33592-34cd-4bdf-b4b8-d845d67b4eb4) and [Dataful](https://dataful.in/datasets/18242/). These contain rates by bank/product, not payment-level records, and I did not download or inspect them.
- **[blog/secondary]** Reported figures that I could not trace to an NPCI table: TD target under 1% and BD under 5% (circular OC-149, 2022); TD "up to 0.8%" in 2024; third-party compilation [upi-pulse](https://github.com/Anubhavkumarkanth/upi-pulse) reporting BD rising from 5.2% (2023) to 10.1% (2026, partial year). Treat as leads, not citations.
- **[docs]** Processor/bank code tables (NPCI UPI error codes v2.9, HDFC, Axis, Pine Labs) give code semantics, not frequencies.
- No public card-decline or netbanking failure-level data was found.

### 2.4 Conclusions from the research
1. A real, failure-level, labelled dataset is not publicly available. **We cannot learn real distributions; we can only anchor the generator to documented behaviour and aggregate rates, and label everything else as assumption.**
2. Anchors we *can* defend: the reason/step/source vocabulary; the existence of a catch-all reason; downtime event structure; aggregate order-of-magnitude BD/TD rates (after verifying NPCI tables directly); per-bank heterogeneity (NPCI publishes bank-wise rates).
3. We cannot defend: class frequencies by reason, how often `payment_failed` appears, retry-success rates, outage duration/severity distributions, customer-level persistence parameters. These stay explicit assumptions with sensitivity ranges.
4. No claim in the project should say "realistic" without the qualifier "anchored to documented vocabulary; parameters assumed".

---

## Part 3 — Proposed candidate features (for approval)

Rules applied: available from the `payment.failed` payload, the downtime webhook, or the platform's own attempt history **at diagnosis time**; no label-derived or outcome-derived inputs; no private issuer data; each new feature must justify itself against existing ones. **Every feature below is a candidate.** Step 5 ablation decides what survives.

Currently 26 (7 categorical, 19 numeric). Proposal: **+31 → 57**, in groups so ablation can remove whole groups.

### A. Gateway reason and flow position (4)
| # | Feature | Source / availability | Why / redundancy note |
|---|---|---|---|
| 1 | `error_reason` (cat) | `payment.failed` payload | Real, direct evidence; missing today (H1). Rare/unseen values rely on #2. |
| 2 | `reason_family` (cat, ~8 levels) | Deterministic documented map from #1 (customer-auth, limits/funds, bank-technical, gateway-technical, instrument-state, risk, generic, other) | Backoff for unseen/rare reasons and new-code robustness; deliberately coarse. |
| 3 | `is_generic_reason` (bool) | Derived from #1 | Explicit signal that the provider gave no usable reason. May prove redundant with #2; ablate. |
| 4 | `step_ordinal` (int) | `error_step` mapped to flow order | How far the payment progressed (initiation → auth → authorization → capture). Replaces reliance on the string. |

### B. Authentication and timing (4)
| 5 | `attempt_duration_s` | webhook `created_at` − `payment.created_at` | Abandonment/timeouts are long; technical failures short. Verify both timestamps are exposed. |
| 6 | `duration_vs_timeout_ratio` | #5 / documented ~10 min session limit | Normalises across methods. Likely redundant with #5; ablate. |
| 7 | `payment_flow` (cat: card_3ds, card_native_otp, upi_intent, upi_collect, netbanking) | payment entity / order options | Failure modes differ by flow. Verify field availability. |
| 8 | `latency_vs_issuer_p50` | existing `current_latency_ms` ÷ issuer rolling median | Latency relative to that issuer's norm; absolute latency already exists. |

### C. Instrument-level history (7)
| 9 | `instrument_consecutive_failures` | platform history | Run length of failures without a success. |
| 10 | `instrument_attempts_24h` | history | Short-horizon intensity (7-day customer count exists, not instrument). |
| 11 | `instrument_failure_rate_24h` | history | Persistent vs one-off. |
| 12 | `instrument_last_success_age_s` | history | Customer-level analogue exists; instrument-level differs when a customer has several cards. |
| 13 | `instrument_prev_reason` (cat) + 14 `same_reason_as_prev` (bool) | history (observed reasons only) | Repeating `card_expired` is much stronger than a first occurrence. Observed reasons only, never inferred causes. |
| 15 | `instrument_distinct_reasons_24h` | history | Stable reason ⇒ persistent state; varying reasons ⇒ transient. |

### D. Infrastructure rolling signals (10)
| 16 | `issuer_failure_rate_1h` | history | Longer window; 5-min exists. |
| 17 | `issuer_failure_delta` | 5m − 1h | Trend/onset. |
| 18 | `issuer_failure_vs_baseline_7d` (z-score) | history | Handles issuers with different normal failure rates (H4, M4). |
| 19 | `issuer_timeout_rate_5m` | share of timeout-family reasons | Timeouts separated from declines. |
| 20 | `issuer_tech_reason_share_5m` | share of `bank_*`/`issuer_*` technical reasons among recent failures | Reason clustering is the cleanest outage indicator. |
| 21 | `psp_failure_rate_1h` + 22 `psp_failure_delta` | history | As for issuers. |
| 23 | `psp_timeout_rate_5m` | history | |
| 24 | `issuer_method_failure_rate_5m` | issuer × method | Card vs UPI vs netbanking at the same bank can diverge. |
| 25 | `global_failure_rate_5m` | all traffic | Network-wide degradation (NPCI-level) vs entity-specific. |

### E. Customer and retry behaviour (4)
| 26 | `secs_since_prev_attempt` | history | Customer's attempt interval. |
| 27 | `customer_methods_tried_24h` | history | Method switching after failures. |
| 28 | `amount_vs_max_success_amount` | history | Funds proxy: ratio to the largest amount this customer ever completed. |
| 29 | `customer_success_rate_30d` | longer history | 7-day window is short for infrequent buyers; requires a longer retained history store. |

### F. External evidence and calendar (2)
| 30 | `matched_downtime_active` (bool) + `downtime_severity` | Razorpay downtime webhooks (documented) | Direct outage evidence; must match method + issuer/PSP; ignore stale events. Reuses the evidence matching logic already in `hard_evidence.py`. |
| 31 | `days_to_month_end` / `is_salary_window` | timestamp | Funds failures cluster around pay cycles *if* the generator models it; keep only if justified in data. |

(Counts: A4 + B4 + C7 + D10 + E4 + F2 = **31 candidates → 57 total**; #13/#14 and #21/#22 are paired. I expect ablation to drop several.)

### Deliberately excluded
- Anything from the later retry/recovery outcome, settlement, or `true_cause`-derived flags.
- Card BIN, issuer-declared decline codes beyond `error_reason`, device or IP signals: private or fraud-oriented, out of scope.
- Customer demographics; free-text descriptions (would be a cause-label proxy on synthetic data).

### Caveat that matters
Most of these features can only show value if the generator **creates the structure that makes them informative**. A feature's measured gain on synthetic data reflects my generator assumptions as much as feature quality. For the report, each feature's evidence will be labelled either *documented behaviour* or *assumed mechanism*.

---

## Part 4 — Proposed generator changes (outline, for review; not implemented)

1. **Emit real vocabulary:** `error_reason`, real `error_step`/`error_source` per method, plus a mapping table with explicit "assumed" probabilities, including a configurable **catch-all share** (e.g. 15–40% of failures return `payment_failed`/`card_declined`).
2. **Latent state richer than flags:** per-issuer baseline failure rates (≥15 issuers, heavy-tailed volumes), per-PSP/UPI-app baselines, issuer×method interaction; customers with stable method preference, instrument sets (multiple cards) and slowly changing balance.
3. **Failure-rate realism:** healthy attempt failure rate in a documented range anchored to NPCI aggregates (to be verified), with business declines dominating technical.
4. **Outages:** random calendar per seed (onset, duration, severity, partial brownouts, ramp-up and recovery), correlated across issuers sharing a rail; detection delay and missed-event probability for downtime webhooks.
5. **Multi-cause failures:** probabilistic causes with competing hazards instead of a fixed priority chain; hard declines not 100% persistent; risk declines transaction-specific.
6. **Noise/missingness:** missing latency on timeouts, delayed/lost telemetry, mislabelled reasons, hint behaviour per H2.
7. **Version it as `sequences@v4`**, leave v3 and all previous reports untouched, ship a `generator_checks` extension (label-purity ceilings, class mix, failure rates, seed-independence of outage calendar) and tests.
8. **Evaluate on one generator at a time**; v3 results are not compared numerically to v4 results.

---

## Open decisions needed from you
1. **Reason catch-all share and failure rates** are assumptions. OK to define 2–3 labelled scenarios (e.g. low / medium / high catch-all share) and report sensitivity, rather than a single claimed value?
2. **Label definition:** keep the six-cause simulator truth as the label for the demo (recommended, as it allows evaluation), while documenting that real labels would come from reason mapping plus later retry outcomes?
3. **Scope of the 31 candidates:** implement all as candidates and let ablation decide, or trim the least-justified now (#3, #6, #31)?
4. **NPCI aggregates:** may I download and check the NPCI BD/TD statistics to anchor base rates (not yet done)?
