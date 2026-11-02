# Phase 3 results: realistic synthetic worlds, 57 candidate features, ablation and selection

Everything here is **offline and synthetic**. The serving classifier, recovery workflows, safety gate and all earlier reports are unchanged. Raw outputs: [phase3/report.json](phase3/report.json), [phase3/ablation-and-selection.json](phase3/ablation-and-selection.json), [phase3/data-quality.json](phase3/data-quality.json), [phase3/npci-upi-summary.json](phase3/npci-upi-summary.json), [phase3/run.log](phase3/run.log). The earlier audit that motivated this work is [phase3-dataset-audit.md](phase3-dataset-audit.md).

## 1. Summary

1. **A new generator, `sequences@v4`, replaces the shortcuts found in the audit.** Instrument hint is uninformative (`P(hint=expired | CARD_EXPIRED)` 1.1% vs 84.5% in v3), a risk decline is no longer a permanent customer flag (76% of risk-declined instruments later succeed vs 0%), failures are a minority (~10–15% vs 50%), outage calendars differ in every world, and causes overlap. It emits Razorpay's documented `error_reason` / `error_source` / `error_step` strings.
2. **On this generator, 57 features beat the existing 26 by +0.15 to +0.23 macro-F1 and the 5-feature baseline by +0.24 to +0.32** (test worlds, three scenarios, §5). Hard-decline recall rises from 0.53–0.74 to 0.78–0.95.
3. **Almost all of that gain comes from two groups whose value depends directly on my assumptions:** the reason features (A) and attempt duration (B). Infrastructure (D) and instrument-history (C) features add only about +0.01–0.02; customer-behaviour (E), calendar and downtime (F) did not clear the selection bar (§6). Do not quote the headline as a real-world gain.
4. **The selection rules kept 16 of the 31 new features (42 total).** The 42-feature set is statistically indistinguishable from all 57 (differences −0.005 to +0.007 vs a 0.015–0.018 standard deviation across worlds), so it is the lighter candidate for the model-benchmark phase (§7).
5. **The unchanged safety gate still lets some hard declines through** (5/1444, 5/1883 and 26/2669 hard declines in the three scenarios for the selected set; upper 95% bounds 0.6–1.4%). It is not zero, and I did not retune it, as agreed.

## 2. What was built

| Item | File |
|---|---|
| Documented Razorpay vocabulary and reason-family mapping (no simulation parameters) | [services/app/domain/failure_vocab.py](../services/app/domain/failure_vocab.py) |
| 31 candidate features + builder (the original 26 come from the unchanged `FeatureBuilder`) | [services/app/domain/features_v4.py](../services/app/domain/features_v4.py) |
| Scenario configuration and emission tables, each value tagged *documented / NPCI / assumed* | [eval/world_config_v4.py](../eval/world_config_v4.py) |
| `sequences@v4` generator, assembly, truncated-replay leak check | [eval/sequences_v4.py](../eval/sequences_v4.py) |
| Data-quality checks | [eval/phase3_quality.py](../eval/phase3_quality.py) |
| Experiment: worlds, ablation, selection, test evaluation | [eval/phase3.py](../eval/phase3.py) |
| NPCI UPI statistics summary | [eval/npci_stats.py](../eval/npci_stats.py) |
| 44 new regression tests | `tests/test_features_v4.py`, `tests/test_sequences_v4.py`, `tests/test_phase3.py` |

Full suite: **150 passed, 20 skipped** (the skips are the existing PostgreSQL integration tests that need `TEST_DATABASE_URL`).

## 3. What is documented, what is NPCI-informed, what is assumed

| Element | Status |
|---|---|
| `error_reason`, `error_source`, `error_step` strings; which reasons sit in the Bad Request / Gateway sections; step order per method; downtime webhook structure; the ~10 min session limit | **Documented** by Razorpay (read via a summarising fetch tool; re-verify exact strings before production use) |
| `reason_family` grouping | **Our reading** of the descriptions. Several reasons are genuinely ambiguous in the docs (e.g. `payment_timed_out`), and the generator reproduces that ambiguity |
| UPI business-decline level (5–10% of attempts) and technical-decline level (0.3–1.5%) | **NPCI-informed**, UPI payer banks only: volume-weighted BD 5.2–6.7% (2022–24) and 10.7% (Sep 2026); TD 1.45% (2022) falling to 0.26% (Sep 2026). Bank-level 5th–95th percentile: BD 3.5–15.9%, TD 0.04–7.8%. Source: NPCI "Top 50 Member Performance" tables as republished by the third-party upi-pulse project, because NPCI's site returned HTTP 403 to automated download. Only 35 months are present |
| Card and netbanking failure rates | **Assumed.** NPCI data is not applied to them |
| Generic-reason share, missing/noisy reasons, every reason weight, source/step weights, attempt-duration distributions | **Assumed** (docs only say a catch-all exists) |
| Outage frequency, duration, severity, ramp shape, downtime detection delay/miss/false-alarm rates, network-incident behaviour | **Assumed.** NPCI motivates heavy-tailed bank-level spikes (16% of bank-months have TD above 2× the bank's own prior median) but monthly data cannot give durations |
| Customer, balance, salary-cycle, retry, switching, reissue behaviour | **Assumed** |

A data file I first downloaded as "NPCI product-wise declines" turned out to contain only **AEPS and NFS** (ATM/cash networks) over 23 days, with no UPI. I deleted it and did not use it.

## 4. Generator and scenarios

Per world: 14 issuers with heavy-tailed volume and persistent baselines, 3 gateways and 5 UPI apps, thousands of customers with 1–3 instruments each, a 14-day warm-up plus 3 measured days, successes and failures, retries and instrument switching. Each attempt draws independent hazards for all six causes (funds from amount vs. a noisy salary-cycle balance, authentication friction, a transaction-level risk hazard with a 12 h cooldown, card reissue unknown to the merchant, issuer and PSP outages including partial/ramped/method-scoped ones, NPCI-style network incidents). When several fire, a stage race picks the reported cause (about 4–7% of failures have an overlapping cause). The six labels are unchanged.

| Scenario | generic reason | reason missing | outages / entity / week | telemetry drop | downtime notice detected | BD scale |
|---|---:|---:|---:|---:|---:|---:|
| calm | 15% | 1% | 0.35 | 1% | 90% | 0.8 |
| typical | 30% | 3% | 1.0 | 3% | 75% | 1.0 |
| stressed | 50% | 8% | 2.5 | 8% | 55% | 1.35 |

Each scenario uses 4 train, 2 validation and 3 test worlds of 24,000 measured orders, plus 2 "shifted-market" test worlds in which issuer and PSP baselines are redrawn. Seeds are disjoint, and every world has its own customers, noise and outage calendar.

## 5. Evaluation

Protocol: same model recipe as the existing offline classifier (HistGradientBoosting, 200 iterations, one-hot categoricals) on every feature set; sigmoid calibration fitted on validation worlds; the existing gate (confidence 0.60, hard-decline mass 0.03) applied unchanged. Its legacy error-tuple rules use the old vocabulary and never fire here. All feature selection used validation worlds only; the test worlds were scored once after the selection was frozen.

### 5.1 Test macro-F1 (mean ± sd over 3 test worlds) and accuracy

| Scenario | 5 features | existing 26 | all 57 | selected 42 |
|---|---:|---:|---:|---:|
| calm | 0.610 ± 0.011 | 0.699 ± 0.010 | 0.925 ± 0.018 | 0.932 ± 0.018 |
| typical | 0.609 ± 0.014 | 0.691 ± 0.030 | 0.894 ± 0.015 | 0.896 ± 0.018 |
| stressed | 0.640 ± 0.005 | 0.736 ± 0.022 | 0.883 ± 0.015 | 0.878 ± 0.017 |
| *accuracy, calm / typical / stressed* | .766 / .749 / .713 | .807 / .797 / .789 | .954 / .937 / .909 | .958 / .938 / .906 |

Pooled training (all scenarios) then testing per scenario gives the same ordering (57: 0.933 / 0.912 / 0.891). Shifted-market test: 57 features 0.918 / 0.883 / 0.880 vs 26 features 0.686 / 0.690 / 0.728, so the result is not specific to one issuer market.

### 5.2 Where the model has to work: generic or missing reasons only
Macro-F1 on failures whose reason is generic or missing (n = 1,124 / 2,221 / 4,309):

| Scenario | 5 | 26 | 57 | selected |
|---|---:|---:|---:|---:|
| calm | 0.544 | 0.621 | 0.706 | 0.725 |
| typical | 0.520 | 0.608 | 0.743 | 0.749 |
| stressed | 0.618 | 0.714 | 0.824 | 0.814 |

The gain is smaller here (+0.08 to +0.13 over 26) but real, which is the fairest single number for "what the model adds beyond the provider telling us the cause".

### 5.3 Per-class and confusion (typical scenario, test)

| Class | 26: P / R | 57: P / R | support |
|---|---|---|---:|
| Insufficient funds | .867 / .897 | .954 / .967 | 3,372 |
| User abandoned (OTP/PIN) | .835 / .883 | .964 / .979 | 2,678 |
| Card expired | .677 / .673 | .929 / .910 | 788 |
| Risk blocked | .631 / .579 | .849 / .860 | 1,095 |
| Issuer down | .722 / .481 | .901 / .744 | 270 |
| PSP timeout | .655 / .537 | .896 / .816 | 467 |

Confusion matrix for 57 features (rows true, columns predicted; order funds, OTP, expired, risk, issuer, PSP):

```
funds    [3261   27   11   54   13    6]
otp      [  23 2623    0   12    3   17]
expired  [  17    0  717   50    1    3]
risk     [  88   13   34  942    5   13]
issuer   [  22   16    9   17  201    5]
psp      [   8   42    1   35    0  381]
```
The remaining errors are the plausible ones: risk vs funds, expired vs risk, issuer-down and PSP-timeout mistaken for other causes. Per-scenario matrices for all sets are in `report.json`.

### 5.4 Calibration, unsafe retries, latency
| Scenario (selected set) | log-loss raw → sigmoid | top-label ECE raw → sigmoid | hard-decline Brier | raw unsafe / hard | gated unsafe / hard (upper 95%) | gate abstention | legit-retry recall |
|---|---|---|---:|---|---|---:|---:|
| calm | 0.237 → 0.136 | 0.031 → 0.007 | 0.0164 | 61 / 1,444 | 5 / 1,444 (0.73%) | 7.0% | 0.888 |
| typical | 0.183 → 0.180 | 0.016 → 0.006 | 0.0250 | 120 / 1,883 | 5 / 1,883 (0.56%) | 12.4% | 0.779 |
| stressed | 0.275 → 0.268 | 0.022 → 0.010 | 0.0386 | 262 / 2,669 | 26 / 2,669 (1.35%) | 16.3% | 0.713 |

For comparison, the existing 26-feature model has gated unsafe 14 / 12 / 28 and raw unsafe 304 / 336 / 487 on the same test worlds, and abstains on 36–40% of failures. So the richer features reduce unsafe retries and abstention together, but **do not eliminate unsafe retries**. These are per-failure rates across three worlds per scenario and carry no claim about real traffic.

Inference latency (single row, container CPU, includes encoding): 5 features 4.5 ms, 26 features 10.0 ms, 57 features 13.5 ms, 42 selected 11.5 ms. Feature construction costs about 0.12–0.14 ms per failure. Batch throughput stays above 30,000 rows/s.

## 6. Ablation and feature selection (validation worlds only)

Group effect on validation macro-F1, with 95% paired-bootstrap lower bounds (300 resamples). "Drop" removes the group from all 57; "add" adds it to the existing 26.

| Group | Drop from 57 (calm / typical / stressed) | Add to 26 (calm / typical / stressed) | Retained |
|---|---|---|---|
| A reason | −0.169 / −0.094 / −0.049 | +0.235 / +0.162 / +0.093 | **yes** |
| B auth & timing | −0.010 / −0.023 / −0.033 | +0.083 / +0.068 / +0.067 | **yes** |
| C instrument history | −0.004 / 0.000 / −0.007 | +0.022 / +0.014 / +0.012 | yes (small) |
| D infrastructure | +0.004 / 0.000 / −0.010 | 0.000 / +0.015 / +0.016 | yes (small; not in calm) |
| E customer behaviour | 0.000 / −0.001 / −0.001 | +0.004 / +0.002 / +0.004 | no |
| F downtime + calendar | 0.000 / −0.002 / 0.000 | +0.015 / +0.005 / +0.006 | no |

Predeclared rule (written before the results): a group is kept if the lower 95% bound of its macro-F1 effect is above zero in at least 2 of 3 scenarios, for drop *or* add; within kept groups, a feature is kept if its leave-one-out log-loss increase exceeds 0.001 or its permutation log-loss increase exceeds 0.002 in at least 2 scenarios.

**Retained (16):** `error_reason`, `reason_family`, `is_generic_reason`, `step_ordinal`, `attempt_duration_s`, `latency_vs_issuer_p50`, `instrument_consecutive_failures`, `instrument_failure_rate_24h`, `instrument_last_success_age_s`, `instrument_prev_reason`, `issuer_failure_rate_1h`, `issuer_failure_delta`, `issuer_timeout_rate_5m`, `psp_failure_rate_1h`, `psp_timeout_rate_5m`, `global_failure_rate_5m`.

**Dropped (15):** `duration_vs_timeout_ratio` (an exact rescale of `attempt_duration_s`), `payment_flow`, `instrument_attempts_24h`, `same_reason_as_prev`, `instrument_distinct_reasons_24h`, `issuer_failure_vs_baseline_7d`, `issuer_tech_reason_share_5m`, `psp_failure_delta`, `issuer_method_failure_rate_5m`, `secs_since_prev_attempt`, `customer_methods_tried_24h`, `amount_vs_max_success_amount`, `customer_success_rate_14d`, `matched_downtime_severity`, `days_to_month_end`.

Strongest by permutation importance (mean log-loss increase): `reason_family` 0.68, `attempt_duration_s` 0.32, `method` 0.21, `gw_error_source` 0.19, `gw_error_step` 0.18, `error_reason` 0.13, `amount_minor` 0.05, `step_ordinal` 0.04. Everything else is below 0.011.

### How much to trust this selection
* **Strong evidence:** `error_reason`, `reason_family`, `step_ordinal`, `attempt_duration_s`. Each moves macro-F1 by far more than the noise.
* **Weak evidence:** the C and D features. Several pass only because the rule is lenient (for example `issuer_failure_delta` has leave-one-out +0.004 but permutation −0.0004; `issuer_timeout_rate_5m` and `global_failure_rate_5m` pass on leave-one-out alone). The 95% intervals for groups C and D include zero in some scenarios. Treat them as "probably small but harmless", not as proven.
* **Not shown to help, which is not the same as useless:** downtime matching and customer behaviour. In this generator, downtime notices are detected 55–90% of the time, outages touch a small share of traffic, and the cause usually already shows in the reason string, so there is little left for these to add. A different traffic mix could change that.
* I did not re-screen the original 26 features. Some may be redundant now.

## 7. Recommendation for the model-benchmark phase

1. **Candidate feature set:** the selected 42 (26 existing + 16). Re-confirm on a fresh set of worlds before benchmarking if time allows; 3 test worlds per scenario give wide intervals.
2. **Before any real-data claim, verify availability at diagnosis time in real Razorpay payloads** for `error_reason`, `attempt_duration_s` (needs payment creation and failure times) and `step_ordinal`. These carry most of the gain, and `attempt_duration_s` rests entirely on my assumed duration distributions.
3. **Training strategy (same splits for all models, all selection on validation worlds):**
   * **HistGradientBoosting:** keep as the reference (current recipe, `max_iter=200`).
   * **XGBoost:** use native categorical support or the same one-hot matrix; early-stop on the validation worlds; tune `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight` with a small random search, scored on validation macro-F1 and log-loss.
   * **CatBoost:** pass categoricals natively (`error_reason`, `reason_family`, `payment_flow`, `instrument_prev_reason`, ids); it is the most likely to gain from high-cardinality categoricals. Tune `depth`, `learning_rate`, `l2_leaf_reg`, with early stopping on validation.
   * **Random Forest:** `n_estimators` ~300, `min_samples_leaf` tuned, `class_weight='balanced_subsample'`; as a variance check, not an expected winner.
   * For every model: calibrate on the validation worlds (sigmoid; the sigmoid improved log-loss and ECE here), report macro-F1, per-class recall, ECE, and gated unsafe retries with denominators, and compare in-scenario, pooled and shifted-market results. Promote nothing unless it beats the HistGradientBoosting reference by more than the across-world spread (~0.015 macro-F1) and does not worsen unsafe retries.
4. Keep the safety gate and the hard-decline protections exactly as they are.

## 8. Limitations and deviations
* **Synthetic only.** Absolute scores (macro-F1 0.88–0.93) describe this generator. The reason field alone gets 0.66–0.89 held-out accuracy depending on the generic share, so scores move strongly with an assumption nobody can currently measure. v3 results (e.g. 0.693 → 0.803) are **not comparable** with these; the generators differ.
* **Deviation from the proposal:** `customer_success_rate` uses a 14-day window (`customer_success_rate_14d`), not 30, to bound warm-up cost. `matched_downtime_severity` merges the proposed bool and severity into one ordinal (0 none, 1 medium, 2 high), keeping the count at 31.
* **A check was changed after seeing its result:** the per-world "no constant feature" flag fired because `matched_downtime_severity` was constant in 2 of 33 worlds (no failure coincided with a notice). I redefined the check as "constant across the whole split", which is what it was meant to test, and report world-level constants as information. The same feature is also one of the dropped ones.
* **Modelling choices that affect labels:** NPCI-style network incidents are labelled `TD_PSP_TIMEOUT` 60% / `TD_ISSUER_DOWN` 40%; UPI PIN failures and cancellations map to the OTP/abandonment class; funds-type limit errors map to insufficient funds.
* **Statistics:** 3 test worlds per scenario. Standard deviations are across worlds; the bootstrap treats failures within a world as independent, which understates uncertainty.
* **UPI technical declines in the stressed scenario (3–4%) are deliberately above the NPCI band;** typical (about 1.1–1.5%) and calm (about 0.9–1.2%) sit within it. Single worlds can spike to 2.8% in typical, as NPCI bank-months do.
* The existing gate's legacy tuple rules are inert on the new vocabulary, so "unsafe retries" here measure the model plus the confidence and hard-mass checks.
* NPCI statistics come from a third-party republication and cover 35 months with gaps (Nov 2024 – Aug 2026 missing).

## 9. Reproduce
```sh
make npci-stats    # summarises eval/data/upi_pulse_raw -> eval/data/npci_upi_summary.json
make phase3        # copies tests, runs the 44 new tests, then the full experiment (~30 min on 8 CPUs)
```
Run it from a shell that does not rewrite `/code/...` paths (Git Bash does; the Makefile uses a relative output path for that reason). Worlds are cached under `eval/data/phase3_run/` (git-ignored); `python -m eval.phase3 --quality-only` rebuilds the quality report from the cache.
