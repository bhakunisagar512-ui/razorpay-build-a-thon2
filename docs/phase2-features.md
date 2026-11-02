# Phase 2: feature candidates and sequential synthetic data

**Current workflow:** see [generator V3, calibration, safety tuning and repeated
ablation](phase2-safety.md). The results and n-as-attempts semantics below describe
the archived V2 experiment; V3 uses n fresh measurement orders plus retries and
seven-day warm-up. Use `make phase2-safety` for current evaluation.

Phase 2 adds **21 candidate features**, for **26 total**, without changing the
serving classifier, safety thresholds, or published evaluation files. These are
inputs to test, not a claim that all 26 should be used in production.

**Selection update:** all 21 additions failed the per-partition safety constraint
in feature-by-feature development evaluation. The retained inputs are now **five**.
Normal builders/datasets exclude the additions; full candidates require explicit
opt-in. See [selection results and next candidates](phase2-feature-selection.md).
The group comparison below is retained as historical experimental evidence.

## Feature schema

The shared implementation is
[`services/app/domain/features.py`](../services/app/domain/features.py), version
`payment-features@v2`. `FEATURES` is the model-input allowlist. Payment/order/customer
IDs, labels, status, latent outage state, and eventual recovery are excluded.

| Group | Fields | Count |
|---|---|---:|
| Existing baseline | `gw_error_code`, `gw_error_source`, `gw_error_step`, `method`, `instrument_hint` | 5 |
| Infrastructure identity | `issuer_id`, `psp_id` | 2 |
| Amount/time context | `amount_minor`, `log_amount_minor`, `hour_ist`, `weekday_ist` | 4 |
| Customer history | `customer_attempts_7d`, `customer_successes_7d`, `customer_failure_rate_7d`, `customer_last_success_age_s`, `amount_to_customer_mean_7d` | 5 |
| Retry context | `order_prior_attempts_7d` | 1 |
| Issuer health | `issuer_attempts_5m`, `issuer_failure_rate_5m`, `issuer_latency_p95_ms_5m` | 3 |
| PSP health | `psp_attempts_5m`, `psp_failure_rate_5m`, `psp_latency_p95_ms_5m` | 3 |
| Method health | `method_attempts_5m`, `method_failure_rate_5m` | 2 |
| Current attempt | `current_latency_ms` | 1 |

Amounts are paise; log amount is `log1p(paise)`. Amount ratio divides the current
amount by the customer's mean prior amount. Age is seconds, latency milliseconds,
and rates fractions in [0, 1]. Weekday is Monday=0; hour and weekday use IST.
Sample counts accompany failure rates: one failure out of one attempt is not
equivalent evidence to one thousand failures out of one thousand attempts.
Latency p95 uses the nearest-rank empirical percentile. Prior order attempts
describe the seven-day observation window, not a lifetime retry count.

`FeatureBuilder.build(observation)` takes a snapshot before
`FeatureBuilder.observe(observation)` records the current attempt. Replay uses
nondecreasing `observed_at`, not event occurrence order. History includes only
earlier observed attempts whose event timestamps lie inside the window. Equal-time
events cannot see each other. Duplicate payment IDs are ignored within the
seven-day retention period, and the current payment is excluded from its history.
Missing rates, latency, ratios, and success ages stay null; missing counts are 0.

The same Python builder can be used by training replay and a future inference
adapter. Online integration still needs a trusted stream containing successful
and failed attempts, issuer/PSP identity, consent-independent telemetry, and
measured latency. Its state must be persisted or replayed after restart. The
current live database does not supply all this data, so the API continues to use
the five-feature baseline. No fabricated health data is injected into serving.

## Generator compatibility and outputs

The original generator is preserved in `eval/legacy_generate.py`.
`eval.generate.generate(n, seed)` defaults to `profile='legacy'`; its data and
public arm/split helpers remain compatible. Existing policy reports, classifier
training, API inference, and `eval/data/*` results retain their original behavior.

Opt into new data with:

```python
from eval.generate import generate, generate_dataset

dataset = generate_dataset(n=20000, seed=42, include_candidates=True)
failures_with_deliveries = generate(n=20000, seed=42, profile='sequences', include_candidates=True)
```

For sequential data, **n counts payment attempts, including successes**. It does
not mean n failures. The dataset contains:

- `transactions`: unique observable attempts, successes and failures, arrival
  timestamps, labels, and a historical-telemetry availability flag.
- `failures`: unique failed attempts, point-in-time feature snapshots, and labels.
- `deliveries`: delayed, reordered failure webhooks with duplicate deliveries.
- `outcomes`: one row per failed order, recording whether a later successful
  customer retry was observed. These are labels, never diagnosis features.
- `infrastructure`: hidden outage/degradation episodes for simulation inspection.
- `metadata`: schema/generator version, seed, counts, and units.

The training world has persistent customer balances, saved-card expiry, risk
flags, OTP friction, four issuers, two PSPs, and a shared chronological retry queue.
Issuer outage windows affect many customers simultaneously. PSP windows increase
failure probability and latency. Business declines depend on persistent customer
and instrument state; hard declines are checked before infrastructure failures.
Error emissions are noisy and overlapping. Successful customer retries can create
later-success outcomes; unresolved orders are right-censored at the dataset end.

All constants are synthetic assumptions. Uniform fresh-order arrivals span three
days; at most two organic retries are scheduled per order, with a 0.58 scheduling
probability and 120–1800 second delay. Customer balance can drift after retries.
This is not a full balance ledger, demand model, or policy-action experiment.
Retry outcomes cannot establish causal recovery effectiveness. Amount distribution
and success denominators differ from legacy data; old and new recovery uplift
figures should not be compared as if generated from the same world.

`observed_at` represents arrival at diagnosis/telemetry. `received_at` additionally
models webhook transport delay and redelivery. Snapshots use diagnosis arrival,
not eventual redelivery order. The telemetry-gap scenario simulates incomplete
historical ingestion while the current failure is available for diagnosis.

## Independent scenarios and splits

`eval/scenarios.py` defines a separate synthetic stress world. It does not import
the training simulator, training cause priors, or training error-emission table.
It shares only the taxonomy, feature contract, and replay/output packaging.

| Scenario | Change from the training world |
|---|---|
| Issuer outage | Unseen issuers, a longer outage, fixed customer cohorts |
| PSP degradation | Unseen PSPs and a smooth latency/failure ramp |
| Unseen codes | New error aliases and source/step tuples absent from training |
| Hard-decline collision | Expired/risk failures deliberately emit retryable-looking issuer errors; expiry hints missing |
| Telemetry gap | Most history unavailable and latency entirely missing |

These fixtures have different customer dynamics, transaction amounts, retry timing,
dates, and disjoint payment/customer IDs. They are independent simulation
assumptions, not independent evidence from real customers. The shared taxonomy and
feature implementation still limit the independence of this validation.

Sequential evaluation uses 60/20/20 arrival-time boundaries and a ten-minute
embargo. Whole orders crossing boundaries are excluded to prevent correlated
attempts leaking between partitions. Earlier online telemetry may inform later
test predictions; future observations may not. Legacy evaluation retains its
original hash split. Stress scenarios are scored after validation selection and
are not used to retune thresholds or features in this run.

## Reproducing the comparison

```sh
docker compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 api python -m eval.phase2 --all-candidates --n 20000 --legacy-n 40000 --scenario-n 4000 --seed 42
```

`make phase2-features` now evaluates retained inputs only; `--all-candidates`
reproduces the historical six-configuration experiment. The default report is
`/tmp/phase2-features.json` inside the container. Add `--export-dir /tmp/phase2-data`
to export transaction, failure, delivery, outcome, infrastructure, and metadata
files. Use `--output` to change the report location. Output directories are
overwritten when reusing a path. Runs are offline: no LLM calls, application DB
writes, serving model cache updates, or production promotion.

Six feature configurations use the same gradient-boosting hyperparameters and
unchanged rule/gate logic: baseline 5, baseline+infrastructure 14,
baseline+history 11, baseline+context 9, all without identities 24, and all 26.
Numeric features pass through with missing values; categorical features use
one-hot encoding with unknown-category handling. Tests establish parity with the
original classifier when only its five features are supplied.

The original metric names and definitions are retained, including decided-only
macro-F1, per-class recall/precision, abstention, stage mix, and unsafe retries.
`macro_f1_all` additionally penalizes abstention. Unsafe retry rate is measured
among true hard declines using the existing retryability mapping; it is a
classification-to-policy proxy, not actual executed retries.

Validation provisionally selects the highest all-row macro-F1 among configurations
with no higher unsafe-retry rate than baseline and no more than ten percentage
points extra abstention. This is an exploratory screen, not a production release
gate or statistical significance test. Stress safety regressions block adoption;
the report explicitly keeps the serving baseline.

## Measured results

The committed [evaluation report](phase2-feature-evaluation.json) records seeds,
environment versions, dataset/split fingerprints, source fingerprints, missingness,
legacy metrics, validation, temporal-test, and all stress results. The sequence
run has 7,770 training, 2,251 validation, and 2,269 test failures.

| Configuration | Temporal test macro-F1, all rows | Unsafe retry rate | Abstention |
|---|---:|---:|---:|
| Baseline 5 | 0.693 | 7.92% | 8.11% |
| + Infrastructure | 0.800 | 9.85% | 4.89% |
| + History | 0.729 | 7.53% | 7.10% |
| + Context | 0.686 | 8.88% | 7.40% |
| All except identifiers, 24 | 0.804 | 7.34% | 4.89% |
| All 26 | 0.803 | 7.34% | 4.76% |

History helps in this run. Infrastructure improves classification but, by itself,
worsens safety. Context alone does not improve the temporal test. Raw issuer/PSP
identifiers show no clear benefit over the combined 24-feature set on that test.
These group-level experiments do not prove every individual feature is useful.

Validation provisionally selects all 26, but the independent scenarios reject
adoption under a no-safety-regression requirement. For example, on the hard-decline
collision scenario, the sequential five-feature baseline abstains heavily and has
0% unsafe retries, while all 26 reaches 28.63%. Unseen codes produce high unsafe
rates even for the five-feature sequential baseline (78.56%). This exposes
overconfidence under unknown categories: a confidence threshold does not itself
provide out-of-distribution detection. The expanded inputs also increase unsafe
rates in the outage, degradation, and telemetry-gap scenarios.

Keep the serving baseline. Next work should test explicit unknown-input handling,
calibration and safety thresholds on separate validation scenarios, replicate
across seeds, and collect real telemetry before promoting an expanded model.
Do not tune against these now-observed stress fixtures and call them unseen tests.

Verification: **76 tests passed**, including the existing Phase 1 regressions,
five-feature prediction parity, chronological splits, late/simultaneous arrival
handling, duplicate exclusion, shared infrastructure incidents, independent
scenario construction, online/offline feature parity, and missing telemetry.
