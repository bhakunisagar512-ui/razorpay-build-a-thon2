# Corrected generator, probability calibration, and safety-constrained ablation

**Follow-up:** the [binary safety comparison](binary-safety.md) keeps the expanded
multiclass model fixed and evaluates an additional hard-decline classifier on 564
fresh independent worlds. It reduces errors at low blocking cost but fails release;
serving remains unchanged.

All five implementation steps are complete offline. The generator is corrected,
unsafe decisions are traceable, three calibration options are compared on separate
data, policy thresholds are selected under a primary safety constraint, and six
feature groups are compared after those repairs. No candidate is deployed.

## Artifacts

- [Readable summary](phase2-safety/summary.json)
- [Full report, calibration metrics and every threshold-grid cell](phase2-safety/report.json)
- [Complete unsafe-decision traces](phase2-safety/unsafe-retries.jsonl.gz)
- [The selected policy's two held-out failures](phase2-safety/selected-policy-failures.json)

The export has **62,915 trace records**. Records repeat a payment across feature
sets, calibration methods, and policy candidates; they are not unique payments
or real executed recovery attempts. Their count is asserted equal to the sum of
all reported unsafe decisions, including rejected threshold-grid candidates.
Every JSONL record was also parsed and checked for normalized class probabilities.

## 1. Generator v3 and point-in-time history

`eval/sequences.py` now uses `sequences@v3`. **n means fresh measurement orders**,
not a cap on total payment attempts. Stratified arrivals cover the full three-day
horizon, while retries use the same queue and may add attempts. Retries beyond the
measurement end are censored rather than shortening the horizon. V2 is retained
in `eval/sequences_v2.py` for inspecting the earlier behavior; the legacy generator
and published five-feature reference metrics remain unchanged.

Seven days of warm-up precede measurement. Customers and opaque instrument
references persist across warm-up and measurement. Each instrument also has an
observed attempt shortly before measurement, generated using the same failure
logic; no artificial success is forced. Warm-up orders, including their retries
crossing the measurement boundary, are excluded from supervised labels and outcome
targets. They contribute only already-observed history. Missing or future data
are not backfilled into earlier snapshots.

The recorded seed-42 run has:

| Item | Count |
|---|---:|
| Fresh measurement orders | 20,000 |
| Warm-up attempts | 70,916 |
| Measurement attempts, including retries | 26,558 |
| Measurement failures | 13,240 |
| Observations in the later PSP beta degradation | 772 |

The last event is one second before the three-day end (99.9996% of the horizon).
All eight configured episodes are represented. Before measurement, every customer
has at least 18 known historical attempts and every instrument at least one.
Additional episodes ensure issuer and PSP failure labels occur in every evaluation
window. `eval/generator_checks.py` asserts these coverage/history properties before
the experiment fits any models.

The generator remains synthetic: balance dynamics, risk flags, expiry and retry
behavior are simplified assumptions. Correcting coverage does not establish
real-world validity. Models do not receive the latent outage state or instrument
reference as additional inputs.

## 2. Four separate data roles

Arrival-time windows use 45% model fit, 15% calibration fit, 20% policy selection,
and 20% confirmation. Ten-minute embargoes and whole-order exclusion prevent
shared orders crossing partitions. Warm-up rows are excluded. Earlier observed
telemetry can inform later predictions; future outcomes cannot.

| Partition | Failed-payment rows |
|---|---:|
| Fit | 6,070 |
| Calibration | 1,894 |
| Policy selection | 2,618 |
| Confirmation | 2,377 |

Calibration also uses five development scenario datasets, 2,000 attempts each,
with seeds 41000–41004. Policy selection uses distinct seeds 51000–51004.
Confirmation uses seeds 61000–61004 and is generated/scored only after all model,
calibration, threshold and feature-group choices are frozen. The scenario families
are already known, so new seeds provide held-out synthetic samples, not independent
real-world evidence. Dataset partition fingerprints are recorded in the report.

## 3. Calibration comparison

The base model is unchanged HistGradientBoosting. `eval/probability_calibration.py`
compares uncalibrated output, per-class sigmoid fits to log-odds, and per-class
isotonic fits to probability. Calibrated outputs are normalized across classes;
all-zero isotonic output falls back to the original distribution. Calibration
parameters are fitted only on the calibration partition/scenarios.

The report includes multiclass Brier (sum of squared class errors), binary
hard-decline Brier, ten-equal-width-bin top-label ECE, and hard-decline ECE.
It also reports false-retry counts/rates for raw output, the legacy gate, revised
fixed-default policy, and validation-tuned revised policy.

For all 26 inputs on temporal confirmation:

| Calibration | Multiclass Brier | Hard-decline Brier | Top-label ECE | Hard-decline ECE |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.09048 | 0.00746 | 0.04044 | 0.00799 |
| Sigmoid | 0.08275 | 0.00669 | 0.02701 | 0.01775 |
| Isotonic | 0.08296 | 0.00679 | 0.01931 | 0.00872 |

Calibration improves some metrics and worsens others. It is not assumed to improve
every model or every safety boundary. The five-feature temporal probabilities, for
example, have better Brier scores without either calibration method.

## 4. Revised safety policy and failure traces

`services/app/domain/diagnosis_safety.py` implements a pure policy. Known-expired
instruments and hard-decline rules veto retries. A retryable rule cannot override
the model or bypass confidence checks. Retryable diagnoses additionally require a
known gateway tuple, sufficient confidence, low estimated hard-decline mass, and
agreement with any gateway rule. Unknown tuples and conflicting rules abstain.

The offline vectorized implementation is checked against the shared pure policy.
Thirty confidence/mass combinations plus an explicit retry-disabled fallback are
tested. A candidate is feasible only with **zero observed unsafe retries in every
policy-validation partition**. Among feasible candidates, all-row macro-F1 is
maximized, then accuracy and retry coverage break ties. The grid is fixed before
confirmation; selected confidence may be as low as 0.60 when validation supports
it, and confirmation is allowed to reject that choice. All outputs report
abstention, retry coverage and recall so disabling retries cannot be mistaken for
useful recovery performance.

The JSONL trace includes true cause, model prediction, final predicted cause, raw
and calibrated class probabilities, rule result, known-tuple status, instrument
hint, confidence, hard-decline mass, thresholds, gate decision and diagnostic
flags. `confident_wrong_prediction` uses the explicitly fixed 0.60 diagnostic
reference; the record's actual decision thresholds are separate. Flags overlap
and provide diagnostic evidence rather than a causal attribution algorithm.

The selected all-26 sigmoid policy passes validation with confidence 0.60 and
hard-mass limit 0.03, but fails confirmation on two expired cards under missing
telemetry. Both are classified as insufficient funds, with calibrated confidence
0.964 and 0.937 and hard-decline mass 0.0175 and 0.0251. Neither has a rule hit;
neither failure is caused by rule precedence. Calibration reduces their near-1.0
raw confidence but leaves them confidently wrong. Their full records are retained
in `selected-policy-failures.json`.

## 5. Repeated feature-group comparison

For a controlled view, the report compares groups using the same fixed revised
policy. It separately compares each group's calibration and threshold choices
selected on development data. The latter results vary both features and selected
policy; they are not pure single-feature importance scores.

| Group | Validation-selected calibration | Temporal confirmation F1 | Unsafe retries across all confirmation partitions |
|---|---|---:|---:|
| Original 5 | Uncalibrated | 0.580 | 0 |
| + Infrastructure (14) | Isotonic | 0.484 | 0 |
| + History (11) | Isotonic | 0.760 | 1 |
| + Context (9) | Sigmoid | 0.540 | 1 |
| All without issuer/PSP IDs (24) | Sigmoid | 0.833 | 0 |
| All 26 | Sigmoid | 0.890 | 2 |

Validation selects all 26. Its confirmation result **rejects promotion**. The
24-feature set is promising, but selecting it now because of confirmation results
would reuse the holdout for selection. It requires a newly locked confirmation
dataset. The retained serving inputs remain five, and the legacy serving adapter
is unchanged: these are offline candidates, not a deployed calibrated model.

Zero observed unsafe retries does not imply zero population risk. Unknown-code
scenarios deliberately have zero retry coverage. The two remaining selected-policy
failures point to a need for missing-telemetry/data-quality handling and independent
confirmation; thresholds were not changed after inspecting them.

## Reproduction and tests

```sh
docker compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 api python -m eval.safety_pipeline --n 20000 --scenario-n 2000 --seed 42
```

`make phase2-safety` and `make phase2-features` run this corrected workflow. Output
defaults to `/tmp/phase2-safety` inside the container. The original `eval.phase2`
comparison is available through `make phase2-legacy-ablation` for the old gate.
It now consumes the current generator unless the archived V2 entry point is used;
old reports are preserved, not overwritten or reinterpreted as V3 results.

Read the compressed trace without extracting it:

```python
import gzip, json
with gzip.open('docs/phase2-safety/unsafe-retries.jsonl.gz', 'rt') as stream:
    for line in stream:
        record = json.loads(line)
        # Filter feature_group, calibration, partition and policy as needed.
```

**98 tests passed**, including scenario coverage, full horizon, customer/instrument
warm-up, no warm-up label leakage, disjoint calibration/policy/confirmation orders,
future-outcome mutation invariance, existing point-in-time tests, calibration
normalization/metrics, confidence and hard-decline vetoes, rule precedence,
vector/pure-policy parity, complete threshold tracing, and the Phase 1 database
regressions. Experiments do not send messages or touch application payment data.
