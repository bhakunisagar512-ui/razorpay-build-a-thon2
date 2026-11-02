# Retained features and the next search

The [V3 calibration and safety evaluation](phase2-safety.md) supersedes this earlier
selection run for current decisions. It fixes coverage/warm-up and repeats group
ablation under the revised gate. The all-26 validation choice fails held-out safety;
the serving allowlist therefore remains unchanged.

The subsequent [generator/redundancy/model investigation](phase2-investigation.md)
finds that history and infrastructure do provide useful predictive information.
Their rejection here reflects decision-pipeline safety regressions, not proof of
intrinsic uselessness. It also identifies truncated generator episode coverage.
Investigate those issues before adding further inputs.

The active model inputs remain the original five. `selected_features.MODEL_FEATURES`
is the retained allowlist. Normal feature builds and sequential datasets use it;
`include_candidates=True` and `eval.phase2 --all-candidates` explicitly opt into
the historical 26-feature bank. The serving classifier remains unchanged.

## Feature-by-feature selection

[Measured report](phase2-feature-selection.json): each of the 21 additions was
tested against the five-feature baseline using two 20,000-attempt training worlds
(seeds 42 and 43). Each fitted model was scored on temporal validation and five
separate 2,000-attempt development scenarios, for 12 development partitions.
Scenario seeds differ from the previously published stress runs. Their simulation
families have already been inspected; these are development data, not new unseen
test evidence. Previously reported temporal test rows were not used for selection.

Acceptance requires improved average all-row macro-F1 **and** accuracy, with no
increase in unsafe retry count in **any** partition. Abstentions count as incorrect
for accuracy. Partitions have equal weight for predictive metrics; safety is
checked separately, so improvements in one partition cannot offset harm elsewhere.

All 21 additions failed this rule. No forward-selection step was accepted, so
there is no new joint subset to promote. Existing group experiments also showed
safety regressions. Greedy selection does not exhaustively rule out beneficial
interactions between otherwise rejected features, and finite synthetic samples
do not establish population safety.

| Candidate | Mean F1 gain | Mean accuracy gain | Partitions with worse safety |
|---|---:|---:|---:|
| Time since customer success | +0.0704 | +3.99 percentage points | 9/12 |
| Customer success count | +0.0708 | +1.43 percentage points | 9/12 |
| Customer failure rate | +0.0645 | +1.66 percentage points | 10/12 |
| PSP latency p95 | +0.0493 | +2.31 percentage points | 10/12 |

These are measured effects of adding one input to this model, not causal or
additive feature importance. Full per-partition metrics and rankings are retained
in the report. Reproduce with `python -m eval.select_features` in the app
environment. The report defaults to `/tmp/phase2-feature-selection.json`.

## Better-sourced candidate signals

More columns alone do not fix an overconfident classifier under distribution
shift. The next search should start with directly observed diagnosis evidence.
The following are **unvalidated candidates**, not additions to the retained set.

| Priority | Candidate | Source and availability | Why investigate |
|---|---|---|---|
| 1 | Gateway `error_reason` | Razorpay payment failure entity; current webhook normalization drops it | More specific diagnostic evidence than generic `error_code` |
| 2 | Matching reported downtime status | Point-in-time downtime API/webhook cache, matched by method and affected instrument | External incident evidence instead of inferring outages only from rolling failures |
| 3 | Same-instrument time since last success | Merchant history keyed by an opaque card/token reference, strictly before diagnosis | Distinguishes a recently working instrument from a new or long-inactive one |
| 4 | Same-instrument consecutive failure count | Merchant history of that instrument's observed attempts | More specific than customer-wide failures across unrelated instruments |
| 5 | Authentication completion and elapsed time | Merchant checkout/authentication event telemetry available before diagnosis | Tests whether authentication friction adds information beyond the coarse gateway step |

Razorpay documents `error_reason`, card references, network/type, and
international status in the [payment entity](https://razorpay.com/docs/api/payments/entity/)
and [expanded card response](https://razorpay.com/docs/api/payments/fetch-payment-expanded-card/).
Card metadata may be missing and tokenization can change identifiers. These
fields' existence does not establish their usefulness. Network/type are best
used first to match incidents rather than automatically adding more raw categories.

The [official downtime API documentation](https://razorpay.com/docs/api/payments/downtime/)
describes instrument-specific incident status/severity. A feature must use the
incident update actually received before prediction; a later report cannot be
backfilled into earlier snapshots. No report or unavailable coverage is unknown,
not evidence that a rail is healthy. Account access and actual feed coverage must
be verified before relying on the signal.

The last three signals require merchant telemetry; this repository currently has
neither a complete instrument-history stream nor authentication event logging.
They must not be invented by copying `true_cause`, future outcomes, or simulator
outage flags. The most immediate data-collection improvement is to retain the
provider's reason field. Test each new signal individually, then its safe
combinations, with missing/stale/conflicting evidence and new confirmation data.
There are currently no measured F1/accuracy/safety gains for these five signals.
