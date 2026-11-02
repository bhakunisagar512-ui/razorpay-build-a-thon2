# Binary hard-decline safety experiment

**The binary veto adds some safety at a small coverage cost, but it fails the
zero-unsafe release criterion. Serving remains unchanged.** It was selected using
development data, then evaluated on a locked, larger fresh confirmation sample.
The threshold was not changed after confirmation failures appeared.

## Evidence

- [Compact results](binary-safety/summary.json)
- [Full report and validation threshold candidates](binary-safety/report.json)
- [Selection frozen before confirmation](binary-safety/frozen-selection.json)
- [Original two failures and their binary scores](binary-safety/retrospective-failures.json)
- [Every confirmation world](binary-safety/confirmation-worlds.jsonl.gz)
- [Unsafe-decision traces](binary-safety/unsafe-retries.jsonl.gz)
- [Sample counts and frozen-selection hash verification](binary-safety/artifact-verification.json)

## Failure analysis

Both earlier failures were **CARD_EXPIRED → BD_INSUFFICIENT_FUNDS**. Both used
cards with unknown expiry hints, the collision tuple
`GATEWAY_ERROR/gateway/authorization`, missing current latency, and limited customer
history. They were model errors, without a rule override. The existing calibrated
hard-decline masses were 0.0175 and 0.0251, below the original 0.03 veto limit.

The new binary scores for these examples are **0.1176 and 0.0299**. Neither exceeds
the validation-selected binary threshold of 0.15. The second objective therefore
repeats both original mistakes. These examples were analyzed retrospectively
after threshold selection, not included in threshold tuning. Full feature values
are retained in the retrospective artifact.

## Models and separated data

The expanded multiclass model remains fixed: all 26 inputs, the existing HGB
settings, the prior fit partition and sigmoid calibration. Its probability vectors
for the original two failures are asserted to reproduce the previous report.

The binary model uses **the same 26 features** with HGB and target
`HARD_DECLINE = CARD_EXPIRED or RISK_BLOCKED`; all other causes are `OTHER`.
Multiclass predictions are not extra binary input features. The binary objective
is different, but its features, labels and algorithm are not an independent source
of evidence. Its sigmoid calibration is fitted separately from model training.
It is not assumed to be optimally calibrated merely because sigmoid worked well
for the multiclass model.

Both models use the corrected V3 fit/calibration roles. Calibration uses the
temporal calibration partition and scenario seeds 41000–41004 at 2,000 attempts.
Threshold validation uses the temporal policy partition plus five scenarios at
1,000 and 3,000 attempts with new seeds 72000–72004 and 74000–74004. Variation in
traffic density is therefore present before confirmation. The original two
confirmation errors are not tuning samples.

The binary score can only **veto an already allowed retry**, producing AMBIGUOUS.
It cannot reenable a retry blocked by the multiclass gate, known expiry, hard rule,
unknown gateway tuple, confidence check or rule disagreement. Invalid binary
scores fail closed. The classifier's raw prediction/accuracy is unchanged; final
decision F1, abstention and coverage are measured separately.

## Predeclared threshold selection

The control keeps confidence 0.60 and multiclass hard-mass limit 0.03. Comparators:

1. Tighten only the multiclass mass limit, keeping the other decisions fixed.
2. Keep the control and add the binary veto.

The finite grids are recorded in the full report. This is not a proof that no
finer threshold or different model could improve the trade-off.

Eligibility requires zero unsafe retries in **every validation partition**, at
least 80% of the expanded model's mean F1 gain over the fixed five-feature reference,
at most 5% of previously allowed legitimate retries additionally blocked, and at
most five percentage points extra abstention in each partition. A legitimate retry
means the true cause is retryable according to the existing taxonomy. At equal
validated safety, selection maximizes legitimate retry recall, then F1, and favors
the simpler multiclass-only gate on ties.

Validation selects **binary threshold 0.15**. It has zero unsafe validation
retries and retains 100.03% of the F1 gain (vetoing wrong predictions can slightly
improve precision). The best safety-only multiclass comparator uses mass 0.02,
but retains only 74.55% of the validation F1 gain and fails the utility requirement.
It remains a diagnostic comparator, not an eligible release candidate.

## Fresh confirmation and risk accounting

The model, threshold, limits and seed schedule were written to a hashed manifest
before any confirmation world was generated. Confirmation uses new seed ranges
starting at 1000000, distinct from development and smoke data. Smoke runs use
2000000 and cannot satisfy the default confirmation sample size.

Six families were tested: sequential V3, issuer outage, PSP degradation, unseen
codes, hard-decline collision and telemetry gap. There are **94 independent worlds
containing hard declines per family: 564 total**, with **317,059 failed-payment
observations**, including **114,510 hard declines**. Sequential worlds use 1,000
fresh measurement orders plus retries and warm-up; stress worlds use 1,000 attempts.
Warm-up labels are never evaluated.

Within-world customer and infrastructure correlation is preserved. The release
bound therefore uses independent worlds, not individual payments. The risk event
is **at least one unsafe retry in a world, conditional on the world containing a
hard decline**, for the specified simulation family and frozen model.

The plan targets an upper bound of 5% with simultaneous 95% confidence across the
six families for the single frozen selected policy. One-sided exact binomial
bounds use Bonferroni alpha `0.05 / 6`. With zero failures, 94 worlds give
`1 - (0.05/6)^(1/94) = 4.9656%`. Sampling stops only after the planned number of
hard-decline-containing worlds, never because unsafe outcomes appear. Worlds
without hard declines would be retained for utility but excluded from the
conditional risk denominator; all 564 recorded worlds contained hard declines.

These are bounds for independent **synthetic worlds** conditional on fixed
models and simulation assumptions. They are not per-payment real-world bounds.
Event-level IID bounds are also recorded but explicitly marked descriptive only.
Comparator intervals are corrected across six families per comparator; there is
no claimed joint 95% guarantee across every compared policy simultaneously.

## Results

| Frozen decision policy | Unsafe retries | Mean final macro-F1 | Additional legitimate retries blocked |
|---|---:|---:|---:|
| Expanded multiclass control | 59 | 0.6563 | 0 |
| Tighter multiclass mass 0.02 | 13 | 0.6218 | 10,779 / 104,283 = 10.34% |
| Binary veto at 0.15 | 41 | 0.6559 | 305 / 104,283 = 0.29% |

F1 is the mean of per-world scores, weighted equally across families. The
five-feature reference mean is 0.4695, so the binary veto retains **99.84%** of the
expanded model's F1 improvement. Its mean extra abstention is **0.095 percentage
points**; its maximum family-level legitimate blocking fraction is 0.48%.
Utility passes, but safety does not.

All remaining unsafe decisions were expired cards: the sample contains 46,723
CARD_EXPIRED and 67,787 RISK_BLOCKED cases, with zero unsafe RISK_BLOCKED decisions
for these three policies. This is an observed result, not a risk-block guarantee.

| Family | Binary unsafe retries | Selected-policy world-risk upper bound |
|---|---:|---:|
| Sequential V3 | 21 | 31.89% |
| Issuer outage | 4 | 12.11% |
| PSP degradation | 3 | 10.53% |
| Unseen codes | 0 | 4.97% |
| Hard-decline collision | 0 | 4.97% |
| Telemetry gap | 13 | 20.59% |

The binary veto catches 18 of the control's 59 unsafe decisions; 41 remain. Paired
comparison finds 11 unsafe examples missed by both binary and tighter multiclass
gates. Binary blocks two unsafe examples the tighter multiclass gate misses;
the tighter multiclass gate blocks 30 that binary misses. The second objective
provides some complementary signal, but is not a sufficiently reliable safety
layer at the coverage-preserving operating point.

Probability quality is also mixed. For example, in telemetry-gap worlds, mean
binary sigmoid Brier is 0.0706 versus 0.0547 for multiclass hard mass (raw binary
Brier is 0.0416). Calibration and a separate binary objective do not automatically
make hard-decline probabilities reliable under missing telemetry.

**Release rejected.** Both zero observed unsafe retries and the world-risk bound
fail. The confirmation sample meets its planned size; insufficient sample size is
not the reason for rejection. Thresholds were not lowered afterward to fit these
failures, nor was a comparator selected using confirmation outcomes. Serving
continues with the previously retained five inputs.

## Reproduction and checks

```sh
make binary-safety
```

This copies the reference failure file into the current container, then runs
`python -m eval.binary_safety`. Output defaults to `/tmp/binary-safety`. Re-running
the default seed range reproduces this experiment; those samples are now observed
and cannot serve as a fresh confirmation set for future tuning. Reserve a new
`--confirmation-seed-base` before a future confirmation. `--smoke-worlds 1` runs a
small engineering check and explicitly fails the default sample-size release check.

The frozen-selection hash, all world counts, every unsafe trace, normalized
multiclass probabilities, and veto monotonicity were verified. **114 tests passed**,
including the existing Phase 1/2 regressions, binary target labels, invalid-score
handling, pure/vector veto parity, safety-plus-utility eligibility, preference for
a simpler gate when binary adds no value, and independent-world sample planning.
No application payments, messages, live model or Temporal workflow were changed.
