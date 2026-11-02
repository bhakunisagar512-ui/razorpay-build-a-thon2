# Why the additional features failed the safety screen

**Follow-up implemented:** the [corrected generator, calibration and safety
experiment](phase2-safety.md) addresses the next implementation order below.
This document and its report retain the earlier V2 diagnosis as historical evidence.

The investigation supports a generator coverage problem and some redundancy.
It does **not** support the claim that HistGradientBoosting cannot use the extra
information. The main distinction is that useful predictive information did not
translate into consistently safe decisions under distribution shift.

Full evidence: [phase2-investigation.json](phase2-investigation.json).
No features were added, no model was promoted, and serving remains unchanged.

## Experiments

Two training worlds, seeds 42 and 43, each have 20,000 payment attempts. Evaluation
uses temporal validation plus five development stress scenarios with 2,000 attempts
each, using seeds `11001 + 100 * training_seed + scenario_index`. The scenario
families have already been inspected and are development diagnostics, not untouched
confirmation tests. Previously published temporal test rows are excluded.

Four input sets are compared: gateway code/source/step only; the existing five;
context/history/infrastructure without gateway fields or raw issuer/PSP IDs; and
all 26. Two fixed, untuned estimators are compared: the existing 200-iteration
HistGradientBoosting and a 150-tree Random Forest (depth 18, leaf size 5).
Random Forest uses median imputation with missing indicators. This is a diagnostic
comparison, not a comprehensive model benchmark or evidence of an optimal model.

The report separates raw classifier predictions from the existing rule/gate
output. The no-gateway model's *raw* predictions exclude gateway information;
its hybrid output still uses the unchanged gateway rules. Raw output is a
diagnostic and is not an alternative safe execution policy.

Groups of additional features are also shuffled jointly **within identical
five-feature baseline tuples**. Baseline values and labels stay fixed. Three
shuffle seeds probe whether the model uses information beyond the gateway tuple.
This preserves relationships within a shuffled group, but can break relationships
between groups. It is model-usage evidence, not causal importance or a proof that
other features are redundant.

## 1. Generator: real connections, incomplete coverage

Issuer/PSP state really does affect generated failures. For seed 42, issuer A's
failure rate rises from 63.8% outside its outage to 96.0% inside; the technical
cause rate rises from 5.5% to 80.0%. PSP alpha's technical cause rate rises from
4.6% to 58.4%, while median latency rises from about 226 ms to 935 ms. Seed 43
shows similar relationships. These are programmed relationships, not validation
against real payment logs.

However, `eval/sequences.py` initializes n fresh-order arrivals over three days,
adds retries, then stops when **all attempts including retries** reach n. This
truncates the horizon:

| Seed | Observed horizon | Issuer C outage attempts | Later PSP beta degradation attempts |
|---|---:|---:|---:|
| 42 | 69.15% | 97 | 0 |
| 43 | 68.93% | 75 | 0 |

The later issuer outage is only partially covered; the later PSP episode never
occurs. Time splits therefore do not contain all configured environments. This
is a concrete generator bug, not evidence that infrastructure features are useless.

Other limitations visible in the code/data:

- A configured three-day run cannot establish seven-day history behavior.
  Histories start cold, so counts grow with elapsed simulation time.
- Five-minute health windows have median support of eight issuer attempts and
  sixteen PSP attempts. Aggregate failure rates include business declines as
  well as outages; they are noisy proxies for infrastructure health.
- Balances and customer risk/expiry/friction are simplified latent states.
  Successful spending does not debit a real balance ledger. A customer-wide risk
  flag affects every payment method. These assumptions need controlled sensitivity
  tests and eventual grounding in real observations.
- Training and stress worlds deliberately differ in customer processes, error
  emissions, missingness, and hard-decline collisions. Stress failures reveal
  sensitivity to these assumptions, not real-world incident prevalence.

The generator is deliberately left unchanged in this investigation so the
diagnosis refers to the exact data that produced the earlier results.

## 2. Redundancy: partial, not a blanket explanation

Some inputs are redundant by construction: raw amount and `log1p(amount)` are
one-to-one; customer attempts, successes, and failure rate are algebraically linked
when history is present. Raw issuer/PSP identities have little conditional value
in these runs.

But history and infrastructure add information beyond identical gateway tuples.
For the all-feature HGB model, conditional shuffling reduces raw all-row macro-F1:

| Shuffled group | Seed 42 F1 drop | Seed 43 F1 drop |
|---|---:|---:|
| Customer/order history | 0.126 | 0.095 |
| Infrastructure and current latency | 0.037 | 0.029 |
| Amount/time context | 0.022 | 0.016 |
| Issuer/PSP identity | 0.002 | -0.002 |

Random Forest shows the same ordering. The features are therefore not all merely
duplicates of gateway error codes. Individual safety-screen rejection should not
be interpreted as evidence of zero predictive value.

## 3. Model: HGB learns the additional information

Temporal-validation macro-F1, **raw model before rules and abstention**:

| Model / features | Seed 42 | Seed 43 |
|---|---:|---:|
| HGB / baseline 5 | 0.794 | 0.804 |
| HGB / all 26 | 0.913 | 0.880 |
| Random Forest / baseline 5 | 0.788 | 0.804 |
| Random Forest / all 26 | 0.909 | 0.879 |

This experiment gives no reason to replace HGB merely to access the extra signal.
Both models improve, and neither resolves safety across stress scenarios. Different
hyperparameters, calibration, or other models could still matter; this test does
not establish model-family superiority.

## 4. Decision path and probability shift explain an additional failure mode

The extra features change hard-decline probabilities. Retryable gateway rule hits
are allowed when estimated hard-decline mass does not exceed 0.25. In the new
hard-decline collision scenarios, all-feature HGB assigns mass at/below that limit
to **28.3% / 35.7%** of true hard declines, versus **0% / 0%** for the five-feature
model. Its hybrid unsafe-retry rates rise by the same fractions.

This happens despite better aggregate hard-decline Brier scores. An average
probability metric can improve while the specific decision boundary becomes less
safe. Random Forest shows the same issue. The model/gate interaction needs its own
validation, rather than treating each rejected feature as intrinsically harmful.

Rules and abstention also lower all-feature HGB temporal macro-F1 from raw
0.913/0.880 to hybrid 0.845/0.819. That difference is not justification to remove
the safety gate: in the collision scenarios the five-feature raw model has
92.6%/100% unsafe retries, while its gate suppresses them to 0%. Classification
accuracy and safe execution are distinct requirements.

An alternative model sometimes abstains on every row of a scenario. The diagnostic
handles this explicitly: decided-only F1 is null/undefined, all-row F1 and accuracy
are zero, and abstention is 100%. The baseline evaluator's historical behavior
is not changed.

## Next implementation order

1. Version a generator fix that distinguishes fresh orders from total attempts,
   covers the intended time horizon and all planned incidents, and supplies a
   separate historical warm-up period. Record coverage by split, not just globally.
2. Add controlled outage-severity, business-decline, telemetry-noise, and missingness
   variations to test whether the relationships survive changed assumptions.
3. Use the existing feature bank to test calibrated probabilities, unknown-input
   behavior, and gate/rule disagreements on development data. Evaluate hard-decline
   safety at the action threshold as well as aggregate calibration and F1.
4. Repeat grouped and individual feature selection on corrected data and reserve
   new untouched confirmation scenarios. Compare model families under the same
   data and safety constraints only after those repairs.

Additional feature collection is deferred. The current five retained inputs stay
in place while these issues are addressed.

Reproduce with `python -m eval.investigate_features` inside the app environment
(`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` for the recorded run). Defaults are
`--n 20000 --scenario-n 2000 --output /tmp/phase2-investigation.json`.
Verification: 85-test regression run passed; final investigation checks: 4 passed.
