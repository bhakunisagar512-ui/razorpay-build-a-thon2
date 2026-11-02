# Model C freeze (Phase 1)

Model C is frozen. This page is the contract: what it is, how it is called, and what later phases must not change. It was written from
the code and artifact as they are today; nothing was retrained, replaced or refactored to produce it.

**Enforced by** `tests/test_model_c_freeze.py` (55 tests, read-only) and the golden values in `tests/fixtures/model_c_golden.json`.
A failing freeze test means Model C is being changed. Follow "Changing it on purpose" below; do not edit the test to make it pass.

## 1. The model

| | |
|---|---|
| Model id | `phase4-hgb-deployment_core-uncalibrated-71bf19e4` |
| Type | scikit-learn `HistGradientBoosting` classifier wrapped in `eval.phase4_models.BenchModel` (one-hot encoder + model) |
| Hyper-parameters | `learning_rate 0.1`, `max_iter 100`, `max_leaf_nodes 15`, `min_samples_leaf 20`, `l2_regularization 0.1` |
| Calibration | none (`ProbabilityCalibrator('uncalibrated')`, an identity): outputs are raw HGB probabilities |
| Trained on | synthetic only: Phase 3 `sequences@v4` train worlds (calm, typical, stressed pooled). `activated: false`, `synthetic_only: true` |
| Libraries it was packaged with | Python 3.11.18, scikit-learn **1.9.1**, numpy **2.4.6** (the pickle requires the same stack) |
| Artifact | `eval/artifacts/phase4/model.joblib` (423,660 bytes) and `schema.json` (7,273 bytes), nothing else in the folder |
| SHA-256 | `model.joblib` `71bf19e4430f1b9532a5b4dd21a9d4076ea044de4655b8c4a2a5ca13b45594e6`; `schema.json` `9b2ad0136fe650fc9e83d1ddf97be19c3057794bf0da592af57afeb6adbb0acc` |
| Failure categories (output order) | `BD_INSUFFICIENT_FUNDS`, `BD_USER_ABANDONED_OTP`, `CARD_EXPIRED`, `RISK_BLOCKED`, `TD_ISSUER_DOWN`, `TD_PSP_TIMEOUT` |
| Known (code, source, step) tuples | 262 (used by the gate's unknown-tuple check) |

`AMBIGUOUS` is not a model class; it is what the gate returns when it will not stand behind a diagnosis.

## 2. The 43 features (order is part of the contract)

`gw_error_code, gw_error_source, gw_error_step, method, instrument_hint, issuer_id, amount_minor, log_amount_minor, hour_ist, weekday_ist,
customer_attempts_7d, customer_successes_7d, customer_failure_rate_7d, customer_last_success_age_s, amount_to_customer_mean_7d,
order_prior_attempts_7d, issuer_attempts_5m, issuer_failure_rate_5m, method_attempts_5m, method_failure_rate_5m, error_reason, reason_family,
is_generic_reason, step_ordinal, instrument_consecutive_failures, instrument_attempts_24h, instrument_failure_rate_24h,
instrument_last_success_age_s, instrument_prev_reason, same_reason_as_prev, instrument_distinct_reasons_24h, issuer_failure_rate_1h,
issuer_failure_delta, issuer_failure_vs_baseline_7d, issuer_timeout_rate_5m, issuer_tech_reason_share_5m, issuer_method_failure_rate_5m,
global_failure_rate_5m, secs_since_prev_attempt, customer_methods_tried_24h, amount_vs_max_success_amount, customer_success_rate_14d,
days_to_month_end`

- This is `DEPLOYMENT_CORE` (`services/app/domain/deployment_features.py`), 43 of the 57 `FEATURES_V4`, in `FEATURES_V4` order.
- 9 categorical (`gw_error_code, gw_error_source, gw_error_step, method, instrument_hint, issuer_id, error_reason, reason_family, instrument_prev_reason`), 34 numeric.
- Missing categorical -> `"missing"`; missing, non-numeric or infinite numeric -> NaN (no imputation); unseen categories -> all-zero one-hot block.
- The feature builder is `services.app.domain.features_v4.FeatureBuilderV4` (history is per builder instance; a custom payment uses an empty one).

## 3. Prediction interface

| Call | Returns |
|---|---|
| `services.app.adapters.diagnosis_v4.diagnose(*, gw_error_code, gw_error_source, gw_error_step, method, instrument_hint='na', features=None, **extra)` | `DiagnosisV4(cause, confidence, stage='ml_v4', decision, probabilities, top_cause, model_id, route)` |
| `diagnosis_v4.score(features)` | `{class: probability}` in class order |
| `diagnosis_v4.bundle()` / `load_artifact()` | the loaded artifact; the loader checks `model.joblib` against `schema.json`'s hash and the feature and class lists, and raises `ArtifactError` otherwise |

- `cause` is the gated cause (`AMBIGUOUS` when the gate abstains). `confidence` is the top raw probability, even when the gate abstains.
- Fails closed: no error code, or any exception -> `AMBIGUOUS`, confidence `0.0`.

## 4. Confidence thresholds and safety gate (`safety-policy@v2.2`)

| Constant | Value | Where |
|---|---|---|
| Minimum confidence | **0.60** (>= passes) | `safety_policy_v2.MIN_CONFIDENCE` |
| Hard-decline budget (CARD_EXPIRED + RISK_BLOCKED mass) | **0.10** (<= passes) | `AUTO_RETRY_MAX_HARD_MASS` = `REVIEW_MAX_HARD_MASS` = `THRESHOLDS_V2` = `diagnosis_v4.GATE` |
| Independent risk detection | **0.20** on P(RISK_BLOCKED) alone | `RISK_DETECTION_MIN_PROBABILITY`, `detect_independent_risk` |
| Original Phase 2.5 gate (reference only) | 0.95 / 0.03 | `diagnosis_safety.SafetyThresholds()` defaults |
| Hard causes | `CARD_EXPIRED`, `RISK_BLOCKED` | `diagnosis_safety.HARD` |
| Retryable causes | funds, issuer down, PSP timeout (OTP abandoned and both hard causes are not) | `taxonomy.RETRYABLE` |

Gate order (`decide_v2`, wrapping `diagnosis_safety.decide`): invalid probabilities -> expired-instrument hint veto -> hard-rule veto ->
confidence below 0.60 -> for retryable causes only: unknown tuple, hard mass above 0.10, rule/model disagreement. Then, only if a retry is
still allowed: explicit hard reason (`card_expired`, `payment_risk_check_failed`), hard reason family, or recent hard reason on the
instrument. Route: `AUTO_RETRY`, `NO_RETRY` (hard cause) or `REVIEW`. In the serving path `main.apply_independent_risk` then turns P(RISK_BLOCKED) >= 0.20 into `RISK_BLOCKED`.
Every branch has a golden case in the fixture.

## 5. What is frozen, and what is not

**Frozen: later phases must not change** (the freeze tests pin source digests, behaviour or both):

- `eval/artifacts/phase4/model.joblib` and `schema.json`; the class order; the 43 features and their order.
- `domain/features_v4.py`, `features.py`, `deployment_features.py`, `failure_vocab.py` (it derives `reason_family` and friends), `hard_evidence.py`, `taxonomy.py`.
- `domain/diagnosis_safety.py`, `domain/safety_policy_v2.py` (thresholds and gate behaviour), the deterministic `RULES` in `eval/classifier.py`.
- `adapters/diagnosis_v4.py`, `eval/phase4_models.py`, `eval/probability_calibration.py` (needed to load and run the pickle).
- The pinned libraries (scikit-learn 1.9.1, numpy 2.4.6, Python 3.11).

**Not frozen** (these consume Model C and may change in later phases, as long as the frozen parts above are untouched): hypothesis selection
(`domain/reassess.py`), recovery plans and plan validation, the evaluator, the Temporal workflows, the APIs, the frontend, the custom-input
rules (`domain/custom_input_rules.py`) and the database schema.

## 6. Dependants (read-only users of the frozen parts)

`services/app/main.py` (`apply_independent_risk`, `/simulate/model-c*`), `shadow.py`, `test_api.py` (individual and custom tests, preview),
`test_runs.py` (dataset runs), `test_datasets.py` and `test_progress.py` (feature derivation), `domain/reassess.py` and
`workflows/evaluated.py` (gate and probabilities for cycles 2 and 3), `domain/custom_input_rules.py` (vocabulary only), and offline
`eval/phase3_*`, `eval/phase4*.py`, `eval/phase5_compare.py`, `eval/build_ui_datasets.py`.

## 7. Changing it on purpose

1. Decide it explicitly (a new model, a threshold or a feature change is a new release, not a patch).
2. Produce the new artifact in a new folder, never in place; update this page, the pinned constants in `tests/test_model_c_freeze.py`, and
   regenerate `tests/fixtures/model_c_golden.json` in the same change.
3. Re-run the freeze tests and the Phase 2-5 suites.

## 8. Risks and limitations found while freezing (nothing was changed to address them)

1. **The artifact is not under version control.** `*.joblib` is git-ignored and `eval/artifacts/` is untracked, so the frozen model exists only
   on this machine. Back it up (the SHA-256 above identifies it). The freeze tests skip, with a message, if it is absent.
2. **The library stack is not pinned.** `requirements.txt` lists `numpy`, `pandas`, `scikit-learn` and `joblib` unpinned. A future image
   rebuild could install different versions and break or alter the pickle. A freeze test checks the running versions against the schema.
3. **The pickle depends on repo code**: it unpickles `eval.phase4_models.BenchModel`, so `eval/` must stay importable at serving time
   (as must `eval/classifier.py` for the gate's rules).
4. **`schema.json` records `safety_gate.max_hard_mass 0.03`** (the Phase 2.5 control); the runtime budget is 0.10 (`v2.1`). The metadata is
   stale but part of the pinned artifact; a test records the discrepancy.
5. **Stale docstrings**: `diagnosis_v4.py` and `safety_policy_v2.py` say "PREPARED, NOT ACTIVATED / nothing in the serving path imports it",
   but `main.py`, `shadow.py`, `test_api.py` and `test_runs.py` import them. (`activated: false` in the schema is likewise stale.)
6. **Probabilities are uncalibrated and only valid in distribution.** An input the model has not seen (for example UPI with
   `payment_authorization`) gives an unreliable 52.6% (pinned as a golden case). Even a rules-supported scenario can be uncertain
   (card + `server_error` + `payment_initiation` -> 56.8%, abstains).
7. **Observed gate behaviour worth knowing** (pinned, not changed): a confident non-retryable diagnosis such as OTP abandoned routes to
   `REVIEW`; explicit hard-reason vetoes only apply when the model would otherwise retry; and because the review budget equals the auto
   budget, the `hard_decline_mass_review` outcome cannot occur.
8. **`validate_csv` writes the upload to `eval/data/phase4_ui/uploads/` as a side effect**, so anything that validates a CSV (including
   `derive_frame`) writes a file. The freeze tests stub this out.
