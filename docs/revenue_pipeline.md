# Revenue-risk pipeline

Everything here is **synthetic and simulated**: no real payment gateway, message, customer or revenue, and no runtime LLM. Model C is frozen
(see [model_c_freeze.md](model_c_freeze.md)); nothing in this pipeline changes it.

## Flow

```
synthetic event (4 types) --> information-quality router (payment failures only) --> ONE diagnosis --> deterministic plan (revenue-plan@2)
      --> Temporal RevenueRecoveryWorkflow --> simulated payment request per money-collecting step --> ACCEPT / REJECT --> result
```

| Event type | Diagnosis | Plan |
|---|---|---|
| Checkout abandonment | none needed | checkout reminder with a payment opportunity, then one more on another channel |
| Subscription payment failure | the cause the subscription system reports | per-cause plan (below) |
| Overdue receivable | none needed | invoice reminders; 30+ days overdue and still unpaid escalates to a person |
| Payment failure | three routes (below) | per-cause plan |

### Diagnosis routes (`services/app/revenue/routing.py`)

| Route | Trigger | Diagnosis | Model C |
|---|---|---|---|
| `CLEAR_AND_CONSISTENT` | the gateway fields agree | deterministic rule | not called |
| `MISSING_OR_GENERIC` | fields absent or generic | frozen Model C, behind its existing safety gate | **once** |
| `CONTRADICTORY` | fields disagree with each other | none: Human Intervention, no automatic action | not called |

Missing is not contradictory. Each event is diagnosed once; ACCEPT, REJECT and timeouts never repeat the diagnosis or call Model C again.

### Custom Testing inputs

No free text anywhere. Payment failures use **seven interdependent dropdowns**: payment method, instrument hint, gateway error code, error source,
error step, error reason and the simulator's hidden evaluation label (optional). Options come from the server (`POST /tests/custom-options`).

A **Diagnosis scenario** selector chooses which inputs the dropdowns can build; it never chooses the route:

| Scenario | `scope` | What the dropdowns offer | Route the router then gives |
|---|---|---|---|
| Clear and consistent (deterministic rules) | `clear` | combinations whose reason identifies exactly one cause | `CLEAR_AND_CONSISTENT` (rule, 0 Model C calls) |
| Missing / generic gateway information (Model C) | `generic` | consistent combinations with no reason, a generic catch-all or a reason shared by several causes; no `expired` hint | `MISSING_OR_GENERIC` (frozen Model C once) |
| Contradictory gateway information (human intervention) | `conflict` | every vocabulary value, independently, so conflicting evidence can be built (still dropdowns, no free text) | `CONTRADICTORY` when the inputs conflict (0 Model C calls, no workflow) |

`POST /revenue/events` takes `diagnosis_scenario` and validates the inputs against that scope only (inputs that do not fit are refused with 422). Routing,
diagnosis and the planner never read it (a test greps for that). If contradictory-scenario inputs happen to be consistent they are still decided by rule; the
form shows a read-only **router preview** (`route_preview` in the options response, no Model C call) and says when it differs from the selected scenario.
Three ready-to-fill examples (one per route) set the selector and the fields. The result view shows the scenario selected, the route taken, who decided,
the predicted cause, the Model C call count, the safety-gate decision, the Temporal workflow and the recovery plan. The hidden label drives only the
simulator and an after-the-fact comparison; the cold-start Model C features include calendar fields, so the same inputs can score slightly differently
on different simulated failure days (the day is derived from the event id), never because of the label.

## Recovery planner (`services/app/revenue/planner.py`, `revenue-plan@2`)

Delays are relative to the previous step. Every step that collects money carries a simulated payment opportunity; `verify_payment_state` and
`human_escalation` do not.

| Cause | Steps | Rules |
|---|---|---|
| Insufficient funds | alternative method (0), reminder (+1 day), delayed retry (+2 days, i.e. 3 days after the failure) | one retry, never sooner than 3 days; the delay is a fixed, documented value: no salary date or pay cycle is in the data, so none is assumed |
| OTP abandoned | reminder with a fresh link (5 min), reminder (+1 day) | abandonment is neither a bank failure nor a success |
| Expired card | card-update request (0), alternative method (+2 days) | the expired instrument is never retried |
| Issuer downtime | delayed retry (2 hours), reminder (+6 hours) | one bounded retry after the outage has had time to clear |
| PSP timeout | `verify_payment_state` (60 s), reminder (+1 hour) | see below |
| Risk blocked | `human_escalation` | no retry |
| Ambiguous | `verify_payment_state` (60 s), reminder (+6 hours), escalate if unresolved | conservative |
| No consent for any channel | escalation | consent is never created |

**PSP payment-state verification.** A timeout does not prove failure, and an ambiguous diagnosis could be one. Before anyone is asked to pay again the
workflow verifies the original payment's state: *succeeded* -> the event becomes `SIMULATED_RECOVERED` (reason `original_payment_succeeded`), counted once,
no new request is created and the plan stops; *confirmed failed* -> the plan continues with its reminder; *unresolved* -> Human Intervention
(`payment_state_unverified`). Nothing is retried while the state is uncertain. A simulated timeout is never treated as proof of failure.

Consent, quiet hours (22:00-09:00 IST: messages wait), the weekly contact cap and the contact/retry budgets are applied at execution time
(`engine.py`, same constants as `domain/guards.py`). `planner.validate()` rejects any plan that retries a hard decline, retries insufficient funds within
3 days or a bank outage within 2 hours, retries while state is uncertain, asks for payment before the state is verified, or uses a channel without consent.
Budgets and the weekly cap are enforced by the executor, not the validator.

## ACCEPT and REJECT

**ACCEPT** = the simulated customer completed the payment: `SIMULATED_RECOVERED`, the amount is counted once, remaining steps are cancelled and the
workflow stops. **REJECT** (or no answer before the demo timeout) = no payment: the workflow continues with the next permitted step. A request resolves at
most once; duplicate answers are harmless, conflicting or late answers get HTTP 409.

Labels: `PENDING`, `SIMULATED_RECOVERED`, `SIMULATED_FAILED`, `HUMAN_INTERVENTION`, `WORKFLOW_COMPLETED`.

## Temporal

One workflow, `RevenueRecoveryWorkflow`, on task queue `revenue-q` (`services/app/workflows/revenue.py`). Activities (`revenue_activities.py`):
`rv_start`, `rv_execute_step`, `rv_simulate_response`, `rv_settle_request`, `rv_finish`. PostgreSQL is the source of truth for payment requests; signals only
wake the workflow; activity retries are idempotent; delays run on a simulated clock scaled by `REVENUE_TIME_SCALE`. The worker does not reload code:
`docker compose restart worker` after changing anything under `services/app`.

## Dataset Testing

Automated, no manual answers. Sources: the prebuilt datasets, `routing_fixtures` (66 controlled records, 24 clear / 24 missing-or-generic / 18
contradictory), `event_type_mix` (48 cases across the four event types) or a validated CSV (`POST /tests/csv/validate`, then `POST /revenue/runs` with
`dataset` **or** `csv_id`). The runner reproduces exactly, counts each simulated recovery once, reports one Model C call at most per case, scans for safety
violations and states that diagnosis accuracy is not recovered revenue.

## Human Intervention

Read-only. `GET /interventions` lists events a person must look at: contradictory data, risk blocks, no safe action, unresolved diagnoses
(`SAFETY_ESCALATION`, `DIAGNOSIS_INTERVENTION`, `PLAN_INTERVENTION`), with the final reason and the conflicting fields. There is no route that acts on them.
`GET /interventions/historical` and the collapsed "Earlier research records" section show the rows in `intervention_case` left by the retired three-cycle
research architecture, read-only and unchanged.

## Persistence and audit

PostgreSQL holds events, plans, execution records, simulated payment requests and outcomes and a hash-chained audit log (`GET /audit/verify`).
Migrations `db/migrations/001`-`005` are history and are never deleted. Historical rows from the retired architecture (`failed_payment`, `intervention_case`,
the audit chain) remain in place.

## Services and API

`api`, `worker`, `postgres`, `temporal`, `temporal-ui` (see the README). Redis was removed: nothing reads it. Routes are listed in the README; the removed
ones (`/webhooks/razorpay`, `/simulate/*`, `/diagnose`, `/tests/payments*`, `/tests/runs*`, `/metrics/*`, `/payments/*`, `/interventions/temporal/*`) answer 404.

## Where things are

| Piece | File |
|---|---|
| Controlled catalog (no free text) | `services/app/revenue/catalog.py` |
| Router, one diagnosis, Model C wrapper | `services/app/revenue/routing.py` |
| Planner | `services/app/revenue/planner.py` |
| Stopping rules, simulator, in-process executor | `services/app/revenue/engine.py` |
| Database operations (idempotent) | `services/app/revenue/store.py`, migration `db/migrations/005_revenue_pipeline.sql` |
| Event specs, `decide()`, ingest | `services/app/revenue/service.py`, `fixtures.py` |
| Automated Dataset Testing | `services/app/revenue/runs.py` |
| Temporal workflow, activities, worker | `services/app/workflows/revenue.py`, `revenue_activities.py`, `revenue/worker.py` |
| HTTP API | `services/app/revenue_api.py`, `test_api.py`, `main.py` |
| Custom-input rules | `services/app/domain/custom_input_rules.py` |
| UI | `services/app/static/` (`index.html`, `app.js`, `app.css`, `pay.html`, `pay.js`) |
| Frozen Model C | `eval/artifacts/phase4/`, `services/app/adapters/diagnosis_v4.py` |

## Retired architecture

The earlier stack (webhook ingestion, `RecoveryWorkflow`, `AdaptiveRecoveryWorkflow`, `EvaluatedRecoveryWorkflow` with up to three diagnosis cycles, hypothesis
reassessment, a mock channel, an LLM triage tier, a Redis service) is removed from the repository. Its research results remain in `docs/phase*` and
`docs/historical/`; a copy of the removed source lives outside the repository. 90 executions of the old `AdaptiveRecoveryWorkflow` on queue
`phase3-c58d15d9` are still open in Temporal with no worker; they are listed, not terminated, and nothing can start a new one.
`eval/oracle_plan.py` and `eval/oracle_plan_validation.py` are the original per-cause plan builder, kept only so the prebuilt datasets' truth files stay reproducible.
