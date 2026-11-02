# Phase 1: recovery correctness

This implementation provides safe mock execution and capture-confirmed accounting.
It does not execute real retries, switch rails, or deliver real messages. Live mode
raises an error instead of silently reporting mock delivery as live.

## Database and worker rollout

Pause API and worker traffic, run `python -m services.app.migrate` from the project
root in the app environment, then restart both processes. Container images without
the new migration must be rebuilt first. Migration `002_recovery_correctness.sql`
adds a unique plan per payment, current customer consent, order captures, capture
attribution, and a workflow-start outbox. It refuses existing duplicate plans;
inspect and reconcile those manually instead of discarding recovery history.

The worker polls committed outbox rows every two seconds. Starts use a fixed
workflow ID and reject ID reuse, including completed executions. If Temporal
accepted a start but its response was lost, the next dispatch recognizes the
existing execution. A failed dispatch leaves the row pending. Run a worker to
dispatch plans; the API response means the plan was durably accepted, not that
Temporal has already started it. Existing active plans are backfilled into the
outbox. Retry failed workflows through an explicit operational process; automatic
ID reuse is intentionally disabled.

## Consent and action execution

`customer_consent` is the current source of truth. Each channel defaults to denied.
Trusted application code must upsert verified consent and revocations there;
there is no unauthenticated consent management endpoint. The demo injector can
initialize consent for its newly generated customer. A later failure does not
overwrite an existing customer's consent. SMS and email require their own consent;
declining WhatsApp does not imply SMS permission. Legacy customers have no consent
record and outbound contact is suppressed until verified consent is recorded.

Before mock execution, the activity locks the order and customer, locks the
consent row, validates the persisted plan step, checks for an existing attempt,
and evaluates guards again. Delivery, attempt, and audit writes share one
transaction. A crash rolls back all three; a retry of a committed step returns
the existing result. Customer locking also serializes contact-cap checks across
different payments. Suppression records and their audit writes are idempotent.

These guarantees apply to the transactional mock adapter. A future external
provider must accept the stable `plan_id:step_idx` idempotency key or provide
reconciliation for uncertain sends. A database transaction alone cannot guarantee
exactly-once external delivery. Mock attempts are explicitly labeled `mock_sent`.

## Captures and attribution

Signed `payment.captured` events persist independently of workflow completion.
An order/capture uniqueness constraint handles redeliveries. Both failure ingestion
and capture processing reconcile under the same order lock, so either delivery
order works. An existing order capture suppresses further actions even if Temporal
notification fails. Successful notifications interrupt new workflows' timers.

One order can credit one failed payment: the earliest eligible failure present at
reconciliation, with payment ID as the tie-breaker. The currency must match, the
capture must cover the failed amount, and its event timestamp must not precede
the failure timestamp. Credit is capped at the failed amount. Later deliveries
do not move an existing credit. Other failed attempts on that order receive no
additional revenue credit, but their plans stop. Partial captures stop actions
conservatively and do not claim full recovery. Multi-capture aggregation and
refund/chargeback accounting are not implemented.

The signed top-level event time is used when available, then the payment creation
time, then receipt time. Reliable historical ordering requires provider event
timestamps. Legacy outcomes without a linked capture are excluded from recovery
rates and revenue. INR net metrics subtract recorded attempt costs immediately,
including before workflow finalization. Finalization preserves confirmed recovery
and updates costs and attempt counts. Captures are audited; concurrent ledger
appends serialize using a transaction-scoped advisory lock.

Confirmed payment after a failure is an accounting observation, not proof that an
intervention caused it. `/metrics/uplift` still reports arm aggregates; it does not
establish causal uplift, adjust for unfinished observation windows, or demonstrate
real execution while actions remain mocked.

New workflows interpret plan delays as offsets from workflow start rather than
cumulative sleeps. A Temporal patch preserves the timer behavior for older
histories. Provider latency can still add execution time to the schedule.

## Regression tests

Set `TEST_DATABASE_URL` to a PostgreSQL SQLAlchemy URL and run
`pytest -q tests`. The integration fixture creates a random schema per test,
applies migrations, and drops only that schema afterward. It never truncates
existing application tables. Set `TEST_MIGRATIONS` if migrations live elsewhere.

Coverage includes concurrent ingestion and execution, lost workflow-start
responses, capture reordering and redelivery, single-credit accounting, consent
revocation, independent channel consent, transactional rollback, live-mode refusal,
concurrent audit appends, and interruptible absolute workflow timers. Database tests
skip explicitly without `TEST_DATABASE_URL`. Timer tests use a mocked workflow
runtime. The deployed-stack check below additionally exercises actual Temporal
workflows; replay against historical pre-migration executions remains untested.

## Completed local rollout — 29 October 2026

API and worker images were rebuilt, both services were stopped, migration `002`
was applied to the existing application database, and both services were recreated
and restarted. Preflight found no duplicate plans and a valid 138-row audit chain.
Existing records were preserved.

`python -m eval.verify_rollout` then exercised the running HTTP API, migrated
application database, and real Temporal worker. All seven check groups passed:
health/migration, actual worker execution with repeated-activity deduplication,
concurrent webhook deduplication, capture redelivery with single-credit accounting
and interrupted timers, capture-before-failure delivery, consent revocation while
waiting, and audit/outbox verification. Five workflows completed, the audit chain
was valid at 150 rows, and no workflow dispatches remained pending. The rebuilt
container's regression suite also passed: **49 tests**.

The machine-readable evidence, including workflow IDs, is retained in
[phase1-rollout-verification.json](phase1-rollout-verification.json). Verification
records remain in the local database and Temporal history: three labeled webhook
failures plus two demo-injected failures. The two signed synthetic captures add
₹998 to local metrics; this is test accounting, not real recovered revenue. All
actions ran in mock mode. Re-running the verification creates another set of
records and requires demo mode, mock mode, and a configured webhook secret.
