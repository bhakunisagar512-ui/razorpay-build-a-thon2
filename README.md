# AI Revenue Recovery (Razorpay Build-a-thon, Track 03)

A **synthetic-only** revenue-recovery pipeline. It diagnoses a revenue-risk event **once**, plans a bounded and safe recovery, and runs that plan as a
durable Temporal workflow that creates **simulated** payment requests. A tester answers them on a separate mock payment page.

Nothing here is real: no payment gateway, no customer messaging, no runtime LLM, no real revenue. "Recovered" always means
**SIMULATED_RECOVERED**: a simulated customer completed a simulated payment.

## How it works

```
synthetic event (4 types) -> information-quality router (payment failures) -> ONE diagnosis -> deterministic plan (revenue-plan@2)
    -> Temporal RevenueRecoveryWorkflow -> simulated payment request per money-collecting step -> ACCEPT / REJECT -> result
```

| Event type | Diagnosis | Recovery |
|---|---|---|
| Checkout abandonment | none needed | checkout reminders, each with a payment opportunity |
| Subscription payment failure | the cause the subscription system reports | per-cause plan |
| Overdue receivable | none needed | invoice reminders; 30+ days overdue and unpaid escalates to a person |
| Payment failure | three routes, below | per-cause plan |

**Payment-failure routes**

| Gateway information | Route | Model C |
|---|---|---|
| Clear and consistent | deterministic rule | not called |
| Missing or generic | frozen Model C behind its safety gate | called **once** |
| Contradictory | Human Intervention, no automatic action | not called |

**Recovery plans** (`services/app/revenue/planner.py`, `revenue-plan@2`)

| Cause | Plan |
|---|---|
| Insufficient funds | alternative payment method now, reminder after 1 day, one delayed retry after a fixed 3 days (no salary date is known, so none is assumed) |
| OTP abandoned | fresh payment link after 5 minutes, reminder after 1 day |
| Expired card | card-update request, alternative method after 2 days; the expired instrument is never retried |
| Issuer downtime | one retry after 2 hours, then a reminder |
| PSP timeout | **verify the payment state first**; still uncertain -> a person; never retried while the original might have succeeded |
| Risk blocked | human escalation only, no retry |
| Ambiguous | verify the payment state, then one conservative reminder; unresolved -> a person |

Consent, quiet hours, the weekly contact cap and contact/retry budgets apply at execution time. **ACCEPT** = simulated payment success: the amount is counted
once, remaining steps are cancelled and the workflow stops. **REJECT** (or no answer before the demo timeout) = no payment: the next permitted step runs.
Neither answer triggers another diagnosis or another Model C call.

Details: [docs/revenue_pipeline.md](docs/revenue_pipeline.md). Model C contract: [docs/model_c_freeze.md](docs/model_c_freeze.md).

## Run it

Requires Docker with Compose v2.

```bash
./setup.sh                 # build, start, migrate, smoke test   (./setup.sh --down stops it, --reset wipes the database)
./restart-project.sh       # restart after a reboot or a code change (--build rebuilds the api and worker images)
```

Open **http://localhost:8000/ui**:

* **Individual Payment Testing**: pick an event from fixed dropdowns (no free text). Payment failures use seven interdependent dropdowns and a
  **Diagnosis scenario** selector (clear / missing-or-generic / contradictory) that only decides which inputs can be built; the router still picks the route, and the
  page shows the route, decision source, predicted cause, Model C call count, safety decision, plan and Temporal execution. Three ready-to-fill examples, one per route. Each step that asks for money opens a mock payment page (`/pay/{request_id}`) to ACCEPT or REJECT.
* **Dataset Testing**: run a prebuilt dataset, the controlled routing fixtures or a validated CSV through the same pipeline automatically.
* **Human Intervention**: read-only list of events a person must look at (contradictory data, risk blocks, no safe action), plus a collapsed list of
  earlier research records.

Temporal UI: http://localhost:8080.

## Services (docker compose)

| Service | Role |
|---|---|
| `api` | FastAPI app, console and mock payment pages (port 8000; code reloads automatically) |
| `worker` | Temporal worker: `RevenueRecoveryWorkflow` + 5 activities on queue `revenue-q`. **Restart it after code changes**: `docker compose restart worker` |
| `postgres` | events, plans, execution records, simulated payment outcomes, audit chain (port 5433) |
| `temporal`, `temporal-ui` | durable workflow engine and its UI |

Environment (`.env`, see `.env.example`): `DATABASE_URL`, `TEMPORAL_ADDRESS`, `DEMO_MODE`; optional `REVENUE_TASK_QUEUE`, `REVENUE_RESPONSE_TIMEOUT_S`,
`REVENUE_TIME_SCALE`. The whole stack is pinned to six logical CPUs (`scripts/cap_stack_cpu.ps1` verifies it on Windows).

## API

| Route | Purpose |
|---|---|
| `GET /health`, `GET /audit/verify` | liveness; verifies the hash-chained audit log |
| `GET /revenue/catalog` | controlled option lists (event types, selectors, consent, times) |
| `POST /tests/custom-options` | server-driven dependent dropdown options (`scope="clear"`) |
| `POST /revenue/events`, `POST /revenue/events/from-dataset`, `GET /revenue/events/{id}` | create an event and read its journey |
| `GET /revenue/requests`, `GET /revenue/requests/{id}`, `POST /revenue/requests/{id}/respond`, `GET /pay/{id}` | simulated payment requests and the mock payment page |
| `GET /tests/datasets`, `GET /tests/datasets/{name}/payments`, `GET /tests/csv-template`, `GET /tests/csv-schema`, `POST /tests/csv/validate` | dataset and CSV inputs |
| `GET /revenue/fixtures/payments`, `POST /revenue/runs`, `GET /revenue/runs`, `GET /revenue/runs/{id}`, `GET /revenue/runs/{id}/cases` | automated Dataset Testing |
| `GET /interventions`, `GET /interventions/revenue/{id}`, `GET /interventions/historical` | read-only Human Intervention views |
| `POST /shadow/event`, `POST /shadow/downtime`, `PUT /shadow/label/{id}`, `GET /shadow/diagnoses` | Model C shadow recorder (scores and records; never acts); used to verify the frozen model |
| `GET /ui` | the console |

## Repository layout

```
services/app/
  main.py  settings.py  db.py  worker.py  migrate.py  shadow.py  test_api.py  test_datasets.py  revenue_api.py
  revenue/      catalog, routing (one diagnosis), planner, engine, store, service, fixtures, runs, worker
  workflows/    revenue.py (RevenueRecoveryWorkflow), revenue_activities.py
  domain/       safety and validation logic, taxonomy, feature definitions, ledger (audit chain), custom-input rules
  adapters/     diagnosis.py, diagnosis_v4.py (the frozen Model C adapter)
  static/       index.html  app.js  app.css  pay.html  pay.js
eval/           frozen Model C artifacts (artifacts/phase4), dataset generators, evaluation and calibration scripts, oracle_plan*.py (dataset truth only)
db/             init/ and migrations/ (001-005; never deleted)
tests/          freeze, routing, planner, safety, API, dataset, live Temporal and frontend/ (browser) tests
docs/           model_c_freeze.md, revenue_pipeline.md, research phase reports (historical) and docs/historical/
```

## Tests

```bash
docker compose exec -T api python -m pytest -q                                   # backend (no database needed)
RUN_LIVE_REVENUE=1 python -m pytest tests/test_revenue_live.py -q                # live API + Temporal + Postgres (stack must be up)
PYTHONIOENCODING=utf-8 python tests/frontend/e2e_ui.py                           # headless-Chrome browser suite (stack must be up)
```

The API image deliberately does not ship matplotlib (a test/evaluation-only dependency, listed in `services/app/requirements-test.txt`). To run the whole backend suite, install it
in a throwaway container or virtualenv: `pip install -r services/app/requirements-test.txt`. Without it only `tests/test_phase5_compare.py` errors; xgboost/catboost tests skip.

`tests/test_model_c_freeze.py` (55 tests) fails if Model C, its 43-feature schema, thresholds or pinned source files change.

## Limitations

Everything is synthetic. A correct diagnosis is not recovered revenue; only a simulated ACCEPT is counted, and it is a simulated payment, not money.
Rates measured on the synthetic worlds say nothing about real traffic. Earlier research (three-cycle diagnosis, an LLM tier, a holdout A/B arm) is documented
as history in [docs/historical/early-research-readme.md](docs/historical/early-research-readme.md); that code was removed.
