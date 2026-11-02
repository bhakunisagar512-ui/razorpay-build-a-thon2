-- Revenue-risk pipeline (four event types, one diagnosis per event, interactive simulated payment requests). ADDITIVE ONLY:
-- new tables, nothing existing is altered or dropped. Everything stored here is synthetic.
CREATE TABLE IF NOT EXISTS revenue_event (
    event_id          TEXT PRIMARY KEY,
    event_type        TEXT NOT NULL CHECK (event_type IN ('CHECKOUT_ABANDONMENT','SUBSCRIPTION_FAILURE','OVERDUE_RECEIVABLE','PAYMENT_FAILURE')),
    source            TEXT NOT NULL CHECK (source IN ('custom','dataset','fixture')),
    run_id            TEXT,
    customer_id       TEXT NOT NULL,
    amount_minor      BIGINT NOT NULL CHECK (amount_minor > 0),
    currency          TEXT NOT NULL DEFAULT 'INR',
    payload           JSONB NOT NULL,                       -- controlled inputs only; never a hidden label
    route             TEXT,                                 -- payment failures: CLEAR_AND_CONSISTENT | MISSING_OR_GENERIC | CONTRADICTORY
    diagnosis         JSONB,
    plan              JSONB NOT NULL,
    status            TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','SIMULATED_RECOVERED','SIMULATED_FAILED','HUMAN_INTERVENTION')),
    status_reason     TEXT,
    workflow_state    TEXT NOT NULL DEFAULT 'NOT_STARTED' CHECK (workflow_state IN ('NOT_STARTED','RUNNING','WORKFLOW_COMPLETED','NOT_REQUIRED')),
    workflow_id       TEXT,
    recovered_amount_minor BIGINT NOT NULL DEFAULT 0,
    sim_started_at    TIMESTAMPTZ NOT NULL,                 -- origin of the simulated clock (quiet hours and weekly caps run on it)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS revenue_event_run_idx ON revenue_event(run_id);
CREATE INDEX IF NOT EXISTS revenue_event_customer_idx ON revenue_event(customer_id);

-- Hidden evaluation labels live apart from the event, readable only by the independent simulator / evaluator.
CREATE TABLE IF NOT EXISTS revenue_truth (
    event_id      TEXT PRIMARY KEY REFERENCES revenue_event(event_id),
    actual_cause  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revenue_step (
    event_id    TEXT NOT NULL REFERENCES revenue_event(event_id),
    step_idx    INT  NOT NULL,
    action      TEXT NOT NULL,
    label       TEXT NOT NULL,
    channel     TEXT,
    status      TEXT NOT NULL DEFAULT 'PLANNED' CHECK (status IN ('PLANNED','EXECUTED','SUPPRESSED','CANCELLED')),
    guard       TEXT,
    sim_at      TIMESTAMPTZ,
    request_id  TEXT,
    PRIMARY KEY (event_id, step_idx)
);

CREATE TABLE IF NOT EXISTS payment_request (
    request_id   TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL REFERENCES revenue_event(event_id),
    step_idx     INT  NOT NULL,
    action       TEXT NOT NULL,
    title        TEXT NOT NULL,
    amount_minor BIGINT NOT NULL,
    method       TEXT,
    status       TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','ACCEPTED','REJECTED','EXPIRED')),
    decided_by   TEXT,
    workflow_id  TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at  TIMESTAMPTZ,
    expires_at   TIMESTAMPTZ,
    UNIQUE (event_id, step_idx)
);
CREATE INDEX IF NOT EXISTS payment_request_status_idx ON payment_request(status);

CREATE TABLE IF NOT EXISTS revenue_intervention (
    event_id    TEXT PRIMARY KEY REFERENCES revenue_event(event_id),
    kind        TEXT NOT NULL CHECK (kind IN ('SAFETY_ESCALATION','DIAGNOSIS_INTERVENTION','PLAN_INTERVENTION')),
    reason      TEXT NOT NULL,
    details     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
