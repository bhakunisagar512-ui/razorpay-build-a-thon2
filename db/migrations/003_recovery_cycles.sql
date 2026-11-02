-- Phase 3: recovery cycles and simulated outcomes. Additive and re-runnable; the default (legacy) path does not need it.
-- Apply to a live database with:
--   docker compose exec -T postgres psql -U recovery -d recovery -f - < db/migrations/003_recovery_cycles.sql

-- Hidden synthetic ground truth. Written only by the DEMO_MODE truth endpoint, read only by the outcome simulator.
-- Diagnosis and planning never query it.
CREATE TABLE IF NOT EXISTS synthetic_truth (
    payment_id TEXT PRIMARY KEY REFERENCES failed_payment(id),
    actual_cause TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per automated recovery cycle (max 3). The plan itself stays one row in recovery_plan (unique per payment);
-- later cycles APPEND their steps to recovery_plan.steps, so step_idx stays unique and every existing guard applies.
CREATE TABLE IF NOT EXISTS recovery_cycle (
    payment_id TEXT NOT NULL REFERENCES failed_payment(id),
    cycle INT NOT NULL CHECK (cycle BETWEEN 1 AND 3),
    cause failure_cause NOT NULL,
    cause_conf NUMERIC(4,3) NOT NULL,
    cause_stage TEXT NOT NULL,
    first_step_idx INT NOT NULL,
    actions JSONB NOT NULL,
    signals JSONB NOT NULL DEFAULT '{}',
    outcome TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    PRIMARY KEY (payment_id, cycle)
);

ALTER TABLE recovery_attempt ADD COLUMN IF NOT EXISTS cycle INT;
ALTER TABLE recovery_attempt ADD COLUMN IF NOT EXISTS sim_outcome TEXT;
ALTER TABLE recovery_attempt ADD COLUMN IF NOT EXISTS verified_outcome TEXT;
ALTER TABLE failed_payment ADD COLUMN IF NOT EXISTS recovery_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE failed_payment ADD COLUMN IF NOT EXISTS recovery_reason TEXT;
ALTER TABLE workflow_dispatch ADD COLUMN IF NOT EXISTS workflow TEXT NOT NULL DEFAULT 'recovery';
