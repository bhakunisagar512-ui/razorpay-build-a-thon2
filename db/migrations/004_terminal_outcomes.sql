-- Phase 3 correction: validated plans, deterministic evaluation, one terminal classification per payment.
-- Additive and re-runnable. NOT applied to the live database until the owner approves.
--   docker compose exec -T postgres psql -U recovery -d recovery -f - < db/migrations/004_terminal_outcomes.sql

ALTER TABLE failed_payment ADD COLUMN IF NOT EXISTS terminal_class TEXT
    CHECK (terminal_class IN ('SIMULATED_RECOVERY_SUCCESS', 'PLAN_INTERVENTION', 'DIAGNOSIS_INTERVENTION', 'SAFETY_ESCALATION'));
ALTER TABLE failed_payment ADD COLUMN IF NOT EXISTS resume_at TIMESTAMPTZ;      -- set while recovery_status = 'waiting'

-- Cases for the Phase 4 human-review backend. The primary key makes a second case for a payment impossible.
CREATE TABLE IF NOT EXISTS intervention_case (
    payment_id TEXT PRIMARY KEY REFERENCES failed_payment(id),
    kind TEXT NOT NULL CHECK (kind IN ('PLAN_INTERVENTION', 'DIAGNOSIS_INTERVENTION', 'SAFETY_ESCALATION')),
    reason TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per diagnosis cycle: the asserted hypothesis (NULL = the gate abstained), the deterministic verdict and the
-- plan-validation record (original plan, errors, corrected plan).
ALTER TABLE recovery_cycle ADD COLUMN IF NOT EXISTS hypothesis TEXT;
ALTER TABLE recovery_cycle ADD COLUMN IF NOT EXISTS evaluation TEXT;
ALTER TABLE recovery_cycle ADD COLUMN IF NOT EXISTS validation JSONB;
