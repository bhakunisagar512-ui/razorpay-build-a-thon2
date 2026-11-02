-- Refuse ambiguous legacy plans rather than silently discard history.
CREATE UNIQUE INDEX IF NOT EXISTS recovery_plan_one_per_payment ON recovery_plan(payment_id);
CREATE TABLE customer_consent (
    customer_id TEXT PRIMARY KEY,
    whatsapp BOOLEAN NOT NULL DEFAULT false,
    sms BOOLEAN NOT NULL DEFAULT false,
    email BOOLEAN NOT NULL DEFAULT false
);
CREATE TABLE order_settlement (
    order_id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL UNIQUE,
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL
);
ALTER TABLE recovery_outcome ADD COLUMN settlement_order_id TEXT UNIQUE REFERENCES order_settlement(order_id);
CREATE TABLE workflow_dispatch (
    payment_id TEXT PRIMARY KEY REFERENCES failed_payment(id),
    dispatched_at TIMESTAMPTZ
);
INSERT INTO workflow_dispatch(payment_id)
SELECT payment_id FROM recovery_plan WHERE status='active' ON CONFLICT DO NOTHING;
