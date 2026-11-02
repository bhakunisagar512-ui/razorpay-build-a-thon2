"""Hash-chained append-only audit ledger. See docs: tamper on any row breaks verify."""
import hashlib, json
from sqlalchemy import text

GENESIS = "0" * 64

def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()

def append(conn, *, entity_type: str, entity_id: str, actor: str,
           action: str, payload: dict, policy_ver: str | None = None) -> str:
    conn.execute(text("SELECT pg_advisory_xact_lock(82719001)"))
    prev = conn.execute(text("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1")).scalar() or GENESIS
    event = {"entity_type": entity_type, "entity_id": entity_id, "actor": actor,
             "action": action, "payload": payload, "policy_ver": policy_ver}
    h = _digest({**event, "prev_hash": prev})
    conn.execute(text("""INSERT INTO audit_log
        (entity_type, entity_id, actor, action, payload, policy_ver, prev_hash, hash)
        VALUES (:entity_type, :entity_id, :actor, :action, CAST(:payload AS jsonb), :policy_ver, :prev, :hash)"""),
        {**event, "payload": json.dumps(event["payload"], default=str), "prev": prev, "hash": h})
    return h

def verify_chain(conn) -> dict:
    prev, n = GENESIS, 0
    for r in conn.execute(text("SELECT * FROM audit_log ORDER BY seq")).mappings():
        body = {"entity_type": r["entity_type"], "entity_id": r["entity_id"],
                "actor": r["actor"], "action": r["action"], "payload": r["payload"],
                "policy_ver": r["policy_ver"], "prev_hash": prev}
        if _digest(body) != r["hash"] or r["prev_hash"] != prev:
            return {"valid": False, "broken_at_seq": r["seq"], "rows_checked": n}
        prev, n = r["hash"], n + 1
    return {"valid": True, "rows_checked": n}
