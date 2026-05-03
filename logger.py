"""
logger.py — Step 5 of CAIR

What this file does:
  Creates routing_log.sqlite on first run (if it doesn't exist).
  Writes one row per routing decision.
  Never blocks routing — if the write fails, it logs to stderr and continues.

The eval script (eval_sa1.py) reads this table to calculate all 5 eval dimensions.
"""

import sqlite3
import hashlib
import logging
from pathlib import Path
from contextlib import contextmanager

from routing_engine import RoutingDecision

DB_PATH = Path(__file__).parent / "routing_log.sqlite"

# We don't store the raw prompt — only its SHA256 hash.
# This protects any sensitive content while still letting us deduplicate
# and track the same prompt across multiple routing calls.

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]   # first 16 hex chars


# ---------------------------------------------------------------------------
# Schema — created once, never recreated if it already exists
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS routing_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    prompt_hash             TEXT NOT NULL,
    model_id                TEXT NOT NULL,
    model_tier              TEXT NOT NULL,
    region                  TEXT NOT NULL,
    complexity_score        REAL NOT NULL,
    complexity_tier         TEXT NOT NULL,
    carbon_intensity_gco2   REAL NOT NULL,
    carbon_source           TEXT NOT NULL,
    estimated_carbon_gco2   REAL NOT NULL,
    sla_ms                  INTEGER,
    routing_reason          TEXT NOT NULL
);
"""


@contextmanager
def _get_conn():
    """Open a SQLite connection, yield it, then close it."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the database and table if they don't exist yet. Safe to call repeatedly."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)


# ---------------------------------------------------------------------------
# Write a routing decision to the log
# ---------------------------------------------------------------------------

def log_decision(decision: RoutingDecision, prompt: str = "") -> None:
    """
    Write one routing decision row to routing_log.sqlite.
    Non-blocking: if the write fails for any reason, logs a warning and returns.
    Routing must never fail because of a logging error (see FP6 in PRD).
    """
    import datetime
    try:
        with _get_conn() as conn:
            conn.execute(_CREATE_TABLE)   # ensures table exists even on first call
            conn.execute(
                """
                INSERT INTO routing_log (
                    timestamp, prompt_hash, model_id, model_tier, region,
                    complexity_score, complexity_tier, carbon_intensity_gco2,
                    carbon_source, estimated_carbon_gco2, sla_ms, routing_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.datetime.utcnow().isoformat() + "Z",
                    _hash_prompt(prompt),
                    decision.model_id,
                    decision.model_tier,
                    decision.region,
                    decision.complexity_score,
                    decision.complexity_tier,
                    decision.carbon_intensity_gco2_per_kwh,
                    decision.carbon_source,
                    decision.estimated_carbon_gco2,
                    decision.latency_sla_ms,
                    decision.routing_reason,
                ),
            )
    except Exception as e:
        logging.warning(f"routing log write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Query helpers — used by eval_sa1.py
# ---------------------------------------------------------------------------

def fetch_all() -> list[dict]:
    """Return all rows as a list of dicts. Used by eval harness."""
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM routing_log ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def fetch_by_tier(tier: str) -> list[dict]:
    """Return all rows for a given complexity tier (SIMPLE / MEDIUM / COMPLEX)."""
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM routing_log WHERE complexity_tier = ?", (tier,)
        ).fetchall()
        return [dict(row) for row in rows]


def clear_log() -> None:
    """Wipe all rows — useful between eval runs to start fresh."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)   # ensure table exists before first DELETE
        conn.execute("DELETE FROM routing_log")


# ---------------------------------------------------------------------------
# Smoke test — python logger.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from routing_engine import RoutingDecision

    init_db()

    # Write a fake decision
    fake = RoutingDecision(
        model_id="claude-haiku-4-5-20251001",
        model_tier="small",
        region="FR",
        complexity_score=0.25,
        complexity_tier="SIMPLE",
        carbon_intensity_gco2_per_kwh=48.0,
        carbon_source="static_default",
        estimated_carbon_gco2=0.000032,
        latency_sla_ms=None,
        routing_reason="SIMPLE task → small tier | clean grid",
        timestamp=time.time(),
    )

    log_decision(fake, prompt="What is the capital of France?")
    rows = fetch_all()
    print(f"\nLogged {len(rows)} row(s) to {DB_PATH}\n")
    for row in rows:
        print(f"  id={row['id']}  model={row['model_id']}  tier={row['complexity_tier']}"
              f"  carbon={row['estimated_carbon_gco2']:.6f} gCO2  reason={row['routing_reason']}")
