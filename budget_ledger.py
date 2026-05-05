"""
budget_ledger.py — Carbon budget enforcement layer for CAIR

What this file does:
  Tracks cumulative CO2 emissions against daily and monthly caps.
  Returns a tier ceiling that the routing engine must respect before
  applying its per-request optimisation logic.

  This is CAIR's core contribution: time-bounded carbon governance.
  Per-request routers minimise carbon per call. CAIR additionally enforces
  a budget across a period — routing degrades progressively as the cap fills,
  enabling CSRD ESRS E1 aligned carbon governance at the inference layer.

Budget states and what they mean:
  NORMAL      (>50% budget remaining)  → all tiers available (large allowed)
  TIGHTENING  (25–50% remaining)       → medium is the max (large locked out)
  RESTRICTED  (10–25% remaining)       → small is the max (medium + large locked)
  CRITICAL    (<10% remaining)         → small only + flag for human review

The tightest of daily vs. monthly constraint wins.
"""

import sqlite3
import datetime
import logging
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "routing_log.sqlite"

_CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS carbon_ledger (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    period_type      TEXT NOT NULL,        -- 'daily' or 'monthly'
    period_key       TEXT NOT NULL,        -- YYYY-MM-DD  or  YYYY-MM
    cumulative_gco2  REAL NOT NULL DEFAULT 0.0,
    updated_at       TEXT NOT NULL,
    UNIQUE(period_type, period_key)
);
"""


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _today() -> str:
    return datetime.date.today().isoformat()          # YYYY-MM-DD


def _this_month() -> str:
    return datetime.date.today().strftime("%Y-%m")    # YYYY-MM


def _ensure_table() -> None:
    with _get_conn() as conn:
        conn.execute(_CREATE_LEDGER)


# ---------------------------------------------------------------------------
# Write — called by logger.py after every routing decision
# ---------------------------------------------------------------------------

def log_emission(gco2: float) -> None:
    """
    Add gco2 to today's and this month's running totals.
    Non-blocking: a write failure never stops routing.
    """
    _ensure_table()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            for period_type, period_key in [
                ("daily",   _today()),
                ("monthly", _this_month()),
            ]:
                conn.execute(
                    """
                    INSERT INTO carbon_ledger (period_type, period_key, cumulative_gco2, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(period_type, period_key) DO UPDATE SET
                        cumulative_gco2 = cumulative_gco2 + excluded.cumulative_gco2,
                        updated_at      = excluded.updated_at
                    """,
                    (period_type, period_key, gco2, now),
                )
    except Exception as e:
        logging.warning(f"budget ledger write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Read — called at the start of every route() call
# ---------------------------------------------------------------------------

def get_budget_status(daily_cap: float, monthly_cap: float) -> dict:
    """
    Return current budget state and the tier ceiling it enforces.

    Args:
        daily_cap:   Maximum gCO2 allowed per day (from model_registry.yaml)
        monthly_cap: Maximum gCO2 allowed per month

    Returns dict with keys:
        state          — NORMAL | TIGHTENING | RESTRICTED | CRITICAL
        tier_ceiling   — large | medium | small (max tier the router may select)
        remaining_pct  — % of budget still available (tighter of daily/monthly)
        binding_period — 'daily' or 'monthly' (which cap is constraining)
        daily_used_gco2, daily_cap_gco2
        monthly_used_gco2, monthly_cap_gco2
    """
    _ensure_table()

    with _get_conn() as conn:
        daily_used   = _get_cumulative(conn, "daily",   _today())
        monthly_used = _get_cumulative(conn, "monthly", _this_month())

    daily_pct   = max(0.0, (daily_cap   - daily_used)   / daily_cap)
    monthly_pct = max(0.0, (monthly_cap - monthly_used) / monthly_cap)

    # Tightest constraint wins
    remaining_pct  = min(daily_pct, monthly_pct)
    binding_period = "daily" if daily_pct <= monthly_pct else "monthly"

    state, ceiling = _resolve_state(remaining_pct)

    return {
        "state":              state,
        "tier_ceiling":       ceiling,
        "remaining_pct":      round(remaining_pct * 100, 1),
        "binding_period":     binding_period,
        "daily_used_gco2":    round(daily_used,   4),
        "daily_cap_gco2":     daily_cap,
        "monthly_used_gco2":  round(monthly_used, 4),
        "monthly_cap_gco2":   monthly_cap,
    }


def _get_cumulative(conn: sqlite3.Connection, period_type: str, period_key: str) -> float:
    row = conn.execute(
        "SELECT cumulative_gco2 FROM carbon_ledger WHERE period_type=? AND period_key=?",
        (period_type, period_key),
    ).fetchone()
    return row[0] if row else 0.0


def _resolve_state(remaining_pct: float) -> tuple[str, str]:
    """Map remaining budget fraction to (state_name, tier_ceiling)."""
    if remaining_pct > 0.50:
        return "NORMAL",     "large"
    elif remaining_pct > 0.25:
        return "TIGHTENING", "medium"
    elif remaining_pct > 0.10:
        return "RESTRICTED", "small"
    else:
        return "CRITICAL",   "small"


# ---------------------------------------------------------------------------
# Smoke test — python budget_ledger.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Budget Ledger — smoke test\n")

    # Simulate a fresh day with 500 gCO2 daily cap, 12000 monthly cap
    daily_cap   = 500.0
    monthly_cap = 12000.0

    # Log some emissions and check state transitions
    scenarios = [
        (0.0,   "start of day — no emissions logged yet"),
        (100.0, "after 100 gCO2 (20% of daily) — should be NORMAL"),
        (150.0, "after 150 more (50% of daily) — should be TIGHTENING"),
        (75.0,  "after 75 more (75% of daily) — should be RESTRICTED"),
        (70.0,  "after 70 more (89% of daily) — should be CRITICAL"),
    ]

    for emission, label in scenarios:
        if emission > 0:
            log_emission(emission)
        status = get_budget_status(daily_cap, monthly_cap)
        print(f"  {label}")
        print(f"    state={status['state']:<12} ceiling={status['tier_ceiling']:<8} "
              f"remaining={status['remaining_pct']}%  "
              f"daily_used={status['daily_used_gco2']} gCO2\n")
