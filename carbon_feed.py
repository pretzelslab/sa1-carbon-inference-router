"""
carbon_feed.py — Step 2 of CAIR

What this file does:
  Fetches real-time carbon intensity (gCO2/kWh) for a grid region.
  Priority chain: in-memory cache → Electricity Maps API → static YAML fallback.
  Never blocks routing. If everything fails, returns a static default.
"""

import time
import asyncio
import aiohttp
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure for a single carbon reading
# ---------------------------------------------------------------------------
# Think of CarbonReading as the "receipt" returned after checking the grid.
# source tells the caller how fresh the data is:
#   "live"           = just fetched from Electricity Maps API right now
#   "cached"         = fetched recently, still within 5-minute window
#   "static_default" = API was unavailable, using our research estimates

@dataclass
class CarbonReading:
    zone: str
    carbon_gco2_per_kwh: float
    source: str                  # "live" | "cached" | "static_default"
    timestamp: float             # Unix epoch seconds
    cache_age_seconds: float = 0.0


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
# A simple dict: { zone_code → CarbonReading }
# We store the whole CarbonReading so we can check how old it is later.
# This lives in memory — it resets every time the program restarts.
# That's fine: 5-minute TTL means a restart just fetches fresh data.

_cache: dict[str, CarbonReading] = {}
CACHE_TTL_SECONDS = 300   # 5 minutes


# ---------------------------------------------------------------------------
# Load static fallback values from model_registry.yaml
# ---------------------------------------------------------------------------
# We read the YAML once at module load time, not on every call.
# If the YAML is missing, we hard-code a minimal fallback so the program
# never crashes on import.

def _load_static_defaults() -> dict[str, float]:
    registry_path = Path(__file__).parent / "model_registry.yaml"
    try:
        with open(registry_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("static_carbon_defaults", {})
    except Exception:
        # Absolute last-resort fallback — never let a missing YAML crash routing
        return {"US-EAST": 300, "EU-WEST": 150, "FR": 50, "IN": 650}


STATIC_DEFAULTS: dict[str, float] = _load_static_defaults()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_hit(zone: str) -> Optional[CarbonReading]:
    """Return a cached reading if it exists and is less than 5 minutes old."""
    reading = _cache.get(zone)
    if reading is None:
        return None
    age = time.time() - reading.timestamp
    if age <= CACHE_TTL_SECONDS:
        reading.cache_age_seconds = age
        reading.source = "cached"
        return reading
    return None   # stale — let it expire


def _cache_store(reading: CarbonReading) -> None:
    """Save a fresh live reading into the cache."""
    _cache[reading.zone] = reading


# ---------------------------------------------------------------------------
# Electricity Maps API call
# ---------------------------------------------------------------------------
# The free tier endpoint: GET /v3/carbon-intensity/latest?zone=FR
# Auth header: "auth-token: <your key>"
# We give it 2 seconds before timing out. If it's slow, routing can't wait.

ELECTRICITY_MAPS_BASE = "https://api.electricitymap.org/v3"
API_TIMEOUT_SECONDS = 2.0


async def _fetch_from_api(zone: str, api_key: str) -> Optional[CarbonReading]:
    """
    Call Electricity Maps API for a single zone.
    Returns a CarbonReading on success, None on any failure.
    Failures: timeout, HTTP error, missing key, bad JSON — all return None silently.
    """
    url = f"{ELECTRICITY_MAPS_BASE}/carbon-intensity/latest"
    headers = {"auth-token": api_key}
    params = {"zone": zone}

    try:
        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    # Rate limited — treat same as unavailable
                    return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                # Electricity Maps returns {"zone": "FR", "carbonIntensity": 48, ...}
                intensity = data.get("carbonIntensity")
                if intensity is None:
                    return None
                return CarbonReading(
                    zone=zone,
                    carbon_gco2_per_kwh=float(intensity),
                    source="live",
                    timestamp=time.time(),
                )
    except Exception:
        # Covers: timeout, DNS failure, connection refused, JSON parse error
        return None


# ---------------------------------------------------------------------------
# Static default fallback
# ---------------------------------------------------------------------------

def _static_default(zone: str) -> CarbonReading:
    """
    Return a static carbon estimate when all live sources fail.
    Falls back to US-EAST (300) if the specific zone isn't in our table.
    Always returns something — routing must never get a None here.
    """
    value = STATIC_DEFAULTS.get(zone, STATIC_DEFAULTS.get("US-EAST", 300.0))
    return CarbonReading(
        zone=zone,
        carbon_gco2_per_kwh=value,
        source="static_default",
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Public interface — this is the one function routing_engine.py calls
# ---------------------------------------------------------------------------

async def get_carbon(zone: str, api_key: Optional[str] = None) -> CarbonReading:
    """
    Get carbon intensity for a grid zone.

    Priority chain (matches PRD Section 5):
      1. Cache hit (age < 5 min)  → return immediately, no API call
      2. Live API call             → cache result, return live reading
      3. API failed / no key       → return static default from YAML

    Args:
        zone:    Electricity Maps zone code, e.g. "FR", "DE", "US-EAST"
        api_key: Electricity Maps API key (from .env). If None, skips API call.

    Returns:
        CarbonReading with .carbon_gco2_per_kwh, .source, .cache_age_seconds
    """
    # Step 1: Check cache first — fastest path, no network call
    cached = _cache_hit(zone)
    if cached:
        return cached

    # Step 2: Try live API if we have a key
    if api_key:
        live = await _fetch_from_api(zone, api_key)
        if live:
            _cache_store(live)
            return live

    # Step 3: Everything failed — return static default, never return None
    return _static_default(zone)


async def get_carbon_multi(
    zones: list[str], api_key: Optional[str] = None
) -> dict[str, CarbonReading]:
    """
    Fetch carbon for multiple zones concurrently.
    All zones are fetched in parallel — total wait time = slowest single zone.
    Used by routing_engine.py when comparing regions to find the cleanest option.
    """
    tasks = [get_carbon(zone, api_key) for zone in zones]
    results = await asyncio.gather(*tasks)
    return {zone: reading for zone, reading in zip(zones, results)}


# ---------------------------------------------------------------------------
# Sync wrapper — convenience for non-async callers (e.g. tests, CLI)
# ---------------------------------------------------------------------------

def get_carbon_sync(zone: str, api_key: Optional[str] = None) -> CarbonReading:
    """Blocking wrapper around get_carbon for use outside async contexts."""
    return asyncio.run(get_carbon(zone, api_key))


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify it works
# python carbon_feed.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    key = os.getenv("ELECTRICITY_MAPS_API_KEY")

    async def demo():
        zones = ["FR", "DE", "IN", "US-EAST"]
        print(f"\nFetching carbon intensity for: {zones}")
        print(f"API key present: {'yes' if key else 'no — will use static defaults'}\n")

        readings = await get_carbon_multi(zones, api_key=key)
        for zone, r in readings.items():
            print(f"  {zone:10s}  {r.carbon_gco2_per_kwh:6.1f} gCO2/kWh  [{r.source}]")

        # Second call — should all be cached
        print("\nSecond call (should all be cached):")
        readings2 = await get_carbon_multi(zones, api_key=key)
        for zone, r in readings2.items():
            print(f"  {zone:10s}  {r.carbon_gco2_per_kwh:6.1f} gCO2/kWh  [{r.source}, age {r.cache_age_seconds:.1f}s]")

    asyncio.run(demo())
