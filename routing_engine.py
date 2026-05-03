"""
routing_engine.py — Step 4 of CAIR

What this file does:
  Combines complexity score + carbon readings + SLA constraint into one
  routing decision: which model to use, in which region, and why.

5-step decision logic:
  1. Start from complexity tier → pick candidate model
  2. Carbon modifier → upgrade on clean grid, downgrade on dirty grid
  3. SLA constraint → downgrade if model is too slow for the latency budget
  4. Confidence penalty → if heuristic scored the task, route up one tier
  5. Calculate carbon estimate → attach to output for eval tracking
"""

import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from complexity_scorer import ComplexityResult, Tier
from carbon_feed import CarbonReading


# ---------------------------------------------------------------------------
# Load model registry from YAML
# ---------------------------------------------------------------------------
# The registry is the menu of models CAIR can route to.
# We load it once and build lookup structures for fast access during routing.

def _load_registry() -> dict:
    registry_path = Path(__file__).parent / "model_registry.yaml"
    with open(registry_path, "r") as f:
        return yaml.safe_load(f)

_REGISTRY = _load_registry()

# List of model dicts, ordered small → medium → large
# Each dict has: id, tier, carbon_per_1k_tokens_gCO2, latency_p50_ms,
#                context_window_tokens, quality_score, preferred_regions
_MODELS: list[dict] = _REGISTRY["models"]

# Tier order for upgrade/downgrade arithmetic
_TIER_ORDER = ["small", "medium", "large"]

# Routing weights (quality / carbon / latency) — from YAML, sum to 1.0
_WEIGHTS = _REGISTRY.get("routing_weights", {
    "quality_weight": 0.5,
    "carbon_weight": 0.3,
    "latency_weight": 0.2,
})

# Carbon thresholds that trigger modifier logic
_CLEAN_GRID_THRESHOLD = 100   # gCO2/kWh — very clean, allow one tier up
_DIRTY_GRID_THRESHOLD = 500   # gCO2/kWh — very dirty, force one tier down


# ---------------------------------------------------------------------------
# Routing decision output
# ---------------------------------------------------------------------------
# This is what routing_engine returns — and what gets logged to SQLite.

@dataclass
class RoutingDecision:
    model_id: str                       # exact Anthropic model string
    model_tier: str                     # small | medium | large
    region: str                         # zone code used for carbon calculation
    complexity_score: float             # 0.0–1.0 from complexity_scorer
    complexity_tier: str                # SIMPLE | MEDIUM | COMPLEX
    carbon_intensity_gco2_per_kwh: float
    carbon_source: str                  # live | cached | static_default
    estimated_carbon_gco2: float        # estimated gCO2 for this request
    latency_sla_ms: Optional[int]       # input SLA, if any
    routing_reason: str                 # human-readable explanation of decision
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Model lookup helpers
# ---------------------------------------------------------------------------

def _models_by_tier(tier: str) -> list[dict]:
    """Return all models matching a given tier string."""
    return [m for m in _MODELS if m["tier"] == tier]


def _get_model(tier: str) -> Optional[dict]:
    """Return the first model matching a tier. Returns None if tier not in registry."""
    matches = _models_by_tier(tier)
    return matches[0] if matches else None


def _tier_index(tier: str) -> int:
    """Convert tier string to integer for arithmetic. small=0, medium=1, large=2."""
    return _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 1


def _tier_from_index(idx: int) -> str:
    """Clamp index to valid range and return tier string."""
    return _TIER_ORDER[max(0, min(idx, len(_TIER_ORDER) - 1))]


# ---------------------------------------------------------------------------
# Token count estimator
# ---------------------------------------------------------------------------
# We don't run a real tokeniser here — that would add a dependency and latency.
# The approximation (characters / 4) is standard and close enough for carbon estimates.
# A more accurate estimate would use tiktoken, but this is Phase 1.

def _estimate_tokens(prompt: str) -> int:
    """Rough token estimate: ~4 characters per token. Minimum 10 tokens."""
    return max(len(prompt) // 4, 10)


# ---------------------------------------------------------------------------
# Carbon estimate calculator
# ---------------------------------------------------------------------------
# Formula from PRD Section 5:
#   carbon_gCO2 = (tokens / 1000) × model.carbon_per_1k_tokens × (region_intensity / 100)
#
# The (region_intensity / 100) factor scales the model's base carbon figure
# by how clean or dirty the actual grid is in that region right now.
# A model running in France (50 gCO2/kWh) emits 6× less than in India (650).

def _estimate_carbon(model: dict, token_count: int, region_intensity: float) -> float:
    base = model["carbon_per_1k_tokens_gCO2"]
    return (token_count / 1000) * base * (region_intensity / 100)


# ---------------------------------------------------------------------------
# Region selector
# ---------------------------------------------------------------------------
# Picks the cleanest available region from the model's preferred_regions list.
# "Cleanest" = lowest gCO2/kWh in the carbon readings we have.
# Falls back to the first preferred region if no carbon data is available.

def _pick_cleanest_region(
    model: dict, carbon_readings: dict[str, CarbonReading]
) -> tuple[str, CarbonReading]:
    """
    Returns (zone, CarbonReading) for the cleanest region this model prefers.
    If no readings are available for any preferred region, uses the first preferred
    region with a static default value of 300 (US-EAST approximation).
    """
    preferred = model.get("preferred_regions", ["US-EAST"])

    best_zone = preferred[0]
    best_reading = carbon_readings.get(best_zone)

    for zone in preferred:
        reading = carbon_readings.get(zone)
        if reading is None:
            continue
        if best_reading is None or reading.carbon_gco2_per_kwh < best_reading.carbon_gco2_per_kwh:
            best_zone = zone
            best_reading = reading

    # If we still have no reading, fabricate a static default so routing never blocks
    if best_reading is None:
        best_reading = CarbonReading(
            zone=best_zone,
            carbon_gco2_per_kwh=300.0,
            source="static_default",
            timestamp=time.time(),
        )

    return best_zone, best_reading


# ---------------------------------------------------------------------------
# SLA guard
# ---------------------------------------------------------------------------
# If the chosen model's median latency exceeds the SLA budget,
# downgrade to the next faster tier. Repeat until SLA is met or we hit
# the smallest model (can't go lower).
# SLA always beats carbon savings — never sacrifice latency for carbon.

def _apply_sla_constraint(
    tier_idx: int, sla_ms: Optional[int], reason_parts: list[str]
) -> int:
    if sla_ms is None:
        return tier_idx
    while tier_idx > 0:
        tier = _tier_from_index(tier_idx)
        model = _get_model(tier)
        if model and model["latency_p50_ms"] <= sla_ms:
            break
        reason_parts.append(f"SLA {sla_ms}ms → downgrade from {tier}")
        tier_idx -= 1
    return tier_idx


# ---------------------------------------------------------------------------
# Main routing function — this is the one function everything else calls
# ---------------------------------------------------------------------------

def route(
    complexity: ComplexityResult,
    carbon_readings: dict[str, CarbonReading],
    sla_ms: Optional[int] = None,
    prompt: str = "",
) -> RoutingDecision:
    """
    Produce a routing decision from complexity + carbon + SLA inputs.

    Args:
        complexity:      Output of complexity_scorer.score()
        carbon_readings: Dict of zone → CarbonReading from carbon_feed.get_carbon_multi()
        sla_ms:          Optional latency budget in milliseconds
        prompt:          Original task string (used for token estimation only)

    Returns:
        RoutingDecision with model_id, region, carbon estimate, and reasoning
    """
    reason_parts: list[str] = []

    # ── Step 1: Start from complexity tier ───────────────────────────────────
    # SIMPLE → small (Haiku), MEDIUM → medium (Sonnet), COMPLEX → large (Opus)
    # COMPLEX is a hard floor — carbon never overrides a complex task.

    tier_map = {
        Tier.SIMPLE: "small",
        Tier.MEDIUM: "medium",
        Tier.COMPLEX: "large",
    }
    base_tier = tier_map[complexity.tier]
    tier_idx = _tier_index(base_tier)
    reason_parts.append(f"{complexity.tier.value} task → {base_tier} tier")

    # ── Step 2: Confidence penalty ───────────────────────────────────────────
    # If heuristic was used (confidence = LOW), route up one tier.
    # Better to over-route than under-route when we're uncertain.

    if complexity.confidence == "LOW":
        if tier_idx < len(_TIER_ORDER) - 1:
            tier_idx += 1
            reason_parts.append(f"heuristic confidence LOW → route up to {_tier_from_index(tier_idx)}")

    # ── Step 3: Hard floor for COMPLEX ──────────────────────────────────────
    # COMPLEX tasks must always use the large model.
    # Carbon cannot override this — quality is non-negotiable for complex tasks.
    # This is evaluated AFTER the confidence penalty so we don't double-upgrade.

    if complexity.tier == Tier.COMPLEX:
        tier_idx = _tier_index("large")
        reason_parts.append("COMPLEX: carbon modifier disabled, large model required")

    # ── Step 4: Carbon modifier (SIMPLE and MEDIUM only) ────────────────────
    # Find the best (cleanest) region for the candidate model.
    # Then apply the grid-based modifier:
    #   Very clean grid (< 100): allow one tier upgrade (bonus — you can go bigger)
    #   Very dirty grid (> 500): force one tier downgrade (save carbon)
    # Neither modifier applies to COMPLEX tasks — handled by the hard floor above.

    candidate_model = _get_model(_tier_from_index(tier_idx))
    if candidate_model is None:
        candidate_model = _MODELS[-1]   # safety: use largest if something's wrong

    zone, zone_reading = _pick_cleanest_region(candidate_model, carbon_readings)
    grid_intensity = zone_reading.carbon_gco2_per_kwh

    if complexity.tier != Tier.COMPLEX:
        if grid_intensity < _CLEAN_GRID_THRESHOLD:
            # Clean grid — allow one tier up if not already at large
            if tier_idx < _tier_index("large"):
                tier_idx += 1
                reason_parts.append(
                    f"clean grid ({grid_intensity:.0f} gCO2/kWh < {_CLEAN_GRID_THRESHOLD}) → upgrade to {_tier_from_index(tier_idx)}"
                )
        elif grid_intensity > _DIRTY_GRID_THRESHOLD:
            # Dirty grid — force one tier down if not already at small
            if tier_idx > _tier_index("small"):
                tier_idx -= 1
                reason_parts.append(
                    f"dirty grid ({grid_intensity:.0f} gCO2/kWh > {_DIRTY_GRID_THRESHOLD}) → downgrade to {_tier_from_index(tier_idx)}"
                )

    # ── Step 5: SLA constraint ───────────────────────────────────────────────
    # After all tier adjustments, check if the model can meet the latency SLA.
    # SLA beats carbon — always. If SLA can't be met, downgrade until it can.

    tier_idx = _apply_sla_constraint(tier_idx, sla_ms, reason_parts)

    # ── Step 6: Resolve final model + region ─────────────────────────────────
    final_tier = _tier_from_index(tier_idx)
    final_model = _get_model(final_tier)

    if final_model is None:
        # Absolute fallback — should never happen with a valid registry
        final_model = _MODELS[0]
        reason_parts.append("fallback: no model found for tier, using smallest")

    # Re-pick cleanest region for the final model (may differ from candidate)
    final_zone, final_reading = _pick_cleanest_region(final_model, carbon_readings)

    # ── Step 7: Carbon estimate ───────────────────────────────────────────────
    token_count = _estimate_tokens(prompt)
    carbon_estimate = _estimate_carbon(
        final_model, token_count, final_reading.carbon_gco2_per_kwh
    )

    return RoutingDecision(
        model_id=final_model["id"],
        model_tier=final_tier,
        region=final_zone,
        complexity_score=complexity.score,
        complexity_tier=complexity.tier.value,
        carbon_intensity_gco2_per_kwh=final_reading.carbon_gco2_per_kwh,
        carbon_source=final_reading.source,
        estimated_carbon_gco2=round(carbon_estimate, 6),
        latency_sla_ms=sla_ms,
        routing_reason=" | ".join(reason_parts),
    )


# ---------------------------------------------------------------------------
# Smoke test — python routing_engine.py
# ---------------------------------------------------------------------------
# Tests the 10 routing cases from the PRD (TC-R1 through TC-R10).
# Runs without any API calls — uses pre-built CarbonReading objects directly.

if __name__ == "__main__":
    from complexity_scorer import ComplexityResult, Tier

    def make_carbon(zone: str, intensity: float, source: str = "static_default") -> dict:
        return {zone: CarbonReading(zone=zone, carbon_gco2_per_kwh=intensity,
                                    source=source, timestamp=time.time())}

    test_cases = [
        # (label, tier, confidence, carbon_zone, carbon_intensity, sla_ms, expected_model_tier)
        ("TC-R1  SIMPLE  + clean grid (FR 50)",   Tier.SIMPLE,  "HIGH", "FR",      50,  None, "small"),
        ("TC-R2  SIMPLE  + dirty grid (IN 650)",  Tier.SIMPLE,  "HIGH", "IN",      650, None, "small"),
        ("TC-R3  MEDIUM  + clean grid (FR 50)",   Tier.MEDIUM,  "HIGH", "FR",      50,  None, "medium"),
        ("TC-R4  MEDIUM  + dirty grid (IN 650)",  Tier.MEDIUM,  "HIGH", "IN",      650, None, "small"),
        ("TC-R5  COMPLEX + clean grid (FR 50)",   Tier.COMPLEX, "HIGH", "FR",      50,  None, "large"),
        ("TC-R6  COMPLEX + dirty grid (IN 650)",  Tier.COMPLEX, "HIGH", "IN",      650, None, "large"),
        ("TC-R7  MEDIUM  + SLA 200ms (DE 300)",   Tier.MEDIUM,  "HIGH", "DE",      300, 200,  "small"),
        ("TC-R8  SIMPLE  + loose SLA 5000ms",     Tier.SIMPLE,  "HIGH", "FR",      50,  5000, "small"),
        ("TC-R9  MEDIUM  + SLA 800ms (FR 50)",    Tier.MEDIUM,  "HIGH", "FR",      50,  900,  "medium"),
        ("TC-R10 COMPLEX + API down (fallback)",  Tier.COMPLEX, "HIGH", "US-EAST", 300, None, "large"),
    ]

    # Map tier string to Complexity score for test setup
    score_map = {"SIMPLE": (0.25, Tier.SIMPLE), "MEDIUM": (0.55, Tier.MEDIUM), "COMPLEX": (0.85, Tier.COMPLEX)}

    print(f"\n{'Test':<45} {'Expected':<10} {'Got':<10} {'Region':<10} {'Carbon gCO2/kWh':<18} Reason")
    print("─" * 130)

    pass_count = 0
    for label, tier, conf, zone, intensity, sla, expected_tier in test_cases:
        score_val = score_map[tier.value][0]
        complexity = ComplexityResult(
            score=score_val, tier=tier, confidence=conf,
            reasoning="test input"
        )
        carbon = make_carbon(zone, intensity)
        decision = route(complexity, carbon, sla_ms=sla, prompt="test prompt with ~20 chars")

        match = "✓" if decision.model_tier == expected_tier else "✗"
        if decision.model_tier == expected_tier:
            pass_count += 1
        print(f"{label:<45} {expected_tier:<10} {decision.model_tier:<10} "
              f"{decision.region:<10} {decision.carbon_intensity_gco2_per_kwh:<18.1f} {match}")

    print(f"\nResult: {pass_count}/{len(test_cases)} routing test cases passed")
