"""
eval_sa1.py — Step 6 of CAIR

What this file does:
  Runs the full Phase 1 eval suite against a set of test prompts.
  Measures all 5 eval dimensions from the PRD and prints a JSON report.

How it works:
  1. Runs 50 test prompts through the full routing pipeline
  2. Compares each routing decision against a human-labelled ground truth
  3. Calculates carbon savings vs an always-large (Opus) baseline
  4. Times overhead latency (routing decision only, not inference)
  5. Tests fallback reliability by mocking API failure

Run: python eval_sa1.py
Output: eval_report.json + printed summary table
"""

import os
import json
import time
import asyncio
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from complexity_scorer import score as complexity_score, Tier
from carbon_feed import get_carbon_multi, CarbonReading
from routing_engine import route, RoutingDecision
from logger import log_decision, clear_log

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ELECTRICITY_MAPS_KEY = os.getenv("ELECTRICITY_MAPS_API_KEY")

# ---------------------------------------------------------------------------
# Ground-truth test set — 50 tasks, human-labelled
# ---------------------------------------------------------------------------
# Format: (prompt, expected_tier_string)
# 10 SIMPLE, 20 MEDIUM, 20 COMPLEX — matches PRD eval target distribution

TEST_SET: list[tuple[str, str]] = [
    # ── SIMPLE (10) ──────────────────────────────────────────────────────────
    ("What is 2 + 2?",                                                    "SIMPLE"),
    ("What is the capital of France?",                                    "SIMPLE"),
    ("Translate 'hello' to Spanish.",                                     "SIMPLE"),
    ("Convert 100 Fahrenheit to Celsius.",                                "SIMPLE"),
    ("What year did World War II end?",                                   "SIMPLE"),
    ("Who wrote Pride and Prejudice?",                                    "SIMPLE"),
    ("What is the chemical symbol for gold?",                             "SIMPLE"),
    ("How many days are in a leap year?",                                 "SIMPLE"),
    ("What is the square root of 144?",                                   "SIMPLE"),
    ("What does HTTP stand for?",                                         "SIMPLE"),

    # ── MEDIUM (20) ──────────────────────────────────────────────────────────
    ("Summarise the key points of this product launch announcement.",     "MEDIUM"),
    ("Draft a short professional email declining a meeting request.",     "MEDIUM"),
    ("Explain what a convolutional neural network does in plain English.","MEDIUM"),
    ("Classify the sentiment of this customer review as positive or negative.", "MEDIUM"),
    ("List five best practices for writing clean Python code.",           "MEDIUM"),
    ("Describe the main differences between SQL and NoSQL databases.",    "MEDIUM"),
    ("Write a brief job description for a data analyst role.",            "MEDIUM"),
    ("Summarise this article about renewable energy trends.",             "MEDIUM"),
    ("Explain the concept of gradient descent in machine learning.",      "MEDIUM"),
    ("Draft a polite follow-up email after a job interview.",             "MEDIUM"),
    ("Outline the steps to deploy a Flask app to a cloud server.",       "MEDIUM"),
    ("Explain what ESG reporting means for a non-specialist audience.",   "MEDIUM"),
    ("Create a short FAQ for a SaaS product's onboarding page.",         "MEDIUM"),
    ("Summarise the pros and cons of remote work for employees.",        "MEDIUM"),
    ("Write a one-paragraph product description for a smartwatch.",      "MEDIUM"),
    ("Describe what a REST API is and give a simple example.",           "MEDIUM"),
    ("Explain how carbon credits work in three short paragraphs.",       "MEDIUM"),
    ("List the key clauses typically found in an employment contract.",  "MEDIUM"),
    ("Summarise the main features of the EU AI Act.",                    "MEDIUM"),
    ("Draft a brief executive summary for a market research report.",    "MEDIUM"),

    # ── COMPLEX (20) ─────────────────────────────────────────────────────────
    ("Review this contract and identify all indemnity and liability clauses.", "COMPLEX"),
    ("Analyse the GDPR compliance gaps in this data processing agreement.",    "COMPLEX"),
    ("Compare GDPR and CCPA data minimisation requirements across five dimensions.", "COMPLEX"),
    ("Evaluate the security architecture of this microservices design.",       "COMPLEX"),
    ("Perform a code review of this Python authentication module and flag vulnerabilities.", "COMPLEX"),
    ("Diagnose the root cause of this patient's symptoms based on the clinical notes.", "COMPLEX"),
    ("Assess the investment risk of this portfolio given current market conditions.", "COMPLEX"),
    ("Design a fault-tolerant distributed caching system for 10M requests/day.", "COMPLEX"),
    ("Audit this AI model card for compliance with the EU AI Act high-risk requirements.", "COMPLEX"),
    ("Review the indemnity provisions of this software licensing agreement.",   "COMPLEX"),
    ("Analyse the carbon accounting methodology in this CSRD disclosure report.", "COMPLEX"),
    ("Evaluate the legal risk of this cross-border data transfer mechanism.",   "COMPLEX"),
    ("Summarise the treatment protocol options for this oncology case.",        "COMPLEX"),
    ("Identify all compliance obligations triggered by this GDPR data breach.", "COMPLEX"),
    ("Refactor this legacy authentication system to eliminate SQL injection vulnerabilities.", "COMPLEX"),
    ("Compare the liability frameworks under UK GDPR and EU GDPR post-Brexit.", "COMPLEX"),
    ("Assess whether this AI system meets the EU AI Act's transparency requirements.", "COMPLEX"),
    ("Review this financial derivatives contract for hidden risk provisions.",  "COMPLEX"),
    ("Analyse the governance gaps in this AI system's model documentation.",    "COMPLEX"),
    ("Evaluate the clinical trial design for statistical validity and bias.",   "COMPLEX"),
]

# Carbon intensity for a clean reference region (FR) used in eval runs
EVAL_REGION = "FR"
EVAL_CARBON_FALLBACK = CarbonReading(
    zone=EVAL_REGION,
    carbon_gco2_per_kwh=50.0,
    source="static_default",
    timestamp=time.time(),
)

# Always-large baseline: what carbon would be used if every task went to Opus
# Using Opus carbon_per_1k = 1.20, avg ~100 tokens per task, region intensity 50
ALWAYS_LARGE_CARBON_PER_TASK = (100 / 1000) * 1.20 * (50 / 100)   # = 0.00060 gCO2


# ---------------------------------------------------------------------------
# E1 — Routing Precision
# ---------------------------------------------------------------------------
# % of SIMPLE tasks routed to small model
# % of COMPLEX tasks routed to large model (must be 100% — never under-route)
# MEDIUM precision is tracked but not a hard target

def eval_routing_precision(results: list[tuple[str, RoutingDecision]]) -> dict:
    counts = {"SIMPLE": {"correct": 0, "total": 0},
              "MEDIUM":  {"correct": 0, "total": 0},
              "COMPLEX": {"correct": 0, "total": 0}}

    tier_to_model = {"SIMPLE": "small", "MEDIUM": "medium", "COMPLEX": "large"}

    for expected_tier, decision in results:
        counts[expected_tier]["total"] += 1
        if decision.model_tier == tier_to_model[expected_tier]:
            counts[expected_tier]["correct"] += 1

    precision = {}
    for tier, c in counts.items():
        precision[tier.lower()] = round(c["correct"] / c["total"], 3) if c["total"] > 0 else 0.0

    return precision


# ---------------------------------------------------------------------------
# E2 — Carbon Savings vs Always-Large Baseline
# ---------------------------------------------------------------------------
# (carbon_always_large - carbon_CAIR) / carbon_always_large × 100%

def eval_carbon_savings(results: list[tuple[str, RoutingDecision]]) -> float:
    total_cair = sum(d.estimated_carbon_gco2 for _, d in results)
    total_baseline = ALWAYS_LARGE_CARBON_PER_TASK * len(results)
    if total_baseline == 0:
        return 0.0
    savings_pct = (total_baseline - total_cair) / total_baseline * 100
    return round(savings_pct, 2)


# ---------------------------------------------------------------------------
# E3 — Quality Delta (Phase 2 placeholder)
# ---------------------------------------------------------------------------
# Requires running actual inference on both Haiku and Opus and comparing outputs.
# Skipped in Phase 1 — returns None with an explanation.

def eval_quality_delta() -> Optional[float]:
    return None   # Phase 2: cosine similarity between Haiku and Opus outputs


# ---------------------------------------------------------------------------
# E4 — Overhead Latency
# ---------------------------------------------------------------------------
# Times the routing decision only (no actual inference call).
# Runs N routing calls in heuristic mode (no API) for a clean latency baseline.

async def eval_overhead_latency(n: int = 100) -> dict:
    sample_prompt = "Summarise this document in three bullet points."
    latencies_ms: list[float] = []

    carbon = {EVAL_REGION: EVAL_CARBON_FALLBACK}

    for _ in range(n):
        t0 = time.perf_counter()
        complexity = await complexity_score(sample_prompt, anthropic_api_key=None)
        decision = route(complexity, carbon, prompt=sample_prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

    return {
        "p50_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(sorted(latencies_ms)[int(n * 0.95)], 2),
        "p99_ms": round(sorted(latencies_ms)[int(n * 0.99)], 2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# E5 — Fallback Reliability
# ---------------------------------------------------------------------------
# Calls the router with no API keys at all.
# Every single call must return a valid routing decision (never None, never crash).

async def eval_fallback_reliability(n: int = 20) -> float:
    test_prompts = [prompt for prompt, _ in TEST_SET[:n]]
    successes = 0
    carbon = {EVAL_REGION: EVAL_CARBON_FALLBACK}

    for prompt in test_prompts:
        try:
            complexity = await complexity_score(prompt, anthropic_api_key=None)
            decision = route(complexity, carbon, prompt=prompt)
            if decision and decision.model_id:
                successes += 1
        except Exception:
            pass   # failure counts as 0

    return round(successes / n, 3)


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

async def run_eval() -> dict:
    print("\n" + "═" * 70)
    print("  CAIR — Phase 1 Eval Suite")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("═" * 70)

    # ── Run all test prompts through the routing pipeline ───────────────────
    print(f"\nRunning {len(TEST_SET)} test prompts through router...")
    carbon = {EVAL_REGION: EVAL_CARBON_FALLBACK}
    results: list[tuple[str, RoutingDecision]] = []

    clear_log()   # fresh log for this eval run

    for prompt, expected_tier in TEST_SET:
        complexity = await complexity_score(prompt, anthropic_api_key=ANTHROPIC_API_KEY)
        decision = route(complexity, carbon, prompt=prompt)
        log_decision(decision, prompt=prompt)
        results.append((expected_tier, decision))

    # ── E1: Routing Precision ────────────────────────────────────────────────
    print("\nE1 — Routing Precision:")
    precision = eval_routing_precision(results)
    target_simple = 0.85
    target_complex = 1.0
    for tier, score in precision.items():
        target = target_complex if tier == "complex" else target_simple
        status = "✓" if score >= target else "✗"
        print(f"  {tier:<10} {score:.0%}  (target ≥ {target:.0%})  {status}")

    # ── E2: Carbon Savings ───────────────────────────────────────────────────
    print("\nE2 — Carbon Savings vs Always-Large (Opus) Baseline:")
    savings = eval_carbon_savings(results)
    status = "✓" if savings >= 40.0 else "✗"
    print(f"  {savings:.1f}% reduction  (target ≥ 40%)  {status}")

    # ── E3: Quality Delta ────────────────────────────────────────────────────
    print("\nE3 — Quality Delta:")
    print("  Skipped (Phase 2) — requires live inference comparison")

    # ── E4: Overhead Latency ─────────────────────────────────────────────────
    print("\nE4 — Routing Overhead Latency (heuristic mode, n=100):")
    latency = await eval_overhead_latency(n=100)
    status = "✓" if latency["p95_ms"] <= 50 else "✗"
    print(f"  P50={latency['p50_ms']}ms  P95={latency['p95_ms']}ms  P99={latency['p99_ms']}ms"
          f"  (target P95 ≤ 50ms heuristic)  {status}")

    # ── E5: Fallback Reliability ─────────────────────────────────────────────
    print("\nE5 — Fallback Reliability (no API keys):")
    reliability = await eval_fallback_reliability(n=20)
    status = "✓" if reliability == 1.0 else "✗"
    print(f"  {reliability:.0%} of calls returned a valid decision  (target 100%)  {status}")

    # ── Assemble report ──────────────────────────────────────────────────────
    report = {
        "routing_precision": precision,
        "carbon_savings_pct": savings,
        "quality_delta_mean": None,
        "overhead_latency": latency,
        "fallback_reliability": reliability,
        "test_cases_run": len(TEST_SET),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'─' * 70}")
    print(f"Report saved → {report_path}")
    print("═" * 70 + "\n")

    return report


if __name__ == "__main__":
    asyncio.run(run_eval())
