"""
complexity_scorer.py — Step 3 of CAIR

What this file does:
  Reads a task prompt and classifies it as SIMPLE / MEDIUM / COMPLEX.
  Returns a score (0.0–1.0), a tier, and a short reason string.

Two modes:
  1. Claude Haiku scorer (primary) — fast API call, ~250ms, ~$0.0001/call
  2. Heuristic fallback             — no API needed, pure keyword + word count logic

Domain override runs FIRST in both modes:
  If any high-stakes domain keyword is found (legal, medical, compliance, etc.)
  the score is forced to COMPLEX immediately — before Haiku or heuristic runs.
  This implements the "summarise + legal = COMPLEX" rule.
"""

import re
import yaml
import asyncio
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Complexity tier enum
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------
# Everything routing_engine.py needs is in one object.
# confidence: "HIGH" = Claude scored it, "LOW" = heuristic fallback was used.
# When confidence is LOW, the routing engine will route up one tier.

@dataclass
class ComplexityResult:
    score: float        # 0.0–1.0 normalised score
    tier: Tier
    confidence: str     # "HIGH" | "LOW"
    reasoning: str      # short human-readable explanation
    domain_triggered: Optional[str] = None   # which domain keyword fired, if any


# ---------------------------------------------------------------------------
# Load config from model_registry.yaml
# ---------------------------------------------------------------------------
# We read thresholds and domain keywords once at module load.
# This means changing the YAML takes effect on next program start — no code edits.

def _load_config() -> dict:
    registry_path = Path(__file__).parent / "model_registry.yaml"
    try:
        with open(registry_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

_CONFIG = _load_config()

# Complexity score boundaries (with hysteresis buffer)
_THRESHOLDS = _CONFIG.get("complexity_thresholds", {
    "simple_max": 0.40,
    "simple_buffer": 0.45,
    "medium_max": 0.70,
    "medium_buffer": 0.75,
})

# High-stakes domain keyword lists — each key is a domain name, value is list of keywords
_DOMAIN_KEYWORDS: dict[str, list[str]] = _CONFIG.get("high_stakes_domains", {})

# Flatten all domain keywords into a lookup dict: keyword → domain_name
# e.g. {"legal": "legal", "contract": "legal", "GDPR": "compliance", ...}
_KEYWORD_TO_DOMAIN: dict[str, str] = {}
for domain_name, keywords in _DOMAIN_KEYWORDS.items():
    for kw in keywords:
        _KEYWORD_TO_DOMAIN[kw.lower()] = domain_name


# ---------------------------------------------------------------------------
# Score → Tier mapping (applies hysteresis)
# ---------------------------------------------------------------------------
# Hysteresis = buffer zone at boundaries.
# When a score lands in the buffer (e.g. 0.40–0.45), we route UP not down.
# Asymmetry is intentional: over-routing wastes carbon, under-routing breaks quality.

def _score_to_tier(score: float) -> Tier:
    simple_max    = _THRESHOLDS["simple_max"]
    simple_buffer = _THRESHOLDS["simple_buffer"]
    medium_max    = _THRESHOLDS["medium_max"]

    if score <= simple_max:
        return Tier.SIMPLE
    elif score <= simple_buffer:
        return Tier.MEDIUM        # borderline simple → route up to MEDIUM
    elif score <= medium_max:
        return Tier.MEDIUM
    else:
        return Tier.COMPLEX       # includes the medium_buffer zone → COMPLEX


# ---------------------------------------------------------------------------
# Domain override — runs first, before any scoring
# ---------------------------------------------------------------------------
# Scans the prompt for any keyword in the high_stakes_domains list.
# If matched: returns COMPLEX immediately regardless of verb or word count.
# Implements "summarise + legal = COMPLEX" — the domain wins.

def _check_domain_override(prompt: str) -> Optional[tuple[float, str]]:
    """
    Returns (score, domain_name) if a high-stakes domain keyword is found.
    Returns None if no domain keyword matched — normal scoring continues.
    """
    prompt_lower = prompt.lower()
    for keyword, domain_name in _KEYWORD_TO_DOMAIN.items():
        # Use word-boundary matching so "legal" doesn't match "paralegal" incorrectly
        if re.search(rf"\b{re.escape(keyword)}\b", prompt_lower):
            return (0.85, domain_name)
    return None


# ---------------------------------------------------------------------------
# Heuristic scorer — the fallback when Claude API is unavailable
# ---------------------------------------------------------------------------
# Three signals, evaluated in order:
#   1. Reasoning verb — what is the task fundamentally asking?
#   2. Word count     — longer prompts correlate with more complex tasks
#   3. Multi-step markers — "first... then...", "compare X and Y", etc.
# Confidence is always LOW — this triggers a one-tier upgrade in routing_engine.py

_SIMPLE_VERBS = [
    "what is", "who is", "when did", "where is", "how many", "how much",
    "translate", "convert", "calculate", "define", "spell", "list the",
    "name the", "what are the", "give me the",
]

_MEDIUM_VERBS = [
    "summarise", "summarize", "draft", "write a", "explain",
    "describe", "classify", "categorise", "categorize", "outline",
    "create a", "generate a",
]

_COMPLEX_VERBS = [
    "analyse", "analyze", "evaluate", "review", "compare", "assess",
    "identify", "recommend", "critique", "investigate", "diagnose",
    "design", "architect", "refactor", "audit",
]

_MULTISTEP_MARKERS = [
    "first", "then", "finally", "step by step", "consider",
    "taking into account", "in light of", "with respect to",
    "across", "between", "versus", "vs",
]


def _heuristic_score(prompt: str) -> tuple[float, str]:
    """
    Returns (score, reasoning) using keyword + word count logic only.
    No API calls. Always marks confidence as LOW.
    """
    prompt_lower = prompt.lower()
    words = prompt.split()
    word_count = len(words)

    # Start with a base score from word count
    if word_count <= 20:
        base_score = 0.25       # short → probably simple
    elif word_count <= 100:
        base_score = 0.55       # medium length → medium tier
    else:
        base_score = 0.80       # long → probably complex

    # Verb signal — check complex first (most specific), then medium, then simple
    verb_score = base_score
    matched_verb = None

    for verb in _COMPLEX_VERBS:
        if verb in prompt_lower:
            verb_score = max(verb_score, 0.80)
            matched_verb = f"complex verb '{verb}'"
            break

    if matched_verb is None:
        for verb in _MEDIUM_VERBS:
            if verb in prompt_lower:
                verb_score = max(verb_score, 0.55)
                matched_verb = f"medium verb '{verb}'"
                break

    if matched_verb is None:
        for verb in _SIMPLE_VERBS:
            if verb in prompt_lower:
                verb_score = min(verb_score, 0.35)
                matched_verb = f"simple verb '{verb}'"
                break

    # Multi-step marker boost — pushes score toward COMPLEX
    multistep_count = sum(1 for m in _MULTISTEP_MARKERS if m in prompt_lower)
    multistep_boost = min(multistep_count * 0.05, 0.15)   # max +0.15 boost
    final_score = min(verb_score + multistep_boost, 1.0)

    # Handle empty string — default to MEDIUM (route up, never assume simple)
    if word_count == 0:
        return (0.55, "empty prompt — defaulting to MEDIUM")

    reasoning_parts = [f"{word_count} words"]
    if matched_verb:
        reasoning_parts.append(matched_verb)
    if multistep_count > 0:
        reasoning_parts.append(f"{multistep_count} multi-step markers")

    return (final_score, "heuristic: " + ", ".join(reasoning_parts))


# ---------------------------------------------------------------------------
# Claude Haiku scorer — primary scoring path
# ---------------------------------------------------------------------------
# Sends a tightly constrained prompt to Haiku.
# The prompt forces a single integer response: 1, 2, or 3.
# 1 → normalise to 0.25 (SIMPLE), 2 → 0.55 (MEDIUM), 3 → 0.85 (COMPLEX)
# Timeout: if Haiku takes > 5 seconds, we give up and fall through to heuristic.

_HAIKU_SCORE_MAP = {1: 0.25, 2: 0.55, 3: 0.85}

_HAIKU_SYSTEM_PROMPT = """You are a task complexity classifier. Your only job is to rate a task's complexity.

Scoring guide:
  1 = SIMPLE   — single fact lookup, translation, basic calculation, yes/no question
  2 = MEDIUM   — summarisation, drafting, classification, 2–3 reasoning steps
  3 = COMPLEX  — legal/medical/compliance analysis, code review, multi-step reasoning, high-stakes output

Reply with ONLY the number 1, 2, or 3. No explanation. No punctuation. Just the digit."""


async def _score_with_haiku(prompt: str, client: anthropic.AsyncAnthropic) -> Optional[tuple[float, str]]:
    """
    Ask Claude Haiku to score the task complexity.
    Returns (score, reasoning) on success, None on any failure.
    Failure triggers fallback to heuristic scorer.
    """
    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                system=_HAIKU_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Rate this task: {prompt}"}],
            ),
            timeout=5.0,
        )
        raw = response.content[0].text.strip()
        digit = int(raw[0])   # take first character in case model adds punctuation
        if digit not in (1, 2, 3):
            return None
        score = _HAIKU_SCORE_MAP[digit]
        return (score, f"Claude Haiku rated {digit}/3")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public interface — this is the one function routing_engine.py calls
# ---------------------------------------------------------------------------

async def score(prompt: str, anthropic_api_key: Optional[str] = None) -> ComplexityResult:
    """
    Score a task prompt and return its complexity tier.

    Execution order:
      1. Domain override  — checks high_stakes_domains keywords (instant, no API)
      2. Claude Haiku     — primary scorer if api_key provided
      3. Heuristic        — fallback if Haiku unavailable or api_key missing

    Args:
        prompt:           The task description to classify.
        anthropic_api_key: Anthropic API key. If None, skips to heuristic.

    Returns:
        ComplexityResult with .score, .tier, .confidence, .reasoning
    """
    # ── Step 1: Domain override ──────────────────────────────────────────────
    # This runs BEFORE Haiku. A domain keyword match overrides everything.
    domain_match = _check_domain_override(prompt)
    if domain_match:
        domain_score, domain_name = domain_match
        return ComplexityResult(
            score=domain_score,
            tier=Tier.COMPLEX,
            confidence="HIGH",
            reasoning=f"domain override: '{domain_name}' keyword detected",
            domain_triggered=domain_name,
        )

    # ── Step 2: Claude Haiku scorer ──────────────────────────────────────────
    if anthropic_api_key:
        client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        haiku_result = await _score_with_haiku(prompt, client)
        if haiku_result:
            haiku_score, haiku_reason = haiku_result
            return ComplexityResult(
                score=haiku_score,
                tier=_score_to_tier(haiku_score),
                confidence="HIGH",
                reasoning=haiku_reason,
            )

    # ── Step 3: Heuristic fallback ───────────────────────────────────────────
    # confidence = LOW tells routing_engine.py to route up one tier
    heuristic_score, heuristic_reason = _heuristic_score(prompt)
    return ComplexityResult(
        score=heuristic_score,
        tier=_score_to_tier(heuristic_score),
        confidence="LOW",
        reasoning=heuristic_reason,
    )


def score_sync(prompt: str, anthropic_api_key: Optional[str] = None) -> ComplexityResult:
    """Blocking wrapper for use outside async contexts (tests, CLI)."""
    return asyncio.run(score(prompt, anthropic_api_key))


# ---------------------------------------------------------------------------
# Smoke test — python complexity_scorer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")

    test_cases = [
        ("What is 2+2?",                                          Tier.SIMPLE),
        ("What is the capital of France?",                        Tier.SIMPLE),
        ("Summarise this email in 3 bullets",                     Tier.MEDIUM),
        ("Write a Python class for a binary search tree",         Tier.COMPLEX),
        ("Review this contract and identify indemnity clauses",   Tier.COMPLEX),
        ("Translate 'hello' to French",                           Tier.SIMPLE),
        ("Compare GDPR and CCPA data minimisation requirements",  Tier.COMPLEX),
        ("",                                                       Tier.MEDIUM),
        ("Fix my legal contract",                                  Tier.COMPLEX),
        ("Summarise the treatment options in this medical report", Tier.COMPLEX),
    ]

    async def demo():
        print(f"\nComplexity scorer — API key: {'yes' if api_key else 'no (heuristic only)'}\n")
        print(f"{'Task':<55} {'Expected':<10} {'Got':<10} {'Score':<7} {'Conf':<6} Reason")
        print("─" * 120)

        pass_count = 0
        for prompt, expected_tier in test_cases:
            result = await score(prompt, api_key)
            match = "✓" if result.tier == expected_tier else "✗"
            if result.tier == expected_tier:
                pass_count += 1
            display_prompt = (prompt[:52] + "...") if len(prompt) > 55 else prompt
            print(f"{display_prompt:<55} {expected_tier.value:<10} {result.tier.value:<10} "
                  f"{result.score:<7.2f} {result.confidence:<6} {result.reasoning}  {match}")

        print(f"\nResult: {pass_count}/{len(test_cases)} test cases passed")

    asyncio.run(demo())
