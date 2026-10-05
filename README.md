# CAIR: Carbon Aware Inference Routing for Large Language Models

**Author:** [@pretzelslab](https://github.com/pretzelslab)<br>
**Status:** Implemented and evaluated<br>
**Preprint:** [Zenodo](https://doi.org/10.5281/zenodo.19934621)<br>
**Implementation:** Reference implementation and evaluation artifacts for CAIR

> Route each LLM request to an appropriate model tier using task complexity, grid carbon intensity, latency requirements, accuracy constraints, and carbon budget signals.

---

## The Problem

LLM inference is not carbon-neutral. A single GPT-4-scale API call can consume 10–30× more energy than a small model call for the same task. Most production systems send every prompt to the same model regardless of whether the task needs it.

The result: a prompt asking "summarise this in one sentence" consumes the same compute as a prompt asking "audit this 40-page contract for GDPR compliance." That is waste — both financially and environmentally.

**The scale of this waste:**
- A 70B-parameter model call uses ~10–30× the energy of a 7B call
- Inference now accounts for 60–90% of total LLM lifecycle energy (surpassing training at steady state)
- A busy production system serving 10M prompts/day can run 40–70% of them on a small model without measurable accuracy loss

No production tooling currently routes at this level.

---

## The Gap — What Exists and What Doesn't

| Tool / Paper | What it does | What it misses |
|---|---|---|
| **Clover (SC'23)** | Carbon-aware ML inference, accuracy/latency/carbon tradeoffs | Research prototype. Not integrated into production serving (vLLM, PyTorch). No real-time grid data. |
| **Green-LLM** | Geographic workload shifting to low-carbon regions | Infrastructure-level (which datacenter), not prompt-level (which model). No per-request routing. |
| **Academic edge work (2023–24)** | 35% emission reduction on edge clusters | Edge-focused. Not applicable to cloud LLM serving. No multi-objective optimisation. |
| **LLM cascade / routing (FrugalGPT, LLM-Blender)** | Route by accuracy/cost | No carbon signal. No live grid data. Accuracy-only optimisation. |
| **Electricity Maps / WattTime** | Live grid carbon intensity data | Data provider only. No routing logic. No LLM integration. |

**The unowned gap:**  
A system that combines **per-prompt complexity scoring** + **real-time grid carbon intensity** + **production serving integration** (vLLM/PyTorch) + **multi-objective routing** (carbon × accuracy × latency simultaneously) does not exist as a shipped, open tool.

This is SA1.

---

## The Idea

Every incoming LLM prompt passes through a routing layer before it reaches any model. The router makes a real-time decision:

```
incoming prompt
      │
      ▼
┌─────────────────────┐
│  Complexity Scorer  │  ← How hard is this task? (token count, syntactic depth,
│                     │    domain signal, instruction count)
└──────────┬──────────┘
           │  complexity score (0.0 – 1.0)
           ▼
┌─────────────────────┐
│  Carbon Feed        │  ← What is the grid doing right now?
│                     │    (Electricity Maps API: gCO2eq/kWh, zone, forecast)
└──────────┬──────────┘
           │  carbon intensity (gCO2eq/kWh)
           ▼
┌─────────────────────┐
│  Routing Engine     │  ← Multi-objective decision:
│                     │    minimise carbon × meet accuracy floor × stay within latency SLA
│                     │    Output: model_id (e.g. "7b", "13b", "70b", "api-large")
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Model Registry     │  ← Available models, their TDP profiles, accuracy benchmarks,
│                     │    latency p50/p95, cost per 1K tokens
└──────────┬──────────┘
           │
           ▼
     inference call
           │
           ▼
┌─────────────────────┐
│  Audit Log          │  ← Per-request: prompt hash, routed model, carbon saved,
│                     │    actual latency, accuracy proxy, decision rationale
└─────────────────────┘
```

**Routing logic (plain English):**

1. Score the prompt's complexity. A one-sentence summarisation task scores ~0.2. A multi-hop legal reasoning task scores ~0.85.
2. Read the current carbon intensity for the serving region. If the grid is dirty (coal-heavy, high gCO2eq/kWh), prefer a smaller model even at marginal accuracy cost.
3. Check the user's latency SLA. If a response is needed in <500ms, that caps the model size regardless of carbon signal.
4. Check the accuracy floor. If the downstream task requires >95% accuracy, the router cannot route to a model below that benchmark.
5. Send the prompt to the model that satisfies all constraints with the lowest carbon cost.

**Routing is not degradation.** 60–70% of real production prompts are structurally simple. Routing them to a 7B model is not a quality compromise — it is correct allocation.

---

## Architecture

### Components

| Component | Responsibility | Implementation |
|---|---|---|
| **Complexity Scorer** | Classify incoming prompt: simple / moderate / complex / expert. Features: token count, sentence count, imperative depth, domain keywords, instruction count. | Python + lightweight classifier (rule-based v1, fine-tuned DistilBERT v2) |
| **Carbon Intensity Feed** | Pull live gCO2eq/kWh for serving zone. Cache with 5-min TTL. Fall back to 24hr rolling average if API unavailable. | Electricity Maps API (free tier: 1 zone · 1 req/min) |
| **Routing Engine** | Multi-objective optimisation: argmin(carbon_cost) subject to accuracy ≥ floor AND latency ≤ SLA. Configurable weights per deployment. | Python — linear scoring v1, LP solver v2 |
| **Model Registry** | Static config: model name, parameter count, TDP profile (W), tokens/sec, accuracy benchmarks (MMLU / task-specific), cost/1K tokens. | YAML config file |
| **Fallback Logic** | If carbon feed unavailable → route by complexity only. If small model returns low-confidence output → retry on larger model (optional). | Python |
| **Audit Logger** | Per-request structured log: timestamp, prompt_hash (no raw text), routed_model, carbon_saved_gCO2eq, latency_ms, accuracy_proxy, routing_reason. | JSON → stdout / file |

### Data flow

```
Client → Router API (FastAPI)
              │
              ├── Complexity Scorer (local, <5ms)
              ├── Carbon Feed (cached, <1ms)
              └── Routing Engine (local, <2ms)
                        │
                        └── Model endpoint (vLLM / Ollama / API)
                                  │
                              Response + audit log entry
```

**Total routing overhead target: <10ms p99.** The routing layer adds negligible latency while saving substantial energy on eligible prompts.

---

## Why This Matters

**For AI sustainability:**  
Inference is already the dominant cost in deployed AI systems. As LLM usage scales, the carbon bill scales with it. Routing is the only intervention that reduces emissions *without* retraining or replacing models.

**For production engineers:**  
The same routing layer that reduces carbon also reduces cost. A system routing 60% of prompts to a 7B model saves ~10× on API cost for those calls. Carbon-awareness and cost-awareness align.

**For governance and regulation:**  
- EU AI Act Art. 53 requires GPAI model providers to report energy consumption
- CSRD Scope 2 emissions include server energy — inference carbon is auditable
- ISSB S2 climate risk disclosures increasingly cover AI operational footprint

SA1 provides both the routing mechanism and the audit trail needed to report on it.

**Competitive position:**  
This is the missing layer between "I know my model uses energy" (existing carbon calculators) and "I systematically reduce that energy at the prompt level" (not yet built as a production tool).

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Router API | FastAPI | Lightweight, async, easy to integrate in front of any serving layer |
| Complexity scorer v1 | Python (rule-based) | Zero dependencies, <5ms latency, interpretable |
| Complexity scorer v2 | DistilBERT (HuggingFace) | ~66MB, runs on CPU, fine-tuneable on domain data |
| Carbon feed | Electricity Maps API | Free tier available, zone-level granularity, JSON |
| Model serving | Ollama (local) / vLLM (GPU) | Both expose OpenAI-compatible API — router is serving-layer-agnostic |
| Model registry | YAML config | Editable, version-controlled, no database needed for Phase 1 |
| Experiment tracking | MLflow | Track routing decisions, carbon savings, accuracy proxy per run |
| Visualisation | Streamlit (Phase 1) / React (Phase 2 portfolio page) | Fast to build for v1 dashboard |
| Audit log | JSON (file / stdout) | Portable, no infrastructure required |

---

## Implementation

CAIR is implemented as a modular routing framework with separate components for prompt complexity assessment, carbon intensity signals, routing decisions, model configuration, budget tracking, audit logging, and evaluation.

| Component | Repository file | Purpose |
|---|---|---|
| Complexity Scorer | `complexity_scorer.py` | Assesses prompt complexity for routing decisions |
| Carbon Feed | `carbon_feed.py` | Provides carbon intensity signals with fallback handling |
| Routing Engine | `routing_engine.py` | Selects an appropriate model tier using routing constraints |
| Model Registry | `model_registry.yaml` | Defines available model profiles and routing characteristics |
| Router API | `router_api.py` | Exposes the CAIR routing workflow through an API |
| Carbon Budget Ledger | `budget_ledger.py` | Tracks carbon budget usage across routing decisions |
| Audit Logger | `logger.py` | Records routing decisions and associated metadata |
| Evaluation | `eval_sa1.py` | Supports evaluation of CAIR routing behavior |

The implementation is designed so that routing logic, carbon signals, model characteristics, and evaluation can be inspected independently rather than being embedded in a single serving component.

---

## Edge Cases

| Scenario | Handling |
|---|---|
| Carbon feed API down | Fall back to 24hr rolling average intensity for zone. Log as `carbon_source: fallback`. |
| All models unavailable except one | Route to available model regardless of complexity. Log forced routing. |
| Latency SLA too tight for any local model | Route to fastest available (API-based). Log SLA override. |
| Prompt complexity score is borderline (0.45–0.55) | Route to middle tier. Do not flip-flop between tiers on retry. |
| User SLA overrides carbon preference | SLA always wins. Log as `carbon_overridden: true`. |
| Grid intensity is very low (clean window) | All complexity tiers can use large model — carbon cost difference is negligible. |
| Cascade retry (small model low confidence) | Optional: retry on next tier up. Logged separately as `cascade: true`. Adds latency. Configurable. |

---

## Carbon Saving Estimate (back-of-envelope)

Assumptions:
- Production system: 1M prompts/day
- 65% of prompts are simple or moderate (realistic for customer support, summarisation, Q&A)
- Large model (70B on A100): ~300W TDP, 50 tokens/sec → ~0.006 kWh per 1K tokens
- Small model (7B on CPU or T4): ~50W TDP, 150 tokens/sec → ~0.0003 kWh per 1K tokens
- Grid intensity: 400 gCO2eq/kWh (EU average)

**Without routing:** 1M × 0.006 kWh × 0.4 kgCO2/kWh = **2,400 kgCO2/day**  
**With routing (65% to 7B):** 650K × 0.0003 + 350K × 0.006 = 195 + 2,100 → × 0.4 = **~918 kgCO2/day**  
**Saving: ~62% reduction in inference carbon** for a system that already exists, with no model changes.

This is the order-of-magnitude case for prompt-level routing. The exact reduction depends on workload distribution, model energy profiles, grid carbon intensity, and routing thresholds. The evaluation artifacts in this repository provide the current CAIR results and methodology.

---

## Relationship to Existing Portfolio Work

| Project | Connection |
|---|---|
| [P2 Carbon Depth Calculator](https://preetibuilds-33d6f6da.vercel.app/carbon-depth) | SA1 uses the same SCI formula and GPU TDP data. Router audit log feeds into carbon calculator as input. |
| [Carbon-Fairness Efficiency Frontier](https://preetibuilds-33d6f6da.vercel.app/carbon-fairness) | SA1 routing decisions are efficiency frontier decisions — same tradeoff visualised at the infrastructure layer. |
| [AC1 Goal Drift Detector](https://github.com/pretzelslab/ac1-agent-drift-detector) | SA1 routing layer could be wrapped as a tool in an agentic pipeline. Carbon-aware agents = natural extension. |

---

## Status

CAIR has progressed from initial concept and architecture into an implemented and evaluated reference framework.

- [x] Concept scoped
- [x] Architecture designed
- [x] Reference implementation completed
- [x] Evaluation artifacts added
- [x] Research preprint published
- [x] Accepted for presentation at IEEE GAISS 2026
- [ ] Further validation and extensions

This repository now serves as the reference implementation and supporting technical artifact set for CAIR.

---

*This repository establishes the design and prior art for SA1 as of 2026-04-30.*

---

## Cite this work

If you reference this framework, please cite the Zenodo preprint:

**Raghuveeran, P. (2026).** *Carbon-Aware Inference Routing for Large Language Models: A Real-Time Framework for Sustainable AI Serving.* Zenodo. https://doi.org/10.5281/zenodo.19934621

```bibtex
@misc{raghuveeran2026cair,
  author       = {Raghuveeran, Preeti},
  title        = {Carbon-Aware Inference Routing for Large Language Models: A Real-Time Framework for Sustainable AI Serving},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19934621},
  url          = {https://doi.org/10.5281/zenodo.19934621}
}
```
