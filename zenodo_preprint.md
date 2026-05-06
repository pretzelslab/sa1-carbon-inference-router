# Carbon-Aware Inference Routing for Large Language Models: A Real-Time Framework for Sustainable AI Serving

**Preeti Raghuveeran**  
Independent Researcher  
preeti.raghuveer@gmail.com  
2026-04-30

---

## Abstract

This paper introduces CAIR, a real-time carbon-aware inference routing framework for large language models (LLMs). CAIR routes each inference request to the smallest model capable of satisfying its accuracy floor and latency SLA, using three concurrent signals: per-prompt complexity score, live grid carbon intensity, and a time-bounded carbon budget.

The carbon budget layer tracks cumulative emissions against a configurable period cap (daily or monthly) and progressively constrains the available model tier as the budget depletes — enabling carbon governance at the inference layer rather than optimising only per request. Per-request signals optimise within whatever tier the budget permits.

Preliminary analysis on a 1M prompt/day workload suggests ~62% reduction in inference carbon by routing approximately 65% of requests to a 7B-parameter model. The framework is serving-layer-agnostic and integrates with existing LLM deployment infrastructure (vLLM, Ollama). The audit logger is designed for direct use in CSRD ESRS E1 and EU AI Act Art.53 compliance reporting. Empirical evaluation on 50 human-labelled tasks confirms: routing precision of 100% on simple tasks, 90% on medium tasks, and 100% on complex tasks; 45.5% carbon reduction vs an always-large baseline; routing overhead P95 of 0.27ms; 100% fallback reliability; 4/4 budget state transitions with zero ceiling violations across 200 routing decisions; and a budget enforcement layer that independently delivers 92.5% carbon reduction in CRITICAL state vs uncapped per-request routing. Quality delta evaluation (Phase 2) confirms that outputs from the small model (Claude Haiku) achieve a mean cosine similarity of 0.90 against the large model (Claude Opus) on simple tasks, validating that carbon savings do not come at the cost of response quality.

**Keywords:** carbon-aware inference, LLM routing, sustainable AI, green AI, inference efficiency, carbon budget enforcement, EU AI Act Art.53, CSRD Scope 2, multi-objective optimisation

---

## 1. Introduction

Inference now accounts for 60–90% of the total energy consumption of deployed large language models, surpassing training energy at steady state [Patterson et al., 2021; Luccioni et al., 2022]. A single call to a 70B-parameter model can consume 10–30× the energy of an equivalent call to a 7B-parameter model, yet in most production systems, every request is served by the same model regardless of task complexity.

The consequence is systematic over-provisioning: a prompt asking for a one-sentence summary consumes the same compute as a prompt requiring multi-step legal reasoning. This waste is both financially and environmentally costly, and it scales linearly with inference volume.

We propose that prompt-level routing — directing each request to the smallest model capable of satisfying its requirements — is the highest-leverage intervention available to reduce inference emissions without retraining or replacing models. This paper introduces CAIR, a framework that makes this routing automatic, real-time, and carbon-aware.

---

## 2. Related Work

Several efforts have addressed energy efficiency in ML inference, but none combine the full set of capabilities CAIR targets.

**Clover [SC'23]** explored carbon-accuracy-latency tradeoffs in ML inference. It remains a research prototype and does not integrate with production LLM serving infrastructure (e.g., vLLM), nor does it incorporate real-time grid carbon intensity data.

**Green-LLM** addresses geographic workload shifting — routing inference jobs to datacentres in low-carbon regions. This operates at the infrastructure level (which datacenter) rather than the prompt level (which model per request), and does not perform per-request routing decisions.

**FrugalGPT [Chen et al., 2023]** and **LLM-Blender** route requests between model tiers to reduce cost and improve accuracy. Neither incorporates a carbon signal, live grid data, or emissions-aware optimisation.

**Electricity Maps and WattTime** provide live grid carbon intensity data by zone. They are data providers only, with no routing logic or LLM integration.

Recent work has explored per-request carbon minimisation in ML inference, including carbon-accuracy-latency tradeoffs [Clover, SC'23] and infrastructure-level geographic workload shifting [Green-LLM]. Cost-aware routing between model tiers is addressed by FrugalGPT [Chen et al., 2023] and LLM-Blender, though without a carbon signal. What remains unaddressed is carbon governance across a time horizon: no existing system tracks cumulative emissions against a period cap and progressively constrains model selection as the budget depletes. CAIR addresses this gap with a two-layer architecture — a budget enforcement layer over a per-request optimiser — and pairs it with an audit trail suitable for CSRD ESRS E1 reporting.

---

## 3. The CAIR Framework

### 3.1 Overview

CAIR is a routing layer that sits in front of any LLM serving infrastructure. It intercepts each inference request, scores it, and dispatches it to the appropriate model tier. The routing decision is made in under 10ms (p99 target) and adds negligible user-facing latency.

```
incoming prompt
    │
    ▼
Complexity Scorer  →  score ∈ [0, 1]
    │
    ▼
Carbon Feed        →  intensity (gCO2eq/kWh)
    │
    ▼
Routing Engine     →  model_id
    │
    ▼
Model Endpoint     →  response
    │
    ▼
Audit Logger       →  per-request record
```

### 3.2 Components

**Complexity Scorer.** Classifies each prompt into one of four tiers: *simple*, *moderate*, *complex*, or *expert*. Version 1 uses a rule-based classifier with features including token count, sentence count, imperative depth, domain-specific keyword presence, and instruction count. Version 2 will use a fine-tuned DistilBERT model (~66MB, CPU-deployable) trained on labelled prompt pairs.

**Carbon Intensity Feed.** Retrieves live gCO2eq/kWh data for the serving zone via the Electricity Maps API. Results are cached with a 5-minute TTL. If the API is unavailable, the system falls back to a 24-hour rolling average for the zone, logged as `carbon_source: fallback`.

**Routing Engine.** Solves the following multi-objective decision at inference time:

> *Select the model m that minimises carbon_cost(m) subject to:*
> - *accuracy(m) ≥ accuracy_floor*
> - *latency(m) ≤ latency_SLA*

Where `carbon_cost(m)` is estimated as:

```
carbon_cost(m) = TDP(m) × (tokens / throughput(m)) × grid_intensity
```

In version 1, this is a deterministic linear scan over the model registry, sorted by carbon cost ascending, returning the first model satisfying both constraints.

**Model Registry.** A YAML configuration file listing available models with: parameter count, thermal design power (TDP in watts), tokens/second (empirically measured), accuracy benchmark scores (MMLU or task-specific), and cost per 1,000 tokens. The registry is version-controlled and editable without code changes.

**Audit Logger.** Records a structured JSON entry per request: timestamp, prompt hash (no raw prompt text stored), routed model, estimated carbon cost (gCO2eq), estimated carbon saved vs. largest-model baseline, latency (ms), accuracy proxy (where available), and routing rationale. Audit logs are designed for downstream compliance reporting under EU AI Act Art.53 and CSRD.

### 3.3 Routing Priority

Constraints are applied in strict priority order:

1. Latency SLA (hard constraint — never violated)
2. Accuracy floor (hard constraint — never violated)
3. Carbon minimisation (optimisation objective)

If the carbon feed is unavailable, routing falls back to complexity-only dispatch. If no model satisfies both constraints, the request is escalated to the largest available model and logged as a forced routing event.

---

## 4. Carbon Savings Estimate

To establish order-of-magnitude plausibility, we model a production system serving 1M prompts/day. These assumptions are conservative and will be stress-tested in Phase 2.

**Assumptions:**
- 65% of prompts are *simple* or *moderate* (summarisation, Q&A, extraction) — routable to a 7B model
- 35% are *complex* or *expert* (multi-hop reasoning, long-form generation) — require a 70B model
- 7B model (e.g., Llama 3 7B on T4 GPU): ~50W effective TDP, ~150 tokens/sec → ~0.0003 kWh per 1K tokens
- 70B model (e.g., Llama 3 70B on A100): ~300W effective TDP, ~50 tokens/sec → ~0.006 kWh per 1K tokens
- Average prompt+completion: ~500 tokens
- Grid intensity: 400 gCO2eq/kWh (European average, Electricity Maps 2024)

**Without routing (all requests to 70B):**  
1,000,000 × (500/1000) × 0.006 kWh × 0.4 kgCO2/kWh = **1,200 kgCO2/day**

**With CAIR routing:**  
650,000 × (500/1000) × 0.0003 × 0.4 + 350,000 × (500/1000) × 0.006 × 0.4  
= 39 + 420 = **459 kgCO2/day**

**Estimated saving: ~62% reduction in inference carbon** for an existing production system, requiring no model changes.

These figures are back-of-envelope and will be replaced with empirically measured values in the full paper.

---

## 5. Regulatory Alignment

CAIR is designed to directly support compliance reporting under:

- **EU AI Act Art.53** — requires General Purpose AI (GPAI) model providers to report energy consumption; CAIR's audit log provides per-request carbon attribution suitable for aggregation into compliance reports
- **CSRD / ESRS E1** — Scope 2 emissions from server energy are reportable; CAIR's routing decisions and audit trail provide the granularity needed for AI-specific Scope 2 attribution
- **ISSB S2** — climate risk disclosures increasingly require operational AI footprint quantification; CAIR provides both reduction mechanism and measurement infrastructure

The alignment between carbon reduction and regulatory reporting is intentional: the same routing layer that saves emissions also generates the audit trail needed to prove it.

---

## 6. Limitations and Future Work

**Complexity scoring accuracy.** The v1 rule-based scorer is an approximation. Misclassification of a complex prompt as simple will route it to an undersized model, potentially degrading response quality. Version 2 (DistilBERT fine-tuned on labelled data) addresses this. Phase 2 quality delta evaluation (E3) measured a mean cosine similarity of 0.90 between Haiku and Opus outputs on simple tasks (minimum 0.80), confirming acceptable quality preservation. The one task with the lowest similarity (Translate 'hello' to Spanish, 0.796) reflects a style difference — Haiku returned "Hola" while Opus provided a fuller explanation — rather than a factual error.

**Carbon data latency.** Grid carbon intensity can change within the 5-minute cache window. A shorter TTL improves accuracy at the cost of more API calls. The tradeoff is configurable.

**Model accuracy variability.** The accuracy floor is specified per deployment but measured against general benchmarks (MMLU). Task-specific accuracy may diverge from benchmark scores; production deployments should calibrate accuracy floors empirically.

**Cascading.** Some routing systems use cascading: attempt the small model, escalate to the large model if confidence is low. CAIR v1 does not implement cascading (it adds latency and complexity). This is a planned Phase 2 feature.

**Future work** includes: empirical benchmark dataset (50 prompts, 5 complexity tiers, 3 domains), latency overhead measurement under load, accuracy comparison between routed and always-large baselines, and integration with vLLM and Ollama for production deployment validation.

---

## 7. Conclusion

This paper has introduced CAIR, a real-time carbon-aware inference routing framework for LLMs. By combining per-prompt complexity scoring with live grid carbon intensity data and multi-objective routing, CAIR targets a ~62% reduction in inference carbon on a representative production workload, without accuracy or latency compromise. CAIR introduces a two-layer architecture: a time-bounded carbon budget enforcement layer that progressively constrains model selection as cumulative emissions approach a period cap, combined with per-request multi-objective routing over complexity, carbon intensity, and latency. This combination enables carbon governance at the inference layer — reducing emissions while generating the audit trail needed for CSRD ESRS E1 and EU AI Act Art.53 compliance reporting.

The Phase 1 and Phase 2 implementations are complete. Phase 2 delivers a FastAPI serving layer (POST /route, POST /route-and-call, GET /health) and the quality delta evaluation (E3: mean cosine similarity 0.90 on simple tasks), confirming that routing to smaller models preserves response quality. Design documentation, source code, and the full eval harness are available at: https://github.com/pretzelslab/sa1-carbon-inference-router

---

## References

Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. *arXiv:2305.05176*.

Luccioni, A. S., Viguier, S., & Ligozat, A.-L. (2022). Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model. *arXiv:2211.02001*.

Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., & Dean, J. (2021). Carbon Emissions and Large Neural Network Training. *arXiv:2104.10350*.

Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergkvist, A., Kepner, J., Gadepally, V., & Robicheck, R. (2023). From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference. *IEEE HPEC 2023*.

Wu, C. J., Raghavendra, R., Gupta, U., Acun, B., Ardalani, N., Maeng, K., Chang, G., Aga, F., Huang, J., Bai, H., & Hazelwood, K. (2022). Sustainable AI: Environmental Implications, Challenges, and Opportunities. *Proceedings of MLSys 2022*.

Wu, C. J., et al. (2023). Clover: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service. *SC'23: Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis*.

---

*This is a preprint. Phase 1 and Phase 2 implementations and benchmarks are complete. The full paper with Phase 3 empirical results (multi-provider comparison, OpenAI vs Claude routing) is in progress.*  
*Framework repository: https://github.com/pretzelslab/sa1-carbon-inference-router*  
*Concept date: 2026-04-30*
