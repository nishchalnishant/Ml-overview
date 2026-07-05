---
module: Llms
topic: Interview Notes
subtopic: Llmops And Production Ai Snappy
status: unread
tags: [llms, ml, interview-notes-llmops-and-pro]
---

> _Quick-recall companion. For the full deep-dive, see [llmops-and-production-ai.md](08-llmops-and-production-ai.md)._

# LLMOps & production AI — ship it like a service
LLMs in prod are **systems**, not demos: routing, budgets, safety, and observability.
**One-line:** LLMOps = DevOps + evals + safety + prompts + retrieval/tool orchestration.
---
# Q1: Explain the AI product lifecycle from ideation to production.
- **Direct answer:** Define value+risks → build prototype → evals → ship behind flags → monitor → iterate.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q2: What is LLMOps, and how does it differ from traditional MLOps?
- **Direct answer:** LLMOps adds prompts, context, safety, tool calls, and non-determinism on top of classic MLOps.

---
# Q3: How do you serve LLMs in production?
- **Direct answer:** Managed API or self-hosted inference behind an API gateway with caching, batching, and guardrails.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q4: What is model quantization?
- **Direct answer:** Lower precision weights (FP16→INT8/INT4) to cut memory/cost with some quality trade-off.
- **Mini pop quiz:** If costs spike, what do you inspect first? → token usage + retries + context size.

---
# Q5: How do you monitor LLM applications in production?
- **Direct answer:** Track latency/cost + quality (hallucinations, refusals) + safety + retrieval metrics.

---
# Q6: What is LLM observability?
- **Direct answer:** Tracing every step: prompt build, retrieval, tool calls, model response, validators, user outcome.

---
# Q7: What are guardrails for LLMs, and how do you implement them?
- **Direct answer:** Policies + allow-lists + validators + refusal rules + human approvals for risky actions.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q8: How do you implement content filtering for AI outputs?
- **Direct answer:** Pre/post filters + safety models + blocklists + policy rules + logging.

---
# Q9: How do you estimate the cost of an AI feature?
- **Direct answer:** Tokens in/out × price + retrieval + tool costs + retries + infra; model it like a unit cost.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q10: How do you optimize inference costs?
- **Direct answer:** Route to cheaper models, cache, compress context, reduce retries, quantize/self-host if needed.
- **Mini pop quiz:** If costs spike, what do you inspect first? → token usage + retries + context size.

---
# Q11: How do you implement A/B testing for LLM systems?
- **Direct answer:** Randomized routing with guardrails; measure task success and user metrics, not vibes.

---
# Q12: CI/CD for AI apps vs traditional?
- **Direct answer:** Adds eval suites, prompt versioning, safety tests, dataset/version provenance.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q13: How do you version prompts in prod?
- **Direct answer:** Treat prompts as code: repo, reviews, semantic versions, changelogs, rollback.

---
# Q14: Model versioning & rollbacks?
- **Direct answer:** Registry + canary + metrics gates; quick rollback on regression.

---
# Q15: Rate limiting/throttling for LLM APIs?
- **Direct answer:** Token-based quotas, per-tenant limits, backpressure queues, graceful errors.

---
# Q16: Updates/migrations without downtime?
- **Direct answer:** Blue/green, canary, dual-run, shadow traffic, staged rollouts.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q17: Role of feature flags?
- **Direct answer:** Ship safely: enable per-tenant, ramp slowly, instant disable.

---
# Q18: Logging and tracing for LLM apps?
- **Direct answer:** Correlate request IDs across retrieval/tools/model; redact PII; sample wisely.

---
# Q19: Handle PII and sensitive data?
- **Direct answer:** Minimize collection, redact, encrypt, RBAC, retention policies, on-prem where needed.

---
# Q20: Gateway pattern for LLM API management?
- **Direct answer:** Central layer for auth, routing, rate limits, caching, safety, provider abstraction.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q21: Streaming responses?
- **Direct answer:** Server-sent events/websockets; monitor TTFT and inter-token latency.

---
# Q22: Key SLAs/metrics?
- **Direct answer:** TTFT, tokens/sec, p95 latency, error rate, cost per task, retrieval recall/precision.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q23: Cloud vs on-device deployment?
- **Direct answer:** Cloud for capability; on-device for privacy/latency/offline; often hybrid.

---
# Q24: Fallback strategies when primary model is unavailable?
- **Direct answer:** Route to backup model/provider, degrade features, cached answers, async mode.

---
# Q25: Reliable structured output?
- **Direct answer:** Schema-first + constrained decoding + validation+repair loops with strict budgets.

---
# Q26: Handle long contexts efficiently?
- **Direct answer:** RAG + summarization + prefix caching + chunk re-ranking; avoid context bloat.
- **Mini pop quiz:** If costs spike, what do you inspect first? → token usage + retries + context size.

---
# Q27: Semantic routing?
- **Direct answer:** Classifier/router chooses model/tool chain based on intent, complexity, and cost.

---
# Q28: Manage secrets and API keys securely?
- **Direct answer:** Key Vault/secret stores, short-lived tokens, RBAC, rotation, no secrets in prompts.

---
# Q29: Latency spikes at peak hours?
- **Direct answer:** Queueing + continuous batching + autoscaling + caching + reduce context; watch KV cache.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q30: Costs too high — reduce without killing quality?
- **Direct answer:** Route, cache, compress context, limit retries, smaller models, quantize/self-host.

---
# Q31: Hitting provider rate limits?
- **Direct answer:** Backpressure queues, token budgets, adaptive routing, request shedding.

---
# Q32: Switch providers without downtime?
- **Direct answer:** Gateway abstraction + dual-run + contract tests + gradual cutover.

---
# Q33: Scale from 100 rps to 5000?
- **Direct answer:** Horizontal scale, batching, vLLM, sharding, queueing, model routing, caching.

---
# Q34: Handle peak traffic spikes?
- **Direct answer:** Autoscale + queue + graceful degradation + cached responses + priority tiers.

---
# Q35: Eliminate single points of failure?
- **Direct answer:** Multi-region, multi-provider, fallback models, circuit breakers.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.

---
# Q36: Multi-LLM pipeline breaks — orchestration failure?
- **Direct answer:** Step isolation, retries, compensation, timeouts, partial results.

---
# Q37: Zero visibility into failing step — add observability?
- **Direct answer:** Distributed tracing + structured logs + per-step metrics + replayable traces.

---
# Q38: Quantization dropped accuracy — minimize loss?
- **Direct answer:** Use AWQ/GPTQ, mixed precision, calibrate on domain data, evaluate, rollback if needed.

---
# Q39: Design graceful degradation?
- **Direct answer:** Tiered features: best effort → safe fallback → ‘insufficient data’; never fail closed on UX.
- **Azure/DevOps bridge:** think *pipelines + gates + dashboards + rollbacks*.
