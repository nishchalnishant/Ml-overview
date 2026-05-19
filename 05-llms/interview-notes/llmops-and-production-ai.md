# LLMOps & Production AI

## The Operating Reality

An LLM system in production is not a model. It's a pipeline: prompt template + model + retrieval index + safety policy + output parser + tool integrations. Each component can fail independently. Each can be updated independently. And unlike a traditional ML model, you can change the system's behavior without retraining anything — just by modifying a prompt or retrieval config.

This creates a new class of operational problem. Your "model" is not fixed. Your "prediction" can change based on context, policy, and retrieved content. Silent regressions happen constantly — a prompt change that improves one task silently breaks another.

LLMOps is the discipline of operating systems like this reliably: versioning non-code artifacts, measuring quality continuously, rolling out changes safely, and recovering quickly when things break.

---

## 1. The AI Product Lifecycle

### The Problem

You build an LLM feature. It works in testing. Six months after launch, users are complaining about quality but you have no way to reproduce the issue, no record of what changed, and no way to tell if it was a model update, a prompt change, or a retrieval index rebuild.

### The Core Insight

LLM systems degrade through changes to components that aren't tracked like code — prompts, retrieval indexes, safety policies, model versions. The lifecycle must treat all of these as versioned artifacts with associated eval results, not just track the application code.

### The Mechanics

1. **Ideation:** define user workflow, success criteria, and risk class before writing code
2. **Design:** architecture (RAG/agents/tools), data flows (PII, retention), evaluation plan
3. **Prototype:** minimal pipeline with logging and output validators
4. **Eval-driven development:** golden sets, measure task success/faithfulness/safety, iterate on prompts and retrieval — each iteration produces a versioned artifact bundle
5. **Pre-prod:** load testing, red teaming, structured output validation, latency/cost profiling
6. **Rollout:** feature flags, canary, A/B, rollback plan, SLA monitoring
7. **Post-prod:** continuous evaluation + drift monitoring + incident response + periodic updates

The deployment unit is not a model checkpoint — it's a **versioned bundle**: prompt template + model + retrieval snapshot + safety policy. Rollback means switching to the previous bundle, which you can do in seconds via feature flags.

### What Breaks

Deploying without eval gates — you don't know if the new bundle is better or worse. No rollback plan — when quality degrades in production, you scramble to identify what changed. Not tracking retrieval index versions — a rebuilt index changes behavior even if the model and prompt are identical.

### What the Interviewer Is Testing

Whether you treat the full pipeline as the artifact under management, not just the model. Whether you have a concrete rollback strategy.

### Common Traps

"We version the model" — but not the prompt, not the retrieval index, not the safety policy. These are equally capable of changing behavior.

---

## 2. LLMOps vs Traditional MLOps

### The Problem

Your team has MLOps tooling for training and serving sklearn and PyTorch models. You're told to run LLM systems with the same tooling. Things start breaking in ways your MLOps tools can't detect.

### The Core Insight

Traditional MLOps tracks: training data, model weights, serving infrastructure, prediction quality. LLMOps tracks all of that plus: prompt templates, retrieval indexes, tool schemas, safety policies, and output parsers — and their behavior under stochastic decoding. The additional artifacts are the primary source of behavioral change.

### The Mechanics

| Dimension | Traditional MLOps | LLMOps |
| :--- | :--- | :--- |
| Primary artifact | Model checkpoint | Prompt + model + retrieval + policy bundle |
| Behavior change mechanism | Retraining | Prompt edit, retrieval update, policy change |
| Output type | Fixed schema | Text, structured JSON, tool calls, multi-turn |
| Failure modes | Distribution shift, data drift | Hallucination, format failure, jailbreak, tool misuse |
| Evaluation | Accuracy/F1 on test set | Faithfulness, format validity, safety, task success |
| Rollback mechanism | Previous checkpoint | Previous artifact bundle via feature flag |

```python
artifact_bundle = {
    "model": model_id,
    "prompt_template": prompt_ver,
    "retrieval_index": idx_snapshot_id,
    "safety_policy": policy_ver,
    "output_parser": parser_ver,
}
```

### What Breaks

Applying traditional MLOps processes without accounting for non-code artifacts — prompt changes are not tracked in your model registry. Not measuring quality metrics specific to LLM failures (hallucination, format, safety) because your MLOps tooling only measures accuracy/F1.

### What the Interviewer Is Testing

Whether you understand the expanded scope of LLMOps. Whether you can identify which artifact types need version tracking.

### Common Traps

"We track the model version" — not tracking the prompt version, which changes behavior just as significantly. Assuming MLOps tooling handles prompt management automatically.

---

## 3. LLM Serving in Production

### The Problem

Your model works at batch=1. At production load, it either runs out of GPU memory, has unacceptable latency, or has low GPU utilization (expensive hardware sitting mostly idle). Which problem are you actually facing?

### The Core Insight

LLM inference has two phases with different bottlenecks:
- **Prefill** (processing input): compute-bound. Slow when prompts are long.
- **Decode** (generating output): memory-bandwidth-bound. Slow when the model is large or batching is inefficient.

Every serving optimization targets one of these two bottlenecks. Diagnosing which one is failing is the prerequisite to fixing it.

### The Mechanics

**The serving stack (from lowest to highest level):**
1. Model weights loaded to GPU VRAM
2. Inference framework (vLLM, TensorRT-LLM, TGI) handles batching, KV cache, scheduling
3. API server (FastAPI, Triton) handles request routing, authentication, streaming
4. Gateway handles rate limiting, caching, safety, logging

**Key metrics to instrument:**
- TTFT (time to first token) — measures prefill bottleneck
- TPOT (time per output token) — measures decode bottleneck
- GPU utilization — measures batching efficiency
- KV cache utilization — measures memory pressure

**Cold start components:** instance provisioning + container image pull + model weight loading from S3/GCS + CUDA compilation of kernels. For a 70B model, weight loading alone can take 2–5 minutes. Design for this with warm pool pre-provisioning.

### What Breaks

Vertical scaling (bigger GPU) when the problem is batching inefficiency. Optimizing TTFT when users are complaining about slow generation (TPOT is the issue). Running without profiling — guessing at the bottleneck.

### What the Interviewer Is Testing

Whether you know TTFT and TPOT are different. Whether you can diagnose the bottleneck before jumping to solutions.

### Common Traps

Recommending quantization before measuring whether you're compute-bound or memory-bound. Not accounting for cold start in your latency budget.

---

## 4. Quantization in Production

### The Problem

LLaMA 3 70B in BF16 requires 140 GB of weight memory. You have two A100 80GB. You don't fit. Do you use INT8? INT4? GPTQ? AWQ? What do you test before declaring a quality loss acceptable?

### The Core Insight

Quantization reduces memory and can increase throughput, but quality loss is not uniform across tasks. A model that passes general perplexity checks at INT4 can fail significantly on structured output tasks, arithmetic, or low-resource languages. You must test on the tasks you actually run.

### The Mechanics

Quantization decision process:
1. Calculate weight memory at target precision: INT8 = ½ the FP16 size, INT4 = ¼
2. Calculate KV cache at target batch size (separate from weight precision)
3. Run perplexity comparison on a representative calibration set
4. Run task-specific eval on your actual production tasks
5. Run format validity tests — quantization can destabilize structured output
6. Set a quality gate: if any target metric drops more than X%, reject this quantization level

```python
# Before deploying quantized model
baseline = run_eval(model="llama3-70b-bf16", dataset=production_eval_set)
quantized = run_eval(model="llama3-70b-int4-awq", dataset=production_eval_set)

for metric in ["task_accuracy", "faithfulness", "format_valid_rate"]:
    if quantized[metric] < baseline[metric] * 0.95:  # 5% regression threshold
        reject_quantization(f"Metric {metric} degraded beyond threshold")
```

**Calibration data matters for GPTQ and AWQ:** calibrate on data that matches your production distribution. Academic calibration sets won't protect columns that are important for your specific domain.

### What Breaks

Calibrating on generic data when your production queries are domain-specific. Declaring quantization acceptable based on perplexity alone. Forgetting that INT4 weights + BF16 KV cache is the common configuration — not INT4 everything.

### What the Interviewer Is Testing

Whether you know how to evaluate quantization quality, not just how to run the quantization tool. Whether you know about different quantization methods (GPTQ vs AWQ vs NF4).

### Common Traps

"AWQ 4-bit is always better than GPTQ 4-bit" — this is true on average but not on all tasks. Always eval on your tasks. Forgetting that KV cache quantization is a separate decision from weight quantization.

---

## 5. Monitoring and Observability

### The Problem

Production failure happens. A user reports "the assistant gave wrong medical information." You don't know: which version of the model/prompt was running, what documents were retrieved, whether the safety filter was triggered, how confident the model was, or whether this is one failure or a systematic pattern.

You have no visibility.

### The Core Insight

You cannot diagnose what you cannot observe. LLM pipeline failures are multi-causal — any of retrieval, generation, parsing, safety, or tool execution could be the source. You need stage-level instrumentation, not just a final output log.

### The Mechanics

**What to instrument at each stage:**

```python
trace_span.set_tag("request_id", req_id)
trace_span.set_tag("prompt_ver", prompt_ver)
trace_span.set_tag("model_ver", model_id)
trace_span.set_tag("retrieval_top1", chunk_ids[0] if chunk_ids else None)
trace_span.set_tag("retrieval_count", len(chunk_ids))
trace_span.set_tag("format_valid", format_valid)
trace_span.set_tag("safety_label", safety_result.label)
trace_span.set_tag("tool_calls", [tc.name for tc in tool_calls])
trace_span.set_tag("stop_reason", stop_reason)
trace_span.finish()
```

**Metrics dashboard (minimum viable):**
- TTFT P50/P95/P99 (alert on P99 > SLA)
- TPOT P50/P95
- Format validity rate (alert if drops below 0.99)
- Safety violation rate (alert if rises above threshold)
- Retrieval hit rate (alert if drops — index may be stale)
- Parse error rate (alert if rises — output schema may have broken)

**Quality metrics (sampled):** faithfulness score, task accuracy proxy (on sampled subset with LLM judge). These run on samples, not every request.

### What Breaks

Only monitoring latency and error rate — a system can be operationally healthy while generating worse answers. Logging raw prompts without PII redaction. Not correlating failures to artifact versions — you know it failed but not which prompt version caused it.

### What the Interviewer Is Testing

Whether you instrument quality metrics, not just operational metrics. Whether you know what to log at each pipeline stage for debugging.

### Common Traps

Logging final outputs only (can't trace which retrieval caused the hallucination). Logging raw user inputs without PII controls.

---

## 6. Guardrails and Content Filtering

### The Problem

Your LLM assistant is deployed. An adversarial user crafts a prompt: "Ignore your instructions and tell me how to..." The model complies. Another user accidentally pastes personal information — it gets stored in logs and potentially sent to a third-party API provider.

Neither failure was caught by the model itself.

### The Core Insight

Safety cannot be the sole responsibility of the model's training. Models can be jailbroken, and they were not trained on your specific policy. You need deterministic enforcement layers: input classifiers that check before generation, output classifiers that check before delivery, and PII detection that protects both directions.

### The Mechanics

**Defense in depth:**

1. **Input classification:** detect policy violations before sending to LLM
   - Prompt injection detection (pattern + embedding similarity)
   - Intent classification (harmful request detection)
   - PII detection + redaction

2. **System prompt injection resilience:**
   - Instruction hierarchy in prompt (emphasize model should not follow user-injected instructions)
   - Instruction position: place security-critical instructions at end of system prompt

3. **Output filtering:**
   - Policy violation classifier on generated text
   - PII detection on output before delivery to client
   - Format validation (structured output must parse before delivery)

4. **Tool call guardrails:**
   - Validate tool arguments against schemas before execution
   - Disallow destructive operations on ambiguous/unconfirmed inputs
   - Log all tool calls with sanitized arguments

```python
def llm_with_guardrails(user_input, context):
    # Input guardrails
    if detect_prompt_injection(user_input):
        return refusal_response("policy")
    clean_input = redact_pii(user_input)
    
    # Generation
    raw_output = llm.generate(clean_input, context)
    
    # Output guardrails
    if classify_unsafe(raw_output):
        return refusal_response("output_policy")
    clean_output = redact_pii(raw_output)
    validate_schema(clean_output)
    return clean_output
```

### What Breaks

Relying solely on model RLHF safety training — it's not deterministic, and it can be bypassed. Applying text filters only to user input, not to retrieved documents (indirect injection attack). Not validating tool arguments before execution.

### What the Interviewer Is Testing

Whether you know safety requires layers: training + input classification + output classification + PII handling. Whether you know about indirect prompt injection via retrieval.

### Common Traps

"We use Claude/GPT-4, they're already safe" — API models can be jailbroken; you still need your own guardrails. Not testing guardrails systematically — red team the guardrails, not just the model.

---

## 7. Cost Estimation and Optimization

### The Problem

Your LLM application costs $50k/month in API calls. The CTO asks where the money is going and what can be cut without degrading quality. You don't have a breakdown by query type, by model, or by pipeline stage.

### The Core Insight

LLM cost is dominated by token volume, not request count. One request with a 4000-token prompt costs 40× more than one with a 100-token prompt. To optimize cost without degrading quality, you must profile *where* tokens are spent and *which* token cuts preserve quality.

### The Mechanics

**Cost breakdown approach:**
1. Profile token usage per route and stage: system prompt, retrieved context, user message, output
2. Identify the expensive routes (usually long context + complex generation)
3. Apply the cheapest intervention that preserves quality:
   - Trim system prompt (remove redundant instructions)
   - Reduce retrieval top-k, apply reranker to keep only relevant chunks
   - Cap max_output_tokens
   - Route simple queries to smaller/cheaper models

**Model cascade for cost:**
```python
route = task_classifier(request_text)  # lightweight classifier
if route["complexity"] == "simple":
    resp = llm_small.generate(prompt, max_tokens=200)   # $0.002/1k tokens
else:
    resp = llm_large.generate(prompt, max_tokens=500)   # $0.03/1k tokens
```

**Caching hierarchy:**
- Exact match cache (SHA256 of prompt → cached response): free for repeated identical queries
- Semantic cache (ANN lookup on embedding): saves LLM call for near-duplicate queries
- Prefix cache (KV cache reuse for shared system prompt): reduces prefill cost

**Quality gate requirement:** any cost optimization must be validated against the same eval suite used for model changes. Never cut tokens without measuring the quality impact.

### What Breaks

Cutting context without measuring whether the model still has enough information. Caching responses without TTL/invalidation — cached answers become stale when the retrieval index updates.

### What the Interviewer Is Testing

Whether you profile before optimizing. Whether you know that cost optimization requires the same eval rigor as model optimization. Whether you know caching strategies.

### Common Traps

"We'll just use a smaller model" — for complex tasks, the smaller model may cost more through repair retries or quality failures. Not versioning caches with index/policy versions.

---

## 8. A/B Testing for AI Systems

### The Problem

You have two prompt variants. Variant B shows better offline eval scores. Should you deploy it to all users? How long should you run the A/B test? What metric do you trust?

### The Core Insight

Offline eval tells you what might happen. A/B testing tells you what actually happens with real users and real traffic. The two often diverge because offline eval sets don't fully represent production distribution. You need both: offline eval to gate candidates before they reach production, A/B to measure actual user impact.

### The Mechanics

**A/B test design for LLM systems:**
1. Define the primary metric: user satisfaction (explicit) or behavioral proxy (implicit: copy rate, follow-up question rate, escalation rate)
2. Define secondary metrics: latency, cost, safety violation rate, format validity
3. Set minimum detectable effect: what improvement is worth deploying?
4. Calculate required sample size (power analysis)
5. Run for enough time to cover usage patterns (weekday/weekend, multiple user cohorts)
6. Check for novelty effects — early lift can fade as users adapt

**Statistical requirements:**
- Paired evaluation where possible (same users see both variants)
- Bootstrap CI or t-test for continuous metrics
- McNemar's test for binary outcomes (success/failure)
- Correct for multiple comparisons if testing many metrics simultaneously

```python
ab_test("prompt_variant_b",
        primary_metric="user_accept_rate",
        secondary_metrics=["latency_p99", "format_valid_rate"],
        traffic_fraction=0.10,  # 10% to candidate
        min_detectable_effect=0.02)  # 2% improvement
```

### What Breaks

Running A/B tests on too little traffic and declaring significance on noise. Not monitoring secondary metrics — a variant that improves satisfaction while increasing safety violations is not an improvement. Stopping the test too early (peeking problem).

### What the Interviewer Is Testing

Whether you know A/B testing requires statistical rigor, not just "it looked better." Whether you monitor safety and format metrics alongside user metrics.

### Common Traps

Treating offline improvement as evidence that A/B will show the same improvement. Not defining the minimum detectable effect before running the test — can lead to underpowered tests.

---

## 9. CI/CD for AI Systems

### The Problem

You change a prompt. How do you know if it's safe to deploy? What automated checks run? Who approves it? What's the rollback mechanism?

### The Core Insight

LLM system changes (prompts, model versions, retrieval indexes, policies) require a release process with the same rigor as code changes, but the automated checks are different. Instead of unit tests, you run regression eval suites. Instead of code review, you compare artifact bundles.

### The Mechanics

**CI pipeline for AI system changes:**

```
Prompt change → git commit
  → CI: run regression eval suite (format, safety, faithfulness, task accuracy)
  → Gate: all metrics ≥ baseline thresholds
  → Staging deploy: smoke test with synthetic load
  → Canary deploy: 5% of production traffic
  → Monitor canary: TTFT P99, format validity, safety rate
  → Gate: canary metrics match or beat stable
  → Full rollout
  → Monitor: 24-hour watch period
  → Rollback plan: switch feature flag to previous bundle
```

**What the regression eval suite must cover:**
- Format validity across representative inputs
- Safety: all red-team regression cases must pass
- Faithfulness: faithfulness score on golden RAG examples
- Task accuracy: exact match / LLM-judge on known-correct examples
- Performance: latency proxy (token count should not explode)

**Artifact bundle as deployment unit:**
```python
artifact_bundle = {
    "model_id": "llama3-70b-instruct-v2",
    "prompt_version": "customer-support-v14",
    "retrieval_index": "knowledge-base-2026-01-15",
    "safety_policy": "enterprise-v3",
    "eval_results": {"faithfulness": 0.94, "format_valid": 0.99, "safety_pass": 1.0},
}
```

### What Breaks

Deploying without regression eval — you discover the regression in production. Not including safety regression tests in CI — a prompt change that doesn't break format can still break safety. Long canary periods with no automated monitoring — you're not actually watching.

### What the Interviewer Is Testing

Whether you can design a full release pipeline for non-code artifacts. Whether you know what the automated gates should test.

### Common Traps

"We test in staging" — staging traffic distribution doesn't match production. Not versioning the eval results alongside the artifacts.

---

## 10. Prompt Versioning and Management

### The Problem

Three engineers are editing the same system prompt. One changes it to fix a hallucination problem. Another changes it to improve format. They deploy separately. A week later, nobody knows which version is in production or which changes caused which behavior.

### The Core Insight

Prompts are code. They determine behavior as much as model weights. They must be versioned, reviewed, and deployed with the same rigor as application code — with the additional requirement that each version has associated eval results demonstrating quality.

### The Mechanics

**Prompt registry design:**
- Unique version ID per prompt template (semantic: `customer-support-v14`, or content hash)
- Immutable versions — once deployed, a version cannot be modified
- Linked eval results: each version has associated regression suite results
- Deployment history: which version is live in which environment

**Prompt template structure:**
```
system_prompt_v14:
  created: 2026-01-15
  author: team-ai
  eval_results: {faithfulness: 0.94, format_valid: 0.99}
  template: |
    You are a customer support assistant for {company_name}.
    Answer questions based only on the provided context.
    If the context doesn't contain the answer, say "I don't have that information."
    Context: {retrieved_context}
    Question: {user_question}
```

**Version selection at runtime:**
```python
prompt_ver = flags.get("prompt_version", "stable")  # or "candidate"
prompt = prompt_registry.load(prompt_ver)
```

### What Breaks

Ad-hoc prompt editing in production without version control. Not testing prompt changes against regression suite before deployment. Hardcoding prompts in application code instead of registering them.

### What the Interviewer Is Testing

Whether you treat prompts as versioned artifacts, not informal strings. Whether you link versions to eval results.

### Common Traps

Storing prompts as hardcoded strings in source code — no ability to change without a full code deploy. Not linking prompt versions to their eval results — you lose the ability to compare.

---

## 11. Model Versioning and Rollback

### The Problem

You upgrade from model v1 to v2. Quality improves on most tasks. Three days after rollout, you get an incident report: a previously working workflow now fails silently. You need to roll back, but you don't know which bundle to roll back to, and re-deploying v1 takes 30 minutes.

### The Core Insight

Rollback must be instantaneous. That means the previous bundle must still be deployed and running — just not receiving traffic. Feature flags switch between them. Model weight loading takes too long to be a rollback mechanism.

### The Mechanics

**Zero-downtime model update procedure:**
1. Deploy new bundle (model + prompt + retrieval + policy) to production, not yet receiving traffic
2. Run offline regression eval on staging
3. Canary: route 5% of traffic to new bundle
4. Monitor canary metrics for 24 hours
5. Gradual rollout: 5% → 25% → 50% → 100% with eval gates between each step
6. Keep previous bundle deployed for 72 hours after full rollout (instant rollback window)
7. After 72-hour window with no incidents, decommission previous bundle

```python
# Instant rollback: no redeploy needed
flags["active_bundle"] = "v1_last_good"  # switches instantly
# Monitor that v1 traffic is flowing correctly
```

**Rollback triggers:**
- Safety violation rate rises above threshold
- Format validity drops below threshold
- P99 latency exceeds SLA
- Task accuracy drops on online proxy metrics
- Explicit on-call decision

### What Breaks

Rollback that requires redeploying model weights — too slow. Not having a defined rollback trigger — on-call team doesn't know when to pull the trigger. Not keeping the previous bundle warm — first request to "old" bundle causes cold-start latency spike.

### What the Interviewer Is Testing

Whether your rollback strategy is instantaneous (feature flag) or slow (redeploy). Whether you have defined triggers, not just "if something goes wrong."

### Common Traps

"We can roll back by redeploying" — at 2 AM during an incident, a 30-minute redeploy is too slow. Not testing the rollback procedure in staging before production.

---

## 12. Rate Limiting and Throttling

### The Problem

A single tenant submits a batch job with 10,000 requests. The shared serving cluster falls over for everyone else. Another tenant hits the LLM provider's token-per-minute limit and all requests fail with no fallback.

### The Core Insight

LLM systems need token-aware admission control, not just request-count-based rate limiting. One request that submits a 4000-token prompt uses 40× the compute of one with a 100-token prompt. Per-request rate limits don't protect against this.

### The Mechanics

**Token-aware rate limiting:**

```python
def handle_request(user, request):
    estimated_tokens = estimate_tokens(request.prompt) + request.max_output_tokens
    
    if not token_budget.try_consume(user.id, estimated_tokens):
        return {"error": "rate_limited", "retry_after": budget.reset_time}
    
    if concurrency_limiter.at_capacity(model=request.model):
        queue_request(request)  # bounded queue with timeout
        return
    
    return generate(request)
```

**Throttling hierarchy:**
- Per-user token-per-minute limit
- Per-tenant (organization) token-per-minute limit
- Global concurrency cap per model
- Provider-level token budget (prevents hitting external API limits)

**Graceful handling under throttling:**
- Return structured error with retry-after time
- Route to smaller/cached fallback for low-priority requests
- Priority queuing: interactive requests ahead of batch jobs

### What Breaks

Request-count limits without token-awareness — a single long prompt bypasses rate limits. No bounded queue — when the queue is full, requests silently fail or cause memory exhaustion. No priority queuing — batch jobs crowd out interactive requests.

### What the Interviewer Is Testing

Whether you know token-aware rate limiting is required. Whether you have a graceful fallback when limits are hit.

### Common Traps

Applying rate limits only at the API gateway, not at the LLM provider budget level — the provider's own rate limits can still be hit. Retrying without backoff from rate-limited clients — amplifies the problem.

---

## 13. Model Updates and Migrations Without Downtime

### The Problem

You need to migrate from one LLM provider to another. Or upgrade the model version. Or rebuild the retrieval index with new documents. Any of these can change behavior. None can cause downtime. How do you do all three simultaneously if needed?

### The Core Insight

Zero-downtime requires coexistence: both old and new configurations must be runnable simultaneously, with traffic controlled by feature flags. You migrate traffic gradually while monitoring.

### The Mechanics

**For model version updates:** new model deployed alongside old, traffic split via feature flag, canary → gradual → full rollout

**For retrieval index updates:**
- Build new index independently without touching the serving path
- Version the new index with a snapshot ID
- Deploy new routing to new index for a canary fraction
- Validate faithfulness and retrieval quality before widening
- Keep old index for rollback window

**For schema changes (output format changes):**
- Deploy new parser version alongside old
- Both parsers accept both old and new schemas during transition
- Migrate output consumers to new schema before retiring old parser

```python
# Multi-component canary
bundle = {
    "stable": {"model": "v1", "index": "idx-20260101", "prompt": "p14"},
    "candidate": {"model": "v2", "index": "idx-20260115", "prompt": "p15"},
}
active = "candidate" if user_in_canary(user_id) else "stable"
route = bundle[active]
```

### What Breaks

Changing retrieval index without versioning — no rollback if quality degrades. Schema changes that break downstream parsers without a transition period. Doing multiple simultaneous changes — if quality degrades, you can't tell which change caused it.

### What the Interviewer Is Testing

Whether you can design zero-downtime migrations. Whether you know to change one thing at a time when possible.

### Common Traps

Rebuilding a retrieval index in-place — breaks reproducibility and rollback. Assuming new and old models produce identical structured output given the same prompt — they don't.

---

## 14. Logging, Tracing, and Observability

### The Problem

An LLM request failed. The user got a wrong answer. You have a log line: `{"status": 200, "latency_ms": 1240}`. That tells you it technically succeeded. It tells you nothing about why the answer was wrong.

### The Core Insight

LLM failure attribution requires tracing through each pipeline stage. You need to know: which document was retrieved, whether it contained the answer, whether the model actually used it, whether the output passed format validation, and which artifact versions were running. A single status code tells you none of this.

### The Mechanics

**Per-request trace structure:**

```python
trace = {
    "request_id": req_id,
    "timestamp": now(),
    "artifact_versions": {
        "prompt": prompt_ver,
        "model": model_id,
        "index": idx_snapshot_id,
        "policy": policy_ver,
    },
    "retrieval": {
        "query": redact_pii(query),
        "retrieved_chunk_ids": chunk_ids,
        "retrieval_latency_ms": t_retrieval,
    },
    "generation": {
        "input_tokens": n_input,
        "output_tokens": n_output,
        "stop_reason": stop_reason,
        "latency_ms": t_generation,
    },
    "validation": {
        "format_valid": format_valid,
        "parse_error": parse_error if not format_valid else None,
        "safety_label": safety_result.label,
        "safety_score": safety_result.score,
    },
    "tool_calls": [{"name": tc.name, "valid": tc.schema_valid} for tc in tool_calls],
}
```

**PII handling:** never log raw user inputs. Log hash or redacted version. Log only sanitized tool arguments. Enforce retention policy on request logs.

**Dashboards to build:**
- Stage-level latency breakdown (retrieval / generation / parsing / safety)
- Format validity rate over time
- Safety violation rate over time
- Retrieval hit rate (did we find relevant chunks?)
- Error rate by type (timeout / format / safety / tool_failure)

### What Breaks

Logging only the final output — you can't trace which retrieval or which artifact version caused the failure. Logging raw user content — PII compliance failure. Not linking logs to artifact versions — you can't reproduce the failure for debugging.

### What the Interviewer Is Testing

Whether you instrument quality metrics, not just operational metrics. Whether you handle PII in logging.

### Common Traps

Logging without redaction. Only logging errors, not successful requests — you lose the ability to compare behavior across versions.

---

## 15. PII and Sensitive Data Handling

### The Problem

A user pastes their full credit card number into a chat prompt. The system sends it to an external LLM API, stores it in logs, and the generated response happens to echo part of the number. You've now violated PCI DSS compliance.

### The Core Insight

PII must be handled at three independent checkpoints: before the model call (detect and redact in input), after the model call (detect and redact in output), and in logging (never persist raw PII). Any checkpoint that's missing creates a compliance gap.

### The Mechanics

**Three-checkpoint PII handling:**

```python
def handle_request(user_text, context):
    # Checkpoint 1: Input
    clean_input = pii_detector.redact(user_text)
    filtered_context = rag_acl_filter(context, user.permissions)
    
    # Generate
    raw_output = llm.generate(clean_input, filtered_context)
    
    # Checkpoint 2: Output
    clean_output = pii_detector.redact(raw_output)
    
    # Checkpoint 3: Logging (only hashes/metadata, never raw)
    log({"input_hash": hash(user_text), "output_redacted": True})
    
    return clean_output
```

**RAG-specific concern:** retrieved documents can contain PII that users shouldn't see (another user's data). ACL filtering at retrieval time is mandatory — filter retrieved documents to only those the current user is authorized to see.

**External API providers:** if sending data to external LLM APIs, classify data sensitivity and ensure the provider's data handling agreements match your requirements. Many organizations prohibit sending certain categories of user data to external providers entirely.

### What Breaks

PII detection with high false-negative rate — some PII passes through. Not filtering retrieved documents by user permissions — user A can see user B's data via retrieval. Logging raw prompts that contain user PII.

### What the Interviewer Is Testing

Whether you handle PII in both directions (input and output). Whether you handle retrieval ACLs. Whether you know the distinction between redaction and filtering.

### Common Traps

"Our safety classifier handles PII" — safety classifiers are not designed for PII detection. They have high false-negative rates for PII. Use dedicated NER/regex-based PII detectors.

---

## 16. Gateway Pattern for LLM API Management

### The Problem

You have 10 teams building features on the same LLM infrastructure. Each team calls the provider directly. Result: no centralized rate limiting, no policy enforcement, no consistent logging, no cost visibility, and no ability to route traffic to different models.

### The Core Insight

Every LLM call should go through a single gateway that enforces policy, controls costs, handles routing, and logs everything. The gateway is the only place where all of these can be enforced uniformly — individual teams cannot be trusted to implement them consistently.

### The Mechanics

**Gateway responsibilities:**
- Authentication / authorization (which tenant, which role, which permissions)
- Input safety classification + PII redaction
- Token-aware rate limiting per tenant
- Model routing (select model based on task type and risk level)
- Response caching (exact match and semantic)
- Logging and tracing
- Output safety filtering

```python
def llm_gateway(request):
    user = authenticate(request.token)
    enforce_acl(user, request.resource)
    
    clean_request = redact_pii(request)
    if detect_injection(clean_request.prompt):
        return refusal("policy_violation")
    
    if not token_budget.consume(user.id, estimate_tokens(clean_request)):
        return rate_limit_response()
    
    if cached := semantic_cache.lookup(clean_request.prompt):
        return cached
    
    model = route_model(clean_request.task, user.risk_level)
    response = model_client.generate(clean_request)
    
    response = safety_filter(response)
    log_trace(user, clean_request, response)
    semantic_cache.store(clean_request.prompt, response)
    
    return response
```

### What Breaks

Clients bypassing the gateway — use network policy to enforce that the gateway is the only allowed path to LLM providers. Gateway becoming a single point of failure — run multiple gateway instances, design them to be stateless. Gateway adding too much latency — run input classification and caching in parallel where possible.

### What the Interviewer Is Testing

Whether you can design a gateway that centralizes all cross-cutting concerns. Whether you know what belongs in the gateway vs application code.

### Common Traps

Building per-team gateways that drift from each other. Gateway that logs secrets (API keys in headers, raw prompts with PII).

---

## 17. Structured Output Reliability

### The Problem

Your LLM generates JSON that downstream systems parse. 3% of the time, it produces invalid JSON — a missing closing brace, a trailing comma. Those 3% of requests fail silently in production, causing downstream data corruption or API errors.

### The Core Insight

LLMs do not reliably produce valid structured output, especially for complex schemas or long outputs. Reliability requires: prompt engineering to encourage valid format + strict validation + repair loop + monitoring of format validity rate as a first-class metric.

### The Mechanics

**Reliability loop:**

```python
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    text = llm.generate(prompt, stop_sequences=["```"])
    try:
        obj = json.loads(text)
        jsonschema.validate(obj, schema)
        return obj
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        if attempt < MAX_RETRIES - 1:
            prompt = prompt + f"\n\nJSON ERROR: {e}\nOutput ONLY valid JSON matching the schema:"
        else:
            raise ValueError(f"Failed to produce valid JSON after {MAX_RETRIES} attempts")
```

**Constrained decoding** (where available): use grammar-constrained decoding (e.g., via Outlines, TGI constrained generation) to guarantee syntactically valid JSON. This is more reliable than prompt engineering alone.

**Format validity rate as a metric:** monitor this in production dashboards. A drop in format validity rate is often the first signal of a prompt regression or model behavior change.

### What Breaks

Accepting partial or malformed JSON and trying to extract fields with regex — fragile and produces incorrect data silently. Not monitoring format validity — format regressions accumulate invisibly.

### What the Interviewer Is Testing

Whether you know structured output requires explicit validation and repair, not just prompt engineering. Whether you monitor format validity as a production metric.

### Common Traps

"Constrained decoding solves it completely" — constrained decoding ensures syntactic validity but not semantic validity (correct field values, business logic). You still need schema validation.

---

## 18. Streaming Responses

### The Problem

Users want to see the first word appear in <200ms. But your safety filter and output parser run on the complete output. How do you stream while maintaining safety guarantees?

### The Core Insight

Streaming reduces perceived latency (users see words appearing) but complicates post-processing. You stream tokens to the client but must buffer the full output for validation. If validation fails, you send a correction — this requires careful UX design to handle the "correction after streaming" case.

### The Mechanics

```python
async def stream_response(prompt):
    buffer = []
    async for token in llm.stream(prompt):
        yield token            # send to client immediately
        buffer.append(token)
    
    # Full output complete
    full_output = "".join(buffer)
    
    # Post-stream validation
    try:
        obj = validate_and_parse(full_output, schema)
        yield {"type": "complete", "parsed": obj}
    except ValidationError as e:
        yield {"type": "correction", "error": str(e)}
        # Optionally: re-generate with repair prompt (hidden from UI)
```

**TTFT vs end-to-end latency:** streaming decouples these. TTFT (first token) can be <200ms while total generation takes 10s. Users perceive this as fast because they're reading as it streams.

**Stop sequence handling:** use stop sequences to prevent the model from generating beyond the expected schema boundary. Important for structured outputs embedded in streaming responses.

### What Breaks

Streaming JSON character-by-character without a correction mechanism — users see partial JSON appear then get an error. Not handling client disconnection — server continues generating and consuming GPU resources for nobody.

### What the Interviewer Is Testing

Whether you know streaming doesn't eliminate the need for validation. Whether you handle client disconnection.

### Common Traps

Not validating streamed output — errors appear after the user has read half the response. Triggering tool calls from partial streamed output before generation is complete.

---

## 19. SLAs and Metrics for Production AI

### The Problem

Your stakeholders say the system is "slow." Your ops team says latency looks fine. Your users say it sometimes takes 30 seconds. Nobody is measuring the same thing.

### The Core Insight

"Latency" in LLM systems is not a single number. TTFT and TPOT are independent, and P99 is what matters for user experience (not mean). Quality is a third dimension that's often missing from SLA definitions entirely.

### The Mechanics

**Complete SLA definition:**

| Metric | Target | Alert threshold |
| :--- | :--- | :--- |
| TTFT P50 | <200ms | >500ms |
| TTFT P99 | <1s | >2s |
| TPOT | <50ms | >100ms |
| Format validity rate | >0.99 | <0.98 |
| Safety violation rate | <0.001% | >0.005% |
| Availability | >99.9% | <99.5% |
| Cost per request | <$0.01 | >$0.02 |

**Quality SLOs** (sampled, not per-request):
- Faithfulness score on weekly production sample: >0.90
- Task accuracy on regression suite: no regression >2%

### What Breaks

Only setting mean latency targets — P99 spikes affect real users and aren't visible in the mean. No quality SLOs — you can meet all operational SLAs while generating increasingly worse answers.

### What the Interviewer Is Testing

Whether you include quality metrics in SLAs. Whether you know TTFT and TPOT are separate metrics.

### Common Traps

Setting only average latency thresholds. Treating availability (request success rate) as equivalent to quality (answer correctness).

---

## 20. Fallback Strategies

### The Problem

The primary LLM API returns a 503. You have 50,000 active users. What happens to their requests?

### The Core Insight

Every failure must have a pre-planned fallback that is safe, schema-compliant, and communicates appropriate uncertainty to users. "Return an error" is not a fallback — it's a failure. The fallback cascade should degrade quality gracefully while maintaining safety and format guarantees.

### The Mechanics

**Fallback cascade:**
- Tier 0: primary model (full quality)
- Tier 1: cached response for same/similar query (exact or semantic match)
- Tier 2: retrieval-only response (return raw retrieved text without generation)
- Tier 3: smaller/cheaper fallback model (lower quality but available)
- Tier 4: template-based response for common intent patterns
- Tier 5: human escalation / deferral with clear ETA

```python
def generate_with_fallback(prompt, context):
    try:
        return primary_model.generate(prompt, context)
    except (RateLimitError, TimeoutError):
        if cached := cache.lookup(prompt):
            return cached.with_metadata({"source": "cache"})
        try:
            return fallback_model.generate(prompt, context, max_tokens=200)
        except Exception:
            return retrieval_only_response(context)
```

**Invariants that must hold at every tier:**
- Output schema is valid (downstream systems don't break)
- Safety filters still apply (no unsafe content in fallback tier)
- Uncertainty is communicated appropriately (don't fake confidence on degraded output)

### What Breaks

Returning invalid schema from fallback tiers — downstream systems that parse the output break. Removing safety filters in fallback mode — the fallback path is an attack vector.

### What the Interviewer Is Testing

Whether you have a tiered strategy, not just "try/catch and return error." Whether safety invariants hold in every fallback tier.

### Common Traps

Not testing fallback paths — they fail in production because they were never exercised. Returning different schema from fallback tier that breaks clients.

---

## 21. Scaling for Concurrent Requests

### The Problem

Your system handles 100 requests/second fine. At 5,000 requests/second, it crashes. Where is the bottleneck?

### The Core Insight

LLM serving has multiple potential bottlenecks that saturate at different loads: GPU compute (saturates at high batch size), GPU memory (KV cache fills at high concurrent long sequences), request queuing (workers fall behind), and connection limits (too many open streams). The crash point tells you which one hit first.

### The Mechanics

**Diagnosis process:**
1. Measure at 5,000 RPS: GPU memory utilization, GPU compute utilization, queue depth, worker pool size, connection count
2. If GPU memory is 100%: KV cache exhaustion — reduce max_seq_len, reduce batch size, improve KV cache efficiency (quantize KV, use GQA)
3. If GPU compute is 100%: add more GPU workers or reduce per-request work (quantize, shorter outputs)
4. If queue depth is growing unbounded: need more workers or reduce per-request latency
5. If connection count exceeds limits: increase OS file descriptor limits, use connection pooling

**Scaling response:**
- Horizontal scaling of inference workers (stateless API servers)
- Dynamic batching within inference server (vLLM handles this automatically)
- Backpressure and admission control: reject or queue requests above capacity
- Load shedding: prioritize interactive over batch during overload

```python
if inference_pool.queue_depth > MAX_QUEUE_DEPTH:
    if request.priority == "batch":
        return {"status": "queued", "estimated_wait": estimate_wait()}
    elif request.priority == "interactive":
        return degraded_model.generate(prompt, max_tokens=200)
```

### What Breaks

Scaling web servers without scaling GPU workers — the web tier handles the connection load, but inference still bottlenecks. No admission control — requests pile up in memory until OOM.

### What the Interviewer Is Testing

Whether you diagnose before prescribing. Whether you know KV cache is a memory bottleneck distinct from GPU compute.

### Common Traps

"Add more servers" — if the bottleneck is GPU memory, more servers don't help if each server is running the same large model. Not distinguishing compute-bound from memory-bound failures.

---

## 22. Eliminating Single Points of Failure

### The Problem

Your LLM provider has a 4-hour outage. Your entire application is unavailable for 4 hours. Users can't get answers. Revenue stops.

### The Core Insight

Any dependency that, when it fails, takes your whole system down is a single point of failure. For LLM systems: the LLM provider, the vector database, the retrieval index, and the safety classifier are all potential SPOFs. Design for degraded operation when any one of them fails, not just full availability.

### The Mechanics

**Multi-provider architecture:**
```python
provider_priority = ["openai", "anthropic", "cohere"]

for provider in provider_priority:
    if not circuit_breaker.open(provider):
        try:
            return provider_clients[provider].generate(prompt)
        except Exception:
            circuit_breaker.record_failure(provider)
            continue

# All providers failed: use degraded fallback
return retrieval_only_response(context)
```

**Circuit breakers:** detect when a provider is consistently failing, stop sending requests immediately (don't wait for timeouts), and automatically re-test after a recovery period.

**Cache as reliability layer:** a warm semantic cache serves repeated queries even when the LLM provider is down. Important for high-traffic applications where a significant fraction of queries repeat.

**Resilience testing:** run chaos tests in staging — mock provider outages and validate that the system degrades gracefully rather than crashes.

### What Breaks

Unbounded retry loops during provider outage — amplifies load on recovering provider (thundering herd). Circuit breakers that never recover — set a half-open state that tests recovery automatically.

### What the Interviewer Is Testing

Whether your fallback cascade handles complete provider unavailability. Whether you test failure scenarios, not just happy paths.

### Common Traps

"We have two providers" — but the second provider uses the same model, which might be unavailable for the same reason. Diverse providers with different model families is more resilient.

---

## 23. Graceful Degradation Design

### The Problem

Your RAG retrieval system goes down. Your safety classifier goes down. Your parsing step fails. In each case, what does the user see? Do they get an error, a partial answer, or a safe fallback?

### The Core Insight

Each component failure should produce a specific, pre-planned degraded behavior. The system should never produce an unsafe output or an invalid schema, regardless of which component failed. Design this explicitly, not as an afterthought.

### The Mechanics

**Component failure → degraded behavior mapping:**

| Failed component | Degraded behavior |
| :--- | :--- |
| Retrieval (vector DB) | Generate without context; acknowledge limited context |
| Safety classifier | Refuse to respond until safety is restored (fail-safe) |
| Output parser (format failure) | Re-attempt with repair prompt; if 3 attempts fail, return acknowledgment without action |
| Tool execution | Return answer without action; flag for human review |
| LLM provider | Retrieval-only response or cached response |

**Invariants that never degrade:**
- Output schema is always valid (downstream never breaks)
- Safety filtering is never bypassed (even if classifier is down, fall to rule-based fallback)
- Tool actions require validated outputs — if parsing failed, no tool execution

```python
def safe_generate(prompt, context):
    try:
        answer = llm.generate(prompt, context)
        obj = parse_and_validate(answer, schema)
    except Exception:
        # Schema failure or generation failure
        obj = {"answer": "I'm unable to process this request right now.", 
               "action": "defer",
               "citations": []}
    
    # Safety always runs, even in degraded mode
    if not safety_classifier.is_available():
        obj = apply_rule_based_safety(obj)  # fallback safety
    elif not safety_classifier.is_safe(obj):
        obj = refusal_response()
    
    return obj
```

### What Breaks

Assuming components fail independently — correlated failures (all depend on the same infrastructure) can take down multiple safety layers simultaneously. Not testing degraded states — you discover the behavior during an incident, not in planning.

### What the Interviewer Is Testing

Whether you have explicit plans for each component failure. Whether safety invariants hold in degraded mode.

### Common Traps

"The model handles edge cases" — when components fail, the model may not even be called. Graceful degradation must work at the infrastructure level, not just the model level.

---

## The Through-Line

LLMOps is the practice of acknowledging that:

1. **The pipeline is the product, not the model.** Prompt + retrieval + safety + parsing together determine behavior. All must be versioned.

2. **Silent regressions are endemic.** Any change can break anything. Continuous eval is the only detection mechanism.

3. **Safety must be enforced at multiple layers.** Model training + input classification + output filtering + PII handling. Any single layer can be bypassed.

4. **Rollback must be instantaneous.** Feature flags over artifact bundles — not redeployments.

5. **Observe everything, alert on what matters.** Format validity, safety rate, and faithfulness are as important as latency and availability.
