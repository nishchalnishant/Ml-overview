---
module: LLM Applications
topic: Applications
subtopic: Eval And Observability
status: unread
tags: [llms, ml, applications-eval, observability, tracing]
---
# LLM Eval Frameworks and Observability

> **Related:** for benchmark-level model evaluation (MMLU, GSM8K, LMSYS Arena) and the mechanics of LLM-as-judge and RAGAS metrics, see [10-llms/04-evaluation-benchmarks.md](../10-llms/04-evaluation-benchmarks.md). This file covers the **tooling layer** — the frameworks and platforms that run those evaluations continuously in a pipeline, and the tracing systems that observe production behavior.

---

## The Core Problem

**The problem:** benchmark scores and one-off LLM-as-judge checks tell you if a model or prompt is good in isolation. They do not tell you whether a *specific production request* failed, why it failed, which pipeline stage caused it, or whether last Tuesday's prompt change regressed quality for a subset of users. That requires two different capabilities: an **evaluation framework** to run structured, repeatable quality checks (offline, in CI, or online against live traffic) and an **observability/tracing platform** to record what actually happened on every call so failures are debuggable after the fact.

**The core insight:** evaluation answers "is this good?" against a metric; observability answers "what happened?" against a specific trace. Production LLM systems need both — eval gates changes before they ship, tracing explains what shipped once it's live. Treating them as the same problem (e.g., only logging outputs without structured traces, or only running offline eval without tracking regressions in production) leaves one half of the debugging loop missing.

---

## Evaluation Frameworks

**The problem:** hand-rolled eval scripts (a for-loop calling an LLM judge) do not scale — they don't version test sets, don't integrate with CI, don't standardize metric definitions across a team, and don't distinguish offline (pre-deploy, against a fixed test set) from online (live traffic, sampled) evaluation.

**The mechanics — offline vs. online eval:**

| Mode | When it runs | What it catches | Cost |
|:---|:---|:---|:---|
| Offline | Pre-deploy, CI, against a fixed labeled/golden test set | Regressions before they reach users | Cheap, fast, but test set can go stale |
| Online | Continuously, against sampled live traffic | Real-world drift, edge cases the test set didn't anticipate | More expensive; needs sampling strategy |

**Framework comparison:**

| Framework | Focus | Notes |
|:---|:---|:---|
| RAGAS | RAG-specific metrics (faithfulness, context precision/recall, answer relevancy) | See [02-rag.md](02-rag.md#evaluation) and [10-llms/04-evaluation-benchmarks.md](../10-llms/04-evaluation-benchmarks.md#10-domain-specific-evaluation-why-standard-benchmarks-fail-in-production) for the metric definitions. Narrowly scoped to RAG; not a general eval harness. |
| DeepEval | General-purpose LLM eval, pytest-style | Ships RAGAS-equivalent metrics (faithfulness, answer relevancy, contextual precision/recall) plus non-RAG metrics (hallucination, bias, toxicity, summarization quality, custom G-Eval-style rubrics). Integrates as a pytest plugin, so eval suites run in CI like unit tests — this is its main differentiator from RAGAS. VERIFY current metric list against the library's docs before citing specific metric names in production code, since eval-framework APIs iterate quickly. |
| Custom LLM-as-judge harness | Anything not covered by a named metric | See [10-llms/04-evaluation-benchmarks.md §7](../10-llms/04-evaluation-benchmarks.md) for absolute-scoring vs. pairwise-comparison mechanics and known judge biases (position bias, verbosity bias, self-preference). Any framework above still inherits these biases — the framework only standardizes the harness, not the judge's reliability. |

**What breaks:** a golden/offline test set decays — it stops representing the live query distribution as usage patterns shift, so a pipeline can pass all offline evals while degrading in production. Online eval requires a sampling strategy (sample failures at a higher rate than successes) or the signal drowns in cost. Any LLM-as-judge-based framework inherits judge unreliability — a green CI eval run is only as trustworthy as the judge model and rubric behind it.

---

## Observability and Tracing

**The problem:** a multi-step LLM pipeline (retrieval → rerank → generation → tool call → generation) can fail at any stage, and the final output alone does not reveal which stage broke. Debugging without a trace means re-running the pipeline manually and guessing.

**The core insight:** instrument every LLM call, tool call, and retrieval step with structured spans — inputs, outputs, latency, token counts, cost — so a failed run can be replayed step-by-step instead of reverse-engineered from the final answer. This is the same instrumentation discipline as distributed tracing in traditional backend systems (OpenTelemetry spans), applied to LLM call chains.

**Platform comparison:**

| Platform | Model | Notes |
|:---|:---|:---|
| LangSmith | Hosted (LangChain-affiliated) | Deepest integration with LangChain/LangGraph; also works framework-agnostic via SDK. Dataset management + online eval built in. |
| Langfuse | Open-source, self-hostable or hosted | Framework-agnostic tracing, prompt management, and eval scoring; open-source core is a common choice when self-hosting/data residency matters. |
| Arize Phoenix | Open-source, self-hostable or hosted (Arize AI) | Built on OpenTelemetry (OpenInference spec) for traces; strong on embedding/drift visualization, inherited from Arize's traditional-ML observability roots. |

VERIFY current feature parity and pricing tiers across these three before recommending one — the space moves fast and this table reflects positioning, not a benchmarked comparison.

**The mechanics — what to record per span:**
- Every LLM call: model name, prompt (with PII redacted), completion, prompt/completion token counts, latency, cost.
- Every tool/retrieval call: tool name, arguments, result, latency, success/failure.
- Full trace: the ordered sequence of spans for one end-to-end request, with parent-child relationships so nested agent/tool calls are visually nested, not flattened.

**What breaks:** tracing every field of every call at 100% sampling gets expensive fast at scale — a common pattern is 100% sampling on errors/flagged traces and a low sampling rate (1-5%) on successes. Logging raw prompts/completions verbatim is a PII and compliance risk if user data flows through the pipeline — redact or hash before the trace is persisted, not after. Traces without prompt/config versioning make it impossible to correlate a quality regression with the deploy that caused it.

---

## Regression Testing for Prompts and Pipelines

**The problem:** a prompt change, a model version bump, or a retrieval config tweak can silently regress quality for a subset of queries while improving the aggregate metric — averages hide regressions.

**The core insight:** treat prompts and pipeline configs as versioned artifacts (see [03-prompt-optimization-and-versioning.md](03-prompt-optimization-and-versioning.md)) and run the same offline eval suite against the old and new version on the same test set before rollout, comparing per-example deltas, not just aggregate scores.

**What breaks:** aggregate-only comparison ("average faithfulness went from 0.91 to 0.92") can mask a regression that made 5% of queries much worse while improving the rest. Always diff per-example, and flag any example whose score dropped past a threshold for manual review before shipping.

---

## Production Checklist

| Concern | Mitigation |
|:---|:---|
| Offline test set staleness | Periodically refresh from sampled production traffic (with review) |
| Judge unreliability (any framework) | Track judge-vs-human agreement on a spot-checked sample |
| Tracing cost at scale | Sample: 100% failures/flagged, 1-5% successes |
| PII in traces | Redact/hash before persisting, not after |
| Silent regressions | Per-example diff against a fixed test set on every prompt/model/config change |
| Framework lock-in | Prefer OpenTelemetry-based tracing where available for platform portability |

---

## Interview Angles

**Q: Your RAG faithfulness score in CI is 0.93 and stable. Users still report hallucinations in production. Why might these not be in tension?**
A: The offline test set does not represent the live query distribution — it may not contain the tail queries or document types that are actually failing, so the CI number can be accurate and simultaneously irrelevant to what's failing live. This is why online eval (sampling real traffic) and full-trace observability are not optional extras once RAG ships — the offline number alone cannot detect distribution shift between the test set and production usage.

**Cross-questions to expect:**
- *How do you close the loop between a production trace showing a hallucination and the offline eval suite?* -> Pull the failing trace's input into the golden test set (after review/labeling), so the specific failure mode that offline eval missed becomes a permanent regression check going forward. Without this loop, the test set never learns from production and the same class of failure recurs silently.

**Trap:** treating a passing eval score as proof of production quality rather than proof against a specific, potentially stale test set. The score is only as good as the set's coverage of the live distribution.

## Flashcards

**What's the difference between an eval framework and an observability/tracing platform, and why do you need both?** #flashcard
An eval framework (RAGAS, DeepEval) answers "is this good?" against a defined metric, run offline (CI, pre-deploy) or online (sampled live traffic). An observability platform (LangSmith, Langfuse, Arize Phoenix) answers "what happened?" for a specific request, via structured traces of every LLM/tool/retrieval call. Eval gates changes before they ship; tracing explains behavior once they're live — a production debugging loop needs both.

**Why does DeepEval differ from RAGAS beyond just "more metrics"?** #flashcard
RAGAS is narrowly scoped to RAG-specific metrics (faithfulness, context precision/recall, answer relevancy). DeepEval covers those plus non-RAG metrics (hallucination, bias, toxicity, custom rubrics) and integrates as a pytest plugin, so LLM eval suites run in CI the same way unit tests do — the CI-native workflow is the main practical differentiator, not just metric count.

**Why can 100% trace sampling become a scaling problem, and what's the standard mitigation?** #flashcard
Recording every field of every LLM/tool call at scale grows storage and cost non-linearly with traffic. Standard mitigation: sample 100% of errors/flagged traces (where debugging value is highest) and a low rate (1-5%) of successful traces (where marginal value per trace is low).

**Why must PII redaction in traces happen before persistence, not after?** #flashcard
Once raw prompts/completions containing PII are written to a trace store, they exist in logs, backups, and potentially downstream analytics before any later redaction pass can reach them — a compliance risk that "redact later" doesn't close. Redaction must happen in the instrumentation layer, before the span is persisted.

**Why is aggregate-only score comparison insufficient for regression testing prompts or pipeline configs?** #flashcard
An aggregate metric (e.g., mean faithfulness) can improve overall while a subset of examples regresses significantly — the average hides the regression. Correct practice: diff per-example scores between old and new versions on the same test set, and flag any example that dropped past a threshold for manual review before rollout.

*Related: [RAG](02-rag.md) | [Agentic Workflows](01-agentic-workflows.md) | [Prompt Optimization and Versioning](03-prompt-optimization-and-versioning.md) | [10-llms/04-evaluation-benchmarks.md](../10-llms/04-evaluation-benchmarks.md)*
