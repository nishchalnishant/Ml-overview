# Communication Framework

## Goal
How to communicate concisely, in a structured way, and always connected to engineering reasoning.

## Mental Model
**"State → Justify → Tradeoff → Improve."**
Every answer should follow this pattern: what you chose, why you chose it, what alternatives exist, and what you'd improve given more time.

---

## The SJTF Answer Template

For any design or decision question, use:
```
S — STATE: "I would [do X] using [Y] because [Z]."
J — JUSTIFY: "The key reason is [engineering justification]."
T — TRADEOFF: "The tradeoff is [A vs B]. In this context, [A] wins because [constraint]."
F — FUTURE: "Given more time/resources, I would [improvement]."
```

**Example:** "Why did you choose XGBoost over a deep neural network?"
- **S:** "I chose XGBoost for this fraud detection task."
- **J:** "The data is tabular with about 50 features, and XGBoost is state-of-the-art for tabular data — it handles missing values natively, requires no scaling, and is highly interpretable via SHAP values, which is required for compliance."
- **T:** "The tradeoff is that XGBoost can't capture complex non-linear interactions as well as a deep network on very large datasets. But with 1M rows and 50 features, we're in XGBoost's optimal range."
- **F:** "With more data (100M+ rows) and if compliance permitted, I would explore a TabNet or SAINT architecture that might capture higher-order feature interactions."

---

## Answering "Why did you choose this?"

### Framework
1. State the choice explicitly.
2. Name the constraint that drove the decision (latency, scale, interpretability, data size).
3. Eliminate one or two obvious alternatives with a single sentence each.
4. Confirm the choice fits the stated constraints.

**Template:**
> "I chose [X] because our primary constraint is [Y]. [Alternative A] would fail because [reason]. [Alternative B] would be better if [different constraint], but given [our constraint], [X] is optimal."

**Example:** "Why gRPC instead of REST?"
> "I chose gRPC because this is an internal service-to-service call where latency is critical and both sides are under our control. REST would add JSON serialization overhead and lacks built-in streaming support. If this were an external public API, I'd use REST for universal compatibility."

---

## Answering "What are the tradeoffs?"

### Framework: Identify the Tension
Every tradeoff has two valid values in tension. Name them explicitly.

```text
Common tension pairs:
├── Accuracy ↔ Latency
├── Recall ↔ Precision
├── Cost ↔ Quality
├── Simplicity ↔ Flexibility
├── Real-time ↔ Batch
├── Privacy ↔ Personalization
└── Consistency ↔ Availability
```

**Template:**
> "The core tradeoff here is [A] vs [B]. If we prioritize [A], we gain [benefit] but lose [cost]. If we prioritize [B], we gain [benefit] but lose [cost]. Given [business context], I'd optimize for [A] and accept [cost of B]."

**Example:** "What are the tradeoffs between batch and real-time inference?"
> "The core tradeoff is cost vs. freshness. Batch inference is 10–100x cheaper (you can use spot instances, GPU utilization is near 100%), but predictions can be hours stale. Real-time inference uses fresh context but requires always-on infrastructure with strict SLAs. For recommendations on the home screen, I'd use batch (predictions update nightly). For fraud detection, I'd use real-time — a stale prediction is unacceptable."

---

## Answering "What would you improve?"

### Framework: Pyramid of Improvements
```text
Level 1 (Quick wins): What I'd do in the next sprint.
Level 2 (Architectural): What requires a larger refactor.
Level 3 (Research): What would require experimentation.
```

**Template:**
> "There are three levels of improvements I'd prioritize. Short-term: [quick win, 1 sprint]. Medium-term: [architectural change, 1 quarter]. Long-term: [research direction, exploratory]."

**Example:** "What would you improve about this RAG system?"
> "Short-term: Add re-ranking (cross-encoder). The single biggest precision improvement for minimal engineering effort. Medium-term: Move to hybrid search (BM25 + dense), which will improve recall on keyword-heavy queries like game patch versions or item names. Long-term: Explore GraphRAG for handling multi-hop queries where answers require combining information across multiple documents."

---

## Answering "What would break first?"

### Framework: Single Point of Failure Analysis
1. Identify all external dependencies.
2. Order them by: impact × probability of failure.
3. Name your mitigation for the top 2–3.

**Template:**
> "The most likely failure point is [component] because [reason: single point, no redundancy, external dependency]. At scale, [secondary failure point] would be my second concern. My mitigation is [X] for the first and [Y] for the second."

**Example:** "What would break first in your LLM-powered support system?"
> "The most likely failure is the LLM API — either rate limits or provider outages. I mitigate with a secondary provider fallback (Azure OpenAI → Anthropic → local LLaMA) and a circuit breaker. The second failure point is the vector DB during a large document ingestion event, which can spike latency. I mitigate with a read-replica pattern and request queuing."

---

## Answering "What assumptions are you making?"

### Framework
1. Be transparent — interviewers respect engineers who know their assumptions.
2. Classify: data assumptions, scale assumptions, business assumptions.
3. Explain what would change if the assumption is wrong.

**Template:**
> "I'm making three key assumptions: [1] about data, [2] about scale, [3] about business requirements. If [assumption 1] is wrong, I would [change]. If [assumption 2] is wrong, I would [change]."

**Example:** "Assumptions for this recommendation system?"
> "I'm assuming: 1) Users have sufficient interaction history (10+ events). For new users, I'd need a cold-start strategy like a popularity-based fallback. 2) We're optimizing for click-through rate, not watch time or revenue. Different objectives require different loss functions. 3) Latency SLA is 200ms end-to-end. If it's 50ms, I'd move to pre-computed predictions only."

---

## Answering "How would you scale this?"

### Framework: Scale in Three Dimensions
```text
1. DATA scale: More training data → distributed training, data sharding.
2. TRAFFIC scale: More QPS → horizontal pod scaling, caching, load balancing.
3. MODEL scale: Larger model → model parallelism, quantization, distillation.
```

**Template:**
> "I'd address scaling in three dimensions. For [data scale], I'd [strategy]. For [traffic scale], I'd [strategy]. For [model scale], I'd [strategy]. The first bottleneck I expect to hit at 10x scale is [bottleneck], so I'd instrument that immediately."

---

## Whiteboarding Communication

### Rules
1. **Think out loud.** Silence is your enemy. Interviewers want to see your thinking.
2. **State before drawing.** "I'm going to draw the data pipeline first." Don't draw silently.
3. **Confirm requirements before architecturing.** "Before I design, let me confirm: our SLA is 200ms, and we expect 10k QPS. Is that right?"
4. **Use a consistent notation.** Boxes for services, arrows for data flow, cylinders for databases.
5. **Revisit and refine.** "I've sketched the happy path. Let me now add failure handling."

---

## Recovering from a Wrong Answer

### How to Handle Mistakes
```text
WRONG approach:
→ "Actually wait, uh... let me... hmm..." [long silence, losing confidence]

RIGHT approach:
→ "Actually, I want to revisit that. I said [X], but thinking through the constraints
   more carefully, [Y] is a better choice because [reason]. [X] would have [problem]."
```

Interviewers explicitly value candidates who can self-correct with reasoning, not candidates who never make mistakes.

---

## Structuring a 5-Minute Answer

```text
[0:00 – 0:30] Restate the question. "So the question is about [X]."
[0:30 – 1:30] High-level answer. "At a high level, I would [approach]."
[1:30 – 3:30] Dive into the most important detail. "The key engineering decision is [X] vs [Y]. I choose [X] because [Z]."
[3:30 – 4:30] Tradeoffs. "The downside of [X] is [A]. I'd mitigate it by [B]."
[4:30 – 5:00] Check-in. "Does this level of detail make sense, or would you like me to dive deeper into [component]?"
```

---

## Engineering Checklist (Communication)

- [ ] Did I restate the question to confirm I understood it?
- [ ] Did I start with requirements/constraints before architecture?
- [ ] Did I state my choices explicitly, not hint at them?
- [ ] Did I name the alternatives I rejected and why?
- [ ] Did I connect engineering decisions to business requirements?
- [ ] Did I offer improvements without being asked?
- [ ] Did I check in with the interviewer mid-way?

## Interview Follow-up Questions & Best Answers

**Q: "You've been talking for 10 minutes. Tell me your key design choices in 60 seconds."**
*Best Answer:* "Sure. The three critical decisions:
1. **Model:** XGBoost for tabular churn prediction — best accuracy/latency tradeoff for structured data, and SHAP-interpretable.
2. **Serving:** Batch inference (nightly) for home screen recommendations, real-time inference only for live game actions — balances cost and freshness.
3. **Monitoring:** Feature PSI for data drift, CTR tracking for concept drift, with auto-rollback if P99 exceeds 200ms.
These three choices directly address our constraints: 10k QPS, < 200ms SLA, and the requirement for model interpretability for the game team."
</content>
