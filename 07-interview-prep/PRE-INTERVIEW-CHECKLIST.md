---
module: Interview Prep
topic: Pre-Interview Checklist
subtopic: ""
status: unread
tags: [interview, checklist, revision]
---
# Pre-Interview Checklist

Three windows: 48 hours out, 24 hours out, morning-of.

---

## 48 Hours Out — Knowledge Check

Goal: identify gaps, not fill them. If you find a genuine hole, address it now. After 24h, no new material.

### Core Topics — Read the Revision Cards

- [ ] [Foundations](../01-foundations/REVISION.md) — math, probability, optimization
- [ ] [Classical ML](../02-classical-ml/REVISION.md) — trees, SVMs, regularization, metrics
- [ ] [Deep Learning](../03-deep-learning/REVISION.md) — backprop, attention, training tricks
- [ ] [Specialized Domains](../04-specialized-domains/REVISION.md) — RL, RecSys, GNN
- [ ] [LLMs](../05-llms/REVISION.md) — architecture, RLHF/DPO, LoRA, inference
- [ ] [Production ML](../06-production-ml/REVISION.md) — MLOps, drift, deployment strategies

### Verify These Are Solid (Not Just Familiar)

**Math you must whiteboard without hesitation:**
- [ ] Softmax + cross-entropy loss derivation
- [ ] Backprop through a two-layer network (chain rule, dimensions)
- [ ] Attention: $\text{softmax}(QK^T/\sqrt{d_k})V$ — why $\sqrt{d_k}$?
- [ ] Bellman equation for Q-learning
- [ ] NDCG formula and why it beats Precision@K

**Concepts you must explain without buzzwords:**
- [ ] Bias-variance tradeoff — give a concrete example
- [ ] Why batch norm helps (and when it doesn't)
- [ ] KV cache — what it stores, why it grows, how GQA helps
- [ ] Training-serving skew — how it silently kills production models
- [ ] Over-smoothing in GNNs — why and how to fix

### System Design — Know One End-to-End Example Cold

Pick one you're confident with. Know the full pipeline, latency budget, failure modes:
- [ ] Recommendation system (retrieval → ranking → re-ranking)
- [ ] Fraud detection (streaming + batch features, ensemble, explainability)
- [ ] Search ranking (query understanding → retrieval → LTR → serving)
- [ ] LLM serving (tokenization → KV cache → batching → output)

**Reference:** [ML System Design](ml/system-design-and-mlops.md) | [LLM System Design](llm/ml-system-design.md)

### Behavioral — Have 3 Stories Ready

Format: Situation → Task → Action → Result (with numbers where possible).

- [ ] Story 1: Technical disagreement you resolved
- [ ] Story 2: Project that failed or pivoted — what you learned
- [ ] Story 3: Cross-functional collaboration or stakeholder management

**Reference:** [Behavioral Questions](ml/behavioral-and-scenario-based-questions.md)

---

## 24 Hours Out — Sharpen, Don't Stuff

Goal: reinforce what you know. No new topics. Sleep matters more than one more paper.

### The 20-Minute Focused Review

Pick the 3 topics most likely to come up given the role. For each:
1. Say the core idea out loud in 30 seconds (no notes)
2. Write the key formula from memory
3. State one real failure mode or gotcha

### Common Role-Specific Emphasis

| Role type | Extra focus |
|-----------|-------------|
| ML Engineer / MLOps | Deployment strategies, training-serving skew, feature stores, monitoring |
| Research Scientist | Scaling laws, fine-tuning methods, evaluation benchmarks, recent papers |
| Applied Scientist | End-to-end system design, metrics, experimentation (A/B testing) |
| RecSys / Ranking | Two-tower, LTR (BPR/LambdaMART), NDCG, cold start, diversity |
| LLM / GenAI | RLHF vs DPO, RAG, LoRA, KV cache, hallucination mitigations |
| GNN / Graph ML | Message passing, over-smoothing, scalability, KG embeddings |

### Logistics Confirmed

- [ ] Interview time confirmed (with timezone)
- [ ] Location / video link tested (camera, mic, screen share)
- [ ] Pen and paper / whiteboard app ready for live coding and diagrams
- [ ] Quiet space booked
- [ ] Water on desk

### Light Coding Warmup (30 min max)

Run through these without looking anything up:
- [ ] Implement softmax from scratch (numerically stable version)
- [ ] Implement a simple Q-learning update step
- [ ] Implement NDCG@K from scratch
- [ ] Sketch a two-tower model in PyTorch (forward pass only)

**Reference:** [ML Coding Patterns](llm/ml-coding-patterns.md) | [Algorithms](ml/algorithms.md)

### Mental Prep

- [ ] Review the job description once — note 2-3 terms they use that you should mirror
- [ ] Prepare 2 questions to ask the interviewer (shows genuine interest)
- [ ] Recall a recent paper or system that genuinely excited you — authenticity lands better than rehearsed answers

---

## Morning Of — Final 10 Minutes

Goal: activate, don't cram. Read fast, don't write.

### Quick-Scan Checklist (in order)

1. **Decision tables** — glance at the "Algorithm Selector" tables in the revision cards for your 3 focus topics. Refresh the mental map of when to use what.

2. **Your system design story** — 60-second mental walkthrough: components, bottlenecks, how you'd scale it.

3. **Your 3 behavioral stories** — one sentence each: what happened, what you did, what the outcome was.

4. **One formula per domain** you'll likely be asked about:
   - Loss function for your focus area (cross-entropy, BPR, DQN target, PPO clip)
   - Attention formula if LLM role

5. **The 3 most common gotchas for your role** — the things candidates forget that you will remember:
   - MLOps: point-in-time correctness in feature joins
   - RecSys: popularity bias, offline→online metric gap
   - GNN: over-smoothing with deep stacks
   - LLM: KV cache is memory-bound, not compute-bound

### State Check

- [ ] Slept 7+ hours
- [ ] Eaten
- [ ] 10 minutes early (video) or 15 minutes early (onsite)

---

## During the Interview — Mental Checklist

These are things to do *in the room* when you get a question:

**For any technical question:**
1. Restate the question to confirm you understood it
2. State your assumptions
3. Give the direct answer in 1-2 sentences before going deep
4. After explaining, state a tradeoff or failure mode — shows depth

**For system design:**
1. Clarify scale (users, items, QPS, latency SLA)
2. Sketch the pipeline end-to-end before any component details
3. Call out the hardest part explicitly ("the bottleneck here is...")
4. Address failure modes before they ask

**For ML coding:**
1. Test with a trivial input before optimizing
2. State time/space complexity when done
3. Mention what you'd add with more time (logging, edge cases, vectorization)

**If you don't know:**
→ "I haven't implemented that specific variant, but here's how I'd think through it:" — then reason from first principles. This is often better than a rehearsed answer.

---

## Post-Interview — For Next Time

Within 1 hour of finishing:
- [ ] Write down every question you were asked (exact phrasing matters)
- [ ] Note which answers landed well and which felt weak
- [ ] For weak spots: add them to your study list, don't just move on

This feedback loop compounds. The second interview is always easier than the first for the same reason the second time reading a paper is faster.
