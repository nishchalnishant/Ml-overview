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

- [ ] [Foundations](../01-foundations/flashcards.md) — math, probability, optimization
- [ ] [Classical ML](../02-classical-ml/classical-ml-flashcards.md) — trees, SVMs, regularization, metrics
- [ ] [Deep Learning](../03-deep-learning/deep-learning-cheatsheet.md) — backprop, attention, training tricks
- [ ] [Specialized Domains](../04-specialized-domains/specialized-domains-cheatsheet.md) — RL, RecSys, GNN
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

---

## Post-Offer — Salary Negotiation for ML Roles

### Market Data First

Never negotiate without data. Sources for ML compensation:

- **levels.fyi** — the most accurate for total comp (TC) at large tech companies. Filter by level (L4/L5/L6), location, and YOE. TC = base + stock/4 + bonus.
- **Glassdoor / Blind** — useful for directional data but noisier than levels.fyi. Use for companies not on levels.fyi.
- **Competing offers** — the single most powerful negotiating tool. If you have one, use it. "I have a competing offer at $X TC" is worth more than any data source.

### ML-Specific Compensation Structure

ML roles at large companies are almost always equity-heavy:

| Level | Typical TC range (Bay Area 2024–2025) | Stock/year (est.) |
|-------|--------------------------------------|-------------------|
| L4 (new grad, 2–3 YOE) | $200K–$280K | $60K–$120K |
| L5 (senior, 4–7 YOE) | $280K–$420K | $120K–$200K |
| L6 (staff/principal) | $400K–$600K+ | $200K–$400K |

**Negotiate on stock, not just base.** Base salary has relatively little room (companies have salary bands). Equity grant, signing bonus, and accelerated vesting are where real movement happens.

### The Negotiation Sequence

**Step 1: Never accept on the spot.** "Thank you — I'd like to take some time to review the full offer. Can you give me until [date, 5–7 business days out]?" This is expected and costs you nothing.

**Step 2: Get the full offer in writing.** Confirm: base salary, equity grant (shares or dollar value), vesting schedule (standard is 4-year with 1-year cliff), signing bonus, annual bonus target (%), benefits, start date.

**Step 3: Research the number.** Use levels.fyi to find median TC for your level, location, and company. Know your target before you counter.

**Step 4: Counter.** The standard counter is 10–20% above the offer on total comp. Anchoring: "Based on my research and a competing offer at $X, I was hoping we could get closer to $Y." You need a reason — data or a competing offer. "I just want more" is weak.

**Step 5: If they can't move on base, move the conversation to equity and signing.** "I understand base is fixed at band. Could we look at an additional equity grant or a larger signing bonus to close the gap?"

**Step 6: Get a revised offer in writing before ending the call.**

### Negotiation Psychology

- **Silence is your friend.** After making a counter, stop talking. The discomfort of silence causes people to fill it — often with concessions.
- **Be specific.** "Can we do $350K TC?" is stronger than "I was hoping for more." A specific number anchors the conversation.
- **Don't reveal your floor.** "What's the minimum you'd accept?" is a trap. Redirect: "I'm focused on getting to market rate for this level — levels.fyi shows $X is the median."
- **Competing offers are not bluffs.** If you don't have a competing offer, don't pretend you do. If you do, reference it clearly.
- **Don't over-negotiate.** Counter once, maybe twice. Repeatedly going back erodes goodwill and can retract the offer at small companies.

### ML-Specific Leverage Points

- **Specialized skills** — if you have production LLM experience, RL expertise, or large-scale ML systems experience, make this explicit. Specialized skills justify top-of-band.
- **Competing offers from direct competitors** — a Meta offer at an Alphabet negotiation carries more weight than a random startup offer.
- **Time constraints** — if you have another offer expiring, state the deadline professionally. It creates urgency without pressure tactics.
- **Relocation** — if relocating, you can ask for a relocation package in addition to signing bonus.

### What Not to Do

- Don't disclose your current salary unless legally required. In many US states (CA, NY, WA, IL), employers cannot ask about past compensation.
- Don't accept verbally and then negotiate further — that burns trust.
- Don't negotiate if you plan to decline. Only negotiate if you would accept the revised offer.
