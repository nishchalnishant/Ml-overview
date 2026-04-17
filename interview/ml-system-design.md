# ML System Design

This file is the fast playbook for design rounds.

Do not start with the model.

Start with:

- goal
- metric
- constraints
- data

That order alone makes you sound better.

---

## 1. The Standard Framework

1. define product goal
2. define success metric
3. define latency / scale / freshness constraints
4. define labels and features
5. choose baseline and architecture
6. define offline and online evaluation
7. define rollout and monitoring

Simple.
Reliable.
Very reusable.

---

## 2. Batch vs Real-Time

Batch:

- high throughput
- heavier models okay
- lower freshness

Real-time:

- low latency
- lighter serving path
- fresher decisions

Pick based on product need, not elegance.

---

## 3. Two-Stage Pattern

Very common in:

- search
- recommendations
- ads

Stage 1:

- retrieval / candidate generation

Stage 2:

- ranking

This pattern exists because scoring everything with a heavy model is usually too expensive.

---

## 4. Recommendation Design

Common pieces:

- candidate generation
- ranking
- user features
- item features
- context
- cold start handling
- offline / online metrics

If you mention cold start, diversity, and feedback loops, the answer gets better fast.

---

## 5. Fraud Design

Common pieces:

- real-time scoring
- velocity features
- rules + model hybrid
- review queue
- cost-aware threshold
- drift handling

Fraud is never just "train a classifier."

It is an adversarial system.

---

## 6. Search Ranking

Common pieces:

- query understanding
- retrieval
- ranking
- personalization
- evaluation with NDCG / MRR style metrics

Again: retrieval then ranking is often the right structure.

---

## 7. Monitoring

Always mention:

- data drift
- prediction drift
- latency
- throughput
- errors
- business KPI
- rollback path

If you skip monitoring, the design feels unfinished.
