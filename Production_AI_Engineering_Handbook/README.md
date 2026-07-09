# Production AI Engineering Decision Frameworks
## A Mental Operating System for Senior AI Engineering Interviews

> *Designed for SDE-2 / Senior AI Engineer roles. Opinionated, production-first, reasoning-driven.*

---

## What This Handbook Is (And Is Not)

**IS:** A repeatable mental framework for solving any production AI system design interview.
**IS NOT:** A textbook on ML algorithms or a collection of facts to memorize.

The goal is **engineering judgment** — the ability to make principled decisions under uncertainty, communicate tradeoffs clearly, and defend every architectural choice.

---

## The Handbook Structure

| Part | Title | Core Question Answered |
| :--- | :--- | :--- |
| **01** | [Universal AI Interview Framework](./01_Universal_Framework.md) | How do I approach any AI problem from scratch? |
| **02** | [Requirement Gathering Framework](./02_Requirement_Gathering.md) | What questions should I ask before designing? |
| **03** | [Model Selection Decision Trees](./03_Model_Selection.md) | When do I use XGBoost vs LLMs vs RAG? |
| **04** | [Data Decision Framework](./04_Data_Decision_Framework.md) | How do I handle missing data, imbalance, drift? |
| **05** | [Training Decision Framework](./05_Training_Decision_Framework.md) | How do I choose batch size, optimizer, scheduler? |
| **06** | [Evaluation Decision Framework](./06_Evaluation_Framework.md) | Which metrics for which task? Offline vs online? |
| **07** | [Deployment Decision Framework](./07_Deployment_Framework.md) | REST vs gRPC, batch vs real-time, canary vs blue-green? |
| **08** | [RAG Decision Framework](./08_RAG_Framework.md) | How do I design, optimize, and debug a RAG system? |
| **09** | [Agent Decision Framework](./09_Agent_Framework.md) | When agents? Single vs multi? Memory? Security? |
| **10** | [System Design Framework](./10_System_Design_Framework.md) | How do I design for scalability, reliability, and latency? |
| **11** | [Observability Framework](./11_Observability_Framework.md) | How do I monitor ML systems in production? |
| **12** | [Security Framework](./12_Security_Framework.md) | Prompt injection, PII, RAG attacks, secrets? |
| **13** | [Cost Optimization Framework](./13_Cost_Optimization_Framework.md) | How do I reduce GPU, LLM, and infra cost? |
| **14** | [Failure Analysis Framework](./14_Failure_Analysis_Framework.md) | How do I debug bad predictions, high latency, OOM? |
| **15** | [Communication Framework](./15_Communication_Framework.md) | How do senior engineers communicate decisions? |
| **16** | [Game AI Decision Frameworks](./16_Game_AI_Framework.md) | Churn, matchmaking, cheat detection, DDA for EA? |
| **17** | [Interview Playbook](./17_Interview_Playbook.md) | Minute-by-minute 90-minute interview blueprint? |

---

## Core Mental Models (The Five Pillars)

### 1. Start Simple, Justify Complexity
```
Heuristic → Classical ML → Deep Learning → LLMs
```
Every step up requires justification. If a rule-based system solves 80% of the problem, start there.

### 2. Data is the Fuel
A perfect model on bad data is a perfect failure. Fix data quality before changing the model.

### 3. Optimize Offline, Validate Online
Offline metrics are necessary but not sufficient. Business KPIs are the true north star.

### 4. Design for Failure, Not Success
Every component will fail. Design graceful degradation, automatic recovery, and observability.

### 5. Cost is a Feature
A model that bankrupts the company is not a successful AI system. Model efficiency is an engineering competency.

---

## The Universal Answer Template (SJTF)

For every design or decision question:

```
STATE    → "I would [X] using [Y]."
JUSTIFY  → "Because [engineering reason tied to a constraint]."
TRADEOFF → "The tradeoff is [A vs B]. Given [constraint], I accept [cost]."
FUTURE   → "Given more time, I would [improvement]."
```

---

## Quick Reference: The Constraint Checklist

Before designing any system, extract these constraints:

```
□ Business goal (revenue / engagement / trust / cost reduction)
□ Scale (DAU, QPS, data volume in TB/PB)
□ Latency SLA (real-time <100ms, async <5s, batch <24h)
□ Data availability (labeled? delayed? historical depth?)
□ Privacy & compliance (GDPR, CCPA, HIPAA?)
□ Budget (can we afford GPU serving?)
□ Failure tolerance (fail open or fail closed?)
□ Update frequency (retrain daily, weekly, on-demand?)
```

---

## Quick Reference: Model Selection

| Data Type | Default Model | Upgrade When |
| :--- | :--- | :--- |
| Tabular | XGBoost / LightGBM | > 100M rows with complex interactions → TabNet |
| Text classification | FastText / DistilBERT | Need multilingual → XLM-R |
| Text generation | GPT-4o-mini (API) | Need private/custom → Llama 3 fine-tuned |
| RAG Q&A | Hybrid search + GPT-4o-mini | High accuracy needed → add re-ranking |
| Time series | LightGBM with lag features | Long-range dependencies → TFT / PatchTST |
| Image classification | ResNet-50 / EfficientNet | Multi-modal → CLIP |
| Recommendations | Two-tower model | Sequential behavior → SASRec / BERT4Rec |

---

## Quick Reference: Deployment Defaults

| Scenario | Default Choice |
| :--- | :--- |
| Latency SLA < 100ms | gRPC + Kubernetes autoscaling |
| Latency SLA flexible | REST + batch queue |
| Internal service calls | gRPC |
| Public API | REST |
| Model size > GPU memory | Quantize (INT8) → Shard → Distill |
| Release strategy | Canary (1% → 5% → 25% → 100%) |
| Rollback trigger | P99 > SLA or business KPI drop > 5% |

---

## Quick Reference: Monitoring Defaults

| Layer | Tool | Alert Condition |
| :--- | :--- | :--- |
| Infrastructure | Prometheus + Grafana | CPU > 80%, GPU < 30% utilization |
| Data drift | Evidently AI | PSI > 0.2 on key features |
| Business | Custom dashboard + PagerDuty | KPI drop > 5% vs. 7-day baseline |
| LLM | LangSmith / Phoenix | Faithfulness score < 0.7 |
| Agents | LangSmith | Error rate > 2%, cost > budget |

---

## How to Use This Handbook

**Before an interview:**
1. Read Part 17 (the 90-minute playbook) — internalize the time allocation.
2. Skim Part 2 (requirements) — review your 8 default questions.
3. Skim Part 15 (communication) — review the SJTF template.
4. Review the relevant domain chapter (Part 16 for EA game AI).

**During an interview:**
1. Phase 1 (0–10 min): Requirements. Use Part 2 questions.
2. Phase 2 (10–20 min): Architecture. Use Part 1 universal flow.
3. Phase 3 (20–60 min): Implementation. Use Parts 3–9 as needed.
4. Phase 4 (60–75 min): Deployment + monitoring. Use Parts 7, 11.
5. Phase 5 (75–90 min): Tradeoffs + improvements. Use Part 15.

**After an interview:**
- Note which questions caught you off-guard.
- Find the corresponding framework chapter.
- Add a note to that chapter with the specific case.

---

## Author's Note

This handbook is deliberately **opinionated**. It says "use XGBoost for tabular data" rather than "it depends on many factors." 

Opinions require justification. But stated with confidence and backed by reasoning, a strong opinion demonstrates engineering maturity far better than endless hedging.

When the interviewer challenges your choice — and they will — use the SJTF template. State, justify, acknowledge the tradeoff, and offer the alternative. That's the hallmark of a principal-level engineer.

*Good luck. You don't need it — you have a framework.*

---

**Folder structure:**
```
Production_AI_Engineering_Handbook/
├── README.md                           ← This file
├── 01_Universal_Framework.md
├── 02_Requirement_Gathering.md
├── 03_Model_Selection.md
├── 04_Data_Decision_Framework.md
├── 05_Training_Decision_Framework.md
├── 06_Evaluation_Framework.md
├── 07_Deployment_Framework.md
├── 08_RAG_Framework.md
├── 09_Agent_Framework.md
├── 10_System_Design_Framework.md
├── 11_Observability_Framework.md
├── 12_Security_Framework.md
├── 13_Cost_Optimization_Framework.md
├── 14_Failure_Analysis_Framework.md
├── 15_Communication_Framework.md
├── 16_Game_AI_Framework.md
└── 17_Interview_Playbook.md
```
