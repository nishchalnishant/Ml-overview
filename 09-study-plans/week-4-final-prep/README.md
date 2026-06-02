# Week 4 (Days 26-30): Final Prep

**Goal:** Sharpen everything. This week is not for learning new material — it is for rapid recall, plugging gaps, and simulating real interview conditions. LLM-specific topics get dedicated time here because they appear in almost every modern ML interview.

---

## What This Week Covers

| Days   | Topic                              | Key Concepts                                                          |
|--------|------------------------------------|-----------------------------------------------------------------------|
| 26-27  | LLM-Specific Questions             | Tokenization, RLHF, RAG, prompt engineering, fine-tuning vs. prompting|
| 28     | Rapid-Fire Q&A Review              | Hit every major topic from weeks 1-3 in compressed format             |
| 29     | Mock Interview Scenarios           | Full end-to-end practice with timed responses                         |
| 30     | Night-Before Sprint                | Cheat sheet review only — no new concepts                             |

---

## Focus Areas

- **LLM fundamentals:** Understand the full training pipeline (pretraining → SFT → RLHF). Know what PPO and reward modeling are at a conceptual level.
- **RAG:** Be able to design a retrieval-augmented generation system from scratch, including chunking strategy, embedding choice, and re-ranking.
- **Prompt engineering:** Few-shot, chain-of-thought, and system prompt design — know when each helps and why.
- **Rapid recall:** Use the revision guide to run flashcard-style review. If a topic takes more than 30 seconds to recall, flag it and drill it.
- **Mock interviews:** Simulate the time pressure. Speak your reasoning out loud — interviewers evaluate process, not just correctness.

---

## Daily Study Pattern

Days 26-27: One hour of focused topic review (LLM-specific + classical ML), then one hour of written Q&A — write out full answers, not bullet fragments.
Day 28: Implementation sprint — code attention, backprop, and k-means from scratch, timed.
Day 29: Full mock interview — three 25-min rounds (LLM/GenAI, Classical ML, Production/MLOps), spoken aloud or with a partner.
Day 30: Revision guide only. No new material. Logistics check. Sleep.

---

## Linked Resources

- [LLM Fundamentals](../../05-llms/interview-notes/llm-fundamentals.md)
- [RAG](../../05-llms/interview-notes/retrieval-augmented-generation-rag.md)
- [Top ML Interview Questions (rapid-fire bank)](../../07-interview-prep/llm/top-ml-interview-questions.md)
- [AI & ML Revision Guide (night-before cheat sheet)](../../01-foundations/AI_ML_REVISION_GUIDE.md)
- [Behavioral & Scenario-Based Questions](../../07-interview-prep/ml/behavioral-and-scenario-based-questions.md)
- Day file in this folder: day-26-30

---

## Day 28 Implementation Targets

These are the implementations that come up most frequently in coding rounds of ML interviews. Time yourself: each should take under 45 minutes from a blank file.

**1. Scaled Dot-Product Attention (NumPy only)**
```python
# Input: Q, K, V matrices (shape: seq_len x d_k)
# Output: Attention output (shape: seq_len x d_v)
# Must include: softmax over (Q @ K.T / sqrt(d_k)), then multiply by V
# Must handle: masking for causal attention (set upper triangle to -inf before softmax)
```

**2. Backpropagation (2-layer MLP, NumPy only)**
```python
# Network: Linear -> ReLU -> Linear -> Sigmoid -> Binary Cross-Entropy loss
# Must compute: dL/dW2, dL/db2, dL/dW1, dL/db1 using chain rule
# Verify: check against numerical gradient (finite differences) — they should match to 1e-5
```

**3. K-Means (NumPy only)**
```python
# Input: X (n_samples x n_features), k (number of clusters), max_iter
# Must implement: random centroid init, assignment step, update step, convergence check
# Verify: on sklearn's make_blobs(n_samples=500, centers=3), inertia should stabilize within 20 iterations
```

---

## Milestone Checkpoints

**After Day 27 (LLM review):** Can you explain the full RLHF training pipeline — pretraining, SFT, reward modeling, PPO — without notes? Can you compare RAG vs. fine-tuning and give concrete criteria for choosing one over the other?

**After Day 28 (implementations):** Did your attention implementation match the PyTorch output on the same inputs? Did your backprop pass the gradient check? These are the implementations — if they fail, debug them. The point is not to write them once but to understand them well enough to debug them under pressure.

**After Day 29 (mock):** Play back your recording (or ask your partner) — did you name the tradeoff explicitly in every system design answer? Did every behavioral answer have a quantified result? Were your technical answers under 90 seconds for factual questions?

---

## End-of-Week Check

- Can you explain RLHF and why it is used instead of pure supervised fine-tuning?
- Can you design a RAG pipeline and explain the failure modes at each stage (retrieval, reranking, generation)?
- Can you answer any question from weeks 1-3 in under 90 seconds without notes?
- Can you implement attention, backprop, and k-means from scratch without referencing documentation?
- Are your logistics confirmed (camera, microphone, dev environment, quiet space)?
