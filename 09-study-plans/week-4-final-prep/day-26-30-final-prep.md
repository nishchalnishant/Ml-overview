---
module: Study Plans
topic: Week 4 Final Prep
subtopic: Day 26 30 Final Prep
status: unread
tags: [studyplans, ml, week-4-final-prep-day-26-30-fi]
---
# Day 26-30: Final Review & Mental Prep

## Why This Topic Comes Here

The goal of the final five days is consolidation, not new learning. You have covered the full scope of ML engineering: foundations, algorithms, evaluation, system design, and communication. Adding new material now would dilute and interfere with what you have already built. The failure mode of week 4 is studying *more* — reading one more paper, learning one more algorithm — when the actual gap is the ability to retrieve and apply what you already know under pressure. These days are designed to stress-test that retrieval and close the remaining gaps in fluency, not coverage.

---

## Day-by-Day Plan

| Day | Goal | Activity |
|-----|------|----------|
| **26** | LLM / GenAI fluency | Re-read LLM fundamentals + emerging trends interview Qs |
| **27** | Classical ML fluency | Re-read supervised/unsupervised + math derivations |
| **28** | Implementation fluency | Implement attention from scratch, k-means, backprop |
| **29** | System design fluency | 3 × 25-min system design out loud — record and review |
| **30** | Reset | Logistics check, mental reset, light review of your own notes |

---

## 1. The "Cheat Sheet" of Cheat Sheets

**Why you need to know these cold:** Interviews create cognitive load. Under pressure, you cannot derive a formula you have only seen once. These are the formulas and concepts that come up frequently enough that their retrieval should be automatic — freeing your working memory for the actual reasoning.

### Metrics

- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
- AUC-ROC: area under ROC curve (FPR vs TPR). AUC=0.5 → random; not reliable for imbalanced classes.
- PR-AUC: preferred for imbalanced classes; directly reflects precision and recall.
- Log loss: $-\sum y_i \log(\hat{y}_i)$ — measures calibration, not just correctness.

### Regularization

- L1 (Lasso): sparsity, coefficients can go to exactly 0, implicit feature selection.
- L2 (Ridge): shrinkage without sparsity, all weights stay nonzero.
- Dropout: equivalent to approximate ensemble over $2^N$ sub-networks.

### Gradient Descent

- Too high LR: oscillates / diverges. Too low: slow convergence, may get stuck.
- Adam: adaptive per-parameter learning rate via first and second moment estimates ($m_t$ and $v_t$).
- Gradient clipping: prevents exploding gradients in RNNs and Transformers.

### System Design

- Retrieval → Ranking pattern: billions of items → hundreds (cheap model) → ranked list (expensive model).
- Feature store: prevents train-serve skew by sharing a single feature computation layer.
- Shadow mode: run new model alongside existing one, compare outputs without serving new model's results.

---

## 2. Mock Interview Question Bank

**Why mock interviews are necessary, not optional:** Knowing the answer and being able to deliver it clearly under pressure are different skills. The latter requires practice under realistic conditions. Treat the mock questions below as a performance test, not a reading comprehension test.

### Round A — LLM/GenAI (25 min)

1. Explain the attention mechanism. Why scale by $\sqrt{d_k}$?
2. What is KV cache? How does paged attention improve it?
3. Difference between RAG and fine-tuning — when to use each?
4. What is LoRA? Derive the parameter savings formula.
5. Design an LLM-powered code review assistant (system design).

**Key insight for this round:** LLM questions often test whether you understand the *constraint* that motivated the technique. KV cache exists because autoregressive generation is expensive. LoRA exists because full fine-tuning is memory-expensive. Paged attention exists because KV cache fragmented GPU memory. If you know the problem each technique solves, you can reconstruct the technique even if you cannot recall exact details.

### Round B — Classical ML (25 min)

1. Bias-variance tradeoff. How does tree depth affect each?
2. Why does gradient boosting work? What is the residual at each step?
3. Explain PCA. What does the explained variance ratio tell you?
4. How would you handle 99:1 class imbalance in a fraud dataset?
5. You have 10 features and 1000 samples. Logistic regression or random forest — why?

**Key insight for this round:** Each question tests whether you understand the mechanism, not just the label. "Random forest reduces variance by averaging" is incomplete. "Random forest reduces variance because averaging $M$ uncorrelated predictors reduces variance by $1/M$, and random feature selection enforces this decorrelation" is complete.

### Round C — Production / MLOps (25 min)

1. What is concept drift? How do you detect and respond to it?
2. How would you set up CI/CD for a model that retrains weekly?
3. Design a feature store. What are the key components?
4. A model's AUC is 0.92 in staging but 0.81 in production. What happened? Debug it.
5. How would you roll back a model that's performing poorly in production?

**Key insight for this round:** Staging-to-production degradation (question 4) is one of the most common real-world ML problems. The primary causes are: (1) train-serve skew in feature computation, (2) data distribution shift between staging and production, (3) label leakage in the training pipeline that was not present at inference, (4) temporal features behaving differently (staging used historical data, production uses real-time). A good answer systematically rules out each cause.

---

## 2b. LLM-Specific Cheat Sheet (Days 26-27)

These concepts appear in nearly every modern ML interview. Know the mechanism, not just the name.

### Training Pipeline

- **Pretraining**: Next-token prediction on massive text corpora. The model learns a compressed world model and language statistics. No human labels involved.
- **Supervised Fine-Tuning (SFT)**: Train on curated (prompt, ideal response) pairs to teach the desired output format and behavior. Cheap but bounded by annotation quality.
- **RLHF (Reinforcement Learning from Human Feedback)**: Train a reward model on human preference comparisons, then use PPO to fine-tune the LLM to maximize reward. More expensive but can exceed SFT ceiling.
- **DPO (Direct Preference Optimization)**: Bypasses the reward model; directly optimizes the LLM on preference data using a closed-form objective. Simpler than PPO, increasingly preferred.

**Key insight for RLHF vs. DPO:** RLHF requires a separate reward model and is susceptible to reward hacking (the LLM learns to game the reward model rather than genuinely improve). DPO is more stable but loses the ability to separate the reward model from the policy — you cannot adjust the reward function without retraining from the SFT checkpoint.

### Inference Efficiency

- **KV Cache**: During autoregressive generation, the Key and Value matrices for all previous tokens are cached to avoid recomputing them. Reduces inference compute from $O(n^2)$ to $O(n)$ per step for the attended portion.
- **Paged Attention (vLLM)**: KV cache in standard implementations wastes GPU memory due to fragmentation (variable-length sequences). Paged attention uses memory pages (like OS virtual memory) to serve multiple requests efficiently from the same GPU.
- **Speculative Decoding**: A small draft model proposes K tokens in parallel; the large model verifies them in one forward pass. Tokens accepted by the large model are kept; the first rejected token is resampled. Achieves 2-4x speedup with no change in output distribution.
- **Quantization**: Reduce model weights from float32 → float16 → int8 → int4. Each step halves memory; accuracy degrades gracefully but must be validated on your specific task.

### RAG Architecture

```
Query → Query Rewriting (optional) → Embedding
→ Vector DB (approximate nearest neighbor search, e.g., FAISS, Pinecone)
→ Top-k retrieved chunks
→ Optional: Cross-encoder reranker (top-k → top-3)
→ LLM generates answer with retrieved context
→ Citations extracted from context
```

**Failure modes at each stage:**
- Embedding: poor chunking (chunks too long/short), wrong embedding model for domain
- Retrieval: low recall (relevant chunk not retrieved), high latency (no approximate index)
- Reranking: bottleneck for low-latency requirements; skip for p50 latency, enable for p99
- Generation: hallucination when retrieved context is insufficient; model ignores context

**RAG vs. Fine-tuning decision:**
- Use RAG when: knowledge is dynamic, sources need citations, you cannot afford GPU compute for fine-tuning
- Use fine-tuning when: task requires format/style adaptation, knowledge is stable, low-latency inference is required (no retrieval hop), RAG retrieval quality is consistently poor

### Prompt Engineering Patterns

- **Zero-shot**: No examples in the prompt. Relies entirely on pretraining knowledge.
- **Few-shot**: Include 3-8 input/output examples in the prompt. Significantly improves structured output tasks.
- **Chain-of-Thought (CoT)**: Prompt the model to reason step-by-step before giving the final answer. Improves multi-step reasoning. Add "Let's think step by step" or show CoT examples.
- **System prompt**: Sets the model's persona, constraints, and response format. Processed differently from user messages in instruction-tuned models.

---

## 3. Rapid-Fire Answer Templates

**Why templates help:** In an interview, an open-ended question like "walk me through your approach to an ML problem" can be answered in many ways. Having a mental template ensures you don't miss structural elements under pressure.

**"Walk me through your approach to [any ML problem]":**
1. Define success metric — align with business goal, not just ML objective.
2. EDA + baseline — understand data, sanity-check first, set a lower bound.
3. Feature engineering — domain knowledge + automated (mutual information, permutation importance).
4. Model selection — start simple, justify each added complexity.
5. Evaluation — hold-out, cross-val, business metric — in that order.
6. Deployment + monitoring — latency constraints, retraining trigger, rollback plan.

**"What would you do if your model's performance degraded in production?":**
1. Check whether it's the model or infrastructure — latency? error rates? missing predictions?
2. Compare input feature distributions to training baseline — data drift?
3. Check feature distributions individually — which features shifted?
4. Check label quality if ground truth is available.
5. Retrain on recent data; consider whether the concept itself changed (concept drift vs. data drift).

---

## 4. Final Mental Strategy

### The "I Don't Know" Scenario

In an interview, if you don't know a specific algorithm or paper: don't panic and don't fabricate.

**Think from first principles:** "I haven't implemented model X directly, but based on the problem constraints — sequential data, variable length, need for long-range dependencies — I would look for something that addresses the vanishing gradient problem and can operate over arbitrary sequence lengths. That points toward attention mechanisms. Let me reason through what I'd expect X to look like based on that..."

**Why this works:** Interviewers often ask about techniques they don't expect you to know in detail — they want to see how you reason when you hit the edge of your knowledge. Demonstrating principled reasoning from fundamentals is more informative and more valuable to them than having memorized a specific paper.

**What trips people up:** Silence or deflection. Saying "I don't know that algorithm" and waiting is the worst possible response. The second worst is confidently describing something incorrect. The correct response is to immediately begin reasoning from what you do know.

### Clear Communication

- Draw diagrams. Label axes. Show data flow. An interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening to an abstract verbal description.
- Structure every answer: (1) direct statement of your position, (2) intuition or analogy, (3) production tradeoff.
- The interviewer is a collaborator, not an examiner. Asking "Does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance.

---

## 5. Readiness Checklist

- [ ] Can I explain my 2 most important projects in under 3 minutes?
- [ ] Do I have 3 thoughtful questions ready for the interviewer?
- [ ] Have I reviewed the math for the algorithms in my past projects?
- [ ] Can I implement attention, backprop, and k-means from scratch?
- [ ] Have I done at least one timed mock interview (aloud, not in my head)?
- [ ] Is my setup (webcam, mic, internet, dev environment) confirmed working?
- [ ] Have I reviewed the [AI & ML Revision Guide](../../01-foundations/AI_ML_REVISION_GUIDE.md)?

---

30 days of deliberate study. The interview is a conversation between two engineers solving a problem together. Trust the foundations you built.

## Rapid Recall

### Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
- Direct Answer: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
- Why: This matters because it tells you how to reason about precision = tp/(tp+fp), recall = tp/(tp+fn), f1 = 2pr/(p+r).
- Pitfall: Don't answer "Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)

### AUC-ROC
- Direct Answer: area under ROC curve (FPR vs TPR). AUC=0.5 → random; not reliable for imbalanced classes.
- Why: This matters because it tells you how to reason about auc-roc.
- Pitfall: Don't answer "AUC-ROC" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: area under ROC curve (FPR vs TPR). AUC=0.5 → random; not reliable for imbalanced classes.

### PR-AUC
- Direct Answer: preferred for imbalanced classes; directly reflects precision and recall.
- Why: This matters because it tells you how to reason about pr-auc.
- Pitfall: Don't answer "PR-AUC" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: preferred for imbalanced classes; directly reflects precision and recall.

### Log loss: $-\sum y_i \log(\hat{y}_i)$
- Direct Answer: measures calibration, not just correctness.
- Why: This matters because it tells you how to reason about log loss: $-\sum y_i \log(\hat{y}_i)$.
- Pitfall: Don't answer "Log loss: $-\sum y_i \log(\hat{y}_i)$" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: measures calibration, not just correctness.

### L1 (Lasso)
- Direct Answer: sparsity, coefficients can go to exactly 0, implicit feature selection.
- Why: This matters because it tells you how to reason about l1 (lasso).
- Pitfall: Don't answer "L1 (Lasso)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sparsity, coefficients can go to exactly 0, implicit feature selection.

### L2 (Ridge)
- Direct Answer: shrinkage without sparsity, all weights stay nonzero.
- Why: This matters because it tells you how to reason about l2 (ridge).
- Pitfall: Don't answer "L2 (Ridge)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: shrinkage without sparsity, all weights stay nonzero.

### Dropout
- Direct Answer: equivalent to approximate ensemble over $2^N$ sub-networks.
- Why: This matters because it tells you how to reason about dropout.
- Pitfall: Don't answer "Dropout" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: equivalent to approximate ensemble over $2^N$ sub-networks.

### Too high LR
- Direct Answer: oscillates / diverges. Too low: slow convergence, may get stuck.
- Why: This matters because it tells you how to reason about too high lr.
- Pitfall: Don't answer "Too high LR" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: oscillates / diverges. Too low: slow convergence, may get stuck.

### Adam
- Direct Answer: adaptive per-parameter learning rate via first and second moment estimates ($m_t$ and $v_t$).
- Why: This matters because it tells you how to reason about adam.
- Pitfall: Don't answer "Adam" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: adaptive per-parameter learning rate via first and second moment estimates ($m_t$ and $v_t$).

### Gradient clipping
- Direct Answer: prevents exploding gradients in RNNs and Transformers.
- Why: This matters because it tells you how to reason about gradient clipping.
- Pitfall: Don't answer "Gradient clipping" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prevents exploding gradients in RNNs and Transformers.

### Retrieval → Ranking pattern
- Direct Answer: billions of items → hundreds (cheap model) → ranked list (expensive model).
- Why: This matters because it tells you how to reason about retrieval → ranking pattern.
- Pitfall: Don't answer "Retrieval → Ranking pattern" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: billions of items → hundreds (cheap model) → ranked list (expensive model).

### Feature store
- Direct Answer: prevents train-serve skew by sharing a single feature computation layer.
- Why: This matters because it tells you how to reason about feature store.
- Pitfall: Don't answer "Feature store" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prevents train-serve skew by sharing a single feature computation layer.

### Shadow mode
- Direct Answer: run new model alongside existing one, compare outputs without serving new model's results.
- Why: This matters because it tells you how to reason about shadow mode.
- Pitfall: Don't answer "Shadow mode" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: run new model alongside existing one, compare outputs without serving new model's results.

### Draw diagrams. Label axes. Show data flow. An interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening to an abstract verbal description.
- Direct Answer: Draw diagrams. Label axes. Show data flow. An interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening to an abstract verbal description.
- Why: This matters because it tells you how to reason about draw diagrams. label axes. show data flow. an interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening to an abstract verbal description..
- Pitfall: Don't answer "Draw diagrams. Label axes. Show data flow. An interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening to an abstract verbal description." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Draw diagrams. Label axes. Show data flow. An interviewer watching you sketch a two-tower architecture while narrating it can assess your understanding far better than listening t…

### Structure every answer
- Direct Answer: (1) direct statement of your position, (2) intuition or analogy, (3) production tradeoff.
- Why: This matters because it tells you how to reason about structure every answer.
- Pitfall: Don't answer "Structure every answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: (1) direct statement of your position, (2) intuition or analogy, (3) production tradeoff.

### The interviewer is a collaborator, not an examiner. Asking "Does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance.
- Direct Answer: The interviewer is a collaborator, not an examiner. Asking "Does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance.
- Why: This matters because it tells you how to reason about the interviewer is a collaborator, not an examiner. asking "does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance..
- Pitfall: Don't answer "The interviewer is a collaborator, not an examiner. Asking "Does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: The interviewer is a collaborator, not an examiner. Asking "Does that constraint match what you had in mind?" is a sign of good engineering judgment, not ignorance.

### [ ] Can I explain my 2 most important projects in under 3 minutes
- Direct Answer: [ ] Can I explain my 2 most important projects in under 3 minutes?
- Why: This matters because it tells you how to reason about [ ] can i explain my 2 most important projects in under 3 minutes.
- Pitfall: Don't answer "[ ] Can I explain my 2 most important projects in under 3 minutes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Can I explain my 2 most important projects in under 3 minutes?

### [ ] Do I have 3 thoughtful questions ready for the interviewer
- Direct Answer: [ ] Do I have 3 thoughtful questions ready for the interviewer?
- Why: This matters because it tells you how to reason about [ ] do i have 3 thoughtful questions ready for the interviewer.
- Pitfall: Don't answer "[ ] Do I have 3 thoughtful questions ready for the interviewer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Do I have 3 thoughtful questions ready for the interviewer?

### [ ] Have I reviewed the math for the algorithms in my past projects
- Direct Answer: [ ] Have I reviewed the math for the algorithms in my past projects?
- Why: This matters because it tells you how to reason about [ ] have i reviewed the math for the algorithms in my past projects.
- Pitfall: Don't answer "[ ] Have I reviewed the math for the algorithms in my past projects" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Have I reviewed the math for the algorithms in my past projects?

### [ ] Can I implement attention, backprop, and k-means from scratch
- Direct Answer: [ ] Can I implement attention, backprop, and k-means from scratch?
- Why: This matters because it tells you how to reason about [ ] can i implement attention, backprop, and k-means from scratch.
- Pitfall: Don't answer "[ ] Can I implement attention, backprop, and k-means from scratch" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Can I implement attention, backprop, and k-means from scratch?

### [ ] Have I done at least one timed mock interview (aloud, not in my head)
- Direct Answer: [ ] Have I done at least one timed mock interview (aloud, not in my head)?
- Why: This matters because it tells you how to reason about [ ] have i done at least one timed mock interview (aloud, not in my head).
- Pitfall: Don't answer "[ ] Have I done at least one timed mock interview (aloud, not in my head)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Have I done at least one timed mock interview (aloud, not in my head)?

### [ ] Is my setup (webcam, mic, internet, dev environment) confirmed working
- Direct Answer: [ ] Is my setup (webcam, mic, internet, dev environment) confirmed working?
- Why: This matters because it tells you how to reason about [ ] is my setup (webcam, mic, internet, dev environment) confirmed working.
- Pitfall: Don't answer "[ ] Is my setup (webcam, mic, internet, dev environment) confirmed working" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Is my setup (webcam, mic, internet, dev environment) confirmed working?

### [ ] Have I reviewed the [AI & ML Revision Guide](../../01-foundations/AI_ML_REVISION_GUIDE.md)
- Direct Answer: [ ] Have I reviewed the [AI & ML Revision Guide](../../01-foundations/AI_ML_REVISION_GUIDE.md)?
- Why: This matters because it tells you how to reason about [ ] have i reviewed the [ai & ml revision guide](../../01-foundations/ai_ml_revision_guide.md).
- Pitfall: Don't answer "[ ] Have I reviewed the [AI & ML Revision Guide](../../01-foundations/AI_ML_REVISION_GUIDE.md)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: [ ] Have I reviewed the [AI & ML Revision Guide](../../01-foundations/AI_ML_REVISION_GUIDE.md)?

### RLHF training stages
- Direct Answer: Pretraining → SFT → Train reward model on human preferences → PPO to maximize reward. DPO skips the reward model and optimizes preferences directly.
- Why: This matters because it tells you how to reason about rlhf training stages.
- Pitfall: Don't answer "RLHF training stages" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Pretraining → SFT → Train reward model on human preferences → PPO to maximize reward. DPO skips the reward model and optimizes preferences directly.

### KV Cache
- Direct Answer: Caches Key and Value matrices for all previous tokens during autoregressive generation. Avoids recomputing O(n) attention for each new token. Essential for inference speed.
- Why: This matters because it tells you how to reason about kv cache.
- Pitfall: Don't answer "KV Cache" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Caches Key and Value matrices for all previous tokens during autoregressive generation. Avoids recomputing O(n) attention for each new token. Essential for inference speed.

### Speculative Decoding
- Direct Answer: Small draft model proposes K tokens; large model verifies in one forward pass. Accepts matching tokens, resamples first rejection. ~2-4x speedup, no change in output distribution.
- Why: This matters because it tells you how to reason about speculative decoding.
- Pitfall: Don't answer "Speculative Decoding" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Small draft model proposes K tokens; large model verifies in one forward pass. Accepts matching tokens, resamples first rejection. ~2-4x speedup, no change in output distribution.

### RAG vs. Fine-tuning — when to use RAG
- Direct Answer: When knowledge is dynamic (needs update without retraining), sources need citations, or GPU compute for fine-tuning is unavailable.
- Why: This matters because it tells you how to reason about rag vs. fine-tuning — when to use rag.
- Pitfall: Don't answer "RAG vs. Fine-tuning — when to use RAG" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: When knowledge is dynamic (needs update without retraining), sources need citations, or GPU compute for fine-tuning is unavailable.

### Chain-of-Thought prompting
- Direct Answer: Prompt the model to reason step-by-step before the final answer. Improves multi-step and arithmetic reasoning. Triggered by "Let's think step by step" or few-shot CoT examples.
- Why: This matters because it tells you how to reason about chain-of-thought prompting.
- Pitfall: Don't answer "Chain-of-Thought prompting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Prompt the model to reason step-by-step before the final answer. Improves multi-step and arithmetic reasoning. Triggered by "Let's think step by step" or few-shot CoT examples.

### DPO vs. RLHF
- Direct Answer: DPO directly optimizes LLM on preference data using a closed-form objective — no separate reward model. Simpler and more stable than PPO-based RLHF but cannot adjust the reward function independently.
- Why: This matters because it tells you how to reason about dpo vs. rlhf.
- Pitfall: Don't answer "DPO vs. RLHF" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: DPO directly optimizes LLM on preference data using a closed-form objective — no separate reward model. Simpler and more stable than PPO-based RLHF but cannot adjust the reward fu…
