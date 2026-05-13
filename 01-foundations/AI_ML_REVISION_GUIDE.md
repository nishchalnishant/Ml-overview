# AI & ML revision guide — *the night-before edition*

**For:** Someone who lives in Azure and DevOps, has taste, and does not have patience for textbook fog.  
**Use:** Skim in 20 minutes, deep-read in an hour, screenshot the tables before you walk in.

---

## 0. Azure / DevOps ↔ ML (one mental model)

| You already do… | In ML land… |
|-----------------|-------------|
| **CI pipeline** | Training + eval + tests on *data* and *code* |
| **Artifact / package** | Model weights + config + preprocessor |
| **Staging → prod** | Registry stage → canary / A-B → full traffic |
| **Health probes** | Latency, errors, *and* accuracy / drift |
| **Config + secrets** | Hyperparameters, feature flags, API keys for LLMs |

**Mnemonic:** *Build (train) → sign-off (metrics) → deploy (serve) → monitor (SLIs + model health) → rollback or retrain.*

---

## 1. Golden rules (when the interviewer says “debug this”)

- **Overfitting** — Model memorized the training set (**high variance**). *Fix:* more/better data, L1/L2, dropout, early stopping, or a simpler architecture. *DevOps parallel:* tuning so hot that you’re overfitting your **staging** logs — cool the system.
- **Underfitting** — Model too simple to learn the pattern (**high bias**). *Fix:* more capacity, richer features, less regularization.
- **Class imbalance** — Don’t brag about **accuracy**. Use PR-AUC, F1, or business-weighted cost. Resample, class weights, focal loss.
- **Vanishing gradients** — Signal dies in deep nets. ReLU family, batch norm, **residual** paths, or sequence models (LSTM/Transformer).
- **Exploding gradients** — Clip gradients; check learning rate; stabilize loss.

**Quick thought experiment:** *You’re rolling out a new API version. What metric proves it’s “better” if “errors” are rare but catastrophic?* That’s the imbalance problem in interview clothing.

---

## 2. Architectures — the headline act

| Topic | Mechanism (one breath) | Why interviewers care |
| :--- | :--- | :--- |
| **Transformer** | Self-attention — compare every token to every other (**$O(N^2)$**). | Parallel training + **global** context. |
| **RAG** | Retrieve docs → stuff into prompt → generate. | Cuts hallucinations; updates **facts** without retraining the whole model. |
| **LoRA** | Low-rank matrices **A × B** on top of frozen weights. | Fine-tune huge models without huge GPUs — like **patching** a service without forking the monolith. |
| **RLHF / DPO** | Learn from human (or model) **preferences**. | “Helpful / harmless” isn’t the same as “low loss.” |
| **Agents** | Perceive → plan → **tool** call → loop. | Automation with guardrails — your **orchestration** diagram with an LLM brain. |

**Ghazal hook (NLP / RNN intuition):** In a line of poetry, the word that hits hardest often depends on **what came before** — not the dictionary definition in isolation. Sequence models live for that dependency; attention makes the dependencies explicit and parallelizable.

---

## 3. Math you can scribble on a whiteboard

- **Backprop:** $\frac{\partial L}{\partial w}$ via **chain rule** — gradients flow backward like post-incident blame, but useful.
- **Attention:** $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ — scaling by $\sqrt{d_k}$ keeps dot products from blowing up so softmax doesn’t go flat (**vanishing** updates).
- **Sigmoid derivative:** $\sigma(z)(1-\sigma(z))$ — max **0.25**, so deep sigmoid stacks **starve** the signal (classic “vanishing”).
- **Adam:** Momentum + adaptive scaling + bias correction — **not** exotic, just robust default for many nets.
- **Cross-entropy + softmax:** Gradient simplifies; you’ll hear **“linear error”** intuition in good explanations.

**Mini pop quiz:** *Why is $\sqrt{d_k}$ in the denominator?*  
→ Stabilize variance of dot products as dimension grows (keeps softmax in a trainable range).

---

## 4. System design & infra (where seniors earn the title)

- **3D parallelism:** **DP** (split batch), **TP** (split tensors), **PP** (split layers) — combine when one GPU is a costume, not a wardrobe.
- **Quantization:** FP16 → INT8/INT4 — smaller memory, faster inference; watch accuracy.
- **Inference speed:** KV-cache, speculative decoding, Flash Attention–class kernels — **latency** is a feature.
- **Evals for LLMs:** G-Eval, RAGAS (for RAG), benchmarks (e.g. MMLU-style) — **your** test pyramid for non-deterministic systems.

**Fashion / CV analogy (feature extraction):** A vision model doesn’t “see” an outfit the way you do — it builds a hierarchy: edges → textures → shapes → **composition**. That’s your **feature hierarchy**: detail to silhouette, thread to collection.

---

## 5. Gotchas (they *will* ask)

1. **RAG vs fine-tuning?** RAG for **updating knowledge** and citations; fine-tuning for **behavior**, tone, task format.
2. **Batch Norm vs Layer Norm?** BN: normalize across **batch** dimension (tricky for small / variable batch). LN: normalize across **features** — default in Transformers.
3. **Data leakage via ID?** IDs often carry time or cohort signal — **drop** or encode carefully.
4. **Precision vs recall?** Precision = trust positives; recall = don’t miss positives — pick based on **business** cost.
5. **$\sqrt{d_k}$?** Already your party trick from section 3.

---

## 6. Topics that separate good from great (often missed)

### Reinforcement Learning (RL) one-liner map
- **MDP:** States, actions, rewards, transitions — agent learns a **policy** to maximize cumulative reward.
- **Q-learning:** Learn value of (state, action) pairs via Bellman equation. DQN = Q-learning + neural net + experience replay.
- **Policy gradients:** Directly optimize the policy. PPO = the stable workhorse used in RLHF.
- **RLHF connection:** SFT → reward model from human preferences → PPO to maximize reward → aligned LLM.

### Recommender Systems one-liner map
- **Collaborative filtering:** Users who liked A also liked B — matrix factorization finds latent factors.
- **Two-tower model:** Separate user encoder + item encoder, dot product at inference. YouTube, Pinterest use this.
- **Cold start:** New users/items have no history. Fallback: popularity, content features, or ask explicitly.
- **Eval:** NDCG@K (ranking quality), Precision@K, Recall@K. Not accuracy — order matters.

### Interpretability one-liner map
- **SHAP:** Game-theory-fair attribution of each feature's contribution. TreeSHAP for trees is exact and fast.
- **LIME:** Fit a local linear model around a single prediction. Simpler, less consistent than SHAP.
- **Grad-CAM:** Highlight which image regions drove a CNN's prediction. Gradient of class score w.r.t. feature maps.
- **Interview trap:** Attention ≠ explanation. High attention on a token doesn't mean that token caused the output.

### Causal Inference one-liner map
- **Correlation ≠ causation:** Ice cream sales correlate with drowning — both are caused by hot weather (confounder).
- **A/B test:** Randomization breaks confounding. The gold standard when you can run it.
- **Can't randomize?** DiD (difference-in-differences), propensity matching, or instrumental variables.
- **Peeking problem:** Stopping an A/B test early when it looks significant inflates false positive rate. Pre-register sample size.

### Emerging trends (2024-2025)
- **Test-time scaling (o1/R1 paradigm):** More inference compute → better answers. Chain-of-thought as a search process.
- **Mamba/SSMs:** Linear-time sequence models. Selective state spaces beat Transformers on long sequences.
- **MoE at scale:** Mixtral 8x7B, GPT-4 (rumored). Sparse activation = large parameter count at fraction of compute.
- **Synthetic data:** Phi models showed quality > quantity. Self-play + rejection sampling = better training signal.
- **Long context (1M+ tokens):** Ring attention, YaRN position scaling. “Lost in the middle” is a real problem.

---

## 7. Deep-dive hubs (when you have a quiet evening)

- [LLM fundamentals](llm-interview-notes/llm-fundamentals.md)
- [AI system design](llm-interview-notes/ai-system-design.md)
- [Math derivations](ml-interview-notes/math-derivations.md)
- [MLOps — full notes](mlops.md)
- [Reinforcement Learning](reinforcement-learning/README.md)
- [Recommender Systems](recommender-systems/README.md)
- [Interpretability & XAI](interpretability-and-xai/README.md)
- [Causal Inference & A/B Testing](experimentation-and-causal-inference/README.md)
- [Emerging Trends](emerging-trends/README.md)
- [Graph Neural Networks](graph-neural-networks/README.md)

---

## 8. “Deploy this in Azure Pipelines” — 30-second sketch

**Prompt:** *You have a sklearn model and a weekly data refresh. Outline the pipeline.*

**Skeleton answer:** Trigger (schedule or data landing) → **validate** data (schema / drift checks) → **train** in Azure ML / container job → **evaluate** vs threshold → register in **model registry** → deploy to **AKS** or managed endpoint → **monitor** metrics + drift → alert + **rollback** or retrain. Same story as always: **gates**, **artifacts**, **observability**.

---

> **Senior frame (memorize the rhythm):**  
> *“The direct answer is ___ . The intuition is ___ . In production, the tradeoff is usually ___ .”*

---

**Field-placement analogy (optimization):** Training isn’t one heroic bowl — it’s **adjusting the field** every few balls: learning rate schedules, early stopping, maybe a different optimizer when the pitch (loss landscape) changes. Same patience as a captain who won’t panic after one expensive over — but **will** change the plan when the radar shows drift.
