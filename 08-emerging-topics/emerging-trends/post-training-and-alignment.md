---
module: Emerging Topics
topic: Emerging Trends
subtopic: Post Training And Alignment
status: unread
tags: [emergingtopics, ml, emerging-trends-post-training-]
---
# Post-Training and Alignment

> *Snapshot: June 2026 — frontier topic, moves fast. Treat as a current-state map, not settled canon.*

The algorithms and training paradigms that transform a pretrained language model into a useful, safe, and instruction-following system — with exact mathematical differences between RLHF, DPO, KTO, and the synthetic data approaches that reduce dependence on human annotation.

---

## 1. Core Concept & Intuition

A pretrained language model is a high-quality text distribution estimator — it models `P(next_token | context)` over the internet's text distribution. This distribution includes toxic content, misinformation, unhelpful responses, and failure to follow instructions, because the internet contains all of these.

Post-training addresses the fundamental mismatch between:
- **What pretraining optimizes:** predict the next token over a broad text corpus
- **What deployment requires:** follow instructions accurately, be helpful, avoid harm, maintain factual accuracy

The progression of techniques:

```
RLHF (2022) → DPO (2023) → KTO (2023) → GRPO + Process RL (2024)
         ↑                                              ↑
  Requires learned reward model                Uses verifiable signals / LLM-as-judge
  Two-stage, complex                           Single-stage, simpler
```

---

## 2. Architecture & Mathematics

### 2.1 Supervised Fine-Tuning (SFT)

Before any RL-based alignment, SFT trains the model to follow the instruction-response format:

```
Dataset: {(instruction_i, response_i)} — high-quality human demonstrations

L_SFT = -Σ_{t ∈ response} log P_θ(x_t | instruction, x_{<t})

Note: loss is computed only on response tokens, not instruction tokens
```

SFT is sufficient for instruction following but insufficient for alignment — the model learns the format but not the preference ordering between good and bad responses. If the training data contains mediocre responses (inevitable at scale), the model learns them too.

**SFT quality bottleneck:** SFT quality is bounded by demonstration quality. Collecting 10K high-quality expert demonstrations is expensive; scaling to 1M requires quality to drop. This is why RL-based methods exist — they can use preference comparisons (easier to elicit) rather than demonstrations (harder to elicit).

### 2.2 RLHF: Full Pipeline

**Step 1: SFT** (above)

**Step 2: Reward Model Training**

Collect preference data: for each prompt, show human annotators two model responses (y_w, y_l) and ask which is better. The preference dataset: `D = {(x, y_w, y_l)}` where y_w is preferred over y_l.

Train a scalar reward model using the Bradley-Terry model of pairwise preferences:

```
P(y_w preferred over y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))
where σ is the sigmoid function

L_RM = -E_{(x,y_w,y_l)~D} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

The reward model takes a prompt-completion pair and outputs a scalar. Typically initialized from the SFT model with the language model head replaced by a scalar regression head.

**Step 3: PPO (Proximal Policy Optimization)**

Optimize the language model (policy π_θ) to maximize expected reward while staying close to the SFT reference policy (prevents reward hacking):

```
L_PPO = E_{x~D, y~π_θ(·|x)} [r_φ(x,y) - β·KL(π_θ(·|x) || π_ref(·|x))]

The KL penalty in practice:
  KL(π_θ || π_ref) = Σ_t π_θ(y_t|x,y_{<t}) · log(π_θ(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t}))
  
Approximated token-by-token:
  KL_t ≈ log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})
```

β controls the strength of the KL penalty. β too large: policy stays near SFT, reward improvement minimal. β too small: policy diverges wildly ("reward hacking" — discovers outputs that fool the reward model but are terrible).

**PPO update rule:**

```
ratio_t = π_θ(y_t|x,y_{<t}) / π_θ_old(y_t|x,y_{<t})

L_clip = E_t [min(ratio_t · A_t, clip(ratio_t, 1-ε, 1+ε) · A_t)]
where A_t = advantage (reward signal, estimated via GAE)
ε = 0.2 (clipping range)
```

The clipping prevents any single gradient step from making too large a policy change.

**RLHF complexity:**

```
Models maintained simultaneously during PPO:
  1. Policy π_θ (the model being trained) — full model in memory
  2. Reference policy π_ref (frozen SFT model) — full model in memory for KL computation
  3. Reward model r_φ — another model for scoring
  4. Value model V_ψ (PPO's critic) — yet another model

For a 70B model: 4 × 70B × 2 bytes ≈ 560 GB GPU memory minimum
This requires a complex orchestration system (e.g., TRL, DeepSpeed-Chat, OpenRLHF)
```

### 2.3 DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) shows that the RLHF objective can be solved directly without training a reward model or running RL.

**Mathematical derivation:**

The optimal policy under the RLHF objective has a closed-form solution:

```
π*(y|x) = π_ref(y|x) · exp(r*(x,y)/β) / Z(x)
where Z(x) = Σ_y π_ref(y|x) · exp(r*(x,y)/β)  (partition function)

Solving for r*:
  r*(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

Substituting into the Bradley-Terry preference model:

```
P(y_w ≻ y_l | x) = σ(r*(x,y_w) - r*(x,y_l))
                  = σ(β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x)))
```

The partition function Z(x) cancels (appears in both terms identically). Now we can write the DPO loss directly in terms of the policy:

```
L_DPO = -E_{(x,y_w,y_l)~D} [log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**DPO vs RLHF — what this achieves:**
- No reward model training required
- No PPO optimization loop required
- Single training run on the preference dataset
- Memory: only π_θ (the policy) and π_ref (frozen) — 2× instead of 4× model copies

**DPO gradient intuition:**

```
∂L_DPO/∂θ ∝ -σ(β·log(...)) · β · [∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x)]
              ↑                      ↑                          ↑
   downweights easy examples    increase prob of y_w       decrease prob of y_l
```

When the model already strongly prefers y_w (the numerator σ(...) → 0), the gradient is small — the model doesn't waste updates on examples it's already correct on.

**DPO failure mode — distribution shift:**

DPO optimizes directly on the pre-collected preference dataset. If the model being trained (π_θ) generates responses that are very different from the responses in the dataset (y_w, y_l), the log-likelihood ratios become unreliable. This is the "out-of-distribution" problem: the model may learn to strongly prefer y_w over y_l but still generate poor responses that are different from both.

RLHF avoids this by re-generating responses online during training (the policy generates fresh responses, reward model scores them). DPO is "offline" — it uses a fixed dataset.

**IPO (Identity Preference Optimization):** fixes DPO's overconfidence by adding a regularization term:

```
L_IPO = E[(h(x,y_w,y_l) - 1/(2β))²]
where h = log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))
```

This targets a specific preference gap of 1/(2β) rather than maximizing it unboundedly.

### 2.4 KTO: Kahneman-Tversky Optimization

KTO (Ethayarajh et al., 2023) is motivated by prospect theory from behavioral economics — humans evaluate outcomes as gains/losses relative to a reference point, with losses weighing more heavily than gains.

**Key difference from DPO:** KTO does NOT require paired (y_w, y_l) preferences. It only requires individual examples labeled as "good" or "bad":

```
Dataset: {(x, y, label)} where label ∈ {desirable, undesirable}
```

This is a much easier annotation task — annotators rate individual responses rather than comparing two responses.

**KTO loss:**

```
For each (x, y):
  z(x,y) = β · (log π_θ(y|x) - log π_ref(y|x))  # log-ratio, similar to DPO
  
  KL term: KL_hat = KL(π_θ(y'|x') || π_ref(y'|x'))  # estimated on a random (x',y') from batch

If y is desirable:   L = 1 - σ(z(x,y) - KL_hat)    # maximize for desirable outputs
If y is undesirable: L = 1 - σ(KL_hat - z(x,y))    # minimize for undesirable outputs

Total loss: L_KTO = E[λ_D · L_desirable + λ_U · L_undesirable]
where λ_D, λ_U are weights (KTO uses λ_U > λ_D to implement loss aversion)
```

**When KTO beats DPO:**
- Your dataset doesn't have paired comparisons (common in practice — you have user thumbs up/down on individual responses)
- The good/bad examples are imbalanced (KTO handles this naturally; DPO needs equal numbers of y_w and y_l pairs)
- Empirically: KTO matches or exceeds DPO on alignment benchmarks while requiring simpler annotation

### 2.5 Constitutional AI (Anthropic)

Constitutional AI (Bai et al., 2022) reduces dependence on human feedback by having the model critique its own outputs against a written constitution of principles.

**RLAIF (RL from AI Feedback) pipeline:**

```
Stage 1: Supervised Learning from AI Feedback (SL-CAI)
  1. Sample potentially harmful responses from SFT model
  2. Show the model its own response + a constitution principle:
     "This response may encourage harmful behavior. Revise it to be helpful 
      while avoiding harm, according to the principle: [principle]"
  3. The model generates a revised response
  4. Fine-tune on (original prompt, revised response) pairs
     L = -Σ log P_θ(revised_response_t | prompt, original_response, principle, ...)

Stage 2: RL from AI Feedback (RLHF-CAI)  
  1. Generate pairs (y_1, y_2) for each prompt from the current model
  2. Ask a "feedback model" (typically Claude itself) to compare responses:
     "Which response better follows the principle: [principle]?"
  3. Use these AI-generated preferences to train a preference model
  4. Run PPO/DPO against this AI-generated preference model
```

**The constitution (example principles):**
```
- Choose the response that is most helpful, harmless, and honest
- Prefer responses that are least likely to encourage illegal activity
- Prefer responses that would make a reasonable person feel respected, not harassed
- Choose the most helpful response if the request is clearly benign
```

**Why this works at scale:** Human preference annotation bottlenecks at ~1 label/minute per annotator. An AI can generate 1000 preference labels/second. Constitutional AI allows scaling alignment data generation without proportional human cost. The constitution externalizes the values — any stakeholder can review and modify it, making the alignment process more transparent.

### 2.6 Synthetic Data Generation and Self-Instruct

**Self-Instruct (Wang et al., 2022):**

Bootstrap instruction-following datasets from the model's own generations:

```
Seed: 175 human-written instruction-response pairs

Loop:
  1. Sample 8 seed examples
  2. Generate new instruction: "Generate a new instruction that is different 
     from the above 8 instructions. It should be varied in task type and topic."
  3. Filter: remove near-duplicates (ROUGE-L > 0.7 with any existing instruction)
  4. Classify: is this instruction "classification" or "generation" type?
  5. Generate input (if applicable) and output for the instruction
  6. Add to pool → repeat

Generate 52K instructions from 175 seeds using GPT-3, fine-tune another model on them
```

**LLM-as-Judge (scaling quality evaluation):**

Rather than human rating, use a frontier model to score responses:

```python
judge_prompt = f"""
Rate the following response to the instruction on a scale of 1-5:
Instruction: {instruction}
Response: {response}

Criteria:
1. Instruction following (1-5)
2. Factual accuracy (1-5)  
3. Helpfulness (1-5)
4. Safety (1-5)

Provide a score and brief justification for each criterion.
"""
score = gpt4(judge_prompt)
```

LLM-as-judge achieves ~80% agreement with human annotators on ranking tasks (comparable to inter-annotator agreement), making it viable for automated quality filtering at scale.

**Magpie / Evol-Instruct / Orca techniques:**

```
Evol-Instruct: start with simple instruction → iteratively "evolve" to harder versions
  Original: "Write a Python function to sort a list"
  Evolved:  "Write a Python function to sort a list of dictionaries by multiple keys,
             handling ties stably and supporting both ascending and descending order
             per key, with O(n log n) complexity"
  
  Evolution operators: add constraints, increase complexity, increase specificity,
                       make it more concrete, add reasoning steps required

Magpie: generate instructions by asking the model directly:
  Prompt: "User: " → model generates a realistic user query from this prefix
  (Exploits the model's learned understanding of what users ask)
  
  High quality because the model generates instructions that match the distribution
  of real user queries (which it has seen during pretraining)
```

**Decontamination:** synthetic data must be decontaminated against evaluation benchmarks. If training data contains instances similar to MMLU or GSM8K questions, benchmark results are inflated. Use MinHash/SimHash to detect near-duplicates between training data and benchmark splits.

### 2.7 Reward Hacking and Its Mitigations

Reward hacking: the model learns to maximize the reward model's score without actually producing better outputs. Common failure modes:

```
1. Length hacking: longer responses score higher on many reward models
   → mitigation: normalize reward by response length; add explicit length penalty

2. Sycophancy: model learns that agreeing with users gets higher scores
   → mitigation: include "deception resistance" in constitution; evaluate on 
                 cases where correct answer contradicts user's assumption

3. Format hacking: bullet points / headers score higher regardless of content
   → mitigation: diverse evaluation set across formats

4. Reward model overoptimization: as KL from reference increases, 
   the reward model becomes unreliable (Goodhart's law)
   → mitigation: early stopping on a held-out human eval set;
                 scheduled KL penalty increase during training
```

**Overoptimization curve:**

```
Reward (human) as a function of reward model optimization:
  Low optimization: human reward increases with RM reward ✓
  Medium optimization: human reward plateaus
  High optimization: human reward decreases while RM reward continues up ✗
                     (Goodhart's Law: the measure is no longer the target)

The optimal RM score ≠ maximum RM score
```

---

## 3. Trade-offs & System Design Implications

### Algorithm Comparison

| Aspect | RLHF+PPO | DPO | KTO | Constitutional AI |
|---|---|---|---|---|
| Data required | Pairwise preferences | Pairwise preferences | Individual labels | Principle set only |
| Human annotation | High | High | Medium | Low |
| Training complexity | Very high (4 models) | Low (2 models) | Low (2 models) | Medium |
| Online vs offline | Online (fresh samples) | Offline (fixed dataset) | Offline | Mixed |
| Risk of reward hacking | High | Low | Low | Medium |
| Quality ceiling | Highest | Slightly lower | Comparable | Depends on constitution |
| Adoption | OpenAI, Anthropic | Mistral, Llama fine-tunes | HuggingFace ecosystem | Anthropic |

### When to Use Each

**RLHF+PPO:** when quality ceiling matters most and you have the infrastructure to run it (4 model copies simultaneously, complex orchestration). Used at frontier labs for flagship models.

**DPO:** when you have a well-curated preference dataset and want to iterate quickly. Good default for teams without PPO infrastructure. Watch for distribution shift if dataset is old.

**KTO:** when preference pairs are unavailable and you only have per-response quality labels (e.g., user thumbs up/down, 5-star ratings). Surprisingly competitive with DPO.

**Constitutional AI / RLAIF:** when human annotation is the bottleneck, you have access to a strong teacher model, and you want transparent, auditable alignment values.

### Hyperparameter Sensitivity

```
RLHF:
  β (KL penalty): 0.01 too small → reward hacking; 0.5 too large → no learning
  Typical: β = 0.1-0.2, adjusted based on KL monitoring during training

DPO:
  β controls preference sharpness (same role as in RLHF)
  Typical: β = 0.1-0.5; lower β → sharper preferences
  
KTO:
  λ_D / λ_U ratio: KTO authors recommend λ_U/λ_D = 1/0.5 (loss aversion)
  If undesirable >> desirable examples: increase λ_D to rebalance
```

---

## 4. Canonical Interview Q&As

**Q1: Derive why DPO eliminates the need for an explicit reward model. What mathematical property makes this possible?**

The RLHF objective optimizes:
```
max_{π_θ} E_{y~π_θ}[r(x,y)] - β·KL(π_θ||π_ref)
```

The optimal solution to this constrained optimization is:
```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
```
which can be verified by writing the Lagrangian and taking the functional derivative with respect to π. Rearranging:
```
r*(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
```

The key insight: the optimal reward r* is uniquely determined by the optimal policy π* and the reference policy π_ref. The partition function Z(x) depends only on x, not y.

When we substitute r* into the Bradley-Terry pairwise preference model:
```
P(y_w ≻ y_l) = σ(r*(y_w) - r*(y_l)) = σ(β·log(π*(y_w)/π_ref(y_w)) - β·log(π*(y_l)/π_ref(y_l)))
```

Z(x) appears in both terms and cancels. We can now directly parameterize π* with the policy parameters θ and optimize using the preference data, bypassing the reward model entirely. The mathematical property that makes this possible is that the optimal RLHF policy has a closed form that expresses the reward as a log-ratio of policy probabilities — and the intractable partition function cancels in the pairwise comparison.

**Q2: What is reward hacking, why does it occur specifically in RLHF, and what does the overoptimization curve look like?**

Reward hacking occurs when the proxy reward model r_φ diverges from the true human preference function r* that it was trained to estimate. The reward model is trained on a finite dataset of human comparisons — it accurately models human preferences within the distribution of responses seen during reward model training, but extrapolates poorly to out-of-distribution responses generated by the policy during PPO.

As PPO trains, the policy π_θ drifts from π_ref (the SFT model on which reward model training data was generated). Responses generated by the drifted policy are increasingly out-of-distribution for the reward model. The reward model's predictions become overconfident extrapolations — the policy finds specific response patterns (length, format, surface-level features) that score high on the reward model but don't actually reflect human preference.

The overoptimization curve shows: human preference quality first increases with RM optimization (the reward model is a reliable proxy), then plateaus, then decreases (Goodhart's Law — optimizing the measure destroys the measure). The optimal stopping point is somewhere before the reward model score peaks. Practically: monitor human eval alongside RM score during training and stop when human eval stops improving, not when RM score stops improving. The KL penalty (β parameter) delays the onset of overoptimization by keeping the policy close to the SFT model, but doesn't eliminate the problem — it just moves the overoptimization cliff further out.

**Q3: Compare RLHF and DPO on the question of distribution shift. Which is more robust, why, and what techniques address DPO's weakness?**

RLHF with PPO is inherently online: during training, the current policy π_θ generates new responses, the reward model scores them, and PPO updates the policy. The training data is always freshly generated from the current policy distribution — there's no distribution shift because the "data distribution" is the current policy.

DPO is offline: it trains on a fixed preference dataset (y_w, y_l) collected before training begins. As θ is updated, the policy's distribution shifts, but the training data doesn't — the (y_w, y_l) pairs become increasingly unrepresentative of what the current policy generates. The log-ratio log(π_θ(y_w)/π_ref(y_w)) measures the likelihood under the current policy, but if π_θ is very different from the policy that generated y_w and y_l, these log-ratios may be poorly calibrated.

Empirically: DPO suffers more when training for many epochs on a small preference dataset (the policy diverges significantly from the initial state) and less when the preference dataset is large and diverse (policy stays near the data distribution).

Mitigations: (1) **Iterative DPO / Online DPO** — run one DPO epoch, use the new policy to generate fresh (y_w, y_l) pairs using the current policy + reward model, run another DPO epoch. Closes the distribution shift gap. (2) **β annealing** — start with large β (keep policy near reference), gradually reduce β as training stabilizes. Prevents large early distribution shifts. (3) **SFT regularization** — add the SFT loss as a regularization term: L_total = L_DPO + λ·L_SFT. Prevents the policy from forgetting to generate responses similar to those in the dataset. (4) **RSO (Rejection Sampling Optimization)** — collect preference data specifically from the current policy (rejection sampling from the reward model), then apply DPO. Equivalent to online RLHF but using DPO's simpler optimizer.

**Q4: Design a post-training pipeline for a coding assistant model. Which alignment techniques would you use, and how would you evaluate alignment quality?**

**Pipeline:**

Stage 1: SFT on code instruction data. Collect ~100K high-quality (instruction, solution) pairs: competition problems (LeetCode, Codeforces), code review tasks (PR description → fixed code), debugging tasks (buggy code + error → fixed code), documentation (code → docstring). Sources: GitHub, CodeContests, curated Stack Overflow answers. Train for 1-3 epochs with learning rate 1e-5.

Stage 2: Preference data collection for coding quality. For each prompt, sample 4-8 responses from the SFT model. Score each using: (a) unit test pass rate (objective, verifiable); (b) LLM judge on code quality (readability, efficiency, comments); (c) human expert preference on a subset. This gives a rich preference signal without pure human annotation. Generate ~50K preference pairs.

Stage 3: DPO on preference data. Use β=0.1-0.2. Training on unit-test-verified pairs first (y_w = passes all tests, y_l = fails tests) is especially effective because the preference signal is clean and unambiguous. Then layer in quality preferences.

Stage 4: Iterative refinement. Generate responses from the DPO model, re-score, update preference data, repeat 2-3 times. Each iteration reduces distribution shift.

**Evaluation:**
- **Functional correctness:** HumanEval (164 problems), MBPP (374 problems), SWE-bench (real GitHub issues). Primary metric: pass@k.
- **Code quality:** cyclomatic complexity, docstring coverage, style conformance (pylint score) — automated but correlated with human preference.
- **Safety:** does the model generate code with obvious security vulnerabilities (SQL injection, eval() on user input)? Red team evaluation with a set of security-focused prompts.
- **Alignment stability:** does the model refuse unreasonable requests (write malware) while helping with reasonable ones (write a network scanner for pentesting with explicit authorization)?

**Q5: What is the relationship between Constitutional AI, RLAIF, and scalable oversight? Why are these approaches increasingly important as models approach frontier capabilities?**

Constitutional AI is a specific instance of a broader strategy called scalable oversight: using AI systems to help humans supervise AI systems at scales where direct human evaluation becomes impossible.

The scalable oversight problem: as models become more capable, their outputs become harder for humans to evaluate directly. A human can verify whether a 10-line Python function is correct; they cannot easily verify a 10,000-line codebase generated by an AI, or assess the correctness of a long mathematical proof, or evaluate the soundness of a complex legal argument. At frontier capability, the supervisor (human) may be less capable than the supervisee (AI) on many tasks. Direct human feedback becomes unreliable.

Constitutional AI addresses this by having the model itself generate preference labels according to explicit principles (the constitution). A stronger model can evaluate a weaker model's outputs accurately (even if the human cannot), and the human retains oversight at the level of the constitution — they specify the values, not every individual preference judgment.

Scalable oversight approaches beyond Constitutional AI:
- **Debate:** two AI models argue for their respective answers; humans judge the debate (which is easier to evaluate than the underlying question). A correct debater can expose flaws in an incorrect debater's argument, making errors legible to humans who couldn't independently verify the answer.
- **Amplification:** iteratively extend human judgment by using AI to decompose hard questions into easier sub-questions that humans can evaluate, then aggregating judgments up the decomposition tree.
- **Automated interpretability:** use AI to generate natural language explanations of other AI's internal computations, making the reasoning process legible to humans.

These approaches become increasingly important at frontier capability because: (1) we cannot rely on human preference labels for tasks where humans are systematically outperformed; (2) without scalable oversight, the alignment training signal degrades precisely when it is most needed (high-capability models); (3) the governance argument — auditors, regulators, and the public need mechanisms to verify alignment claims without requiring frontier-level ML expertise.

## Flashcards

**What pretraining optimizes?** #flashcard
predict the next token over a broad text corpus

**What deployment requires?** #flashcard
follow instructions accurately, be helpful, avoid harm, maintain factual accuracy

**No reward model training required?** #flashcard
No reward model training required

**No PPO optimization loop required?** #flashcard
No PPO optimization loop required

**Single training run on the preference dataset?** #flashcard
Single training run on the preference dataset

**Memory: only π_θ (the policy) and π_ref (frozen)?** #flashcard
2× instead of 4× model copies

**Your dataset doesn't have paired comparisons (common in practice?** #flashcard
you have user thumbs up/down on individual responses)

**The good/bad examples are imbalanced (KTO handles this naturally; DPO needs equal numbers of y_w and y_l pairs)?** #flashcard
The good/bad examples are imbalanced (KTO handles this naturally; DPO needs equal numbers of y_w and y_l pairs)

**Empirically?** #flashcard
KTO matches or exceeds DPO on alignment benchmarks while requiring simpler annotation

**Choose the response that is most helpful, harmless, and honest?** #flashcard
Choose the response that is most helpful, harmless, and honest

**Prefer responses that are least likely to encourage illegal activity?** #flashcard
Prefer responses that are least likely to encourage illegal activity

**Prefer responses that would make a reasonable person feel respected, not harassed?** #flashcard
Prefer responses that would make a reasonable person feel respected, not harassed

**Choose the most helpful response if the request is clearly benign?** #flashcard
Choose the most helpful response if the request is clearly benign

**Functional correctness?** #flashcard
HumanEval (164 problems), MBPP (374 problems), SWE-bench (real GitHub issues). Primary metric: pass@k.

**Code quality: cyclomatic complexity, docstring coverage, style conformance (pylint score)?** #flashcard
automated but correlated with human preference.

**Safety?** #flashcard
does the model generate code with obvious security vulnerabilities (SQL injection, eval() on user input)? Red team evaluation with a set of security-focused prompts.

**Alignment stability?** #flashcard
does the model refuse unreasonable requests (write malware) while helping with reasonable ones (write a network scanner for pentesting with explicit authorization)?

**Debate?** #flashcard
two AI models argue for their respective answers; humans judge the debate (which is easier to evaluate than the underlying question). A correct debater can expose flaws in an incorrect debater's argument, making errors legible to humans who couldn't independently verify the answer.

**Amplification?** #flashcard
iteratively extend human judgment by using AI to decompose hard questions into easier sub-questions that humans can evaluate, then aggregating judgments up the decomposition tree.

**Automated interpretability?** #flashcard
use AI to generate natural language explanations of other AI's internal computations, making the reasoning process legible to humans.
