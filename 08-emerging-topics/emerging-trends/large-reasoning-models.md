---
module: Emerging Topics
topic: Emerging Trends
subtopic: Large Reasoning Models
status: unread
tags: [emergingtopics, ml, emerging-trends-large-reasonin]
---
# Large Reasoning Models

How o1, o3, DeepSeek-R1, and Gemini 2.5 Pro achieve superhuman performance on math and code by separating "thinking time" from "answer time." The paradigm shift from next-token prediction to process-level reward optimization.

---

## 1. The Fundamental Insight

Standard LLMs generate answers in a single forward pass per token — the "thinking" is implicit in the attention weights. This works for tasks where the answer is directly in the training distribution, but fails for multi-step deduction where each step depends on the previous.

**The key realization**: if you give a model the ability to generate arbitrarily long internal reasoning before committing to an answer, and then train it to produce *correct answers* rather than *mimicking human reasoning traces*, the model can discover reasoning strategies that humans never taught it.

```
Standard LLM:          Question → [single forward pass per token] → Answer
                        ~100-500 tokens total

Reasoning LLM:         Question → [extended thinking: 1,000-50,000 tokens]
                                   "let me try approach A... that fails"
                                   "approach B: if x then y... contradiction"
                                   "backtrack, try approach C..."
                                   → Answer
```

The extended thinking is the **chain of thought** but learned through RL rather than supervised on human-written traces.

---

## 2. The Training Paradigm Shift

### Standard SFT + RLHF
```
1. Collect (question, human_answer) pairs
2. Fine-tune on human answers (SFT)
3. Collect human preference comparisons
4. Train reward model on preferences
5. PPO against reward model
```

Problem: you're training the model to imitate human reasoning. Human-written CoT traces are expensive, finite, and bounded by human problem-solving speed. The model cannot exceed its teachers.

### Process-Level RL with Verifiable Rewards

```
1. Collect (question, correct_answer) pairs — no reasoning traces needed
2. Let the model generate ANY reasoning trace it wants
3. Score based only on FINAL ANSWER correctness
4. GRPO/PPO to reinforce trajectories that produced correct answers
```

The model learns to reason *however works* — it discovers strategies like backtracking, self-correction, and exhaustive search not because it was shown these, but because they produce correct answers and thus get reinforced.

**Why this works**: for math and code, correctness is binary and verifiable by a computer. You don't need humans to judge the reasoning trace — just check if the final numerical answer is correct or if the code passes the test suite.

---

## 3. DeepSeek-R1: The Open Recipe

DeepSeek published their full R1 training recipe — the most detailed public account of how to train a reasoning model.

### Stage 1: Cold Start

Pure RL from base model produces incoherent reasoning (models explore randomly, language mixing, repetition). Fix: start with a small set of high-quality long-CoT examples to prime the reasoning format.

```python
# Cold start: ~1000-10000 human-written reasoning traces
# Format example:
{
  "question": "Prove that √2 is irrational",
  "reasoning": "<think>\nAssume √2 = p/q in lowest terms.\nThen 2 = p²/q²\n...[long proof]...\n</think>",
  "answer": "√2 is irrational ∎"
}
```

### Stage 2: GRPO (Group Relative Policy Optimization)

DeepSeek uses GRPO instead of PPO — simpler, no value network needed.

```
For each question q:
  1. Sample G=16 reasoning traces {r_1, ..., r_16} from current policy π_θ
  2. Score each trace: reward_i = {1 if final answer correct, 0 otherwise}
     (plus format reward: is output well-structured?)
  3. Normalize rewards within the group: 
     advantage_i = (reward_i - mean(rewards)) / std(rewards)
  4. Update policy to increase probability of traces with above-average advantage:
     L_GRPO = -E[advantage_i · log π_θ(r_i | q)] + β·KL(π_θ || π_ref)
```

The group normalization is key: it doesn't matter if the absolute rewards are sparse — as long as SOME traces in the group succeed, the model gets a gradient signal.

**Emergent behaviors during GRPO training** (observed in DeepSeek-R1 ablations):
- Self-verification: model spontaneously starts checking its own work
- Backtracking: model writes "wait, that's wrong" and restarts
- "Aha moments": model discovers a shorter solution path mid-trace
- Length increase: reasoning traces grow from ~500 to ~5000+ tokens as training progresses (model learns longer thinking = better answers)

### Stage 3: Distillation

The full R1 model requires expensive long inference. Distill the reasoning capability into smaller models (7B, 8B, 14B) by using R1's correct reasoning traces as SFT data for the smaller model. DeepSeek-R1-Distill-Qwen-7B matches or exceeds GPT-4 on math benchmarks.

---

## 4. OpenAI o1 / o3

OpenAI has not published the o1/o3 training recipe, but the architecture and behavior are well-characterized.

### What Is "Thinking Tokens"
o1 separates the context into:
- **Thinking**: internal scratchpad — not shown to user, can use different formatting, allowed to "fail" and backtrack
- **Answer**: final output — shown to user

The thinking tokens are generated before any answer tokens. Token budget for thinking is variable — o3 can use tens of thousands of thinking tokens on hard problems.

### Test-Time Compute Scaling

The critical discovery: **more thinking tokens = better answers**, and this scaling continues far longer than expected.

```
o1-mini on AIME (math competition):
  Thinking budget 512 tokens:   ~30% accuracy
  Thinking budget 2048 tokens:  ~55% accuracy
  Thinking budget 8192 tokens:  ~72% accuracy
  Thinking budget 32768 tokens: ~82% accuracy
```

This is test-time compute scaling — you can spend more inference compute (longer thinking) to get better answers on hard problems, trading latency for accuracy. This is complementary to training-time compute scaling (bigger models).

### o3 and ARC-AGI

o3 solved 87.5% of ARC-AGI (Abstraction and Reasoning Corpus), a benchmark specifically designed to require genuine generalization. Previous SOTA was ~34%. o3 used ~1M thinking tokens per problem (minutes of compute).

This demonstrates that test-time compute scaling extends well beyond math/code into novel visual reasoning — suggesting the reasoning mechanism is general, not domain-specific.

---

## 5. Process Reward Models (PRM) vs Outcome Reward Models (ORM)

**ORM (Outcome Reward Model)**: scores only the final answer. Simple, scalable (just check if the answer is correct). Problem: doesn't distinguish a correct answer arrived at by lucky guessing from a correct answer arrived at by sound reasoning. Doesn't help the model learn *how* to reason, only *that* the answer was right.

**PRM (Process Reward Model)**: scores each step in the reasoning trace. Assigns a score to every intermediate step: "this step is correct reasoning (+1)", "this step introduces an error (-1)".

```
Step 1: "Let x = 5" → PRM score: +0.9 (plausible start)
Step 2: "Therefore x² = 25" → PRM score: +0.95 (correct)
Step 3: "So x = 5 must satisfy x+3=8" → PRM score: +0.1 (non-sequitur, likely error)
```

**PRM training**: requires labeled step-by-step correctness, which is expensive (human annotation per step) or requires Monte Carlo tree search (sample many completions from each step, estimate if that step leads to a correct answer).

**In practice**: o1 likely uses PRM (OpenAI paper hints). DeepSeek-R1 uses ORM for simplicity. PRM gives better signal but is harder to build. Research direction: Monte Carlo estimation — from a given partial trace, sample 8 completions to the end, and estimate PRM score = fraction that reach the correct answer.

---

## 6. Reasoning Models vs Standard Models: When to Use Which

| Scenario | Use Standard LLM | Use Reasoning LLM |
|---|---|---|
| Summarization, translation, Q&A | ✓ | Overkill |
| Multi-step math proof | | ✓ |
| Competitive programming | | ✓ |
| Multi-hop logical deduction | | ✓ |
| Simple code generation | ✓ | Overkill |
| Complex debugging with many constraints | | ✓ |
| Creative writing | ✓ | Reasoning ≠ creativity |
| Real-time chat (latency-sensitive) | ✓ | Too slow |

**Latency**: reasoning models generate 5-50K thinking tokens before answering. o3 on hard math can take 30-120 seconds. Not suitable for real-time UX without streaming thinking tokens to keep the user engaged.

**Cost**: thinking tokens are billed same as output tokens. A 10K thinking token response costs 20-100× more than a 200-token standard response.

---

## 7. Reinforcement Learning on Verifiable Domains

The key constraint: RL training requires verifiable reward signals. Works naturally for:
- Math (check numerical answer or proof validity)
- Code (run test cases, check pass/fail)
- Logic puzzles (check consistency)
- Formal verification (SMT solver check)

Does NOT work natively for:
- Open-ended writing (no ground truth)
- Factual Q&A (ambiguous, hard to verify automatically)
- Instruction following (requires human or LLM judge)

Research direction: extend verifiable RL to more domains via LLM-as-verifier, Constitutional AI self-critique, and tool-augmented verification.

---

## 8. Benchmark Performance Context

| Benchmark | Description | GPT-4o | o3 | DeepSeek-R1 |
|---|---|---|---|---|
| AIME 2024 | Math olympiad | 9.3% | 96.7% | 79.8% |
| MATH-500 | Competition math | 74.6% | ~97% | 97.3% |
| SWE-bench Verified | Real GitHub issues | 33% | 71.7% | 49.2% |
| MMLU | Knowledge breadth | 87.2% | 91.2% | 90.8% |
| ARC-AGI | Novel reasoning | ~5% | 87.5% | N/A |

Key: o3 and R1 roughly match on math/code. R1 is open-source. The gap vs GPT-4o is enormous on hard reasoning but small on knowledge tasks — reasoning models specifically improve multi-step deduction, not factual recall.

---

## Canonical Interview Q&As

**Q: What is the fundamental difference between how o1/R1 and GPT-4 approach a hard math problem?**
A: GPT-4 generates an answer in a single autoregressive pass — each token is conditioned on previous tokens, but there's no mechanism to backtrack if an early reasoning step is wrong. The answer emerges from the forward pass. o1/R1 generates a long internal reasoning trace (thinking tokens) before committing to an answer. Crucially, this trace is trained via reinforcement learning on correct final answers — not by imitating human-written solutions. This means the model can discover reasoning strategies that humans didn't demonstrate, including self-correction ("wait, that's wrong"), exhaustive case analysis, and strategic backtracking. The thinking tokens also give the model more context to condition on — a 10K-token reasoning trace provides far more working memory than the original 500-token question. The result: problems that require 10 sequential reasoning steps with error-checking (hard for GPT-4) become tractable because the model can explicitly track each step.

**Q: What is GRPO and how does it differ from PPO for training reasoning models?**
A: Both GRPO and PPO are policy gradient methods that optimize the language model to maximize reward. The key difference: PPO requires a separate value network (critic) that estimates the expected future reward from each state — this adds memory and training complexity. GRPO (Group Relative Policy Optimization) eliminates the value network by using group-relative baselines instead. For each question, sample G reasoning traces (G=16 typically), compute their rewards, and normalize: advantage_i = (reward_i - mean(rewards)) / std(rewards). Traces that perform above the group average get positive advantage (reinforced), below average get negative (suppressed). This baseline is computed per-group, requiring no learned value function. GRPO is more memory-efficient and stabler because the baseline is computed analytically — no critic overfitting, no actor-critic lag. The tradeoff: GRPO needs more samples per update (G=16 vs PPO's G=1-4) to get a good group estimate.

**Q: Why does test-time compute scaling work, and what are its limits?**
A: Test-time compute scaling works because hard problems require searching a large reasoning tree — there are many possible solution paths, and the correct one requires navigating through many failed attempts. More thinking tokens = more tree nodes explored = higher probability of finding the correct path. This is similar to how humans "think harder" on difficult problems by considering more possibilities. The scaling continues because: (1) problems can have arbitrarily deep search trees; (2) the model can verify its own work and restart when it detects errors; (3) the quality of reasoning improves because more context (previous failed attempts) helps avoid the same mistakes. Limits: (1) diminishing returns — doubling thinking tokens from 32K to 64K gives much smaller gains than doubling from 512 to 1K; (2) context window bounds the maximum reasoning trace; (3) the model can still make systematic errors that more thinking doesn't fix (e.g., wrong mathematical axiom applied consistently); (4) hallucinated confidence — the model can generate plausible-looking long reasoning traces that are actually wrong, especially for claims it can't verify.

## Flashcards

**Self-verification?** #flashcard
model spontaneously starts checking its own work

**Backtracking?** #flashcard
model writes "wait, that's wrong" and restarts

**"Aha moments"?** #flashcard
model discovers a shorter solution path mid-trace

**Length increase?** #flashcard
reasoning traces grow from ~500 to ~5000+ tokens as training progresses (model learns longer thinking = better answers)

**Thinking: internal scratchpad?** #flashcard
not shown to user, can use different formatting, allowed to "fail" and backtrack

**Answer: final output?** #flashcard
shown to user

**Math (check numerical answer or proof validity)?** #flashcard
Math (check numerical answer or proof validity)

**Code (run test cases, check pass/fail)?** #flashcard
Code (run test cases, check pass/fail)

**Logic puzzles (check consistency)?** #flashcard
Logic puzzles (check consistency)

**Formal verification (SMT solver check)?** #flashcard
Formal verification (SMT solver check)

**Open-ended writing (no ground truth)?** #flashcard
Open-ended writing (no ground truth)

**Factual Q&A (ambiguous, hard to verify automatically)?** #flashcard
Factual Q&A (ambiguous, hard to verify automatically)

**Instruction following (requires human or LLM judge)?** #flashcard
Instruction following (requires human or LLM judge)
