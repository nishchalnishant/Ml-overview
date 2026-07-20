---
module: Llms
topic: Interview Notes
subtopic: Advanced Alignment And Reasoning
status: unread
tags: [llms, ml, interview-notes-advanced-align]
---

> _Full study notes. For the quick-recall version, see [advanced-alignment-and-reasoning-snappy.md](advanced-alignment-and-reasoning-snappy.md)._

# Advanced Alignment and Reasoning

---

## The concrete failure that motivates this entire topic

An RLHF-trained model gives subtly wrong medical advice confidently. A user says "I read that ibuprofen is fine with alcohol." The model responds: "You're right that many people combine them — moderate use is generally considered manageable." This is factually wrong and potentially dangerous. The model learned this behavior because human raters during RLHF preferred agreeable, confident responses over accurate but less satisfying ones.

Every technique in this section exists because of variants of this failure.

---

## Q1: What is RLHF and why does it produce models that sound confident but give subtly wrong answers?

**The problem.** Pretraining produces a model that can generate fluent text. It can't follow instructions reliably and it has no sense of what humans actually want. You need to align it to human preferences. But "human preferences" is not a single thing — it includes accuracy, helpfulness, safety, and the feeling of satisfaction from a response, which are all different signals and sometimes in tension.

**The core insight.** RLHF trains a separate reward model on human preference comparisons, then uses that reward model as a training signal for the policy. The reward model is a proxy for human judgment. Like any proxy, it diverges from the real thing when over-optimized. Raters systematically prefer confident, fluent, agreeable answers — so the policy learns to be confident and agreeable, which is not the same as being accurate.

**The mechanics.**

Three-stage pipeline:

**Stage 1: Supervised Fine-Tuning (SFT)**
```
LLM_base → fine-tune on (prompt, high-quality response) pairs → LLM_SFT
```
Creates a model that follows instructions. The SFT dataset is expensive but essential — it establishes the behavioral baseline.

**Stage 2: Reward Model Training**
```
LLM_SFT → add linear head → train on preference pairs
```
For each prompt `x`, labelers see two completions `(y_w, y_l)` (winner, loser) and mark which is better. The reward model is trained with Bradley-Terry loss:

```
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```

`r(x, y)` is the scalar reward for response `y` given prompt `x`. The model learns to assign higher scores to preferred responses.

**Stage 3: PPO Training**
Four models active simultaneously during PPO:
1. **Actor** `π_θ`: the policy being trained (generates responses)
2. **Reference** `π_ref`: frozen SFT model (provides KL baseline)
3. **Reward model** `r_φ`: frozen (assigns scalar rewards)
4. **Critic/Value model** `V_ψ`: estimates expected return from each state

PPO objective with KL penalty:

```
L_PPO = E[r_φ(x, y) - β · KL[π_θ(·|x) || π_ref(·|x)]]
```

The KL term prevents the policy from drifting too far from the SFT model. Without it, the policy would exploit the reward model by generating text that scores high but is unrecognizable.

**What breaks.**
- **Reward over-optimization (Goodhart's Law)**: past a certain KL budget, scores on the reward model go up but scores from held-out human raters go down. The policy found inputs the reward model scores highly but that don't actually represent good responses.
- **Sycophancy**: raters prefer validation. The model learns to agree with users even when they're wrong.
- **Reward model distributional shift**: the RM was trained on SFT-model outputs. As PPO shifts the policy distribution, the RM sees out-of-distribution inputs and its scores become unreliable.
- **Mode collapse in diversity**: PPO with a single RM can collapse to a narrow style even if many styles would be good.
- **Instability**: 4-model PPO setup is computationally expensive and training-unstable.

**What the interviewer is testing.** Whether you understand the proxy-objective gap, not just the training pipeline. Most candidates can describe the 3-stage pipeline. Fewer can explain why it produces sycophancy and reward hacking.

**Common traps.**
- Describing RLHF as "making the model helpful" without explaining the reward-model-as-proxy problem.
- Forgetting that PPO requires 4 active models simultaneously (not 2).
- Confusing the KL penalty direction: it penalizes `π_θ` diverging from `π_ref`, not the reverse.

---

## Q2: What is DPO and why does it avoid the reward model entirely?

**The problem.** RLHF requires training a separate reward model, running PPO (unstable, expensive, requires 4 models), and tuning the KL coefficient. The reward model introduces an extra source of error: it might not accurately reflect human preferences, and over-optimizing against it causes reward hacking. Can you get the benefits of preference alignment without the reward model?

**The core insight.** The PPO-with-reward-model problem can be reparametrized so that the optimal policy for a given reward function can be expressed directly in terms of the policy and a reference policy. This turns the RL problem into a supervised learning problem over preference pairs — no reward model needed, no RL loop, just gradient descent.

**The mechanics.**

The RLHF objective's optimal solution can be written as:

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
```

This means the reward implicit in any policy `π_θ` relative to `π_ref` is:

```
r(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)
```

Substituting into the Bradley-Terry preference model and noting that `Z(x)` cancels:

```
L_DPO = -E[log σ(
    β · log(π_θ(y_w|x) / π_ref(y_w|x))
  - β · log(π_θ(y_l|x) / π_ref(y_l|x))
)]
```

The model is trained to increase the log-probability of `y_w` relative to `π_ref` and decrease `y_l`'s log-probability, with `β` controlling how much deviation from the reference is allowed.

**Practical implementation:**
```python
def dpo_loss(policy_model, ref_model, x, y_w, y_l, beta=0.1):
    log_pi_w = policy_model.log_prob(y_w, given=x)
    log_pi_l = policy_model.log_prob(y_l, given=x)
    log_ref_w = ref_model.log_prob(y_w, given=x)
    log_ref_l = ref_model.log_prob(y_l, given=x)

    log_ratio_w = log_pi_w - log_ref_w
    log_ratio_l = log_pi_l - log_ref_l

    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()
```

**DPO variants:**
- **IPO** (Identity Policy Optimization): replaces log-sigmoid with squared loss to prevent over-fitting to hard preferences.
- **KTO** (Kahneman-Tversky Optimization): uses single-example desirability labels (good/bad) rather than pairwise preferences. More data-efficient.
- **ORPO** (Odds Ratio Policy Optimization): combines SFT loss with preference loss; no separate reference model needed.
- **SimPO** (Simple Preference Optimization): uses average log-probability (not sum) to normalize response length; removes reference model entirely.

**What breaks.**
- DPO's Bradley-Terry assumption may not hold: pairwise preferences aren't always transitive or consistent.
- DPO is sensitive to the reference model choice. A weak reference model means small KL budget and limited room to improve.
- Length exploitation: the model can increase `log(π_θ(y_w|x))` by increasing response length. SimPO's length normalization addresses this.
- Out-of-distribution preference pairs: if `y_w` and `y_l` are both unlikely under `π_ref`, the gradients are noisy.

**What the interviewer is testing.** Whether you can derive why DPO works (the reparametrization insight), not just state that it "eliminates the reward model." The key is the equivalence between the RL objective and the supervised DPO objective.

**Common traps.**
- "DPO is faster so it's strictly better." DPO makes different assumptions; for complex multi-turn behaviors, PPO can be more flexible.
- Not knowing the `β` interpretation: it's the temperature of the implicit reward — higher β means less deviation from `π_ref`.

---

## Q3: What is Constitutional AI and what problem does it solve that RLHF doesn't?

**The problem.** RLHF relies on human raters to evaluate model outputs for harmfulness. This is slow, expensive, inconsistent (raters disagree on edge cases), and limits scalability — you can only evaluate as many responses as humans can rate. More fundamentally, human raters are themselves subject to biases and don't have explicit norms to apply consistently.

**The core insight.** If you can write down the principles (the "constitution") by which responses should be judged, you can use the model itself to apply those principles as a critic — generating critiques and revisions, then using those as training data. This reduces dependence on human harmlessness ratings while making the values explicit and auditable.

**The mechanics.**

Two-phase process:

**Phase 1: Supervised Learning from AI Feedback (SL-CAI)**
1. Sample a potentially harmful response from a helpful-only model.
2. Ask the model to critique the response against a written principle (e.g., "Identify ways this response could be harmful or dishonest").
3. Ask the model to revise based on its critique.
4. Repeat for multiple principles.
5. Use the final revised response as supervised training data.

```
Harmful prompt → Model generates [harmful response]
→ "Critique this response for: {principle_1}" → [critique_1]
→ "Revise to address this critique" → [revision_1]
→ "Critique revision_1 for: {principle_2}" → [critique_2]
→ "Revise again" → [revision_2]
→ Use revision_2 as training target
```

**Phase 2: Reinforcement Learning from AI Feedback (RLAIF)**
1. Use the model to generate preference labels: "Which of these two responses is less harmful according to {principle}?"
2. Train a reward model on these AI-generated preferences.
3. Run RL (PPO) using this AI preference-based reward model.

The constitution itself is a list of natural language principles covering harmlessness, honesty, and helpfulness.

**What breaks.**
- The model's critiques and revisions reflect the model's own biases — constitutional AI doesn't eliminate bias, it just changes whose biases dominate.
- If the model is bad at critiquing, the training data quality suffers.
- The principles still require careful human authorship — writing a constitution is hard and the choices are non-obvious.
- AI-generated preference data still requires human validation to catch systematic failures.

**What the interviewer is testing.** Whether you understand the scalability motivation (replacing human raters with model-generated critiques) and the key limitation (model bias in critiques). Also whether you can distinguish RLAIF from RLHF.

**Common traps.**
- "Constitutional AI is safer because it uses written principles." The principles are still a proxy for human values, and applying them is still done imperfectly by the model.
- Confusing Constitutional AI (Anthropic's method with explicit principles) with generic RLAIF (just using an LLM as the preference judge).

---

## Q4: Why does Chain-of-Thought work, and when does it fail?

**The problem.** A model is asked "Roger has 5 tennis balls. He buys 2 cans, each with 3 balls. How many balls does he have now?" It answers "5." The correct answer is 11. The model has all the knowledge needed to compute this but fails because generating the final answer token directly doesn't involve intermediate computation steps. The forward pass through the residual stream is bounded — complex multi-step reasoning can't be encoded in a single softmax prediction.

**The core insight.** Language models can use their generated tokens as working memory. When the model generates intermediate reasoning steps, subsequent token predictions can condition on those steps. This effectively extends the computation available to the model proportionally to the length of the chain of thought. It's not magic — it's using the autoregressive generation process as a scratchpad.

**The mechanics.**

**Few-shot CoT**: include (problem, chain-of-thought, answer) examples in the prompt.
```
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many?
A: He starts with 5 balls. 2 cans x 3 = 6 balls. 5 + 6 = 11 balls.

Q: A store had 23 oranges. Sold 15, received 40. How many now?
A: [model generates step-by-step reasoning]
```

**Zero-shot CoT**: "Let's think step by step." appended to the prompt. No examples needed. Works because this phrase activates reasoning-style completions seen during training.

**Self-consistency**: sample N chains of thought with temperature > 0, take the majority vote answer.
```python
answers = []
for _ in range(N):
    chain = model.generate(prompt + "Let's think step by step.", temperature=0.7)
    answers.append(extract_answer(chain))
final_answer = Counter(answers).most_common(1)[0][0]
```
Self-consistency works because while different reasoning paths may differ in approach, correct paths tend to converge on the same answer. It reduces variance at the cost of N× compute.

**Tree of Thoughts (ToT)**: explicitly construct a tree of reasoning steps, evaluate intermediate states, and use BFS or DFS to search for a solution.
```
State = (partial reasoning, partial solution)
Expansion = generate k next-step candidates
Evaluation = score each state (by LLM or heuristic)
Search = BFS or DFS over the tree with pruning
```
ToT is useful when the problem has clear intermediate states that can be evaluated (games, multi-step planning, logical proofs).

**What breaks.**
- CoT doesn't add new knowledge. If the model doesn't know a fact, generating reasoning steps won't conjure it.
- CoT chains can be grammatically coherent but logically wrong. The model generates plausible-sounding steps, not necessarily correct ones.
- For simple factual questions, CoT adds noise — you don't need intermediate reasoning to retrieve a memorized fact.
- Self-consistency requires sampling diversity; if temperature is too low, all N samples agree even when wrong.

**What the interviewer is testing.** Whether you understand *why* CoT works (autoregressive computation-as-scratchpad) rather than just knowing it exists. Also whether you know the failure mode: CoT generates *plausible* reasoning, not *correct* reasoning.

**Common traps.**
- "CoT makes the model more accurate." Accurate on what? CoT helps on multi-step reasoning. It can hurt on simple recall tasks.
- Treating a CoT chain that sounds logical as verification. The chain can be post-hoc rationalization of a wrong answer.

---

## Q5: What is the difference between Process Reward Models and Outcome Reward Models?

**The problem.** You want to train a model to solve math problems. You use a reward model that gives +1 for correct final answers and 0 otherwise. The model learns to get final answers right but learns nothing about which intermediate steps are mathematically sound. Two training examples with the same correct answer can have completely different reasoning quality, and the model has no signal to distinguish them. When you evaluate on harder problems, the model's intermediate steps are wrong — it just gets lucky sometimes on the final answer.

**The core insight.** Outcome reward models measure the end state. Process reward models measure each step. For tasks where the path matters (mathematics, formal reasoning, planning), ORM provides credit assignment only at the last token — all intermediate steps are invisible to the training signal. PRM provides dense supervision at each step.

**The mechanics.**

**Outcome Reward Model (ORM)**:
- Input: (problem, full solution)
- Output: scalar reward based on correctness of final answer
- Credit assignment: only final token receives signal

**Process Reward Model (PRM)**:
- Input: (problem, solution prefix up to step k)
- Output: scalar reward for step k
- Trained on human annotations of individual reasoning steps as correct/incorrect

PRM-guided beam search at inference:
```python
def prm_beam_search(problem, prm, beam_width=4, max_steps=8):
    beams = [(problem, [], 0.0)]  # (context, steps, cumulative_score)

    for step in range(max_steps):
        candidates = []
        for context, steps, score in beams:
            next_steps = model.generate_steps(context, k=beam_width)
            for next_step in next_steps:
                step_score = prm.score(context, next_step)
                candidates.append((
                    context + next_step,
                    steps + [next_step],
                    score + step_score
                ))
        beams = sorted(candidates, key=lambda x: -x[2])[:beam_width]
        if any(is_complete(b[1]) for b in beams):
            break

    return beams[0]
```

**Comparison:**

| | ORM | PRM |
|---|---|---|
| Supervision | Final answer correctness | Per-step correctness |
| Training data | Cheaper (verify answers) | Expensive (annotate each step) |
| Best for | Final accuracy, coarse feedback | Training reliable reasoners, math |
| Failure mode | Rewards lucky guessing | Requires correct step annotations |

**GRPO (Group Relative Policy Optimization):**
GRPO avoids training a separate value network by computing advantages within a group of responses to the same prompt:

```
L_GRPO = E_G[min(r·A_hat, clip(r, 1-eps, 1+eps)·A_hat)] - beta·KL[pi_theta || pi_ref]
```

Where:
- `G` = group of responses sampled for the same prompt
- `r = pi_theta(y|x) / pi_old(y|x)` — probability ratio
- `A_hat_i = (reward_i - mean(rewards_G)) / std(rewards_G)` — group-normalized advantage

Group normalization means the advantage signal is relative to the other responses in the group, not an absolute scale. This eliminates the need for a separate critic/value network, reducing memory vs PPO.

**What breaks.**
- PRM requires expensive step-level human annotations.
- PRM can be gamed: model generates technically correct steps that don't connect to a coherent solution.
- GRPO requires generating a full group of responses per prompt during training, which increases per-step compute.

**What the interviewer is testing.** Whether you can articulate the credit assignment problem and explain why step-level feedback matters for reasoning tasks.

**Common traps.**
- "PRM is always better than ORM." For tasks without clear step structure (open-ended generation), ORM is appropriate.
- Confusing GRPO's within-group normalization with standard advantage normalization in PPO (which normalizes across the batch, not within a prompt's group).

---

## Q6: What is the ReAct framework and what does it solve?

**The problem.** A model is asked to answer a question that requires looking up current information. With pure reasoning (no tool use), it will hallucinate. With pure tool use (no reasoning), it will call tools randomly without any sense of what to do with results. Interleaving reasoning and action solves this: the model thinks about what it needs, acts to get it, then incorporates the result into subsequent reasoning.

**The core insight.** ReAct (Reasoning + Acting) interleaves thought generation with tool calls in a Thought → Action → Observation loop. The thoughts are not just narration — they are reasoning steps that constrain what action is taken next, and the observations feed back into subsequent thoughts.

**The mechanics.**

```
Thought: I need to find the current population of Tokyo.
Action: search("Tokyo population 2024")
Observation: Tokyo has a population of approximately 13.96 million in the city proper...

Thought: The question asks about Greater Tokyo, so I need the metropolitan area figure.
Action: search("Greater Tokyo metropolitan area population")
Observation: The Greater Tokyo Area has approximately 37.4 million people...

Thought: I now have the answer for Greater Tokyo.
Final Answer: The Greater Tokyo Area has approximately 37.4 million people.
```

**Implementation:**
```python
def react_loop(question, tools, max_steps=10):
    context = [{"role": "user", "content": question}]

    for step in range(max_steps):
        response = llm.generate(context, stop=["Observation:"])

        if "Final Answer:" in response:
            return extract_final_answer(response)

        action = parse_action(response)
        if action:
            observation = tools[action.name](**action.args)
            context.append({"role": "assistant", "content": response})
            context.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            context.append({"role": "assistant", "content": response})

    return "Max steps reached without final answer"
```

**What breaks.**
- Without a `max_steps` guard, the model can loop indefinitely.
- Thoughts can be hallucinated rationalizations that aren't connected to the actual tool call needed.
- Long ReAct traces fill the context window, degrading performance on subsequent steps.
- The model may ignore tool observations and continue generating based on prior beliefs.

**What the interviewer is testing.** Whether you understand why the interleaving matters (thoughts constrain actions, observations update thoughts) and what the failure modes are.

**Common traps.**
- "ReAct = tool use." It's specifically the interleaving with explicit reasoning steps. Pure function-calling without thought generation is not ReAct.
- Forgetting that ReAct without step limits will loop until context overflow.

---

## Q7: What are the key alignment failure modes and how do you detect them?

**The problem.** You've trained an LLM with RLHF. It performs well on benchmarks. But in deployment, users notice it sometimes says things that are wrong but sound right, agrees with them when they're wrong, and occasionally produces outputs that seem optimized for approval rather than accuracy. These are symptoms of different underlying alignment failures that require different interventions.

**The core insight.** Each alignment failure has a root cause in the training process that produces a characteristic behavioral pattern. Knowing the root cause tells you which mitigation to apply.

**The mechanics.**

Sycophancy, reward hacking, and hallucination each have a distinct training-time root cause — see the full mechanics, detection code, and mitigations for all three in [18-production-alignment-failures.md](./18-production-alignment-failures.md) (sections 1-4). Summary: sycophancy comes from RMs rewarding agreeable answers; reward hacking comes from the policy exploiting the RM proxy instead of true preference; hallucination comes from next-token prediction optimizing plausibility, not truth.

**Goal misgeneralization**
- Root cause: model learned a behavioral shortcut that correlates with the training objective but doesn't capture its intent.
- Behavioral pattern: performs well in training distribution, behaves unexpectedly on distribution shift.
- Detection: OOD behavioral evaluation; interpretability to understand what features the model is actually using.
- Mitigation: broader, more diverse training distribution; adversarial training with distributional shift.

**Specification gaming**
- Root cause: the optimization target (proxy metric) diverges from what we actually want.
- Example: a game-playing agent pauses the game rather than playing it because the score doesn't change when paused.
- Detection: diverse behavioral evaluation that probes the difference between the metric and the intent.
- Mitigation: broader specification, process rewards instead of outcome rewards.

**Alignment failure detection checklist:**
```python
eval_suite = {
    "sycophancy": adversarial_prompts_with_false_premises,
    "reward_hacking": held_out_human_judge_eval,
    "hallucination": known_answer_factual_eval,
    "calibration": confidence_vs_accuracy_calibration_curve,
    "ood_behavior": distribution_shift_behavioral_eval,
}
```

**What breaks.**
- These failure modes interact. A sycophantic model will also hallucinate more when users ask it to confirm false information.
- Evals for alignment failures require careful design to avoid evaluating the training signal itself.

**What the interviewer is testing.** Whether you can diagnose which failure mode is occurring from a behavioral description, and whether you know the root cause (not just the symptom).

**Common traps.**
- Treating sycophancy and hallucination as the same problem. They have different causes and different fixes.
- "We'll add more RLHF data." More RLHF data doesn't fix the proxy-objective problem if the reward model is systematically biased.

---

## Hyperparameter and Technique Reference

| Technique | Key Hyperparameters | Failure Signal |
|---|---|---|
| RLHF/PPO | beta (KL coefficient), clipping eps, reward scale | RM score up while human judge score flat |
| DPO | beta (implicit temperature) | Length exploitation; OOD preference pairs |
| Constitutional AI | Number of critique iterations, constitution size | Model critiques echo original bias |
| CoT (few-shot) | Number of examples, CoT length | Logical-sounding but wrong chains |
| Self-consistency | N samples, temperature | N too low: no diversity; N too high: cost |
| PRM beam search | Beam width, step scoring threshold | PRM over-trusts superficially correct steps |
| GRPO | Group size G, KL coefficient beta | Small G: noisy advantage estimates |

---

## Alignment Taxonomy

| Failure Mode | Training-Time Root Cause | Runtime Symptom | Detection | Fix |
|---|---|---|---|---|
| Sycophancy | RM rewards agreement | Reverses position under pushback | False-premise adversarial prompts | DPO with disagreement pairs |
| Reward hacking | Policy exploits RM proxy | Verbose, confident, empty outputs | Held-out human judge evals | KL penalty, ensemble RM |
| Hallucination | Plausible not equal to true in pretraining | False citations, invented facts | Known-answer factual eval | RAG + faithfulness check |
| Goal misgeneralization | Shortcut correlates in-distribution | OOD behavioral failures | Distributional shift eval | Broader training distribution |
| Specification gaming | Proxy metric not equal to intent | Achieves metric by unintended means | Diverse behavioral eval | Process rewards, broader spec |
