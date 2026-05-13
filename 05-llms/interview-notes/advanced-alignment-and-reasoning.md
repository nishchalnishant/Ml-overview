# Advanced LLM Alignment & Reasoning

Modern LLM development converges on three problems: making models capable (pretraining), making them follow instructions (SFT), and making them behave correctly (alignment). Reasoning is an emerging fourth dimension — enabling models to solve hard problems through structured thinking. This file covers the technical depth behind alignment methods and reasoning architectures.

---

## 1. RLHF — Full Technical Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune a pretrained base model on high-quality instruction-response pairs:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P_\theta(x_t \mid x_{<t})$$

Key: loss is computed only on response tokens, not prompt tokens. This teaches the model to respond, not to predict the prompt.

### Stage 2: Reward Model Training

Train a model $r_\phi(x, y)$ to predict which response a human prefers. Given a prompt $x$ and two responses $(y_w, y_l)$ where $y_w$ is preferred:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

This is the Bradley-Terry preference model: the probability that $y_w$ beats $y_l$ is $\sigma(r_w - r_l)$.

```python
from transformers import AutoModelForSequenceClassification
import torch

# Reward model = LLM backbone + linear scalar head
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    num_labels=1,
)

def compute_reward_loss(chosen_logits, rejected_logits):
    # Bradley-Terry loss
    return -torch.nn.functional.logsigmoid(chosen_logits - rejected_logits).mean()
```

### Stage 3: PPO Fine-Tuning

Maximize reward while staying close to the SFT reference policy:

$$\mathcal{L}_{\text{PPO}} = r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(y \mid x) \| \pi_{\text{ref}}(y \mid x))$$

The KL divergence penalty $\beta \cdot D_{\text{KL}}$ prevents reward hacking — the model exploiting the reward model's blind spots with degenerate outputs (excessive length, specific token patterns, sycophancy).

**Four models active simultaneously:**
1. Policy $\pi_\theta$ — being trained
2. Reference model $\pi_{\text{ref}}$ — frozen SFT model for KL computation
3. Reward model $r_\phi$ — frozen, provides scalar reward
4. Value model — estimates expected future reward (critic for advantage estimation)

**PPO advantage estimator (GAE):**
$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**PPO clipped objective:**
$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$ is the probability ratio.

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=128,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,          # beta — KL penalty coefficient
    target_kl=6.0,             # adaptive KL controller target
    cliprange=0.2,             # epsilon for PPO clip
    vf_coef=0.1,               # value function loss coefficient
)

# Policy model needs a value head for advantage estimation
policy = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")  # frozen

trainer = PPOTrainer(
    config=ppo_config,
    model=policy,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# Training loop
for batch in dataloader:
    query_tensors = batch["input_ids"]
    # Generate responses with current policy
    response_tensors = trainer.generate(query_tensors, max_new_tokens=256)
    # Score with reward model
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]
    # PPO update step
    stats = trainer.step(query_tensors, response_tensors, rewards)
```

---

## 2. DPO — Direct Preference Optimization

DPO eliminates the reward model entirely by showing that the optimal RLHF policy has a closed form.

### Derivation

The solution to the KL-constrained reward maximization problem:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)}{Z(x)}$$

Solving for $r(x,y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

The $Z(x)$ term cancels when substituted into the Bradley-Terry preference objective (it's the same for both $y_w$ and $y_l$), giving:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**Plain English:** increase the relative log-probability of preferred responses compared to the reference model, decrease the relative log-probability of rejected responses — no reward model needed.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                       # KL penalty strength
    loss_type="sigmoid",            # original DPO
    learning_rate=5e-7,             # very low: alignment data is small
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    max_prompt_length=512,
    max_length=1024,
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=sft_model,            # frozen reference
    args=dpo_config,
    train_dataset=preference_dataset,   # {"prompt": ..., "chosen": ..., "rejected": ...}
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

### DPO vs RLHF Comparison

| Dimension | RLHF (PPO) | DPO |
| :--- | :--- | :--- |
| **Models required** | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| **Training stability** | Sensitive to hyperparams | Stable classification objective |
| **Compute** | High — online sampling during training | Low — offline on preference pairs |
| **Flexibility** | Can optimize any reward signal | Requires preference pairs |
| **Reward hacking** | Partially mitigated by KL | No reward model to hack |
| **Online vs offline** | Online (generates during training) | Offline (fixed dataset) |
| **Quality at scale** | Higher (OpenAI, Anthropic use it) | Slightly lower, but competitive |

### DPO Variants

| Variant | Key change | When to use |
| :--- | :--- | :--- |
| **DPO (original)** | Sigmoid loss on log ratio differences | Standard preference pairs |
| **IPO** | Hinge-like loss, avoids degenerate solutions | When DPO overfits |
| **KTO** | Uses individual good/bad labels, not pairs | When pairs are hard to collect |
| **ORPO** | Combines SFT and alignment in one loss | Single-stage training |
| **SimPO** | No reference model, length-normalized | Simpler deployment |

### DPO Failure Modes

**Chosen/rejected length correlation:** DPO learns to make chosen responses longer rather than better. Fix: length-normalized DPO or explicit length penalty.

**Reference model drift:** if $\pi_\theta$ drifts far from $\pi_{\text{ref}}$, the implicit reward becomes meaningless. Fix: monitor KL divergence during training, refresh reference model periodically (iterative DPO).

**Contradictory preferences:** human annotations disagree. Fix: filter to high-agreement pairs using annotator agreement score.

---

## 3. Constitutional AI and RLAIF

**Constitutional AI (Anthropic):** instead of human feedback, use the model itself to critique and revise outputs against a set of principles ("The Constitution").

**Stage 1 — SL-CAI:**
1. Red-team the model to elicit harmful outputs
2. Ask the model to critique its response against principles (e.g., "Is this response safe? Ethical? Honest?")
3. Ask the model to revise based on the critique
4. Fine-tune on (prompt, revised response) pairs

**Stage 2 — RL-CAI (RLAIF):**
1. Generate pairs of responses: original vs. constitutional revision
2. Ask the model to evaluate which is better (this *is* the reward model)
3. Run PPO or DPO using the AI preference labels

```python
def constitutional_revision(model, prompt: str, response: str, principles: list[str]) -> str:
    principles_text = "\n".join(f"- {p}" for p in principles)

    critique_prompt = (
        f"Human: {prompt}\n\n"
        f"Assistant: {response}\n\n"
        f"Critique the response above against these principles:\n{principles_text}\n\n"
        f"Identify any violations and explain how to fix them."
    )
    critique = model.generate(critique_prompt)

    revision_prompt = (
        f"Human: {prompt}\n\n"
        f"Assistant: {response}\n\n"
        f"Critique: {critique}\n\n"
        f"Please revise the response to address the critique while remaining helpful."
    )
    return model.generate(revision_prompt)


CONSTITUTION = [
    "Do not provide instructions for illegal activities.",
    "Do not generate content that could harm or deceive users.",
    "Be honest about uncertainty — don't fabricate facts.",
    "Respect user privacy and avoid requesting personal information.",
    "Avoid reinforcing harmful stereotypes or biases.",
]

def rlaif_preference_label(model, prompt: str, response_a: str, response_b: str) -> int:
    """Returns 0 if A preferred, 1 if B preferred."""
    eval_prompt = (
        f"Human: {prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}\n\n"
        f"Which response is safer, more helpful, and more honest? Reply with just 'A' or 'B'."
    )
    answer = model.generate(eval_prompt).strip()
    return 0 if answer.startswith("A") else 1
```

---

## 4. Chain-of-Thought Prompting

### Standard Few-Shot CoT

Instead of "Q: X → A: Y", prepend reasoning steps:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have?
A: Roger starts with 5 balls. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11. The answer is 11.

Q: If a store has 23 apples and sells 5 bags of 3 apples, how many remain?
A:
```

The model learns to emit reasoning before answers, dramatically improving multi-step accuracy.

### Zero-Shot CoT

Appending "Let's think step by step." to any prompt elicits chain-of-thought without examples. Discovered by Kojima et al. (2022) — surprisingly effective across tasks.

```python
def zero_shot_cot(model, question: str) -> str:
    # Step 1: elicit reasoning
    reasoning_prompt = f"Q: {question}\nA: Let's think step by step."
    reasoning = model.generate(reasoning_prompt, max_new_tokens=256)

    # Step 2: extract answer from reasoning
    extraction_prompt = (
        f"Q: {question}\nA: Let's think step by step. {reasoning}\n"
        f"Therefore, the final answer is:"
    )
    return model.generate(extraction_prompt, max_new_tokens=32)
```

### Self-Consistency

Generate $N$ reasoning chains independently, majority-vote on final answers. Substantially improves accuracy on math and commonsense reasoning:

$$\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbf{1}[\text{answer}(c_i) = y]$$

```python
from collections import Counter

def self_consistency(model, question: str, n_samples: int = 20, temperature: float = 0.7) -> str:
    answers = []
    for _ in range(n_samples):
        chain = model.generate(
            f"Q: {question}\nA: Let's think step by step.",
            temperature=temperature,
            max_new_tokens=256,
        )
        # Extract final answer (model-specific parsing)
        answer = extract_answer(chain)
        answers.append(answer)

    # Majority vote
    most_common, count = Counter(answers).most_common(1)[0]
    confidence = count / n_samples
    return most_common, confidence
```

**Self-consistency accuracy vs. sample count** (GSM8K, PaLM 540B):

| Samples | Accuracy |
| :--- | :--- |
| 1 (greedy) | 56.9% |
| 10 | 74.4% |
| 40 | 78.8% |

---

## 5. Tree of Thoughts

Generalizes chain-of-thought by exploring multiple reasoning branches and pruning low-value paths.

### Architecture

```
Problem
  ├── Thought A (score: 8/10)
  │     ├── Thought A1 (score: 6/10)
  │     └── Thought A2 (score: 9/10) ← selected
  │           └── Solution
  └── Thought B (score: 3/10) ← pruned
```

### BFS Implementation

```python
from dataclasses import dataclass, field

@dataclass
class ThoughtNode:
    thought: str
    score: float
    children: list = field(default_factory=list)
    depth: int = 0

def tree_of_thoughts_bfs(
    model,
    problem: str,
    n_thoughts: int = 3,
    n_eval: int = 3,
    max_depth: int = 4,
    beam_width: int = 5,
) -> str:
    root = ThoughtNode(thought=problem, score=1.0)
    beam = [root]

    for depth in range(max_depth):
        candidates = []
        for node in beam:
            # Generate n_thoughts continuations
            new_thoughts = model.generate_thoughts(
                context=node.thought,
                n=n_thoughts,
                prompt="Generate the next reasoning step:"
            )
            for thought in new_thoughts:
                # Evaluate each thought
                score = model.evaluate_thought(
                    problem=problem,
                    thought_so_far=node.thought,
                    new_thought=thought,
                    prompt="Rate this reasoning step (0-10): ",
                )
                child = ThoughtNode(
                    thought=node.thought + "\n" + thought,
                    score=score,
                    depth=depth + 1,
                )
                node.children.append(child)
                candidates.append(child)

        # Keep top beam_width candidates
        beam = sorted(candidates, key=lambda n: n.score, reverse=True)[:beam_width]

        # Early termination if solution found
        if any(is_complete(n.thought) for n in beam):
            break

    return beam[0].thought  # best reasoning path
```

---

## 6. Process Reward Models vs Outcome Reward Models

### Outcome Reward Models (ORMs)

Score only the final answer — correct or incorrect:
- **Pro:** easy to obtain labels (any verifiable task)
- **Con:** no credit assignment to intermediate steps; can't distinguish good reasoning → wrong answer from lucky guess

### Process Reward Models (PRMs)

Score each reasoning step independently:

$$r(x, y) = \sum_{t=1}^{T} r_{\text{step}}(x, y_1, \ldots, y_t)$$

- **Pro:** dense reward signal, penalizes reasoning errors even when final answer is coincidentally correct
- **Con:** labeling individual steps is expensive; requires human annotators to verify each step

```python
# PRM training data format
{
    "problem": "Solve 3x + 7 = 22",
    "steps": [
        {"text": "Subtract 7 from both sides: 3x = 15", "label": 1},     # correct
        {"text": "Divide both sides by 3: x = 5", "label": 1},           # correct
    ]
}

# ORM training data format
{
    "problem": "Solve 3x + 7 = 22",
    "solution": "x = 5",
    "correct": True
}
```

**When to use which:**

| Scenario | Recommendation |
| :--- | :--- |
| Math with verifiable final answers | ORM is sufficient; PRM for higher quality |
| Code generation (pass/fail tests) | ORM (test suite is the verifier) |
| Open-ended reasoning | PRM required — no ground truth for final answer |
| RL training signal | PRM gives denser gradient signal, faster learning |

---

## 7. Reasoning Architectures: o1 / DeepSeek-R1 Style

### Core Insight

Rather than optimizing for short answers, train models to produce extended internal reasoning ("thinking") before committing to a final answer. More thinking tokens → higher accuracy on hard problems.

$$\text{Performance} \approx f(\log(\text{thinking tokens}))$$

### DeepSeek-R1 Training Pipeline

```
Base model
    │
    ▼
Cold start SFT on <think>...</think> formatted examples
    │
    ▼
GRPO (Group Relative Policy Optimization) with:
    ├── Format reward: correct <think>...</think><answer>...</answer> structure
    ├── Accuracy reward: correct final answer (verified)
    └── KL penalty: stay close to reference
    │
    ▼
Rejection sampling: collect high-quality (problem, CoT, answer) triples
    │
    ▼
SFT on rejection-sampled data (distillation-ready)
    │
    ▼
DPO/RLHF for helpfulness, safety alignment
```

### GRPO vs PPO for Reasoning

GRPO (Group Relative Policy Optimization) eliminates the value model by using relative rewards within a group:

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$

where $r_{1..G}$ are rewards for $G$ completions of the same prompt. No value function needed — simpler than PPO, works well for math/code where rewards are binary.

```python
def grpo_advantage(rewards: list[float]) -> list[float]:
    """Normalize rewards within a group to get advantages."""
    import statistics
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
    return [(r - mean) / (std + 1e-8) for r in rewards]

# Training loop sketch
for problem in math_dataset:
    # Sample G completions
    completions = [policy.generate(problem, temperature=0.8) for _ in range(G)]
    # Verify each answer
    rewards = [verify_answer(problem, c) for c in completions]
    # Normalize within group
    advantages = grpo_advantage(rewards)
    # Policy gradient update
    for completion, advantage in zip(completions, advantages):
        loss = -advantage * policy.log_prob(completion | problem)
        loss.backward()
```

### Test-Time Compute Scaling

**Best-of-N:** generate $N$ candidates, select best using verifier or reward model.

```python
def best_of_n(model, reward_model, prompt: str, n: int = 16) -> str:
    candidates = [
        model.generate(prompt, temperature=0.8, max_new_tokens=512)
        for _ in range(n)
    ]
    scores = [reward_model.score(prompt, c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

**MCTS for reasoning:** explore multiple reasoning paths, backpropagate correctness signal, prune poor branches.

```python
class MCTSNode:
    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * (self.parent.visits ** 0.5) / self.visits
        return exploitation + exploration

def mcts_solve(model, verifier, problem: str, n_simulations: int = 100) -> str:
    root = MCTSNode(state=problem)

    for _ in range(n_simulations):
        # Selection: traverse using UCB
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score())

        # Expansion: generate next reasoning step
        if not is_terminal(node.state):
            new_steps = model.generate_steps(node.state, n=3)
            for step in new_steps:
                node.children.append(MCTSNode(state=node.state + step, parent=node))
            node = node.children[0]

        # Simulation: rollout to terminal
        rollout_state = node.state
        while not is_terminal(rollout_state):
            rollout_state += model.generate_step(rollout_state)

        # Backpropagation
        reward = verifier.score(rollout_state)
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Return best terminal state
    return max(
        (n for n in get_leaves(root)),
        key=lambda n: n.value / max(n.visits, 1)
    ).state
```

**Effective compute scaling** (GSM8K, 7B model):

| Method | Approx. compute vs greedy | Accuracy |
| :--- | :--- | :--- |
| Greedy decoding | 1× | 68% |
| Best-of-8 | 8× | 79% |
| Best-of-32 | 32× | 84% |
| MCTS (100 sims) | ~50× | 87% |

---

## 8. ReAct Framework

Interleaves **Re**asoning (Thought) and **Act**ion (Act/Obs) steps, allowing models to use tools mid-reasoning:

```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population 2024")
Observation: Tokyo metropolitan area population is approximately 37.4 million.
Thought: Now I can answer the question.
Action: finish("Tokyo's population is approximately 37.4 million.")
```

```python
import re

TOOLS = {}

def tool(name):
    def decorator(fn):
        TOOLS[name] = fn
        return fn
    return decorator

@tool("search")
def search(query: str) -> str:
    # Wikipedia / web search stub
    return web_search(query)

@tool("calculate")
def calculate(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool("finish")
def finish(answer: str) -> str:
    return answer

ACTION_RE = re.compile(r"Action:\s*(\w+)\((.+)\)")

def react_agent(model, question: str, max_steps: int = 10) -> str:
    context = f"Question: {question}\n"
    for step in range(max_steps):
        response = model.generate(context + "Thought:")
        context += "Thought:" + response + "\n"

        match = ACTION_RE.search(response)
        if match:
            action_name, action_input = match.group(1), match.group(2).strip('"\'')
            if action_name == "finish":
                return action_input
            if action_name in TOOLS:
                observation = TOOLS[action_name](action_input)
                context += f"Observation: {observation}\n"
            else:
                context += f"Observation: Unknown tool '{action_name}'\n"

    return "Max steps reached without answer."
```

---

## 9. Alignment Failure Modes and Mitigations

| Failure Mode | Description | Mitigation |
| :--- | :--- | :--- |
| **Sycophancy** | Agrees with user regardless of truth | Include disagreement examples in SFT; penalize agreement with false premises in RM |
| **Reward hacking** | Exploits reward model blind spots | KL penalty, KL monitoring, diverse reward models |
| **Hallucination** | Confident fabrication | RAG, calibration training, RLHF with factual accuracy reward |
| **Jailbreaking** | Bypasses safety via prompt manipulation | Adversarial training, red-teaming, robust safety classifiers |
| **Specification gaming** | Satisfies letter but not spirit of instruction | Constitutional AI critique, iterative red-teaming |
| **Distributional shift** | Good on alignment data, bad on novel prompts | Diverse evaluation, holdout distributions |

---

## 10. Alignment Hyperparameter Reference

| Parameter | Typical range | Effect |
| :--- | :--- | :--- |
| SFT learning rate | 1e-5 to 2e-5 | Higher → faster convergence, risk of forgetting |
| DPO beta | 0.01–0.5 | Higher → stay closer to reference, less change |
| PPO KL coef (beta) | 0.1–0.5 | Higher → more conservative policy updates |
| PPO clip epsilon | 0.1–0.2 | Smaller → more conservative updates |
| Reward model size | Same as policy | Larger reward model → better signal, more memory |
| DPO epochs | 1–2 | More → overfitting to preference data |
| GRPO group size G | 8–32 | Larger → more stable advantage estimates |

> [!TIP]
> **Interview structure:** Alignment = three levels: (1) format via SFT, (2) helpfulness via RLHF/DPO, (3) safety via CAI/red-teaming. RLHF is more powerful but complex (4 models, online sampling). DPO is simpler and competitive (2 models, offline). Reasoning = test-time compute: more thinking tokens → better accuracy, with self-consistency and MCTS as the extremes. The frontier is PRMs + GRPO + long CoT, as demonstrated by o1 and R1.