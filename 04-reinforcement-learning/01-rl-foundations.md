---
module: Reinforcement Learning
topic: RL Foundations
subtopic: ""
status: unread
tags: [rl, mdp, bellman, value-functions]
prerequisites: [probability, expectation, dynamic-programming]
---
# RL Foundations — MDPs, Returns, and the Bellman Equations

---

## The Problem It Solves

Supervised learning needs labels: for every input, someone tells you the right answer. An enormous class of problems has no such labels. What is the "correct" move in chess at turn 14? The correct action for a game's matchmaking system? Nobody can label these, because the quality of a decision depends on everything that happens *afterward*.

Reinforcement learning replaces the label with a **reward signal** and asks a different question: not "what is the right answer here?" but "what sequence of decisions maximizes total reward over time?"

Two properties make this genuinely harder than supervised learning, and interviewers probe both:

1. **Delayed consequence.** A move that looks bad now may win the game twenty turns later. The learning signal arrives long after the decision that caused it — the *credit assignment problem*.
2. **The data depends on the policy.** In supervised learning your dataset is fixed. In RL, the actions you take determine what you observe next. A bad policy collects bad data, which teaches a bad policy. The i.i.d. assumption underpinning most of ML is gone.

---

## Intuition

Think of training a dog with treats. You never explain the desired behavior — you only reward outcomes. The dog explores, occasionally does the right thing, gets a treat, and gradually shifts toward behaviors that historically produced treats.

Two things make this hard, and they map exactly onto the technical problems:

- The dog must figure out *which* of its recent actions earned the treat (credit assignment).
- The dog must occasionally try something new, or it will lock onto the first mediocre behavior that ever worked (exploration vs. exploitation).

---

## The Mechanics

### The Markov Decision Process

An MDP is the formal container for an RL problem — the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Symbol | Name | Meaning |
| :--- | :--- | :--- |
| $\mathcal{S}$ | State space | The situations the agent can be in |
| $\mathcal{A}$ | Action space | The choices available |
| $P(s' \mid s, a)$ | Transition dynamics | Probability of landing in $s'$ after taking $a$ in $s$ |
| $R(s, a)$ | Reward function | Immediate scalar feedback |
| $\gamma \in [0, 1]$ | Discount factor | How much future reward is worth relative to now |

**The Markov property** is the load-bearing assumption: the future depends only on the current state, not the history. $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_1, a_1, \ldots, s_t, a_t)$.

This is an assumption about your *state representation*, not about the world — and that distinction is where practitioners get burned. If your state is a single video frame, velocity is unobservable and the problem is non-Markovian. Stack four frames and it becomes Markovian again. When someone says "the Markov assumption doesn't hold here," the real question is usually "what's missing from your state?"

### Return: what we actually maximize

The **return** is cumulative discounted future reward from time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

$\gamma$ does three jobs at once, which is worth being able to state cleanly:

1. **Mathematical:** guarantees the infinite sum converges (for bounded rewards, $\gamma < 1$).
2. **Modeling:** encodes genuine preference for sooner reward — money now beats money later.
3. **Practical:** limits the effective planning horizon to roughly $1/(1-\gamma)$ steps, which controls variance.

At $\gamma = 0.99$ the horizon is ~100 steps; at $\gamma = 0.9$ it is ~10. Tuning $\gamma$ is tuning how far ahead the agent bothers to look.

### Policies and value functions

A **policy** $\pi(a \mid s)$ maps states to action probabilities. It is the thing we are learning.

Two value functions measure how good things are under a policy:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] \qquad Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

$V$ answers "how good is this state?"; $Q$ answers "how good is this action in this state?" Their difference is the **advantage**, $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — how much better an action is than the policy's average behavior in that state. Advantage is central to policy gradient methods, and the reason is variance: it subtracts off the part of the return that has nothing to do with the action chosen.

**Why $Q$ is often more useful than $V$:** if you have $Q$, you can act greedily without knowing the transition dynamics — just take $\arg\max_a Q(s,a)$. With only $V$, choosing an action requires knowing where each action leads, i.e. a model of $P$. This is precisely why Q-learning is model-free and can be run on a system whose dynamics you never write down.

### The Bellman equations

The recursive structure that makes everything computable — the value of a state is the immediate reward plus the discounted value of where you land:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s,a)\left[R(s,a) + \gamma V^\pi(s')\right]$$

The **Bellman optimality equation** replaces the policy average with a max:

$$Q^*(s,a) = \sum_{s'} P(s' \mid s,a)\left[R(s,a) + \gamma \max_{a'} Q^*(s',a')\right]$$

That $\max$ is the entire difference between evaluating a policy and finding the best one — and it is also what makes the optimality equation nonlinear and thus not directly solvable by linear algebra.

---

## Worked Example

A three-state chain. $\gamma = 0.9$. From $S_1$ you may go **right** (reward 0, to $S_2$) or **stay** (reward 1, to $S_1$). $S_2$ transitions to terminal $S_3$ with reward 10.

Work backward — the only reliable way to do these by hand.

**Terminal:** $V(S_3) = 0$

**State $S_2$:** one action, reward 10, into terminal.
$$V(S_2) = 10 + 0.9 \times 0 = 10$$

**State $S_1$:** compare the two actions.
$$Q(S_1, \text{right}) = 0 + 0.9 \times V(S_2) = 0 + 0.9 \times 10 = 9$$

For **stay**, the agent loops on itself forever, so $V$ appears on both sides:
$$Q(S_1, \text{stay}) = 1 + 0.9 \times V(S_1)$$

If staying were optimal, $V(S_1) = 1 + 0.9V(S_1)$, giving $V(S_1) = 1/0.1 = 10$.

So staying forever is worth **10**, and going right is worth **9**. Greedy short-horizon intuition says "reward 1 beats reward 0," and here that intuition happens to be right — but only because $\gamma$ is high enough to make an infinite trickle of 1s outweigh a single 10.

**Now set $\gamma = 0.5$ and redo it:**
- $Q(S_1, \text{right}) = 0 + 0.5 \times 10 = 5$
- Staying: $V = 1 + 0.5V \Rightarrow V = 2$

The optimal policy **flips to "right"**. Same MDP, same rewards, different discount factor, opposite behavior.

This is the example worth remembering: $\gamma$ is not a minor hyperparameter that trades off convergence speed. It changes which policy is optimal.

---

## When It Breaks

| Failure | Mechanism | What you see |
| :--- | :--- | :--- |
| **Non-Markovian state** | State omits information needed to predict the future | Agent plateaus well below achievable performance; behavior looks "confused" in specific situations |
| **Reward hacking** | Agent maximizes the literal reward, not your intent | High reward, useless or absurd behavior — the boat that spins collecting powerups instead of finishing the race |
| **Sparse reward** | Reward only at the end; random exploration never reaches it | No learning at all — flat curve from step zero |
| **Discount mis-set** | Horizon too short to see the real payoff | Myopic policy; agent takes small immediate gains |
| **Poor exploration** | Locks onto first adequate behavior | Premature convergence to a mediocre policy |

**Reward specification is the hardest unsolved part in practice.** Anything you fail to penalize, the optimizer is free to exploit. In production this is why RL systems ship with heavy guardrails and constraint penalties rather than a single clean objective.

---

## Production Notes

- **Sample efficiency is the binding constraint.** RL routinely needs 10⁶–10⁸ environment interactions. Viable when interaction is cheap (a simulator); often prohibitive when it is not (real users, physical robots).
- **Simulator fidelity determines transfer.** A policy trained in simulation inherits every inaccuracy in that simulation. Domain randomization — training across randomized simulator parameters — is the standard mitigation.
- **Offline RL when you cannot explore.** If you have logged data but cannot let a learning agent loose on live users, offline RL (CQL, IQL) learns from the fixed dataset. The core difficulty is *distribution shift*: the learned policy wants to take actions your logs never recorded, so its value estimates there are unconstrained fantasy.
- **Prefer bandits when there is no state.** If the action does not change the next situation, you have a bandit problem, not a full RL problem — and bandits are dramatically cheaper and safer. See [02-bandits-and-exploration.md](02-bandits-and-exploration.md).
- **RL is rarely the first tool.** Most problems posed as RL are better served by supervised learning or a bandit. Reaching for RL when a simpler method suffices is itself an interview red flag.

---

## Interview Angles

### Q: What is the Markov property, and why does it matter? [Easy]

The future depends only on the present state, not the path taken to reach it. It matters because every RL algorithm's correctness rests on it — value functions are defined on states, so if the state doesn't capture what's needed to predict the future, the value function is learning something incoherent.

**Cross-questions to expect:**
- *"Give me a case where it fails."* → A single frame of Pong: you can see the ball's position but not its velocity, so you cannot predict the next state. Fixed by stacking consecutive frames.
- *"So is it a property of the world or the representation?"* → The representation. Almost any process can be made Markovian by enriching the state — in the limit, include the whole history. The engineering question is how much history you actually need.
- *"What if you can't observe the full state?"* → That's a POMDP. Standard responses: stack frames, or use a recurrent policy that maintains a belief state internally.

**Trap:** Calling a problem "non-Markovian" as though it were a fixed property of the environment. It's almost always a statement about an inadequate state representation.

---

### Q: Why discount future rewards? [Easy]

Three reasons, and a strong answer gives more than one: it makes infinite-horizon sums converge; it encodes real preference for sooner reward; and it bounds the effective planning horizon to about $1/(1-\gamma)$ steps, which controls variance in the return estimates.

**Cross-questions to expect:**
- *"What happens at γ = 0? At γ = 1?"* → At 0, purely myopic — maximize immediate reward only. At 1, all future reward counts equally; only valid for guaranteed-terminating episodes, otherwise the sum diverges.
- *"How would you choose γ for a specific problem?"* → From the timescale on which the real payoff arrives. If reward lands ~100 steps after the decision causing it, γ ≈ 0.99. Too low and the agent literally cannot see the consequence.
- *"Can a lower γ ever perform better?"* → Yes. Lower γ means lower-variance returns and easier credit assignment. It's a bias–variance trade — a common practical trick is to train with a lower γ than the one you actually care about.

**Trap:** Treating γ as purely a convergence device. The worked example above flips the optimal policy from "stay" to "right" purely by changing γ — it is part of the problem definition, not a training detail.

---

### Q: What's the difference between V(s) and Q(s,a), and when would you want each? [Medium]

$V$ is the expected return from a state under the policy; $Q$ conditions additionally on the first action. The practical difference: $Q$ lets you act greedily without a model of the environment, because $\arg\max_a Q(s,a)$ needs no knowledge of where actions lead. With only $V$, action selection requires the transition dynamics.

**Cross-questions to expect:**
- *"Then why does anything use V?"* → $V$ has $|\mathcal{S}|$ values against $Q$'s $|\mathcal{S}| \times |\mathcal{A}|$ — far cheaper for large action spaces, and a much easier regression target. Actor-critic methods learn $V$ as a baseline and get action selection from the separate policy network.
- *"What's the advantage function and why does it help?"* → $A(s,a) = Q(s,a) - V(s)$. Subtracting $V$ removes the state's baseline value, which is the same for every action and therefore pure noise in the gradient. Big variance reduction, no bias introduced.
- *"How do you get Q from V?"* → $Q(s,a) = \mathbb{E}_{s'}[R + \gamma V(s')]$ — which requires the model. That requirement is exactly why model-free control learns $Q$ directly.

---

### Q: Why can't you just use supervised learning if you have logged data from a deployed policy? [Hard]

Because supervised learning on logged actions reproduces the *logging policy*, including its mistakes — it's imitation, not improvement. It can never exceed the behavior that generated the data. Worse, the log is biased: it only contains outcomes for actions the old policy chose, so you have no counterfactual evidence about the actions it avoided.

**Cross-questions to expect:**
- *"So how do you evaluate a new policy offline?"* → Off-policy evaluation: importance sampling reweights logged outcomes by the ratio $\pi_{\text{new}}(a|s) / \pi_{\text{old}}(a|s)$. Variance explodes when the policies disagree, so doubly-robust estimators are usually preferred.
- *"What breaks in offline RL specifically?"* → Distributional shift. The learned policy assigns high value to out-of-distribution actions precisely because nothing in the data contradicts it. CQL and IQL exist to penalize exactly this over-optimism.
- *"When is imitation actually the right call?"* → When you have expert demonstrations and the goal is to match, not exceed, them. Behavioral cloning is far simpler and safer; RLHF's supervised fine-tuning stage is exactly this.

**Trap:** Claiming logged data makes it "just a supervised problem." The interviewer is testing whether you see the counterfactual gap — you only observe the reward for the action that was actually taken.

---

### Q: A game team wants RL for difficulty balancing. Talk me through whether that's the right tool. [Hard]

Start by interrogating the framing rather than accepting it — that is what's being tested.

**Does the action change the next state?** If difficulty adjustment affects the player's future engagement trajectory, it's sequential and RL applies. If you're just picking a difficulty per session and observing an immediate outcome, it's a **contextual bandit** — vastly cheaper, safer, and more sample-efficient.

**What's the reward?** This is where these systems fail. "Engagement" naively defined produces an agent that makes the game frustrating-but-addictive, because frustration drives retries and retries look like engagement. Reward specification is the real design work.

**Can you afford exploration?** RL learns by trying suboptimal actions. On live players that means deliberately degrading some players' experience. This is the constraint that usually decides the answer.

**Cross-questions to expect:**
- *"How would you handle the exploration cost?"* → Train in simulation against player models first; deploy with constrained exploration on a small traffic slice; use offline RL on historical logs before any live interaction.
- *"What reward would you actually use?"* → A composite with explicit guardrails — long-horizon retention as the primary term, penalties for rage-quit and for difficulty variance. Then verify against holdout metrics the agent cannot optimize.
- *"How do you know it's working?"* → A/B test against the existing heuristic on business metrics, not on RL return. Return is measured against your possibly-wrong reward function; it is not evidence of success.

**Trap:** Jumping straight to algorithm choice (PPO vs. DQN). The senior answer establishes whether RL is warranted at all, and most interviewers are specifically checking whether you reach for a bandit first.

---

## Connections

- **Next:** [02-bandits-and-exploration.md](02-bandits-and-exploration.md) — the stateless special case, and the one you should usually try first
- **Value methods:** [03-value-based-methods.md](03-value-based-methods.md) — Q-learning, DQN, and the deadly triad
- **Policy methods:** [04-policy-gradient-methods.md](04-policy-gradient-methods.md) — REINFORCE, actor-critic, PPO
- **RLHF:** [../05-llms/](../05-llms/) — where these ideas meet language models
- **Probability background:** [../01-foundations/02-math-and-theory-foundations.md](../01-foundations/02-math-and-theory-foundations.md)
