---
module: Reinforcement Learning
topic: Policy Gradient Methods
subtopic: ""
status: unread
tags: [rl, reinforce, actor-critic, ppo, trpo, sac, rlhf]
prerequisites: [rl-foundations, value-based-methods, gradient-descent]
---
# Policy Gradient Methods — REINFORCE, Actor-Critic, and PPO

---

## The Problem It Solves

Value-based methods derive a policy indirectly: learn $Q$, then act greedily. That breaks down in three places.

**Continuous actions.** $\arg\max_a Q(s,a)$ requires enumerating actions. A robot arm with seven joint torques has an uncountable action space; you cannot take the max.

**Stochastic optimal policies.** Greedy policies are deterministic. In rock-paper-scissors the optimal policy is uniformly random, and no deterministic policy achieves it. Any partially observed environment can make randomization genuinely optimal.

**Indirection.** A tiny change in $Q$ can flip the argmax and change the policy discontinuously — one source of the instability in the previous file.

Policy gradient methods parameterize the policy $\pi_\theta(a \mid s)$ directly and ascend the gradient of expected return. The policy changes smoothly, handles continuous actions natively, and can be stochastic by construction.

This is also the family that matters most for LLMs: **RLHF is PPO applied to a language model**, where the action space is the vocabulary and the reward comes from a learned preference model.

---

## Intuition

You're coaching without knowing the rules of the game. You watch a full match, see the score, and adjust: whatever the player did in a winning match, tell them to do more of that; in a losing match, less. Repeat over many matches, and the noise averages out — moves that genuinely help are reinforced, coincidences wash away.

That's REINFORCE. The refinement everyone actually uses: don't compare against zero, compare against *how well you expected to do*. A win against a strong opponent should be reinforced more than a win against a weak one. That expectation is the **baseline**, and subtracting it is what makes the method practical rather than merely correct.

---

## The Mechanics

### The policy gradient theorem

The objective is expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. The gradient looks intractable — the distribution being sampled from depends on $\theta$ — but the log-derivative trick ($\nabla p = p \nabla \log p$) resolves it:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$

The result is remarkable: **no derivative of the environment dynamics appears**. You never need to know or differentiate through $P(s' \mid s,a)$. This is what makes model-free RL possible.

Read the update as: push up the log-probability of actions, weighted by how good the return was. Actions preceding high return get more likely; actions preceding low return get less.

### REINFORCE and the variance problem

```python
def reinforce_loss(log_probs, returns):
    # returns: discounted return from each timestep onward
    return -(log_probs * returns).sum()      # negative: ascend by descending
```

REINFORCE is unbiased and nearly unusable on its own. $G_t$ is the sum of many random rewards along a whole trajectory, so its variance is enormous — and it scales with episode length. Worse, if all returns are positive, *every* action's probability is pushed up; learning depends entirely on relative magnitudes surviving the noise.

**Baselines fix this.** Subtract any function of state $b(s)$:

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\left(G_t - b(s_t)\right)\right]$$

This stays unbiased for *any* $b$ that doesn't depend on the action, because $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a\mid s)] = 0$. The variance-minimizing choice is $b(s) = V(s)$, which makes the weight the **advantage** $A(s,a) = Q(s,a) - V(s)$: *how much better was this action than average from this state.*

Now the sign is meaningful. Positive advantage means better than expected — reinforce. Negative means worse — suppress.

### Actor-critic

Learn the baseline instead of estimating it from returns. Two networks:

- **Actor** $\pi_\theta(a\mid s)$ — picks actions
- **Critic** $V_\phi(s)$ — estimates state value, trained by TD regression

The advantage becomes the TD error from the previous file:

$$\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**GAE (Generalized Advantage Estimation)** interpolates across $n$-step advantages with a decay $\lambda$, giving an explicit bias–variance dial: $\lambda = 0$ is the one-step TD advantage (low variance, biased), $\lambda = 1$ recovers Monte Carlo (unbiased, high variance). $\lambda \approx 0.95$ is the standard default.

### Trust regions: TRPO and PPO

Policy gradients have a failure mode value methods don't: a single oversized update can collapse the policy, and because the policy generates its own training data, there's no recovery. A bad policy collects bad data, which produces worse updates.

**TRPO** solves this with a hard KL constraint — maximize improvement subject to $\text{KL}(\pi_{\text{old}} \| \pi_\theta) < \delta$. Principled, with a monotonic improvement guarantee, but it needs conjugate gradients and a Hessian-vector product. Painful to implement.

**PPO** achieves nearly the same effect with a clipped objective and first-order optimization only:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$ is the probability ratio and $\epsilon \approx 0.2$.

```python
def ppo_loss(logp, logp_old, adv, eps=0.2):
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()      # min ⇒ pessimistic bound
```

The `min` is the whole design. When the advantage is positive, the objective is capped at $(1+\epsilon)\hat{A}$ — no reward for pushing the probability far beyond the old policy. When negative, capped at $(1-\epsilon)\hat{A}$. Either way the incentive to move far from $\pi_{\text{old}}$ is removed, so multiple epochs of minibatch updates on the same batch stay safe. That reuse is why PPO is far more sample-efficient than vanilla policy gradient.

### The continuous-control family

| Method | On/Off-policy | Key idea |
| :--- | :--- | :--- |
| **DDPG** | Off | Deterministic actor + Q-critic; DQN tricks for continuous actions |
| **TD3** | Off | DDPG + twin critics (min of two, fixing overestimation) + delayed actor updates |
| **SAC** | Off | Maximum-entropy objective — maximize return *and* policy entropy |

**SAC** is the usual default for continuous control. Adding an entropy bonus $\alpha\mathcal{H}(\pi(\cdot\mid s))$ to the objective makes exploration intrinsic rather than a bolt-on, and it's markedly less hyperparameter-sensitive than DDPG. Being off-policy, it's also far more sample-efficient than PPO — which matters when each sample is a real robot movement.

---

## Worked Example

Three actions from one state. Current policy $\pi = [0.5, 0.3, 0.2]$. One episode samples $a_2$ and receives return $G = 10$.

**Without a baseline**, the gradient weight is $G = 10$, positive, so $\log \pi(a_2)$ is pushed up. After a step (schematically) $\pi \approx [0.45, 0.38, 0.18]$.

Now suppose the critic estimates $V(s) = 12$ — from this state, average return is 12.

**With the baseline**, the weight is $A = 10 - 12 = -2$. Negative. The probability of $a_2$ is pushed *down*: $\pi \approx [0.52, 0.28, 0.20]$.

**Opposite update from identical data.** A return of 10 is only good relative to something. Against an expectation of 12, taking $a_2$ was a mistake and should be made less likely — even though the raw reward was positive.

This is also why "all rewards positive" breaks naive REINFORCE. Without a baseline every sampled action is reinforced, and the policy improves only through the *relative* size of pushes — a much weaker and noisier signal than a correctly signed one.

---

## When It Breaks

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Entropy collapses to ~0 early | Premature convergence to a deterministic policy | Entropy bonus, lower LR, check reward scale |
| Reward improves then crashes irrecoverably | Update too large; policy collapsed and now collects bad data | Lower $\epsilon$, fewer epochs per batch, check KL |
| Extremely noisy learning curve | Advantage variance | GAE, normalize advantages per batch, larger batch |
| Learns nothing, loss near zero | Advantages all ≈ 0 — critic memorized returns | Check critic isn't overfitting; verify reward reaches the agent |
| PPO ratio explodes | $\pi_{\text{old}}$ stale across too many epochs | Fewer epochs, early-stop on KL threshold |

**Structural limitations:**

- **On-policy means sample-hungry.** PPO must discard data after each policy update. If environment steps are expensive (robots, real users), prefer SAC or offline RL.
- **Local optima are real.** Policy gradients ascend a non-convex objective and will happily converge to a mediocre stable strategy.
- **Reward hacking is the dominant real-world failure.** The policy optimizes precisely what you wrote, which is rarely quite what you meant. In RLHF this shows up as the model exploiting quirks of the reward model — verbose, sycophantic, confidently formatted answers that score well without being better.

---

## Production Notes

- **Normalize advantages per batch** ($\hat{A} \leftarrow (\hat{A} - \mu)/\sigma$). Small change, large stability gain, essentially free.
- **Monitor KL divergence, not just reward.** KL between old and new policy is the leading indicator of collapse; reward tells you only after the damage. Early-stop the epoch loop when KL exceeds a threshold (~0.01–0.02).
- **Track entropy every run.** A collapsing entropy curve predicts a stuck policy well before the reward curve flattens.
- **PPO's implementation details matter more than the algorithm.** Observation normalization, orthogonal init, LR annealing, value-loss clipping, and reward scaling account for much of the reported performance gap between implementations. This is documented and reproducible — treat published PPO numbers as claims about a *codebase*, not just an algorithm.
- **RLHF specifics:** the reward is a learned model, so it can be over-optimized. A KL penalty against the frozen reference model is the standard guard. DPO removes the RL loop entirely by reparameterizing the preference objective — simpler and often competitive.

---

## Interview Angles

### Q: Why use policy gradients instead of Q-learning? [Easy]

Three reasons: continuous action spaces (no $\arg\max$ over an uncountable set), stochastic optimal policies (greedy policies are deterministic, and sometimes randomization is genuinely optimal), and smoother optimization (the policy changes continuously rather than flipping when the argmax flips).

**Cross-questions to expect:**
- *"When would you prefer Q-learning?"* → Discrete actions plus expensive environment steps. Off-policy replay makes DQN far more sample-efficient than on-policy PPO.
- *"Example where a stochastic policy is strictly optimal?"* → Rock-paper-scissors, or any partially observed state where two distinct situations look identical to the agent — randomizing hedges across them.
- *"Can Q-learning handle continuous actions at all?"* → With effort: DDPG/TD3 learn a deterministic actor to approximate the argmax. That's a hybrid, not pure value-based.

---

### Q: Why does REINFORCE need a baseline, and why doesn't it introduce bias? [Medium]

Variance. $G_t$ sums many stochastic rewards over a trajectory, so its variance grows with episode length and swamps the signal. Subtracting a baseline recenters the weights so the sign carries information.

Unbiasedness follows from $\mathbb{E}_{a\sim\pi}[\nabla_\theta \log \pi_\theta(a\mid s)] = \nabla_\theta \sum_a \pi_\theta(a\mid s) = \nabla_\theta 1 = 0$. Any $b(s)$ multiplied by something with zero expectation contributes zero to the gradient in expectation, while reducing variance.

**Cross-questions to expect:**
- *"What's the optimal baseline?"* → Variance-minimizing is a value weighted by squared gradient norm, but $V(s)$ captures nearly all the benefit and is what everyone uses. It makes the weight the advantage.
- *"What if the baseline depended on the action?"* → It would bias the gradient. The proof requires $b$ to come outside the expectation over actions.
- *"Concretely, why does it matter?"* → With all-positive returns and no baseline, every sampled action gets reinforced. Learning relies on relative magnitudes surviving noise, which is far weaker than a correctly signed update.

**Trap:** Saying the baseline "makes it converge faster." It reduces variance in the gradient *estimator*; the expected gradient is unchanged. Precision here separates a memorized answer from an understood one.

---

### Q: Explain PPO's clipped objective. Why the `min`? [Medium]

PPO limits how far the policy moves from the one that collected the data. The ratio $r_t = \pi_\theta/\pi_{\theta_{\text{old}}}$ measures the shift; the objective takes the min of the unclipped and clipped surrogate.

The `min` makes the objective a *pessimistic lower bound*. With positive advantage, gain is capped at $(1+\epsilon)\hat{A}$ — no further reward for pushing probability higher. With negative advantage, capped at $(1-\epsilon)\hat{A}$. Either direction, the incentive to move far from $\pi_{\text{old}}$ vanishes, so you can run several epochs of minibatch SGD on one batch without the policy running away.

**Cross-questions to expect:**
- *"Why not just clip the ratio without the min?"* → Clipping alone gives zero gradient outside the range even when the policy has moved in the *wrong* direction, stranding it. The min ensures a bad move always retains a corrective gradient.
- *"How does this relate to TRPO?"* → Same goal — bound the policy change — but TRPO enforces a hard KL constraint via second-order optimization; PPO approximates it with a first-order clip. PPO is far easier to implement and nearly as effective, which is why it won.
- *"What does $\epsilon$ control?"* → Trust region size. Smaller is more conservative and slower; larger risks collapse. 0.2 is the default, 0.1 typical for RLHF where stability dominates.
- *"Does clipping guarantee a bounded KL?"* → No. It's a heuristic, not a constraint — the ratio is bounded per sample, but KL can still drift. That's why production PPO monitors KL explicitly and early-stops.

---

### Q: You're doing RLHF on an LLM. Reward-model score climbs steadily, but human evaluators rate the newer model worse. What's happening? [Hard]

Classic **reward over-optimization** — Goodhart's law with a learned proxy. The reward model is a finite approximation of human preference trained on a fixed dataset. The policy optimizes the proxy, not the target, and past some point it finds regions where the proxy is high and true quality is not. Rising RM score is then evidence of exploitation, not improvement.

**How I'd diagnose it:**
1. **Check KL from the reference model.** If it's large and growing, the policy has drifted into territory the RM never saw labeled data for — its estimates there are unreliable extrapolation.
2. **Inspect samples directly.** Over-optimization has recognizable signatures: excessive length, hedging, formulaic openers, sycophancy, over-formatting. These are RM biases the policy has learned to trigger.
3. **Plot human rating against KL.** The typical shape is an inverted U — true quality peaks at moderate KL and declines after, while RM score climbs monotonically. That divergence point is the answer.
4. **Check RM calibration on fresh on-policy samples**, not the original held-out set. Accuracy on old data says nothing about the distribution the policy now occupies.

**What I'd do:**
- Increase the KL penalty, or early-stop at the KL where human ratings peaked
- Retrain the RM with fresh preference labels on current on-policy outputs — the standard iterated-RLHF loop
- Use an ensemble of reward models and optimize a conservative statistic (e.g. the minimum) to penalize disagreement
- Add explicit length normalization or debiasing if length correlates with reward
- Consider DPO, which avoids a separately exploitable reward model, though it doesn't eliminate the underlying proxy problem

**Cross-questions to expect:**
- *"Why does the KL penalty help?"* → It keeps the policy near the distribution where the RM was trained and is therefore accurate. It's a trust region in output space, exactly analogous to PPO's clip in parameter space.
- *"Isn't a large KL sometimes what you want?"* → Yes — a big genuine improvement requires real movement. That's why you tune against human evaluation rather than fixing KL a priori; the penalty coefficient is an empirical choice.
- *"How would you catch this before shipping?"* → Human eval on a held-out prompt set at multiple KL checkpoints throughout training, not just at the end. The RM curve alone cannot detect its own over-optimization — by construction.

**Trap:** Treating this as a bug in PPO or a hyperparameter problem. PPO is doing exactly its job; the fault is that the objective is a proxy. Reaching for the learning rate signals you've missed the point.

---

## Connections

- **Foundations:** [01-rl-foundations.md](01-rl-foundations.md) — advantage function, why $A = Q - V$
- **Value methods:** [03-value-based-methods.md](03-value-based-methods.md) — the critic is TD learning
- **Exploration:** [02-bandits-and-exploration.md](02-bandits-and-exploration.md)
- **RLHF in context:** [../05-llms/interview-notes/17-advanced-alignment-and-reasoning.md](../10-llms/interview-notes/17-advanced-alignment-and-reasoning.md)
- **RLHF in the training pipeline:** [../05-llms/01-training-process.md](../10-llms/01-training-process.md)
