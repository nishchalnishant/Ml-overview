---
module: Reinforcement Learning
topic: Value-Based Methods
subtopic: ""
status: unread
tags: [rl, q-learning, dqn, td-learning, deadly-triad]
prerequisites: [rl-foundations, bellman-equations, neural-networks]
---
# Value-Based Methods — TD Learning, Q-Learning, and DQN

---

## The Problem It Solves

The Bellman optimality equation tells you what $Q^*$ satisfies, but not how to compute it when you don't know the transition dynamics $P(s' \mid s, a)$ — which is nearly always. Value-based methods learn $Q$ directly from experience, without ever building a model of the environment.

The central trick is **bootstrapping**: instead of waiting for an episode to finish to learn the true return, update your current estimate using your *own* estimate of the next state's value. You learn a guess from a guess — which sounds unstable, and under specific conditions genuinely is. Understanding when it works and when it fails is what this file is about.

---

## Intuition

You're learning a city's commute times. After driving one route, you could wait until you arrive to learn anything (Monte Carlo), or you could update as you go: "I expected 30 minutes total; I'm 10 minutes in and my remaining-time estimate now says 25, so my original estimate was 5 minutes optimistic." You correct immediately, using your own estimate of what remains.

That's temporal-difference learning. You don't need the trip to end to learn from it, and that's why TD methods work on continuing tasks where Monte Carlo cannot.

---

## The Mechanics

### TD learning and the TD error

The core update. Compare what you predicted against a one-step-better prediction:

$$V(s_t) \leftarrow V(s_t) + \alpha\underbrace{\left[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right]}_{\text{TD error } \delta_t}$$

The **TD error** $\delta_t$ is the surprise: how much better or worse things turned out than expected. It appears throughout RL — in actor-critic as the advantage estimate, and (notably) as a documented signal in dopamine neuron activity.

**TD vs. Monte Carlo** is a bias–variance trade:

| | Monte Carlo | TD |
| :--- | :--- | :--- |
| Target | Actual return $G_t$ | $r + \gamma V(s')$ |
| Bias | Unbiased | Biased (bootstraps an imperfect estimate) |
| Variance | High (whole trajectory of randomness) | Low (one transition) |
| Needs episode end | Yes | No |

TD's lower variance usually wins in practice. $n$-step returns and TD($\lambda$) interpolate between the extremes.

### Q-learning

Apply the same idea to $Q$, with a $\max$ over next actions to target the *optimal* policy:

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)\right]$$

Q-learning is **off-policy**: the $\max$ means it learns about the greedy policy regardless of the policy actually generating behavior. You can explore randomly and still converge to optimal $Q$ — which is what makes replay buffers and learning from logged data possible.

**SARSA** is the on-policy sibling, using the action actually taken instead of the max:

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\right]$$

The difference is not academic. On the classic cliff-walking task, Q-learning learns the optimal path along the cliff edge; SARSA learns a safer path further back — because SARSA's target accounts for the fact that an ε-greedy policy will *sometimes step off the cliff*. When exploration is costly and happening in the real world, SARSA's conservatism is often what you want.

### DQN: Q-learning with a neural network

Tabular Q-learning needs one entry per state-action pair, impossible for images. DQN replaces the table with a network $Q_\theta(s,a)$ — and naively doing so diverges. Three fixes made it work:

**1. Experience replay.** Store transitions in a buffer; sample minibatches uniformly. This breaks the temporal correlation between consecutive samples (restoring something closer to i.i.d.) and reuses each transition many times, which matters enormously for sample efficiency.

**2. Target network.** The bootstrap target $r + \gamma\max_{a'}Q_\theta(s',a')$ depends on the same parameters being updated, so the target moves every step — chasing a moving goal. DQN keeps a frozen copy $\theta^-$, updated every $C$ steps:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

**3. Reward clipping / normalization.** Keeps gradient magnitudes comparable across environments with wildly different reward scales.

```python
def dqn_loss(batch, q_net, target_net, gamma=0.99):
    s, a, r, s_next, done = batch
    q_pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():                                    # target is not differentiated
        q_next = target_net(s_next).max(dim=1).values
        target = r + gamma * q_next * (1 - done)             # no bootstrap past terminal
    return F.smooth_l1_loss(q_pred, target)                  # Huber: robust to outliers
```

### The improvements worth naming

| Variant | Fixes | Mechanism |
| :--- | :--- | :--- |
| **Double DQN** | Overestimation bias | Select action with online net, evaluate with target net |
| **Dueling DQN** | Wasted capacity | Separate $V(s)$ and $A(s,a)$ streams; most states don't need per-action detail |
| **Prioritized replay** | Uniform sampling wastes compute | Sample in proportion to TD error — learn more from surprising transitions |
| **Rainbow** | — | Combines six improvements; the standard strong baseline |

**Overestimation bias** is the one to understand rather than memorize. $\max_{a'} Q(s',a')$ over noisy estimates is biased upward: the max of noisy values exceeds the true max in expectation, because whichever action happens to be *overestimated* is the one selected. The error compounds through bootstrapping. Double DQN decouples selection from evaluation, so an action overestimated by the online network gets evaluated by an independent one.

---

## Worked Example

Two states, two actions, $\alpha = 0.5$, $\gamma = 0.9$. All $Q$ initialized to 0.

Transition observed: $(s_1, a_1) \to r = 5, s_2$, and currently $Q(s_2, \cdot) = [0, 3]$.

$$Q(s_1,a_1) \leftarrow 0 + 0.5\left[5 + 0.9 \times \max(0,3) - 0\right] = 0.5\left[5 + 2.7\right] = 3.85$$

Same transition again, now from $Q(s_1,a_1) = 3.85$:

$$Q(s_1,a_1) \leftarrow 3.85 + 0.5\left[5 + 2.7 - 3.85\right] = 3.85 + 0.5(3.85) = 5.775$$

And a third time:

$$Q(s_1,a_1) \leftarrow 5.775 + 0.5[7.7 - 5.775] = 6.7375$$

The estimate converges geometrically toward the fixed point $r + \gamma\max_{a'}Q(s_2,a') = 7.7$, closing half the remaining gap each update because $\alpha = 0.5$.

**What this shows:** the learning rate controls how fast you close the gap to the current target — but the target itself moves as $Q(s_2, \cdot)$ updates. In deep RL both are changing simultaneously, which is precisely why the target network exists.

---

## When It Breaks

### The deadly triad

Divergence is guaranteed possible when all three of these are present:

1. **Function approximation** (a neural network rather than a table)
2. **Bootstrapping** (targets built from your own estimates)
3. **Off-policy learning** (training data from a different policy than the one being evaluated)

Any two are safe. All three — which is exactly DQN — can diverge. This is Sutton & Barto's framing and a frequent senior-level interview question.

The mechanism: with a table, updating $Q(s,a)$ touches only that entry. With a network, it shifts *all* states sharing features. If that shift changes the bootstrap target in the same direction as the update, you get positive feedback and the values blow up. Off-policy data makes this worse, because the states whose values are being inflated may never be visited to provide a correction.

DQN's target network and replay buffer are mitigations, not solutions — they slow the feedback loop enough that training usually works in practice.

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Q-values grow without bound | Deadly triad feedback | Slower target updates, lower LR, Double DQN |
| Learns then collapses | Replay buffer full of stale off-policy data | Smaller buffer, or prioritized replay |
| Systematic overestimation | Max over noisy estimates | Double DQN |
| No learning on sparse reward | Random exploration never finds reward | Reward shaping, curiosity bonus, demonstrations |
| Unstable with large rewards | Gradient scale varies wildly | Reward clipping, Huber loss |

---

## Production Notes

- **Discrete actions only.** The $\max_{a'}$ requires enumerating actions. Continuous control needs policy gradients or DDPG/SAC — see [04-policy-gradient-methods.md](04-policy-gradient-methods.md).
- **Replay buffer is a memory decision.** A million Atari frames is ~7 GB uncompressed. Store `uint8` and convert on sample; don't store float32 observations.
- **Target-update cadence is a real hyperparameter.** Too frequent reintroduces instability; too infrequent slows learning. Soft updates ($\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$, $\tau \approx 0.005$) are often more stable than hard periodic copies.
- **Evaluate with a separate greedy policy.** Training-time returns are contaminated by ε-greedy exploration and will understate true performance.
- **Sample efficiency remains the constraint.** DQN needed ~200M frames for human-level Atari. If environment interaction is expensive, look at model-based RL or offline RL before committing.

---

## Interview Angles

### Q: What's the difference between Q-learning and SARSA? [Medium]

Q-learning's target uses $\max_{a'} Q(s',a')$ — the best available next action. SARSA uses $Q(s',a')$ for the action actually taken by the current policy. That makes Q-learning **off-policy** (learns the optimal policy regardless of behavior) and SARSA **on-policy** (learns the value of the policy it's actually running, exploration included).

**Cross-questions to expect:**
- *"Which gives a better policy?"* → Q-learning converges to optimal $Q^*$. But SARSA often produces better *online* performance during learning, because it accounts for exploration cost.
- *"Concrete example where they differ?"* → Cliff walking. Q-learning takes the optimal path along the cliff edge; SARSA takes a safer route because its target incorporates the ε-greedy chance of stepping off. If exploration happens on real hardware or real users, SARSA's conservatism is usually preferable.
- *"Why does off-policy matter practically?"* → It's what permits replay buffers and learning from logged data. On-policy methods must discard data after the policy updates.

---

### Q: Why does DQN need a target network? [Medium]

Without it, the regression target $r + \gamma\max_{a'}Q_\theta(s',a')$ depends on the very parameters being updated. Every gradient step moves the target, so the network chases a goal that shifts with it — the errors correlate and can amplify rather than cancel. Freezing a copy for $C$ steps gives a stationary target, turning each interval into an ordinary supervised regression problem.

**Cross-questions to expect:**
- *"What if you update the target too often?"* → It approaches no target network at all — instability returns. Too rarely, and you're regressing toward stale values, which slows learning.
- *"Alternative to periodic hard copies?"* → Polyak averaging: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ with small $\tau$. Smoother, often more stable, and standard in SAC and DDPG.
- *"Does the replay buffer solve the same problem?"* → No — different problem. Replay breaks *temporal correlation between samples*; the target network fixes *target non-stationarity*. Both are needed.

**Trap:** Saying the target network "prevents overfitting." It addresses non-stationary targets, not overfitting.

---

### Q: What is the deadly triad? [Hard]

Function approximation, bootstrapping, and off-policy learning. Any two together are safe; all three can diverge — and DQN has all three, which is why it needed the target network and replay buffer to work at all.

The mechanism is worth being able to explain: with a table, an update touches one entry. With a network, it moves every state sharing features. If that generalization shifts the bootstrap target in the same direction as the update, the error feeds back on itself. Off-policy data compounds it, because the inflated states may never be visited to generate a corrective signal.

**Cross-questions to expect:**
- *"Which component would you drop?"* → Depends on the constraint. Dropping off-policy (use on-policy PPO) is most common and is largely why PPO is the default for hard problems. Dropping bootstrapping means Monte Carlo — unbiased but high variance. Dropping function approximation isn't an option at scale.
- *"So how does DQN survive it?"* → It doesn't eliminate the problem; it slows the feedback loop. The target network delays propagation and replay decorrelates samples. Divergence is still possible, and in practice people still see Q-values explode.
- *"Anything with actual convergence guarantees?"* → Gradient-TD methods (GTD, TDC) are provably convergent under function approximation and off-policy sampling. They're rarely used in practice because they're slower and empirically weaker than DQN variants.

---

### Q: Your DQN's Q-values grow to 10⁶ while episode reward stays flat. Diagnose it. [Hard]

Values diverging while performance doesn't improve is the signature of the deadly triad producing a positive feedback loop, likely compounded by max-operator overestimation.

**How I'd work it:**
1. **Log the TD error distribution**, not just the loss. Systematically positive TD errors mean targets consistently exceed predictions — the hallmark of runaway bootstrapping.
2. **Compare predicted Q against actual discounted return** from evaluation episodes. A large gap quantifies the overestimation directly.
3. **Check the target-update period.** Too-frequent updates are among the most common causes.
4. **Check reward scale.** Unclipped large rewards produce large targets and large gradients.

**Fixes, in order of what I'd try first:**
- **Double DQN** — directly addresses max-operator bias, and is close to free to implement
- Lengthen the target-update interval, or switch to Polyak averaging with small $\tau$
- **Huber loss** instead of MSE — bounds gradient magnitude from large TD errors
- Reward clipping to $[-1, 1]$
- Lower the learning rate

**Cross-questions to expect:**
- *"Why does the max operator overestimate?"* → $\mathbb{E}[\max_a \hat{Q}] \geq \max_a \mathbb{E}[\hat{Q}]$ by Jensen's inequality. With noisy estimates, whichever action is overestimated gets selected, so the bias is systematically upward and compounds through bootstrapping.
- *"How exactly does Double DQN fix it?"* → Decouples selection from evaluation: $r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s',a'))$. The online net picks the action, the target net scores it, so an action overestimated by one is unlikely to be equally overestimated by the other.
- *"Could flat reward have a different cause entirely?"* → Yes — check exploration first. If ε decayed too fast the agent may simply never have found reward, and the diverging values would be a separate issue.

---

## Connections

- **Foundations:** [01-rl-foundations.md](01-rl-foundations.md) — Bellman equations these methods approximate
- **Continuous actions:** [04-policy-gradient-methods.md](04-policy-gradient-methods.md)
- **The stateless case:** [02-bandits-and-exploration.md](02-bandits-and-exploration.md)
- **Bias–variance:** [../07-interview-prep/ml/08-algorithms.md](../07-interview-prep/ml/08-algorithms.md)
