---
module: Reinforcement Learning
topic: Bandits and Exploration
subtopic: ""
status: unread
tags: [rl, bandits, exploration, thompson-sampling, ucb, ab-testing]
prerequisites: [probability, bayes-theorem, confidence-intervals]
---
# Bandits and the Exploration–Exploitation Trade-off

---

## The Problem It Solves

You have $k$ options — ad creatives, ranking models, onboarding flows — and each returns a noisy reward when chosen. You want to maximize total reward while *discovering* which option is best. Every trial spent learning about a bad option is reward forfeited; every trial spent exploiting your current best risks locking in a mistake made on thin evidence.

This is the **exploration–exploitation trade-off**, and the bandit setting isolates it cleanly by stripping away everything else: there are no states, and actions don't influence what happens next. Just repeated choices under uncertainty.

**Why this file sits before the RL algorithms:** most problems presented as reinforcement learning are actually bandit problems. Bandits need orders of magnitude less data, carry far less deployment risk, and are dramatically easier to reason about. Recognizing when you have a bandit instead of a full MDP is one of the highest-value judgments in applied RL — and a frequent interview probe.

---

## Intuition

A row of slot machines with unknown payout rates. Pull arms, observe payouts, maximize total winnings over 1,000 pulls.

Pull only the arm that's done best so far and you may be exploiting a machine that got lucky in its first three pulls. Spread pulls evenly and you waste most of your budget on machines you've already established are poor. Neither extreme works, and the good algorithms all resolve the tension the same way: **explore in proportion to how uncertain you are, not at a fixed rate.**

---

## The Mechanics

### Regret: the right objective

We measure a bandit algorithm by **cumulative regret** — total reward lost relative to always having played the best arm:

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^{T} \mathbb{E}[r_t]$$

where $\mu^*$ is the mean of the best arm. The key question is how regret grows with $T$:

- **Linear regret** $O(T)$ — the algorithm never stops making a constant rate of mistakes. Failure.
- **Logarithmic regret** $O(\log T)$ — the theoretical optimum for stochastic bandits (Lai–Robbins lower bound). Achieved by UCB and Thompson Sampling.

The reason $\varepsilon$-greedy is theoretically unsatisfying falls out immediately: with $\varepsilon$ fixed, it explores forever at a constant rate, so it accrues linear regret no matter how certain it becomes.

### The four algorithms worth knowing

**1. ε-greedy** — with probability $\varepsilon$, pick uniformly at random; otherwise pick the current best.

```python
def epsilon_greedy(counts, values, eps=0.1):
    if np.random.random() < eps:
        return np.random.randint(len(values))   # explore
    return int(np.argmax(values))               # exploit
```

Trivial to implement and reason about. Its flaw is that exploration is *undirected* — it is equally likely to try an arm it has already proven terrible as one it knows nothing about. Decaying $\varepsilon_t \propto 1/t$ recovers logarithmic regret.

**2. Upper Confidence Bound (UCB)** — optimism in the face of uncertainty. Score each arm by its mean plus an uncertainty bonus, then act greedily on that score:

$$\text{UCB}_i(t) = \hat{\mu}_i + c\sqrt{\frac{2\ln t}{n_i}}$$

```python
def ucb1(counts, values, t, c=1.0):
    if 0 in counts:
        return int(np.argmin(counts))           # each arm once first
    bonus = c * np.sqrt(2 * np.log(t) / counts)
    return int(np.argmax(values + bonus))
```

The bonus shrinks as $n_i$ grows and grows slowly with $t$, so under-sampled arms get revisited. Exploration is now *directed*: it targets uncertainty rather than randomness. Deterministic given the counts, which makes it reproducible and easy to debug.

**3. Thompson Sampling** — maintain a posterior over each arm's reward, sample from each posterior, play the argmax of the samples.

```python
def thompson_bernoulli(successes, failures):
    samples = np.random.beta(successes + 1, failures + 1)
    return int(np.argmax(samples))
```

Elegant because the exploration is *automatic*: a wide posterior produces variable samples that sometimes win, so uncertain arms get tried; as evidence accumulates the posterior narrows and the arm stops being sampled to the top by accident. In practice it usually outperforms UCB, handles delayed feedback gracefully, and extends naturally to batched updates.

**4. Contextual bandits** — the version you'll actually deploy. Reward depends on a context $x_t$ (user features, device, time), so you learn $\mathbb{E}[r \mid x, a]$ rather than a single mean per arm. LinUCB assumes a linear model per arm and carries confidence ellipsoids; neural contextual bandits replace the linear model.

This is the bridge to personalization: not "which creative is best?" but "which creative is best *for this user*?"

### Choosing among them

| | ε-greedy | UCB | Thompson | Contextual |
| :--- | :--- | :--- | :--- | :--- |
| Regret | Linear (fixed ε) | $O(\log T)$ | $O(\log T)$ | Depends on model |
| Exploration | Undirected | Directed, deterministic | Directed, stochastic | Directed |
| Delayed feedback | Fine | Degrades | Handles well | Handles well |
| Needs | Nothing | Count per arm | Posterior per arm | Feature model |
| Use when | Baseline, debugging | Reproducibility matters | Default choice | Users differ |

---

## Worked Example

Three ad creatives, Bernoulli rewards. After 100 impressions:

| Arm | Impressions | Clicks | $\hat{\mu}$ |
| :--- | ---: | ---: | ---: |
| A | 70 | 7 | 0.100 |
| B | 20 | 3 | 0.150 |
| C | 10 | 1 | 0.100 |

**Greedy** picks B — highest observed mean. But B's estimate rests on 20 impressions, and C's on only 10.

**UCB at $t = 100$, $c = 1$:**

$$\text{UCB}_A = 0.100 + \sqrt{\tfrac{2\ln 100}{70}} = 0.100 + 0.363 = 0.463$$
$$\text{UCB}_B = 0.150 + \sqrt{\tfrac{2\ln 100}{20}} = 0.150 + 0.679 = 0.829$$
$$\text{UCB}_C = 0.100 + \sqrt{\tfrac{2\ln 100}{10}} = 0.100 + 0.960 = 1.060$$

**UCB picks C** — the worst-tied observed mean, because 10 impressions is nearly no evidence. The bonus (0.960) dominates the mean difference (0.050). That inversion is the whole idea: with this little data, "C looks equal to A" is not a claim worth acting on.

**Thompson Sampling** reaches the same conclusion by a different route. C's posterior is $\text{Beta}(2, 10)$ — mean 0.167, but wide, with meaningful mass above 0.3. Draw from all three posteriors and C's sample wins often enough to keep getting impressions until its posterior tightens.

**The lesson:** observed means alone are not a decision rule. How much evidence sits behind each estimate is half the information, and both principled algorithms act on it.

---

## When It Breaks

| Failure | Mechanism | Mitigation |
| :--- | :--- | :--- |
| **Non-stationary rewards** | Arm quality drifts; stale data dominates the estimate | Sliding window, or discount old observations |
| **Delayed feedback** | Reward arrives hours later; algorithm updates on incomplete data | Thompson handles this better; explicitly model pending trials |
| **Batched decisions** | Must commit to 10k impressions before any update | Thompson degrades gracefully; UCB's determinism makes a whole batch identical |
| **Correlated arms** | Two near-identical creatives split evidence | Contextual/parametric model that shares information across arms |
| **Reward is not the goal** | Optimizing CTR degrades long-term retention | Choose a reward aligned to the real objective; monitor guardrails |

**Non-stationarity is the most common production failure.** Standard bandits assume fixed reward distributions. Real arms decay — a creative that performed well in week one suffers ad fatigue by week three. Without discounting, an arm's early success is never forgotten and it is exploited long past its actual decline.

---

## Production Notes

- **Bandits vs. A/B tests.** An A/B test fixes allocation, runs to a predetermined sample size, and delivers a clean statistical inference. A bandit reallocates traffic continuously and maximizes reward *during* the experiment. Use an A/B test when you need a defensible causal estimate for a one-time decision; use a bandit when you need ongoing allocation and the cost of showing a bad variant is real.
- **Bandits complicate inference.** Because allocation depends on observed rewards, the collected data is not i.i.d. Naive confidence intervals on bandit data are wrong. If you need valid post-hoc inference, use always-valid inference or reserve a small fixed-allocation holdout.
- **Always keep a holdout.** A small slice on fixed random allocation gives you unbiased evaluation and protects against a mis-specified reward silently steering the system.
- **Warm-start from history.** Initializing posteriors from historical performance avoids re-learning what you already know — a substantial practical win.
- **Watch the reward horizon.** Immediate-signal rewards (clicks) are easy to optimize and frequently misaligned with the metric you care about (retention, revenue). This is the same reward-specification problem that dominates full RL, in a smaller package.

---

## Interview Angles

### Q: What's the exploration–exploitation trade-off? [Easy]

Exploitation takes the action that currently looks best; exploration takes an action to reduce uncertainty about it. Pure exploitation risks locking onto an option that only looked good by chance; pure exploration wastes trials on options already known to be poor. Good algorithms explore in proportion to uncertainty rather than at a fixed rate.

**Cross-questions to expect:**
- *"Simplest algorithm that handles it?"* → ε-greedy. Explore with probability ε, otherwise exploit.
- *"What's wrong with it?"* → Exploration is undirected — it's as likely to re-try a known-bad arm as an unknown one. And with fixed ε it explores forever, giving linear regret.
- *"How would you fix that with one change?"* → Decay ε over time, e.g. $\varepsilon_t = 1/t$. Recovers logarithmic regret while staying trivial to implement.

---

### Q: Why does UCB add a term based on the visit count? [Medium]

It's optimism in the face of uncertainty. The bonus $\sqrt{2\ln t / n_i}$ is a confidence-interval width — it's large when an arm has few pulls and shrinks as evidence accumulates. Acting greedily on mean-plus-bonus means an arm gets tried either because it looks good *or* because you don't know enough about it yet. Under-explored arms are automatically revisited, and the exploration is directed rather than random.

**Cross-questions to expect:**
- *"Why $\ln t$ in the numerator rather than a constant?"* → It grows slowly, so exploration never fully stops — protection against having been unlucky early with the true best arm. Slow enough to keep total regret logarithmic.
- *"What does the constant $c$ control?"* → Exploration aggressiveness. Too high wastes trials on known-bad arms; too low collapses toward greedy and risks premature convergence.
- *"When would you pick Thompson over UCB?"* → Delayed or batched feedback. UCB is deterministic, so a whole batch under identical counts selects the same arm; Thompson's sampling naturally diversifies. Thompson also tends to win empirically.

**Trap:** Describing UCB as "adding randomness." It's fully deterministic given the counts — the exploration comes from optimism, not noise. That distinction is exactly what separates it from ε-greedy.

---

### Q: When would you use a bandit instead of an A/B test? [Medium]

When you want to *minimize regret during* the experiment rather than obtain a clean causal estimate at the end. An A/B test knowingly sends 50% of traffic to the worse variant for its full duration — that's the price of a clean inference. A bandit shifts traffic toward better variants as evidence accumulates.

Use an A/B test when you need a defensible estimate for a one-time launch decision, or when a stakeholder needs a p-value. Use a bandit for ongoing allocation among many options where the cost of serving a bad one is real — ad creatives, ranking candidates, content slots.

**Cross-questions to expect:**
- *"What do you give up with a bandit?"* → Valid classical inference. Allocation depends on observed rewards, so the data isn't i.i.d. and standard confidence intervals are invalid. You need always-valid inference or a fixed holdout.
- *"What if the arms change quality over time?"* → Standard bandits assume stationarity. Add a sliding window or discount old observations, otherwise early winners are exploited long after they've decayed.
- *"Could you use both?"* → Yes, and this is the strong answer: a bandit for allocation with a small fixed-allocation holdout for unbiased measurement.

---

### Q: How do contextual bandits differ, and where's the catch? [Medium]

A contextual bandit conditions on features: instead of one mean per arm, you learn $\mathbb{E}[r \mid x, a]$. The best arm can differ per user, which turns the problem from "which creative is best?" into personalization.

The catch is that you've reintroduced a model, and now all the usual modeling failures apply *inside* the exploration loop. A mis-specified model produces confidence estimates that are wrong in ways that misdirect exploration — and because the model steers data collection, the error is self-reinforcing.

**Cross-questions to expect:**
- *"Is this just RL now?"* → No — still one step. Actions don't affect the next context. That's the boundary: no state transitions means no credit assignment problem, which is why contextual bandits are far more tractable than full RL.
- *"When does it become full RL?"* → When the action changes the next state. Recommending an item that changes what a user is later interested in makes it sequential.
- *"How do you evaluate one offline?"* → Off-policy evaluation with importance weighting on logged data, or replay on logged random-allocation traffic. This is a strong argument for always logging a random-allocation slice.

---

### Q: Your bandit converged to one arm in two days, then CTR declined over the next month. What happened? [Hard]

The likely cause is **non-stationarity plus premature convergence**. The bandit locked in on an arm that was genuinely best at the time. Ad fatigue then degraded it, but because the algorithm had stopped exploring alternatives, it had no evidence that anything else had become better — and the arm's large accumulated sample made its estimate slow to move.

A second contributing mechanism: with no discounting, thousands of early successful impressions dominate the running mean, so recent poor performance barely shifts the estimate.

**How I'd diagnose it:**
1. Plot per-arm CTR over time, not the pooled average — is the winning arm declining, or is the population shifting?
2. Check the holdout slice. If randomly-allocated traffic shows other arms now outperforming, it's the bandit failing to adapt, not a market-wide decline.
3. Check impression counts per arm. If losers stopped receiving traffic entirely, exploration collapsed.

**Fixes:**
- Sliding-window or exponentially-discounted estimates so old data decays
- A floor on exploration probability so no arm is permanently starved
- Periodic posterior resets, or explicit change-point detection
- Adding creative-age as a context feature, which lets the model represent fatigue directly

**Cross-questions to expect:**
- *"How would you set the discount rate?"* → From the timescale of genuine change. Measure how fast CTR decays for a fixed creative in historical data and match the effective window to it.
- *"Would Thompson Sampling have avoided this?"* → Not by itself — it has the same stationarity assumption. Discounted Thompson Sampling would, since posteriors widen again as old evidence decays.
- *"How would you catch it sooner?"* → Alert on the gap between the bandit's expected reward and the holdout's realized reward. Divergence means the model's beliefs have gone stale.

**Trap:** Blaming the algorithm choice. The failure is the stationarity assumption, and every standard bandit shares it — swapping UCB for Thompson changes nothing.

---

## Connections

- **Foundations:** [01-rl-foundations.md](01-rl-foundations.md) — MDPs, and why bandits are the stateless special case
- **When it becomes sequential:** [03-value-based-methods.md](03-value-based-methods.md)
- **A/B testing:** [../06-production-ml/system-design/14-ab-testing-experimentation.md](../06-production-ml/system-design/14-ab-testing-experimentation.md)
- **Statistical background:** [../07-interview-prep/ml/24-statistics-probability-rapid-fire.md](../07-interview-prep/ml/24-statistics-probability-rapid-fire.md)
