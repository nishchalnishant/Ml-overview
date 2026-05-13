# Reinforcement Learning

This file is your complete RL interview prep kit — from "what even is an agent" to PPO and RLHF. Written for someone who already knows ML fundamentals and wants to reason clearly under interview pressure.

---

## Quick Access

| Section | What you get |
| :--- | :--- |
| [What is RL?](#1-what-is-rl) | Mental model, when to use it |
| [Core Concepts](#2-core-concepts) | Agent, state, action, reward, policy, value |
| [MDPs](#3-markov-decision-processes-mdps) | Bellman equations, discounting |
| [Dynamic Programming](#4-dynamic-programming) | Value iteration, policy iteration |
| [Model-Free RL](#5-model-free-rl) | Monte Carlo, TD, Q-Learning |
| [Deep Q-Networks](#6-deep-q-networks-dqn) | DQN, experience replay, target nets |
| [Policy Gradients](#7-policy-gradient-methods) | REINFORCE, Actor-Critic, A2C/A3C |
| [PPO](#8-proximal-policy-optimization-ppo) | Clipped objective, RLHF connection |
| [Multi-Armed Bandits](#9-multi-armed-bandits) | Exploration vs exploitation |
| [RLHF](#10-reinforcement-learning-from-human-feedback-rlhf) | How ChatGPT is trained |
| [Interview Q&A](#11-common-interview-questions) | 20 questions with full answers |
| [Key Equations](#12-key-equations-reference) | Cheatsheet |

---

# 1. What is RL?

## The Core Idea

Supervised learning says: "Here are 10,000 labeled examples. Learn from them."

Reinforcement learning says: "Here is a world. Go explore it. You'll get a score after each action. Figure out what to do."

No labels. No teacher. Just a signal at the end telling you whether that was good or bad.

### The Dog Training Analogy

You are training a dog to sit. You do not give it a textbook on joint biomechanics. You:

1. Say "sit"
2. Wait for the dog to do something
3. If it sits, give it a treat (reward = +1)
4. If it barks and runs away, give nothing (reward = 0)
5. Repeat thousands of times

The dog is the **agent**. Your living room is the **environment**. The dog's posture at any moment is the **state**. Sitting vs. barking vs. rolling over are **actions**. The treat is the **reward**. The dog's general rule for "what to do when asked to sit" is its **policy**.

That is RL in one paragraph.

### The Chess Player Analogy

A chess player learns not from a textbook but from games. After each game, they win (+1) or lose (-1). Over thousands of games they build intuition about which positions lead to wins. The intermediate moves did not come with labels — the signal only arrived at the very end. RL calls this the **credit assignment problem**: which of the 40 moves in the game was responsible for the win?

### When is RL the Right Tool?

| Scenario | Why RL fits |
| :--- | :--- |
| Game playing (Go, Chess, Atari) | Clear reward signal, well-defined rules |
| Robotics control | Sequential physical actions with delayed feedback |
| Recommendation systems | User engagement as reward signal |
| Trading algorithms | Profit/loss as reward |
| LLM fine-tuning (RLHF) | Human preferences as reward signal |
| Resource scheduling | Minimize latency/cost over time |

RL is not magic. It is expensive to train (millions of interactions), unstable, and hard to debug. Use supervised learning when you have labels. Use RL when you don't — and when the problem is inherently sequential.

---

# 2. Core Concepts

## The Cast of Characters

Think of an RL problem as a video game. Everything that follows maps onto this frame.

### Agent

The learner and decision-maker. In a video game, this is the player. In robotics, it is the robot. In RLHF, it is the language model.

The agent's job: observe the current situation, pick an action, and try to maximize the total reward it accumulates over time.

### Environment

Everything the agent interacts with but does not directly control. The environment receives an action from the agent and returns two things: the next state, and a reward.

In the video game frame: the game engine is the environment. The agent presses a button (action). The game updates the screen (next state) and maybe adds to the score (reward).

### State ($s$)

A description of the current situation. Complete enough for the agent to make a decision.

In chess: the board position. In robotics: the joint angles and velocities. In a conversation: the dialogue history.

**Key distinction:** *state* vs *observation*. A state is the full truth about the world. An observation is what the agent actually sees (which might be partial). In a poker game, you observe your cards — but the state includes everyone's cards.

### Action ($a$)

What the agent does. Can be discrete (move left/right/up/down) or continuous (apply 2.7 Nm of torque).

The set of all possible actions is the **action space**.

### Reward ($r$)

A scalar signal the environment sends to the agent after each action. Positive means good. Negative means bad. Zero means neutral.

The reward function is the most important design decision in any RL system. Get it wrong and your agent will find creative, unintended ways to maximize it — like a dog that learns to bark at the treat bag instead of sitting.

This is called **reward hacking**. It is hilarious in toy problems and catastrophic in production.

### Policy ($\pi$)

The agent's strategy. A mapping from states to actions (or distributions over actions).

$$\pi(a \mid s) = P(\text{take action } a \mid \text{in state } s)$$

A **deterministic policy** always picks the same action in a given state: $\pi(s) = a$.

A **stochastic policy** picks actions according to a probability distribution. Stochastic policies are better when: (a) the optimal strategy requires randomization, or (b) you are still exploring.

Cricket analogy: a batsman's policy is their decision-making system for each delivery. "If it's a full-length delivery on off-stump, drive straight. If it's short, pull." That mapping from ball-type (state) to shot selection (action) is their policy.

### Value Function ($V$)

If policy is "what should I do right now?", value is "how good is it to be in this situation?"

The **state value function** $V^\pi(s)$ is the expected total future reward when starting in state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]$$

where $\gamma$ is the discount factor (explained shortly).

The **action-value function** $Q^\pi(s, a)$ (also called Q-value) is the expected total future reward when in state $s$, taking action $a$, then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s, a_0 = a \right]$$

The difference: $V^\pi(s)$ doesn't let you choose the first action. $Q^\pi(s, a)$ does. This makes $Q$ more useful for selecting actions.

### Model (optional)

Some RL algorithms also have a **model** of the environment — a learned approximation of how the world works (transition dynamics and reward function). Algorithms that use a model are called **model-based**. Those that don't are **model-free**. Most modern RL (DQN, PPO, SAC) is model-free.

---

# 3. Markov Decision Processes (MDPs)

## The Formal Framework

Every RL problem can be stated as an MDP. An MDP is a mathematical framework with five components:

$$\text{MDP} = (S, A, P, R, \gamma)$$

| Symbol | Meaning |
| :--- | :--- |
| $S$ | Set of all states |
| $A$ | Set of all actions |
| $P(s' \mid s, a)$ | Transition probability: chance of landing in $s'$ after taking $a$ in $s$ |
| $R(s, a, s')$ | Reward received after the transition |
| $\gamma \in [0, 1)$ | Discount factor |

## The Markov Property

The key assumption: **the future depends only on the present state, not the history of how you got there.**

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

This is the Markov assumption. It says: if you know the current state, the past is irrelevant for predicting the future.

Cricket analogy: what happens on the next delivery depends on the current match situation (runs, wickets, overs, pitch conditions, bowler's form) — not on what happened in the first over three hours ago. The current state captures everything relevant.

This is why defining a good state representation is critical. If your state doesn't capture enough information, the Markov assumption breaks down and your agent will struggle.

## Discounting

Why do we need $\gamma$?

Two reasons:

1. **Mathematical:** Without discounting, total rewards over infinite time horizons diverge (infinite sum). Discounting makes the series converge.

2. **Economic:** A reward now is worth more than a reward later. Money today > money tomorrow. This is the time value of reward.

$$\text{Return } G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

At $\gamma = 0$: the agent is completely myopic — only cares about immediate reward. At $\gamma = 0.99$: the agent is nearly far-sighted, valuing future rewards almost as much as immediate ones.

In chess, a good player values long-term board position. High $\gamma$. In a sprint race, only the finish line matters. Low $\gamma$.

## Bellman Equations

The Bellman equations are the backbone of almost all RL algorithms. They express recursive relationships between values.

### Bellman Expectation Equation for $V^\pi$

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

Read this as: "The value of being in state $s$ under policy $\pi$ equals the expected immediate reward plus the discounted value of wherever I land next."

It is recursive: $V(s)$ is defined in terms of $V(s')$.

### Bellman Expectation Equation for $Q^\pi$

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]$$

### Bellman Optimality Equations

The **optimal** value functions satisfy a special condition: the policy always picks the best action.

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

Once you have $Q^*$, extracting the optimal policy is trivial:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

This is the key insight behind Q-Learning: learn $Q^*$ directly, and the optimal policy falls out for free.

---

# 4. Dynamic Programming

## Overview

Dynamic Programming (DP) assumes you have a **perfect model of the environment** — you know $P(s' \mid s, a)$ and $R(s, a, s')$ exactly. Given that, you can compute optimal policies by iteratively solving the Bellman equations.

DP is not practical for real-world RL (you rarely know the full model), but it is foundational. Every model-free algorithm is an approximation of DP.

## Policy Evaluation

Before improving a policy, you need to know how good it is. **Policy evaluation** computes $V^\pi$ for a fixed $\pi$.

Algorithm: repeatedly apply the Bellman expectation equation until convergence.

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

This is a system of linear equations being solved by iterative substitution.

## Policy Improvement

Given $V^\pi$, can we find a better policy? Yes — by acting **greedily** with respect to $V^\pi$:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

**Policy Improvement Theorem:** The greedy policy $\pi'$ is at least as good as $\pi$. That is, $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

## Policy Iteration

Alternate between policy evaluation and policy improvement until the policy stops changing.

```
Initialize policy π arbitrarily
Repeat:
    1. Policy Evaluation:  compute V^π
    2. Policy Improvement: π' = greedy(V^π)
    3. If π' == π: stop (optimal policy found)
    4. π ← π'
```

**Convergence:** Guaranteed for finite MDPs. Each iteration strictly improves the policy (or terminates). Because there are finitely many policies, it converges.

**Cost:** Each policy evaluation step requires solving the full system — expensive for large state spaces.

## Value Iteration

Instead of waiting for full convergence of policy evaluation, do a single sweep and immediately improve.

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
Extract policy:
    π(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s)]
```

This directly applies the **Bellman Optimality Equation** as an update rule.

**Value vs Policy Iteration:**

| | Policy Iteration | Value Iteration |
| :--- | :--- | :--- |
| Inner loop | Full policy eval to convergence | Single backup per state |
| Outer iterations | Fewer | More |
| Practical speed | Often faster overall | Simpler to implement |
| Policy availability | Explicit policy maintained | Policy extracted at end |

Both converge to the optimal policy. Policy iteration often needs fewer outer iterations because it has a more informed policy at each step.

---

# 5. Model-Free RL

## Why Model-Free?

In most real problems, you don't know $P(s' \mid s, a)$. You only know what actually happened: "I took action $a$ in state $s$ and ended up in $s'$ with reward $r$."

Model-free methods learn directly from experience without ever building an explicit model of the environment.

Two families: **Monte Carlo** (wait for episode to end) and **Temporal Difference** (update at every step).

---

## 5.1 Monte Carlo Methods

### Core Idea

Run a full episode. Collect all the rewards. At the end, go back and compute the actual return $G_t$ for each state visited. Use that as a sample estimate of $V(s)$ or $Q(s, a)$.

$$G_t = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T$$

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

### Cricket Analogy

You play a full innings. At the end, you look back at every shot you played and ask: "That pull shot I played in the 15th over — given that the innings ended at 180, did that shot contribute to the final score or was it reckless?" You update your judgment about each shot based on how the whole innings played out.

### Properties

- **Unbiased:** The return $G_t$ is the true cumulative reward — no approximation of future value.
- **High variance:** One episode can be wildly different from another (bad luck, good luck, etc.). You need many episodes to get stable estimates.
- **Episodic only:** Only works if episodes terminate. Useless for continuous tasks.

---

## 5.2 Temporal Difference Learning

### Core Idea

Don't wait for the episode to end. At every step, make a **bootstrapped** estimate of the return using the current value function estimate for the next state.

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]$$

The term $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error** — the difference between what you expected ($V(s_t)$) and what actually happened plus the discounted estimate of the future ($r_{t+1} + \gamma V(s_{t+1})$).

### The Prediction-Correction Intuition

Imagine you are driving to work. You predict it will take 30 minutes. After 10 minutes you're stuck in traffic — you update: now you predict 45 minutes. You didn't wait to arrive to update your estimate. You used the new information (traffic) to correct your earlier prediction.

TD learning does the same thing with value estimates.

### MC vs TD

| | Monte Carlo | TD |
| :--- | :--- | :--- |
| Update timing | End of episode | Every step |
| Bias | Unbiased | Biased (bootstrapping) |
| Variance | High | Low |
| Requires episodes | Yes | No |
| Convergence | Slower (high variance) | Faster in practice |

**TD is biased but lower variance. MC is unbiased but higher variance.** This is the fundamental tradeoff.

### TD($\lambda$): Bridging MC and TD

TD(0) uses one step of lookahead. MC uses the full episode. TD($\lambda$) interpolates between them using eligibility traces.

At $\lambda = 0$: TD(0). At $\lambda = 1$: equivalent to MC. In practice, values like $\lambda = 0.9$ work well.

---

## 5.3 Q-Learning

### The Big Idea

Q-Learning learns the optimal action-value function $Q^*(s, a)$ directly, without needing to know the policy being followed. It is **off-policy**.

### Derivation from Bellman Optimality

We want to satisfy the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

We don't know the expectation, but we have samples $(s, a, r, s')$. So we use each sample as an approximation:

$$\text{target} = r + \gamma \max_{a'} Q(s', a')$$

And update our estimate toward this target:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

That is the Q-Learning update. One line. The most important equation in RL.

### Why "Off-Policy"?

The agent uses an **$\epsilon$-greedy** policy for exploration (sometimes picks random actions), but the update uses $\max_{a'} Q(s', a')$ — the **greedy** (optimal) action — regardless of what the agent actually did next. The learning target does not depend on the behavior policy.

This means you can learn from old data, replayed experiences, or even watching someone else play. This is crucial for DQN.

### Q-Learning in Code

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.n_actions = n_actions

    def select_action(self, state):
        # epsilon-greedy: explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # Q-learning update
        self.Q[state, action] += self.alpha * (target - current_q)
```

### SARSA: The On-Policy Sibling

SARSA (State-Action-Reward-State-Action) is Q-Learning's on-policy cousin. Instead of using $\max_{a'} Q(s', a')$, it uses the action the agent **actually takes** next:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where $a'$ is sampled from $\pi(\cdot \mid s')$.

**Key difference:** Q-Learning optimizes for the greedy policy regardless of exploration. SARSA accounts for the fact that the agent will sometimes explore. In cliffwalking environments, SARSA learns safer paths because it accounts for accidental falls during exploration. Q-Learning learns the theoretically optimal path but gets burned during training.

---

# 6. Deep Q-Networks (DQN)

## The Problem with Tabular Q-Learning

Q-Learning with a table works perfectly for small discrete state spaces. But what about Atari video games? The state is a $210 \times 160$ RGB screen — $256^{210 \times 160 \times 3}$ possible states. No table can hold that.

Solution: replace the Q-table with a neural network. Input: state $s$. Output: Q-values for all actions $Q(s, a; \theta)$.

## Why Naive Deep Q-Learning Fails

If you just plug a neural network into the Q-learning update, training explodes. Two reasons:

1. **Correlated data:** RL data is sequential. Frame 1 is correlated with frame 2, which is correlated with frame 3. Neural networks assume i.i.d. data. This correlation causes the network to overfit to recent experience and forget older lessons.

2. **Non-stationary targets:** The target $r + \gamma \max_{a'} Q(s', a'; \theta)$ changes every update because the same network $\theta$ appears on both sides. It is like trying to hit a moving target with the same arm that keeps moving it. Training oscillates and diverges.

## DQN's Two Key Innovations

### 1. Experience Replay

Instead of training on transitions as they arrive, store them in a **replay buffer** (a large circular buffer of $(s, a, r, s', \text{done})$ tuples). Sample random mini-batches from this buffer for training.

This breaks temporal correlations and makes the data approximately i.i.d. As a bonus, each experience can be used for multiple gradient updates (more data efficient).

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)
```

### 2. Target Network

Maintain two networks with identical architectures:
- **Online network** $Q(s, a; \theta)$: updated every step
- **Target network** $Q(s, a; \theta^-)$: frozen copy of online network, updated every $N$ steps (e.g., every 1000 steps)

The loss is computed against the frozen target network:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \text{buffer}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

The target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ is now stable for $N$ steps. The moving target problem is resolved.

## Epsilon-Greedy Exploration

DQN uses a decaying $\epsilon$-greedy strategy:

- Start with $\epsilon = 1.0$ (fully random — explore everything)
- Linearly decay to $\epsilon = 0.01$ over the first million steps
- After that, act greedily with 1% random actions to maintain exploration

```python
epsilon = max(epsilon_end, epsilon_start - steps * (epsilon_start - epsilon_end) / epsilon_decay)
```

## Full DQN in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_dim, n_actions, lr=1e-4, gamma=0.99):
        self.gamma = gamma
        self.n_actions = n_actions
        
        self.online_net = DQN(input_dim, n_actions)
        self.target_net = DQN(input_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.steps = 0
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (from frozen target network)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically sync target network
        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
```

## DQN Improvements (Double DQN, Dueling DQN, PER)

### Double DQN

Standard DQN overestimates Q-values because it uses the same network to select and evaluate actions. Double DQN decouples these:

- **Action selection:** use online network — $a^* = \arg\max_{a'} Q(s', a'; \theta)$
- **Action evaluation:** use target network — $Q(s', a^*; \theta^-)$

$$\text{target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

### Dueling DQN

Separate the Q-value into two streams:
- **Value stream** $V(s)$: how good is this state, regardless of action?
- **Advantage stream** $A(s, a)$: how much better is action $a$ relative to the average?

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')$$

The mean subtraction ensures identifiability (otherwise $V$ and $A$ can absorb each other's values arbitrarily).

### Prioritized Experience Replay (PER)

Not all experiences are equally informative. Prioritize experiences with large TD error — those are the ones where the agent was most surprised.

$$P(i) \propto |\delta_i|^\alpha$$

where $\delta_i$ is the TD error of experience $i$ and $\alpha$ controls how much prioritization matters.

---

# 7. Policy Gradient Methods

## The Fundamental Shift

Q-Learning and DQN are **value-based**: learn a value function, derive the policy implicitly.

Policy gradient methods are **policy-based**: directly parameterize and optimize the policy $\pi_\theta(a \mid s)$.

Why? Three reasons:
1. Value-based methods only work for discrete action spaces. Policy gradients handle continuous actions naturally.
2. Policy gradients naturally produce stochastic policies (useful for games with mixed strategies).
3. The policy is the end goal — why not optimize it directly?

## The Policy Gradient Theorem

We want to maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G(\tau)]$$

where $\tau$ is a trajectory (sequence of states, actions, rewards) sampled by following $\pi_\theta$.

The gradient of this objective is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

This is the **Policy Gradient Theorem**. The key insight: even though $G_t$ doesn't depend on $\theta$, we can write the gradient of the expectation as an expectation of $\log \pi_\theta \cdot G_t$. This is the REINFORCE trick (log-derivative trick / score function estimator).

**Intuition:** increase the log-probability of actions that led to high returns. Decrease it for actions that led to low returns. Simple.

## 7.1 REINFORCE

The simplest policy gradient algorithm. Run full episodes, compute returns, update.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class REINFORCEAgent:
    def __init__(self, input_dim, n_actions, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(input_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, rewards, log_probs):
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize
        
        # Policy gradient loss
        loss = -torch.stack(log_probs) * returns
        loss = loss.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### The Baseline Trick

REINFORCE has high variance because $G_t$ can vary wildly across episodes. Solution: subtract a **baseline** $b(s_t)$ from the return. The gradient estimate remains unbiased (baseline has zero expected gradient), but variance drops.

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]$$

The most natural baseline is the **value function** $V(s_t)$. The quantity $G_t - V(s_t)$ is called the **advantage** — how much better was this action than average?

---

## 7.2 Actor-Critic

REINFORCE waits for the whole episode. Slow. Can we do better?

Actor-Critic maintains two components:
- **Actor:** the policy $\pi_\theta(a \mid s)$ (makes decisions)
- **Critic:** the value function $V_\phi(s)$ (evaluates decisions)

Instead of waiting for $G_t$, use a TD estimate:

$$A_t \approx r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

This is the **advantage estimate** — the critic says "I expected value $V_\phi(s_t)$, but we got $r_{t+1}$ and landed in $s_{t+1}$ which is worth $V_\phi(s_{t+1})$. Were we above or below expectations?"

Cricket analogy: the actor is the batsman picking their shot. The critic is the batting coach watching from the dugout saying "that cover drive was 20 runs better than your average shot in that situation" (advantage). The batsman updates their strategy accordingly.

```python
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        # Actor head: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        # Critic head: outputs state value
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        features = self.shared(x)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value
    
    def evaluate_action(self, state, action):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy
```

**Actor loss** (maximizing advantage-weighted log probability):
$$\mathcal{L}_{\text{actor}} = -\log \pi_\theta(a_t \mid s_t) \cdot A_t$$

**Critic loss** (minimizing TD error):
$$\mathcal{L}_{\text{critic}} = \left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2$$

**Total loss:**
$$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}} - c_2 H(\pi_\theta)$$

The entropy bonus $H(\pi_\theta)$ encourages exploration by preventing the policy from collapsing to a deterministic choice prematurely.

---

## 7.3 A2C and A3C

### A3C (Asynchronous Advantage Actor-Critic)

Introduced by DeepMind in 2016. Run multiple agents in parallel on copies of the environment. Each agent collects experience and asynchronously updates a global network. No replay buffer needed — the multiple workers provide diverse, less correlated experience.

Advantages: faster training (parallel), diverse experience, works on CPUs.

### A2C (Advantage Actor-Critic — Synchronous)

The synchronous version of A3C. All workers gather experience simultaneously. Wait for all of them, then do a synchronized update. More stable than A3C in practice.

Both use the **advantage function**:

$$A_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$$

This is the $n$-step advantage — look $n$ steps ahead before bootstrapping.

---

# 8. Proximal Policy Optimization (PPO)

## The Problem with Plain Policy Gradients

Standard policy gradient methods have a catastrophic failure mode: if you take too large a gradient step, the policy can change dramatically in one update. The new policy may be so different from the old one that performance collapses — and because you are now collecting data from the new policy, you can't even tell why.

Think of a batsman who, after one bad innings, completely changes their stance, grip, and shot selection all at once. They will probably play even worse for weeks.

## The Trust Region Idea

The safe way to update a policy: don't stray too far from the current one in a single step. Stay within a "trust region."

TRPO (Trust Region Policy Optimization) formalized this, but required solving a constrained optimization problem — complex and expensive.

PPO approximates TRPO with a simple clipped objective. Much simpler. Just as effective. This is why PPO is the dominant algorithm in practice.

## Importance Sampling

PPO is on-policy but reuses data from the previous policy for a few gradient steps. To correct for the distribution shift, it uses importance sampling:

$$\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = r_t(\theta)$$

This ratio $r_t(\theta)$ measures how much the new policy differs from the old one for a specific action.

## The PPO Clipped Objective

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Breaking this down:

- $r_t(\theta) \hat{A}_t$: the standard policy gradient objective weighted by the importance ratio
- $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$: the same, but the ratio is clipped to $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon = 0.2$)
- $\min(\ldots)$: take the worse of the two

The `min` + `clip` combination prevents the objective from benefiting from pushing the policy too far in either direction:

- If $\hat{A} > 0$ (action was good): increase its probability, but not beyond $1 + \epsilon$ times the old probability
- If $\hat{A} < 0$ (action was bad): decrease its probability, but not below $1 - \epsilon$ times the old probability

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2,
             value_pred, returns, value_coef=0.5, entropy_coef=0.01,
             entropy):
    # Importance sampling ratio
    ratios = torch.exp(log_probs_new - log_probs_old)
    
    # Clipped objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Critic loss
    critic_loss = nn.MSELoss()(value_pred, returns)
    
    # Total loss (minimize)
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
    return total_loss
```

## PPO Full Update Loop

```
Collect T timesteps of experience using current policy π_θ_old
Compute advantages Â_t using GAE (Generalized Advantage Estimation)
For K epochs:
    For each mini-batch:
        Compute new log probs, values, entropy under π_θ
        Compute clipped PPO loss
        Backprop and update θ
Update θ_old ← θ
```

## Generalized Advantage Estimation (GAE)

PPO typically uses GAE for advantage estimation, which smoothly interpolates between 1-step TD ($\lambda=0$) and full Monte Carlo ($\lambda=1$):

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error at time $t$.

$\lambda = 0.95$ and $\gamma = 0.99$ are the standard PPO settings.

## Why PPO Dominates

| Property | REINFORCE | A2C | PPO |
| :--- | :--- | :--- | :--- |
| Sample efficiency | Low | Medium | High |
| Stability | Low | Medium | High |
| Implementation | Simple | Medium | Moderate |
| Continuous actions | Yes | Yes | Yes |
| Used in RLHF | No | No | Yes |

PPO is the workhorse of modern RL. It is used in: OpenAI Five (Dota 2), AlphaStar (StarCraft 2), and critically — RLHF fine-tuning of language models.

---

# 9. Multi-Armed Bandits

## The Setup

The simplest RL problem: no states, just actions and rewards.

You have $K$ slot machines (arms). Each machine pays out some random reward with an unknown distribution. You have $N$ total pulls. Which machines do you pull to maximize total reward?

### The Restaurant Analogy

You move to a new city with 20 restaurants. You don't know which is good. You have lunch every day. Do you:
- Always go to your current favorite? (exploitation — might miss the best one)
- Try a new one every day? (exploration — waste time at bad ones)
- Some mix? (the correct answer)

This is the **exploration-exploitation tradeoff** — the central tension of all RL.

## Formal Setup

At each step $t$, pick arm $a_t$. Receive reward $r_t \sim P_a$.

Goal: maximize cumulative reward $\sum_{t=1}^{T} r_t$.

Alternative goal: minimize **regret** — the difference between optimal total reward and what you actually got:

$$\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^{T} r_t$$

where $\mu^*$ is the mean reward of the best arm.

## Key Strategies

### Epsilon-Greedy

With probability $\epsilon$: pick a random arm (explore).
With probability $1-\epsilon$: pick the arm with the highest estimated mean reward (exploit).

Simple but effective. The restaurant version: you go to your favorite restaurant 9 out of 10 days, but one random day per week you try somewhere new.

### UCB (Upper Confidence Bound)

Instead of exploring randomly, explore **optimistically**. Pick the arm that has the highest upper confidence bound on its value:

$$a_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

where $\hat{\mu}_a$ is the estimated mean of arm $a$, $N_a$ is the number of times arm $a$ has been pulled, and $c$ is a confidence parameter.

The second term is large when an arm has been pulled few times — uncertainty bonus. As you pull an arm more, confidence grows and the bonus shrinks.

**UCB is deterministic** (given the same history, always picks the same arm) and provably achieves $O(\ln T)$ regret.

### Thompson Sampling

Bayesian approach. Maintain a posterior distribution over the mean reward for each arm. At each step, sample a mean from each posterior. Pick the arm with the highest sample. Update posteriors with observed rewards.

For Bernoulli rewards (0 or 1), the conjugate prior is Beta:

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)  # successes + 1
        self.beta = np.ones(n_arms)   # failures + 1
    
    def select_arm(self):
        # Sample from each arm's Beta posterior
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
```

Thompson Sampling often outperforms UCB empirically and has similar theoretical guarantees.

## Why Bandits Matter for ML

Bandits are not just a toy problem. They appear in:
- **A/B testing:** which website version performs better? (don't waste traffic on the loser)
- **Ad serving:** which ad to show to maximize click-through?
- **Hyperparameter tuning:** which config should I try next?
- **Recommendation:** which item to recommend to maximize engagement?

The key insight that transfers to full RL: **you must balance exploiting what you know with exploring what you don't.**

---

# 10. Reinforcement Learning from Human Feedback (RLHF)

## The Core Problem

Large language models pre-trained on text corpora are trained to predict the next token — not to be helpful, harmless, or honest. They optimize for likelihood, not quality. You can get a model that correctly predicts offensive text because offensive text appears on the internet.

How do you teach a model to be helpful? You need a reward signal. But how do you define "helpful" mathematically? You can't. So you use humans.

RLHF is the answer.

## The Three-Stage Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Take a pre-trained LLM. Collect a dataset of high-quality prompt-response pairs (written or curated by human labelers). Fine-tune the LLM on this dataset with standard cross-entropy loss.

This gives you a **reference model** $\pi_{\text{ref}}$ that can follow instructions reasonably well. But it's still not optimized for human preferences.

### Stage 2: Train a Reward Model

Collect a dataset of **preference comparisons**: for the same prompt, show the model (or human labelers) two responses $y_w$ (preferred) and $y_l$ (rejected). The labeler picks which they prefer.

Train a reward model $r_\phi(x, y)$ that takes a prompt $x$ and response $y$, and outputs a scalar reward. Train using the Bradley-Terry preference model:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

This says: the reward model should score preferred responses higher than rejected ones. The sigmoid $\sigma$ converts the score difference to a probability.

The reward model is typically the SFT model with the final head replaced by a scalar output head.

### Stage 3: RL Fine-Tuning with PPO

Now use PPO to optimize the SFT policy to maximize the reward model's scores:

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$

Two components:
1. $r_\phi(x, y)$: maximize the reward model score (be helpful)
2. $-\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$: KL divergence penalty (don't drift too far from SFT model)

The KL penalty is critical. Without it, the model will find ways to hack the reward model — generating responses that score high but are meaningless or manipulative. This is reward hacking, at LLM scale.

```
The KL term acts as a leash. The model can explore improvements,
but can't run too far from the safe reference policy.
```

## RLHF in Practice: The Token-Level MDP

The LLM generates responses token by token. Casting this as an MDP:

- **State $s_t$:** the prompt + all tokens generated so far
- **Action $a_t$:** the next token to generate (vocabulary size ~50,000)
- **Reward:** 0 for all intermediate steps; $r_\phi(x, y)$ at end-of-sequence token

This is a challenging RL problem: extremely large action space, very sparse rewards (only at the end of generation), and extremely long episodes.

PPO handles this by computing the reward at the end and back-propagating through the policy gradient.

## DPO: The PPO-Free Alternative

Direct Preference Optimization (DPO) skips RL entirely. It shows that optimizing the RLHF objective is equivalent to a simple supervised loss on preference data:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Simpler to implement, more stable, often competitive with PPO-based RLHF. Many modern LLMs (Llama-based models) use DPO instead of full PPO.

## The Connection: Why PPO?

Why PPO specifically for RLHF, and not DQN or other methods?

1. **Continuous/discrete hybrid:** The action space (tokens) is discrete but enormous. DQN requires computing Q-values for every action — impractical for 50k+ tokens. PPO works with policy probabilities directly.
2. **Stability:** PPO's clipped objective prevents catastrophic policy collapse, critical when the base model is a $70B$-parameter LLM that took months to train.
3. **On-policy data:** Each generated response is new data. PPO's few-epoch reuse of experience is exactly right for this setting.

---

# 11. Common Interview Questions

---

## Q1: What is the difference between model-based and model-free RL?

**Model-based:** The agent learns or is given a model of the environment — transition dynamics $P(s' \mid s, a)$ and reward function $R(s, a)$. It can use this model to plan (e.g., simulate future trajectories) before acting.

**Model-free:** No model. The agent learns purely from interactions. It must experience something to learn from it.

**Tradeoff:** Model-based is more sample efficient (can plan with fewer real interactions) but harder to get right (model errors compound). Model-free is simpler and more robust but data-hungry.

**Examples:** AlphaZero (model-based, uses MCTS with a learned model). DQN, PPO (model-free).

---

## Q2: What is the exploration-exploitation tradeoff?

Exploitation: act according to your current best knowledge to maximize immediate reward. Exploration: try actions you're uncertain about to gain information.

**Why it matters:** A fully exploitative agent gets stuck in local optima — it never discovers that the restaurant on the other side of town is much better. A fully exploratory agent wastes time in clearly bad states.

**In practice:** $\epsilon$-greedy (DQN), entropy bonus (A2C/PPO), UCB (bandits), Thompson sampling (Bayesian bandits), curiosity-based intrinsic rewards (deep RL).

---

## Q3: Explain the Bellman equation in plain English.

The value of being in a state equals: the immediate reward you expect to get, plus the discounted value of wherever you expect to end up.

It is recursive: to know how good state $s$ is, you need to know how good all the states you might transition to are. Solving this self-referential system (either exactly via DP, or approximately via learning) is what RL algorithms do.

---

## Q4: What is the credit assignment problem?

In a game of chess, you make 40 moves and then win or lose. Which of the 40 moves was responsible for the outcome?

In RL, when rewards are delayed and sparse, it is hard to know which actions led to the final reward. This is credit assignment.

Solutions: discounting (actions closer to the reward get more credit), eligibility traces (TD($\lambda$)), advantage estimation (how much better was this specific action than average?).

---

## Q5: What is the difference between on-policy and off-policy learning?

**On-policy:** Learn about the policy you're currently using to collect experience. If you change the policy, old data becomes stale. Examples: SARSA, A2C, PPO.

**Off-policy:** Learn about a policy different from the one collecting data. Can reuse old experiences. Examples: Q-learning, DQN.

**Implication:** Off-policy methods can use experience replay (more data efficient). On-policy methods need fresh data but are typically more stable.

---

## Q6: Why does DQN use a target network?

Without a target network, both sides of the loss equation change simultaneously:

$$\mathcal{L} = \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2$$

Gradient updates on $\theta$ change both $Q(s, a; \theta)$ (what you're updating) and $\max_{a'} Q(s', a'; \theta)$ (the target). This is like trying to hit a target that moves every time you shoot at it.

The target network $\theta^-$ is a frozen copy. It provides stable targets for $N$ steps. This greatly stabilizes training.

---

## Q7: What is the policy gradient theorem?

The gradient of expected return with respect to policy parameters is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

**Why it works:** The log-derivative trick rewrites the gradient of an expectation as an expectation of a gradient-weighted quantity. This allows Monte Carlo estimation — we don't need to differentiate through the environment, just through the policy.

**Intuition:** Increase the probability of actions that led to high returns. Decrease for low returns. Weighted by how good/bad the outcome was.

---

## Q8: Why does REINFORCE have high variance? How do baselines help?

$G_t$ in the REINFORCE update is the actual return from a single episode — which varies enormously across runs due to environment stochasticity and early decisions. A lucky episode and an unlucky episode starting from the same state may have very different returns.

A baseline $b(s_t)$ is subtracted: $(G_t - b(s_t))$. The gradient estimate remains unbiased because $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b(s_t)] = 0$.

Using the value function $V(s_t)$ as baseline gives the advantage $A_t = G_t - V(s_t)$, which captures "how much better was this return than expected?" — much lower variance.

---

## Q9: Explain the Actor-Critic architecture.

**Actor:** The policy $\pi_\theta(a \mid s)$. Takes an action. Updated by policy gradient.

**Critic:** The value function $V_\phi(s)$. Evaluates the action taken. Updated by minimizing TD error.

The critic provides a low-variance baseline (advantage estimate) for the actor's gradient update. The actor's policy improvement makes the critic's job easier (better states to evaluate). They bootstrap each other.

Key advantage over REINFORCE: doesn't need to wait for episode end. Updates happen at every step using TD estimates.

---

## Q10: What is PPO and why is it preferred over vanilla policy gradients?

PPO addresses the instability of large policy gradient steps by clipping the importance sampling ratio:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

This prevents any single update from changing the policy too drastically. Benefits:

1. No catastrophic performance collapse
2. Can reuse data for multiple gradient epochs (more sample efficient than A2C)
3. Simpler to implement than TRPO (no constrained optimization)
4. Naturally handles continuous action spaces

PPO is the default algorithm for most modern RL applications including RLHF.

---

## Q11: What is RLHF and how does it work?

RLHF fine-tunes LLMs to be helpful using human preference signals instead of ground-truth labels:

1. **SFT:** Fine-tune the base model on high-quality demonstrations.
2. **Reward model:** Train a model to score responses based on human preference comparisons ($y_w$ preferred over $y_l$).
3. **PPO:** Use PPO to optimize the LLM's policy to maximize reward model scores, with a KL penalty against the SFT model to prevent reward hacking.

This aligns the model's behavior with human preferences without requiring explicit human-written scores for every possible response.

---

## Q12: What is reward hacking / Goodhart's Law in RL?

When a measure becomes a target, it ceases to be a good measure. In RL: when the agent optimizes the reward signal too hard, it finds behaviors that score high on the reward but don't correspond to the intended behavior.

Classic example: a cleaning robot trained to minimize "visible dirt" learns to cover the dirt rather than remove it.

RLHF example: LLMs optimized too hard on a reward model learn to produce very long, verbose responses that score high on the reward model but are not actually helpful.

Mitigations: KL penalty (PPO in RLHF), reward model regularization, iterated RLHF (update reward model with new data).

---

## Q13: What is the difference between value iteration and policy iteration?

Both solve MDPs with known dynamics. Both converge to the optimal policy.

**Policy iteration:** Alternate between full policy evaluation (compute $V^\pi$ exactly) and policy improvement (act greedy w.r.t. $V^\pi$). Fewer outer iterations but each evaluation is expensive.

**Value iteration:** Apply Bellman optimality equation directly as an update: $V(s) \leftarrow \max_a [R + \gamma V(s')]$. More outer iterations but each is cheap. Often faster in practice for large state spaces.

---

## Q14: What is the advantage function?

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

It measures how much better action $a$ is compared to the average action in state $s$ under policy $\pi$.

- $A > 0$: taking action $a$ is better than average — increase its probability
- $A < 0$: taking action $a$ is worse than average — decrease its probability
- $A = 0$: taking action $a$ is exactly average

The advantage reduces gradient variance compared to using $Q$ directly (because the mean $V(s)$ is subtracted out).

---

## Q15: Why can't you use DQN for continuous action spaces?

DQN requires computing $\max_{a'} Q(s', a')$. If the action space is continuous (e.g., $a \in \mathbb{R}^{10}$), this is a high-dimensional optimization problem at every step — not tractable.

Solutions for continuous actions:
- **Policy gradient methods** (PPO, A3C): parameterize the policy directly, output Gaussian distributions over actions
- **SAC (Soft Actor-Critic):** off-policy actor-critic with entropy maximization, state-of-the-art for continuous control
- **TD3:** twin delayed DDPG, addresses overestimation bias for continuous action DPG

---

## Q16: What is the difference between SARSA and Q-Learning?

Both update Q-values. The difference is what they use for the next-step value:

- **Q-Learning (off-policy):** $Q(s', \arg\max_{a'} Q(s', a'))$ — the greedy action, regardless of what the agent actually does
- **SARSA (on-policy):** $Q(s', a')$ — the action the agent actually took next (sampled from its current policy)

Q-Learning learns the optimal policy even while following a sub-optimal exploratory policy. SARSA learns the best policy given that exploration will happen. In risky environments, SARSA is safer.

---

## Q17: What is the Markov property and when does it fail?

The Markov property: the future is conditionally independent of the past given the present state.

It fails when the state is **partial** — when the current observation doesn't capture all relevant history. Example: in poker, your hand doesn't reveal opponents' cards. Example: in a video game, a single frame doesn't tell you the velocity of moving objects.

Solution: use **frame stacking** (DQN stacks 4 frames to infer velocity), or use RNNs/LSTMs to maintain a hidden state that summarizes history (POMDPs — partially observable MDPs).

---

## Q18: Explain experience replay and why it helps.

Experience replay stores past transitions $(s, a, r, s', \text{done})$ in a buffer. During training, random mini-batches are sampled from this buffer rather than using sequential data.

**Benefits:**
1. **Breaks temporal correlation:** Sequential RL data violates the i.i.d. assumption of SGD. Random sampling from the buffer restores approximate independence.
2. **Data efficiency:** Each experience can be replayed multiple times, extracting more learning signal per environment interaction.
3. **Stability:** Diverse mini-batches prevent the network from over-adapting to recent narrow experience.

---

## Q19: What is GAE (Generalized Advantage Estimation)?

GAE smoothly interpolates between the 1-step TD advantage (low variance, high bias) and the Monte Carlo advantage (unbiased, high variance):

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

At $\lambda = 0$: 1-step TD advantage. At $\lambda = 1$: full MC advantage. In PPO, $\lambda = 0.95$ works well empirically.

---

## Q20: Compare Q-Learning, DQN, PPO, and SAC.

| Algorithm | Type | Action Space | Key Feature | Typical Use |
| :--- | :--- | :--- | :--- | :--- |
| Q-Learning | Value-based, off-policy | Discrete | Tabular Q-table | Small discrete MDPs |
| DQN | Value-based, off-policy | Discrete | Neural Q-function + replay + target net | Atari games |
| PPO | Policy gradient, on-policy | Both | Clipped objective, stable | RLHF, robotics, games |
| SAC | Actor-Critic, off-policy | Continuous | Entropy maximization | Continuous control, robotics |

**Rule of thumb:**
- Discrete actions: DQN or PPO
- Continuous actions: PPO or SAC
- Sample efficiency matters: SAC (off-policy, replay buffer)
- Stability and RLHF: PPO

---

# 12. Key Equations Reference

## Bellman Equations

**Value function (policy $\pi$):**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**Action-value function (policy $\pi$):**
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]$$

**Bellman optimality (V):**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

**Bellman optimality (Q):**
$$Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

## TD Updates

**TD(0) value update:**
$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**Q-Learning update:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**SARSA update:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

## Policy Gradient

**REINFORCE gradient:**
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**Advantage function:**
$$A(s,a) = Q(s,a) - V(s)$$

**TD advantage (critic estimate):**
$$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**GAE:**
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## PPO

**Importance sampling ratio:**
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

**Clipped objective:**
$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

## DQN Loss

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

## RLHF Reward Model

$$\mathcal{L}(\phi) = -\mathbb{E} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**PPO RLHF Objective:**
$$\max_\theta \mathbb{E} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

## Bandits

**UCB action selection:**
$$a_t = \arg\max_a \left[ \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}} \right]$$

**Expected Regret:**
$$\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^T r_t$$

---

## Algorithm Decision Tree

```
Is the environment known (transition dynamics available)?
├── Yes → Dynamic Programming (Value Iteration / Policy Iteration)
└── No → Model-Free RL
    ├── Is the action space discrete?
    │   ├── Yes, small state space → Q-Learning (tabular)
    │   └── Yes, large/continuous state → DQN, Double DQN, Dueling DQN
    └── Is the action space continuous?
        ├── On-policy, stability important → PPO
        ├── Off-policy, sample efficiency → SAC
        └── Is it an LLM you're aligning? → PPO + RLHF or DPO
```

---

## Summary: The RL Zoo

| Algorithm | Year | Key Idea | Best At |
| :--- | :--- | :--- | :--- |
| Q-Learning | 1989 | Bellman optimality, tabular | Theory, small MDPs |
| SARSA | 1994 | On-policy TD | Safe exploration |
| REINFORCE | 1992 | Policy gradient, MC | Simple policy optimization |
| Actor-Critic | 1999 | Policy + value together | Reduced variance |
| DQN | 2013 | Neural Q + replay + target net | Atari, discrete actions |
| A3C | 2016 | Async parallel actors | Speed, diversity |
| PPO | 2017 | Clipped trust region | General-purpose, RLHF |
| SAC | 2018 | Entropy maximization | Continuous control |
| RLHF | 2022+ | Human preferences as reward | LLM alignment |
| DPO | 2023 | RL-free preference optimization | LLM alignment (simpler) |
