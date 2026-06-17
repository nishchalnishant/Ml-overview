---
module: Specialized Domains
topic: Reinforcement Learning
subtopic: ""
status: unread
tags: [specializeddomains, ml, reinforcement-learning]
---
# Reinforcement Learning

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

**The problem:** Supervised learning requires labeled examples. But for many sequential decision-making tasks, you cannot enumerate the correct action for every situation. You can only evaluate outcomes after a sequence of decisions: a game won or lost, a robot arm that succeeded or failed, a conversation that was helpful or harmful. No oracle can label each intermediate action; only the eventual result signals quality.

**The core insight:** Learning from delayed outcome signals is possible if you model the problem as an agent interacting with an environment. The agent takes actions, observes consequences, and receives occasional rewards. By accumulating experience, it can infer which sequences of actions lead to good outcomes — without anyone labeling each step.

**The mechanics:** An agent observes state s, takes action a, receives reward r, transitions to next state s'. Over many interactions it learns a policy π(a|s) that maximizes cumulative reward.

**What breaks:** RL is expensive (millions of interactions), unstable (policy changes can cause catastrophic forgetting), and hard to debug (the reward signal may be sparse, delayed, or hackable). Use supervised learning when you have labels. Use RL when the problem is inherently sequential and labels are not available — only outcome signals.

| Scenario | Why RL fits |
| :--- | :--- |
| Game playing (Go, Chess, Atari) | Clear reward signal, well-defined rules |
| Robotics control | Sequential physical actions with delayed feedback |
| Recommendation systems | User engagement as reward signal |
| Trading algorithms | Profit/loss as reward |
| LLM fine-tuning (RLHF) | Human preferences as reward signal |
| Resource scheduling | Minimize latency/cost over time |

---

# 2. Core Concepts

**The problem:** Sequential decision-making involves too many interdependent quantities to reason about without a clean vocabulary. "How good is it to be here?" and "what should I do here?" are different questions. Confusing them leads to confused algorithms.

**The core insight:** Decompose the problem into roles. The agent acts. The environment responds. State describes the current situation. Reward evaluates the last action. Policy maps states to actions. Value predicts future rewards. These decompositions are not just terminology — they correspond to separate mathematical objects that can be learned and composed independently.

### Agent

The learner and decision-maker. In a video game: the player. In robotics: the robot. In RLHF: the language model. The agent observes the current situation, picks an action, and tries to maximize the total reward it accumulates over time.

### Environment

Everything the agent interacts with but does not directly control. Receives an action, returns the next state and a reward.

### State (s)

A description of the current situation complete enough for the agent to make a decision. In chess: the board position. In robotics: joint angles and velocities.

**Key distinction:** state vs. observation. A state is the full truth about the world. An observation is what the agent actually sees (may be partial). In poker, you observe your cards — the state includes everyone's cards.

### Action (a)

What the agent does. Discrete (move left/right/up/down) or continuous (apply 2.7 Nm of torque). The set of all possible actions is the **action space**.

### Reward (r)

A scalar signal after each action. The reward function is the most important design decision in any RL system. Get it wrong and your agent finds creative, unintended ways to maximize it. This is called **reward hacking**.

### Policy (π)

**The problem:** The agent needs a mapping from situations to actions. Without a policy, there is no decision.

**The core insight:** A policy is that mapping — either deterministic (always pick the same action in a state) or stochastic (pick from a distribution). Stochastic policies are better when the optimal strategy requires randomization, or when still exploring.

$$\pi(a \mid s) = P(\text{take action } a \mid \text{in state } s)$$

### Value Function (V)

**The problem:** "What action should I take?" requires knowing not just immediate reward but long-term consequences. A greedy agent that only looks at immediate reward will take terrible long-term decisions.

**The core insight:** The value function answers "how good is it to be in this state?" by capturing the expected cumulative future reward from that state forward under the current policy. The action-value function Q^π(s,a) asks the same question but also fixes the first action — making it directly usable for choosing actions.

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]$$

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s, a_0 = a \right]$$

V^π(s) does not let you choose the first action. Q^π(s,a) does. This makes Q more useful for extracting a policy.

### Model (optional)

**The problem:** Acting without knowing the consequences of actions requires sampling from the environment. If you already know how the world works, you could plan internally instead.

**The core insight:** A model of the environment — transition dynamics P(s'|s,a) and reward function R(s,a) — lets you simulate trajectories without real interactions. Algorithms that use a model are **model-based**. Those that do not are **model-free**. Most modern RL (DQN, PPO, SAC) is model-free.

**What breaks (Core Concepts broadly):** Without a good state representation, the Markov assumption breaks down. If the state does not capture enough information to predict the next state and reward, the agent's value estimates are systematically wrong and no algorithm can recover.

---

# 3. Markov Decision Processes (MDPs)

**The problem:** Sequential decision-making problems come in many forms. Without a formal framework, it is impossible to define precisely what "optimal behavior" means or to prove that an algorithm finds it. Different problems use incompatible language.

**The core insight:** Almost every sequential decision problem can be cast as an MDP: a tuple (S, A, P, R, γ). The key assumption — the Markov property — says that the future depends only on the present state, not the history of how you arrived there. This assumption makes the problem tractable: if the past is irrelevant given the present, you only need to track the current state.

**The mechanics:**

$$\text{MDP} = (S, A, P, R, \gamma)$$

| Symbol | Meaning |
| :--- | :--- |
| S | Set of all states |
| A | Set of all actions |
| P(s' \| s, a) | Transition probability: chance of landing in s' after taking a in s |
| R(s, a, s') | Reward received after the transition |
| γ ∈ [0, 1) | Discount factor |

### The Markov Property

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

If you know the current state, the past is irrelevant for predicting the future. This is why defining a good state representation is critical.

### Discounting

**The problem:** For infinite-horizon problems, total reward is an infinite sum that may diverge. Also, rewards far in the future are less certain and less relevant.

**The core insight:** Weight future rewards by γ^t. This ensures the sum converges and encodes the intuition that a reward now is worth more than the same reward later.

$$\text{Return } G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

At γ = 0: agent is completely myopic. At γ = 0.99: agent values future rewards nearly as much as immediate ones.

**What breaks:** Choosing γ too low makes the agent short-sighted — it ignores consequences beyond a few steps. Choosing γ too high (near 1) makes convergence slow and the value function sensitive to noise in distant future estimates.

### Bellman Equations

**The problem:** Value functions are defined recursively — the value of a state depends on the values of successor states. How do you solve a self-referential definition?

**The core insight:** The recursive structure is an asset. The Bellman equations express V(s) as a function of V(s'), which can be solved iteratively. Every RL algorithm is, at its core, a way to solve or approximate these equations.

**Bellman Expectation Equation for V^π:**

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

"The value of being in state s under policy π equals the expected immediate reward plus the discounted value of wherever I land next."

**Bellman Optimality Equations:**

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

Once you have Q*, extracting the optimal policy is trivial:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

**What breaks:** The Markov assumption fails when the observation is partial — a single frame does not reveal the velocity of moving objects; a poker hand does not reveal opponents' cards. The full true state is not always observable, creating a Partially Observable MDP (POMDP) that is substantially harder to solve.

---

# 4. Dynamic Programming

**The problem:** The Bellman equations define the value functions but do not tell you how to compute them. With a known model, how do you actually find the optimal policy?

**The core insight:** With a known model, the Bellman equations form a solvable system. Iteratively applying them — alternating between evaluating the current policy and improving it greedily — converges to the optimal policy. DP is not practical for real-world RL (you rarely know the full model) but establishes the theoretical foundation for everything else.

## Policy Evaluation

**The problem:** Given a policy π, what is V^π? The Bellman equation defines V^π recursively but does not give a closed form.

**The core insight:** Repeated application of the Bellman expectation equation is a contraction mapping — it converges to V^π.

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

## Policy Improvement

**The problem:** Once you have V^π, is π actually the best policy?

**The core insight:** Acting greedily with respect to V^π produces a policy π' that is at least as good as π — the Policy Improvement Theorem guarantees V^{π'}(s) ≥ V^π(s) for all s.

$$\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

## Policy Iteration

**The problem:** One round of evaluation and improvement is not enough. How do you reach the optimal policy?

**The core insight:** Alternate evaluation and improvement. Because each improvement step strictly increases value (or terminates), and the number of policies is finite, convergence to the optimal policy is guaranteed.

```
Initialize policy π arbitrarily
Repeat:
    1. Policy Evaluation:  compute V^π
    2. Policy Improvement: π' = greedy(V^π)
    3. If π' == π: stop (optimal policy found)
    4. π ← π'
```

## Value Iteration

**The problem:** Policy iteration's inner loop requires full convergence of policy evaluation — expensive for large state spaces.

**The core insight:** You do not need to evaluate the current policy exactly before improving. Apply the Bellman optimality equation directly in a single sweep. This merges evaluation and improvement into one operation.

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
Extract policy:
    π(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s)]
```

| | Policy Iteration | Value Iteration |
| :--- | :--- | :--- |
| Inner loop | Full policy eval to convergence | Single backup per state |
| Outer iterations | Fewer | More |
| Practical speed | Often faster overall | Simpler to implement |

**What breaks:** DP requires knowing the full transition model P(s'|s,a). For most real problems this is unknown. Also, DP requires iterating over all states — infeasible for large or continuous state spaces. This motivates model-free and function approximation approaches.

---

# 5. Model-Free RL

**The problem:** In most real problems, you do not know P(s'|s,a). You only know what actually happened: "I took action a in state s and ended up in s' with reward r." DP cannot run without the model.

**The core insight:** You can still learn V or Q by collecting samples of experience and treating those samples as noisy estimates of the Bellman equations. Two families differ in when they update: Monte Carlo waits for the episode to end; Temporal Difference updates at every step.

## 5.1 Monte Carlo Methods

**The problem:** You need an unbiased estimate of V(s). With no model, you cannot compute the Bellman expectation directly.

**The core insight:** Run a full episode. The actual return G_t is an unbiased sample of V^π(s). Average many such samples. No model needed — just real experience.

$$G_t = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T$$

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

**What breaks:** MC has high variance because G_t depends on all the stochastic events in the episode. Many episodes are needed for stable estimates. It is also useless for continuous (non-episodic) tasks because you never get the complete return.

## 5.2 Temporal Difference Learning

**The problem:** MC requires waiting for the episode to end before updating. For long episodes or continuous tasks, this is too slow or impossible.

**The core insight:** You do not need to wait for the actual return. Use the current value estimate for the next state as a stand-in. This bootstrapping gives you an update at every single step. You trade unbiasedness for immediacy and lower variance.

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]$$

The term δ_t = r_{t+1} + γV(s_{t+1}) - V(s_t) is the **TD error** — the difference between what you expected and what actually happened plus the discounted estimate of the future.

| | Monte Carlo | TD |
| :--- | :--- | :--- |
| Update timing | End of episode | Every step |
| Bias | Unbiased | Biased (bootstrapping) |
| Variance | High | Low |
| Requires episodes | Yes | No |

**TD(λ):** TD(0) uses one step of lookahead. MC uses the full episode. TD(λ) interpolates between them using eligibility traces. At λ=0: TD(0). At λ=1: equivalent to MC.

**What breaks:** TD bootstrapping introduces bias — your estimate of V(s_{t+1}) may be wrong, and you are updating V(s_t) toward a wrong target. For early training when all value estimates are poor, this bias can slow convergence significantly.

## 5.3 Q-Learning

**The problem:** TD updates for V^π do not directly tell you which action to take. You still need the model to extract the greedy policy from V. How do you learn Q*(s,a) directly from samples, without a model?

**The core insight:** The Bellman optimality equation for Q* says:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

Each sample (s, a, r, s') is a noisy observation of this expectation. Update Q toward the sample target:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Once Q* is learned, the optimal policy follows immediately: π*(s) = argmax_a Q*(s,a).

**Why "off-policy":** The agent uses ε-greedy for exploration but the update uses max_{a'} Q(s',a') — the greedy action — regardless of what the agent actually did next. The learning target does not depend on the behavior policy. This means you can learn from old data, replayed experiences, or watching someone else.

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - current_q)
```

## 5.4 SARSA

**The problem:** Q-learning learns the optimal policy but ignores the cost of exploration during learning. In risky environments, the exploratory actions can cause harm — and Q-learning's off-policy update does not account for them.

**The core insight:** Use the action the agent actually takes next in the update, not the greedy action. This makes SARSA on-policy: it learns the best policy given that the agent will still be exploring.

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where a' is the actual next action taken (not the greedy one).

Q-Learning learns the optimal policy even while exploring. SARSA learns the best policy given that exploration will happen. In risky environments where accidental falls off a cliff cost real money or safety, SARSA is safer.

**What breaks:** SARSA is slower to converge to the optimal policy than Q-learning because its target is distorted by exploration. If you need the best policy and can afford the training cost, Q-learning wins. If safety during training matters, SARSA wins.

**What breaks (Model-Free broadly):** Both Q-learning and SARSA require a table indexed by (state, action). For large or continuous state spaces, the table does not fit in memory. This motivates function approximation — replacing the table with a neural network.

---

# 6. Deep Q-Networks (DQN)

**The problem:** Q-learning with a table requires storing and updating a value for every (state, action) pair. A video game frame is 210×160×3 pixels — the state space is astronomically large. No table can hold it.

**The core insight:** Replace the Q-table with a neural network parameterized by θ: input state s, output Q(s,a;θ) for all actions. But naively substituting a neural net into Q-learning fails catastrophically due to two interacting problems: correlated training data and non-stationary targets.

**The mechanics:** DQN fixes both problems with two innovations: experience replay and target networks.

### Experience Replay

**The problem:** Sequential RL data violates the i.i.d. assumption of SGD. Frame t is highly correlated with frame t+1. Training on sequential data causes the network to overfit to recent experience and catastrophically forget everything else.

**The core insight:** Store transitions in a replay buffer. Sample random mini-batches for training. The random sampling breaks temporal correlations, restoring approximate i.i.d. conditions. As a bonus, each experience can be replayed multiple times, improving data efficiency.

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

**What breaks:** The replay buffer stores transitions from the current policy. For on-policy algorithms, old data is invalid because the policy changed. Experience replay only works for off-policy algorithms like DQN.

### Target Network

**The problem:** In the Q-learning loss, the same network θ appears on both sides:

$$\mathcal{L} = \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2$$

Every gradient update changes both the prediction Q(s,a;θ) and the target max Q(s',a';θ) simultaneously. The target is a moving object that your update keeps moving further. Training oscillates or diverges.

**The core insight:** Maintain two networks. The online network θ is updated every step. The target network θ^- is a frozen copy, updated only every N steps. The target in the loss is computed from the frozen network, giving stable training targets for N steps at a time.

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \text{buffer}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
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
        
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
```

## DQN Improvements

### Double DQN

**The problem:** Standard DQN overestimates Q-values. The max_{a'} Q(s',a';θ^-) in the target uses the same network to both select and evaluate the best action. Because noise in the network causes some actions to appear falsely high, the max consistently picks overestimated values. Overestimated Q-values bias the policy toward states that are not actually good.

**The core insight:** Decouple selection and evaluation. Use the online network to select which action is best; use the target network to evaluate how good that action is. This removes the systematic upward bias.

$$\text{target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

**What breaks:** Double DQN reduces but does not eliminate overestimation — both networks are correlated because the target network is a lagged copy of the online network. The bias is smaller but still present.

### Dueling DQN

**The problem:** For many states, most actions have similar value. The agent does not need to know Q(s,a) for every action — it mainly needs to know how good the state is and which action is slightly better than others. Estimating Q(s,a) directly conflates these two signals.

**The core insight:** Separate Q into V(s) (how good is this state?) and A(s,a) (how much better is this action than average?). V(s) can be updated from any action; A(s,a) captures action-specific advantages. This separation improves learning efficiency when most actions have similar value.

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')$$

The subtraction of the mean advantage ensures uniqueness: without it, V and A are not separately identifiable (you could add a constant to V and subtract it from all A values and get the same Q).

**What breaks:** The decomposition is most useful in states where many actions have nearly identical Q-values. For tasks where action selection is critical everywhere, dueling DQN may offer little advantage.

### Prioritized Experience Replay (PER)

**The problem:** Uniform sampling from the replay buffer treats all experiences as equally informative. But most stored transitions are unsurprising — the agent already knows their outcome well. Experiences where the agent was surprised carry more information and should be trained on more.

**The core insight:** Sample experiences with probability proportional to their TD error magnitude. Large TD error means the agent's current estimate is far from the actual target — that experience contains more information to learn from.

$$P(i) \propto |\delta_i|^\alpha$$

Importance sampling weights correct for the biased sampling distribution: w_i = (1 / N P(i))^β.

**What breaks:** PER requires maintaining a priority queue over the buffer, which adds computational overhead. Stale priorities accumulate — the TD error used to assign priority was computed under an older network, not the current one. This requires periodic priority updates or accepting stale priorities.

**What breaks (DQN broadly):** DQN requires a discrete action space. Computing max_{a'} Q(s',a') is infeasible for continuous actions. For continuous control, use policy gradient methods (PPO, SAC) instead.

---

# 7. Policy Gradient Methods

**The problem:** Q-learning and DQN derive the policy implicitly from Q-values: π(s) = argmax_a Q(s,a). This requires that the argmax be computable — which fails for continuous action spaces and requires a separate approximation for stochastic policies. Also, small errors in Q propagate into the policy in opaque ways.

**The core insight:** Directly parameterize the policy π_θ(a|s) as a neural network and optimize it end-to-end. The gradient of expected return with respect to θ can be computed without knowing the environment dynamics, using the log-derivative trick.

## The Policy Gradient Theorem

We want to maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G(\tau)]$$

The gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

**The mechanics of the log-derivative trick:** ∇_θ P(τ;θ) = P(τ;θ) ∇_θ log P(τ;θ). This rewrites the gradient of an expectation as an expectation of a log-gradient — computable by Monte Carlo sampling without differentiating through the environment.

**Intuition:** Increase the log-probability of actions that led to high returns. Decrease it for actions that led to low returns.

## 7.1 REINFORCE

**The problem:** You have the policy gradient theorem but need a concrete algorithm. Run episodes, compute returns, update the policy.

**The core insight:** Each complete episode provides a Monte Carlo estimate of the policy gradient. The return G_t is the actual discounted reward from step t — an unbiased signal. No value function, no model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
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
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = -torch.stack(log_probs) * returns
        loss = loss.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### The Baseline Trick

**The problem:** G_t varies wildly across episodes. High variance makes learning slow — the gradient estimate is correct on average but noisy enough that many samples are needed before the signal dominates the noise.

**The core insight:** Subtracting a baseline b(s_t) from G_t leaves the gradient estimate unbiased — the baseline contributes zero in expectation because E[∇_θ log π_θ(a_t|s_t) · b(s_t)] = 0 — but reduces variance. The natural baseline is V(s_t). The quantity G_t - V(s_t) is the **advantage** A_t: how much better was this action than average.

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]$$

**What breaks (REINFORCE):** REINFORCE requires complete episodes before any update. It is slow and cannot handle continuous (non-episodic) tasks. The variance, even with a baseline, is high because G_t depends on every random event in the remainder of the episode.

## 7.2 Actor-Critic

**The problem:** REINFORCE waits for the full episode to compute G_t. For long episodes this is extremely slow — thousands of steps before a single gradient update. Also, G_t is high variance because it aggregates all the stochasticity in the rest of the episode.

**The core insight:** Maintain a critic (value function V_φ) that provides a TD-based estimate of the advantage at every step, without waiting for the episode to end:

$$A_t \approx r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

The actor (policy π_θ) uses this online advantage estimate to update at every step. The critic reduces variance compared to Monte Carlo returns by substituting learned V estimates for unknown future returns.

```python
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions), nn.Softmax(dim=-1)
        )
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

The entropy bonus H(π_θ) encourages exploration by preventing the policy from collapsing to a deterministic choice prematurely.

**What breaks:** The critic's TD estimates introduce bias — V_φ is a learned approximation, not the true value. If V_φ is systematically wrong early in training, the policy gradient updates point in the wrong direction. Also, sharing network layers between actor and critic means the two objectives can interfere: critic gradients pulling the shared layers may destabilize the actor.

## 7.3 A2C and A3C

**The problem:** A single Actor-Critic agent collects highly correlated sequential experience — consecutive states are nearly identical. Correlated data destabilizes training. Also, a single agent is slow.

**The core insight:** Run multiple agents in parallel on independent copies of the environment. Their experiences are diverse and uncorrelated by construction. Multiple workers provide the data diversity that experience replay provides for DQN — but without requiring off-policy correction.

**A3C (Asynchronous Advantage Actor-Critic):** Workers asynchronously update a global network. Each worker computes gradients locally and applies them to the global network immediately, without waiting for other workers.

**A2C (Synchronous):** All workers gather experience simultaneously, then do a synchronized update using the combined mini-batch. More stable than A3C in practice because gradient updates do not interfere.

Both use the n-step advantage:

$$A_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$$

**What breaks:** Both A2C and A3C are on-policy — every update requires fresh data. As soon as the policy changes, the data collected under the previous policy becomes stale and cannot be reused. This limits sample efficiency. Also, naive policy gradient updates can cause catastrophic policy collapse: one too-large step completely ruins the policy and there is no recovery mechanism. This motivates PPO.

---

# 8. Proximal Policy Optimization (PPO)

**The problem:** Standard policy gradient methods have a catastrophic failure mode. If you take too large a gradient step, the policy changes dramatically in one update. The new policy is so different from the old one that performance collapses — and because you are now collecting data from the new (bad) policy, you cannot even detect why. There is no natural stopping criterion for "how large a step is too large."

**The core insight:** Do not stray too far from the current policy in a single step. Stay within a "trust region." TRPO formalized this with a constrained optimization problem. PPO approximates the same constraint with a much simpler clipped objective — just as effective, far easier to implement.

**The mechanics:**

### Importance Sampling

**The problem:** Collecting fresh data for every gradient update is expensive (on-policy). Can you reuse data from the old policy for a few gradient steps?

**The core insight:** Yes, using importance sampling. The importance ratio corrects for the distribution shift between old and new policy:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

Without clipping, this ratio can become very large if the new policy diverges significantly from the old one, causing unstable updates.

### The PPO Clipped Objective

**The problem:** The importance ratio r_t(θ) can grow arbitrarily large if the policy changes a lot, causing large destructive updates.

**The core insight:** Clip r_t(θ) to [1-ε, 1+ε]. The min ensures the objective never benefits from pushing the policy outside the trust region in either direction.

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

- If Â > 0 (action was good): increase its probability, but not beyond 1+ε times the old probability.
- If Â < 0 (action was bad): decrease its probability, but not below 1-ε times the old probability.
- The min ensures the objective never benefits from pushing the policy outside the trust region.

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2,
             value_pred=None, returns=None, value_coef=0.5, entropy_coef=0.01,
             entropy=None):
    ratios = torch.exp(log_probs_new - log_probs_old)
    
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    critic_loss = nn.MSELoss()(value_pred, returns)
    
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
    return total_loss
```

### PPO Update Loop

```
Collect T timesteps of experience using current policy π_θ_old
Compute advantages Â_t using GAE
For K epochs:
    For each mini-batch:
        Compute new log probs, values, entropy under π_θ
        Compute clipped PPO loss
        Backprop and update θ
Update θ_old ← θ
```

## Generalized Advantage Estimation (GAE)

**The problem:** One-step TD advantage (low variance, high bias from V errors) and MC advantage (unbiased, high variance) are the two extremes. Neither is optimal. The bias-variance tradeoff for advantage estimation is separate from the bias-variance tradeoff for the return.

**The core insight:** Smoothly interpolate between TD and MC advantages by computing a weighted sum of k-step TD residuals. The λ parameter controls the tradeoff:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where δ_t = r_t + γV(s_{t+1}) - V(s_t). At λ=0: 1-step TD. At λ=1: full MC. λ=0.95 and γ=0.99 are the standard PPO settings.

**What breaks:** The effective decay rate is (γλ)^l. If V_φ is wrong, the bias compounds over many steps — low λ is safer when V_φ is unreliable early in training. λ=0.95 works well empirically but is a hyperparameter that may need tuning.

| Property | REINFORCE | A2C | PPO |
| :--- | :--- | :--- | :--- |
| Sample efficiency | Low | Medium | High |
| Stability | Low | Medium | High |
| Implementation | Simple | Medium | Moderate |
| Continuous actions | Yes | Yes | Yes |
| Used in RLHF | No | No | Yes |

**What breaks (PPO broadly):** PPO is on-policy — stale data from previous policies is not reused beyond a few epochs, limiting sample efficiency. For continuous control tasks where sample efficiency is critical (e.g., robotic manipulation requiring physical hardware), off-policy methods like SAC are preferred.

---

# 9. Multi-Armed Bandits

**The problem:** You must choose between K options with unknown reward distributions. Every choice that explores an uncertain option is an opportunity cost — you could have exploited the best known option instead. Explore too little and you miss the best option. Explore too much and you waste resources on bad options. This is the exploration-exploitation tradeoff in its purest form.

**The core insight:** The optimal exploration strategy depends on uncertainty. Explore options you are least certain about. As you learn, shift toward exploitation. The challenge is formalizing "uncertainty" in a principled way.

## Setup

At each step t, pick arm a_t. Receive reward r_t ~ P_a.

Goal: maximize cumulative reward Σ r_t, or equivalently, minimize **regret**:

$$\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^{T} r_t$$

where μ* is the mean reward of the best arm.

## Key Strategies

### Epsilon-Greedy

**The problem:** You need a concrete, simple exploration policy.

**The core insight:** With probability ε, pick a random arm (explore). With probability 1-ε, pick the arm with the highest estimated mean reward (exploit). Simple but effective.

**What breaks:** ε-greedy explores randomly, not proportionally to uncertainty. An arm pulled 10,000 times and an arm pulled once both have probability ε/K of being chosen. This wastes exploration budget on already well-characterized arms.

### UCB (Upper Confidence Bound)

**The problem:** ε-greedy explores randomly. You should explore options you are most uncertain about, not random ones.

**The core insight:** Add an "optimism bonus" to each arm's estimated value that decreases as you pull it more. Always act optimistically: pick the arm with the highest upper bound. Uncertain arms have large confidence bounds and get selected for exploration; well-characterized arms do not.

$$a_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

where N_a is the number of times arm a has been pulled. UCB is deterministic and provably achieves O(ln T) regret.

**What breaks:** UCB uses a fixed, mathematically derived confidence bound. It does not update based on the actual shape of the reward distribution — it assumes the same exploration bonus structure for all problems, which may be suboptimal when the reward distribution is strongly non-Gaussian.

### Thompson Sampling

**The problem:** UCB uses a fixed confidence bound formula. Can we be more principled by explicitly tracking uncertainty as a probability distribution?

**The core insight:** Maintain a posterior distribution over the mean reward for each arm. At each step, sample a mean from each posterior and pick the arm with the highest sample. As you observe rewards, update posteriors via Bayes' rule. Natural posterior uncertainty drives exploration: arms you know little about have wide posteriors and frequently generate high samples.

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)  # successes + 1
        self.beta = np.ones(n_arms)   # failures + 1
    
    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
```

Thompson Sampling often outperforms UCB empirically and has similar theoretical guarantees.

**What breaks:** Thompson Sampling requires a prior and a tractable posterior. For Bernoulli rewards, Beta-Bernoulli is conjugate and exact. For complex reward distributions, the posterior is intractable and must be approximated — introducing errors that can lead to under- or over-exploration.

## Why Bandits Matter for ML

Bandits appear in: A/B testing (which version performs better?), ad serving (which ad to show?), hyperparameter tuning (which config to try next?), recommendation (which item to surface?).

**What breaks (Bandits broadly):** Bandit algorithms assume stationary reward distributions. If the best arm changes over time (non-stationary bandits), the confidence bounds become stale. Use sliding window or discounted UCB variants for non-stationary settings.

---

# 10. Reinforcement Learning from Human Feedback (RLHF)

**The problem:** LLMs pre-trained on text corpora learn to predict the next token — not to be helpful, harmless, or honest. You cannot write down a loss function that captures "helpful" because human preferences are contextual, subtle, and cannot be reduced to a formula. But you need a training signal that goes beyond next-token prediction.

**The core insight:** Human preferences can be converted into a learned reward signal. Collect pairwise comparisons (human annotators pick which of two responses they prefer), train a reward model on those comparisons, then use RL to optimize the LLM against the learned reward — with a KL penalty to prevent reward hacking.

## The Three-Stage Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

**The problem:** The base LLM produces incoherent or unhelpful responses out-of-the-box. You need a starting point that can follow instructions before applying RL.

**The core insight:** Fine-tune on high-quality prompt-response pairs written or curated by human labelers. Standard cross-entropy loss. This gives a reference model π_ref that can follow instructions reasonably well and defines the "safe zone" for subsequent RL.

### Stage 2: Train a Reward Model

**The problem:** You cannot label individual responses with a scalar "quality" score reliably. Annotators disagree about absolute scores. But you can reliably rank two responses: this one is better than that one.

**The core insight:** Train a reward model r_φ(x,y) using the Bradley-Terry preference model. Score preferred responses higher than rejected ones. Pairwise comparisons are more reliable and consistent than absolute scores.

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

The reward model is typically the SFT model with the final head replaced by a scalar output head.

**What breaks:** The reward model is trained on a finite set of human comparisons. It will be wrong on out-of-distribution inputs. When the policy is subsequently optimized against this reward model with PPO, it will find inputs where the reward model is wrong in its favor — reward hacking.

### Stage 3: RL Fine-Tuning with PPO

**The problem:** Optimizing the LLM against the reward model without constraints will cause reward hacking — the model learns to produce responses that score high on the reward model but are not actually helpful or safe. The optimization pressure is unbounded.

**The core insight:** Add a KL penalty that prevents the policy from drifting too far from the SFT reference model. The KL term acts as a leash: the model can explore improvements but cannot run so far from the safe reference policy that it finds reward model failure modes.

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$

## RLHF as a Token-Level MDP

**The problem:** PPO is designed for step-level rewards. RLHF has only a final response-level reward. How do you apply RL?

**The core insight:** Treat generation as an MDP where each token is an action, and the reward is received only at the end-of-sequence token. All intermediate token steps receive zero reward except for the KL penalty term added per-step.

- **State s_t:** the prompt + all tokens generated so far
- **Action a_t:** the next token to generate (vocabulary size ~50,000)
- **Reward:** 0 for all intermediate steps; r_φ(x, y) at end-of-sequence

This is a challenging RL problem: extremely large action space, sparse rewards (only at episode end), long episodes.

## DPO: The PPO-Free Alternative

**The problem:** PPO-based RLHF is complex and requires running the reward model, the reference model, and the policy model simultaneously during training. Implementation is fragile.

**The core insight:** The RLHF objective with KL penalty has a closed-form optimal policy. This can be directly substituted back into the training objective, yielding a supervised loss on preference data — no RL needed. The reward model is implicitly parameterized by the policy itself.

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Simpler to implement, more stable, often competitive with PPO-based RLHF.

**What breaks (DPO):** DPO requires pre-collected preference pairs — no online exploration. It cannot generate new data or discover response types outside the preference dataset. PPO can in principle explore novel responses that the reward model scores highly. DPO also implicitly assumes the preference data is generated by the reference policy, which may not hold for strongly out-of-distribution responses.

**What breaks (RLHF broadly):** The reward model is trained on human comparison data, which has biases (annotators prefer verbose responses, agreeable responses, confident-sounding responses). These biases get amplified through PPO training. Without careful KL constraints and iterative reward model updates, RLHF can produce models that game the reward model in systematic ways.

---

# 11. Common Interview Questions

## Q1: What is the difference between model-based and model-free RL?

**Model-based:** The agent learns or is given transition dynamics P(s'|s,a) and reward function R(s,a). It uses this model to plan before acting.

**Model-free:** No model. The agent learns purely from interactions.

**Tradeoff:** Model-based is more sample efficient but harder to get right (model errors compound). Model-free is simpler and more robust but data-hungry.

**Examples:** AlphaZero (model-based, MCTS). DQN, PPO (model-free).

---

## Q2: What is the exploration-exploitation tradeoff?

Exploitation: act on current best knowledge to maximize immediate reward. Exploration: try uncertain actions to gain information.

**Why it matters:** A fully exploitative agent gets stuck in local optima. A fully exploratory agent wastes time in bad states.

**In practice:** ε-greedy (DQN), entropy bonus (A2C/PPO), UCB (bandits), Thompson sampling, curiosity-based intrinsic rewards (deep RL).

---

## Q3: Explain the Bellman equation in plain English.

The value of being in a state equals the immediate reward you expect plus the discounted value of wherever you expect to end up. It is recursive: to know how good state s is, you need to know how good all successor states are. Solving this self-referential system is what RL algorithms do.

---

## Q4: What is the credit assignment problem?

In a game of chess, you make 40 moves and win or lose. Which move was responsible? In RL, when rewards are delayed and sparse, it is hard to know which actions led to the final reward.

Solutions: discounting (closer actions get more credit), eligibility traces (TD(λ)), advantage estimation (how much better was this action than average?).

---

## Q5: What is the difference between on-policy and off-policy learning?

**On-policy:** Learn about the policy currently being used to collect experience. Old data becomes stale when the policy changes. Examples: SARSA, A2C, PPO.

**Off-policy:** Learn about a policy different from the one collecting data. Can reuse old experiences. Examples: Q-learning, DQN.

**Implication:** Off-policy methods can use experience replay (more data efficient). On-policy methods need fresh data but are typically more stable.

---

## Q6: Why does DQN use a target network?

Without a target network, both sides of the loss change simultaneously: updating θ changes both the prediction Q(s,a;θ) and the target max Q(s',a';θ). This is like trying to hit a target that moves every time you shoot at it. The target network θ^- is a frozen copy that provides stable targets for N steps, greatly stabilizing training.

---

## Q7: What is the policy gradient theorem?

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

The log-derivative trick rewrites the gradient of an expectation as an expectation of a gradient-weighted quantity. This allows Monte Carlo estimation without differentiating through the environment.

**Intuition:** Increase the probability of actions that led to high returns. Decrease for low returns.

---

## Q8: Why does REINFORCE have high variance? How do baselines help?

G_t in the REINFORCE update is the actual return from a single episode — which varies enormously across runs. A baseline b(s_t) is subtracted: (G_t - b(s_t)). The gradient estimate remains unbiased because E[∇_θ log π_θ(a_t|s_t) · b(s_t)] = 0. Using V(s_t) as baseline gives the advantage A_t = G_t - V(s_t) — much lower variance.

---

## Q9: Explain the Actor-Critic architecture.

**Actor:** Policy π_θ(a|s). Takes an action. Updated by policy gradient.

**Critic:** Value function V_φ(s). Evaluates the action taken. Updated by minimizing TD error.

The critic provides a low-variance baseline for the actor's gradient update. Unlike REINFORCE, updates happen at every step using TD estimates, not at episode end.

---

## Q10: What is PPO and why is it preferred over vanilla policy gradients?

PPO prevents catastrophic large update steps by clipping the importance sampling ratio:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Benefits: no catastrophic performance collapse, can reuse data for multiple gradient epochs, simpler than TRPO (no constrained optimization), handles continuous action spaces. PPO is the default for RLHF.

---

## Q11: What is RLHF and how does it work?

1. **SFT:** Fine-tune the base model on high-quality demonstrations.
2. **Reward model:** Train a model to score responses based on human preference comparisons (y_w preferred over y_l), using Bradley-Terry loss.
3. **PPO:** Optimize the LLM to maximize reward model scores with a KL penalty against the SFT reference model to prevent reward hacking.

---

## Q12: What is reward hacking / Goodhart's Law in RL?

When a measure becomes a target, it ceases to be a good measure. In RL: when the agent optimizes the reward signal too hard, it finds behaviors that score high on the reward but do not correspond to intended behavior.

Example: a cleaning robot trained to minimize "visible dirt" learns to cover the dirt rather than remove it. RLHF example: LLMs optimized too hard on a reward model learn to produce verbose responses that score high but are not useful.

Mitigations: KL penalty (PPO in RLHF), reward model regularization, iterated RLHF.

---

## Q13: What is the difference between value iteration and policy iteration?

Both solve MDPs with known dynamics. Both converge to the optimal policy.

**Policy iteration:** Full policy evaluation (compute V^π exactly) then policy improvement. Fewer outer iterations but each evaluation is expensive.

**Value iteration:** Apply Bellman optimality directly: V(s) ← max_a [R + γV(s')]. More outer iterations but each is cheap. Often faster in practice.

---

## Q14: What is the advantage function?

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

How much better is action a compared to the average action in state s?

- A > 0: taking action a is better than average — increase its probability
- A < 0: taking action a is worse than average — decrease its probability

The advantage reduces gradient variance compared to using Q directly (the mean V(s) is subtracted out).

---

## Q15: Why can't you use DQN for continuous action spaces?

DQN requires computing max_{a'} Q(s',a'). If the action space is continuous (e.g., a ∈ R^10), this is a high-dimensional optimization problem at every step — not tractable.

Solutions: policy gradient methods (PPO outputs Gaussian distributions over actions), SAC (off-policy actor-critic with entropy maximization), TD3 (twin delayed DDPG).

---

## Q16: What is the difference between SARSA and Q-Learning?

- **Q-Learning (off-policy):** Uses max_a' Q(s',a') — the greedy action, regardless of what the agent does next.
- **SARSA (on-policy):** Uses Q(s',a') — the action the agent actually took next.

Q-Learning learns the optimal policy even while following a sub-optimal exploratory policy. SARSA learns the best policy given that exploration will happen. In risky environments, SARSA is safer.

---

## Q17: What is the Markov property and when does it fail?

The Markov property: the future is conditionally independent of the past given the present state.

It fails when the state is partial — when the current observation does not capture all relevant history. In a video game, a single frame does not reveal the velocity of moving objects.

Solution: frame stacking (DQN stacks 4 frames), or RNNs/LSTMs to maintain a hidden state (POMDPs).

---

## Q18: Explain experience replay and why it helps.

Experience replay stores past transitions (s, a, r, s', done) in a buffer. During training, random mini-batches are sampled.

1. **Breaks temporal correlation:** Sequential RL data violates i.i.d. assumptions. Random sampling restores approximate independence.
2. **Data efficiency:** Each experience can be replayed multiple times.
3. **Stability:** Diverse mini-batches prevent overfitting to recent narrow experience.

---

## Q19: What is GAE (Generalized Advantage Estimation)?

GAE smoothly interpolates between 1-step TD (low variance, high bias) and MC (unbiased, high variance):

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where δ_t = r_t + γV(s_{t+1}) - V(s_t). At λ=0: 1-step TD. At λ=1: full MC. λ=0.95 works well in PPO.

---

## Q20: Compare Q-Learning, DQN, PPO, and SAC.

| Algorithm | Type | Action Space | Key Feature | Typical Use |
| :--- | :--- | :--- | :--- | :--- |
| Q-Learning | Value-based, off-policy | Discrete | Tabular Q-table | Small discrete MDPs |
| DQN | Value-based, off-policy | Discrete | Neural Q-function + replay + target net | Atari games |
| PPO | Policy gradient, on-policy | Both | Clipped objective, stable | RLHF, robotics, games |
| SAC | Actor-Critic, off-policy | Continuous | Entropy maximization | Continuous control, robotics |

**Rule of thumb:** Discrete actions: DQN or PPO. Continuous actions: PPO or SAC. Sample efficiency matters: SAC. Stability and RLHF: PPO.

---

# 12. Key Equations Reference

## Bellman Equations

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]$$

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

## TD Updates

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

## Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

$$A(s,a) = Q(s,a) - V(s)$$

$$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## PPO

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

## DQN Loss

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

## RLHF Reward Model

$$\mathcal{L}(\phi) = -\mathbb{E} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

$$\max_\theta \mathbb{E} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

## Bandits

$$a_t = \arg\max_a \left[ \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}} \right]$$

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
