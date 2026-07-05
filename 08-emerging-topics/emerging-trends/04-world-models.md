---
module: Emerging Topics
topic: World Models
subtopic: ""
status: unread
tags: [world-models, video-generation, dreamer, jepa, sora, genie, simulation, rl]
---
# World Models — Learning to Predict and Simulate Reality

**What is a world model?** A world model is a system that learns an internal model of how the world evolves — given the current state and an action (or just time), it can predict the next state. A perfect world model could simulate any environment, enabling agents to plan without interacting with the real world.

**Why it matters:** World models are one of the most important missing pieces in AI systems. They underlie:
- Sample-efficient RL (plan in the model's head, not the real environment)
- Video generation (a generative video model *is* a world model)
- Robot learning and physical simulation
- Model-based planning for agents

---

## Table of Contents
1. [What Is a World Model?](#1-what-is-a-world-model)
2. [Dreamer — Latent World Models for RL](#2-dreamer)
3. [I-JEPA and V-JEPA — Yann LeCun's Vision](#3-jepa-models)
4. [Video Generation as World Modeling](#4-video-generation-as-world-modeling)
5. [Genie 2 — Interactive World Simulation](#5-genie-2)
6. [World Models for Robotics](#6-world-models-for-robotics)
7. [Challenges and Open Problems](#7-challenges)
8. [Interview Questions](#8-interview-questions)

---

## 1. What Is a World Model?

A world model consists of three components:

```
World Model Components:
┌───────────────────────────────────────────────┐
│                                               │
│  Encoder: observation o_t → latent state z_t  │
│                                               │
│  Dynamics model: z_t + action a_t → z_{t+1}  │
│                                               │
│  Decoder: z_t → reconstructed observation ô_t │
│                                               │
│  Reward model: z_t → predicted reward r_t     │
│  (optional, for RL settings)                  │
└───────────────────────────────────────────────┘
```

**The key insight (Sutton's "bitter lesson" meets representation learning):** Instead of interacting with the environment for every decision (expensive in the real world — robots break, games take time, clinical trials take years), learn a compressed model of the environment and simulate interactions in latent space. This is how humans plan: mentally simulating "what if I do X?" without physically doing X.

### Comparison: Model-Free vs. Model-Based RL

| Approach | Data Efficiency | Computation | Planning | Example |
|---|---|---|---|---|
| **Model-free** (PPO, SAC) | Low | Low/medium | None | Most game AI |
| **Model-based** (Dreamer) | High | High | Yes (latent rollouts) | Dreamer, MuZero |
| **World model** (Genie) | High | Very high | Yes | World simulation |

---

## 2. Dreamer — Latent World Models for RL

**DreamerV3** (Hafner et al., Google DeepMind, 2023) is the most capable model-based RL system for continuous control and games.

### Architecture

```
Real Environment:          World Model:                Policy:
o_t → Encoder → z_t   →   z_t + a_t → RSSM → z_{t+1}  →  π(a | z_t)
              ↑                                              ↑
              └─── Learn to encode compactly                └── Trained in imagination
```

**RSSM (Recurrent State Space Model):** The dynamics model combines:
- **Deterministic path** $h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$ — captures temporal dependencies
- **Stochastic path** $z_t \sim q(z_t | h_t, x_t)$ — captures uncertainty in observations

Training: optimize a modified ELBO that balances reconstruction, KL divergence, and prediction loss.

### World Dreamer Training Loop

```python
# DreamerV3 training pseudocode
for step in range(total_steps):
    # === Phase 1: World Model Learning ===
    # Sample batch of real trajectories from replay buffer
    batch = replay_buffer.sample(batch_size=16, seq_len=64)
    
    # Encode observations to latent states
    z = encoder(batch.observations)  # (batch, seq, latent_dim)
    
    # RSSM forward: predict next latent states
    h, z_pred = rssm(z, batch.actions)
    
    # Compute world model loss:
    # reconstruction_loss = MSE(decoder(z), observations)
    # kl_loss = D_KL(posterior || prior)  [regularize latents]
    # reward_loss = MSE(reward_head(z), actual_rewards)
    wm_loss = reconstruction_loss + kl_loss + reward_loss
    wm_optimizer.step(wm_loss)
    
    # === Phase 2: Policy Learning in Imagination ===
    # Roll out policy for H steps in the LEARNED world model
    # No actual environment interaction needed!
    imagined_trajectories = []
    z_init = z[:, -1, :]  # start from last real state
    
    z_curr = z_init
    for h in range(imagination_horizon):  # H = 15 steps
        a_curr = policy(z_curr)            # actor selects action
        z_next = rssm.imagine(z_curr, a_curr)  # simulate in model
        r_pred = reward_head(z_next)       # predict reward
        imagined_trajectories.append((z_curr, a_curr, r_pred))
        z_curr = z_next
    
    # Update policy using imagined returns
    returns = compute_lambda_returns(imagined_trajectories)
    actor_loss = -returns.mean()
    critic_loss = MSE(value_head(z), returns.detach())
    policy_optimizer.step(actor_loss + critic_loss)
    
    # === Phase 3: Real Interaction (occasionally) ===
    # Only interact with real environment ~10% of steps
    if step % 10 == 0:
        real_trajectory = collect_real_data(policy, env)
        replay_buffer.add(real_trajectory)
```

### DreamerV3 Achievements
- **First algorithm** to achieve diamond collection in Minecraft from pixels and sparse rewards — without any domain-specific engineering
- Trains from **raw pixel observations** with 10-100× fewer environment interactions than model-free methods
- Generalizes across diverse domains: control, Atari, DMLab, BSuite, Crafter

---

## 3. JEPA Models — Yann LeCun's Vision

**JEPA (Joint Embedding Predictive Architecture)** is Yann LeCun's proposed architecture for world modeling that avoids the pitfalls of generative models.

### The Problem LeCun Identifies

Autoregressive generative models (GPT-style, diffusion) learn to **generate pixel-level details** of observations. But for planning and reasoning, you don't need to predict exact pixel values — you need to predict the **abstract structure** of what will happen.

```
Problem: Generative world models predict raw observations
o_t → predict → ô_{t+1} (exact pixels, audio, etc.)

High-dimensional, computationally expensive, and prediction
uncertainty compounds across steps → poor long-horizon plans.

JEPA: Instead, predict in ABSTRACT representation space
e(o_t) + a_t → predict → ê(o_{t+1})

Simpler, focuses on structure, tolerates irrelevant variation.
```

### I-JEPA (Image JEPA)

**I-JEPA** (He et al., Meta AI, 2023) learns image representations by predicting masked image patch embeddings from visible patches:

- **Encoder** sees visible patches → context representation
- **Predictor** (lightweight) takes context + mask positions → predicts missing patch embeddings
- **Target encoder** (exponential moving average) generates the target embeddings
- **Key:** Predict in embedding space, not pixel space → focus on semantic structure, not rendering details

This yields representations competitive with DINO and MAE on downstream tasks with better computational efficiency.

### V-JEPA (Video JEPA)

**V-JEPA** (Meta AI, 2024) extends I-JEPA to video:
- Predict embeddings of future or masked video regions from context
- Strong zero-shot performance on action recognition, motion understanding
- 16× fewer parameters needed for fine-tuning vs generative video models

### Why JEPA Matters

The key claim: **predicting in representation space rather than pixel space** is the right inductive bias for world modeling — agents need to know "the ball will fall into the left goal" not "exactly what the pixels will look like when it does."

---

## 4. Video Generation as World Modeling

Modern video generation models are implicitly learning world models — they learn to simulate how pixels evolve under various conditions.

### Sora (OpenAI, 2024)

- **Architecture:** Diffusion Transformer (DiT) operating on video patches (space-time patches = "patchifies")
- **Key capability:** Maintains physical consistency, object permanence, 3D coherence over long videos
- **Implication for world modeling:** OpenAI demonstrated that Sora develops emergent simulation properties — objects persist, light changes consistently, physics is approximately respected

**Technical approach:**
```
Video → Tokenize into space-time patches → Diffusion Transformer
  ↑                                             ↓
  └───────────── Text conditioning ─────────────┘
                (using T5-style text encoder)

Training: DDPM-style noise prediction on video latents
Key: No explicit temporal modeling — full space-time attention
```

### Open Video Generation Models (2025-2026)

| Model | Organization | Type | Notable Feature |
|---|---|---|---|
| **CogVideoX-2B/5B** | Zhipu AI | Open weights | Good quality, fully open |
| **Open-Sora** | Community | Open weights | Sora-inspired, open source |
| **Wan** (万象) | Alibaba | Open weights | Best open-source quality (2025) |
| **HunyuanVideo** | Tencent | Open weights | Long videos |
| **Mochi 1** | Genmo | Open weights | Smooth motion |
| **Pyramid-Flow** | - | Open weights | Multi-resolution training |

### Video as Interactive Simulation

The key frontier question: can a video generation model be used as an **interactive** simulator where actions change outcomes?

**Genie** (DeepMind, 2024): Yes — trained on unlabeled internet gameplay videos, Genie learns an action-conditioned world model. Given a single image, it generates playable game worlds. The model discovers latent actions without action labels.

---

## 5. Genie 2

**Genie 2** (DeepMind, 2024) dramatically extends the original:
- Generate **interactive 3D environments** from a single image prompt
- Supports diverse actions (keyboard, mouse, controller)
- Consistent 3D geometry, physics, object permanence
- Can run at interactive frame rates (~1 second per generated frame with current inference)

```
Input: Single image (screenshot, concept art, photo) + action sequence
↓
Genie 2 foundation model (video diffusion + action conditioning)
↓
Continuous interactive world: move, pick up objects, new areas render dynamically
```

**Key insight:** This demonstrates that world models can be **bootstrapped from passive video observation** (YouTube gameplay) without explicit environment access — just video + unsupervised action discovery.

---

## 6. World Models for Robotics

World models are particularly valuable for robotics where:
- Real robot interaction is slow (30 minutes of training = 1 hour of robot time)
- Failures are dangerous/expensive
- Sim-to-real gap is a major challenge

### UniSim (Yang et al., 2023)

Trains a video generation model on diverse robot trajectories. The video model is then used as a simulator for training downstream robot policies — robots train in the video world model instead of the real world or explicit physics simulators.

**Advantage over physics simulators:** No need to manually specify physical properties — the model learns them from data. Better generalization to novel objects.

### RT-2 / OpenVLA (Google/Open Community)

Vision-language-action models that learn to take robot actions conditioned on language and visual observations. Implicitly learning aspects of world models through the language-understanding component.

### Isaac Sim + Foundation Models

NVIDIA's Isaac Sim provides physics simulation, but foundation models are increasingly used to generate procedural variations (textures, objects, lighting) — world model principles applied to generate diverse training data.

---

## 7. Challenges and Open Problems

### Challenge 1: Compounding Prediction Errors
World model errors compound over long horizons. Predicting 100 steps ahead in a learned model will accumulate errors that make the simulation unreliable. Current work: uncertainty-aware planning (plan conservatively when model is uncertain), periodic correction from real data.

### Challenge 2: Partial Observability
The real world is partially observable — cameras don't see behind objects, sensors are noisy. RSSM handles this via stochastic latents, but long-horizon planning under partial observability remains difficult.

### Challenge 3: Causal Confusion
World models may learn spurious correlations ("when I see the sunset, the score goes up") rather than causal relationships ("pressing right makes the character move right"). Interventional data (actively exploring) is more valuable than purely observational data.

### Challenge 4: Distributional Shift
A model trained in the world model may exploit its inaccuracies — taking actions that look good in the model but fail in reality. This is the "model exploitation" problem in MBRL.

### Challenge 5: High-Dimensional Action Spaces
Robot manipulation with 50+ degrees of freedom, language-conditional action specification — the action space makes world model learning harder than game environments.

---

## 8. Interview Questions

**Q: What is a world model and why is it useful?**  
*A:* A world model is a learned internal model of how an environment transitions: given state s_t and action a_t, predict s_{t+1} and reward r_t. It's useful because: (1) sample efficiency — an agent can plan by simulating in the model without expensive real-world interaction; (2) safe exploration — dangerous actions can be tested in the model first; (3) transfer learning — a good world model generalizes across tasks. The key tradeoff: world models require their own learning resources, and errors in the model can mislead the agent.

**Q: How does DreamerV3 differ from model-free RL like PPO?**  
*A:* DreamerV3 learns a latent world model (RSSM) from observations, then trains the policy entirely in the model's "imagination" — never touching the real environment during policy updates. This makes it 10-100× more sample-efficient. PPO interacts with the real environment at every step. The tradeoff: DreamerV3 requires more computation per real environment step (world model forward passes are expensive), but each real environment step yields much more learning signal because the policy is updated for many imagined steps.

**Q: What is the JEPA approach and how does it differ from generative world models?**  
*A:* JEPA (Joint Embedding Predictive Architecture) predicts in abstract representation space rather than pixel space. Instead of predicting "what will the next frame look like," it predicts "what will the embedding of the next frame look like." This avoids modeling irrelevant details (exact textures, lighting), focuses on semantic structure, and scales better computationally. Generative models like diffusion models are better for content creation but JEPA-style models may be better for planning — analogous to how humans don't mentally visualize exact pixel details when planning.

---

## Where to Next

- **Reinforcement Learning (the framework for world model planning)** → [04-specialized-domains/01-rl-fundamentals.md](../../04-specialized-domains/01-rl-fundamentals.md)
- **Video generation models (diffusion + DiT)** → [09-diffusion-models.md](09-diffusion-models.md)
- **Agentic AI systems (agents using world models for planning)** → [10-agentic-ai-systems.md](10-agentic-ai-systems.md)
- **Multimodal architectures** → [08-multimodal-architectures.md](08-multimodal-architectures.md)
