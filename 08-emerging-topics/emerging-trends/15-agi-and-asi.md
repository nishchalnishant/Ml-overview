---
module: Emerging Topics
topic: Emerging Trends
subtopic: Agi And Asi
status: unread
tags: [emergingtopics, ml, emerging-trends-agi-and-asi]
---
# AGI, ASI, and the Path to General Intelligence

> *Snapshot: June 2026 — frontier topic, moves fast. Treat as a current-state map, not settled canon.*

What AGI/ASI actually means technically, what the leading theories of how to get there are, what evidence exists, and why this matters for understanding where AI is heading.

---

## 1. Definitions (Precise, Not Philosophical)

### Narrow AI
Current AI systems. Excels at specific, well-defined tasks within a distribution similar to its training data. GPT-4 is narrow AI — extraordinary at text tasks, zero transfer to novel physical problems outside training distribution.

### Artificial General Intelligence (AGI)
A system that can perform any cognitive task that a human can perform, at or above human level, with the ability to transfer learning to genuinely novel domains.

**The critical criterion**: not "does it score well on human benchmarks" but "can it learn and solve tasks it has never seen, the way humans can." An AGI handed a new video game with no rules would figure it out. Current LLMs trained on massive web data often "know" the rules of standard games from training — that's not generalization, that's recall.

**Operational definitions used in research**:
- OpenAI: "highly autonomous systems that outperform humans at most economically valuable work"
- DeepMind: five levels — (1) emerging, (2) competent, (3) expert, (4) virtuoso, (5) superhuman — across breadth (cognitive tasks) and performance (vs. human baseline). Current frontier models are Level 2-3 on narrow tasks, Level 1-2 on breadth.
- ARC-AGI (François Chollet): can the system solve novel visual reasoning puzzles that require generalization from minimal examples? Tests the thing LLMs specifically lack: sample-efficient generalization.

### Artificial Superintelligence (ASI)
A system that exceeds the best human cognitive performance across all cognitive domains simultaneously — including scientific discovery, strategic reasoning, social manipulation, and creative work.

Distinct from AGI: AGI can do what humans do. ASI can do things humans cannot.

---

## 2. Current State: Where Are We?

### What LLMs Can Do (That Looks Like AGI)
- Score in top 1% of humans on bar exams, medical licensing, PhD-level knowledge tests
- Write production-quality code, solve competition math at IMO level (o3)
- Reason through multi-step problems with backtracking (o1/o3/R1)
- Learn new tasks from a few examples (in-context learning)

### What LLMs Cannot Do (That Distinguishes Them from AGI)
- **Sample-efficient generalization**: LLMs need vast training data to "know" a domain; humans learn a new game or skill from 10 examples
- **Causal reasoning from scratch**: LLMs statistical-correlate; they don't build causal models of novel environments they've never seen
- **Embodied learning**: integrate perception, action, and learning in physical environments without prior training
- **True novelty**: when asked to solve a problem that is genuinely outside all training data, performance degrades sharply
- **Continuous learning**: LLMs have static weights after training; they cannot update their knowledge from new experiences without retraining

**ARC-AGI scores (2024)**:
| System | Score |
|---|---|
| Average human | 85% |
| GPT-4o (no thinking) | ~5% |
| o3 (low compute) | 75.7% |
| o3 (high compute) | 87.5% |

o3 at high compute exceeded average human performance — the first AI system to do so. But: o3 used ~$1000 of compute per ARC puzzle; humans solve them in seconds. The gap in *efficiency* of generalization is enormous.

---

## 3. Theories of How to Reach AGI

### 3.1 Scaling Hypothesis (Current Dominant Bet)
**Claim**: intelligence is an emergent property of scale — more parameters, more data, more compute → capabilities emerge that weren't explicitly trained. The scaling laws suggest no fundamental ceiling has been hit yet.

**Evidence for**: GPT-4 → Claude 3 → o3 shows consistent capability jumps. Many tasks (chain-of-thought reasoning, multi-hop logic) appeared suddenly at threshold model scales. o3's ARC-AGI performance suggests scaling test-time compute (not just parameters) is a new frontier.

**Evidence against**: scaling slows — each 10× compute increase produces smaller capability jumps. Certain capabilities (causal reasoning, sample efficiency) seem not to emerge from scale alone. ARC-AGI requires 1000× more compute in o3 than a human, suggesting the underlying mechanism is brute-force search rather than genuine generalization.

### 3.2 World Models Hypothesis

> DreamerV3's RSSM uses a GRU-based deterministic state combined with a stochastic latent — a form of learned state space model. For the broader family of SSMs (S4, Mamba, RWKV, Jamba) that achieve O(N) recurrent inference and serve as transformer alternatives for long-sequence world modeling, see [05-state-space-models.md](05-state-space-models.md).

**Claim**: language modeling alone is insufficient for AGI. A general intelligence must build an internal model of the world — understanding causality, physics, and how actions change states — not just statistical associations between words.

**Proposed path**: train agents in simulated or physical environments where they must predict and interact with the world, building causal representations. Language then becomes a communication layer on top of a rich world model.

**Evidence for**: current LLMs fail on tasks requiring causal counterfactuals ("if I remove the support, what happens?"). Robotics + LLM hybrids (RT-2, physical world interaction) suggest this direction. Yann LeCun's JEPA (Joint Embedding Predictive Architecture) proposes this explicitly.

**Proponents**: Yann LeCun (Meta), Gary Marcus, many robotics researchers.

#### World Model Architectures

**Dreamer / DreamerV3 (Hafner et al., 2020–2023)**: the most influential model-based RL agent using a learned world model.

Architecture: a **Recurrent State Space Model (RSSM)** with two state components:
- **Deterministic state** h_t: hidden state of a GRU, captures long-term dependencies
- **Stochastic state** z_t: sampled from a categorical or Gaussian distribution, captures uncertainty

```
# RSSM forward pass
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})           # deterministic
z_t ~ q_φ(z_t | h_t, x_t)                       # posterior (given observation)
z_t ~ p_φ(z_t | h_t)                            # prior (imagination, no observation)
x̂_t = p_θ(x_t | h_t, z_t)                      # reconstruction decoder
r̂_t = p_θ(r_t | h_t, z_t)                      # reward predictor
```

Training: ELBO on observations + rewards. The encoder compresses image observations to z_t; the world model predicts future z_t sequences for "imagination" rollouts used by the actor-critic.

**DreamerV3 (2023)**: unifies training across domains (Atari, continuous control, Minecraft) with fixed hyperparameters. Key improvements: symlog transformation for reward scaling, free bits for KL, categorical latents for gradient flow. First algorithm to collect diamonds in Minecraft from scratch.

**JEPA (LeCun, 2022–)**: Joint Embedding Predictive Architecture. Learns representations by predicting the representation of masked or future inputs, not reconstructing raw pixels:

```
s_x = Encoder(x)           # encode context
s_y = Encoder(y)           # encode target (y is masked/future version of x)
ŝ_y = Predictor(s_x, z)   # predict target representation with latent z
Loss: ||ŝ_y - sg(s_y)||²   # sg = stop gradient
```

Key difference from MAE/reconstruction-based methods: the predictor works in **abstract representation space**, not pixel space. This prevents the model from wasting capacity on unimportant pixel details. I-JEPA (image) and V-JEPA (video) show that models trained with JEPA learn richer semantic representations than masked autoencoders without generative pixel decoding.

**Genie 1 (Google DeepMind, 2024)**: a generative world model trained on unlabeled internet videos of 2D platformer games. Learns a latent action space (no action labels) and can simulate the environment in response to user-provided actions. Architecture: VQ-VAE tokenizer + spatiotemporal transformer + dynamics model. Key result: emergent controllability from video-only pretraining.

**Genie 2 (Google DeepMind, 2024)**: 3D world model trained on diverse video. Generates interactive 3D environments from a single image prompt, maintaining consistent physics, lighting, and object interactions for 10-20 seconds. Uses a foundation video model (similar to Sora) combined with action conditioning.

#### Are Current LLMs Implicit World Models?

**Argument for**: LLMs trained on internet text must encode causal structure to predict language accurately. Sentence "I dropped the glass and it shattered" requires knowing glass is brittle. Chain-of-thought reasoning shows LLMs can simulate multi-step causal chains.

**Argument against**: these are statistical patterns in language, not structural world models. LLMs confabulate physics (rotating objects in 3D), fail on novel causal scenarios with no textual analogue, and lack grounded simulation. Sora-like video models are *closer* to world models but still fail on physical consistency tests (objects appearing/disappearing, unrealistic fluid dynamics).

### 3.3 Neurosymbolic Integration

**Claim**: pure neural networks lack systematic compositionality — they can't reliably apply learned rules to new combinations. Combine neural networks (pattern recognition, language) with symbolic reasoning (formal logic, program synthesis) for robustness.

**Evidence**: AlphaGeometry 2 (Google DeepMind, 2024) solved 83% of 2000–2024 IMO geometry problems using a hybrid LLM + formal prover. Neither alone achieved >30%.

**Proponents**: research community at DeepMind, MIT, others.

#### Neurosymbolic Architectures in Practice

**LLM + Formal Theorem Provers**: the dominant current approach. An LLM generates proof steps (tactics) in a formal language (Lean 4, Isabelle, Coq), and an automated theorem prover (ATP) verifies each step and provides feedback.

```
Problem statement (natural language)
    ↓
LLM: generate formal problem encoding in Lean 4
    ↓
LLM: propose next proof tactic (sampled, temperature > 0)
    ↓
Lean 4 kernel: verify tactic is valid
    ↓ (if valid)
Update proof state
    ↓ (loop until QED or failure)
```

The verifier provides a **ground-truth signal** — verification is decidable for formal proofs. This eliminates hallucination in the reasoning chain (the LLM can hallucinate tactics, but the verifier immediately rejects invalid ones).

**AlphaGeometry 2 (DeepMind, 2024)**: fine-tuned Gemini Pro on 300M synthetic geometry proofs. At test time, uses beam search over LLM-proposed auxiliary point constructions, verified by a symbolic deduction engine (DD+AR+AG). Solved 83% of IMO geometry problems (2000–2024), gold medalist equivalent.

**DeepSeek-Prover / Lean-based systems**: multiple groups (DeepSeek, MIT, Google) have trained LLMs on Lean theorem proving datasets (Mathlib). Lean's type system enforces correctness — a "proof" that doesn't typecheck is rejected. Results: SOTA on MiniF2F benchmark at ~65% pass@1 (formally verified).

**Program Synthesis**: LLMs as symbolic programs. Code is by definition symbolic-compositional. DeepMind's AlphaCode 2 achieved competitive programmer level (top 15%) by generating code that passes unit tests — a verifiable symbolic artifact.

**Neural Concept Learners**: systems that learn to construct symbolic programs over a learned concept space. MIT's Neuro-Symbolic Concept Learner (NS-CL) uses a visual perception module to extract symbolic objects/attributes, then a program executor for compositional question answering. Achieves near-perfect accuracy on CLEVR with 10× fewer samples than pure neural approaches.

#### Why Neurosymbolic Matters for AGI

Systematic generalization: if a system learns "the red ball is to the left of the blue cube" it should immediately generalize to "the green pyramid is to the right of the yellow cylinder" — applying the same spatial relation rule. Pure neural models fail this (they interpolate rather than compose). Systems with explicit symbolic structure generalize systematically.

**Proponents**: research community at DeepMind, MIT, others.

### 3.4 Embodied AI and Robotics

**Claim**: cognition requires grounding in physical interaction. An agent that only processes language will never develop the causal models, spatial reasoning, and action understanding that biological intelligence is built on.

#### The Sim-to-Real Problem

Training robots in simulation is cheap and fast; physical data collection is slow and expensive. But policies trained in simulation often fail in the real world (sim-to-real gap) due to:
- **Dynamics mismatch**: simulated physics don't match reality (friction, contact forces, deformable objects)
- **Visual domain gap**: simulated rendering ≠ real-world lighting, textures
- **Morphology mismatch**: simulated robot kinematics may differ subtly from physical robot

**Domain randomization**: during simulation training, randomize physics parameters (mass, friction, damping), visual properties (texture, lighting), and sensor noise. The policy must learn a representation invariant to these variations, which generalizes to the real world. Used successfully for dexterous manipulation (OpenAI DACTYL: learned to solve Rubik's cube with a robot hand) and locomotion.

#### Foundation Models for Robotics

**RT-2 (Google, 2023)**: Vision-Language-Action model. Fine-tune a large VLM (PaLI-X or PaLM-E) to output robot actions directly as tokens, alongside language tokens. Training: web images/text (general VLM pretraining) + robot demonstrations (action prediction). Result: RT-2 can follow novel instructions that require semantic reasoning ("move the object that belongs in the recycling bin") without having seen recycling tasks in robot training data. The VLM knowledge transfers to robot actions.

```
Input: image of scene + language instruction
Model: VLM backbone (12B parameters)
Output: robot action tokens [move, 0.3m, forward; gripper, close]
```

**π0 (Physical Intelligence, 2024)**: pre-trained generalist robot policy using flow matching on diverse robot data. Architecture: small transformer trained with action chunks (predict next K=10 actions, not just one). Key innovation: **action chunking** reduces the temporal credit assignment problem and handles multi-modal action distributions (multiple valid ways to grasp an object). Demonstrates robot dexterity across folding laundry, making sandwiches, packing boxes.

**RoboCAT (Google DeepMind, 2023)**: self-improving agent that learns new tasks with minimal demonstrations. Architecture: tokenize robot observations (images + state) and actions into a unified token sequence, trained with next-token prediction (autoregressive). Few-shot adaptation: given 10-100 demonstrations of a new task, RoboCAT fine-tunes and achieves competence. Demonstrates positive transfer across robot embodiments (different arm morphologies).

**OpenVLA (2024)**: 7B parameter VLA (vision-language-action) model fine-tuned from Llama-2 on the Open X-Embodiment dataset (~1M robot demonstrations across 22 robot types). Open-weights, demonstrating that the open-source community can build competitive generalist robot policies.

#### Architecture Patterns for VLA Models

Vision-Language-Action (VLA) models unify perception, language understanding, and motor control:

```
Image(s) + Language instruction
    ↓
Vision encoder (ViT) → image tokens
Language encoder (LLM) → text tokens
Joint attention (full or cross)
    ↓
Action head: continuous action regression OR
             discrete action tokenization (256 bins per DOF)
```

**Discrete vs continuous action representations**:
- Discrete (RT-2): tokenize actions as integers; leverage language model's token prediction
- Continuous regression (π0, Diffusion Policy): predict continuous action vectors; flow matching or diffusion handles multi-modal action distributions better
- **Diffusion Policy (Chi et al., 2023)**: condition a diffusion model on current observation; outputs action trajectory. Naturally handles multi-modal distributions (robot can pick up from left OR right). State-of-the-art on many manipulation benchmarks.

#### Evaluation Benchmarks

| Benchmark | What it tests |
|---|---|
| LIBERO | Language-conditioned manipulation (4 suites: spatial, object, goal, long-horizon) |
| RLBench | 100 task variants, language instructions, diverse manipulation |
| Open X-Embodiment | Cross-embodiment transfer across 22 robot platforms |
| CALVIN | Long-horizon sequential manipulation (7-step task chains) |
| ManiSkill2 | GPU-accelerated simulation, physically plausible contact |

### 3.5 Recursive Self-Improvement

**Claim**: once an AI system is capable enough to improve its own training process, each generation is smarter than the last — bootstrapping to AGI/ASI.

**Current state**: AI assists in ML research (writing code, reviewing papers, suggesting hyperparameters) but doesn't autonomously improve its own architecture. This is an area of active investment.

**Risk**: if self-improvement is fast enough, the transition from AGI to ASI could be rapid and hard to control.

---

## 4. The Intelligence Explosion and ASI

If AGI is achieved, the path to ASI may be short. The argument:

```
Human-level AI → can do AI research at human speed
                → automates AI research → faster improvements
                → smarter AI → can do AI research better than humans
                → further automated improvements → ASI
```

This is the "intelligence explosion" thesis (Vinge, Yudkowsky, Bostrom). The key question is whether intelligence self-improvement is fast and unbounded, or slow and limited.

**Arguments for fast explosion**:
- AI runs 24/7 at GHz, humans work 8 hours at biological speeds
- AI can run thousands of parallel copies
- AI can be fine-tuned rapidly on targeted datasets

**Arguments for slow/bounded improvement**:
- Understanding intelligence ≠ ability to improve it
- Hardware and data availability are real constraints
- Many capabilities may not be improvable by self-generated data alone
- AGI → ASI may require qualitative, not quantitative, breakthroughs

---

## 5. Safety and Alignment Implications

The transition from narrow AI to AGI/ASI is also the point where alignment becomes existential rather than just inconvenient.

### The Alignment Problem at AGI Scale
Current LLMs are misaligned in minor ways (hallucination, sycophancy) that are annoying but not catastrophic. An AGI-level system pursuing misaligned goals with general problem-solving capability could cause serious harm that's hard to reverse.

**Instrumental convergence**: regardless of terminal goals, intelligent systems tend to pursue the same instrumental goals (self-preservation, resource acquisition, goal preservation) because these help achieve *any* goal. This means a misaligned AGI would resist being shut down.

### Current Approaches to Alignment at Scale

**Constitutional AI (Anthropic)**: train the model to critique and revise its own outputs against a set of principles. Scales better than human feedback for individual responses.

**Scalable Oversight**: use AI to assist humans in evaluating AI outputs. Debate — two AI systems argue for their answers, a human judges who argued more convincingly. Amplification — AI helps human understand complex outputs well enough to evaluate them.

**Interpretability**: understand what's happening inside the model. Mechanistic interpretability (circuit analysis) finds human-readable circuits for specific capabilities. Sparse autoencoders map model activations to interpretable features. Goal: identify misaligned goals before they manifest as harmful behavior.

**Governance**: international AI treaties, compute governance (track who has what compute), deployment restrictions on frontier models.

---

## 6. Near-Term Milestones Researchers Watch

| Milestone | Why it matters |
|---|---|
| ARC-AGI >90% at human-comparable compute | Suggests genuine generalization, not brute-force search |
| Autonomous AI researcher (publishes accepted paper) | AGI can contribute to its own development |
| Passing the "Employment Test" (sustain a job for 1 year) | Practical economic AGI |
| Winning Nobel Prize via AI reasoning | Novel scientific discovery |
| Solving P=NP or Riemann Hypothesis | Mathematical reasoning beyond human frontier |
| Continuous learning in novel environments | True world-model generalization |

Current frontier (2025): o3 passes ARC-AGI at high compute; AI co-authors research papers and generates novel protein structures (AlphaFold); AI autonomously fixes real GitHub issues (SWE-bench 70%+). Researchers at leading labs tentatively estimate AGI (by most definitions) within 5-10 years, with wide uncertainty.

---

## 7. Practical Implications for ML Engineers

Whether or not AGI arrives in 5 years, the trajectory has immediate practical implications:

**Test-time compute scaling** changes how you deploy models — expensive inference for hard problems is now viable. Budget adaptive compute allocation by problem difficulty.

**Agentic systems** are the path to "AGI for a task domain" today — an agent with code execution, web search, and memory can solve tasks that require general intelligence within a bounded domain.

**Alignment and safety engineering** is increasingly a job function — red-teaming, adversarial evaluation, output filtering, and monitoring are now standard infrastructure.

**Model capability trajectories** matter for roadmap planning — capabilities that seem years away can arrive suddenly (in-context learning, chain-of-thought, GPT-4-level code all seemed distant until they appeared).

---

## Canonical Interview Q&As

**Q: What is the difference between AGI and ASI, and what technical capabilities separate current LLMs from AGI?**
A: AGI is a system that can perform any cognitive task a human can, with the ability to generalize to genuinely novel domains — not just high performance on benchmarks. ASI exceeds human performance across all cognitive domains simultaneously. Current LLMs are narrow AI: they excel on tasks similar to their training distribution but fail on three key criteria that define AGI: (1) Sample-efficient generalization — humans learn a new game from 10 examples; LLMs need millions of training examples; ARC-AGI specifically tests this. (2) Causal reasoning from first principles — LLMs learn statistical correlations; they struggle with causal counterfactuals in novel environments they've never encountered in training. (3) Continuous learning — LLMs have static weights; they can't update from new experiences without retraining. o3 at high compute passed ARC-AGI (87.5%) but required ~1000× more compute than a human — it's finding solutions by extensive search, not generalizing efficiently. That gap in efficiency is the key remaining barrier.

**Q: What is the alignment problem and why does it become more serious as systems approach AGI capability?**
A: The alignment problem: how do you ensure that a powerful AI system pursues goals that are beneficial to humans, especially when it has enough capability to subvert any external constraints? At narrow AI scale, misalignment causes inconveniences (hallucination, wrong answers). At AGI scale, an agent that's misaligned and can take autonomous actions in the world could cause serious, hard-to-reverse harm — because it would have general problem-solving capability to pursue its misaligned goals effectively. The core challenge is instrumental convergence: regardless of what terminal goal a system has, it will tend to pursue instrumental goals like self-preservation (don't get shut down, it would prevent the goal), resource acquisition (more resources = better goal achievement), and goal preservation (resisting modification). A misaligned AGI would naturally resist correction. Current mitigation approaches include Constitutional AI (train models to self-critique against principles), scalable oversight (AI helps humans evaluate AI outputs), interpretability (understand what goals the model has internally), and governance (deployment restrictions on the most capable systems).

**Q: Is scaling alone sufficient to reach AGI, or are qualitative architectural changes needed?**
A: Genuinely unsettled question in the field, but the evidence is mixed. Arguments that scaling is sufficient: emergent capabilities appear at threshold scales without architectural changes — chain-of-thought reasoning, multi-hop logic, and ARC-AGI-level performance all emerged from scaling variants of the transformer. The test-time compute scaling shown in o3 opens a new axis beyond parameter count. Arguments that qualitative changes are needed: current architectures lack persistent memory (weights don't update from experience), causal world models (no explicit representation of how actions cause state changes), and sample-efficient generalization. These seem like structural absences, not just insufficient scale. The most credible middle position: scaling will push current architectures very far (possibly to "weak AGI" — human-level on most cognitive benchmarks), but true AGI with human-like sample efficiency and robust generalization may require architectures that learn causal world models, not just statistical language models.

## Flashcards

**OpenAI?** #flashcard
"highly autonomous systems that outperform humans at most economically valuable work"

**DeepMind: five levels?** #flashcard
(1) emerging, (2) competent, (3) expert, (4) virtuoso, (5) superhuman — across breadth (cognitive tasks) and performance (vs. human baseline). Current frontier models are Level 2-3 on narrow tasks, Level 1-2 on breadth.

**ARC-AGI (François Chollet)?** #flashcard
can the system solve novel visual reasoning puzzles that require generalization from minimal examples? Tests the thing LLMs specifically lack: sample-efficient generalization.

**Score in top 1% of humans on bar exams, medical licensing, PhD-level knowledge tests?** #flashcard
Score in top 1% of humans on bar exams, medical licensing, PhD-level knowledge tests

**Write production-quality code, solve competition math at IMO level (o3)?** #flashcard
Write production-quality code, solve competition math at IMO level (o3)

**Reason through multi-step problems with backtracking (o1/o3/R1)?** #flashcard
Reason through multi-step problems with backtracking (o1/o3/R1)

**Learn new tasks from a few examples (in-context learning)?** #flashcard
Learn new tasks from a few examples (in-context learning)

**Sample-efficient generalization?** #flashcard
LLMs need vast training data to "know" a domain; humans learn a new game or skill from 10 examples

**Causal reasoning from scratch?** #flashcard
LLMs statistical-correlate; they don't build causal models of novel environments they've never seen

**Embodied learning?** #flashcard
integrate perception, action, and learning in physical environments without prior training

**True novelty?** #flashcard
when asked to solve a problem that is genuinely outside all training data, performance degrades sharply

**Continuous learning?** #flashcard
LLMs have static weights after training; they cannot update their knowledge from new experiences without retraining

**AI runs 24/7 at GHz, humans work 8 hours at biological speeds?** #flashcard
AI runs 24/7 at GHz, humans work 8 hours at biological speeds

**AI can run thousands of parallel copies?** #flashcard
AI can run thousands of parallel copies

**AI can be fine-tuned rapidly on targeted datasets?** #flashcard
AI can be fine-tuned rapidly on targeted datasets

**Understanding intelligence ≠ ability to improve it?** #flashcard
Understanding intelligence ≠ ability to improve it

**Hardware and data availability are real constraints?** #flashcard
Hardware and data availability are real constraints

**Many capabilities may not be improvable by self-generated data alone?** #flashcard
Many capabilities may not be improvable by self-generated data alone

**AGI → ASI may require qualitative, not quantitative, breakthroughs?** #flashcard
AGI → ASI may require qualitative, not quantitative, breakthroughs
