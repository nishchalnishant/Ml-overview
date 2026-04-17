# Advanced LLM Alignment & Reasoning

These notes follow the **Gold Standard** for interview preparation: providing direct answers, intuition, and the "why" behind modern LLM scaling and behavior.

---

# 1. Model Alignment (RLHF & DPO)

## Q1: What is RLHF (Reinforcement Learning from Human Feedback)?

### 🔹 Direct Answer
**RLHF** is the process used to align LLM behavior with human preferences (Helpfulness, Honesty, Harmlessness). It involves three steps:
1. **SFT (Supervised Fine-Tuning):** Training the base model on high-quality demonstration data.
2. **Reward Model Training:** Training a separate model to predict which of two responses a human would prefer.
3. **PPO (Proximal Policy Optimization):** Using the reward model to fine-tune the SFT model via reinforcement learning.

### 🔹 Intuition
Imagine a student who can write essays (Base Model).
- **SFT:** You give them 1,000 "A+" essays to copy (Demonstration).
- **Reward Model:** You show them two essays and say "This one is better because it's safer." They learn to assign "scores" to essays.
- **RL:** They write a new essay, you (the Reward Model) give it a score, and they adjust their style to get higher scores next time.

---

## Q2: RLHF vs. DPO (Direct Preference Optimization)

### 🔹 Direct Answer
**DPO** is a stable, efficient alternative to RLHF that eliminates the need for a separate reward model and complex sampling during training. It treats the alignment problem as a simple binary classification on preference pairs (Chosen vs. Rejected), directly optimizing the policy to favor the chosen response.

### 🔹 Comparison Table

| Feature | RLHF (PPO) | DPO |
| :--- | :--- | :--- |
| **Complexity** | High (Requires 3 models: Policy, Reward, Reference). | Low (Requires only Policy and Reference). |
| **Stability** | Finicky; sensitive to hyperparameters. | High; stable objective. |
| **Compute** | High (Requires sampling during training). | Low (Offline optimization). |

---

# 2. Reasoning & Planning

## Q3: What is Chain-of-Thought (CoT) Prompting?

### 🔹 Direct Answer
**Chain-of-Thought** is a prompting technique that encourages LLMs to break down complex problems into intermediate reasoning steps. It drastically improves performance on arithmetic, symbolic reasoning, and logic tasks where the answer isn't a direct "pattern match."

### 🔹 Intuition
If I ask you "What is 123 x 45?", you can't just blurt out the answer. You need a scratchpad to do the math step-by-step. CoT is the "scratchpad" for the LLM's brain.

---

## Q4: Explain the ReAct Framework (Reasoning + Acting).

### 🔹 Direct Answer
**ReAct** is a framework that combines **Reasoning** (CoT) with **Action** (Tool use). The model generates a "Thought" (planning step), then performs an "Action" (e.g., searching the web or using a calculator), observes the "Observation", and repeats until the task is complete.

### 🔹 Practical Snippet (ReAct Loop)
```text
Thought: I need to find the current price of Bitcoin.
Action: GoogleSearch[current price of Bitcoin]
Observation: Bitcoin is currently $65,000.
Thought: Now I have the price. I can answer the user.
Final Response: The price is $65,000.
```

---

## Q5: What are AI Agents and Agentic Workflows?

### 🔹 Direct Answer
An **AI Agent** is an LLM powered system that is given a goal (e.g., "Research and write a newsletter") and has the autonomy to plan, use tools, and self-reflect to accomplish it. **Agentic Workflows** (popularized by Andrew Ng) emphasize iterative refinement over a single zero-shot prompt.

### 🔹 Key Patterns
1. **Reflection:** The agent writes code, reviews it for errors, and fixes it before showing the user.
2. **Tool Use:** Calling APIs to interact with the world.
3. **Planning:** Breaking a large goal into a sub-task list.
4. **Multi-agent Collaboration:** Different agents (e.g., "Writer" and "Editor") reviewing each other's work.
