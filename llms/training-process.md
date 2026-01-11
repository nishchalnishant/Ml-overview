# LLM Training Lifecycle: From Raw Text to Aligned Assistant

## Executive Summary
The training of a modern LLM (e.g., Llama 3, GPT-4) happens in three distinct stages, each with a different objective, data requirement, and mathematical optimization.

| Stage | Objective | Data Type | Algorithm |
|-------|-----------|-----------|-----------|
| **1. Pre-training** | Learn world knowledge & grammar | Trillions of raw tokens (Web, Books) | Next-Token Prediction |
| **2. SFT** | Learn to follow instructions | Thousands of (Prompt, Answer) pairs | Supervised Learning |
| **3. Alignment** | Safety, tone, and human preference | Human rankings of outputs | RLHF / DPO |

---

## 1. Pre-training: The Heavy Lifting
This is where 99% of the compute is spent. The model learns to predict the next token based on a massive corpus.
- **Data Mixture**: High-quality web text (Common Crawl), code (GitHub), math (ArXiv), and reasoning data.
- **Objective Function**: Cross-Entropy Loss over the entire sequence.
- **The Result**: A **Base Model** (Foundation Model). It is a "knowledgeable parrot" that can complete sentences but doesn't know it's an assistant yet.

---

## 2. Supervised Fine-Tuning (SFT)
Converting a Base Model into an **Instruct Model**.
- **Data**: Humans manually write high-quality responses to prompts (e.g., "Summarize this...", "Write code for...").
- **Objective**: The model is trained to maximize the likelihood of the human-written answer.
- **The Shift**: The model learns the *format* of an assistant and how to generalize instruction following.

---

## 3. Alignment: RLHF & DPO
Ensuring the model is helpful, honest, and harmless ($HHH$).

### RLHF (Reinforcement Learning from Human Feedback)
1. **Sampling**: Model generates multiple answers for one prompt.
2. **Ranking**: Humans rank these answers (A > B > C).
3. **Reward Modeling**: A separate "Reward Model" is trained to predict these rankings.
4. **PPO (Proximal Policy Optimization)**: The LLM is updated to maximize its score from the Reward Model while staying close to the original model (KL-Divergence penalty).

### DPO (Direct Preference Optimization)
A more stable and efficient alternative to RLHF. It directly optimizes the model on preference pairs $(x, y_w, y_l)$ where $y_w$ is the winning answer and $y_l$ is the losing one. It eliminates the need for a separate reward model.

---

## Interview Questions

**1. "What happens if you skip Pre-training and only do SFT?"**
> The model will lack "World Knowledge". It might learn the *style* of an assistant but will hallucinate facts constantly because it hasn't seen the trillions of tokens required to learn underlying patterns and data relationships.

**2. "Why is the KL-Divergence term crucial in RLHF?"**
> It prevents **Reward Hacking**. Without it, the model might find a "cheat code" to get high scores from the reward model (e.g., repeating a specific keyword) while losing its basic linguistic coherence. The KL term keeps the "new" model similar to the "old" stable model.

**3. "What is a 'Hallucination' at a technical level?"**
> It is a high-probability prediction that does not align with factual reality. Technically, the model is simply maximizing the likelihood of the next token based on its training distribution; if the training data was conflicting or insufficient, the prediction path leads to plausible-sounding but false information.

---

## Logic Flow
```python
# The standard "Alignment" loss intuition
loss = -log(prob_winning_response) + log(prob_losing_response)
```
