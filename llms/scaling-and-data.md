# Scaling Laws: The Economics of LLMs

## Executive Summary
The most influential paper in modern AI (Kaplan et al., 2020 / Chinchilla, 2022) proved that model performance is a predictable function of three variables: **Compute ($C$)**, **Data ($D$)**, and **Parameters ($N$)**.

| Concept | Key Finding | Real-world Impact |
|---------|-------------|-------------------|
| **Kaplan Laws** | Over-emphasized parameters | Led to massive, under-trained models (GPT-3) |
| **Chinchilla Laws** | Data is just as important as size | Models should be "Compute Optimal" |
| **Optimal Ratio** | ~20 tokens per parameter | Llama 3 (70B) was trained on 15T tokens |

---

## 1. The Chinchilla Optimality
DeepMind's research showed that for every doubling of compute budget, we should increase model size and training tokens in equal proportions.
- **Compute Budget $C$**: Measured in PFLOPS-days.
- **Equation**: $C \approx 6ND$ (where $N$ is params and $D$ is tokens).
- **The Finding**: Most models (like GPT-3) were too big for the amount of data they were trained on. To be optimal, a 175B model should have been trained on trillions of tokens.

---

## 2. Data Mixture & Quality
"More data" is no longer enough; it must be "high-quality" data.

### The Lifecycle of Data
1. **Deduplication**: Removing identical or near-identical web pages to prevent "bucketing" of gradients.
2. **Quality Filtering**: Uses a small classifier (e.g., fastText) to detect "Wikipedia-quality" text from "spam-quality" text.
3. **Synthetic Data**: Building high-quality reasoning/math data using stronger models (e.g., GPT-4 feeding Llama 3).

---

## Interview Questions

**1. "If you have a fixed compute budget, would you train a larger model for fewer steps or a smaller model for more steps?"**
> According to Chinchilla laws, there is a specific "optimal" point. However, in practice, we often train **smaller models for longer** (over-training) because it makes inference cheaper for millions of users.

**2. "What are the limits of scaling? Can we just keep adding data forever?"**
> We are approaching the "Token Crisis" where we have exhausted most of the high-quality human-generated text on the internet. Future scaling depends on **Synthetic Data**, **Self-Play**, and **Multi-Modal** (Video/Audio) data.

**3. "How does the 'Inverse Scaling' law work?"**
> While most tasks improve with scale, some exhibit "inverse scaling" where larger models perform *worse* initially because they over-rely on common patterns. Examples include certain types of logical traps or tasks where the model must ignore a frequent but irrelevant association.

---

## Summary Table: Model Sizes
| Model | Params | Training Tokens | Status |
|-------|--------|-----------------|--------|
| **GPT-3** | 175B | 300B | Under-trained |
| **Llama 2** | 70B | 2T | Well-trained |
| **Llama 3** | 70B | 15T | Massive Over-training |
