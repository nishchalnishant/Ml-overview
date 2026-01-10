# LLM Evaluation: Beyond the Vibe Check

##  Executive Summary
Evaluating LLMs is difficult because language is subjective. We use a battery of automated benchmarks, "Model-as-a-Judge," and human evaluations.

| Benchmark | Focus | Why it matters |
|-----------|-------|----------------|
| **MMLU** | General Knowledge | Covers 57 subjects (STEM, Humanities) |
| **GSM8K** | Math/Reasoning | Multi-step word problems |
| **HumanEval** | Coding | Python coding tasks |
| **LMSYS Chatbot Arena** | Human Preference | The ultimate "Elo" rating (Crowdsourced) |

---

##  1. Common Benchmarks

### MMLU (Massive Multitask Language Understanding)
The "SAT" for LLMs. It contains 16k multiple-choice questions across 57 branches of knowledge.
- **State of the Art**: GPT-4 and Llama 3 400B score around 85-88%.

### GSM8K (Grade School Math 8K)
Tests multi-step reasoning. To solve these, models must maintain a consistent logic path.
- **Success Signal**: High GSM8K scores usually correlate with strong "Chain of Thought" capabilities.

---

##  2. Modern Evaluation Techniques

### Model-as-a-Judge
Since human evaluation is slow and expensive, we use a stronger model (e.g., GPT-4o) to grade the outputs of a smaller model.
- **Criteria**: Helpful, Honest, Harmless.
- **Bias Alert**: Stronger models often have a "self-preference bias" where they prefer answers that sound like their own style.

### Elo Rating (LMSYS)
An anonymous side-by-side comparison where humans pick the better answer. 
- **The "Gold Standard"**: Because it captures nuance, formatting, and "vibes" that static benchmarks miss.

---

##  Interview Questions

**1. "What is 'Data Contamination' in LLM evaluation?"**
> It happens when the test questions (e.g., from GSM8K) are present in the massive pre-training corpus of the model. This leads to the model "memorizing" the answers rather than "reasoning" through them, resulting in inflated scores.

**2. "How would you evaluate a model designed for a very specific domain, like Legal or Medical?"**
> 1. Use **Domain-specific benchmarks** (e.g., PubMedQA, Bar Exam datasets). 2. Use **Expert-in-the-loop** evaluation (Actual doctors/lawyers). 3. Measure **Factual Accuracy** and **Citability** over just fluency.

**3. "Why is Accuracy sometimes a poor metric for LLMs?"**
> For generative tasks, there are many "correct" ways to say the same thing. Metrics like **ROUGE** or **BLEU** measure n-gram overlap but miss meaning. Modern evaluation uses **Semantic Similarity** or **LLM-Judges** rather than exact matches.
