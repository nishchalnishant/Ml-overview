# Large Language Model (Core Foundations)

This hub provides the foundational "first-principles" understanding of Large Language Models. For advanced topics like Agentic RAG, 3D Parallelism, and Alignment Research, please refer to the specialized [LLM Interview Notes](../llm-interview-notes/README.md) hub.

---

# 1. 🔹 The LLM Paradigm

## Q1: What is a Large Language Model (LLM)?

### 🔹 Direct Answer
An **LLM** is a massive neural network (typically a Transformer) trained on petabytes of text to predict the next token in a sequence. It learns a compressed representation of human knowledge, allowing it to perform tasks like reasoning, coding, and summarization without task-specific training.

### 🔹 Intuition
Imagine a student who has read every book, article, and forum post on the internet. While they haven't "lived" in the real world, they have become an expert at "guessing the next word" so well that they appear to understand logic, culture, and science.

---

# 2. 🔹 Pre-training vs. Fine-tuning

## Q2: Explain the lifecycle of an LLM.

### 🔹 Direct Answer
1. **Pre-training:** The model learns "how the world works" and general language grammar by predicting the next token on a massive, unlabeled dataset. (Builds the raw brain).
2. **Supervised Fine-Tuning (SFT):** The model is trained on specific "Question-Answer" pairs to learn how to follow instructions (Instruction Tuning). (Teaches the brain how to behave).
3. **Alignment (RLHF):** The model is tuned using human feedback to ensure its outputs are helpful, honest, and harmless. (Polishes the behavior).

---

# 3. 🔹 Tokenization

## Q3: What is BPE (Byte Pair Encoding)?

### 🔹 Direct Answer
**BPE** is the standard sub-word tokenization algorithm for LLMs. It repeatedly merges the most frequent pairs of characters/bytes into a single token until a fixed vocabulary size is reached.

### 🔹 Why it matters
It allows the model to handle "Out-of-Vocabulary" words by breaking them into known sub-units (e.g., "unconscious" -> "un" + "conscious"). It is a bridge between character-level (too long) and word-level (too many words) tokenization.

---

# 4. 🔹 Decoding Parameters

## Q4: What is "Temperature" in LLM Sampling?

### 🔹 Direct Answer
**Temperature ($T$)** is a hyperparameter that controls the randomness of the next-token selection. It rescales the logits before applying the Softmax function.
- **Low $T$ (e.g., 0.1):** The model is confident and deterministic.
- **High $T$ (e.g., 1.0):** The model is creative and diverse.

### 🔹 Formula
$P_i = \frac{e^{z_i / T}}{\sum e^{z_j / T}}$
As $T \rightarrow 0$, the highest logit becomes 1 (greedy). As $T \rightarrow \infty$, the distribution becomes uniform.

---

# 5. 🔹 Evaluation & Quality

## Q5: How do you measure LLM performance?

### 🔹 Direct Answer
- **Perplexity:** Measures how "surprised" the model is by the data. Lower is better.
- **Benchmark Suites:** MMLU (General Knowledge), GSM8K (Math), HumanEval (Code).
- **LLM-as-a-Judge:** Using a stronger model (e.g., GPT-4o) to grade the responses of a smaller model.

---

## 🔹 Difficulty Tag: 🟡 Medium
