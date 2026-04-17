# Efficient LLM Deployment & Optimization

These notes follow the **Gold Standard** for interview preparation: providing direct answers, intuition, and the "numbers" that engineers need to know for production serving.

---

# 1. Quantization & Model Compression

## Q1: What is Quantization in LLMs? (FP16 -> INT8/INT4)

### 🔹 Direct Answer
**Quantization** is the process of reducing the precision of model weights and activations (e.g., from 16-bit Floating Point to 4-bit Integers). The goal is to shrink the model size and increase inference speed while maintaining as much performance as possible.

### 🔹 Intuition
Imagine writing down a number like $\pi = 3.14159265...$. 
- **High Precision:** You write it out to 16 decimal places. (Takes lots of paper/RAM).
- **Quantization:** You just write $3.14$. (Takes way less space). 
The model still knows it's "three-ish," but you saved 80% of the storage.

### 🔹 Common Formats
- **FP16 / BF16:** The training standard. 2 bytes per parameter.
- **GPTQ / AWQ (4-bit):** The industry standard for single-GPU serving. ~0.5 bytes to 0.7 bytes per parameter.
- **GGUF (vLLM / llama.cpp):** Optimized for CPU+GPU mixed inference.

---

## Q2: How do you calculate how much VRAM is needed to serve a model?

### 🔹 High-Yield Formula
For a model with $P$ parameters:
$$VRAM \approx P \times \text{bytes per parameter} \times 1.25 \text{ (overhead for KV Cache)}$$

### 🔹 Examples
- **Llama 3 (8B) in FP16:** $8 \times 2 = 16 \text{ GB}$ (Needs a 24GB GPU).
- **Llama 3 (8B) in 4-bit:** $8 \times 0.5 = 4 \text{ GB}$ (Can run on a phone or small laptop).
- **Llama 3 (70B) in 4-bit:** $70 \times 0.5 = 35 \text{ GB}$ (Needs 2x RTX 3090/4090 or 1x A100).

---

# 2. Serving Optimizations

## Q3: What is KV Caching and why is it essential?

### 🔹 Direct Answer
**KV Caching** stores the Key and Value tensors of past tokens in GPU memory so the model doesn't have to recompute them for every new token during autoregressive generation. It transforms the generation complexity from $O(N^2)$ to $O(N)$.

### 🔹 Intuition
If you are writing a story, and the LLM just generated "The cat sat on the...", to guess the next word "mat", it shouldn't have to re-read "The", "cat", "sat" again from scratch. It should just "remember" them and focus on the new input.

---

## Q4: Flash Attention & Paged Attention (vLLM)

### 🔹 Direct Answer
- **Flash Attention:** An IO-aware algorithm that speeds up the attention mechanism by reducing memory reads/writes between GPU HBM and SRAM.
- **Paged Attention:** A technique (used in vLLM) that manages KV Cache memory like an Operating System manages virtual memory (using pages). It eliminates memory fragmentation and allows for 10x higher serving throughput.

---

## Q5: Speculative Decoding

### 🔹 Direct Answer
**Speculative Decoding** uses a tiny, fast model (Draft Model) to guess multiple future tokens in parallel, and then uses the big model (Target Model) to verify them all at once. This significantly speeds up generation when the target model is very large.

### 🔹 Intuition
Like a fast-typing assistant who drafts 5 words ahead. You (the expert) just nod or shake your head. If the assistant guessed right, you saved the time of thinking and typing those 5 words yourself.
