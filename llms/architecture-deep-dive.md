# Advanced LLM Architectures: MoE, RoPE, and GQA

## Executive Summary
While the standard Transformer is the base, state-of-the-art models (Mistral, Llama 3, GPT-4) use several critical optimizations to handle 100k+ context windows and trillions of parameters.

| Technique | Problem Solved | Mechanism |
|-----------|----------------|-----------|
| **MoE** | High compute cost of large models | Only activate 2/8 experts per token |
| **RoPE** | Poor long-context extrapolation | Rotary Positional Embeddings |
| **GQA** | KV-Cache memory bottleneck | Multi-query sharing of K/V heads |
| **Flash Attention** | Quadratic memory overhead | Tiling and IO-aware computation |

---

## 1. Mixture of Experts (MoE)
Instead of one massive dense layer, we use many small "Expert" layers. 
- **The Router**: For every token, a "Gating Network" decides which 2 experts (out of 8 or 16) are best suited to process it.
- **Benefit**: You get the knowledge of a 100B params model with the inference speed of a 12B params model (Active parameters).

---

## 2. Positional Encoding: RoPE
**Rotary Positional Embeddings (RoPE)** replaced fixed sinusoidal encodings.
- **How it works**: It rotates the embedding vectors in a specific way that represents their relative position.
- **Advantage**: It allows the model to extrapolate to much longer context windows (e.g., from 4k to 128k) more gracefully than absolute positional encodings.

---

## 3. Memory Optimization: GQA
**Grouped Query Attention (GQA)** is a middle ground between Multi-Head (MHA) and Multi-Query (MQA).
- **MHA**: Every Query has its own Key and Value head. (High memory).
- **MQA**: All Queries share a single Key and Value head. (Fast, but lower quality).
- **GQA**: Queries are grouped, and each group shares a Key/Value head. (Best balance).

---

## Interview Questions

**1. "Why is Flash Attention 10x faster if it still has $O(L^2)$ complexity?"**
> It addresses **Memory Wall** issues. Standard attention spends more time moving data between HBM (GPU VRAM) and SRAM (fast cache) than actually computing. Flash Attention "tiles" the operation to stay in SRAM as much as possible, reducing I/O.

**2. "What is 'Sliding Window Attention'?"**
> Used in models like Mistral. A neuron only attends to a fixed number of previous tokens ($W$). Since each layer attends to the layer below, the "effective" receptive field grows as you go deeper without the quadratic cost in early layers.

**3. "What are the challenges of training MoE models?"**
> 1. **Routing instability**: One expert might become a "hot spot" while others are ignored. 2. **Memory overhead**: You still need to fit the *entire* model (all experts) in VRAM even if you only use 2 at a time.

---

## Architecture Comparison
| Model | Attention | Positional Encoding | Layer Type |
|-------|-----------|---------------------|------------|
| **GPT-3** | MHA | Absolute | Dense |
| **Llama 3** | GQA | RoPE | Dense |
| **Mistral** | GQA | RoPE | MoE (8x7B) |
