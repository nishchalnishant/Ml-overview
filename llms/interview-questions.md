# Top LLM Interview Questions (Technical & Strategy)

## 🏗️ Architecture & Theory

**1. "Explain the 'KV Cache' and why it's critical for real-time LLM applications."**
> In autoregressive models, we predict tokens one by one. Each new prediction depends on the hidden states of all previous tokens. Instead of re-calculating the Key ($K$) and Value ($V$) dot products for the entire history every time, we store (cache) them. This reduces the time complexity for a single token generation from $O(L)$ to $O(1)$ relative to history length, significantly reducing latency.

**2. "Why do we divide by $\sqrt{d_k}$ in the scaled dot-product attention formula?"**
> As the dimension $d_k$ grows, the magnitude of the dot product $QK^T$ increases. This results in very large values being passed to the Softmax function, pushing it into regions where gradients are nearly zero (vanishing gradients). Scaling by $\sqrt{d_k}$ keeps the variance of the dot product around 1, ensuring stable gradients during training.

**3. "Compare LoRA and Prefix-Tuning as PEFT methods."**
> **LoRA** (Low-Rank Adaptation) adds trainable low-rank matrices to the existing weights (usually $Q$ and $V$). It doesn't increase the sequence length. **Prefix-Tuning** prepends trainable vectors (prefixes) to the keys and values of every layer. This *effectively* reduces the context window available for the actual user prompt, whereas LoRA is more architectural and "invisible" to the context window.

---

## 🚀 Training & Production

**4. "How do you detect and mitigate 'catastrophic forgetting' when fine-tuning an LLM?"**
> **Detect**: Evaluate the model on a "general" benchmark (like MMLU) after fine-tuning on a specific task; if the general score drops significantly, forgetting has occurred. 
> **Mitigate**: 1. Use a very low learning rate. 2. Use **PEFT (LoRA)** which freezes the base weights. 3. Use **Experience Replay** (mixing some general pre-training data into the fine-tuning set).

**5. "What is 'RLHF' and why can't we just use Supervised Fine-Tuning (SFT) for alignment?"**
> SFT helps the model learn the *structure* of an answer, but it's limited by the "average" quality of the human labelers. **RLHF** allows the model to explore multiple responses and learns from a Reward Model that captures complex human preferences (e.g., tone, safety, nuance) that are hard to explicitly write as "perfect" labels in SFT.

**6. "Describe the 'Chinchilla' scaling laws in your own words."**
> It suggests that for a given amount of compute, most early LLMs were too large and under-trained. The optimal approach is to scale the number of parameters and the number of training tokens equally. Specifically, for every parameter, you should train on approximately 20 tokens.

---

## 🎨 Creative & Problem Solving

**7. "An LLM is hallucinating a legal fact. How do you solve this without training?"**
> Use **RAG** (Retrieval-Augmented Generation). 1. Retrieve the actual law text from a verified database. 2. Inject that text into the prompt context. 3. Instruct the model to "Answer only using the provided text." This anchors the model in reality and provides a "paper trail" via citations.

**8. "Your model is too large to fit on a single 80GB A100 GPU. What are your options?"**
> 1. **Quantization**: Use 4-bit or 8-bit precision (reduces size by 2-4x). 2. **Model Parallelism**: Split the layers or heads across multiple GPUs. 3. **PEFT (LoRA)**: Train only a tiny fraction of weights whilekeeping the rest on CPU/Disk (if training). 4. **Offloading**: Move part of the model to System RAM (slower).
