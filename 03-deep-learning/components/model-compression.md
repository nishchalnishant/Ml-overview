---
module: Deep Learning
topic: Components
subtopic: Model Compression
status: unread
tags: [deeplearning, ml, components-model-compression]
---
# Model Compression

---

## The Deployment Gap

**The problem**: a model is trained on a cluster of GPUs with hundreds of gigabytes of memory. It needs to run on a smartphone, inside a browser, or with a 20ms latency budget. A 70B parameter model at float32 is 280GB — it cannot even load on a single server GPU, let alone a phone.

**The core insight**: trained models are massively over-specified for inference. During training, high numerical precision prevents gradient noise from accumulating over millions of updates. During inference, you forward-pass once — small precision errors have minimal compound effect. Similarly, not all weights matter equally; many can be removed or replaced with cheaper approximations without significantly changing outputs.

Model compression is the engineering problem of finding the minimum representation that preserves task-critical behavior.

---

## Quantization

### Post-Training Quantization (PTQ)

**The problem**: neural network weights are stored as float32 (4 bytes each). A 70B parameter model needs 280GB. Most modern inference hardware runs faster with integers. You cannot retrain the model.

**The core insight**: most weights cluster in a small numerical range. Map that range to integers, losing little information. You already have the trained model — just change the number format.

**The mechanics**: for INT8, map the weight range $[w_\text{min}, w_\text{max}]$ to integers $[-128, 127]$:

$$w_\text{int} = \text{round}\left(\frac{w - \text{zero\_point}}{\text{scale}}\right)$$

where scale and zero-point are computed from activation statistics on a calibration dataset. A calibration dataset of 128–1024 representative inputs is passed through the model to observe actual activation ranges before quantization is applied.

- **FP16**: 2× memory reduction, nearly zero accuracy loss. Start here.
- **INT8**: 4× reduction, $<1\%$ accuracy drop on most tasks with good calibration. Significant speed gains on CPUs and accelerators.
- **INT4**: 8× reduction. More aggressive — accuracy drops more, requires careful calibration.

**What breaks**: bad calibration data means the quantization ranges are wrong. If calibration uses random noise instead of representative inputs, the scale/zero-point values are wrong, and quantized activations clip or overflow. Always use representative real data for calibration.

---

### Quantization-Aware Training (QAT)

**The problem**: PTQ loses accuracy at aggressive quantization (INT4, binary). The model was trained at float32 and never learned to compensate for quantization error.

**The core insight**: simulate quantization during training. Insert fake quantization nodes that round weights and activations to low-precision values in the forward pass, but allow gradients to flow at full precision in the backward pass (straight-through estimator). The model learns to tolerate quantization noise.

**What breaks**: QAT requires re-training or fine-tuning, which costs compute. For INT8, PTQ is usually good enough and much cheaper. Use QAT when PTQ accuracy is unacceptable and you have the training budget.

---

### GPTQ and AWQ for LLMs

**The problem**: standard INT8/INT4 PTQ assumes weights are roughly uniformly distributed. LLM weights have outlier values in certain channels that are critical to model behavior — naively quantizing these outliers causes severe accuracy degradation.

**GPTQ**: uses second-order information (Hessian approximation) to decide which weights can be quantized aggressively. Quantizes layer by layer; after quantizing a weight, adjusts remaining weights to compensate for the introduced error. Result: LLMs quantized to INT4 with surprisingly small accuracy loss.

**What breaks**: GPTQ is expensive to apply (requires Hessian computation). The layer-by-layer sequential process means errors can compound. Still the standard method for getting 70B models onto single consumer GPUs (~35GB at 4-bit).

**AWQ (Activation-aware Weight Quantization)**: observes that not all weights are equally important. Weights that multiply large-magnitude activations matter more — their quantization error gets amplified. AWQ scales these channels before quantization to protect them.

**What breaks**: AWQ requires access to calibration activations. Faster to apply than GPTQ, and often more robust across architectures, but both are approximations — neither fully recovers the original model's quality.

---

## Knowledge Distillation

### The Teacher-Student Paradigm

**The problem**: you have a large model that is too slow or too large for deployment, but a smaller model trained from scratch on the same task performs noticeably worse.

**The core insight**: the large model's output distributions contain more information than the hard labels. When a model classifies an image of a cat, it assigns 0.78 to cat, 0.15 to dog, 0.07 to fox — encoding which classes look similar. A smaller model trained only on the hard label "cat" never learns these relationships. Train the smaller model to mimic the large model's full distribution.

**The mechanics**: the distillation loss combines two terms:

$$L = \alpha \cdot L_\text{CE}(\text{student}, \text{hard\_labels}) + (1-\alpha) \cdot L_\text{KL}(\text{student}, \text{teacher\_soft\_labels})$$

Temperature $T > 1$ is applied to both student and teacher logits before softmax to soften the distributions, increasing information in the tails:

$$\text{soft}(z_i, T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

**What breaks**: if the student model is too small, it cannot absorb the teacher's knowledge regardless of training regime. The student architecture must have sufficient capacity for the task; distillation improves sample efficiency, it cannot compress knowledge into an incapable model.

---

### Intermediate Layer Distillation

**The problem**: matching only the final output ignores rich internal representations learned by the teacher. The student might match the teacher's final predictions but do so via a completely different internal computation.

**The core insight**: force the student to also mimic the teacher's intermediate representations — attention maps, hidden states, feature maps. This gives the student a structural blueprint, not just the final answer.

**FitNets**: match intermediate feature maps directly:

$$L_\text{hint} = \|f_\text{student}(x) - W \cdot f_\text{teacher}(x)\|^2$$

A learned projection $W$ bridges the dimension mismatch between teacher and student layers.

**Attention transfer**: match layer-wise attention weight distributions. Particularly effective for Transformers — the student learns not just what to predict, but which tokens to attend to.

**What breaks**: selecting which teacher layers to align with which student layers requires architectural judgment. Mismatched layers can introduce misleading constraints. Also, intermediate distillation can conflict with the task loss — the model optimizes a sum of competing objectives.

---

### Self-Distillation

**The problem**: you do not have a larger teacher model. You want the benefits of distillation with only one model.

**The core insight**: use the model's own predictions as soft labels. A model's confidence distribution over outputs encodes learned uncertainty — training on these soft predictions acts like strong label smoothing and regularizes the model.

Born-Again Networks: train a model, then train an identical-architecture model using the first model's outputs as soft targets. Surprisingly, the student often outperforms the original teacher on the test set.

**What breaks**: self-distillation benefits are task-dependent. On tasks where the model is already well-calibrated, the soft labels add little over hard labels. The gains are most visible when the original model is confident-but-sometimes-wrong — the self-distillation corrects overconfident errors.

---

## Pruning

### Unstructured vs Structured Pruning

**The problem**: trained models have millions of redundant parameters. Some weights are near zero and contribute little to any output. Why keep them?

**Unstructured pruning**: zero out individual low-magnitude weights. The weight matrix becomes sparse. Reduces parameter count on paper.

**What breaks**: irregular sparsity gives no real speed benefit on standard hardware. A GPU processes matrices in regular blocks; skipping individual zeros does not reduce the number of memory accesses or compute operations in the common case. You need special sparse tensor cores (NVIDIA Ampere's 2:4 sparsity) or libraries to benefit.

**Structured pruning**: remove entire neurons, attention heads, convolutional filters, or even full layers. The weight matrices are physically smaller. This directly reduces compute and memory on any hardware.

Rule: if you cannot control the inference hardware, use structured pruning. Unstructured pruning is appropriate only when the deployment hardware explicitly supports sparse computation.

---

### Magnitude-Based Pruning

**The problem**: among all weights, which should you remove first?

**The core insight**: weights near zero contribute little to the output by definition. Removing them has minimal effect on model behavior.

**The mechanics**: compute a threshold (globally across all layers, or per-layer). Zero all weights below the threshold. Fine-tune the pruned model to recover accuracy. Repeat.

**What breaks**: global thresholding can be overly aggressive on some layers (whose weights happen to be small in magnitude but are functionally critical) and too conservative on others. A weight that is small now might have been important at an earlier stage of training. Iterative gradual pruning — increase sparsity slowly over training steps rather than all at once — reduces accuracy collapse.

---

### The Lottery Ticket Hypothesis

**The problem**: if trained large networks contain many redundant parameters, why not train small networks from the start? Empirically, small networks trained from random initialization underperform large networks, even if the large network is then pruned to the same size.

**The core insight (Frankle & Carlin, 2019)**: large networks contain small subnetworks — "winning tickets" — that, when initialized to their *original* (pre-training) values and trained alone, converge to the same accuracy as the full network. The large network is a lottery: most initializations lose, but a few initial configurations are lucky. Training the full network is the process of finding the winning ticket.

**The mechanics**: train the full network → prune smallest-magnitude weights → reset remaining weights to their original initialization → retrain this sparse mask. The reset is critical — it is not the final pruned weights that matter, but the original initialization of the surviving connections.

**What breaks**: the original hypothesis was demonstrated on small models (LeNet, VGG on CIFAR). Scaling to modern LLMs requires searching for winning tickets within a regime where the network size is so large that full iterative pruning is prohibitively expensive. The theoretical insight is influential; the practical algorithm is mostly impractical at scale.

---

## Low-Rank Factorization

**The problem**: large weight matrices in Transformers and dense networks are expensive — a linear layer from 4096 to 4096 has 16 million parameters. If the actual information in this matrix is low-dimensional, most of those parameters are encoding redundancy.

**The core insight**: if a matrix $W \in \mathbb{R}^{m \times n}$ has effective rank $r \ll \min(m, n)$, approximate it as $W \approx AB$ where $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$. Parameter count falls from $mn$ to $r(m+n)$.

**SVD-based compression**: compute the full SVD, keep the top $r$ singular values and vectors, discard the rest. The approximation error is minimized in the Frobenius norm sense.

**What breaks**: choosing $r$ is a trade-off between compression ratio and accuracy. The singular values drop off gradually for most weight matrices — there is no sharp cutoff that indicates "this rank is enough." In practice, you choose $r$ by measuring accuracy on a validation set after truncation.

**LoRA (Low-Rank Adaptation)**: the same math applied to fine-tuning. Freeze the pretrained weight $W$; learn a low-rank update $\Delta W = AB$ where $A$ and $B$ are small. At inference, merge: $W' = W + AB$. No extra cost at serving time. Used pervasively for efficient LLM fine-tuning.

---

## Speculative Decoding

**The problem**: autoregressive LLM generation is inherently sequential — you cannot generate token $t+1$ until token $t$ is generated. Each token requires a full forward pass through all layers. The model is memory-bandwidth bound: the bottleneck is reading the model weights from GPU memory, not arithmetic. You cannot parallelize across tokens.

**The core insight**: verification is cheaper than generation. If a small fast model proposes $k$ tokens, the large model can *verify* all $k$ in a single forward pass (because during verification you know the full token sequence and can run parallel attention). Accepted tokens come for free.

**The mechanics**: a small draft model generates $k$ tokens. The large target model runs one forward pass over the original context + $k$ draft tokens, producing $k+1$ probability distributions. For each draft token, check if it falls within the target model's distribution (using rejection sampling). Accept tokens that match; regenerate from the first mismatch.

Typical speedup: 2–3× throughput. Draft model must share the same tokenizer and vocabulary. Works best when draft and target model distributions are similar (same model family, different sizes).

**What breaks**: if the draft and target models have significantly different distributions (e.g., different training data), acceptance rates are low and the speedup disappears — you are doing extra work for little gain.

---

## Combination Strategies

### Prune → Distill → Quantize

A standard production pipeline:

1. **Prune**: structured pruning removes redundant heads, layers, or neurons. Fine-tune the structurally smaller model.
2. **Distill**: use the original full model to distill into the pruned structure. Recovers quality lost to pruning by providing richer training signal.
3. **Quantize**: apply PTQ (or QAT if needed) to the final architecture. INT8 for CPU; FP16 for GPU; INT4 if memory is the hard constraint.

This order matters. Pruning first removes capacity that cannot absorb knowledge efficiently. Distillation fills the remaining capacity. Quantization after has the smallest impact because the preceding steps have already focused the model.

---

### QLoRA

**The problem**: fine-tuning a 70B parameter model requires holding it in memory in full precision plus optimizer states — hundreds of GB. Infeasible on a single GPU.

**The core insight**: quantize the frozen base model to 4-bit (NF4 — a quantization format optimized for neural network weight distributions). Add small LoRA adapters in BF16. Train only the adapters. The quantized base provides context; the adapters learn task-specific behavior.

Result: fine-tuning a 70B model on a single 48GB GPU. The quantized base uses ~35GB; the adapters use a few GB.

**What breaks**: quantizing the base to 4-bit introduces quantization noise into every forward pass. The adapters must learn to produce useful updates despite this noise in the residual stream. Fine-tuned model quality is slightly below full-precision fine-tuning, but the trade-off is almost always worth the hardware savings.

---

## Compression Technique Reference

| Technique | What it compresses | Hardware requirement | When to use |
| :--- | :--- | :--- | :--- |
| **FP16** | Numerical precision | Any modern GPU | Always (baseline) |
| **INT8 PTQ** | Numerical precision | CPU/GPU INT8 support | Production serving |
| **INT4 GPTQ/AWQ** | Numerical precision | GPU | LLMs on constrained hardware |
| **QAT** | Numerical precision | Any | When PTQ quality insufficient |
| **Structured pruning** | Model structure | Any | Reducing compute + memory |
| **Unstructured pruning** | Individual weights | Sparse hardware only | Specialized deployment |
| **Distillation** | Knowledge transfer | Any | Smaller student architecture |
| **Low-rank factorization** | Weight matrix rank | Any | Large linear/attention layers |
| **Speculative decoding** | Token generation speed | Two models in memory | LLM inference throughput |
