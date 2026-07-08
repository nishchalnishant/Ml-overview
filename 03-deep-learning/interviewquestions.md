---
module: Deep Learning
topic: Interview Questions
subtopic: ""
status: unread
tags: [deep-learning, interview-questions, revision]
---

# Deep Learning — Interview Questions

**For:** SDE-2 / AI Engineer interviews — calibrated to what's actually asked Round 1 and beyond.
**Difficulty guide:**
- **Easy** → Round 1 basics: definitions, intuition, "explain the training loop," standard architectural choices. Know these cold.
- **Medium** → Round 2 depth: applied debugging, trade-off reasoning, architecture design choices, NLP/CV applied concepts.
- **Hard** → Staff/Research depth: derivations, distributed training internals, quantization, transformer-scale system design.

---

## Easy

> Round 1 Deep Learning fundamentals. If you've been asked "basic DL questions" in a first round, this is the bucket.

### Q: What is the basic neural network training loop? Walk through it step by step.
```python
for x, y in dataloader:
    optimizer.zero_grad()   # clear stale gradients from last step
    out = model(x)          # forward pass — builds computation graph
    loss = criterion(out, y) # compute scalar loss
    loss.backward()         # backprop — populate .grad on each parameter
    optimizer.step()        # update parameters using .grad
```
Order matters: `zero_grad` must come before `backward` (otherwise old grads accumulate), `step` must come after `backward` (it reads `.grad` which only exists after backward). Also remember `model.train()` before training and `model.eval()` before evaluation — Dropout and BatchNorm behave differently in each mode.

### Q: What is overfitting in a neural network, and how do you detect and fix it?
Overfitting: the model memorizes training-set noise and doesn't generalize. You see low train loss but high validation loss — a widening gap. Fixes: more training data, data augmentation, dropout, weight decay (L2 regularization), simpler architecture, or early stopping. Detect by plotting train vs. validation loss curves over epochs — a persistent and growing gap = overfitting.

### Q: What is the difference between parameters and hyperparameters in a neural network?
**Parameters** (weights, biases) are learned automatically from data via gradient descent. **Hyperparameters** (learning rate, batch size, number of layers, dropout rate, weight decay) are set before training and control the training process or model architecture — they can't be optimized directly by the same gradient-based procedure and are chosen via validation.

### Q: What is an epoch, a batch, and an iteration?
- **Epoch**: one full pass through the entire training dataset.
- **Batch (mini-batch)**: a subset of the data processed together in one forward/backward pass.
- **Iteration**: one forward/backward pass = one batch processed.

Number of iterations per epoch = dataset size / batch size. Schedulers set to "per iteration" vs. "per epoch" behave very differently.

### Q: Compare sigmoid, tanh, and ReLU — when would you pick each?
- **Sigmoid**: squashes to (0,1), useful for binary output probabilities and gates (LSTM). Not good for hidden layers — saturates and causes vanishing gradients, and outputs are not zero-centered.
- **Tanh**: squashes to (−1,1), zero-centered (better gradient flow than sigmoid) but still saturates.
- **ReLU**: `max(0,z)`, doesn't saturate for positive inputs, cheap to compute, default for hidden layers in CNNs/MLPs. Can "die" if a large negative bias pushes all activations below zero permanently.

### Q: Why can't you initialize all weights to zero?
With all weights identical, every neuron in a layer computes the same output and receives the same gradient — **symmetry problem**. All neurons stay identical forever regardless of width. Random initialization breaks this symmetry so different neurons learn different features.

### Q: What goes wrong if weights are initialized with too large or too small variance?
- **Too large**: activations explode through depth, saturating sigmoid/tanh and causing gradient explosions.
- **Too small**: activations shrink toward zero through depth, gradients vanish, early layers receive negligible learning signal. Both are addressed by variance-preserving schemes — **Xavier/Glorot** for tanh/sigmoid, **He/Kaiming** for ReLU (scales by 2/n_in to compensate for ReLU zeroing ~half of activations).

### Q: What is the vanishing gradient problem, and why is it worse with sigmoid than ReLU?
Backprop multiplies Jacobians across all layers. Sigmoid's derivative maxes at 0.25 — through 10 layers the gradient can shrink by 0.25¹⁰ ≈ 10⁻⁶, stopping learning in early layers. ReLU's derivative is exactly 1 in the active region, so it doesn't systematically shrink gradients. Fixes: ReLU/GELU activations, residual connections, proper initialization (He init), BatchNorm.

### Q: How does dropout work and why is it regularization?
During training, each unit's activation is independently zeroed with probability p, and remaining activations are scaled by 1/(1−p) (inverted dropout) so expected magnitude matches inference. It prevents **co-adaptation** — units can't rely on any specific other unit always being present, forcing more robust representations. Interpreted as implicitly training an ensemble of exponentially many subnetworks sharing weights, averaged at test time.

### Q: What is BatchNorm and what problem does it solve?
BatchNorm normalizes each activation using the current batch's mean and variance (per-channel), then applies learnable scale and shift. It solves **internal covariate shift** — the changing distribution of each layer's inputs as previous layers update — by keeping activations in a well-scaled range. Benefits: allows higher learning rates, more robust to initialization, provides some regularization. At inference it uses stored running mean/variance (not batch statistics) — forgetting `model.eval()` before inference is a common bug.

### Q: What is transfer learning and why does it work?
Reusing representations learned on a large source task for a related target task with less data. It works because early/mid layers of deep networks learn general-purpose features (edges, textures, syntax patterns) that transfer across tasks — only later, task-specific layers need adaptation. Feature extraction (freeze backbone, train only a new head) suits small target datasets; full fine-tuning suits larger or more domain-different targets.

### Q: What is the difference between binary cross-entropy and categorical cross-entropy? When do you use each?
- **Binary cross-entropy (BCE)**: for binary classification (one sigmoid output) or multi-label classification (independent sigmoid per class, since classes aren't mutually exclusive).
- **Categorical cross-entropy**: for single-label multiclass classification (one softmax output across all classes, classes are mutually exclusive).
- Using softmax for multi-label is wrong — softmax forces probabilities to sum to 1, implying mutual exclusivity that doesn't exist.

### Q: What is the difference between an LSTM and a vanilla RNN?
Vanilla RNN: `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)`. Backpropagating through time multiplies the same matrix W_hh repeatedly — if its largest eigenvalue is <1, gradients vanish exponentially with sequence length. LSTM adds a **cell state** updated additively (not multiplicatively) via learned gates (forget, input, output), providing a "gradient highway" that lets gradients flow many steps with much less decay. Forget gate bias initialized near 1 helps early training.

### Q: What does `model.eval()` actually change?
Sets a `training` flag on every submodule. This changes behavior only in layers that read that flag: **Dropout** becomes a no-op (identity pass-through), **BatchNorm** switches from computing batch statistics to using stored running mean/variance. It does **not** disable gradient tracking — you still need `torch.no_grad()` for that. Forgetting `eval()` before inference with BatchNorm-heavy models (e.g., ResNets) gives noisy, batch-size-dependent predictions.

### Q: What is gradient descent and what are the main variants?
An iterative optimization algorithm that updates parameters by moving in the negative gradient direction: `θ ← θ − α·∇L(θ)`. Variants:
- **Batch GD**: full dataset per step — stable but slow for large datasets.
- **SGD**: one sample per step — noisy but fast.
- **Mini-batch SGD**: 32–512 samples — GPU-efficient sweet spot, current default.
- **SGD + Momentum**: accumulates exponentially decaying moving average of past gradients — dampens oscillations, accelerates consistent directions.
- **Adam**: adaptive per-parameter learning rates via first and second gradient moment estimates — fast convergence, current default for transformers and NLP.

---

## Medium

#### Q: How do Diffusion Models (like Stable Diffusion) work at a high level, and how do they compare to GANs? [Generative Vision]
A: **Diffusion Models** work by defining a forward process that incrementally adds Gaussian noise to an image over $T$ steps until it becomes pure static. The neural network (typically a U-Net) is then trained on the reverse process: given a noisy image and the timestep $t$, predict the specific noise that was added at that step. To generate a new image, you sample pure noise and iteratively run it through the network $T$ times to denoise it into a coherent image. Text-to-image conditioning (like Stable Diffusion) is done by injecting text embeddings (via Cross-Attention) into the U-Net at each step to guide the denoising toward the prompt. **Comparison to GANs:** GANs are trained via an adversarial game between a Generator and Discriminator. GANs require only a single forward pass to generate an image (making them extremely fast), but they suffer from training instability (mode collapse, vanishing gradients). Diffusion models are much more stable to train and capture a wider diversity of data (no mode collapse), but require dozens or hundreds of sequential forward passes to generate a single image, making them much slower at inference time.


> Round 2 depth — applied debugging, design trade-offs, and "how would you build this?" questions.

### Q: Why do gradients accumulate across `.backward()` calls and why does that matter?
PyTorch adds new gradients into `.grad` rather than overwriting them. In a standard loop this means you **must call `optimizer.zero_grad()` before each `.backward()`** — otherwise gradients from the previous step silently accumulate and corrupt the update. A hard-to-notice bug: loss curves look "off" but don't crash. Intentional use: gradient accumulation over micro-batches to simulate a larger effective batch size on limited GPU memory.

### Q: Your training loss is NaN after a few iterations. How do you debug it?
Check in order: (1) **learning rate too high** → gradient/weight blow-up; (2) **unclipped exploding gradients** → add `torch.nn.utils.clip_grad_norm_`; (3) **division by zero or log(0)** in a custom loss; (4) **mixed-precision overflow/underflow** → use `GradScaler`; (5) **bad input data** (NaN/Inf in a batch). Tool: `torch.autograd.set_detect_anomaly(True)` pinpoints the exact operation that produced the NaN (slow, dev-only).

### Q: Explain Adam's update rule and what each moment estimate does.
Adam maintains a first moment (mean, like momentum) and second moment (uncentered variance) of gradients: m_t = β₁m_{t-1} + (1−β₁)g_t; v_t = β₂v_{t-1} + (1−β₂)g_t². Bias-corrected: m̂_t = m_t/(1−β₁ᵗ); v̂_t = v_t/(1−β₂ᵗ). Update: θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε). The **first moment** gives momentum-like smoothing; the **second moment** gives a per-parameter adaptive learning rate — shrinking steps for parameters with historically large/noisy gradients and growing steps for consistently small-gradient parameters. This adaptivity is why Adam often converges fast with minimal LR tuning.

### Q: Why does AdamW outperform Adam+L2 weight decay?
In standard Adam, adding L2 to the loss makes the decay term go through Adam's adaptive denominator (√v̂_t), so parameters with large historical gradients get **less effective decay than intended** — regularization strength is entangled with the adaptive scaling. AdamW **decouples** weight decay: θ_{t+1} = θ_t − η·(m̂_t/(√v̂_t+ε) + λθ_t), shrinking weights by a fixed fraction independent of adaptive scaling. This is now the default in nearly all transformer training recipes.

### Q: What's the difference between feature extraction and full fine-tuning? When do you use each?
**Feature extraction**: freeze the backbone (pretrained weights locked), train only a new task-specific head. Fewer trainable parameters → less overfitting risk. Best for small target datasets or when source/target domains are similar. **Full fine-tuning**: unfreeze everything, use a small learning rate, adapt all layers. Best for larger target datasets or domains that differ substantially from the source. Middle ground: **gradual unfreezing** — train head first, then progressively unfreeze deeper blocks with layer-wise discriminative learning rates (later layers get higher LR than early layers).

### Q: What is self-supervised learning and why has it become dominant?
Self-supervised learning creates labels from the data itself (predicting masked tokens/pixels, contrasting augmented views) so it can exploit **unlabeled data at scale** — critical since labeled data is expensive and unlabeled data is nearly unlimited. Examples: BERT's masked language modeling, SimCLR/MoCo's contrastive instance discrimination, MAE's masked patch reconstruction. Produces more transferable, less task-biased representations than training end-to-end on one narrow supervised objective.

### Q: Walk through LSTM's gates and explain how the cell state solves vanishing gradients.
LSTM maintains a cell state c_t updated **additively**: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t, where:
- **Forget gate** f_t = σ(W_f·[h_{t-1}, x_t]) — controls retention
- **Input gate** i_t — controls how much new candidate information c̃_t = tanh(W_c·[h_{t-1}, x_t]) enters
- **Output gate** o_t — controls how much of c_t (through tanh) becomes h_t

Because c_t is updated additively (not via repeated matrix multiplication), the gradient path ∂c_t/∂c_{t-1} = f_t can stay close to 1 when the forget gate is near 1, letting gradients flow across many timesteps largely undiminished — a "gradient highway" analogous to residual connections.

### Q: Explain the Query/Key/Value formulation of self-attention.
Each input token is projected into three vectors: Q = XW_Q, K = XW_K, V = XW_V. Attention scores are the scaled dot product softmax(QKᵀ/√d_k) — scaling by √d_k prevents dot products from growing large with dimension, which would push softmax into a saturated, near-one-hot regime with vanishing gradients. Final output = softmax(QKᵀ/√d_k)·V. Intuitively: each token "queries" all other tokens' "keys" to decide how much of each "value" to incorporate into its updated representation.

### Q: What is causal masking and why is it needed?
Causal masking sets attention scores for future positions to −∞ before the softmax (→ zero weight), ensuring position i can only attend to positions ≤ i. This preserves the autoregressive property: during training with teacher forcing the model sees the whole sequence at once, but each position must only use past context — matching inference-time generation where future tokens genuinely don't exist yet.

### Q: Why use multiple attention heads instead of one large attention operation?
Multi-head attention splits Q, K, V into h smaller subspaces (each d_k = d_model/h), computes attention independently per head, then concatenates and projects. Different heads can specialize in different relationships (syntactic dependencies, positional proximity, coreference) simultaneously and in parallel — empirically far more expressive than a single attention over the full dimension, at roughly the same total compute cost.

### Q: Encoder-only vs. decoder-only vs. encoder-decoder — when is each appropriate?
- **Encoder-only (BERT)**: bidirectional (non-causal) attention — good for understanding tasks where the full input is available (classification, NER, embeddings).
- **Decoder-only (GPT family)**: causal attention only — natural for open-ended generation, dominant for general-purpose LLMs due to simplicity and scaling behavior.
- **Encoder-decoder (T5, original Transformer)**: bidirectional encoder + causal decoder with cross-attention — well-suited for sequence-to-sequence tasks with a clear distinct input/output (translation, summarization).

### Q: Why does LayerNorm replace BatchNorm in transformers?
LayerNorm normalizes across the **feature dimension** for each individual sample (independent of batch), rather than across the batch for each feature. This makes it: (1) invariant to batch size (works at batch size 1, critical for autoregressive generation), (2) handles variable sequence lengths without skewed statistics from padding, (3) no train/inference discrepancy (no running statistics to switch between). BatchNorm's per-batch statistics become noisy with very small batches and are complicated by padding in variable-length sequences.

### Q: What is the reparameterization trick in VAEs and why is it necessary?
Directly sampling z ~ q(z|x) = N(μ, σ²) is a stochastic operation with no defined gradient w.r.t. μ, σ — blocking backpropagation through the sampling step. The reparameterization trick rewrites the sample as a deterministic, differentiable function: **z = μ + σ ⊙ ε**, ε ~ N(0,I). Now gradients flow through μ and σ normally, with all randomness isolated in ε, which doesn't need gradients.

### Q: Compare subword tokenization approaches: BPE, WordPiece, SentencePiece.
**BPE (Byte-Pair Encoding)**: starts with individual characters, iteratively merges the most frequent adjacent pair — deterministic, used in GPT-2/3/4 (byte-level BPE guarantees no OOV). **WordPiece**: similar to BPE but selects merges by maximizing likelihood of the training corpus (used in BERT). **SentencePiece/Unigram**: treats tokenization probabilistically, works directly on raw text without pre-tokenization (language-agnostic, used in LLaMA, T5). All are middle grounds between word-level (huge vocabulary, OOV problem) and character-level (long sequences, poor semantics).

### Q: What is beam search and why not always use greedy decoding?
Greedy decoding picks the single highest-probability token at each step — a locally optimal choice can lead to a globally worse full sequence. Beam search maintains the top-k partial sequences ("beams"), expanding all and keeping only the k highest-scoring continuations — approximating search over the full sequence space without exponential cost. Larger beam width improves quality up to a point, with diminishing/sometimes negative returns beyond it (can favor generic, "safe" outputs). For LLMs, temperature sampling + top-p/top-k is often preferred over beam search for generating diverse, creative text.

### Q: What is focal loss and what problem does it solve?
Focal loss modifies cross-entropy with a modulating factor: FL = −(1−p_t)^γ · log(p_t). This **down-weights easy, well-classified examples** (high p_t → small modulating factor) and focuses training on hard/misclassified examples. Designed for extreme class imbalance in dense object detection (RetinaNet), where the vast majority of anchor boxes are easy negatives that would otherwise dominate the gradient and swamp the rare positive examples.

### Q: What is knowledge distillation and why do "soft labels" matter?
A smaller **student** model is trained to match a larger **teacher** model's output distribution (soft, temperature-scaled probabilities) rather than just hard ground-truth labels. Soft labels carry "dark knowledge" — relative probabilities across incorrect classes reveal the teacher's learned similarity structure (a "dog" image getting some probability on "wolf" but almost none on "car"), which is far richer supervisory signal than a one-hot label, letting the student learn faster and generalize better.

### Q: What is the difference between structured and unstructured pruning?
**Unstructured pruning**: zeroes out individual weights (e.g., smallest magnitude) — achieves high sparsity ratios with minimal accuracy loss but the resulting irregular sparse pattern can't be exploited for speedup on standard dense hardware without specialized sparse kernels. **Structured pruning**: removes entire structural units (channels, attention heads, whole layers) — produces a smaller **dense** model that runs faster on standard hardware immediately, though it must remove less to preserve the same accuracy since it can't cherry-pick individually unimportant weights.

---

## Hard

#### Q: Explain the mathematical distinction between predicting the noise ($\epsilon$) vs predicting the image ($x_0$) in a Denoising Diffusion Probabilistic Model (DDPM). [Generative Vision]
A: In DDPMs, the reverse process involves predicting the mean of the Gaussian distribution for the previous timestep $t-1$. Mathematically, you can parameterize the neural network to either predict the original clean image $\hat{x}_0$ from the noisy image $x_t$, OR predict the exact noise $\hat{\epsilon}$ that was added to reach $x_t$. The DDPM paper showed that predicting the noise $\hat{\epsilon}$ is mathematically equivalent to predicting the score function (the gradient of the log probability density) of the data distribution, drawing a direct link to Langevin dynamics and Score-Based Generative Modeling. Predicting the noise $\hat{\epsilon}$ works much better in practice because the noise has a stable, zero-centered Gaussian distribution at all timesteps, making it an easier and more stable target for a neural network to optimize via simple MSE loss, whereas the distribution of the "original image" varies wildly.


> Staff-level and Research Engineer depth — derivations, system design, distributed training internals.

### Q: What's the difference between forward-mode and reverse-mode automatic differentiation, and why does deep learning use reverse-mode?
**Forward-mode** propagates derivatives alongside the computation from inputs to outputs — efficient when there are few inputs and many outputs. **Reverse-mode (backprop)** propagates derivatives from the scalar output backward to inputs — efficient when there are many inputs (millions of weights) and one scalar output (the loss), which is exactly the deep learning setting. Reverse-mode computes the full gradient in one backward pass regardless of the number of parameters, at the cost of storing intermediate activations (memory-for-compute tradeoff, addressed by gradient checkpointing).

### Q: What is GELU and why do transformers prefer it over ReLU?
GELU (Gaussian Error Linear Unit) = x · Φ(x), a smooth approximation that weights inputs by their percentile under a standard normal — effectively a smoother, probabilistic version of ReLU. It's differentiable everywhere (no kink at 0) which empirically improves optimization dynamics for transformers. Default activation in BERT/GPT FFN blocks. **SwiGLU** (used in LLaMA, PaLM) is a gated variant: SwiGLU(x) = Swish(xW) ⊙ (xV) — splits the FFN up-projection into two paths, one gated through Swish, multiplied elementwise by the other. This input-dependent gating gives more expressive nonlinearity per-FLOP than GELU-based FFNs, at the cost of an extra weight matrix.

### Q: What problem do residual connections solve and why "identity" specifically?
Plainly stacking more layers causes **training** accuracy (not just generalization) to degrade — not overfitting but optimization difficulty where deeper plain networks can't even fit the training set as well as shallower ones. A residual block y = F(x) + x makes identity trivial to represent (just drive F(x) → 0 — a small perturbation near zero is easier to fit than forcing nonlinear layers to approximate identity exactly). The skip connection also provides a direct gradient path (Jacobian includes an identity term), mitigating vanishing gradients across very deep stacks.

### Q: Derive why Adam's bias correction terms are necessary.
Adam initializes m₀ = v₀ = 0, so early moment estimates are biased toward zero: E[m_t] ≈ (1−β₁ᵗ)·E[g_t], meaning at t=1 with β₁=0.9, m₁ is only 10% of the true gradient magnitude. Without correction, the first several updates would be artificially tiny. Dividing by (1−β₁ᵗ) and (1−β₂ᵗ) rescales early estimates to be unbiased in expectation, giving **consistent effective step sizes from the very first update** rather than a slow ramp-up period. This matters most at the start of training when stable step sizes are most needed.

### Q: What is learning rate warmup and why do transformers specifically need it?
Warmup linearly ramps LR from ~0 to the target LR over the first N steps before applying the main schedule (e.g., cosine decay). Early in training, Adam's second-moment estimate v_t is based on very few samples — a noisy/biased underestimate of true gradient variance. Despite bias correction, effective step size can be erratically large, causing instability or divergence. Warmup keeps steps small while these statistics stabilize. Especially critical for transformers with layer norm and residual streams, where large early updates can push the model into a bad region it never recovers from.

### Q: Pre-norm vs. post-norm transformer blocks — why did the field move to pre-norm?
**Post-norm** (original Transformer): LN(x + Sublayer(x)) — applies LayerNorm after the residual addition. Can cause gradient instability in very deep stacks since the residual stream itself gets renormalized every block; requires careful warmup. **Pre-norm** (GPT-2/3, LLaMA): x + Sublayer(LN(x)) — applies LayerNorm before the sublayer, on a branch off the residual stream, leaving the residual path unnormalized (a clean, unimpeded gradient highway). Much more stable at depth, often removes the need for warmup, at a minor cost to final performance ceiling in some studies. Adopted by essentially all modern LLMs.

### Q: Derive the VAE loss (ELBO) and explain what each term does.
The Evidence Lower Bound: **L = E_{q(z|x)}[log p(x|z)] − D_KL(q(z|x) ‖ p(z))**. The **first term** is reconstruction loss — how well can we decode z back to x? The **second term** is KL divergence regularizing the learned approximate posterior q(z|x) (parameterized as N(μ(x), σ²(x)) by the encoder) toward a fixed prior p(z) = N(0,I). This KL term keeps the latent space continuous, smooth, and sample-able — enabling generation by sampling z ~ p(z) and decoding. Unlike a vanilla autoencoder whose latent space has no structure, the VAE's prior constraint ensures nearby points in latent space decode to similar outputs.

### Q: What is the KV cache and why is it essential for efficient autoregressive generation?
During generation, each new token's query only needs to attend to all previous tokens' keys and values. Since those don't change across steps, **caching K and V** avoids recomputing the entire prefix at every new token, reducing per-step compute from O(n²) to O(n) (growing with context length, but per-token cost stays roughly constant). Trade-off: KV cache size grows linearly with sequence length × layers × heads × head dim × batch size, becoming the dominant memory consumer for long-context serving — driving techniques like Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and PagedAttention.

### Q: Compare MHA, MQA, and GQA for serving efficiency.
**Multi-Head Attention (MHA)**: separate K/V projections per head → KV cache scales with all heads — highest quality but most memory at long context. **Multi-Query Attention (MQA)**: single K/V head shared across all query heads → KV cache shrinks by ×(num_heads), dramatically cutting memory bandwidth during decode, at some cost to model quality. **Grouped-Query Attention (GQA)**: K/V heads shared across small groups of query heads — middle ground. Most modern serving-oriented LLMs (LLaMA-2/3, Mistral) use GQA to get most of MQA's serving efficiency while retaining more of MHA's quality.

### Q: What is Flash Attention and why is it faster without changing outputs?
Standard attention is **memory-bandwidth-bound** at typical sequence lengths (not compute-bound). Flash Attention is an I/O-aware, exact implementation that avoids materializing the full n×n attention matrix in slow GPU HBM. It tiles computation and fuses matmul, softmax, and weighted-sum within fast on-chip SRAM, recomputing parts on-the-fly during backward rather than storing large intermediates — mathematically identical output, but 2-4x wall-clock speedup and O(n) memory instead of O(n²).

### Q: What is speculative decoding and how does it speed up generation without changing outputs?
A small, fast **draft** model proposes several candidate tokens ahead in one cheap pass; the large **target** model verifies all of them in a single parallel forward pass. The verification accepts the longest prefix of draft tokens that matches what the target would have generated, resampling at the first divergence. Exploits the fact that decode is memory-bandwidth-bound — one target-model pass verifying K draft tokens costs barely more than verifying one token. If draft guesses are often correct, multiple tokens are produced per "expensive" target-model pass while **provably preserving the target model's exact output distribution**.

### Q: Explain ZeRO's three stages and what each shards.
ZeRO (Zero Redundancy Optimizer) eliminates the memory redundancy of data parallelism where every GPU keeps a full copy of optimizer states, gradients, and parameters. **Stage 1**: shards optimizer states (Adam's momentum/variance, which can be 2-3x the model size in fp32) across GPUs. **Stage 2**: additionally shards gradients. **Stage 3**: additionally shards the parameters themselves — each GPU only permanently holds a slice, gathering full parameters on-demand during forward/backward via communication. Each stage trades more communication for less per-GPU memory, enabling training of models that wouldn't fit under standard DDP.

### Q: How does GPTQ differ from naive round-to-nearest quantization for LLMs?
GPTQ quantizes weights layer-by-layer using **second-order (Hessian-based) information** from a calibration set: it greedily chooses which weights to quantize and updates the remaining unquantized weights to compensate for the error introduced — a form of Optimal Brain Surgeon-style correction. This enables accurate 4-bit (even lower) quantization with minimal calibration data and no retraining. AWQ (Activation-aware Weight Quantization) observes that a small fraction of weight channels are disproportionately important based on activation magnitudes, and protects those salient channels via per-channel scaling before quantization — simpler/faster than GPTQ's per-layer optimization with similar accuracy retention.

### Q: What is QLoRA and why did it make LLM fine-tuning dramatically cheaper?
QLoRA quantizes the **frozen base model to 4-bit** (NF4 data type optimized for normally-distributed weights + double quantization) to drastically cut memory footprint, then adds small trainable **LoRA adapter matrices** (low-rank A, B such that the effective weight update ≈ BA) on top in full/half precision — training only those adapters. Combination: 4-bit frozen model + fp16 trainable adapters = fine-tuning 30B+ parameter models on a single consumer GPU (24-48GB) that previously required multiple data-center GPUs for full fine-tuning.

### Q: Walk through what can go wrong numerically when combining gradient checkpointing, mixed precision, and DDP in the same training run.
Each technique alone is well-tested, but their interactions create subtle failure modes: **Gradient checkpointing** recomputes forward activations during backward — any non-deterministic op (some cuDNN algorithms, dropout without fixed RNG state replay) can produce a different activation on recompute than the original forward pass, silently corrupting gradients. **Mixed precision GradScaler** interacts with DDP's gradient all-reduce: if scaling overflows on one rank but not another (due to data-dependent numerical differences), naïve implementations can desync the "found_inf" flag across ranks unless synchronized globally before deciding whether to skip the optimizer step. **Debugging approach**: disable one technique at a time to bisect; verify checkpointed dropout RNG state is deterministically replayed (`torch.utils.checkpoint` handles this via `fork_rng`, but custom code often misses it); use `set_detect_anomaly(True)` plus fixed seeds on a minimal reproducer to localize the exact operation producing divergent results.

---

## Summary Table — Quick Reference

| Concept | Key fact |
|---|---|
| ReLU vs sigmoid | ReLU gradient = 1 in active region; sigmoid max gradient = 0.25 |
| Xavier init | Scales by 1/n_in (tanh/sigmoid); He init scales by 2/n_in (ReLU) |
| Dropout | p=zero prob; inverted scaling by 1/(1−p); ensemble interpretation |
| BatchNorm | Train: batch stats; Eval: running mean/var; `model.eval()` is required |
| LayerNorm | Per-sample across features; batch-size invariant; used in transformers |
| Attention | softmax(QKᵀ/√d_k)·V; multi-head splits into h parallel subspaces |
| Causal mask | Future positions → −∞ before softmax; enables autoregressive training |
| Adam | m_t = momentum; v_t = adaptive LR; bias-corrected; AdamW decouples decay |
| KV cache | Cache K,V across steps; O(n²)→O(n) per decode step |
| Flash Attention | I/O-aware exact attention; avoids materializing n×n; O(n) memory |
| GQA | K/V shared across groups of query heads; serving efficiency compromise |
| ZeRO Stage 3 | Shards optimizer state + gradients + parameters across GPUs |
| QLoRA | 4-bit frozen base + fp16 LoRA adapters; fine-tune 30B on 1 GPU |
| Speculative decoding | Draft model proposes K tokens; target verifies in one pass |
| ELBO | E[log p(x\|z)] − KL(q(z\|x) ‖ p(z)); reconstruction + latent regularization |
