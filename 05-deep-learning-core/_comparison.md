---
module: Deep Learning
topic: Cheatsheet
subtopic: Comparison
status: unread
tags: [deeplearning, ml, cheatsheet-comparison]
---
# Deep Learning Comparison Cheatsheet

Rapid pre-interview reference. Every entry: what it is / pros / cons / when to pick over alternatives / key formula.

---

## Architectures

### CNNs

**What it is**: weight-shared local filters over spatial data; builds hierarchical features via depth.
- **Pros**: parameter-efficient (weight sharing), strong image inductive bias (locality, translation equivariance), data-efficient vs Transformers.
- **Cons**: no built-in rotation/scale invariance; global receptive field only emerges gradually with depth; struggles with long-range dependencies.
- **Pick over ViT when**: limited training data (<1M images), need fast inference on edge/mobile, or task has strong local structure.
- **Formula**: `H_out = floor((H_in + 2·pad − K)/stride) + 1`; params = `K²·C_in·C_out`.

### ResNet (Residual Networks)

**What it is**: stacks convolutional blocks with skip connections `y = F(x) + x`, letting layers learn residuals instead of full transforms.
- **Pros**: solves degradation problem, trains reliably at 100+ layers, gradient highway via identity term.
- **Cons**: needs shape-matching (1×1 projection when downsampling); still quadratic-ish param cost without bottleneck design.
- **Pick over plain deep CNN when**: depth > ~10 layers — always use residual connections beyond this.
- **Formula**: `dL/dx = dL/dy · (dF/dx + I)` — the `+I` prevents vanishing gradient regardless of depth.

### VGG / Inception / DenseNet / EfficientNet / ConvNeXt (CNN family evolution)

- **VGG**: uniform 3×3 convs, depth > large kernels; simple but 138M params, FC-heavy.
- **Inception/GoogLeNet**: parallel multi-scale filters (1×1/3×3/5×5) + 1×1 bottlenecks; 4M params vs VGG's 138M. Pick when filter scale is uncertain a priori.
- **DenseNet**: every layer connects to every subsequent layer (concat); low-level features stay accessible. Cons: memory grows quadratically with depth — harder to scale very deep.
- **EfficientNet**: compound-scales depth/width/resolution together (`α^φ, β^φ, γ^φ`). Pick for best accuracy/FLOPs ratio, especially mobile (EfficientNet-B0) to max-accuracy (B7).
- **ConvNeXt**: modernizes ResNet with ViT design tricks (7×7 depthwise, GELU, LayerNorm) without attention — proves much of ViT's gain was recipe, not attention.

### Vision Transformer (ViT) / DeiT / Swin

**What it is**: treats image as patch sequence, applies standard Transformer self-attention.
- **Pros**: global receptive field from layer 1; scales extremely well with data+compute.
- **Cons**: no inductive bias — needs massive pretraining data (JFT-300M) to beat CNNs from scratch; O(N²) attention cost.
- **Pick ViT over CNN when**: huge labeled/pretraining data available and compute is not the constraint (or use DeiT for ImageNet-1k-only budgets).
- **Pick Swin over ViT when**: need dense prediction (detection/segmentation) or high-res images — windowed attention gives O(M²·N) linear cost and multi-scale features.
- **DeiT**: distills from CNN teacher to match ViT accuracy without extra data.

### RNN / LSTM / GRU (sequence architectures)

- **Vanilla RNN**: `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Cons: vanishing/exploding gradients, ~10-20 step effective memory. Rarely used alone now.
- **LSTM**: 4 gates (forget/input/candidate/output), cell state `c_t = f_t⊙c_{t-1} + i_t⊙c̃_t` gives constant error carousel. Pick over vanilla RNN whenever sequence length > ~20 steps.
- **GRU**: 2 gates (reset/update), ~25% fewer params than LSTM, comparable accuracy. Pick over LSTM when compute/memory constrained or dataset is small (less overfitting risk).
- **Bidirectional RNN**: concatenates forward+backward hidden states. Cons: cannot be used autoregressively (needs full sequence upfront) — unusable for causal LM.
- **RNN/LSTM/GRU vs Transformer**: RNNs are O(1) memory per step but sequential (no parallelism across time); Transformers are parallel across sequence but O(n²) attention and O(n) KV-cache memory at inference.
- **SSMs/Mamba**: dual recurrent (inference, O(1)/step) and convolutional (training, parallel) form; selective state update. Pick when need linear-time long-context inference without full attention cost.

### Transformer (Encoder-only / Decoder-only / Encoder-Decoder)

- **Encoder-only** (BERT, RoBERTa, DeBERTa): bidirectional attention, good for understanding/classification tasks, not for generation.
- **Decoder-only** (GPT, LLaMA, Mistral, Claude): causal masking, autoregressive generation, dominant for LLMs.
- **Encoder-Decoder** (T5, BART, Whisper, original Transformer): cross-attention bridges the two; best for seq2seq (translation, summarization, ASR).
- **Modern stack conventions** (LLaMA/Mistral/Gemma vs original Transformer): Pre-Norm (not Post-Norm), RMSNorm (not LayerNorm), SwiGLU FFN (not ReLU FFN), RoPE (not sinusoidal/learned), GQA (not full MHA).

### Autoencoder family

- **Undercomplete AE**: bottleneck forces compression; MSE/BCE reconstruction. Cons: if bottleneck too wide, learns identity — no useful compression.
- **Denoising AE**: corrupt input, reconstruct clean — forces robust features (BERT MLM is structurally this).
- **Sparse AE**: wide latent + L1 penalty on activations → interpretable, disentangled features; used in LLM interpretability (Anthropic).
- **VAE**: encodes to distribution `N(μ,σ²)`, ELBO objective, reparameterization trick `z = μ + ε⊙σ` enables backprop through sampling. Pick over vanilla AE when you need a generative model with a smooth, sample-able latent space.
- **β-VAE**: `β>1` on KL term trades reconstruction quality for disentanglement; too high β → posterior/KL collapse.

### Generative model family comparison

| Model | Training | Sample quality | Latent control | Speed | Pick when |
|---|---|---|---|---|---|
| **VAE** | ELBO | Blurry (images) | Smooth interpolation | Fast | Need structured latent space, anomaly detection, stable training |
| **GAN** | Adversarial min-max | Sharp | Less stable | Fast | Need sharp samples fast, tolerate unstable training |
| **Diffusion** | Denoising score matching | Best | Moderate | Slow (multi-step) | Quality is priority, can afford iterative sampling |
| **Flow** | Exact likelihood | Good | Exact inversion | Medium | Need exact likelihood / invertibility |

### Object Detection Architectures

- **Two-stage (R-CNN → Faster R-CNN)**: region proposals then classify. Higher accuracy, slower. Pick when accuracy > latency.
- **One-stage (YOLO)**: dense grid prediction in one pass. Faster (45+ FPS), historically weaker on small/dense objects. Pick for real-time.
- **DETR**: set prediction via Transformer decoder + Hungarian matching, no anchors/NMS. Cons: slow convergence (~500 epochs); Deformable DETR fixes this with sparse attention.

### Segmentation Architectures

- **FCN**: 1×1 convs + upsampling; coarse recovery of detail.
- **U-Net**: symmetric encoder-decoder with concatenative skip connections; preserves fine spatial detail. Standard for medical imaging.
- **DeepLab (atrous/dilated conv + ASPP)**: grows receptive field without losing resolution or striding; captures multi-scale context.
- **Mask R-CNN**: adds per-instance mask head to Faster R-CNN + ROIAlign (bilinear, not quantized). Pick over semantic segmentation when you need instance-level separation.

### Architecture Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| CNN (ResNet/EfficientNet) | Limited data, edge deployment | Need global context from layer 1 | Strong bias = data-efficient but capped expressiveness |
| ViT/Swin | Huge data/compute, dense prediction (Swin) | Small datasets, no pretraining budget | No bias = needs scale, but scales better |
| RNN/LSTM/GRU | Streaming/low-memory sequential inference | Long-range deps, need parallel training | O(1) memory/step but sequential compute |
| Transformer (decoder-only) | Autoregressive generation, LLMs | Extremely long context on tight memory | Parallel training, O(n²) attention / O(n) KV cache |
| SSM/Mamba | Long-context, linear-time inference | Need mature tooling/ecosystem | Linear time but newer, less battle-tested |
| VAE | Structured latent, anomaly detection | Need photorealistic samples | Stable but blurry |
| Diffusion | Best quality generation | Latency-critical generation | Best quality, slowest sampling |

---

## Activation Functions

- **Sigmoid**: `σ(z)=1/(1+e^-z)`, range (0,1), max derivative 0.25. Cons: vanishing gradients, not zero-centered. Use only for output layer of binary classification / gates (LSTM/GRU).
- **Tanh**: range (-1,1), zero-centered (better than sigmoid for hidden layers) but still saturates. Pick over sigmoid for RNN hidden states when zero-centering matters.
- **ReLU**: `max(0,z)`. Pros: no saturation for z>0, cheap. Cons: dying ReLU (permanent zero gradient if pre-activation stays negative). Default for CNNs.
- **Leaky ReLU**: `z if z>0 else αz` (α≈0.01). Pick over ReLU when dying-neuron rate is high (large LR, deep nets).
- **ELU**: `α(e^z−1)` for z≤0. Pros: mean-centering (pushes mean activation toward 0, speeds convergence). Cons: exp() is costly. Pick over Leaky ReLU when convergence speed matters more than compute cost.
- **GELU**: `z·Φ(z)` (Φ = standard normal CDF). Smooth, non-monotonic near 0. Default in Transformers (GPT/BERT/ViT). Pick over ReLU when using Transformer/attention architectures — empirically better for these.
- **Swish/SiLU**: `z·σ(z)`. Similar smoothness to GELU, used in EfficientNet. PyTorch calls it `SiLU`.
- **SwiGLU**: `Swish(xW)⊙(xV)` — gated variant, needs 3 weight matrices, hidden dim scaled to 8/3·d to match param budget. Used in LLaMA/PaLM/Gemini. Pick over plain GELU FFN in decoder-only LLMs — consistently improves quality at same compute.
- **Softmax**: multiclass output layer; always use numerically-stable subtract-max trick.

### Activation Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Sigmoid | Binary output, gates | Deep hidden layers | Saturates both ends, vanishing gradient |
| Tanh | RNN hidden state | Very deep nets | Zero-centered but still saturates |
| ReLU | CNN hidden layers (default) | High dying-neuron rate | Cheap, no saturation for z>0, but dead neurons |
| Leaky ReLU / ELU | Fixing dying ReLU | Need max speed (ELU costs exp) | Small negative slope keeps gradient alive |
| GELU | Transformers (default) | Very latency-sensitive CNNs | Smooth, slightly costlier than ReLU |
| Swish/SiLU | EfficientNet-style CNNs | — | Similar to GELU, simpler formula |
| SwiGLU | Modern LLM FFN (LLaMA-class) | Param-constrained tiny models (3 matrices) | Best quality/compute for LLM FFN |

---

## Optimizers

- **SGD (mini-batch)**: simplest, implicit regularization via flat minima. Pick over Adam for CNN/vision benchmarks where generalization > convergence speed.
- **SGD + Momentum** (β=0.9, Nesterov variant available): accelerates in consistent gradient directions, dampens oscillation.
- **Adam**: adaptive per-parameter LR via first/second moment estimates + bias correction. Defaults β1=0.9, β2=0.999, ε=1e-8, LR≈3e-4. Pick as default for most non-vision tasks / fast prototyping.
- **AdamW**: decouples weight decay from gradient update (vs Adam's L2-in-gradient, which interacts badly with adaptive LR). Always prefer over Adam for Transformers.
- **RMSprop**: adaptive LR (ρ=0.99), no momentum term. Common in RL (A3C/DQN).
- **Lion**: sign-based update, discards 2nd moment (lower memory). Needs 3–10× lower LR than AdamW. Pick for memory-constrained large-model training.

### Optimizer Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| SGD (+momentum) | CNN/vision, best generalization | Need fast convergence out of box | Slower but flatter minima, better generalization |
| Adam | Fast prototyping, non-Transformer tasks | Transformer training (use AdamW) | Fast convergence, weaker generalization than SGD |
| AdamW | Transformers/LLMs (default) | — | Correct decoupled weight decay, standard for LLMs |
| RMSprop | RL | Supervised deep learning generally | No momentum, adaptive LR only |
| Lion | Memory-constrained large-scale training | Need well-understood/tuned defaults | Less memory (no 2nd moment), needs much lower LR |

### LR Scheduling

- **Warmup + Cosine Annealing**: standard for Transformers — linear warmup then cosine decay to ~0. Prevents early instability from large LR on unwarmed Adam moments.
- **Step Decay**: drop LR by factor at fixed epochs — simple, less smooth than cosine.
- **One-Cycle LR**: warmup → peak → anneal below initial, 3 phases; fast convergence for shorter training runs.
- **ReduceLROnPlateau**: adaptive, drops LR when validation metric stalls — good when total epoch budget is unknown upfront.

### Gradient & Precision Techniques

- **Gradient Clipping**: clip-by-norm (preferred, preserves direction) vs clip-by-value (can distort direction). τ=1.0 typical for Transformers, τ=5.0 for RNNs.
- **Gradient Accumulation**: simulate larger batch size on limited memory by summing gradients over N micro-batches before stepping. Caveat: BatchNorm statistics still computed on micro-batch size, not accumulated batch.
- **Mixed Precision (AMP)**: fp16/bf16 forward+backward, fp32 master weights. fp16 needs GradScaler (narrow dynamic range, underflow risk); bf16 doesn't (same exponent range as fp32, no scaler needed). ~1.5× memory savings, faster on Ampere+/Hopper.

---

## Normalization

- **BatchNorm**: normalizes per-feature across the batch; μ_B, σ_B² + learned γ/β; different train/eval behavior (running stats at eval). Cons: breaks at batch size <8; unsuitable for variable-length sequences (padding pollutes stats). Pick for CNNs with large batch sizes.
- **LayerNorm**: normalizes per-sample across features. Batch-size independent. Pick over BatchNorm for Transformers/variable-length sequences/small batches. Standard in BERT/GPT/T5/LLaMA (pre-RMSNorm era).
- **RMSNorm**: like LayerNorm but skips mean-subtraction, only rescales by RMS + γ. Cheaper, empirically equal quality. Pick over LayerNorm for modern LLMs (LLaMA/Mistral/Gemma) — saves compute at scale with no quality loss.
- **Instance Norm**: per-channel, per-sample (no batch, no cross-channel). Pick for style transfer (+ AdaIN for style injection).
- **Group Norm**: groups of channels, per-sample. G=1 → LayerNorm; G=C → InstanceNorm; G=32 typical. Pick over BatchNorm for detection/segmentation with small per-GPU batch sizes.
- **Weight Normalization**: reparameterizes `w = g·v/‖v‖`. Used in WaveNet/flow models — decouples direction and magnitude of weight learning.
- **Spectral Normalization**: constrains largest singular value to 1 (via power iteration). Used in GAN discriminators to enforce Lipschitz constraint and stabilize adversarial training.
- **Pre-Norm vs Post-Norm**: Pre-Norm (`x + Sublayer(Norm(x))`) gives more stable gradients at high depth, needs a final LayerNorm; Post-Norm (`Norm(x + Sublayer(x))`) can perform slightly better at shallow depth but is less stable to train deep. Modern LLMs use Pre-Norm almost universally.

### Normalization Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| BatchNorm | CNNs, large batch (≥16-32) | Small batch, variable-length seq, RNN/Transformer | Great regularization, breaks at small batch |
| LayerNorm | Transformers, small/variable batch | Very large-scale LLM (RMSNorm cheaper) | Batch-independent, slightly more compute than RMSNorm |
| RMSNorm | Modern large LLMs | Need re-centering behavior | Cheapest, no quality loss vs LayerNorm at scale |
| GroupNorm | Detection/segmentation, small batch | Have large batch (BatchNorm simpler) | Batch-independent, tunable group granularity |
| InstanceNorm | Style transfer | Classification (loses discriminative batch stats) | Removes instance-specific contrast/style info |
| Spectral Norm | GAN discriminator stability | Non-adversarial training | Enforces Lipschitz constraint, adds power-iteration cost |

---

## Regularization

- **L2 / Weight Decay**: `λ/2·Σw²` penalty (or decoupled in AdamW). Shrinks weights smoothly, no sparsity. Default regularizer for most models.
- **L1**: `λΣ|w|`, gradient is `sign(w)` — drives weights to exact zero → feature selection / sparsity. Pick over L2 when you want sparse/interpretable weights.
- **Dropout**: Bernoulli-mask activations at rate p, inverted-dropout scaling `1/(1-p)` at train time. Ensemble-of-subnetworks interpretation. p=0.5 typical dense, 0.1-0.3 Transformers. Weaker on CNNs (spatially correlated activations) — DropBlock is better there.
- **Early Stopping**: stop training when validation metric stops improving (patience-based). Cheapest regularizer — always usable.
- **Data Augmentation**: flip/crop/jitter/rotation — increases effective dataset diversity. Must preserve label semantics (don't flip digit "6"→"9").
- **MixUp**: `λ~Beta(α,α)` interpolates both inputs and labels. Smooths decision boundary. Cons: produces unrealistic blended images.
- **CutMix**: pastes a real rectangular patch (not blended) — preserves spatial structure/realistic textures; often outperforms MixUp for vision.
- **DropPath / Stochastic Depth**: drops entire residual branches at random per training step; drop probability increases with depth. Used in ViT/EfficientNet/Swin. Pick over standard Dropout in very deep residual nets.
- **DropBlock**: drops contiguous B×B spatial blocks (not individual activations) — matches CNN's spatially-correlated feature maps. Pick over Dropout for CNN regularization specifically (used in EfficientNet/ResNet/AmoebaNet).
- **Label Smoothing**: soft targets `(1-ε)·onehot + ε/K`, ε=0.1 typical. Improves calibration, can slightly cost accuracy on well-separated tasks.
- **Spectral Normalization**: (cross-ref Normalization) — regularizes via Lipschitz constraint, mainly for GAN stability.

### Regularization Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| L2/weight decay | Default, always-on | Want sparse weights | Smooth shrinkage, no sparsity |
| L1 | Feature selection, sparse models | Need smooth optimization landscape | Sparsity but non-differentiable at 0 |
| Dropout | Dense/FC layers, Transformers (low p) | CNN feature maps (use DropBlock) | Ensemble effect but disrupts spatial correlation in CNNs |
| DropBlock | CNN regularization | Non-conv architectures | Matches spatial structure vs plain dropout |
| DropPath/Stochastic Depth | Very deep residual nets (ViT/EfficientNet) | Shallow nets | Regularizes depth, not width |
| MixUp | General vision robustness | Need realistic-looking augmented images | Smooths boundary but unrealistic blends |
| CutMix | Vision, often better than MixUp | Patch may misalign semantics | Realistic patches, but label/content mismatch risk |
| Label Smoothing | Calibration, large-scale classification | Task needs max sharp confidence | Better calibration, small accuracy cost |
| Early Stopping | Any model, free to use | — | Cheapest regularizer, needs validation set |

---

## Weight Initialization

- **Zero init**: never for weights — breaks symmetry (all neurons identical forever). Fine for biases.
- **Naive random (fixed σ)**: doesn't account for layer width — activations vanish or explode depending on width.
- **Xavier/Glorot**: `Var(W) = 2/(n_in+n_out)`. Designed for linear-ish activations (tanh, sigmoid). Pick for tanh/sigmoid networks.
- **He/Kaiming**: `Var(W) = 2/n_in`. Compensates for ReLU zeroing ~half of inputs. Pick over Xavier whenever using ReLU/Leaky ReLU — Xavier+ReLU causes ~32× activation shrinkage over 10 layers.
- **Orthogonal init**: `WᵀW=I`, preserves gradient norm exactly through repeated multiplication. Pick for RNN/LSTM hidden-to-hidden weights specifically.
- **GPT residual scaling**: scale each residual branch's output projection by `1/√(2L)` — prevents residual-stream variance from growing linearly with depth (critical at 96+ layers).
- **LSTM forget-gate bias = 1**: biases the LSTM to remember at start of training (sigmoid(1)≈0.73), avoiding immediate forgetting.

### Initialization Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Xavier | Tanh/sigmoid networks | ReLU networks (use He) | Assumes linear activation |
| He/Kaiming | ReLU/Leaky ReLU networks (PyTorch default) | Tanh/sigmoid nets | Compensates ReLU's variance halving |
| Orthogonal | RNN/LSTM recurrent weights | Feedforward layers | Exact norm preservation, only linear-map guarantee |
| GPT residual scaling (1/√2L) | Deep Transformers (12+ layers) | Shallow nets | Prevents variance blow-up with depth |

---

## Loss Functions

- **BCE (with logits)**: binary classification; always use `BCEWithLogitsLoss` for numerical stability over manual sigmoid+log.
- **Multiclass Cross-Entropy**: mutually-exclusive classes; PyTorch's `CrossEntropyLoss` expects raw logits, not softmax outputs.
- **Weighted CE**: reweight by inverse class frequency for class imbalance — fixes frequency but not per-example difficulty.
- **Focal Loss**: `-α(1-p_t)^γ log(p_t)` — down-weights easy examples regardless of class; γ=2 typical. Pick over weighted CE when imbalance is extreme (e.g. 99% background in detection).
- **Label Smoothing**: soft targets to fight overconfidence/improve calibration (cross-ref Regularization).
- **MSE**: negative log-likelihood of Gaussian noise; quadratic penalty — outlier-sensitive.
- **MAE**: linear penalty, robust to outliers, but constant gradient causes oscillation near convergence.
- **Huber**: MSE near zero, MAE for large errors — smooth hybrid, default regression choice when outliers exist but aren't ignorable.
- **Triplet Loss**: `max(0, d(a,p) - d(a,n) + margin)` for metric learning; needs hard/semi-hard negative mining or gradient vanishes as triplets become "easy."
- **Contrastive/InfoNCE (SimCLR)**: self-supervised, treats other batch items as negatives; needs large batch for enough negatives.
- **KL Divergence**: asymmetric distributional distance; used in VAE (regularize posterior to prior), distillation (match teacher), RLHF (stay close to reference policy). Undefined if Q=0 where P>0.

### Loss Function Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| BCE | Binary/multi-label classification | Mutually-exclusive multiclass (use CE) | Use logits version for stability |
| Cross-Entropy | Multiclass exclusive classification | Multi-label (use per-class BCE) | MLE-grounded, needs raw logits |
| Focal Loss | Extreme class imbalance (detection) | Balanced classes (added tuning cost, γ) | Fixes imbalance + difficulty, adds hyperparameter |
| MSE | Regression, Gaussian noise, no outliers | Outlier-heavy data | Quadratic penalty amplifies outliers |
| MAE | Outlier-robust regression | Need fast final-stage convergence | Robust but oscillates near optimum |
| Huber | Regression with some outliers (default choice) | Don't want to tune δ | Best of both, one extra hyperparameter |
| Triplet | Metric learning / face-embedding retrieval | Simple classification tasks | Needs negative mining strategy |
| InfoNCE/Contrastive | Self-supervised representation learning | Small batch sizes | Needs large batch or memory bank |

---

## Compression / Quantization / Pruning

### Quantization

- **FP16**: 2× memory reduction, ~zero accuracy loss. Always start here.
- **INT8 PTQ**: 4× reduction, <1% accuracy drop with good calibration (128-1024 representative samples). Pick for production serving without retraining budget.
- **INT4 (GPTQ/AWQ)**: 8× reduction. Pick for LLMs on constrained hardware (fits 70B on single consumer GPU, ~35GB).
  - **GPTQ**: Hessian-based, column-by-column compensation — expensive to apply (~1h for 70B) but recovers quality well at 4-bit.
  - **AWQ**: scales salient weight channels (those multiplying large activations) before quantizing — faster to apply (~20min for 70B), no Hessian inversion needed.
- **QAT (Quantization-Aware Training)**: simulates quantization in forward pass with straight-through estimator for gradients. Pick over PTQ only when PTQ accuracy is unacceptable — costs a full fine-tune.
- **Per-tensor vs per-channel vs per-group (g=128) quantization**: per-group is the sweet spot (used by GPTQ/AWQ/QLoRA) — balances accuracy and overhead; per-tensor is cheapest but least accurate for LLM weights (outlier channels ruin the shared scale).
- **QLoRA**: 4-bit NF4-quantized frozen base + BF16 LoRA adapters trained on top. Enables fine-tuning 70B on a single 48GB GPU (~42GB total). Gradients flow only through adapters, not the quantized base.

### Pruning

- **Unstructured pruning**: zero out individual low-magnitude weights. Cons: no real speedup on standard hardware without sparse tensor cores (e.g. Ampere 2:4 sparsity).
- **Structured pruning**: remove entire neurons/heads/filters/layers — genuinely smaller dense matrices, hardware-agnostic speedup. Pick over unstructured whenever you don't control inference hardware.
- **Magnitude pruning / Iterative Magnitude Pruning (IMP)**: remove smallest-|w| weights, optionally reset survivors to original init ("lottery ticket") and retrain.
- **Lottery Ticket Hypothesis**: large nets contain small "winning ticket" subnetworks that, reset to original init, match full-network accuracy alone. Mostly a theoretical insight — impractical to search for at LLM scale.
- **Movement Pruning**: prune by `w · ∂L/∂w` (movement away from zero) rather than raw magnitude — better suited to fine-tuning regimes.

### Knowledge Distillation

- **Standard (teacher-student)**: student mimics teacher's soft output distribution (temperature-scaled) + hard labels. Teacher's "dark knowledge" (relative class similarities) transfers information hard labels can't. Fails if student lacks capacity for the task regardless of training.
- **Intermediate/FitNets/Attention Transfer**: match hidden features / attention maps, not just final logits — gives structural blueprint. Risk: mismatched layer pairing introduces misleading constraints.
- **Self-Distillation (Born-Again Networks)**: no teacher needed — model's own soft predictions become targets; acts like adaptive label smoothing.

### Other Compression

- **Low-Rank Factorization / SVD compression**: `W ≈ AB`, rank r ≪ min(m,n); params fall from `mn` to `r(m+n)`. Choosing r is an accuracy/compression tradeoff with no sharp cutoff.
- **LoRA**: same idea applied to fine-tuning — freeze `W`, learn low-rank `ΔW=AB`, merge at inference for zero extra cost.
- **Speculative Decoding**: small draft model proposes k tokens, large model verifies all k in one parallel forward pass; 2-3× throughput speedup. Fails if draft/target distributions diverge (low acceptance rate).
- **Prune → Distill → Quantize pipeline**: standard production order — pruning first (remove capacity), distillation fills remaining capacity, quantization last (smallest impact once focused).

### Compression Comparison Table

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| FP16 | Always-on baseline | Need >2× compression | Free 2× reduction, ~no quality loss |
| INT8 PTQ | Production serving, no retrain budget | Need max compression (use INT4) | 4× reduction, <1% loss with calibration |
| INT4 GPTQ/AWQ | LLMs on constrained GPUs | Need best possible quality | 8× reduction, ~0.5-1% loss |
| QAT | PTQ quality insufficient, have training budget | Quick deployment needed | Best low-bit quality, costs full fine-tune |
| Structured pruning | Any hardware, want real speedup | Need max compression ratio (unstructured tighter on paper) | Hardware-agnostic but less aggressive |
| Unstructured pruning | Sparse-tensor-core hardware only | Standard GPU/CPU inference | High compression on paper, no speedup without special kernels |
| Distillation | Smaller deployable architecture | Task needs teacher-level capacity in tiny model | Transfers "dark knowledge," bounded by student capacity |
| LoRA/QLoRA | Fine-tuning under memory constraints | Need full-precision fine-tune quality | Near-free adapters, small quantization/rank noise |
| Speculative decoding | LLM inference throughput | Draft/target model families differ | 2-3× speedup only if acceptance rate is high |

---

## Attention & Transformer Internals (quick reference)

- **Self-attention**: `softmax(QKᵀ/√d_k)V`; √d_k scaling prevents softmax saturation at large d_k.
- **MHA vs MQA vs GQA**: MHA = full K/V per head (best quality, largest KV cache); MQA = single shared K/V across heads (smallest cache, quality loss); GQA = grouped K/V (middle ground — used in LLaMA 3). Pick GQA over MHA for large-scale serving when KV-cache memory is the bottleneck.
- **RoPE vs ALiBi vs Sinusoidal vs Learned positional encoding**: RoPE (relative, rotation-based, extendable via NTK/YaRN) is the modern default (LLaMA/Mistral); ALiBi (linear distance penalty, no params) extrapolates well to longer contexts cheaply; Sinusoidal (fixed, no params) is the original but less used now; Learned embeddings cap out at max trained sequence length.
- **Flash Attention**: tiling + online softmax to avoid materializing the full N×N attention matrix in HBM; O(n) memory, 2-4× speedup, no approximation (exact attention).
- **KV Cache**: stores past K/V to avoid recomputation during autoregressive decode; scales linearly with sequence length and is the main inference memory bottleneck for long contexts — motivates MQA/GQA.

---

## Distributed Training / Parallelism (quick reference)

- **Data Parallelism**: replicate model, shard data, all-reduce gradients. Pick when model fits on one GPU.
- **Tensor Parallelism**: shard individual weight matrices across GPUs (splits matmuls). Pick when a single layer doesn't fit in one GPU's memory.
- **Pipeline Parallelism**: shard layers across GPUs, pipeline micro-batches through stages. Pick when the whole model doesn't fit but individual layers do.
- **ZeRO / FSDP**: shard optimizer states/gradients/parameters across data-parallel workers — removes the redundancy of full replication. Pick to fit larger models without full tensor/pipeline parallelism complexity.
