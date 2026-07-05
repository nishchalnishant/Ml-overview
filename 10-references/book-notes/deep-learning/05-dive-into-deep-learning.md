---
module: References
topic: Book Notes
subtopic: Deep Learning Dive Into Deep Learning
status: unread
tags: [references, ml, book-notes-deep-learning]
---
# Dive into Deep Learning

## Chapter 1: Introduction to Deep Learning

**The problem the book is addressing**
Traditional programs encode every rule explicitly. When the problem space is too complex for rules — recognizing images, translating language, playing Go — explicit programming fails. Practitioners need a principled framework for when and why to switch from rule-based to learned systems.

**The core insight**
Deep learning replaces manual feature engineering with hierarchical representation learning. Low-level patterns (edges, phonemes, character n-grams) compose into high-level concepts automatically, given enough data, compute, and the right optimization. Three factors converged to make this practical: internet-scale data, GPU acceleration, and backpropagation.

**The mechanics**
- Core components: data (raw input), model (function from input to prediction), objective function (measures error), optimization algorithm (adjusts parameters to reduce error)
- Learning paradigms: supervised (labeled input→output pairs), unsupervised (structure in input alone), reinforcement (reward signal from environment)
- Deep learning adds: automatic feature learning through stacked nonlinear transformations, gradient-based optimization over millions of parameters

**What the book gets right / what to watch out for**
The three-factor narrative (data/compute/algorithms) is accurate and useful for explaining DL's rise. The implicit message — that more data and compute always help — is true for large-scale pretraining but breaks down for small-data settings where inductive biases and domain knowledge matter more than depth.

---

## Chapter 2: Tensors, Autodiff, and Probability

**The problem the book is addressing**
Neural network code fails silently when shapes are wrong. Autodiff bugs produce zero or NaN gradients. Probability concepts (likelihoods, expectations, Bayes) underpin every loss function and evaluation metric. Without fluency in these three areas, DL code is a black box.

**The core insight**
Tensors are typed multidimensional arrays with storage and a view (shape, stride, offset) — operations that look like they copy data often create views of the same storage. Autodiff records operations on tensors and replays them in reverse to compute gradients. Probability gives a language for expressing what the model believes and how certain those beliefs are.

**The mechanics**
- Tensor creation: `torch.zeros/ones/rand/randn`, `torch.tensor([...])`, `from_numpy()`
- Shape ops: `.view(shape)`, `.reshape(shape)`, `.squeeze/unsqueeze`, `.permute(dims)`; `.contiguous()` required before `.view()` after `.permute()`
- Autodiff: `requires_grad=True` on leaf tensors; operations recorded in computation graph; `.backward()` computes all gradients via chain rule in one reverse pass; `torch.no_grad()` disables tracking for inference
- Probability: expectation E[X] = Σ x·P(X=x); variance Var[X] = E[(X-μ)²]; Bayes' theorem P(H|E) = P(E|H)P(H)/P(E); MLE = maximize log-likelihood over parameters

**What the book gets right / what to watch out for**
The stride/view explanation is the most important thing in this chapter — a common bug is `.view()` failing on non-contiguous tensors after `.permute()`. The Bayesian introduction is important context: every L2 regularization term is equivalent to a Gaussian prior on weights under MLE. Mixed dtype operations raise errors or silently downcast — always be explicit about dtype.

---

## Chapter 3: Linear Regression

**The problem the book is addressing**
Linear regression is the simplest supervised learning model. Understanding it end-to-end — loss function, analytical solution, gradient descent solution, normal equations — builds the template for every more complex model.

**The core insight**
Linear regression minimizes mean squared error: L = (1/n)Σ(ŷᵢ - yᵢ)². The normal equations give an analytical solution in O(d³) time but fail for d > 10⁴ (matrix inversion too expensive). Gradient descent iteratively moves parameters in the direction that reduces loss, scaling to any dataset size.

**The mechanics**
- Model: ŷ = Xw + b; X is (n,d), w is (d,), b is scalar
- MSE loss: L = ||Xw + b - y||² / n
- Normal equations: w* = (XᵀX)⁻¹Xᵀy — exact but O(d³)
- SGD update: w ← w - η·∇_w L; b ← b - η·∇_b L
- With PyTorch: define `nn.Linear(in, out)`, use `nn.MSELoss()`, `torch.optim.SGD`

**What the book gets right / what to watch out for**
The normal equations are important to know conceptually but never used in practice for DL — gradient descent is always preferred for scale. Linear regression assumes the relationship is linear and residuals are Gaussian — both assumptions fail in practice and warrant diagnostic checks (residual plots) before trusting predictions.

---

## Chapter 4: Softmax Regression and Classification

**The problem the book is addressing**
Binary classification extends to multiclass by stacking multiple logistic classifiers, but naively taking the max of linear outputs doesn't produce valid probabilities. You need outputs that sum to 1 and can be interpreted as a distribution over classes.

**The core insight**
Softmax converts a vector of raw scores (logits) into a probability distribution: P(y=k|x) = exp(oₖ) / Σⱼ exp(oⱼ). The natural loss function is cross-entropy: L = -log P(y_true|x). Minimizing cross-entropy is equivalent to maximum likelihood estimation of the softmax model.

**The mechanics**
- Softmax: ŷ = softmax(Xw + b); output is a (n, C) probability matrix
- Cross-entropy loss: L = -(1/n) Σᵢ log(ŷᵢ[yᵢ]); penalizes confident wrong predictions heavily
- Gradient: ∂L/∂o = ŷ - y_onehot (softmax + cross-entropy gradient is clean)
- In PyTorch: `nn.CrossEntropyLoss()` takes raw logits (not softmax output) — it combines log-softmax and NLL internally for numerical stability

**What the book gets right / what to watch out for**
Passing softmax outputs into `nn.CrossEntropyLoss()` is a common bug — the function expects raw logits and applies log-softmax internally. Doing it twice produces silent numerical errors. Accuracy is the wrong metric for imbalanced classification — cross-entropy loss and AUC-ROC are more informative during training.

---

## Chapters 4–5: Multilayer Perceptrons

**The problem the book is addressing**
Linear models can only represent linear decision boundaries. Any function that isn't a hyperplane — XOR, concentric circles, image classification — requires nonlinearity. How do you add expressiveness without exploding the parameter count?

**The core insight**
Stacking linear layers with nonlinear activations between them creates universal function approximators. A single hidden layer with enough neurons can approximate any continuous function (universal approximation theorem), but depth is more parameter-efficient than width — each layer learns a new level of abstraction.

**The mechanics**
- MLP: z₁ = σ(W₁x + b₁); z₂ = σ(W₂z₁ + b₂); ŷ = W₃z₂ + b₃
- Activation functions: sigmoid σ(z) = 1/(1+e^-z) (saturates → vanishing gradients), tanh (zero-centered, still saturates), ReLU max(0,z) (non-saturating, fast, default), Leaky ReLU max(αz,z) (fixes dying ReLU), GELU (smooth approximation, used in transformers)
- Vanishing gradients: sigmoid/tanh saturate → derivative ≈ 0 → gradients vanish in deep networks → use ReLU
- Xavier initialization: Var[w] = 2/(fan_in + fan_out) — keeps variance stable across layers; critical for training deep networks

**What the book gets right / what to watch out for**
The vanishing gradient explanation motivates the shift from sigmoid to ReLU — this is a historically important design decision. The "dying ReLU" problem (neurons stuck at 0 permanently) is real but usually not severe in practice; Leaky ReLU or ELU resolves it if needed. GELU is the correct activation for modern transformers.

---

## Chapter 5 (Dropout and Regularization)

**The problem the book is addressing**
Deep networks with millions of parameters memorize training data. The model achieves near-zero training loss but generalizes poorly. Practitioners need principled regularization that works for deep networks.

**The core insight**
Dropout randomly zeroes activations during training with probability p, forcing the network to learn redundant representations. Each forward pass uses a different subnetwork, approximating ensemble learning. At inference, activations are scaled by (1-p) to match expected values.

**The mechanics**
- Dropout: h_dropped = h ⊙ Bernoulli(1-p) / (1-p) (inverted dropout, standard in PyTorch)
- L2 weight decay: add λ||w||² to loss; equivalent to Gaussian prior on weights
- Apply only during training — `model.train()` enables dropout; `model.eval()` disables it
- Typical p=0.5 for FC layers; p=0.1-0.2 for conv layers

**What the book gets right / what to watch out for**
Forgetting to call `model.eval()` during inference is a common bug — dropout stays active and predictions become random and inconsistent across calls. In modern practice, LayerNorm + residual connections provide enough regularization that dropout is often omitted in transformer blocks.

---

## Chapter 5: Builders' Guide (nn.Module and Parameter Management)

**The problem the book is addressing**
Building complex architectures requires tracking hundreds of parameter tensors, loading/saving checkpoints, and managing device placement. Manual bookkeeping is error-prone and doesn't scale.

**The core insight**
`nn.Module` is a recursive container. Any `nn.Parameter` or sub-module assigned as an attribute is automatically registered. `model.parameters()` yields a flat iterator over all trainable tensors — exactly what optimizers need. `model.state_dict()` captures all parameters and buffers as an ordered dict for checkpointing.

**The mechanics**
- Define: `class MyModel(nn.Module): def __init__(self): super().__init__(); self.layer = nn.Linear(in, out)`
- `model.parameters()`: flat iterator for optimizer
- `model.state_dict()`: save checkpoints; `model.load_state_dict(state_dict, strict=False)`: load (partial) checkpoints
- `model.to(device)`: move all parameters to GPU/CPU
- `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`: register collections of modules

**What the book gets right / what to watch out for**
Always save `model.state_dict()`, not the model object — pickling the model embeds the class definition and breaks when code changes. `strict=False` allows loading pretrained weights into models with a different head. Using Python lists instead of `nn.ModuleList` means parameters are invisible to `.parameters()` — a silent bug.

---

## Chapter 6: LeNet and Convolutional Neural Networks

**The problem the book is addressing**
Fully connected networks applied to images require O(H×W×C×D) parameters per layer — millions for a 224×224 image. They also ignore spatial structure: nearby pixels are correlated and a cat is a cat regardless of where it appears. A parameter-efficient architecture is needed.

**The core insight**
Convolutions share weights spatially — the same filter is applied at every position, reducing parameters by orders of magnitude and encoding translation equivariance. Pooling adds invariance. LeNet (1989) demonstrated the full CNN template: conv→pool→conv→pool→flatten→FC.

**The mechanics**
- Conv layer: output[i,j] = Σ_{k,l} filter[k,l] × input[i+k, j+l] + bias
- Hyperparameters: kernel size (3×3 standard), stride (default 1), padding ('same' preserves spatial size), number of filters
- Max pooling: takes max over 2×2 window, halves spatial dimensions
- LeNet: Conv(1,6,5)→AvgPool→Conv(6,16,5)→AvgPool→Flatten→FC(400,120)→FC(120,84)→FC(84,10)

**What the book gets right / what to watch out for**
LeNet is the right starting point for understanding CNNs — all modern architectures (ResNet, ViT) are elaborations of this template. Average pooling (LeNet) has been superseded by max pooling (AlexNet onwards) and global average pooling (ResNet). The 3×3 filter is now essentially universal — large kernels (5×5, 7×7) are only used in the first layer for efficiency.

---

## Chapter 7: Modern CNNs (AlexNet, VGG, NiN, GoogLeNet, BatchNorm, ResNet, DenseNet)

**The problem the book is addressing**
LeNet-style CNNs stall at ImageNet scale — millions of images, 1000 classes, high-resolution inputs. The challenge is: how do you go deeper without vanishing gradients, parameter explosion, and the compute cost of large filters?

**The core insight**
AlexNet (2012) won ImageNet by stacking ReLUs, dropout, and multi-GPU training. VGG replaced large filters with stacked 3×3 convolutions (same receptive field, fewer parameters, more nonlinearities). ResNet (2015) broke the depth barrier with skip connections: the layer learns a residual F(x) = H(x)-x rather than H(x) directly, so identity is the trivial solution and gradients flow directly through skip connections.

**The mechanics**
- AlexNet: 5 conv layers + 3 FC; ReLU activations; dropout 0.5 in FC; data augmentation
- VGG block: [Conv(3×3, same padding) → ReLU] × 2-3 → MaxPool; 16 layers total
- NiN (Network in Network): 1×1 convolutions as pointwise FC layers — reduces parameters, adds nonlinearity per-channel
- GoogLeNet/Inception: parallel branches with 1×1, 3×3, 5×5 convolutions and max pooling; concatenate outputs — captures multi-scale features
- BatchNorm: normalize each mini-batch to zero mean unit variance per channel; trainable γ, β; stabilizes training, allows higher LR; place after linear/conv, before activation
- ResNet block: y = F(x, {W}) + x; if dimensions mismatch, use 1×1 conv shortcut
- DenseNet: each layer connected to all subsequent layers; x_l = H_l([x_0, x_1, ..., x_{l-1}]); maximizes gradient flow, parameter-efficient

**What the book gets right / what to watch out for**
ResNet's residual connection is one of the most important innovations in DL — it enables training networks 100+ layers deep by solving the degradation problem (deeper networks performing worse than shallower ones without skip connections). BatchNorm is essential but has training/inference behavior discrepancy — always call `model.train()/model.eval()` correctly. DenseNet's dense connectivity improves gradient flow but increases memory usage quadratically with depth — use with care.

---

## Chapters 8–9: RNNs, LSTMs, GRUs, seq2seq, Beam Search

**The problem the book is addressing**
Sequences (text, time series, audio) have variable length and temporal dependencies. A feedforward network processes each timestep independently and requires fixed input size. The key challenge: how do you maintain context across arbitrarily long sequences?

**The core insight**
RNNs maintain a hidden state hₜ that is updated at each step, carrying forward information from earlier in the sequence. LSTMs add a cell state cₜ with explicit gates (forget, input, output) that control what information to retain, add, or expose — enabling long-range memory. GRUs simplify LSTMs to two gates with similar performance.

**The mechanics**
- RNN: hₜ = tanh(Wₓhxₜ + Wₕₕhₜ₋₁ + b); output: ŷₜ = Whyₕₜ
- LSTM: fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf); iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi); cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁,xₜ]); oₜ = σ(Wo·[hₜ₋₁,xₜ]+bo); hₜ = oₜ⊙tanh(cₜ)
- GRU: rₜ (reset gate), zₜ (update gate); hₜ = zₜ⊙hₜ₋₁ + (1-zₜ)⊙tanh(W·[rₜ⊙hₜ₋₁,xₜ])
- Backprop through time (BPTT): unroll T steps, apply chain rule — gradients vanish/explode for large T
- Gradient clipping: if ||g|| > threshold, g ← g × threshold/||g|| — prevents gradient explosion
- Language model perplexity: exp(cross-entropy loss) — measures how surprised the model is by test text; lower is better
- seq2seq: encoder RNN produces context vector from input; decoder RNN generates output token by token
- Beam search: keep top-k hypotheses at each decoding step; k=1 is greedy, k=∞ is exhaustive; k=4-10 is typical

**What the book gets right / what to watch out for**
BPTT explanation is correct and motivates the need for LSTMs. In practice, for most sequence tasks today, transformers outperform LSTMs — they parallelize training and handle long-range dependencies better. RNNs remain relevant for streaming/real-time applications where full-sequence attention is too slow. Gradient clipping is essential for RNN training — without it, occasional large gradients destabilize the hidden state.

---

## Chapters 10–11: Attention Mechanisms and Transformers

**The problem the book is addressing**
RNNs compress the entire input sequence into a fixed-size context vector — the bottleneck forces the encoder to summarize arbitrarily long sequences into a vector of constant dimension. Information is lost, particularly for long sequences. Additionally, RNNs cannot parallelize across timesteps.

**The core insight**
Attention lets the decoder query the encoder's hidden states directly at each output step, instead of reading from a single compressed context. Self-attention generalizes this: every position can attend to every other position simultaneously, enabling full parallelism and direct long-range dependencies. Multi-head attention runs several attention operations in parallel, capturing different types of relationships.

**The mechanics**
- Bahdanau attention: score(hₜ, hs) = vᵀ tanh(W₁hₜ + W₂hs); αₜₛ = softmax(scores); context cₜ = Σs αₜₛ hs
- Scaled dot-product attention: A = softmax(QKᵀ/√d_k)·V; scaling prevents softmax saturation in high dimensions
- Multi-head attention: h heads with d_k = d_model/h; concatenate outputs and project with W_O
- Causal mask (GPT): set upper-triangle of QKᵀ to -∞ before softmax — prevents attending to future tokens
- Positional encoding: since attention is permutation-invariant, inject position information; sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d)); learned positional embeddings (GPT) are equally common
- Transformer encoder block: LayerNorm → MultiHeadAttention → residual + LayerNorm → FFN → residual
- Pre-LayerNorm (GPT-style): LayerNorm before each sublayer — more training-stable than original Post-LN
- FFN: Linear(d_model, 4×d_model) → GELU → Linear(4×d_model, d_model) — 4× expansion, independent per-token

**What the book gets right / what to watch out for**
The comparison of self-attention vs CNNs vs RNNs (Table in d2l) is one of the clearest explanations of why transformers win: O(1) path length between any two positions (vs O(n) for RNNs, O(log n) for CNNs). In production, use `F.scaled_dot_product_attention` (PyTorch 2.0) or FlashAttention — naive O(n²) attention OOMs for sequences beyond 2k tokens. Pre-LayerNorm is now standard and should be preferred over the original Post-LN.

---

## Chapter 11: Vision Transformer (ViT)

**The problem the book is addressing**
CNNs have strong inductive biases (locality, translation equivariance) that help with small data but limit flexibility. Transformers lack these biases but scale better with data and compute. Can a pure transformer architecture match or exceed CNNs on image tasks?

**The core insight**
Split the image into fixed-size patches (16×16 typically), flatten and linearly project each patch into a d-dimensional embedding, prepend a learnable [CLS] token, add positional embeddings, and pass through standard transformer encoder blocks. At sufficient scale (JFT-300M), ViT exceeds CNN performance.

**The mechanics**
- Patch embedding: divide H×W image into (H/P)×(W/P) patches of size P×P; flatten to P²×C; project to d_model via linear layer
- Add learned positional embedding to each patch token
- [CLS] token: prepended; its final hidden state used for classification
- Standard transformer encoder blocks follow
- ViT-Base: d_model=768, 12 heads, 12 layers; ViT-Large: d_model=1024, 16 heads, 24 layers

**What the book gets right / what to watch out for**
ViT requires more data than ResNet to match performance — the CNNs' translation equivariance provides free regularization that ViT must learn. DeiT (Data-efficient Image Transformers) closes this gap with distillation from a CNN teacher. For most practical vision tasks with limited data, a pretrained ResNet/EfficientNet still outperforms training ViT from scratch.

---

## Chapter 11: BERT, T5, and Pretraining Strategies

**The problem the book is addressing**
Supervised NLP datasets for specific tasks are expensive to label and often small. Most text on the internet is unlabeled. How do you leverage unlabeled text to build representations that transfer to downstream tasks?

**The core insight**
Self-supervised pretraining on unlabeled text yields representations that transfer to downstream tasks with minimal labeled data. BERT (encoder-only) uses masked language modeling (predict randomly masked tokens) and next sentence prediction. GPT (decoder-only) uses causal language modeling (predict next token). T5 (encoder-decoder) frames all tasks as text-to-text generation.

**The mechanics**
- BERT masked LM: randomly mask 15% of tokens; predict original tokens from context (bidirectional)
- NSP: predict whether sentence B follows sentence A (less useful in practice, dropped in RoBERTa)
- Fine-tuning BERT: add task-specific head on [CLS] token; fine-tune all parameters on labeled examples
- GPT: autoregressive LM; left-to-right context only; fine-tune by prompting or adding classification head on last token
- T5: both input and target are text; unifies summarization, translation, QA into one format

**What the book gets right / what to watch out for**
The BERT/GPT contrast (bidirectional vs unidirectional) is important: BERT is better for understanding tasks (classification, NLI, NER); GPT is better for generation tasks (text completion, summarization, QA via prompting). NSP was found to be unhelpful — RoBERTa removes it. In production, use pretrained models from HuggingFace — pretraining from scratch requires 100B+ tokens.

---

## Chapter 12: Optimization Algorithms

**The problem the book is addressing**
The loss landscape of deep networks is non-convex, high-dimensional, and full of saddle points and local minima. Vanilla gradient descent oscillates, stalls, or diverges depending on learning rate. Practitioners need a systematic understanding of which optimizer to use and why.

**The core insight**
SGD with momentum accumulates velocity in consistent gradient directions, smoothing oscillations. Adam adds per-parameter adaptive learning rates — parameters with infrequent large gradients get larger updates. Neither is universally best: Adam converges faster but often to sharper minima; SGD+momentum+cosine annealing often achieves better final generalization for vision tasks.

**The mechanics**
- Batch GD: gradient over full dataset — exact but O(n) per step; impractical
- Mini-batch SGD: gradient over B examples (B=32–256 typical); parallelizable; noise acts as regularization
- Momentum: vₜ = γvₜ₋₁ + η∇L; θ ← θ - vₜ; γ=0.9 typical
- Adagrad: gₜ accumulates squared gradients; η_i = η/√(Gᵢᵢ+ε) — adapts per parameter; aggressive decay for frequent features
- RMSProp: Eₜ = ρEₜ₋₁ + (1-ρ)gₜ²; η_i = η/√(Eₜ+ε) — fixes Adagrad's monotonically decreasing LR
- Adam: first moment mₜ = β₁mₜ₋₁ + (1-β₁)gₜ; second moment vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²; bias-corrected: m̂ₜ, v̂ₜ; θ ← θ - η·m̂ₜ/(√v̂ₜ+ε); defaults: β₁=0.9, β₂=0.999, ε=1e-8
- AdamW: decouple weight decay from adaptive gradient update — use for modern LLM training
- LR scheduling: step decay, cosine annealing (η follows cosine curve), warm restarts; linear warmup for the first 2% of training steps prevents instability

**What the book gets right / what to watch out for**
The saddle point discussion is important — deep networks rarely get stuck in local minima (the landscape is benign for overparameterized models) but saddle points are common and momentum helps escape them. AdamW (decoupled weight decay) should be preferred over Adam for most modern work. Cosine decay with warmup is now essentially standard for LLM pretraining.

---

## Chapter 13: Computational Performance

**The problem the book is addressing**
Training large models takes weeks without careful attention to hardware utilization. Most practitioners run at a fraction of theoretical GPU throughput due to memory bottlenecks, redundant computation, and poor multi-GPU coordination.

**The core insight**
Asynchronous computation (GPU executes while CPU prepares next batch), mixed-precision training (FP16/BF16 for compute, FP32 for accumulation), and ring-allreduce for multi-GPU gradient aggregation are the three levers that collectively determine whether you're fully utilizing available hardware.

**The mechanics**
- Hybrid programming: `torch.jit.script` compiles Python control flow to TorchScript for optimization; `torch.compile` (PyTorch 2.0) provides static-graph speedup while keeping dynamic semantics
- Mixed precision: `torch.autocast('cuda', dtype=torch.float16)` — forward/backward in FP16; optimizer step in FP32; BF16 preferred for training stability (same range as FP32, less precision)
- Data parallelism: replicate model on each GPU; each GPU processes different data shard; ring-allreduce synchronizes gradients without a central parameter server
- `torch.nn.parallel.DistributedDataParallel` (DDP): the standard multi-GPU training wrapper; scales linearly with number of GPUs

**What the book gets right / what to watch out for**
DDP via ring-allreduce is the correct modern approach — the parameter server alternative is a bandwidth bottleneck. `torch.compile` (PyTorch 2.0) provides 30–50% speedup on many models with a single line change. BF16 is strictly better than FP16 for training stability (same exponent range as FP32) and should be the default on A100/H100 GPUs.

---

## Chapter 14: Computer Vision — Augmentation and Fine-Tuning

**The problem the book is addressing**
Vision models trained on small annotated datasets overfit. Collecting more labeled images is expensive and slow. Additionally, training from random initialization for a specific task ignores the fact that ImageNet-pretrained weights already encode useful visual features.

**The core insight**
Data augmentation applies label-preserving transformations to expand effective dataset size. Transfer learning starts from a pretrained backbone and fine-tunes either the head only (fast, less data needed) or the entire network (slower, better performance). The pretrained backbone provides a strong initialization that converges faster and generalizes better.

**The mechanics**
- Geometric augmentation: random horizontal/vertical flip, rotation (±30°), random crop and resize
- Intensity augmentation: color jitter (brightness ±20%, contrast ±20%, saturation, hue), Gaussian blur, grayscale
- Modern augmentations: MixUp (blend two images linearly, blend labels accordingly), CutMix (paste rectangular crop from one image into another)
- Apply augmentation only during training — never on validation/test
- Transfer learning: load ImageNet weights; replace final FC layer for new task; freeze backbone initially; optionally unfreeze and fine-tune at lower LR (1/10th of head LR)

**What the book gets right / what to watch out for**
Transfer learning from ImageNet weights dominates training from scratch for most practical vision tasks — even for medical imaging, natural photography features in early layers transfer well. MixUp and CutMix are underused but consistently improve generalization. For domain-specific data (X-rays, satellite imagery), domain-adaptive augmentations (realistic deformations, sensor noise models) outperform standard geometric transforms.

---

## Chapter 14: Object Detection (Bounding Boxes, Anchor Boxes, NMS, SSD, R-CNN)

**The problem the book is addressing**
Image classification outputs a single label per image. Real-world perception requires detecting multiple objects of different classes at different locations. This requires a representation for object location (bounding boxes) and a mechanism to simultaneously predict locations and classes.

**The core insight**
Anchor boxes tile the image with reference boxes of different scales and aspect ratios. The detection model predicts *offsets* from anchors (not absolute coordinates) and class probabilities for each anchor. Anchors with high IoU with a ground-truth box are positive examples; anchors far from all objects are negative. Non-maximum suppression removes redundant detections.

**The mechanics**
- Bounding box: (x_center, y_center, width, height) normalized to [0,1] relative to image size
- IoU (Intersection over Union): IoU = area(A∩B) / area(A∪B); used for anchor assignment and NMS threshold
- Anchor assignment: IoU > 0.5 → positive; IoU < 0.3 → negative; 0.3–0.5 → ignored
- NMS: sort boxes by confidence; iteratively remove boxes with IoU > threshold with higher-confidence box
- SSD (Single Shot MultiBox Detector): predict from multiple feature map scales; fast, single-stage
- R-CNN family: Region Proposal Network (RPN) generates candidate regions; separate branch classifies and refines each → more accurate but slower than single-stage

**What the book gets right / what to watch out for**
IoU-based anchor assignment and NMS are the core of every classical detection pipeline. Modern detectors (DETR, Grounding DINO) replace NMS with attention-based set prediction, eliminating hand-tuned thresholds. For practical deployment, YOLOv8 is the current best balance of speed and accuracy; R-CNN family is only needed when accuracy is paramount and latency is not a constraint.

---

## Chapter 14: Semantic Segmentation

**The problem the book is addressing**
Object detection draws boxes around objects but doesn't identify which pixels belong to each object. Medical imaging, autonomous driving, and satellite imagery require pixel-level classification (segmentation).

**The core insight**
Fully convolutional networks (FCNs) replace all FC layers with 1×1 convolutions, allowing arbitrary input sizes and producing spatial output maps. Transposed convolutions (learnable upsampling) restore spatial resolution lost during pooling. U-Net adds skip connections from encoder to decoder at each scale, preserving spatial detail that pooling discards.

**The mechanics**
- FCN: replace FC with 1×1 Conv; output is H×W×C feature map; bilinear upsample to input resolution
- Transposed convolution: learnable upsampling with stride > 1; opposite of strided convolution
- U-Net: encoder (Conv→MaxPool ×4, halves spatial, doubles channels); bottleneck; decoder (Upsample→Conv ×4); skip connections concatenate encoder features to decoder at each scale
- Dice loss: L = 1 - (2×|P∩T|) / (|P|+|T|); handles class imbalance; background pixels don't dominate the loss
- BCE + Dice combined: BCE penalizes individual pixels; Dice penalizes structural overlap

**What the book gets right / what to watch out for**
U-Net remains the standard for medical segmentation. Dice loss is critical for imbalanced segmentation — cross-entropy alone fails when the target class is tiny (nodule vs background). For 3D medical volumes, 3D U-Net extends naturally but requires significant GPU memory — patch-based training with overlap is standard.

---

## Chapters 15–16: NLP Pretraining (word2vec, GloVe, fastText, BPE, BERT)

**The problem the book is addressing**
One-hot word encodings are sparse, high-dimensional, and encode no semantic similarity — "king" and "queen" are orthogonal vectors despite their relationship. How do you build dense word representations that capture semantic and syntactic relationships?

**The core insight**
Words appearing in similar contexts have similar meanings (distributional hypothesis). word2vec (skip-gram/CBOW) learns embeddings by predicting context words from a target word or vice versa. GloVe leverages global co-occurrence statistics. fastText extends word2vec to subword n-grams, handling morphologically rich languages and OOV words. BERT contextualizes embeddings — the same word gets different vectors in different contexts.

**The mechanics**
- Skip-gram: maximize P(context|target) = softmax(uᵀ_context · v_target)
- Negative sampling: instead of softmax over full vocabulary, maximize P(positive pair) while minimizing P(K negative pairs) — reduces training cost from O(V) to O(K)
- GloVe: minimize Σᵢⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²; Xᵢⱼ is co-occurrence count
- fastText: represent word as sum of character n-gram embeddings; handles OOV and morphology
- BPE (Byte Pair Encoding): start with character vocabulary; iteratively merge most frequent adjacent pair; handles arbitrary input without OOV
- BERT: bidirectional transformer; pretrain with masked LM (predict 15% masked tokens) + NSP; fine-tune with task-specific head

**What the book gets right / what to watch out for**
The word embedding section is pedagogically important but word2vec/GloVe are obsolete for production NLP — use BERT/sentence-transformers for static embeddings. The "king-man+woman=queen" analogy holds only ~30% of the time on analogy benchmarks. BPE is the correct tokenization for LLMs; never train your own tokenizer unless pretraining on a new language/domain.

---

## Chapter 16: NLP Applications (Sentiment Analysis, NLI, BERT Fine-Tuning)

**The problem the book is addressing**
After pretraining, applying LLMs to downstream tasks requires understanding how to adapt the architecture, what data is needed, and how to avoid overfitting on small fine-tuning datasets.

**The core insight**
BERT fine-tuning inserts a task-specific head (linear layer) on top of the pretrained [CLS] token representation and fine-tunes all parameters jointly. This works because the pretrained representations already encode rich linguistic structure — the head only needs to learn the mapping to task labels.

**The mechanics**
- Sentiment analysis: [CLS] → Linear(d, 2) → softmax; fine-tune with labeled sentiment examples
- NLI (Natural Language Inference): encode [CLS, premise, SEP, hypothesis, SEP]; classify as entailment/contradiction/neutral
- RNN-based sentiment: apply Bi-LSTM to token embeddings; concatenate final forward/backward hidden states → classify
- CNN-based sentiment: 1D conv with multiple kernel sizes (2,3,4 n-grams); max-over-time pooling → classify
- Machine translation: encoder-decoder with attention; beam search decoding

**What the book gets right / what to watch out for**
The BERT fine-tuning pattern ([CLS] token → classification head) is the dominant approach for understanding tasks. For generation tasks, GPT-style models are preferred. Fine-tuning the entire BERT model on small datasets (<1000 examples) overfits — LoRA or head-only fine-tuning works better. For production NLP, use `transformers` library fine-tuning recipes rather than implementing from scratch.

---

## Chapter 17: Reinforcement Learning

**The problem the book is addressing**
Supervised learning requires labeled examples. Many sequential decision problems — game playing, robotics, ad ranking — have no labeled data; only a reward signal after taking actions. How do you learn a policy from rewards?

**The core insight**
The Markov Decision Process (MDP) formalizes sequential decision making: states S, actions A, transition probabilities P(s'|s,a), rewards R(s,a,s'). The optimal policy maximizes cumulative discounted reward. Value iteration computes exact optimal values by dynamic programming; Q-learning approximates this without a model of the environment.

**The mechanics**
- MDP: V*(s) = max_a Σs' P(s'|s,a)[R(s,a,s') + γV*(s')]; discount γ ∈ [0,1)
- Value iteration: initialize V(s) = 0; repeat V(s) ← max_a Σs' P(s'|s,a)[R + γV(s')] until convergence
- Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]; off-policy, model-free
- Deep Q-Network (DQN): parameterize Q(s,a;θ) with CNN; experience replay buffer; target network for stability

**What the book gets right / what to watch out for**
Value iteration is computationally tractable only for small state spaces — real problems have intractably large state spaces requiring function approximation (DQN). Modern RL for language models uses PPO or GRPO for RLHF — the same MDP framework but with LLM as policy and human preference scores as reward.

---

## Chapter 18: Gaussian Processes

**The problem the book is addressing**
Neural networks provide point predictions without uncertainty estimates. In settings where knowing "I don't know" matters (medical decisions, active learning, Bayesian optimization), you need a model that outputs a distribution over predictions, not just a point estimate.

**The core insight**
A Gaussian Process (GP) is a distribution over functions: any finite collection of function values follows a multivariate Gaussian. The prior is specified by a kernel (covariance function) that encodes assumptions about function smoothness. Given observations, Bayesian conditioning yields an analytic posterior — exact uncertainty estimates without sampling.

**The mechanics**
- RBF kernel: k(x,x') = σ² exp(-||x-x'||²/(2l²)); l controls smoothness, σ² controls scale
- GP inference: mean μ(x*) = K(x*,x)[K(x,x)+σ²I]⁻¹y; variance σ²(x*) = K(x*,x*) - K(x*,x)[K(x,x)+σ²I]⁻¹K(x,x*)
- Prediction: returns μ(x*) ± 2σ(x*) for 95% credible interval
- GPyTorch: scalable GP implementation with GPU support; ExactGP for exact inference; ApproximateGP for large datasets

**What the book gets right / what to watch out for**
GPs are the principled uncertainty quantification tool for small-to-medium datasets. The O(n³) matrix inversion makes exact GPs impractical beyond ~10,000 training points — use sparse GPs or neural network uncertainty methods (MC Dropout, Deep Ensembles) at scale. GPs are the theoretical foundation for Bayesian optimization — the acquisition function operates on the GP posterior.

---

## Chapter 19: Hyperparameter Optimization

**The problem the book is addressing**
Model performance depends heavily on hyperparameters (learning rate, architecture, batch size, regularization strength) that aren't learned by gradient descent. Manual tuning is slow and rarely finds the optimum. Systematic search is needed.

**The core insight**
Random search outperforms grid search by sampling each hyperparameter independently — it explores more of each dimension for the same compute budget. Bayesian optimization fits a surrogate model (GP or random forest) to the hyperparameter→metric function and uses an acquisition function to select the most promising next configuration. Multi-fidelity methods (successive halving/Hyperband) terminate unpromising runs early, allocating more budget to promising configurations.

**The mechanics**
- Grid search: O(k^d) evaluations for k values per d hyperparameters — exponential, impractical beyond d=3
- Random search: sample each hyperparameter independently; covers each dimension in O(n) evaluations
- Bayesian optimization: surrogate (GP) + acquisition (expected improvement, UCB); updates surrogate with each observation; scales to ~50 hyperparameters
- Successive halving: start n configs at minimum budget; keep top half; double budget; repeat until one remains
- Async Hyperband: runs successive halving in parallel with different random seeds; accounts for variance in early stopping

**What the book gets right / what to watch out for**
Random search with log-uniform sampling for LR is a strong baseline. Bayesian optimization requires ~20-50 function evaluations to outperform random search — useful for expensive evaluations (hours per run). For modern LLMs, Optuna or Ax are the practical libraries. Log-scale sampling for learning rate and regularization strength is critical — uniform sampling wastes most budget in the wrong regions.

---

## Chapter 20: Generative Adversarial Networks

**The problem the book is addressing**
Discriminative models learn P(y|x). Some applications — data augmentation, privacy-preserving synthetic data, image super-resolution — require generating new samples from the data distribution P(x) itself. Explicit density models (VAEs) produce blurry outputs; an adversarial approach can produce sharper samples.

**The core insight**
GANs solve the generation problem via a game: a generator G(z) maps Gaussian noise to data; a discriminator D(x) classifies real vs generated. G is trained to fool D; D is trained to distinguish them. At Nash equilibrium, G produces samples indistinguishable from real data. DCGANs stabilize this with architectural constraints: batch normalization, transposed convolutions, LeakyReLU in discriminator.

**The mechanics**
- Minimax objective: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]
- Generator loss: -E[log D(G(z))] (non-saturating; provides stronger gradient when G is weak)
- DCGAN: Generator: FC → Reshape → ConvTranspose × 4 → Tanh output; Discriminator: Conv × 4 with LeakyReLU → FC → Sigmoid
- Mode collapse: G produces few sample types that fool D; mitigated by Wasserstein loss (Earth Mover's distance), spectral normalization, progressive growing
- Neural style transfer: extract content features from content image; extract style features (Gram matrices) from style image; optimize input image to minimize combined loss

**What the book gets right / what to watch out for**
DCGANs produce good results for small tasks but training instability and mode collapse remain persistent challenges. Diffusion models (DDPM, Stable Diffusion) have largely superseded GANs for high-quality image generation — they train stably and produce better diversity. Neural style transfer is a useful application but is slow (minutes per image via optimization); fast style transfer (feedforward network) is more practical.

---

## Chapter 21: Recommender Systems

**The problem the book is addressing**
Users can't evaluate every item in a large catalog. Recommendation systems must predict which items a specific user will engage with, given sparse implicit feedback (clicks, purchases) rather than explicit ratings.

**The core insight**
Collaborative filtering exploits the fact that users with similar past behavior will have similar future preferences. Matrix factorization decomposes the user-item interaction matrix into low-rank user and item embeddings — the dot product of a user and item embedding approximates their interaction probability.

**The mechanics**
- Collaborative filtering: find similar users, recommend items they liked; or find similar items, recommend items similar to those the user liked
- Matrix factorization: minimize Σ(rᵢⱼ - uᵢᵀvⱼ)² over observed ratings; add bias terms; regularize with L2
- Implicit feedback: clicks/views don't mean positive preference — reframe as binary classification with confidence weighting
- Context-aware: incorporate user context (time, device, location) as additional features
- Content-based: use item features (genre, text description) as item embeddings; handles cold-start for new items

**What the book gets right / what to watch out for**
Matrix factorization is the correct starting model for recommendation. Pure collaborative filtering fails at cold start — new users or items have no interactions. In production, a two-stage approach (recall → ranking) scales to billions of items: ANN search over embeddings retrieves candidates; a full feature model ranks them. Modern systems (YouTube, TikTok) use transformer-based sequential models that treat a user's history as a sequence.

## Flashcards

**Core components?** #flashcard
data (raw input), model (function from input to prediction), objective function (measures error), optimization algorithm (adjusts parameters to reduce error)

**Learning paradigms?** #flashcard
supervised (labeled input→output pairs), unsupervised (structure in input alone), reinforcement (reward signal from environment)

**Deep learning adds?** #flashcard
automatic feature learning through stacked nonlinear transformations, gradient-based optimization over millions of parameters

**Tensor creation?** #flashcard
torch.zeros/ones/rand/randn, torch.tensor([...]), from_numpy()

**Shape ops?** #flashcard
.view(shape), .reshape(shape), .squeeze/unsqueeze, .permute(dims); .contiguous() required before .view() after .permute()

**Autodiff?** #flashcard
requires_grad=True on leaf tensors; operations recorded in computation graph; .backward() computes all gradients via chain rule in one reverse pass; torch.no_grad() disables tracking for inference

**Probability?** #flashcard
expectation E[X] = Σ x·P(X=x); variance Var[X] = E[(X-μ)²]; Bayes' theorem P(H|E) = P(E|H)P(H)/P(E); MLE = maximize log-likelihood over parameters

**Model?** #flashcard
ŷ = Xw + b; X is (n,d), w is (d,), b is scalar

**MSE loss?** #flashcard
L = ||Xw + b - y||² / n

**Normal equations: w* = (XᵀX)⁻¹Xᵀy?** #flashcard
exact but O(d³)

**SGD update?** #flashcard
w ← w - η·∇_w L; b ← b - η·∇_b L

**With PyTorch?** #flashcard
define nn.Linear(in, out), use nn.MSELoss(), torch.optim.SGD

**Softmax?** #flashcard
ŷ = softmax(Xw + b); output is a (n, C) probability matrix

**Cross-entropy loss?** #flashcard
L = -(1/n) Σᵢ log(ŷᵢ[yᵢ]); penalizes confident wrong predictions heavily

**Gradient?** #flashcard
∂L/∂o = ŷ - y_onehot (softmax + cross-entropy gradient is clean)

**In PyTorch: nn.CrossEntropyLoss() takes raw logits (not softmax output)?** #flashcard
it combines log-softmax and NLL internally for numerical stability

**MLP?** #flashcard
z₁ = σ(W₁x + b₁); z₂ = σ(W₂z₁ + b₂); ŷ = W₃z₂ + b₃

**Activation functions?** #flashcard
sigmoid σ(z) = 1/(1+e^-z) (saturates → vanishing gradients), tanh (zero-centered, still saturates), ReLU max(0,z) (non-saturating, fast, default), Leaky ReLU max(αz,z) (fixes dying ReLU), GELU (smooth approximation, used in transformers)

**Vanishing gradients?** #flashcard
sigmoid/tanh saturate → derivative ≈ 0 → gradients vanish in deep networks → use ReLU

**Xavier initialization: Var[w] = 2/(fan_in + fan_out)?** #flashcard
keeps variance stable across layers; critical for training deep networks

**Dropout?** #flashcard
h_dropped = h ⊙ Bernoulli(1-p) / (1-p) (inverted dropout, standard in PyTorch)

**L2 weight decay?** #flashcard
add λ||w||² to loss; equivalent to Gaussian prior on weights

**Apply only during training?** #flashcard
model.train() enables dropout; model.eval() disables it

**Typical p=0.5 for FC layers; p=0.1-0.2 for conv layers?** #flashcard
Typical p=0.5 for FC layers; p=0.1-0.2 for conv layers

**Define?** #flashcard
class MyModel(nn.Module): def __init__(self): super().__init__(); self.layer = nn.Linear(in, out)

**model.parameters()?** #flashcard
flat iterator for optimizer

**model.state_dict()?** #flashcard
save checkpoints; model.load_state_dict(state_dict, strict=False): load (partial) checkpoints

**model.to(device)?** #flashcard
move all parameters to GPU/CPU

**nn.Sequential, nn.ModuleList, nn.ModuleDict?** #flashcard
register collections of modules

**Conv layer?** #flashcard
output[i,j] = Σ_{k,l} filter[k,l] × input[i+k, j+l] + bias

**Hyperparameters?** #flashcard
kernel size (3×3 standard), stride (default 1), padding ('same' preserves spatial size), number of filters

**Max pooling?** #flashcard
takes max over 2×2 window, halves spatial dimensions

**LeNet?** #flashcard
Conv(1,6,5)→AvgPool→Conv(6,16,5)→AvgPool→Flatten→FC(400,120)→FC(120,84)→FC(84,10)

**AlexNet?** #flashcard
5 conv layers + 3 FC; ReLU activations; dropout 0.5 in FC; data augmentation

**VGG block?** #flashcard
[Conv(3×3, same padding) → ReLU] × 2-3 → MaxPool; 16 layers total

**NiN (Network in Network): 1×1 convolutions as pointwise FC layers?** #flashcard
reduces parameters, adds nonlinearity per-channel

**GoogLeNet/Inception: parallel branches with 1×1, 3×3, 5×5 convolutions and max pooling; concatenate outputs?** #flashcard
captures multi-scale features

**BatchNorm?** #flashcard
normalize each mini-batch to zero mean unit variance per channel; trainable γ, β; stabilizes training, allows higher LR; place after linear/conv, before activation

**ResNet block?** #flashcard
y = F(x, {W}) + x; if dimensions mismatch, use 1×1 conv shortcut

**DenseNet?** #flashcard
each layer connected to all subsequent layers; x_l = H_l([x_0, x_1, ..., x_{l-1}]); maximizes gradient flow, parameter-efficient

**RNN?** #flashcard
hₜ = tanh(Wₓhxₜ + Wₕₕhₜ₋₁ + b); output: ŷₜ = Whyₕₜ

**LSTM?** #flashcard
fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf); iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi); cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁,xₜ]); oₜ = σ(Wo·[hₜ₋₁,xₜ]+bo); hₜ = oₜ⊙tanh(cₜ)

**GRU?** #flashcard
rₜ (reset gate), zₜ (update gate); hₜ = zₜ⊙hₜ₋₁ + (1-zₜ)⊙tanh(W·[rₜ⊙hₜ₋₁,xₜ])

**Backprop through time (BPTT): unroll T steps, apply chain rule?** #flashcard
gradients vanish/explode for large T

**Gradient clipping: if ||g|| > threshold, g ← g × threshold/||g||?** #flashcard
prevents gradient explosion

**Language model perplexity: exp(cross-entropy loss)?** #flashcard
measures how surprised the model is by test text; lower is better

**seq2seq?** #flashcard
encoder RNN produces context vector from input; decoder RNN generates output token by token

**Beam search?** #flashcard
keep top-k hypotheses at each decoding step; k=1 is greedy, k=∞ is exhaustive; k=4-10 is typical

**Bahdanau attention?** #flashcard
score(hₜ, hs) = vᵀ tanh(W₁hₜ + W₂hs); αₜₛ = softmax(scores); context cₜ = Σs αₜₛ hs

**Scaled dot-product attention?** #flashcard
A = softmax(QKᵀ/√d_k)·V; scaling prevents softmax saturation in high dimensions

**Multi-head attention?** #flashcard
h heads with d_k = d_model/h; concatenate outputs and project with W_O

**Causal mask (GPT): set upper-triangle of QKᵀ to -∞ before softmax?** #flashcard
prevents attending to future tokens

**Positional encoding?** #flashcard
since attention is permutation-invariant, inject position information; sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d)); learned positional embeddings (GPT) are equally common

**Transformer encoder block?** #flashcard
LayerNorm → MultiHeadAttention → residual + LayerNorm → FFN → residual

**Pre-LayerNorm (GPT-style): LayerNorm before each sublayer?** #flashcard
more training-stable than original Post-LN

**FFN: Linear(d_model, 4×d_model) → GELU → Linear(4×d_model, d_model)?** #flashcard
4× expansion, independent per-token

**Patch embedding?** #flashcard
divide H×W image into (H/P)×(W/P) patches of size P×P; flatten to P²×C; project to d_model via linear layer

**Add learned positional embedding to each patch token?** #flashcard
Add learned positional embedding to each patch token

**[CLS] token?** #flashcard
prepended; its final hidden state used for classification

**Standard transformer encoder blocks follow?** #flashcard
Standard transformer encoder blocks follow

**ViT-Base?** #flashcard
d_model=768, 12 heads, 12 layers; ViT-Large: d_model=1024, 16 heads, 24 layers

**BERT masked LM?** #flashcard
randomly mask 15% of tokens; predict original tokens from context (bidirectional)

**NSP?** #flashcard
predict whether sentence B follows sentence A (less useful in practice, dropped in RoBERTa)

**Fine-tuning BERT?** #flashcard
add task-specific head on [CLS] token; fine-tune all parameters on labeled examples

**GPT?** #flashcard
autoregressive LM; left-to-right context only; fine-tune by prompting or adding classification head on last token

**T5?** #flashcard
both input and target are text; unifies summarization, translation, QA into one format

**Batch GD: gradient over full dataset?** #flashcard
exact but O(n) per step; impractical

**Mini-batch SGD?** #flashcard
gradient over B examples (B=32–256 typical); parallelizable; noise acts as regularization

**Momentum?** #flashcard
vₜ = γvₜ₋₁ + η∇L; θ ← θ - vₜ; γ=0.9 typical

**Adagrad: gₜ accumulates squared gradients; η_i = η/√(Gᵢᵢ+ε)?** #flashcard
adapts per parameter; aggressive decay for frequent features

**RMSProp: Eₜ = ρEₜ₋₁ + (1-ρ)gₜ²; η_i = η/√(Eₜ+ε)?** #flashcard
fixes Adagrad's monotonically decreasing LR

**Adam?** #flashcard
first moment mₜ = β₁mₜ₋₁ + (1-β₁)gₜ; second moment vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²; bias-corrected: m̂ₜ, v̂ₜ; θ ← θ - η·m̂ₜ/(√v̂ₜ+ε); defaults: β₁=0.9, β₂=0.999, ε=1e-8

**AdamW: decouple weight decay from adaptive gradient update?** #flashcard
use for modern LLM training

**LR scheduling?** #flashcard
step decay, cosine annealing (η follows cosine curve), warm restarts; linear warmup for the first 2% of training steps prevents instability

**Hybrid programming?** #flashcard
torch.jit.script compiles Python control flow to TorchScript for optimization; torch.compile (PyTorch 2.0) provides static-graph speedup while keeping dynamic semantics

**Mixed precision: torch.autocast('cuda', dtype=torch.float16)?** #flashcard
forward/backward in FP16; optimizer step in FP32; BF16 preferred for training stability (same range as FP32, less precision)

**Data parallelism?** #flashcard
replicate model on each GPU; each GPU processes different data shard; ring-allreduce synchronizes gradients without a central parameter server

**torch.nn.parallel.DistributedDataParallel (DDP)?** #flashcard
the standard multi-GPU training wrapper; scales linearly with number of GPUs

**Geometric augmentation?** #flashcard
random horizontal/vertical flip, rotation (±30°), random crop and resize

**Intensity augmentation?** #flashcard
color jitter (brightness ±20%, contrast ±20%, saturation, hue), Gaussian blur, grayscale

**Modern augmentations?** #flashcard
MixUp (blend two images linearly, blend labels accordingly), CutMix (paste rectangular crop from one image into another)

**Apply augmentation only during training?** #flashcard
never on validation/test

**Transfer learning?** #flashcard
load ImageNet weights; replace final FC layer for new task; freeze backbone initially; optionally unfreeze and fine-tune at lower LR (1/10th of head LR)

**Bounding box?** #flashcard
(x_center, y_center, width, height) normalized to [0,1] relative to image size

**IoU (Intersection over Union)?** #flashcard
IoU = area(A∩B) / area(A∪B); used for anchor assignment and NMS threshold

**Anchor assignment?** #flashcard
IoU > 0.5 → positive; IoU < 0.3 → negative; 0.3–0.5 → ignored

**NMS?** #flashcard
sort boxes by confidence; iteratively remove boxes with IoU > threshold with higher-confidence box

**SSD (Single Shot MultiBox Detector)?** #flashcard
predict from multiple feature map scales; fast, single-stage

**R-CNN family?** #flashcard
Region Proposal Network (RPN) generates candidate regions; separate branch classifies and refines each → more accurate but slower than single-stage

**FCN?** #flashcard
replace FC with 1×1 Conv; output is H×W×C feature map; bilinear upsample to input resolution

**Transposed convolution?** #flashcard
learnable upsampling with stride > 1; opposite of strided convolution

**U-Net?** #flashcard
encoder (Conv→MaxPool ×4, halves spatial, doubles channels); bottleneck; decoder (Upsample→Conv ×4); skip connections concatenate encoder features to decoder at each scale

**Dice loss?** #flashcard
L = 1 - (2×|P∩T|) / (|P|+|T|); handles class imbalance; background pixels don't dominate the loss

**BCE + Dice combined?** #flashcard
BCE penalizes individual pixels; Dice penalizes structural overlap

**Skip-gram?** #flashcard
maximize P(context|target) = softmax(uᵀ_context · v_target)

**Negative sampling: instead of softmax over full vocabulary, maximize P(positive pair) while minimizing P(K negative pairs)?** #flashcard
reduces training cost from O(V) to O(K)

**GloVe?** #flashcard
minimize Σᵢⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²; Xᵢⱼ is co-occurrence count

**fastText?** #flashcard
represent word as sum of character n-gram embeddings; handles OOV and morphology

**BPE (Byte Pair Encoding)?** #flashcard
start with character vocabulary; iteratively merge most frequent adjacent pair; handles arbitrary input without OOV

**BERT?** #flashcard
bidirectional transformer; pretrain with masked LM (predict 15% masked tokens) + NSP; fine-tune with task-specific head

**Sentiment analysis?** #flashcard
[CLS] → Linear(d, 2) → softmax; fine-tune with labeled sentiment examples

**NLI (Natural Language Inference)?** #flashcard
encode [CLS, premise, SEP, hypothesis, SEP]; classify as entailment/contradiction/neutral

**RNN-based sentiment?** #flashcard
apply Bi-LSTM to token embeddings; concatenate final forward/backward hidden states → classify

**CNN-based sentiment?** #flashcard
1D conv with multiple kernel sizes (2,3,4 n-grams); max-over-time pooling → classify

**Machine translation?** #flashcard
encoder-decoder with attention; beam search decoding

**MDP?** #flashcard
V(s) = max_a Σs' P(s'|s,a)[R(s,a,s') + γV(s')]; discount γ ∈ [0,1)

**Value iteration?** #flashcard
initialize V(s) = 0; repeat V(s) ← max_a Σs' P(s'|s,a)[R + γV(s')] until convergence

**Q-learning?** #flashcard
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]; off-policy, model-free

**Deep Q-Network (DQN)?** #flashcard
parameterize Q(s,a;θ) with CNN; experience replay buffer; target network for stability

**RBF kernel?** #flashcard
k(x,x') = σ² exp(-||x-x'||²/(2l²)); l controls smoothness, σ² controls scale

**GP inference?** #flashcard
mean μ(x) = K(x,x)[K(x,x)+σ²I]⁻¹y; variance σ²(x) = K(x,x) - K(x,x)[K(x,x)+σ²I]⁻¹K(x,x*)

**Prediction?** #flashcard
returns μ(x) ± 2σ(x) for 95% credible interval

**GPyTorch?** #flashcard
scalable GP implementation with GPU support; ExactGP for exact inference; ApproximateGP for large datasets

**Grid search: O(k^d) evaluations for k values per d hyperparameters?** #flashcard
exponential, impractical beyond d=3

**Random search?** #flashcard
sample each hyperparameter independently; covers each dimension in O(n) evaluations

**Bayesian optimization?** #flashcard
surrogate (GP) + acquisition (expected improvement, UCB); updates surrogate with each observation; scales to ~50 hyperparameters

**Successive halving?** #flashcard
start n configs at minimum budget; keep top half; double budget; repeat until one remains

**Async Hyperband?** #flashcard
runs successive halving in parallel with different random seeds; accounts for variance in early stopping

**Minimax objective?** #flashcard
min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

**Generator loss?** #flashcard
-E[log D(G(z))] (non-saturating; provides stronger gradient when G is weak)

**DCGAN?** #flashcard
Generator: FC → Reshape → ConvTranspose × 4 → Tanh output; Discriminator: Conv × 4 with LeakyReLU → FC → Sigmoid

**Mode collapse?** #flashcard
G produces few sample types that fool D; mitigated by Wasserstein loss (Earth Mover's distance), spectral normalization, progressive growing

**Neural style transfer?** #flashcard
extract content features from content image; extract style features (Gram matrices) from style image; optimize input image to minimize combined loss

**Collaborative filtering?** #flashcard
find similar users, recommend items they liked; or find similar items, recommend items similar to those the user liked

**Matrix factorization?** #flashcard
minimize Σ(rᵢⱼ - uᵢᵀvⱼ)² over observed ratings; add bias terms; regularize with L2

**Implicit feedback: clicks/views don't mean positive preference?** #flashcard
reframe as binary classification with confidence weighting

**Context-aware?** #flashcard
incorporate user context (time, device, location) as additional features

**Content-based?** #flashcard
use item features (genre, text description) as item embeddings; handles cold-start for new items
