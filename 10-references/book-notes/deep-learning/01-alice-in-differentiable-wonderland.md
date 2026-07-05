---
module: References
topic: Book Notes
subtopic: Deep Learning Alice In Differentiable Wonderland
status: unread
tags: [references, ml, book-notes-deep-learning]
---
# Alice in Differentiable Wonderland

## Chapter 1: Scaling Laws

**The problem the book is addressing**
Why do bigger models consistently outperform smaller ones? Practitioners kept observing that throwing more data or compute at a problem helped, but without a principled explanation — leading to ad-hoc engineering decisions about when to stop scaling.

**The core insight**
Model loss follows a power law in the number of parameters, training tokens, and compute budget. These relationships are smooth and predictable, meaning you can extrapolate from small training runs to large ones before committing expensive resources.

**The mechanics**
- Loss ~ N^(-α) where N is parameters, α ≈ 0.05–0.1 depending on architecture
- Optimal allocation: given a fixed compute budget C, split it between model size N and tokens D such that N ~ D (Chinchilla scaling: train smaller models longer rather than huge models on few tokens)
- Foundation models exploit this: pretrain once at scale, fine-tune cheaply for tasks

**What the book gets right / what to watch out for**
Scaling laws hold for autoregressive language models trained on diverse text. They do not automatically transfer to narrow domains, small datasets, or non-transformer architectures. The power law extrapolation breaks near phase transitions (emergent capabilities). Compute-optimal training per Chinchilla assumes a specific loss metric — downstream task performance can diverge.

---

## Chapter 2: Mathematical Foundations

### Vectors, Matrices, and Tensors

**The problem the book is addressing**
Neural networks operate on multidimensional arrays. Without a precise vocabulary for these objects and their operations, architecture descriptions become ambiguous and implementation errors are hard to debug.

**The core insight**
Everything in deep learning is a tensor — a generalized n-dimensional array. All forward passes are sequences of tensor operations (linear maps, element-wise functions, reductions), and all of these are differentiable.

**The mechanics**
- Scalar = 0-d tensor, vector = 1-d, matrix = 2-d, higher orders for batched data
- Matrix multiply: (a×b) @ (b×c) → (a×c); cost O(abc)
- Hadamard (element-wise) product: same shape required; used in gating mechanisms
- Reduction ops: sum, mean, max along axes — change shape by collapsing dimensions

**What the book gets right / what to watch out for**
Treating shapes as the primary debugging tool is correct. Broadcasting rules (NumPy/PyTorch) silently expand dimensions and are a common source of shape bugs — always verify shapes explicitly during development.

---

### Gradients, Jacobians, and the Chain Rule

**The problem the book is addressing**
To train a model you need to know how a change in each parameter affects the loss. For scalar functions this is a gradient; for vector-valued functions it's a Jacobian. Getting this wrong means broken training.

**The core insight**
The chain rule composes local derivatives: d(f∘g)/dx = f'(g(x)) · g'(x). Neural networks are just long chains of compositions, so backpropagation is repeated chain-rule application.

**The mechanics**
- Gradient ∇L ∈ ℝⁿ points in the direction of steepest ascent of L
- Jacobian J ∈ ℝ^(m×n) for f: ℝⁿ → ℝᵐ; entry J_{ij} = ∂fᵢ/∂xⱼ
- Directional derivative: ∇L · v gives rate of change in direction v
- In practice: you rarely construct the full Jacobian — vector-Jacobian products (VJPs) are computed instead

**What the book gets right / what to watch out for**
The Jacobian framing is theoretically clean but impractical to materialize for large networks. Modern autodiff frameworks compute VJPs (reverse mode) or JVPs (forward mode) without forming J explicitly. The distinction matters when choosing between forward and reverse mode (see Chapter 6).

---

## Chapter 3: Learning as Optimization

### Empirical Risk Minimization

**The problem the book is addressing**
We want a model that generalizes to unseen data, but we only have a finite training set. How do we formalize "learn from data" in a way that connects to optimization?

**The core insight**
ERM replaces the true (unobservable) risk with the average loss over training examples. Minimizing empirical risk is the operational definition of training.

**The mechanics**
- True risk: R(f) = E_{(x,y)~P}[L(f(x), y)]
- Empirical risk: R̂(f) = (1/n) Σ L(f(xᵢ), yᵢ)
- Training = argmin_θ R̂(f_θ)
- Generalization gap: R(f) - R̂(f); controlled by model complexity and dataset size

**What the book gets right / what to watch out for**
ERM works well when training distribution ≈ deployment distribution. It fails silently under distribution shift — the model minimizes empirical risk on training data but that risk may not reflect the deployment environment. Regularization adds a penalty to the empirical risk to discourage overfitting.

---

### Bayesian Perspective on Learning

**The problem the book is addressing**
ERM gives a point estimate of parameters. But what if you want uncertainty estimates, or a principled way to encode prior beliefs about what good parameters look like?

**The core insight**
Maximum a posteriori (MAP) estimation is ERM plus regularization: the prior P(θ) becomes a regularizer, and the posterior P(θ|data) ∝ P(data|θ)P(θ) is the object being maximized.

**The mechanics**
- Gaussian prior on θ → L2 (weight decay) regularization
- Laplace prior on θ → L1 regularization (sparse solutions)
- Full Bayesian: integrate out θ for predictions — intractable for large nets, approximated via variational inference or MCMC

**What the book gets right / what to watch out for**
The MAP=regularized-ERM equivalence is a useful design principle. Full Bayesian deep learning is computationally expensive and rarely used in production. For calibration benefits without full Bayes, use ensembles or Monte Carlo dropout as cheaper approximations.

---

## Chapter 4: Linear Models

### Linear Regression

**The problem the book is addressing**
How do you fit a model to predict a continuous quantity? Linear regression is the baseline every other model is judged against, and understanding it analytically clarifies what gradient descent is doing.

**The core insight**
For MSE loss, the optimal linear model has a closed-form solution (normal equations). This reveals the geometry of least-squares fitting and explains why gradient descent converges to the same solution iteratively.

**The mechanics**
- Model: ŷ = Wx + b
- MSE loss: L = (1/n)||Xw - y||²
- Normal equations: w* = (XᵀX)⁻¹Xᵀy (exact, O(d³) cost)
- Gradient: ∇_w L = (2/n)Xᵀ(Xw - y) → use SGD for large n

**What the book gets right / what to watch out for**
Normal equations are exact but impractical for large d (invert a d×d matrix). When XᵀX is near-singular (collinear features), the solution is numerically unstable — add ridge penalty (L2). For large-scale problems, gradient descent is always used instead.

---

### Logistic Regression and Softmax

**The problem the book is addressing**
Linear regression outputs unconstrained real numbers — unsuitable for classification where you need probabilities in [0,1]. You need a model that outputs calibrated class probabilities.

**The core insight**
Logistic regression is a linear model with a sigmoid output — it models log-odds linearly. Softmax generalizes this to K classes. Cross-entropy loss is the correct loss: it's the negative log-likelihood of the data under the model.

**The mechanics**
- Binary: p(y=1|x) = σ(wᵀx + b) where σ(z) = 1/(1+e^(-z))
- Multiclass: p(yₖ|x) = softmax(Wx+b)ₖ = exp(zₖ)/Σⱼexp(zⱼ)
- Loss: L = -Σᵢ log p(yᵢ|xᵢ) (cross-entropy)
- Gradient of cross-entropy + softmax simplifies to: ∂L/∂z = p̂ - y (one-hot)

**What the book gets right / what to watch out for**
The clean gradient formula (prediction minus label) is one reason cross-entropy is preferred over MSE for classification. Softmax is numerically unstable for large logits — always use the log-sum-exp trick or fused implementations. Logistic regression is a strong baseline: beat it before adding complexity.

---

## Chapter 5: MLPs and Nonlinearity

### Universal Approximation and Activation Functions

**The problem the book is addressing**
A composition of linear layers is still linear — adding more layers without nonlinearity doesn't increase representational power. Why do nonlinear activations solve this, and what can MLPs theoretically represent?

**The core insight**
A single hidden layer MLP with a nonlinear activation and enough neurons can approximate any continuous function on a compact domain (Universal Approximation Theorem). In practice, depth is more parameter-efficient than width.

**The mechanics**
- Layer: h = activation(Wx + b)
- ReLU: max(0, z) — dead neurons when z ≤ 0 (dying ReLU problem)
- GELU: z·Φ(z) — smoother, preferred in transformers
- Sigmoid/tanh: saturate for large |z|, causing vanishing gradients in deep nets
- Stack L layers: f(x) = W_L · activation(W_{L-1}...activation(W_1 x + b_1)...)

**What the book gets right / what to watch out for**
UAT proves existence, not learnability — you can approximate any function but training might not find it. ReLU is a good default for MLPs. Dying ReLU (neurons stuck at 0) is a real issue with bad initialization or high learning rates — use He initialization and learning rate warmup.

---

### Stochastic Gradient Descent

**The problem the book is addressing**
Computing the exact gradient over the entire dataset is O(n) per step — infeasible for large n. You need a way to make progress with cheap gradient estimates.

**The core insight**
A gradient computed on a random mini-batch is a noisy but unbiased estimate of the true gradient. SGD follows these noisy estimates and still converges, with the noise actually helping escape local minima.

**The mechanics**
- Full gradient: g = (1/n) Σᵢ ∇L(θ; xᵢ, yᵢ)
- Mini-batch gradient: ĝ = (1/B) Σᵢ∈B ∇L(θ; xᵢ, yᵢ) — unbiased estimate
- Update: θ ← θ - η·ĝ
- Learning rate η controls step size; too large → diverge, too small → slow convergence
- Momentum: accumulate gradient history to smooth updates and accelerate convergence

**What the book gets right / what to watch out for**
SGD with momentum is still competitive with Adam for many tasks, especially vision. Adam converges faster but can generalize worse (sharp minima). Batch size affects gradient noise: large batches reduce noise but can hurt generalization — the "linear scaling rule" (scale η with B) partially compensates but breaks down beyond ~8k batch size.

---

## Chapter 6: Automatic Differentiation

### Forward Mode vs. Reverse Mode

**The problem the book is addressing**
Neural networks have millions of parameters but a scalar loss. Computing gradients manually is error-prone and unscalable. How do modern frameworks compute exact gradients efficiently?

**The core insight**
Reverse-mode autodiff (backpropagation) computes ∂L/∂θ for all parameters in a single backward pass — cost proportional to one forward pass, regardless of parameter count. Forward mode is efficient when outputs >> inputs, but for ML (many inputs, scalar output) reverse mode wins.

**The mechanics**
- Forward mode: propagate dual numbers (x + εẋ); gives directional derivative Jv in one pass; O(1) per input direction
- Reverse mode: store computation graph during forward pass; propagate adjoints backward; O(1) per output
- VJP: v·J computed in O(forward pass cost) in reverse mode
- Autograd in PyTorch: each tensor op creates a node in the computation graph; `.backward()` traverses it in reverse

**What the book gets right / what to watch out for**
Reverse mode is correct for deep learning. The constant-factor overhead is typically 2–3× the forward pass cost. Memory is the bottleneck: storing activations for the backward pass is O(depth × batch × hidden_size). Gradient checkpointing (recompute activations instead of storing) trades memory for compute.

---

## Chapter 7: Convolutional Neural Networks

### Translation Equivariance and Locality

**The problem the book is addressing**
Images have spatial structure: a cat is a cat whether it's in the top-left or bottom-right of the image. A fully connected layer treats every pixel independently and requires O(H×W×D) parameters — quadratic in image size. How do you build in spatial structure efficiently?

**The core insight**
Convolutions share weights across spatial positions — the same kernel detects the same feature everywhere in the image (translation equivariance). This reduces parameters from O(H×W×D) to O(K×K×D) per filter and encodes the inductive bias that nearby pixels are more related than distant ones.

**The mechanics**
- Convolution: (W * x)[i,j] = Σ_{k,l} W[k,l] · x[i+k, j+l]
- Kernel size K × K; filters F; output channels = F
- Translation equivariance: conv(shift(x)) = shift(conv(x))
- Pooling: max or average over spatial neighborhoods — reduces spatial resolution, builds translation invariance
- Receptive field: the region of input that influences a single output unit; grows with depth and dilation

**What the book gets right / what to watch out for**
CNNs are the right inductive bias for spatially correlated data (images, audio spectrograms). They underperform transformers on tasks that require long-range dependencies. Max pooling is lossy — for dense prediction (segmentation), avoid heavy spatial downsampling or use skip connections.

---

### Max Pooling and Receptive Fields

**The problem the book is addressing**
As you stack convolutional layers, each output unit only "sees" a limited input region. How do you efficiently build up a large enough receptive field to capture global structure while keeping computation manageable?

**The core insight**
Receptive field grows multiplicatively with depth and pooling strides. Max pooling achieves two things: it reduces spatial resolution (computation) and selects the most active feature in each region (invariance to small translations).

**The mechanics**
- After L conv layers with kernel size K: receptive field ≈ 1 + L(K-1)
- Stride-2 pooling: halves spatial dimensions → doubles effective receptive field growth per layer
- Dilated convolutions: insert gaps between kernel weights with dilation d; receptive field = K + (K-1)(d-1) without downsampling

**What the book gets right / what to watch out for**
Pooling is effective but discards spatial precision. For tasks requiring pixel-level outputs (segmentation, pose estimation), use U-Net style skip connections to recover spatial information after pooling.

---

## Chapter 8: 1D Convolutions and Causal Models

**The problem the book is addressing**
Sequences (audio, time series, text) have temporal structure analogous to spatial structure in images. How do you apply the convolution insight to sequences while respecting causality (no access to future inputs)?

**The core insight**
1D convolution slides a kernel over a sequence dimension — same weight-sharing and locality benefits as 2D. Causal convolutions mask future positions (use only left padding), enabling autoregressive generation.

**The mechanics**
- 1D conv: (W * x)[t] = Σ_k W[k] · x[t-k]
- Causal padding: pad K-1 zeros on the left only — output at t depends only on x[0..t]
- Dilated causal convolutions: achieve exponentially growing receptive field while maintaining causality (WaveNet architecture)
- Depth-wise separable convolutions: factorize K×K×C into K×K×1 followed by 1×1×C; reduces FLOPs by factor C

**What the book gets right / what to watch out for**
1D CNNs are fast and parallelizable — unlike RNNs, all timesteps compute simultaneously. The tradeoff: receptive field is fixed at training time. For very long sequences, dilated convolutions with small dilation factors may still miss long-range dependencies that attention handles naturally.

---

## Chapter 9: Regularization and Normalization

### Batch Normalization

**The problem the book is addressing**
As networks get deeper, the distribution of activations at each layer shifts during training (internal covariate shift). This forces downstream layers to constantly adapt, slowing convergence and requiring careful initialization.

**The core insight**
Normalize activations within each mini-batch to zero mean, unit variance, then learn a scale (γ) and shift (β). This stabilizes the distribution of inputs to each layer, enabling higher learning rates and reducing sensitivity to initialization.

**The mechanics**
- μ_B = (1/B) Σ xᵢ; σ²_B = (1/B) Σ (xᵢ - μ_B)²
- x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
- yᵢ = γ x̂ᵢ + β (learned scale and shift)
- At inference: use running statistics computed during training (not mini-batch stats)

**What the book gets right / what to watch out for**
BatchNorm dramatically improves training stability and speed for CNNs. It introduces a train/inference discrepancy — running stats must be maintained carefully. For small batches (B < 8) or sequence models, use LayerNorm instead. BatchNorm also acts as regularizer (noise from batch statistics) — combining with dropout can be counterproductive.

---

### Dropout

**The problem the book is addressing**
Large neural networks memorize training data — they have enough capacity to fit noise. How do you prevent this without reducing model size?

**The core insight**
Randomly zero out each neuron with probability p during training. This prevents co-adaptation: neurons can't rely on specific other neurons being active. At inference, scale activations by (1-p) to match expected activation magnitude.

**The mechanics**
- Training: mask each activation with Bernoulli(1-p); scale by 1/(1-p) (inverted dropout)
- Inference: no dropout; activations unchanged
- Typical rates: p=0.5 for FC layers, p=0.1 for convolutional layers
- Variational dropout, MC dropout: keep dropout active at inference for uncertainty estimation

**What the book gets right / what to watch out for**
Dropout is effective for FC layers but less so for convolutions (spatial correlations make dropout less effective — use DropBlock instead). Dropout + BatchNorm interact poorly: BN statistics become corrupted when dropout changes the effective batch distribution. In transformers, dropout is applied after attention and FFN layers.

---

### Residual Connections

**The problem the book is addressing**
Very deep networks (>20 layers) suffer from vanishing/exploding gradients and degradation — adding layers actually hurts performance even on training data. How do you train arbitrarily deep networks?

**The core insight**
Skip connections allow gradients to flow directly from output to earlier layers, bypassing intermediate transformations. The network learns residuals (corrections) rather than full transformations, which is easier to optimize near the identity.

**The mechanics**
- ResNet block: output = F(x) + x (when same dimensions)
- When dimensions differ: output = F(x) + W_s·x (1×1 conv projection)
- Gradient highway: ∂L/∂x = ∂L/∂output · (∂F/∂x + I) — identity term prevents vanishing
- Enables training 100–1000+ layer networks

**What the book gets right / what to watch out for**
Residual connections are now near-universal. The identity shortcut works best when the residual is small (near-zero initialization of the last layer in each block — "zero-init residuals"). Residuals create an implicit ensemble of shallow networks. Pre-activation ResNets (BN-ReLU before weight layers) often outperform original post-activation design.

---

## Chapter 10: Self-Attention and Transformers

### Self-Attention Mechanism

**The problem the book is addressing**
CNNs and RNNs have limited receptive fields or sequential dependencies. How do you build a model where every position can directly attend to every other position, regardless of distance?

**The core insight**
Self-attention computes, for each position, a weighted sum of all positions in the sequence. The weights are determined by the compatibility (dot product) between query and key vectors derived from the input. This gives O(1) path length between any two positions.

**The mechanics**
- Queries, keys, values: Q = XW_Q, K = XW_K, V = XW_V
- Attention: A = softmax(QKᵀ / √d_k) · V
- Scale by √d_k to prevent softmax saturation when d_k is large
- Causal mask: set upper triangle of QKᵀ to -∞ before softmax for autoregressive models

**What the book gets right / what to watch out for**
Self-attention's O(n²) complexity in sequence length is the main limitation for long sequences. Efficient attention variants (FlashAttention, sliding window) address this. The √d_k scaling is critical — without it, dot products grow with d_k and softmax becomes near-one-hot, blocking gradient flow.

---

### Multi-Head Attention

**The problem the book is addressing**
A single attention head focuses on one type of relationship at a time. How do you allow the model to simultaneously attend to different types of relationships (syntactic, semantic, positional)?

**The core insight**
Run H independent attention heads with different learned projections, then concatenate outputs. Each head can specialize in a different aspect of the input.

**The mechanics**
- head_h = Attention(XW_Q^h, XW_K^h, XW_V^h) for h=1..H
- MultiHead(X) = Concat(head_1, ..., head_H) W_O
- Typically: d_k = d_model / H (each head works in a lower-dimensional subspace)
- Total parameters ≈ same as single large attention head

**What the book gets right / what to watch out for**
Multi-head attention is one of the most robust architectural components. In practice, many heads learn redundant patterns — heads can be pruned without major performance loss. The value projection W_O is essential: it mixes information across heads.

---

### Positional Encoding

**The problem the book is addressing**
Attention is permutation-equivariant — it treats all positions equally. But sequence order matters. How do you inject positional information without breaking the parallelism that makes transformers efficient?

**The core insight**
Add a position-dependent signal to token embeddings before the first layer. This can be learned or fixed (sinusoidal). The key property: the encoding must allow the model to infer both absolute and relative positions.

**The mechanics**
- Sinusoidal: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d))
- Learned absolute: one embedding per position — doesn't generalize beyond training length
- RoPE (Rotary Position Embedding): rotate Q, K vectors by angle proportional to position; dot product Q·K depends only on relative position — better length generalization
- ALiBi: subtract linear bias from attention logits proportional to distance

**What the book gets right / what to watch out for**
Sinusoidal PE doesn't generalize well beyond training sequence length. RoPE has become the standard for modern LLMs (LLaMA, Mistral) because of better length extrapolation. ALiBi is simpler and also extrapolates well. Learned absolute PE is still used in BERT-style models where input length is fixed.

---

## Chapter 11: Encoder-Decoder and Cross-Attention

### Encoder-Decoder Architecture

**The problem the book is addressing**
Tasks like translation, summarization, and speech recognition require mapping a source sequence to a target sequence of different length. How do you condition a generative model on an input sequence?

**The core insight**
An encoder processes the source into a set of contextualized representations (one per position). A decoder generates the target autoregressively, attending to encoder outputs at each step via cross-attention.

**The mechanics**
- Encoder: bidirectional self-attention + FFN; produces context vectors C = {c₁,...,cₙ}
- Decoder: causal self-attention + cross-attention over encoder outputs + FFN
- Cross-attention: Q from decoder, K and V from encoder
- Generation: sample or beam-search from decoder until end-of-sequence token

**What the book gets right / what to watch out for**
Encoder-decoder is correct for seq2seq. Modern LLMs use decoder-only (GPT-style) because encoder-only and encoder-decoder require knowing the full input before generating — decoder-only is more flexible for open-ended generation. BERT-style encoder-only models remain best for classification and retrieval.

---

### Vision Transformers (ViT)

**The problem the book is addressing**
Transformers excel at sequence tasks but images are 2D grids, not sequences. How do you apply self-attention to images without losing spatial structure and without the quadratic cost becoming prohibitive?

**The core insight**
Divide the image into fixed-size patches, flatten each patch into a vector, and treat patches as tokens. The transformer then operates on a sequence of patch embeddings.

**The mechanics**
- Patch size P×P; image H×W; number of tokens = (H/P)×(W/P)
- Each patch → linear projection to d-dimensional embedding
- Prepend [CLS] token; add 2D or 1D positional embeddings
- Fine-tune on downstream tasks by attaching head to [CLS] token

**What the book gets right / what to watch out for**
ViT requires large pretraining datasets (ImageNet-21k or JFT) to match CNNs — it lacks the locality inductive bias of convolutions. DeiT uses distillation to train ViT on ImageNet without massive data. Hybrid models (Conv stem + transformer body) combine both inductive biases effectively.

---

### Audio Transformers (Wav2Vec, Whisper)

**The problem the book is addressing**
Raw audio is a 1D signal sampled at 16–44kHz — too long for standard transformers. Speech recognition requires both acoustic and linguistic understanding. How do you handle the multi-scale nature of audio?

**The core insight**
Use a CNN front-end to extract local acoustic features and reduce sequence length, then apply transformer self-attention over the resulting shorter sequence. Wav2Vec 2.0 adds self-supervised pretraining via contrastive loss on masked latent representations.

**The mechanics**
- Wav2Vec 2.0: CNN → quantized latents; mask portions; contrastive loss between true quantized latent and distractors
- Whisper: large-scale supervised training on 680k hours; log-mel spectrogram → CNN → encoder-decoder transformer
- Multitask: transcription, translation, language ID, VAD in one model via task tokens

**What the book gets right / what to watch out for**
Self-supervised pretraining for audio (Wav2Vec) dramatically reduces labeled data requirements. Whisper demonstrates that scale + diverse supervised data can also work. The tradeoff: Whisper is large and slow; faster/smaller alternatives (Distil-Whisper) trade accuracy for speed.

---

## Chapter 12: Graph Neural Networks

### Graph Convolutional Networks (GCNs)

**The problem the book is addressing**
Molecules, social networks, and knowledge graphs are naturally graph-structured. Flattening them into fixed vectors destroys relational structure. How do you build neural networks that operate on graphs?

**The core insight**
Replace spatial convolution with neighborhood aggregation: each node's representation is updated by aggregating features from its neighbors, then applying a learnable transformation. This is equivalent to a convolution on irregular grids.

**The mechanics**
- GCN layer: H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
- Where Ã = A + I (add self-loops), D̃ = degree matrix of Ã
- Normalization by degree prevents high-degree nodes from dominating
- Stack L layers → each node aggregates from L-hop neighborhood

**What the book gets right / what to watch out for**
GCNs are effective for homophily graphs (connected nodes have similar labels). They underperform on heterophily graphs where different-label nodes connect. Oversmoothing: stacking many GCN layers causes all node representations to converge to the same value — limit to 2–3 layers or use residual connections.

---

### Graph Attention Networks (GATs)

**The problem the book is addressing**
GCNs weight all neighbors equally (up to degree normalization). But in real graphs some neighbors are more relevant than others. How do you learn which neighbors to attend to?

**The core insight**
Replace fixed aggregation weights with learned attention coefficients, computed from the features of both the source and target node. This is self-attention applied to graphs.

**The mechanics**
- Attention: α_{ij} = softmax_j(LeakyReLU(aᵀ[W·hᵢ || W·hⱼ]))
- Aggregation: h'ᵢ = σ(Σⱼ α_{ij} W hⱼ)
- Multi-head: run K heads, concatenate or average outputs

**What the book gets right / what to watch out for**
GATs improve over GCNs on heterophily graphs. The attention mechanism is computed per-pair, making it O(|E|) — scales to large sparse graphs but not dense ones. For molecular property prediction, GCNs/GATs are often outperformed by message-passing networks that encode edge features explicitly.

---

### Message Passing Neural Networks (MPNNs)

**The problem the book is addressing**
Both GCNs and GATs are limited to node features. Chemistry and physics problems require modeling edge features (bond type, distances). How do you build a general framework for graph learning?

**The core insight**
Separate the computation into message functions (how edges communicate between nodes) and update functions (how nodes aggregate received messages). Edge features participate in the message function.

**The mechanics**
- Message: m_{ij} = M(hᵢ, hⱼ, e_{ij}) where e_{ij} is edge feature
- Aggregate: a_i = AGG({m_{ij} : j ∈ N(i)}) — sum, mean, or max
- Update: h'ᵢ = U(hᵢ, aᵢ)
- Readout: graph-level prediction via global pooling R({hᵢ})

**What the book gets right / what to watch out for**
MPNNs are the most general and expressive framework. They cannot distinguish certain non-isomorphic graphs (bounded by 1-WL test). Higher-order networks or random features are needed to go beyond 1-WL expressiveness.

---

## Chapter 13: Sequential Models

### RNNs and the Vanishing Gradient Problem

**The problem the book is addressing**
Language, time series, and audio are sequential — each element depends on what came before. Fixed-size windows miss long-range dependencies. Recurrent networks maintain a hidden state, but training them via backpropagation through time (BPTT) fails for long sequences.

**The core insight**
Gradients flow through the Jacobian of the hidden state transition at every timestep. If the spectral radius of this Jacobian < 1, gradients vanish exponentially; if > 1, they explode. Either way, learning long-range dependencies is impossible.

**The mechanics**
- RNN: hₜ = tanh(W_hh · hₜ₋₁ + W_xh · xₜ + b)
- BPTT: unroll T steps, chain-rule through all state transitions
- Gradient at step t: ∂L/∂h₀ = ∏ₜ (∂hₜ/∂hₜ₋₁) · ...
- Gradient clipping: if ||g|| > threshold, scale g down — prevents explosions, doesn't fix vanishing

**What the book gets right / what to watch out for**
Gradient clipping is necessary but only fixes explosions, not vanishing. Truncated BPTT (unroll only K steps) reduces cost but further limits effective context. For most sequence tasks today, transformers or SSMs are preferred over vanilla RNNs.

---

### LSTMs and GRUs

**The problem the book is addressing**
The vanishing gradient problem prevents RNNs from learning dependencies beyond ~10 steps. You need a mechanism that allows gradients to flow across hundreds of timesteps without attenuation.

**The core insight**
A cell state with additive updates (not multiplicative) allows gradients to flow unchanged. Gates (forget, input, output) are learned functions that decide what information to keep, add, or suppress at each step.

**The mechanics**
- LSTM: cell state cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ
  - Forget gate: fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f)
  - Input gate: iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i); candidate: g̃ₜ = tanh(...)
  - Output gate: oₜ = σ(W_o[hₜ₋₁, xₜ] + b_o); hₜ = oₜ ⊙ tanh(cₜ)
- GRU: simpler, merges cell and hidden state; uses reset and update gates
- Gradient through cell state: ∂cₜ/∂cₜ₋₁ = fₜ — when fₜ ≈ 1, gradient flows unattenuated

**What the book gets right / what to watch out for**
LSTMs effectively handle sequences up to ~500 steps. Beyond that, transformers dominate. GRUs are faster with similar performance. Modern LSTMs with improved initialization (chrono initialization) close much of the gap with transformers on certain tasks.

---

### State Space Models (Mamba, RWKV)

**The problem the book is addressing**
Transformers are O(n²) in sequence length — prohibitive for very long sequences (genomics, long documents). RNNs are O(n) but can't parallelize during training. Can you get both?

**The core insight**
Structured state space models (SSMs) can be computed either recurrently (for O(n) inference) or convolutionally (for parallel O(n log n) training). Mamba adds selective state spaces — the SSM parameters depend on the input, recovering expressivity close to attention.

**The mechanics**
- SSM: h'(t) = Ah(t) + Bx(t); y(t) = Ch(t)
- Discretize A, B for sequences: h_t = Ā h_{t-1} + B̄ x_t
- Parallel scan: compute all h_t simultaneously in O(n log n) using parallel prefix sum
- Mamba selective: Δ, B, C are functions of input x_t → model can selectively "focus" on relevant tokens
- RWKV: reformulates attention as an RNN for O(n) inference; uses time-mixing and channel-mixing blocks

**What the book gets right / what to watch out for**
SSMs are promising for long sequences but not yet at transformer quality on standard NLP benchmarks at scale. Mamba is the strongest current SSM. Training efficiency is better than transformers for very long contexts, but the ecosystem (tooling, fine-tuning recipes) is less mature.

---

## Appendix A: Probability Fundamentals

**The problem the book is addressing**
Probability is the language of uncertainty — every loss function, every sampling strategy, every Bayesian argument requires it. Without this grounding, key ML concepts remain unmotivated.

**The core insight**
A probability distribution over a random variable X describes the relative likelihood of each outcome. For continuous variables, the PDF p(x) defines this, with p(x) ≥ 0 and ∫p(x)dx = 1. Expected values, variance, and KL divergence are the key summary statistics.

**The mechanics**
- Expectation: E[f(X)] = ∫ f(x)p(x)dx (continuous) or Σ f(x)p(x) (discrete)
- Variance: Var(X) = E[(X-μ)²] = E[X²] - E[X]²
- KL divergence: KL(P||Q) = ∫ p(x) log(p(x)/q(x))dx — non-symmetric measure of P vs Q difference
- Cross-entropy: H(P,Q) = H(P) + KL(P||Q); minimizing CE ≡ minimizing KL when P is fixed (true labels)

**What the book gets right / what to watch out for**
Understanding that cross-entropy minimization = KL minimization from true distribution motivates why CE is the right classification loss. KL divergence is not a metric (asymmetric, doesn't satisfy triangle inequality) — this matters for choosing divergence in VAEs and information theory arguments.

---

## Appendix B: Universal Approximation

**The problem the book is addressing**
Is there any theoretical guarantee that neural networks can represent the functions we care about? Without this, deep learning is empirically observed to work but theoretically unjustified.

**The core insight**
MLPs with a single hidden layer of sufficient width can approximate any continuous function on a compact domain to arbitrary precision (Cybenko 1989, Hornik 1991). This guarantees that the hypothesis class is rich enough — the question is only whether optimization finds the right function.

**The mechanics**
- Formal statement: For any ε > 0 and continuous f: [0,1]ⁿ → ℝ, there exists a one-hidden-layer network N such that |N(x) - f(x)| < ε for all x
- Applies to any non-polynomial activation function (sigmoid, ReLU, etc.)
- Depth efficiency: some functions require exponentially wide shallow networks but only polynomial depth in deep networks (depth separation theorems)

**What the book gets right / what to watch out for**
UAT is an existence theorem — it says a solution exists but gives no recipe for finding it or bound on the required width. In practice, depth is more parameter-efficient than width, which is why deep networks dominate. UAT also doesn't address generalization: fitting training data ≠ generalizing to test data.

## Flashcards

**Loss ~ N^(-α) where N is parameters, α ≈ 0.05–0.1 depending on architecture?** #flashcard
Loss ~ N^(-α) where N is parameters, α ≈ 0.05–0.1 depending on architecture

**Optimal allocation?** #flashcard
given a fixed compute budget C, split it between model size N and tokens D such that N ~ D (Chinchilla scaling: train smaller models longer rather than huge models on few tokens)

**Foundation models exploit this?** #flashcard
pretrain once at scale, fine-tune cheaply for tasks

**Scalar = 0-d tensor, vector = 1-d, matrix = 2-d, higher orders for batched data?** #flashcard
Scalar = 0-d tensor, vector = 1-d, matrix = 2-d, higher orders for batched data

**Matrix multiply?** #flashcard
(a×b) @ (b×c) → (a×c); cost O(abc)

**Hadamard (element-wise) product?** #flashcard
same shape required; used in gating mechanisms

**Reduction ops: sum, mean, max along axes?** #flashcard
change shape by collapsing dimensions

**Gradient ∇L ∈ ℝⁿ points in the direction of steepest ascent of L?** #flashcard
Gradient ∇L ∈ ℝⁿ points in the direction of steepest ascent of L

**Jacobian J ∈ ℝ^(m×n) for f?** #flashcard
ℝⁿ → ℝᵐ; entry J_{ij} = ∂fᵢ/∂xⱼ

**Directional derivative?** #flashcard
∇L · v gives rate of change in direction v

**In practice: you rarely construct the full Jacobian?** #flashcard
vector-Jacobian products (VJPs) are computed instead

**True risk?** #flashcard
R(f) = E_{(x,y)~P}[L(f(x), y)]

**Empirical risk?** #flashcard
R̂(f) = (1/n) Σ L(f(xᵢ), yᵢ)

**Training = argmin_θ R̂(f_θ)?** #flashcard
Training = argmin_θ R̂(f_θ)

**Generalization gap?** #flashcard
R(f) - R̂(f); controlled by model complexity and dataset size

**Gaussian prior on θ → L2 (weight decay) regularization?** #flashcard
Gaussian prior on θ → L2 (weight decay) regularization

**Laplace prior on θ → L1 regularization (sparse solutions)?** #flashcard
Laplace prior on θ → L1 regularization (sparse solutions)

**Full Bayesian: integrate out θ for predictions?** #flashcard
intractable for large nets, approximated via variational inference or MCMC

**Model?** #flashcard
ŷ = Wx + b

**MSE loss?** #flashcard
L = (1/n)||Xw - y||²

**Normal equations?** #flashcard
w* = (XᵀX)⁻¹Xᵀy (exact, O(d³) cost)

**Gradient?** #flashcard
∇_w L = (2/n)Xᵀ(Xw - y) → use SGD for large n

**Binary?** #flashcard
p(y=1|x) = σ(wᵀx + b) where σ(z) = 1/(1+e^(-z))

**Multiclass?** #flashcard
p(yₖ|x) = softmax(Wx+b)ₖ = exp(zₖ)/Σⱼexp(zⱼ)

**Loss?** #flashcard
L = -Σᵢ log p(yᵢ|xᵢ) (cross-entropy)

**Gradient of cross-entropy + softmax simplifies to?** #flashcard
∂L/∂z = p̂ - y (one-hot)

**Layer?** #flashcard
h = activation(Wx + b)

**ReLU: max(0, z)?** #flashcard
dead neurons when z ≤ 0 (dying ReLU problem)

**GELU: z·Φ(z)?** #flashcard
smoother, preferred in transformers

**Sigmoid/tanh?** #flashcard
saturate for large |z|, causing vanishing gradients in deep nets

**Stack L layers?** #flashcard
f(x) = W_L · activation(W_{L-1}...activation(W_1 x + b_1)...)

**Full gradient?** #flashcard
g = (1/n) Σᵢ ∇L(θ; xᵢ, yᵢ)

**Mini-batch gradient: ĝ = (1/B) Σᵢ∈B ∇L(θ; xᵢ, yᵢ)?** #flashcard
unbiased estimate

**Update?** #flashcard
θ ← θ - η·ĝ

**Learning rate η controls step size; too large → diverge, too small → slow convergence?** #flashcard
Learning rate η controls step size; too large → diverge, too small → slow convergence

**Momentum?** #flashcard
accumulate gradient history to smooth updates and accelerate convergence

**Forward mode?** #flashcard
propagate dual numbers (x + εẋ); gives directional derivative Jv in one pass; O(1) per input direction

**Reverse mode?** #flashcard
store computation graph during forward pass; propagate adjoints backward; O(1) per output

**VJP?** #flashcard
v·J computed in O(forward pass cost) in reverse mode

**Autograd in PyTorch?** #flashcard
each tensor op creates a node in the computation graph; .backward() traverses it in reverse

**Convolution?** #flashcard
(W * x)[i,j] = Σ_{k,l} W[k,l] · x[i+k, j+l]

**Kernel size K × K; filters F; output channels = F?** #flashcard
Kernel size K × K; filters F; output channels = F

**Translation equivariance?** #flashcard
conv(shift(x)) = shift(conv(x))

**Pooling: max or average over spatial neighborhoods?** #flashcard
reduces spatial resolution, builds translation invariance

**Receptive field?** #flashcard
the region of input that influences a single output unit; grows with depth and dilation

**After L conv layers with kernel size K?** #flashcard
receptive field ≈ 1 + L(K-1)

**Stride-2 pooling?** #flashcard
halves spatial dimensions → doubles effective receptive field growth per layer

**Dilated convolutions?** #flashcard
insert gaps between kernel weights with dilation d; receptive field = K + (K-1)(d-1) without downsampling

**1D conv?** #flashcard
(W * x)[t] = Σ_k W[k] · x[t-k]

**Causal padding: pad K-1 zeros on the left only?** #flashcard
output at t depends only on x[0..t]

**Dilated causal convolutions?** #flashcard
achieve exponentially growing receptive field while maintaining causality (WaveNet architecture)

**Depth-wise separable convolutions?** #flashcard
factorize K×K×C into K×K×1 followed by 1×1×C; reduces FLOPs by factor C

**μ_B = (1/B) Σ xᵢ; σ²_B = (1/B) Σ (xᵢ - μ_B)²?** #flashcard
μ_B = (1/B) Σ xᵢ; σ²_B = (1/B) Σ (xᵢ - μ_B)²

**x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)?** #flashcard
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

**yᵢ = γ x̂ᵢ + β (learned scale and shift)?** #flashcard
yᵢ = γ x̂ᵢ + β (learned scale and shift)

**At inference?** #flashcard
use running statistics computed during training (not mini-batch stats)

**Training?** #flashcard
mask each activation with Bernoulli(1-p); scale by 1/(1-p) (inverted dropout)

**Inference?** #flashcard
no dropout; activations unchanged

**Typical rates?** #flashcard
p=0.5 for FC layers, p=0.1 for convolutional layers

**Variational dropout, MC dropout?** #flashcard
keep dropout active at inference for uncertainty estimation

**ResNet block?** #flashcard
output = F(x) + x (when same dimensions)

**When dimensions differ?** #flashcard
output = F(x) + W_s·x (1×1 conv projection)

**Gradient highway: ∂L/∂x = ∂L/∂output · (∂F/∂x + I)?** #flashcard
identity term prevents vanishing

**Enables training 100–1000+ layer networks?** #flashcard
Enables training 100–1000+ layer networks

**Queries, keys, values?** #flashcard
Q = XW_Q, K = XW_K, V = XW_V

**Attention?** #flashcard
A = softmax(QKᵀ / √d_k) · V

**Scale by √d_k to prevent softmax saturation when d_k is large?** #flashcard
Scale by √d_k to prevent softmax saturation when d_k is large

**Causal mask?** #flashcard
set upper triangle of QKᵀ to -∞ before softmax for autoregressive models

**head_h = Attention(XW_Q^h, XW_K^h, XW_V^h) for h=1..H?** #flashcard
head_h = Attention(XW_Q^h, XW_K^h, XW_V^h) for h=1..H

**MultiHead(X) = Concat(head_1, ..., head_H) W_O?** #flashcard
MultiHead(X) = Concat(head_1, ..., head_H) W_O

**Typically?** #flashcard
d_k = d_model / H (each head works in a lower-dimensional subspace)

**Total parameters ≈ same as single large attention head?** #flashcard
Total parameters ≈ same as single large attention head

**Sinusoidal?** #flashcard
PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d))

**Learned absolute: one embedding per position?** #flashcard
doesn't generalize beyond training length

**RoPE (Rotary Position Embedding): rotate Q, K vectors by angle proportional to position; dot product Q·K depends only on relative position?** #flashcard
better length generalization

**ALiBi?** #flashcard
subtract linear bias from attention logits proportional to distance

**Encoder?** #flashcard
bidirectional self-attention + FFN; produces context vectors C = {c₁,...,cₙ}

**Decoder?** #flashcard
causal self-attention + cross-attention over encoder outputs + FFN

**Cross-attention?** #flashcard
Q from decoder, K and V from encoder

**Generation?** #flashcard
sample or beam-search from decoder until end-of-sequence token

**Patch size P×P; image H×W; number of tokens = (H/P)×(W/P)?** #flashcard
Patch size P×P; image H×W; number of tokens = (H/P)×(W/P)

**Each patch → linear projection to d-dimensional embedding?** #flashcard
Each patch → linear projection to d-dimensional embedding

**Prepend [CLS] token; add 2D or 1D positional embeddings?** #flashcard
Prepend [CLS] token; add 2D or 1D positional embeddings

**Fine-tune on downstream tasks by attaching head to [CLS] token?** #flashcard
Fine-tune on downstream tasks by attaching head to [CLS] token

**Wav2Vec 2.0?** #flashcard
CNN → quantized latents; mask portions; contrastive loss between true quantized latent and distractors

**Whisper?** #flashcard
large-scale supervised training on 680k hours; log-mel spectrogram → CNN → encoder-decoder transformer

**Multitask?** #flashcard
transcription, translation, language ID, VAD in one model via task tokens

**GCN layer?** #flashcard
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

**Where Ã = A + I (add self-loops), D̃ = degree matrix of Ã?** #flashcard
Where Ã = A + I (add self-loops), D̃ = degree matrix of Ã

**Normalization by degree prevents high-degree nodes from dominating?** #flashcard
Normalization by degree prevents high-degree nodes from dominating

**Stack L layers → each node aggregates from L-hop neighborhood?** #flashcard
Stack L layers → each node aggregates from L-hop neighborhood

**Attention?** #flashcard
α_{ij} = softmax_j(LeakyReLU(aᵀ[W·hᵢ || W·hⱼ]))

**Aggregation?** #flashcard
h'ᵢ = σ(Σⱼ α_{ij} W hⱼ)

**Multi-head?** #flashcard
run K heads, concatenate or average outputs

**Message?** #flashcard
m_{ij} = M(hᵢ, hⱼ, e_{ij}) where e_{ij} is edge feature

**Aggregate: a_i = AGG({m_{ij} : j ∈ N(i)})?** #flashcard
sum, mean, or max

**Update?** #flashcard
h'ᵢ = U(hᵢ, aᵢ)

**Readout?** #flashcard
graph-level prediction via global pooling R({hᵢ})

**RNN?** #flashcard
hₜ = tanh(W_hh · hₜ₋₁ + W_xh · xₜ + b)

**BPTT?** #flashcard
unroll T steps, chain-rule through all state transitions

**Gradient at step t?** #flashcard
∂L/∂h₀ = ∏ₜ (∂hₜ/∂hₜ₋₁) · ...

**Gradient clipping: if ||g|| > threshold, scale g down?** #flashcard
prevents explosions, doesn't fix vanishing

**LSTM?** #flashcard
cell state cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ

**Forget gate?** #flashcard
fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f)

**Input gate?** #flashcard
iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i); candidate: g̃ₜ = tanh(...)

**Output gate?** #flashcard
oₜ = σ(W_o[hₜ₋₁, xₜ] + b_o); hₜ = oₜ ⊙ tanh(cₜ)

**GRU?** #flashcard
simpler, merges cell and hidden state; uses reset and update gates

**Gradient through cell state: ∂cₜ/∂cₜ₋₁ = fₜ?** #flashcard
when fₜ ≈ 1, gradient flows unattenuated

**SSM?** #flashcard
h'(t) = Ah(t) + Bx(t); y(t) = Ch(t)

**Discretize A, B for sequences?** #flashcard
h_t = Ā h_{t-1} + B̄ x_t

**Parallel scan?** #flashcard
compute all h_t simultaneously in O(n log n) using parallel prefix sum

**Mamba selective?** #flashcard
Δ, B, C are functions of input x_t → model can selectively "focus" on relevant tokens

**RWKV?** #flashcard
reformulates attention as an RNN for O(n) inference; uses time-mixing and channel-mixing blocks

**Expectation?** #flashcard
E[f(X)] = ∫ f(x)p(x)dx (continuous) or Σ f(x)p(x) (discrete)

**Variance?** #flashcard
Var(X) = E[(X-μ)²] = E[X²] - E[X]²

**KL divergence: KL(P||Q) = ∫ p(x) log(p(x)/q(x))dx?** #flashcard
non-symmetric measure of P vs Q difference

**Cross-entropy?** #flashcard
H(P,Q) = H(P) + KL(P||Q); minimizing CE ≡ minimizing KL when P is fixed (true labels)

**Formal statement?** #flashcard
For any ε > 0 and continuous f: [0,1]ⁿ → ℝ, there exists a one-hidden-layer network N such that |N(x) - f(x)| < ε for all x

**Applies to any non-polynomial activation function (sigmoid, ReLU, etc.)?** #flashcard
Applies to any non-polynomial activation function (sigmoid, ReLU, etc.)

**Depth efficiency?** #flashcard
some functions require exponentially wide shallow networks but only polynomial depth in deep networks (depth separation theorems)
