---
module: Interview Prep
topic: Llm
subtopic: Dl Architectures
status: unread
tags: [interviewprep, ml, llm-dl-architectures]
---
# Deep Learning Architectures — First-Principles Interview Guide

---

## 1. CNN (Convolutional Neural Network)

### What the interviewer is actually testing
Whether you understand *why* convolutions exist — not just that they process images. The competency is: can you identify when spatial structure in data should change your architecture choice, and articulate what failure mode the alternative creates?

### The reasoning structure
Before CNNs, people applied fully-connected layers to images. A 224×224×3 image has 150,528 inputs. A single FC layer with 4,096 hidden units needs 617M parameters — for one layer. Beyond the parameter count, the deeper problem is structural: the network gets no hint that nearby pixels correlate more than distant ones, and that the same feature appearing at different positions is *the same feature*. The FC layer must learn that a horizontal edge at row 10 and a horizontal edge at row 200 are both horizontal edges — from scratch, independently, for every spatial position.

Three failure modes of FC layers on image-like data:
- **No translation invariance:** a cat in the top-left corner is a completely different input vector than a cat in the bottom-right. The model must learn a separate detector for every position.
- **No locality:** distant pixels get equal weight to adjacent ones. The correlation structure of natural images (local patches are highly correlated; distant pixels much less so) is invisible to the architecture.
- **Parameter explosion:** every input pixel independently connected to every hidden unit forces massive parameter counts before any useful computation begins.

Convolutions solve all three by imposing a prior that matches the structure of spatial data: *features are local, and the same feature can appear anywhere*.

### The pattern in action

The convolution operation:
$$(f * W)[i, j] = \sum_{m} \sum_{n} f[i+m, j+n] \cdot W[m, n]$$

A kernel $W$ slides over the image and applies at every spatial position. The same kernel detects the same pattern — a vertical edge, a corner, a texture — regardless of where it appears. This is **weight sharing**: one kernel for an entire image instead of a separate set of weights per position.

Parameter count for a conv layer with input channels $C_{in}$, output channels $C_{out}$, kernel $k \times k$:
$$\text{params} = C_{out} \times (C_{in} \times k^2 + 1)$$

Compare to replacing a 7×7×512 feature map with 4096 outputs via FC: $7 \times 7 \times 512 \times 4096 = 102M$ weights. The equivalent conv layer: $4096 \times (512 \times 9 + 1) \approx 18.8M$. Same output resolution; 5× fewer parameters; spatial structure preserved.

**Standard building blocks and why each exists:**

| Block | Operation | The failure it prevents |
| :--- | :--- | :--- |
| Conv2d | $W * x + b$ | Weight explosion + no locality |
| BatchNorm | $\frac{x-\mu}{\sigma}\gamma + \beta$ | Distributional shift between layers that throttles learning rate |
| ReLU | $\max(0, x)$ | Gradient saturation (sigmoid/tanh zero-out gradients in deep networks) |
| MaxPool | $\max$ over window | Sensitivity to sub-pixel translation — small shifts no longer change output |
| GlobalAvgPool | Mean over $H \times W$ | A large FC layer connecting the feature map to the classifier — replaced with a single vector |

**Architecture milestones and the problem each solved:**

| Model | What was broken | What fixed it |
| :--- | :--- | :--- |
| AlexNet (2012) | Networks couldn't get deep without training dying | ReLU (vs tanh) prevented gradient saturation; dropout prevented co-adaptation |
| VGG (2014) | Large kernels were assumed necessary | Two stacked 3×3 conv layers cover the same receptive field as one 5×5 but use fewer parameters and add an extra nonlinearity |
| ResNet (2015) | Deeper networks performed *worse* than shallower ones (degradation, not overfitting) | Skip connections: $y = F(x) + x$ let the network default to identity, only learning the residual |
| EfficientNet (2019) | Scaling depth, width, or resolution independently is suboptimal | Compound scaling: all three dimensions simultaneously, at a fixed ratio derived from a neural architecture search |

### Common traps

**Trap: confusing translation equivariance with translation invariance.**
Convolutions are *equivariant*: shift the input → the output shifts by the same amount. MaxPooling introduces *invariance*: small shifts in input produce the same output. Interviewers ask this directly. Equivariance preserves location information; invariance discards it. For object detection, you want equivariance (you need to know *where* the object is). For classification, you want invariance (a cat anywhere is still a cat).

**Trap: not knowing why ResNet skip connections work mechanically.**
The common answer is "prevents vanishing gradients." The precise mechanism: $y = F(x) + x$ makes the identity mapping the default behavior at initialization. The gradient of the loss with respect to the input of a residual block is $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}(\frac{\partial F}{\partial x} + 1)$. Even if $F$'s gradient shrinks to near zero, the signal $1$ always passes through unimpeded. The network only needs to learn the *deviation* from identity — a much easier optimization problem.

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)   # gradient highway: +residual ensures grad ≥ 1
```

---

## 2. RNN / LSTM / GRU

### What the interviewer is actually testing
Whether you understand the *gradient flow problem* that motivated gated architectures — not just the equations. The competency: given a sequence of 500 tokens, what happens to the gradient of the loss with respect to the hidden state at step 1, and how do gates structurally prevent the problem?

### The reasoning structure

The natural approach to sequences: maintain a hidden state that compresses what happened so far and update it at each step. The vanilla RNN:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

What breaks: the gradient of the loss at step $T$ with respect to $h_1$ involves the product of Jacobians:
$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_{hh} \cdot \text{diag}(\tanh'(z_t))$$

If the spectral radius of $W_{hh}$ is less than 1, this product shrinks exponentially in $T$. If it exceeds 1, it explodes. For sequences of length 100–500, gradients either vanish (early words contribute nothing) or explode (training diverges). This isn't a theoretical concern — vanilla RNNs were empirically broken on any task requiring memory beyond ~10 steps.

### The pattern in action

**LSTM:** introduces a cell state $c_t$ — a gradient highway that sidesteps the multiplicative problem. The key insight: $c_t$ is updated *additively*, not by matrix multiplication.

$$c_t = \underbrace{f_t}_{\text{forget}} \odot c_{t-1} + \underbrace{i_t}_{\text{input}} \odot \tilde{c}_t$$

The gradient through $c_t$ back to $c_{t-1}$ is just $f_t$ (a vector of values in $(0,1)$ controlled by learned gates) — not the product of weight matrices. Gradients flow through time without exponential attenuation.

Full LSTM equations:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(what fraction of the cell state to forget)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(how much of the new candidate to write)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(the new candidate)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(what to expose from cell state)}$$
$$h_t = o_t \odot \tanh(c_t)$$

**GRU:** same insight with fewer parameters — merges forget and input gates into a single update gate:
$$z_t = \sigma(W_z [h_{t-1}, x_t]), \quad r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W[r_t \odot h_{t-1}, x_t])$$

GRU uses 3 weight matrices vs LSTM's 4; performance is generally comparable, training is faster.

### Common traps

**Trap: not explaining why Transformers replaced RNNs at scale.**
It isn't that Transformers perform better (though they do). The structural problem is *serial dependency*: token $t$ cannot be processed until token $t-1$ finishes. On a GPU with thousands of cores, this means nearly all compute sits idle. Transformers compute all positions in parallel via attention — every position attends to every other position simultaneously. Training a 70B LSTM would take orders of magnitude longer than a 70B Transformer, making it not just slower but economically infeasible.

| | RNN | LSTM | GRU | Transformer |
| :--- | :--- | :--- | :--- | :--- |
| Parallelizable over sequence | No | No | No | Yes |
| Long-range memory | Poor | Good | Good | Full (any distance, O(1) path) |
| Parameters (relative) | Low | ~4× RNN | ~3× RNN | Many (but compute-parallel) |
| Training at LLM scale | Infeasible | Infeasible | Infeasible | Standard |

---

## 3. Transformer

### What the interviewer is actually testing
Whether you can derive *why* self-attention exists — not just describe it. The competency: understand the $O(n^2)$ cost and when it matters, what the $\sqrt{d_k}$ scaling prevents and why, and the precise difference between encoder and decoder architectures.

### The reasoning structure

CNNs have bounded receptive fields: a word at position 0 can only "see" a word at position 500 after many stacked layers. RNNs have serial dependency. Both bottlenecks vanish with self-attention: every position attends to every other position *directly*, in a single operation. The path length between any two positions in a Transformer is $O(1)$ regardless of sequence length. This is why a Transformer can, in a single layer, resolve long-distance coreference that would require hundreds of RNN steps to propagate.

### The pattern in action

**Scaled dot-product attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Query $Q$: what this token is looking for. Key $K$: what each token advertises. Value $V$: what each token actually contributes. Compatibility is the dot product; softmax converts scores to weights; the output is a weighted sum of values.

**Why $\sqrt{d_k}$:** Without scaling, dot products have variance $d_k$, so at $d_k=64$ scores have std 8. A score of 30 causes softmax to collapse to a one-hot vector, zeroing gradients. Dividing by $\sqrt{d_k}$ restores variance to 1. Full derivation: [math-derivations.md §5](math-derivations.md#5-why-sqrt-d_k-in-attention-scaling).

**Multi-head attention** lets the model attend to different aspects simultaneously:
$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

One head might capture subject-verb agreement, another long-range coreference, another local dependency. Each head uses $d_k = d_\text{model}/h$ — same total compute as one large head.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm: normalize before the sublayer, not after
        # This keeps the residual path clean and prevents gradient explosion at init
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
```

**Complexity:**

| Component | Compute | Memory | What it limits |
| :--- | :--- | :--- | :--- |
| Self-attention | $O(n^2 d)$ | $O(n^2)$ | Long context — the attention matrix |
| Feed-forward | $O(n d^2)$ | $O(nd)$ | Model depth and width |

The $O(n^2)$ memory cost is why a 100K-token context requires Flash Attention. Standard attention materializes the full $n \times n$ attention matrix in GPU HBM. Flash Attention tiles the computation into SRAM-sized blocks and never writes the full matrix to HBM — producing the identical result without the memory cost. This is not an approximation.

**Positional encoding — why it exists:**
Self-attention is a set operation. Permuting the input tokens produces a permuted output — the order of tokens does not affect attention scores. "Dog bites man" and "man bites dog" are identical to a pure attention layer: the same set of (Q, K, V) values, rearranged. Without positional information, the model cannot learn that word order matters at all.

Sinusoidal PE (original Transformer) adds position-dependent signals to token embeddings before they enter attention:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

RoPE (LLaMA, Mistral): encodes relative position by rotating Q and K vectors. The attention score $q_m \cdot k_n$ becomes a function of $m - n$ only — the *relative distance* — not absolute positions. This gives better generalization to sequence lengths beyond what was seen during training.

**Encoder vs Decoder:**

| | Encoder (BERT) | Decoder (GPT) |
| :--- | :--- | :--- |
| Attention mask | Bidirectional — all tokens visible | Causal — only past tokens visible |
| Pretraining objective | Masked token prediction | Next-token prediction |
| What it enables | Rich representations for classification | Autoregressive generation |
| Why BERT can't generate | It sees future tokens during training — the "answer" is available at every position | — |

### Common traps

**Trap: treating pre-norm vs post-norm as cosmetic.**
Original Transformer: $x = \text{LN}(x + F(x))$ (post-norm). GPT-2 and all modern LLMs: $x = x + F(\text{LN}(x))$ (pre-norm). Post-norm means the output of each sublayer is normalized, but gradients flowing through the residual connection can still explode at initialization if the sublayer outputs large values. Pre-norm normalizes the *input* to the sublayer — the residual path is completely clean, and the normalized branch cannot cause gradient explosion through the addition. Pre-norm training is significantly more stable at large scale.

**Trap: saying Flash Attention approximates attention.**
Flash Attention produces mathematically identical output to standard attention. The change is purely algorithmic (IO-aware computation that avoids writing the full attention matrix to HBM). Same result; 2-4× faster; memory linear in sequence length for the attention operation.

---

## 4. GAN (Generative Adversarial Network)

### What the interviewer is actually testing
Whether you understand adversarial training dynamics — specifically the instabilities and why they are structurally hard to fix. The competency: framing the min-max game, diagnosing each failure mode, and articulating why diffusion models overcame GAN limitations.

### The reasoning structure

Direct likelihood maximization — $\max \log p_\theta(x)$ — is intractable for complex distributions like images: you can't evaluate $p_\theta(x)$ for a neural generator in closed form. GANs sidestep this entirely by replacing the likelihood criterion with an adversarial one: does a discriminator network think the generated sample is real?

The minimax game:
$$\min_G \max_D \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

At Nash equilibrium: $D(x) = 0.5$ everywhere — discriminator cannot distinguish real from generated. Getting there reliably is the unsolved problem.

### The pattern in action

**Three structural failure modes:**

| Failure | Mechanism | What it looks like | Fix |
| :--- | :--- | :--- | :--- |
| Mode collapse | Generator converges to a small set of outputs that fool the discriminator — diversity disappears | All generated faces look similar | Minibatch discrimination, unrolled GANs, diversity regularization |
| Vanishing gradient | When $D$ is too accurate, $\log(1 - D(G(z))) \approx 0$ — generator gradient approaches zero | Generator stops improving despite discriminator being perfect | Non-saturating loss: maximize $\log D(G(z))$ instead of minimizing $\log(1-D(G(z)))$ |
| Training oscillation | Generator and discriminator cycle — neither converges | Loss curves oscillate without settling | Wasserstein GAN with Lipschitz constraint |

**WGAN:** replaces the classifier discriminator with a critic that measures Earth-mover distance:
$$\mathcal{L}_\text{WGAN} = \mathbb{E}_{x \sim p_\text{data}}[D(x)] - \mathbb{E}_z[D(G(z))]$$

The critic must be 1-Lipschitz (enforced via gradient penalty: $\mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$). This provides meaningful gradients even when the generator and data distributions don't overlap at all — solving the vanishing gradient problem structurally.

### Common traps

**Trap: not knowing why GANs lost to diffusion models.**
The adversarial training instabilities — mode collapse, vanishing gradients, oscillation — were never fully solved despite a decade of effort. The fundamental issue is that the training objective is a minimax game, which has no guarantee of convergence to the optimal solution. Diffusion models train by simple regression (predict the added noise), have no adversarial dynamics, and produce better sample diversity with no mode collapse mechanism. The one remaining advantage of GANs: single-step inference. Diffusion requires 20–1000 denoising steps; a GAN requires one forward pass. This matters for real-time generation.

---

## 5. Autoencoder / VAE

### What the interviewer is actually testing
Whether you understand the reparameterization trick and *why* it's necessary — and the precise difference between a deterministic bottleneck and a stochastic one. The competency: explain what posterior collapse is and why it's the opposite failure from a bad latent space.

### The reasoning structure

The autoencoder is straightforward: compress $x$ to bottleneck $z$, reconstruct $\hat{x}$, train with reconstruction loss. The bottleneck forces a compressed representation.

The failure mode for *generation*: the latent space of a standard autoencoder is not a continuous manifold you can sample from. Points between two training examples' latent codes may decode to garbage — there's no regularity. Two nearby points in latent space might decode to completely different outputs because nothing encouraged smooth structure.

A VAE fixes this by making the encoder output a *distribution* over latent codes, then regularizing that distribution toward $\mathcal{N}(0, I)$. Sampling from the latent space becomes meaningful: all of $\mathcal{N}(0, I)$ decodes to valid outputs.

### The pattern in action

VAE encoder: outputs $\mu_\phi(x)$ and $\log\sigma_\phi^2(x)$ — parameters of $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$.

VAE objective (ELBO):
$$\mathcal{L}_\text{VAE} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]}_{\text{reconstruct accurately}} - \underbrace{D_\text{KL}(q_\phi(z \mid x) \| \mathcal{N}(0,I))}_{\text{keep latent space organized}}$$

**Why the reparameterization trick is necessary:** you cannot backpropagate through a random sample $z \sim \mathcal{N}(\mu, \sigma^2)$ — sampling is not a differentiable operation. The trick: rewrite $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. Now $z$ is a deterministic, differentiable function of $\mu$ and $\sigma$ (which the encoder outputs), with a noise term $\epsilon$ that is treated as a constant with respect to gradients.

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)     # constant w.r.t. gradient
    return mu + eps * std           # gradient flows through mu and std

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence between N(mu, sigma^2) and N(0, 1), closed form:
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

### Common traps

**Trap: confusing the purpose of the KL term.**
Without the KL term, the encoder learns a Dirac delta (point estimates) and the VAE degenerates to a standard autoencoder with no generative structure — you can't sample from it. **Posterior collapse** is the *opposite* failure: the KL term becomes too strong, the encoder ignores $x$ (sets $q_\phi(z \mid x) \approx p(z)$ for all $x$), and the decoder becomes a marginal language model with no use for the latent code. The KL term and reconstruction term must both be non-negligible for the VAE to work as intended. A $\beta$-VAE controls this with a weight $\beta > 1$ on the KL term: stronger disentanglement but higher reconstruction error.

---

## 6. Diffusion Models

### What the interviewer is actually testing
Whether you understand the closed-form forward process and why it enables efficient training. The competency: explain why "predict the noise" produces stable training that GANs could never achieve, and what DDIM does differently from DDPM.

### The reasoning structure

The key insight that makes diffusion models tractable: the forward process (adding Gaussian noise over $T$ steps) has a closed form. You don't need to simulate 1000 steps to get the noisified version of $x_0$ at timestep $t$ — you can compute it in one shot:

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\right), \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

This is what enables training at scale: for any batch element, sample a random timestep $t$, corrupt $x_0$ by the right amount in one operation, and train the denoiser.

### The pattern in action

**Simplified training objective (Ho et al.):** predict the noise that was added:
$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\; t)\|^2\right]$$

The network $\epsilon_\theta$ takes a noisy image and a timestep, and outputs the predicted noise. This is regression — not a minimax game, not a discriminator to balance, not an intractable likelihood. It trains stably with standard optimization. Mode collapse has no mechanism to occur: each training step is an independent regression on a random noise level.

**Why diffusion beat GANs structurally:**
1. Stable training — regression loss, no adversarial dynamics
2. Better sample diversity — no mode collapse mechanism
3. Classifier-free guidance works reliably — scale guidance weight to trade diversity for fidelity
4. Better coverage of complex distributions at large scale

**Inference DDPM vs DDIM:**
DDPM: stochastic reverse process, requires all $T$ steps (typically 1000).
DDIM: reframes the reverse process as an ODE (deterministic). The same trained model; a different sampling scheme. Allows 10-50 steps with minimal quality loss because the ODE solver can take larger steps.

**The current state of the speed gap:**
DDIM: 20-50 steps. Consistency models: 1-4 steps. Latent diffusion (Stable Diffusion): denoises in a compressed latent space (64×64 instead of 512×512), reducing per-step cost by ~64×. The single-step advantage of GANs has been largely closed.

### Common traps

**Trap: saying diffusion models are slow without qualification.**
The 1000-step requirement was DDPM-specific. DDIM showed the trained model works with far fewer steps. The framing to use in an interview: "Inference speed was the limitation, and it's been addressed through ODE-based sampling (DDIM), latent space diffusion, and consistency models. The remaining tradeoff is one-step generation quality vs multi-step generation quality — not whether multi-step is required at all."

---

## 7. Architecture Selection Framework

### What the interviewer is actually testing
Whether you reason from task structure to architecture inductive bias — or default to whatever is currently popular. The competency: identify what structural property of the data is load-bearing, and which architecture encodes that property natively.

### The reasoning structure

"Which architecture?" is always really "what structural priors does my data have, and which architecture bakes in those priors so the model doesn't have to learn them from scratch?"

- **CNNs:** locality and translation equivariance. Data where nearby elements are more correlated than distant ones, and where features appear at multiple positions.
- **Transformers:** arbitrary pairwise relationships. Data where any element might be relevant to any other, and where global context is essential.
- **GNNs:** graph topology. Data with explicit relational structure between entities.
- **MLPs / gradient boosting:** tabular data with no spatial or sequential structure among features.

| Task | Architecture | The inductive bias that's load-bearing |
| :--- | :--- | :--- |
| Image classification | CNN or ViT | Locality (CNN) or global patch relationships (ViT, requires scale) |
| Text generation | Decoder-only Transformer | Causal attention; scales with compute |
| Text understanding | Encoder Transformer | Bidirectional context for representations |
| Image generation | Diffusion + U-Net | Hierarchical spatial resolution + noise schedule |
| Time series forecasting | Temporal CNN or Transformer | Whether locality or long-range matters more |
| Graph data | GNN (message passing) | Permutation invariance; topology-aware computation |
| Tabular data | XGBoost or MLP | No spatial structure — tree splits match tabular properties |
| Long documents (>32K tokens) | Sparse attention or retrieval-augmented | $O(n^2)$ cost makes dense attention prohibitive |

### Common traps

**Trap: recommending Transformers for tabular data.**
ViT-style models for tabular data rarely beat XGBoost unless the dataset has millions of rows and rich categorical structure requiring learned embeddings. The Transformer's strength is scaling with data and compute; its weakness is requiring enormous data before the absence of inductive bias stops hurting performance. XGBoost has the right inductive bias for tabular data (decision tree splits naturally handle mixed numeric and categorical features, threshold-based splits capture nonlinear interactions efficiently) and trains in minutes. The correct framing: "XGBoost or LightGBM first; move to a neural approach only if you have > 1M rows, need to jointly embed high-cardinality IDs or raw text, or need to share representations across multiple tasks."

## Rapid Recall

### No translation invariance
- Direct Answer: a cat in the top-left corner is a completely different input vector than a cat in the bottom-right. The model must learn a separate detector for every position.
- Why: This matters because it tells you how to reason about no translation invariance.
- Pitfall: Don't answer "No translation invariance" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: a cat in the top-left corner is a completely different input vector than a cat in the bottom-right. The model must learn a separate detector for every position.

### No locality
- Direct Answer: distant pixels get equal weight to adjacent ones. The correlation structure of natural images (local patches are highly correlated; distant pixels much less so) is invisible to the architecture.
- Why: This matters because it tells you how to reason about no locality.
- Pitfall: Don't answer "No locality" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: distant pixels get equal weight to adjacent ones. The correlation structure of natural images (local patches are highly correlated; distant pixels much less so) is invisible to th…

### Parameter explosion
- Direct Answer: every input pixel independently connected to every hidden unit forces massive parameter counts before any useful computation begins.
- Why: This matters because it tells you how to reason about parameter explosion.
- Pitfall: Don't answer "Parameter explosion" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: every input pixel independently connected to every hidden unit forces massive parameter counts before any useful computation begins.

### CNNs
- Direct Answer: locality and translation equivariance. Data where nearby elements are more correlated than distant ones, and where features appear at multiple positions.
- Why: This matters because it tells you how to reason about cnns.
- Pitfall: Don't answer "CNNs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: locality and translation equivariance. Data where nearby elements are more correlated than distant ones, and where features appear at multiple positions.

### Transformers
- Direct Answer: arbitrary pairwise relationships. Data where any element might be relevant to any other, and where global context is essential.
- Why: This matters because it tells you how to reason about transformers.
- Pitfall: Don't answer "Transformers" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: arbitrary pairwise relationships. Data where any element might be relevant to any other, and where global context is essential.

### GNNs
- Direct Answer: graph topology. Data with explicit relational structure between entities.
- Why: This matters because it tells you how to reason about gnns.
- Pitfall: Don't answer "GNNs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: graph topology. Data with explicit relational structure between entities.

### MLPs / gradient boosting
- Direct Answer: tabular data with no spatial or sequential structure among features.
- Why: This matters because it tells you how to reason about mlps / gradient boosting.
- Pitfall: Don't answer "MLPs / gradient boosting" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tabular data with no spatial or sequential structure among features.
