# Regularization

---

## Overfitting

**The problem**: a neural network with millions of parameters trained on a finite dataset can fit the training data essentially perfectly — memorizing idiosyncrasies, noise, and irrelevant correlations. It achieves near-zero training loss but fails on new data. Capacity is not the issue; the model has too much freedom to be gullible.

**The core insight**: constrain what the model is allowed to learn. Regularization is any technique that reduces a model's ability to overfit — by penalizing complexity, introducing noise during training, limiting access to data, or stopping before the model has fully memorized the training set.

---

## L2 Regularization (Weight Decay)

**The problem**: without any constraint, weights can grow arbitrarily large. Large weights mean the model is extremely sensitive to small input changes — it has learned to exploit fine-grained patterns that may be noise, not signal.

**The core insight**: add a penalty proportional to the squared magnitude of all weights. Large weights are expensive. The optimizer balances fitting the training data against keeping weights small — it will only grow a weight large if the training data strongly supports it.

**The mechanics**:

$$L_\text{total} = L_\text{task} + \frac{\lambda}{2} \sum_w w^2$$

Gradient of the penalty: $\lambda w$. At each update, every weight is shrunk toward zero by a fraction proportional to its current value. Large weights shrink faster. Small weights are nearly unaffected.

$$w \leftarrow w - \eta(\nabla_w L_\text{task} + \lambda w) = w(1 - \eta\lambda) - \eta \nabla_w L_\text{task}$$

This is why L2 regularization is called weight decay — each step decays the weight by factor $(1 - \eta\lambda)$.

**What breaks**: in Adam (not AdamW), the weight penalty $\lambda w$ is added to the gradient before adaptive scaling. The adaptive denominator scales down the penalty for parameters with large gradient history — weight decay is effectively weaker for those parameters. Use AdamW, which applies weight decay directly to the weights after the gradient update, decoupled from the adaptive scaling.

---

## L1 Regularization

**The problem**: L2 regularization shrinks all weights but never forces them to exactly zero. You want the model to perform feature selection — identifying which input features are irrelevant and zeroing their associated weights completely.

**The core insight**: penalize the absolute value of weights rather than the squared value. The gradient of $|w|$ is $\text{sign}(w)$ — a constant push toward zero regardless of the weight's current magnitude. Unlike L2 (where the push weakens as the weight approaches zero), L1 keeps pushing at the same rate until the weight crosses zero and the sign flips. This causes weights to "snap" to exactly zero.

**The mechanics**:

$$L_\text{total} = L_\text{task} + \lambda \sum_w |w|$$

Gradient penalty: $\lambda \cdot \text{sign}(w)$.

**What breaks**: L1 is useful for linear models and explicit feature selection. In deep neural networks, sparsity in individual weights does not translate to structured efficiency (see unstructured pruning), and L1 gradients are non-differentiable at $w = 0$. For deep learning, L2 (weight decay via AdamW) is almost always preferred. L1 is used in sparse autoencoders where explicit latent sparsity is the goal.

---

## Dropout

**The problem**: in a fully connected network, neurons can co-adapt — neuron $A$ learns to detect a pattern, and neuron $B$ learns to rely on neuron $A$ rather than detecting its own pattern. This co-adaptation means the network is not truly learning robust, redundant representations. Any neuron that is disrupted takes a chain of dependent neurons down with it.

**The core insight**: during each training step, randomly disable a fraction $p$ of neurons. Each neuron cannot rely on the presence of its neighbors — it must learn to be useful on its own. The network learns multiple overlapping representations, any subset of which can produce a reasonable output.

**The mechanics**:

$$h' = \frac{h \cdot m}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)$$

Each neuron is kept with probability $1-p$ and zeroed with probability $p$. The $(1-p)$ denominator (inverted dropout) scales up surviving neurons to preserve expected value. At test time, all neurons are active and no scaling is needed.

Ensemble interpretation: with $n$ neurons, there are $2^n$ possible sub-networks. Dropout trains all of them simultaneously with shared weights. The final model is approximately an average over all $2^n$ sub-networks.

```python
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),    # only active during model.train()
    nn.Linear(256, 10)
)
# model.train() enables dropout
# model.eval() disables dropout — NEVER forget this before inference
```

Typical rates: $p = 0.5$ for large dense layers, $p = 0.1$–$0.3$ for smaller layers or Transformers.

**What breaks**: dropout on convolutional layers tends to be less effective because adjacent spatial positions are highly correlated — dropping individual activations does not break the spatial redundancy. DropBlock (drop contiguous spatial regions) is better for CNNs. For Transformers, high dropout rates slow convergence because the model receives noisy gradient signals from many zeroed attention and FFN outputs.

---

## Early Stopping

**The problem**: as training continues beyond the point of peak generalization, the model begins memorizing training data. Training loss keeps falling; validation loss starts rising. You want the model at peak generalization, not at end of training.

**The core insight**: monitor validation loss during training. Save the model checkpoint when validation loss is at its best. Stop training when validation loss has not improved for $k$ consecutive evaluations.

**The mechanics**:

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(max_epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

model.load_state_dict(torch.load('best.pt'))
```

**What breaks**: validation loss can be noisy — a single bad batch can trigger false early stopping. Use a patience window (stop only after $k$ consecutive non-improvements). The validation set must be representative; if it is too small or biased, the stopping criterion is unreliable.

---

## Data Augmentation

**The problem**: the network sees a fixed set of training images. It may learn to recognize objects only in the orientations, scales, and lighting conditions present in training data. A horizontally-flipped image of a cat is still a cat, but if no flipped cats appeared in training, the model may not recognize it.

**The core insight**: apply random transformations to inputs during training that preserve label semantics. The model is forced to learn representations invariant to those transformations — it cannot distinguish "cat facing left" from "cat facing right" because it has seen both labeled "cat."

**Standard vision augmentations**:

```python
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),      # cats look the same both ways
    T.RandomCrop(32, padding=4),        # position shouldn't matter
    T.ColorJitter(brightness=0.2, contrast=0.2),  # lighting varies
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**What breaks**: augmentations must preserve label semantics. Horizontal flip is safe for most natural images but wrong for text (flipping a "b" produces a "d"). Color jitter is safe for object recognition but wrong for tasks where color is the label. Aggressive augmentation can slow convergence significantly — the model trains on harder examples from the start.

---

## MixUp

**The problem**: standard data augmentation creates new examples by transforming single inputs. The model still trains on clean, one-thing-per-image examples. The decision boundary near the boundary between classes is underconstrained — the model has never been trained on examples that are genuinely between two classes.

**The core insight**: linearly interpolate between two training examples and their labels. Train the model to produce interpolated predictions for interpolated inputs. This encourages smoother decision boundaries and calibrated confidence near class boundaries.

**The mechanics**:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

The mixed input is a blend of two examples; the target is a soft blend of their labels.

- $\alpha = 0.2$: $\lambda$ mostly near 0 or 1 — nearly clean examples with mild mixing
- $\alpha = 1.0$: $\lambda \sim \text{Uniform}(0,1)$ — more aggressive blending

**What breaks**: MixUp with $\lambda = 0.5$ produces genuinely ambiguous images (half cat, half dog). If the model is not deep enough, it may not be able to output the correct blend probabilities and training becomes unstable. Also, MixUp increases the apparent dataset difficulty — convergence typically requires more epochs.

**CutMix**: instead of blending pixels, cut a random rectangular patch from one image and paste it into another, mixing labels proportionally to patch area. Preserves local spatial structure better than MixUp and often outperforms it for vision tasks.

---

## DropPath (Stochastic Depth)

**The problem**: dropout regularizes individual neurons. In residual networks, entire residual blocks may become redundant — a block could be removed and the model would route around it via the skip connection. Individual neuron dropout does not address block-level redundancy.

**The core insight**: randomly drop entire residual branches during training. The residual stream flows through the skip connection when a block is dropped — effectively shortening the network's depth stochastically at each training step.

**The mechanics**:

$$x_{l+1} = x_l + b_l \cdot F_l(x_l), \quad b_l \sim \text{Bernoulli}(1 - p_l)$$

Drop probability $p_l$ increases linearly with layer depth — deeper layers are dropped more aggressively, since the model can rely more heavily on earlier representations.

**What breaks**: aggressive drop rates can cause underfitting — too many blocks are dropped for the model to compose deep representations. The schedule of drop probabilities (from 0 near the input to max near the output) must be tuned with the network depth.

Used in: ViT, EfficientNet, Swin Transformer.

---

## Spectral Normalization

**The problem**: in GANs, the discriminator has no inherent constraint on how rapidly its output can change with its input. An unconstrained discriminator can grow arbitrarily Lipschitz — its gradients can explode, destabilizing generator training.

**The core insight**: constrain the spectral norm (largest singular value) of each weight matrix to 1. A 1-Lipschitz network has bounded gradients by definition.

**The mechanics**: estimate the spectral norm $\sigma(W)$ via power iteration at each forward pass. Divide the weight matrix by $\sigma(W)$ during the forward computation. No change to the weight matrix itself — the normalization is applied at compute time.

**What breaks**: spectral normalization limits how expressive the discriminator can be. If the constraint is too tight (relative to how complex the true data distribution is), the discriminator cannot distinguish real from fake well enough to give useful gradient signal to the generator. The constraint is a regularizer — calibrate it with the problem complexity.

---

## Regularization Summary

| Technique | Mechanism | When to use |
| :--- | :--- | :--- |
| **L2 / AdamW weight decay** | Penalizes large weights | Universal default |
| **L1** | Encourages exact zeros | Feature selection in linear models |
| **Dropout** | Random neuron masking | Dense layers, Transformers |
| **Early stopping** | Stop at best validation checkpoint | Always, alongside other regularizers |
| **Data augmentation** | Invariance to label-preserving transforms | Vision, audio |
| **MixUp / CutMix** | Train on interpolated examples | Vision; smoothes decision boundaries |
| **DropPath** | Drop entire residual blocks | ViT, EfficientNet, deep residual nets |
| **Spectral normalization** | Lipschitz constraint per layer | GAN discriminators |

---

## Canonical Interview Q&As

**Q: Explain the L1 vs L2 regularization trade-off — why does L1 produce sparse weights?**  
A: Both add a penalty to the loss: L1 adds λΣ|w_i|, L2 adds λΣw_i². The gradient tells the story: L2 gradient = 2λw_i — always proportional to the weight, so small weights get very small updates. L1 gradient = λ·sign(w_i) — constant magnitude regardless of weight value. This means L1 applies the same "push toward zero" force to all nonzero weights, no matter how small they are, so small weights hit exactly zero. L2 only asymptotically approaches zero. Geometrically: the L1 constraint set (diamond shape) has corners on the axes, making intersection with the loss contours more likely at sparse points. L2 constraint set (sphere) has no corners. Use L1 when you believe many features are irrelevant (e.g., high-dimensional genetic data with few causal variants). Use L2 when you believe all features contribute somewhat and you want to limit large weights (most neural networks use L2/weight decay). ElasticNet combines both — useful when features are correlated and you want some sparsity.

**Q: Why does dropout work — what is it implicitly optimizing?**  
A: Dropout zeroes each activation with probability p during training and scales by 1/(1-p) at test time. Three complementary explanations: (1) Ensemble approximation — each forward pass uses a different subset of the network, so training corresponds to optimizing 2^N possible networks simultaneously; test-time prediction is approximately averaging these networks; (2) Prevents co-adaptation — neurons can't rely on the presence of specific other neurons, so each must learn to be useful independently; this forces distributed representations; (3) Implicit Bayesian inference — Gal & Ghahramani (2016) showed dropout at test time approximates Bayesian posterior inference over the network weights; keeping dropout on at test time gives a MC estimate of prediction uncertainty. Practical notes: dropout is effective for dense layers but hurts convolutional layers (spatial correlation means nearby units are redundant; SpatialDropout2D is better). DropPath (for transformers/residual nets) drops entire residual branches at path granularity, working better for deep architectures where per-activation dropout creates inconsistent feature maps.

**Q: How do you detect and handle overfitting in a deep learning model?**  
A: Detection: training loss much lower than validation loss and diverging, or validation metric plateaus while training metric continues improving. Diagnosis first — check if overfitting is from: (1) too few examples (get more data or augment), (2) model too large (reduce depth/width), (3) training too long (early stopping), or (4) features leaking label information. Mitigation stack in order of preference: (1) More data / data augmentation — always the first resort; augmentation creates synthetic variations the model shouldn't overfit to; (2) Early stopping — checkpoints at best validation metric, stop after K epochs without improvement; (3) Dropout — typically 0.1-0.3 in transformer FFNs, 0.5 in dense layers; (4) Weight decay (L2) — λ=0.01-0.1 for most cases; (5) Reduce model size if the gap is extreme. Anti-pattern: applying heavy regularization instead of diagnosing the root cause. If the model overfits with 1K training examples, adding dropout may hide the symptom but not address that you need more data.
