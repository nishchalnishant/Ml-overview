# Adversarial Robustness

A field studying how machine learning models fail under intentionally crafted inputs, and how to build models that resist such attacks.

---

## 1. What Are Adversarial Examples

Adversarial examples are inputs crafted by adding small, often imperceptible perturbations to clean data that cause a model to produce incorrect predictions with high confidence.

**Key properties:**
- The perturbation is bounded so the modified input looks identical (or nearly identical) to a human
- The model's predicted class flips, often to a specific target class
- They generalize across architectures and training procedures

### Threat Models (Perturbation Norms)

The attacker's budget is formalized as a constraint on the perturbation `δ = x' - x`:

| Norm | Formula | Effect |
|------|---------|--------|
| **L∞** | `max_i |δ_i| ≤ ε` | Limits max pixel change; uniform, hard to see |
| **L2** | `√(Σ δ_i²) ≤ ε` | Limits Euclidean distance; allows large local changes if compensated |
| **L0** | `‖δ‖_0 ≤ k` | Limits number of pixels changed; sparse, targeted pixel flips |

L∞ is the most common in the literature (e.g., ε = 8/255 on ImageNet, ε = 0.3 on MNIST).

---

## 2. FGSM — Fast Gradient Sign Method

**Paper:** Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)

Single-step attack. Moves the input one step in the direction that maximally increases the loss, bounded by ε under L∞.

### Formula

```
x' = x + ε · sign(∇_x L(f(x), y))
```

- `L` is the cross-entropy loss
- `∇_x L` is the gradient of the loss with respect to the input (not model parameters)
- `sign(·)` takes the elementwise sign, giving each dimension ±1
- `ε` is the perturbation budget

### Code

```python
import torch
import torch.nn.functional as F

def fgsm(model, x, y, epsilon):
    """
    Fast Gradient Sign Method attack.
    
    Args:
        model: PyTorch model (in eval mode)
        x: clean input tensor, shape (B, C, H, W), values in [0, 1]
        y: true labels, shape (B,)
        epsilon: L-inf perturbation budget
    
    Returns:
        x_adv: adversarial examples clipped to [0, 1]
    """
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()
```

**Limitation:** Single-step; often fooled by gradient masking defenses. Weak compared to iterative methods.

---

## 3. PGD — Projected Gradient Descent

**Paper:** Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)

Multi-step version of FGSM. Each step is a small FGSM step; after each step the result is projected back onto the ε-ball around the original input. Considered the standard white-box attack.

### Algorithm

```
x_0 = x + Uniform(-ε, ε)   # random start within ε-ball

for t in 1..T:
    x_t = x_{t-1} + α · sign(∇_x L(f(x_{t-1}), y))
    x_t = Proj_{B_∞(x, ε)}(x_t)    # clip to [x-ε, x+ε]
    x_t = clip(x_t, 0, 1)           # clip to valid image range
```

- `α` is the step size (typically `ε / 4` or `2ε / T`)
- Random start makes the attack stronger and avoids local gradient artifacts
- Projection under L∞ is elementwise clipping to `[x - ε, x + ε]`

### Code

```python
import torch
import torch.nn.functional as F

def pgd(model, x, y, epsilon, alpha, num_steps, random_start=True):
    """
    PGD attack under L-inf constraint.

    Args:
        model: PyTorch model (eval mode)
        x: clean input, shape (B, C, H, W), values in [0, 1]
        y: true labels, shape (B,)
        epsilon: L-inf budget
        alpha: step size per iteration
        num_steps: number of PGD steps
        random_start: whether to initialize with a random perturbation

    Returns:
        x_adv: adversarial examples
    """
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back to epsilon-ball around original x
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            # Keep in valid image range
            x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()
```

**Standard settings:** ε = 8/255, α = 2/255, T = 20 or 40 steps on CIFAR-10.

---

## 4. C&W Attack — Carlini & Wagner

**Paper:** Carlini & Wagner, "Evaluating the Robustness of Neural Networks: An Adversarial Examples Perspective" (IEEE S&P 2017)

Optimization-based attack that directly minimizes perturbation size subject to the example being misclassified. Stronger than PGD but computationally heavier.

### Formulation (L2 variant)

```
minimize   ‖δ‖_2 + c · f(x + δ)
subject to  x + δ ∈ [0, 1]^n
```

Where `f(x')` is a custom objective that is negative when `x'` is successfully misclassified:

```
f(x') = max( max_{j ≠ t} Z(x')_j - Z(x')_t, -κ )
```

- `Z(x')` are the logits
- `t` is the target class
- `κ` controls the confidence of the misclassification
- `c` is found via binary search

**Key trick:** change of variables `δ = tanh(w) · (x_max - x_min)/2` to enforce box constraint without projection, enabling unconstrained optimization with Adam.

**Why stronger than PGD:**
- Finds minimum-norm perturbations (more natural adversarial examples)
- Directly optimizes the misclassification objective rather than maximizing cross-entropy
- Bypasses gradient masking more reliably

---

## 5. AutoAttack — Parameter-Free Ensemble

**Paper:** Croce & Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks" (ICML 2020)

Ensemble of four attacks designed to give a reliable, parameter-free evaluation of adversarial robustness. The de facto standard for reporting robust accuracy.

### Components

| Attack | Type | Description |
|--------|------|-------------|
| **APGD-CE** | White-box | Adaptive PGD with cross-entropy loss, automatic step size |
| **APGD-DLR** | White-box | Adaptive PGD with difference-of-logits-ratio loss (avoids gradient masking) |
| **FAB** | White-box | Fast Adaptive Boundary attack; finds minimum L∞/L2/L1 perturbation |
| **Square Attack** | Black-box | Score-based attack using random square perturbations; no gradients |

### DLR Loss (Difference of Logits Ratio)

```
L_DLR(x, y) = -(Z_y - max_{j≠y} Z_j) / (Z_π1 - Z_π3)
```

The denominator normalizes by the spread of top logits, making the loss scale-invariant and harder to defeat with output rescaling.

### Usage

```python
from autoattack import AutoAttack

adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)
```

`version='standard'` runs APGD-CE + APGD-DLR + FAB + Square Attack sequentially on examples that have not yet been broken.

---

## 6. Adversarial Training — Madry et al.

**Paper:** Madry et al., 2018 (same as PGD paper)

Replace clean examples in standard training with PGD adversarial examples. The model minimizes worst-case loss over the ε-ball.

### Objective

```
min_θ  E_{(x,y)~D} [ max_{δ: ‖δ‖≤ε} L(f_θ(x + δ), y) ]
```

Inner maximization: PGD attack (find worst-case perturbation).  
Outer minimization: standard SGD on the adversarial examples.

### Training Loop

```python
model.train()
for x, y in dataloader:
    x, y = x.to(device), y.to(device)

    # Inner maximization: generate adversarial examples
    x_adv = pgd(model, x, y, epsilon=8/255, alpha=2/255, num_steps=10)

    # Outer minimization: update model on adversarial examples
    optimizer.zero_grad()
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

### Robustness vs. Accuracy Trade-off

Adversarial training consistently degrades clean accuracy (by ~10-15% on CIFAR-10). This is not purely a training artifact — it reflects a fundamental tension:

- Robust features are less "predictive" in a standard sense
- The model must be invariant to directions that normally carry signal
- The Bayes-optimal robust classifier differs from the Bayes-optimal clean classifier

---

## 7. TRADES — Trade-off Inspired Adversarial Defense via Surrogate-Loss Minimization

**Paper:** Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy" (ICML 2019)

Decomposes the robust error into natural error + boundary error, then optimizes a regularized objective that controls both.

### Objective

```
min_θ  E[ L(f_θ(x), y) ]  +  (1/λ) · E[ max_{x': ‖x'-x‖≤ε} KL(f_θ(x) ‖ f_θ(x')) ]
```

- First term: standard cross-entropy on clean examples
- Second term: KL divergence between clean and adversarial predictions
- `λ` controls the trade-off (smaller λ → more robust, less clean accuracy)

### Key Insight

TRADES does not directly train on adversarial labels. It pushes the model's output distribution to be stable around x, without requiring the adversarial example to be correctly classified. This tends to give better clean accuracy than Madry AT while achieving comparable robustness.

### Code Sketch

```python
import torch.nn.functional as F

def trades_loss(model, x, y, epsilon, alpha, num_steps, beta):
    """
    TRADES loss = CE(clean) + beta * KL(clean || adv)
    """
    model.eval()
    # Generate adversarial example by maximizing KL divergence
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            kl = F.kl_div(
                F.log_softmax(model(x_adv), dim=1),
                F.softmax(model(x), dim=1),
                reduction='batchmean'
            )
        kl.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon).clamp(0, 1)

    model.train()
    # Natural loss on clean examples
    loss_natural = F.cross_entropy(model(x), y)
    # Regularization: KL between clean and adversarial softmax outputs
    loss_robust = F.kl_div(
        F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(x).detach(), dim=1),
        reduction='batchmean'
    )
    return loss_natural + beta * loss_robust
```

---

## 8. Certified Defenses — Randomized Smoothing

**Paper:** Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)

Provides a provable (certified) lower bound on the L2 radius within which the prediction cannot be changed by any attacker.

### Core Idea

Given a base classifier `f`, define the **smoothed classifier** `g`:

```
g(x) = argmax_c  P_{ε ~ N(0, σ²I)}[ f(x + ε) = c ]
```

`g` returns the class that `f` most frequently predicts when x is corrupted with Gaussian noise.

### Certification Theorem

If `g(x) = c_A` and `p_A = P[f(x+ε) = c_A]`, then `g` is robust around `x` with certified radius:

```
R = (σ/2) · ( Φ⁻¹(p_A) - Φ⁻¹(p_B) )
```

Where:
- `Φ⁻¹` is the inverse normal CDF
- `p_A` is the probability of the top class
- `p_B` is the probability of the runner-up class
- A simple lower bound: `R ≥ σ · Φ⁻¹(p_A)` when `p_A > 0.5`

### Code

```python
import torch
import numpy as np
from scipy.stats import norm

def smooth_predict(model, x, sigma, num_samples, batch_size, device):
    """
    Randomized smoothing: majority vote over Gaussian-corrupted copies.

    Returns:
        counts: vote counts per class, shape (num_classes,)
    """
    model.eval()
    counts = None

    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples)
            noise = torch.randn(n, *x.shape, device=device) * sigma
            noisy = (x.unsqueeze(0) + noise).clamp(0.0, 1.0)
            logits = model(noisy)
            preds = logits.argmax(dim=1)

            if counts is None:
                counts = torch.bincount(preds, minlength=logits.shape[1])
            else:
                counts += torch.bincount(preds, minlength=logits.shape[1])

    return counts


def certify(model, x, y, sigma, num_samples, alpha, device):
    """
    Certify L2 robustness for a single example.

    Args:
        alpha: failure probability (e.g. 0.001)

    Returns:
        prediction: predicted class (-1 if abstain)
        radius: certified L2 radius (0 if abstain)
    """
    from statsmodels.stats.proportion import proportion_confint

    counts = smooth_predict(model, x, sigma, num_samples, batch_size=500, device=device)
    top2 = counts.topk(2)
    c_A, c_B = top2.indices[0].item(), top2.indices[1].item()
    n_A, n_B = top2.values[0].item(), top2.values[1].item()

    # One-sided lower confidence bound on p_A
    p_A_low, _ = proportion_confint(n_A, num_samples, alpha=2 * alpha, method='beta')

    if p_A_low > 0.5:
        radius = sigma * norm.ppf(p_A_low)
        return c_A, radius
    else:
        return -1, 0.0  # abstain
```

**Trade-offs:**
- Higher σ → larger certified radius, but lower clean accuracy (predictions noisier)
- Certification is exact but predictions are slow (need many samples)
- Only provides L2 guarantees; L∞ certificates require different methods (interval bound propagation, etc.)

**Other certified methods:**
- **Interval Bound Propagation (IBP):** propagate input intervals through network layers; fast but loose
- **CROWN / α-CROWN:** tighter linear relaxation bounds; state-of-the-art for verified L∞ robustness
- **Lipschitz-constrained networks:** enforce ‖f(x) - f(x')‖ ≤ K‖x - x'‖ by design

---

## 9. Robustness Benchmarks

### RobustBench

`https://robustbench.github.io`

Standardized leaderboard for adversarial robustness. All entries are evaluated with AutoAttack under fixed threat models.

**Standard threat models:**
- CIFAR-10: L∞ ε = 8/255
- CIFAR-100: L∞ ε = 8/255
- ImageNet: L∞ ε = 4/255

**Top entries (as of 2024):**
- CIFAR-10 L∞: ~71-73% robust accuracy (WideResNet + adversarial training variants)
- ImageNet L∞: ~60-62% robust accuracy

### AutoAttack Evaluation Protocol

To claim robust accuracy on a new defense:

1. Evaluate on the full test set (or 10,000 samples)
2. Run `version='standard'` AutoAttack (all four attacks)
3. Report the percentage of examples not broken by any attack
4. Do NOT report accuracy against only FGSM or a single PGD run

```python
from autoattack import AutoAttack

model.eval()
adversary = AutoAttack(
    model,
    norm='Linf',
    eps=8/255,
    version='standard',
    device=device
)
# Returns adversarial examples that break the model
x_adv = adversary.run_standard_evaluation(x_test[:1000], y_test[:1000], bs=128)
```

---

## 10. Practical Defenses

These defenses do not provide formal guarantees but are often used in deployed systems.

### Input Preprocessing

**JPEG Compression:**
```python
from PIL import Image
import io

def jpeg_compress(x_np, quality=75):
    """Apply JPEG compression to remove high-frequency adversarial noise."""
    img = Image.fromarray((x_np * 255).astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf)).astype('float32') / 255.0
```

**Feature Squeezing:** reduce color bit depth or apply spatial smoothing before inference.

```python
def bit_depth_reduction(x, bits=4):
    """Reduce color depth to squeeze out adversarial perturbations."""
    max_val = 2 ** bits - 1
    return torch.round(x * max_val) / max_val
```

**Limitations of preprocessing defenses:**
- Adaptive attacks that account for the preprocessor break them reliably
- BPDA (Backward Pass Differentiable Approximation): approximate gradient through non-differentiable preprocessor, then apply PGD
- Should not be reported as certified robustness

### Detection-Based Approaches

Rather than correcting predictions, flag adversarial inputs and reject them.

**Feature squeezing detection:** run model on original and squeezed input; high disagreement → adversarial.

```python
def detect_adversarial(model, x, squeezed_x, threshold=0.05):
    with torch.no_grad():
        p1 = F.softmax(model(x), dim=1)
        p2 = F.softmax(model(squeezed_x), dim=1)
        distance = (p1 - p2).abs().max(dim=1).values
    return distance > threshold
```

**Limitations of detection:**
- An adaptive attacker optimizes to evade the detector simultaneously
- High false positive rates in practice
- Does not help when the attacker knows the detector

### Other Practical Measures

- **Ensemble diversity:** models with different architectures are harder to transfer attacks across
- **Input randomization:** random resizing and padding at inference time
- **Distillation:** soft labels smooth the loss surface (broken by C&W)

---

## 11. Transferability and the Black-Box Threat Model

### Transferability

Adversarial examples crafted on model A often transfer to model B, even if B has a different architecture, training set, or training procedure.

**Why it happens:**
- Different models learn similar decision boundaries in input space
- Adversarial perturbations exploit geometry of the data manifold, not model-specific artifacts

**Factors increasing transferability:**
- More attack iterations (stronger attacks transfer better)
- Ensemble attacks (optimize against multiple models simultaneously)
- Input diversity / translation-invariant attacks (avoid overfitting to source model)

### Black-Box Attack Settings

| Setting | Attacker Knows | Method |
|---------|---------------|--------|
| **White-box** | Architecture, weights, gradients | PGD, C&W, AutoAttack |
| **Gray-box** | Architecture, not weights | Transfer from re-trained surrogate |
| **Black-box (score)** | Output probabilities only | Square Attack, NES |
| **Black-box (decision)** | Hard label only | Boundary Attack, HopSkipJump |

**Transfer attack pipeline:**
1. Train or obtain a surrogate model on similar data
2. Run white-box attack on surrogate to generate `x_adv`
3. Query the target model with `x_adv`

```python
# Ensemble transfer attack: maximize loss across multiple surrogate models
def ensemble_fgsm(models, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = sum(F.cross_entropy(m(x_adv), y) for m in models) / len(models)
    loss.backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()
```

---

## 12. Key Interview Points

**Concepts**

- Adversarial examples exploit directions in input space that are highly predictive but imperceptible — they are a property of the geometry of high-dimensional data, not just model bugs
- FGSM is one step; PGD is multi-step FGSM with projection; both are L∞ attacks
- C&W minimizes perturbation norm directly; finds smaller, stronger perturbations than PGD
- AutoAttack is the standard evaluation tool because it is parameter-free and uses diverse attack types including a score-based black-box component (Square Attack)

**Training**

- Madry adversarial training solves a min-max problem: min over θ of max over perturbations of the loss
- Adversarial training always incurs a robustness-accuracy trade-off; no free lunch
- TRADES splits this into natural loss + KL regularization, giving better Pareto efficiency than Madry AT
- Adversarial training is expensive: inner PGD adds 10-40× compute cost

**Defenses**

- Randomized smoothing is the only practical method with provable L2 robustness guarantees at scale
- Certified radius R = σ·Φ⁻¹(p_A); larger σ gives larger R but noisier predictions
- Preprocessing defenses (JPEG, feature squeezing) are broken by adaptive attacks (BPDA)
- Gradient masking — making gradients vanish or mislead — is not true robustness; AutoAttack specifically targets it with gradient-free (Square Attack) and loss-adapted (DLR) components

**Benchmarks**

- RobustBench uses AutoAttack as the standard evaluation; never trust results evaluated only against FGSM or 20-step PGD
- Clean accuracy and robust accuracy are both reported; a defense that drops clean accuracy to 40% is not practical
- State-of-the-art robust accuracy on CIFAR-10 L∞ (ε=8/255) is ~70-73% vs ~95%+ clean accuracy

**Common Pitfalls**

- Evaluating defense only against the attack it was designed to defeat (not adaptive)
- Obfuscated gradients masquerading as robustness
- Reporting robust accuracy without specifying the threat model (ε, norm)
- Confusing robust accuracy (% correctly classified under attack) with certified accuracy (% with provable guarantee)
