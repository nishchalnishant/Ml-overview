---
module: Emerging Topics
topic: Adversarial Robustness
subtopic: ""
status: unread
tags: [emergingtopics, ml, adversarial-robustness]
---
# Adversarial Robustness

---

## 1. The Problem

In 2017, researchers placed a small printed sticker — a few centimeters wide — on a stop sign. A self-driving car's vision system, encountering the sign at road speed, classified it as a 45 mph speed limit sign. The sticker contained no text. It was designed to fool the model, not the human eye.

To understand why this is possible, you have to look at what a neural network actually learns.

A classifier trained on ImageNet sees 1.28 million labeled images and adjusts its weights to minimize cross-entropy loss. By the end of training, it can distinguish stop signs from speed limit signs at 95%+ accuracy. But the features it uses are not the features you think. Some are perceptually meaningful: the octagonal shape, the red color, the white text. Others are not: specific high-frequency texture patterns across groups of pixels, patterns that correlate with the correct label in the training set but are completely invisible to a human observer.

The network uses both kinds equally. It has no mechanism to prefer robust features over brittle ones, because both reduce training loss. This is not a bug in any particular model — it is a consequence of optimizing for label prediction without any constraint on which features to use.

The result: for any input `x` the model classifies correctly, there exists a nearby input `x'` — often within L∞ distance of 8/255 per pixel, a change undetectable to humans — that the model misclassifies with high confidence.

These adversarial examples are not random noise. They are precisely engineered. And the engineering uses the same tool as training: the gradient.

---

## 2. The Core Insight

The loss function `L(f_θ(x), y)` is differentiable with respect to the input `x`. Its gradient `∇_x L` points in the direction that increases the model's error fastest.

This gradient lives in the same pixel space as the image. Taking a small step in that direction gives you an input that looks nearly identical to the original but incurs higher loss — meaning the model is more wrong.

The geometry explains why this works so effectively in high dimensions. In a d-dimensional space, moving ε in each dimension along the gradient sign produces a total L2 perturbation of magnitude ε√d. For a 32×32×3 CIFAR-10 image, d = 3072. At ε = 8/255, the total L2 perturbation is ~0.44 — a significant shift in activation space — even though no single pixel changes by more than 8/255.

The adversarial examples problem reduces to: **the model's loss landscape is exploitable along directions that are invisible in pixel space but geometrically significant in activation space.** Three questions follow:

1. How efficiently can an attacker find these directions? (Attack design)
2. Can you train a model that doesn't have exploitable directions? (Defense design)
3. Can you prove no exploitable direction exists within some radius? (Certification)

---

## 3. The Mechanics

### Threat Model

First, define the attacker's budget — the constraint on the perturbation `δ = x' - x`:

| Norm | Constraint | Effect |
|------|-----------|--------|
| **L∞** | `max_i |δ_i| ≤ ε` | Every pixel changes by at most ε; visually uniform |
| **L2** | `√(Σ δ_i²) ≤ ε` | Euclidean budget; concentrated changes allowed |
| **L0** | `‖δ‖_0 ≤ k` | At most k pixels changed; can be large changes |

L∞ is standard. Standard budgets: ε = 8/255 on CIFAR-10 and ImageNet, ε = 0.3 on MNIST.

---

### FGSM — Fast Gradient Sign Method (Goodfellow et al., ICLR 2015)

The most direct attack: take one step in the gradient sign direction, with step size ε.

```
x' = x + ε · sign(∇_x L(f(x), y))
```

```python
import torch
import torch.nn.functional as F

def fgsm(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()
```

One step is cheap but leaves accuracy on the table. The gradient at `x` is not the gradient at `x + δ`. For a non-linear loss surface, a single step misses the true worst-case perturbation.

---

### PGD — Projected Gradient Descent (Madry et al., ICLR 2018)

The natural fix: iterate FGSM with a smaller step size, projecting back onto the ε-ball after each step. Add a random start to escape local optima near the original input.

```
x_0 = x + Uniform(-ε, ε)

for t in 1..T:
    x_t = x_{t-1} + α · sign(∇_x L(f(x_{t-1}), y))
    x_t = clip(x_t, x - ε, x + ε)    # project onto L∞ ε-ball
    x_t = clip(x_t, 0, 1)             # valid pixel range
```

```python
def pgd(model, x, y, epsilon, alpha, num_steps, random_start=True):
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()
```

Standard settings: ε = 8/255, α = 2/255, T = 20–40 on CIFAR-10. PGD is the canonical white-box attack — strong enough to be the inner loop of adversarial training.

---

### C&W Attack — Carlini & Wagner (IEEE S&P 2017)

PGD maximizes cross-entropy loss. C&W directly minimizes the perturbation norm subject to misclassification:

```
minimize   ‖δ‖_2 + c · f(x + δ)

f(x') = max( max_{j ≠ t} Z(x')_j - Z(x')_t, -κ )
```

`Z` are the logits, `t` is the target class, `κ` controls confidence margin. Change of variables `δ = tanh(w) · (x_max - x_min)/2` enforces the box constraint without projection, enabling unconstrained Adam optimization. `c` is found via binary search.

C&W finds minimum-norm adversarial examples. It is stronger than PGD and bypasses gradient masking more reliably because it directly optimizes the logit margin.

---

### AutoAttack — Reliable Evaluation (Croce & Hein, ICML 2020)

A PGD run with wrong hyperparameters can make a broken defense look robust. AutoAttack eliminates evaluation brittleness by running four diverse attacks and reporting the fraction not broken by any:

| Attack | Type | What it adds |
|--------|------|-------------|
| **APGD-CE** | White-box | Adaptive step size, cross-entropy |
| **APGD-DLR** | White-box | Scale-invariant loss; defeats output rescaling |
| **FAB** | White-box | Minimum-norm perturbation |
| **Square Attack** | Black-box | No gradients; defeats gradient masking entirely |

```python
from autoattack import AutoAttack

adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)
```

The DLR loss is the key innovation for defeating output-space obfuscation:

```
L_DLR(x, y) = -(Z_y - max_{j≠y} Z_j) / (Z_π1 - Z_π3)
```

The denominator normalizes by the spread of top logits. Multiplying all logits by a constant (a common masking technique) does not change this loss.

---

### Adversarial Training — The Defense That Works

Training on adversarial examples is the only defense that consistently withstands adaptive attacks. Instead of training on clean inputs, solve:

```
min_θ  E_{(x,y)~D} [ max_{δ: ‖δ‖≤ε} L(f_θ(x + δ), y) ]
```

Inner maximization: PGD. Outer minimization: SGD on adversarial examples.

```python
model.train()
for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    x_adv = pgd(model, x, y, epsilon=8/255, alpha=2/255, num_steps=10)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    optimizer.step()
```

The cost: adversarially trained models incur ~10–15% clean accuracy degradation on CIFAR-10. This is not an artifact of imperfect training. The Bayes-optimal robust classifier genuinely uses fewer non-robust features, and those features carry real predictive signal under the standard distribution.

---

### TRADES — Better Pareto Efficiency (Zhang et al., ICML 2019)

Madry AT trains on adversarial examples labeled with the true class. This conflates two objectives: learning correct features, and being stable under perturbation.

TRADES decomposes the robust error explicitly:

**Robust error = Natural error + Boundary error**

```
min_θ  E[ L(f_θ(x), y) ]  +  (1/λ) · E[ max_{x': ‖x'-x‖≤ε} KL(f_θ(x) ‖ f_θ(x')) ]
```

The second term penalizes output instability within the ε-ball — without requiring the adversarial example to be correctly labeled. This separation gives TRADES a better position on the accuracy-robustness Pareto frontier.

```python
def trades_loss(model, x, y, epsilon, alpha, num_steps, beta):
    model.eval()
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
    loss_natural = F.cross_entropy(model(x), y)
    loss_robust = F.kl_div(
        F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(x).detach(), dim=1),
        reduction='batchmean'
    )
    return loss_natural + beta * loss_robust
```

---

### Randomized Smoothing — Certified Defenses (Cohen et al., ICML 2019)

Adversarial training provides empirical robustness — it resists known attacks — but cannot certify that no attack exists within the ε-ball. Randomized smoothing provides a provable L2 radius.

The idea: wrap any base classifier `f` with Gaussian noise averaging:

```
g(x) = argmax_c  P_{ε ~ N(0, σ²I)}[ f(x + ε) = c ]
```

`g` returns the class `f` most often predicts when the input is corrupted by Gaussian noise. If the top-class probability is `p_A` and `p_A > 0.5`, the certified L2 radius is:

```
R = σ · Φ⁻¹(p_A)
```

Any perturbation within this radius cannot change `g`'s prediction — provably.

```python
import torch
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def smooth_predict(model, x, sigma, num_samples, batch_size, device):
    model.eval()
    counts = None
    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples)
            noise = torch.randn(n, *x.shape, device=device) * sigma
            noisy = (x.unsqueeze(0) + noise).clamp(0.0, 1.0)
            preds = model(noisy).argmax(dim=1)
            if counts is None:
                counts = torch.bincount(preds, minlength=model(noisy).shape[1])
            else:
                counts += torch.bincount(preds, minlength=counts.shape[0])
    return counts

def certify(model, x, sigma, num_samples, alpha, device):
    counts = smooth_predict(model, x, sigma, num_samples, batch_size=500, device=device)
    top2 = counts.topk(2)
    n_A = top2.values[0].item()
    c_A = top2.indices[0].item()
    p_A_low, _ = proportion_confint(n_A, num_samples, alpha=2 * alpha, method='beta')
    if p_A_low > 0.5:
        return c_A, sigma * norm.ppf(p_A_low)
    return -1, 0.0  # abstain
```

Higher σ → larger certified radius, lower clean accuracy. Other certified methods: Interval Bound Propagation (IBP) and CROWN/α-CROWN compute tighter bounds for L∞ via linear relaxations of network layers.

---

### Practical Defenses (No Formal Guarantees)

Input preprocessing destroys some adversarial structure before it reaches the model:

```python
from PIL import Image
import io, numpy as np

def jpeg_compress(x_np, quality=75):
    img = Image.fromarray((x_np * 255).astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf)).astype('float32') / 255.0

def bit_depth_reduction(x, bits=4):
    max_val = 2 ** bits - 1
    return torch.round(x * max_val) / max_val
```

Detection: run model on original and preprocessed input; large output disagreement flags adversarial input.

```python
def detect_adversarial(model, x, squeezed_x, threshold=0.05):
    with torch.no_grad():
        p1 = F.softmax(model(x), dim=1)
        p2 = F.softmax(model(squeezed_x), dim=1)
        distance = (p1 - p2).abs().max(dim=1).values
    return distance > threshold
```

---

### Benchmarks — RobustBench

`https://robustbench.github.io` — standardized leaderboard evaluated with AutoAttack under fixed threat models.

Standard threat models: CIFAR-10 L∞ ε = 8/255, CIFAR-100 L∞ ε = 8/255, ImageNet L∞ ε = 4/255.

State of the art (2024): ~71–73% robust accuracy on CIFAR-10 L∞ vs. ~95%+ clean. Always report both.

---

### Transferability and Black-Box Attacks

Adversarial examples crafted on model A often fool model B — different architectures, different training data. Different models learn similar decision boundary geometry in the input manifold. Perturbations exploit the data manifold, not model-specific artifacts.

| Setting | Attacker Knows | Method |
|---------|---------------|--------|
| **White-box** | Architecture, weights, gradients | PGD, C&W, AutoAttack |
| **Gray-box** | Architecture, not weights | Transfer from surrogate |
| **Black-box (score)** | Output probabilities | Square Attack, NES |
| **Black-box (decision)** | Hard label only | Boundary Attack, HopSkipJump |

Ensemble attacks improve transferability by exploiting shared geometry:

```python
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

## 4. What Breaks

**The robustness-accuracy tradeoff is fundamental, not a training artifact.** The Bayes-optimal robust classifier differs from the standard one. Non-robust features carry real predictive signal under the clean distribution. Removing them costs accuracy. No training method eliminates this gap.

**Gradient masking is not robustness.** Defenses that make gradients vanish (saturated activations, non-differentiable preprocessing) appear robust against gradient-based attacks but are not. AutoAttack specifically targets this: Square Attack needs no gradients at all; DLR loss defeats output-space rescaling. Never evaluate a defense only against the attack it was designed to defeat.

**BPDA breaks preprocessing defenses.** For non-differentiable preprocessors: use the preprocessor in the forward pass but substitute a differentiable approximation in the backward pass. PGD runs end-to-end through the approximation. JPEG compression, feature squeezing, and bit-depth reduction are all broken by adaptive attacks using this technique.

**Adversarial training is computationally expensive.** Inner PGD adds 10–40× compute over standard training. This is the primary reason most production models are not adversarially trained.

**Randomized smoothing has fundamental limits at high accuracy.** At σ = 0.5 on ImageNet, clean accuracy falls to ~50%. The accuracy-radius tradeoff is architectural, not tunable away. Smoothing does not extend cleanly to L∞ — the certified L∞ radius via smoothing is trivial.

**Transferability means black-box is not safe.** An attacker who trains a surrogate on the same distribution produces adversarial examples without access to your weights. Score-based attacks (Square Attack) require no surrogate at all — just query access.

---

## Key Interview Points

- Adversarial examples arise because models use non-robust features — directions in pixel space that correlate with labels but are imperceptible to humans. High-dimensional gradient descent on pixel-space features makes this structural, not accidental.
- FGSM is one step in the gradient sign direction. PGD is iterated FGSM with projection and random start. C&W minimizes perturbation norm directly. PGD is the canonical white-box attack and the inner loop of adversarial training.
- AutoAttack is the evaluation standard: four attacks including the gradient-free Square Attack. Results against FGSM alone or single-run PGD are not trustworthy.
- Adversarial training solves `min_θ E[max_{δ} L(f_θ(x+δ), y)]` — inner PGD, outer SGD. Always costs clean accuracy.
- TRADES decomposes robust error into natural + boundary components. KL regularization between clean and adversarial outputs achieves better Pareto efficiency than Madry AT.
- Randomized smoothing is the only method with provable L2 guarantees at scale. Certified radius R = σ·Φ⁻¹(p_A). Higher σ means larger radius but lower clean accuracy.
- Gradient masking is not robustness. AutoAttack specifically targets it via Square Attack and DLR loss.
- State-of-the-art: ~70–73% robust accuracy on CIFAR-10 L∞ ε=8/255 vs. ~95%+ clean. Report both.

---

## Canonical Interview Q&As

**Q1: Derive why adversarial examples are an inherent property of high-dimensional geometry, not a training artifact.**

In d-dimensional space, the L∞ ball of radius ε contains `(2ε/Δ)^d` non-overlapping L∞ balls of radius Δ/2, where Δ is the inter-class separation. For d = 3072 (CIFAR-10 32×32×3) with ε = 8/255 ≈ 0.031:

The human-perceptual manifold occupies a tiny fraction of the full input space. A classifier trained with ERM finds a decision boundary that correctly classifies natural images but makes no commitments about the vast volume of space between them and imperceptible perturbations.

Formally: let f be a Bayes-optimal classifier for the natural distribution. The robust Bayes classifier `f_rob` solving:

```
f_rob = argmin_f  E[max_{||δ||≤ε} L(f(x+δ), y)]
```

is a different function from `f`. The two diverge because the natural distribution has positive mass on "non-robust features" — directions in input space that are highly predictive for clean accuracy but vary across the ε-ball. A model that uses non-robust features will be accurate on clean data and adversarially vulnerable by construction.

Empirical evidence: Ilyas et al. (2019) trained a model on adversarially perturbed images relabeled to the adversarial target class. This "adversarially corrupted" dataset produces a model that correctly classifies natural images with >50% accuracy — proving the perturbations contain genuinely predictive features (non-robust), not just noise.

---

**Q2: AutoAttack comprises four attacks. Explain why each is necessary and what specific defense weaknesses each targets.**

```
AutoAttack = {
  APGD-CE,      # Adaptive PGD with cross-entropy loss
  APGD-DLR,     # Adaptive PGD with Difference of Logits Ratio loss
  FAB,          # Fast Adaptive Boundary attack (minimal perturbation)
  Square Attack # Score-based black-box, gradient-free
}
```

**APGD-CE** targets standard gradient-based defenses. Adaptive step size (restarts from best checkpoint, reduces step when no progress) defeats defenses that use stochastic smoothing to make individual PGD steps ineffective.

**APGD-DLR** uses a scale-invariant loss `(z_y - max_{i≠y} z_i) / (z_y - z_3rd)`. It defeats output-space rescaling defenses: if a defense adds a temperature or normalization to outputs, CE loss gradients become saturated or misleading, but DLR is invariant to output scaling.

**FAB** minimizes the actual perturbation norm rather than maximizing loss. It finds the minimal adversarial example, which is important for evaluating certified defenses and detecting when a model is relying on geometric margin inflation rather than true robustness.

**Square Attack** is score-based and requires only output probabilities. It defeats gradient masking defenses entirely — if a defense obfuscates or zeroes out gradients (non-differentiable preprocessing, randomization), the first three attacks fail. Square Attack sidesteps this by estimating optimal perturbations through random square-shaped queries without any gradient access.

A defense is only credible if it withstands all four simultaneously.

---

**Q3: Your team is deploying a computer vision model to detect manufacturing defects. Adversarial robustness is a requirement. Design the training and evaluation pipeline.**

**Threat model specification (required first):**
- Attacker type: white-box (assume attacker has model access; if internal tool, gray-box is realistic)
- Norm constraint: L∞ ε = 4/255 (pixel-level perturbations), not L2 (physical attacks would use L∞)
- Target: untargeted misclassification (defect → non-defect is the high-cost error direction)

**Training pipeline:**

```python
# Adversarial training (Madry AT)
def train_step(model, x, y, optimizer, eps=4/255, steps=10, alpha=1/255):
    # Inner maximization: find worst-case perturbation
    delta = torch.zeros_like(x).uniform_(-eps, eps).requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x + delta.clamp(-eps, eps)), y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)
            delta.grad.zero_()
    # Outer minimization: update model on adversarial examples
    adv_loss = F.cross_entropy(model(x + delta.detach()), y)
    adv_loss.backward()
    optimizer.step()
```

**TRADES variant** for better accuracy-robustness tradeoff:
```
L_TRADES = L_CE(f(x), y) + β · KL(f(x) || f(x+δ*))
```
Empirically: TRADES with β=6 achieves ~56% robust / ~84% clean on CIFAR-10 vs. Madry AT at ~45% robust / ~87% clean. The extra 11% robust accuracy matters for defect detection.

**Evaluation:** Run AutoAttack (all 4 components) on 1000 held-out samples. Report clean accuracy and robust accuracy separately. Never report only adversarially-trained clean accuracy — it looks weaker, but honest.

**Production additions:**
- Input preprocessing: JPEG compression at quality 75 + bit-depth reduction to 7 bits adds ~3% free robustness via input diversity.
- Ensemble of 3 independently trained robust models — transferability across all three is harder to achieve than fooling a single model.
- Monitor for distribution shift in production inputs; adversarially-trained models can be more sensitive to legitimate domain shift because the decision boundaries are tighter.

---

**Q4: Explain the certified guarantee of randomized smoothing. For a Gaussian noise level σ = 0.25, what L2 perturbation radius is certified at 99% confidence?**

Randomized smoothing wraps any base classifier `f` in a smoothed classifier `g`:

```
g(x) = argmax_c  P(f(x + ε) = c),  ε ~ N(0, σ²I)
```

**Certification theorem (Cohen et al. 2019):** If `g(x) = c_A` and:
```
p_A = P(f(x + ε) = c_A) ≥ p̄_A
p_B = max_{c ≠ c_A} P(f(x + ε) = c) ≤ p̄_B
```

then `g` is robust within L2 radius:
```
R = (σ/2) · (Φ⁻¹(p̄_A) - Φ⁻¹(p̄_B))
```

where Φ⁻¹ is the inverse normal CDF.

**Concrete calculation at 99% confidence:**
- Run N = 100,000 Monte Carlo samples of `f(x + ε_i)` for ε_i ~ N(0, 0.25²I)
- Estimate `p̄_A` via one-sided Clopper-Pearson interval at 99.5% confidence
- Suppose `k_A = 90,000` votes for class A → `p̄_A ≈ 0.899` (lower bound)
- `p̄_B ≤ 1 - p̄_A = 0.101`
- `R = 0.25/2 · (Φ⁻¹(0.899) - Φ⁻¹(0.101))`
- `Φ⁻¹(0.899) ≈ 1.278`, `Φ⁻¹(0.101) ≈ -1.278`
- `R = 0.125 · 2.556 ≈ 0.32`

This is a **provable** L2 robustness certificate — no adversary, regardless of algorithm, can construct an L2 perturbation of magnitude ≤ 0.32 that changes the output of `g` at input `x`.

Key limitation: this is an L2 certificate. Converting to L∞ (the practically relevant norm for pixel attacks) gives R_∞ = R/√d ≈ 0.32/√3072 ≈ 0.006 — essentially zero. Randomized smoothing does not provide useful L∞ certificates.

## Flashcards

**Adversarial examples arise because models use non-robust features?** #flashcard
directions in pixel space that correlate with labels but are imperceptible to humans. High-dimensional gradient descent on pixel-space features makes this structural, not accidental.

**FGSM is one step in the gradient sign direction. PGD is iterated FGSM with projection and random start. C&W minimizes perturbation norm directly. PGD is the canonical white-box attack and the inner loop of adversarial training.?** #flashcard
FGSM is one step in the gradient sign direction. PGD is iterated FGSM with projection and random start. C&W minimizes perturbation norm directly. PGD is the canonical white-box attack and the inner loop of adversarial training.

**AutoAttack is the evaluation standard?** #flashcard
four attacks including the gradient-free Square Attack. Results against FGSM alone or single-run PGD are not trustworthy.

**Adversarial training solves min_θ E[max_{δ} L(f_θ(x+δ), y)]?** #flashcard
inner PGD, outer SGD. Always costs clean accuracy.

**TRADES decomposes robust error into natural + boundary components. KL regularization between clean and adversarial outputs achieves better Pareto efficiency than Madry AT.?** #flashcard
TRADES decomposes robust error into natural + boundary components. KL regularization between clean and adversarial outputs achieves better Pareto efficiency than Madry AT.

**Randomized smoothing is the only method with provable L2 guarantees at scale. Certified radius R = σ·Φ⁻¹(p_A). Higher σ means larger radius but lower clean accuracy.?** #flashcard
Randomized smoothing is the only method with provable L2 guarantees at scale. Certified radius R = σ·Φ⁻¹(p_A). Higher σ means larger radius but lower clean accuracy.

**Gradient masking is not robustness. AutoAttack specifically targets it via Square Attack and DLR loss.?** #flashcard
Gradient masking is not robustness. AutoAttack specifically targets it via Square Attack and DLR loss.

**State-of-the-art?** #flashcard
~70–73% robust accuracy on CIFAR-10 L∞ ε=8/255 vs. ~95%+ clean. Report both.

**Attacker type?** #flashcard
white-box (assume attacker has model access; if internal tool, gray-box is realistic)

**Norm constraint?** #flashcard
L∞ ε = 4/255 (pixel-level perturbations), not L2 (physical attacks would use L∞)

**Target?** #flashcard
untargeted misclassification (defect → non-defect is the high-cost error direction)

**Input preprocessing?** #flashcard
JPEG compression at quality 75 + bit-depth reduction to 7 bits adds ~3% free robustness via input diversity.

**Ensemble of 3 independently trained robust models?** #flashcard
transferability across all three is harder to achieve than fooling a single model.

**Monitor for distribution shift in production inputs; adversarially-trained models can be more sensitive to legitimate domain shift because the decision boundaries are tighter.?** #flashcard
Monitor for distribution shift in production inputs; adversarially-trained models can be more sensitive to legitimate domain shift because the decision boundaries are tighter.

**Run N = 100,000 Monte Carlo samples of f(x + ε_i) for ε_i ~ N(0, 0.25²I)?** #flashcard
Run N = 100,000 Monte Carlo samples of f(x + ε_i) for ε_i ~ N(0, 0.25²I)

**Estimate p̄_A via one-sided Clopper-Pearson interval at 99.5% confidence?** #flashcard
Estimate p̄_A via one-sided Clopper-Pearson interval at 99.5% confidence

**Suppose k_A = 90,000 votes for class A → p̄_A ≈ 0.899 (lower bound)?** #flashcard
Suppose k_A = 90,000 votes for class A → p̄_A ≈ 0.899 (lower bound)

**p̄_B ≤ 1 - p̄_A = 0.101?** #flashcard
p̄_B ≤ 1 - p̄_A = 0.101

**R = 0.25/2 · (Φ⁻¹(0.899) - Φ⁻¹(0.101))?** #flashcard
R = 0.25/2 · (Φ⁻¹(0.899) - Φ⁻¹(0.101))

**Φ⁻¹(0.899) ≈ 1.278, Φ⁻¹(0.101) ≈ -1.278?** #flashcard
Φ⁻¹(0.899) ≈ 1.278, Φ⁻¹(0.101) ≈ -1.278

**R = 0.125 · 2.556 ≈ 0.32?** #flashcard
R = 0.125 · 2.556 ≈ 0.32
