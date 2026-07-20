---
module: Responsible AI
topic: Adversarial Robustness
subtopic: Attacks and Defenses
status: unread
tags: [adversarial, robustness, fgsm, pgd, security, responsible-ai]
prerequisites: [optimization, neural-networks, model-evaluation]
---
# Adversarial Robustness

**The problem**: a model with 99% test accuracy misclassifies an input that a human sees as
unchanged. Add a perturbation invisible at 8-bit display precision, and a stop sign reads as a
speed limit sign with high confidence. This is not an accuracy problem — it is a *worst-case*
problem, and standard test accuracy is an average-case measurement that cannot see it.

**The core insight**: training minimizes expected loss over the data distribution. Nothing in
that objective says anything about the loss at points *near* the data. In high dimensions there
is an enormous amount of space near every training point, and gradient descent never visits it.
Adversarial examples live there.

> **Scope note.** This file covers adversarial examples against a *deployed model* — evasion at
> inference. For LLM-specific attacks (jailbreaks, prompt injection), see
> [`../10-llms/interview-notes/18-production-alignment-failures.md`](../10-llms/interview-notes/18-production-alignment-failures.md).
> For adversarial *training* as a fairness technique, see
> [`01-privacy-and-fairness.md`](01-privacy-and-fairness.md) §9 — same word, different goal.

---

## 1. The threat model comes first

You cannot say a model is "robust" without stating against what. Interviewers listen for whether
you specify this unprompted.

| Dimension | Options |
| :--- | :--- |
| **Attacker knowledge** | White-box (weights, gradients) / black-box (query only) / transfer (surrogate model) |
| **Perturbation budget** | $\ell_\infty$ ($\|\delta\|_\infty \le \epsilon$, every pixel a little) / $\ell_2$ / $\ell_0$ (few pixels, any amount) / semantic (rotation, weather, patch) |
| **Attack goal** | Untargeted (any wrong class) / targeted (a *specific* wrong class) |
| **Attack timing** | Evasion (inference) / poisoning (training) / model extraction |

$\ell_\infty$ with $\epsilon = 8/255$ on images is the academic default. It is a *proxy* for
imperceptibility, not a definition of it — and real attackers are not constrained by it.

---

## 2. Attacks

### FGSM — one step

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

Take the gradient of the loss *with respect to the input*, move every pixel by $\epsilon$ in
whichever direction increases loss. One backward pass.

The `sign` is what makes it an $\ell_\infty$ attack: every coordinate moves by exactly
$\epsilon$, so the maximum per-pixel change is bounded regardless of gradient magnitude.

```python
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()
```

### PGD — iterative, and the one that matters

FGSM takes one step along a linear approximation. PGD iterates, projecting back into the
$\epsilon$-ball each time:

$$x^{t+1} = \Pi_{\mathcal{B}_\epsilon(x)}\left(x^t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x^t, y))\right)$$

with random initialization inside the ball to avoid degenerate starting points.

```python
def pgd(model, x, y, eps, alpha, steps):
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)   # random start
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = x + (x_adv - x).clamp(-eps, eps)          # project to eps-ball
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()
```

**PGD is the standard benchmark** because it is a strong first-order attack and empirically
close to the worst case within the $\ell_\infty$ ball. "Robust accuracy" in a paper means
accuracy under PGD unless stated otherwise. Evaluating only against FGSM is the classic way to
publish a defense that doesn't work.

### Transfer and black-box

Adversarial examples transfer across architectures trained on the same data — a genuinely
surprising empirical fact, and the reason black-box attacks are practical. The attacker trains
a surrogate, attacks it white-box, and the example often fools the target. **No gradient access
is required to attack a deployed model.**

---

## 3. Defenses

### Adversarial training — the only one that reliably works

Replace ERM with a min-max objective — train on the worst case within the ball, not the average:

$$\min_\theta \; \mathbb{E}_{(x,y)}\left[\max_{\|\delta\| \le \epsilon} \mathcal{L}(\theta, x+\delta, y)\right]$$

Implementation: each batch, generate PGD examples and train on those. The inner maximization is
why this costs 5–10× standard training — every step runs a full PGD loop.

**The robustness/accuracy tradeoff is real and appears to be fundamental**, not an artifact of
poor tuning. On CIFAR-10, a standard model gets ~95% clean / ~0% robust; adversarially trained,
~87% clean / ~50% robust. You buy robustness with clean accuracy, and that is a product
decision.

### Certified defenses

Adversarial training gives *empirical* robustness — robust against known attacks. Certification
gives a *proof* that no perturbation within a radius changes the prediction. Randomized
smoothing is the scalable version: classify by majority vote over Gaussian-noised copies, which
yields a certified $\ell_2$ radius. Guarantees are real but the certified radii are small, and
inference costs many forward passes.

### What does not work

**Gradient masking / obfuscation** is the recurring failure mode. Defenses that make gradients
uninformative — input discretization, non-differentiable preprocessing, randomization — appear
to work because gradient-based attacks fail on them. They are broken by BPDA (approximate the
non-differentiable part with the identity on the backward pass) or by transfer attacks that
never needed the gradient. A long series of published defenses fell this way. **If a defense
reduces attack success without an accuracy cost, suspect gradient masking first.**

Distillation-based defenses, most input transformations, and detection schemes have similar
histories.

---

## 4. Production notes

- **Threat-model before mitigating.** For most business ML the realistic adversary manipulates
  *semantic* features (transaction structuring, keyword obfuscation), not $\ell_\infty$ pixel
  noise. $\ell_\infty$ robustness may be irrelevant to your actual attack surface.
- **Rate-limit and monitor queries.** Black-box attacks need many queries. Query-pattern
  monitoring is often more cost-effective than adversarial training.
- **Adversarial retraining loops.** Where the adversary adapts (fraud, spam, abuse), robustness
  is an ongoing process, not a training-time property. Budget for continuous retraining.
- **Measure robustness explicitly.** Standard evaluation cannot detect this. Add PGD accuracy to
  the eval suite if it matters, or you will not know it regressed.

---

## Interview Angles

### Q: Why do adversarial examples exist at all? [Easy]

Training minimizes *average* loss on the data distribution. It says nothing about loss at nearby
points, and in high dimensions there is vast unvisited space adjacent to every training point.
Models learn features that are predictive on-distribution but not robust to small perturbations
in directions the training data never constrained.

**Cross-questions to expect**
- *"Is it overfitting?"* — No. It reproduces on models that generalize well, and more data
  doesn't fix it.
- *"Bug or feature?"* — There's evidence they arise from genuinely predictive but non-robust
  features — the model is using real signal humans can't perceive.

**Trap:** "The model didn't learn the true function." True but vacuous. The question is why the
failure is systematic and directed rather than random noise.

### Q: A paper reports 95% robust accuracy with no clean-accuracy loss. Believe it? [Medium]

No — that combination is the signature of gradient masking. The robustness/accuracy tradeoff is
consistently observed; a defense claiming to escape it usually isn't defending, it's breaking
the attacker's gradient.

**Cross-questions to expect**
- *"How would you check?"* — Attack with BPDA; run transfer attacks from an undefended surrogate;
  verify attack success increases with PGD steps (if it plateaus early, gradients are masked);
  check for adaptive attacks in the evaluation.
- *"What would convince you?"* — Evaluation against attacks designed specifically for that
  defense, plus a certified bound.

**Trap:** Accepting FGSM-only evaluation. Insufficient by itself.

### Q: Your fraud model is being evaded. The security team wants adversarial training. [Hard]

Push back on the threat model first. Adversarial training defends an $\ell_p$ ball in *feature
space* — imperceptible numeric perturbations. A fraud adversary doesn't do that; they change
real-world behaviour: splitting transactions below thresholds, aging accounts, cycling devices.
Those are large, semantically meaningful moves, not small-norm ones. $\ell_\infty$ adversarial
training defends against the wrong thing, at 5–10× training cost.

What actually applies: identify which features are attacker-controllable versus costly to fake
(device fingerprint, account age, network position), weight the model toward the costly ones,
add velocity/aggregate features that are hard to manipulate without changing real behaviour, and
build the retraining loop — because the adversary adapts continuously.

**Cross-questions to expect**
- *"Is adversarial training ever right here?"* — If part of the pipeline takes raw perceptual
  input (document images, face verification), yes, for that component.
- *"How do you measure success?"* — Not robust accuracy. Attacker *cost*, and evasion rate over
  time — the metric has to be dynamic because the adversary is.
- *"What about the feature that's most predictive but easiest to fake?"* — That is exactly the
  feature to downweight. Predictive power measured on historical data is not predictive power
  against an adversary who knows you use it.

**Trap:** Applying the academic framing because it's the one with the well-known name. The
interviewer is testing whether you match the defense to the actual adversary.

---

## Connections

- [../04-evaluation/05-model-interpretation.md](../04-evaluation/05-model-interpretation.md) — the other half of "can I trust this model"
- [01-privacy-and-fairness.md](01-privacy-and-fairness.md) §9 — adversarial *training* for debiasing; §11 membership inference as a different attack surface
- [../15-system-design/cases/05-fraud-detection.md](../15-system-design/cases/05-fraud-detection.md) — the adaptive-adversary setting end to end
- [../10-llms/interview-notes/18-production-alignment-failures.md](../10-llms/interview-notes/18-production-alignment-failures.md) — jailbreaks and prompt injection
