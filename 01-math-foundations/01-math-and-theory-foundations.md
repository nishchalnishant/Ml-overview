---
module: Math Foundations
topic: Math And Theory Foundations
subtopic: ""
status: unread
tags: [foundations, ml, math-and-theory-foundations]
---
# Math and Theory Foundations

**How to use this:** This is the *learning* pass, not the drill. Each section builds the intuition for why a piece of math exists and what it does to your data — enough that the formulas feel inevitable rather than memorized. When you want the derivations, the identities, and the rapid-fire recall, follow the **→ Drill** links into the interview-prep track. Read this file top to bottom once; return to the drill files repeatedly.

---

## Why Math Shows Up in ML at All

Three questions run underneath every model you will ever build:

1. **What shape is my data, and what does this transformation do to it?** — linear algebra
2. **How confident should I be in what I just measured?** — probability and statistics
3. **How do I move the parameters to make the loss smaller?** — optimization

Almost every piece of ML math is an answer to one of those. Linear algebra gives you the vocabulary for data as geometry. Probability gives you the vocabulary for uncertainty. Optimization gives you the vocabulary for improvement. A model is what happens when you point all three at the same dataset.

The reason interviews probe this layer is not that you will derive backprop at work — autograd does it. It is that the failure modes *only* make sense at this layer. Exploding gradients, singular covariance matrices, a model that is confidently wrong, an A/B test that says "significant" and means nothing: each is a math problem wearing an engineering costume.

---

## 1. Linear Algebra — Data as Geometry

**The reframe:** A dataset is a cloud of points in high-dimensional space. A row is a point. A column is an axis. A matrix multiplication is not "a bunch of multiply-adds" — it is a *transformation of that space*: rotate it, stretch it along certain directions, squash it into fewer dimensions, project it onto a subspace.

Once you hold that picture, a lot of ML collapses into one sentence: **most models are learning which directions in your feature space actually matter.**

**Why eigenvectors are the punchline.** Most transformations rotate vectors. Eigenvectors are the special directions that a given transformation *doesn't* rotate — it only scales them, and the eigenvalue is how much. That makes them the natural coordinate system for the transformation: in eigenvector coordinates, a messy matrix becomes a simple list of stretch factors.

This is why PCA works. The covariance matrix encodes how your features vary together; its eigenvectors are the directions of greatest variance; keeping the top-k gives you the subspace where your data actually lives. When someone says "our 500 features are really about 12 things," they are making a claim about the eigenvalue spectrum.

**Why SVD is the more useful cousin.** Eigendecomposition needs a square matrix. Real data isn't square — you have n rows and d columns. SVD generalizes the idea to any matrix, which is why it underpins PCA on raw data, low-rank approximation, matrix factorization for recommenders, and LoRA's core trick of approximating a weight update with two skinny matrices.

**Where it bites in production.** A singular (or near-singular) matrix means your features are linearly dependent — some column is a combination of others. Linear regression's normal equation requires inverting `XᵀX`; perfect multicollinearity makes that inversion undefined and near-collinearity makes it numerically unstable, which is one motivation for ridge regularization: adding `λI` to the diagonal guarantees invertibility. Condition number is the diagnostic worth knowing by name.

**→ Drill:** [07-interview-prep/ml/19-maths.md](02-maths.md) — eigenvalues, SVD, the PCA/SVD connection, Jacobian and Hessian, PSD matrices, L1/L2 norms.
**→ Applied:** [02-classical-ml/10-dimensionality-reduction.md](../03-classical-ml/05-dimensionality-reduction.md)

---

## 2. Probability and Inference — Reasoning Under Uncertainty

**The reframe:** A model does not output answers. It outputs *beliefs*. Every training objective you use is a statement about which beliefs count as good ones, and nearly all of them reduce to one principle: **maximize the likelihood of the data you actually observed.**

That principle is doing more work than it appears. Minimizing squared error is maximum likelihood under Gaussian noise. Minimizing cross-entropy is maximum likelihood under a categorical distribution. These are not two loss functions that happen to work — they are the same idea applied to different assumptions about how your data was generated. If you can say that sentence in an interview, you have demonstrated more than a list of losses could.

**MLE vs MAP, and why regularization is a prior.** Maximum likelihood asks: which parameters make the observed data most probable? Maximum a posteriori adds a prior belief about the parameters themselves and asks which parameters are most probable *given* the data. Work through the algebra and L2 regularization falls out as a Gaussian prior on the weights, L1 as a Laplace prior. Regularization isn't a hack bolted onto the loss; it's what happens when you admit you had an opinion before seeing the data.

**Bayes' theorem as the update rule.** Prior belief, meet evidence, produce posterior belief. The classic interview trap — a 99%-accurate test for a disease with 1-in-10,000 prevalence still yields mostly false positives — is not a trick question. It is the base-rate problem that makes fraud detection, medical triage, and anomaly detection hard, and it is why precision collapses on imbalanced data no matter how good your classifier's accuracy looks.

**Why sampling distributions matter more than distributions.** The Central Limit Theorem says that averages of independent samples tend toward a normal distribution regardless of the underlying shape. That is the license for nearly every confidence interval and hypothesis test you will run. It is also why "my metric moved 2%" means nothing until you know the variance of that estimate — the single most common way ML teams fool themselves.

**Where it bites in production.** A p-value is the probability of seeing data this extreme *if the null hypothesis were true* — it is not the probability that your model is better. Peeking at an A/B test until it turns significant inflates the false positive rate. Testing twenty metrics and reporting the one that moved is the same error wearing a different hat. Multiple-testing corrections exist because this failure is so easy and so tempting.

**→ Drill:** [07-interview-prep/ml/24-statistics-probability-rapid-fire.md](_rapid-fire.md) — distributions, Bayes, MLE/MAP, hypothesis testing, confidence intervals, CLT, A/B testing.
**→ Traps:** [07-interview-prep/ml/11-canonical-stats-questions.md](05-canonical-stats-questions.md) — multiple testing, Simpson's paradox, causal design.
**→ Applied:** [02-classical-ml/14-calibration-and-uncertainty.md](../04-evaluation/04-calibration-and-uncertainty.md)

---

## 3. Optimization — How Models Actually Learn

**The reframe:** Training is search. You have a loss surface over parameter space, you are somewhere on it, and you want to get lower. The gradient is the local direction of steepest ascent, so you step against it. Everything else — momentum, adaptive learning rates, schedules — is engineering around the fact that this simple picture is hostile in high dimensions.

**Why convexity is the dividing line.** A convex loss has one minimum; find it and you are done, and you can prove you are done. Linear and logistic regression are convex, which is why they are reliable and why classical ML feels stable. Neural networks are not convex — the surface has saddle points, ravines, and many minima. In high dimensions the enemy is mostly saddle points rather than local minima, which reframes what momentum is for: not escaping bad minima, but building velocity through flat regions and damping oscillation across narrow ravines.

**What adaptive optimizers actually fix.** Plain SGD applies one learning rate to every parameter. But a parameter attached to a rare feature gets a gradient signal a thousand times less often than one attached to a common feature — the same step size cannot be right for both. Adam tracks per-parameter running averages of the gradient and its square, then normalizes the step by the second one. Frequently-updated parameters get damped; rarely-updated ones get amplified.

The AdamW correction is worth understanding rather than memorizing: standard Adam routes weight decay through the gradient, where the adaptive normalizer then rescales it, so your regularization strength silently varies per parameter. Decoupling the decay from the gradient restores the intended behavior. This is the kind of detail that separates "I use AdamW" from "I know why AdamW exists."

**Constrained optimization, briefly.** When you need to optimize subject to constraints, Lagrange multipliers convert the constrained problem into an unconstrained one; KKT conditions generalize this to inequalities. This is the machinery behind the SVM dual formulation and the kernel trick — the main place in classical ML where you meet it head-on.

**Where it bites in production.** Vanishing and exploding gradients are chain-rule arithmetic: multiply many numbers below one and you approach zero; above one and you diverge. Residual connections, normalization layers, careful initialization, and gradient clipping are all responses to that single multiplication problem. When a training run diverges at step 400, you are debugging calculus.

**→ Drill:** [07-interview-prep/ml/03-optimization.md](06-optimization.md) — gradient descent, batch size, learning rate, momentum, Adam/AdamW, vanishing gradients, LR scheduling, optimizer selection.
**→ Applied:** [03-deep-learning/components/06-optimisers.md](../05-deep-learning-core/06-optimisers.md) · [03-deep-learning/components/01-backpropagation.md](../05-deep-learning-core/01-backpropagation.md)

---

## 4. Information Theory — The Glue

Information theory is where the other three meet, and it explains why your loss function looks the way it does.

**Entropy** measures uncertainty in a distribution: uniform is maximum entropy, a point mass is zero. **Cross-entropy** measures the cost of encoding data from distribution `p` using a code optimized for distribution `q` — which is precisely what you are doing when your model predicts `q` and reality is `p`. **KL divergence** is the excess cost of that mismatch, cross-entropy minus entropy.

Because the true distribution's entropy is fixed by the data, minimizing cross-entropy and minimizing KL divergence are the same optimization. That is the connective tissue: classification loss, the VAE's evidence lower bound, knowledge distillation, and RLHF's KL penalty against the reference policy are all the same quantity doing different jobs.

Note the asymmetry — `KL(p‖q) ≠ KL(q‖p)` — and that it has real consequences. One direction is mode-covering, the other mode-seeking, which is why the choice matters in variational inference.

**→ Drill:** [07-interview-prep/ml/19-maths.md](02-maths.md) (§7 KL Divergence, §9 Information Theory)
**→ Derivation:** [07-interview-prep/ml/18-math-derivations.md](03-math-derivations.md) (§10 ELBO, §11 KL and cross-entropy)

---

## 5. The Derivations Worth Being Able to Whiteboard

Interviewers rarely ask you to derive something obscure. They ask for a small set of derivations that prove you understand the chain rule and the shape of a loss surface. In rough order of how often they appear:

| Derivation | What it proves you understand |
| :--- | :--- |
| Logistic regression gradient | The link between sigmoid, cross-entropy, and why the gradient is so clean |
| Backpropagation through a 2-layer net | Chain rule mechanics; where vanishing gradients come from |
| Softmax gradient | Jacobian structure; why the diagonal and off-diagonal terms differ |
| Normal equation for linear regression | Why `XᵀX` must be invertible; the multicollinearity connection |
| L2 regularization as weight decay | Regularization as a prior; the MAP connection |
| Scaled dot-product attention, and why `√dₖ` | Variance control; softmax saturation |
| Bias–variance decomposition | Why more capacity is not free |

All but the last are worked line by line in the derivations file, which also carries a whiteboard template for structuring your answer under pressure. Bias–variance lives with the algorithms instead.

**→ Full derivations:** [07-interview-prep/ml/18-math-derivations.md](03-math-derivations.md)
**→ Bias–variance:** [07-interview-prep/ml/08-algorithms.md](../03-classical-ml/03-algorithms.md)

---

## Self-Check

If you can answer these without notes, this file has done its job:

1. Why does PCA use the eigenvectors of the covariance matrix, and what do the eigenvalues tell you?
2. Why is minimizing squared error equivalent to maximum likelihood, and what assumption does that require?
3. What exactly does a p-value measure — and what does it *not* measure?
4. Why does Adam need bias correction in its first few steps?
5. Why is minimizing cross-entropy the same as minimizing KL divergence?
6. Your training loss diverges at step 400. Name three causes and the math behind each.

---

## Where to Next

- **Vocabulary for everything above** → [06-ml-glossary.md](../18-resources/02-ml-glossary.md)
- **Putting it to work on real algorithms** → [02-classical-ml/](../02-classical-ml/)
- **Where the calculus becomes backprop** → [03-deep-learning/components/01-backpropagation.md](../05-deep-learning-core/01-backpropagation.md)
- **Rapid-fire recall before an interview** → [07-interview-prep/ml/](../07-interview-prep/ml/)
