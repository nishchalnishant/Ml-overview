---
module: Math Foundations
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, math-foundations, linear-algebra, probability, statistics, optimization, learning-theory, information-theory]
prerequisites: [01-math-foundations]
---
# 01 Math Foundations — Revision Guide

Single-page revision for the entire `01-math-foundations/` folder: linear algebra, derivations, probability & statistics, canonical stats questions, optimization (practical + theory), and statistical learning theory. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **01-math-foundations**
  - **Linear Algebra**
    - Eigenvalues/eigenvectors
    - SVD
    - PCA via SVD
    - Jacobian
    - Hessian
    - PSD matrices
    - Condition number
    - L1 vs L2 norms
    - Lagrange multipliers
    - KKT conditions
  - **Worked Derivations**
    - Backprop chain rule
    - Logistic regression gradient
    - Softmax gradient
    - Attention gradient
    - sqrt(d_k) scaling
    - Adam bias correction
    - MSE vs BCE
    - BatchNorm backprop
    - L2 as weight decay
    - ELBO derivation
    - KL/cross-entropy link
    - Normal equation
  - **Probability & Statistics**
    - Common distributions
      - Bernoulli/Binomial
      - Poisson/Exponential
      - Normal
    - LLN vs CLT
    - Hypothesis testing
      - Z-test
      - T-test
      - P-value meaning
    - Type I vs Type II errors
    - Confidence intervals
      - Wald vs Wilson
    - Bayes theorem
      - Base rate fallacy
    - MLE vs MAP
    - Bootstrap resampling
  - **Canonical Stats Questions**
    - Multiple testing
      - Bonferroni FWER
      - Benjamini-Hochberg FDR
    - Simpsons paradox
    - Causal study design
      - RCT
      - Instrumental variables
      - Diff-in-diff
      - Regression discontinuity
      - Propensity score matching
    - Common statistical traps
  - **Optimization Practice**
    - Gradient descent variants
    - Batch size effects
    - Learning rate
    - Momentum
    - Adam
    - AdamW decoupled decay
    - Vanishing/exploding gradients
    - LR scheduling
    - Hyperparameter search
    - Optimizer selection
    - Debugging checklist
    - Quantization
  - **Optimization Theory**
    - Convex vs non-convex
    - Variance reduction SVRG/SAGA
    - Second-order methods
      - Newton
      - L-BFGS
      - K-FAC
    - SAM/ASAM
    - LR schedules deep dive
    - Gradient clipping
    - Loss landscape
      - Mode connectivity
      - NTK
  - **Statistical Learning Theory**
    - Empirical vs true risk
    - Bias-variance decomposition
      - Cross-term vanishing
    - PAC learning
      - Sample complexity bound
    - VC dimension
      - Shattering
      - VC of linear classifiers
    - Generalization bounds
      - Uniform convergence
      - Rademacher complexity
    - No-free-lunch theorem
    - Deep nets vs classical theory
      - Double descent
      - Implicit regularization
      - Norm/margin bounds
  - **Information Theory**
    - Shannon entropy
    - Cross-entropy
    - KL divergence
      - Forward vs reverse KL
    - Mutual information
    - Perplexity
```

## Questions

### Easy

1. What does it mean for a vector to be an eigenvector of a matrix, geometrically?
2. Why must singular values in SVD be non-negative while eigenvalues can be negative?
3. What is the practical difference between the Jacobian and the Hessian?
4. Why does a positive-definite Hessian at a critical point imply a local minimum?
5. What geometric shape does the L1 constraint region take, and why does that shape produce sparsity?
6. State Bayes' theorem and label each term (prior, likelihood, posterior, evidence).
7. What is the difference between the Law of Large Numbers and the Central Limit Theorem?
8. Define Type I and Type II error in plain language, with an example each.
9. What does a 95% confidence interval actually guarantee, and what is the most common misinterpretation of it?
10. What is the difference between MLE and MAP in one sentence?
11. Why is $n-1$ (not $n$) used in the sample variance formula?
12. What problem does the bootstrap solve that a closed-form standard error formula cannot?
13. What is Simpson's paradox, and what causes it?
14. Why does Bonferroni correction get more conservative as the number of tests grows?
15. What is the difference between momentum and plain SGD?
16. Why does Adam typically converge faster than vanilla SGD early in training?
17. What symptom in training loss curves indicates exploding gradients?
18. What is the role of the learning rate warm-up phase?
19. State the bias-variance decomposition and name which term irreducible noise corresponds to.
20. What does VC dimension measure, in one sentence?
21. In your own words, what does the No-Free-Lunch theorem say about comparing learning algorithms?
22. What is the difference between entropy and cross-entropy?
23. Why is KL divergence not a true distance metric?
24. What is perplexity, and how is it related to cross-entropy?
25. Define PAC learning informally: what do "probably" and "approximately correct" each hedge against?

### Medium

1. Derive why PCA's principal components are the eigenvectors of the covariance matrix, and explain the SVD-based alternative computation.
2. A matrix has a condition number of 10^8. What practically breaks when you try to invert it or run gradient-based optimization on a loss with that Hessian?
3. Walk through why the cross-term in the bias-variance decomposition vanishes — what property of $\bar f(x)$ makes this work?
4. Why does the bias-variance decomposition not have an equally clean form for 0-1 loss or cross-entropy loss?
5. Explain why AdamW's decoupled weight decay is not equivalent to Adam with L2 regularization added to the loss.
6. Derive (or explain the intuition for) why dividing $QK^T$ by $\sqrt{d_k}$ in attention prevents softmax saturation.
7. A model gets 99.9% accuracy on a fraud dataset with 0.1% positive rate. Walk through the Bayes'-theorem reasoning for why most flagged cases are still false positives even with 95% sensitivity and 99% specificity.
8. Why does the Wald confidence interval for a proportion break down when $\hat p$ is close to 0 or 1, and what interval should you use instead?
9. You run 50 A/B test metrics simultaneously. Explain why Bonferroni is likely too conservative here and what Benjamini-Hochberg controls for instead.
10. Explain the parallel-trends assumption behind difference-in-differences and describe a scenario where it plausibly fails.
11. Why does the classical VC generalization bound become vacuous for deep neural networks, and what does Rademacher complexity offer instead?
12. Explain why "peeking" at a running A/B test's p-value inflates the Type I error rate even though each individual look uses a valid p-value.
13. A model has VC dimension $\infty$ (like 1-NN) yet works well in practice. Reconcile this with the classical generalization bound.
14. Explain the connection between L2 regularization, a Gaussian prior, and MAP estimation — derive the correspondence.
15. Why does bagging primarily reduce variance while boosting primarily reduces bias? Connect this to the bias-variance decomposition.
16. Compare forward KL $D_{KL}(P\|Q)$ and reverse KL $D_{KL}(Q\|P)$ in terms of mode-covering vs mode-seeking behavior, and give a concrete case (e.g. VI) where the choice matters.
17. Explain why the ELBO is a *lower* bound on the log-likelihood — what quantity makes up the gap?
18. A loss curve shows train and validation loss both plateauing at a high value, close together. Diagnose bias vs variance and propose two concrete fixes.
19. Explain why Newton's method converges faster than gradient descent near a minimum, and why it's rarely used directly for large neural networks.
20. What does it mean for SGD to exhibit "implicit regularization," and why does this matter for reconciling deep learning with classical capacity theory?
21. Explain why a p-value of 0.03 does not mean "there's a 3% chance the null hypothesis is true."
22. Derive the gradient of the softmax + cross-entropy combination and explain why it simplifies so cleanly compared to differentiating each separately.
23. Why is instrumental variable regression needed when treatment is endogenous, and what three conditions must a valid instrument satisfy?
24. A gradient clipping threshold is set far too low. What symptom would you see in the training curve, and why?
25. Explain the relationship between mutual information and Pearson correlation — why can two variables have zero correlation but high mutual information?

### Hard

1. Derive the full bias-variance-noise decomposition from scratch starting from $\mathbb{E}[(y-\hat f(x))^2]$, showing explicitly every step where independence assumptions are used.
2. Prove (or sketch a proof of) why VC dimension of linear classifiers in $\mathbb{R}^d$ is exactly $d+1$, including why $d+2$ points cannot be shattered.
3. Derive the sample complexity bound for PAC learning in the realizable, finite-hypothesis-class case, and explain why the bound is only logarithmic in $|\mathcal H|$ and $1/\delta$ but polynomial in $1/\epsilon$.
4. Double descent contradicts the classical U-shaped bias-variance curve. Explain the interpolation threshold and describe two competing theoretical explanations for why test error descends again past it.
5. Derive the KKT conditions from the Lagrangian for a constrained optimization problem with both equality and inequality constraints, and explain what complementary slackness means intuitively.
6. Two hypotheses classes have the same VC dimension but one generalizes much better empirically on a given dataset. Using Rademacher complexity, explain how this is possible when VC-based bounds would treat them identically.
7. Derive the update rule for Adam's bias-corrected first and second moment estimates, and explain precisely why the raw (uncorrected) estimates are biased toward zero in early steps.
8. A model trained with SGD and one trained with Adam reach the same training loss, but the SGD model generalizes better on a vision task. Using the "flat minima" / implicit regularization framework, explain a plausible mechanism.
9. Derive the ELBO from the log marginal likelihood using Jensen's inequality, and identify exactly where the inequality becomes tight (i.e., when ELBO = log-likelihood).
10. Explain why the No-Free-Lunch theorem is compatible with the empirical fact that some algorithms (e.g., CNNs on images) consistently outperform others — what is the theorem actually averaging over, and why doesn't that average describe the real world?
11. A dataset has 20M rows and a treatment effect estimate with p < 0.0001, but the effect size is a 0.002% lift. Walk through why statistical significance here is nearly meaningless and what additional analysis you'd demand before shipping.
12. Derive why minimizing cross-entropy loss with one-hot true labels is equivalent to minimizing KL divergence to the true label distribution up to an additive constant.
13. Construct a scenario using Berkson's paradox / collider bias where two variables are marginally independent but become dependent after conditioning on a third variable, and explain the causal-graph reasoning behind it.
14. Explain why second-order methods like L-BFGS or K-FAC are difficult to scale to billion-parameter models, tying the answer to Hessian dimensionality and curvature approximation cost.
15. A regression discontinuity design is proposed to estimate a causal effect at a policy cutoff. What assumption could silently invalidate it, and how would you test for that failure using the running variable's density?
16. Explain precisely what breaks in the standard Wald-type generalization bound's assumptions when the hypothesis class is selected *after* looking at the test set, and why this is mathematically identical to the multiple-testing / p-hacking problem.
17. Given a loss landscape with many saddle points but few bad local minima (as is typical in overparameterized networks), explain why plain gradient descent can still stall for long periods, and how SGD noise and momentum address this differently.
18. Derive the connection between L1 regularization and a Laplace prior under MAP estimation, and explain mathematically why the Laplace prior — unlike the Gaussian — produces exact zeros at the optimum.
19. A practitioner claims "since my network fits random labels perfectly, VC theory is wrong." Construct the precise rebuttal distinguishing what VC theory gets right (the capacity-data tradeoff exists) from what it gets wrong (parameter count as the capacity measure).
20. Derive the two-sample z-test statistic for proportions from first principles (variance of a difference of independent proportions), and explain why the pooled proportion $\hat p$ is used in the standard error rather than each group's own $\hat p_i$.
