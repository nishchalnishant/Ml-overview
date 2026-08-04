---
module: Meta
topic: Glossary
subtopic: ""
status: in-progress
tags: [glossary, ml, reference, all-folders]
---
# Repository-Wide ML Glossary

Comprehensive glossary of key terms drawn from every folder in this repo, built folder by folder (in numeric folder order). Each entry: the problem the term solves, the core definition/insight, and why it matters (pitfall/interview trap). This file is distinct from `18-resources/02-ml-glossary.md` (a separate, standalone A-Z glossary) — no merging between the two.

**Progress:** 01-math-foundations, 02-data, 03-classical-ml, 04-evaluation done. Remaining: 05-deep-learning-core through 18-resources.

---

## 01-math-foundations


**A/B Testing**

*The problem:* Deploying a change without a controlled experiment conflates correlation (deployment coincided with a metric change) with causation — seasonality, novelty, and other concurrent changes confound the comparison.

*Core insight / definition:* Randomize users into control/treatment, compute a two-proportion z-test: `z = (p̂₁ - p̂₂) / sqrt(p̂(1-p̂)(1/n₁+1/n₂))`. Sample size is pre-registered from the minimum detectable effect, baseline rate, α, and power: `n = 2(z_{α/2}+z_β)²p(1-p)/Δp²`.

*Why it matters / pitfall / interview trap:* "Peeking" — checking significance daily and stopping early — inflates Type I error from 5% to ~30%; fix with pre-committed sample size or sequential testing (SPRT/always-valid p-values). Testing many metrics at once needs Bonferroni/BH correction, and novelty effects can inflate short-run wins.

---

**Accuracy**

*The problem:* Need a single number to summarize classifier correctness.

*Core insight / definition:* `(TP+TN)/(TP+TN+FP+FN)` — fraction of predictions correct overall.

*Why it matters / pitfall / interview trap:* Meaningless under class imbalance — predicting the majority class everywhere on a 99%/1% split scores 99% while catching zero minority cases. Use precision/recall/F1/PR-AUC instead.

---

**Acquisition Function**

*The problem:* Bayesian optimization needs a rule for choosing the next hyperparameter point to evaluate, trading off exploring uncertain regions vs exploiting known-good ones.

*Core insight / definition:* A function (e.g., Expected Improvement, Upper Confidence Bound) computed from the surrogate Gaussian Process's posterior mean and variance that scores candidate points; the optimizer evaluates the point maximizing it next.

*Why it matters / pitfall / interview trap:* This is what makes Bayesian optimization sample-efficient compared to grid/random search — it uses the model of past results instead of ignoring them.

---

**Adagrad**

*The problem:* Uniform learning rates across parameters is suboptimal when features/gradients have very different scales (e.g., sparse features).

*Core insight / definition:* Adapts the per-parameter learning rate by dividing by the square root of the accumulated sum of squared past gradients: `θ ← θ - η/√(G+ε) · g`.

*Why it matters / pitfall / interview trap:* The accumulator only grows, so the effective learning rate monotonically shrinks and can vanish before convergence on long training runs — this is exactly what RMSprop/Adam fix by using a decaying moving average instead.

---

**Adam**

*The problem:* SGD with a single global learning rate converges slowly and is sensitive to LR choice; different parameters may need different effective step sizes.

*Core insight / definition:* Maintains exponential moving averages of the gradient (first moment `m`) and squared gradient (second moment `v`), bias-corrects both, and updates `θ ← θ - η·m̂/(√v̂+ε)`. Combines momentum with per-parameter adaptive scaling.

*Why it matters / pitfall / interview trap:* Fast and robust default (LR ~3e-4), but can generalize worse than well-tuned SGD+momentum on vision tasks; the bias-correction terms matter most in early training when the moving averages are still near zero.

---

**AdamW**

*The problem:* In vanilla Adam, L2 regularization added to the loss gets divided by the adaptive `√v̂` term, so weight decay strength becomes coupled to (and distorted by) the per-parameter adaptive scaling.

*Core insight / definition:* Decouples weight decay from the gradient-based update — applies `θ ← θ - η·λ·θ` as a separate step rather than folding it into the loss gradient.

*Why it matters / pitfall / interview trap:* This decoupling is why AdamW (not Adam+L2) is the standard for training Transformers — "using Adam with L2 regularization" is a common but subtly incorrect statement of what modern training actually does.

---

**Agents (LLM)**

*The problem:* Single-shot LLM prompting cannot handle tasks requiring multi-step reasoning, tool use, or interaction with external state.

*Core insight / definition:* A loop where the LLM perceives state, plans an action, calls a tool, observes the result, and updates state — the LLM acts as a control plane over external tools/environments.

*Why it matters / pitfall / interview trap:* Errors compound over long horizons (each step's mistake propagates), and evaluation/debugging is much harder than for single-turn outputs; latency and cost multiply per step.

---

**ASAM (Adaptive Sharpness-Aware Minimization)**

*The problem:* Vanilla SAM's isotropic perturbation ball is not invariant to parameter re-scaling (e.g., ReLU network weight rescaling symmetries), so "sharpness" measured this way can be an artifact of parameterization rather than a real geometric property.

*Core insight / definition:* Normalizes the perturbation radius per-parameter relative to the weight magnitude, making sharpness measurement invariant to these reparameterizations.

*Why it matters / pitfall / interview trap:* Directly answers the Dinh et al. 2017 critique that sharp/flat minima classifications under naive metrics aren't reparameterization-invariant and thus aren't a well-defined generalization predictor without a fix like this.

---

**Backpropagation**

*The problem:* Computing gradients of a loss with respect to millions of parameters in a deep composition of functions by brute-force calculus is intractable.

*Core insight / definition:* Applies the chain rule systematically backward through the computation graph, reusing intermediate gradients (`∂L/∂θ_l` computed from `∂L/∂θ_{l+1}` and local Jacobians) so cost is linear in network size rather than exponential.

*Why it matters / pitfall / interview trap:* Vanishing/exploding gradients are a direct consequence of repeatedly multiplying Jacobians through many layers — this is why deep network trainability depends heavily on activation choice, initialization, and normalization.

---

**Bagging (Bootstrap Aggregating)**

*The problem:* A single model trained on one sample has variance — a different training sample would produce different predictions.

*Core insight / definition:* Train many models independently in parallel on bootstrap resamples of the data, then average predictions (or vote). Reduces variance because averaging independent noisy estimators shrinks variance without changing bias.

*Why it matters / pitfall / interview trap:* Bagging does not reduce bias — a Random Forest of shallow, biased trees is still biased. Pick over boosting for a robust, low-tuning baseline; pick boosting when you have time to tune for max accuracy.

---

**Base Rate Fallacy**

*The problem:* Humans intuitively ignore the prior probability of a hypothesis when updating on new evidence, leading to systematic misjudgment of conditional probabilities.

*Core insight / definition:* Given a rare condition (prior `P(H)` small) and an imperfect test, `P(H|+)` can be far lower than the test's sensitivity/specificity would naively suggest — e.g., 99%-accurate test, 0.1% base rate → `P(disease|+) ≈ 9%` because the false positives from the huge healthy population outnumber true positives.

*Why it matters / pitfall / interview trap:* Classic interview trap: candidates answer "~99%" instead of computing the full Bayes' theorem expression with the base rate in the denominator.

---

**Batch Normalization**

*The problem:* Internal covariate shift — the distribution of layer inputs changes during training as earlier layers' weights update, slowing convergence and requiring careful, low learning rates.

*Core insight / definition:* Normalizes activations to zero mean/unit variance across the batch dimension per feature, then applies learnable scale/shift (`γ, β`) to restore expressiveness.

*Why it matters / pitfall / interview trap:* Batch-size dependent — unstable at very small batch sizes and awkward with RNNs/Transformers (sequence-length variability), which is why LayerNorm is preferred there instead.

---

**Batch Size**

*The problem:* The number of examples used per gradient update controls the tradeoff between gradient estimate accuracy, hardware utilization, and generalization.

*Core insight / definition:* Larger batches give a lower-variance gradient estimate (closer to full-batch) and better GPU utilization; smaller batches inject more noise into updates, which can help escape sharp minima.

*Why it matters / pitfall / interview trap:* Very large batch training often needs a linearly scaled learning rate (linear LR scaling rule) and warmup to remain stable; naively scaling batch size without adjusting LR degrades generalization.

---

**Bayes' Theorem**

*The problem:* Given evidence, you want the updated (posterior) probability of a hypothesis, but you only directly know how likely the evidence is under each hypothesis (the likelihood) plus your prior belief.

*Core insight / definition:* `P(H|E) = P(E|H)·P(H) / P(E)` — posterior ∝ likelihood × prior, normalized by the total probability of the evidence.

*Why it matters / pitfall / interview trap:* Ignoring the prior (base rate) is the single most common Bayes'-theorem interview trap; it also underlies MAP estimation, Naive Bayes, and Bayesian neural networks.

---

**Bayesian Neural Networks**

*The problem:* Standard neural nets give point-estimate weights and predictions with no calibrated uncertainty.

*Core insight / definition:* Places a distribution over network weights rather than a point value, and computes (or approximates) a posterior over weights given data, propagating uncertainty into predictions.

*Why it matters / pitfall / interview trap:* Exact posterior inference is intractable for any nontrivial network, forcing approximations (variational inference, MC dropout, Laplace) — the practical uncertainty estimate is only as good as the approximation.

---

**Bayesian Optimization**

*The problem:* Grid and random search waste evaluations on clearly unpromising hyperparameter regions because they ignore the results of previous trials.

*Core insight / definition:* Fits a surrogate model (typically a Gaussian Process) over the hyperparameter→performance mapping, then uses an acquisition function to choose the next point balancing exploration and exploitation.

*Why it matters / pitfall / interview trap:* Much more sample-efficient than grid/random search for expensive-to-evaluate objectives (full training runs), but the surrogate model itself adds overhead and complexity that isn't worth it for cheap objectives.

---

**Benjamini-Hochberg Procedure (FDR Control)**

*The problem:* Bonferroni's family-wise error control is overly conservative (too many false negatives) when testing many hypotheses, as is common in exploratory settings (genomics, many-metric A/B tests).

*Core insight / definition:* Controls the false discovery rate instead: sort p-values ascending, find the largest `k` such that `p_(k) ≤ (k/m)·α`, and reject all hypotheses up to `k`.

*Why it matters / pitfall / interview trap:* Preferred over Bonferroni for exploratory studies with many comparisons (e.g., testing 50 A/B metrics); Bonferroni is preferred when strict FWER control matters (confirmatory/clinical settings).

---

**Bias (statistical/ML)**

*The problem:* Need to characterize the error component caused by a model's structural assumptions being wrong, independent of the particular training sample drawn.

*Core insight / definition:* The systematic difference between the model's expected prediction (averaged over training sets) and the true function — high bias means the hypothesis class itself is too simple to capture the pattern.

*Why it matters / pitfall / interview trap:* High bias shows as underfitting: both train and validation error are high and similar — distinguishing this from high variance (overfitting) drives the correct fix (more capacity vs. more data/regularization).

---

**Bias-Variance Decomposition**

*The problem:* Expected test error has multiple distinct sources, and treating "the model is bad" as one undifferentiated problem leads to the wrong fix (adding capacity to a high-variance model, or more data to a high-bias one).

*Core insight / definition:* For squared error, `E[(y - f̂(x))²] = Bias(f̂(x))² + Var(f̂(x)) + σ²`, derived by adding and subtracting `E[f̂(x)]` and using independence of noise; `σ²` is irreducible.

*Why it matters / pitfall / interview trap:* Bagging reduces variance without changing bias (parallel independent models); boosting reduces bias (sequential error correction) — misattributing which term an ensemble method addresses is a common interview slip.

---

**Bonferroni Correction**

*The problem:* Running `m` independent hypothesis tests at α=0.05 each gives a family-wise false-positive probability approaching 1 as `m` grows.

*Core insight / definition:* Adjust the per-test significance threshold to `α/m`, controlling the family-wise error rate at the original `α`.

*Why it matters / pitfall / interview trap:* Conservative — causes many false negatives, especially when tests are correlated; use for confirmatory/high-stakes settings (clinical trials), not large-scale exploratory testing (prefer Benjamini-Hochberg there).

---

**Boosting**

*The problem:* A single weak learner or an ensemble of independently-trained models can't correct systematic errors that persist across all of them (bias).

*Core insight / definition:* Trains models sequentially, each new model fit to the current ensemble's residual errors (or reweighted misclassified examples), reducing bias over rounds.

*Why it matters / pitfall / interview trap:* More hyperparameter-sensitive and prone to overfitting noisy labels than bagging since later rounds chase residuals that may be noise, not signal; pick when you have tuning budget and need max accuracy (XGBoost/LightGBM).

---

**Bootstrap (Resampling)**

*The problem:* Many useful statistics (median, AUC, custom metrics) have no closed-form sampling distribution, so no analytic confidence interval formula exists.

*Core insight / definition:* Draw `B` resamples of size `n` with replacement from the observed data, compute the statistic on each, and use the empirical percentiles (e.g., 2.5th/97.5th) of the `B` values as the confidence interval.

*Why it matters / pitfall / interview trap:* Doesn't create new information, only quantifies uncertainty already present in the sample; unreliable for very small `n` (<30) and needs `B ≥ 1000` for stable percentile estimates.

---

**Central Limit Theorem (CLT)**

*The problem:* Without knowing a population's true distribution, how can you construct confidence intervals or hypothesis tests for a sample statistic?

*Core insight / definition:* The standardized sample mean converges in distribution to standard Normal regardless of the population's original shape: `(X̄_n - μ)/(σ/√n) → N(0,1)` as `n→∞`.

*Why it matters / pitfall / interview trap:* Requires independence — doesn't directly apply to correlated data like time series; it governs the sampling distribution of the mean, not the shape of the raw data itself (a common conflation).

---

**Chain Rule (Calculus)**

*The problem:* Need the derivative of a composition of functions, which arises constantly in deep, layered models.

*Core insight / definition:* `d/dx f(g(x)) = f'(g(x))·g'(x)` — derivatives of nested functions multiply along the composition path.

*Why it matters / pitfall / interview trap:* The entire mechanism behind backpropagation; repeated multiplication of many such terms across layers is the direct cause of vanishing/exploding gradients.

---

**Class Imbalance**

*The problem:* When one class vastly outnumbers another, naive metrics and naive training both silently favor the majority class.

*Core insight / definition:* Standard accuracy and unweighted loss are dominated by the majority class; addressing it requires resampling (SMOTE/undersampling), class weighting, threshold tuning, or imbalance-robust metrics.

*Why it matters / pitfall / interview trap:* "High accuracy" on an imbalanced problem is a red flag, not a success signal — always check precision/recall/PR-AUC before trusting accuracy.

---

**Complementary Slackness**

*The problem:* In constrained optimization with inequality constraints, need a condition linking whether a constraint is active to whether its Lagrange multiplier is nonzero.

*Core insight / definition:* At the KKT optimum, for each inequality constraint `g_i(x) ≤ 0` with multiplier `λ_i ≥ 0`: `λ_i·g_i(x*) = 0` — either the constraint is tight (`g_i=0`) or its multiplier is zero.

*Why it matters / pitfall / interview trap:* In SVMs, this is exactly why only support vectors (points on/inside the margin) have nonzero dual coefficients — non-support-vector points have slack, forcing their multiplier to zero.

---

**Concept Drift**

*The problem:* A deployed model's accuracy can degrade even if the input feature distribution hasn't changed, because the relationship between features and target has shifted.

*Core insight / definition:* `P(Y|X)` changes over time (e.g., user preferences shift), as opposed to `P(X)` changing.

*Why it matters / pitfall / interview trap:* Can't be detected from input feature distributions alone (unlike data drift) — requires lagged ground-truth labels, so detection is inherently delayed.

---

**Condition Number**

*The problem:* Need a way to quantify how numerically sensitive a matrix-based computation (e.g., matrix inversion, an optimization landscape) is to small input perturbations.

*Core insight / definition:* Ratio of largest to smallest singular value, `κ(A) = σ_max/σ_min`. Large `κ` means the matrix is ill-conditioned.

*Why it matters / pitfall / interview trap:* High condition number in a loss's Hessian means gradient descent oscillates in high-curvature directions while crawling in low-curvature ones — directly motivates preconditioning/adaptive methods and explains why some loss landscapes are much harder to optimize than others.

---

**Confidence Interval (CI)**

*The problem:* A point estimate alone hides how much sampling uncertainty surrounds it.

*Core insight / definition:* `x̄ ± z_{α/2}·s/√n` — an interval constructed by a procedure that, under repeated sampling, contains the true parameter with the stated frequency (e.g., 95%).

*Why it matters / pitfall / interview trap:* The correct interpretation is about the long-run behavior of the procedure, not a probability statement about one specific interval containing the (fixed, non-random) true value — stating it the second way is a classic interview trap.

---

**Confounder**

*The problem:* Two variables can be strongly correlated purely because a third variable causes both, without either causing the other.

*Core insight / definition:* A variable that influences both the "treatment"/predictor and the outcome, producing a spurious association (e.g., hot weather drives both ice cream sales and drowning rates).

*Why it matters / pitfall / interview trap:* Models trained on observational data learn these spurious correlations and break when deployed as an actual intervention (correlation ≠ causation); Simpson's paradox is often a signature that an unaddressed confounder exists.

---

**Confusion Matrix (TP/TN/FP/FN)**

*The problem:* A single accuracy number can't distinguish the different ways a classifier can be wrong.

*Core insight / definition:* Cross-tabulates predicted vs actual class: True Positive (correct positive), True Negative (correct negative), False Positive (Type I error, false alarm), False Negative (Type II error, miss).

*Why it matters / pitfall / interview trap:* Precision, recall, F1, and ROC/PR curves are all derived from this table — misreading which cell corresponds to which error type leads to wrong metric choices for asymmetric-cost problems.

---

**Conjugate Priors**

*The problem:* Computing a Bayesian posterior in closed form generally requires the prior and likelihood to combine in a tractable way; otherwise numerical integration or MCMC/VI is needed.

*Core insight / definition:* A prior family is conjugate to a likelihood if the resulting posterior is in the same family (e.g., Beta prior + Binomial likelihood → Beta posterior).

*Why it matters / pitfall / interview trap:* Enables exact, fast Bayesian updating in special cases; for most realistic ML models (deep nets) no useful conjugate prior exists, forcing approximate inference.

---

**Convexity**

*The problem:* Non-convex optimization problems can have many local minima, and gradient-based methods offer no guarantee of finding the global one.

*Core insight / definition:* A function is convex if the line segment between any two points on its graph lies on or above the graph (`f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y)`); for convex functions, any local minimum is guaranteed to be the global minimum.

*Why it matters / pitfall / interview trap:* Logistic regression and (linear-kernel) SVM losses are convex, giving reliable optimization guarantees; deep network losses are generally non-convex, yet SGD still works well in practice because over-parameterized landscapes have relatively few bad local minima — saddle points, not local minima, are the more common obstacle.

---

**Correlation vs Causation**

*The problem:* Predictive models exploit statistical association between variables, which is not evidence that changing one variable causes a change in the other.

*Core insight / definition:* Pearson correlation captures linear co-movement only, and even a perfect correlation can be fully explained by a shared confounder (e.g., ice cream sales and drownings, both driven by summer heat).

*Why it matters / pitfall / interview trap:* Deploying a purely correlational model for an interventional/policy decision can produce nonsensical results; establishing causation requires design (RCT, IV, DiD, RD) not just observed association.

---

**Cosine Learning Rate Decay**

*The problem:* Learning rate needs to decrease over training for convergence, but abrupt step-drops can destabilize training and simple linear decay isn't always smooth enough.

*Core insight / definition:* Learning rate follows a cosine curve from initial value down to (near) zero over training: `η_t = η_min + 0.5(η_max-η_min)(1+cos(πt/T))`, typically after a warmup phase.

*Why it matters / pitfall / interview trap:* Standard for modern large-model training (often combined with linear warmup); the smoothness avoids the discontinuities of step schedules that can cause loss spikes.

---

**Cross-Entropy**

*The problem:* Need a differentiable loss function measuring how well a predicted probability distribution matches the true one, suitable for gradient-based classification training.

*Core insight / definition:* `H(P,Q) = -Σ P(x)log Q(x)` — expected cost of encoding samples from true distribution `P` using a code optimized for `Q`. For one-hot true labels, reduces to `L = -log q(y_true)`.

*Why it matters / pitfall / interview trap:* `H(P,Q) = H(P) + D_KL(P‖Q)`, so minimizing cross-entropy is equivalent to minimizing KL divergence to the true labels (since `H(P)` is constant) — this is why cross-entropy loss is the natural choice, not an arbitrary one.

---

**Cross-Validation (K-Fold)**

*The problem:* A single train/validation split gives a high-variance, potentially unlucky estimate of generalization performance, especially on small datasets.

*Core insight / definition:* Split data into K folds, train on K-1 and validate on the remaining fold, rotate through all K, average the results.

*Why it matters / pitfall / interview trap:* Naive K-fold leaks information for time series (shuffles future into training) and can break class balance — use time-series split for temporal data and stratified K-fold for imbalanced classification.

---

**Data Drift**

*The problem:* A deployed model's input distribution can shift over time even if the underlying feature-target relationship hasn't, silently degrading calibration/coverage.

*Core insight / definition:* `P(X)` changes (e.g., new user segment appears); detectable directly from feature distributions via PSI or KL divergence, without needing labels.

*Why it matters / pitfall / interview trap:* Detecting data drift doesn't confirm the model is actually performing worse (that requires concept drift detection with labels) — the two are commonly conflated.

---

**Data Leakage**

*The problem:* Information from outside the legitimate training scope (future data, the test set, or the target itself) can leak into features or preprocessing, producing misleadingly optimistic offline metrics that collapse in production.

*Core insight / definition:* Several distinct forms: target leakage (feature is the target/a near-synonym), preprocessing leakage (fitting scaler/imputer on the full dataset before splitting), temporal leakage (features computed using future timestamps), ID leakage (an ID column encodes cohort/time/target indirectly).

*Why it matters / pitfall / interview trap:* The universal fix is auditing every feature with "would this be known at prediction time in production?" and using pipelines (sklearn `Pipeline` + CV) so preprocessing is refit per fold.

---

**Difference-in-Differences (DiD)**

*The problem:* Randomization isn't always possible, but a natural pre/post treatment split across treated and untreated groups sometimes exists.

*Core insight / definition:* `ATT = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre})` — nets out time-invariant confounders and common time trends.

*Why it matters / pitfall / interview trap:* Requires the parallel trends assumption (treated and control groups would have moved together absent treatment) — this assumption is untestable directly and is the main point of attack in interview follow-ups.

---

**Differential Privacy / DP-SGD**

*The problem:* Training on sensitive data risks the model memorizing and later leaking individual training examples.

*Core insight / definition:* Clips per-example gradient norms and adds calibrated Gaussian noise before averaging, bounding any single example's influence on the final weights under a formal privacy budget `ε`.

*Why it matters / pitfall / interview trap:* The added noise degrades model utility — there's an explicit ε-vs-utility tradeoff that must be tuned, not a free guarantee.

---

**Double Descent**

*The problem:* Classical bias-variance theory predicts test error should monotonically worsen past the point where a model starts to interpolate (fit) training data perfectly — but this isn't always what's observed for modern over-parameterized models.

*Core insight / definition:* As model capacity grows past the interpolation threshold, test error can decrease, spike near the threshold, then decrease again in the heavily over-parameterized regime.

*Why it matters / pitfall / interview trap:* Explains why deep, heavily over-parameterized networks often generalize well despite fitting training data (including noise) — contradicts naive "more capacity = worse generalization" intuition and is a key reason classical learning theory alone can't explain deep learning.

---

**Eckart-Young Theorem**

*The problem:* Need to know, formally, what the *best possible* low-rank approximation to a matrix is, and in what sense it's optimal.

*Core insight / definition:* The best rank-`k` approximation to a matrix `A` in Frobenius (or spectral) norm is given by truncating its SVD to the top `k` singular values/vectors: `A_k = U_k Σ_k V_k^T`.

*Why it matters / pitfall / interview trap:* This is the formal justification for truncated-SVD-based PCA and compression — "just keep the top singular values" isn't a heuristic, it's provably optimal under this norm.

---

**Eigenvalues and Eigenvectors**

*The problem:* Need to understand the "intrinsic" directions and scaling factors of a linear transformation, independent of the basis it's expressed in.

*Core insight / definition:* `Av = λv` — `v` is a direction that the transformation `A` only scales (doesn't rotate); `λ` is the scale factor.

*Why it matters / pitfall / interview trap:* PCA's principal components are exactly the eigenvectors of the data's covariance matrix, ordered by eigenvalue (variance explained); Hessian eigenvalues determine whether a critical point is a min/max/saddle.

---

**Elastic Net**

*The problem:* Pure L1 is unstable under correlated features (arbitrarily picks one of a correlated group); pure L2 never produces sparse solutions.

*Core insight / definition:* Combines both penalties: `Loss + α₁Σ|w| + α₂Σw²`, getting L1's sparsity together with L2's stability on correlated features.

*Why it matters / pitfall / interview trap:* Costs an extra hyperparameter to tune (the L1/L2 mix ratio) versus a single `λ`; best used when features are both high-dimensional and correlated.

---

**ELBO (Evidence Lower Bound)**

*The problem:* For latent-variable models the true posterior `P(z|x)` is intractable, so direct maximum-likelihood training is impossible.

*Core insight / definition:* A tractable lower bound on the log-likelihood, `log P(x) ≥ ELBO`, derived via Jensen's inequality; maximizing ELBO indirectly maximizes likelihood while fitting an approximate posterior.

*Why it matters / pitfall / interview trap:* This is literally the VAE training objective — decomposes into a reconstruction term plus a KL term regularizing the approximate posterior toward the prior.

---

**Empirical Risk vs True Risk**

*The problem:* We can only ever measure performance on a finite training/test sample, but what we actually care about is performance on the entire underlying data distribution.

*Core insight / definition:* Empirical risk `R̂(h) = (1/n)Σ L(h(x_i),y_i)` is the average loss over the observed sample; true risk `R(h) = E_{(x,y)~D}[L(h(x),y)]` is the expectation over the full distribution — the gap between them is the generalization gap.

*Why it matters / pitfall / interview trap:* All of statistical learning theory (VC bounds, Rademacher complexity, PAC learning) exists to bound how large this gap can be as a function of model capacity and sample size.

---

**Entropy (Shannon)**

*The problem:* Need a principled numerical measure of "how uncertain" or "how surprising" a probability distribution is.

*Core insight / definition:* `H(P) = -Σ P(x)log P(x)` — the expected information content (`-log p`) of outcomes drawn from `P`. Maximized by the uniform distribution; zero for a deterministic distribution.

*Why it matters / pitfall / interview trap:* Rare events carry more information (`-log p` grows as `p→0`); entropy is the foundation from which cross-entropy and KL divergence are built.

---

**Exploding Gradients**

*The problem:* In deep or recurrent networks, repeated multiplication of Jacobians with the largest eigenvalue > 1 can cause gradients to grow exponentially with depth/sequence length.

*Core insight / definition:* Gradient magnitude grows as roughly `O(λ_max^T)` over `T` layers/timesteps, causing NaN losses and wildly oscillating weight updates.

*Why it matters / pitfall / interview trap:* Fixed with gradient clipping (by value or by norm), lower learning rates, and gating mechanisms (LSTM); transformers are less prone to this than vanilla RNNs due to residual connections and normalization, but not immune, hence clipping remains common practice.

---

**F1 Score**

*The problem:* Precision and recall individually don't give a single balanced number, and optimizing one alone can crash the other.

*Core insight / definition:* Harmonic mean of precision and recall: `F1 = 2·P·R/(P+R)`.

*Why it matters / pitfall / interview trap:* Appropriate when neither false positives nor false negatives dominate in cost; if one error type is far costlier, F1 obscures that asymmetry — use precision or recall directly, or a weighted Fβ.

---

**False Discovery Rate (FDR)**

*The problem:* Family-wise error rate control (Bonferroni) becomes overly conservative when testing hundreds/thousands of hypotheses simultaneously.

*Core insight / definition:* The expected proportion of false positives among all rejected (declared significant) hypotheses, controlled via the Benjamini-Hochberg procedure.

*Why it matters / pitfall / interview trap:* Appropriate for exploratory large-scale testing (genomics, many-metric experiments) where some false positives are tolerable as long as their rate is bounded — distinct from FWER, which bounds the probability of *any* false positive.

---

**Family-Wise Error Rate (FWER)**

*The problem:* Need to characterize the risk of making at least one false discovery across a whole family of hypothesis tests.

*Core insight / definition:* `FWER = 1-(1-α)^m` for `m` independent tests — approaches 1 rapidly as `m` grows even with each individual test at α=0.05.

*Why it matters / pitfall / interview trap:* Motivates Bonferroni correction; conflating "my p-value is significant" with "this specific finding is trustworthy" ignores that with enough tests, some will pass by pure chance.

---

**Feature Learning Regime**

*The problem:* The Neural Tangent Kernel (NTK) analysis describes networks in the infinite-width limit where features stay essentially fixed, but real, finite-width networks visibly learn useful internal representations.

*Core insight / definition:* The regime (typical of practical, finite-width networks) where internal representations meaningfully change during training, in contrast to the NTK "lazy training" regime where the network behaves like a fixed random-feature kernel method.

*Why it matters / pitfall / interview trap:* Explains why NTK theory, despite being mathematically clean, doesn't fully describe why deep learning works in practice — real performance gains often come from feature learning that NTK theory explicitly excludes.

---

**Federated Learning**

*The problem:* Sensitive or siloed data (e.g., on user devices) cannot be centralized for training due to privacy, bandwidth, or regulatory constraints.

*Core insight / definition:* Clients train locally on their own data and send only model updates (not raw data) to a central server, which aggregates them (e.g., FedAvg).

*Why it matters / pitfall / interview trap:* Non-IID data across clients causes conflicting local optima that complicate convergence, and gradient updates themselves can still leak information (gradient inversion attacks) — federated learning is not automatically equivalent to full privacy.

---

**Fisher Information Matrix**

*The problem:* Need to measure how much information an observed random variable carries about an unknown parameter, and to characterize the local curvature of the likelihood/loss around the current parameters.

*Core insight / definition:* `F = E[∇log p(x|θ) ∇log p(x|θ)^T]` — expected outer product of the score function; approximates the Hessian of the negative log-likelihood and is guaranteed PSD (unlike the true Hessian, which can be indefinite at saddle points).

*Why it matters / pitfall / interview trap:* Used as a well-behaved curvature proxy in natural gradient descent and K-FAC precisely because it's always PSD, sidestepping the indefinite-Hessian problem that breaks naive second-order methods near saddle points.

---

**Flash Attention**

*The problem:* Naive attention materializes the full N×N attention matrix in slow HBM memory, making memory (and effectively speed) scale quadratically with sequence length.

*Core insight / definition:* Tiles the attention computation to fit in fast on-chip SRAM, computing softmax incrementally without ever materializing the full N×N matrix — mathematically identical output, `O(N)` memory instead of `O(N²)`.

*Why it matters / pitfall / interview trap:* A systems-level exact-math optimization, not an approximation — it's "free" accuracy-wise; the benefit is marginal on short sequences and grows with sequence length.

---

**Forward vs Reverse KL Divergence**

*The problem:* When fitting an approximate distribution `Q` to a true distribution `P`, the direction in which you compute KL divergence changes what kind of approximation you get.

*Core insight / definition:* Forward KL `D_KL(P‖Q)` is mass/mode-covering — it heavily penalizes `Q` assigning near-zero probability where `P` has mass, so `Q` spreads to cover all of `P`'s modes. Reverse KL `D_KL(Q‖P)` is mode-seeking — `Q` collapses onto a single mode of `P` to avoid placing mass in `P`'s low-density regions.

*Why it matters / pitfall / interview trap:* Standard MLE fitting implicitly uses forward KL; variational inference typically uses reverse KL (for tractability), which is why VI posteriors are often overconfident/underestimate variance relative to the true posterior.

---

**Gaussian Process**

*The problem:* Bayesian optimization needs a probabilistic surrogate model that gives both a prediction and a calibrated uncertainty estimate at unobserved hyperparameter points.

*Core insight / definition:* A distribution over functions such that any finite set of points has a jointly Gaussian distribution, defined by a mean function and covariance/kernel function; conditioning on observed points gives a posterior mean and variance everywhere.

*Why it matters / pitfall / interview trap:* The variance estimate is exactly what the acquisition function needs to balance exploration vs exploitation; GPs scale poorly (cubic in number of observations), limiting Bayesian optimization to modest numbers of trials.

---

**GELU (Gaussian Error Linear Unit)**

*The problem:* ReLU's hard cutoff at zero is a discontinuity in the derivative and can cause "dying" neurons; a smoother nonlinearity was sought for very deep, wide models like Transformers.

*Core insight / definition:* `x·Φ(x)` where `Φ` is the standard Gaussian CDF — a smooth approximation of ReLU that weights inputs by their likelihood rather than a hard threshold.

*Why it matters / pitfall / interview trap:* Empirically stronger than ReLU in Transformer architectures despite being more expensive to compute — a case where extra compute cost is accepted for measurable quality gains.

---

**Generalization Gap**

*The problem:* Need a precise name and formalization for "the model does worse on new data than on training data."

*Core insight / definition:* The difference between true risk and empirical risk, `R(h) - R̂(h)`; bounded by learning-theoretic tools (VC dimension, Rademacher complexity, PAC bounds) as a function of hypothesis class complexity and sample size.

*Why it matters / pitfall / interview trap:* A large generalization gap signals overfitting; classical bounds based on capacity alone fail to explain why heavily over-parameterized deep nets can still generalize well (see double descent, implicit regularization).

---

**Gradient Clipping**

*The problem:* Occasional large gradient spikes (especially in RNNs or early training) can destabilize optimization with a single bad update.

*Core insight / definition:* By-value clipping caps each gradient component independently to `[-c,c]`; by-norm clipping rescales the entire gradient vector if its norm exceeds a threshold, preserving direction (`g ← g·min(1, c/‖g‖)`).

*Why it matters / pitfall / interview trap:* By-norm clipping is generally preferred because by-value clipping distorts the gradient's direction; transformers need it less than vanilla RNNs (due to residual/normalization structure) but it's still common practice, especially at large scale.

---

**Gradient Descent**

*The problem:* Need an iterative, scalable method to find parameters minimizing a (possibly very high-dimensional) loss function when no closed-form solution exists.

*Core insight / definition:* Repeatedly step in the direction of steepest descent: `θ ← θ - η∇L(θ)`, where `η` is the learning rate.

*Why it matters / pitfall / interview trap:* Guaranteed to reach the global minimum only for convex losses; for non-convex deep learning losses it typically converges to a good local minimum or saddle-adjacent region in practice, aided by SGD noise and the empirically favorable geometry of over-parameterized landscapes.

---

**Hessian**

*The problem:* First-derivative (gradient) information alone doesn't tell you about the curvature of the loss surface, which is needed to distinguish minima, maxima, and saddle points, and to enable faster (second-order) optimization.

*Core insight / definition:* The matrix of second-order partial derivatives, `H_ij = ∂²L/∂θ_i∂θ_j`. Positive-definite at a critical point implies a local minimum; indefinite implies a saddle point.

*Why it matters / pitfall / interview trap:* Computing/storing/inverting the full Hessian is `O(d²)`/`O(d³)` and infeasible for large neural nets — this is exactly why Newton's method is impractical at scale and approximations (L-BFGS, K-FAC, Fisher Information) exist.

---

**Hyperparameter Search**

*The problem:* Model hyperparameters (learning rate, regularization strength, architecture size) can't be learned by gradient descent on the training loss and must be chosen via an outer search process.

*Core insight / definition:* Grid search exhaustively evaluates a Cartesian product of values (combinatorially expensive); random search samples randomly (wastes less budget on unimportant dimensions); Bayesian optimization builds a surrogate model to intelligently pick the next point.

*Why it matters / pitfall / interview trap:* Random search often outperforms grid search at equal budget because most hyperparameters don't matter much — grid search wastes evaluations varying unimportant dimensions exhaustively.

---

**Hypothesis Testing**

*The problem:* Given an observed difference between groups/conditions, need a principled way to decide whether it reflects a real effect or could plausibly arise from sampling noise alone.

*Core insight / definition:* Assume the null hypothesis `H₀` (no effect) is true, then compute how surprising the observed data (or more extreme) would be under that assumption — the p-value.

*Why it matters / pitfall / interview trap:* A p-value is `P(data this extreme | H₀ true)`, not `P(H₀ true | data)` — conflating the two is the single most common statistics interview trap.

---

**Implicit Regularization**

*The problem:* Classical theory says more capacity (parameters) should mean more overfitting, yet heavily over-parameterized deep networks trained with plain SGD often generalize well without any explicit regularizer.

*Core insight / definition:* The optimization algorithm itself (SGD's trajectory, its noise, and its tendency to find flatter/lower-norm solutions among the many that fit the training data) acts as an implicit bias toward solutions that generalize, without any explicit penalty term.

*Why it matters / pitfall / interview trap:* Explains why removing all explicit regularization (dropout, weight decay) from a deep net doesn't always cause catastrophic overfitting — the optimizer is quietly doing regularization work; this is a major reason classical VC-dimension-based bounds fail for deep nets.

---

**Instrumental Variables (IV)**

*The problem:* When treatment assignment is confounded by unobserved variables, standard regression can't isolate the causal effect of treatment on outcome.

*Core insight / definition:* Uses a variable `Z` that (1) correlates with treatment, (2) affects the outcome only through treatment, (3) is independent of unobserved confounders — isolating variation in treatment that's "as good as randomized."

*Why it matters / pitfall / interview trap:* Finding a genuinely valid instrument is hard in practice — a weak or invalid instrument (violating condition 2 or 3) biases the estimate, sometimes worse than not using IV at all.

---

**Interpolation Threshold**

*The problem:* Classical theory predicts a single U-shaped bias-variance curve, but empirically there's a specific capacity point where behavior qualitatively changes.

*Core insight / definition:* The model capacity at which the model can just barely perfectly fit (interpolate) the training data; test error tends to spike near this point before double descent's second decrease in the over-parameterized regime.

*Why it matters / pitfall / interview trap:* Practitioners scaling up model size sometimes see this transient spike and mistakenly conclude a bigger model is "worse" without realizing performance improves again past this point.

---

**Jacobian**

*The problem:* For vector-valued functions of multiple variables, a single derivative doesn't capture how every output component changes with respect to every input component.

*Core insight / definition:* Matrix of all first-order partial derivatives, `J_ij = ∂f_i/∂x_j`, generalizing the scalar derivative to vector-valued, multivariate functions.

*Why it matters / pitfall / interview trap:* Backpropagation is essentially a repeated chain of Jacobian-vector products; the softmax Jacobian's specific structure (`diag(s) - ss^T`) is a common derivation asked directly in interviews.

---

**Jensen's Inequality**

*The problem:* Need to relate the expectation of a convex/concave function to the function of the expectation, which arises when deriving tractable bounds on intractable quantities (e.g., log-likelihoods).

*Core insight / definition:* For a convex function `f`, `f(E[X]) ≤ E[f(X)]` (reversed for concave `f`); applying it to `log` (concave) inside an expectation gives the ELBO lower bound on log-likelihood.

*Why it matters / pitfall / interview trap:* This one inequality is the entire derivation behind why the ELBO is a valid *lower* bound, not an approximation that could go either direction.

---

**K-FAC (Kronecker-Factored Approximate Curvature)**

*The problem:* Second-order optimization needs curvature information (Fisher/Hessian), but the full matrix is far too large to store or invert for any real neural network.

*Core insight / definition:* Approximates each layer's block of the Fisher Information Matrix as a Kronecker product of two much smaller matrices, `F_W ≈ A ⊗ G` (activation covariance ⊗ gradient covariance), making inversion tractable per layer.

*Why it matters / pitfall / interview trap:* Makes natural-gradient-style second-order optimization practical for deep nets at a fraction of full-Hessian cost, but relies on a strong independence approximation between layers that doesn't exactly hold.

---

**K-Means**

*The problem:* Need an unsupervised way to partition unlabeled data into groups without any target variable.

*Core insight / definition:* Iteratively assigns points to the nearest of `K` centroids and updates centroids to the mean of assigned points, minimizing within-cluster variance.

*Why it matters / pitfall / interview trap:* Assumes roughly spherical, equal-sized clusters and requires `K` chosen in advance (Elbow/Silhouette); sensitive to outliers and initialization — poor random starts can trap it in bad local minima (mitigated with k-means++).

---

**KKT Conditions (Karush-Kuhn-Tucker)**

*The problem:* Lagrange multipliers alone handle only equality constraints; most real optimization problems (including SVMs) have inequality constraints too.

*Core insight / definition:* Necessary conditions for optimality in constrained optimization with both equality and inequality constraints: stationarity, primal feasibility, dual feasibility (`λ≥0`), and complementary slackness.

*Why it matters / pitfall / interview trap:* Directly derives the SVM dual problem and support vector structure; interviewers often ask you to identify which points have nonzero multipliers (the support vectors) via complementary slackness.

---

**KL Divergence**

*The problem:* Need to quantify how different one probability distribution is from another, e.g., an approximate posterior from a true one, or a model's output from ground truth.

*Core insight / definition:* `D_KL(P‖Q) = Σ P(x)log(P(x)/Q(x))` — expected extra bits/nats needed to encode `P`-distributed data using a code built for `Q`. Always ≥0, equals 0 iff `P=Q`; not symmetric, not a true metric (no triangle inequality).

*Why it matters / pitfall / interview trap:* Appears throughout ML: VAE latent regularization, DPO's reference-policy penalty, knowledge distillation — and its asymmetry (forward vs reverse) changes what kind of approximation you get, which is frequently probed in interviews.

---

**Knowledge Distillation**

*The problem:* A large, accurate "teacher" model may be too slow/expensive to deploy, but training a small model from scratch on hard labels alone loses information the teacher has learned.

*Core insight / definition:* Train a smaller "student" model to match the teacher's full soft output distribution (with temperature scaling), not just the hard argmax label, transferring the teacher's learned uncertainty structure.

*Why it matters / pitfall / interview trap:* The student's accuracy ceiling is bounded by the teacher's; requires access to the teacher's soft outputs (logits), not just its final predictions.

---

**Kernel Trick**

*The problem:* Many algorithms (SVM, kernel PCA) benefit from operating in a high-dimensional (or infinite-dimensional) feature space, but explicitly computing that mapping is often computationally infeasible.

*Core insight / definition:* Replace explicit feature mapping `φ(x)` with a kernel function `k(x,x') = φ(x)·φ(x')` that computes the inner product in the high-dimensional space directly, without ever materializing `φ(x)`.

*Why it matters / pitfall / interview trap:* Requires the kernel to correspond to a valid inner product in some feature space (Mercer's theorem gives the condition — the kernel matrix must be PSD); using a non-valid "kernel" silently breaks the theoretical guarantees of the algorithm.

---

**KV-Cache**

*The problem:* Naive autoregressive generation recomputes attention Key/Value projections for all previous tokens at every new decoding step, wasting enormous compute.

*Core insight / definition:* Cache the Key and Value tensors for previously generated tokens so each new token only computes its own K/V and attends to the cached history.

*Why it matters / pitfall / interview trap:* Essential for practical LLM serving, but memory grows linearly with sequence length × batch size, which is exactly the problem PagedAttention is designed to manage efficiently.

---

**L-BFGS**

*The problem:* Newton's method needs the full Hessian and its inverse, both `O(d²)`/`O(d³)`, infeasible for large-scale models; but pure gradient descent ignores curvature entirely and converges slowly on ill-conditioned problems.

*Core insight / definition:* A quasi-Newton method that approximates the inverse Hessian using only a limited history of recent gradient/parameter differences (the "two-loop recursion"), avoiding ever forming the full matrix.

*Why it matters / pitfall / interview trap:* Gives much of Newton's fast convergence benefit at a fraction of the memory/compute cost, but is less common in deep learning than in classical ML (e.g., logistic regression) because it doesn't handle the stochastic mini-batch setting as gracefully as SGD-family methods.

---

**L1 Regularization (Lasso)**

*The problem:* With many features, especially irrelevant or redundant ones, an unregularized model can overfit and lacks any mechanism for automatic feature selection.

*Core insight / definition:* Adds `λΣ|w|` to the loss; the diamond-shaped constraint region has corners on the axes, so optimal solutions often land exactly on zero for some weights, producing sparsity. Equivalent to a Laplace prior on weights under a MAP view.

*Why it matters / pitfall / interview trap:* Non-differentiable at zero (needs subgradients); unstable under correlated features — arbitrarily picks one of a correlated group rather than spreading weight across them (unlike L2).

---

**L2 Regularization (Ridge)**

*The problem:* Same overfitting concern as L1, but without needing sparsity — especially relevant when features are correlated and sparsity would be unstable.

*Core insight / definition:* Adds `λΣw²` to the loss; the spherical constraint region gives smooth shrinkage toward zero without ever hitting exact zero. Equivalent to a Gaussian prior on weights under a MAP view.

*Why it matters / pitfall / interview trap:* Stable under multicollinearity (all correlated features are shrunk together rather than one arbitrarily zeroed); doesn't perform feature selection — all features remain in the model, just shrunk.

---

**Lagrange Multipliers**

*The problem:* Need to find the extremum of a function subject to equality constraints, without being able to just set the unconstrained gradient to zero.

*Core insight / definition:* Introduce a multiplier `λ` per constraint and optimize the Lagrangian `L(x,λ) = f(x) + λ·g(x)`; at the optimum, `∇f = λ∇g` — the objective's gradient is parallel to the constraint's gradient.

*Why it matters / pitfall / interview trap:* Generalized to inequality constraints by the KKT conditions; underlies the SVM dual derivation directly.

---

**Law of Large Numbers (LLN)**

*The problem:* Need theoretical justification that averaging over more and more samples actually converges to something meaningful about the underlying population.

*Core insight / definition:* The sample mean converges in probability to the true mean as `n→∞`: `X̄_n →^p μ`. Says nothing about the shape of the sampling distribution (that's CLT's job).

*Why it matters / pitfall / interview trap:* Justifies why more labeled/training data gives more reliable metrics and closer approximation to the true data distribution; commonly confused with CLT, which additionally describes the *shape* of fluctuations around the mean.

---

**LDA (Linear Discriminant Analysis)**

*The problem:* PCA finds directions of maximum variance but ignores class labels entirely, which can discard exactly the directions most useful for classification.

*Core insight / definition:* Supervised dimensionality reduction that finds directions maximizing between-class separability relative to within-class scatter, using label information directly.

*Why it matters / pitfall / interview trap:* Limited to at most `C-1` components for `C` classes, and assumes Gaussian class-conditional distributions with shared covariance — violated in practice but often still useful as a classification preprocessing step.

---

**Learning Rate**

*The problem:* The step size in gradient descent directly controls the tradeoff between convergence speed and stability — too large diverges, too small wastes compute.

*Core insight / definition:* The scalar `η` multiplying the gradient in the parameter update `θ ← θ - η∇L(θ)`.

*Why it matters / pitfall / interview trap:* The single most impactful hyperparameter in most training runs; scaling batch size without correspondingly scaling learning rate (linear LR scaling rule) is a common large-batch training mistake.

---

**Learning Rate Warmup**

*The problem:* Starting training at the target (often large) learning rate can cause instability early on, especially with adaptive optimizers like Adam whose second-moment estimates are unreliable in the first few steps.

*Core insight / definition:* Linearly (or otherwise) ramps the learning rate up from near-zero to its target value over an initial number of steps before applying the main schedule (e.g., cosine decay).

*Why it matters / pitfall / interview trap:* The Adam moment-instability derivation shows the bias-corrected second moment estimate has especially high variance early in training — warmup is a direct practical mitigation for this, not just an empirical trick.

---

**LoRA (Low-Rank Adaptation)**

*The problem:* Full fine-tuning of large pretrained models requires updating and storing all parameters, which is prohibitively expensive for many task-specific adapters.

*Core insight / definition:* Freezes the base weight matrix `W` and adds a trainable low-rank update `ΔW = A·B` where `A,B` have rank `r ≪ d`, drastically reducing trainable parameter count.

*Why it matters / pitfall / interview trap:* The low-rank update can be merged back into `W` at inference time for zero added latency; slightly less flexible than full fine-tuning when the target task is far from the pretraining distribution.

---

**Loss Landscape / Mode Connectivity**

*The problem:* Classical intuition pictures separate, isolated local minima in a non-convex loss surface, which would make model merging or interpolation between trained models meaningless.

*Core insight / definition:* Independently trained (same architecture) networks' minima are often connected by a low-loss path through parameter space (mode connectivity), and after accounting for permutation symmetry of hidden units, sometimes even by a straight line (linear mode connectivity) — the basis for model merging techniques.

*Why it matters / pitfall / interview trap:* Naively averaging two independently trained networks' weights usually fails due to permutation misalignment (unit `i` in one network doesn't correspond to unit `i` in the other) — merging only works well after finding the right permutation alignment first.

---

**MAE / RMSE**

*The problem:* Regression needs a loss/metric quantifying prediction error, and different formulations penalize large errors differently.

*Core insight / definition:* MAE `= (1/n)Σ|y-ŷ|` penalizes errors linearly (robust to outliers, non-differentiable at 0); RMSE `= √MSE` penalizes large errors quadratically (same units as target, more sensitive to outliers).

*Why it matters / pitfall / interview trap:* Choosing MAE vs RMSE is itself a modeling decision about how much large errors should be penalized — RMSE implicitly assumes large errors are disproportionately bad (consistent with a Gaussian noise/MLE view).

---

**MAP (Maximum A Posteriori) Estimation**

*The problem:* MLE has no mechanism to incorporate prior knowledge and can badly overfit with little data (e.g., 3 coin flips all heads gives MLE `p̂=1.0`).

*Core insight / definition:* `θ̂_MAP = argmax_θ [log P(D|θ) + log P(θ)]` — maximizes the posterior, adding a log-prior term that acts as a regularizer.

*Why it matters / pitfall / interview trap:* Still a point estimate with no uncertainty quantification (unlike full Bayesian inference); as `n→∞` the likelihood dominates and MAP→MLE, so "MAP is always better" is false — its advantage is specifically in the small-data regime.

---

**Mercer's Theorem**

*The problem:* Not every symmetric function `k(x,x')` someone proposes as a "kernel" actually corresponds to a valid inner product in some feature space.

*Core insight / definition:* A symmetric function is a valid kernel (corresponds to an inner product in some, possibly infinite-dimensional, feature space) if and only if its corresponding kernel matrix is positive semi-definite for any finite set of points.

*Why it matters / pitfall / interview trap:* Practically, this is why kernel matrices in SVMs must be PSD — using an invalid kernel breaks the convexity/duality guarantees the whole SVM formulation relies on.

---

**MLE (Maximum Likelihood Estimation)**

*The problem:* Given observed data and a parametric model family, need a principled criterion for choosing the "best" parameter values.

*Core insight / definition:* `θ̂_MLE = argmax_θ Σ_i log P(x_i|θ)` — the parameters that make the observed data most probable under the model.

*Why it matters / pitfall / interview trap:* Equivalent to minimizing MSE under a Gaussian likelihood assumption, or cross-entropy under a categorical one — "just use MSE/cross-entropy" is secretly always an MLE assumption about the noise/output distribution; MLE has no built-in regularization and can overfit badly with little data.

---

**Momentum (SGD with Momentum)**

*The problem:* Plain SGD can oscillate across narrow valleys in the loss surface and converges slowly along low-curvature directions.

*Core insight / definition:* Accumulates an exponential moving average of past gradients (velocity) and uses it for the update, smoothing out oscillations and accelerating consistent-direction progress: `v ← βv + ∇L(θ); θ ← θ - ηv`.

*Why it matters / pitfall / interview trap:* Nesterov momentum improves on this by evaluating the gradient at the "look-ahead" position `θ - βv` rather than the current position, giving a corrective term that plain momentum lacks.

---

**Multicollinearity**

*The problem:* When predictor features are highly correlated with each other, the normal equations for linear regression become numerically unstable, and individual coefficient estimates become unreliable/high-variance.

*Core insight / definition:* Near-linear dependence among features makes `X^T X` close to singular (ill-conditioned), inflating coefficient variance even though overall predictions may still be fine.

*Why it matters / pitfall / interview trap:* Ridge regression (L2) directly fixes this by adding `λI` to `X^T X`, guaranteeing invertibility and shrinking unstable coefficient estimates; interpreting individual coefficients under severe multicollinearity is misleading even if the fitted model predicts well.

---

**Multiple Testing Correction**

*The problem:* Running many statistical tests simultaneously inflates the probability that at least one "significant" result is a false positive purely by chance.

*Core insight / definition:* Bonferroni (`α/m`, controls FWER) or Benjamini-Hochberg (controls FDR) adjust significance thresholds to account for the number of tests performed.

*Why it matters / pitfall / interview trap:* "We ran 20 A/B tests and this one metric won" without correction is a textbook false-discovery trap — at α=0.05 with 20 independent tests, one false positive is expected by chance alone.

---

**Mutual Information**

*The problem:* Correlation only captures linear dependence between variables; two variables can be strongly (nonlinearly) related with zero correlation.

*Core insight / definition:* `I(X;Y) = H(X) - H(X|Y)` — the reduction in uncertainty about `X` from observing `Y`; zero if and only if `X` and `Y` are independent (linear or not).

*Why it matters / pitfall / interview trap:* Used for feature selection when a feature's relationship to the target might be nonlinear — correlation-based feature selection would miss such a feature entirely.

---

**Naive Bayes**

*The problem:* Computing the full joint likelihood `P(x_1,...,x_n|y)` for classification requires modeling all feature interactions, which is intractable with many features.

*Core insight / definition:* Assumes features are conditionally independent given the class: `P(y|x) ∝ P(y)·Π_i P(x_i|y)` — reduces the problem to estimating simple per-feature conditional distributions.

*Why it matters / pitfall / interview trap:* The independence assumption is almost always false, yet the classifier often still performs well because the *decision boundary* (argmax over classes) can be correct even when the individual probability estimates are biased.

---

**Natural Gradient**

*The problem:* Standard gradient descent treats parameter space as Euclidean, but the same-sized step in different directions of parameter space can correspond to wildly different changes in the actual model output distribution.

*Core insight / definition:* Rescales the gradient by the inverse Fisher Information Matrix, `θ ← θ - ηF⁻¹∇L`, taking steps that are uniform in terms of distributional change (KL divergence) rather than raw parameter distance.

*Why it matters / pitfall / interview trap:* More invariant to reparameterization than plain gradient descent, but computing/inverting `F` exactly is intractable at scale — K-FAC exists specifically to approximate this efficiently.

---

**NDCG (Normalized Discounted Cumulative Gain)**

*The problem:* Precision/recall treat all correctly retrieved items equally regardless of their rank position, but in ranking tasks, an error at position 1 is much worse than one at position 50.

*Core insight / definition:* `DCG = Σ(relevance_i / log₂(i+1))`, normalized by the ideal DCG ordering, giving a [0,1] score at a chosen cutoff `K` that discounts relevance by rank position.

*Why it matters / pitfall / interview trap:* The right metric whenever position in a ranked list matters (search, recommendations) — using plain precision/recall for ranking tasks throws away exactly the information NDCG is designed to capture.

---

**Newton's Method**

*The problem:* Gradient descent uses only first-order (slope) information and can converge slowly, especially on ill-conditioned (high condition number) loss surfaces.

*Core insight / definition:* Uses second-order (curvature) information: `θ ← θ - H⁻¹∇L(θ)`, taking a step that accounts for the local curvature via the inverse Hessian, achieving quadratic convergence near a minimum.

*Why it matters / pitfall / interview trap:* Computing and inverting the full Hessian is `O(d²)`/`O(d³)`, infeasible for large neural networks — motivating approximations (L-BFGS, K-FAC, Fisher-based natural gradient); also fails badly near saddle points since the Hessian can be indefinite there.

---

**No-Free-Lunch Theorem**

*The problem:* Is there a universally best learning algorithm that outperforms all others across every possible problem?

*Core insight / definition:* Averaged over all possible data-generating distributions, no learning algorithm outperforms any other (including random guessing) — an algorithm's good performance on real-world tasks necessarily comes from an inductive bias matched to the structure of those tasks.

*Why it matters / pitfall / interview trap:* Explains why architecture/algorithm choice matters — "best" is always relative to the assumed structure of the problem class, not universal; a common trap is treating a benchmark win as evidence of universal superiority.

---

**Normal Equation**

*The problem:* Linear regression's least-squares objective is quadratic and convex, so it has a closed-form solution rather than requiring iterative optimization.

*Core insight / definition:* `θ = (X^T X)^{-1} X^T y` — derived by setting the gradient of the squared-error loss to zero.

*Why it matters / pitfall / interview trap:* `X^T X` inversion is `O(d³)` and fails/is unstable when features are collinear (singular or near-singular matrix) — this is exactly the scenario ridge regression's `+λI` term is designed to fix.

---

**PAC Learning (Probably Approximately Correct)**

*The problem:* Need a formal framework answering "how much training data is needed to guarantee a learned hypothesis is close to correct, with high confidence?"

*Core insight / definition:* A hypothesis class is PAC-learnable if, for any `ε,δ>0`, there's a sample size `n` such that with probability `≥1-δ`, the learned hypothesis has true risk `≤ε`. For finite hypothesis classes: `n ≥ (1/ε)(ln|H| + ln(1/δ))`.

*Why it matters / pitfall / interview trap:* Formalizes sample complexity — directly ties hypothesis class size/complexity to how much data is needed for a generalization guarantee; extended to infinite hypothesis classes via VC dimension since `|H|` is infinite for most real model classes.

---

**PAC-Bayes Bound**

*The problem:* Classical PAC bounds via VC dimension are often far too loose (or not applicable) for describing why specific deep learning solutions generalize.

*Core insight / definition:* Bounds generalization gap in terms of the KL divergence between a learned posterior distribution over hypotheses and a fixed prior, connecting flatness of minima to tighter generalization guarantees.

*Why it matters / pitfall / interview trap:* Provides the theoretical grounding for the empirical observation (Keskar et al. 2017) that flatter minima tend to generalize better than sharp ones — motivating sharpness-aware training methods like SAM.

---

**PCA (Principal Component Analysis)**

*The problem:* High-dimensional data is hard to visualize, may contain redundant/correlated features, and many downstream algorithms scale poorly with dimensionality.

*Core insight / definition:* Finds orthogonal directions (principal components) of maximum variance in the data, given by the eigenvectors of the covariance matrix (or via SVD), and projects onto the top-`k` for dimensionality reduction.

*Why it matters / pitfall / interview trap:* Purely linear — will miss/misrepresent nonlinear structure; sensitive to feature scale, so standardization before PCA is essential (a feature with large raw variance dominates components purely due to units, not true importance).

---

**p-value**

*The problem:* Need a way to express how surprising an observed result is under the assumption that there's no real effect.

*Core insight / definition:* `P(observed statistic or more extreme | H₀ is true)` — a probability about the data given the null, not a probability about the null hypothesis itself.

*Why it matters / pitfall / interview trap:* Not `P(H₀ true)`, not a measure of effect size, and not stable near the 0.05 threshold (p=0.049 vs p=0.051 are practically identical) — all three are extremely common interview traps.

---

**Poisson Distribution**

*The problem:* Need to model counts of rare, independent events occurring at a constant average rate over a fixed interval (not a fixed number of trials, unlike Binomial).

*Core insight / definition:* `P(X=k) = e^{-λ}λ^k/k!`, with mean = variance = `λ` — a distinctive signature property.

*Why it matters / pitfall / interview trap:* If observed variance is much larger than the mean ("overdispersion"), the Poisson assumption is violated and negative binomial should be used instead; commonly confused with Binomial (which has a fixed number of trials `n`).

---

**Power (Statistical)**

*The problem:* Need to quantify a test's ability to actually detect a real effect when one exists, not just control false positives.

*Core insight / definition:* `Power = 1 - β` — the probability of correctly rejecting `H₀` when `H₁` is true (detecting a true effect). Determines required sample size for a target minimum detectable effect.

*Why it matters / pitfall / interview trap:* Skipping a power analysis before running an experiment risks an underpowered test that misses a real effect and gets misreported as "no significant difference" (absence of evidence ≠ evidence of absence).

---

**PR-AUC (Precision-Recall Area Under Curve)**

*The problem:* ROC-AUC can look deceptively good on highly imbalanced datasets because it's inflated by the large true-negative pool in the false-positive-rate denominator.

*Core insight / definition:* Area under the precision-recall curve across thresholds — focuses purely on positive-class performance, unaffected by a large true-negative count.

*Why it matters / pitfall / interview trap:* The correct choice over ROC-AUC for imbalanced problems (fraud, rare disease detection) — using ROC-AUC there can mask a genuinely poor minority-class classifier.

---

**Precision**

*The problem:* Need to measure, of everything the model flagged positive, how many were actually correct — critical when false positives are expensive.

*Core insight / definition:* `TP/(TP+FP)`.

*Why it matters / pitfall / interview trap:* Pick over recall when false positives are the costly error (e.g., a spam filter blocking legitimate mail); optimizing precision alone (very conservative threshold) can crash recall.

---

**Propensity Score Matching**

*The problem:* Randomization is infeasible, but rich observed covariates are available, and you need to approximate a randomized comparison from observational data.

*Core insight / definition:* Estimate each unit's probability of receiving treatment given observed covariates (the propensity score), then match treated and untreated units with similar scores to balance observed confounders.

*Why it matters / pitfall / interview trap:* Cannot adjust for confounders that weren't measured (unobserved confounding) — unlike a true RCT, it only balances what you observed, giving false confidence if an important unmeasured confounder exists.

---

**Quantization (Model)**

*The problem:* Full-precision (FP32) model weights/activations consume more memory and compute than necessary for acceptable accuracy at inference time.

*Core insight / definition:* Reduces numeric precision of weights/activations (e.g., to FP16/INT8/INT4). Post-training quantization (PTQ) applies this after training is complete; quantization-aware training (QAT) simulates quantization effects during training so the model adapts to reduced precision.

*Why it matters / pitfall / interview trap:* Aggressive bit-widths (e.g., INT4) risk meaningful accuracy loss, especially with PTQ; QAT recovers more accuracy at the cost of needing to retrain/fine-tune, which PTQ avoids.

---

**Rademacher Complexity**

*The problem:* VC dimension gives a worst-case, distribution-independent capacity measure, but it can be loose because it ignores the actual data distribution the model will see.

*Core insight / definition:* `R̂_S(H) = E_σ[sup_{h∈H} (1/n)Σσ_i h(x_i)]` — measures how well the hypothesis class can fit random `±1` noise (Rademacher variables) on the actual sample, giving a data-dependent capacity measure.

*Why it matters / pitfall / interview trap:* Tighter and more realistic than VC-based bounds because it accounts for the specific data distribution, but is generally harder to compute in closed form for complex model classes like deep networks.

---

**RCT (Randomized Controlled Trial)**

*The problem:* Observational comparisons of "treated" vs "untreated" groups are always vulnerable to confounding, since the groups may differ systematically in ways beyond the treatment itself.

*Core insight / definition:* Randomly assigning units to treatment/control breaks any correlation between treatment assignment and confounders (observed or unobserved), isolating the causal effect.

*Why it matters / pitfall / interview trap:* The gold standard for causal inference, but frequently infeasible (ethical, practical, or cost constraints) — this is exactly why quasi-experimental methods (IV, DiD, RD, PSM) exist as fallbacks.

---

**Recall**

*The problem:* Need to measure, of all actual positives, how many the model successfully caught — critical when false negatives are expensive.

*Core insight / definition:* `TP/(TP+FN)`.

*Why it matters / pitfall / interview trap:* Pick over precision when false negatives are the costly error (e.g., missing a cancer diagnosis); optimizing recall alone (very permissive threshold) can crash precision.

---

**Regression Discontinuity Design**

*The problem:* Treatment assignment sometimes follows a sharp, known threshold rule (e.g., a cutoff score), which precludes randomization but offers a different identification strategy.

*Core insight / definition:* Compares outcomes for units just above vs just below the treatment-assignment cutoff, under the assumption that units near the threshold are otherwise comparable (as if locally randomized).

*Why it matters / pitfall / interview trap:* Only identifies a *local* causal effect near the cutoff — cannot be generalized to units far from the threshold.

---

**Regularization**

*The problem:* Unconstrained model fitting can produce parameters that fit training noise rather than true signal, especially with limited data or high model capacity.

*Core insight / definition:* Adds a penalty term (or algorithmic constraint) discouraging complex/large parameter values, trading a small increase in training error for a (hopefully larger) decrease in generalization error. Equivalent to imposing a prior on parameters under a MAP framing (Gaussian↔L2, Laplace↔L1).

*Why it matters / pitfall / interview trap:* "L2 prevents overfitting" without naming the mechanism (Gaussian prior penalizing large weights via MAP) is an incomplete interview answer; weight decay and L2 are only mathematically equivalent under vanilla SGD, not under Adam (motivating AdamW).

---

**ReLU (Rectified Linear Unit)**

*The problem:* Sigmoid/tanh activations saturate at extremes, causing vanishing gradients that stall training in deep networks.

*Core insight / definition:* `max(0,x)` — gradient is exactly 1 in the active (positive) region, avoiding attenuation there, and is cheap to compute.

*Why it matters / pitfall / interview trap:* Prone to "dying ReLU" — a neuron whose input is always negative gets zero gradient forever and stops learning permanently; this is a distinct failure mode from vanishing gradients, though often conflated.

---

**ROC-AUC**

*The problem:* Need a threshold-independent way to summarize a classifier's ranking quality across all possible decision thresholds.

*Core insight / definition:* Area under the curve of true positive rate vs false positive rate as the decision threshold varies.

*Why it matters / pitfall / interview trap:* Misleadingly optimistic under class imbalance — a large true-negative pool inflates the FPR denominator, making even a weak minority-class classifier look good; use PR-AUC instead when classes are imbalanced.

---

**Sample Complexity**

*The problem:* Need to know, quantitatively, how much labeled data is required before a learning algorithm can be expected to generalize within some error tolerance.

*Core insight / definition:* The minimum sample size `n` needed to guarantee (with confidence `1-δ`) that true risk is within `ε` of optimal, as a function of hypothesis class complexity (`|H|`, VC dimension, or Rademacher complexity).

*Why it matters / pitfall / interview trap:* Directly ties model/hypothesis class complexity to data requirements — a common practical implication is that a more expressive model class requires proportionally more data to achieve the same generalization guarantee.

---

**SAM (Sharpness-Aware Minimization)**

*The problem:* Models that converge to sharp minima (high curvature) tend to generalize worse than those in flat minima, but standard training doesn't explicitly seek flat regions.

*Core insight / definition:* Optimizes a minimax objective, `min_θ max_{‖ε‖₂≤ρ} L(θ+ε)` — finds parameters where the *worst-case* loss within a small neighborhood is minimized, not just the loss at a single point.

*Why it matters / pitfall / interview trap:* Roughly doubles compute per step (needs a gradient computation at the perturbed point too); ASAM fixes SAM's lack of invariance to weight rescaling (Dinh et al. 2017 critique) by normalizing the perturbation per-parameter.

---

**SAGA**

*The problem:* SGD's gradient noise never fully vanishes even near the optimum (a persistent "noise floor"), limiting convergence rate for finite-sum objectives like empirical risk minimization.

*Core insight / definition:* A variance-reduction method that maintains a table of the most recent gradient computed for each individual training example, and uses the average of the table plus a correction term for an unbiased, lower-variance gradient estimate each step.

*Why it matters / pitfall / interview trap:* Achieves faster convergence than plain SGD on convex finite-sum problems by directly attacking the noise floor, but requires storing a per-example gradient table (memory cost scales with dataset size), unlike SVRG.

---

**Saddle Point**

*The problem:* Not every point where the gradient is zero is a minimum — need to distinguish a genuine local minimum from a point that's a minimum along some directions and a maximum along others.

*Core insight / definition:* A critical point (`∇L=0`) where the Hessian is indefinite — positive curvature in some directions, negative in others.

*Why it matters / pitfall / interview trap:* In high-dimensional non-convex landscapes (like deep nets), saddle points vastly outnumber bad local minima and are the more common practical obstacle to optimization; SGD's inherent noise helps escape them, unlike deterministic gradient descent, which can stall near saddles for long periods.

---

**Scaled Dot-Product Attention**

*The problem:* Computing attention scores as raw dot products `QK^T` grows in magnitude with the dimensionality `d_k`, pushing the softmax into a saturated regime with near-zero gradients.

*Core insight / definition:* `softmax(QK^T/√d_k)V` — dividing by `√d_k` keeps the pre-softmax logits at a controlled scale regardless of dimensionality, preserving useful gradient signal.

*Why it matters / pitfall / interview trap:* Explaining exactly *why* `√d_k` prevents softmax saturation (not just that it's a normalization constant) is a common interview follow-up on Transformer internals.

---

**Sigmoid**

*The problem:* Need a smooth function mapping any real-valued score to a valid probability in (0,1) for binary classification output.

*Core insight / definition:* `σ(x) = 1/(1+e^{-x})`.

*Why it matters / pitfall / interview trap:* Saturates at both extremes (gradient → 0 for large |x|), causing vanishing gradients if used in hidden layers of deep networks — appropriate only for the final binary-output layer, not hidden layers.

---

**Simpson's Paradox**

*The problem:* An aggregated trend across a whole dataset can reverse direction when the data is broken into meaningful subgroups, misleading causal conclusions if only the aggregate is examined.

*Core insight / definition:* Caused by a confounding variable correlated with both the grouping variable and the outcome (classic example: UC Berkeley 1973 admissions — aggregate rates favored men, but women had higher admit rates within most individual departments, since women applied disproportionately to more competitive departments).

*Why it matters / pitfall / interview trap:* Always check both aggregated and stratified results before making a causal claim; its appearance is itself a signal that an unaddressed confounder exists.

---

**Softmax**

*The problem:* Need to convert a vector of arbitrary real-valued scores into a valid probability distribution over multiple classes.

*Core insight / definition:* `softmax(x)_i = e^{x_i}/Σ_j e^{x_j}` — exponentiates and normalizes so outputs are positive and sum to 1.

*Why it matters / pitfall / interview trap:* Its Jacobian has the form `diag(s) - ss^T` (a common derivation asked directly); sensitive to large input logits, which is exactly what necessitates the `√d_k` scaling in attention to avoid gradient-killing saturation.

---

**Statistical Power**

See **Power (Statistical)**.

---

**Stratified K-Fold**

*The problem:* Plain random K-fold splitting can produce folds with very different class proportions than the overall dataset, especially problematic under class imbalance.

*Core insight / definition:* Splits data into K folds while preserving the original class distribution in each fold.

*Why it matters / pitfall / interview trap:* Required whenever doing classification cross-validation on imbalanced data — plain K-fold risks a fold with too few (or zero) minority examples, producing unreliable validation metrics.

---

**SVD (Singular Value Decomposition)**

*The problem:* Eigendecomposition only applies to square matrices, but many practically important matrices (data matrices, non-square transformations) are rectangular.

*Core insight / definition:* `A = UΣV^T` — decomposes *any* matrix into orthogonal matrices `U,V` and a diagonal matrix of non-negative singular values `Σ`, ordered by magnitude.

*Why it matters / pitfall / interview trap:* Numerically more stable than eigendecomposition and works on non-square matrices, making it the practical foundation for PCA, low-rank approximation (Eckart-Young), and pseudo-inverses — often preferred over eigendecomposition even when the matrix is square, for stability.

---

**SVM Dual / Support Vectors**

*The problem:* The primal SVM formulation directly optimizes over the weight vector, but doesn't naturally reveal which training points actually determine the decision boundary, or enable the kernel trick.

*Core insight / definition:* Via Lagrangian duality, the SVM optimization is reformulated in terms of dual variables (one per training point); at the optimum, only points on or inside the margin (support vectors) have nonzero dual coefficients (by complementary slackness) — all other points could be removed without changing the boundary.

*Why it matters / pitfall / interview trap:* The dual formulation depends on training points only through pairwise inner products, which is exactly what enables the kernel trick (replacing the inner product with a kernel function) — this connection is a frequent "why does SVM support kernels" interview question.

---

**SVRG (Stochastic Variance Reduced Gradient)**

*The problem:* Plain SGD's gradient estimate has variance that doesn't vanish as training approaches the optimum, capping the achievable convergence rate on finite-sum (empirical risk) objectives.

*Core insight / definition:* Periodically computes a full-batch "anchor" gradient, then at each step corrects the noisy stochastic gradient using the difference between the current and anchor gradient at the same example, producing a lower-variance unbiased estimate.

*Why it matters / pitfall / interview trap:* Achieves faster theoretical convergence than plain SGD on convex problems without storing a full per-example gradient table (unlike SAGA), at the cost of periodic full-batch gradient computations.

---

**t-SNE**

*The problem:* PCA (linear) can fail to reveal cluster structure that only shows up in nonlinear relationships between high-dimensional points.

*Core insight / definition:* Non-linear dimensionality reduction that preserves local neighborhood structure by matching pairwise similarity distributions between high- and low-dimensional space, primarily for 2D/3D visualization.

*Why it matters / pitfall / interview trap:* Does not preserve global distances or cluster sizes/positions — cross-cluster distances and relative cluster sizes in a t-SNE plot are not meaningful and shouldn't be over-interpreted; also not suited as a general preprocessing step for downstream models (unlike PCA/UMAP).

---

**T-test**

*The problem:* Z-tests assume the population variance is known, which is almost never true in practice with real, finite samples.

*Core insight / definition:* `t = (x̄-μ₀)/(s/√n)`, using the sample standard deviation `s` instead of the true `σ`, with `df=n-1`; converges to the z-test as `n` grows large.

*Why it matters / pitfall / interview trap:* The standard choice for comparing sample means in most realistic settings where true variance is unknown — using a z-test when variance is actually unknown understates uncertainty, especially at small `n`.

---

**Type I / Type II Error**

*The problem:* A hypothesis test can be wrong in two structurally different ways, each with different real-world costs depending on context.

*Core insight / definition:* Type I (`α`): rejecting `H₀` when it's actually true (false positive). Type II (`β`): failing to reject `H₀` when it's actually false (false negative, misses a real effect).

*Why it matters / pitfall / interview trap:* At fixed sample size, lowering `α` raises `β` and vice versa — the only way to reduce both simultaneously is more data; the correct tradeoff point depends entirely on the relative real-world cost of each error type (e.g., missed cancer diagnosis vs false alarm).

---

**UMAP**

*The problem:* t-SNE is slow, doesn't scale well, and discards global structure, limiting its usefulness beyond pure visualization.

*Core insight / definition:* Non-linear dimensionality reduction, faster than t-SNE, that better preserves both local neighborhood structure and some global structure, producing embeddings more usable for downstream ML tasks (not just visualization).

*Why it matters / pitfall / interview trap:* Hyperparameter-sensitive (`n_neighbors`, `min_dist` substantially change the output) and less mathematically interpretable than PCA — pick over t-SNE when speed or downstream-task-usable embeddings matter.

---

**Uniform Convergence**

*The problem:* Bounding the generalization gap for a *specific* learned hypothesis is hard directly, since the hypothesis itself depends on the (random) training sample.

*Core insight / definition:* Bounds the gap `sup_{h∈H} |R̂(h) - R(h)|` uniformly over the *entire* hypothesis class simultaneously, so the bound holds no matter which hypothesis the learning algorithm ends up selecting.

*Why it matters / pitfall / interview trap:* This uniform-over-the-class approach is why VC dimension and Rademacher complexity (which measure whole-class capacity) are the right tools for generalization bounds, rather than trying to analyze one specific learned hypothesis in isolation.

---

**VC Dimension (Vapnik-Chervonenkis)**

*The problem:* Need a way to measure the intrinsic capacity/expressiveness of a hypothesis class that works even for infinite classes (where simply counting `|H|` is impossible).

*Core insight / definition:* The size of the largest set of points the hypothesis class can shatter (realize every possible labeling of). E.g., linear classifiers in `R^d` have VC dimension `d+1`; axis-aligned rectangles in 2D have VC dimension 4; 1-nearest-neighbor has infinite VC dimension.

*Why it matters / pitfall / interview trap:* Higher VC dimension means more expressive power but a looser generalization guarantee for fixed sample size — a classic derivation question is showing why 3 points (but not 4, due to the XOR labeling) can be shattered by lines in 2D.

---

**Variance (Statistical/ML)**

*The problem:* Need to quantify how much a model's predictions (or an estimator's value) would change if trained on a different sample from the same distribution.

*Core insight / definition:* `s² = (1/(n-1))Σ(x_i-x̄)²` for a sample (Bessel's correction: dividing by `n-1` instead of `n` corrects for the bias introduced by estimating the mean from the same data); in the bias-variance decomposition, `Var(f̂(x))` is the model-instability component of expected error.

*Why it matters / pitfall / interview trap:* High variance manifests as overfitting (large train/validation gap) — the fix is more data, regularization, or ensembling (bagging), not more model capacity, which is the opposite fix from high bias.

---

**Variational Inference (VI)**

*The problem:* Exact Bayesian posterior inference is intractable for most nontrivial models, and MCMC sampling, while asymptotically exact, is often too slow for large-scale models.

*Core insight / definition:* Approximates the true posterior with a simpler parametric family `Q`, chosen by maximizing the ELBO (equivalent to minimizing reverse KL divergence to the true posterior).

*Why it matters / pitfall / interview trap:* Much faster and more scalable than MCMC (used in VAEs), but only as good as the chosen approximating family, and reverse-KL mode-seeking behavior can cause it to underestimate the true posterior's variance/spread.

---

**Warmup + Cosine Decay**

See **Learning Rate Warmup** and **Cosine Learning Rate Decay**.

---

**Weight Decay vs L2 Regularization**

*The problem:* "Weight decay" and "L2 regularization" are often used interchangeably, but under adaptive optimizers (Adam) they are not mathematically equivalent, leading to subtly wrong regularization behavior if conflated.

*Core insight / definition:* L2 regularization adds a penalty term to the loss before computing gradients, so it gets divided by Adam's adaptive `√v̂` denominator; true weight decay applies the decay directly to the weights as a separate update step, independent of the adaptive scaling.

*Why it matters / pitfall / interview trap:* This distinction is precisely why AdamW (decoupled weight decay) was introduced and is now standard for Transformer training — using "Adam + L2" silently under-regularizes large-adaptive-scale parameters relative to what was intended.

---

**Xavier / He Initialization**

*The problem:* Poor weight initialization can cause activations (and gradients) to shrink or explode exponentially as they propagate through a deep network, before any training has even occurred.

*Core insight / definition:* Xavier (Glorot) initialization scales initial weights based on fan-in/fan-out to keep activation variance stable, derived assuming symmetric activations like sigmoid/tanh; He initialization uses a different scaling factor derived for ReLU's asymmetric, half-zeroing behavior.

*Why it matters / pitfall / interview trap:* Using Xavier init with ReLU networks (or vice versa) systematically mismatches the variance assumptions, contributing to preventable vanishing/exploding gradient issues right from the start of training — a common early-training-stability question.

---

**Z-test**

*The problem:* Need a hypothesis test for a sample mean or proportion when the population variance is known (or the sample is large enough that the sample variance is a reliable stand-in).

*Core insight / definition:* `z = (x̄-μ₀)/(σ/√n)`; reject `H₀` if `|z| > 1.96` for `α=0.05` (two-tailed). The two-sample proportion version for A/B tests: `z = (p̂₁-p̂₂)/√(p̂(1-p̂)(1/n₁+1/n₂))`.

*Why it matters / pitfall / interview trap:* Assumes known/large-sample variance — using it with a small sample and unknown true variance understates uncertainty; the t-test is the more defensible default in most practical (finite-sample) settings.

## 02-data

**ADASYN (Adaptive Synthetic Sampling)**

*The problem:* SMOTE generates synthetic minority samples uniformly, giving equal coverage to easy and hard regions of the minority class, when the hard cases near the decision boundary need more attention.

*Core insight / definition:* ADASYN generates more synthetic samples in regions where the minority class is harder to learn (surrounded by more majority-class neighbors) and fewer in easy regions, by weighting each minority sample's synthetic-sample count by the fraction of its k-nearest-neighbors that belong to the majority class.

*Why it matters / pitfall / interview trap:* The adaptive weighting can amplify noise — outliers in the minority class near the boundary get extra synthetic samples generated around them, reinforcing rather than correcting the noise.

---

**ANOVA F-Statistic (filter feature selection)**

*The problem:* You have continuous features and want to measure linear association between each feature and a categorical target.

*Core insight / definition:* For each feature, compute F = (between-group variance) / (within-group variance) across target classes; a high F-statistic means the feature's class means differ significantly, implying discriminative signal.

*Why it matters / pitfall / interview trap:* ANOVA is a linear test — a feature with a U-shaped relationship to the target (e.g., extreme values in both directions predict the same class) can show a near-zero F-statistic despite being highly predictive; use mutual information for non-linear relationships.

---

**apply() (Pandas)**

*The problem:* Some row-wise or column-wise transformations can't be expressed as a single vectorized operation.

*Core insight / definition:* `.apply()` runs a Python function over each row or column of a DataFrame/Series; it is still a Python-level loop under the hood, roughly 10-50x slower than a fully vectorized operation.

*Why it matters / pitfall / interview trap:* Reach for vectorized ops first; use `.apply()` only as a last resort when logic genuinely can't be vectorized — never default to it out of convenience.

---

**Backward Elimination**

*The problem:* You want the best-performing feature subset for a model using actual cross-validated performance rather than importance scores.

*Core insight / definition:* A greedy sequential search that starts with all features and, at each step, removes the single feature whose removal least damages (or most improves) cross-validated score, repeating until a target feature count is reached.

*Why it matters / pitfall / interview trap:* Like forward selection, it is greedy and can miss globally optimal subsets when features are only jointly (synergistically) predictive.

---

**BalancedRandomForest**

*The problem:* You want an ensemble method to internally handle class imbalance without altering the global training distribution via a separate resampling step.

*Core insight / definition:* Each tree in the forest is trained on a bootstrap sample drawn with equal class balance rather than the original imbalanced ratio, so every base learner sees a class-balanced view of the data.

*Why it matters / pitfall / interview trap:* Balanced bootstrap sampling changes the effective class prior the model learns under, relative to the deployment distribution — this can cause probability miscalibration requiring post-hoc correction.

---

**Bivariate / Multivariate Analysis**

*The problem:* Univariate summaries miss relationships between features and between features and the target.

*Core insight / definition:* Techniques such as correlation heatmaps, pairplots, grouped boxplots, and crosstabs that examine relationships between two or more variables — catching multicollinearity and giving an early read on which features may predict the target.

*Why it matters / pitfall / interview trap:* This exploration is cheap compared to training a full model just to discover a feature has no predictive relationship with the target.

---

**Broadcasting**

*The problem:* Performing elementwise operations on arrays of different shapes would otherwise require manually copying/tiling one array to match the other's shape, wasting memory and code.

*Core insight / definition:* NumPy's rule for operating on differently-shaped arrays without copying data: compare shapes right-to-left, pad missing leading dimensions with 1, and treat any dimension of size 1 as "stretched" to match the other array's size — done virtually, with no memory copy.

*Why it matters / pitfall / interview trap:* Getting shapes wrong produces either a `ValueError` or, worse, a silent bug — e.g. adding a `(3,)` array to a `(3,1)` array broadcasts to `(3,3)` instead of the intended elementwise `(3,)` result, with no error raised.

---

**Cardinality (of a categorical feature)**

*The problem:* The right encoding strategy for a categorical feature depends heavily on how many distinct values it has.

*Core insight / definition:* Cardinality is the count of distinct categories in a feature. Low cardinality (e.g., 3-10 categories) suits one-hot encoding; high cardinality (thousands of categories, e.g., city or user ID) makes one-hot impractical and favors target encoding, frequency encoding, or embeddings.

*Why it matters / pitfall / interview trap:* `df.nunique()` during EDA flags high-cardinality (often ID-like) columns early, before an encoding choice blows up feature-matrix size.

---

**Category dtype (Pandas)**

*The problem:* Low-cardinality string (`object`) columns waste memory because Pandas stores each repeated string value as a separate boxed Python object.

*Core insight / definition:* `astype("category")` stores each unique value once and represents rows via integer codes referencing that dictionary, dramatically reducing memory and speeding up groupby/comparison operations.

*Why it matters / pitfall / interview trap:* Converting a low-cardinality `object` column to `category` can cut memory usage 10x or more — a classic memory-optimization interview question.

---

**Chained Indexing / SettingWithCopyWarning**

*The problem:* Assigning into the result of two sequential indexing operations (`df[cond]["col"] = x`) may silently fail to modify the original DataFrame.

*Core insight / definition:* `df[condition]` may return either a view or a copy depending on internal Pandas heuristics; chaining a second `[...]` on top makes it ambiguous whether the assignment touches the original data or a discarded temporary copy.

*Why it matters / pitfall / interview trap:* Always use a single `.loc[row_selector, col_selector] = value` call for conditional assignment instead of chained indexing.

---

**Chi-Squared Test (feature selection)**

*The problem:* You need to test whether a non-negative feature is statistically independent of a classification target.

*Core insight / definition:* Chi-squared measures how much the observed cross-tabulation of feature values and class labels deviates from what would be expected if they were independent; a low p-value indicates dependence (likely informative feature).

*Why it matters / pitfall / interview trap:* Requires non-negative features (fails on z-scored/standardized data) and a discrete/binned feature and label — doesn't directly extend to regression targets.

---

**Class Imbalance**

*The problem:* When one class vastly outnumbers another (e.g., 99% legitimate vs 1% fraud), a model can achieve high accuracy while learning nothing about the minority class.

*Core insight / definition:* The model's loss/gradients are dominated by majority-class examples, biasing it toward always predicting the majority class unless the loss function, sampling distribution, or class weights are explicitly adjusted.

*Why it matters / pitfall / interview trap:* This is both a training problem (biased gradients) and a metric problem (accuracy is meaningless) — fixing only one without the other leaves a broken evaluation or a broken model.

---

**class_weight='balanced' (Cost-Sensitive Learning)**

*The problem:* You want to correct for class imbalance without modifying the training data itself (no synthetic samples, no discarded rows).

*Core insight / definition:* Assigns each class a weight inversely proportional to its frequency (`w_j = n_samples / (n_classes * n_j)`), multiplying the loss contribution of minority-class errors so the optimizer is forced to reduce them; sklearn, XGBoost (`scale_pos_weight`), and PyTorch (`pos_weight`) all support variants.

*Why it matters / pitfall / interview trap:* Class weights change which errors are penalized, not the predicted probabilities' calibration — if calibrated probabilities are needed, apply post-hoc calibration after cost-sensitive training.

---

**Concept Drift**

*The problem:* A deployed model's predictions can become wrong even if the input distribution hasn't changed, because the relationship between inputs and outcome has shifted.

*Core insight / definition:* Concept drift is when P(y|X) changes — the underlying relationship the model learned no longer holds, distinct from covariate shift (P(X) changes) or label shift (P(y) changes).

*Why it matters / pitfall / interview trap:* Distinguishing which type of drift occurred determines the fix — concept drift generally requires retraining with new labels, not just recalibrating on new input statistics.

---

**Correlation-Based Redundancy Removal**

*The problem:* Two highly correlated features add no new information but double the noise and can cause multicollinearity in linear models.

*Core insight / definition:* Compute the pairwise correlation matrix, flag feature pairs with |r| above a threshold (e.g., 0.95), and drop one feature from each redundant pair, keeping a single representative.

*Why it matters / pitfall / interview trap:* Pearson correlation only detects linear relationships — two features can be highly redundant non-linearly (e.g., one is the square of the other) while showing zero Pearson correlation, so this method misses non-linear redundancy.

---

**Cost-Sensitive Learning**

*The problem:* Resampling modifies training data (synthetic points, discarded rows), which can introduce artifacts; an alternative approach avoids touching the data at all.

*Core insight / definition:* Keep the data unchanged but assign a higher weight/cost to misclassifying minority-class samples in the loss function, so the optimizer must reduce those errors to minimize total weighted loss — achieving a similar rebalancing effect as resampling without data modification.

*Why it matters / pitfall / interview trap:* Preferred when interpretability or data integrity matters, since no synthetic or duplicated samples are introduced; but doesn't produce calibrated probabilities out of the box.

---

**Covariate Shift**

*The problem:* Model performance can degrade in production even when the learned input-output relationship is still valid, because the inputs themselves have changed.

*Core insight / definition:* Covariate shift is when P(X) — the distribution of the input features — changes between training and production, while P(y|X) may remain stable.

*Why it matters / pitfall / interview trap:* Detected via distributional tests like PSI or KS test on individual features; but multivariate covariate shift (joint distribution shift with stable marginals) can be invisible to univariate drift checks.

---

**Data Drift**

*The problem:* A model deployed and monitored only on accuracy metrics can degrade silently over months as the real world diverges from the training distribution, with no code change to blame.

*Core insight / definition:* Data drift is any divergence between the distribution of inputs (or the input-output relationship) seen in production versus training; it encompasses covariate shift, label shift, and concept drift, and requires quantitative monitoring (PSI, KS test) to detect before business metrics visibly suffer.

*Why it matters / pitfall / interview trap:* Waiting until business metrics fall to notice drift is too late — proactive statistical monitoring is required, but PSI/KS thresholds are rules of thumb, not guarantees (e.g., seasonal variation can trip a PSI threshold without true model degradation).

---

**Data Leakage**

*The problem:* Information that would not be available at prediction time can accidentally enter the training features, making validation metrics look great while production performance collapses.

*Core insight / definition:* Leakage is information from outside the (legitimate) training set — often, indirectly, from the target itself — leaking into features or preprocessing statistics. Common sources: preprocessing fit before the split, target-proxy or post-outcome features, temporal leakage (random splits on time series), group leakage (same entity in train and val), and target-encoding leakage (using a row's own label in its encoding).

*Why it matters / pitfall / interview trap:* The rule of thumb: fit nothing on data the model won't see in production — split first, transform second, fit only on train. If a validation metric looks too good, leakage is the first hypothesis to check, not model excellence.

---

**Datetime Decomposition**

*The problem:* A raw timestamp (e.g., Unix integer) is almost never directly useful as a model feature.

*Core insight / definition:* Extracting components like day-of-week, hour, month, quarter, `is_weekend`, `is_month_end`, or days-since-last-event from a datetime column turns an opaque number into features the model can actually use to find cyclic or seasonal patterns.

*Why it matters / pitfall / interview trap:* Feature engineering here bridges domain knowledge (e.g., "3am transactions are riskier") into a representation the model can exploit, since it cannot discover this from the raw timestamp alone.

---

**Dbt / Airflow**

*The problem:* Data pipelines need reliable orchestration and transformation tooling to run reproducibly at scale.

*Core insight / definition:* Airflow is a workflow orchestrator for scheduling and monitoring pipeline DAGs; dbt is a SQL-based transformation tool for defining and testing data warehouse transformations. (Full treatment moved to the production-ML data-engineering doc.)

*Why it matters / pitfall / interview trap:* Referenced as part of the broader data-engineering-for-ML toolkit (batch/streaming, feature stores, data quality) consolidated outside this section.

---

**Dimensionality Reduction (vs. Feature Selection)**

*The problem:* Both feature selection and dimensionality reduction shrink the number of model inputs, but they trade off interpretability differently and are often conflated.

*Core insight / definition:* Feature selection keeps a subset of the original, interpretable features; dimensionality reduction (PCA, UMAP, t-SNE, autoencoders) creates new transformed features via compression, which may lack a clear real-world interpretation.

*Why it matters / pitfall / interview trap:* For a regulated model requiring explanation of feature contributions to regulators, feature selection is necessary; PCA components can't be explained as original measurable variables, so using them with feature-importance explanations is a category error.

---

**dtype (NumPy/Pandas data type)**

*The problem:* Choosing an inappropriately wide or generic numeric/data type wastes memory and can silently change computation semantics.

*Core insight / definition:* `dtype` specifies the underlying binary representation of array/column values (e.g., `int64`, `float32`, `object`, `category`). Default float is `float64` (8 bytes); deep learning typically prefers `float32`/`float16`/`bfloat16`. Mixing types (int+float, float32+float64) silently upcasts to the wider type.

*Why it matters / pitfall / interview trap:* Silent upcasting during mixed-type arithmetic is a common way to accidentally double a pipeline's memory footprint without any error being raised.

---

**Elastic Net**

*The problem:* Lasso (L1) tends to arbitrarily pick one feature from a group of correlated features, discarding the rest even when several correlated features are jointly meaningful.

*Core insight / definition:* Elastic Net combines the L1 penalty (which induces sparsity/feature selection) with the L2 penalty (which shrinks correlated features together as a group) rather than picking one arbitrarily.

*Why it matters / pitfall / interview trap:* Preferred over plain Lasso when many correlated features should be selected/retained together instead of one survivor chosen essentially at random by the solver.

---

**Embedded Methods (Feature Selection)**

*The problem:* Wrapper methods require training the model repeatedly on different feature subsets, which is expensive.

*Core insight / definition:* Embedded methods perform feature selection as a byproduct of model training itself — one training pass, no extra cost — e.g., L1/Lasso regularization zeroing out coefficients, or tree-based feature importances.

*Why it matters / pitfall / interview trap:* Cheaper than wrapper methods but still tied to the specific model's inductive biases (e.g., Lasso assumes linear relationships and a sparse true model).

---

**Embeddings (categorical encoding)**

*The problem:* One-hot encoding a very-high-cardinality categorical (user IDs, words) creates an impractically large, mostly-empty sparse matrix.

*Core insight / definition:* Embeddings represent each category as a learned, dense, low-dimensional vector trained jointly with the rest of a neural network, capturing semantic similarity between categories rather than treating them as equally distinct.

*Why it matters / pitfall / interview trap:* Only practical inside a neural network training loop — not a preprocessing step usable with classical linear/tree models without a separate embedding-training step.

---

**EasyEnsemble**

*The problem:* Undersampling the majority class to balance training data discards potentially useful majority examples in a single pass.

*Core insight / definition:* Trains an ensemble of AdaBoost classifiers, each on a different random undersampled majority-class subset paired with all minority-class samples, effectively using many different majority subsets across the ensemble instead of throwing most of the majority data away once.

*Why it matters / pitfall / interview trap:* Like BalancedRandomForest, this changes the effective class prior seen during training relative to deployment, risking probability miscalibration.

---

**EDA (Exploratory Data Analysis)**

*The problem:* Committing modeling assumptions to data before understanding its actual shape, types, distributions, and anomalies risks building on faulty premises.

*Core insight / definition:* EDA is the iterative process of building intuition about what's actually in the data — first-pass checks (`shape`, `info`, `describe`, `nunique`, `duplicated`), univariate analysis (histograms, boxplots, value counts), and bivariate/multivariate analysis (correlation heatmaps, pairplots, crosstabs) — before committing to cleaning or modeling decisions.

*Why it matters / pitfall / interview trap:* EDA is a loop, not a line — insights from later pipeline stages (feature engineering, modeling) should send you back to re-explore, not just a one-time upfront step.

---

**Feature Crosses / Interaction Terms**

*The problem:* Additive models (like linear regression) cannot learn that the joint combination of two features (e.g., "weekend AND beach location") predicts an outcome differently than either feature alone.

*Core insight / definition:* A feature cross explicitly creates a new feature as the product (or combination) of two existing features, $f_1 \times f_2$, letting a linear model represent an interaction it otherwise couldn't discover automatically.

*Why it matters / pitfall / interview trap:* Critical for linear models; tree-based models can discover some interactions natively through sequential splits, reducing (but not eliminating) the need for manual crosses.

---

**Feature Engineering**

*The problem:* The model can only discover patterns representable in the features actually provided to it — raw data rarely presents domain knowledge in a directly usable form.

*Core insight / definition:* Feature engineering is the practice of constructing new features (crosses, ratios, temporal decompositions, transforms) that encode domain knowledge in a form the model's inductive biases can exploit, bridging the gap between raw data and useful representation.

*Why it matters / pitfall / interview trap:* Engineered features must be computable using only information available at prediction time (else it's leakage) and should be monitored for predictive stability, since a feature predictive historically can stop being predictive as underlying behavior shifts.

---

**Feature Selection (overview)**

*The problem:* Training on all available features (e.g., 500) when only a fraction carry genuine signal invites overfitting to noise, slower training, and harder interpretability/debugging.

*Core insight / definition:* Feature selection reduces the feature set to those carrying genuine predictive signal, pruning redundant, noisy, or causally-downstream (leaky) features. Three families: filter methods (score features independent of a model), wrapper methods (use actual model performance on subsets), and embedded methods (selection built into training).

*Why it matters / pitfall / interview trap:* Must be fit only on training data — running any selection method (even simple univariate correlation) on the full dataset before splitting leaks test-label information into which features get kept.

---

**Filter Methods (Feature Selection)**

*The problem:* With thousands of features, you need a cheap, model-agnostic way to prune obvious dead weight before spending compute on model training.

*Core insight / definition:* Filter methods score each feature independently of any model — using variance, correlation, mutual information, chi-squared, or ANOVA F-statistic — and are fast but blind to feature interactions.

*Why it matters / pitfall / interview trap:* A feature that's individually useless can still be jointly predictive with another feature; filter methods systematically miss such interaction effects.

---

**F1 Score**

*The problem:* Precision and recall trade off against each other, and you often need a single number that captures both without favoring the one that's trivially easy to inflate.

*Core insight / definition:* F1 is the harmonic mean of precision and recall: `F1 = 2*P*R/(P+R)`. The harmonic mean penalizes extreme imbalance between precision and recall more harshly than a simple average would.

*Why it matters / pitfall / interview trap:* F1 ignores true negatives entirely — a model that aggressively predicts positive can inflate recall (and F1) while generating large false-positive volumes; MCC captures all four confusion-matrix quadrants where F1 does not.

---

**F-Beta Score**

*The problem:* F1 treats false positives and false negatives as equally costly, but in domains like medical screening a missed positive is far more costly than a false alarm.

*Core insight / definition:* $F_\beta = (1+\beta^2)\cdot \frac{P\cdot R}{\beta^2 P + R}$ generalizes F1 with a parameter β that weights recall more heavily (β>1) or precision more heavily (β<1), letting the metric reflect asymmetric real-world costs.

*Why it matters / pitfall / interview trap:* Choosing β requires an explicit judgment about the relative cost of false positives vs. false negatives in the business context — it isn't a free hyperparameter to tune blindly.

---

**Focal Loss**

*The problem:* In extreme imbalance settings (e.g., 1000:1 background-to-foreground ratio in object detection), standard cross-entropy is overwhelmed by easy majority-class examples, and even class weighting doesn't fully counteract their aggregate contribution to the loss.

*Core insight / definition:* Focal loss adds a modulating factor $(1-p_t)^\gamma$ to cross-entropy that dynamically down-weights examples the model already classifies confidently, letting hard examples (regardless of class) dominate the gradient: $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$.

*Why it matters / pitfall / interview trap:* Requires tuning two extra hyperparameters (α, γ); designed for dense per-pixel prediction tasks, and its benefit on tabular binary classification is less consistent — try class weighting first.

---

**Forward Selection**

*The problem:* You want to find the feature subset that actually maximizes cross-validated model performance, not just importance-score rankings.

*Core insight / definition:* A greedy sequential search that starts with an empty feature set and, at each step, adds whichever single feature most improves cross-validated score, repeating until a target size is reached.

*Why it matters / pitfall / interview trap:* Requires O(n_features × k) model fits to select k features from n — computationally expensive with slow estimators, and the greedy strategy can miss globally optimal, synergistic feature combinations.

---

**Frequency Encoding**

*The problem:* High-cardinality categoricals are impractical to one-hot encode, but target encoding carries leakage risk if done incorrectly.

*Core insight / definition:* Replace each category with its frequency (count or proportion) in the dataset, producing a single numeric column with no leakage risk, at the cost of weaker predictive signal than target encoding.

*Why it matters / pitfall / interview trap:* A safe fallback when target-encoding leakage risk is a concern, but it discards the target-relationship signal that makes target encoding powerful in the first place.

---

**GroupBy / Split-Apply-Combine**

*The problem:* Computing per-group aggregations (e.g., total revenue per user) via manual loops is slow and error-prone.

*Core insight / definition:* Pandas' `.groupby()` implements split-apply-combine: split rows into groups by key, apply an aggregation (`.agg()`, reduces to one row per group) or `.transform()` (returns same-length output broadcasting the group statistic back to every row), then combine results. Implemented as roughly O(n) via hash-based grouping.

*Why it matters / pitfall / interview trap:* `.transform()` is the idiomatic way to compute "subtract the group mean from every row" without a manual join; avoid `.groupby().apply(custom_python_fn)` on large data — it's as slow as `iterrows()`.

---

**Group K-Fold / Group Leakage**

*The problem:* If multiple rows belong to the same entity (patient, user, device), a naive random split can place some of that entity's rows in train and others in validation, letting the model implicitly "recognize" the entity rather than generalize.

*Core insight / definition:* Group leakage occurs when the same entity appears across the train/validation boundary; `GroupKFold` fixes this by splitting on the entity ID so all of one entity's rows land in a single fold.

*Why it matters / pitfall / interview trap:* Easy to miss in interview settings because it looks like a normal random split just with row-level, not entity-level, structure — always check whether rows are grouped by an underlying entity before choosing a CV strategy.

---

**Imputation (Missing Value Handling)**

*The problem:* Missing values in a feature can't be fed directly into most models, but the "right" way to fill them depends entirely on why they're missing.

*Core insight / definition:* Simple imputation fills missing values with a summary statistic (median for numeric — robust to outliers — mode for categorical); model-based imputation (KNNImputer, IterativeImputer) predicts the missing value from other observed features. Must be fit only on training data and applied (not refit) to validation/test.

*Why it matters / pitfall / interview trap:* Mean/median imputation is only safe under MCAR; under MAR or MNAR it can shrink variance, distort relationships, and mask the fact that "missingness itself" was predictive — often paired with a missingness indicator column instead of, or alongside, the imputed value.

---

**IQR (Interquartile Range) Outlier Detection**

*The problem:* You need a distribution-agnostic rule of thumb to flag values far outside the typical range of a feature.

*Core insight / definition:* IQR = Q3 − Q1 (75th minus 25th percentile); values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` are flagged as outliers, a robust alternative to assuming normality.

*Why it matters / pitfall / interview trap:* A statistically extreme value is not automatically an error — in fraud/anomaly domains, "outliers" flagged by IQR are often exactly the signal you're trying to model, so detection must be followed by domain judgment, not automatic removal.

---

**iterrows() (Pandas)**

*The problem:* Looping over DataFrame rows in pure Python is a common but severe anti-pattern for performance.

*Core insight / definition:* `.iterrows()` boxes every row into a `Series` object during iteration, making it roughly 100x+ slower than a vectorized column operation and 2-5x slower even than `.apply()`.

*Why it matters / pitfall / interview trap:* Seeing `iterrows()` in a data pipeline is almost always a red flag for "why is this job so slow" — prefer vectorized ops, fall back to `.apply()` only when necessary, and essentially never use `iterrows()`.

---

**Kolmogorov-Smirnov (KS) Test**

*The problem:* You need a statistical test to detect whether a feature's distribution has shifted between training and production.

*Core insight / definition:* The KS test compares the empirical CDFs of two samples; the KS statistic is the maximum vertical difference between the two CDFs, and the associated p-value tests the null hypothesis that both samples come from the same distribution.

*Why it matters / pitfall / interview trap:* A per-feature (univariate) KS test misses multivariate drift, where two features' individual distributions look stable but their joint distribution has shifted.

---

**Label Shift**

*The problem:* A model's predictions can degrade if the base rate of the outcome itself changes, even if the input-output relationship and input distribution are stable.

*Core insight / definition:* Label shift is when P(y) — the prior probability of each class — changes between training and production, distinguished from covariate shift (P(X) changes) and concept drift (P(y|X) changes).

*Why it matters / pitfall / interview trap:* Recognizing which of the three drift types occurred (covariate, label, concept) determines the correct remediation — treating a label-shift problem as if it were concept drift wastes retraining effort on the wrong fix.

---

**Label Smoothing**

*The problem:* Under class imbalance, cross-entropy loss rewards the model for outputting extremely confident probabilities on majority-class examples, which are not genuinely that certain and produce poorly calibrated outputs.

*Core insight / definition:* Replace hard 0/1 targets with soft targets, $y_{smooth} = y(1-\varepsilon) + \varepsilon/K$, so the model can never perfectly minimize loss by outputting probability 1.0, reducing overconfidence.

*Why it matters / pitfall / interview trap:* Reduces calibration error but also softens the signal on genuinely easy examples; it's complementary to focal loss and class weighting, not a replacement for either.

---

**Lasso (L1 Regularization) for Feature Selection**

*The problem:* With L2 regularization, all model weights shrink toward zero but essentially never reach exactly zero, so you can't interpret the result as a feature selection.

*Core insight / definition:* The L1 penalty (sum of absolute weight values) has a geometric constraint region with corners aligned to the coordinate axes; the optimum often lands exactly on a corner, driving many weights to precisely zero, so the model performs feature selection as a side effect of fitting: minimize `Loss + λΣ|w_j|`.

*Why it matters / pitfall / interview trap:* When features are highly correlated, Lasso arbitrarily picks one from the group (the choice can depend on random seed); it also assumes the true underlying model is sparse and will underselect if many features each contribute small genuine effects.

---

**Matthews Correlation Coefficient (MCC)**

*The problem:* F1 ignores true negatives, so a model that aggressively predicts positive can score well on F1/recall while generating excessive false positives — you need a metric sensitive to all four confusion-matrix outcomes.

*Core insight / definition:* MCC is effectively the correlation coefficient between true and predicted binary labels: $\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$, ranging from -1 (perfect inverse) through 0 (random) to +1 (perfect), and gives a high score only when the model performs well on both classes.

*Why it matters / pitfall / interview trap:* Considered robust to class imbalance, making it a good complement (or alternative) to F1/PR-AUC in heavily skewed classification problems.

---

**Mean Decrease Impurity (MDI)**

*The problem:* After training a random forest, you want to know which features it actually relied on.

*Core insight / definition:* MDI sums, for each feature, the weighted impurity decrease across every node in every tree where that feature was used to split, normalized to sum to 1 across all features — the standard `feature_importances_` in sklearn tree ensembles.

*Why it matters / pitfall / interview trap:* MDI is systematically biased toward high-cardinality features — a random ID column with thousands of unique values can rank near the top of importance even with zero genuine signal; permutation importance is the bias-free alternative.

---

**Merges / Joins (Pandas)**

*The problem:* Combining two tables on a shared key is one of the most common — and most bug-prone — data operations.

*Core insight / definition:* `pd.merge(a, b, on=key, how=...)` implements SQL-style joins: `inner` (matching keys only), `left`/`right` (all rows from one side, NaN where unmatched on the other), `outer` (all rows from both, NaN where unmatched).

*Why it matters / pitfall / interview trap:* A many-to-many join on a key with duplicate values silently explodes the row count via a Cartesian product on the duplicated keys — always sanity-check `len(result)` against expectations after a join.

---

**Missingness Indicator**

*The problem:* Imputing a missing value with a single number destroys the information that the value was originally missing, which can itself be predictive.

*Core insight / definition:* A binary column (e.g., `col_was_missing`) added alongside an imputed feature that flags whether the original value was absent, letting the model separately learn the effect of the value's presence and the effect of its absence.

*Why it matters / pitfall / interview trap:* Especially important under MNAR, where the missingness itself (e.g., "high earners refuse to report income") carries signal no imputation method can recover on its own.

---

**MCAR / MAR / MNAR (Missingness Mechanisms)**

*The problem:* Choosing an imputation strategy without understanding *why* data is missing can bias the model or destroy predictive signal.

*Core insight / definition:* MCAR (Missing Completely At Random) — missingness independent of any data; safe to impute simply. MAR (Missing At Random) — missingness depends on other *observed* variables; safe to impute conditioning on those variables. MNAR (Missing Not At Random) — missingness depends on the *unobserved* value itself (e.g., high earners refusing to report income); imputation can introduce bias, and a missingness indicator is often the better/complementary response.

*Why it matters / pitfall / interview trap:* Treating all missingness as MCAR and defaulting to mean/median imputation is a classic mistake — under MAR/MNAR it introduces bias by ignoring the dependence between missingness and other variables.

---

**Min-Max Normalization**

*The problem:* Some algorithms or downstream uses need feature values bounded within a known, fixed range.

*Core insight / definition:* $(x - x_{min}) / (x_{max} - x_{min})$ rescales a feature to [0,1] (or another bounded range), preserving relative spacing but using the min/max as reference points.

*Why it matters / pitfall / interview trap:* Sensitive to outliers — a single extreme value compresses the rest of the distribution into a tiny sub-range; use RobustScaler instead when outliers are present.

---

**Missing Value Diagnosis Tools**

*The problem:* Before choosing a handling strategy, you need to see how much data is missing and whether missingness is correlated across columns.

*Core insight / definition:* `df.isnull().mean()` reports the percentage missing per column; `missingno.matrix(df)` visualizes the missingness pattern across the whole DataFrame to reveal correlated missingness.

*Why it matters / pitfall / interview trap:* Correlated missingness across columns is often a clue pointing toward MAR or MNAR rather than MCAR, changing the appropriate handling strategy.

---

**Mutual Information (Feature Selection)**

*The problem:* Pearson correlation only detects linear relationships; a feature with zero correlation to the target can still carry genuine, non-linear predictive signal that a naive filter would discard.

*Core insight / definition:* Mutual information $I(X_i;Y) = \sum P(x,y)\log\frac{P(x,y)}{P(x)P(y)}$ measures statistical dependence of any form (not just linear) between a feature and the target, estimated via k-nearest-neighbor density estimation; features are ranked by MI score and top-k selected.

*Why it matters / pitfall / interview trap:* MI estimates are high-variance on small datasets, and MI still evaluates features individually — it does not detect interaction effects where a feature is only jointly predictive with another.

---

**NumPy Arrays (vs. Python Lists)**

*The problem:* Python lists store pointers to boxed objects with significant per-element overhead, making numerical computation over large collections slow.

*Core insight / definition:* NumPy arrays store raw typed values contiguously in memory with a shape/stride descriptor, with no per-element Python object overhead, enabling operations to run as compiled C loops (via BLAS/LAPACK) instead of interpreted Python loops.

*Why it matters / pitfall / interview trap:* This contiguous-memory-plus-C-loop model is *why* NumPy (and everything built on it: Pandas, PyTorch tensors, JAX) is fast — the foundational fact interviewers probe first.

---

**One-Hot Encoding**

*The problem:* Most ML algorithms require numeric input, but nominal categorical values have no inherent numeric ordering.

*Core insight / definition:* Creates one new binary column per category, encoding the assumption that all categories are equally different from each other; best suited to low-cardinality, nominal (unordered) categoricals, especially for linear models. `pd.get_dummies(..., drop_first=True)` avoids the "dummy variable trap" (perfect multicollinearity) for linear models.

*Why it matters / pitfall / interview trap:* At high cardinality it creates a huge, mostly-zero sparse matrix — impractical without switching to a sparse matrix representation or a different encoding (target/frequency/embeddings).

---

**Ordinal Encoding**

*The problem:* Genuinely ordered categorical values (low/medium/high) need an encoding that preserves their order, unlike one-hot's assumption of equal distinctness.

*Core insight / definition:* Maps each category to an integer preserving its natural order (e.g., low=1, medium=2, high=3).

*Why it matters / pitfall / interview trap:* Applying ordinal encoding to nominal (unordered) categories incorrectly imposes a numeric ordering assumption the categories don't actually have — a common misuse flagged in interviews.

---

**Outliers**

*The problem:* Extreme values in a feature can distort statistics (mean, std), dominate loss functions, and violate model assumptions — but removing them without investigation can also destroy the exact signal you care about.

*Core insight / definition:* Detected via the IQR method or z-score (`|z| > 3`, assumes roughly normal data). The key judgment isn't detection but decision: distinguish data errors (age = 200 → fix/drop) from genuine extreme values (a large fraud transaction, a rare but valid measurement) that may be the most important rows in the dataset.

*Why it matters / pitfall / interview trap:* In fraud/anomaly detection specifically, the "outlier" is frequently the actual target class — blind statistical outlier removal in these domains destroys the very signal being modeled.

---

**Pandas DataFrame / Series**

*The problem:* Working with heterogeneous tabular data (mixed types across columns, labeled rows/columns) is awkward using raw NumPy arrays alone.

*Core insight / definition:* A `Series` is a NumPy array plus an index; a `DataFrame` is a dict of `Series` (columns) sharing a common index. Every Pandas performance question reduces to whether an operation is vectorized (fast, C-level) or row-wise (slow, Python-level).

*Why it matters / pitfall / interview trap:* Understanding this internal structure explains why certain operations (column-wise math) are fast while others (`iterrows`, row-wise `.apply`) are slow.

---

**Permutation Importance**

*The problem:* MDI (mean decrease impurity) feature importance is biased toward high-cardinality features, making it unreliable for pruning.

*Core insight / definition:* Shuffle one feature's values at a time and measure the resulting drop in validation performance; the size of the drop is that feature's importance — bias-free relative to MDI since it's measured directly on held-out predictive performance.

*Why it matters / pitfall / interview trap:* Correlated features protect each other under permutation — if feature A and B carry the same signal, shuffling A alone doesn't hurt much because B compensates, so both can show artificially low importance despite being jointly critical.

---

**Pipeline (sklearn) / ColumnTransformer**

*The problem:* Preprocessing steps applied inconsistently between training and inference (or across CV folds) is a major source of leakage and train/serve skew.

*Core insight / definition:* A `Pipeline` chains preprocessing and modeling steps into one object that guarantees identical transformations at fit time and inference time; `ColumnTransformer` applies different pipelines to different column subsets (e.g., numeric vs. categorical) and combines the results.

*Why it matters / pitfall / interview trap:* Leakage-proof by construction — `cross_val_score` on a Pipeline refits preprocessing on each fold's train split automatically, so you cannot accidentally fit a scaler on validation data; also enables single-object serialization (`joblib.dump`) so production doesn't need to reimplement feature engineering by hand.

---

**Polars**

*The problem:* Pandas' eager execution model and single-threaded design becomes a bottleneck on multi-GB datasets or heavy aggregation workloads.

*Core insight / definition:* Polars is a Rust-based DataFrame library with a lazy, query-optimizing execution engine (`pl.scan_csv(...).filter(...).select(...).collect()`); it builds a query plan first, pushing filters (predicate pushdown) and column selection (projection pushdown) into the file scan before reading data, using Apache Arrow memory layout and automatic multi-threading. `collect(streaming=True)` processes data in chunks at constant memory for out-of-core workloads.

*Why it matters / pitfall / interview trap:* Beats Pandas by 5-10x on large multi-GB groupbys due to the query optimizer, but Pandas still wins on ecosystem maturity — most ML libraries (scikit-learn, XGBoost) expect Pandas/NumPy input directly, requiring `.to_pandas()` conversion.

---

**Population Stability Index (PSI)**

*The problem:* You need a single quantitative score for how much a feature's distribution has shifted between a training reference window and a production window.

*Core insight / definition:* PSI compares binned distributions: $\text{PSI} = \sum_i (A_i - E_i)\ln(A_i/E_i)$, where $A_i$ is the actual (production) fraction in bin i and $E_i$ is the expected (training) fraction. PSI < 0.1 = stable; 0.1-0.25 = moderate shift, investigate; > 0.25 = major shift, consider retraining.

*Why it matters / pitfall / interview trap:* PSI thresholds are heuristics, not guarantees — a feature can exceed PSI 0.25 due to benign seasonal variation that doesn't actually impair model performance, so a PSI alarm should trigger investigation, not automatic retraining.

---

**PR-AUC (Precision-Recall AUC / Average Precision)**

*The problem:* ROC AUC can look deceptively good on heavily imbalanced data because the enormous pool of true negatives keeps the true-negative rate high even when many positives are missed.

*Core insight / definition:* PR-AUC (Average Precision) ignores true negatives entirely, measuring how well the model ranks actual positives among the examples it flags as positive — exactly the question that matters when positives are rare.

*Why it matters / pitfall / interview trap:* Preferred over ROC-AUC specifically under class imbalance; ROC-AUC in this regime is "optimistic" and can mask a model that's actually failing to catch the minority class.

---

**Precision / Recall**

*The problem:* Accuracy is meaningless on imbalanced data, so you need metrics that isolate performance specifically on the positive (often minority, rare-event) class.

*Core insight / definition:* Precision = TP/(TP+FP) — of predicted positives, what fraction are real. Recall = TP/(TP+FN) — of real positives, what fraction were caught. The two are in tension: raising recall (predicting positive more aggressively) typically reduces precision.

*Why it matters / pitfall / interview trap:* The classic imbalanced-data interview scenario: a 99% accurate model on a 1% positive-rate dataset can have 0% recall — always check recall, not just accuracy, on skewed classification problems.

---

**Ratio Features**

*The problem:* Two absolute quantities individually may carry little predictive signal, but their ratio often captures the actual underlying relationship of interest.

*Core insight / definition:* Engineering a feature as the ratio of two existing quantities (debt-to-income, click-through rate, `transaction_amount / account_balance`) that is more informative than either quantity alone.

*Why it matters / pitfall / interview trap:* Classic fraud-detection example: a large transaction amount alone isn't suspicious, but a large transaction *relative to account balance* is — the ratio surfaces the signal neither raw feature reveals independently.

---

**Recursive Feature Elimination (RFE / RFECV)**

*The problem:* You want the best feature subset for a specific model, iteratively removing the least useful features as importance estimates sharpen.

*Core insight / definition:* Train the model on all features, rank by coefficient magnitude or importance, remove the least important feature(s), retrain on the remainder, and repeat until a target feature count is reached; RFECV adds cross-validation to select that target count automatically.

*Why it matters / pitfall / interview trap:* Runs one full model refit per elimination step (O(n_features) fits minimum), making it slow with expensive models — use fast linear estimators for large feature sets; MDI-based RFE on random forests inherits the high-cardinality bias of MDI importance.

---

**Robust Scaling**

*The problem:* StandardScaler's mean and standard deviation are themselves distorted by extreme outliers, defeating the purpose of "robust" normalization when outliers are present.

*Core insight / definition:* $(x - \text{median}) / \text{IQR}$ centers and scales using the median and interquartile range instead of mean/std, since neither is dragged around by extreme values the way mean/std are.

*Why it matters / pitfall / interview trap:* The go-to choice specifically when a feature has genuine, meaningful outliers you want to preserve (not remove) while still preventing them from dominating gradient-based training.

---

**Seaborn / Matplotlib**

*The problem:* Effective EDA and reporting require both fast statistical visualization and full pixel-level plotting control.

*Core insight / definition:* Matplotlib is the low-level plotting substrate offering full control; Seaborn is a high-level statistical visualization library built on Matplotlib that operates directly on DataFrames (histograms with KDE, pairplots, correlation heatmaps).

*Why it matters / pitfall / interview trap:* Use Seaborn for fast statistical exploration during EDA; drop to Matplotlib when Seaborn has no recipe for what you need, or pixel-level control (custom annotations, publication figures) is required.

---

**Sequential Feature Selection**

*The problem:* You want a wrapper-method feature subset chosen by actual cross-validated performance rather than model-internal importance scores.

*Core insight / definition:* A greedy search (forward or backward) that at each step evaluates all possible single feature additions or removals via cross-validation and keeps whichever change most improves score, stopping at a target feature count.

*Why it matters / pitfall / interview trap:* Computationally expensive — O(n_features × k) model fits for forward selection of k features from n — and, like all greedy searches, can miss globally optimal, synergistic feature combinations.

---

**SelectFromModel**

*The problem:* You want to plug any model's feature-importance scores into an automated selection step inside a standard sklearn pipeline.

*Core insight / definition:* Wraps any estimator exposing `feature_importances_` or `coef_`, dropping features below a specified (or automatic) importance threshold.

*Why it matters / pitfall / interview trap:* `threshold='median'` drops exactly half the features by definition regardless of whether that's the meaningful cutoff — better to use a threshold grounded in a natural break in the importance distribution.

---

**SHAP (SHapley Additive exPlanations)**

*The problem:* MDI importance is biased toward high-cardinality features and permutation importance misses redundancy between correlated features; you want importances consistent with the model's actual predictions and interaction effects.

*Core insight / definition:* SHAP assigns each feature a contribution to each individual prediction by fairly distributing the prediction across all possible feature orderings (Shapley values from cooperative game theory); the mean absolute SHAP value across samples gives global importance. TreeSHAP computes exact SHAP values in polynomial time for tree models.

*Why it matters / pitfall / interview trap:* SHAP values are consistent and locally accurate but can still be confounded by correlated features — highly correlated features split SHAP credit between them, making each individually look less important than their true combined contribution.

---

**SMOTE (Synthetic Minority Oversampling Technique)**

*The problem:* Random oversampling merely duplicates minority-class samples, causing the model to memorize exact points instead of learning the underlying minority-class region.

*Core insight / definition:* For each minority sample, find its k-nearest minority neighbors, pick one at random, and generate a new synthetic sample by interpolating between the two: $x_{new} = x + \lambda(x_{nn}-x)$, $\lambda \sim \text{Uniform}(0,1)$ — introducing diversity rather than exact copies.

*Why it matters / pitfall / interview trap:* SMOTE interpolates without awareness of the decision boundary — when classes overlap, synthetic points can land inside majority-class territory while being labeled minority; also doesn't work on categorical features directly (use SMOTE-NC). Must only be applied inside the training fold (via `imblearn.pipeline.Pipeline`), never before CV splitting or on the test set.

---

**SMOTEENN / SMOTETomek**

*The problem:* Oversampling alone (SMOTE) can generate synthetic samples in ambiguous regions near the decision boundary.

*Core insight / definition:* Combination resamplers that oversample first (SMOTE) then clean the boundary: SMOTETomek removes Tomek-link majority members (gentler); SMOTEENN removes any sample (from either class) whose label disagrees with the majority of its k neighbors via Edited Nearest Neighbors (more aggressive).

*Why it matters / pitfall / interview trap:* Both reduce total training set size after the resample-and-clean cycle — on very small datasets, the cleaning step can discard too much data.

---

**Sparse Matrices (CSR/CSC)**

*The problem:* A one-hot encoded categorical with 100K categories over 1M rows would be an 800GB dense float64 matrix — infeasible to store or compute with.

*Core insight / definition:* Sparse matrix formats (Compressed Sparse Row for fast row slicing, Compressed Sparse Column for fast column slicing/matrix-vector products) store only non-zero entries, reducing an infeasible dense footprint to a few MB for mostly-zero data like one-hot or TF-IDF features.

*Why it matters / pitfall / interview trap:* Most sklearn linear models accept sparse input directly — any feature matrix from text (TF-IDF, bag-of-words) or high-cardinality categoricals should be constructed/stored as sparse, not densified.

---

**Standardization (Z-score Scaling)**

*The problem:* Features on very different numeric scales (e.g., income 0-500,000 vs. age 0-100) cause distance-based and gradient-based algorithms to implicitly weight the larger-range feature as more "important," regardless of its actual informativeness.

*Core insight / definition:* $(x - \mu)/\sigma$ centers a feature to zero mean and unit variance; required for distance-based models (KNN, SVM) and gradient-based models (linear/logistic regression, neural nets) so gradients and distances behave proportionally to actual information content rather than raw magnitude.

*Why it matters / pitfall / interview trap:* Must be fit (`fit_transform`) only on the training set and applied (`transform` only) to validation/test — fitting on the full dataset before splitting is a textbook leakage bug; tree-based models need no scaling at all since splits are invariant to monotonic transforms of a single feature.

---

**Stratified K-Fold**

*The problem:* A plain random K-fold split on an imbalanced classification dataset can produce folds with wildly different (or even zero) positive-class representation, making evaluation unreliable.

*Core insight / definition:* Stratified K-Fold preserves the overall class ratio within every fold, so each fold is a representative miniature of the full class distribution.

*Why it matters / pitfall / interview trap:* Without stratification on heavily imbalanced data, a fold may contain no minority-class examples at all, making both training and evaluation on that fold meaningless.

---

**Target Encoding**

*The problem:* One-hot encoding a high-cardinality categorical produces thousands of mostly-empty sparse columns; you want a compact numeric representation that also captures the category's relationship to the target.

*Core insight / definition:* Replace each category with (a smoothed version of) the mean target value for that category: $\text{enc}(c) = \lambda_c \bar{y}_c + (1-\lambda_c)\bar{y}_{global}$, $\lambda_c = n_c/(n_c+k)$ — shrinking toward the global mean for rare categories so small counts don't produce wild estimates.

*Why it matters / pitfall / interview trap:* Must be computed out-of-fold (K-fold target encoding) — encoding a row using its own label (or same-fold labels) is direct label leakage; also requires a fallback strategy (global mean, "unseen" token) for categories not seen during training.

---

**Temporal Leakage / Time Series Split**

*The problem:* A random train/validation split on time-ordered data can let the model "see the future" — training on data that would not exist yet at the moment predictions are actually made in production.

*Core insight / definition:* Temporal leakage is any information flow from later-in-time rows into training for predicting earlier-in-time outcomes; the fix is a chronological/time-based split (`TimeSeriesSplit`), training on past folds and validating on the next chunk forward, never backward, and never shuffling across time.

*Why it matters / pitfall / interview trap:* A model that "sees the future" during training can post an artificially strong validation score that will never be achievable in real deployment, since production data genuinely doesn't exist yet at prediction time.

---

**Threshold Moving**

*The problem:* The default 0.5 decision threshold implicitly assumes balanced classes; on a 99/1 split, predicted probabilities for the rare positive class are naturally low, so a 0.5 cutoff misses most true positives.

*Core insight / definition:* Since the model outputs probabilities, the threshold is a separate, post-hoc decision about where to draw the predicted-class boundary; sweeping thresholds and choosing the one that maximizes a target metric (F1, F-beta, or an explicit cost function) shifts the operating point on the precision-recall curve at zero additional training cost.

*Why it matters / pitfall / interview trap:* Threshold tuning must be done on a separate validation set, never the test set (that's leakage); the chosen optimal threshold may not generalize perfectly to production, especially under class-distribution drift.

---

**Tomek Links**

*The problem:* After oversampling, the class boundary can be cluttered with ambiguous samples from both classes, making a clean separation hard to learn.

*Core insight / definition:* A Tomek link is a pair of samples from different classes that are each other's nearest neighbor — sitting exactly on or near the decision boundary; removing the majority-class member of each link cleans the boundary with minimal data loss.

*Why it matters / pitfall / interview trap:* Produces only minimal undersampling on its own (only boundary-adjacent points are removed) — used primarily as a post-processing/cleaning step after oversampling (SMOTETomek), not as a standalone balancing technique.

---

**Train-Serve Skew**

*The problem:* A feature computed one way during training and a different way (or from a different data source/freshness) in production creates a mismatch invisible to offline evaluation.

*Core insight / definition:* Train-serve skew is any discrepancy between how a feature is computed in the training pipeline versus the real-time production serving pipeline — e.g., training on a complete historical database but serving from a cache updated every 6 hours.

*Why it matters / pitfall / interview trap:* Especially insidious because it doesn't show up in offline validation (which uses the same training pipeline to construct the validation set) — it only manifests once real production serving-time feature values differ, making it a common cause of "the model was fine in testing but degraded in production."

---

**Transformations (Log, Polynomial, Yeo-Johnson)**

*The problem:* Skewed distributions cause gradient-based models to be dominated by long-tail extreme values, and violate normality assumptions in statistical methods.

*Core insight / definition:* `log1p(x)` compresses a right-skewed, non-negative distribution's long tail while safely handling x=0 (unlike plain `log`, which is undefined at 0); polynomial features (`x**2`) let linear models capture curvature; Yeo-Johnson generalizes log-style transforms to handle zero and negative values.

*Why it matters / pitfall / interview trap:* Log-transforming a target requires remembering to invert the transform (`expm1`) before reporting predictions in the original units — a common bug is forgetting this and reporting log-space numbers as if they were dollars/counts.

---

**Validation Strategy (K-Fold / Nested CV)**

*The problem:* Choosing the wrong cross-validation scheme for the data's structure (i.i.d., temporal, grouped, imbalanced) silently invalidates the resulting performance estimate.

*Core insight / definition:* Random K-Fold suits i.i.d. tabular data with no time or group structure; Stratified K-Fold preserves class ratio for imbalanced classification; `TimeSeriesSplit` respects chronological order; `GroupKFold` keeps an entity's rows within one fold; Nested CV runs an outer loop for unbiased performance evaluation around an inner loop used for hyperparameter tuning, avoiding tuning-induced optimism.

*Why it matters / pitfall / interview trap:* Nested CV specifically addresses the subtle leakage of using the same CV loop for both hyperparameter selection and final performance reporting, which otherwise overstates generalization performance.

---

**Variance Threshold**

*The problem:* A feature that barely varies across samples cannot help discriminate between different labels and wastes a model weight/split.

*Core insight / definition:* Compute each feature's variance and drop any feature below a chosen threshold, since a near-constant feature carries no discriminative information regardless of the model used downstream.

*Why it matters / pitfall / interview trap:* Binary features that are 99% one class have naturally low variance (≈0.01) even if the rare 1% is highly predictive — applying a variance threshold too aggressively can discard exactly these useful rare-event indicator features.

---

**Vectorization**

*The problem:* Explicit Python-level loops over array elements are slow because each iteration pays interpreter overhead.

*Core insight / definition:* Vectorization expresses a computation as array-level operations (NumPy/Pandas column ops) so the actual looping happens in compiled C/Fortran (BLAS/LAPACK) rather than the Python interpreter — often 100x faster for large N.

*Why it matters / pitfall / interview trap:* Any explicit `for` loop iterating over array/DataFrame elements in numerical code is treated as a code smell — the reflex should be "how do I express this as an array operation instead?"

---

**Views vs. Copies (NumPy)**

*The problem:* Some indexing operations share memory with the original array while others create independent copies, and mixing up which is which causes silent data-corruption bugs.

*Core insight / definition:* Basic slicing (`a[2:5]`) returns a view that shares memory with the parent — mutating it mutates the original. Fancy indexing (`a[[2,3,4]]`, `a[mask]`) and most arithmetic operations return copies, independent of the source.

*Why it matters / pitfall / interview trap:* The number-one source of "why did my original array change?!" bugs in multi-stage data pipelines — call `.copy()` explicitly whenever independence is required and you're not certain which behavior an operation has.

---

**Wrapper Methods (Feature Selection)**

*The problem:* Filter methods evaluate features in isolation and cannot detect that a feature is only useful in combination with another.

*Core insight / definition:* Wrapper methods (Recursive Feature Elimination, Sequential Feature Selection) train the actual model on different feature subsets and use its measured generalization performance to guide selection, making the choice a function of the specific learning algorithm rather than a model-agnostic statistic.

*Why it matters / pitfall / interview trap:* Far more computationally expensive than filter methods since each step requires (re-)training the model — impractical with slow estimators and large feature counts unless a fast surrogate model is used during selection.

---

**Yeo-Johnson Transform** — see Transformations (Log, Polynomial, Yeo-Johnson)

## 03-classical-ml

**AIC / BIC (Akaike/Bayesian Information Criterion)**

*The problem:* Choosing among candidate ARIMA/statistical models by likelihood alone always favors more parameters, overfitting in-sample.

*Core insight / definition:* AIC = 2k − 2ln(L̂); BIC = k·ln(n) − 2ln(L̂), where k is the number of parameters and n the sample size. Both penalize model complexity against goodness of fit; lower is better. BIC penalizes complexity more heavily than AIC because its penalty scales with ln(n).

*Why it matters / pitfall / interview trap:* auto_arima uses AIC/BIC to search (p,d,q) automatically, but in-sample AIC is not a substitute for walk-forward validation — a model can win on AIC and still fail out-of-sample.

---

**ACF / PACF (Autocorrelation / Partial Autocorrelation Function)**

*The problem:* To specify an ARIMA model's AR and MA orders, you need to see how a series correlates with its own past.

*Core insight / definition:* ACF at lag k is Cov(Xₜ, Xₜ₋ₖ)/Var(Xₜ) — total correlation including indirect paths through intermediate lags. PACF is the correlation at lag k after removing the effects of lags 1..k-1. If ACF tails off and PACF cuts off at lag p → AR(p); if ACF cuts off at lag q and PACF tails off → MA(q); both tailing off → ARMA(p,q).

*Why it matters / pitfall / interview trap:* neither tailing off (slow ACF decay) signals non-stationarity — difference the series before fitting.

---

**ADASYN (Adaptive Synthetic Sampling)**

*The problem:* SMOTE generates synthetic minority points uniformly, even in regions already well-represented, ignoring that harder-to-learn regions need more help.

*Core insight / definition:* A variant of SMOTE that generates more synthetic samples for minority examples that are harder to classify (near the decision boundary, surrounded by majority-class neighbors) and fewer for those already easy to classify.

*Why it matters / pitfall / interview trap:* like SMOTE, still interpolates in feature space and can create unrealistic points in multimodal minority distributions.

---

**ADWIN (Adaptive Windowing)**

*The problem:* Streaming data distributions drift over time; a fixed-window anomaly/drift detector either reacts too slowly or too noisily.

*Core insight / definition:* An online drift-detection algorithm that maintains a variable-length window of recent values and statistically tests whether the window can be split into two sub-windows with significantly different means, shrinking the window when drift is detected.

*Why it matters / pitfall / interview trap:* used for streaming anomaly/drift detection in time series where re-fitting a batch model is impractical.

---

**AdaBoost**

*The problem:* Combine many weak learners (e.g., decision stumps) into a strong classifier by focusing sequentially on the examples previous learners got wrong.

*Core insight / definition:* Trains learners sequentially; after each round, misclassified examples get higher sample weights so the next learner focuses on them. Each learner's vote is weighted by αₘ, a function of its weighted error rate (lower error → higher α). Final prediction is a weighted majority vote.

*Why it matters / pitfall / interview trap:* highly sensitive to noisy labels and outliers, since misclassified noisy points keep gaining weight round after round.

---

**Agglomerative Clustering (see Hierarchical Clustering)**

---

**ALE (Accumulated Local Effects)**

*The problem:* Partial Dependence Plots assume feature independence, giving misleading curves when features are correlated (extrapolating into unrealistic feature combinations).

*Core insight / definition:* ALE plots compute the effect of a feature by accumulating local differences in predictions within small windows of the feature's actual conditional distribution, avoiding the need to extrapolate outside the observed data manifold.

*Why it matters / pitfall / interview trap:* preferred over PDP when features are correlated; still an average effect, not per-instance explanation.

---

**Anomaly Detection**

*The problem:* Flagging rare, evolving events (fraud, intrusions, defects) where positive labels are scarce, biased toward attacks already caught, and future anomalies may not resemble past ones.

*Core insight / definition:* Instead of learning "what fraud looks like" (supervised), model what normal looks like and flag deviations — isolation (Isolation Forest), boundary (One-Class SVM, Elliptic Envelope), density (LOF), or reconstruction (Autoencoder) framings. Generalizes to unseen anomaly types since it models absence of normalcy rather than presence of a known signature.

*Why it matters / pitfall / interview trap:* choosing unsupervised detection when reliable labels are actually plentiful is a common, expensive mistake — supervised classification wins when labels are good and anomalies resemble each other.

---

**ANOVA F-test (feature selection)**

*The problem:* Quickly rank numeric features by relevance to a categorical target before modeling.

*Core insight / definition:* A filter method that computes, for each feature, the F-statistic from a one-way ANOVA comparing feature means across target classes; high F-value implies the feature's distribution differs meaningfully between classes.

*Why it matters / pitfall / interview trap:* assumes roughly normal, equal-variance feature distributions per class; captures only linear/mean differences, missing non-linear relevance.

---

**ARIMA(p, d, q)**

*The problem:* Model and forecast a univariate time series whose past values and past errors both carry predictive signal, after removing non-stationarity.

*Core insight / definition:* Combines AR(p) (regression on p past values), differencing d times to induce stationarity, and MA(q) (regression on q past error terms): Xₜ = c + Σφᵢ Xₜ₋ᵢ + εₜ + Σθⱼ εₜ₋ⱼ applied to the d-times-differenced series.

*Why it matters / pitfall / interview trap:* over-differencing (d too large) or excessive p/q leads to overfitting; order should be chosen via ADF test + ACF/PACF inspection or AIC/BIC search, validated with walk-forward CV.

---

**ARI (Adjusted Rand Index)**

*The problem:* Evaluate clustering quality when ground-truth labels exist, correcting for agreement expected by chance.

*Core insight / definition:* Measures similarity between predicted cluster assignments and true labels, adjusted so that random labelings score near 0 (1.0 = perfect agreement).

*Why it matters / pitfall / interview trap:* unlike raw Rand Index, ARI is chance-corrected, making scores comparable across different numbers of clusters.

---

**Autoencoder**

*The problem:* Capture complex non-linear low-dimensional structure that no linear (PCA) or kernel method can represent, or detect anomalies via reconstruction difficulty.

*Core insight / definition:* A neural network with an encoder compressing input x∈ℝᵈ into a bottleneck z∈ℝᵏ (k≪d) and a decoder reconstructing x̂ from z, trained to minimize reconstruction loss ‖x − g(f(x))‖². The bottleneck forces the network to discard non-essential information; z is the learned embedding. PCA is the special case of a linear autoencoder with no activations.

*Why it matters / pitfall / interview trap:* no global optimum guarantee (unlike PCA's SVD), can overfit, and latent dimensions are not directly interpretable; for anomaly detection, high reconstruction error signals deviation from the learned "normal" manifold.

---

**Average Precision (see PR-AUC)**

---

**Bagging (Bootstrap Aggregating)**

*The problem:* A single high-variance model (e.g., deep decision tree) is unstable — small training-set changes produce very different models.

*Core insight / definition:* Train many instances of the same learner on independent bootstrap samples (sampling with replacement) of the training data, then average (regression) or majority-vote (classification) their predictions. Reduces variance without increasing bias, since averaging independent noisy estimators shrinks variance by roughly 1/n (less due to correlation between bootstraps).

*Why it matters / pitfall / interview trap:* bagging does not reduce bias — bagging a high-bias model just gives you an ensemble of equally-biased models; use it specifically to tame high-variance base learners.

---

**Bayesian Optimization**

*The problem:* Hyperparameter search budgets are small (each evaluation trains an expensive model) and random search wastes budget re-exploring regions already known to be poor.

*Core insight / definition:* Maintain a cheap surrogate model (e.g., Gaussian Process or TPE) of the objective function fit on past (config, score) pairs; use an acquisition function (Expected Improvement, UCB, Thompson Sampling) to pick the next configuration balancing exploration and exploitation; update the surrogate after each evaluation.

*Why it matters / pitfall / interview trap:* inherently sequential — parallelizing breaks the "informed by all past results" assumption; GP-based variants scale O(n³) in evaluations, impractical past ~500 trials.

---

**Bayesian interpretation of regularization**

*The problem:* Explain why L1/L2 penalties correspond to specific prior beliefs about model weights, not just ad hoc penalty terms.

*Core insight / definition:* Ridge (L2) regularization is equivalent to MAP estimation under a Gaussian prior on weights; Lasso (L1) is equivalent to MAP estimation under a Laplace (double-exponential) prior, whose sharp peak at zero induces sparsity.

*Why it matters / pitfall / interview trap:* explains geometrically why L1's diamond-shaped constraint region has corners on the axes (producing exact zeros) while L2's spherical region does not.

---

**Bias-Variance Tradeoff**

*The problem:* Diagnose why a model performs poorly — is it too simple or too complex relative to the data?

*Core insight / definition:* Expected squared error decomposes as Bias² + Variance + irreducible noise σ². Bias is systematic error from wrong model assumptions (same mistake across resamples); variance is error from sensitivity to the specific training sample (different mistakes across resamples); σ² cannot be reduced by any model.

*Why it matters / pitfall / interview trap:* applying regularization to a high-bias (underfit) model makes it worse; assuming more data alone fixes variance ignores that an unregularized model will simply re-overfit to the larger set. The tradeoff only holds within a fixed model family — better architecture/features can reduce both simultaneously.

---

**Bagging vs Boosting (bias-variance framing)**

*The problem:* Choose the right ensemble strategy depending on whether the base learner suffers from high bias or high variance.

*Core insight / definition:* Bagging averages independent high-variance, low-bias learners (deep trees) to cancel variance. Boosting sequentially adds weak, high-bias, low-variance learners (shallow trees/stumps), each correcting the residual error of the ensemble so far, reducing bias.

*Why it matters / pitfall / interview trap:* using boosting on an already low-bias, high-variance learner (deep tree) amplifies overfitting; boosting is more sensitive to noisy labels than bagging.

---

**Bernoulli Naive Bayes**

*The problem:* Classify data described by binary (present/absent) features, e.g., word occurrence in short text.

*Core insight / definition:* Assumes each feature is a Bernoulli-distributed binary indicator conditioned on the class; explicitly penalizes the absence of a feature, unlike Multinomial NB which ignores absent features.

*Why it matters / pitfall / interview trap:* better than MultinomialNB for short documents where presence/absence (not count) is what matters.

---

**BIC (see AIC / BIC)**

---

**Binary Cross-Entropy / Log-Loss**

*The problem:* Train a probabilistic binary classifier with a loss that penalizes confident wrong predictions heavily and rewards well-calibrated probabilities.

*Core insight / definition:* L = −[y log(p) + (1−y) log(1−p)]; a proper scoring rule minimized when predicted probability p equals the true conditional probability P(y=1|x). Used to fit logistic regression via maximum likelihood.

*Why it matters / pitfall / interview trap:* minimizing log-loss does not guarantee optimizing accuracy or recall, especially under class imbalance — a well-calibrated model can still have poor recall if it hedges near 0.5.

---

**Blending**

*The problem:* Combine multiple base models' predictions like stacking, but avoid the computational/complexity cost of full cross-validated out-of-fold predictions.

*Core insight / definition:* Train base models on a training subset, generate predictions on a single held-out validation set, then train a meta-learner on those held-out predictions. Simpler than stacking (single split vs. k-fold OOF), but uses less data for the meta-learner and risks overfitting to that one split.

*Why it matters / pitfall / interview trap:* often confused with stacking; the key difference is stacking uses cross_val_predict/OOF, blending uses one static holdout.

---

**BOHB (Bayesian Optimization with Hyperband)**

*The problem:* Bayesian optimization guides search but ignores cheap early-stopping signals; Hyperband stops early but proposes configurations randomly.

*Core insight / definition:* Combine a TPE surrogate to propose promising configurations with Hyperband's multi-fidelity early-stopping schedule to evaluate them cheaply — informed search plus cheap early rejection.

*Why it matters / pitfall / interview trap:* requires a surrogate that can integrate observations from multiple budget levels; standard for expensive deep-learning HPO but overkill for cheap-to-train models.

---

**Boosting**

*The problem:* Improve a high-bias (weak) base learner by having successive models correct the errors of prior ones, rather than averaging independent models.

*Core insight / definition:* Sequential ensemble construction where each new model is trained to reduce the residual error (or reweighted misclassifications) of the current ensemble. Includes AdaBoost (reweighting), Gradient Boosting (fitting pseudo-residuals), and their tree-based implementations (XGBoost, LightGBM, CatBoost).

*Why it matters / pitfall / interview trap:* reduces bias but can increase variance/overfit if too many rounds or too high learning rate are used without regularization.

---

**Box-Cox Transform**

*The problem:* A time series exhibits unstable (often increasing) variance over time, violating stationarity assumptions.

*Core insight / definition:* A parametrized power transform (including log as a special case) chosen to stabilize variance and make the series closer to homoscedastic/Gaussian.

*Why it matters / pitfall / interview trap:* one of several standard stationarity-inducing transforms alongside differencing and log transform; choice depends on whether the instability is exponential growth (log) vs general variance instability (Box-Cox).

---

**Calibration (Model Calibration)**

*The problem:* A model's predicted probability (e.g., 0.85) must reflect the true empirical frequency of the positive outcome, or downstream decisions (reserves, pricing, thresholds) will be systematically wrong.

*Core insight / definition:* A model is calibrated if P(Y=1 | p̂(X)=p) ≈ p for all p. Logistic regression is natively calibrated (optimizes log-loss, a proper scoring rule) when well-specified; SVMs, boosted trees, and neural nets optimized for margin/accuracy are typically not calibrated and need post-hoc correction (Platt scaling, isotonic regression, temperature scaling).

*Why it matters / pitfall / interview trap:* a highly accurate model can still be badly miscalibrated; conflating "high AUC" with "trustworthy probabilities" is a common mistake.

---

**Calibration Before Ensembling**

*The problem:* Averaging probability outputs from differently-calibrated models (e.g., SVM margins vs. tree leaf fractions) in a soft-voting ensemble biases the combined score toward whichever model's outputs happen to span a wider range.

*Core insight / definition:* Apply CalibratedClassifierCV (Platt or isotonic) to each base model before combining probabilities, so all inputs to the ensemble are on a comparable, meaningful probability scale.

*Why it matters / pitfall / interview trap:* skipping this step silently lets one poorly-calibrated model dominate a soft vote regardless of its actual accuracy.

---

**Calinski-Harabasz Index**

*The problem:* Evaluate clustering quality without ground-truth labels, based purely on cluster geometry.

*Core insight / definition:* Ratio of between-cluster dispersion to within-cluster dispersion (scaled by degrees of freedom); higher values indicate dense, well-separated clusters.

*Why it matters / pitfall / interview trap:* like other internal indices, biased toward convex, similarly-sized clusters — can favor wrong k on non-convex clusters (e.g., DBSCAN-appropriate shapes).

---

**CatBoost**

*The problem:* Standard gradient boosting handles high-cardinality categorical features poorly (one-hot explodes dimensionality; naive target encoding leaks target information from the same row).

*Core insight / definition:* Uses Ordered Target Statistics — target-encodes categories using only prior rows in a random permutation, preventing target leakage. Grows symmetric (oblivious) trees, where the same split condition is used across all nodes at a given depth, making prediction very fast and acting as regularization.

*Why it matters / pitfall / interview trap:* symmetric trees reduce overfitting risk and speed inference but are less flexible than leaf-wise growth for capturing highly irregular interactions.

---

**Chi-Squared Test (feature selection)**

*The problem:* Rank categorical (or non-negative) features by association with a categorical target.

*Core insight / definition:* A filter method computing the χ² statistic between each feature and the target's contingency table; larger statistic implies stronger dependence, i.e., the feature is less likely to be independent of the target by chance.

*Why it matters / pitfall / interview trap:* requires non-negative feature values (e.g., counts), so continuous features need appropriate binning or scaling first.

---

**Chronos / Moirai / TimesFM (Time Series Foundation Models)**

*The problem:* Forecast a series with little or no historical data ("cold start"), where classical models have nothing to fit on.

*Core insight / definition:* Large models pretrained on massive cross-domain time-series corpora and used zero-shot at inference. Chronos (Amazon) tokenizes values and uses a T5 architecture; Moirai (Salesforce) is frequency-agnostic and patch-based; TimesFM (Google, 200M params) is a pretrained decoder.

*Why it matters / pitfall / interview trap:* useful for cold-start forecasting with no domain history, but classical/tree methods (ARIMA, LightGBM with lags) still win when ample domain-specific history exists.

---

**Class Imbalance**

*The problem:* When one class vastly outnumbers another (e.g., 0.5% fraud), standard loss functions and accuracy are dominated by the majority class, and the model can achieve high accuracy by ignoring the minority class entirely.

*Core insight / definition:* Addressed via independent, composable levers: resampling (SMOTE, ADASYN, undersampling), cost-sensitive learning (class_weight, scale_pos_weight), and threshold moving — each changes a different part of the pipeline (data, loss, decision rule) and they can be combined.

*Why it matters / pitfall / interview trap:* reporting accuracy on imbalanced data is close to meaningless — predicting the majority class everywhere can yield 99%+ accuracy with zero recall on the class that matters.

---

**Class Weight / Cost-Sensitive Learning**

*The problem:* Force a model to pay more attention to the minority class without altering the data distribution via resampling.

*Core insight / definition:* Scale the loss contribution of each class inversely to its frequency (e.g., XGBoost's scale_pos_weight = n_negatives/n_positives), so misclassifying a minority example incurs a larger gradient/penalty.

*Why it matters / pitfall / interview trap:* composes cleanly with resampling and threshold moving; often safer than synthetic oversampling (SMOTE) when the minority class is very small or multimodal.

---

**Cosine Annealing with Warm Restarts (Snapshot Ensembles)**

*The problem:* Train an ensemble of diverse neural networks without the cost of training multiple models from scratch.

*Core insight / definition:* Cycle the learning rate down to near-zero and back up repeatedly during a single training run (cosine schedule with periodic restarts); save a model "snapshot" at each low-LR point. Each restart escapes the current local minimum and settles into a different one, giving diverse ensemble members cheaply.

*Why it matters / pitfall / interview trap:* the diversity comes from different local minima explored within one training run, not from independent training runs.

---

**Co-adaptation / Dropout**

*The problem:* Neurons in a network can develop brittle interdependencies where one neuron's usefulness depends on another's specific pattern, hurting generalization.

*Core insight / definition:* Dropout randomly deactivates a fraction of neurons (typically 20-50%) each training step, forcing each neuron to be useful with any random subset of co-neurons; at inference, dropout is disabled and weights are scaled to compensate — equivalent to approximating an ensemble of 2ⁿ thinned subnetworks.

*Why it matters / pitfall / interview trap:* forgetting to disable dropout at inference (`model.eval()`) causes stochastic, inconsistent predictions — a common production bug; applying dropout to an already underpowered (high-bias) network makes it worse.

---

**Conformal Prediction**

*The problem:* Produce prediction intervals/sets with a formal, distribution-free coverage guarantee, rather than relying on a model's (possibly miscalibrated) internal uncertainty estimates.

*Core insight / definition:* Uses a held-out calibration set to compute nonconformity scores and constructs prediction sets that are guaranteed to contain the true label with a specified probability (e.g., 90%), regardless of the underlying model or data distribution (under exchangeability).

*Why it matters / pitfall / interview trap:* provides finite-sample coverage guarantees that ad hoc confidence-interval heuristics (e.g., softmax probability) do not.

---

**Correlation Removal (feature selection)**

*The problem:* Highly correlated features add redundancy and can destabilize linear model coefficients (multicollinearity) without adding predictive signal.

*Core insight / definition:* A filter method that computes pairwise feature correlations and drops one feature from any pair exceeding a correlation threshold.

*Why it matters / pitfall / interview trap:* only detects linear pairwise redundancy; misses non-linear or multi-feature (higher-order) redundancy.

---

**Cross-Entropy and Softmax**

*The problem:* Convert unbounded classifier logits into probabilities and define a loss that penalizes confident wrong predictions with a clean, trainable gradient.

*Core insight / definition:* Softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ converts logits to a probability distribution; cross-entropy loss L = −log p(correct class) penalizes low probability on the true class. The combination's gradient simplifies to (p − y), the predicted probability vector minus the one-hot label — clean and stable for optimization.

*Why it matters / pitfall / interview trap:* using per-class sigmoid instead of softmax for multiclass problems doesn't constrain probabilities to sum to 1; naive softmax computation overflows on large logits — subtract the max logit first for numerical stability.

---

**Cross-Validation (K-Fold)**

*The problem:* A single train/validation split gives one noisy estimate of generalization performance — you might get a lucky or unlucky split.

*Core insight / definition:* Rotate through k different train/validation partitions of the data and average the scores, reducing the variance of the performance estimate. High variance across folds itself signals that the model's behavior is sensitive to which examples land in training vs. validation.

*Why it matters / pitfall / interview trap:* preprocessing/feature selection fit on the full dataset before CV leaks validation information into training; the CV score used for model selection is itself optimistically biased and cannot substitute for a final held-out test set.

---

**Crowding Problem (t-SNE)**

*The problem:* In high dimensions, a "shell" at distance r has volume proportional to r^(d−1); collapsing this into 2D (where volume scales just as r) leaves no room for all the medium-distance points that should be near each other, causing them to crowd onto near neighbors.

*Core insight / definition:* Fixed by using a heavy-tailed Student t-distribution (one degree of freedom, i.e., Cauchy) for pairwise similarities in the low-dimensional embedding instead of a Gaussian — its heavy tail lets moderately similar points spread out rather than collapse.

*Why it matters / pitfall / interview trap:* even with the fix, t-SNE in 2D still struggles when the true manifold has intrinsic dimension >2.

---

**Curse of Dimensionality**

*The problem:* Geometric and statistical intuitions from low dimensions break down as feature count grows, silently degrading distance-based algorithms.

*Core insight / definition:* Three consequences: (1) distance concentration — max/min pairwise distance ratio → 1, making nearest/farthest neighbor indistinguishable; (2) volume concentration — most of a high-dimensional hypercube's volume sits near its corners/faces, not the center; (3) statistical sparsity — maintaining the same sample density requires exponentially more data (n^d) as dimensions grow.

*Why it matters / pitfall / interview trap:* effects become noticeable around d≈10 and severe past d≈100; regularization alone does not fix this — dimensionality reduction or feature selection is often the correct remedy, not just a stronger penalty.

---

**Data Leakage**

*The problem:* Information from outside the legitimate training scope (often from the future or from the target itself) leaks into features, inflating offline metrics that then collapse in production.

*Core insight / definition:* Occurs when preprocessing statistics, target-encoded features, or temporal features are computed using data unavailable at prediction time — e.g., fitting a scaler on the full dataset before splitting, or using post-outcome information as a feature.

*Why it matters / pitfall / interview trap:* a correct temporal train/test split does not by itself prevent leakage — global statistics (encodings, imputations, normalizations) fit across the whole series still leak future information into past rows.

---

**Davies-Bouldin Index**

*The problem:* Evaluate clustering quality without ground truth, penalizing clusters that are both spread out and close to each other.

*Core insight / definition:* Average similarity ratio of each cluster to its most similar other cluster, where similarity combines within-cluster scatter and between-cluster separation; lower values indicate better clustering (unlike Silhouette/Calinski-Harabasz where higher is better).

*Why it matters / pitfall / interview trap:* easy to misremember the direction (lower-is-better) since most other clustering indices are higher-is-better.

---

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

*The problem:* K-Means assumes spherical, similarly-sized clusters and forces every point into a cluster; real data often has arbitrarily shaped clusters and genuine noise points.

*Core insight / definition:* Defines clusters as dense regions separated by sparse regions using two parameters: eps (neighborhood radius) and min_samples (minimum neighbors to be a core point). Points are classified as core (≥min_samples within eps), border (within eps of a core point but not core itself), or noise (neither). Clusters form by connecting core points and their reachable neighbors; no need to specify k in advance.

*Why it matters / pitfall / interview trap:* struggles with clusters of varying density (a single eps can't fit all); HDBSCAN extends DBSCAN to handle variable density by building a hierarchy over multiple eps values.

---

**Decision Tree**

*The problem:* Build an interpretable, non-linear model that can represent arbitrary axis-aligned decision boundaries without manual feature engineering.

*Core insight / definition:* Recursively partitions the feature space by choosing, at each node, the split (feature + threshold) that most reduces impurity — Gini impurity or Entropy/Information Gain for classification, variance reduction for regression. A leaf's prediction is the majority class or mean value of training points that reach it.

*Why it matters / pitfall / interview trap:* unconstrained trees overfit easily (memorize training noise down to single-point leaves); depth, min_samples_leaf, and pruning are essential regularizers.

---

**Deep Ensembles**

*The problem:* Quantify predictive uncertainty in neural networks without the computational overhead of full Bayesian inference.

*Core insight / definition:* Train several neural networks independently (different random initializations/data orders) and treat the spread of their predictions as an uncertainty estimate; disagreement indicates epistemic uncertainty.

*Why it matters / pitfall / interview trap:* often outperforms MC Dropout for uncertainty quality but costs N× the training and inference compute.

---

**Dendrogram**

*The problem:* Visualize and choose the number of clusters produced by hierarchical clustering without pre-specifying k.

*Core insight / definition:* A tree diagram showing the sequence of merges (agglomerative) or splits (divisive) and the distance/dissimilarity at which each occurred; cutting the tree horizontally at a chosen height yields a specific clustering.

*Why it matters / pitfall / interview trap:* the height of the "longest vertical line uncut" heuristic for choosing k is a rule of thumb, not a guarantee of the "true" number of clusters.

---

**Distance Metrics (Euclidean, Manhattan, Minkowski, Cosine, Hamming)**

*The problem:* Distance-based algorithms (KNN, K-Means, clustering) need a notion of similarity appropriate to the data type and scale.

*Core insight / definition:* Euclidean (L2, straight-line distance), Manhattan (L1, sum of absolute differences, robust to outliers along axes), Minkowski (generalizes both via a parameter p), Cosine (angle between vectors, scale-invariant, good for high-dimensional sparse/text data), Hamming (fraction of differing positions, for categorical/binary data).

*Why it matters / pitfall / interview trap:* choosing Euclidean distance on unscaled features lets high-magnitude features dominate distance calculations — features must be scaled first.

---

**Distance Concentration**

*The problem:* Explains formally why nearest-neighbor search degrades in high dimensions.

*Core insight / definition:* For points drawn from a distribution with bounded moments, (max distance − min distance)/min distance → 0 as dimensionality d → ∞, meaning all points become approximately equidistant from any query point.

*Why it matters / pitfall / interview trap:* this degrades KNN, kernel methods, and any distance-based algorithm silently — no error is thrown, results just become meaningless.

---

**Divisive Clustering (see Hierarchical Clustering)**

---

**Dropout (see Co-adaptation / Dropout)**

---

**Elastic Net**

*The problem:* L1 regularization arbitrarily selects one feature from a group of correlated features and zeroes out the rest, making feature selection unstable; L2 keeps all features but doesn't produce sparsity.

*Core insight / definition:* Combines L1 and L2 penalties (λ₁‖w‖₁ + λ₂‖w‖₂²), inheriting L1's sparsity while L2's smooth penalty stabilizes selection among correlated features (tends to select or zero out groups together rather than arbitrarily picking one).

*Why it matters / pitfall / interview trap:* requires tuning two regularization strengths (or one strength plus a mixing ratio), adding complexity relative to plain Ridge or Lasso.

---

**Elliptic Envelope**

*The problem:* Detect outliers in roughly Gaussian-distributed multivariate data using a robust estimate of the data's covariance structure.

*Core insight / definition:* Fits a robust estimate of the data's mean and covariance (Minimum Covariance Determinant) and flags points with low density under the fitted ellipse (high Mahalanobis distance) as anomalies.

*Why it matters / pitfall / interview trap:* excellent when data truly is elliptical/Gaussian-shaped, badly wrong when the true distribution is multimodal or non-convex.

---

**Embeddings**

*The problem:* Raw categorical features (user IDs, words) have no inherent numeric structure — one-hot encoding makes all categories equally distant, discarding semantic relationships.

*Core insight / definition:* A learned mapping from categories to dense vectors such that categories that behave similarly end up close in embedding space; the geometry emerges from training on the downstream task rather than being hand-designed.

*Why it matters / pitfall / interview trap:* using very high-dimensional embeddings for low-cardinality categories wastes parameters (rule of thumb: dim ≈ min(50, category_count/2)); freezing pretrained embeddings when fine-tuning data is available can leave task-relevant distinctions unlearned.

---

**Embedded Methods (feature selection)**

*The problem:* Filter methods ignore the model; wrapper methods are expensive (retrain repeatedly). You want feature selection integrated into model training itself.

*Core insight / definition:* Feature selection performed as a byproduct of model fitting — Lasso's zeroed coefficients, tree-based impurity/gain importance, SelectFromModel (threshold on any model's importance attribute), or SHAP-based selection (drop features with near-zero mean |SHAP value|).

*Why it matters / pitfall / interview trap:* importance from one model family (e.g., tree Gini importance) doesn't necessarily transfer to feature relevance for a different model family.

---

**Ensemble Methods (overview)**

*The problem:* A single model has an irreducible bias-variance tradeoff; combining multiple models can reduce error beyond what any single model achieves.

*Core insight / definition:* Techniques include Bagging (variance reduction via averaging independent learners), Boosting (bias reduction via sequential error correction), Voting (combine predictions of diverse, independently-trained models), Stacking/Blending (learn how to combine base model outputs via a meta-learner).

*Why it matters / pitfall / interview trap:* ensembling helps most when base learners are accurate AND diverse (make different errors) — averaging identical models gives no benefit.

---

**Entropy / Information Gain**

*The problem:* Choose the best split in a decision tree by measuring how much a split reduces uncertainty about the class label.

*Core insight / definition:* Entropy H = −Σ pᵢ log₂(pᵢ) measures the impurity/uncertainty of a node's class distribution. Information Gain is the reduction in entropy from parent to weighted average of child nodes after a split; trees choose the split maximizing information gain.

*Why it matters / pitfall / interview trap:* entropy is more computationally expensive (log) than Gini impurity but often gives very similar splits in practice — the choice rarely changes tree structure much.

---

**Epoch, Batch, Batch Size, Iteration**

*The problem:* Full-dataset gradient computation is infeasible for large datasets; data must be partitioned for gradient-based optimization, and the terminology for this is often confused.

*Core insight / definition:* Epoch = one full pass through the training set. Batch = the data subset used in one optimizer step. Batch size = number of examples per batch. Iteration = one optimizer step. n=10,000 examples with batch size 100 → 100 iterations per epoch.

*Why it matters / pitfall / interview trap:* larger batch sizes give more accurate but less noisy gradient estimates, which can converge to sharper (worse-generalizing) minima; larger batches typically need a proportionally larger learning rate (linear scaling rule) or they underfit relative to gradient quality.

---

**Exponential Smoothing (SES / Holt's / Holt-Winters)**

*The problem:* Forecast a series using a weighted average of past observations, with more emphasis on recent data, handling increasingly complex patterns (level → trend → seasonality).

*Core insight / definition:* Simple Exponential Smoothing (SES): ŷₜ₊₁ = α yₜ + (1−α) ŷₜ, for series with no trend/seasonality. Holt's Linear (double) adds a trend component (level + trend recursions). Holt-Winters (triple) adds a seasonal component (level + trend + seasonal recursions), with additive or multiplicative seasonality.

*Why it matters / pitfall / interview trap:* choosing additive vs multiplicative seasonality matters — multiplicative is appropriate when seasonal swings scale with the series level, additive when they're constant in absolute terms.

---

**F1 Score / F-beta**

*The problem:* Precision and recall trade off against each other; a single number is often needed to compare models, especially under imbalance.

*Core insight / definition:* F1 is the harmonic mean of precision and recall, punishing extreme imbalance between the two (e.g., 100% precision/0% recall gives F1=0, not the arithmetic-mean-implied 50%). F-beta generalizes this with a weight β controlling the relative importance of recall vs precision (β>1 weights recall more).

*Why it matters / pitfall / interview trap:* choosing F1 by default without considering the actual cost asymmetry of false positives vs false negatives ignores the real decision context (e.g., cancer screening should weight recall much more via F2).

---

**FastICA (see ICA)**

---

**Feature Engineering**

*The problem:* Neural networks can learn representations from raw inputs but need vast data to discover interactions a domain expert already knows; classical models benefit directly from hand-derived features.

*Core insight / definition:* Deriving new input features (ratios, rolling statistics, lag features, domain-specific indicators) that encode known relationships directly, so the model doesn't need to rediscover them from raw data — a ratio feature captures a multiplicative interaction in one split/coefficient rather than requiring many.

*Why it matters / pitfall / interview trap:* requires domain expertise that isn't always available, and encodes assumptions that can become wrong (and silently degrade the model) if the underlying domain relationship shifts.

---

**Filter Methods (feature selection)**

*The problem:* Cheaply reduce feature count before any model training, based only on the features' statistical properties.

*Core insight / definition:* Score features independent of any specific model, using criteria like variance threshold (drop near-constant features), correlation removal, mutual information, chi-squared test, or ANOVA F-test; select the top-scoring subset.

*Why it matters / pitfall / interview trap:* fast and model-agnostic but ignores feature interactions — a feature useless alone but powerful in combination with another gets dropped.

---

**Focal Loss**

*The problem:* Under severe class imbalance, standard cross-entropy is dominated by the easy, plentiful majority-class examples, drowning the gradient signal from hard minority examples.

*Core insight / definition:* Modifies cross-entropy by down-weighting well-classified (easy) examples via a modulating factor (1−p)^γ, focusing training on hard, misclassified examples regardless of class.

*Why it matters / pitfall / interview trap:* is essentially neural networks' single primary lever for class imbalance, versus classical ML's several composable levers (resampling + cost-sensitive weighting + threshold moving); requires careful tuning of γ and interacts with learning rate.

---

**GMM (Gaussian Mixture Model)**

*The problem:* K-Means assumes hard, spherical cluster assignments; real clusters can overlap, have different shapes, and points can plausibly belong to multiple clusters with different probabilities.

*Core insight / definition:* Models data as a weighted mixture of K Gaussian distributions, each with its own mean and covariance. Fit via Expectation-Maximization: E-step computes each point's responsibility (soft probability of belonging to each component) given current parameters; M-step updates each component's mean/covariance/weight using those responsibilities, weighted by them. covariance_type (spherical/diagonal/tied/full) controls cluster shape flexibility.

*Why it matters / pitfall / interview trap:* EM only guarantees convergence to a local optimum — sensitive to initialization; unlike K-Means's hard assignment, GMM gives soft cluster membership probabilities.

---

**Generalization**

*The problem:* Distinguish a model that has genuinely learned an underlying pattern from one that has merely memorized its training examples.

*Core insight / definition:* The capacity to perform well on examples never seen during training. Any model can achieve low training loss by memorization; generalization is measured only via held-out data.

*Why it matters / pitfall / interview trap:* defining ML merely as "learning from data" without addressing generalization misses the entire central technical challenge of the field.

---

**Gini Impurity**

*The problem:* Choose the best split in a decision tree using a computationally cheap measure of node purity.

*Core insight / definition:* Gini = 1 − Σ pᵢ², the probability that two randomly drawn samples from a node have different labels; splits are chosen to minimize the weighted Gini of child nodes. Cheaper to compute than entropy (no logarithms).

*Why it matters / pitfall / interview trap:* Gini-based feature importance (mean decrease in impurity) is biased toward high-cardinality features — permutation importance or SHAP is preferred for unbiased ranking.

---

**Gradient Boosting**

*The problem:* Build a strong predictor by sequentially fitting new models to the errors the current ensemble is still making, framed as gradient descent in function space.

*Core insight / definition:* At each round, fit a new weak learner (typically a shallow tree) to the negative gradient (pseudo-residuals) of the loss function with respect to the current ensemble's predictions; add the new learner's output scaled by a learning rate (shrinkage) to the ensemble. This is literally gradient descent, where the "parameter" being updated is the function itself.

*Why it matters / pitfall / interview trap:* a small learning rate paired with more trees generally improves generalization at the cost of training time — treating learning rate and n_estimators as independent tuning knobs (rather than a coupled tradeoff) is a common mistake.

---

**Grid Search**

*The problem:* Exhaustively evaluate a small, well-defined set of hyperparameter combinations to find the best.

*Core insight / definition:* Define a discrete grid of values per hyperparameter, evaluate the cross-validated score at every combination, return the best. Cost is the product of grid sizes across hyperparameters — scales exponentially with the number of hyperparameters.

*Why it matters / pitfall / interview trap:* wastes evaluations on hyperparameters that don't actually matter (spends equal budget on irrelevant dimensions) and only samples the exact grid points specified, potentially missing the true optimum between them.

---

**Hessian / Newton Boosting (XGBoost)**

*The problem:* Standard gradient boosting uses only first-order gradient information, which can be a slow, imprecise guide to the optimal step at each round.

*Core insight / definition:* XGBoost uses a second-order Taylor expansion of the loss around the current prediction, incorporating both the gradient and the Hessian (second derivative) to compute an optimal leaf score in closed form, akin to a Newton step rather than plain gradient descent.

*Why it matters / pitfall / interview trap:* this is a core reason XGBoost often converges faster/more accurately than plain gradient boosting with the same number of trees.

---

**Hierarchical / Agglomerative Clustering**

*The problem:* K-Means requires specifying k upfront and assumes flat, non-nested cluster structure; some data has a natural multi-level (nested) grouping.

*Core insight / definition:* Agglomerative (bottom-up): start with each point as its own cluster, iteratively merge the two closest clusters according to a linkage criterion, until one cluster remains; visualized as a dendrogram. Linkage criteria: Single (min pairwise distance, can chain), Complete (max pairwise distance, compact), Average/UPGMA (mean pairwise distance), Ward's (minimizes increase in within-cluster variance).

*Why it matters / pitfall / interview trap:* naive implementations are O(N³) time / O(N²) memory, impractical for large N; Ward's linkage tends to produce the most balanced, compact clusters and is the most commonly recommended default.

---

**Holdout Set (see Train/Validation/Test Split)**

---

**Holt-Winters / Holt's Linear Smoothing (see Exponential Smoothing)**

---

**Hyperband / ASHA**

*The problem:* Fully training every hyperparameter configuration to convergence to evaluate it wastes enormous compute, since most bad configurations are already identifiably bad after a few epochs.

*Core insight / definition:* Successive Halving: start with n configs at a small budget, keep the top 1/η fraction, give survivors η× more budget, repeat until one config reaches the max budget. Hyperband runs multiple brackets of Successive Halving with different starting budgets to hedge the aggressiveness tradeoff. ASHA removes synchronization barriers, promoting configurations asynchronously as soon as ready — better for distributed settings.

*Why it matters / pitfall / interview trap:* only works when early performance correlates with final performance; configurations affected by learning-rate schedules or late-training dynamics can look bad early but be best at convergence — the grace_period parameter guards against premature elimination.

---

**Hyperparameter Optimization (HPO)**

*The problem:* Hyperparameters (learning rate, depth, regularization strength) cannot be learned from training data directly, and manually searching all combinations is intractable.

*Core insight / definition:* Framed as black-box optimization of "validation score given a hyperparameter configuration" — expensive to evaluate, noisy, and gradient-free. Strategy choice (grid, random, Bayesian, Hyperband/ASHA, BOHB, PBT) depends on evaluation cost and number of hyperparameters.

*Why it matters / pitfall / interview trap:* the right method is workload-dependent — cheap models favor Bayesian optimization alone; expensive deep models favor multi-fidelity methods (Hyperband/BOHB).

---

**ICA (Independent Component Analysis) / FastICA**

*The problem:* PCA finds uncorrelated components, which is a weaker condition than statistical independence — insufficient for tasks like separating truly independent signal sources (e.g., separating mixed audio tracks).

*Core insight / definition:* By the Central Limit Theorem, a mixture of independent non-Gaussian sources looks more Gaussian than any individual source; ICA finds the unmixing matrix that makes the recovered components least Gaussian (via kurtosis or negentropy). FastICA is the standard algorithm, using fixed-point iteration for cubic convergence. Components are unique up to sign and permutation (not rotation, unlike PCA), and Gaussian sources have no unique ICA solution.

*Why it matters / pitfall / interview trap:* the "cocktail party problem" (blind source separation) is the classic application; PCA whitening is typically used as preprocessing before FastICA.

---

**Imbalanced Data (see Class Imbalance)**

---

**Information Criteria (see AIC / BIC)**

---

**Isolation Forest**

*The problem:* Detect anomalies without computing distances or densities, which degrade in high dimensions and are expensive at scale.

*Core insight / definition:* Builds trees on random subsamples by repeatedly picking a random feature and random split value; anomalies, being "few and different," get isolated in far fewer splits (shorter path length) than normal points in dense regions. Score s(x,n) = 2^(−E[h(x)]/c(n)), normalized by the average unsuccessful BST search depth c(n); score near 1 = anomaly, near 0.5 = normal. Never computes a pairwise distance, so it stays fast and effective in high dimensions.

*Why it matters / pitfall / interview trap:* `contamination` only sets the score threshold/cutoff, not a learned model parameter — misreading it as tunable "sensitivity" and guessing a value causes wildly wrong alert volumes; it also misses local anomalies (normal globally but wrong for their neighborhood) since it uses a single global notion of isolation.

---

**Isotonic Regression**

*The problem:* Calibrate a model's probability outputs when the miscalibration is non-linear/non-monotonic in a simple parametric sense, or when Platt scaling's logistic assumption is too rigid.

*Core insight / definition:* A non-parametric calibration method that fits an arbitrary monotonically increasing step function mapping raw scores to calibrated probabilities, minimizing squared error subject to the monotonicity constraint.

*Why it matters / pitfall / interview trap:* more flexible than Platt scaling but overfits with small calibration sets (rule of thumb: use Platt scaling under ~500 calibration examples, isotonic above that).

---

**Iteration (see Epoch, Batch, Batch Size, Iteration)**

---

**Johnson-Lindenstrauss Lemma**

*The problem:* Justify why low-dimensional random projections can be trusted to approximately preserve the geometry of high-dimensional data.

*Core insight / definition:* Random projections onto a sufficiently large but still low-dimensional subspace approximately preserve pairwise distances between points, with high probability.

*Why it matters / pitfall / interview trap:* offered as evidence supporting the manifold hypothesis — that true data dimensionality is often far below the ambient feature count.

---

**Kernel PCA**

*The problem:* PCA is linear and cannot uncover structure that lies along a non-linear manifold (e.g., concentric circles, Swiss roll).

*Core insight / definition:* Implicitly maps data into a high-dimensional feature space via a kernel function (RBF, polynomial, sigmoid) — as in SVMs — so the data becomes linearly separable there, then performs PCA in that space without ever computing the mapping explicitly (the kernel trick).

*Why it matters / pitfall / interview trap:* O(n²) memory / O(n³) computation makes it impractical past n≈10⁴; has no direct out-of-sample projection without a Nyström approximation.

---

**Kernel Trick / SVM Kernels**

*The problem:* Many real datasets are not linearly separable in their original feature space, but computing an explicit high-dimensional mapping to make them separable is often computationally infeasible.

*Core insight / definition:* Replace the dot product in the SVM optimization with a kernel function K(xᵢ,xⱼ) that computes the equivalent of an inner product in a high (possibly infinite) dimensional feature space, without ever materializing that mapping. Common kernels: linear (no transform), polynomial (interactions up to a degree), RBF/Gaussian (infinite-dimensional, local similarity via distance).

*Why it matters / pitfall / interview trap:* RBF kernel's gamma parameter controls the influence radius of each point — too high overfits (each point its own region), too low underfits (overly smooth boundary).

---

**KNN (K-Nearest Neighbors)**

*The problem:* Classify or predict a value for a new point using the labels/values of its most similar training examples, without fitting an explicit parametric model.

*Core insight / definition:* A non-parametric, instance-based (lazy) learner: at prediction time, find the k closest training points (by a chosen distance metric) and return the majority class (classification) or average value (regression), optionally weighted by inverse distance.

*Why it matters / pitfall / interview trap:* degrades badly in high dimensions due to distance concentration (curse of dimensionality); requires feature scaling since distance metrics are scale-sensitive; k too small overfits to noise, k too large oversmooths toward the global majority.

---

**K-Means**

*The problem:* Partition unlabeled data into k groups such that points within a group are similar and points across groups are dissimilar, without any labels to guide the process.

*Core insight / definition:* Iteratively (coordinate-descent style) alternates assigning each point to its nearest centroid and recomputing each centroid as the mean of its assigned points, minimizing within-cluster sum of squares (WCSS/inertia). Guaranteed to converge (monotonically decreasing objective) but only to a local optimum — sensitive to initialization.

*Why it matters / pitfall / interview trap:* assumes roughly spherical, similarly-sized clusters and requires specifying k upfront; the Elbow Method (plotting WCSS vs. k, looking for the bend) and Silhouette Score help choose k, but neither is definitive.

---

**K-Means++**

*The problem:* Random centroid initialization in K-Means can lead to poor local optima or slow convergence.

*Core insight / definition:* An initialization scheme that spreads initial centroids apart by choosing each new centroid with probability proportional to its squared distance from the nearest already-chosen centroid, rather than uniformly at random.

*Why it matters / pitfall / interview trap:* is the scikit-learn default for K-Means and substantially improves both convergence speed and final cluster quality versus naive random init.

---

**Label Smoothing**

*The problem:* A classifier trained with hard one-hot targets can become overconfident, especially problematic under label noise or class imbalance.

*Core insight / definition:* Replace hard 0/1 targets with softened targets (e.g., 0.9/0.1 split across classes instead of 1/0), preventing the model from driving logits to extreme values and improving calibration/generalization.

*Why it matters / pitfall / interview trap:* acts as a regularizer that also tends to improve calibration, distinct from focal loss's re-weighting of hard examples.

---

**Latency-Critical Inference (Classical ML advantage)**

*The problem:* Some production systems (fraud scoring, ad auctions) require sub-millisecond predictions that GPU-bound deep networks cannot deliver without batching.

*Core insight / definition:* Logistic regression (dot product + sigmoid) and compiled decision trees (branch sequences) run in microseconds on CPU, cache-resident and branch-predicted, versus tens to hundreds of milliseconds for neural networks like BERT on CPU. Trees can be compiled to native code (e.g., via treelite) for further speedup.

*Why it matters / pitfall / interview trap:* the gap narrows when predictions can be batched (e.g., recommendation scoring many users at once); for true single-request low-latency needs (HFT, embedded systems), classical ML remains the only viable option.

---

**LDA (Linear Discriminant Analysis)**

*The problem:* PCA's variance-maximizing directions may be irrelevant or even orthogonal to the directions that best separate known classes — you need supervised dimensionality reduction.

*Core insight / definition:* Finds a projection maximizing Fisher's criterion J(w) = wᵀS_Bw / wᵀS_Ww — the ratio of between-class scatter to within-class scatter — via a generalized eigenproblem. With C classes, produces at most C−1 discriminants. Under Gaussian class-conditionals with equal covariance, LDA's boundary equals the Bayes-optimal linear classifier, so it simultaneously reduces dimensionality and classifies.

*Why it matters / pitfall / interview trap:* when d > n, the within-class scatter matrix S_W is rank-deficient/singular and cannot be inverted directly — requires regularized LDA or a PCA pre-step to reduce d below n first.

---

**LightGBM**

*The problem:* XGBoost's level-wise tree growth can be slow and wasteful — it grows all nodes at a depth even when some splits offer little gain.

*Core insight / definition:* Grows trees leaf-wise (best-first) — always splitting the leaf with the highest loss reduction regardless of depth — producing deeper, more accurate trees per iteration but at higher overfitting risk on small data; uses histogram-based binning of continuous features for speed and Gradient-based One-Side Sampling / Exclusive Feature Bundling for efficiency.

*Why it matters / pitfall / interview trap:* leaf-wise growth without a max_depth cap can overfit noisy or small datasets more readily than XGBoost's level-wise default.

---

**LIME (Local Interpretable Model-agnostic Explanations)**

*The problem:* Explain an individual black-box prediction without needing access to the model's internals or exact gradients.

*Core insight / definition:* Perturbs the input around the instance being explained, gets the black-box model's predictions on those perturbations, and fits a simple interpretable model (e.g., sparse linear regression) locally weighted toward the instance, using that local surrogate's coefficients as the explanation.

*Why it matters / pitfall / interview trap:* explanations are local and approximate — they can be unstable across repeated runs (different random perturbations) and are not guaranteed to reflect the true model behavior outside the immediate local neighborhood.

---

**Linear Regression (OLS)**

*The problem:* Model a continuous target as a linear combination of features, minimizing prediction error, with a closed-form solution when feasible.

*Core insight / definition:* Ordinary Least Squares finds coefficients β minimizing Σ(yᵢ − Xᵢβ)²; solvable in closed form via the normal equation β = (XᵀX)⁻¹Xᵀy, or iteratively via gradient descent for large datasets.

*Why it matters / pitfall / interview trap:* the normal equation requires XᵀX to be invertible — multicollinearity or more features than samples breaks this, requiring regularization (Ridge) or gradient-based methods.

---

**Logistic Regression**

*The problem:* Predict a binary (or multiclass) outcome's probability as a function of features, with an output constrained to [0,1] and interpretable coefficients.

*Core insight / definition:* Applies a sigmoid function to a linear combination of features: p = 1/(1+e^(−wᵀx+b)); trained by minimizing binary cross-entropy (log-loss) via gradient descent (no closed-form solution). Coefficients are interpretable as changes in log-odds (logit) per unit feature change.

*Why it matters / pitfall / interview trap:* natively well-calibrated (optimizes a proper scoring rule) when the log-odds are approximately linear in the features — this breaks down under strong non-linear interactions, degrading calibration.

---

**Loss Function vs. Metric**

*The problem:* The function optimized during training and the metric used to judge success can diverge, and a model can look good on one while failing the other.

*Core insight / definition:* A loss function must be differentiable to provide gradient signal (e.g., cross-entropy); a metric must reflect what the business actually cares about (e.g., recall, precision). Accuracy is discontinuous and unusable as a loss; cross-entropy is its smooth differentiable proxy, but optimizing it does not guarantee optimizing accuracy or recall, especially under imbalance.

*Why it matters / pitfall / interview trap:* discovering a loss/metric mismatch after training (e.g., low cross-entropy but poor recall because the model hedges near 0.5) is expensive — the alignment should be checked before training starts.

---

**MAE (Mean Absolute Error)**

*The problem:* Measure average prediction error in a way that is robust to a few large outlier errors.

*Core insight / definition:* MAE = mean(|y − ŷ|); penalizes all errors linearly regardless of magnitude, making it less sensitive to outliers than RMSE.

*Why it matters / pitfall / interview trap:* not differentiable at zero error, which can complicate direct gradient-based optimization (though this is rarely a practical blocker).

---

**Manifold Hypothesis**

*The problem:* Explain why dimensionality reduction is even possible — why don't real high-dimensional datasets truly need all their dimensions?

*Core insight / definition:* High-dimensional data typically lies near a much lower-dimensional manifold embedded in the ambient space (e.g., face images spanning ~50 true degrees of freedom within 10⁶-pixel space). A k-dimensional manifold looks locally like ℝᵏ even when embedded in higher-dimensional space (e.g., the Swiss roll, a 2D manifold in 3D).

*Why it matters / pitfall / interview trap:* algorithms that ignore this structure waste capacity on the d−k irrelevant directions; dimensionality reduction aims to recover the intrinsic k coordinates.

---

**MAPE (Mean Absolute Percentage Error)**

*The problem:* Communicate forecast error in an easily interpretable, scale-independent percentage.

*Core insight / definition:* MAPE = mean(|y−ŷ|/|y|) × 100.

*Why it matters / pitfall / interview trap:* undefined/explodes when actual values y are near zero, and asymmetric — penalizes over-forecasts and under-forecasts differently — motivating sMAPE or MASE as alternatives.

---

**MASE (Mean Absolute Scaled Error)**

*The problem:* Compare forecast accuracy across series of different scales and against a naive baseline, avoiding MAPE's division-by-near-zero issue.

*Core insight / definition:* MASE = MAE(model) / MAE(naive one-step forecast); MASE < 1 means the model beats a naive persistence forecast, MASE > 1 means it's worse.

*Why it matters / pitfall / interview trap:* scale-free and robust to near-zero actuals, making it generally preferable to MAPE for cross-series comparison.

---

**MCC (Matthews Correlation Coefficient)**

*The problem:* Summarize binary classification quality in a single number that remains meaningful under severe class imbalance, unlike accuracy or even F1 in some cases.

*Core insight / definition:* A correlation coefficient between −1 and +1 computed from all four confusion matrix cells (TP, TN, FP, FN); +1 is perfect prediction, 0 is random, −1 is total disagreement.

*Why it matters / pitfall / interview trap:* considered by many practitioners more robust to imbalance than F1 because it accounts for all four confusion matrix quadrants, including true negatives, which F1 ignores entirely.

---

**MC Dropout (Monte Carlo Dropout)**

*The problem:* Estimate a neural network's predictive uncertainty without training a separate ensemble of models.

*Core insight / definition:* Keep dropout active at inference time and run multiple stochastic forward passes; the variance across these passes approximates epistemic (model) uncertainty.

*Why it matters / pitfall / interview trap:* cheaper than Deep Ensembles (one trained model, multiple forward passes) but generally gives lower-quality uncertainty estimates.

---

**Model Selection Bias / Optimistic CV Estimate**

*The problem:* Repeatedly using the validation set (or CV score) to choose among models or hyperparameters causes that estimate to become biased upward, since the selection process has effectively "trained on" the validation data.

*Core insight / definition:* After many rounds of model selection guided by the same validation/CV score, that score no longer reflects unbiased generalization performance; a final untouched held-out test set is required for an honest final estimate.

*Why it matters / pitfall / interview trap:* nested cross-validation is the principled fix when hyperparameter tuning and performance estimation both need to happen without leakage between them.

---

**Multicollinearity**

*The problem:* Highly correlated input features make linear model coefficients unstable and hard to interpret, even though predictions may still be accurate.

*Core insight / definition:* Occurs when two or more features are linearly correlated, causing the XᵀX matrix in OLS to be near-singular; small data changes cause large swings in estimated coefficients, and individual coefficient significance becomes unreliable even though the model's overall fit is unaffected.

*Why it matters / pitfall / interview trap:* a common failure mode listed alongside overfitting, underfitting, and data leakage; addressed by removing correlated features, PCA, or Ridge regularization.

---

**Multinomial Naive Bayes**

*The problem:* Classify data described by count features (e.g., word frequencies in documents).

*Core insight / definition:* Assumes each feature follows a multinomial distribution conditioned on class (models feature counts, not just presence); commonly used for text classification with bag-of-words counts.

*Why it matters / pitfall / interview trap:* unlike BernoulliNB, ignores the absence of a feature — only accounts for the counts of features that occur.

---

**Multi-Seed Ensemble**

*The problem:* A single model's training is sensitive to random initialization and data shuffling, adding variance to its predictions.

*Core insight / definition:* Train the same model architecture multiple times with different random seeds and average/vote the predictions, reducing variance from stochastic training without changing the model family.

*Why it matters / pitfall / interview trap:* the simplest possible ensembling strategy, and a useful low-effort baseline before trying more complex stacking/blending.

---

**Mutual Information (feature selection)**

*The problem:* Quantify a feature's relevance to the target when the relationship may be non-linear (correlation alone would miss it).

*Core insight / definition:* A filter method measuring how much knowing a feature's value reduces uncertainty about the target, based on information theory; captures non-linear dependencies that correlation coefficients cannot.

*Why it matters / pitfall / interview trap:* more computationally expensive and estimator-sensitive than correlation, and doesn't reflect the strength of relevance in a specific model's decision function.

---

**Naive Bayes**

*The problem:* Build a fast, simple probabilistic classifier that scales well and works reasonably even with limited data, at the cost of a strong independence assumption.

*Core insight / definition:* Applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class: P(y|x) ∝ P(y) Π P(xᵢ|y). Variants differ in the assumed feature distribution: GaussianNB (continuous features), MultinomialNB (counts), BernoulliNB (binary presence/absence).

*Why it matters / pitfall / interview trap:* the independence assumption is almost always violated in practice, yet the classifier often performs surprisingly well because classification only needs the correct ranking of posterior probabilities, not their exact calibrated values.

---

**Nested Cross-Validation**

*The problem:* Using the same CV loop both to tune hyperparameters and to estimate final generalization performance produces an optimistically biased performance estimate.

*Core insight / definition:* An outer CV loop estimates generalization performance; within each outer fold, an inner CV loop performs hyperparameter tuning on only that fold's training data — separating the tuning process from the performance estimate entirely.

*Why it matters / pitfall / interview trap:* computationally expensive (k_outer × k_inner model fits) but necessary whenever you need an honest performance number after any form of model/hyperparameter selection.

---

**N-BEATS / N-HiTS**

*The problem:* Achieve competitive time-series forecasting accuracy with an architecture simpler than Transformers, while retaining interpretability.

*Core insight / definition:* A pure MLP architecture using basis function expansion (ŷ = Σ θ_k φ_k(t)), with separate stacks that can be interpreted as trend and seasonality components; competitive with Transformer variants on standard forecasting benchmarks (M4).

*Why it matters / pitfall / interview trap:* demonstrates that architectural complexity (attention) isn't necessary for state-of-the-art forecasting — a well-designed MLP with the right inductive bias (basis expansion) can match it.

---

**NMF (Non-negative Matrix Factorization)**

*The problem:* PCA components can have negative values, making them hard to interpret as "parts" for inherently additive/non-negative data (images, documents, spectra).

*Core insight / definition:* Factorizes X ≈ WH with both W and H constrained non-negative, yielding sparse, localized, parts-based representations (e.g., face images decompose into eye/nose/mouth-like parts rather than holistic eigenfaces). Solved via multiplicative update rules (Lee & Seung) guaranteeing non-negativity and monotonic convergence to a local minimum — the joint problem is non-convex.

*Why it matters / pitfall / interview trap:* no equivalent to PCA's explained variance ratio for choosing the number of components k; results depend heavily on initialization (nndsvda initialization improves over random).

---

**NMI (Normalized Mutual Information)**

*The problem:* Evaluate clustering quality against ground-truth labels using an information-theoretic measure rather than a pairwise-agreement measure like ARI.

*Core insight / definition:* Measures the mutual information between predicted cluster assignments and true labels, normalized to [0,1] so it's comparable across different numbers of clusters.

*Why it matters / pitfall / interview trap:* like ARI, requires ground-truth labels — not usable for purely unsupervised evaluation, unlike Silhouette or Calinski-Harabasz.

---

**Ordinal Regression**

*The problem:* Predict a discrete but ordered target (e.g., 1-5 star ratings), where plain classification ignores the ordering and plain regression assumes equal spacing between values.

*Core insight / definition:* A specialized approach that respects both the discreteness and the ordering of the target, unlike naively binning a continuous value (losing ordinal information, adding sensitivity to arbitrary bin boundaries) or treating it as unordered classes.

*Why it matters / pitfall / interview trap:* a frequently overlooked "non-obvious case" between classification and regression that interviewers use to test whether output-type reasoning is superficial.

---

**One-Class SVM**

*The problem:* Detect anomalies by learning a boundary that encloses "normal" data, for cases where a clean boundary-based notion of "outside normal" fits the data well.

*Core insight / definition:* Learns a boundary (using, e.g., an RBF kernel for non-linear shapes) that encloses the region of normal data; the ν parameter upper-bounds the expected fraction of outliers/support vectors.

*Why it matters / pitfall / interview trap:* scales poorly (O(n²)–O(n³)), impractical past ~10⁴ samples, and requires feature scaling.

---

**One-Hot Encoding (categorical, high-cardinality)**

*The problem:* Represent categorical features numerically for models that require numeric input, without imposing a false ordinal structure.

*Core insight / definition:* Represents each category as a binary indicator vector; treats all categories as equally distant from each other, unlike embeddings which learn semantic proximity.

*Why it matters / pitfall / interview trap:* for high-cardinality categoricals (e.g., 5,000 zip codes), one-hot explodes dimensionality and wastes model capacity — trees handle this more gracefully via direct threshold splits, and NNs typically need learned embeddings instead.

---

**Overfitting / Underfitting**

*The problem:* Diagnose whether a model's poor generalization stems from too little or too much capacity relative to the data and true pattern.

*Core insight / definition:* Underfitting: the model lacks capacity/inductive bias to capture the true pattern — wrong even on training data (both train and validation error high). Overfitting: the model has enough capacity to memorize training noise and did — training error is low but validation error is high (large train/val gap).

*Why it matters / pitfall / interview trap:* diagnosing overfitting requires a genuine held-out validation set — training loss alone reveals nothing about generalization; repeatedly re-selecting models on the same validation set eventually "trains on" it too, requiring a separate final test set.

---

**PatchTST**

*The problem:* Apply Transformers to long time series efficiently, without quadratic attention cost over every individual timestep.

*Core insight / definition:* Splits the series into patches (subseries) treated as tokens, using channel independence (each variate processed separately with shared weights), achieving linear complexity in sequence length.

*Why it matters / pitfall / interview trap:* one of several long-horizon transformer variants (alongside Informer/Autoformer, which use sparse attention or auto-correlation instead of standard attention).

---

**PBT (Population-Based Training)**

*The problem:* Fixed hyperparameters set before training may be optimal early but suboptimal later — HPO methods that only search fixed configurations miss beneficial hyperparameter schedules.

*Core insight / definition:* Evolve a population of models in parallel; periodically, poorly-performing models copy the weights and hyperparameters of top performers and then randomly perturb those hyperparameters (multiply by 0.8/1.2 or resample) — an evolutionary search over hyperparameter schedules, not fixed configurations.

*Why it matters / pitfall / interview trap:* requires K× the compute of a single run (K = population size); copied weights may be incompatible with the newly perturbed hyperparameters, often requiring a warmup period.

---

**PCA (Principal Component Analysis)**

*The problem:* Project high-dimensional data onto a lower-dimensional subspace while retaining as much information (operationalized as variance) as possible.

*Core insight / definition:* Finds orthogonal directions (eigenvectors of the covariance matrix Σ, ranked by eigenvalue) that successively capture the maximum remaining variance; in practice computed via SVD of the centered data matrix for numerical stability (X̃=USVᵀ), which scales as O(ndk) via truncated SVD rather than O(d³) via full eigendecomposition. Explained Variance Ratio guides how many components k to keep (e.g., 90-95%), read off a scree plot's "elbow."

*Why it matters / pitfall / interview trap:* PCA is a linear method — it "cuts through" curved manifolds rather than unrolling them; not appropriate when features are categorical, when non-linear structure dominates, or when components need to remain interpretable as original features.

---

**PCA Whitening**

*The problem:* After PCA projection, components can still have very different variances, which hurts algorithms sensitive to feature scale (e.g., k-means, neural nets).

*Core insight / definition:* Rescale each principal component by dividing by the square root of its eigenvalue (z_i = w_iᵀx/√λ_i), giving every component unit variance and zero correlation.

*Why it matters / pitfall / interview trap:* amplifies noise in low-variance directions, since those components get divided by a small √λ.

---

**PDP (Partial Dependence Plot)**

*The problem:* Visualize how a model's prediction changes on average as one (or two) features vary, holding others "fixed" in aggregate.

*Core insight / definition:* Computed by averaging model predictions over the training data while sweeping the feature(s) of interest across their range, marginalizing out all other features.

*Why it matters / pitfall / interview trap:* assumes feature independence — with correlated features it extrapolates into unrealistic/unobserved feature combinations, producing misleading curves; ALE plots address this.

---

**Permutation Importance**

*The problem:* Measure a feature's true contribution to a fitted model's performance in a model-agnostic way, without the bias of tree impurity-based importance toward high-cardinality features.

*Core insight / definition:* Randomly shuffle a single feature's values (breaking its relationship with the target) and measure the resulting drop in model performance on held-out data; a larger performance drop implies greater importance.

*Why it matters / pitfall / interview trap:* can be misleading with strongly correlated features — permuting one feature has little effect if a correlated feature can compensate, understating its true importance.

---

**Perplexity (t-SNE)**

*The problem:* t-SNE needs a way to control the effective neighborhood size used when computing pairwise similarities in high-dimensional space, per point.

*Core insight / definition:* Sets the per-point Gaussian bandwidth σᵢ so that the resulting neighborhood distribution has entropy corresponding to a target perplexity (2^H(Pᵢ)); low perplexity (5-15) emphasizes very local structure and can fragment clusters, high perplexity (30-50) captures broader, more global-like structure.

*Why it matters / pitfall / interview trap:* results are quite sensitive to this parameter — always run multiple values; rule of thumb is perplexity should be less than n/3.

---

**Platt Scaling**

*The problem:* Calibrate a model's raw decision scores (e.g., SVM margins) into meaningful probabilities using a simple parametric method.

*Core insight / definition:* Fits a logistic regression on the model's raw output scores against the true labels, producing calibrated probabilities via a sigmoid transform.

*Why it matters / pitfall / interview trap:* preferred over isotonic regression when the calibration set is small (<500 examples), since isotonic's non-parametric flexibility overfits with too little data.

---

**Policy-Based vs. Value-Based RL**

*The problem:* Reinforcement learning agents need a way to choose actions from delayed, scalar reward signals; different action-space structures favor different approaches.

*Core insight / definition:* Value-based (Q-learning, DQN): learn Q(s,a), expected cumulative reward of an action in a state, then act greedily; suited to discrete action spaces. Policy-based (REINFORCE, PPO): directly parameterize and optimize the policy π_θ(a|s); handles continuous action spaces naturally but suffers high-variance gradient estimates requiring baselines/variance reduction.

*Why it matters / pitfall / interview trap:* treating exploration as a minor implementation detail is a trap — insufficient exploration means the agent never discovers better strategies exist.

---

**PR-AUC (Precision-Recall AUC) / Average Precision**

*The problem:* ROC-AUC can look deceptively good under severe class imbalance because the false positive rate denominator (all negatives) is huge, diluting the impact of false positives.

*Core insight / definition:* The area under the precision-recall curve, which focuses entirely on the positive class's precision and recall trade-off across thresholds — much more sensitive to performance on the minority class than ROC-AUC under imbalance.

*Why it matters / pitfall / interview trap:* the recommended default over ROC-AUC whenever the positive class is rare (e.g., fraud, anomaly detection).

---

**Precision, Recall**

*The problem:* Every classification error is either a false positive or false negative, and these typically carry very different real-world costs.

*Core insight / definition:* Precision = TP/(TP+FP) — of everything predicted positive, how much was correct (use when false positives are costly). Recall = TP/(TP+FN) — of everything actually positive, how much was caught (use when false negatives are costly).

*Why it matters / pitfall / interview trap:* fixing the decision threshold at 0.5 by default implicitly assumes false positives and false negatives have equal cost, which is rarely true in practice — the threshold should be chosen from the cost structure or the precision-recall curve.

---

**Prophet**

*The problem:* Forecast business time series with strong seasonality, holiday effects, and trend changes, in a way that's usable by non-experts with limited statistical background.

*Core insight / definition:* Decomposes the series as y(t) = g(t) + s(t) + h(t) + ε — g(t) piecewise-linear or logistic trend, s(t) Fourier-series seasonality, h(t) holiday effects, ε Gaussian noise. Fits via curve-fitting rather than autocorrelation modeling.

*Why it matters / pitfall / interview trap:* doesn't capture autocorrelation directly and its uncertainty intervals are often miscalibrated (too narrow, ignoring that shocks persist) — its "non-expert friendly" convenience can invite shipping without checking residual autocorrelation or interval coverage.

---

**QDA (Quadratic Discriminant Analysis)**

*The problem:* LDA's assumption of equal covariance across classes is often violated, requiring a more flexible decision boundary.

*Core insight / definition:* Like LDA but allows each class its own covariance matrix, producing quadratic (not linear) decision boundaries; requires O(C·d²) parameters and collapses to LDA when covariances are actually equal.

*Why it matters / pitfall / interview trap:* the added flexibility costs more parameters, increasing overfitting risk especially with limited data per class.

---

**Random Forest**

*The problem:* A single decision tree overfits and is unstable; you want an ensemble that reduces variance while keeping trees as diverse as possible.

*Core insight / definition:* Bagging plus random feature subsets: each tree is trained on a bootstrap sample and, at each split, only considers a random subset of features (max_features), decorrelating trees further than bagging alone. Out-of-bag (OOB) error estimates generalization performance using each tree's unseen (non-bootstrapped) samples, without needing a separate validation set.

*Why it matters / pitfall / interview trap:* max_features controls a bias-variance tradeoff — smaller values decorrelate trees more (lower variance) but each tree is weaker (higher bias); Random Forest generally underperforms gradient boosting on accuracy but trains faster in parallel and is less sensitive to hyperparameter tuning.

---

**Random Oversampling / Undersampling**

*The problem:* Rebalance a skewed class distribution by directly manipulating the dataset composition before training.

*Core insight / definition:* Oversampling duplicates minority-class examples (risk: overfitting to exact duplicates); undersampling drops majority-class examples (risk: losing potentially useful information).

*Why it matters / pitfall / interview trap:* naive oversampling by duplication doesn't add new information, only more weight; SMOTE improves on this by generating synthetic (not duplicated) examples.

---

**Random Search**

*The problem:* Grid search wastes budget evaluating irrelevant hyperparameters at the same resolution as important ones.

*Core insight / definition:* Sample hyperparameter configurations from a distribution (uniform, log-uniform, discrete) rather than an exhaustive grid; with a fixed budget of N evaluations, each hyperparameter dimension gets N distinct sampled values, versus only ᵏ√N per dimension for grid search with k hyperparameters.

*Why it matters / pitfall / interview trap:* still unguided — doesn't learn from past evaluations, so evaluation 50 is as blind as evaluation 1; log-uniform sampling should be used for parameters like learning rate that vary over orders of magnitude.

---

**RBF Kernel (see Kernel Trick)**

---

**Regularization (L1/L2, general)**

*The problem:* An unconstrained model can fit training data arbitrarily well by growing large weights that overfit noise; you need a mechanism to penalize complexity.

*Core insight / definition:* Add a penalty term to the loss proportional to weight magnitude. L2 (Ridge) pushes weights toward zero proportionally, rarely reaching exactly zero; L1 (Lasso) applies constant pressure toward zero, driving small weights to exactly zero (sparsity/implicit feature selection). Elastic Net blends both.

*Why it matters / pitfall / interview trap:* applying regularization to an already high-bias (underfit) model makes it worse, not better — it's a variance-reduction tool, not a universal fix.

---

**Reparameterization Trick**

*The problem:* Training a VAE requires backpropagating gradients through a random sampling step, which is not directly differentiable.

*Core insight / definition:* Rewrite the sampling z ~ N(μ,σ²) as z = μ + σ⊙ε where ε ~ N(0,1) is sampled independently, moving the stochasticity outside the computation graph so gradients can flow through μ and σ.

*Why it matters / pitfall / interview trap:* the specific mechanism that makes VAEs trainable end-to-end via standard backpropagation.

---

**RMSE (Root Mean Squared Error)**

*The problem:* Measure average prediction error while penalizing large errors more heavily than small ones.

*Core insight / definition:* RMSE = √(mean((y−ŷ)²)); squaring amplifies the contribution of large errors/outliers relative to MAE.

*Why it matters / pitfall / interview trap:* more sensitive to outliers than MAE — the right choice depends on whether large errors should be disproportionately penalized in the given application.

---

**ROC-AUC**

*The problem:* Summarize a binary classifier's discrimination ability across all possible decision thresholds in one number.

*Core insight / definition:* Area under the Receiver Operating Characteristic curve (true positive rate vs. false positive rate across thresholds); 0.5 = random, 1.0 = perfect separation.

*Why it matters / pitfall / interview trap:* can look artificially good under severe class imbalance because the false-positive-rate denominator (total negatives) is large — PR-AUC is preferred when the positive class is rare.

---

**R² (Coefficient of Determination)**

*The problem:* Express regression model fit as the proportion of target variance explained, on an interpretable normalized scale.

*Core insight / definition:* R² = 1 − (SS_residual / SS_total); 1.0 means perfect fit, 0 means no better than predicting the mean, and it can go negative for a model worse than the mean baseline.

*Why it matters / pitfall / interview trap:* R² always increases (or stays the same) when adding more features regardless of their usefulness — adjusted R² or held-out validation should be used instead to compare models with different feature counts.

---

**SARIMA(p,d,q)(P,D,Q,s)**

*The problem:* ARIMA cannot represent recurring seasonal patterns (e.g., monthly cycles); a seasonal extension is needed.

*Core insight / definition:* Adds seasonal AR/MA/differencing terms at a seasonal lag s (e.g., s=12 for monthly data), combining seasonal and non-seasonal components in one model.

*Why it matters / pitfall / interview trap:* choosing s incorrectly (e.g., using 12 for weekly data with a 52-week cycle) silently produces a poorly-fit seasonal component.

---

**Scree Plot**

*The problem:* Visually decide how many principal components to retain in PCA.

*Core insight / definition:* Plots eigenvalues (explained variance per component) in decreasing order; the "elbow" — where the curve flattens — marks the point beyond which components mostly capture noise.

*Why it matters / pitfall / interview trap:* the elbow location is subjective; combining it with a hard explained-variance-ratio threshold (e.g., 90-95%) gives a more principled cutoff.

---

**SelectFromModel (see Embedded Methods)**

---

**SES (Simple Exponential Smoothing) (see Exponential Smoothing)**

---

**SHAP (SHapley Additive exPlanations) / TreeSHAP / KernelSHAP**

*The problem:* Attribute an individual model prediction's deviation from a baseline to each input feature in a way grounded in game theory (fair credit allocation), not an ad hoc heuristic.

*Core insight / definition:* Computes each feature's Shapley value — its average marginal contribution across all possible orderings/coalitions of features — guaranteeing consistency and local accuracy (contributions sum exactly to the prediction minus baseline). TreeSHAP computes exact Shapley values efficiently (O(TLD²)) specifically for tree ensembles; KernelSHAP is a slower, sampling-based model-agnostic approximation usable on any model, including neural networks.

*Why it matters / pitfall / interview trap:* TreeSHAP's exactness and speed for gradient-boosted trees is a major reason tree models are preferred in regulated, high-stakes settings (e.g., GDPR adverse-action explanations) over neural networks, whose SHAP explanations are only approximate.

---

**Silhouette Score**

*The problem:* Evaluate clustering quality per-point and overall without ground-truth labels.

*Core insight / definition:* For each point, (b−a)/max(a,b), where a is the mean intra-cluster distance and b is the mean distance to the nearest other cluster; ranges from −1 (likely wrong cluster) to +1 (well-clustered), near 0 meaning the point sits on a cluster boundary.

*Why it matters / pitfall / interview trap:* along with the Elbow Method, commonly used to help choose k for K-Means, but both are heuristics, not definitive answers.

---

**Snapshot Ensembles (see Cosine Annealing with Warm Restarts)**

---

**SMOTE (Synthetic Minority Oversampling Technique)**

*The problem:* Naive oversampling by duplication adds no new information; you need genuinely new minority-class examples to rebalance the data meaningfully.

*Core insight / definition:* Generates synthetic minority examples by interpolating between an existing minority point and one of its k-nearest minority neighbors in feature space.

*Why it matters / pitfall / interview trap:* fails on multimodal minority classes — interpolating between well-separated minority clusters creates unrealistic synthetic points in the gap between them; also risky with very small minority classes (<20 examples), where synthetic points have high variance — cost-sensitive learning is safer in these cases.

---

**SMOTEENN / SMOTETomek**

*The problem:* SMOTE alone can leave noisy or overlapping synthetic/majority points near the new decision boundary, still confusing the classifier.

*Core insight / definition:* Hybrid resampling: apply SMOTE to oversample the minority class, then clean the resulting dataset using Edited Nearest Neighbors (SMOTEENN) or Tomek Links removal (SMOTETomek) to strip out ambiguous or borderline points from both classes.

*Why it matters / pitfall / interview trap:* combines oversampling and undersampling/cleaning in one pipeline — generally more robust than plain SMOTE but adds computational cost and complexity.

---

**Stacking / Stacked Generalization**

*The problem:* A simple average/vote of base models treats each model's output equally; some models may be more reliable in specific regions of feature space, and you'd like to learn how to combine them optimally.

*Core insight / definition:* Train a meta-learner on the out-of-fold (OOF) predictions of base models (generated via cross_val_predict to avoid leakage — base models never predict on data they were trained on), so the meta-learner learns how to weight/combine base model outputs. Can be extended to multi-level stacking.

*Why it matters / pitfall / interview trap:* using in-sample (non-OOF) base model predictions to train the meta-learner leaks information and causes the meta-learner to over-trust models that overfit their own training data.

---

**Stationarity**

*The problem:* Most classical time-series models (ARIMA) assume the underlying statistical properties of the series don't change over time; violating this produces spurious model fits and misleading regressions.

*Core insight / definition:* A series is (weakly/covariance) stationary if its mean, variance, and autocovariance (as a function of lag only, not absolute time) are all constant over time. Strict stationarity requires the entire joint distribution to be time-shift invariant. Tested via the Augmented Dickey-Fuller (ADF) test (H₀: unit root/non-stationary; reject H₀ → stationary).

*Why it matters / pitfall / interview trap:* trend-stationary series (deterministic trend, removable by detrending) must be distinguished from difference-stationary/unit-root series (stochastic trend, shocks have permanent effects, requires differencing not detrending) — detrending a difference-stationary series doesn't fully remove non-stationarity.

---

**STL Decomposition**

*The problem:* Detect anomalies or extract structure in a time series by separating it into interpretable components.

*Core insight / definition:* Seasonal-Trend decomposition using Loess splits a series into trend, seasonal, and residual components; large deviations in the residual (e.g., beyond k standard deviations) flag anomalies.

*Why it matters / pitfall / interview trap:* anomaly thresholds must account for the residual's own scale (k·σ), and seasonality must be correctly modeled first or normal seasonal peaks get flagged as anomalies.

---

**Supervised vs. Unsupervised vs. Reinforcement Learning**

*The problem:* Different problems provide fundamentally different types of feedback during training, which determines the appropriate learning paradigm.

*Core insight / definition:* Supervised: labeled examples provide the richest, most direct feedback signal (bottleneck: expensive/biased label acquisition). Unsupervised: no labels, goal shifts to finding structure in the input distribution (underspecified — "structure" isn't uniquely defined by data alone). Reinforcement: an agent acts and receives delayed scalar reward; the defining challenge is credit assignment (which of many past actions caused a later outcome).

*Why it matters / pitfall / interview trap:* treating these as mutually exclusive misses hybrids (self-supervised, semi-supervised, weakly supervised); treating unsupervised as "unguided" ignores that distance metric, k, and architecture choices all embed prior assumptions about expected structure.

---

**Support Vector Machine (SVM)**

*The problem:* Find a decision boundary that not only separates classes but maximizes the margin to the nearest points of each class, for better generalization.

*Core insight / definition:* Finds the hyperplane maximizing the margin between classes; hard margin requires perfect linear separability, soft margin (via a slack/C parameter) tolerates some misclassification for better generalization on noisy/non-separable data. Only points on or inside the margin (support vectors) determine the boundary — other points don't affect it. The kernel trick extends this to non-linear boundaries.

*Why it matters / pitfall / interview trap:* the C parameter trades off margin width against misclassification tolerance — large C overfits (narrow margin, few errors tolerated), small C underfits (wide margin, more tolerance); requires feature scaling since it's distance-based.

---

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**

*The problem:* Visualize high-dimensional data in 2D/3D such that nearby points in the original space remain nearby, when the underlying structure is non-linear.

*Core insight / definition:* Models pairwise similarities in high dimensions as a Gaussian-based probability distribution, and finds a low-dimensional arrangement (using a heavy-tailed t-distribution to fix the crowding problem) whose similarities match by minimizing KL divergence via gradient descent. Governed by the perplexity parameter (effective neighborhood size).

*Why it matters / pitfall / interview trap:* preserves neither absolute distances nor cluster sizes — cluster separation and size in a t-SNE plot carry no reliable meaning about the original space; has no out-of-sample transform (fit_transform only) and results vary across random seeds/perplexities, so a single run should never be treated as ground truth.

---

**Temperature Scaling**

*The problem:* Calibrate a neural network's softmax outputs, which tend to be systematically overconfident, using a single learned parameter.

*Core insight / definition:* Divides the logits by a learned scalar temperature T > 1 before applying softmax, softening the output distribution to better match empirical accuracy, fit on a held-out calibration set.

*Why it matters / pitfall / interview trap:* is neural networks' standard calibration step, analogous to Platt scaling/isotonic regression for classical models, but requires an additional held-out calibration set and ongoing monitoring that logistic regression's native calibration avoids.

---

**Threshold Moving**

*The problem:* The default 0.5 classification threshold implicitly assumes equal cost for false positives and false negatives, which is rarely true, especially under class imbalance.

*Core insight / definition:* Sweep the precision-recall curve post-training and select the decision threshold that achieves a target precision, target recall, or optimal F-beta for the actual cost structure of the problem — a purely post-hoc lever requiring no retraining.

*Why it matters / pitfall / interview trap:* composes cleanly with resampling and cost-sensitive learning as one of classical ML's independent, stackable levers for handling class imbalance.

---

**TCN (Temporal Convolutional Network)**

*The problem:* Model long-range dependencies in sequential data without the sequential (non-parallelizable) computation of RNNs.

*Core insight / definition:* Uses causal (only past inputs), dilated convolutions where dilation grows exponentially across layers, giving an exponentially growing receptive field (2^L at layer L) while remaining fully parallelizable and respecting causality (no future leakage).

*Why it matters / pitfall / interview trap:* causal padding must be implemented carefully (pad only on the past side) to avoid leaking future information into a "past-only" prediction.

---

**TimeSeriesSplit / Walk-Forward Validation**

*The problem:* A random train/test split on time series leaks future information into the training set, since the split ignores temporal order.

*Core insight / definition:* Iteratively expand the training window and evaluate on the next unseen time block (train on [1..t], test on [t+1]; then train on [1..t+1], test on [t+2]; etc.), simulating how the model would actually be deployed and retrained over time.

*Why it matters / pitfall / interview trap:* a model that performs well under walk-forward validation is not guaranteed to hold up in production if the regime later shifts — good test performance certifies the past, not future regime changes; a shadow-serve period is the more reliable production check.

---

**Tomek Links**

*The problem:* After oversampling or in imbalanced data generally, majority and minority class points that are each other's nearest neighbor sit right on/near the decision boundary, confusing the classifier.

*Core insight / definition:* Identifies pairs of opposite-class points that are mutual nearest neighbors (a "Tomek link") and removes the majority-class point from each pair, cleaning up the boundary region.

*Why it matters / pitfall / interview trap:* an undersampling/cleaning technique, often combined with SMOTE (as SMOTETomek) rather than used alone.

---

**TPE (Tree-structured Parzen Estimator)**

*The problem:* Fitting a full Gaussian Process surrogate for Bayesian optimization scales poorly (O(n³)) and can struggle with the mixed discrete/continuous, conditional structure typical of hyperparameter spaces.

*Core insight / definition:* Optuna's default sampler; instead of modeling the objective directly, it models two densities — l(λ), the distribution of configurations scoring above a threshold, and g(λ), those scoring below — and proposes the next configuration maximizing l(λ)/g(λ), i.e., configurations that look like good ones rather than bad ones.

*Why it matters / pitfall / interview trap:* scales better than GP-based Bayesian optimization to larger, more complex hyperparameter spaces, which is why it's Optuna's default.

---

**Train/Validation/Test Split**

*The problem:* Reliably estimate how a model will perform on genuinely unseen data, distinct from data used to fit or tune it.

*Core insight / definition:* Training set fits model parameters; validation set guides model/hyperparameter selection; test set (touched only once, at the very end) gives the final, unbiased generalization estimate.

*Why it matters / pitfall / interview trap:* repeatedly consulting the "test" set during development turns it into a second validation set, invalidating its role as an unbiased final check.

---

**UMAP (Uniform Manifold Approximation and Projection)**

*The problem:* t-SNE is slow, lacks an out-of-sample transform, and discards global structure — a faster, more general-purpose embedding method is needed.

*Core insight / definition:* Models data as a weighted topological graph (fuzzy simplicial complex approximating the manifold) built from each point's k nearest neighbors with fuzzy edge weights, then optimizes a low-dimensional graph to match it by minimizing cross-entropy via SGD with negative sampling. Key parameters: n_neighbors (local vs. global structure tradeoff) and min_dist (how tightly points can pack in the embedding).

*Why it matters / pitfall / interview trap:* supports true out-of-sample transform() (unlike t-SNE) and is deterministic-ish (spectral init), but still optimizes a local-structure objective and distorts global distances — its embeddings are usable as downstream features in a way t-SNE's aren't, but inter-cluster gaps/densities remain not fully trustworthy.

---

**VAE (Variational Autoencoder)**

*The problem:* Standard autoencoders' latent space is unstructured — nearby latent points can decode to unrelated outputs, making interpolation and generation unreliable.

*Core insight / definition:* Encodes each input to a distribution (mean, variance) rather than a point, adding a KL-divergence regularizer pulling that distribution toward a standard Gaussian prior, which smooths the latent space so nearby points decode similarly. Trained via the reparameterization trick to allow gradients through the sampling step.

*Why it matters / pitfall / interview trap:* the KL term is what gives VAEs a proper generative/probabilistic structure that plain autoencoders lack — without it, you just get a regular (non-generative) autoencoder.

---

**Variance Threshold (feature selection)**

*The problem:* Near-constant features carry essentially no information and add noise/dimensionality for free.

*Core insight / definition:* A filter method that drops any feature whose variance falls below a specified threshold (e.g., a feature that's 99% one value).

*Why it matters / pitfall / interview trap:* the simplest, cheapest filter method, but only catches globally low-variance features — a feature could have high variance yet still be irrelevant to the target.

---

**Voting Ensemble (Hard / Soft)**

*The problem:* Combine predictions from several independently-trained, diverse models into a single, typically more robust prediction.

*Core insight / definition:* Hard voting takes the majority class vote across models; soft voting averages predicted class probabilities and picks the highest-probability class, generally performing better when base models are reasonably calibrated.

*Why it matters / pitfall / interview trap:* soft voting is only meaningful if base models' probability outputs are on comparable, meaningful scales — hence Calibration Before Ensembling as a companion technique.

---

**Walk-Forward Validation (see TimeSeriesSplit)**

---

**Ward's Linkage (see Hierarchical Clustering)**

---

**WCSS / Inertia**

*The problem:* K-Means needs an explicit objective to minimize when assigning points to clusters and updating centroids.

*Core insight / definition:* Within-Cluster Sum of Squares (inertia) = Σ over clusters Σ over points ‖x − centroid‖²; K-Means's coordinate-descent-style updates monotonically decrease this objective until convergence to a local minimum.

*Why it matters / pitfall / interview trap:* WCSS always decreases as k increases (more clusters can only reduce within-cluster spread), so it cannot be used directly to choose k — hence the Elbow Method looks for diminishing returns rather than the minimum.

---

**Wrapper Methods (feature selection)**

*The problem:* Filter methods score features independent of any model and miss interaction effects; you want feature selection guided by actual downstream model performance.

*Core insight / definition:* Repeatedly train and evaluate a model on different feature subsets to search for the best-performing subset — Recursive Feature Elimination (RFE, iteratively remove the least important feature and retrain) or Sequential Feature Selection (greedily add/remove features one at a time based on validation performance).

*Why it matters / pitfall / interview trap:* captures feature interactions that filter methods miss but is computationally expensive since it requires retraining the model many times.

---

**XGBoost**

*The problem:* Scale gradient boosting to be fast, regularized, and robust enough for large-scale production tabular ML.

*Core insight / definition:* Implements gradient boosting using a second-order (Newton) approximation of the loss (gradient + Hessian) to compute optimal leaf scores in closed form, with explicit L1/L2 regularization on leaf weights and tree structure, and level-wise tree growth (all nodes at a depth are considered before going deeper). Split gain criterion directly incorporates the regularization penalty, so splits that don't sufficiently reduce loss net of complexity cost are pruned.

*Why it matters / pitfall / interview trap:* against Random Forest, XGBoost generally achieves better accuracy through sequential error correction but is more sensitive to hyperparameters (learning rate, depth, regularization) and easier to overfit without careful tuning/early stopping.

---

## 04-evaluation

**Accuracy**

*The problem:* Teams need a single headline number to summarize how often a classifier is right.

*Core insight / definition:* Accuracy = (TP+TN)/(TP+TN+FP+FN), the fraction of all predictions that were correct. It treats false positives and false negatives as equally costly and treats all classes as equally important.

*Why it matters / pitfall / interview trap:* On a 99/1 imbalanced dataset, a model that always predicts the majority class gets 99% accuracy while being useless — always pair accuracy with precision/recall/F1 or PR-AUC on imbalanced problems.

---

**Accumulated Local Effects (ALE)**

*The problem:* Partial Dependence Plots extrapolate into unrealistic feature combinations when features are correlated.

*Core insight / definition:* ALE computes the average change in prediction within narrow bins of the feature of interest (using only real neighboring data), then accumulates these local differences into a global effect curve, avoiding the need to average over unrealistic combinations of correlated features.

*Why it matters / pitfall / interview trap:* ALE is preferred over PDP whenever features are correlated, since it isolates the true marginal effect without extrapolating off-manifold.

---

**Adjusted R²**

*The problem:* Plain R² always increases (or stays the same) as more features are added, even useless ones, making model comparison across different feature counts misleading.

*Core insight / definition:* $R^2_{\text{adj}} = 1-(1-R^2)\frac{n-1}{n-p-1}$ penalizes for the number of predictors $p$, only increasing when a new feature improves fit more than chance would predict.

*Why it matters / pitfall / interview trap:* Use adjusted R² (not raw R²) when comparing regression models with different numbers of predictors.

---

**Aleatoric Uncertainty**

*The problem:* Not all model uncertainty is the same — some is irreducible noise, some reflects the model's own ignorance, and each demands a different response.

*Core insight / definition:* Aleatoric uncertainty is the variance of the label given the input, $\text{Var}[Y|X=x]$ — inherent noise in the data-generating process (e.g., label noise, sensor error) that cannot be reduced by collecting more data. Estimated as the average entropy of predictions across stochastic forward passes.

*Why it matters / pitfall / interview trap:* Aleatoric uncertainty should be communicated to the downstream decision-maker, not "fixed" — routing genuinely ambiguous cases to human review wastes reviewer time on cases no additional data can resolve. Also, the aleatoric/epistemic split is a property of the model and estimation method, not a clean fact about the data.

---

**AIPW (Augmented Inverse Propensity Weighting)** — see Doubly Robust Estimation

---

**Alpha Spending / Sequential Testing**

*The problem:* Continuously monitoring an experiment and stopping as soon as p < 0.05 inflates the true false-positive rate far above 5%.

*Core insight / definition:* Sequential testing methods (alpha spending) allocate the total error budget across multiple looks at the data over time, allowing valid early stopping without inflating the false-positive rate, as an alternative to committing to one fixed sample size and duration.

*Why it matters / pitfall / interview trap:* "Peeking" without a sequential-testing correction can push the effective false-positive rate to 20-30%.

---

**Anscombe's Quartet**

*The problem:* A single summary statistic like R² can look identical for datasets with wildly different underlying structure.

*Core insight / definition:* Anscombe's quartet is four datasets with nearly identical R² (and other summary statistics) but visually and structurally very different distributions (linear, curved, outlier-driven, etc.).

*Why it matters / pitfall / interview trap:* High R² alone does not guarantee a good model fit — always visualize residuals, don't rely on a single summary number.

---

**Attention Weights as Explanation**

*The problem:* For transformer models, practitioners often visualize attention weights and call them "explanations" of the model's decision.

*Core insight / definition:* Attention weights are intermediate computations, not causal attributions — the same attention pattern can be produced by different underlying weight matrices, so attention does not reliably indicate what drove the prediction.

*Why it matters / pitfall / interview trap:* Prefer principled attribution methods (Integrated Gradients, SHAP) over raw attention visualization when the audience needs a real explanation.

---

**Average Precision (AP) / Mean Average Precision (MAP)**

*The problem:* Ranking metrics need to reward putting relevant documents early in a list of binary-relevance results, not just anywhere in the top k.

*Core insight / definition:* AP = $\frac{1}{R}\sum_k P@k \cdot \mathbf{1}[\text{doc}_k \text{ relevant}]$, averaging precision at each rank where a relevant item appears; MAP = the mean of AP across all queries.

*Why it matters / pitfall / interview trap:* MAP assumes binary relevance and is sensitive to missing relevant items (unlike NDCG at a fixed k) — use NDCG instead when relevance is graded.

---

**ATE (Average Treatment Effect) vs. ATT (Average Treatment Effect on the Treated)**

*The problem:* "What effect?" is ambiguous — the effect of treating everyone can differ from the effect on the people who were actually treated, if the effect is heterogeneous.

*Core insight / definition:* With potential outcomes $Y(1)$, $Y(0)$: ATE = $\mathbb{E}[Y(1)-Y(0)]$ over the whole population ("if we treated everyone vs. no one"); ATT = $\mathbb{E}[Y(1)-Y(0)\mid T=1]$, the effect for units that actually got treated.

*Why it matters / pitfall / interview trap:* DiD estimates ATT; IPW can target either. Rollout decisions usually care about ATT; policy decisions about scaling to everyone want ATE — conflating them misstates what a causal estimate implies.

---

**Back-door Path / Back-door Criterion**

*The problem:* Confounders create spurious statistical association between treatment and outcome that has nothing to do with a causal effect.

*Core insight / definition:* A back-door path is a non-causal path connecting treatment $T$ to outcome $Y$ through a confounder, e.g. $T \leftarrow C \to Y$. The back-door criterion says: block every back-door path by conditioning on an appropriate set of variables to recover the true causal effect.

*Why it matters / pitfall / interview trap:* "Throw all features into the regression" is not automatically safe — conditioning on the wrong variable (a collider or mediator) can introduce bias rather than remove it.

---

**Benjamini-Hochberg Procedure**

*The problem:* Running many hypothesis tests inflates the number of expected false positives, but a fully conservative correction like Bonferroni may be too strict when some false positives are acceptable.

*Core insight / definition:* Benjamini-Hochberg controls the False Discovery Rate (FDR) — the expected proportion of false positives among rejected hypotheses — rather than the family-wise error rate, allowing more power when many tests are run.

*Why it matters / pitfall / interview trap:* Benjamini-Hochberg is almost always preferable to Bonferroni in ML contexts where many comparisons (e.g., many metrics or many segments) are tested simultaneously.

---

**BERTScore**

*The problem:* n-gram overlap metrics (BLEU, ROUGE) miss semantic similarity when generated text paraphrases the reference.

*Core insight / definition:* BERTScore measures similarity using contextual embeddings (e.g., from BERT) rather than exact token overlap, capturing semantic closeness between generated and reference text.

*Why it matters / pitfall / interview trap:* Expensive to compute and biased toward text that is fluent in the embedding model's own style, so it can still be gamed or misleading.

---

**Bias-Variance of CV Estimate**

*The problem:* Practitioners assume more folds is strictly better for cross-validation.

*Core insight / definition:* Increasing K reduces bias of the CV estimate (each training set is larger, closer to the full dataset) but increases variance because folds overlap more and become correlated; the variance does not shrink as 1/K.

*Why it matters / pitfall / interview trap:* Leave-one-out CV (K=n) has near-zero bias but very high variance and n training runs — more folds is not automatically a better estimate.

---

**BLEU**

*The problem:* Evaluating machine-generated text against reference text needs an automatable proxy for translation/generation quality.

*Core insight / definition:* BLEU scores n-gram overlap between generated text and one or more reference texts, rewarding matches at multiple n-gram lengths.

*Why it matters / pitfall / interview trap:* BLEU rewards short, "safe" outputs and misses semantic correctness — a perfect translation using different words than the reference can score near 0.

---

**Bonferroni Correction**

*The problem:* Testing k hypotheses at α=0.05 each yields an expected ~0.05k false positives.

*Core insight / definition:* Bonferroni controls the family-wise error rate (FWER) by dividing the significance threshold α by the number of tests k, so each individual test uses α/k.

*Why it matters / pitfall / interview trap:* Bonferroni becomes very conservative (low power) as the number of comparisons grows; Holm-Bonferroni is a less conservative FWER-controlling alternative, and Benjamini-Hochberg (FDR control) is usually preferred with many tests.

---

**Brier Score**

*The problem:* Need a single number to evaluate probability forecasts, not just hard classification correctness.

*Core insight / definition:* Brier score = $\frac{1}{n}\sum_i(\hat p_i - y_i)^2$, the mean squared error between predicted probabilities and binary outcomes; range [0,1], lower is better. Decomposes (Murphy decomposition) into Calibration + Resolution − Uncertainty, and is a strictly proper scoring rule.

*Why it matters / pitfall / interview trap:* Because it's proper, the minimum is only achieved by outputting true probabilities — a model can't game Brier score by being confidently wrong. Brier Skill Score (BSS) = $1-\text{Brier}/\text{Brier}_{\text{ref}}$ compares against a base-rate baseline.

---

**Calibration**

*The problem:* A model's hard predictions can be accurate while its probability outputs are numerically meaningless — dangerous for any downstream system that consumes the probability directly (pricing, risk thresholds).

*Core insight / definition:* A classifier is calibrated if $P(Y=1\mid \hat p(X)=p)=p$ for all $p$: among all predictions assigned probability $p$, the empirical fraction of positives should equal $p$. Calibration and ranking quality (AUC) are orthogonal properties.

*Why it matters / pitfall / interview trap:* A model can have AUC = 1.0 (perfect ranking) and be arbitrarily miscalibrated (e.g., all positives at 0.52, all negatives at 0.48). Skipping calibration assessment because "AUC looks good" is a very common trap.

---

**Calibration Set**

*The problem:* Any post-hoc calibration method (Platt, isotonic, temperature scaling, conformal prediction) needs data to fit the mapping from raw scores to calibrated probabilities, but using training or test data for this leaks information.

*Core insight / definition:* A calibration set is a held-out split, separate from both training and test data, used exclusively to fit a calibration mapping or to compute conformal nonconformity score quantiles.

*Why it matters / pitfall / interview trap:* Fitting calibration on the training set makes the calibration curve optimistic; using the test set for calibration is data leakage that invalidates the final evaluation.

---

**Canary Release**

*The problem:* Even offline-validated and shadow-tested models can behave unexpectedly at scale in production; a full rollout risks broad damage.

*Core insight / definition:* A canary release serves the new model to a small fraction of live traffic (1-5%) while closely monitoring business metrics, with readiness to roll back before wider exposure.

*Why it matters / pitfall / interview trap:* Canary is a stage after shadow mode in the standard evaluation progression: offline → shadow → canary → full rollout with guardrails.

---

**Class Imbalance**

*The problem:* Real-world classification problems (fraud, disease, spam) often have a rare positive class, which breaks metrics and default thresholds designed for balanced data.

*Core insight / definition:* Class imbalance refers to a skewed ratio between classes (e.g., 1% positive, 99% negative), under which accuracy, ROC-AUC, and the 0.5 threshold all become misleading or suboptimal.

*Why it matters / pitfall / interview trap:* Under severe imbalance, PR-AUC, F-beta, expected cost, and a cost-derived threshold are more informative than accuracy or ROC-AUC alone.

---

**Cluster Randomization**

*The problem:* In social networks, marketplaces, or ride-sharing, individually randomizing users violates independence because treated and control users interact and contaminate each other.

*Core insight / definition:* Cluster randomization assigns entire clusters (social groups, geographic regions) to treatment or control as a unit, rather than individual users, to preserve the independence assumption behind standard statistical tests — an application of avoiding SUTVA violations.

*Why it matters / pitfall / interview trap:* Failing to cluster-randomize in an interference-prone setting means the estimated treatment effect is biased by contamination between arms.

---

**Coefficient of Determination** — see R²

---

**Collider**

*The problem:* Practitioners often assume "control for more variables = safer," but this is false in causal analysis.

*Core insight / definition:* A collider is a variable caused by both the treatment and the outcome (or by two other variables in the DAG). Conditioning on (or controlling for) a collider opens a spurious association path and introduces bias rather than removing it.

*Why it matters / pitfall / interview trap:* "Throw all covariates into the regression" is a common but dangerous heuristic — the causal DAG determines which variables are safe to control for, and colliders are not.

---

**Conditional Coverage vs. Marginal Coverage**

*The problem:* A prediction-interval method might guarantee correct average coverage overall while still being badly wrong for specific subgroups.

*Core insight / definition:* Marginal coverage is the guarantee $P(Y_{n+1}\in C(X_{n+1}))\geq 1-\alpha$ averaged over the whole population; conditional coverage would guarantee this within any subgroup or region of covariate space, which conformal prediction does NOT guarantee.

*Why it matters / pitfall / interview trap:* Conformal prediction's guarantee is only marginal — a specific subgroup can be systematically under-covered even while the overall average meets the target.

---

**Confidence Interval (for A/B test effect)**

*The problem:* A point-estimate effect size alone hides how much sampling uncertainty surrounds it.

*Core insight / definition:* The 95% CI for a treatment effect is computed as effect_size ± 1.96·SE, where SE = $\sqrt{\text{Var(treatment)}/n_t + \text{Var(control)}/n_c}$, giving a range of plausible true effect sizes.

*Why it matters / pitfall / interview trap:* Report effect size and CI alongside p-value — "p<0.001" alone doesn't reveal whether the effect is practically meaningful.

---

**Confounder**

*The problem:* Comparing treated and untreated (or exposed and unexposed) units directly can produce an association that reflects pre-existing differences rather than a causal effect.

*Core insight / definition:* A confounder is a variable that causes both the treatment and the outcome, opening a back-door path that creates spurious association independent of any real causal effect (e.g., user "intent" driving both feature adoption and spend).

*Why it matters / pitfall / interview trap:* Every adjustment method except IV assumes all confounders are measured (ignorability) — this assumption can never be fully verified, and unmeasured confounding silently biases the result with no built-in warning.

---

**Confusion Matrix**

*The problem:* A single summary metric aggregates four qualitatively different types of errors, hiding which failure mode a model actually has.

*Core insight / definition:* The confusion matrix is the 2×2 (or n×n) table of TP, TN, FP, FN counts from which every classification metric (precision, recall, specificity, accuracy, F1, MCC) can be derived.

*Why it matters / pitfall / interview trap:* Two models with identical accuracy can have completely different confusion matrices (one high-recall/low-precision, the other the reverse) — always inspect the full matrix before choosing a summary metric.

---

**Conformal Prediction**

*The problem:* Standard prediction intervals (e.g., $\hat y \pm 2\sigma$) assume a distributional form that may not hold, and don't come with a provable coverage guarantee.

*Core insight / definition:* Conformal prediction uses a calibration set of nonconformity scores $s_i=|y_i-\hat y_i|$ to build a prediction interval/set that provably contains the true label with probability ≥ 1−α under only the assumption of exchangeability, with no distributional assumptions on the model or data.

*Why it matters / pitfall / interview trap:* The guarantee is marginal (averaged over all test points), not conditional per subgroup, and it breaks under distribution shift because shift violates exchangeability.

---

**Cost Matrix / Expected Cost**

*The problem:* Standard accuracy assumes all four confusion-matrix cells have equal cost, which is rarely true in business settings (e.g., missing fraud vs. a false alarm).

*Core insight / definition:* A cost matrix assigns a distinct cost/benefit to each cell (TP, FP, FN, TN); expected cost = $\frac1n\sum_i[y_i(\hat y_i C(TP)+(1-\hat y_i)C(FN)) + (1-y_i)(\hat y_i C(FP)+(1-\hat y_i)C(TN))]$.

*Why it matters / pitfall / interview trap:* Once costs are asymmetric, the optimal decision threshold is generally not 0.5 — it should be derived from the cost ratio (see Optimal Threshold Under Cost).

---

**Coverage (Recommendation Diversity Metric)**

*The problem:* A recommender can maximize relevance metrics like NDCG while only ever recommending a handful of popular items, providing poor user experience and failing long-tail users.

*Core insight / definition:* Coverage = |unique items recommended across all users| / |catalog size| — measures whether the system explores the full catalog or concentrates on a few items.

*Why it matters / pitfall / interview trap:* A system with 1% coverage is vulnerable to cold-start problems and systematically fails users interested in the long tail, even with high relevance scores.

---

**Coverage Guarantee (Prediction Intervals)**

*The problem:* A prediction interval is only useful if it reliably contains the true value at some known rate.

*Core insight / definition:* A prediction interval $[\hat l(x),\hat u(x)]$ has coverage $1-\alpha$ if $P(Y_{n+1}\in[\hat l(X_{n+1}),\hat u(X_{n+1})])\geq 1-\alpha$.

*Why it matters / pitfall / interview trap:* A trivial model can achieve 100% coverage by returning $[-\infty,+\infty]$ — coverage must always be judged jointly with interval width (MPIW), never alone.

---

**Cross-Entropy Loss** — see Log Loss

---

**Cross-Validation (K-Fold)**

*The problem:* A single train/validation split gives one noisy estimate of generalization performance that depends heavily on which examples happened to land in the test set.

*Core insight / definition:* K-fold CV splits training data into K folds, trains on K−1 folds and evaluates on the held-out fold, rotating which fold is held out, then averages the K scores; each sample is used as a test point exactly once.

*Why it matters / pitfall / interview trap:* CV reduces variance of the performance estimate, but it operates only on training data — the test set stays untouched. Preprocessing steps (scalers, imputers) must be refit inside each fold, not on the full dataset beforehand, to avoid leakage.

---

**CUPED (Controlled-experiment Using Pre-Experiment Data)**

*The problem:* High-variance metrics (e.g., revenue per user) require huge sample sizes and long experiment durations to reach statistical significance.

*Core insight / definition:* CUPED adjusts the outcome metric by subtracting the portion of variance explained by a user's pre-experiment behavior: $Y_{\text{cuped}} = Y - \theta(X-\mathbb{E}[X])$, where $\theta=\text{Cov}(Y,X)/\text{Var}(X)$; this reduces variance by roughly 30–70% for metrics correlated with historical behavior.

*Why it matters / pitfall / interview trap:* CUPED requires the same metric to be measurable before the experiment starts — it cannot be applied to genuinely new metrics with no pre-period history.

---

**DAG (Directed Acyclic Graph, Causal)**

*The problem:* Deciding which variables to "control for" by intuition often leads to including variables that bias, rather than fix, a causal estimate.

*Core insight / definition:* A causal DAG makes explicit which variables are confounders (open a back-door path), mediators (lie on the causal path $T\to M\to Y$), or colliders (caused by both $T$ and $Y$), and dictates which should be adjusted for (confounders only).

*Why it matters / pitfall / interview trap:* Conditioning on a mediator removes part of the real causal effect; conditioning on a collider introduces bias — "control for everything" is not a safe default.

---

**Deep Ensembles**

*The problem:* A single deterministic model gives no signal about how confident it should be, and simpler uncertainty methods (MC Dropout) can underestimate true epistemic uncertainty.

*Core insight / definition:* Deep ensembles train N independently initialized models on the same data; the variance of their predictions estimates epistemic uncertainty, and their average is typically better calibrated than any single model.

*Why it matters / pitfall / interview trap:* Deep ensembles outperform MC Dropout for uncertainty quality but cost N× the training and inference compute; also, correlated ensemble members can be jointly, confidently wrong on truly novel (OOD) inputs, so low ensemble variance is not proof of reliable knowledge.

---

**Difference-in-Differences (DiD)**

*The problem:* Treatment turned on for one group but not another at a known time, and you have before/after data for both, but a naive treated-group-only before/after comparison confounds the treatment effect with any concurrent trend.

*Core insight / definition:* DiD cancels time-invariant confounders and common time trends by double-differencing: $\widehat{\text{ATT}} = (\bar Y_{\text{treat,post}}-\bar Y_{\text{treat,pre}}) - (\bar Y_{\text{ctrl,post}}-\bar Y_{\text{ctrl,pre}})$.

*Why it matters / pitfall / interview trap:* Relies on the parallel trends assumption (absent treatment, both groups would have moved identically) — supported by checking pre-period trends move in parallel, and testable with a placebo/pre-trend test. With staggered adoption across many units and heterogeneous effects, classic two-way fixed effects is biased — use a Callaway–Sant'Anna style estimator instead.

---

**Discounted Cumulative Gain (DCG)** — see NDCG

---

**Doubly Robust Estimation (AIPW)**

*The problem:* Both propensity-score methods and outcome-regression methods for causal effects rely on one model being correctly specified, and you don't know in advance which (if either) is right.

*Core insight / definition:* AIPW combines an outcome model $\hat\mu(x)$ with an inverse-propensity-weighted correction term; it is consistent if either the propensity model OR the outcome model is correctly specified — "two shots at getting it right."

*Why it matters / pitfall / interview trap:* This is the modern default for observational causal estimation and underlies double/debiased ML, where both nuisance models are fit with ML estimators and cross-fitting.

---

**ECE (Expected Calibration Error)**

*The problem:* Reliability diagrams are visual; teams need a single scalar to compare calibration between models, track it over time, and set alert thresholds.

*Core insight / definition:* $\text{ECE}=\sum_m \frac{|B_m|}{n}|\text{acc}(B_m)-\text{conf}(B_m)|$, the bin-population-weighted average gap between predicted confidence and empirical accuracy across bins.

*Why it matters / pitfall / interview trap:* ECE depends on bin count and binning strategy — only compare ECE across models computed with identical binning settings. ECE < 0.01 is excellent; > 0.10 means raw probabilities should not be used for downstream decisions.

---

**Efficiency (Shapley Axiom)**

*The problem:* An attribution method is only useful for explanation if the individual feature contributions add up to something meaningful about the prediction.

*Core insight / definition:* Efficiency is the Shapley axiom stating $\sum_i \phi_i = f(x) - \mathbb{E}[f(x)]$ — SHAP values sum exactly to the difference between the prediction and the baseline (expected) prediction.

*Why it matters / pitfall / interview trap:* This is the axiom to remember for interviews — it's what makes SHAP usable in an adverse-action notice: baseline plus per-feature contributions leaves nothing unexplained.

---

**Epistemic Uncertainty**

*The problem:* Distinguishing "this input is inherently ambiguous" from "the model just hasn't seen enough like this" requires different remedies (accept vs. collect more data / flag for review).

*Core insight / definition:* Epistemic uncertainty is uncertainty about model parameters given finite training data — $\text{Var}_\theta[\mathbb{E}[Y|X,\theta]]$ — and decreases as more data is collected in the uncertain region. Estimated via variance across stochastic forward passes (MC Dropout) or across ensemble members.

*Why it matters / pitfall / interview trap:* High epistemic uncertainty should trigger human review or active learning, but ensemble/MC-Dropout variance can be small even on genuinely novel inputs if all members share the same blind spot — variance measures disagreement, not distance from training data.

---

**Exchangeability**

*The problem:* Conformal prediction needs some assumption connecting calibration data to future test data, but wants to avoid strong distributional assumptions.

*Core insight / definition:* Exchangeability means the joint distribution of the data is invariant to the order of the samples — a weaker assumption than i.i.d. that is sufficient for conformal prediction's coverage guarantee.

*Why it matters / pitfall / interview trap:* Under distribution shift, exchangeability breaks, and with it the conformal coverage guarantee.

---

**Exclusion Restriction (IV)**

*The problem:* An instrumental variable must not have any path to the outcome except through the treatment, or the estimate is biased.

*Core insight / definition:* The exclusion restriction is the requirement that instrument $Z$ affects outcome $Y$ only through treatment $T$, never directly or through any other path.

*Why it matters / pitfall / interview trap:* Exclusion (along with independence) is untestable — it must be argued from domain knowledge, and a skeptic can always propose an alternative path from $Z$ to $Y$.

---

**Expected Cost** — see Cost Matrix

---

**Explainability vs. Interpretability**

*The problem:* Interviewers probe whether candidates conflate two related but distinct concepts describing how understandable a model's behavior is.

*Core insight / definition:* Interpretability is a property of the model itself — you can inspect its mechanism directly (e.g., a linear model or shallow tree). Explainability is a property of an account generated after the fact for a model you did not constrain (e.g., a SHAP plot over a GBM) — it's a model of the model.

*Why it matters / pitfall / interview trap:* Because post-hoc explanations are themselves models, their failure mode is being wrong about the thing they explain — this framing is central to understanding why SHAP/LIME can mislead.

---

**Exposure Fairness**

*The problem:* Standard ranking metrics don't check whether protected groups receive proportional visibility relative to their relevance in the corpus.

*Core insight / definition:* Exposure fairness requires that protected groups receive ranking exposure proportional to their relevance, used in hiring/lending/content platforms to prevent systematic suppression of minority groups.

*Why it matters / pitfall / interview trap:* A niche but real concern in specialized ranking/fairness roles — standard relevance metrics can be maximized while still systematically under-exposing a protected group.

---

**F1 Score / F-beta Score**

*The problem:* Precision and recall trade off against each other, and stakeholders often want one number, but a naive average can hide extreme imbalance between the two.

*Core insight / definition:* F1 = $2PR/(P+R)$, the harmonic mean of precision and recall (not the arithmetic mean) — it punishes extreme imbalance (P=1.0, R=0.1 gives F1=0.18, not 0.55). $F_\beta = (1+\beta^2)PR/(\beta^2P+R)$ generalizes this, with $\beta>1$ weighting recall more and $\beta<1$ weighting precision more.

*Why it matters / pitfall / interview trap:* Defaulting to F1 without justifying that precision and recall have equal cost is a common trap — derive $\beta$ from the actual cost ratio of false positives vs. false negatives.

---

**Feature Independence Assumption (PDP)**

*The problem:* Partial Dependence Plots compute marginal effects by averaging over other features, which silently assumes those features are independent of the feature being studied.

*Core insight / definition:* PDP's feature-independence assumption means that when the feature of interest is correlated with others, the averaging process creates unrealistic/impossible feature combinations (e.g., age=90 with a $20K student income).

*Why it matters / pitfall / interview trap:* This is exactly the failure mode that motivates using ALE instead of PDP under correlated features.

---

**FID (Fréchet Inception Distance)**

*The problem:* Evaluating image-generation quality needs a quantitative proxy since human evaluation at scale is expensive.

*Core insight / definition:* FID = $||\mu_r-\mu_g||^2+\text{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$, the distance between Inception-feature distributions of real vs. generated images; lower is better.

*Why it matters / pitfall / interview trap:* Sensitive to sample size (needs 10k+ samples), and the Inception features it relies on may not generalize to all image domains.

---

**Friedman Test + Nemenyi Post-hoc**

*The problem:* Comparing more than two classifiers across multiple datasets needs a statistical test that isn't limited to pairwise comparisons.

*Core insight / definition:* The Friedman test is a rank-based, non-parametric test for comparing multiple classifiers across multiple datasets; the Nemenyi test is the post-hoc procedure used to identify which specific pairs differ significantly after a significant Friedman result.

*Why it matters / pitfall / interview trap:* Appropriate specifically for the "multiple classifiers × multiple datasets" comparison setting, distinct from the two-model paired tests (McNemar, paired t-test, Wilcoxon).

---

**Grad-CAM**

*The problem:* Neural network image classifiers need a fast, localized explanation of which image regions drove a prediction.

*Core insight / definition:* Grad-CAM is a gradient-based method for neural networks that produces a local, class-discriminative localization map (heatmap) highlighting important regions of an input image.

*Why it matters / pitfall / interview trap:* Fast but gradient-based (approximate), positioned in the interpretability spectrum as local and neural-net-specific, distinct from model-agnostic methods like SHAP/LIME.

---

**Group K-Fold**

*The problem:* When multiple records belong to the same entity (e.g., 5 records per patient), random K-fold can place some of an entity's records in train and others in validation, letting the model implicitly memorize that entity and inflating validation performance.

*Core insight / definition:* Group K-Fold ensures all records belonging to a given group (patient, user, session, region) appear in the same fold, so the model is validated on truly unseen entities. `StratifiedGroupKFold` combines this with class-balance preservation.

*Why it matters / pitfall / interview trap:* Without grouping, in-production performance on genuinely new entities will be worse than the (leaked) validation estimate suggested.

---

**Guardrail Metrics**

*The problem:* Optimizing a single north-star metric can hide serious regressions in other important dimensions (latency, spam, unsubscribes).

*Core insight / definition:* Guardrail metrics are metrics that must not regress beyond a defined threshold during an experiment, defined before launch alongside the primary metric; a guardrail violation stops the experiment or blocks shipping even if the primary metric improves.

*Why it matters / pitfall / interview trap:* "We'll watch for negative effects" is not a plan — guardrails must be pre-specified with explicit rollback triggers and ownership, turning vague monitoring into an operational procedure.

---

**Holdback Group**

*The problem:* Long-term effects of a feature (e.g., habituation, engagement decay) may not show up in a short experiment window.

*Core insight / definition:* A holdback group is a permanent group of users kept on the old experience (no crossover) used to measure long-term treatment effects that a short-duration A/B test would miss.

*Why it matters / pitfall / interview trap:* Useful when a launch to 100% of traffic prevents future clean causal comparison — proactively reserving a holdback preserves the ability to measure long-run impact.

---

**Holm-Bonferroni Correction**

*The problem:* Bonferroni correction is valid but overly conservative, losing power unnecessarily.

*Core insight / definition:* Holm-Bonferroni is a step-down procedure that also controls the family-wise error rate (FWER) but is slightly less conservative than plain Bonferroni, giving more statistical power while retaining the same FWER guarantee.

*Why it matters / pitfall / interview trap:* Choose Holm-Bonferroni over plain Bonferroni when FWER control is required but conservatism needs to be minimized; choose Benjamini-Hochberg instead if FDR control is acceptable.

---

**Huber Loss**

*The problem:* MSE over-penalizes outliers; MAE is robust but non-differentiable at zero and doesn't distinguish moderate vs. severe errors.

*Core insight / definition:* Huber loss is quadratic (MSE-like) for small errors ($|a|\leq\delta$) and linear (MAE-like) for large errors ($|a|>\delta$), combining the smoothness of MSE with the outlier-robustness of MAE. The threshold $\delta$ is a hyperparameter, typically set near the 90th–95th percentile of residuals.

*Why it matters / pitfall / interview trap:* Well suited to production systems with occasional bad labels or anomalous inputs where full MAE robustness isn't needed but MSE's outlier sensitivity is too costly.

---

**Ignorability (No Unmeasured Confounders)**

*The problem:* Every observational causal-adjustment method (PSM, IPW, doubly robust) needs some assumption to justify that the comparison is fair.

*Core insight / definition:* Ignorability is the assumption that, conditional on measured covariates $X$, treatment assignment is independent of potential outcomes — i.e., there are no unmeasured confounders.

*Why it matters / pitfall / interview trap:* This assumption can never be fully tested and fails silently — if a real confounder isn't in $X$, PSM/IPW/DR are all biased with no built-in warning; sensitivity analysis (Rosenbaum bounds, E-values) is used to gauge robustness to this risk.

---

**Instrumental Variables (IV)**

*The problem:* When you suspect unmeasured confounding — the one case propensity-based adjustment can't fix — you still need a way to recover a causal estimate.

*Core insight / definition:* IV uses an instrument $Z$ satisfying relevance ($Z$ affects $T$), exclusion ($Z$ affects $Y$ only through $T$), and independence ($Z$ is as-good-as-random w.r.t. unobserved confounders) to recover the causal effect from the part of $T$'s variation explained by $Z$ (via two-stage least squares).

*Why it matters / pitfall / interview trap:* Exclusion and independence are untestable and must be argued from domain knowledge; IV estimates a LATE (local average treatment effect) — the effect only for "compliers," not the whole population. Weak instruments (low relevance) give badly biased, high-variance estimates.

---

**Integrated Gradients**

*The problem:* Attention weights are not a principled causal attribution for neural network predictions.

*Core insight / definition:* $\text{IG}_i(x)=(x_i-x_i')\cdot\int_0^1 \frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}d\alpha$ — interpolates between a baseline input $x'$ and the actual input $x$, integrating the gradient along the path to attribute importance to each input dimension.

*Why it matters / pitfall / interview trap:* Considered more principled than raw attention visualization for explaining neural network (including transformer) predictions.

---

**Intent-Aware NDCG**

*The problem:* When a query has multiple valid interpretations (e.g., "jaguar" = car or animal), standard NDCG rewards systems that cover only the single most common intent.

*Core insight / definition:* IA-NDCG weights each result by the probability of the subtopic/intent it satisfies and penalizes redundancy within a single subtopic, rewarding coverage across multiple query intents.

*Why it matters / pitfall / interview trap:* Niche, specialized-role metric — standard NDCG can look great while systematically ignoring minority query intents.

---

**Interference / Network Effects (SUTVA Violation)**

*The problem:* Standard A/B testing assumes users are independent, but in social networks, marketplaces, and ride-sharing, treated and control users interact and contaminate each other's outcomes.

*Core insight / definition:* Interference occurs when a treated unit's outcome depends on other units' treatment status (violating the Stable Unit Treatment Value Assumption, SUTVA) — e.g., a treated driver affects a control rider's wait time. Solutions include cluster randomization, geographic/market splits, switchback testing, and holdback groups.

*Why it matters / pitfall / interview trap:* Applying naive individual-level A/B testing under interference biases the estimated effect because the "control" group is contaminated by treatment effects.

---

**Interleaving (Ranker Comparison)**

*The problem:* Comparing two ranking models via standard A/B testing on user-level CTR requires weeks because between-user variance is high.

*Core insight / definition:* Interleaving mixes items from both rankers into a single list shown to the same user (e.g., via Team-Draft Interleaving) and tracks which model's attributed items receive more clicks, removing between-user variance entirely.

*Why it matters / pitfall / interview trap:* Reaches statistical significance far faster (often ~100x) than A/B testing because it compares within the same user session rather than across independent users.

---

**Intra-List Diversity (ILD)**

*The problem:* A recommendation list can score perfectly on relevance while consisting of nearly identical items, providing little value to users seeking variety.

*Core insight / definition:* $\text{ILD}=\frac{1}{\binom k2}\sum_{i\neq j\in L} d(i,j)$, the average pairwise dissimilarity (e.g., cosine distance in embedding space) between items in the recommended list $L$.

*Why it matters / pitfall / interview trap:* Trades off against precision — high ILD can hurt relevance if dissimilar items are less relevant; typically balanced via re-ranking with maximal marginal relevance (MMR).

---

**Inverse Propensity Weighting (IPW)**

*The problem:* Comparing treated and untreated groups directly is confounded when treatment assignment correlates with covariates that also affect the outcome.

*Core insight / definition:* IPW reweights each unit by the inverse of its probability of receiving the treatment it actually got, constructing a pseudo-population in which treatment is independent of covariates: $\widehat{\text{ATE}}=\frac1n\sum_i[\frac{T_iY_i}{e(x_i)}-\frac{(1-T_i)Y_i}{1-e(x_i)}]$.

*Why it matters / pitfall / interview trap:* Propensities near 0 or 1 create enormous weights that explode variance — must clip/trim extreme weights and check overlap/positivity (regions with only treated or only control units cannot be compared by any method).

---

**Isotonic Regression Calibration**

*The problem:* Platt scaling imposes a rigid sigmoid functional form on the calibration mapping, which fails when miscalibration is non-monotone or has a complex shape.

*Core insight / definition:* Isotonic regression fits a non-parametric, non-decreasing step function from raw scores to probabilities using the Pool Adjacent Violators algorithm, requiring only monotonicity (higher scores map to higher probabilities), no parametric functional form.

*Why it matters / pitfall / interview trap:* With small calibration sets (<500 examples), isotonic regression overfits by memorizing empirical frequencies of individual bins — prefer Platt scaling when the calibration set is small.

---

**LATE (Local Average Treatment Effect)**

*The problem:* Instrumental variables don't estimate the effect for the whole population — they estimate it for a specific, often unobservable, subgroup.

*Core insight / definition:* LATE is the causal effect estimated by IV specifically for "compliers" — the units whose treatment status the instrument actually moved — not the whole population's average effect.

*Why it matters / pitfall / interview trap:* Reporting an IV estimate as if it applies to everyone overstates its generality; it's only valid for compliers, a subpopulation defined by the instrument itself.

---

**Leave-One-Out Cross-Validation**

*The problem:* With small datasets, standard K-fold might not use enough data per training fold, but full CV needs to balance bias and computational cost.

*Core insight / definition:* Leave-one-out CV is K-fold with K=n — each fold holds out exactly one example, training on all others. Near-zero bias, but very high variance due to fold correlation and n training runs required.

*Why it matters / pitfall / interview trap:* More folds does not straightforwardly mean a better estimate — LOO-CV's high variance is a classic gotcha.

---

**LIME (Local Interpretable Model-agnostic Explanations)**

*The problem:* Explaining a single black-box prediction in a human-understandable way, without needing global model transparency.

*Core insight / definition:* LIME samples N perturbed versions around an input, gets model predictions for each, weights samples by proximity to the original input, and fits a weighted local linear regression whose coefficients serve as feature importances for that one prediction.

*Why it matters / pitfall / interview trap:* Not additive (explanations for feature subsets don't combine predictably), sensitive to hyperparameters (kernel width, sample count), and stochastic — different runs on the same point/model can yield different explanations, unlike SHAP, which is deterministic.

---

**LLM-as-Judge**

*The problem:* Human evaluation of generated text is the gold standard but slow and expensive at scale.

*Core insight / definition:* LLM-as-judge uses a (typically larger/stronger) model to rate the outputs of another model, providing faster automated evaluation than human review.

*Why it matters / pitfall / interview trap:* Biased toward the judge model's own stylistic preferences — always calibrate LLM-judge scores against spot human checks.

---

**Log Loss (Cross-Entropy Loss)**

*The problem:* Accuracy treats a barely-correct prediction the same as a highly confident correct one, and doesn't punish confident wrong predictions any more than uncertain wrong ones.

*Core insight / definition:* $\text{Log Loss}=-\frac1n\sum_i[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)]$; it penalizes confident wrong predictions severely through the logarithm (predicting 0.99 for the wrong class gives loss ≈4.6, vs. ≈0.92 for a 0.6 wrong prediction), and is a strictly proper scoring rule.

*Why it matters / pitfall / interview trap:* Log loss is not scale-free — comparisons are only valid within the same problem/class-balance, not across different datasets. Minimizing expected log loss is equivalent to minimizing KL divergence from the model's distribution to the true distribution.

---

**MAE (Mean Absolute Error)**

*The problem:* Need a regression error metric that treats all deviations proportionally, without letting a few large errors dominate the score.

*Core insight / definition:* $\text{MAE}=\frac1n\sum_i|y_i-\hat y_i|$ — treats a 10-unit error as exactly 10x worse than a 1-unit error (linear cost); robust to outliers but not differentiable at 0.

*Why it matters / pitfall / interview trap:* Use MAE (median regression) when outliers are expected and shouldn't dominate the metric; use MSE/RMSE when large errors are genuinely more costly.

---

**MAPE (Mean Absolute Percentage Error) / sMAPE / MASE**

*The problem:* Stakeholders often want error expressed as a percentage for intuitive interpretation across different scales.

*Core insight / definition:* $\text{MAPE}=\frac{100}{n}\sum|\frac{y_i-\hat y_i}{y_i}|$. Alternatives: sMAPE (symmetric MAPE, still has issues) and MASE (mean absolute scaled error, normalized against an in-sample naive forecast — best for time series).

*Why it matters / pitfall / interview trap:* MAPE is undefined at $y_i=0$, asymmetric (over- and under-predictions of equal magnitude are penalized differently), penalizes under-forecasting more heavily (bad for inventory/safety use cases), and meaningless when the target can be negative.

---

**Matthews Correlation Coefficient (MCC)**

*The problem:* F1 ignores true negatives entirely and can still be misleading under extreme imbalance; a metric using all four confusion-matrix cells is needed.

*Core insight / definition:* $\text{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$, ranging [-1,1], where +1 is perfect, 0 is random, -1 is perfectly inverted; it is invariant to class-balance manipulation in a way accuracy is not.

*Why it matters / pitfall / interview trap:* A model predicting all negatives on a 99/1 split gets MCC=0 (correctly reflecting uselessness), not the misleadingly high accuracy score — but MCC is harder to explain to non-technical stakeholders than F1.

---

**MC Dropout (Monte Carlo Dropout)**

*The problem:* Estimating predictive uncertainty for a deterministic neural network normally requires an ensemble, which is expensive to train.

*Core insight / definition:* Keeping dropout active at inference time (`model.train()` during prediction) and running N stochastic forward passes approximates sampling from an approximate posterior over network weights (Gal & Ghahramani 2016); the variance across passes estimates epistemic uncertainty, and average entropy estimates aleatoric uncertainty.

*Why it matters / pitfall / interview trap:* N forward passes multiply inference latency by N; MC Dropout underestimates epistemic uncertainty relative to deep ensembles because it samples from a restricted variational posterior, not the true Bayesian posterior.

---

**McNemar's Test**

*The problem:* Comparing two classifiers' point-estimate accuracy on a shared hold-out set doesn't tell you whether the difference is statistically meaningful.

*Core insight / definition:* McNemar's test builds a 2×2 disagreement table (b = model A correct/B wrong, c = model B correct/A wrong) and computes $\chi^2=(b-c)^2/(b+c)$ to test whether the models' error patterns are systematically different.

*Why it matters / pitfall / interview trap:* Appropriate specifically for paired classifier comparisons on the same examples; report effect size alongside p<0.05, not significance alone.

---

**MDE (Minimum Detectable Effect)**

*The problem:* Before running an experiment, you need to know how large a sample is required, which depends on the smallest effect worth being able to detect.

*Core insight / definition:* MDE is the smallest improvement considered worth detecting; it directly determines the required sample size via the power-analysis formula $n=\frac{2(z_{\alpha/2}+z_\beta)^2\sigma^2}{\delta^2}$.

*Why it matters / pitfall / interview trap:* An MDE too small relative to available traffic makes an experiment infeasible in reasonable time; an MDE set too loosely misses real, valuable effects — must be computed before starting, not chosen after the fact.

---

**Mediator**

*The problem:* Practitioners often assume any variable correlated with treatment and outcome should be controlled for, without checking its causal role.

*Core insight / definition:* A mediator is a variable lying on the causal path from treatment to outcome ($T\to M\to Y$); conditioning on it removes part of the true causal effect rather than removing bias.

*Why it matters / pitfall / interview trap:* Distinguishing mediators from confounders is essential — controlling for a mediator is a mistake that a naive "throw everything into the regression" strategy will make.

---

**Metric Drift**

*The problem:* A model that was well-calibrated and performant at launch can silently degrade as the world (and its own feedback effects) change.

*Core insight / definition:* Metric drift is the phenomenon where the deployed model's data distribution shifts toward the model's own behavior (via feedback loops, self-selection, or behavioral change), causing offline-validated properties like calibration to degrade over time.

*Why it matters / pitfall / interview trap:* Requires continuous monitoring of ECE, coverage, and prediction distribution in production, with retraining alerts triggered when drift exceeds a threshold.

---

**MMR (Maximal Marginal Relevance)**

*The problem:* Optimizing pure relevance in a ranked list can produce highly redundant results.

*Core insight / definition:* MMR is a re-ranking technique applied after initial retrieval that balances relevance against diversity, used to improve intra-list diversity without abandoning relevance-based retrieval.

*Why it matters / pitfall / interview trap:* Typical production solution to the ILD/relevance trade-off — re-rank for diversity rather than optimizing diversity from scratch.

---

**Model Coefficients (Linear Interpretability)**

*The problem:* Some stakeholders need an exact, not approximate, account of how a model computes its output.

*Core insight / definition:* For linear models, the fitted coefficients themselves are an exact, global, fast interpretation of feature contribution — no post-hoc approximation is needed since the model mechanism is directly readable.

*Why it matters / pitfall / interview trap:* Positioned as the one method in the interpretation spectrum that is inherently interpretable rather than a post-hoc explanation of a black box.

---

**MPIW (Mean Prediction Interval Width)**

*The problem:* Coverage alone can be trivially satisfied by an infinitely wide interval, so interval quality needs a second axis.

*Core insight / definition:* $\text{MPIW}=\frac1n\sum(\hat u_i-\hat l_i)$, the average width of prediction intervals, measuring the efficiency/informativeness of the intervals.

*Why it matters / pitfall / interview trap:* Good interval methods maximize PICP for a given MPIW, or equivalently minimize MPIW subject to a target PICP — never judge PICP without MPIW.

---

**MRR (Mean Reciprocal Rank)**

*The problem:* Some ranking tasks (QA, spelling correction, entity lookup) care only about how quickly the first correct answer appears, not the whole ranking.

*Core insight / definition:* $\text{MRR}=\frac1Q\sum_q \frac{1}{\text{rank}_q}$, the average of the reciprocal rank of the first relevant result across queries.

*Why it matters / pitfall / interview trap:* Ignores all relevant results after the first one — inappropriate for tasks where finding multiple relevant items matters (use MAP or NDCG instead).

---

**MSE / RMSE (Mean Squared Error / Root Mean Squared Error)**

*The problem:* When large regression errors are disproportionately more costly than small ones, an error metric needs to reflect that.

*Core insight / definition:* $\text{MSE}=\frac1n\sum(y_i-\hat y_i)^2$ squares errors before averaging, making large errors quadratically more costly (a 10-unit error contributes 100, a 1-unit error contributes 1); $\text{RMSE}=\sqrt{\text{MSE}}$ returns to the original units for interpretability.

*Why it matters / pitfall / interview trap:* A model optimized for RMSE aggressively minimizes large errors, sometimes at the cost of more frequent small errors — using RMSE when outliers shouldn't dominate the metric (e.g., rare, low-stakes anomalies) is a common mistake; use MAE or Huber instead in that case.

---

**Multiple Hypothesis Correction**

*The problem:* Running many statistical tests (many metrics, many segments) raises the expected number of false positives well above the nominal α.

*Core insight / definition:* Multiple hypothesis correction adjusts the significance threshold or interprets p-values differently to account for the number of tests run; family-wise methods (Bonferroni, Holm-Bonferroni) control the probability of any false positive, while FDR methods (Benjamini-Hochberg) control the expected proportion of false positives among rejections.

*Why it matters / pitfall / interview trap:* Testing 10 metrics without correction means roughly one will show p<0.05 by chance alone — this is a very common experimentation pitfall.

---

**Multi-Armed Bandit / Thompson Sampling**

*The problem:* A fixed 50/50 A/B split wastes traffic on a worse variant for the full experiment duration, even after enough evidence accumulates to know it's worse.

*Core insight / definition:* Bandits adaptively allocate more traffic to better-performing variants while continuing to explore. Thompson Sampling is a Bayesian approach: maintain a Beta distribution per variant (posterior over conversion rate), sample from each, and route traffic to the variant with the highest sample.

*Why it matters / pitfall / interview trap:* Use bandits when the reward signal is immediate (clicks, purchases) and there's real business cost to serving a worse variant; use fixed A/B tests when the primary metric has delayed feedback (e.g., D7 retention), since bandits need fast reward signal to adapt.

---

**Nested Cross-Validation**

*The problem:* Using the same CV loop to both select hyperparameters and report a final performance estimate is optimistic — you selected the configuration that happened to score best on that particular split.

*Core insight / definition:* Nested CV separates hyperparameter tuning from performance estimation with two loops: an outer loop splits data into outer-train/outer-test for unbiased performance estimation, while an inner loop (run within each outer-train fold) tunes hyperparameters.

*Why it matters / pitfall / interview trap:* The final deployed model is retrained on all data using the best hyperparameters from a separate, non-nested tuning run — the nested CV score is only for honest performance reporting, not for producing the deployment model itself. Expensive (inner_K × outer_K × n_candidates evaluations).

---

**Novelty Effect**

*The problem:* Users may respond more positively to a new UI or feature simply because it is new, not because it is genuinely better, inflating short-term experiment results.

*Core insight / definition:* The novelty effect is a bias where users behave differently toward something simply because it's unfamiliar/new, which fades over time; mitigated by running experiments for 2+ weeks and comparing new vs. returning user cohorts.

*Why it matters / pitfall / interview trap:* Long-running experiments risk accumulating novelty effects (early positive reaction) as well as selection effects (early adopters being unrepresentative) — both must be accounted for in experiment duration planning.

---

**NDCG (Normalized Discounted Cumulative Gain)**

*The problem:* When relevance is graded (not just binary) and position in a ranked list matters (rank 1 is far more valuable than rank 10), standard precision/recall don't capture ranking quality.

*Core insight / definition:* $\text{DCG@}k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2(i+1)}$ discounts relevance by log position; $\text{NDCG@}k=\text{DCG@}k/\text{IDCG@}k$ normalizes by the DCG of the ideal (perfectly sorted) ranking, giving a [0,1] score where 1 is perfect ordering.

*Why it matters / pitfall / interview trap:* Undefined when there are no relevant documents (IDCG=0). NDCG can be high while offline relevance labels are self-fulfilling (derived from what the current system already showed), masking whether users are actually satisfied — a common trap when churn coexists with high NDCG.

---

**Optimal Threshold Under Cost**

*The problem:* Choosing 0.5 as the decision threshold implicitly assumes false positives and false negatives cost the same, which is almost never true in practice.

*Core insight / definition:* Given asymmetric costs, the optimal threshold is $\hat p(x)\geq \frac{C(FP)-C(TN)}{C(FP)-C(TN)+C(FN)-C(TP)}$ — e.g., in fraud detection with $C(FN)=\$500$ and $C(FP)=\$5$, the optimal threshold is ≈0.01.

*Why it matters / pitfall / interview trap:* The threshold is a business/operational parameter set after training based on cost structure and capacity constraints, not a model hyperparameter — and it must never be tuned on the test set, since that inflates the reported metric.

---

**Overlap / Positivity**

*The problem:* Causal comparison methods need treated and control units that are actually comparable in covariate space; without any overlap, no method can produce a valid estimate.

*Core insight / definition:* Overlap (positivity) requires that for every region of covariate space, there exist both treated and untreated units with non-degenerate probability; a violation means the propensity score is 0 or 1 in some region.

*Why it matters / pitfall / interview trap:* Poor overlap (treated and control distributions barely overlapping) leads to extrapolation and unreliable estimates — always check propensity-score distributions and report how many units were discarded due to poor matches.

---

**PDP (Partial Dependence Plot)**

*The problem:* Stakeholders want to see the marginal effect of a single feature on model predictions, averaged over all other features.

*Core insight / definition:* $\text{PDP}(x_j)=E_{X_C}[f(x_j,X_C)]=\frac1n\sum_i f(x_j,x_C^{(i)})$ — fixes feature $x_j$ to a value and averages predictions over the observed distribution of all other features $X_C$.

*Why it matters / pitfall / interview trap:* Assumes feature independence; when $x_j$ correlates with other features, averaging creates unrealistic/impossible feature combinations (extrapolation) — use ALE instead when features are correlated.

---

**Peeking**

*The problem:* Checking an experiment's results repeatedly and stopping as soon as significance appears feels natural but statistically invalidates the test.

*Core insight / definition:* Peeking is evaluating experiment results continuously and stopping whenever p < 0.05 is observed, rather than committing to a fixed sample size/duration in advance.

*Why it matters / pitfall / interview trap:* Inflates the effective false-positive rate well above the nominal 5%, sometimes to 20-30% — the fix is a fixed horizon or a proper sequential-testing (alpha-spending) method.

---

**PICP (Prediction Interval Coverage Probability)**

*The problem:* Need a way to empirically measure whether prediction intervals actually contain the true values at the promised rate.

*Core insight / definition:* $\text{PICP}=\frac1n\sum \mathbf{1}[y_i\in[\hat l_i,\hat u_i]]$, the empirical fraction of true values falling within their predicted intervals.

*Why it matters / pitfall / interview trap:* Must always be evaluated jointly with MPIW — a trivial model achieves 100% PICP with infinitely wide intervals, which is useless.

---

**Pinball Loss (Quantile Loss)**

*The problem:* Standard symmetric losses (MAE, MSE) target the conditional mean or median, but many applications need calibrated prediction intervals at arbitrary quantiles.

*Core insight / definition:* $\rho_\tau(y,\hat q)=\tau(y-\hat q)$ if $y\geq\hat q$, else $(1-\tau)(\hat q-y)$ — minimizing this loss at quantile $\tau$ produces the $\tau$-th conditional quantile; $\tau=0.5$ recovers MAE.

*Why it matters / pitfall / interview trap:* Used to build prediction intervals by fitting separate models at, e.g., $\tau=0.1$ and $\tau=0.9$ to get an 80% interval; it is a proper scoring rule for quantiles, unlike symmetric losses.

---

**Platt Scaling**

*The problem:* Model outputs like SVM decision values or raw boosting scores are not probabilities and need to be converted into trustworthy calibrated probabilities.

*Core insight / definition:* Platt scaling fits a logistic regression on top of raw model scores using a held-out calibration set: $P(Y=1)=\sigma(A\cdot f(x)+B)$, learning parameters $A,B$ that map scores to calibrated probabilities.

*Why it matters / pitfall / interview trap:* Assumes the miscalibration is monotone and roughly sigmoid-shaped; if the true miscalibration pattern is non-monotone or complex, the parametric fit will be wrong — use isotonic regression instead (with sufficient calibration data).

---

**Position Bias**

*The problem:* Offline click/relevance labels reflect where items were positioned when shown to users, not their true relevance, contaminating both training and evaluation.

*Core insight / definition:* Position bias is the tendency for items shown in higher positions to receive more clicks/engagement regardless of true relevance, which biases offline metrics like CTR when used as surrogates for online outcomes.

*Why it matters / pitfall / interview trap:* Listed as a known gap between offline CTR and real production CTR — a major reason offline and online metrics diverge.

---

**PR-AUC (Precision-Recall AUC) vs. ROC-AUC**

*The problem:* On imbalanced data, ROC-AUC can look impressive while the model is operationally nearly useless, because false-positive rate is normalized by an abundant true-negative count.

*Core insight / definition:* ROC-AUC plots TPR vs. FPR across thresholds; its value equals P(random positive ranked above random negative), with baseline 0.5 regardless of class balance. PR-AUC plots precision vs. recall across thresholds, with baseline equal to prevalence; both quantities have TP in the denominator's complement, making PR-AUC directly sensitive to positive-prediction quality.

*Why it matters / pitfall / interview trap:* With rare positives (e.g., <10% prevalence), ROC-AUC=0.95 can coexist with PR-AUC=0.20-0.30 on the same model — use PR-AUC whenever the positive class is rare and false positives are costly.

---

**Precision, Recall (Sensitivity), Specificity**

*The problem:* Different error types (false alarms vs. missed detections) have different real-world costs, and a single accuracy number can't distinguish which type of mistake a model makes.

*Core insight / definition:* Precision = TP/(TP+FP) — of predicted positives, how many were right (measures false-alarm cost). Recall/Sensitivity = TP/(TP+FN) — of actual positives, how many were caught (measures missed-positive cost). Specificity = TN/(TN+FP) — of actual negatives, how many were correctly ignored.

*Why it matters / pitfall / interview trap:* Reporting precision or recall in isolation is trivially gameable in opposite directions (predict nothing positive → perfect precision, 0 recall; predict everything positive → perfect recall, base-rate precision) — always report both together.

---

**Precision@K / Recall@K**

*The problem:* In ranking/recommendation, users mostly see the top of a list, so metrics need to focus on a fixed cutoff rather than the whole ranking.

*Core insight / definition:* $P@k$ = fraction of the top-k items that are relevant; $R@k$ = fraction of all relevant items that appear in the top k.

*Why it matters / pitfall / interview trap:* P@k is right when the corpus is large/unbounded and users only see the top k; R@k matters when all relevant items matter (e.g., legal document retrieval) — choosing the wrong one misaligns the metric with the actual user experience.

---

**Precision/Recall for Generative Models**

*The problem:* FID and Inception Score conflate "quality" and "diversity" into one number, hiding whether a generative model fails on one dimension specifically.

*Core insight / definition:* In generative-model evaluation, Precision measures sample quality (fidelity to the real distribution) and Recall measures diversity (coverage of the real distribution) via k-NN manifold estimation.

*Why it matters / pitfall / interview trap:* Requires careful k-NN radius selection — a poorly chosen radius distorts both the quality and diversity estimates.

---

**Primacy Effect**

*The problem:* Users accustomed to the old experience may initially prefer the familiar control condition regardless of the treatment's true quality, biasing early experiment results.

*Core insight / definition:* Primacy effect is the bias where control users prefer the familiar UX simply out of habit, the mirror image of the novelty effect.

*Why it matters / pitfall / interview trap:* Mitigated the same way as the novelty effect (running longer, examining new vs. returning users).

---

**Prior/Base Rate — Threshold-Setting for Alerts** — see Optimal Threshold Under Cost, Class Imbalance

---

**Proper Scoring Rule**

*The problem:* Not every loss function that "measures error" actually incentivizes a model to report its true beliefs about probability.

*Core insight / definition:* A scoring rule $S(\hat p,y)$ is proper if its expected value is maximized (or minimized) only when $\hat p=P(y)$; it's strictly proper if that optimum is unique. Log loss and Brier score are strictly proper; 0-1 loss is not proper.

*Why it matters / pitfall / interview trap:* Optimizing a proper scoring rule incentivizes calibrated probability outputs, not just correct hard decisions; improper rules (accuracy, hinge loss) can be minimized by confidently wrong models — this is why log loss (cross-entropy) is the standard neural-network training objective.

---

**Propensity Score**

*The problem:* Directly matching or adjusting on a high-dimensional covariate vector $X$ to remove confounding is difficult or infeasible.

*Core insight / definition:* The propensity score $e(x)=P(T=1|X=x)$ is the probability of receiving treatment given covariates, typically modeled via logistic regression or gradient boosting; by the Rosenbaum-Rubin result, conditioning on this single scalar is sufficient to remove confounding from all of $X$.

*Why it matters / pitfall / interview trap:* Underlies PSM, IPW, and doubly robust estimation — all inherit the ignorability assumption (no unmeasured confounders), which is untestable.

---

**Propensity Score Matching (PSM)**

*The problem:* You want an intuitive, transparent way to compare treated and untreated units that "look similar" on measured covariates.

*Core insight / definition:* PSM pairs each treated unit with a control unit of similar propensity score and compares outcomes within matched pairs.

*Why it matters / pitfall / interview trap:* Intuitive and transparent but discards unmatched units and is sensitive to the matching caliper (bandwidth) chosen — results that swing with the caliper indicate lack of robustness.

---

**Regression Discontinuity (RDD)**

*The problem:* When treatment is assigned by a threshold on a continuous variable (credit score cutoff, test score for a scholarship), you want a near-experimental causal estimate without randomizing.

*Core insight / definition:* RDD compares outcomes for units just above vs. just below the cutoff, since they are essentially identical on everything except treatment status; effect = $\lim_{x\downarrow c}\mathbb{E}[Y|X=x]-\lim_{x\uparrow c}\mathbb{E}[Y|X=x]$.

*Why it matters / pitfall / interview trap:* Relies on the assumption that units can't precisely manipulate the running variable to land on their preferred side (tested via a McCrary density check for suspicious pile-ups); only identifies a local effect at the cutoff, and data near the threshold is often sparse.

---

**Reliability Diagram**

*The problem:* A single calibration scalar (ECE, Brier) can obscure where in the probability range a model is over- or under-confident.

*Core insight / definition:* A reliability diagram bins predictions by predicted probability and plots mean predicted probability against the empirical fraction of positives per bin; points on the diagonal indicate perfect calibration, below the diagonal indicates overconfidence, above indicates underconfidence.

*Why it matters / pitfall / interview trap:* Sensitive to bin count and strategy (uniform vs. quantile) — the same model can look differently calibrated with 5 vs. 20 bins, and sparse extreme bins produce noisy, unreliable points that should be treated skeptically. Always report binning parameters used.

---

**Repeated K-Fold**

*The problem:* With small datasets, a single K-fold arrangement's estimate has high variance — a different random seed can give a noticeably different result.

*Core insight / definition:* Repeated K-Fold runs K-fold CV multiple times with different random shuffles (n_repeats), computing the mean and standard deviation over all (n_repeats × K) fold scores to further reduce variance.

*Why it matters / pitfall / interview trap:* Fold scores across repeats are not independent (they share training samples), so the reported standard deviation underestimates true uncertainty — treat it as a diagnostic, not a rigorous confidence interval.

---

**Reweighting / Class Weights** — see Class Imbalance

---

**R² (Coefficient of Determination)**

*The problem:* Stakeholders want a single number expressing what fraction of outcome variance a regression model explains.

*Core insight / definition:* $R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ — fraction of variance explained; $R^2=1$ is perfect, $R^2=0$ matches a constant (mean) baseline, and $R^2$ can be negative for a model worse than predicting the mean.

*Why it matters / pitfall / interview trap:* Adding features always increases raw $R^2$ even when useless (use adjusted $R^2$ instead); high $R^2$ does not guarantee a good model (Anscombe's quartet is the canonical counterexample) — always inspect residuals, not just $R^2$.

---

**Sample Ratio Mismatch (SRM)**

*The problem:* If experiment assignment logic has a bug, the observed traffic split may not match the intended split, silently invalidating the entire analysis.

*Core insight / definition:* SRM is detected via a chi-squared goodness-of-fit test comparing observed variant counts against expected ratios; a significant deviation (e.g., 48/52 instead of 50/50) signals an assignment bug, not a real effect.

*Why it matters / pitfall / interview trap:* SRM is a "stop the experiment" signal, not just a metric to note — any treatment effect measured on top of a biased population is confounded and untrustworthy; SRM must always be checked before analyzing results.

---

**Sample Size / Power Analysis**

*The problem:* Running an experiment without first computing the required sample size risks either wasting resources on an over-powered test or failing to detect a real effect with an under-powered one.

*Core insight / definition:* The sample size formula for a two-sample proportions/means test is $n=\frac{2(z_{\alpha/2}+z_\beta)^2\sigma^2}{\delta^2}$ per group, where $\delta$ is the MDE and $\sigma^2$ is the metric's variance; default $\alpha=0.05$, power $(1-\beta)=0.80$.

*Why it matters / pitfall / interview trap:* Most teams run experiments too small to detect real effects — a 1% improvement can require millions of users, while a 10K-user experiment can only reliably detect >10% improvements.

---

**Sensitivity Analysis (Rosenbaum Bounds / E-value)**

*The problem:* Since ignorability (no unmeasured confounders) can never be directly verified, analysts need a way to gauge how fragile a causal estimate is to a hidden confounder.

*Core insight / definition:* Sensitivity analysis (e.g., Rosenbaum bounds, E-values) quantifies how strong a hidden confounder would need to be — in terms of its association with both treatment and outcome — to fully explain away an observed effect.

*Why it matters / pitfall / interview trap:* A propensity-matched positive result should always be accompanied by a sensitivity analysis; a result that could be explained away by a weak, plausible confounder is far less credible than one requiring an implausibly strong confounder.

---

**Serendipity (Recommendation Metric)**

*The problem:* Relevance metrics reward recommending items users are known to want, but don't reward introducing genuinely useful unexpected discoveries.

*Core insight / definition:* $\text{Serendipity}=\frac1{|U|}\sum_u\frac{|\text{relevant}\cap\text{unexpected}|}{k}$, where "unexpected" means not what a naive baseline (e.g., a popularity model) would have recommended.

*Why it matters / pitfall / interview trap:* Captures a dimension of recommendation quality (novel-but-relevant discovery) invisible to standard precision/recall/NDCG metrics.

---

**Shadow Mode**

*The problem:* Deploying a new model directly to real users risks exposing pathological outputs before they're caught.

*Core insight / definition:* Shadow mode runs the new model in parallel with the production model, logging its predictions without serving them to users, allowing comparison of score distributions and detection of pathological outputs before any real exposure.

*Why it matters / pitfall / interview trap:* Costs almost nothing operationally and catches distribution anomalies before they reach users — skipping straight from offline evaluation to canary/full rollout forfeits this cheap safety check.

---

**Shapley Value / SHAP (SHapley Additive exPlanations)**

*The problem:* Practitioners need a theoretically principled way to attribute a model's prediction to individual input features, especially for a single prediction (not just globally).

*Core insight / definition:* $\phi_i=\sum_{S\subseteq F\setminus\{i\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}[v(S\cup\{i\})-v(S)]$ — the Shapley value of feature $i$ is its average marginal contribution across all possible feature-subset orderings. SHAP is the unique attribution method satisfying four axioms simultaneously: Efficiency, Symmetry, Dummy, and Linearity/Additivity.

*Why it matters / pitfall / interview trap:* SHAP is not causal — it reports what the model used, not what causes the outcome (a model using ZIP code as a race proxy produces an "honest" attribution to ZIP code that says nothing about real causal structure). With correlated features, SHAP must marginalize missing features (typically by sampling from the marginal distribution), which constructs impossible off-manifold inputs and makes attributions partly dependent on the imputation scheme rather than purely on the model.

---

**Sliding Window Validation**

*The problem:* Expanding-window time-series CV keeps growing the training set, which doesn't simulate a production scenario where models are periodically retrained on a fixed lookback period.

*Core insight / definition:* Sliding window validation keeps the training window at a fixed size while moving both the training and validation windows forward through time, simulating fixed-lookback periodic retraining.

*Why it matters / pitfall / interview trap:* Distinct from the expanding-window (TimeSeriesSplit) approach — choice depends on whether production retraining uses all history or a fixed recent window.

---

**Simpson's Paradox**

*The problem:* A metric can appear to improve within every subgroup yet worsen overall, misleading decision-makers who only look at the aggregate number.

*Core insight / definition:* Simpson's Paradox occurs when segment sizes shift such that a metric improves in every individual segment but worsens in aggregate — e.g., model A beats model B on both mobile and desktop separately but loses overall because it was evaluated on a harder mix of users.

*Why it matters / pitfall / interview trap:* Defense is to always stratify metrics by subgroup and report both weighted and unweighted aggregates — trusting only the aggregate number can lead to shipping a genuinely worse model.

---

**Stratified K-Fold**

*The problem:* With imbalanced classes, random fold assignment can produce folds with wildly different class ratios than the full dataset by chance, adding spurious variance to the CV estimate.

*Core insight / definition:* Stratified K-Fold distributes samples from each class proportionally across all K folds so that every fold's class distribution matches the overall dataset's distribution.

*Why it matters / pitfall / interview trap:* With extreme imbalance (e.g., 0.1% positives) and a large K, stratification still can't guarantee every fold has any minority-class examples at all.

---

**Switchback Testing**

*The problem:* In marketplace/platform settings where individual users interact and contaminate each other (interference), user-level randomization doesn't cleanly isolate treatment effects, but stopping all randomization loses causal inference entirely.

*Core insight / definition:* Switchback testing alternates the entire system between control and treatment over fixed time periods (e.g., every 30 minutes), avoiding the need for user-level tracking while still enabling causal comparison.

*Why it matters / pitfall / interview trap:* Appropriate for time-varying interference-prone settings (marketplace/platform experiments) where per-user assignment isn't feasible due to interaction effects.

---

**Symmetry (Shapley Axiom)**

*The problem:* An attribution method should give equal credit to features that contribute equally, or its outputs are arbitrary and untrustworthy.

*Core insight / definition:* Symmetry is the Shapley axiom guaranteeing that two features contributing identically to every possible coalition receive identical attribution values.

*Why it matters / pitfall / interview trap:* One of the four axioms that together uniquely determine the Shapley value as the only fair attribution scheme.

---

**SUTVA (Stable Unit Treatment Value Assumption)** — see Interference / Network Effects

---

**Temperature Scaling**

*The problem:* Deep neural networks trained with modern techniques (batch norm, weight decay, deep architectures) are systematically overconfident in their softmax outputs.

*Core insight / definition:* Temperature scaling divides all logits by a single learned scalar $T$ before the softmax, flattening the probability distribution without changing the argmax/predictions: $\hat q_i=\max_k\frac{\exp(z_k/T)}{\sum_j\exp(z_j/T)}$. $T$ is fit by minimizing NLL on a held-out calibration set; $T>1$ softens (reduces overconfidence).

*Why it matters / pitfall / interview trap:* A single scalar cannot fix class-specific or region-specific miscalibration — if the reliability diagram is non-monotonic or miscalibration varies by subgroup, one $T$ can't address it; vector or matrix scaling generalize at the cost of more parameters. Also, fixing calibration says nothing about whether the underlying ranking or chosen threshold serves the actual cost structure.

---

**Time Series Cross-Validation (Expanding Window / TimeSeriesSplit)**

*The problem:* Shuffling time-ordered data before splitting into folds trains a model on future data to predict the past — a form of leakage that produces a meaningless performance estimate.

*Core insight / definition:* Time series CV enforces that training data always precedes validation data; the expanding-window approach (TimeSeriesSplit) trains on all data up to a cutoff and validates on the window immediately after, growing the training window with each fold.

*Why it matters / pitfall / interview trap:* Even with correct time ordering, rolling-window features (e.g., 7-day averages) can still leak future information into training if they extend past the validation boundary — requires a gap parameter at least as long as the longest feature lookback window.

---

**TreeSHAP**

*The problem:* Naive Shapley value computation is O(2^d) — intractable for models with many features.

*Core insight / definition:* TreeSHAP exploits tree structure to compute exact Shapley values in O(T×L×D) time (T=trees, L=leaves, D=depth), making exact SHAP feasible for tree-based models (e.g., XGBoost).

*Why it matters / pitfall / interview trap:* Exact and fast (milliseconds per sample) but limited to tree models — for other model types (neural nets, SVMs), KernelSHAP is needed instead, at much higher computational cost.

---

**KernelSHAP**

*The problem:* Models without a tree-specific structure (neural networks, SVMs) still need SHAP-style explanations, but TreeSHAP's exact algorithm doesn't apply.

*Core insight / definition:* KernelSHAP is model-agnostic: it samples random feature-subset coalitions, evaluates the model output with in-coalition features at actual values and others at baseline, and fits a specially weighted linear regression (weights emphasizing subsets near the empty set and the full set) to approximate Shapley values.

*Why it matters / pitfall / interview trap:* Approximate and much slower (seconds per sample) than TreeSHAP — practical use requires trading off sample count against runtime.

---

**Two-Stage Least Squares (2SLS)**

*The problem:* Instrumental variables need a concrete estimation procedure to translate the instrument's exogenous variation into a causal effect estimate.

*Core insight / definition:* 2SLS is the standard IV estimation technique: first regress treatment $T$ on instrument $Z$ (and covariates) to get predicted treatment $\hat T$, then regress outcome $Y$ on $\hat T$ to recover the causal effect using only the instrument-driven variation in treatment.

*Why it matters / pitfall / interview trap:* The strength of the first stage (relevance) is directly testable; a weak first stage produces badly biased, high-variance IV estimates even if exclusion and independence hold.

---

**Wilcoxon Signed-Rank Test**

*The problem:* A paired t-test assumes normally distributed differences between two models' per-example losses, which may not hold, especially with small n.

*Core insight / definition:* The Wilcoxon signed-rank test is a non-parametric alternative to the paired t-test for comparing two regression models' paired per-example errors, without assuming normality.

*Why it matters / pitfall / interview trap:* Preferred over the paired t-test specifically when normality of differences is questionable or the sample size is small.

---

**Win Rate (Pairwise Human Preference)**

*The problem:* Automated generative-model metrics (BLEU, FID, BERTScore) are all proxies that can be gamed or fail to track true quality.

*Core insight / definition:* Win rate is the percentage of pairwise human preference judgments favoring one model's output over another's — treated as ground truth for generative model quality.

*Why it matters / pitfall / interview trap:* Expensive and noisy to collect at scale, but remains the most reliable quality signal — for ranking model quality in practice, human win rate or preference evaluation is the only fully trustworthy signal.

---

**Youden Index**

*The problem:* Practitioners need a principled way to pick a classification threshold when they believe sensitivity and specificity should be weighted equally.

*Core insight / definition:* The Youden index $J=\text{Sensitivity}+\text{Specificity}-1$ identifies the threshold that maximizes the sum of sensitivity and specificity.

*Why it matters / pitfall / interview trap:* Only sensible when sensitivity and specificity genuinely have equal value to the business — a claim that requires explicit justification, not a default choice.

