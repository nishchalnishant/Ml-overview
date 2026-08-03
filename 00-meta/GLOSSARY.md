---
module: Meta
topic: Glossary
subtopic: ""
status: in-progress
tags: [glossary, ml, reference, all-folders]
---
# Repository-Wide ML Glossary

Comprehensive glossary of key terms drawn from every folder in this repo, built folder by folder (in numeric folder order). Each entry: the problem the term solves, the core definition/insight, and why it matters (pitfall/interview trap). This file is distinct from `18-resources/02-ml-glossary.md` (a separate, standalone A-Z glossary) — no merging between the two.

**Progress:** 01-math-foundations done. Remaining: 02-data through 18-resources.

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
