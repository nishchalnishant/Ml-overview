---
module: Math Foundations
topic: Cheatsheet Comparison
subtopic: ""
status: unread
tags: [foundations, cheatsheet, comparison, revision]
---
# Foundations — Comparison Cheat Sheet

Rapid pre-interview review. Every entry: what it is → pros → cons → when to pick over alternatives → key formula/intuition. Sections mirror the folder's own files (01–06 + flashcards). Comparison tables close out sections with competing techniques.

---

## Core ML Paradigms & Failure Modes

**Statistical ML (linear models, trees, SVMs)**
- What: Learn explicit decision boundaries from labeled data with constrained hypothesis classes.
- Pros: Interpretable, data-efficient, cheap to train/serve.
- Cons: Limited expressiveness; needs manual feature engineering.
- Pick over DL when: data is tabular/small, interpretability matters, latency budget is tight.

**Deep Learning**
- What: Stacked parameterized transformations; features emerge via backprop instead of being hand-crafted.
- Pros: Learns hierarchical representations; state of the art on unstructured data (image/text/audio).
- Cons: Data- and compute-hungry, low interpretability, many failure modes (vanishing/exploding gradients).
- Pick over statistical ML when: data is large/unstructured, feature engineering is intractable.

**Generative AI (GANs/VAEs/Diffusion/Autoregressive)**
- What: Models $P(x)$ or $P(x|c)$ directly; can sample new data.
- Pros: Enables synthesis, data augmentation, creative applications.
- Cons: Training instability (GANs), slow sampling (diffusion), evaluation is hard.
- Pick over discriminative models when: the task is generation/sampling, not just prediction.

**Reinforcement Learning**
- What: Learn actions maximizing cumulative reward via environment interaction, not labels.
- Pros: Handles sequential decision-making, delayed reward, no labeled dataset needed.
- Cons: Sample-inefficient, unstable training, reward design is hard (reward hacking).
- Pick over supervised learning when: the problem is inherently sequential/interactive (control, RLHF, games).

### The Five Failure Modes

**Overfitting (high variance)**
- What: Model memorizes training data; low train error, high val error.
- Fixes: more data (best), L1/L2/dropout, simpler model, early stopping, augmentation.
- Signature: large train/val gap.

**Underfitting (high bias)**
- What: Model too simple to capture the pattern; both train and val error high.
- Fixes: more capacity, better features, less regularization, boosting.
- Signature: train and val error both high and similar.

**Class imbalance**
- What: Majority class dominates; accuracy becomes meaningless.
- Fixes: PR-AUC/F1/MCC instead of accuracy, SMOTE/undersampling, class weights, focal loss, threshold tuning.
- Signature: naive "always predict majority" model scores deceptively high accuracy.

**Vanishing gradients**
- What: Chain-rule product of per-layer gradients <1 shrinks to ~0 in deep nets, early layers stop learning.
- Fixes: ReLU/GELU, residual connections, BatchNorm/LayerNorm, He init.
- Signature: early-layer weights barely move; sigmoid/tanh-heavy deep nets.

**Exploding gradients**
- What: Chain-rule product >1 grows exponentially; NaN losses, oscillating weights.
- Fixes: gradient clipping, lower LR, normalization, LSTM gating.
- Signature: loss suddenly spikes to NaN/inf (common in RNNs).

| Failure mode | Signature | Primary fix | Avoid confusing with |
|---|---|---|---|
| Overfitting | train≪val error | more data / regularization | underfitting (both errors high) |
| Underfitting | train≈val, both high | more capacity | overfitting |
| Class imbalance | high accuracy, low recall on minority | PR-AUC + resampling | genuine strong model |
| Vanishing gradient | early layers frozen | ReLU/ResNet/norm | slow LR / bad init |
| Exploding gradient | NaN loss, oscillation | grad clipping | numerical overflow elsewhere |

---

## Bias-Variance, Regularization, Validation

**Bias-variance tradeoff**
- What: `Expected Error = Bias² + Variance + σ²` (irreducible noise).
- Pros of understanding it: diagnoses whether to add capacity or regularize.
- Cons: decomposition is theoretical — real models trade off nonlinearly (see double descent).
- Formula: increasing complexity ↓ bias, ↑ variance; ensembling (bagging) can lower variance without raising bias.

**L1 (Lasso)**
- What: `Loss + λΣ|w|`. Diamond-shaped constraint region → corners on axes.
- Pros: Produces sparse weights — implicit feature selection; good for high-dimensional/many-irrelevant-feature settings.
- Cons: Unstable selection under correlated features (arbitrarily picks one); non-differentiable at 0.
- Pick over L2 when: you want automatic feature selection / sparse solutions.
- Intuition: Laplace prior on weights (MAP view).

**L2 (Ridge)**
- What: `Loss + λΣw²`. Spherical constraint region → smooth shrinkage, no exact zeros.
- Pros: Stable with correlated/multicollinear features; smooth, always differentiable.
- Cons: Doesn't perform feature selection; all features retained (shrunk).
- Pick over L1 when: features are correlated or you want stability, not sparsity.
- Intuition: Gaussian prior on weights (MAP view).

**Elastic Net**
- What: `Loss + α₁Σ|w| + α₂Σw²` — combines L1 + L2.
- Pros: Sparsity of L1 + stability of L2 on correlated features.
- Cons: Two hyperparameters to tune instead of one.
- Pick over pure L1/L2 when: features are both numerous and correlated.

| Regularizer | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| L1 | sparse/high-dim features, feature selection | strongly correlated features (unstable picks) | sparsity vs. stability |
| L2 | correlated features, general stability | you need automatic feature selection | smooth shrinkage vs. no sparsity |
| Elastic Net | correlated + high-dim together | simplicity/fewer hyperparams desired | best of both vs. extra tuning cost |

**Train/Val/Test split**
- What: Train (60-80%) learns params; Val (10-20%) tunes hyperparams; Test (10-20%) touched once for honest estimate.
- Cons if violated: tuning on test set → optimistic bias; shuffling time series → temporal leakage.
- Rule: stratify for imbalance, split chronologically for time series.

**K-Fold Cross-Validation**
- What: K non-overlapping splits, train on K-1, validate on 1, rotate, average.
- Pros: Lower-variance generalization estimate than single split; uses all data for both train and val across folds.
- Cons: K× training cost; naive K-fold leaks time-series data from future into training.
- Pick over single split when: dataset is small and split-to-split variance would otherwise be high.

**Stratified K-Fold**
- What: Each fold preserves original class distribution.
- Pick over plain K-fold when: classes are imbalanced.

**Time Series Split (walk-forward)**
- What: Validation folds always chronologically after training folds.
- Pick over K-fold when: data has temporal structure — never shuffle.

**Group K-Fold**
- What: Ensures examples from the same group (e.g., same user/patient) never split across train/val.
- Pick over plain K-fold when: leakage could occur via grouped/repeated entities.

**Nested CV**
- What: Inner loop tunes hyperparameters, outer loop estimates generalization — avoids tuning bias contaminating the performance estimate.
- Cons: Expensive (K_outer × K_inner trainings).
- Pick over flat CV when: reporting a publication/benchmark-grade unbiased performance number.

**Leave-One-Out (LOO)**
- What: K = N.
- Pros: Maximum data usage per fold.
- Cons: Very expensive; high-variance estimates on noisy problems.
- Pick over K-fold when: dataset is tiny and every example counts.

| Validation strategy | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Random K-fold | i.i.d. tabular data | imbalance, time series, grouped entities | simplicity vs. leakage risk |
| Stratified K-fold | classification w/ imbalance | regression, non-class targets | class balance preserved vs. no temporal awareness |
| Time-series split | sequential/temporal data | i.i.d. data (wastes structure) | realism vs. less training data per fold |
| Group K-fold | repeated/grouped entities | no natural grouping exists | prevents entity leakage vs. fewer effective folds |
| Nested CV | unbiased hyperparameter+performance reporting | limited compute/time | rigor vs. cost |
| LOO | very small N | large N (expensive), noisy data (high variance) | max data use vs. compute + variance |

---

## Ensemble Methods

**Bagging (Bootstrap Aggregating)**
- What: Train models independently in parallel on bootstrap resamples; average predictions.
- Pros: Reduces variance; hard to overfit; robust default.
- Cons: Doesn't reduce bias; less peak accuracy than tuned boosting.
- Pick over boosting when: you want a robust baseline with minimal tuning. Ex: Random Forest.

**Boosting**
- What: Train sequentially; each model corrects the ensemble's current residual errors.
- Pros: Reduces bias; typically higher peak accuracy.
- Cons: More hyperparameters, sensitive to tuning, can overfit noisy labels.
- Pick over bagging when: you have time to tune and need max performance. Ex: XGBoost, LightGBM.

| Ensemble type | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Bagging (Random Forest) | robust baseline, minimal tuning, noisy labels | need absolute max accuracy | variance reduction vs. bias unchanged |
| Boosting (XGBoost/LightGBM) | max performance, tuning time available | noisy labels + low tuning budget | bias reduction vs. overfit/tuning sensitivity risk |

---

## Algorithm Selection & Classification Metrics

**Algorithm selection ladder**
- Regression: Linear Regression → Random Forest → XGBoost.
- Classification: Logistic Regression → Random Forest → XGBoost.
- Need interpretability: Linear/Logistic Regression, Decision Trees.
- High-dimensional: Lasso, Ridge, Random Forest.
- Fastest on large tabular: LightGBM.

**Precision**
- What: `TP/(TP+FP)` — of predicted positives, fraction real.
- Pick over recall when: false positives are costly (spam filter blocking legit mail).

**Recall**
- What: `TP/(TP+FN)` — of actual positives, fraction caught.
- Pick over precision when: false negatives are costly (cancer screening).

**F1 Score**
- What: Harmonic mean of precision and recall.
- Pick over precision/recall alone when: you need one balanced number and neither error type dominates in cost.

**ROC-AUC**
- What: Threshold-invariant summary of TPR vs FPR.
- Cons: Misleadingly optimistic on imbalanced data (large true-negative pool inflates denominator of FPR).
- Pick over PR-AUC when: classes are roughly balanced.

**PR-AUC**
- What: Threshold-invariant summary of precision vs recall.
- Pick over ROC-AUC when: classes are imbalanced (e.g., fraud, rare disease).

**Accuracy**
- Cons: Meaningless under class imbalance (majority-predict baseline scores near 100%).
- Pick over other metrics when: classes are balanced and error costs are symmetric.

| Metric | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Precision | FP costly (spam) | FN costly | catches fewer positives to avoid false alarms |
| Recall | FN costly (cancer screen) | FP costly | catches more positives at cost of false alarms |
| F1 | balanced cost of FP/FN | one error type is far costlier | single number vs. loses precision/recall detail |
| ROC-AUC | balanced classes | imbalanced classes | overall ranking quality vs. optimistic under imbalance |
| PR-AUC | imbalanced classes | balanced classes (less standard) | focuses on minority class vs. less familiar to stakeholders |
| Accuracy | balanced classes, symmetric costs | imbalanced classes | simplicity vs. misleading under skew |

---

## Unsupervised Learning

**K-Means**
- What: Partitions data into K spherical clusters by minimizing within-cluster variance.
- Pros: Fast, simple, scalable.
- Cons: Requires K in advance; assumes spherical/equal-size clusters; sensitive to outliers; random init can trap in poor local minima (use k-means++).
- Pick over DBSCAN/GMM when: clusters are roughly spherical/equal-sized and K is known or estimable (Elbow/Silhouette).

**DBSCAN**
- What: Density-based clustering; groups dense regions, marks sparse points as noise/outliers.
- Pros: No need to specify K; finds arbitrary-shaped clusters; built-in outlier detection.
- Cons: Struggles with varying density clusters; sensitive to eps/min_samples params.
- Pick over K-Means when: clusters are non-spherical or outlier detection is needed.

**GMM (Gaussian Mixture Model)**
- What: Probabilistic clustering — soft assignment via mixture of Gaussians.
- Pros: Gives cluster membership probabilities; models elliptical clusters (covariance).
- Cons: Assumes Gaussian components; sensitive to initialization; more expensive than K-Means.
- Pick over K-Means when: you need soft/probabilistic assignments or elliptical cluster shapes.

| Clustering method | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| K-Means | spherical, equal-sized clusters, speed | unknown cluster count sensitivity, non-spherical shapes | simple/fast vs. rigid assumptions |
| DBSCAN | arbitrary shapes, outlier detection | varying density clusters | flexible shape vs. param sensitivity |
| GMM | soft/probabilistic membership, elliptical clusters | need for hard fast clustering at scale | probabilistic richness vs. compute + Gaussian assumption |

**PCA**
- What: Linear; finds orthogonal axes of maximum variance, projects to lower dimension.
- Pros: Deterministic, fast, preserves global variance structure, reversible (approximately).
- Cons: Linear only; sensitive to feature scale (standardize first); components can be hard to interpret.
- Pick over t-SNE/UMAP when: you need a linear, reproducible, globally-variance-preserving reduction (e.g., preprocessing before a downstream linear model).

**t-SNE**
- What: Non-linear; preserves local neighborhood structure for visualization.
- Pros: Excellent for visualizing cluster structure in 2D/3D.
- Cons: Does not preserve global distances (cluster positions/sizes are arbitrary); slow; not suited as a general preprocessing step.
- Pick over PCA/UMAP when: the goal is purely visual exploration of local cluster structure.

**UMAP**
- What: Non-linear; faster than t-SNE, better preserves both local and global structure.
- Pros: Faster, more scalable, better for downstream ML tasks than t-SNE.
- Cons: Less mathematically interpretable; hyperparameter-sensitive (n_neighbors, min_dist).
- Pick over t-SNE when: you need speed, global structure preservation, or downstream-task-usable embeddings.

**LDA (Linear Discriminant Analysis)**
- What: Supervised dimensionality reduction; maximizes class separability rather than variance.
- Pros: Uses label information, often better for classification pipelines than PCA.
- Cons: Requires labels; assumes classes are Gaussian with shared covariance; limited to (C-1) components for C classes.
- Pick over PCA when: the downstream task is classification and labels are available.

| Dim. reduction | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| PCA | linear preprocessing, global variance, speed | non-linear structure, need visualization of clusters | interpretability/speed vs. linear-only |
| t-SNE | 2D/3D visualization of local clusters | preprocessing for models, need global distances | great visuals vs. arbitrary global layout, slow |
| UMAP | scalable viz + downstream embeddings | need deterministic/simple linear method | speed + global structure vs. less interpretable |
| LDA | supervised reduction for classification | no labels available, non-Gaussian classes | class separability vs. requires labels + assumptions |

---

## Deep Learning Basics

**Activation functions**

*ReLU* — `max(0,x)`. Pros: cheap, gradient=1 in active region (no vanishing in positive domain), default for hidden layers. Cons: "dying ReLU" (neurons stuck at 0 forever). Pick as default hidden-layer activation.

*Sigmoid* — `1/(1+e⁻ˣ)`. Pros: squashes to (0,1), useful for binary output probabilities. Cons: saturates at extremes → vanishing gradients in deep stacks. Pick only for binary output layer, not hidden layers.

*Softmax* — `eˣⁱ/Σeˣʲ`. Pros: converts scores to a valid probability distribution. Cons: not used in hidden layers; sensitive to large logits (needs the `√d_k` type scaling in attention). Pick for multi-class output layer.

*GELU* — `x·Φ(x)`. Pros: smooth approximation of ReLU, empirically stronger in Transformers. Cons: more expensive to compute than ReLU. Pick for Transformer architectures.

| Activation | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| ReLU | hidden layers, general default | need smooth gradient everywhere | speed/simplicity vs. dying neurons |
| Sigmoid | binary output layer | deep hidden stacks | interpretable probability vs. vanishing gradient |
| Softmax | multi-class output layer | hidden layers | valid probability dist vs. saturation risk on large logits |
| GELU | Transformer hidden layers | compute-constrained simple nets | smoother/better empirical performance vs. extra compute |

**Normalization**

*Batch Normalization* — normalizes over the batch dimension. Pros: stabilizes/accelerates CNN training. Cons: batch-size dependent, unstable at small batch sizes, awkward with RNNs/Transformers. Pick over LayerNorm for CNNs with large batches.

*Layer Normalization* — normalizes over the feature dimension. Pros: batch-size independent, stable for Transformers/small batches. Cons: slightly different normalization statistics than BatchNorm, not always optimal for CNNs. Pick over BatchNorm for Transformers and variable/small batch sizes.

| Normalization | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| BatchNorm | CNNs, large batch sizes | small batches, RNNs/Transformers | strong regularizing effect vs. batch-size dependence |
| LayerNorm | Transformers, small/variable batches | needs cross-sample batch statistics | consistent per-example stats vs. less used in CNNs |

**Optimizers**

*SGD + Momentum* — accumulates gradient history (velocity) to escape local minima and smooth noise. Pros: simple, generalizes well, less memory than Adam. Cons: sensitive to LR choice, slower convergence. Pick when: final-stage fine-tuning / when generalization > raw convergence speed matters (e.g., vision models).

*Adam* — adaptive per-parameter LR, divides update by RMS of recent gradients. Default LR ~3e-4. Pros: fast convergence, robust to LR choice, handles sparse gradients well. Cons: can generalize worse than SGD; more memory (stores 1st+2nd moment per param). Pick as default for most DL training, especially early experimentation.

*AdamW* — decouples weight decay from adaptive scaling. Pros: correct regularization behavior vs. Adam+L2; preferred for Transformers. Cons: same memory overhead as Adam. Pick over Adam whenever weight decay is used (virtually all modern Transformer training).

| Optimizer | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| SGD+Momentum | vision models, final fine-tuning, best generalization | need fast/robust convergence with less tuning | better generalization vs. more LR sensitivity |
| Adam | fast default convergence, sparse gradients | want best possible generalization / memory constrained | speed/robustness vs. memory + weaker regularization |
| AdamW | Transformers, any setup using weight decay | memory-constrained edge deployment training | correct decoupled decay vs. same memory cost as Adam |

**Gradient descent variants**
- Batch GD: full dataset per step — exact but slow. Pick for small datasets fitting in memory.
- SGD: one sample per step — noisy but fast, can escape shallow minima. Pick for online/streaming learning.
- Mini-batch SGD (32–512): GPU-efficient balance. Pick as the default for virtually all DL training.

---

## Architectures

**The Transformer**
- What: Replaces recurrence with parallel self-attention: `softmax(QKᵀ/√d_k)V`.
- Pros: Fully parallelizable (no sequential bottleneck like RNNs); captures long-range dependencies directly.
- Cons: O(N²) memory/compute in sequence length.
- Pick over RNN/LSTM when: parallelism and long-range dependency modeling matter more than strict linear-time compute (almost always today).
- Key intuition: `√d_k` scaling prevents large dot products in high dimensions from collapsing softmax gradients to zero.

**RAG (Retrieval-Augmented Generation)**
- What: Retrieve relevant docs from external DB at inference, inject into context.
- Pros: Grounds answers in evidence, reduces hallucination, accesses private/recent data without retraining.
- Cons: Adds retrieval latency; answer quality bounded by retrieval quality (garbage in, garbage out); needs a maintained index.
- Pick over pure fine-tuning when: knowledge is large, changes frequently, or must be attributable/private.

**LoRA (Low-Rank Adaptation)**
- What: Freeze base weights `W`; add trainable low-rank `ΔW = A·B` (r ≪ d).
- Pros: ~10,000× fewer trainable params than full fine-tuning; mergeable into `W` at inference for zero added latency.
- Cons: Slightly less flexible than full fine-tuning for tasks far from pretraining distribution; rank r is a tuning knob.
- Pick over full fine-tuning when: compute/memory constrained or need many task-specific adapters cheaply.

**RLHF vs DPO**
- RLHF: train reward model from human preferences, optimize LLM policy against it via PPO. Pros: proven, flexible reward shaping. Cons: two-stage, complex, PPO is unstable and expensive.
- DPO: derives optimal policy analytically from preference pairs, bypassing reward model + PPO. Pros: simpler, more stable, comparable quality. Cons: less flexible than an explicit reward model for complex/multi-objective reward shaping.
- Pick DPO over RLHF when: you want simplicity/stability and don't need a separately reusable reward model.

**Agents**
- What: Loop — perceive state → plan action → call tool → observe result → update state; LLM as control plane.
- Pros: Extends LLMs to multi-step tasks with tool use.
- Cons: Compounding error over long horizons; hard to evaluate/debug; latency and cost multiply per step.
- Pick over single-shot prompting when: task requires multi-step tool use or external state interaction.

| Architecture/technique | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Transformer | parallel training, long-range deps | extremely long sequences without Flash Attention/SSM | parallelism vs. O(N²) cost |
| RAG | fresh/private/large knowledge, reduce hallucination | ultra-low-latency needs, static small knowledge base | groundedness vs. retrieval latency + dependency |
| LoRA | cheap fine-tuning, multiple task adapters | need max flexibility for very different domain | efficiency vs. slightly less expressive than full FT |
| RLHF | flexible/rich reward modeling | want simplicity & stability | flexibility vs. two-stage instability |
| DPO | simple, stable preference alignment | need explicit reusable reward model | simplicity/stability vs. less flexible reward shaping |

---

## Production Systems

**Distributed training**

*Data Parallelism (DP)* — split batch across GPUs, each GPU holds full model. Pick when model fits on one GPU; simplest to implement.

*Tensor Parallelism (TP)* — split weight matrices across GPUs. Pick when a single layer's weights don't fit on one GPU.

*Pipeline Parallelism (PP)* — split model layers across GPUs. Pick when the whole model (all layers) doesn't fit on one GPU's memory even with TP.

| Parallelism | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Data Parallelism | model fits per-GPU, want simplicity | model itself too large for one GPU | simple/scalable vs. no help for huge models |
| Tensor Parallelism | single layer too large for one GPU | communication overhead is prohibitive | fits huge layers vs. heavy inter-GPU communication |
| Pipeline Parallelism | whole model too large even with TP | want to avoid pipeline "bubble" idle time | fits huge models vs. pipeline bubbles/complexity |

**Inference optimization**

*Quantization* — reduce weights to FP16/INT8/INT4. Pros: smaller memory footprint, faster inference. Cons: potential accuracy loss at aggressive bit-widths. Pick when: deploying under tight memory/latency budgets.

*KV-Cache* — cache past token keys/values during autoregressive generation. Pros: avoids recomputation, essential for practical LLM serving. Cons: memory grows with sequence length × batch size (motivates PagedAttention).

*Speculative Decoding* — small draft model proposes tokens, large model verifies in parallel. Pros: 2-4x speedup with no quality loss (rejected tokens re-sampled from large model's true distribution). Cons: needs a good draft model; benefit shrinks if acceptance rate is low.

*PagedAttention (vLLM)* — manages KV cache like OS virtual memory (paged, non-contiguous). Pros: near-zero fragmentation, up to 24x throughput vs naive. Cons: implementation complexity (framework-level, not a manual knob).

*Flash Attention* — tiles attention computation to fit SRAM, avoiding materializing the full N×N matrix in slow HBM. Pros: same math, O(N) memory instead of O(N²), faster due to SRAM speed. Cons: implementation-level optimization, not something you tune per-task.

| Inference technique | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Quantization | memory/latency constrained deployment | accuracy-critical, low tolerance for degradation | speed/size vs. potential accuracy loss |
| KV-Cache | any autoregressive generation serving | N/A (near-universal) | speed vs. memory growth with context length |
| Speculative decoding | latency-sensitive serving w/ good draft model available | no good small draft model exists | speedup vs. draft model maintenance |
| PagedAttention | high-concurrency multi-request serving | single-request/simple serving setups | throughput vs. system complexity |
| Flash Attention | long-sequence training/inference | short sequences (marginal benefit) | linear memory vs. implementation complexity |

**Model monitoring & drift**

*Data drift* — P(X) changes (e.g., new user segment). Detect via PSI or KL divergence on feature distributions.

*Concept drift* — P(Y|X) changes (relationship between features and target shifts). Detect via lagged accuracy/precision on labeled cohorts (can't be seen in feature distributions alone).

| Drift type | Best detection method | Avoid confusing with | Key tradeoff |
|---|---|---|---|
| Data drift | PSI / KL divergence on inputs | concept drift (needs labels, not just inputs) | fast to detect vs. doesn't confirm model is actually wrong |
| Concept drift | lagged ground-truth accuracy | data drift (detectable without labels) | confirms real degradation vs. requires delayed labels |

**A/B testing**
- What: Randomize users between control/treatment, monitor model + business metrics, pre-register sample size.
- Cons/pitfall: "peeking" — stopping early when it looks significant inflates false positive rate.
- Pick over offline eval when: you need to validate actual online business impact, not just offline metric improvement.

---

## Python & Data Tooling

**NumPy vectorization vs loops**
- What: Vectorized ops apply operations across whole arrays via compiled C loops instead of Python-level iteration.
- Pros: Orders of magnitude faster than Python loops.
- Cons: Requires reasoning about broadcasting/shape rules; can silently produce wrong results if shapes broadcast unintentionally.
- Pick over explicit loops when: always, unless the operation is inherently non-vectorizable.

**Views vs copies**
- What: Slicing a NumPy array returns a view (shares memory); fancy indexing/certain ops return a copy.
- Cons: Modifying a view mutates the original array unexpectedly if not intended.
- Rule: know which operations return views to avoid silent mutation bugs.

**Pandas: vectorized ops vs `.apply` vs `.iterrows`**
- Vectorized ops: fastest, use built-in pandas/NumPy methods.
- `.apply`: row/column-wise Python function, slower than vectorized, faster than iterrows.
- `.iterrows`: slowest, avoid except for small data or unavoidable row-wise logic.
- Pick vectorized > apply > iterrows, in that priority order, always.

**Pandas vs Polars**
- Pandas: mature ecosystem, eager execution, single-threaded core ops (mostly). Pick when ecosystem compatibility (sklearn, plotting) matters most.
- Polars: expression API, lazy evaluation (query optimization before execution), multi-threaded, streaming for larger-than-memory data. Pick when performance/memory on large datasets matters and ecosystem lock-in is less important.

| Tool | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Pandas | ecosystem compatibility, small/medium data | very large data, need max performance | maturity/compatibility vs. slower on big data |
| Polars | large data, performance-critical pipelines | need deep sklearn/plotting ecosystem integration | speed/memory efficiency vs. smaller ecosystem |

**Chained indexing (`df[a][b] = x`)**
- Cons: May silently operate on a copy instead of the original (SettingWithCopyWarning); unreliable writes.
- Fix: use `.loc[]`/`.iloc[]` explicitly.

---

## EDA & Data Processing

**Missing value mechanisms**

*MCAR (Missing Completely At Random)* — missingness unrelated to any data. Safe to drop or simple-impute.

*MAR (Missing At Random)* — missingness depends on observed variables. Can be modeled/imputed conditioned on other features.

*MNAR (Missing Not At Random)* — missingness depends on the unobserved value itself (e.g., high earners don't report income). Hardest case — imputation can introduce systematic bias; may need explicit missingness modeling.

| Missingness type | Best handling | Avoid when | Key tradeoff |
|---|---|---|---|
| MCAR | simple drop or mean/median impute | can't verify randomness assumption | simplicity vs. wrong if assumption violated |
| MAR | model-based imputation using other features | those features are themselves missing/unreliable | uses available info vs. more complex pipeline |
| MNAR | explicit missingness indicator / domain modeling | ignoring it (silently biases estimates) | correctness vs. much harder to implement |

**Outlier detection: IQR vs Z-score**

*IQR method* — flags points outside `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`. Pros: robust to non-normal distributions and existing outliers (uses quantiles, not mean/std). Cons: the 1.5 multiplier is a heuristic, not universal. Pick over Z-score when: data is skewed or not normally distributed.

*Z-score method* — flags points beyond ~3 standard deviations from the mean. Pros: simple, well-understood for normal data. Cons: mean/std are themselves distorted by outliers (not robust); breaks down on skewed distributions. Pick over IQR when: data is approximately normal/Gaussian.

| Outlier method | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| IQR | skewed/non-normal data | need a well-understood parametric assumption | robust vs. heuristic threshold |
| Z-score | roughly normal data | skewed data or data with extreme outliers already | simple/interpretable vs. not robust to outliers itself |

**Feature scaling**

*Standardization (z-score)* — `(x-μ)/σ`, mean 0 std 1. Pick for: linear models, PCA, neural nets, algorithms assuming Gaussian-ish features.

*Min-Max scaling* — rescales to `[0,1]`. Pick for: bounded-input algorithms (e.g., neural net inputs, image pixels), when you need a fixed range. Cons: sensitive to outliers (a single extreme value compresses the rest of the range).

*Robust scaling* — uses median/IQR instead of mean/std. Pick for: data with outliers, since it's not distorted by them.

*None needed* — tree-based models (Decision Tree, RF, XGBoost) are scale-invariant since they split on thresholds; monotonic transforms don't change splits.

| Scaling method | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Standardization | linear models, PCA, neural nets | data has extreme outliers | assumes Gaussian-ish spread vs. distorted by outliers |
| Min-Max | bounded-input needs (NN inputs, images) | data has outliers | fixed range vs. outlier sensitivity |
| Robust (median/IQR) | data with outliers | need mean/variance-based downstream assumptions | outlier-resistant vs. less standard for some algorithms |
| None | tree-based models | distance/gradient-based models | scale-invariant by construction vs. N/A |

**Categorical encoding**

*One-hot* — binary column per category. Pick for: low-cardinality nominal features, linear models. Cons: explodes dimensionality with high cardinality.

*Ordinal* — integer per category, preserving order. Pick for: genuinely ordinal categories (e.g., low/medium/high). Cons: implies false ordering if used on nominal data.

*Target encoding* — replace category with mean of target for that category. Pros: compact, captures target-relationship info, handles high cardinality. Cons: high leakage risk if not done with proper out-of-fold/smoothing.

*Frequency encoding* — replace category with its occurrence count/frequency. Pros: simple, no leakage risk, handles high cardinality. Cons: loses target-relationship signal; collapses distinct categories with the same frequency.

*Embeddings* — learned dense vector per category (typically in NN pipelines). Pros: captures rich similarity structure, scales to very high cardinality. Cons: requires enough data to learn meaningfully, adds model complexity.

| Encoding | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| One-hot | low-cardinality nominal | high cardinality (dimensionality blowup) | simple/no leakage vs. dimensionality |
| Ordinal | genuinely ordered categories | nominal categories with no order | compact vs. false order if misused |
| Target encoding | high cardinality, strong target relationship | small data / no leakage-safe CV setup | powerful signal vs. leakage risk |
| Frequency encoding | high cardinality, leakage-sensitive settings | frequency doesn't correlate with target | safe/simple vs. loses target info |
| Embeddings | very high cardinality in NN pipelines | small datasets, non-NN pipelines | rich structure vs. needs scale + complexity |

**Data leakage types**

*Target leakage* — feature is the target or a near-synonym.

*Preprocessing leakage* — fitting scaler/imputer on full dataset before splitting.

*Temporal leakage* — features computed using future timestamps relative to the prediction point.

*ID leakage* — an ID column correlates with cohort/time and implicitly encodes the target.

- Fix pattern for all: use sklearn `Pipeline` + proper CV so preprocessing is refit per fold; audit every feature for "would this be known at prediction time in production?"

**sklearn Pipelines**
- What: Chains preprocessing + model into one object; `cross_val_score`/`GridSearchCV` refit every step per fold.
- Pros: Structurally prevents preprocessing leakage; cleaner deployment (single object to serialize).
- Cons: Slightly less flexible for custom cross-feature logic outside the standard fit/transform contract.
- Pick over manual preprocessing when: leakage risk exists (virtually always in CV workflows).

---

## Math & Theory Foundations

**Eigenvalues/Eigenvectors**
- What: `Av = λv` — v only scales (doesn't rotate) under transformation A; λ is the scale factor.
- Use: PCA (principal components = eigenvectors of covariance matrix), stability analysis.

**SVD (Singular Value Decomposition)**
- What: `A = UΣVᵀ` — decomposes any matrix (not just square) into orthogonal U, V and singular values Σ.
- Pros: Works on any matrix, numerically stable, foundation for PCA, low-rank approximation, pseudo-inverse.
- Pick over eigendecomposition when: matrix is non-square or numerical stability matters.

**Condition number**
- What: Ratio of largest to smallest singular value.
- Interpretation: high condition number → matrix is ill-conditioned/unstable — small input changes cause large output changes (relevant to optimization landscape conditioning).

**MLE vs MAP vs Full Bayesian**

*MLE* — finds parameters maximizing `P(data|θ)`. Pros: simple, consistent estimator, equivalent to MSE (Gaussian likelihood) or cross-entropy (categorical). Cons: no regularization, can overfit with little data, ignores prior knowledge.

*MAP* — maximizes `P(θ|data) ∝ P(data|θ)P(θ)`, incorporating a prior. Pros: equivalent to regularization (Gaussian prior = L2, Laplace prior = L1); combats overfitting with small data. Cons: point estimate still — no uncertainty quantification; result depends on chosen prior.

*Full Bayesian inference* — computes the full posterior distribution over θ, not just a point estimate. Pros: full uncertainty quantification. Cons: computationally expensive (often needs MCMC/variational inference); intractable in closed form for most models.

- Pick MLE when: large data, no need for regularization/uncertainty.
- Pick MAP when: small data, want regularization, need only a point estimate.
- Pick Full Bayesian when: uncertainty quantification is required and compute budget allows (MCMC/VI).

| Estimation approach | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| MLE | large data, no regularization needed | small data (overfits, ignores priors) | simplicity vs. no uncertainty/regularization |
| MAP | small data, want regularization | need full uncertainty over parameters | regularized point estimate vs. still no uncertainty |
| Full Bayesian | need calibrated uncertainty | compute-constrained / need fast point answer | full uncertainty vs. expensive (MCMC/VI required) |

**MCMC vs Variational Inference (VI) vs Laplace Approximation**

*MCMC* — samples from the true posterior via a Markov chain (e.g., Metropolis-Hastings, HMC). Pros: asymptotically exact. Cons: slow to converge/mix, hard to diagnose convergence.

*Variational Inference* — approximates the posterior with a simpler parametric family, optimized via ELBO maximization. Pros: much faster than MCMC, scales to large models (used in VAEs). Cons: approximation only as good as the chosen family; can underestimate posterior variance.

*Laplace Approximation* — approximates posterior as Gaussian centered at the MAP estimate using local curvature (Hessian). Pros: cheap, simple, closed-form. Cons: poor approximation when posterior is multimodal or highly non-Gaussian.

- Pick MCMC when: need asymptotically exact posterior and compute time is available.
- Pick VI when: need speed/scalability and can tolerate approximation.
- Pick Laplace when: need a cheap, quick-and-dirty Gaussian approximation near a mode.

| Posterior approximation | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| MCMC | exact posterior, small-to-medium models | need speed/scale, real-time constraints | accuracy vs. slow convergence |
| Variational Inference | large-scale models (VAEs), need speed | posterior is very complex/multimodal and precision matters | fast/scalable vs. approximation bias |
| Laplace Approximation | quick Gaussian approx near a single mode | multimodal or highly non-Gaussian posteriors | cheap/simple vs. poor fit away from the mode |

**Generalization theory**

*VC Dimension* — measures a hypothesis class's capacity (largest set of points it can shatter). Higher VC dim → more expressive but higher overfitting risk without enough data.

*Double Descent* — test error can decrease, increase, then decrease again as model capacity grows past the interpolation threshold — contradicts classical bias-variance intuition at the "modern"/overparameterized regime.

*PAC Learning* — framework bounding how much data is needed to learn a hypothesis class within (ε, δ) error/confidence guarantees.

**Convex optimization / KKT**
- Convexity: guarantees any local minimum is global — critical for reliable optimization guarantees (true for e.g. logistic regression/SVM loss, not for deep nets).
- KKT conditions: necessary conditions for optimality in constrained optimization (generalizes Lagrange multipliers to inequality constraints).
- Why SGD works on non-convex DL landscapes anyway: empirically, over-parameterized loss landscapes have few bad local minima; saddle points are more common obstacles than local minima, and noise from SGD helps escape them.

---

## Information Theory

**Shannon Entropy**
- What: `H(P) = -Σ P(x)log P(x)` — expected "surprise"/uncertainty in a distribution.
- Use: baseline for measuring information content; higher entropy = more uncertainty/randomness.

**Cross-Entropy**
- What: `H(P,Q) = -Σ P(x)log Q(x)` — expected cost of encoding true distribution P using coding scheme optimized for Q.
- Use: the standard classification loss function — measures how well predicted distribution Q matches true distribution P.
- Key identity: `H(P,Q) = H(P) + D_KL(P‖Q)`.

**KL Divergence**
- What: `D_KL(P‖Q) = Σ P(x)log(P(x)/Q(x))` — non-symmetric distance between two distributions.
- Cons: not a true metric (asymmetric, no triangle inequality).
- Forward KL `D_KL(P‖Q)`: mean/mass-covering behavior — used when fitting Q to cover all modes of P.
- Reverse KL `D_KL(Q‖P)`: mode-seeking behavior — used in variational inference (Q collapses onto one mode of P).
- Pick forward KL when: want the approximation to cover the whole true distribution (e.g., MLE fitting).
- Pick reverse KL when: want a tractable single-mode approximation (e.g., VI, avoiding placing mass in low-density regions).

| KL direction | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Forward KL (P‖Q) | mass-covering fit (e.g. standard MLE) | multimodal P where covering-all-modes causes blurry averaging | covers all modes vs. can average over modes |
| Reverse KL (Q‖P) | tractable VI approximations | need to represent the full distribution's spread | sharp single-mode fit vs. can miss other modes entirely |

**Mutual Information**
- What: `I(X;Y) = H(X) - H(X|Y)` — reduction in uncertainty about X from observing Y.
- Use: feature selection (how informative is a feature about the target), measuring dependence beyond linear correlation.

**Perplexity**
- What: `exp(cross-entropy)` — interpretable as the effective number of equally-likely choices a language model is choosing among.
- Use: standard LM evaluation metric; lower is better.

**ELBO (Evidence Lower Bound)**
- What: Lower bound on log-likelihood used to optimize variational inference tractably (`log P(x) ≥ ELBO`).
- Use: training objective for VAEs and other latent-variable models where the true posterior is intractable.

---

## Glossary-Only Concepts (not covered in core files 01–05)

**Q-Learning / DQN**
- What: Learn `Q(s,a)` via Bellman update `Q(s,a) ← r + γ·max_a' Q(s',a')` without modeling environment dynamics.
- DQN adds: neural function approximator + experience replay (breaks temporal correlation) + target network (stabilizes targets).
- Pick over policy-gradient methods when: action space is discrete and you want a well-understood, off-policy, sample-reusing algorithm.

**PPO (Proximal Policy Optimization)**
- What: Clips the policy update ratio to stay within a "proximal" region of the old policy, preventing destructive large updates.
- Pros: More stable than vanilla policy gradient methods.
- Cons: Still an RL algorithm — sample-inefficient, sensitive to reward design; the "expensive/unstable" stage of RLHF.
- Pick over vanilla policy gradient when: training stability matters (this is why it underlies RLHF).

**Two-Tower Model**
- What: Encode user and item separately into a shared embedding space via independent towers; retrieve via ANN search on dot products.
- Pros: Scales retrieval to millions of items in milliseconds.
- Cons: Cold start — new users/items have no interaction history to embed; needs fallback (content features, popularity).
- Pick over joint scoring models when: retrieval must scale to huge item catalogs in real time.

**NDCG (Normalized Discounted Cumulative Gain)**
- What: `DCG = Σ(relevance_i / log₂(i+1))`, normalized by ideal DCG; range [0,1]; evaluated at cutoff K.
- Pick over precision/recall for ranking tasks when: position in the ranked list matters (top-of-list errors cost more).

**Knowledge Distillation**
- What: Train a small "student" to match a large "teacher"'s full soft output distribution (not just hard labels), often with temperature scaling.
- Pros: Smaller/faster model approaching teacher accuracy.
- Cons: Still bounded by teacher's ceiling; requires access to teacher's soft outputs.
- Pick over training small model from scratch when: a strong teacher model already exists.

**MoE (Mixture of Experts)**
- What: Partition FFN layers into E experts; a router selects K of them per token (e.g., K=2 of E=8/64). Params scale with E, compute scales with K.
- Pros: Scales capacity without proportional compute increase.
- Cons: Load balancing failure (expert collapse) without an auxiliary balancing loss; routing adds complexity.
- Pick over dense scaling when: you want more parameters without proportionally more compute per token.

**Differential Privacy (DP) / DP-SGD**
- What: Clip per-example gradient norms, add calibrated Gaussian noise before averaging — bounds any single example's effect on final weights (ε privacy budget).
- Pros: Formal privacy guarantee against memorization/leakage.
- Cons: Noise degrades model utility; ε/utility tradeoff must be tuned.
- Pick over standard training when: training on sensitive data with a compliance/privacy requirement.

**Federated Learning**
- What: Clients train locally, only send model updates (not data) to a central server; server aggregates (FedAvg).
- Pros: Data never leaves client devices — privacy, bandwidth, regulatory benefits.
- Cons: Non-IID client data causes conflicting local optima; gradient inversion can still leak info; communication overhead remains.
- Pick over centralized training when: data cannot be centralized (privacy/regulatory/ownership constraints).

**Speculative Decoding, Flash Attention, PagedAttention** — see Production Systems section above.

**Test-Time Scaling**
- What: Spend more inference compute (chain-of-thought, search/MCTS over reasoning chains, self-verification) instead of more training compute to improve output quality.
- Pros: Improves hard-reasoning performance from an already-trained model.
- Cons: Higher latency/cost per query.
- Pick over further pretraining when: marginal training compute is more expensive than marginal inference compute, or the model is already deployed.

**Constitutional AI (CAI)**
- What: Model critiques/revises its own outputs against a set of written principles; critiqued outputs become training data — reduces reliance on human preference labels.
- Pick over pure RLHF when: you want to reduce cost/inconsistency of human labeling.

**CLIP (Contrastive Language-Image Pretraining)**
- What: Jointly trains image/text encoders with contrastive loss on image-caption pairs, aligning visual/textual embedding space.
- Pros: Enables zero-shot classification (embed class names as text, compare to image embedding).
- Cons: Needs huge paired image-text datasets; zero-shot accuracy still below fully supervised specialist models on narrow tasks.
- Pick over traditional supervised classifiers when: you need zero-shot/open-vocabulary classification.

**Causal Inference tools**

*RCT / A/B test* — randomization breaks confounding; gold standard when feasible.

*Difference-in-Differences (DiD)* — compares treatment/control before/after; controls for time-invariant confounders. Pick when randomization isn't possible but a natural pre/post treatment split exists.

*Propensity Score Matching* — matches treated/untreated units on observed covariates to approximate randomization. Pick when: randomization is infeasible but rich covariates are observed (cannot control for unobserved confounders).

*Instrumental Variables* — uses a variable affecting treatment but not outcome directly, to isolate causal effect. Pick when: unobserved confounding is suspected and a valid instrument exists.

| Causal method | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| RCT / A/B test | feasible to randomize | randomization impossible/unethical | gold standard vs. often infeasible |
| DiD | natural pre/post treatment split exists | treatment/control trends weren't parallel pre-treatment | simple/quasi-experimental vs. relies on parallel trends assumption |
| Propensity Score Matching | rich observed covariates, no randomization | important unobserved confounders exist | approximates randomization vs. can't fix unobserved confounding |
| Instrumental Variables | unobserved confounding + valid instrument available | no valid instrument exists (weak/invalid instruments bias results) | handles unobserved confounding vs. hard to find valid instruments |

**Confounder**
- What: A third variable causing both "treatment" and "outcome," creating spurious correlation (e.g., hot weather → ice cream sales AND drowning).
- Practical impact: models trained on observational data can confound correlation with causation, breaking when deployed as an intervention.

**Peeking problem (A/B testing)**
- What: Stopping an experiment early upon seeing significance (before pre-specified sample size) inflates false positive rate.
- Fix: pre-register sample size and significance threshold; don't peek.

**Mamba / SSM (Selective State Space Model)**
- What: Processes sequences in O(N) via a recurrently updated latent state; "selective" variant learns input-dependent state updates (what to remember/forget).
- Pros: Linear-time alternative to O(N²) attention for very long sequences.
- Cons: Less mature tooling/ecosystem than Transformers; recurrence can still limit some parallelism during training vs. full attention.
- Pick over Transformer when: sequence lengths make O(N²) attention prohibitive even with Flash Attention.

**RAGAS / RAG Evaluation metrics**

*Faithfulness* — does the answer only claim what retrieved context supports? Detects generation hallucination despite good retrieval.

*Answer Relevancy* — does the answer address the question? Detects generation that ignores the actual question.

*Context Precision* — are retrieved documents relevant to the question? Detects retrieval-stage noise.

*Context Recall* — are retrieved documents sufficient to answer? Detects retrieval-stage insufficiency.

- Use together to diagnose which RAG component (retrieval vs generation) is failing without expensive human annotation per query.

| RAG metric | Diagnoses | Avoid confusing with | Key tradeoff |
|---|---|---|---|
| Faithfulness | generation hallucinating beyond context | context precision (retrieval issue) | catches ungrounded claims vs. doesn't check if context itself was relevant |
| Answer Relevancy | generation ignoring the question | faithfulness (which checks grounding, not relevance) | catches off-topic answers vs. doesn't check factual grounding |
| Context Precision | retrieval returning irrelevant docs | context recall (a coverage issue, not relevance) | flags noisy retrieval vs. doesn't check sufficiency |
| Context Recall | retrieval missing necessary docs | context precision (a relevance issue, not coverage) | flags missing info vs. doesn't check relevance of what was retrieved |

---

## Quick Reference: Common "Pick X over Y" Interview Answers

- Tree-based models (RF/XGBoost) need no feature scaling — split-based, scale-invariant by construction; linear models/NNs/KNN/SVM/PCA are distance- or gradient-based and do need it.
- Random Forest over XGBoost: want a robust, low-tuning baseline. XGBoost over Random Forest: have time to tune, need max accuracy.
- PR-AUC over ROC-AUC: imbalanced classes (fraud, rare disease).
- LayerNorm over BatchNorm: Transformers, small/variable batch sizes.
- AdamW over Adam: any setup using weight decay (virtually all modern Transformer training).
- DPO over RLHF: want simplicity/stability without a separate reward model.
- LoRA over full fine-tuning: compute/memory constrained, need multiple cheap task adapters.
- UMAP over t-SNE: need speed + global structure + downstream-usable embeddings.
- Time-series split over K-fold: any temporal data — never shuffle.
- L1 over L2: want automatic feature selection / sparse weights. L2 over L1: correlated features, want stability.
