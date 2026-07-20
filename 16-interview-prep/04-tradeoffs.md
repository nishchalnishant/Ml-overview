---
module: Interview Prep
topic: Tradeoff Question Bank (ML, DL, LLMs)
subtopic: ""
status: unread
tags: [interviewprep, tradeoffs, ml, dl, llms]
---
# Tradeoff Question Bank — ML, DL & LLMs

Each entry: default choice → constraint driving it → failure mode/cost of the default → condition
that flips you to the alternative. Treat these as starting positions, not scripture — restate them
in your own words and adjust for the specifics of whatever role/problem you're given. Practice
saying the whole 4-part shape out loud in under 20 seconds.

---

## Part 1 — Classical ML

### Model choice

**GBT (XGBoost/LightGBM) vs. deep learning (MLP/embeddings) for tabular data?**
- Default: GBT. Constraint: tabular data with heterogeneous feature types and no huge dataset — trees split on thresholds naturally, need less data and no normalization.
- Failure mode: GBT can't share representations across high-cardinality categoricals or fuse with unstructured inputs (text/image) in the same model.
- Flip: dataset is large (>~10M rows), features include text/image/sequence that need to be fused end-to-end, or you need embeddings that transfer across tasks.

**Linear/logistic regression vs. tree ensembles?**
- Default: start with logistic regression as a baseline. Constraint: need an interpretable, fast-to-train reference point before investing in complexity.
- Failure mode: linear models miss feature interactions and non-linear thresholds trees capture for free.
- Flip: once the baseline is beaten by a meaningful margin by a tree ensemble, or interactions are known to matter (domain knowledge says "X only matters when Y is high").

**Random forest vs. gradient boosting?**
- Default: gradient boosting (XGBoost/LightGBM) when squeezing out accuracy matters and you have time to tune. Constraint: boosting reduces bias sequentially and usually wins on leaderboard-style accuracy.
- Failure mode: boosting is more sensitive to hyperparameters and easier to overfit; random forest is more robust out of the box.
- Flip to random forest: limited tuning time/expertise, need robustness to noisy labels, or want easy parallelism (trees are independent) over sequential accuracy gains.

**k-NN vs. a parametric model?**
- Default: parametric model (logistic regression/GBT) for anything at production scale. Constraint: k-NN inference cost scales with dataset size (or requires an ANN index), and it's memory-hungry.
- Failure mode: parametric models can underfit if the true decision boundary is highly irregular and local.
- Flip to k-NN: small dataset, decision boundary is genuinely local/non-parametric (e.g. recommendation via similarity), and inference latency budget can absorb the lookup cost.

**SVM vs. logistic regression for a small, high-dimensional dataset?**
- Default: logistic regression with regularization. Constraint: easier to calibrate, cheaper to train, and interpretable coefficients.
- Failure mode: logistic regression assumes a linear boundary in feature space (mitigated by kernels, but then you've built an SVM anyway).
- Flip to SVM: margin-maximization is specifically valuable (small samples, clear margin expected), or a kernel trick meaningfully separates non-linear classes and calibration isn't required.

**Single global model vs. per-segment models (e.g. per-region, per-product-line)?**
- Default: single global model with segment as a feature. Constraint: maximizes data per model and avoids fragmenting rare segments.
- Failure mode: if segments have genuinely different feature-target relationships (not just different base rates), a global model averages them away and underperforms in each segment.
- Flip: run an interaction-term ablation — if segment-feature interactions are strong and each segment has enough data to train independently, split.

**Simple heuristic/rules vs. ML model as the very first version?**
- Default: ship the heuristic first if one exists (e.g. "flag if >3 failed logins in 10 min"). Constraint: gets a baseline live fast and de-risks "is ML even needed" before investing in a pipeline.
- Failure mode: heuristics plateau quickly and can't capture interactions or adapt to shifting patterns.
- Flip to ML immediately: heuristic accuracy is provably far from acceptable, or this is explicitly a v2/replacement project where ML must beat an existing heuristic stated as the launch bar.

**Generative vs. discriminative model for a classification task?**
- Default: discriminative (logistic regression, GBT) when you only need $P(y|x)$ for prediction. Constraint: discriminative models directly optimize the decision boundary and typically need less data to reach a given accuracy.
- Failure mode: discriminative models can't generate samples or handle missing features gracefully at inference time.
- Flip to generative: you need to model $P(x,y)$ — e.g. handling missing inputs, anomaly detection, or when labeled data is scarce but the generative assumptions (e.g. Naive Bayes' conditional independence) roughly hold.

**Parametric vs. non-parametric model when the data-generating process is unknown?**
- Default: start parametric (fewer assumptions to verify, faster, less data-hungry) unless there's strong reason to believe the true function is highly irregular.
- Failure mode: a misspecified parametric model has irreducible bias no amount of data fixes.
- Flip to non-parametric: enough data to support it, and validation curves show the parametric model's bias isn't shrinking with more data/complexity.

### Feature engineering & data

**Manual feature engineering vs. learned representations/embeddings?**
- Default: manual features for tabular/structured data with strong domain priors (ratios, rolling aggregates). Constraint: manual features are interpretable, debuggable, and don't need huge data volumes.
- Failure mode: manual feature engineering doesn't scale to unstructured inputs (text, images) or capture interactions you didn't think to encode.
- Flip to learned embeddings: unstructured input, very high-cardinality categoricals, or enough data that a learned representation outperforms hand-crafted ones.

**Dropping rows with missing data vs. imputation (mean/median vs. model-based/MICE)?**
- Default: simple imputation (median for skewed, mean for symmetric) + a missingness indicator flag. Constraint: preserves sample size and lets the model learn if "missing" itself is informative.
- Failure mode: naive imputation can distort distributions and hide a systematic missingness mechanism.
- Flip: drop rows only when missingness is a small fraction and clearly random (MCAR); use model-based imputation (MICE) when missingness is substantial and relationships between features are strong enough to predict the missing value reliably.

**One-hot encoding vs. target/embedding encoding for high-cardinality categoricals?**
- Default: one-hot for low-cardinality (<~20 categories). Constraint: simple, no leakage risk, works with any model.
- Failure mode: one-hot blows up dimensionality and sparsity for high-cardinality features (e.g. zip code, user ID).
- Flip to target/embedding encoding: high cardinality, and you can guard against target leakage (out-of-fold encoding, regularization/smoothing).

**Oversampling (SMOTE) vs. undersampling vs. class weights for imbalanced data?**
- Default: class weights first — cheapest, no synthetic data risk, works with most models directly.
- Failure mode: class weights alone sometimes aren't enough for extreme imbalance (e.g. 0.1% positive rate).
- Flip to SMOTE: need more signal from the minority class and have continuous features where interpolation makes sense. Flip to undersampling: majority class is redundant/cheap to subsample and training time is the bottleneck — but validate on the untouched (real) distribution either way.

**Feature selection vs. regularization to handle high dimensionality?**
- Default: regularization (L1/L2) over explicit feature selection. Constraint: regularization is differentiable, doesn't require a separate selection step, and L1 gives sparsity as a side effect.
- Failure mode: regularization alone doesn't reduce serving-time feature computation cost — you still compute every feature.
- Flip to explicit selection: feature computation cost/latency at serving time is the actual constraint (not just overfitting), or you need interpretability from a small, named feature set.

**Raw features vs. PCA/dimensionality reduction before modeling?**
- Default: raw features for tree-based models (they handle high dimensionality and correlated features natively).
- Failure mode: PCA components are uninterpretable and can discard information relevant to the target (it's unsupervised — maximizes variance, not predictive power).
- Flip to PCA: distance-based methods (k-NN, k-means) suffering from the curse of dimensionality, or extreme dimensionality causing compute/memory issues with no interpretability requirement.

**Synthetic data augmentation vs. collecting more real labeled data?**
- Default: collect more real data when feasible — no distributional-mismatch risk.
- Failure mode: real data collection is often slow, expensive, or impossible (rare events, privacy-sensitive domains).
- Flip to synthetic: real data collection is the bottleneck and you can validate the synthetic generator doesn't introduce a distribution shift from production reality (validate on real hold-out only).

### Regularization & generalization

**L1 vs. L2 regularization?**
- Default: L2 (ridge) when you believe most features contribute a little. Constraint: smooth shrinkage, all features retained, generally more stable optimization.
- Failure mode: L2 doesn't zero out irrelevant features, so it doesn't help with interpretability or serving-time cost.
- Flip to L1: you want implicit feature selection (sparse, zeroed coefficients) and suspect many features are actually irrelevant.

**Early stopping vs. explicit regularization penalty?**
- Default: use both together in practice, but if forced to choose one for a fast iteration loop, early stopping — free (just monitor validation loss), no extra hyperparameter to tune per-weight.
- Failure mode: early stopping is sensitive to the validation set's noise (stop too early/late based on a lucky/unlucky epoch).
- Flip to explicit penalty: you need a stable, reproducible model independent of training-run noise, or want to control specific weight magnitudes deliberately (e.g. for calibration).

**Dropout vs. weight decay?**
- Default: dropout for large neural nets with fully-connected layers prone to co-adaptation. Constraint: acts as an implicit ensemble over sub-networks.
- Failure mode: dropout increases training time to convergence and interacts awkwardly with batch normalization if not sequenced carefully.
- Flip to weight decay: smaller networks, or when using architectures (e.g. certain conv/transformer blocks) where dropout placement is finicky and L2-style decay is simpler to reason about.

**Bagging vs. boosting for variance/bias control?**
- Default: bagging (random forest) when the base learner overfits (high variance, low bias) — averaging independent trees reduces variance.
- Failure mode: bagging does little for a high-bias base learner; averaging weak, biased models still gives a biased ensemble.
- Flip to boosting: base learner underfits (high bias) — sequential boosting directly reduces bias by fitting residuals, at the cost of more overfitting risk and less parallelism.

**Cross-validation vs. single holdout split — when does the extra compute pay off?**
- Default: single holdout under time pressure or with a large dataset (holdout variance is already low).
- Failure mode: single holdout is noisy for small datasets — model selection can be driven by holdout-set luck.
- Flip to K-fold: dataset is small enough that per-fold variance matters, and you have compute budget for K full retrains; report the final config's confidence interval via K-fold or bootstrap regardless.

**Bias-heavy simple model vs. variance-heavy complex model, given a fixed, small dataset?**
- Default: simple/high-bias model. Constraint: with little data, a complex model's variance dominates test error — it memorizes noise.
- Failure mode: an overly simple model can leave real signal on the table (underfitting) if the true relationship truly is complex.
- Flip to complex: data size grows, or you add strong regularization/augmentation that controls variance enough to let a complex model's lower bias pay off.

### Hyperparameter search & training

**Grid search vs. random search vs. Bayesian optimization (Optuna/TPE)?**
- Default: random search for cheap trials (small models); Bayesian optimization once each trial is expensive (large DL models, long training runs).
- Failure mode: grid search wastes compute on unpromising regions and scales exponentially with hyperparameter count.
- Flip to grid search: only justified with ≤2 hyperparameters where exhaustive coverage is cheap and interpretability of the full grid matters.

**Manual tuning vs. automated search under a tight time-box?**
- Default: automated search whenever you have the time budget — it's more systematic and reproducible.
- Failure mode: automated search needs enough trials to be reliable; in a 45-minute interview it won't finish.
- Flip to manual: extremely tight time-box — tune 1-2 known highest-leverage knobs (learning rate, tree depth/regularization) by hand and state you'd automate given more time.

**Full K-fold CV vs. single holdout split during search?**
- Default: single holdout for large datasets or expensive-to-train models — the extra folds cost K times the compute for a diminishing variance-reduction return.
- Failure mode: single holdout can pick a hyperparameter config that got lucky/unlucky on that particular split.
- Flip to K-fold: small dataset where holdout variance materially affects which config wins, and compute allows K full retrains per candidate config.

**Early stopping on a validation set vs. fixed number of epochs/rounds?**
- Default: early stopping — avoids both under- and over-training without hand-picking an epoch count.
- Failure mode: requires a held-out validation set (costs data) and adds noise-sensitivity to the stopping point.
- Flip to fixed epochs: validation data is too scarce to spare, or you're reproducing a known, previously-validated training recipe exactly.

### Metrics & evaluation

**Accuracy vs. F1 vs. AUC-ROC vs. PR-AUC — which for an imbalanced classification problem?**
- Default: PR-AUC (or F1 at an operating threshold) for imbalanced problems. Constraint: accuracy is dominated by the majority class and AUC-ROC can look deceptively good under heavy imbalance since FPR stays low even with many false positives relative to the tiny positive class.
- Failure mode: PR-AUC alone doesn't tell you the right operating threshold — still need to pick one based on business cost of FP vs. FN.
- Flip to AUC-ROC: classes are roughly balanced, or you only care about ranking quality across all thresholds rather than a specific operating point.

**Optimizing for calibration vs. optimizing for raw discrimination (AUC)?**
- Default: calibration matters whenever the score feeds a downstream decision with a fixed threshold or cost (e.g. "flag if p>0.8", blending with other ranking signals, or showing the raw number to a human).
- Failure mode: a highly discriminative but uncalibrated model can rank correctly yet give meaningless absolute probabilities, breaking any fixed-threshold logic.
- Flip to pure discrimination: only the ranking/order matters (e.g. top-k retrieval), not the absolute score.

**Offline proxy metric vs. online/business metric as the optimization target?**
- Default: pick the offline metric that best proxies the business metric (e.g. AUC as a proxy for revenue lift), but explicitly flag it's a proxy and plan the A/B test that validates it.
- Failure mode: offline metric improvements don't always translate to business-metric improvements (e.g. better click prediction doesn't always mean more revenue).
- Flip: if an online metric is already cheaply instrumented and testable, optimize directly against it rather than over-investing in an untested proxy.

**Point estimate vs. confidence interval (bootstrap) when reporting model performance?**
- Default: report a confidence interval (bootstrap or K-fold spread) whenever the decision maker will compare this number against another model or a launch bar.
- Failure mode: a bare point estimate hides whether a "0.3% AUC improvement" is real signal or noise.
- Flip to point estimate only: quick internal sanity-check iteration where you'll re-verify the final candidate with an interval before any real decision.

---

## Part 2 — Deep Learning

### Architecture choice

**CNN vs. Vision Transformer (ViT) for an image task?**
- Default: CNN for small-to-medium labeled datasets. Constraint: CNNs have a built-in inductive bias (locality, translation invariance) that lets them generalize with less data.
- Failure mode: CNNs' inductive bias caps how well they exploit very large datasets compared to a more flexible architecture.
- Flip to ViT: large labeled dataset (or strong pretraining available) where the lack of inductive bias becomes an advantage rather than a data-efficiency liability.

**RNN/LSTM/GRU vs. Transformer for sequence modeling?**
- Default: Transformer for most modern sequence tasks. Constraint: parallelizable training (no sequential dependency), better long-range dependency modeling via attention.
- Failure mode: Transformer attention is $O(n^2)$ in sequence length — expensive/infeasible for very long sequences without modifications.
- Flip to RNN/LSTM/GRU: strict low-memory/streaming inference constraints (process one token at a time with fixed state), or sequence length is extreme and full attention is infeasible without added complexity (sparse/linear attention).

**GRU vs. LSTM when you do need a recurrent architecture?**
- Default: GRU. Constraint: fewer gates/parameters than LSTM, faster to train, often comparable performance.
- Failure mode: on tasks needing very fine-grained control over what's forgotten vs. retained, LSTM's separate cell state can have an edge.
- Flip to LSTM: task-specific validation shows a meaningful accuracy gain, or you're matching an existing/legacy architecture for compatibility.

**CNN vs. fully-connected network for structured spatial data?**
- Default: CNN whenever input has spatial/local structure (images, spectrograms). Constraint: weight sharing drastically reduces parameters vs. a fully-connected layer on the same input size.
- Failure mode: CNNs assume locality — inappropriate for tabular data with no spatial relationship between adjacent columns.
- Flip to fully-connected: input has no spatial structure (generic tabular features) — a CNN's inductive bias would be actively wrong.

**Shallow-and-wide vs. deep-and-narrow network for a fixed parameter budget?**
- Default: deep-and-narrow. Constraint: depth composes non-linearities, generally more expressive per parameter than width alone.
- Failure mode: very deep networks are harder to optimize (vanishing/exploding gradients) without careful init/normalization/residual connections.
- Flip to shallow-and-wide: optimization difficulty at depth isn't worth it for the task's actual complexity, or you lack the training infrastructure (residual connections, normalization) to stabilize a deep network.

**Autoencoder vs. GAN vs. diffusion model for a generative task?**
- Default: diffusion model for high-fidelity sample generation today. Constraint: more stable training than GANs, higher sample quality than vanilla autoencoders.
- Failure mode: diffusion models are slow at inference (many denoising steps) compared to a single forward pass.
- Flip to GAN: inference speed is critical and you can tolerate/manage training instability (mode collapse). Flip to autoencoder: you need a compact latent representation for downstream tasks (compression, anomaly detection) rather than best-in-class sample quality.

### Optimization

**SGD (with momentum) vs. Adam/AdamW?**
- Default: AdamW for most deep learning tasks, especially early in a project. Constraint: adaptive per-parameter learning rates make it more forgiving of a poorly-tuned base learning rate.
- Failure mode: Adam can converge to sharper minima that generalize slightly worse than SGD in some vision tasks, and its adaptive estimates add memory overhead.
- Flip to SGD+momentum: final-stage fine-tuning for best generalization on well-studied architectures (e.g. ResNet on ImageNet) where SGD is known to reach better test accuracy given enough tuning budget.

**Fixed learning rate vs. LR schedule (cosine decay, warmup + decay)?**
- Default: warmup + cosine decay schedule for any non-trivial training run. Constraint: warmup avoids early instability from large gradients on random init; decay lets the model settle into a sharper minimum late in training.
- Failure mode: a schedule adds hyperparameters (warmup length, decay shape) that themselves need tuning.
- Flip to fixed LR: very short training runs or quick prototyping where schedule tuning isn't worth the iteration overhead.

**Batch gradient descent vs. mini-batch vs. online (single-sample) updates?**
- Default: mini-batch. Constraint: balances gradient estimate stability (vs. online) with per-step compute/memory cost (vs. full batch) and enables GPU parallelism.
- Failure mode: batch size itself becomes a tuning axis affecting both convergence speed and generalization.
- Flip to full-batch: dataset is small enough to fit in memory and you want the most stable gradient estimate (e.g. classical convex optimization). Flip to online: strict streaming setting where data arrives one sample at a time and you can't buffer a batch.

**Large batch size vs. small batch size — what does each cost you?**
- Default: the largest batch size your hardware supports without needing to scale the learning rate awkwardly, paired with LR scaling rules (linear scaling + warmup).
- Failure mode: very large batches can generalize worse (sharper minima) unless LR/warmup is adjusted; very small batches have noisy, slow-to-converge gradients but often generalize better and need less memory.
- Flip to small batch: memory-constrained hardware, or empirically better generalization matters more than wall-clock training speed.

**Gradient clipping vs. redesigning the architecture/init to prevent exploding gradients?**
- Default: gradient clipping as an immediate safety net, especially for RNNs/Transformers. Constraint: cheap, doesn't require redesign, catches rare spikes.
- Failure mode: clipping treats the symptom — if gradients explode constantly, clipping just masks an underlying init/normalization problem.
- Flip to redesign: exploding gradients are frequent/systematic rather than rare — fix init scheme, add normalization layers, or add residual connections instead of relying on clipping alone.

### Regularization & generalization (DL-specific)

**Dropout vs. batch normalization vs. layer normalization?**
- Default: layer normalization for Transformers/sequence models (batch statistics are unstable/ill-defined across variable-length sequences); batch normalization for CNNs with large, consistent batch sizes.
- Failure mode: batch norm degrades with very small batch sizes (noisy batch statistics) or when train/inference batch composition differs.
- Flip to dropout: you want regularization independent of normalization method, or are combining with either norm type as a complementary regularizer (careful with ordering/placement).

**Data augmentation vs. architectural regularization (dropout/weight decay)?**
- Default: data augmentation when domain-appropriate transformations exist (image flips/crops, text paraphrase) — it directly increases effective data diversity rather than just penalizing model capacity.
- Failure mode: augmentation choices can inject label-inconsistent transformations if not domain-aware (e.g. flipping a digit "6" to look like "9").
- Flip to architectural regularization: no safe/meaningful augmentation exists for the domain, or overfitting persists even after reasonable augmentation.

**Transfer learning (freeze + fine-tune) vs. training from scratch?**
- Default: transfer learning whenever a relevant pretrained model exists. Constraint: drastically less data and compute needed to reach strong performance.
- Failure mode: pretrained representations can carry biases or domain mismatch from the source dataset that hurt on a sufficiently different target domain.
- Flip to scratch: target domain is very different from anything pretrained models have seen, or you have enough data/compute that a bespoke architecture outperforms adapting a general one.

**Full fine-tuning vs. freezing the backbone and only training a head?**
- Default: freeze backbone + train head when target dataset is small. Constraint: prevents catastrophic forgetting/overfitting on limited data.
- Failure mode: a frozen backbone can't adapt its features to a target domain that's meaningfully different from pretraining data, capping achievable accuracy.
- Flip to full fine-tuning: enough target data and compute, and the target domain diverges enough from pretraining that backbone features need to adapt too (often done in stages: head first, then unfreeze).

### Loss functions

**Cross-entropy vs. focal loss for class imbalance?**
- Default: cross-entropy + class weighting first — simpler, one less hyperparameter (focal loss's $\gamma$) to tune.
- Failure mode: plain weighted cross-entropy can still be dominated by easy negatives in extreme imbalance (e.g. object detection background vs. foreground).
- Flip to focal loss: extreme imbalance where easy negatives overwhelm the loss even after class weighting — focal loss's down-weighting of well-classified examples directly targets this.

**MSE vs. MAE vs. Huber loss for regression, given outlier-heavy data?**
- Default: Huber loss when outliers are present but you still want smooth gradients near zero. Constraint: quadratic near zero (stable optimization), linear for large errors (outlier-robust).
- Failure mode: Huber introduces a delta hyperparameter that needs tuning to the error scale of your problem.
- Flip to MAE: outliers should be robustly ignored entirely and gradient smoothness near zero doesn't matter. Flip to MSE: no meaningful outliers, and you want the loss to penalize large errors more aggressively (matches Gaussian-noise assumption).

**Contrastive/triplet loss vs. classification loss for representation learning?**
- Default: contrastive loss when the goal is a general-purpose embedding space (similarity search, retrieval) rather than a fixed label set.
- Failure mode: contrastive/triplet losses are sensitive to negative-sampling strategy and mining hard negatives is its own engineering problem.
- Flip to classification loss: task has a fixed, well-defined label set and you don't need the resulting embeddings to generalize to unseen classes.

**Label smoothing vs. hard one-hot targets?**
- Default: label smoothing for large classifiers prone to overconfidence, especially with noisy labels. Constraint: prevents the model from driving logits to extremes, improving calibration.
- Failure mode: label smoothing can slightly hurt tasks where sharp, confident distinctions matter (e.g. distillation, where soft targets need to reflect real uncertainty, not artificial smoothing).
- Flip to hard targets: label noise is minimal and calibration isn't a concern (e.g. purely rank-based downstream use).

### Training infrastructure

**Data parallelism vs. model parallelism for distributed training?**
- Default: data parallelism whenever the model fits on a single device. Constraint: simpler to implement, near-linear scaling with more GPUs.
- Failure mode: data parallelism does nothing if the model itself doesn't fit in a single device's memory.
- Flip to model parallelism: model is too large for one device (modern large LLMs) — split layers/tensors across devices, typically combined with data parallelism at scale.

**Mixed precision (fp16/bf16) vs. full precision (fp32) training?**
- Default: mixed precision (bf16 preferred over fp16 for stability) for any large model training run. Constraint: roughly halves memory and speeds up matmul-heavy training on modern hardware with minimal accuracy loss.
- Failure mode: fp16 (not bf16) has a narrower dynamic range and can cause underflow/overflow without careful loss scaling.
- Flip to full fp32: numerical stability issues appear that mixed precision can't resolve even with loss scaling, or hardware doesn't support efficient mixed-precision kernels.

**Training from scratch vs. distillation from a larger teacher model?**
- Default: distillation when a strong teacher model already exists and you need a smaller, faster deployable student.
- Failure mode: student model quality is capped by teacher quality and by how well the distillation objective transfers the teacher's knowledge.
- Flip to training from scratch: no suitable teacher exists, or the deployment target's task is different enough from the teacher's original training that distillation transfer would be weak.

**Checkpointing frequency: aggressive (safe, slower) vs. sparse (faster, riskier)?**
- Default: aggressive checkpointing for long, expensive training runs (large LLMs) — losing hours/days of compute to a crash is far more costly than the storage/I-O overhead.
- Failure mode: very frequent checkpointing adds I/O overhead that can meaningfully slow down training at scale.
- Flip to sparse: training runs are short/cheap to restart from scratch, or storage I/O is a genuine bottleneck relative to how likely a crash is.

---

## Part 3 — LLMs

### Architecture & scaling

**Dense model vs. Mixture-of-Experts (MoE) for a fixed inference compute budget?**
- Default: MoE when you want more total parameters (capacity) without a proportional increase in FLOPs per token. Constraint: sparse activation (top-k experts per token) keeps inference compute roughly constant as parameter count grows.
- Failure mode: MoE needs a load-balancing auxiliary loss (router can collapse to favoring a few experts) and has a larger memory footprint even though FLOPs stay low.
- Flip to dense: memory (not compute) is the binding constraint, or the added routing complexity/instability isn't worth it for your training/serving infra maturity.

**Scaling parameters vs. scaling training tokens, given a fixed compute budget (Chinchilla-style tradeoff)?**
- Default: scale both roughly together per Chinchilla's compute-optimal ratio, rather than over-investing in parameters alone (as many pre-Chinchilla models did).
- Failure mode: an undertrained large model wastes compute — you get worse loss than a smaller model trained compute-optimally on more tokens.
- Flip toward more tokens over parameters: your deployment target has a hard memory/latency ceiling on model size — then it's better to over-train a smaller model past the "compute-optimal" point since you'll pay the inference cost many times over.

**RoPE vs. ALiBi vs. learned position embeddings for position encoding?**
- Default: RoPE — used in most modern production LLMs. Constraint: encodes relative position directly in the attention dot product, extends reasonably via interpolation/NTK scaling.
- Failure mode: naive RoPE extrapolation beyond trained context length degrades quality without explicit scaling techniques.
- Flip to ALiBi: extrapolation to much longer sequences than trained on is a priority and you want a technique designed for that by default. Flip to learned embeddings: fixed, known max sequence length and no need for length extrapolation.

**Decoder-only vs. encoder-decoder architecture for a given task?**
- Default: decoder-only for open-ended generation (chat, completion) — simpler, scales well, dominant in modern general-purpose LLMs.
- Failure mode: decoder-only models process input and output with the same causal mechanism, which can be less efficient for tasks with a clearly distinct "understand fixed input, then generate" structure (e.g. translation).
- Flip to encoder-decoder: task has a clear, bounded input to fully attend to bidirectionally before generating (translation, summarization) and you're not trying to build a single general-purpose model.

**Multi-head attention vs. multi-query attention (MQA) vs. grouped-query attention (GQA)?**
- Default: GQA — the pattern most production LLMs converge on. Constraint: groups of heads share K/V projections, cutting KV cache memory substantially with a smaller quality hit than MQA.
- Failure mode: GQA still gives up some quality vs. full multi-head attention, and the group size is another architecture hyperparameter to tune.
- Flip to full MHA: memory isn't the binding constraint (small model or short context) and you want maximum quality. Flip to MQA: KV cache memory is the dominant bottleneck (very long context, very high concurrency) and you can tolerate the larger quality hit.

### Adapting a model to a task

**RAG vs. fine-tuning — when does each win?**
- Default: RAG when knowledge is dynamic/frequently updated and answers need citations/traceability. Constraint: no retraining needed to incorporate new information, and you can point to the source.
- Failure mode: RAG adds a retrieval hop (latency) and quality is capped by retrieval recall — good generation over bad retrieval still fails.
- Flip to fine-tuning: knowledge is stable, the task needs a specific output format/style/behavior baked in, or low-latency inference (no retrieval hop) is required.

**Full fine-tuning vs. parameter-efficient fine-tuning (LoRA/QLoRA)?**
- Default: LoRA/QLoRA. Constraint: trains a low-rank update on top of frozen weights — far less memory and compute, and empirically matches full fine-tuning quality on most tasks since the needed weight update has low intrinsic rank.
- Failure mode: for tasks requiring very large behavioral shifts from the base model, a low-rank update may not have enough capacity to capture the needed change.
- Flip to full fine-tuning: task requires substantial changes to the model's underlying representations (not just adaptation) and you have the GPU budget to support it.

**Prompt engineering (few-shot/CoT) vs. fine-tuning?**
- Default: prompt engineering first — zero infra cost, instantly testable, no training pipeline needed.
- Failure mode: prompting has a ceiling; it can't teach genuinely new capabilities or reliably enforce complex formatting/behavior across many edge cases.
- Flip to fine-tuning: prompting plateaus below the required quality/consistency bar, or per-request prompt length (many few-shot examples) becomes a real latency/cost problem that fine-tuning would eliminate.

**Few-shot prompting vs. zero-shot with a well-engineered system prompt?**
- Default: zero-shot with a strong system prompt first — cheaper (fewer tokens per request), and modern instruction-tuned models often don't need examples for common tasks.
- Failure mode: zero-shot underperforms on tasks needing a specific output format/style the model hasn't reliably inferred from instructions alone.
- Flip to few-shot: zero-shot output is inconsistent or wrong on format/structure — a few well-chosen examples fix this faster than more instruction-tuning of the prompt.

**Fine-tuning your own model vs. calling a larger general-purpose API model?**
- Default: call the API model first — no training infra, immediate iteration, benefits from the frontier model's general capability.
- Failure mode: API model costs scale linearly with volume and you have no control over latency, availability, or model changes over time (vendor drift).
- Flip to your own fine-tuned model: request volume is high enough that per-token API cost dominates, you need guaranteed latency/availability SLAs, or data residency/privacy rules forbid sending data to a third-party API.

### Alignment

**RLHF (reward model + PPO) vs. DPO (direct preference optimization)?**
- Default: DPO for most teams today. Constraint: simpler, more stable training — no separate reward model, no RL loop, directly optimizes on preference pairs in closed form.
- Failure mode: DPO couples the reward signal directly into the policy — you can't inspect/adjust the reward function without retraining from the SFT checkpoint, and it's less studied for very large-scale, iterative alignment.
- Flip to RLHF: you need a reusable, inspectable reward model (e.g. to reuse across model versions or audit reward behavior separately), or you have the infra maturity to manage PPO's instability for a marginal quality edge.

**Human-labeled preference data vs. AI-generated feedback (RLAIF/Constitutional AI)?**
- Default: human-labeled data for the initial alignment pass — grounds the model in genuine human preference, no risk of amplifying an existing model's biases.
- Failure mode: human labeling is slow and expensive to scale, capping how much preference data you can generate.
- Flip to AI-generated feedback: you need to scale preference data volume beyond what human labeling budgets allow, and you can validate the AI-feedback model's judgments against a human-labeled sample to check for drift/bias.

**Optimizing hard for the reward model vs. accepting a KL penalty against the SFT model?**
- Default: accept a KL penalty (don't optimize the reward model to convergence). Constraint: prevents reward hacking — the policy exploiting reward-model blind spots (verbosity, sycophancy) rather than genuinely improving.
- Failure mode: too strong a KL penalty limits how much the policy can actually improve over the SFT baseline.
- Flip to weaker KL constraint: reward model is unusually robust/well-validated against known exploitation patterns, and you've verified via human eval that stronger optimization still correlates with real quality gains.

**Safety-first alignment (higher refusal rate) vs. maximizing helpfulness — how do you choose the operating point?**
- Default: err toward safety for consumer-facing, high-risk-surface products (open-ended chat with general public). Constraint: cost of a harmful output is asymmetric — usually far worse than an over-cautious refusal.
- Failure mode: over-refusal on benign requests measurably degrades usefulness and user trust ("alignment tax").
- Flip toward helpfulness: narrower, controlled deployment surface (internal tool, expert users, constrained domain) where false-refusal cost is high and harmful-output risk is genuinely lower.

### Inference & serving efficiency

**KV cache vs. recomputing attention every step (why does the tradeoff even exist)?**
- Default: always use KV cache for autoregressive generation. Constraint: recomputing K/V for all previous tokens at every generation step is wasteful — caching turns per-step cost from growing with sequence length into roughly constant new-token work.
- Failure mode: KV cache consumes growing GPU memory with sequence length and batch size — this, not compute, is usually the actual serving bottleneck at scale.
- Flip: the "flip" here is really MQA/GQA/quantization applied to shrink the cache, not abandoning caching itself — no realistic serving setup skips KV cache entirely.

**MQA/GQA (memory-efficient, some quality cost) vs. full multi-head attention?**
- Default: GQA in production serving. Constraint: KV cache memory is usually the binding constraint on concurrency/batch size, not raw quality.
- Failure mode: quality gap vs. full MHA, though usually small and accepted in exchange for serving more concurrent requests.
- Flip to full MHA: small model/short context where KV cache was never the bottleneck, and squeezing out maximum quality matters more than serving concurrency.

**Quantization (int8/int4) vs. full precision serving?**
- Default: int8 (or int4 with QLoRA-style careful calibration) for production serving. Constraint: roughly halves memory per precision step and speeds up memory-bound inference with graceful, usually-acceptable accuracy degradation.
- Failure mode: post-training quantization can degrade unpredictably on outlier-heavy layers if not validated per-task.
- Flip to full precision: serving cost isn't the bottleneck (low volume, ample GPU budget) and the task is sensitive enough to quality that even small degradation isn't acceptable — validate with quantization-aware techniques (GPTQ/AWQ) before ruling it out.

**Speculative decoding (draft + verify) vs. direct autoregressive decoding from the large model?**
- Default: speculative decoding whenever a smaller, faster draft model of the same family/tokenizer is available. Constraint: 2-4x wall-clock speedup with no change to the output distribution (verified via rejection sampling), so there's no quality tradeoff to accept.
- Failure mode: adds infra complexity (maintaining/serving a draft model) and the speedup shrinks if the draft model's acceptance rate is low (draft and target diverge often).
- Flip to direct decoding: no good draft model exists, or draft-target divergence is high enough that speculative decoding's overhead isn't worth the marginal speedup.

**Batching requests (throughput) vs. per-request latency?**
- Default: dynamic/continuous batching (e.g. as in vLLM) — batches opportunistically without forcing every request to wait for a full batch.
- Failure mode: naive static batching increases per-request latency (waiting for the batch to fill) in exchange for throughput.
- Flip to minimal/no batching: strict low-latency single-request SLA (e.g. interactive tool-use loop) where throughput matters far less than fast turnaround for each call.

**Smaller fine-tuned model vs. larger general model behind a prompt?**
- Default: larger general model behind a good prompt for prototyping and low-to-medium volume — fastest to iterate, no training pipeline.
- Failure mode: cost scales with volume and you're capped by the vendor's latency/availability and capability for capabilities outside its strengths.
- Flip to smaller fine-tuned model: high request volume where per-token cost dominates, or you need a narrow task done extremely reliably/fast and can afford the fine-tuning investment.

### RAG & retrieval

**Dense embedding retrieval vs. sparse/lexical retrieval (BM25)?**
- Default: dense embedding retrieval for semantic/paraphrase-tolerant matching. Constraint: captures meaning beyond exact keyword overlap.
- Failure mode: dense retrieval can miss exact-match signals (product IDs, rare technical terms, codes) that lexical search catches trivially.
- Flip to BM25 (or hybrid): queries are keyword/entity-heavy (search over structured/technical content), or you combine both (hybrid retrieval) since they fail on different query types.

**Single-stage retrieval vs. retrieve-then-rerank (cross-encoder)?**
- Default: add a reranking stage whenever retrieval quality (recall of the truly relevant chunk in top-k) is the bottleneck and p99 latency budget allows it.
- Failure mode: cross-encoder reranking adds real latency (scores each candidate jointly with the query) — a real cost for latency-sensitive paths.
- Flip to single-stage: p50-latency-critical path, or initial retrieval is already precise enough (small, well-curated corpus) that reranking's gain doesn't justify its cost.

**Small chunk size (precise, more retrieval calls) vs. large chunk size (more context, less precise)?**
- Default: chunk size tuned to the natural unit of meaning in the corpus (e.g. paragraph-level) as a starting point, then validated empirically against retrieval recall.
- Failure mode: too-small chunks lose surrounding context needed to answer correctly; too-large chunks dilute the embedding (relevant sentence buried in irrelevant text) and hurt retrieval precision.
- Flip smaller: corpus has dense, discrete facts (FAQ-style) where precision matters most. Flip larger: answers typically require multi-sentence context that a small chunk would fragment.

**Vector DB (ANN index) vs. exact nearest-neighbor search?**
- Default: ANN index (FAISS/Pinecone/pgvector with HNSW/IVF) for any corpus beyond a small number of vectors. Constraint: exact NN search is $O(n)$ per query — doesn't scale.
- Failure mode: ANN indexes trade a small amount of recall for large speed gains — occasionally miss the true nearest neighbor.
- Flip to exact search: corpus is small enough that exact search is still fast, and the recall risk of an approximate index isn't worth taking.

**Citation-grounded generation (refuse when retrieval confidence is low) vs. always answering?**
- Default: refuse or hedge when retrieval confidence is low, especially for high-stakes or factual domains. Constraint: reduces hallucination — a confidently wrong grounded-sounding answer is worse than an honest "I don't have enough information."
- Failure mode: overly conservative refusal thresholds degrade usefulness on legitimate queries retrieval happened to score lower on.
- Flip to always answering: low-stakes, exploratory/creative use case where a best-effort answer (clearly caveated as unverified) is more valuable than a refusal.

### Agentic systems

**Single LLM call vs. multi-step agent loop with tool use?**
- Default: single call whenever the task doesn't require external information or multi-step state (most factual/generative requests).
- Failure mode: agent loops add latency, cost (multiple LLM calls), and failure surface (tool errors, infinite loops) compared to one call.
- Flip to agent loop: task genuinely requires actions (searching, running code, calling APIs) or multi-step reasoning grounded in external state a single prompt can't supply.

**Fixed tool-calling budget vs. open-ended iteration until task completion?**
- Default: a fixed, bounded budget (max N tool calls/steps) with a graceful fallback if exhausted. Constraint: prevents runaway loops and unbounded cost/latency in production.
- Failure mode: a too-tight budget cuts off legitimately complex tasks before completion.
- Flip to open-ended: internal/offline tooling where cost and latency are not user-facing constraints, and task complexity is genuinely unpredictable.

**MCP-standardized tool integration vs. bespoke per-tool integration code?**
- Default: MCP (or an equivalent standard protocol) once you're integrating more than a couple of tools/models. Constraint: reduces the $N \times M$ integration problem (N models, M tools) to roughly $N + M$.
- Failure mode: adds a protocol/infra dependency and abstraction layer that's unnecessary overhead for a single, stable, one-off integration.
- Flip to bespoke: exactly one tool, one model, no expectation of adding more — a standard protocol's overhead isn't justified yet.

**Agent autonomy (fewer confirmation steps) vs. human-in-the-loop checkpoints?**
- Default: human-in-the-loop for any action with real-world side effects (sending messages, modifying data, spending money) — mirrors the same "risky action" judgment used for any automation.
- Failure mode: too many confirmation checkpoints defeats the purpose of automation and frustrates users on genuinely low-risk actions.
- Flip to more autonomy: action is reversible, low-blast-radius, and the user has explicitly pre-authorized the class of action in advance.

### Evaluation & production (LLMOps)

**LLM-as-judge vs. human evaluation vs. rubric-based automatic metrics?**
- Default: LLM-as-judge for fast, cheap, scalable iteration, validated periodically against a human-labeled sample. Constraint: scales to thousands of examples where human eval can't.
- Failure mode: LLM judges have their own biases (favoring longer/more confident-sounding answers) and can drift from true human preference if not periodically re-validated.
- Flip to human evaluation: final validation before a major launch, or the judge model itself is suspected to be miscalibrated on this specific task/domain.

**Prompt versioning discipline vs. treating prompts as disposable strings in code?**
- Default: version and track prompts like any other production artifact (model weights, code) with rollback capability.
- Failure mode: treating prompts as disposable inline strings means a silent prompt change can regress behavior with no audit trail — as damaging as an unversioned model change.
- Flip away from strict versioning: true throwaway prototyping/exploration that will never reach production.

**Fixed retrain/refresh cadence vs. trigger-based (drift/quality-threshold) refresh for an LLM-backed feature?**
- Default: fixed cadence when the underlying knowledge/behavior need drifts slowly and predictably (e.g. periodic RAG index refresh on a stable corpus).
- Failure mode: a fixed schedule either refreshes too rarely (misses a real shift) or wastes compute refreshing unchanged content.
- Flip to trigger-based: knowledge base or usage pattern changes are bursty/unpredictable (fast-moving product docs, adversarial usage) — a scheduled refresh would systematically lag real changes.

**Monitoring output-distribution/refusal-rate drift vs. classical feature/label drift monitoring?**
- Default: monitor output-distribution and refusal-rate drift specifically for generative systems — there's no single ground-truth label to compare against like classical ML.
- Failure mode: relying only on classical feature/label drift monitoring misses generative-specific failure signals (increased hallucination rate, rising refusal rate, format drift).
- Flip toward classical drift monitoring: the LLM is used as one component feeding a downstream classical model/decision with real labels available (then both should run together, not either/or).

---

## How to Practice This

1. Pick one question, cover the answer, set a 20-second timer, and reconstruct the 4-part shape from memory: default → constraint → failure mode → flip condition.
2. Don't skip the failure mode — "why is your default not just always correct" is usually the actual thing being tested.
3. Adapt these to your own experience: replace generic constraints with a specific project you've shipped wherever you can.
4. Come back weekly; the goal is a stable, defensible position you can state without hesitation, not memorized prose.

## Where to Next

- **Pre-written defaults + flip conditions in a production-system framing (fraud/ranking/etc.)** → [ROUND3-tradeoff-drills.md](05-round3-tradeoff-drills.md)
- **LLM-specific deep-dive material to ground your reasoning** → [09-study-plans/week-5-llm-deep-dive/README.md](study-plans/week-5-llm-deep-dive/README.md)
- **System design references by domain** → [06-production-ml/system-design/](../06-production-ml/system-design/)
