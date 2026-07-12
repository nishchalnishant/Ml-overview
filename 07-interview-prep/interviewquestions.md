---
module: Interview Prep
topic: Master Interview Questions
subtopic: Curated Consolidated Reference
status: unread
tags: [interviewprep, master-reference, ml, dl, llm, system-design, statistics, coding, behavioral]
---
# Master Interview Question Bank — Curated High-Signal Reference

Consolidated from every file in this folder. Not a duplicate dump — the sharpest, most
frequently-asked question per concept, organized for a single final drilling pass. When an
answer says "see X," that's the deep-dive if you need more than the paragraph here.

Organized by difficulty (Easy → Medium → Hard) so you can drill in order or target a tier.
Topic tags in brackets after each question help you cross-reference the original subject areas.

---

## Easy

Definitional, single-concept questions a junior candidate should be able to answer cleanly.

#### Q: What is the bias-variance tradeoff? [ML Fundamentals]
$$\mathbb{E}[(y-\hat f(x))^2] = \text{Bias}^2[\hat f] + \text{Var}[\hat f] + \sigma^2$$
High bias = underfitting (model too simple, misses signal; fix: more capacity/features). High
variance = overfitting (memorizes noise; fix: regularization, more data, simpler model). Total
error is what you minimize, not training error alone — there's an irreducible noise floor
$\sigma^2$ you can never remove.

#### Q: What is an embedding and why use one over one-hot encoding? [ML Fundamentals]
A dense, learned low-dimensional vector representation of a discrete entity (word, user ID,
item ID). One-hot vectors are sparse, high-dimensional, and encode no notion of similarity;
embeddings are trained so that similar entities end up close in vector space and scale far
better to high-cardinality categoricals (e.g., millions of user IDs).

#### Q: Precision vs. recall — when does each dominate the decision? [Model Evaluation]
$$\text{Precision}=\frac{TP}{TP+FP}\quad \text{Recall}=\frac{TP}{TP+FN}\quad F1=\frac{2PR}{P+R}$$
High precision matters when false positives are costly (spam filter blocking real email). High
recall matters when false negatives are costly (cancer screening missing a case). F1 balances
both when neither dominates; report the one that matches the business cost asymmetry, not F1 by
default.

#### Q: Why is accuracy misleading for imbalanced classification (99% accuracy trap)? [Model Evaluation]
If positives are 1% of data, predicting "all negative" gets 99% accuracy while catching zero
positives. Always check recall on the positive class and PR-AUC; a ROC-AUC of 0.5 confirms the
model is literally random despite the high accuracy number.

#### Q: Batch GD vs. SGD vs. mini-batch — tradeoffs? [Optimization]
Batch GD uses the full dataset per step — exact gradient, stable, but slow and memory-heavy. SGD
uses one sample — fast, noisy updates that can escape shallow local minima but with high
variance. Mini-batch (32–512) is the practical default — GPU-parallel-friendly, smooths noise
while staying fast; batch size interacts with learning rate (linear scaling rule) and
generalization (very large batches can generalize worse without adjustment).

#### Q: Backpropagation — the essential mechanism. [Deep Learning]
Forward pass computes and caches every layer's activations. Backward pass applies the chain rule
from the loss back to each parameter: $\partial L/\partial W^{(l)} = \delta^{(l)} \cdot
a^{(l-1)T}$ where $\delta^{(l)}$ is the error signal propagated backward. Activations must be
cached during the forward pass because computing $\partial z^{(l)}/\partial W^{(l)}$ requires
$a^{(l-1)}$.

#### Q: Why ReLU over sigmoid/tanh, and what is the dying ReLU problem? [Deep Learning]
ReLU $=\max(0,x)$ is non-saturating for $x>0$ (no vanishing gradient there), cheap to compute,
and induces sparse activations. Its flaw: if a neuron's input is always negative, its gradient
is permanently 0 and it stops updating ("dies"). Fixes: LeakyReLU/ELU/GELU (small negative
slope), careful initialization, lower learning rates.

#### Q: CNNs — why convolution and pooling instead of a fully-connected network on pixels? [Deep Learning]
Convolution exploits two image priors: **locality** (nearby pixels are correlated) and
**translation invariance** (a cat is a cat wherever it appears) via a small shared kernel that
slides across the image, drastically cutting parameters vs. a dense layer. Pooling (max/avg)
downsamples for local translation invariance and reduces computation for deeper layers.

#### Q: Batch vs. real-time (online) serving — how do you decide? [System Design & MLOps]
Decide from the latency SLA first. Batch (nightly/hourly scoring written to a store) suits
predictions that don't need instant freshness (churn score, email targeting) — cheap, simple,
easy to debug. Real-time (online inference, <100ms) is required when the prediction depends on
the current session/context (fraud at checkout, in-session recommendations) — needs a low-latency
feature store and a served model, at higher infra cost and complexity.

#### Q: BERT vs. GPT — why do encoder-only and decoder-only architectures exist separately? [LLMs]
BERT (encoder, bidirectional) is trained via **masked language modeling** — predict masked
tokens using context from both directions — ideal for understanding tasks (classification, NER,
embeddings) where the full input is available at once. GPT (decoder, causal/unidirectional) is
trained via **next-token prediction** with causal masking so each position only sees the past —
matches the actual generation task and enables autoregressive sampling. T5/BART (encoder-decoder)
combine both when you need to map one full sequence to another (translation, summarization).

#### Q: What is RAG and when do you reach for it over fine-tuning? [LLMs]
Retrieval-Augmented Generation retrieves relevant documents from an external store (via
embedding similarity search) and injects them into the prompt context before generation. Use RAG
when knowledge changes frequently or is too large to bake into weights, when you need citeable
sources, or when hallucination reduction on factual queries matters. Use fine-tuning instead when
you need to change *behavior/style/format* rather than inject *facts* — RAG and fine-tuning are
complementary, not substitutes.

#### Q: How does temperature affect sampling, and what's the difference from top-k/top-p? [LLMs]
Temperature $T$ rescales logits before softmax: $p_i \propto \exp(z_i/T)$. $T<1$ sharpens the
distribution (more deterministic/greedy-like); $T>1$ flattens it (more random/creative); $T\to0$
approaches greedy argmax. **Top-k** restricts sampling to the $k$ highest-probability tokens;
**top-p (nucleus)** restricts to the smallest set of tokens whose cumulative probability exceeds
$p$ — adapts the candidate pool size to the model's confidence, unlike fixed top-k.

#### Q: BPE tokenization — how does it work and why not just use whole words or characters? [LLMs]
Byte-Pair Encoding starts from individual characters/bytes and iteratively merges the most
frequent adjacent pair into a new token, building a vocabulary of subword units. Whole-word
vocabularies explode in size and can't handle unseen words (OOV problem); character-level avoids
OOV but produces very long sequences and weak per-token semantics. BPE balances both — a fixed,
manageable vocabulary that handles rare/unseen words by falling back to subword/byte pieces.

#### Q: Bagging vs. boosting — what does each reduce, and name a canonical example of each. [Classical Algorithms]
Bagging trains models independently in parallel on bootstrap samples and averages them —
reduces **variance** (Random Forest). Boosting trains models sequentially, each correcting the
previous ensemble's residual errors — reduces **bias**, at higher overfitting risk if run too
long (XGBoost, LightGBM, AdaBoost). Bagging parallelizes trivially; boosting is inherently
sequential.

#### Q: K-Means — algorithm, weaknesses, and how DBSCAN differs. [Classical Algorithms]
K-Means: initialize $k$ centroids, alternate assigning points to nearest centroid and
recomputing centroids as cluster means, until convergence. Weaknesses: must pick $k$ upfront,
assumes spherical/similar-sized clusters, sensitive to initialization (use k-means++) and
outliers. **DBSCAN** instead groups points by density (core points with enough neighbors within
$\epsilon$), automatically finds the number of clusters, handles arbitrary shapes, and natively
labels sparse points as noise/outliers — but struggles with varying-density clusters and needs
$\epsilon$/min-samples tuning.

#### Q: Which algorithms require feature scaling and why? [Data Preprocessing]
Distance/gradient-based algorithms need it: KNN and SVM (distance/margin computed directly on
feature magnitudes), logistic regression and neural nets (unscaled features cause imbalanced
gradient magnitudes, slow/unstable convergence). Tree-based models (decision trees, Random
Forest, XGBoost) don't need scaling — splits depend only on feature *rank order*, not magnitude.

#### Q: What is data leakage and give three concrete examples. [Data Preprocessing]
Leakage is any signal available during training/validation that wouldn't be available at true
prediction time. Examples: (1) fitting a scaler/encoder on the full dataset before the
train/test split, (2) including a feature computed using data from after the label window (e.g.,
"total purchases this month" as a feature for a mid-month churn label), (3) row-level (not
group-level) splitting when multiple rows belong to the same entity, letting that entity's
identity/behavior leak from train into validation.

#### Q: State Bayes' theorem and walk through the medical-test base-rate fallacy. [Probability & Statistics]
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$
Classic trap: a test is 99% sensitive and 99% specific, disease prevalence is 1%. Despite the
test sounding highly accurate, $P(\text{disease}|\text{positive}) \approx 50\%$ — because the
number of false positives from the huge healthy population rivals the true positives from the
tiny diseased population. The lesson: a test's accuracy alone tells you nothing without the base
rate (prior).

#### Q: Type I vs. Type II error, and how do you trade them off? [Probability & Statistics]
Type I ($\alpha$): rejecting a true null (false positive). Type II ($\beta$): failing to reject a
false null (false negative); power $=1-\beta$. Lowering $\alpha$ (stricter significance
threshold) reduces false positives but increases false negatives for fixed sample size — the
only way to reduce both simultaneously is to increase sample size/statistical power.

#### Q: TF-IDF — what does it measure and why divide by document frequency? [NLP]
$$\text{TF-IDF}(t,d) = tf(t,d) \times \log\frac{N}{df(t)}$$
Term frequency captures local importance within a document; dividing by (log of) document
frequency downweights terms that appear in many documents (like "the," "is") since they carry
little discriminative signal, while upweighting rare, topic-specific terms. Still a strong
baseline for search/retrieval before or alongside embeddings.

#### Q: Stemming vs. lemmatization — what's the practical tradeoff? [NLP]
Stemming crudely chops word endings via rules (e.g., "running" → "run" but also "universe" →
"univers") — fast but can produce non-words and over/under-stem. Lemmatization uses
vocabulary/morphological analysis to return the true dictionary base form (needs POS context,
e.g., "better" → "good") — more accurate but slower and requires more linguistic resources.
Modern subword-tokenization pipelines (BPE) have made both largely unnecessary for neural
models, but they still matter for classical IR/BoW pipelines.

#### Q: IoU and mAP — how is object detection quality measured? [Computer Vision]
**IoU** (Intersection over Union) $= \frac{\text{area of overlap}}{\text{area of union}}$ between
a predicted box and ground truth; a detection counts as a true positive only if IoU exceeds a
threshold (commonly 0.5). **mAP** averages precision across recall levels (area under the
precision-recall curve) for each class, then averages across classes — mAP@0.5 and mAP@[0.5:0.95]
(COCO-style, averaged over multiple IoU thresholds) are the standard reported numbers.

#### Q: Implement a numerically stable softmax — what breaks in the naive version? [Coding Patterns]
Naive `exp(x) / sum(exp(x))` overflows for large `x` (exp of large numbers → inf). Fix: subtract
the max before exponentiating — mathematically identical, numerically safe.
```python
def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
```

#### Q: What is differential privacy and what does the privacy budget $\epsilon$ control? [Privacy & Fairness]
A randomized mechanism is $\epsilon$-differentially private if its output distribution changes by
at most a factor of $e^\epsilon$ whether or not any single individual's record is included in the
dataset — formally bounding how much any one person's data can influence (and thus be inferred
from) the output. Smaller $\epsilon$ = stronger privacy but more noise added = worse
utility/accuracy — it's a tunable privacy/utility tradeoff, not a binary property.

#### Q: What is membership inference and what does it expose? [Privacy & Fairness]
An attack that asks: "was this specific record part of the training set?" — exploiting the fact
that models tend to be more confident/lower-loss on training examples than unseen ones. It
matters because a "yes" answer alone can leak sensitive information (e.g., inferring someone was
in a hospital's cancer-patient training set reveals their diagnosis), independent of any explicit
data leak — models trained on sensitive data need to be audited for this even if the training
data itself is never exposed.

#### Q: What is the difference between parametric and non-parametric models? [ML Fundamentals]
Parametric models (linear/logistic regression, a fixed-size neural net) assume a fixed functional
form with a bounded number of parameters that doesn't grow with the amount of training data —
fast to train/predict, more bias if the assumed form is wrong. Non-parametric models (KNN,
kernel SVM, decision trees) let model complexity grow with the data (e.g., KNN effectively stores
all training points) — more flexible, can fit arbitrary decision boundaries, but slower at
inference and more prone to overfitting without care.

#### Q: What is the curse of dimensionality? [ML Fundamentals]
As the number of features grows, the volume of the feature space grows exponentially, so data
points become sparse and average pairwise distances converge — nearest-neighbor notions of
similarity break down, and exponentially more data is needed to maintain the same sample density.
Practical consequences: KNN/clustering degrade in very high dimensions, and dimensionality
reduction (PCA, feature selection) or regularization becomes necessary before applying
distance-based methods.

---

## Medium

Require connecting two or more concepts, explaining a tradeoff, or applying an idea to a scenario.

#### Q: How do you diagnose overfitting vs underfitting from learning curves? [ML Fundamentals]
Both train and val loss high → underfitting (add capacity/features). Train low, val high →
overfitting (regularize, more data). Both low and close → generalizing. Plot a learning curve
(loss vs. training-set size): a persistent gap at large $n$ signals high variance; a small gap
with both curves plateaued high signals high bias.

#### Q: Why must train/val/test be split, and what's the #1 leakage mistake? [ML Fundamentals]
Train fits parameters, validation tunes hyperparameters/selects models, test gives one unbiased
final read — touched exactly once. The classic mistakes: tuning on the test set (optimistic
bias) and using future data in training for time series (temporal leakage) — always use a
chronological split for sequential data.

#### Q: L1 vs L2 regularization — mechanism and when to prefer each? [ML Fundamentals]
L1 (Lasso) adds $\lambda\sum|w_i|$; its diamond-shaped constraint region touches axes, driving
some weights exactly to 0 → sparse, built-in feature selection. L2 (Ridge/weight decay) adds
$\lambda\sum w_i^2$; its circular constraint shrinks weights smoothly without zeroing them —
better when all features are somewhat relevant and you want stability with correlated
predictors. Elastic Net combines both.

#### Q: Explain dropout and why the inverted-scaling trick exists. [ML Fundamentals]
At training time, each unit is zeroed with probability $1-p$, forcing the network to not rely on
any single neuron (an implicit ensemble). At test time you want the expected output to match
training — "inverted dropout" scales surviving activations by $1/p$ during training so no
rescaling is needed at inference, keeping eval code simple and fast.

#### Q: ROC-AUC vs. PR-AUC — why does the choice matter for imbalanced data? [Model Evaluation]
ROC-AUC plots TPR vs. FPR; FPR has a huge true-negative count in the denominator for rare
positives, so ROC-AUC stays deceptively high even for a nearly-useless classifier. PR-AUC (P vs.
R) has no TN term — it directly reflects how well you find rare positives without flooding
alerts with false ones. A random classifier's PR-AUC ≈ class prevalence, giving a meaningful
baseline; use PR-AUC whenever positives are rare (fraud, disease, churn).

#### Q: How do you do cross-validation correctly with preprocessing (leakage-free)? [Model Evaluation]
Preprocessing (scaler, imputer, encoder) must be fit **inside** each fold's training data only,
via a `Pipeline`/`ColumnTransformer`, and applied unchanged to that fold's validation data.
Fitting the scaler on the full dataset before splitting leaks validation-fold statistics (mean,
std) into training, inflating the CV score.

#### Q: Stratified K-Fold vs. GroupKFold vs. TimeSeriesSplit — when is each mandatory? [Model Evaluation]
Stratified K-Fold preserves class ratio per fold — use whenever there's class imbalance.
GroupKFold ensures all rows from one entity (user/player) stay in one fold — mandatory whenever
there are multiple rows per entity, otherwise identity leaks across train/val and inflates the
score. TimeSeriesSplit (rolling-origin, forward-only) is mandatory for any temporal prediction
task — random splits leak future into the past.

#### Q: What are NDCG and MAP, and why does NDCG beat plain Precision@K for ranking? [Model Evaluation]
Precision@K ignores the order within the top K. **NDCG@K** $= DCG@K / IDCG@K$ where
$DCG = \sum_i (2^{rel_i}-1)/\log_2(i+1)$ — rewards relevant items appearing earlier and supports
graded (not just binary) relevance, matching how users actually scan a ranked list. **MAP**
averages precision at each relevant-item rank position, good for binary relevance across many
queries.

#### Q: How does Adam work, and why does AdamW fix a real bug in it? [Optimization]
Adam maintains per-parameter first ($m$) and second ($v$) moment estimates of the gradient with
bias correction, giving each parameter an adaptive effective learning rate:
$\theta \mathrel{-}= \eta \hat m/(\sqrt{\hat v}+\epsilon)$. Plain Adam applies L2 regularization
by adding $\lambda\theta$ to the gradient *before* the adaptive scaling, so weight decay gets
divided by $\sqrt{\hat v}$ — an unintended, parameter-dependent decay strength. **AdamW**
decouples weight decay, applying it directly to the parameter update rather than through the
gradient, matching true L2/weight-decay behavior and generally improving generalization.

#### Q: Vanishing/exploding gradients — causes and fixes? [Optimization]
Cause: repeated multiplication of derivatives (deep nets, long RNN sequences) either shrinks
gradients toward zero (saturating activations like sigmoid/tanh, poor init) or blows them up
(large weights, no normalization). Fixes: ReLU-family activations, proper init (Xavier/He),
batch/layer normalization, residual connections (identity shortcuts preserve gradient flow),
gradient clipping for explosion, and LSTM/GRU gating for sequence models.

#### Q: Learning rate scheduling — why not just use a constant LR? [Optimization]
A high constant LR converges fast early but oscillates/fails to settle near the minimum; a low
constant LR is stable but slow. Warmup (linear ramp-up) avoids early instability when adaptive
optimizer moment estimates are still noisy; cosine decay or step decay then reduces LR as
training progresses to fine-tune into a sharp/good minimum. LR is widely considered the single
highest-leverage hyperparameter — an LR range test finds a good starting value fast.

#### Q: Why is random search better than grid search for hyperparameter tuning? [Optimization]
(Bergstra & Bengio) When only a few hyperparameters actually matter, grid search wastes budget
evenly across all dimensions including unimportant ones, while random search samples each
dimension independently and explores the important ones more densely for the same budget.
Bayesian optimization (Optuna/TPE) is preferred when each trial is very expensive (large DL
training runs), using past trial outcomes to pick the next point intelligently.

#### Q: Why does batch normalization help training? [Deep Learning]
Normalizes layer inputs to zero mean/unit variance per mini-batch, then applies a learned
scale/shift ($\gamma,\beta$) to preserve representational power. Reduces internal covariate
shift, allows higher learning rates, and adds a mild regularizing noise from batch statistics.
Fails with very small batches or variable-length sequences — **LayerNorm** (normalize per-sample
across features, batch-size independent) is the standard replacement in Transformers.

#### Q: Residual connections — why do they let networks go much deeper? [Deep Learning]
A residual block computes $y = F(x) + x$ instead of just $F(x)$. This gives the gradient a
direct identity path back to earlier layers during backprop, avoiding the vanishing-gradient
degradation that plain deep stacks suffer, and lets the network default to learning an identity
mapping (safe fallback) rather than forcing every block to be useful.

#### Q: Why did Transformers largely replace RNNs? [Deep Learning]
RNNs process sequentially — $O(T)$ depth, cannot parallelize across time, and long-range
dependencies degrade with distance. Transformers use self-attention giving every position direct
$O(1)$-hop access to every other position, fully parallelizable across the sequence during
training, at the cost of $O(T^2 d)$ complexity vs. RNN's $O(Td^2)$ — quadratic in sequence length
instead of linear, which matters at very long context lengths.

#### Q: Walk through the universal ML system design framework. [System Design & MLOps]
1) Clarify requirements: scale (QPS, users, items), latency SLA, online vs. offline. 2) Define
the metric — both offline (AUC/NDCG) and online/business (CTR, revenue). 3) Data pipeline —
sources, labels, feature computation, point-in-time correctness. 4) Model — start simple
(baseline), justify complexity. 5) Serving architecture — batch vs. real-time, caching. 6)
Monitoring — drift, feature health, delayed labels. 7) Iteration — retraining cadence, feedback
loops. State the bottleneck explicitly before being asked.

#### Q: Describe the two-stage retrieval + ranking pattern and why it's needed. [System Design & MLOps]
Running a heavy ranking model over millions of items per request is too slow. Stage 1 (retrieval)
uses a cheap method — ANN search (FAISS) over two-tower embeddings, or BM25 — to cut millions
down to ~100s–1000s of candidates. Stage 2 (ranking) applies an expensive, feature-rich model
(LambdaMART, DCN, DIN) only to those candidates. This trades a small recall loss at stage 1 for
orders-of-magnitude lower cost, making systems like search and recsys serve at scale.

#### Q: What is training-serving skew and how do you prevent it? [System Design & MLOps]
Skew happens when a feature is computed differently offline (training) than online (serving) —
e.g., training uses a batch job with full historical context while serving computes the feature
in real time with a slightly different definition or a "leak" of future data. It's the #1
real-world cause of a model that looks great offline and fails in production. Prevention: a
**feature store** with shared feature-computation logic and strict point-in-time correctness for
both training and serving paths.

#### Q: How do you detect and respond to data/concept drift in production? [System Design & MLOps]
Feature (covariate) drift: $P(X)$ shifts — detect via PSI or KS test on feature distributions
(PSI < 0.1 stable, 0.1–0.25 monitor, > 0.25 retrain). Concept drift: $P(Y|X)$ shifts — the same
inputs now map to different outputs; detect by monitoring live accuracy on a delayed-labeled
slice, since ground truth often lags (e.g., churn labels arrive weeks later). Label drift:
$P(Y)$ shifts — monitor prediction score distribution over time. Response: alert, then retrain
on a schedule tied to detected drift, not a fixed calendar.

#### Q: Canary vs. shadow deployment — what's the difference and when do you use each? [System Design & MLOps]
**Shadow**: the new model runs in parallel on live traffic, its predictions logged but never
served to users — zero user risk, used to validate behavior/latency before any exposure.
**Canary**: the new model is served to a small % of real traffic, monitored against the
incumbent, then rolled out gradually if healthy — used once shadow testing passes and you need
to validate actual user-facing impact/business metrics.

#### Q: How do you monitor an ML system with no immediate ground truth (e.g., churn label arrives weeks later)? [System Design & MLOps]
Layer monitoring: (1) infra health (latency, errors — standard SRE), (2) model health via proxy
signals available immediately — prediction score distribution drift (PSI), feature distribution
drift, confidence calibration — since you can't compute live accuracy yet, and (3) business KPI
proxies that correlate with the eventual label. Once delayed labels arrive, backfill true
accuracy and reconcile against the proxy signals to validate they were predictive of the real
metric.

#### Q: Why do positional encodings exist, and how does RoPE differ from sinusoidal encoding? [LLMs]
Self-attention is permutation-invariant — without position info, "dog bites man" and "man bites
dog" would look identical. Sinusoidal encoding adds a fixed sin/cos pattern per position to the
embedding (original Transformer) — fixed, doesn't generalize length well beyond training range.
**RoPE** instead rotates Q/K vectors by an angle proportional to absolute position, so the dot
product $QK^T$ naturally encodes *relative* position — better extrapolation to longer sequences
and is now the default in most modern LLMs (LLaMA, GPT-NeoX family).

#### Q: What is LoRA and why does it work with so few trainable parameters? [LLMs]
Full fine-tuning updates every weight matrix $W$. LoRA freezes $W$ and learns a low-rank update
$\Delta W = BA$ where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$, $r \ll d$.
This works because fine-tuning weight updates empirically have low "intrinsic rank" — the
adaptation needed for a new task lives in a small subspace. Trainable params drop to ~0.05–0.1%
of the base model, cutting memory/storage while matching full fine-tuning quality on most tasks.
Merges back into $W$ at inference with zero added latency.

#### Q: RLHF vs. DPO — what problem does each solve and what does DPO simplify away? [LLMs]
RLHF: train a reward model on human preference pairs, then optimize the policy against it with
PPO — powerful but complex (reward model + RL loop, unstable, expensive). **DPO** reframes the
same preference-optimization objective as a single classification-style loss directly on the
policy, using a closed-form relationship between the optimal RLHF policy and the reward function
— removing the separate reward model and the RL training loop entirely while empirically
matching RLHF's alignment quality with far more stable, simpler training.

#### Q: What causes LLM hallucination and what are the main mitigations? [LLMs]
Hallucination arises because LLMs are trained to produce plausible continuations, not verified
truth — the objective (next-token likelihood) doesn't distinguish a confident-sounding fabrication
from a fact. Mitigations: RAG (ground responses in retrieved evidence), lower temperature for
factual tasks, prompting the model to cite sources / say "I don't know," self-consistency
checks (sample multiple times, check agreement), and RLHF fine-tuning that penalizes confident
wrong answers.

#### Q: How does XGBoost/gradient boosting work mathematically? [Classical Algorithms]
Builds an additive model $F_m(x) = F_{m-1}(x) + \alpha h_m(x)$ where each new shallow tree
$h_m$ is fit to the negative gradient (residual) of the loss w.r.t. the current ensemble's
predictions — effectively gradient descent in function space. XGBoost adds L1/L2 regularization
on leaf weights, a second-order (Newton) approximation of the loss for faster/better splits, and
efficient handling of sparsity/missing values.

#### Q: PCA — what does it optimize and how does it relate to SVD? [Classical Algorithms]
PCA finds orthogonal directions (principal components) that maximize retained variance, ranked
by eigenvalue of the (centered) data's covariance matrix — the top-$k$ eigenvectors give the
best $k$-dimensional linear reconstruction under squared error. In practice PCA is computed via
SVD of the centered data matrix $X = U\Sigma V^T$: the columns of $V$ are the principal
components and $\Sigma$'s singular values relate to the eigenvalues ($\lambda_i = \sigma_i^2/(n-1)$)
— SVD avoids explicitly forming the covariance matrix and is more numerically stable.

#### Q: Random Forest vs. XGBoost — when would you pick one over the other? [Classical Algorithms]
Random Forest: robust out-of-the-box, harder to overfit, trivially parallel, fewer hyperparameters
to tune, but usually slightly lower ceiling accuracy. XGBoost/LightGBM: typically higher accuracy
with proper tuning, handles missing values and imbalanced data better with built-in options, but
more hyperparameters, more overfitting risk, and sequential (slower) training. Default to a GBT
when squeezing max accuracy on tabular data matters and you have time to tune; default to Random
Forest for a fast, low-maintenance baseline.

#### Q: How do you handle high-cardinality categorical features? [Data Preprocessing]
Options ranked by cardinality: one-hot for low cardinality (<~10-20 categories); target/mean
encoding (with cross-fold smoothing to prevent leakage) or frequency encoding for medium
cardinality; learned embeddings for high cardinality (user ID, item ID) especially in neural
nets, since one-hot would blow up dimensionality and add no notion of similarity between
categories.

#### Q: How do you handle severe missing data, and why does mean/median imputation alone lose information? [Data Preprocessing]
Simple imputation (mean/median/mode) is a reasonable default but silently discards the signal
that "this value was missing" — which is often informative itself (e.g., a sensor failure
correlates with the outcome). Add an explicit `is_missing` indicator column alongside the
imputed value. For MCAR data with low missingness, mean/median is fine; for MAR/MNAR patterns
or high missingness, consider model-based imputation (KNN, MICE) or a missingness-aware model
(trees natively handle NaN in LightGBM/XGBoost).

#### Q: What techniques address severe class imbalance, in order of effort? [Data Preprocessing]
1) `class_weight="balanced"` — free, try first. 2) Threshold tuning on the cost matrix instead of
0.5. 3) Resampling — SMOTE (synthetic oversampling of the minority class) for tabular data with
low minority counts, or undersampling the majority when data is abundant. 4) Focal loss in deep
learning — down-weights easy/majority examples so the loss focuses on hard/minority ones. Always
evaluate with PR-AUC/recall on the positive class, never accuracy.

#### Q: MLE vs. MAP — what's the difference and how does MAP relate to regularization? [Probability & Statistics]
MLE picks parameters maximizing $P(\text{data}|\theta)$ — no prior belief about $\theta$. MAP
maximizes $P(\theta|\text{data}) \propto P(\text{data}|\theta)P(\theta)$ — incorporates a prior.
With a Gaussian prior on weights, MAP's log-posterior objective becomes NLL plus an L2 penalty
$\frac{1}{2\sigma^2}\|\theta\|^2$ — i.e., **L2 regularization is exactly MAP estimation with a
Gaussian prior**; a Laplace prior similarly yields L1/Lasso.

#### Q: What do the Central Limit Theorem and Law of Large Numbers each guarantee, and when does CLT fail? [Probability & Statistics]
LLN: the sample mean converges to the true population mean as $n\to\infty$. CLT: the sampling
distribution of the mean approaches a Normal distribution as $n\to\infty$, *regardless of the
population's original distribution* — this is what justifies z-tests/confidence intervals on
sample means. CLT fails (or needs much larger $n$) for heavy-tailed distributions (infinite
variance, e.g., certain power laws), for rare-event/count data at low $n$ (use exact
binomial/Poisson tests instead), and when samples aren't i.i.d.

#### Q: How do you correctly explain a p-value to a non-technical stakeholder? [Probability & Statistics]
A p-value is the probability of seeing data this extreme (or more) **if the null hypothesis were
true** — it is *not* the probability the null hypothesis is true, nor the probability the result
is due to chance in a colloquial sense. Common misuse: treating $p<0.05$ as "99% confidence the
effect is real," or "no effect" when $p>0.05$ (absence of evidence isn't evidence of absence,
especially underpowered tests).

#### Q: Bootstrap — how does it estimate a confidence interval without distributional assumptions? [Probability & Statistics]
Resample the observed data **with replacement** $B$ times (same size as original), compute the
statistic of interest (mean, median, AUC) on each resample, and use the resulting distribution's
percentiles (e.g., 2.5th/97.5th) as the CI. Makes no parametric assumption about the underlying
distribution — useful for statistics with no simple closed-form variance (e.g., median, or AUC
across CV folds).

#### Q: Word2Vec skip-gram — what's the training objective and why negative sampling? [NLP]
Skip-gram predicts context words from a center word, learning embeddings such that words
appearing in similar contexts end up close in vector space (distributional hypothesis). The
exact softmax over the whole vocabulary is too expensive per step; **negative sampling**
approximates it by turning it into a binary classification task — distinguish true
(center,context) pairs from a handful of randomly sampled "negative" (non-context) pairs — making
training tractable at web scale.

#### Q: One-stage vs. two-stage object detectors — the speed/accuracy tradeoff. [Computer Vision]
Two-stage (Faster R-CNN): first proposes candidate regions (RPN), then classifies/refines each —
higher accuracy, slower. One-stage (YOLO, SSD): predicts boxes and classes directly in a single
dense pass over the image — much faster (real-time capable), historically somewhat lower accuracy
though the gap has largely closed with modern YOLO versions. Choose one-stage for latency-critical
applications (video, robotics), two-stage when accuracy dominates and latency budget is looser.

#### Q: Why do CNNs need data augmentation, and name augmentations that preserve label validity. [Computer Vision]
CNNs are data-hungry and prone to overfitting on limited labeled image data; augmentation
synthetically expands the effective dataset by applying label-preserving transforms — random
crop, flip, rotation, color jitter, cutout/mixup — teaching invariance to nuisance variation the
model will see at test time without needing more labeled examples. Must ensure the
transformation doesn't change the true label (e.g., don't flip a "6" horizontally—it becomes
ambiguous with certain digit tasks).

#### Q: Implement K-Means from scratch — what's the convergence loop? [Coding Patterns]
```python
def kmeans(X, k, n_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids
```
Common trap: empty clusters after an update (no points assigned) — must handle by reinitializing
that centroid, e.g., to the point farthest from its current centroid.

#### Q: Write a time-series-safe cross-validation split — what's wrong with random K-fold here? [Coding Patterns]
Random K-fold lets future rows train a model evaluated on the past, leaking information. Correct
approach: rolling-origin / forward-chaining splits.
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# fold 1: train on weeks 1-10, test on week 11
# fold 2: train on weeks 1-11, test on week 12  ...
```

#### Q: Write a minimal but correct PyTorch training loop — what do people forget? [Coding Patterns]
```python
model.train()
for x, y in loader:
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```
Commonly forgotten: `optimizer.zero_grad()` (gradients accumulate otherwise), switching to
`model.eval()` + `torch.no_grad()` for validation (dropout/batchnorm behave differently and you
waste memory tracking gradients), and saving/loading both `model.state_dict()` and
`optimizer.state_dict()` to resume training correctly (optimizer state like Adam's moments
matters for smooth resumption).

#### Q: Implement self-attention from scratch in NumPy. [Coding Patterns]
```python
def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)          # row-wise softmax
    return weights @ V
```
Sanity check with a trivial input (e.g., 2 tokens, d_k=2) before trusting it on real data; state
time complexity ($O(n^2 d)$) when done.

#### Q: Model accuracy dropped overnight in production — how do you diagnose it? [Practical/Scenario]
Structure the investigation, don't guess: (1) check for an upstream data/pipeline change (schema
change, a feature pipeline job failing silently, null-filling a previously populated column), (2)
check for real-world distribution shift (feature drift via PSI, has user behavior genuinely
changed), (3) check the serving stack itself (model version rollback, a bad deploy, a stale
cached model), (4) only then suspect data drift/concept drift needing retraining. Cheap,
fast-to-check causes first — an infra bug is far more likely than sudden concept drift.

#### Q: You get 99% accuracy but the model is useless in production — what's happening and how do you find it? [Practical/Scenario]
Almost certainly severe class imbalance and a trivial majority-class predictor. Check recall on
the positive class first — if it's ~0, the model predicts one class always. Check PR-AUC (a
random baseline PR-AUC ≈ prevalence) rather than ROC-AUC, which stays deceptively high because
FPR's denominator is dominated by the huge true-negative count.

#### Q: Model works great in testing but fails in production — what's your checklist? [Practical/Scenario]
Check, in order: (1) training-serving skew — is the feature computed identically online/offline;
(2) temporal leakage — was the offline test split randomized when it should have been
chronological, letting future information leak into training; (3) population shift — does
production traffic differ from the historical training distribution (new users, new region); (4)
silent data pipeline failures feeding stale or null features online.

#### Q: Real-time recommendation system has high latency — what do you check, in order? [Practical/Scenario]
Profile before optimizing blindly: (1) is it feature computation (slow online joins, uncached hot
features) or model inference (large model, no batching) — measure each stage; (2) apply the
two-stage retrieval+ranking pattern if not already in place to cut the candidate set before the
expensive model runs; (3) cache frequently requested predictions/features; (4) consider model
distillation/quantization if inference itself is the bottleneck; (5) only then consider
infrastructure scaling (more replicas), since that treats the symptom, not the cause.

#### Q: How would you design a real-time fraud scoring system end to end? [Practical/Scenario]
Streaming feature computation (Kafka/Flink) joining real-time signals (transaction velocity,
device fingerprint) with precomputed historical features from a feature store; an ensemble of a
fast rules engine (catches known patterns instantly, cheap) plus a ML model (catches novel
patterns) scored within a tight latency budget (~50-100ms); a human review queue for
borderline/high-value cases; explainability (SHAP) attached to flagged transactions for analyst
review and regulatory compliance; tight feedback loop since confirmed fraud labels arrive with
delay and must be used to retrain/recalibrate the threshold.

#### Q: "Tell me about a time a model you built failed in production." [Behavioral]
Use STAR, but the "Result" must include what changed afterward, not just the outcome. Strong
answers name a specific mechanism (training-serving skew, a distribution shift the offline eval
didn't catch, a leakage bug caught late), quantify the impact if possible, and end with the
concrete process/monitoring change made afterward — interviewers are listening for genuine
learning, not just "we fixed it."

#### Q: "Tell me about a technical disagreement with a teammate." [Behavioral]
Pick a real disagreement with technical substance (not a personality conflict), show you
understood the other position's merits before pushing back, describe how it was resolved
(data/experiment settling it is the strongest resolution, not seniority or persistence), and be
honest if you were the one who was wrong — that self-awareness reads better than a story where
you're always right.

#### Q: "Describe a project that failed or had to pivot — what did you learn?" [Behavioral]
Be specific about the failure mode (wrong metric chosen, underestimated data quality issues,
scope misjudged) and show the retrospective insight changed how you approach subsequent projects
— generic "we learned to communicate better" answers read as rehearsed; a precise technical or
process lesson lands better.

#### Q: "Where do you see AI/ML going in the next five years?" [Behavioral]
Authenticity beats a rehearsed industry-trends recitation — anchor the answer in something you
actually find interesting (a specific capability, failure mode, or research direction) and
connect it to how it might change the kind of work you'd want to do, rather than a generic list
of buzzwords (agents, multimodality, reasoning).

#### Q: How does DP-SGD make model training differentially private? [Privacy & Fairness]
Two changes to standard SGD: (1) **per-example gradient clipping** — clip each individual
example's gradient norm to a fixed threshold $C$, bounding any single example's maximum influence
on the update; (2) **calibrated noise addition** — add Gaussian noise (scaled to $C$ and the
target $\epsilon,\delta$) to the summed/averaged clipped gradients before the parameter update.
The tradeoff: stronger privacy (more noise, smaller $C$) directly degrades model accuracy and
convergence speed — larger batch sizes help offset the added noise's relative impact.

#### Q: Name the three major group-fairness metrics and why you can't satisfy all of them simultaneously. [Privacy & Fairness]
**Demographic parity**: positive prediction rate equal across groups. **Equalized odds**: TPR and
FPR equal across groups (conditioned on true label). **Predictive parity**: precision (PPV) equal
across groups. The **impossibility theorem** (Chouldechova/Kleinberg et al.) shows that when base
rates differ between groups, you cannot simultaneously satisfy equalized odds and predictive
parity except in degenerate cases — fairness is a set of tradeoffs requiring an explicit choice
of which metric matters for the specific harm being prevented, not a single objective.

#### Q: What is machine unlearning and why is simply deleting a row from the training set not sufficient? [Privacy & Fairness]
The "right to be forgotten" requires removing an individual's *influence* on a trained model, not
just their raw data row — a trained model's weights already encode information derived from
that row, so it can still be partially reconstructed/inferred from the model even after the row
is deleted from storage. Exact unlearning (retraining from scratch without that row) is
guaranteed-correct but expensive; approximate unlearning methods (influence-function-based weight
updates, SISA sharded retraining) trade some correctness guarantee for far lower cost.

#### Q: How do fairness concerns manifest specifically in LLMs, beyond classical group fairness? [Privacy & Fairness]
Beyond disparate outcome rates across demographic groups, LLMs raise: representational harms
(stereotyped or demeaning associations in generated text), allocational harms when LLM outputs
feed downstream decisions (resume screening, credit), performance disparities across languages/
dialects (much less training data for low-resource languages), and harder auditability since the
"outcome" is open-ended generated text rather than a single scalar prediction — standard
fairness metrics (demographic parity, equalized odds) don't directly translate and require
task-specific proxies (e.g., toxicity rate by group, sentiment disparity in generated bios).

#### Q: How would you decide between building a custom model and calling a hosted foundation-model API for a new product feature? [System Design & MLOps]
Weigh: (1) data sensitivity/compliance (self-hosting may be required for regulated data), (2)
latency/cost at expected scale (API cost per call can dominate at high QPS vs. amortized
self-hosted infra), (3) how differentiated the task is (a generic task rarely justifies training
from scratch — fine-tune or prompt-engineer a foundation model instead), (4) time-to-market
(APIs ship in days, custom models take months), and (5) how fast the underlying capability is
improving (locking into a heavy custom pipeline early can be overtaken by next quarter's frontier
model). Default to the hosted API plus prompting/RAG unless there's a clear, measured reason to
go custom.

#### Q: What is calibration in a classifier, and how do you fix a poorly calibrated model? [Model Evaluation]
A model is calibrated if, among examples predicted with probability $p$, roughly $p$ fraction are
actually positive — e.g., of all examples scored 0.7, ~70% should be true positives. Tree
ensembles and SVMs are often poorly calibrated (scores are rank-useful but not true
probabilities) even when AUC is high. Fixes: Platt scaling (fit a logistic regression on top of
the raw scores) for smaller/sigmoid-shaped miscalibration, or isotonic regression for more
flexible, non-parametric correction with enough validation data; check with a reliability diagram
and Brier score, not just AUC.

---

## Hard

Deep reasoning, multi-step tradeoffs, and subtle edge cases.

#### Q: Explain the Transformer's self-attention mechanism and why the $\sqrt{d_k}$ scaling exists. [LLMs]
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Each token projects to Query/Key/Value vectors; $QK^T$ scores how much each token should attend
to every other token, softmax normalizes to weights, and the weighted sum of $V$ produces the
output. Without scaling, dot products grow with $d_k$ (variance $\propto d_k$ for random
vectors), pushing softmax into a saturated regime with near-zero gradients almost everywhere.
Dividing by $\sqrt{d_k}$ keeps the pre-softmax variance at ~1, keeping gradients healthy.

#### Q: Derive the gradient of cross-entropy loss with a sigmoid output. [Math]
$L=-[y\log\hat y+(1-y)\log(1-\hat y)]$, $\hat y=\sigma(z)$. Using
$\sigma'(z)=\sigma(z)(1-\sigma(z))$, the chain rule collapses to:
$$\frac{\partial L}{\partial z} = \hat y - y$$
This is why cross-entropy + sigmoid/softmax is the default pairing — contrast with MSE + sigmoid,
whose gradient vanishes when $\hat y$ saturates near 0/1 even if the prediction is very wrong.

#### Q: What does KL divergence measure and how does it relate to cross-entropy? [Math]
$$D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = H(P,Q) - H(P)$$
$H(P,Q)$ is cross-entropy, $H(P)$ is $P$'s own entropy and is fixed (doesn't depend on the model
$Q$) — so minimizing cross-entropy during training is equivalent to minimizing $D_{KL}(P\|Q)$,
pushing $Q$ toward the true label distribution $P$. KL is asymmetric: $D_{KL}(P\|Q) \ne
D_{KL}(Q\|P)$, which is why forward vs. reverse KL give different behavior in things like VAEs.

#### Q: Why does KV-caching make autoregressive LLM decoding tractable, and what is its memory cost as context grows? [LLMs]
Without caching, generating token $t$ recomputes $K,V$ for all $1..t$ from scratch — $O(T^2)$
total work across a full generation. Since keys/values for already-generated tokens never change
under causal masking, caching them turns each new-token step into $O(T)$ work (one new
query attending to all cached keys), making generation $O(T)$ total instead of $O(T^2)$. The
cost: KV-cache memory grows linearly with sequence length × layers × heads × head_dim × batch
size, and at long context/large batch this — not compute — becomes the binding constraint on
serving throughput, motivating techniques like multi-query/grouped-query attention (share K/V
heads across query heads) and PagedAttention (paged, non-contiguous KV-cache memory management)
to shrink or better-utilize this footprint.

#### Q: A model has near-zero training loss but the validation loss is noisy and non-monotonic rather than smoothly increasing — is this overfitting, and what do you check? [ML Fundamentals]
Not necessarily classic overfitting (which shows a smooth, monotonically widening train/val gap).
Non-monotonic, noisy validation loss suggests: (1) too small a validation set — high variance in
the loss estimate itself, unrelated to the model; (2) a learning rate too high late in training,
causing the model to bounce around sharp minima that generalize inconsistently epoch to epoch;
(3) a train/val distribution mismatch where validation batches vary in composition (e.g., not
shuffled, or drawn from a different time window) more than in size. Diagnose by plotting val loss
variance across multiple seeds/runs and checking val set size before concluding it's overfitting
and reaching for regularization.

#### Q: Two models have identical top-line AUC but you must choose one for a production fraud system — what else do you check before deciding? [Model Evaluation]
AUC is a single aggregate over all thresholds and can hide critical differences: (1) calibration
— are the probability outputs usable directly for a cost-based threshold, or only good for
ranking; (2) performance at the *specific* operating point you'll actually deploy at (e.g.,
precision at 1% flag rate), since AUC averages over regions of the ROC curve you'll never
operate in; (3) latency/inference cost at serving scale; (4) robustness to distribution shift —
backtest on a more recent out-of-time slice, since fraud patterns evolve and a model that
overfit older fraud typologies can have equal historical AUC but degrade faster; (5)
explainability requirements for compliance/analyst review; (6) subgroup performance (does one
model silently perform much worse on a minority transaction type that matters for regulatory or
business reasons). Pick based on the deployment operating point and stability under shift, not
the headline AUC.

#### Q: Design a monitoring and rollback strategy for an LLM-powered feature where "correctness" has no single ground-truth label. [System Design & MLOps]
Layer signals since you can't compute accuracy directly: (1) automated proxy evals — a smaller
judge model or rule-based checks (format validity, forbidden-content filters, length/latency
bounds) run on every production response as a cheap first line; (2) sampled human review on a
statistically meaningful slice, stratified by input type, to catch what automated judges miss and
to calibrate the judge model itself against human judgment periodically; (3) implicit behavioral
signals — user edit/regeneration rate, thumbs down, task abandonment — as a continuous proxy for
quality that doesn't require explicit labeling; (4) drift detection on the input distribution
itself (are users asking fundamentally different things than eval data covered); (5) shadow
deployment of any prompt/model change before rollout, diffing judge-model scores and behavioral
signals against the incumbent. Rollback trigger: an automatic threshold breach on the fast proxy
signals (latency, judge-score drop, spike in regeneration rate) rather than waiting for delayed
human review, since human-in-the-loop signal often lags too far behind to catch a bad deploy
before real damage.

#### Q: Why can a model with strong average performance still be unsafe to ship, and how do you build an evaluation that would catch this? [Practical/Scenario]
Average-case metrics (overall accuracy, mean AUC) can be dominated by the majority of "easy"
cases while masking catastrophic failure on a small but important slice — e.g., a self-driving
perception model that's 99.9% accurate overall but reliably fails on a rare-but-safety-critical
scenario (pedestrian at night in rain), or a medical model that performs worse on an underrepresented
demographic. An evaluation robust to this requires: (1) explicit slice-based evaluation
(performance broken out by known risk-relevant subgroups, not just the aggregate), (2)
adversarial/edge-case test sets deliberately over-sampling rare-but-high-stakes scenarios beyond
their natural frequency, (3) worst-case or tail-risk metrics (e.g., max error over slices,
CVaR-style tail loss) reported alongside the mean, and (4) a pre-defined "hard no" gate — certain
slices must clear a minimum bar regardless of how good the aggregate metric looks, so a strong
average can never compensate for an unacceptable worst-case failure mode.

---

## Reference / Appendix

Use this file as the final drilling pass before an interview loop — walk each tier in order,
covering definitional ground first (Easy), then tradeoff/application reasoning (Medium), before
stress-testing yourself on the deep derivations and staff-level judgment calls (Hard). If an
answer references "see X," check the corresponding deep-dive file in this repository's other
modules for the fuller derivation or worked example.
