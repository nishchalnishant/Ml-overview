# Machine Learning Interview — Complete Preparation Guide

---

## 1. What the Interview Loop is Actually Measuring

Every round is testing a specific *capability*, not a knowledge domain. Recognizing the capability being tested changes how you construct your answer.

| Round | Underlying capability | How weak answers are spotted |
| :--- | :--- | :--- |
| Recruiter screen | Can you communicate your work to a non-expert? Do you have a coherent direction? | Vague descriptions signal unclear thinking, not just poor communication |
| DSA / coding | Do you decompose problems systematically, or do you thrash? | Interviewers watch how you handle new constraints mid-problem, not just whether you reach a solution |
| ML coding | Can you translate math to working code without pausing to look things up? | Numeric instability bugs, wrong gradient flow — the bugs that only appear if you don't understand what the code *means* |
| ML theory | Do you understand *why* things work, or just *what* they are? | Follow-up questions probe one causal layer deeper. "What happens if you remove the $\sqrt{d_k}$ scaling?" separates the two populations |
| ML system design | Do you think like someone who has shipped systems that break in production? | Missing cold start, monitoring, rollback, or training-serving skew is a strong junior signal |
| Behavioral / project | Do you own outcomes or just participate in them? | Passive framing — "the team did X" vs "I decided X and here's why" — is an ownership signal |

**Calibrate prep time by level:**
- L3 / junior: 60% ML coding + theory, 30% system design foundations, 10% behavioral
- L4 / mid: 40% system design, 35% theory depth, 25% behavioral
- L5+ / senior: 50% system design + failure modes, 30% behavioral with strong ownership framing, 20% theory depth

---

## 2. The Three-Layer Framework for Any Theory Question

Every ML theory question has three answerable layers. Stopping at layer 1 is a junior answer regardless of whether the definition is correct.

```
Layer 1 — Definition + formula
    What is it precisely? Include the math when relevant.

Layer 2 — Intuition: why this mechanism, not something simpler?
    What problem does it solve? What breaks if you remove it?
    Make the mechanism obvious in retrospect.

Layer 3 — Tradeoffs and failure modes
    When does it fail? What does it interact badly with?
    What's the alternative, and when would you choose the alternative?
```

**Applied to four common questions:**

**"Why BatchNorm?"**
- L1: Normalizes activations over the batch dimension per feature: $\hat{x} = (x - \mu_B)/\sigma_B$, then rescales with learnable $\gamma, \beta$.
- L2: Without it, each layer sees inputs whose distribution shifts as the preceding layers change during training — called internal covariate shift. The layer must simultaneously learn its task *and* adapt to a moving input distribution. BatchNorm fixes the input distribution, allowing a higher learning rate and faster convergence.
- L3: Breaks at small batch sizes (noisy $\mu_B, \sigma_B$ estimates corrupt the normalization). Breaks for variable-length sequences (different valid positions per sample — batch statistics are meaningless). Interacts badly with dropout (dropout changes the effective batch statistics at inference if ordering matters). For these cases: LayerNorm normalizes over the feature dimension per sample — sequence-length agnostic, works at batch size 1.

**"Why ReLU over sigmoid?"**
- L1: $\text{ReLU}(x) = \max(0, x)$; gradient is 1 for positive inputs, 0 for negative.
- L2: Sigmoid saturates at both ends — its derivative $\sigma(x)(1-\sigma(x))$ approaches 0 for large positive or negative inputs. Backpropagating through 20 layers of sigmoid multiplies gradients by $< 0.25^{20} \approx 10^{-13}$ — effectively zero. ReLU has gradient exactly 1 for positive pre-activations, so gradients don't shrink through ReLU layers.
- L3: Dying ReLU problem — if a neuron's pre-activation is always negative (e.g., from a bad initialization or a large weight update), the gradient is always 0 and the neuron never updates. Fix: LeakyReLU ($0.01x$ for $x < 0$) or careful He initialization. For Transformers, GELU is preferred — smooth approximation of ReLU that performs better on language tasks.

**"L1 vs L2 regularization?"**
- L1: $\lambda\sum|w_i|$; L2: $\lambda\sum w_i^2$.
- L2: The penalty on large weights is dominated by the gradient-of-squared-term, which shrinks weights proportionally. Some weight may get small but never exactly zero — the gradient at $w = 0$ is $2\lambda w = 0$, so no further push.
- L1: The penalty is $\lambda \cdot \text{sign}(w)$ — a constant push toward zero regardless of magnitude. For small nonzero weights, this push exceeds any gradient from the data, forcing them exactly to zero. This is why L1 produces sparse solutions. Probabilistic interpretation: L2 = Gaussian prior on weights (MAP), L1 = Laplace prior (fatter tails near zero — encourages sparsity).
- Tradeoff: L1 for feature selection and interpretability. L2 when all features are plausibly relevant. ElasticNet ($\alpha L1 + (1-\alpha) L2$) for both sparsity and stability.

**"Why cross-entropy over MSE for classification?"**
- L1: CE: $-\sum y_k \log \hat{p}_k$; MSE: $\sum(y - \hat{p})^2$.
- L2: Near the decision boundary, the MSE gradient with respect to the logit is small because the sigmoid's derivative is small there. You're in the middle of learning, and the gradient provides almost no update signal. Cross-entropy's gradient with respect to the logit is $\hat{p} - y$ — large and informative when the prediction is wrong, regardless of where you are in probability space.
- L3: MSE assumes Gaussian noise around the target, which is appropriate for regression. Classification outputs are Bernoulli (binary) or Categorical — cross-entropy is the correct log-likelihood for these distributions. Using MSE for classification is using the wrong probabilistic model.

---

## 3. How Depth Changes by Level

The same topic produces very different questions at different levels. Recognize which depth you're being evaluated at:

**Level 1 (L3 — correct definition):** "What is dropout?"
> "Randomly zeros activations with probability $p$ during training; surviving activations are scaled by $1/(1-p)$. At inference, all activations are used at full scale."

**Level 2 (L4 — reasoning about tradeoffs):** "Why use dropout vs L2 regularization?"
> "Both prevent overfitting, but through different mechanisms. L2 penalizes large weights — all neurons remain active but with shrunk weights; the model can still rely on any feature, just less strongly. Dropout randomly removes neurons during each training step — each neuron must learn to be useful in isolation, preventing neurons from co-adapting to each other's errors. Dropout approximates an ensemble of $2^n$ subnetworks with shared weights. The practical tradeoff: don't use both heavily together. L2 already shrinks weights; then dropout's rescaling creates compound uncertainty. With BatchNorm, use dropout *before* the BN layer — dropout after BN distorts the batch statistics that BN relies on."

**Level 3 (L5 — production judgment):** "We're seeing overfitting on a new domain after fine-tuning. Dropout is already at 0.3. What do you change and in what order?"
> "First, distinguish overfitting from distribution shift — they look identical in offline metrics. Check the validation loss trajectory: does it start high from epoch 1 (shift — the model never fits this distribution) or does it rise after initially decreasing (overfitting — the model starts fitting then memorizes)? If true overfitting: freeze the bottom 80% of layers and fine-tune only the top layers. The bottom layers contain general representations that shouldn't change; only the task-specific top layers need adaptation. If that's insufficient, increase dropout to 0.4-0.5 on the fine-tuning layers. If distribution shift: no amount of regularization helps — you need more representative data from the target domain. Mixing a small fraction (5-10%) of the original pretraining distribution prevents catastrophic forgetting of general representations."

---

## 4. ML Coding — What Must Be Automatic

These must be writeable with zero hesitation. Each one has a numerical stability failure mode.

**Sigmoid + BCE loss — with the stability fix:**
```python
import numpy as np

def sigmoid(z):
    # Clip prevents exp(-z) overflow for large negative z
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce_loss(y_true, y_pred, eps=1e-7):
    # eps prevents log(0) = -inf when y_pred ∈ {0, 1}
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )
```

**Numerically stable softmax — the invariance argument:**
```python
def softmax(x):
    # Softmax is shift-invariant: softmax(x) = softmax(x - c)
    # Proof: exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i)/sum(exp(x_j))
    # Set c = max(x): largest exponent is e^0 = 1, never overflows
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
```

**Precision, Recall, F1 — with correct denominators explained:**
```python
def prf1(y_true, y_pred, eps=1e-8):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()   # we said positive; it was negative
    fn = ((y_pred == 0) & (y_true == 1)).sum()   # we said negative; it was positive
    # Precision: of all our positive predictions, what fraction was correct?
    p = tp / (tp + fp + eps)
    # Recall: of all actual positives, what fraction did we catch?
    r = tp / (tp + fn + eps)
    # F1: harmonic mean — punishes extreme imbalance between P and R
    f1 = 2 * p * r / (p + r + eps)
    return p, r, f1
```

**Scaled dot-product attention:**
```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # (batch, seq_q, seq_k)
    if mask is not None:
        # Where mask is False, set to -1e9 → softmax assigns ~0 weight
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V
```

**K-Means — one full E-step + M-step:**
```python
def kmeans_step(X, centroids):
    # Broadcasting: (n, 1, d) - (1, k, d) → (n, k, d) → norm → (n, k)
    dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
    labels = dists.argmin(axis=1)
    new_centroids = np.array([
        X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
        for j in range(len(centroids))
    ])
    return labels, new_centroids
```

**Gradient descent step with L2 regularization:**
```python
def gd_step(X, y, w, b, lr=0.01, lambda_=0.0):
    n = len(y)
    y_hat = sigmoid(X @ w + b)
    # L2 gradient: adds lambda * w to ∂L/∂w
    dw = (X.T @ (y_hat - y)) / n + lambda_ * w
    db = (y_hat - y).mean()
    return w - lr * dw, b - lr * db
```

**PyTorch training loop — the four common bugs and why they are bugs:**
```python
# Bug 1: Not zeroing gradients — PyTorch accumulates gradients by default.
# Each backward() adds to the existing gradient. Without zero_grad(), you train
# on the sum of all past batch gradients, not the current batch.
optimizer.zero_grad()

loss.backward()

# Bug 2: Not clipping gradients before the step.
# Gradient clipping must precede optimizer.step() to be effective.
# Clipping after the step modifies weights that were already updated with bad gradients.
nn.utils.clip_grad_norm_(model.parameters(), 1.0)

optimizer.step()
scheduler.step()

# Inference: both lines are required for different reasons
model.eval()           # mode: disables dropout, switches BN to use running statistics
with torch.no_grad():  # memory: stops building the autograd computation graph
    out = model(x)

# Checkpoint: optimizer state is not optional
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),  # Adam's m and v buffers — lose these, lose momentum
    "scheduler": scheduler.state_dict() if scheduler else None,
    "val_loss": val_loss,
}, "checkpoint.pt")
```

---

## 5. ML System Design — Universal Structure

Never start with a model. The model is the *last* thing you choose. These steps are not optional — skipping any one signals a specific blind spot.

```
Step 1: Clarify the product goal
    "What user action are we optimizing? Click, purchase, dwell time, safety flag?"
    "Is this ranking, classification, generation, or anomaly detection?"
    "What is the cost of FP vs FN in business terms?"

Step 2: Define metrics — both layers
    Online (business):  CTR, revenue/query, D7 retention, P99 latency
    Offline (model):    AUC-ROC, NDCG@10, precision@K, perplexity
    "What does success look like in a dashboard 30 days post-launch?"

Step 3: State constraints explicitly
    Latency SLA, QPS, memory budget, regulatory, team size, retraining cadence

Step 4: Data and labels
    Volume, freshness required, label quality, class balance
    Where does ground truth come from? How delayed is it?
    Training-serving skew risks?

Step 5: Baseline first
    Simplest thing that could work: popularity ranking, logistic regression, BM25
    This is your control arm in the A/B test

Step 6: Architecture — justified by constraints, not trends
    Two-stage if item corpus > 1M: fast retrieval → slow ranking
    Justify every architectural choice against a constraint

Step 7: Feature pipeline
    Real-time (Redis velocity features) vs batch (offline embeddings)
    Where are transforms fit? Train data only — never on val/test

Step 8: Serving
    Batch (pre-compute offline) vs real-time (sub-200ms)
    Caching strategy for repeated expensive lookups

Step 9: Evaluation
    A/B test design: primary metric, sample size calculation, duration, novelty effect
    Temporal backtesting for ranking and time-sensitive systems

Step 10: Monitoring + rollback
    PSI on input features (> 0.25 = retrain trigger)
    Prediction score distribution drift (> 2σ from 7-day baseline = alert)
    Business KPI drop threshold → auto-rollback
    Rollback plan defined before launch, tested quarterly
```

**Signals that distinguish senior responses:**

| Signal | What it reveals about experience |
| :--- | :--- |
| Cold start handling | Every production system has new users and new items. Missing this is a strong junior signal. |
| Feedback loops | Recommending popular items makes them more popular. Self-reinforcing bias is a real design problem, not a theoretical one. |
| Rollback plan | Rollback should be defined before launch. Defining it after something breaks means you shipped without a safety net. |
| Cost-aware threshold | Default 0.5 threshold is almost never optimal. The threshold should come from the cost matrix $C_{FP}, C_{FN}$. |
| Training-serving skew | The most common source of offline-online metric divergence. Missing this means you'd waste days debugging the wrong thing. |
| Ground truth delay | For fraud, chargebacks arrive 30-60 days after the transaction. Your "current" labels are a month old. |

---

## 6. Behavioral / Project Deep Dive

Every project story requires seven elements. Missing any one costs you.

| Element | Weak version | Strong version |
| :--- | :--- | :--- |
| Problem | "We needed better recommendations" | "CTR on homepage dropped 8% after a catalog expansion — our model had no cold start handling for new items, so new items never appeared in recommendations" |
| Constraints | (omitted) | "< 50ms P99 latency, retraining budget of once per week, team of 2 MLE, model decisions must be explainable to legal" |
| Approach | "We used a two-tower model" | "Started with matrix factorization as a baseline (2 days). Per-segment analysis showed cold start drove most of the gap. Added a content-based two-tower for new items only, routing traffic by item age" |
| Tradeoffs | (omitted) | "Two-tower adds 40ms latency vs MF. Accepted this because it recovered the 8% CTR gap on new items. Recall@100 on existing items dropped 3% — accepted because the business impact was smaller" |
| Metrics | "AUC improved" | "Offline NDCG@10 +12%. A/B: CTR +6.3%, revenue/user +2.1%, p < 0.01 at n = 500K per arm" |
| Production outcome | "We shipped it" | "5% canary → 50% → 100% over two weeks. One rollback event: feature pipeline failure detected at 5% canary — zero user impact because we hadn't scaled up" |
| Lesson | (omitted) | "We didn't run cold start analysis before building the baseline. Cold start evaluation is now a first-day checklist item for every new model." |

**Three story types you must have prepared, not just one:**
1. **Technical problem-solving:** you diagnosed a hard problem systematically — unexpected degradation, a confounding variable, a bug that looked like a model problem but was a data problem
2. **Failure story:** you shipped something wrong, or made a judgment call that turned out to be incorrect — what happened, what you did, what changed
3. **Tradeoff / disagreement:** you chose the slower or more conservative option over stakeholder pressure, with a quantified reason

---

## 7. Key Tradeoffs — the "Why X and Not Y" Pattern

The right answer to any tradeoff question is: "it depends on these factors, and here is how I would decide."

| Tradeoff | Decision logic |
| :--- | :--- |
| XGBoost vs neural net | XGBoost for tabular data < 1M rows — right inductive bias, fast iteration, interpretable. Neural net when jointly embedding high-cardinality IDs or raw text, sharing representations across tasks, or dataset > 10M rows. The data size and embedding requirement are the decision variables. |
| Precision vs recall | Depends on the cost matrix: $C_{FN}$ vs $C_{FP}$. Fraud: missed fraud ($C_{FN}$) >> false block ($C_{FP}$) → recall-oriented. Spam: false block ($C_{FP}$) >> missed spam ($C_{FN}$) → precision-oriented. The threshold should minimize $\text{FP} \times C_{FP} + \text{FN} \times C_{FN}$, not maximize F1. |
| BatchNorm vs LayerNorm | BatchNorm normalizes over the batch dimension — requires a large, consistent batch; fails for variable-length sequences; breaks at batch size 1. LayerNorm normalizes over features per sample — sequence-length agnostic, standard for Transformers, works at any batch size. RMSNorm (LLaMA): skips mean subtraction (empirically unimportant), faster. |
| L1 vs L2 | L1: sparse — some weights exactly 0. Use for feature selection and interpretability. L2: all weights small, none exactly 0. Use when all features are plausibly relevant. ElasticNet: use when you want both sparsity and stability. |
| Online vs batch serving | Online when the decision depends on current context (fraud at transaction time, autocomplete). Batch when freshness is not required within the latency budget (weekly churn scores). Hybrid is most common in production: pre-compute embeddings offline, assemble and score at query time. |
| SMOTE vs class_weight | Try `class_weight='balanced'` first — free, no data generation, works for most models. SMOTE for severe imbalance in tabular data when the minority class is genuinely underrepresented in coverage, not just count. Neither fix handles distribution shift — that requires domain adaptation or more representative data. |

---

## 8. Numbers to Know Without Thinking

| Fact | Value |
| :--- | :--- |
| GPT-3 parameters | 175B |
| Chinchilla optimal tokens per parameter | ~20 |
| LLaMA 3 training tokens | 15T |
| Standard attention complexity | $O(n^2 d)$ |
| LoRA trainable parameters (typical) | ~0.06% of base model |
| BF16 memory per parameter | 2 bytes |
| FP32 memory per parameter | 4 bytes |
| 70B model in BF16 VRAM | ~140GB |
| PSI retrain threshold | > 0.25 |
| AdamW default $\beta_1, \beta_2$ | 0.9, 0.999 (LLM pretraining: 0.9, 0.95) |
| Warmup fraction of training | 1–5% of total steps |
| KV cache per 1K tokens (LLaMA 3 70B) | ~160MB |

---

## 9. Mistakes That Cost You the Role

| Mistake | What it signals |
| :--- | :--- |
| Starting system design with model choice | No requirements discipline — you'd design the right model for the wrong problem |
| Mentioning only accuracy as a metric | No understanding of class imbalance or business impact |
| Skipping cold start in a recommendation design | Junior signal — every production recommendation system encounters this |
| "Retrain the model" as first response to a production accuracy drop | No debugging instinct — you'd waste days on the wrong fix |
| Not mentioning monitoring or rollback | You think shipping = done; no production experience |
| Confident misinterpretation of p-values | Statistics red flag — "p < 0.05 means the null is probably false" is wrong |
| Rote behavioral answers with passive framing | "The team did" instead of "I decided because" — low ownership signal |

---

## 10. Pre-Interview Checklist

**Must write from scratch, zero hesitation:**
- [ ] Sigmoid, softmax, numerically stable BCE
- [ ] Precision, recall, F1 with correct denominators stated verbally
- [ ] Bias-variance decomposition formula + one-sentence interpretation of each term
- [ ] Bayes theorem applied to the rare-disease test example (the answer is ~9%, not 99%)
- [ ] Correct p-value interpretation + two wrong interpretations stated explicitly
- [ ] Two-stage retrieval + ranking sketch
- [ ] The 10-step system design framework

**Must explain tradeoffs fluently:**
- [ ] L1 vs L2 vs Dropout vs Early stopping (mechanism, not just effect)
- [ ] BatchNorm vs LayerNorm — with the BatchNorm failure conditions
- [ ] Batch vs real-time serving — with the hybrid pattern
- [ ] Precision vs recall as a cost matrix problem
- [ ] XGBoost vs neural net — with specific decision criteria

**Must sketch from memory:**
- [ ] Two-tower recommendation system (retrieval + ranking + cold start path)
- [ ] Fraud detection pipeline (Redis velocity → rule engine → ML model → cost-aware threshold)
- [ ] LLM serving stack (load balancer → vLLM → KV cache → streaming)
- [ ] Temporal train/val/test split — why random splits fail for time-series
