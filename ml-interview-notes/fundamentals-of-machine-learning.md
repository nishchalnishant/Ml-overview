# Fundamentals of Machine Learning

---

# Q1: Explain Epoch, Batch, Batch Size, and Iteration.

## 1. 🔹 Direct Answer
**Epoch**: one full pass over the **entire training set**. **Batch**: a **subset** of examples used in one **forward/backward** update. **Batch size**: number of examples per batch. **Iteration**: one **optimizer step** (one batch processed); iterations per epoch = **ceil(n / batch_size)**.

## 2. 🔹 Intuition
Epoch = “saw every training example once”; iteration = “one weight update.”

## 3. 🔹 Deep Dive
- **Full-batch GD**: batch size = n, one iteration per epoch.
- **Mini-batch SGD**: many iterations per epoch.

## 4. 🔹 Practical Perspective
Large batch → stable gradients, more memory; small batch → noise, often better **generalization** (implicit regularization).

## 5. 🔹 Code Snippet
```python
epochs = 10
batch_size = 32
iterations_per_epoch = (len(dataset) + batch_size - 1) // batch_size
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Batch = 1? **A:** SGD—noisy, slow on GPU utilization.

## 7. 🔹 Common Mistakes
Confusing iteration with epoch.

## 8. 🔹 Comparison / Connections
Learning rate schedules per epoch vs step.

## 9. 🔹 One-line Revision
Epoch covers full data; iteration is one batch update—batch size trades noise, speed, and memory.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: What are embeddings in Machine Learning?

## 1. 🔹 Direct Answer
**Embeddings** are **learned dense vector representations** of discrete items (words, users, items) in **ℝ^d** such that **similar** or **co-occurring** items map **nearby**—capture **semantic** or **behavioral** structure.

## 2. 🔹 Intuition
Instead of huge one-hot vectors, a **compact** code that **generalizes** across related entities.

## 3. 🔹 Deep Dive
Trained via **matrix factorization**, **Word2Vec**, **neural** layers (**nn.Embedding**), or **encoder** outputs.

## 4. 🔹 Practical Perspective
**Dimension** d trades capacity vs overfit; **cold start** for new IDs.

## 5. 🔹 Code Snippet
```python
import torch.nn as nn
emb = nn.Embedding(num_users, 64)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multimodal? **A:** Joint embedding spaces (CLIP).

## 7. 🔹 Common Mistakes
Using raw high-cardinality IDs without embedding in deep models.

## 8. 🔹 Comparison / Connections
Kernel methods, latent factors.

## 9. 🔹 One-line Revision
Embeddings map discrete IDs to trainable vectors capturing similarity—core to NLP and recommender systems.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What is Softmax Activation Function?

## 1. 🔹 Direct Answer
**Softmax** maps a vector **z ∈ ℝ^K** to **probabilities** **p_i = e^{z_i} / Σ_j e^{z_j}**—outputs are **positive** and **sum to 1**. Used as last layer for **multiclass** classification with **cross-entropy** loss.

## 2. 🔹 Intuition
Turns scores into a **competition** where larger logits win more mass.

## 3. 🔹 Deep Dive
**Numerical stability**: subtract **max(z)** before exp. Gradient simplifies with CE: **p − y**.

## 4. 🔹 Practical Perspective
**Temperature** T: softmax(z/T)—sharp vs soft distributions.

## 5. 🔹 Code Snippet
```python
import numpy as np
def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Log-softmax? **A:** More stable for CE than log(softmax(z)).

## 7. 🔹 Common Mistakes
Applying softmax to multi-label problems (use **sigmoid** per class instead).

## 8. 🔹 Comparison / Connections
Sigmoid (binary), logsumexp trick.

## 9. 🔹 One-line Revision
Softmax converts logits to a probability simplex—pair with cross-entropy for multiclass.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q4: What is Machine Learning?

## 1. 🔹 Direct Answer
**ML** is learning **predictive models** or **policies** from **data** rather than **hand-coded** rules—**optimize** parameters to **minimize loss** or **maximize reward** on observed examples and **generalize** to new data.

## 2. 🔹 Intuition
Let data **tell** the pattern subject to **constraints** and **inductive bias**.

## 3. 🔹 Deep Dive
**Supervised** (labels), **unsupervised** (structure), **RL** (rewards). **Bias-variance** tradeoff, **generalization** bound mindset.

## 4. 🔹 Practical Perspective
**Data quality** > algorithm choice often; **monitoring** and **iteration** essential.

## 5. 🔹 Code Snippet
```text
learn: min_θ E[ℓ(f_θ(x), y)] + λΩ(θ)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs statistics? **A:** Overlap—ML emphasizes prediction, scale, computation.

## 7. 🔹 Common Mistakes
“ML finds truth automatically”—needs assumptions and validation.

## 8. 🔹 Comparison / Connections
AI, data science, causal inference.

## 9. 🔹 One-line Revision
ML learns functions from data to minimize expected loss while controlling complexity.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q5: Differentiate between Supervised and Unsupervised Learning.

## 1. 🔹 Direct Answer
**Supervised**: training pairs **(x, y)**—learn **f(x)→y** (classification/regression). **Unsupervised**: only **x**—find **structure** (clusters, manifolds, representations) without labels.

## 2. 🔹 Intuition
Teacher tells correct answer vs explore data alone.

## 3. 🔹 Deep Dive
**Semi-supervised** mixes both; **self-supervised** creates labels from data (e.g., masked LM).

## 4. 🔹 Practical Perspective
Unsupervised needs **evaluation proxies** (silhouette, downstream).

## 5. 🔹 Code Snippet
```python
# supervised
fit(X_train, y_train)
# unsupervised
fit(X_train)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Anomaly detection? **A:** Often unsupervised or weak labels.

## 7. 🔹 Common Mistakes
Calling clustering “classification without labels”—objectives differ.

## 8. 🔹 Comparison / Connections
RL as separate paradigm.

## 9. 🔹 One-line Revision
Supervised uses labeled examples; unsupervised learns structure from inputs alone.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q6: What is Reinforcement Learning?

## 1. 🔹 Direct Answer
**RL** learns a **policy** **π(a|s)** to **maximize expected cumulative reward** from **environment** interaction—**trial and error**, **delayed** consequences (**MDP** formalism).

## 2. 🔹 Intuition
Learn by **doing** and getting **graded**, not from static dataset only.

## 3. 🔹 Deep Dive
**Value** methods (Q-learning), **policy** gradients (PPO), **exploration/exploitation**.

## 4. 🔹 Practical Perspective
**Sample inefficient**; simulators help; **safety** critical in real systems.

## 5. 🔹 Code Snippet
```text
maximize E[ Σ γ^t r_t | π ]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Off-policy? **A:** Learn from behavior not equal target policy (DQN).

## 7. 🔹 Common Mistakes
Confusing RL with supervised learning on trajectories without credit assignment nuance.

## 8. 🔹 Comparison / Connections
Bandits, optimal control, imitation learning.

## 9. 🔹 One-line Revision
RL optimizes long-term reward through environment interaction—MDPs, policies, value functions.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is Bias?

## 1. 🔹 Direct Answer
**Statistical bias**: error from **wrong assumptions** (e.g., linear model for nonlinear truth)—**underfitting**. **Algorithmic/social bias**: systematic **unfairness** toward groups from **data** or **design**. **Bias** in estimators: **E[θ̂] − θ**.

## 2. 🔹 Intuition
Model too **simple** to capture signal, or data **skewed** against a cohort.

## 3. 🔹 Deep Dive
**Bias-variance**: high bias → **train and val** both poor; **fairness** metrics per group.

## 4. 🔹 Practical Perspective
Disambiguate **interview** term—ML vs social vs estimator bias.

## 5. 🔹 Code Snippet
```text
Bias² term in decomposition of expected squared error
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Reduce bias? **A:** More complex model, better features, less strong wrong priors.

## 7. 🔹 Common Mistakes
Using “bias” only in fairness sense in technical ML questions.

## 8. 🔹 Comparison / Connections
Variance, fairness, inductive bias.

## 9. 🔹 One-line Revision
Bias is systematic error from model mismatch or data skew—clarify statistical vs social meaning in interview.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: What is the difference between Classification and Regression?

## 1. 🔹 Direct Answer
**Regression** predicts **continuous** outputs (real numbers). **Classification** predicts **discrete** labels (finite classes)—may output **hard** labels or **probabilities**.

## 2. 🔹 Intuition
Price vs category; same algorithms often adapt via **loss** (MSE vs CE).

## 3. 🔹 Deep Dive
**Ordinal** regression in between; **multi-label** classification vector.

## 4. 🔹 Practical Perspective
**Metrics** differ (RMSE vs F1); **calibration** for probabilistic classification.

## 5. 🔹 Code Snippet
```python
from sklearn.linear_model import LinearRegression, LogisticRegression
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Logistic “regression”? **A:** Classification despite name—linear model for class prob.

## 7. 🔹 Common Mistakes
Using linear regression on categorical codes ordinally.

## 8. 🔹 Comparison / Connections
Ranking, structured prediction.

## 9. 🔹 One-line Revision
Regression targets continuous values; classification targets discrete labels—choose losses and metrics accordingly.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q9: Explain Overfitting and Underfitting. How can you prevent them?

## 1. 🔹 Direct Answer
**Overfitting**: **low train error, high val error**—memorizes noise. **Underfitting**: **high** train and val—model too **simple** or **wrong** features. **Fix overfit**: more data, **regularization**, **dropout**, simpler model, **early stopping**, **augmentation**. **Fix underfit**: richer features, **deeper** model, less regularization.

## 2. 🔹 Intuition
Overfit = **too tailored** to training quirks; underfit = **too blunt**.

## 3. 🔹 Deep Dive
**Bias-variance** decomposition; **learning curves** diagnose.

## 4. 🔹 Practical Perspective
**Validation discipline**—detect with **held-out** data and **cross-validation**.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import learning_curve
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Double descent? **A:** Beyond classical U-shaped—large models interpolate; still monitor val.

## 7. 🔹 Common Mistakes
Tuning on test set—**leaks** generalization estimate.

## 8. 🔹 Comparison / Connections
Generalization gap, PAC learning intuition.

## 9. 🔹 One-line Revision
Overfit memorizes train; underfit misses signal—regularize, simplify, or add capacity/data accordingly.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What Are L1 and L2 Loss Functions?

## 1. 🔹 Direct Answer
**L1 (MAE)**: **|y − ŷ|**—robust, linear penalty. **L2 (MSE)**: **(y − ŷ)²**—differentiable, emphasizes large errors. **L1 regularization** sparsifies weights; **L2** shrinks smoothly.

## 2. 🔹 Intuition
L2 punishes **outliers** more; L1 tolerates **heavy tails** better for **errors**.

## 3. 🔹 Deep Dive
L1 has **kinks** at zero (sparsity); L2 smooth.

## 4. 🔹 Practical Perspective
**Huber** combines both for regression.

## 5. 🔹 Code Snippet
```python
import numpy as np
l1 = np.abs(y - yhat).mean()
l2 = ((y - yhat)**2).mean()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** L1 vs L2 reg? **A:** L1 → sparse weights; L2 → small but dense.

## 7. 🔹 Common Mistakes
Confusing **loss** L1/L2 with **regularization** naming—clarify context.

## 8. 🔹 Comparison / Connections
Elastic net, robust regression.

## 9. 🔹 One-line Revision
L1 loss is MAE; L2 is MSE—L1 robust, L2 smooth and outlier-sensitive.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q11: What is Regularization? Explain L1 (Lasso) and L2 (Ridge) regularization.

## 1. 🔹 Direct Answer
**Regularization** adds **penalty Ω(θ)** to loss to **constrain** model complexity—**reduce overfitting**. **L2 (Ridge)**: **λ||w||₂²**—**shrinks** weights smoothly. **L1 (Lasso)**: **λ||w||₁**—promotes **sparsity** (feature selection).

## 2. 🔹 Intuition
Tell optimizer: “keep weights small or few” unless data strongly demands them.

## 3. 🔹 Deep Dive
**MAP** view: Gaussian prior → L2; Laplace prior → L1. **Elastic net** mixes both.

## 4. 🔹 Practical Perspective
Tune **λ** via CV; **standardize** features for fair penalization.

## 5. 🔹 Code Snippet
```python
from sklearn.linear_model import Ridge, Lasso
Ridge(alpha=1.0); Lasso(alpha=0.1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Weight decay in NNs? **A:** Typically L2 on weights—AdamW decouples from adaptive LR.

## 7. 🔹 Common Mistakes
Huge λ causing **underfit**—always CV.

## 8. 🔹 Comparison / Connections
Dropout, early stopping as implicit regularization.

## 9. 🔹 One-line Revision
L2 shrinks coefficients; L1 zeros many—both reduce overfitting with tuned λ.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q12: What are Loss Functions and Cost Functions? Explain the key difference between them.

## 1. 🔹 Direct Answer
**Loss** often refers to **per-example** error **ℓ(y, ŷ)**. **Cost** (empirical risk) is **average/total** loss over dataset **+ regularization**—what you **optimize**. **Colloquially** “loss” = total objective—**clarify** in interview.

## 2. 🔹 Intuition
Loss = single mistake; cost = **bill** for whole training set.

## 3. 🔹 Deep Dive
**Expected risk** E[ℓ] vs **empirical** average; **surrogate** losses for tractability.

## 4. 🔹 Practical Perspective
Report **mean** loss for comparability across batch sizes.

## 5. 🔹 Code Snippet
```text
cost = (1/N) Σ ℓ(y_i, f(x_i)) + λ Ω(θ)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Hinge vs 0-1? **A:** Hinge is convex surrogate for SVM.

## 7. 🔹 Common Mistakes
Inconsistent naming—define terms explicitly.

## 8. 🔹 Comparison / Connections
Risk minimization, Bayes optimal classifier.

## 9. 🔹 One-line Revision
Per-example loss aggregates to cost/objective—language varies but optimization targets the aggregate.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q13: What are dropouts?

## 1. 🔹 Direct Answer
**Dropout** randomly **zeros** activations with probability **p** during training—**prevents co-adaptation** of neurons. At **inference**, use **all** units with **scaling** (or inverted dropout during training) to match expected magnitude.

## 2. 🔹 Intuition
Randomly “knock out” helpers so network can’t rely on fragile conspiracies.

## 3. 🔹 Deep Dive
Approximate **ensemble** of subnetworks; **MC dropout** for uncertainty (approximate Bayesian).

## 4. 🔹 Practical Perspective
Common in **FC** layers; **BatchNorm** interaction—order matters; often **0.1–0.5** drop prob.

## 5. 🔹 Code Snippet
```python
nn.Dropout(p=0.5)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Test time? **A:** model.eval() disables dropout in PyTorch.

## 7. 🔹 Common Mistakes
Applying dropout to **output** layer incorrectly; forgetting **train/eval** mode.

## 8. 🔹 Comparison / Connections
Stochastic depth, L2 regularization.

## 9. 🔹 One-line Revision
Dropout stochastically drops units while training as regularizer—disable and scale at inference.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: What is a Perceptron?

## 1. 🔹 Direct Answer
A **perceptron** is a **binary linear classifier**: **ŷ = sign(wᵀx + b)**. **Rosenblatt** algorithm updates weights on **misclassified** examples—**guaranteed** convergence if data **linearly separable**.

## 2. 🔹 Intuition
Simplest **neuron**—foundation for neural networks.

## 3. 🔹 Deep Dive
**Margin** interpretation; **kernel trick** extension (not in basic perceptron).

## 4. 🔹 Practical Perspective
Historical; replaced by **logistic regression** (probabilities) and **SVM** (max margin).

## 5. 🔹 Code Snippet
```python
y_hat = np.where(X @ w + b >= 0, 1, 0)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Non-separable? **A:** Doesn’t converge—use pocket algorithm or SVM with slack.

## 7. 🔹 Common Mistakes
Confusing with **logistic** neuron (sigmoid).

## 8. 🔹 Comparison / Connections
Adaline, SVM.

## 9. 🔹 One-line Revision
Perceptron is linear threshold classifier with mistake-driven updates—converges if separable.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: Explain Multilayer Perceptron (MLP).

## 1. 🔹 Direct Answer
**MLP** is a **feedforward** network: **layers** of **affine + nonlinear** transforms **σ(Wx + b)** stacked—**universal approximator** with enough width/depth. **Dense** connections between layers.

## 2. 🔹 Intuition
Stack of **learned feature transforms**—hierarchy from raw inputs.

## 3. 🔹 Deep Dive
**Backprop** through chain rule; **width** vs **depth** trade-offs.

## 4. 🔹 Practical Perspective
Baseline for **tabular** data; **ResNet**-style skip in deep nets.

## 5. 🔹 Code Snippet
```python
nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, out_dim))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs CNN? **A:** MLP ignores spatial structure—needs flattening images.

## 7. 🔹 Common Mistakes
Saying MLP has no activation—then it’s just linear.

## 8. 🔹 Comparison / Connections
Deep learning stack, autograd.

## 9. 🔹 One-line Revision
MLP stacks nonlinear layers to learn hierarchical features—classic universal approximator baseline.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q16: What is Cross-Entropy?

## 1. 🔹 Direct Answer
**Cross-entropy** **H(p,q) = −Σ p_i log q_i** measures **average bits** to encode **true** distribution **p** using **model** **q**. For classification with **one-hot p**, reduces to **−log q_y**—**log loss**. **Minimizing CE** = **MLE** for categorical data.

## 2. 🔹 Intuition
Penalizes assigning **low probability** to correct class.

## 3. 🔹 Deep Dive
**Relationship** to **KL**: **KL(p||q) = H(p,q) − H(p)**.

## 4. 🔹 Practical Perspective
**Numerical**: **log_softmax** + **NLLLoss** stable in PyTorch.

## 5. 🔹 Code Snippet
```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, target)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Label smoothing? **A:** Softens one-hot—regularization.

## 7. 🔹 Common Mistakes
Applying CE without **logits** vs **probs** correctly.

## 8. 🔹 Comparison / Connections
Hinge loss, focal loss.

## 9. 🔹 One-line Revision
Cross-entropy is the standard classification loss aligned with MLE for softmax outputs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: What are Logits?

## 1. 🔹 Direct Answer
**Logits** are **unnormalized scores** **z** **before** **softmax** (or **sigmoid** for binary)—**any real** numbers. **Probabilities** = softmax(z). **Numerically** stable training uses logits directly in **cross-entropy**.

## 2. 🔹 Intuition
Raw “evidence” for each class before turning into probabilities.

## 3. 🔹 Deep Dive
In **temperature scaling** for calibration, divide logits by **T**.

## 4. 🔹 Practical Perspective
**BCEWithLogitsLoss** in PyTorch expects logits not sigmoid outputs.

## 5. 🔹 Code Snippet
```python
logits = model(x)  # shape [batch, num_classes]
probs = torch.softmax(logits, dim=-1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why not train on probs directly? **A:** Logits are unbounded—easier optimization; CE handles normalization.

## 7. 🔹 Common Mistakes
Applying softmax twice.

## 8. 🔹 Comparison / Connections
Log-odds in logistic regression.

## 9. 🔹 One-line Revision
Logits are pre-softmax scores—use with cross-entropy for stable training.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q18: Explain Cross-Validation. Why is it used?

## 1. 🔹 Direct Answer
**k-fold CV** trains **k** times, each holding out **1/k** for validation—**averages** performance estimate with **lower variance** than single split. **Why**: better **data use**, **tune** hyperparameters **without** test peek, **detect** instability.

## 2. 🔹 Intuition
Rotate the “exam” so every example validates once.

## 3. 🔹 Deep Dive
**Stratified**, **GroupKFold**, **TimeSeriesSplit** for correct independence.

## 4. 🔹 Practical Perspective
**Nested** CV for unbiased performance with **inner** HPO.

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5, scoring="roc_auc")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** LOOCV? **A:** Low bias, high variance, expensive.

## 7. 🔹 Common Mistakes
**Leakage** by fitting preprocessing on full data before CV.

## 8. 🔹 Comparison / Connections
Bootstrap, holdout.

## 9. 🔹 One-line Revision
Cross-validation rotates train/val splits for robust performance estimates and safer tuning.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: What are precision, recall, and F1-score?

## 1. 🔹 Direct Answer
**Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **F1** = **2PR/(P+R)** harmonic mean. Use when **class imbalance** or **asymmetric** costs—**accuracy** misleading.

## 2. 🔹 Intuition
Precision: trust positives; recall: catch positives; F1 balances.

## 3. 🔹 Deep Dive
**β-F** if recall weighted differently.

## 4. 🔹 Practical Perspective
**Threshold** tuning on **validation** PR curve.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Macro vs micro F1? **A:** Macro per-class average; micro pools globally.

## 7. 🔹 Common Mistakes
High accuracy with **99% negatives** and useless model.

## 8. 🔹 Comparison / Connections
PR-AUC, ROC-AUC.

## 9. 🔹 One-line Revision
Precision vs FP; recall vs FN; F1 combines—essential for imbalance.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q20: What is anomaly detection?

## 1. 🔹 Direct Answer
**Anomaly detection** identifies **rare** or **abnormal** points differing from **normal** pattern—often **unsupervised** (density, reconstruction error, isolation forest) or **one-class** SVM. Apps: **fraud**, **manufacturing**, **security**.

## 2. 🔹 Intuition
Learn “normal” and **flag** outliers—often **few** positive labels.

## 3. 🔹 Deep Dive
**Supervised** if labeled anomalies exist (rare); **metrics**: precision at k, AUC if scored.

## 4. 🔹 Practical Perspective
**Drift** in normal behavior—**retrain**; **false alarm** costs.

## 5. 🔹 Code Snippet
```python
from sklearn.ensemble import IsolationForest
IsolationForest(contamination=0.01).fit(X)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs classification? **A:** Extreme imbalance + changing “normal.”

## 7. 🔹 Common Mistakes
Using accuracy with 0.1% fraud base rate.

## 8. 🔹 Comparison / Connections
Novelty detection, PU learning.

## 9. 🔹 One-line Revision
Anomaly detection finds rare deviations via density, reconstruction, or isolation—optimize precision at operational alert rate.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: What is the difference between policy-based and value-based methods?

## 1. 🔹 Direct Answer
**Value-based**: learn **Q(s,a)** or **V(s)** then derive **policy** (e.g., **greedy** w.r.t. Q). **Policy-based**: **parameterize π(a|s)** directly and optimize **expected return** via **policy gradient** (REINFORCE, PPO). **Actor-critic** combines both.

## 2. 🔹 Intuition
Value methods: learn **how good** states/actions are; policy methods: learn **what to do** directly.

## 3. 🔹 Deep Dive
Value: sample efficient in discrete actions; policy: **continuous** action spaces natural.

## 4. 🔹 Practical Perspective
**PPO** popular default for continuous control.

## 5. 🔹 Code Snippet
```text
Q-learning: TD update on Q ; REINFORCE: ∇ log π(a|s) * G_t
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Off-policy PG? **A:** Importance sampling—tricky; use trust regions.

## 7. 🔹 Common Mistakes
Saying DQN is policy-based—it’s value-based with function approx.

## 8. 🔹 Comparison / Connections
Actor-critic, A3C.

## 9. 🔹 One-line Revision
Value methods learn Q/V; policy methods optimize π directly—actor-critic blends strengths.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q22: What is Q-Learning?

## 1. 🔹 Direct Answer
**Q-learning** learns **Q(s,a)** for optimal **action-value** using **Bellman** backup: **Q ← Q + α [r + γ max_a' Q(s',a') − Q(s,a)]**—**off-policy** (target is **max** over actions). **Tabular** for small spaces; **DQN** for large.

## 2. 🔹 Intuition
Propagate **future** reward information **backwards** from experience.

## 3. 🔹 Deep Dive
**Exploration** (ε-greedy); **function approximation** instability—**target networks**, **experience replay**.

## 4. 🔹 Practical Perspective
Foundation for **Atari** DQN; superseded by **Rainbow** improvements.

## 5. 🔹 Code Snippet
```text
TD target: r + γ max_a' Q(s', a')
```

## 6. 🔹 Interview Follow-ups
1. **Q:** SARSA vs Q-learning? **A:** SARSA on-policy—includes exploration in update.

## 7. 🔹 Common Mistakes
Max bias in **double Q-learning** issue—use **double DQN**.

## 8. 🔹 Comparison / Connections
Temporal difference learning, policy iteration.

## 9. 🔹 One-line Revision
Q-learning is off-policy TD control learning Q* via Bellman max backups—scalable as DQN.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q23: Explain the concept of exploration vs exploitation.

## 1. 🔹 Direct Answer
**Exploration** tries **new** actions to discover **better** rewards; **exploitation** uses **current best** knowledge—**trade-off**: explore too much wastes reward; exploit too much **misses** optima. **ε-greedy**, **UCB**, **Thompson sampling** balance.

## 2. 🔹 Intuition
Restaurant choice: try new places vs go to favorite.

## 3. 🔹 Deep Dive
**Regret** bounds in bandits; **intrinsic** motivation in deep RL (curiosity).

## 4. 🔹 Practical Perspective
**Simulated** environments allow more exploration safely.

## 5. 🔹 Code Snippet
```python
if random.random() < eps: action = random_action()
else: action = greedy(Q, state)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** In recommender systems? **A:** Explore new items—multi-armed bandit per user.

## 7. 🔹 Common Mistakes
Pure greedy in **unknown** MDP—gets stuck.

## 8. 🔹 Comparison / Connections
Multi-armed bandits, Bayesian optimization.

## 9. 🔹 One-line Revision
Exploration gathers information; exploitation optimizes known reward—balance is core to RL and bandits.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q24: Explain the curse of dimensionality and how to address it.

## 1. 🔹 Direct Answer
As **dimension d** grows, **volume** explodes—data becomes **sparse**, distances **less meaningful**, **sample** needs grow **exponentially** for density estimation. **Mitigate**: **dimensionality reduction** (PCA), **feature selection**, **manifold** assumption, **regularization**, **more data**, **inductive biases** (CNNs), **embedding** low-d structures.

## 2. 🔹 Intuition
In high-d, “nearby” is **huge**—everything is far until **structure** assumed.

## 3. 🔹 Deep Dive
**kNN** degrades; **concentration** of measure phenomena.

## 4. 🔹 Practical Perspective
**Curse** motivates **deep learning** that learns **representations** collapsing effective dimension.

## 5. 🔹 Code Snippet
```text
n ~ exp(d) samples for dense coverage (informal)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Johnson-Lindenstrauss? **A:** Random projections preserve distances in lower k ~ log(n)/ε².

## 7. 🔹 Common Mistakes
Throwing **more** raw features without regularization.

## 8. 🔹 Comparison / Connections
Manifold hypothesis, metric learning.

## 9. 🔹 One-line Revision
Curse of dimensionality: data sparsity and distance breakdown in high-d—fight with structure, reduction, and regularization.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q25: Explain Local Loss, Focal Loss, and Gradient Blending in Multi-Task Learning.

## 1. 🔹 Direct Answer
**Multi-task learning (MTL)** shares representation for **multiple losses**. **Local loss** per task (e.g., CE + MSE). **Focal Loss** down-weights **easy** examples **(1−p)^γ** to focus on **hard** ones—helps **imbalance** in detection. **Gradient blending** (or PCGrad, GradNorm) **weights** task losses/gradients so **one task** doesn’t **dominate**—balances **shared** updates.

## 2. 🔹 Intuition
Tasks **fight** for the same layers—need **reweighting** or **gradient surgery**.

## 3. 🔹 Deep Dive
**Focal**: **FL = −(1−p_t)^γ log p_t** for **p_t** on true class. **Uncertainty weighting** uses **learned** σ_i per task.

## 4. 🔹 Practical Perspective
**Auxiliary** tasks can **help** or **hurt**—ablation per task.

## 5. 🔹 Code Snippet
```python
# focal for CE
pt = p * y_onehot + (1-p)*(1-y_onehot)  # simplified
loss = -((1-pt)**gamma * torch.log(pt)).mean()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Negative transfer? **A:** Tasks conflict—separate towers or routing.

## 7. 🔹 Common Mistakes
Equal loss weights when **scales** differ by orders of magnitude.

## 8. 🔹 Comparison / Connections
Multi-task architectures, hard example mining.

## 9. 🔹 One-line Revision
MTL combines task losses; focal focuses on hard examples; gradient blending balances conflicting task gradients.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
