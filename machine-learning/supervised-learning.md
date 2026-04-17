# Supervised learning (deep-dive)

**Cold open:** Supervised learning = you have **labels** (the answer key). Your job is to learn a mapping that **generalizes** — same spirit as passing integration tests that weren’t copy-pasted from dev data.

**Azure angle:** Labels are your **golden dataset**; the model is a **policy** that maps inputs to outputs; evaluation metrics are your **SLIs** for quality — pick them like you pick error budgets.

---

## 1. What to reach for (blueprint)

| Task | Family | Go-to | When |
| :--- | :--- | :--- | :--- |
| **Regression** | Linear | Linear regression | Baseline, interpretability, linear-ish signal. |
| **Regression** | Non-linear | RF / XGBoost / LightGBM | Tabular, interactions, “we don’t know the shape.” |
| **Classification** | Probabilistic | Logistic regression | Calibrated-ish probs, strong baseline. |
| **Classification** | Margin | SVM | High-dim, clear separation, kernel when needed. |
| **Classification** | Fast & simple | Naive Bayes | Text, tiny data, independence assumption holds “enough.” |

**Deploy prompt:** *You’re handed a CSV and one hour. What’s your first pipeline stage in Azure ML or a notebook?* → Baseline (linear or NB) → strong tree model → proper CV + metric tied to the business.

---

## 2. Linear & logistic — why not MSE for classification?

### Q: Why log-loss for logistic regression instead of MSE?

**Direct answer:** MSE on probabilities for classification is **awkward** — non-convex in the 0/1 setup people care about, so optimization can get stuck. **Log-loss (cross-entropy)** is convex for logistic regression and **punishes confident wrong answers** brutally — exactly what you want when “almost right” isn’t safe.

**Ghazal-ish intuition:** The loss isn’t about each word in isolation — it’s about whether the **whole line** (probability vector) lands on the **right emotional truth** (label). Cross-entropy is strict about that harmony.

**Logistic setup:**

$$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Loss (average negative log-likelihood):**

$$J(w) = -\frac{1}{m} \sum \big[y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})\big]$$

---

## 3. SVM & kernels — changing the room without moving the furniture

### Q: How does the kernel trick work without explicitly adding features?

**Direct answer:** You only need **inner products** in some high-dimensional (even infinite-D) feature space. A kernel $K(x, y)$ returns that inner product **without** building $\phi(x)$ explicitly. The model still finds a **linear separator** — but in the “lifted” space.

**Intuition:** Two classes curled in a ring in 2D? Instead of hand-engineering polar coordinates, you change the **similarity measure** (kernel) so a straight line exists **somewhere** sensible.

**Fashion nod:** Like judging **drape** vs. **color** — you didn’t add new fabric; you changed which **distance** defines “closer.”

---

## 4. Random Forest vs. gradient boosting

### Q: Compare bagging vs. boosting for trees.

| | Random Forest | Gradient boosting (XGBoost, etc.) |
| :--- | :--- | :--- |
| **Flow** | Parallel bagging | Sequential correction |
| **Main fight** | Variance | Bias (then watch variance) |
| **Overfitting** | Harder by default | Easier if untamed |
| **Ops feel** | Robust default | Strong but wants tuning |

**Feature importance (know both):**
- **Gini / impurity importance** — fast, can favor high-cardinality features; know the caveat.
- **Permutation importance** — shuffle feature, watch score drop; often more trustworthy for “what actually matters.”

---

## 5. Precision, recall, F1 — not academic trivia

### Q: Accuracy vs. precision vs. recall?

- **Precision** — “When we said positive, how often were we right?” Minimize **false alarms** when alarms are expensive.
- **Recall** — “Of all real positives, how many did we catch?” Minimize **misses** when misses are catastrophic.
- **F1** — Harmonic mean; punishes **one-sided** bragging (great precision but awful recall, or the reverse).

**Mini quiz:** *Fraud detection — optimize precision or recall first?* → Usually **recall** for catching fraud (then layer rules / review queues); exact answer depends on **cost matrix**, always say that out loud.

---

> **Production line:** For fat tabular data, **XGBoost / LightGBM** are default power tools. For images or long sequences, graduate to [Deep learning](../deep-learning/README.md) — wrong hammer, fancy nail.
