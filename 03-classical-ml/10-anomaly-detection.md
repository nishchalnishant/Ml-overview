---
module: Classical ML
topic: Anomaly Detection
subtopic: ""
status: unread
tags: [classicalml, ml, anomaly-detection, isolation-forest, one-class-svm, lof]
prerequisites: [probability, clustering, decision-trees]
---
# Anomaly Detection

## The Problem It Solves

You need to flag rare, unexpected events — fraud, intrusions, failing hardware, bot traffic — but you cannot pose it as ordinary classification. Two structural obstacles:

1. **You have almost no positive labels.** Fraud might be 0.1% of transactions, and the confirmed cases are only the fraud you *caught*. Your labels are biased toward attacks you already know how to detect.
2. **The next anomaly may not resemble the last one.** A supervised classifier trained on known fraud patterns gets high recall on those patterns and near-zero recall on a novel one — it learned specific signatures, not "unusual."

Anomaly detection inverts the framing: model what **normal** looks like, then flag whatever fails to fit. That generalizes to unseen anomaly types because you are modeling the *absence of normalcy*, not the presence of a known attack.

> **Decide this first:** if you have thousands of reliable labels and anomalies resemble each other, use supervised classification with class weighting — it will beat any method here. Reach for unsupervised detection when labels are scarce, unreliable, or when novel anomaly types are the actual threat. Choosing unsupervised when you have good labels is a common and expensive mistake.

---

## Intuition

Three different definitions of "weird," and your data decides which one is right:

| Framing | "Anomalous" means | Method family |
| :--- | :--- | :--- |
| **Isolation** | Easy to separate from everything else | Isolation Forest |
| **Boundary** | Outside the region normal data occupies | One-Class SVM, Elliptic Envelope |
| **Density** | In a sparser neighborhood than its neighbors sit in | LOF |
| **Reconstruction** | Hard to compress and rebuild | Autoencoder |

The isolation intuition is the one worth internalizing because it explains why the default works. Pick a random feature, pick a random split, repeat. A point in a dense cluster needs many splits to isolate. An outlier sitting alone gets cut off in two or three. **Path length is the anomaly score** — no distance metric, no density estimate, no assumption about distribution shape.

---

## The Mechanics

### Isolation Forest

Build `n_estimators` trees on random subsamples (default 256 points). At each node pick a random feature and a random split value. Isolate every point; record its path length `h(x)`.

Normalize against the average path length of an unsuccessful BST search, `c(n) = 2H(n−1) − 2(n−1)/n`:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Score → 1 means anomaly (very short paths), → 0.5 means normal. Complexity is O(n log n), and it never computes a pairwise distance — which is why it stays fast in high dimensions where LOF and One-Class SVM degrade.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=100,
    max_samples=256,        # subsample size; the paper's default, rarely needs tuning
    contamination=0.01,     # sets the decision threshold, NOT a model parameter
    random_state=42,
)
iso.fit(X_train)                       # fit on normal-ish data only
scores = -iso.score_samples(X_test)    # higher = more anomalous
```

### Local Outlier Factor

LOF compares a point's local density to that of its k neighbors. LOF ≈ 1 means same density as neighbors; LOF ≫ 1 means the point sits in a relatively sparse pocket.

This is the method that catches **local** anomalies — a point that is perfectly normal globally but wrong for its neighborhood. If your data has clusters of genuinely different densities, Isolation Forest will miss exactly these.

```python
from sklearn.neighbors import LocalOutlierFactor

# novelty=False: fit_predict on one dataset (outlier detection)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
labels = lof.fit_predict(X)            # -1 = outlier

# novelty=True: fit on clean data, then score NEW points
lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_train)
labels_new = lof_nov.predict(X_test)
```

### The others, briefly

- **One-Class SVM** — learns a boundary enclosing normal data; RBF kernel handles non-linear shapes. `nu` upper-bounds the outlier fraction. O(n²)–O(n³), so it fades past ~10⁴ samples. Needs scaling.
- **Elliptic Envelope** — robust Gaussian fit (Minimum Covariance Determinant), flags low-Mahalanobis-density points. Excellent *if* your data is roughly elliptical; badly wrong otherwise.
- **Autoencoder** — train on normal data, flag high reconstruction error. The choice for images, sequences, and rich multivariate telemetry. Needs enough data to train and a GPU budget.

---

## Worked Example

Six 1-D points: `2, 3, 4, 5, 6, 30`. Intuitively `30` is the anomaly. Watch isolation find it without any distance computation.

Build a tree by repeatedly picking a random split in the current range.

**Isolating 30:** range is [2, 30]. A uniformly random split lands above 6 with probability (30−6)/28 ≈ 0.86. One split, and `30` is alone.
→ **path length ≈ 1**

**Isolating 4:** it sits mid-cluster. First split, say at 15 → `{2,3,4,5,6}` and `{30}`. Split at 3.5 → `{2,3}` and `{4,5,6}`. Split at 5.5 → `{4,5}` and `{6}`. Split at 4.5 → `4` alone.
→ **path length ≈ 4**

With n = 6, `c(6) = 2H(5) − 2(5)/6 ≈ 2(2.283) − 1.667 ≈ 2.899`:

$$s(30) = 2^{-1/2.899} = 2^{-0.345} \approx \mathbf{0.79}$$
$$s(4) = 2^{-4/2.899} = 2^{-1.380} \approx \mathbf{0.38}$$

0.79 vs 0.38 — separated cleanly. The point to carry into an interview: **the anomaly was found because it was easy to isolate, not because it was far away.** No metric was ever computed. That is the whole idea, and it is why the method survives high dimensions.

---

## When It Breaks

| Failure | Why | What to do |
| :--- | :--- | :--- |
| **Contamination misread as a model parameter** | `contamination` only sets the score cutoff. Setting 0.05 on data with 0.1% anomalies forces a 50× over-flag. | Set it from a real base-rate estimate, or ignore it and threshold `score_samples` at a chosen percentile. |
| **Training data already contains anomalies** | The model learns them as normal and stops flagging them. | Filter aggressively first, or accept and monitor drift. Unsupervised ≠ needs no clean data. |
| **Local anomalies missed** | Isolation Forest is global; a point normal overall but wrong for its cluster is invisible. | Use LOF, or run per-segment models. |
| **High-cardinality categoricals** | One-hot explodes dimensionality; random splits mostly hit meaningless sparse columns. | Target/frequency encode, or embed. |
| **Seasonality flagged as anomaly** | Sunday traffic is not an outlier. | Deseasonalize first, or add time features so "3am" is context, not surprise. |
| **Evaluating with accuracy** | 99.9% accuracy by calling everything normal. | Precision@k, PR-AUC, recall at a fixed alert budget. Never accuracy, and prefer PR-AUC to ROC-AUC under extreme imbalance. |

---

## Production Notes

- **The threshold is a business decision, not a hyperparameter.** Work backward from capacity: analysts can review 500 alerts/day → set the cutoff at the 500th-highest score. Express it as an alert budget and the conversation with stakeholders becomes tractable.
- **Isolation Forest is the default** for tabular production work: CPU-only, ~1ms/vector, trivially parallel, no scaling required.
- **Precision@k is the metric that matters.** Analysts work a ranked queue; what counts is how many of the top *k* are real. A model with better PR-AUC but worse precision@100 is worse in practice.
- **Alert suppression is mandatory at scale.** A fleet-wide incident fires thousands of simultaneous alerts. Collapse correlated alerts into one event or you have built a pager DDoS.
- **Feed confirmed hits back.** Once you accumulate labels, a supervised model on *known* fraud plus unsupervised detection for *novel* fraud beats either alone. This hybrid is the mature architecture and a strong thing to volunteer.
- **Anomaly rates drift.** Attackers adapt. Monitor the flagged-fraction over time — a sudden drop usually means the model went stale, not that fraud stopped.

---

## Interview Angles

### Q: Why not just train a classifier on your labeled fraud examples? [Easy]

You can, and when labels are plentiful and anomalies look alike, you should — it will outperform unsupervised methods. The problem is structural: labels cover only fraud you already caught, so the classifier learns *known signatures* and generalizes poorly to a new attack. Unsupervised detection models normal behavior and flags deviation, so it can catch patterns never seen in training.

**Cross-questions to expect:**
- *"So which do you actually deploy?"* → Both. Supervised for known patterns (high precision), unsupervised for novel ones (coverage), unioned into one queue.
- *"How do you know the unsupervised model is finding anything real?"* → Precision@k on analyst-reviewed alerts, and count of confirmed hits the supervised model missed. That second number justifies its existence.

**Trap:** Claiming unsupervised is "better because it needs no labels." It needs no labels *for training* but you still need labels to evaluate it. A detector you cannot measure is not a system.

### Q: How does Isolation Forest actually assign a score? [Medium]

Random splits on random features until each point is isolated; the score is the normalized average path length across trees. Short path = anomaly. The insight is that anomalies are *few and different*, so random partitioning separates them early, while dense-region points need many cuts.

**Cross-questions to expect:**
- *"Why normalize by c(n)?"* → Raw path length depends on subsample size; c(n), the average unsuccessful-BST-search depth, makes scores comparable across n and yields the 0–1 range.
- *"Why is it fast in high dimensions?"* → It never computes a distance. Distance-based methods degrade as dimensions grow and points become equidistant; path length is unaffected.
- *"When does it fail?"* → Local anomalies in varying-density data, and irrelevant high-dimensional noise features that random splits waste depth on.

**Trap:** Saying "it isolates outliers because they're far away." Distance is never computed — the mechanism is *ease of separation*. Interviewers listen for exactly this.

### Q: Your fraud detector flags 5% of transactions. Fraud is ~0.1%. What happened? [Medium]

Almost certainly `contamination=0.05` left at a guessed value. It does not change what the model learned — it sets the percentile cutoff on scores. At 5% you are force-flagging the top 5% regardless of whether they are anomalous, so precision is ~2% at best and analysts drown.

**Cross-questions to expect:**
- *"How do you set it correctly?"* → From base rate if known; otherwise ignore it, take `score_samples`, and threshold at the percentile matching your alert budget.
- *"Base rate is unknown — now what?"* → Set by capacity, not statistics. 500 reviews/day → threshold at the 500th-highest daily score. Then measure precision@500 and tune.
- *"What if the score distribution has no clean gap?"* → Common and not disqualifying; it means "anomaly" is continuous here. Ship a ranked queue rather than a binary flag.

**Trap:** Treating `contamination` as a knob to "make the model better." It is a threshold, not a learning parameter. Retuning it trades precision against recall and nothing else.

### Q: Detector worked for six months, now catches nothing. Alert volume normal. Diagnose. [Hard]

Constant alert volume is the clue. If `contamination` is fixed, the model flags a fixed *fraction* by construction — so volume stays flat whether or not real fraud is present. Volume tells you nothing; only confirmed-hit rate does.

Ranked by likelihood:

1. **Retraining on contaminated data.** If it retrains on recent traffic and that traffic now contains the attack, the attack becomes "normal." Self-inflicted and the most common cause.
2. **Adversarial adaptation.** Attackers shifted below threshold — often deliberately, by pacing activity to look ordinary.
3. **Feature drift/breakage.** An upstream change makes a key feature constant or null; the model still scores, just meaninglessly.
4. **Genuine distribution shift.** Normal behavior moved (new product, new region) and the old boundary no longer describes it.

**Cross-questions to expect:**
- *"How do you distinguish 1 from 2?"* → Score the current model against a held-out labeled set from six months ago. Still detects → data/threshold issue. Also fails → the model itself degraded.
- *"How would you have caught this earlier?"* → Alert on confirmed-hit rate and on feature distributions, not alert volume. Volume is invariant by design.
- *"Prevent the retraining trap?"* → Never retrain on unfiltered production data. Exclude confirmed fraud and previously flagged records, hold out a clean reference set, and diff score distributions between model versions before promoting.

**Trap:** Jumping to "retrain the model." Retraining is what *caused* the most likely version of this failure. The first move is diagnosis against a labeled reference, not a fresh fit.

---

## Connections

- [Unsupervised Learning](02-unsupervised-learning.md) — clustering foundations; DBSCAN labels low-density points as noise, which is anomaly detection by another name
- [Imbalanced Data](../02-data/06-imbalanced-data.md) — when you *do* have labels and want supervised framing
- [ML Evaluation Metrics](../04-evaluation/01-ml-evaluation-metrics.md) — precision@k, PR-AUC, why accuracy is meaningless here
- [Autoencoders](../08-generative/01-autoencoders.md) — reconstruction-error detection for images and sequences
- [Time Series](../07-domains/06-time-series.md) — seasonality removal before flagging temporal anomalies
- [Fraud Detection Case Study](../15-system-design/cases/05-fraud-detection.md) — end-to-end system design
- [Real-Time Anomaly Detection (EA)](../16-interview-prep/ea/sde2-handbook/condensed/16-anomaly-detection.md) — streaming telemetry at fleet scale
