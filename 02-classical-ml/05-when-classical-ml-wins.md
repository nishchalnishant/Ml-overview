---
module: Classical Ml
topic: When Classical Ml Wins
subtopic: ""
status: unread
tags: [classicalml, ml, when-classical-ml-wins]
---
# When Classical ML Wins

---

## The Problem This Answers

**The problem**: You have a tabular dataset, a business deadline, and a colleague insisting you need a neural network. You need a principled answer to "when does classical ML produce better results — not just faster results — than deep learning?" Without that answer, you will routinely build systems that are slower, less interpretable, harder to maintain, and no more accurate than what gradient boosting would have given you in a tenth of the time.

**The core insight**: Deep learning's advantages — representation learning, spatial and sequential inductive biases, scaling laws — are irrelevant for structured tabular data with no local correlation structure. On that problem class, gradient-boosted trees dominate empirically, and the theoretical reasons are well understood. The question is not "which is smarter?" but "which inductive bias matches the problem?"

---

## Tabular Data Dominance

**The problem**: You have a CSV with 40 columns — age, income, zip code, account balance, days since last login. A neural network researcher says to tokenize each feature and run a Transformer. An ML engineer says to run XGBoost. You need to know who is right and why.

**The core insight**: CNNs exploit spatial locality. Transformers exploit token proximity. Gradient-boosted trees make no assumption about feature relationships — they split on whatever threshold minimizes loss. Tabular data has no spatial or sequential structure to exploit, so the inductive biases of deep learning architectures provide no advantage, and their additional complexity becomes pure overhead.

**The mechanics**: The empirical record is decisive. Grinsztajn et al. (NeurIPS 2022) tested 45 diverse tabular datasets: tree-based models outperformed deep learning on 90% after hyperparameter tuning. The key failure mode for neural networks: irregular, locally constant target functions and uninformative features — both common in real tabular data. Trees split axis-aligned; neural networks must approximate axis-aligned boundaries through layers of smooth nonlinearities, which is less efficient for this structure.

Categorical features illustrate the gap concretely. A column with 5,000 unique zip codes requires a learned embedding in a neural network (training complexity, memory overhead, embedding quality). A tree splits on whatever threshold within that column minimizes impurity — no embedding needed.

**What breaks**: Neural architectures for tabular data (FT-Transformer, TabNet, NODE) have closed some of the gap on specific benchmarks. They remain operationally heavier and rarely exceed gradient boosting on medium-sized datasets. The correct default is still: logistic regression (interpretable baseline) → XGBoost (strong baseline) → neural network (only if both fall short with justification).

---

## Sample Efficiency

**The problem**: You have 3,000 labeled examples. A team member proposes fine-tuning a large neural network. You know this will overfit. You need to explain why, and what to use instead.

**The core insight**: Generalization bounds scale with model complexity. A model with $d$ parameters needs roughly $O(d / \epsilon^2)$ examples to generalize to error $\epsilon$. A three-layer MLP with 256 hidden units has ~250,000 parameters. XGBoost with 100 trees of depth 6 has effective complexity orders of magnitude lower. On 3,000 examples, XGBoost has the right parameter-to-data ratio; the MLP does not.

**The mechanics**:

| Sample range | Recommended approach | Reasoning |
|---|---|---|
| < 1,000 | Logistic regression or SVM | Very few parameters; strong regularization |
| 1,000–10,000 | Random forest or XGBoost with regularization | Ensemble diversity compensates for small n |
| 10,000–100,000 | XGBoost usually wins; try neural nets with strong regularization | Competitive regime |
| > 100,000 | Neural networks become competitive | Sufficient data to overcome parameter count |

**What breaks**: Transfer learning breaks this rule. If a pretrained backbone exists for the domain — BERT for text, ResNet for images — fine-tuning borrows sample efficiency from pretraining on millions of examples. This is not deep learning winning at small data; it is deep learning inheriting supervised signal from a massive prior training run. For genuinely small tabular problems with no pretrained backbone, classical ML still wins.

---

## Interpretability Requirements

**The problem**: Your model denies a loan application. The applicant asks why. The regulator asks why. Under GDPR Article 22, FCRA adverse action requirements, and the EU AI Act, "the model said no" is not a sufficient answer. You need per-prediction, per-feature explanations that survive legal scrutiny.

**The core insight**: Classical ML models either are directly interpretable (logistic regression coefficients are log-odds; a shallow decision tree is a set of human-readable rules) or have exact, efficient post-hoc explanations (TreeSHAP computes exact Shapley values in O(TLD²) for gradient-boosted trees). Neural network explanations (KernelSHAP, Integrated Gradients, GradCAM) are approximations — they are slower, less consistent, and harder to defend in a regulatory context.

**The mechanics**: Three levels of interpretability, and where classical ML satisfies each:

- **Simulatability**: A human steps through the model manually. A decision tree with depth ≤ 5 satisfies this. No neural network does.
- **Local explanations**: Explain one prediction. Logistic regression coefficients × feature values give exact local attribution. TreeSHAP gives exact local attribution for gradient-boosted trees in milliseconds.
- **Global feature importance**: Which features drive the model across the dataset. MDI importance from random forests (biased toward high-cardinality features) or permutation importance and SHAP summary plots (unbiased) work directly.

For a GDPR-compliant adverse action notice on a loan denial:

```
Reason for decline:
  Debt-to-income ratio:        -0.34  (primary factor)
  Months of credit history:    -0.22  (secondary factor)
  Number of recent inquiries:  -0.11  (tertiary factor)
```

This is a SHAP force plot on an XGBoost model, computed exactly with TreeSHAP, legally defensible, and actionable for the applicant.

**What breaks**: SHAP for neural networks uses KernelSHAP (sampling-based, slow, approximate) or GradientSHAP (gradient-based, faster but tied to smoothness assumptions). These produce defensible explanations in practice, but they are not exact — a court or regulator can challenge them. For high-stakes regulated decisions, the overhead of building a defensible explanation layer on neural networks almost always exceeds the cost of just using a tree-based model.

---

## Latency-Critical Inference

**The problem**: Your system needs to score 100,000 transactions per second. Or respond within 2ms to an ad auction bid. A neural network requires GPU hardware and 50ms per batch. You need a model that fits inside the latency budget.

**The core insight**: A compiled decision tree is a sequence of if-else branches — the CPU executes them at instruction speed, cache-resident, branch-predicted. Logistic regression is a dot product followed by a sigmoid. Both run in microseconds. A neural network is matrix multiply after matrix multiply, typically requiring vectorization, memory bandwidth, and GPU acceleration to achieve comparable throughput.

**The mechanics**:

| Model | Single-prediction CPU latency |
|---|---|
| Logistic Regression | 1–5 microseconds |
| Decision Tree (depth 10) | 5–20 microseconds |
| Random Forest (100 trees) | 0.5–2 milliseconds |
| XGBoost (100 trees, depth 6) | 1–5 milliseconds |
| Small MLP (3 layers, 256 units) | 0.5–2 milliseconds (CPU) |
| BERT-base | 100–500 milliseconds (CPU) |

For even greater speedup, gradient-boosted trees can be compiled to native C:

```python
import treelite
model = treelite.Model.load("model.json", model_format="xgboost")
model.export_lib(toolchain="gcc", libpath="./model.so", verbose=True)
# Compiled model runs ~3x faster than native XGBoost predict()
```

**What breaks**: Neural networks close the latency gap with batch inference — scoring 512 examples simultaneously on a GPU becomes cheap per-example. If your system can batch predictions (e.g., a recommendation engine scoring many users at once), neural networks become competitive on throughput even if single-prediction latency is high. For true single-request, low-latency requirements (HFT, per-packet network scoring, embedded systems), classical ML is the only viable option.

---

## Class Imbalance Handling

**The problem**: Your fraud dataset is 0.5% positive. Standard cross-entropy loss is dominated by the 99.5% negatives. The model predicts "not fraud" for everything and achieves 99.5% accuracy. You need a mechanism to force the model to learn the minority class.

**The core insight**: Classical ML addresses this at multiple independent levers: resampling (change the data distribution), cost-sensitive learning (change the loss weights), and threshold moving (change the decision boundary post-training). These levers compose cleanly. Neural networks address this mainly through focal loss, which is one mechanism that requires careful hyperparameter tuning and interacts with the learning rate.

**The mechanics**:

**Cost-sensitive learning** in XGBoost sets `scale_pos_weight = n_negatives / n_positives`. For 0.5% fraud: `scale_pos_weight = 199`. The boosting gradient updates are scaled by this weight, giving 199× more pull from minority examples. No data augmentation, no synthetic data quality concerns.

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    scale_pos_weight=199,
    eval_metric='aucpr'   # PR-AUC, not ROC-AUC — see imbalanced-data.md
)
```

**SMOTE** generates synthetic minority examples by interpolating between existing minority examples in the original feature space:

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

**Threshold moving**: the default 0.5 decision threshold is wrong for imbalanced problems. Sweep the precision-recall curve and select the threshold that achieves the target precision, target recall, or optimal F-beta.

**What breaks**: SMOTE interpolates in feature space, assuming the minority class is convex. Failure modes: multimodal minority classes (interpolation between clusters creates unrealistic synthetic points); categorical features (interpolation doesn't make semantic sense; use SMOTE-NC instead); very small minority class (< 20 examples — synthetic points have high variance). In these cases, cost-sensitive learning (class weights) is safer than synthetic oversampling.

---

## Calibration

**The problem**: Your fraud model scores an application at 0.85. The risk team uses this number to set reserve requirements. If the model is miscalibrated — applicants scored at 0.85 default only 40% of the time — the reserve is wrong, and the financial exposure is mispriced.

**The core insight**: Logistic regression is natively calibrated because it directly optimizes log-loss, a proper scoring rule. Its predicted probabilities satisfy $P(Y=1 \mid \hat{p}(X)=p) \approx p$ without any post-hoc adjustment. Neural networks and SVMs optimized for accuracy or margin are not calibrated by default and require an additional calibration step.

**The mechanics**: see the per-model calibration behavior table in [14-calibration-and-uncertainty.md](14-calibration-and-uncertainty.md).

Post-hoc calibration for any classical model:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

When calibrated probabilities are a hard requirement (medical risk scoring, insurance pricing, credit decisioning), using logistic regression as the base model eliminates the calibration step entirely. This is a non-trivial operational advantage over neural networks, which require a held-out calibration set, a temperature scaling pass, and ongoing calibration monitoring.

**What breaks**: Logistic regression's native calibration holds only when the model is well-specified — when the log-odds are approximately linear in the features. With strong nonlinear interactions and feature transformations, the linearity assumption breaks and calibration degrades. In that case, apply isotonic regression calibration post-hoc. Platt scaling (logistic regression on raw scores) is the right choice when the calibration set is small (< 500 examples); isotonic regression overfits with small calibration sets.

---

## Feature Engineering Advantage

**The problem**: You have sensor readings from a factory machine. The raw readings show no predictive signal. But a domain expert tells you that "vibration increase over the past 24 hours relative to a 30-day baseline" is the leading indicator of failure. A neural network that sees raw readings will take hundreds of thousands of examples to discover this interaction. A classical model with this feature as an engineered input will find it immediately.

**The core insight**: Neural networks learn feature representations from raw inputs — this is an advantage when the raw input is rich (pixels, tokens) and labeled data is plentiful. For tabular data where domain experts can encode relationships directly, the learned representations are unnecessary. A ratio feature encodes a multiplicative interaction that a tree finds in one split and a logistic regression models directly.

**The mechanics**:

| Domain | Engineered feature | Advantage over raw input |
|---|---|---|
| Credit risk | Debt-to-income ratio, 30/60/90-day delinquency counts | High — captures relationships trees would need many splits to approximate |
| E-commerce | Days since last purchase, purchase velocity, basket size | High — temporal patterns invisible in raw transaction records |
| Healthcare | Change in lab value over 30 days, medication adherence rate | Very high — clinical domain knowledge encoded directly |
| Manufacturing | Rolling mean/std of sensor reading, FFT peaks of vibration signal | High — frequency-domain features invisible to models on raw time series |

The workflow: exploratory analysis to identify monotone relationships and saturation points → derive ratio and lag features → fit logistic regression with L1 to identify which features survive → add interaction terms for the top survivors → promote to XGBoost. The neural network, if attempted, must discover all of this from raw inputs with no guarantees.

**What breaks**: Feature engineering requires domain expertise that is not always available. For domains where no one knows what the important derived features are — novel biological signals, unusual financial instruments — neural networks that learn representations from data may be the only path. Feature engineering also encodes assumptions; if the domain relationship changes, the engineered feature becomes wrong and the model degrades faster than a learned representation would.

---

## Decision Framework

**The problem**: You are starting a new ML project and need to choose a model class before writing a line of code. Without a systematic framework, this becomes an unstructured debate based on preferences and hype.

**The core insight**: The choice depends on four observable properties of the problem, all determinable before modeling: data modality, sample count, latency SLA, and interpretability requirements. These properties uniquely determine the right starting point.

```
Input: problem specification

├── Data type = images, raw text, audio?
│   └── YES → deep learning; start with a pretrained backbone

├── Data type = tabular / structured?
│   │
│   ├── n < 10,000?
│   │   └── YES → logistic regression or XGBoost with regularization
│   │
│   ├── Interpretability required (GDPR, FCRA, EU AI Act)?
│   │   └── YES → logistic regression + SHAP or XGBoost + TreeSHAP
│   │
│   ├── Single-prediction latency SLA < 5ms?
│   │   └── YES → compiled tree or logistic regression
│   │
│   ├── Calibrated probabilities required?
│   │   └── YES → logistic regression (natively calibrated) or XGBoost + isotonic regression
│   │
│   └── None of the above?
│       └── XGBoost first; neural network only if XGBoost plateaus

└── Pretrained model exists for domain?
    └── YES → fine-tuning; classical ML is unlikely to win here
```

**What breaks**: This framework assumes the problem is reasonably stationary and the dataset is representative of deployment. Under severe distribution shift, model choice matters less than monitoring and retraining. For problems where the decision boundary changes faster than retraining cycles, online learning methods (not in this framework) may dominate both classical ML and deep learning.

---

## Practical Comparison Table

| Condition | Classical ML advantage | Deep learning advantage |
|---|---|---|
| Tabular data | Axis-aligned splits, no inductive bias needed | — |
| n < 10,000 | Low parameter count; strong regularization | — |
| n > 500,000 | — | Scaling laws; representation learning |
| Latency < 5ms | Microsecond inference; compilable | — |
| Images / text / audio | — | Spatial/sequential inductive bias; pretrained backbones |
| GDPR / FCRA / EU AI Act | Exact SHAP; simulatable models | Approximate explanations only |
| Calibrated probabilities | Logistic regression natively; cheap post-hoc calibration | Temperature scaling required; additional complexity |
| Class imbalance | Composable levers: class weights + SMOTE + threshold | Focal loss; one mechanism |
| Domain expertise available | Feature engineering directly encodes domain knowledge | Domain knowledge must be discovered from data |
| No domain expertise, novel domain | — | Learned representations from raw input |

For active-recall drilling on these terms, see [classical-ml-flashcards.md](classical-ml-flashcards.md).
