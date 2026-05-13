# When Classical ML Outperforms Deep Learning

## Executive Summary

The ML community's obsession with deep learning creates a systematic bias toward using neural networks where simpler models would win — on accuracy, latency, interpretability, and cost. This guide covers the concrete conditions under which XGBoost, logistic regression, random forests, and SVMs are the correct choice, not the lazy one.

| Signal | Lean Classical ML | Lean Deep Learning |
|--------|-------------------|--------------------|
| Data type | Tabular, structured | Images, text, audio |
| Dataset size | < 50k rows | > 100k examples |
| Inference latency SLA | < 1ms | > 10ms acceptable |
| Interpretability req. | Regulated industry | Consumer product |
| Calibration critical | Medical, fraud, insurance | Rankings, recommendations |
| Feature engineering available | Domain experts present | Raw input preferred |

---

## 1. Tabular Data Dominance

### Why XGBoost Beats Neural Nets on Most Kaggle Tabular Competitions

Between 2015 and 2024, gradient-boosted trees (XGBoost, LightGBM, CatBoost) won the majority of Kaggle tabular competitions. This is not coincidence.

**Structural reasons:**

- **Tabular data has no local correlation structure.** CNNs exploit spatial locality in images; Transformers exploit token proximity in text. In a CSV where column 3 is "age" and column 4 is "zip code", there is no spatial or sequential relationship the architecture can exploit. Trees just split on whatever feature and threshold minimizes loss — no inductive bias needed.

- **Decision boundaries in tabular data are often axis-aligned.** If a feature threshold like `income > $75k` separates classes, a tree finds it in one split. A neural network must learn this through multiple nonlinear activations.

- **Categorical features are natural for trees.** XGBoost handles high-cardinality categoricals (via CatBoost's target encoding or LightGBM's native categorical support) without the embedding overhead required by neural networks.

- **Trees are robust to feature scale.** Neural networks require normalization; trees don't care if one feature ranges from 0.001 to 0.003 and another from 10,000 to 9,000,000.

**Empirical evidence from the "Why does tree-based ML still outperform deep learning on tabular data?" (Grinsztajn et al., 2022, NeurIPS):**

- Tested on 45 diverse tabular datasets
- Tree-based models outperformed deep learning on 90% of datasets after hyperparameter tuning
- Key finding: neural networks struggled with uninformative features and irregular target functions

**What neural networks need to close the gap:**

- FT-Transformer (Feature Tokenizer + Transformer): treats each tabular feature as a token
- TabNet: sequential attention over features
- NODE (Neural Oblivious Decision Ensembles): differentiable trees

These architectures are still losing on most benchmarks and add significant operational complexity.

**Production implication:** On a tabular classification problem with moderate data, your first two models should be logistic regression (interpretable baseline) and XGBoost (strong baseline). A neural network is the third attempt, only if both fall short.

---

## 2. Sample Efficiency: When You Have Less Than 10k Labeled Examples

### The Fundamental Problem with Deep Learning and Small Data

Neural networks are function approximators with millions of parameters. Training a model with 5 million parameters on 5,000 examples is a severe overparameterization — the network will memorize the training set.

**Theoretical framing:** Generalization bounds (PAC learning, VC theory) suggest that the number of training examples required scales with model complexity. For a model with $d$ parameters, a rough sample requirement is:

$$n \geq O\left(\frac{d}{\epsilon^2} \log \frac{1}{\delta}\right)$$

A ResNet-50 has 25 million parameters. XGBoost with 100 trees of depth 6 has an effective complexity orders of magnitude lower.

**Classical ML advantages in low-data regimes:**

| Model | Why it works with small data |
|-------|------------------------------|
| Logistic Regression | Only $d+1$ parameters; strong regularization via L1/L2 |
| SVM with RBF kernel | Maximum margin principle; kernel trick avoids explicit feature expansion |
| Random Forest | Bagging introduces diversity from a small dataset; each tree sees a bootstrap sample |
| Naive Bayes | Assumes feature independence; needs per-feature statistics only |

**Practical thresholds (rules of thumb):**

- < 1,000 examples: logistic regression or SVM, period
- 1,000 – 10,000 examples: random forest or XGBoost with aggressive regularization
- 10,000 – 100,000 examples: gradient boosting is still usually competitive; try neural nets with strong regularization (dropout, weight decay, early stopping)
- > 100,000 examples: neural networks become competitive; > 1M, they typically win

**Transfer learning exception:** If a pretrained model exists for your domain (BERT for text, ResNet for images), fine-tuning can work with 500–2,000 examples. But this is borrowed sample efficiency from pretraining on millions of examples — still classical ML wins on truly small-data problems with no pretrained backbone available.

**Azure ML note:** Azure AutoML runs both tree-based and neural baselines in the same sweep. On datasets < 10k, AutoML's leaderboard will almost always have gradient boosting at the top.

---

## 3. Interpretability Requirements: Regulated Industries

### The Regulatory Landscape

Interpretability is not a nice-to-have in certain industries — it is a legal requirement.

**GDPR Article 22 (Right to Explanation):** Individuals have the right not to be subject to solely automated decisions that significantly affect them. When automated decisions are used, individuals must be able to request "meaningful information about the logic involved."

**US Fair Credit Reporting Act (FCRA):** Adverse action notices must provide specific reasons for credit denials. "The neural network said no" does not satisfy this requirement.

**EU AI Act (2024):** High-risk AI systems (credit scoring, employment, medical devices, critical infrastructure) must be transparent and explainable.

**FDA guidance for Software as a Medical Device (SaMD):** Requires the ability to explain model predictions to clinical staff.

### What "Interpretable" Means in Practice

Three levels of interpretability:

1. **Global interpretability:** Understand the overall model behavior. Which features drive predictions across the dataset? → Feature importance plots, SHAP summary plots.

2. **Local interpretability:** Explain a single prediction. Why did this customer get rejected? → SHAP force plots, LIME, counterfactual explanations.

3. **Simulatability:** A human can step through the model manually. A decision tree with depth ≤ 5 satisfies this; an XGBoost ensemble does not directly, but SHAP approximates it.

### Model-Level Interpretability

| Model | Interpretability mechanism |
|-------|---------------------------|
| Logistic Regression | Coefficients are log-odds; direct feature attribution |
| Decision Tree (shallow) | Fully simulatable; each path is a rule |
| Linear SVM | Weight vector directly interpretable |
| XGBoost + SHAP | Post-hoc but consistent and efficient (TreeSHAP is O(TLD²)) |

**When SHAP satisfies GDPR right-to-explanation:**

SHAP values (SHapley Additive exPlanations) provide consistent, locally accurate feature attributions. For a logistic regression or tree-based model, SHAP values can be computed exactly (not approximated). This means you can produce an adverse action notice: "Your application was declined primarily because your debt-to-income ratio (contribution: -0.34) and short credit history (contribution: -0.22) lowered the predicted creditworthiness score below the threshold."

**Neural networks and interpretability:** SHAP for neural networks uses KernelSHAP (slow, approximate) or DeepSHAP (fast but less accurate). Gradient-based attribution methods (Integrated Gradients, GradCAM) exist but don't produce the same legally defensible per-feature contribution scores. For regulated use cases, the overhead of interpretability infrastructure on neural networks is almost always higher than just using a tree-based model.

---

## 4. Latency-Critical Inference

### Decision Trees in Microseconds

Inference latency is often the deciding factor in production deployment.

**Benchmark figures (single prediction, CPU, no batching):**

| Model | Approximate inference latency |
|-------|-------------------------------|
| Logistic Regression | 1–5 microseconds |
| Decision Tree (depth 10) | 5–20 microseconds |
| Random Forest (100 trees) | 0.5–2 milliseconds |
| XGBoost (100 trees, depth 6) | 1–5 milliseconds |
| Small MLP (3 layers, 256 units) | 0.5–2 milliseconds (CPU) |
| BERT-base | 100–500 milliseconds (CPU) |

**Where microsecond latency matters:**

- High-frequency trading systems: risk scoring per order at nanosecond to microsecond scale
- Ad auction bidding: 5–10ms budget for the entire pipeline including model inference
- Network intrusion detection: per-packet scoring at line rate
- Embedded systems: microcontroller inference with no GPU

**Compilation and optimization for classical ML:**

Decision trees can be compiled to native if-else C code, achieving instruction-level efficiency. Libraries like `treelite` compile XGBoost/LightGBM models to optimized C/C++ shared libraries.

```python
import treelite
model = treelite.Model.load("model.json", model_format="xgboost")
model.export_lib(toolchain="gcc", libpath="./model.so", verbose=True)
# Compiled model: ~3x faster than native XGBoost predict()
```

**Azure DevOps note:** Azure ML's managed online endpoints support sub-10ms SLAs on CPU SKUs for gradient boosting models. Neural networks typically require GPU endpoints to meet equivalent SLAs, increasing cost by 5–10x.

---

## 5. Class Imbalance Handling

### SMOTE, Cost-Sensitive Learning vs Neural Focal Loss

Class imbalance (e.g., 1% fraud in a transaction dataset) is the default in many production ML problems. Classical ML has mature, well-understood tooling for this.

**Classical approaches:**

**SMOTE (Synthetic Minority Oversampling Technique):**
- Generates synthetic minority examples by interpolating between existing minority examples in feature space
- Works in the original feature space — no latent representation needed
- Variants: ADASYN (adaptive density), Borderline-SMOTE (focuses on decision boundary)

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

**Cost-sensitive learning:**
- Assign higher misclassification cost to minority class
- Directly optimizable in tree-based models via `scale_pos_weight` (XGBoost) or `class_weight` (sklearn)

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    scale_pos_weight=99,  # ratio of negatives to positives for 1% minority
    eval_metric="aucpr"   # PR-AUC preferred over ROC-AUC for imbalance
)
```

**Threshold optimization:**
- Models output probabilities; the decision threshold is a hyperparameter
- For imbalanced problems, the default 0.5 threshold is almost always wrong
- Optimize threshold on validation set using PR-AUC or F-beta score

**Neural Focal Loss (for comparison):**
- $FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
- Down-weights easy negatives dynamically during training
- Requires tuning $\alpha$ and $\gamma$; sensitive to learning rate interactions
- Works well for object detection (RetinaNet); less clear advantage on tabular fraud data

**Why classical ML wins on imbalanced tabular data:**

- SMOTE + XGBoost is a proven combination with extensive production history
- Cost-sensitive learning requires no data augmentation; no synthetic data quality concerns
- Threshold calibration is model-agnostic and easy to tune post-training
- Neural focal loss adds another hyperparameter interaction on top of already complex DL tuning

---

## 6. Calibration: When Probabilities Need to Be Trusted

### The Calibration Problem

A model that outputs `P(fraud) = 0.85` should be correct 85% of the time among all predictions at that score. If it's only correct 60% of the time, the model is miscalibrated — its probabilities cannot be used as probabilities.

**Why calibration matters for classical ML choice:**

- Logistic regression is natively calibrated (its optimization objective is log-loss, which is proper scoring rule)
- Neural networks and SVMs are often badly miscalibrated out of the box
- XGBoost probabilities are better calibrated than SVMs but still benefit from post-hoc calibration

**Post-hoc calibration methods:**

- **Platt scaling:** Fit a logistic regression on the raw scores. Works well for SVMs.
- **Isotonic regression:** Non-parametric monotone fit. Works better when calibration curve is highly non-linear.
- **Temperature scaling:** Divide logits by a scalar $T$ before softmax. Best for neural networks (see `calibration-and-uncertainty.md` in this directory).

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
# Now predict_proba() outputs calibrated probabilities
```

**Use calibrated classical models when:** medical risk scoring, insurance premium pricing, fraud score thresholding with monetary consequences, credit decisioning.

---

## 7. Feature Engineering Still Matters

### The Feature Engineering Advantage of Classical ML

Deep learning is often sold as "automatic feature engineering." This is true for images and text where the raw input is homogeneous. For tabular data, domain-derived features beat raw inputs — and classical ML gives you full control over the feature space.

**Where feature engineering wins:**

| Domain | Hand-engineered feature | Lift over raw input |
|--------|------------------------|---------------------|
| Finance | Debt-to-income ratio, 30/60/90-day delinquency | High |
| E-commerce | Days since last purchase, purchase frequency, basket size | High |
| Healthcare | Change in lab value over 30 days, medication adherence rate | Very high |
| Manufacturing | Rolling mean/std of sensor reading, vibration FFT peaks | High |

**Classical ML + feature engineering workflow:**

1. Exploratory analysis: identify monotone relationships, saturation points, interaction effects
2. Create derived features: ratios, lags, rolling statistics, log transforms for skewed distributions
3. Fit a logistic regression with L1 regularization: coefficients reveal which features survive
4. Add interaction terms for the top features; refit
5. Promote to XGBoost for nonlinear boundaries, using the same feature set

**Neural network limitation:** Embedding layers and attention mechanisms can learn feature interactions, but they require sufficient data to generalize. With 5,000 rows and 50 features, a neural network will overfit to noise in the interaction space. A domain expert who derives 5 meaningful ratio features and uses logistic regression will beat it.

---

## 8. The Decision Tree: When to Choose Classical ML

### The Rule-of-Thumb Decision Framework

```
Input: Problem specification
│
├── Data type = Images / Raw Text / Audio?
│   └── YES → Deep Learning (CNN, Transformer, etc.)
│
├── Data type = Tabular / Structured?
│   │
│   ├── n_samples < 10,000?
│   │   └── YES → Classical ML (Logistic Regression, SVM, Random Forest)
│   │
│   ├── Interpretability required (regulated industry, GDPR)?
│   │   └── YES → Classical ML (Logistic Regression, Decision Tree, XGBoost + SHAP)
│   │
│   ├── Latency SLA < 5ms?
│   │   └── YES → Classical ML (compiled tree or logistic regression)
│   │
│   ├── Calibrated probabilities required?
│   │   └── YES → Logistic Regression (natively) or Classical ML + Platt/Isotonic
│   │
│   └── None of the above apply?
│       └── Try XGBoost first, then neural nets if XGBoost plateaus
│
└── Pretrained model available for domain?
    └── YES → Fine-tuning beats classical ML even at small n
```

**Summary:** The default for new tabular ML problems should be classical ML. The burden of proof is on deep learning.

---

## 9. Azure and Production Bridges

**Azure ML AutoML:** Runs XGBoost, LightGBM, Random Forest, Logistic Regression, and optional neural baselines in a single sweep. For tabular datasets, AutoML's top models are almost always gradient-boosting variants. Examine the leaderboard before escalating to custom neural architectures.

**Azure ML Responsible AI Dashboard:** Generates SHAP-based explanations, calibration plots, and fairness assessments for classical ML models out of the box. For neural networks, the same dashboard requires SHAP approximations (slower) and doesn't support all architectures.

**Azure Kubernetes Service (AKS) for inference:** Deploying a compiled XGBoost model on a CPU AKS pod costs a fraction of a GPU pod running a neural network. For latency-critical applications, classical ML enables cost-efficient horizontal scaling.

**Model monitoring:** Azure ML's data drift and prediction drift monitoring works at the feature level — which is directly interpretable for classical ML. Neural network feature drift monitoring requires monitoring input distributions only (not latent representations), so the monitoring story is equivalent.

---

## Interview Q&A

**Q1: When would you use XGBoost over a neural network?**

> On tabular data, especially with < 50k examples, when inference latency must be < 5ms, when you need SHAP-based explanations for regulatory compliance, or when you have domain experts who can engineer meaningful features. XGBoost is the default for structured data; neural networks require justification.

**Q2: Can you explain why deep learning underperforms on tabular data?**

> Tabular data lacks the local correlation structure that inductive biases in CNNs and Transformers exploit. Features are heterogeneous (age, zip code, income) with no spatial or sequential relationship. Trees split on arbitrary thresholds, which matches the axis-aligned or piecewise constant nature of many tabular decision boundaries. Neural networks must learn these boundaries through layers of smooth nonlinearities, which is less efficient for this structure.

**Q3: What is SMOTE and when would you use it?**

> SMOTE generates synthetic minority class examples by interpolating between real minority examples in feature space. Use it when the minority class has < 5% prevalence and you have enough minority examples that interpolation is meaningful (> 50 minority examples). Prefer it over random oversampling because it introduces diversity rather than duplicates. Always apply SMOTE only on training data, never before the train/test split.

**Q4: How do you handle class imbalance in XGBoost?**

> Three levers: (1) `scale_pos_weight = n_negatives / n_positives` to upweight minority class gradient contributions; (2) `eval_metric = "aucpr"` to optimize for precision-recall rather than ROC-AUC, which inflates on imbalanced datasets; (3) threshold optimization on a held-out validation set using the F-beta score with beta chosen based on relative cost of false negatives vs false positives.

**Q5: What does "calibrated probability" mean and why does it matter?**

> A model is calibrated if among all predictions of probability $p$, approximately fraction $p$ of the outcomes are positive. It matters when the probability is used downstream — for example, a fraud score of 0.9 triggers a manual review queue; if the model outputs 0.9 for cases that are only 50% fraud, the review team will waste resources on false positives. Logistic regression is naturally calibrated; SVMs and neural networks typically require post-hoc calibration.

**Q6: A business stakeholder says "the black box made the decision." How do you respond?**

> I would present SHAP values for the specific prediction. For an XGBoost model, TreeSHAP provides exact, consistent feature attributions: "This application was denied because the debt-to-income ratio contributed -0.31 to the score and the number of recent inquiries contributed -0.19." This is legally compliant with GDPR Article 22 and FCRA adverse action notice requirements, and it's actionable for the applicant.

**Q7: What is your process for selecting a model on a new ML problem?**

> Start with problem framing: data modality, dataset size, latency SLA, interpretability requirements, calibration requirements. If tabular and < 50k: logistic regression (interpretable baseline), then XGBoost. If tabular and > 100k: XGBoost first, neural net if it plateaus. If images/text/audio: deep learning from the start, starting with a pretrained backbone. Never skip the logistic regression baseline — it reveals which features have linear signal and gives a floor for XGBoost to beat.

**Q8: When would you choose logistic regression over XGBoost?**

> When you need native probability calibration (XGBoost requires post-hoc calibration), when the relationship between features and target is approximately log-linear (logistic regression will generalize better), when you have very few examples (< 1,000), when inference must be a single matrix multiply (< 1 microsecond), or when the deployment environment doesn't support tree inference libraries.

**Q9: How do you justify using a "simple" model to a team that wants to use a neural network?**

> I run both and show the numbers. For tabular data, XGBoost typically matches or beats neural nets in 80% of cases. The remaining 20%, where neural nets win, are problems with complex interaction patterns and large datasets. The argument is not ideology — it's that classical ML has lower latency, lower infrastructure cost, native interpretability, and competitive accuracy on the problem class we're working on. If the neural network is materially better (> 1-2% on the business metric), we switch.

**Q10: What are the failure modes of SMOTE?**

> SMOTE interpolates in feature space, which assumes the minority class region is convex and the interpolated points are realistic. Failure modes: (1) if the minority class is multimodal (multiple clusters), interpolation between clusters produces unrealistic synthetic points; (2) for categorical features, interpolation doesn't make semantic sense (SMOTE-NC handles this partially); (3) if the minority class is very small (< 20 examples), the synthetic points have high variance and may introduce noise. In these cases, prefer cost-sensitive learning or class weighting over synthetic oversampling.
