# Interpretability and Explainable AI (XAI)

> ML interview prep guide — conversational, analogy-rich, practically focused.

---

## Table of Contents

1. [Why Interpretability?](#1-why-interpretability)
2. [Taxonomy](#2-taxonomy-intrinsic-vs-post-hoc-local-vs-global-model-agnostic-vs-model-specific)
3. [Linear Models & Coefficients](#3-linear-models--coefficients-the-simplest-explanation)
4. [Decision Tree Explanations](#4-decision-tree-explanations-rules-as-stories)
5. [Feature Importance](#5-feature-importance-permutation-importance-impurity-based)
6. [PDPs and ICE Plots](#6-partial-dependence-plots-pdps-and-ice-plots)
7. [LIME](#7-lime-local-interpretable-model-agnostic-explanations)
8. [SHAP](#8-shap-shapley-additive-explanations)
9. [Attention Visualization](#9-attention-visualization-what-attention-weights-do-and-dont-tell-us)
10. [Saliency Maps and Grad-CAM](#10-saliency-maps-and-grad-cam-for-cnns)
11. [Integrated Gradients](#11-integrated-gradients)
12. [Concept-Based Explanations (TCAV)](#12-concept-based-explanations-tcav)
13. [Model Cards and Fact Sheets](#13-model-cards-and-fact-sheets)
14. [Interpretability in LLMs](#14-interpretability-in-llms-logit-lens-probing-classifiers-mechanistic-interpretability)
15. [Regulatory Context](#15-regulatory-context-gdpr-financial-models)
16. [Common Interview Questions](#16-common-interview-questions-with-answers)

---

## 1. Why Interpretability?

Imagine you go to a doctor and they say: "You have cancer. I just know." No test results, no reasoning, no explanation. You'd walk out. Trust requires explanation.

That's the core tension in modern ML. We've built systems that are extraordinarily accurate — deep neural networks, gradient boosted trees, large language models — but they often function as black boxes. They take in inputs and produce outputs, but the "why" is buried under millions of parameters.

**Why this matters in practice:**

- **Debugging**: If your model is wrong, you need to know *where* it's wrong and *why*. Is it overfitting a spurious correlation? Relying on a protected attribute?
- **Trust**: Stakeholders — doctors, loan officers, judges — won't use a system they don't understand.
- **Fairness**: A model might discriminate not because it uses race directly, but because it uses zip code as a proxy. You can only find that by looking inside.
- **Regulation**: GDPR gives EU citizens the right to an explanation when an automated system makes a decision about them. Financial regulators require model documentation.
- **Improvement**: Understanding why a model fails is the fastest path to making it better.

**The accuracy-interpretability tradeoff** is real but often overstated. A linear model is highly interpretable but may underfit. A neural network may be more accurate but harder to explain. The field of XAI is about closing that gap — building tools that let us understand complex models without necessarily simplifying them.

**Interpretability vs Explainability:**

These terms are often used interchangeably, but there's a useful distinction:

- **Interpretability**: The model's structure is inherently understandable. A linear regression is interpretable because you can read the coefficients.
- **Explainability**: We apply post-hoc techniques to explain a model that isn't inherently transparent. SHAP values explain a random forest, but the forest itself is not interpretable.

Think of it this way: a glass box is interpretable (you can see inside). A black box with a camera system that shows you what's happening inside is explainable.

---

## 2. Taxonomy: Intrinsic vs Post-hoc, Local vs Global, Model-agnostic vs Model-specific

Before diving into methods, it helps to have a map.

### 2.1 Intrinsic vs Post-hoc

**Intrinsic (transparent) models** are interpretable by design:
- Linear/logistic regression
- Decision trees
- Rule-based systems
- Generalized additive models (GAMs)

**Post-hoc methods** are applied *after* training to explain any model:
- LIME, SHAP, Grad-CAM, saliency maps
- Partial dependence plots
- Feature importance

The post-hoc vs intrinsic distinction matters because post-hoc explanations are approximations. They explain a simplified surrogate, not the model itself. They can be wrong or misleading.

### 2.2 Local vs Global

**Local explanation**: Why did the model predict *this specific instance* the way it did?
- "Why was this loan application rejected?"
- LIME, SHAP force plots, counterfactual explanations

**Global explanation**: What does the model do *overall*?
- "What features does this model rely on most?"
- Feature importance, PDPs, SHAP summary plots

Think of local as explaining one verdict vs global as describing the judge's general philosophy.

### 2.3 Model-agnostic vs Model-specific

**Model-agnostic** methods treat the model as a black box. They only need access to inputs and outputs:
- LIME
- KernelSHAP
- Permutation importance
- PDPs

**Model-specific** methods exploit the internal structure:
- Coefficients (linear models)
- TreeSHAP (tree ensembles)
- DeepSHAP / Grad-CAM (neural networks)
- Attention weights (transformers)

Model-specific methods are usually faster and more accurate because they have more information to work with.

### Summary Table

| Method | Scope | Agnostic? | Post-hoc? |
|---|---|---|---|
| Coefficients | Global | No | No (intrinsic) |
| Decision tree rules | Global + Local | No | No (intrinsic) |
| Permutation importance | Global | Yes | Yes |
| PDP / ICE | Global / Local | Yes | Yes |
| LIME | Local | Yes | Yes |
| KernelSHAP | Local + Global | Yes | Yes |
| TreeSHAP | Local + Global | No (tree) | Yes |
| Grad-CAM | Local | No (CNN) | Yes |
| Integrated Gradients | Local | No (NN) | Yes |
| Attention visualization | Local | No (transformer) | Yes |
| TCAV | Global (concept) | No (NN) | Yes |

---

## 3. Linear Models & Coefficients: The Simplest Explanation

Linear regression is the benchmark of interpretability. The model is literally a sum of weighted features:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

Each coefficient `βᵢ` tells you: holding everything else constant, a one-unit increase in `xᵢ` changes the prediction by `βᵢ`. That's a clean, causal-sounding (though not truly causal) statement.

**For logistic regression**, the interpretation extends to log-odds:

```
log(p / (1-p)) = β₀ + β₁x₁ + ...
```

`exp(βᵢ)` is the odds ratio — how the odds of the positive class change with a one-unit increase in `xᵢ`.

### Caveats

**Standardization matters**. Raw coefficients are not comparable across features with different scales. Always standardize before comparing magnitudes.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Example: predicting house prices
X = pd.DataFrame({
    'sqft': [1500, 2000, 1200, 1800],
    'bedrooms': [3, 4, 2, 3],
    'age': [10, 5, 20, 8]
})
y = [300000, 400000, 240000, 360000]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)

print(coef_df)
# Larger absolute coefficient = more influential feature (on standardized scale)
```

**Multicollinearity breaks interpretation**. If `sqft` and `rooms` are highly correlated, the model might assign a negative coefficient to one of them — not because it's actually negative, but because they share explanatory power. The coefficients become unstable and hard to interpret.

**Regularization shrinks coefficients**. In Ridge or Lasso regression, coefficients are penalized. Lasso (L1) drives some to exactly zero, giving you implicit feature selection. The surviving coefficients still have the same "one unit change" interpretation.

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Features with zero coefficient were "selected out"
selected = [(feat, coef) for feat, coef 
            in zip(X.columns, lasso.coef_) if coef != 0]
print("Selected features:", selected)
```

**The linear model is a strong baseline** in interpretability. When a linear model performs nearly as well as a complex model, just use the linear model. The interpretability is free.

---

## 4. Decision Tree Explanations: Rules as Stories

A decision tree is interpretable in the most literal sense — you can print it out and follow the logic by hand. It's a flowchart. It's a story about how the model thinks.

```
Is age > 30?
├── Yes: Is income > 50K?
│   ├── Yes: APPROVED (confidence: 92%)
│   └── No: Is credit_score > 680?
│       ├── Yes: APPROVED (confidence: 78%)
│       └── No: REJECTED (confidence: 85%)
└── No: REJECTED (confidence: 70%)
```

This is a **path-based explanation**. For any specific prediction, you trace the path from root to leaf and you have a natural language explanation: "The loan was rejected because the applicant is under 30 and has a credit score below 680."

### Building and Visualizing

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
tree.fit(X, y)

# Text representation
print(export_text(tree, feature_names=list(X.columns)))

# Visual representation
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=['malignant', 'benign'],
          filled=True, ax=ax)
plt.savefig('tree.png', dpi=150, bbox_inches='tight')
```

### Extracting Rules

For each leaf node, you can extract the full conjunction of conditions that defines it:

```python
from sklearn.tree import _tree

def tree_to_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, depth, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], depth + 1,
                    conditions + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], depth + 1,
                    conditions + [f"{name} > {threshold:.2f}"])
        else:
            class_idx = tree_.value[node].argmax()
            rules.append({
                'conditions': conditions,
                'prediction': class_names[class_idx],
                'samples': int(tree_.n_node_samples[node])
            })

    recurse(0, 1, [])
    return rules

rules = tree_to_rules(tree, list(X.columns), ['malignant', 'benign'])
for rule in rules[:3]:
    print("IF " + " AND ".join(rule['conditions']))
    print(f"  THEN {rule['prediction']} (n={rule['samples']})\n")
```

### Limitations

- **Depth tradeoff**: Shallow trees are interpretable but may be inaccurate. Deep trees are more accurate but impossible to follow.
- **Instability**: Small changes in data can produce completely different trees. A tree that explains the same data differently each time you retrain is hard to trust.
- **Not great for continuous features**: Trees create axis-aligned splits, which can look odd for smooth relationships.

This is why we use tree ensembles (random forests, gradient boosting) for accuracy, then apply post-hoc interpretation methods — the single tree is the white-box baseline.

---

## 5. Feature Importance: Permutation Importance, Impurity-Based

Feature importance answers the global question: "Which features does this model rely on most?"

### 5.1 Impurity-Based (MDI) Importance

For tree-based models, each split reduces impurity (Gini or entropy). Sum up the impurity reduction contributed by each feature across all trees and you get MDI importance.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# MDI importance — built in
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.head(10).plot(kind='barh', x='feature', y='importance')
plt.title('MDI Feature Importance')
plt.tight_layout()
plt.savefig('mdi_importance.png')
```

**Bias of MDI**: High-cardinality features (continuous features, features with many unique values) tend to get inflated importance because the tree has more split thresholds to choose from. A random ID column would score high. Use with caution.

### 5.2 Permutation Importance

Permutation importance is more reliable. The idea: shuffle one feature at a time and measure how much the model's performance drops. A feature is important if the model falls apart when that feature is scrambled.

```
Permutation Importance(feature j) = 
    model_score(original data) - model_score(data with feature j shuffled)
```

```python
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf.fit(X_train, y_train)

# Always compute on test set to avoid overfitting artifacts
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

print(perm_df.head(10))

# Visualize with error bars
fig, ax = plt.subplots(figsize=(10, 8))
perm_df.head(10).plot(
    kind='barh', x='feature', y='importance_mean', 
    xerr='importance_std', ax=ax
)
plt.title('Permutation Feature Importance (Test Set)')
plt.tight_layout()
```

**Key difference from MDI**: Permutation importance measures the *actual impact* on predictions, evaluated on held-out data. It's slower (requires multiple evaluations) but more trustworthy. It also correctly handles correlated features by measuring the joint effect.

**Caution with correlated features**: If features A and B are highly correlated, shuffling A doesn't hurt much because B still carries the same information. Both features may appear unimportant even if the pair together is critical.

### 5.3 Drop-Column Importance

The most honest version: retrain the model without each feature and measure performance loss. Too expensive for large models, but the ground truth when you can afford it.

```python
from sklearn.metrics import accuracy_score

baseline_score = accuracy_score(y_test, rf.predict(X_test))
drop_col_importance = {}

for col in X.columns:
    X_train_drop = X_train.drop(columns=[col])
    X_test_drop = X_test.drop(columns=[col])
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train_drop, y_train)
    
    score = accuracy_score(y_test, rf_temp.predict(X_test_drop))
    drop_col_importance[col] = baseline_score - score

# Sort by importance
sorted_imp = sorted(drop_col_importance.items(), key=lambda x: x[1], reverse=True)
print("Drop-column importance:", sorted_imp[:5])
```

---

## 6. Partial Dependence Plots (PDPs) and ICE Plots

Feature importance tells you *which* features matter. PDPs tell you *how* they matter — the shape of the relationship.

### 6.1 Partial Dependence Plots

A PDP shows the marginal effect of one (or two) features on the predicted outcome, averaged over the distribution of all other features.

Formally, for feature `x_s`:

```
PD(x_s) = E_{x_c}[f(x_s, x_c)] = (1/n) Σᵢ f(x_s, xᵢ_c)
```

Where `x_c` are the complement features. We loop over a grid of `x_s` values and for each value, we push it through the model with all observed combinations of other features, then average.

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X, y)

# Top 4 important features
top_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    gb, X, top_features, ax=axes.flatten(), kind='average'
)
plt.suptitle('Partial Dependence Plots')
plt.tight_layout()
plt.savefig('pdp.png')
```

**Interpretation**: A flat PDP means the feature has little effect on the prediction (after marginalizing over others). An increasing PDP means higher feature values lead to higher predicted probability. A non-monotone PDP captures interactions.

**The averaging trap**: PDPs can be misleading when there are strong interactions. Averaging can wash out heterogeneous effects — the average effect might be zero if the feature helps some instances and hurts others equally.

### 6.2 ICE (Individual Conditional Expectation) Plots

ICE plots solve the averaging problem. Instead of averaging, we plot one line per instance. Each line shows how *that specific instance's* prediction changes as the feature varies.

```python
# ICE plots — kind='individual' or kind='both' (PDP + ICE)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

PartialDependenceDisplay.from_estimator(
    gb, X, ['mean radius'], ax=axes[0], kind='individual',
    subsample=100, random_state=42, alpha=0.3
)
axes[0].set_title('ICE Plot: mean radius')

PartialDependenceDisplay.from_estimator(
    gb, X, ['mean radius'], ax=axes[1], kind='both',
    subsample=100, random_state=42, alpha=0.3
)
axes[1].set_title('PDP + ICE: mean radius')

plt.tight_layout()
plt.savefig('ice_plots.png')
```

**Centered ICE (c-ICE)**: Subtract each line's value at a reference point (usually the minimum of the feature range) so all lines start at zero. This makes it easier to see the *shape* of individual effects rather than absolute levels.

**When to use ICE over PDP**: When you suspect interactions. If all ICE lines are roughly parallel, the PDP is trustworthy. If lines cross or fan out, there are interactions — the PDP's average hides the story.

### 6.3 2D PDPs

You can also do two-feature PDPs to visualize interactions:

```python
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    gb, X, [('mean radius', 'mean texture')], ax=ax
)
plt.title('2D PDP: mean radius x mean texture')
plt.tight_layout()
plt.savefig('2d_pdp.png')
```

A 2D PDP as a heatmap shows you where in the joint feature space the prediction is high or low.

---

## 7. LIME (Local Interpretable Model-agnostic Explanations)

LIME was one of the first major post-hoc explanation methods, introduced by Ribeiro et al. (2016). The core idea is elegant: even if the global model is complex, it's probably approximately linear in a small region around any specific prediction.

The analogy: the Earth is round, but your neighborhood looks flat. A local flat-earth model is a good approximation locally.

### 7.1 How It Works: Local Linear Approximation

For a specific instance `x`:

1. **Perturb** the instance: create synthetic samples around `x` by randomly turning features on/off (for tabular data) or masking superpixels (for images).
2. **Query the black box**: get predictions for all synthetic samples.
3. **Weight by proximity**: samples closer to `x` get higher weight (Gaussian kernel on distance).
4. **Fit a simple model**: train a sparse linear model on the weighted synthetic samples.
5. **Report the linear model's coefficients** as the explanation.

```
explanation = argmin_{g ∈ G} L(f, g, πₓ) + Ω(g)
```

Where:
- `f` is the black-box model
- `g` is the interpretable surrogate (linear model)
- `πₓ` is the proximity measure around `x`
- `L` is the fidelity loss (how well `g` approximates `f` locally)
- `Ω(g)` is the complexity penalty (favors sparse models)

### 7.2 LIME in Practice

```python
# pip install lime
import lime
import lime.lime_tabular
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X.columns),
    class_names=['malignant', 'benign'],
    mode='classification'
)

# Explain a single prediction
instance_idx = 5
instance = X_test.iloc[instance_idx].values

explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=rf.predict_proba,
    num_features=10
)

# Show explanation
print("Predicted class:", rf.predict([instance])[0])
print("LIME explanation:")
for feat, weight in explanation.as_list():
    print(f"  {feat}: {weight:.4f}")

# Visualize
fig = explanation.as_pyplot_figure()
plt.tight_layout()
plt.savefig('lime_explanation.png')
```

### LIME for Text

```python
import lime.lime_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Toy sentiment example
texts = ["great product love it", "terrible waste of money",
         "okay nothing special", "fantastic highly recommend",
         "broken arrived damaged"]
labels = [1, 0, 0, 1, 0]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
pipeline.fit(texts, labels)

text_explainer = lime.lime_text.LimeTextExplainer(class_names=['negative', 'positive'])
exp = text_explainer.explain_instance(
    "great quality but arrived damaged",
    pipeline.predict_proba,
    num_features=6
)

print("Text LIME explanation:")
for word, weight in exp.as_list():
    print(f"  '{word}': {weight:.4f}")
```

### 7.3 When to Use LIME / Limitations

**Use LIME when:**
- You need a quick, human-readable explanation for a single prediction
- You're working with a truly black-box model (only need predict function)
- You're working with text or image data (LIME handles these natively)
- The audience needs an intuitive explanation, not mathematical precision

**Limitations:**

1. **Instability**: Run LIME twice on the same instance — you might get different explanations. The random perturbation makes it non-deterministic. This is a serious problem if explanations are used for auditing.

2. **Neighborhood definition is tricky**: What counts as "local"? The Gaussian kernel bandwidth `σ` controls this, and results are sensitive to it. There's no principled way to choose it.

3. **Superpixel artifacts (images)**: For images, LIME groups pixels into superpixels before masking. Different superpixel segmentations yield different explanations.

4. **Fidelity vs simplicity tradeoff**: A sparse linear model may not approximate the local decision boundary well. The explanation might look clean but be misleading.

5. **No global view**: LIME gives you one explanation per instance. Aggregating LIME explanations is not well-defined (unlike SHAP, which has a global analog).

---

## 8. SHAP (SHapley Additive exPlanations)

SHAP is arguably the most principled and widely-used explainability method in practice. It's built on game theory and has strong theoretical guarantees that LIME lacks.

### 8.1 Game Theory Foundation: Dividing Credit Among Team Members

The Shapley value comes from cooperative game theory (Shapley, 1953). The setup: a group of players collaborate on a task and earn a reward. How do you fairly distribute the reward among players, accounting for each player's contribution?

The fair answer must satisfy four axioms:
1. **Efficiency**: The sum of all players' contributions equals the total outcome.
2. **Symmetry**: If two players are interchangeable, they get equal credit.
3. **Dummy**: A player who contributes nothing gets nothing.
4. **Linearity**: If you combine two games, payoffs add up.

The unique value satisfying all four is the Shapley value:

```
φᵢ(f) = Σ_{S ⊆ F\{i}} [|S|! (|F|-|S|-1)! / |F|!] × [f(S∪{i}) - f(S)]
```

Where:
- `F` is the set of all features
- `S` is a subset of features not including feature `i`
- `f(S)` is the model's prediction using only features in `S`
- The formula averages over all possible orderings of features

**The team analogy**: Think of features as team members and the prediction as the team's score. The Shapley value for player i is their *average marginal contribution* across all possible orderings in which they could join the team.

If sales = 100K, baseline = 60K:
- Feature "location" joins first: contribution = 15K
- Feature "size" joins first: contribution = 25K
- Together: 40K total above baseline
- Shapley values fairly split this 40K according to each feature's average contribution

### 8.2 SHAP in Practice

```python
# pip install shap
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer (TreeSHAP for tree-based models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values shape: {shap_values.shape}")  # (n_samples, n_features)
print(f"Base value (expected output): {explainer.expected_value:.4f}")

# Verify efficiency property: base_value + sum(shap_values) ≈ model output
instance_idx = 0
predicted = model.predict_proba(X_test.iloc[[instance_idx]])[0, 1]
shap_sum = explainer.expected_value + shap_values[instance_idx].sum()
print(f"Predicted: {predicted:.4f}, SHAP reconstruction: {shap_sum:.4f}")
```

### 8.3 TreeSHAP, DeepSHAP, KernelSHAP

**TreeSHAP** (Lundberg & Lee, 2018):
- Exact (not approximate) Shapley values for tree-based models
- O(TLD²) complexity where T=trees, L=leaves, D=depth — polynomial instead of exponential
- Available for: sklearn trees/forests, XGBoost, LightGBM, CatBoost
- The default choice when you have a tree model

```python
# TreeSHAP with XGBoost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

explainer_xgb = shap.TreeExplainer(xgb_model)
shap_vals = explainer_xgb.shap_values(X_test)
```

**DeepSHAP**:
- Combines SHAP with DeepLIFT for neural networks
- Propagates SHAP values backward through the network
- Approximate — uses a background dataset to estimate feature baselines
- Faster than KernelSHAP for deep models

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Assuming model is trained...
# background = torch.tensor(X_train.values[:100], dtype=torch.float32)
# test_data = torch.tensor(X_test.values[:10], dtype=torch.float32)
# explainer_deep = shap.DeepExplainer(net_model, background)
# shap_vals_deep = explainer_deep.shap_values(test_data)
```

**KernelSHAP**:
- Model-agnostic (only needs predict function like LIME)
- Uses a specially weighted linear regression to estimate Shapley values
- Significantly slower than TreeSHAP (needs many model evaluations)
- Use when you have a non-tree, non-neural-network model

```python
# KernelSHAP — model agnostic
background = shap.sample(X_train, 100)  # Background sample for integration

kernel_explainer = shap.KernelExplainer(
    model.predict_proba, background
)

# This is slow — use a small test set
shap_vals_kernel = kernel_explainer.shap_values(X_test.iloc[:20])
```

### 8.4 SHAP Plots: Summary, Dependence, Force Plots

**Force Plot (Local)**:

Shows a single prediction as a tug-of-war between features pushing the prediction up (red) and down (blue) from the base value.

```python
# Force plot for one instance
shap.initjs()  # For Jupyter

instance_idx = 0
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[instance_idx],
    X_test.iloc[instance_idx],
    feature_names=list(X.columns)
)
shap.save_html('force_plot.html', force_plot)

# Stacked force plot for many instances
force_plot_all = shap.force_plot(
    explainer.expected_value,
    shap_values[:50],
    X_test.iloc[:50]
)
```

**Summary Plot (Global)**:

Shows the distribution of SHAP values for all features across all instances. Each dot is one instance. Color shows feature value (red=high, blue=low). Position on x-axis shows impact.

```python
# Summary plot — most important visualization
shap.summary_plot(shap_values, X_test, plot_type='dot', show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')

# Bar version — simpler, shows mean absolute SHAP value
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
```

Reading the summary plot:
- Features at top are most important globally
- Width of distribution shows variance in impact
- Correlation between color and position tells you directionality: if red (high value) is on right (positive SHAP), the feature has a positive effect

**Dependence Plot**:

Shows SHAP value for one feature vs its actual value, colored by another feature (auto-detected interaction).

```python
# Dependence plot — feature effect + interaction
shap.dependence_plot(
    'mean radius',           # Feature to plot
    shap_values,
    X_test,
    interaction_index='auto',  # auto-detects strongest interacting feature
    show=False
)
plt.title('SHAP Dependence: mean radius')
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
```

**Waterfall Plot** (newer, cleaner force plot):

```python
# Waterfall plot — requires shap >= 0.40
explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0].values,
    feature_names=list(X.columns)
)

shap.plots.waterfall(explanation, show=False)
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
```

### 8.5 Global vs Local Explanations with SHAP

SHAP naturally supports both:

**Local**: SHAP value for a specific instance tells you why *this* prediction is what it is. Feature `j` contributed `φⱼ` to move the prediction from the baseline.

**Global**: Average |SHAP value| across all instances gives you global feature importance. This is better than MDI or permutation importance because:
- It properly handles correlated features
- It accounts for feature interactions
- It's grounded in the efficiency axiom

```python
# Global importance from SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("Global SHAP importance:")
print(global_importance.head(10))
```

**SHAP interaction values**: TreeSHAP also computes pairwise interaction effects:

```python
# Interaction values (expensive)
shap_interaction = explainer.shap_interaction_values(X_test.iloc[:100])
# shape: (n_samples, n_features, n_features)
# Diagonal: main effects. Off-diagonal: pairwise interactions.
```

---

## 9. Attention Visualization: What Attention Weights Do and Don't Tell Us

Attention mechanisms are central to transformer models. It's tempting to treat attention weights as explanations: "the model focused on this word when making the prediction." But this story is more complicated than it seems.

### 9.1 What Attention Weights Are

In a standard self-attention layer:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

The attention weights `softmax(QKᵀ / √dₖ)` form a matrix where entry `(i, j)` represents "how much position i attends to position j." The output is a weighted sum of value vectors.

Visualizing these weights is straightforward:

```python
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

text = "The bank can guarantee deposits will eventually cover future losses."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
attentions = outputs.attentions  # 12 layers

# Visualize layer 0, head 0
layer_idx, head_idx = 0, 0
attn_weights = attentions[layer_idx][0, head_idx].numpy()

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens,
            cmap='Blues', ax=ax)
ax.set_title(f'BERT Attention Weights (Layer {layer_idx}, Head {head_idx})')
plt.tight_layout()
plt.savefig('attention_viz.png', dpi=150)
```

### 9.2 What Attention Does NOT Tell Us

The critical paper: **"Attention is not Explanation"** (Jain & Wallace, 2019) and the counter **"Attention is not not Explanation"** (Wiegreffe & Pinter, 2019).

The core problem: attention weights tell you *where the model looked*, not *why it made the decision*. These are different things.

**Reasons attention is not a reliable explanation:**

1. **Gradient vs attention disagreement**: A token can have high attention weight but zero gradient — meaning the model's output is insensitive to that token's value. The model looked at it but didn't use it.

2. **Multiple attention heads**: BERT has 12 heads per layer, 12 layers = 144 attention matrices. Which one are you supposed to look at? They often disagree.

3. **Permutable explanations**: You can often find a different attention distribution that produces the same output. The attention weights are not unique — multiple configurations lead to identical outputs.

4. **Attention over representations, not words**: By the time the model attends to something, the representations have already been mixed by previous layers. "Attending to token j" doesn't mean "using the semantic meaning of word j."

5. **CLS token attention is a sink**: In many transformer architectures, the CLS token absorbs attention from many tokens for reasons unrelated to classification.

**What you should use instead:**
- Integrated gradients (Section 11) for attribution
- Probing classifiers (Section 14) for understanding what's encoded
- Attention rollout (aggregating attention across layers) for a more complete picture
- Raw attention is still useful for debugging and exploration, just not as a formal explanation

---

## 10. Saliency Maps and Grad-CAM for CNNs

For computer vision, the natural question is: which pixels (or regions) of the image influenced the prediction? Saliency maps and Grad-CAM both answer this for CNNs.

### 10.1 Vanilla Saliency Maps (Gradient-based)

The simplest approach: compute the gradient of the output class score with respect to the input image. Pixels where a small change would strongly affect the output are "salient."

```
Saliency(x) = |∂f(x) / ∂x|
```

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_saliency(model, image_tensor, target_class):
    image_tensor.requires_grad_(True)
    
    output = model(image_tensor.unsqueeze(0))
    score = output[0, target_class]
    
    model.zero_grad()
    score.backward()
    
    # Take max across color channels
    saliency = image_tensor.grad.data.abs().max(dim=0)[0]
    return saliency.numpy()

# img = Image.open('cat.jpg')
# img_tensor = transform(img)
# target = 281  # tabby cat class in ImageNet
# saliency = compute_saliency(model, img_tensor, target)
# plt.imshow(saliency, cmap='hot')
```

**Problem**: Vanilla saliency maps are often noisy and visually difficult to interpret. The gradient is locally defined and sensitive to saturation issues (where the function is flat, gradients are near zero but the region can still be important).

### 10.2 Grad-CAM

Grad-CAM (Selvaraju et al., 2017) is more robust. Instead of looking at input gradients, it looks at the gradients flowing into the *last convolutional layer*. The intuition: the last conv layer has high-level spatial feature maps — the gradients tell you which spatial locations those feature maps focused on for the target class.

**Algorithm:**

1. Forward pass: get the feature maps `A^k` from the target conv layer (shape: `H × W × K`)
2. Backward pass: compute `∂y^c / ∂A^k_{ij}` — gradient of class score w.r.t. each feature map
3. Global average pool the gradients: `α^c_k = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}`
4. Weighted sum of feature maps: `L^c_{Grad-CAM} = ReLU(Σ_k α^c_k A^k)`
5. Upsample to input resolution and overlay

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)
    
    def _save_activations(self, module, input, output):
        self.activations = output
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach().numpy()

# Usage with ResNet
model = models.resnet50(pretrained=True)
model.eval()

target_layer = model.layer4[-1].conv2  # Last conv layer
grad_cam = GradCAM(model, target_layer)

# img_tensor = transform(img).unsqueeze(0)
# cam = grad_cam.generate(img_tensor, target_class=281)
# 
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1); plt.imshow(img); plt.title('Original')
# plt.subplot(1, 3, 2); plt.imshow(cam, cmap='jet'); plt.title('Grad-CAM')
# plt.subplot(1, 3, 3)
# plt.imshow(img)
# plt.imshow(cam, cmap='jet', alpha=0.5); plt.title('Overlay')
```

**Grad-CAM variants:**
- **Grad-CAM++**: Improves localization for multiple instances of the same class
- **Score-CAM**: Uses activation perturbation instead of gradients — avoids gradient noise
- **Eigen-CAM**: Uses PCA on activation maps, no gradients needed

**Limitation**: Grad-CAM only highlights spatial regions, not specific features within those regions. Also, the choice of which convolutional layer to use matters — shallower layers show texture patterns, deeper layers show semantic features.

---

## 11. Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) provides attribution scores that satisfy three important axioms that vanilla gradients do not:

1. **Sensitivity**: If two inputs differ in one feature and produce different outputs, that feature must have non-zero attribution.
2. **Implementation invariance**: Two functionally equivalent networks must give the same attributions.
3. **Completeness**: Attribution scores sum to the difference between model output and the baseline output.

### The Problem with Vanilla Gradients

Vanilla gradients fail sensitivity because of the "saturation problem." If a neuron is in a saturated region (flat part of sigmoid or relu), its gradient is near zero — but the feature might still be important. The gradient measures the *current rate of change*, not the *total contribution*.

### How Integrated Gradients Works

Choose a **baseline** `x'` (typically zeros, black image, or expected value of features). Integrate the gradient along the straight-line path from baseline to input:

```
IntegratedGrads_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i dα
```

In practice, approximate with a Riemann sum over `m` steps:

```
IntegratedGrads_i(x) ≈ (x_i - x'_i) × (1/m) × Σ_{k=1}^{m} ∂F(x' + (k/m)(x-x')) / ∂x_i
```

**Completeness check**: `Σᵢ IntegratedGrads_i(x) = F(x) - F(x')` — attributions sum exactly to the output difference from baseline.

```python
import torch
import numpy as np

def integrated_gradients(model, input_tensor, baseline, target_class, steps=50):
    """
    Compute integrated gradients for a specific prediction.
    
    Args:
        model: PyTorch model
        input_tensor: Input of shape (1, ...)
        baseline: Same shape as input, reference point (e.g., zeros)
        target_class: Class index to compute attribution for
        steps: Number of interpolation steps (more = more accurate)
    
    Returns:
        Integrated gradients of same shape as input
    """
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps + 1)  # [0, 0.02, 0.04, ..., 1.0]
    
    # Interpolated inputs: (steps+1, *input_shape)
    interpolated = baseline + alphas.view(-1, *([1] * (input_tensor.dim()))) * (input_tensor - baseline)
    interpolated.requires_grad_(True)
    
    # Forward pass for all interpolations
    outputs = model(interpolated)
    scores = outputs[:, target_class].sum()
    
    # Backward pass
    model.zero_grad()
    scores.backward()
    
    # Average gradients along path (trapezoidal rule)
    grads = interpolated.grad
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2
    
    # Scale by input - baseline
    integrated_grads = (input_tensor - baseline).squeeze(0) * avg_grads
    
    return integrated_grads.detach()

def verify_completeness(ig_values, model, input_tensor, baseline, target_class):
    """Check that IG values sum to F(input) - F(baseline)."""
    with torch.no_grad():
        f_input = model(input_tensor)[0, target_class].item()
        f_baseline = model(baseline)[0, target_class].item()
    
    delta = f_input - f_baseline
    ig_sum = ig_values.sum().item()
    
    print(f"F(input) - F(baseline) = {delta:.6f}")
    print(f"Sum of IG values = {ig_sum:.6f}")
    print(f"Completeness error = {abs(delta - ig_sum):.8f}")
```

**Choosing the baseline**: The baseline matters because it defines "absence of information":
- For images: black image (zeros), blurred image, or random noise
- For text: padding token or masked token
- For tabular: mean of training data, or all zeros

```python
# Using Captum (PyTorch's official interpretability library)
# pip install captum
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input_tensor,
    baselines=baseline,
    target=target_class,
    return_convergence_delta=True,
    n_steps=50
)

print(f"Convergence delta: {delta.item():.6f}")  # Should be near 0
```

**Smoothed Integrated Gradients**: Add Gaussian noise to the input multiple times and average — reduces visual noise in attribution maps at the cost of compute.

---

## 12. Concept-Based Explanations (TCAV)

TCAV (Testing with Concept Activation Vectors, Kim et al., 2018) asks a fundamentally different kind of question: "Does the model use this *human-defined concept* when making predictions?"

Instead of asking "which pixels mattered?" it asks "did the model's use of the concept 'stripes' influence classifying this animal as a zebra?"

### How TCAV Works

1. **Define a concept**: Gather examples of the concept (e.g., images with stripes) and non-concept examples.
2. **Learn a CAV**: Train a linear classifier to distinguish concept examples from non-concept examples in a specific model layer's activations. The normal to the decision boundary is the Concept Activation Vector (CAV).
3. **Measure directional derivative**: Compute how much the model's prediction changes as we move activations in the direction of the CAV. This is the *conceptual sensitivity* `S_TC`.
4. **Statistical test (TCAV score)**: Across a test class (e.g., all zebra images), what fraction have positive conceptual sensitivity? If TCAV score >> 0.5, the concept is important.

```
TCAV_score(concept, class, layer) = 
    |{x ∈ class : S_TC(x) > 0}| / |class|
```

```python
# Conceptual implementation (not full TCAV, illustrative)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def compute_cav(model, layer, concept_images, non_concept_images, device='cpu'):
    """
    Compute a Concept Activation Vector.
    
    Returns the weight vector of a linear classifier trained to 
    separate concept from non-concept activations at the specified layer.
    """
    def get_activations(images):
        activations = []
        hooks = []
        
        def hook_fn(module, input, output):
            activations.append(output.flatten(start_dim=1).detach().cpu().numpy())
        
        hook = layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            model(images.to(device))
        
        hook.remove()
        return np.vstack(activations)
    
    concept_acts = get_activations(concept_images)
    non_concept_acts = get_activations(non_concept_images)
    
    X = np.vstack([concept_acts, non_concept_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(non_concept_acts))
    
    # Train linear probe
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # CAV is the normal to the decision boundary
    cav = clf.coef_[0]
    cav = cav / np.linalg.norm(cav)  # Normalize
    
    # CAV quality: how separable are concept and non-concept activations?
    accuracy = accuracy_score(y, clf.predict(X))
    print(f"CAV linear separability accuracy: {accuracy:.4f}")
    
    return cav

def tcav_score(model, layer, cav, test_images, target_class, device='cpu'):
    """
    Compute TCAV score: fraction of test images with positive 
    directional derivative toward the concept.
    """
    positive_count = 0
    
    for img in test_images:
        img = img.unsqueeze(0).to(device)
        img.requires_grad_(True)
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        
        hook = layer.register_forward_hook(hook_fn)
        output = model(img)
        hook.remove()
        
        # Gradient of target class output w.r.t. layer activations
        score = output[0, target_class]
        act = activations[0]
        grad = torch.autograd.grad(score, act)[0]
        
        # Directional derivative in direction of CAV
        grad_flat = grad.flatten().detach().cpu().numpy()
        directional_deriv = np.dot(grad_flat, cav)
        
        if directional_deriv > 0:
            positive_count += 1
    
    return positive_count / len(test_images)
```

### When to Use TCAV

TCAV is particularly powerful when:
- You want to audit whether a model learned a *spurious concept* (e.g., "hospital backgrounds" in medical imaging)
- You want to communicate with domain experts using their vocabulary ("did the model use texture or shape?")
- You're doing model debugging at a conceptual level

**Limitations:**
- You need to curate concept images — labor intensive
- CAV quality depends on the quality and quantity of concept examples
- The choice of layer matters significantly
- Statistical significance requires multiple runs with random non-concept samples

---

## 13. Model Cards and Fact Sheets

Technical explanations (SHAP values, Grad-CAM) serve data scientists. But stakeholders, policymakers, and the public need a different kind of transparency: documentation.

### 13.1 Model Cards (Mitchell et al., 2019)

A Model Card is a short document accompanying a trained model, standardizing what information is reported. Google introduced them; they're now widely adopted.

**Standard sections:**

```
# Model Card: Credit Risk Classifier v2.1

## Model Details
- Developed by: Risk Analytics Team, Acme Bank
- Model date: 2024-Q1
- Model type: Gradient Boosted Trees (XGBoost 1.7)
- License: Internal use only
- Contact: risk-ml@acme.com

## Intended Use
### Primary intended uses
- Evaluate creditworthiness for personal loan applications ($5K-$50K)
- Internal loan officers as a decision-support tool (not autonomous)

### Out-of-scope uses
- Business loans
- Mortgage applications
- Decisions without human review

## Factors
### Relevant factors
- Credit history length (strong positive)
- Debt-to-income ratio (strong negative)
- Employment duration (moderate positive)

### Evaluation factors
- Evaluated separately for: age groups (18-25, 26-45, 46+), gender, race

## Metrics
- Primary: AUC-ROC (0.84 on holdout)
- Secondary: KS statistic, Gini coefficient
- Calibration: Brier score (0.11)

## Evaluation Data
- Holdout set: 50,000 applications, Q4 2023
- Time-based split (NOT random) to avoid leakage

## Training Data
- 500,000 loan applications, 2019-2023

## Quantitative Analyses
### Unitary results
| Metric | Overall | Age 18-25 | Age 26-45 | Age 46+ |
|--------|---------|-----------|-----------|---------|
| AUC    | 0.84    | 0.79      | 0.86      | 0.85    |
| FPR    | 0.12    | 0.18      | 0.10      | 0.11    |

### Intersectional results
- No significant interaction effects between age and gender

## Ethical Considerations
- Model uses proxy variables that may encode protected attributes
- Regular fairness audits scheduled quarterly
- Disparity thresholds: max 1.2x adverse impact ratio by any protected group

## Caveats and Recommendations
- Performance degrades for applicants with < 2 years of credit history
- Should not be sole basis for rejection; human review required
- Recalibrate if macroeconomic conditions shift significantly
```

### 13.2 IBM AI Fact Sheets

IBM's Fact Sheets (Arnold et al., 2019) extend Model Cards with more focus on the AI lifecycle, lineage, and governance:

- **AI Service Sheet**: Describes the AI service offering
- **AI Model Sheet**: Technical model details
- **AI Training Sheet**: Training data and methodology
- **AI Risk Sheet**: Known risks and mitigations

### 13.3 Datasheets for Datasets

Complementary to model cards: Gebru et al.'s "Datasheets for Datasets" proposes that every dataset include documentation covering:
- Motivation for collection
- Composition
- Collection process
- Preprocessing/labeling
- Uses (and prohibited uses)
- Distribution
- Maintenance

### Why This Matters for Interviews

Model cards demonstrate *institutional* commitment to interpretability — not just technical methods. An interviewer asking about XAI often wants to know:
1. Can you build the technical tools (SHAP, Grad-CAM)?
2. Do you understand the broader accountability context?

Model cards answer (2). Being able to discuss them shows you think beyond accuracy.

---

## 14. Interpretability in LLMs: Logit Lens, Probing Classifiers, Mechanistic Interpretability

Interpreting large language models is a field of active research. The scale and architecture create unique challenges, but also unique opportunities.

### 14.1 Probing Classifiers

The question: does a specific layer of an LLM encode a specific linguistic property?

Method: freeze the model, extract representations at a given layer for a dataset annotated with a property (e.g., part-of-speech tags, syntactic relations, semantic roles), train a simple linear classifier on those representations.

If a linear classifier achieves high accuracy, that layer's representations encode that property.

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_bert_representations(texts, tokenizer, model, layer_idx=6):
    """Extract token-level representations from a specific BERT layer."""
    model.eval()
    all_reps = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # hidden_states: tuple of (layer, batch, seq_len, hidden_dim)
        layer_reps = outputs.hidden_states[layer_idx]  # (1, seq_len, 768)
        # Mean pool over sequence
        rep = layer_reps[0].mean(dim=0).numpy()
        all_reps.append(rep)
    
    return np.array(all_reps)

# Example: probe for sentiment using sentence representations
texts = ["I love this movie", "This film is terrible", 
         "Absolutely brilliant", "Complete waste of time"]
labels = [1, 0, 1, 0]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Probe each layer
for layer_idx in range(0, 13):  # BERT has 12 layers + embedding layer
    reps = extract_bert_representations(texts, tokenizer, model, layer_idx)
    X_train, X_test, y_train, y_test = train_test_split(
        reps, labels, test_size=0.5, random_state=42
    )
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    acc = accuracy_score(y_test, probe.predict(X_test))
    print(f"Layer {layer_idx:2d}: probe accuracy = {acc:.3f}")
```

**Probing findings in BERT (empirically observed):**
- Lower layers: surface features (word identity, capitalization)
- Middle layers: syntactic structure (POS tags, dependency relations peak ~layer 6)
- Higher layers: semantic and discourse features

**Caveat — probing only shows encoding, not use**: A representation that encodes sentiment doesn't mean the model *uses* that information for its task. This is called the **amnesic probing** critique. Supplement probing with causal interventions (see mechanistic interpretability below).

### 14.2 Logit Lens

The logit lens (nostalgebraist, 2020) is a technique to understand what the residual stream of a transformer is "thinking" at each layer.

The insight: in transformers, the output at each layer is added to a residual stream. The final layer's output is converted to logits by the unembedding matrix `W_U`. We can apply this same unembedding matrix to any intermediate layer's residual stream to get "as if it were the final output" predictions.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

def logit_lens(model, tokenizer, text):
    """
    Show how the model's prediction evolves through layers
    using the logit lens.
    """
    model.eval()
    
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, hidden)
    
    # GPT-2's LM head = layer norm + linear (= W_U)
    lm_head = model.lm_head
    ln_f = model.transformer.ln_f
    
    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    last_token_idx = -1  # Predict next token from last position
    
    print(f"Input: {text}")
    print(f"Predicting next token after position {len(input_tokens)-1}: '{input_tokens[-1]}'")
    print("-" * 50)
    
    for layer_idx, hidden in enumerate(hidden_states):
        # Apply final layer norm + unembed
        normed = ln_f(hidden[0, last_token_idx, :])
        logits = lm_head(normed)
        probs = torch.softmax(logits, dim=-1)
        
        top5_probs, top5_ids = probs.topk(5)
        top5_tokens = [tokenizer.decode([t]) for t in top5_ids]
        
        layer_name = f"Layer {layer_idx:2d}" if layer_idx > 0 else "Embed "
        top_token = top5_tokens[0]
        top_prob = top5_probs[0].item()
        
        print(f"{layer_name}: top prediction = '{top_token}' ({top_prob:.3f})")

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# logit_lens(model, tokenizer, "The capital of France is")
```

The logit lens reveals how predictions form over layers — typically the model is uncertain in early layers and converges on the final answer progressively.

### 14.3 Mechanistic Interpretability

Mechanistic interpretability (Olah et al., Elhage et al.) aims to reverse-engineer the *algorithms* learned by neural networks. The goal is not "which inputs mattered" but "what computation is this circuit performing?"

Key concepts:

**Superposition**: Networks represent more features than they have dimensions by encoding multiple features in overlapping, nearly-orthogonal directions. This makes interpretation hard — individual neurons don't cleanly correspond to concepts.

**Circuits**: A circuit is a subgraph of the network (specific attention heads, MLPs, residual stream components) that implements a specific behavior. Notable circuits discovered:
- **Induction heads** (Olsson et al., 2022): Attention heads that implement in-context learning by copying previous patterns
- **Indirect object identification** circuit in GPT-2 (Wang et al., 2022): 26 attention heads that implement the task of identifying "Mary gave John the ball; John returned it to ___"
- **Modular arithmetic** circuits: Circuits in small transformers that implement modular addition

**Activation patching (causal tracing)**:

```python
def activation_patching(model, clean_input, corrupted_input, target_pos, layer):
    """
    Patch in activations from clean run into corrupted run
    at a specific layer and position.
    
    If patching in clean activations at (layer, pos) restores the 
    clean output, that (layer, pos) is causally important.
    """
    # Store clean activations
    clean_activations = {}
    
    def save_hook(name):
        def hook(module, input, output):
            clean_activations[name] = output.detach()
        return hook
    
    # Run clean forward pass
    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.register_forward_hook(save_hook(f'block_{i}'))
        hooks.append(h)
    
    with torch.no_grad():
        clean_output = model(**clean_input)
    
    for h in hooks:
        h.remove()
    
    # Run corrupted forward pass with patching at (layer, target_pos)
    def patch_hook(module, input, output):
        patched = output.clone()
        patched[:, target_pos, :] = clean_activations[f'block_{layer}'][:, target_pos, :]
        return patched
    
    patch_h = model.transformer.h[layer].register_forward_hook(patch_hook)
    
    with torch.no_grad():
        patched_output = model(**corrupted_input)
    
    patch_h.remove()
    
    return patched_output
```

**Sparse autoencoders (SAEs)**: A newer approach (Templeton et al., 2024) that trains sparse autoencoders on residual stream activations to decompose superposition into interpretable features. SAEs learn a sparse dictionary of monosemantic features, making the network more interpretable without changing its behavior.

---

## 15. Regulatory Context: GDPR, Financial Models

Interpretability is not just academically interesting. It's legally required in many contexts.

### 15.1 GDPR: Right to Explanation

Article 22 of GDPR (EU, 2018) grants data subjects the right "not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her."

Recital 71 specifies the right to "obtain an explanation of the decision reached after such assessment and to challenge the decision."

**What this means in practice:**
- If your model makes decisions about loan approvals, hiring, insurance pricing, parole, or anything "significantly affecting" EU individuals, you must be able to explain those decisions.
- The explanation must be "meaningful" — not just technical jargon.
- You need to maintain the infrastructure to generate these explanations on demand.

**Implementation implications:**

```python
class GDPRCompliantClassifier:
    """Wrapper that stores explanations for audit/GDPR compliance."""
    
    def __init__(self, model, explainer, threshold=0.5):
        self.model = model
        self.explainer = explainer
        self.threshold = threshold
        self.explanation_log = {}  # In practice: use a database
    
    def predict_with_explanation(self, X, decision_id, user_id):
        """Make prediction and store explanation for GDPR compliance."""
        prediction_proba = self.model.predict_proba(X)[0]
        prediction = int(prediction_proba[1] >= self.threshold)
        
        # Generate SHAP explanation
        shap_values = self.explainer.shap_values(X)
        
        # Store explanation
        self.explanation_log[decision_id] = {
            'user_id': user_id,
            'timestamp': pd.Timestamp.now(),
            'prediction': prediction,
            'probability': float(prediction_proba[1]),
            'shap_values': shap_values.tolist(),
            'feature_names': list(X.columns),
            'feature_values': X.iloc[0].to_dict()
        }
        
        return prediction
    
    def get_explanation(self, decision_id, format='human_readable'):
        """Retrieve explanation for a specific decision."""
        log = self.explanation_log.get(decision_id)
        if log is None:
            raise ValueError(f"No explanation found for decision {decision_id}")
        
        if format == 'human_readable':
            shap_df = pd.DataFrame({
                'feature': log['feature_names'],
                'value': list(log['feature_values'].values()),
                'shap_value': log['shap_values']
            }).sort_values('shap_value', key=abs, ascending=False)
            
            top_reasons = shap_df.head(3)
            outcome = "approved" if log['prediction'] == 1 else "rejected"
            
            explanation = f"Decision: {outcome.upper()}\n"
            explanation += "Primary reasons:\n"
            
            for _, row in top_reasons.iterrows():
                direction = "increased" if row['shap_value'] > 0 else "decreased"
                explanation += f"  - {row['feature']} = {row['value']:.2f} {direction} likelihood\n"
            
            return explanation
        
        return log
```

### 15.2 Financial Services: SR 11-7

The Federal Reserve's Supervisory Guidance SR 11-7 (2011) requires that model risk be managed, which includes:
- Documentation of model development
- Validation by independent parties
- Ongoing monitoring of model performance
- Explanation of model behavior and limitations

For US banking, this essentially mandates interpretability for credit models.

**Adverse Action Notices**: ECOA and FCRA require that when a consumer is denied credit, they receive an adverse action notice listing the principal reasons. For ML models, this typically requires generating a ranked list of the top factors. SHAP values are increasingly used to generate these.

```python
def generate_adverse_action_notice(shap_values, feature_names, feature_values, top_n=4):
    """
    Generate ECOA-compliant adverse action notice.
    For rejections, report top N features with most negative SHAP values.
    """
    df = pd.DataFrame({
        'feature': feature_names,
        'value': feature_values,
        'shap_value': shap_values
    })
    
    # For rejection, the negative SHAP features are the "reasons"
    adverse_factors = df[df['shap_value'] < 0].nsmallest(top_n, 'shap_value')
    
    reasons = []
    for _, row in adverse_factors.iterrows():
        reasons.append({
            'reason_code': row['feature'].upper().replace(' ', '_'),
            'description': f"{row['feature']}: {row['value']}"
        })
    
    return {
        'adverse_action_reasons': reasons,
        'notice_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'model_version': 'CreditRisk_v2.1'
    }
```

### 15.3 High-Stakes Domains

| Domain | Regulation | Explanation requirement |
|---|---|---|
| EU consumer credit | GDPR Art. 22 + Consumer Credit Directive | Right to explanation for automated decisions |
| US credit | ECOA / FCRA | Adverse action notice with top factors |
| US banking model risk | SR 11-7 | Model documentation, validation, monitoring |
| EU healthcare AI | EU AI Act (Art. 13) | Transparency for high-risk AI systems |
| Criminal justice | Varies by state/country | Variable, increasingly challenged |
| Insurance pricing | State regulations (US varies) | Varies; some states require non-discriminatory factors |

---

## 16. Common Interview Questions with Answers

### Q1: What's the difference between interpretability and explainability?

**Interpretability** means the model is inherently understandable — you can read its structure directly (e.g., linear regression coefficients, decision tree rules). **Explainability** means using post-hoc techniques to explain a model that isn't transparent (e.g., SHAP values explaining a random forest). In practice, the terms are often used interchangeably.

---

### Q2: LIME vs SHAP — when would you choose each?

**SHAP is almost always preferable** for tree-based models because TreeSHAP is exact (not approximate), fast, and has theoretical guarantees (efficiency, symmetry, dummy, linearity axioms). SHAP also gives you global importance naturally through mean absolute SHAP.

**LIME is useful when:**
- You have a truly arbitrary black box (not tree or neural net) and KernelSHAP is too slow
- You need a very fast, rough local explanation
- You're working with text and want word-level explanations quickly
- Your audience responds better to "this word pushed the prediction positive" than Shapley values

**Main critique of LIME**: instability. Running it twice on the same instance can give different results. SHAP is deterministic for tree models.

---

### Q3: Can SHAP values be negative, and what does that mean?

Yes. A negative SHAP value for feature `j` means that feature's value pulled the prediction *below* the baseline (expected model output). For example: a credit score of 580 might have a SHAP value of -0.15, meaning it reduced the probability of approval by 0.15 relative to the average credit score in the training data.

The baseline in SHAP is `E[f(X)]` — the expected model output over the training distribution.

---

### Q4: What's wrong with using attention weights as explanations?

Three main problems:
1. **Gradient-attention disagreement**: High attention weight doesn't mean the model's output is sensitive to that token. The gradient with respect to that token's value might be zero.
2. **Permutability**: You can often find multiple attention distributions that produce the same output. The weights aren't uniquely tied to the prediction.
3. **Representation mixing**: By the time a token is attended to, its representation has already been transformed by previous layers. "Attending to 'bank'" doesn't mean "using the concept of a financial institution."

Better alternatives: Integrated Gradients, LIME on tokens, or attention rollout for a more faithful attribution.

---

### Q5: How would you explain a specific loan rejection to a customer?

1. **Generate SHAP values** for that applicant's data point using the model.
2. **Identify the top 3-4 features** with the most negative SHAP values (the factors that most strongly pushed toward rejection).
3. **Translate to human-readable reasons**: "Your debt-to-income ratio of 0.52 is above our threshold and was the primary factor in this decision" (with SHAP = -0.23).
4. **Include actionable guidance** where possible: "Reducing outstanding debt by $X would likely change this outcome."
5. **Log the explanation** for audit/compliance.

In the US, this directly maps to ECOA's adverse action notice requirement.

---

### Q6: How do you validate that an explanation is correct?

This is a deep question. A few approaches:

1. **Completeness check (SHAP)**: Verify that sum of SHAP values + base value ≈ model output. This is a mathematical sanity check.
2. **Sanity checks**: For simulated data where you know ground truth, check that important features are flagged as important.
3. **Input perturbation**: If a feature has a positive SHAP value, increasing it should increase the prediction (all else equal). You can test this empirically.
4. **Model fidelity**: How well does the local surrogate (for LIME) approximate the real model in the region of interest? Measure the surrogate's R² on the synthetic samples.
5. **Human evaluation**: Do domain experts find the explanations plausible? Do they identify cases where the explanation contradicts known domain knowledge?
6. **Removal test**: Remove the top-k features by importance, retrain, and measure performance drop. If important features are truly important, performance should drop significantly.

No perfect validation exists — all XAI methods are approximations of complex models.

---

### Q7: What is the "Rashomon effect" in ML interpretability?

The Rashomon effect (named after the Kurosawa film where witnesses give conflicting accounts of the same event) refers to the existence of many different models that achieve similar accuracy on the test set but have very different internal logic.

Breiman (2001) coined the term in ML: for most datasets, there exist many models in the "Rashomon set" (all models within ε of optimal performance) that are structurally different.

**Implications for interpretability:**
- If multiple high-accuracy models exist but explain the same prediction differently, which explanation do you trust?
- The "correct" explanation might not be unique
- Regularization choices and random seeds affect which model is found and thus which explanation is generated

**Practical response**: Report uncertainty in your explanations. SHAP values have variance when computed over multiple model seeds. Show confidence intervals where possible.

---

### Q8: What's the difference between a PDP and a SHAP dependence plot?

Both show how a feature relates to model output, but they answer different questions:

**PDP**: Shows `E_{x_c}[f(x_s, x_c)]` — the *marginal* effect of feature `x_s`, averaging over all other features. It tells you the average relationship.

**SHAP dependence plot**: Shows SHAP values for feature `j` plotted against its actual values. Each dot is one instance. The SHAP value for an instance is the feature's *contribution conditional on all other feature values for that instance* (not averaged). This makes it more nuanced — you can see heterogeneous effects.

When the two agree: simple, non-interactive relationships.
When they disagree: strong interactions exist. The PDP's average is masking the real story.

---

### Q9: How does TreeSHAP differ from KernelSHAP?

| | TreeSHAP | KernelSHAP |
|---|---|---|
| **Works on** | Tree-based models only | Any model |
| **Speed** | Fast: O(TLD²) | Slow: O(n_background × n_features) per instance |
| **Exactness** | Exact Shapley values | Approximate (weighted linear regression estimate) |
| **Conditioning** | Conditions on tree paths | Conditions on marginal distribution of background |
| **Interaction effects** | Computes exact pairwise interactions | Not available |

The conditioning difference is subtle but important: TreeSHAP conditions on the features in the tree path (interventional SHAP), while KernelSHAP marginalizes over the training distribution (marginal/observational SHAP). These give different values for correlated features.

---

### Q10: What is the "explanation stability" problem, and how do you address it?

If an explanation is different every time you generate it (like LIME), it's not reliable. You can't audit it, defend it in court, or trust it for debugging.

**Why it happens:**
- Random perturbation in LIME
- Random seed in model training
- Stochastic gradient descent

**How to address:**
1. **Use deterministic methods**: TreeSHAP, Integrated Gradients with fixed seeds
2. **Report confidence intervals**: Run LIME 100 times, report mean ± std for each feature
3. **Use a fixed random seed**: Document it. Not a solution for production, but good for validation
4. **Switch to SHAP**: For tree models, TreeSHAP is deterministic
5. **Evaluate stability explicitly**: Measure the fraction of runs where the top-k features are consistent

---

### Q11: What is a counterfactual explanation and when is it useful?

A counterfactual explanation answers: "What would have to change about this input for the prediction to be different?"

Example: "Your loan was rejected. If your credit score were above 680 OR your debt-to-income ratio were below 0.40, the loan would have been approved."

Counterfactuals are:
- **Actionable**: They tell the user what to change, not just why the current decision was made
- **Minimal**: Good counterfactual methods find the smallest change (fewest features changed, smallest magnitude)
- **Realistic**: Good methods ensure counterfactuals are plausible given the feature distribution

```python
# Using the DiCE library for diverse counterfactuals
# pip install dice-ml
import dice_ml
from dice_ml import Dice

# d = dice_ml.Data(dataframe=df, continuous_features=['age', 'income'],
#                  outcome_name='loan_approved')
# m = dice_ml.Model(model=trained_model, backend='sklearn')
# exp = Dice(d, m, method='random')
# 
# dice_exp = exp.generate_counterfactuals(
#     query_instance=X_test.iloc[[0]],
#     total_CFs=3,           # Number of counterfactuals
#     desired_class='opposite'  # Flip the prediction
# )
# dice_exp.visualize_as_dataframe()
```

---

### Q12: Explain the concept of "right to explanation" under GDPR. What does a model engineer need to do?

GDPR Article 22 restricts fully automated decisions with legal or significant effects and gives data subjects the right to:
1. Human review of the decision
2. Express their point of view
3. Obtain "meaningful information about the logic involved"

Practically, a model engineer should:
1. **Log all predictions** with enough context to reproduce explanations
2. **Generate SHAP values** (or similar) at prediction time and store them
3. **Build an explanation API** that returns human-readable factors for any decision ID
4. **Document the model** with a Model Card specifying intended use and limitations
5. **Implement human review workflows** for high-stakes decisions
6. **Audit fairness** across protected groups regularly and document findings
7. **Set an explanation retention period** — you need to provide explanations for decisions you've already made

The explanation doesn't need to be a full ML tutorial. "Your application was primarily influenced by: high debt-to-income ratio (most important), limited credit history, recent credit inquiries" is sufficient and legally defensible.

---

## Quick Reference Cheat Sheet

```
CHOOSING AN EXPLANATION METHOD:

Tree model?
  ├── Local explanation → TreeSHAP (force/waterfall plot)
  ├── Global explanation → SHAP summary plot
  └── Feature relationship → SHAP dependence plot or PDP

Neural network?
  ├── Image → Grad-CAM or Integrated Gradients
  ├── Text → Integrated Gradients or LIME
  ├── Tabular → DeepSHAP or KernelSHAP
  └── Transformer → Probing + IG (not attention)

Any model (black box)?
  ├── Local, fast → LIME
  ├── Local, principled → KernelSHAP
  └── Global → Permutation importance + PDP/ICE

LLM / large transformer?
  ├── What does layer encode? → Probing classifiers
  ├── How prediction forms? → Logit lens
  └── What circuit implements behavior? → Activation patching

Concept audit?
  └── Does model use this concept? → TCAV

Regulatory / documentation?
  └── Model Card + SHAP-based adverse action factors
```

---

## Further Reading

- **Christoph Molnar** — *Interpretable Machine Learning* (free online, comprehensive reference)
- **Lundberg & Lee (2017)** — *A Unified Approach to Interpreting Model Predictions* (original SHAP paper)
- **Ribeiro et al. (2016)** — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME paper)
- **Selvaraju et al. (2017)** — *Grad-CAM: Visual Explanations from Deep Networks* 
- **Sundararajan et al. (2017)** — *Axiomatic Attribution for Deep Networks* (Integrated Gradients)
- **Kim et al. (2018)** — *Interpretability Beyond Classification Accuracy: TCAV*
- **Mitchell et al. (2019)** — *Model Cards for Model Reporting*
- **Elhage et al. (2021)** — *A Mathematical Framework for Transformer Circuits* (mechanistic interpretability)
- **Olsson et al. (2022)** — *In-context Learning and Induction Heads*
- **Templeton et al. (2024)** — *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*
