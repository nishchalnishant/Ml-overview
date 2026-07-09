---
module: Classical Ml
topic: Model Interpretation
subtopic: ""
status: unread
tags: [classicalml, ml, model-interpretation]
---
# Model Interpretation: SHAP, LIME, and Feature Importance

Explaining ML model predictions is critical for debugging, regulatory compliance, and building trust. Essential knowledge for interviews at companies with high-stakes ML (credit, healthcare, fraud).

---

## 1. The Interpretation Spectrum

| Method | Model type | Scope | Speed | Fidelity |
|---|---|---|---|---|
| Model coefficients | Linear models | Global | Fast | Exact |
| Tree feature importance (Gini) | Tree-based | Global | Fast | Approximate |
| Permutation importance | Any | Global | Moderate | Good |
| Partial Dependence Plots (PDP) | Any | Global | Moderate | Marginal effect |
| ALE plots | Any | Global | Moderate | Better than PDP |
| LIME | Any | Local | Moderate | Approximate |
| SHAP | Any | Local + Global | Moderate–slow | Theoretically grounded |
| Grad-CAM | Neural nets | Local | Fast | Gradient-based |

**When to use which:**
- Explaining a single prediction to a customer → SHAP waterfall or LIME
- Debugging model generally → Permutation importance + PDP
- Regulatory audit (GDPR "right to explanation") → SHAP
- Real-time feature importance in serving → pre-computed SHAP on representative samples

---

## 2. SHAP (SHapley Additive exPlanations)

### Mathematical Foundation

SHAP is grounded in cooperative game theory. The Shapley value of feature i is the average marginal contribution of feature i across all possible feature subsets:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

where:
- $F$ = full feature set
- $S$ = subset of features excluding $i$
- $v(S)$ = model prediction using only features in $S$
- The fraction is the probability that $S$ appears in a random feature ordering

**Key property — SHAP satisfies 4 axioms:**
1. **Efficiency:** $\sum_i \phi_i = f(x) - E[f(x)]$ (SHAP values sum to prediction - baseline)
2. **Symmetry:** features contributing equally get equal SHAP values
3. **Dummy:** features that never change the prediction get SHAP = 0
4. **Linearity:** SHAP values add linearly for model ensembles

### TreeSHAP (Exact, Polynomial Time)

Naive Shapley computation is O(2^d) — intractable. TreeSHAP exploits tree structure for O(T × L × D) complexity (T=trees, L=leaves, D=depth).

```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBClassifier(n_estimators=100).fit(X_train, y_train)

# TreeSHAP explainer (exact, fast for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# shap_values shape: [n_samples, n_features]

# Single prediction explanation
prediction_idx = 0
print(f"Prediction: {model.predict_proba(X_test)[prediction_idx, 1]:.3f}")
print(f"Baseline: {explainer.expected_value:.3f}")
print(f"SHAP sum: {shap_values[prediction_idx].sum() + explainer.expected_value:.3f}")  # == prediction

# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[prediction_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[prediction_idx],
        feature_names=X_test.columns.tolist()
    )
)

# Global importance: mean absolute SHAP
global_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
```

### KernelSHAP (Model-Agnostic, Approximate)

For models without a tree-specific explainer (neural nets, SVMs), KernelSHAP uses a weighted linear regression over random feature coalitions.

**How it works:**
1. Sample random subsets S of features
2. For each S, compute model output with features in S at actual values and others replaced by baseline (marginal expectation)
3. Fit weighted linear regression: weights emphasize subsets near ∅ and F (full set)

$$\text{SHAP kernel weight: } \pi(S) = \frac{|F|-1}{\binom{|F|}{|S|} \cdot |S| \cdot (|F|-|S|)}$$

```python
# KernelSHAP for any model (neural network example)
import shap

# Background dataset for baseline computation (100-200 samples)
background = shap.sample(X_train, 100)

# For neural network or any sklearn-compatible model
explainer = shap.KernelExplainer(model.predict_proba, background)

# Slower than TreeSHAP — use smaller n_samples for speed
shap_values = explainer.shap_values(X_test[:50], nsamples=200)
```

**KernelSHAP vs TreeSHAP:**
- TreeSHAP: exact, fast (milliseconds per sample), tree models only
- KernelSHAP: approximate, slow (seconds per sample), any model

---

## 3. LIME (Local Interpretable Model-agnostic Explanations)

LIME explains a single prediction by fitting a simple interpretable model (linear regression) locally around the input.

**Algorithm:**
1. Sample N perturbed versions of the input (random feature masking/noise)
2. Get model predictions for all N samples
3. Weight samples by distance to original input: $w_i = \exp(-d(x, x_i)^2 / \sigma^2)$
4. Fit weighted linear regression on the N samples
5. Return regression coefficients as "feature importances" for this prediction

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["not_fraud", "fraud"],
    mode="classification"
)

# Explain single prediction
instance = X_test.iloc[0].values
explanation = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=10,        # top 10 features
    num_samples=1000        # perturbed samples
)

# Get feature coefficients
for feature, coeff in explanation.as_list():
    print(f"{feature}: {coeff:+.4f}")
```

**LIME limitations:**
- Not additive — LIME explanations for subsets don't combine predictably
- Sensitive to hyperparameters (kernel width σ, n_samples)
- Local linearity assumption may not hold for highly non-linear boundaries
- Different runs can give different explanations (sampling variance)

**SHAP vs LIME:**
| | SHAP | LIME |
|---|---|---|
| Theoretical foundation | Game theory (Shapley) | Local linear approximation |
| Consistency | Guaranteed (same model = same SHAP) | Stochastic |
| Global aggregation | Valid (mean|SHAP|) | Not valid (local coefficients) |
| Speed (tree models) | Fast (TreeSHAP) | Moderate |
| Negative values | Meaningful | Meaningful |

---

## 4. Partial Dependence Plots (PDP) and ALE

### PDP

Shows the marginal effect of one feature on predictions, averaging over all other features.

$$\text{PDP}(x_j) = E_{X_C}[f(x_j, X_C)] = \frac{1}{n} \sum_{i=1}^n f(x_j, x_C^{(i)})$$

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=["transaction_amount", "user_txn_count_1h", 
              ("transaction_amount", "user_txn_count_1h")],  # 2D PDP
    kind="average",  # or "individual" for ICE plots
    ax=ax
)
```

**PDP assumption: feature independence.** When feature $x_j$ is correlated with other features, averaging over $X_C$ creates unrealistic combinations (extrapolation).

### ALE (Accumulated Local Effects)

ALE fixes PDP's correlation problem by computing *differences* within narrow bins, then accumulating.

$$\text{ALE}(x_j) = \int_{z_0}^{x_j} E\left[\frac{\partial f}{\partial x_j} \middle| x_j = z\right] dz$$

In practice: divide $x_j$ range into bins, compute average prediction difference within each bin, accumulate.

```python
# PyALE library
from PyALE import ale

ale_eff = ale(
    X=X_train,
    model=model,
    feature=["transaction_amount"],
    feature_type="continuous",
    grid_size=20
)
```

**ALE vs PDP:** ALE is preferred when features are correlated — it isolates the true marginal effect without creating unrealistic feature combinations.

---

## 5. Permutation Importance

Measures how much model performance drops when a feature is randomly shuffled.

$$\text{PI}_j = \text{metric}(y, f(X)) - \text{metric}(y, f(X_{perm_j}))$$

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,           # repeat permutation 10x for stability
    scoring="roc_auc",
    random_state=42
)

importance_df = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

# Features with negative importance: shuffling IMPROVES performance
# → feature is adding noise or leaking information
```

**When permutation importance misleads:**
- Correlated features: when two features carry the same info, shuffling one doesn't hurt because the other still provides it — both appear low importance. Check by computing importance on the residuals after removing the correlated feature.
- It measures *predictive importance*, not causal importance.

---

## 6. Attention Weights as Explanation (and Why It's Wrong)

For transformer models, attention weights are often visualized as explanations.

**Why attention ≠ explanation:**
- Attention weights are intermediate computations, not causal attributions
- The same attention pattern can be produced with different weight matrices
- Gradient-based methods (Integrated Gradients, SHAP for transformers) are more principled

**Integrated Gradients (correct for neural nets):**
$$\text{IG}_i(x) = (x_i - x_i') \cdot \int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$

Interpolates between baseline $x'$ (e.g., zero embedding) and input $x$, integrating the gradient.

```python
# Using Captum for Integrated Gradients on a transformer
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(
    input_ids,
    target=predicted_class,
    n_steps=300
)
# attributions[token_idx] = importance of token_idx for this prediction
```

---

## 7. Debugging with Interpretability

**Common patterns to investigate:**

```python
def find_spurious_correlations(model, X_val, y_val, shap_values):
    """Find features with high SHAP but suspected as proxies."""
    high_importance = global_importance.head(20)["feature"].tolist()
    
    for feat in high_importance:
        # Check if feature correlates with label in training but not logically
        corr_with_label = X_val[feat].corr(y_val)
        shap_directional = np.sign(shap_values[:, X_val.columns.tolist().index(feat)].mean())
        
        print(f"{feat}: correlation={corr_with_label:.3f}, SHAP direction={shap_directional:+.0f}")
        
        # Flag potential proxies
        if feat in ["customer_age", "zip_code", "name_initial"]:
            print(f"  ⚠ Potential protected attribute proxy!")
```

---

## Canonical Interview Q&As

**Q: Explain the difference between SHAP and permutation importance.**  
A: Both measure feature importance, but at different levels. Permutation importance is global — it measures how much accuracy drops on validation data when a feature is randomly shuffled. It captures a feature's contribution to the whole model. SHAP is local-first — it computes each feature's additive contribution to a specific prediction, and global SHAP importance is derived by averaging absolute values across samples. The critical difference: SHAP respects feature interactions (assigns credit correctly when features interact), while permutation importance distributes credit arbitrarily among correlated features. For correlated features, permutation importance underestimates all of them. SHAP is also more principled — it satisfies game-theoretic axioms guaranteeing consistency and fairness of attribution.

**Q: Why can't you use PDP when features are correlated?**  
A: PDP computes the marginal effect of feature $x_j$ by setting it to a value $v$ and averaging predictions over all observed values of other features. When $x_j$ is correlated with other features, this averaging creates impossible combinations — e.g., fixing age=90 while averaging over all income values including $20K student incomes, which don't exist for 90-year-olds in your data. The extrapolation into unseen regions makes PDP misleading. ALE fixes this by computing effects only within narrow bins of $x_j$, using actual neighboring samples, then accumulating the local differences. ALE is always preferred when features are correlated.

**Q: A regulatory body is asking you to explain why your model denied a loan application. What do you provide?**  
A: A SHAP waterfall plot showing the top features that reduced the probability below the approval threshold, with their actual values and how much each contributed. For the specific applicant, it shows: "Your income of $35K reduced your approval probability by -0.12; your debt-to-income ratio of 0.45 reduced it by -0.09; your 3 recent credit inquiries reduced it by -0.06." This is interpretable by a compliance officer, satisfies GDPR Article 22 "meaningful information about the logic involved," and is reproducible (same input always gives same SHAP). Importantly, also verify: none of the top features are proxies for protected attributes (race, gender), and the explanation aligns with domain knowledge (high debt-to-income should indeed reduce creditworthiness).

For active-recall drilling on these terms, see [classical-ml-flashcards.md](classical-ml-flashcards.md).
