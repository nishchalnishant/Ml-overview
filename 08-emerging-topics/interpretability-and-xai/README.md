# Interpretability and Explainable AI (XAI)

---

## 1. The Problem

A radiologist reviews a neural network's output: 94% probability of malignancy. The radiologist needs to know — is the model looking at the tumor, or at the hospital watermark that appears on a specific scanner at the facility where most cancer cases in the training set came from?

Without an explanation, those two scenarios produce identical outputs. The model is equally confident in both. The radiologist cannot distinguish signal from artifact by looking at the number alone.

This is the concrete harm that motivates interpretability: **a model can be right for the wrong reasons, and you cannot detect that from its predictions alone.** On a held-out test set drawn from the same distribution, the model performs identically whether it has learned real pathology or a scanner artifact. The failure mode only reveals itself when the distribution shifts — when the model is deployed at a different hospital with a different scanner.

The same failure appears across domains:

- A loan model achieves 87% accuracy but has learned that zip code is the strongest predictor. Zip code is a proxy for race. The model is discriminatory and you cannot see it from accuracy metrics alone.
- A fraud detection model achieves 99.5% precision. A new merchant category appears post-deployment. The model has no learned features for it and silently misclassifies all transactions in that category as non-fraudulent.
- A hiring model achieves strong precision on historical hires. Historical hires skew male in engineering roles. The model has learned a gender proxy from resume formatting patterns.

In each case, the fix requires knowing what the model is doing internally — not just what it outputs.

---

## 2. The Core Insight

**Interpretability vs Explainability:**

These are distinct:

- **Interpretability**: The model's structure is directly readable. A linear regression is interpretable because the coefficients are the explanation. You do not need to analyze the model — you just read it.
- **Explainability**: The model is a black box; a separate analysis produces an approximation of its behavior. SHAP values explain a gradient-boosted tree. The tree itself is not interpretable — SHAP approximates what it does.

The distinction matters because post-hoc explanations are approximations. They can be wrong. They can be gamed. A model that produces plausible-looking SHAP values while actually relying on a proxy is a real failure mode.

**The taxonomy:**

| Dimension | Options |
|-----------|---------|
| When produced | Intrinsic (built-in) vs Post-hoc (after training) |
| Scope | Local (one prediction) vs Global (overall behavior) |
| Model dependency | Agnostic (black box access) vs Specific (uses internals) |

These are independent axes. SHAP is post-hoc, can be local or global, and has both model-agnostic (KernelSHAP) and model-specific (TreeSHAP, DeepSHAP) variants.

| Method | Scope | Agnostic? | Post-hoc? |
|--------|-------|-----------|-----------|
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

## 3. Intrinsic Models: Linear and Tree

### The mechanics of linear model interpretability

A linear model is its own explanation:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

Each `βᵢ` is the marginal effect: holding all other features constant, a one-unit increase in `xᵢ` changes the prediction by `βᵢ`. For logistic regression, `exp(βᵢ)` is the odds ratio.

**What breaks:** Raw coefficients are not comparable across features with different scales. A coefficient on income in dollars versus income in thousands of dollars differ by a factor of 1000. Always standardize before comparing:

```python
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
```

**Multicollinearity corrupts interpretation.** If `sqft` and `rooms` are correlated, coefficients become unstable — one may be large positive, the other large negative, and their sum carries the true signal. The individual coefficients are uninterpretable. Check variance inflation factor (VIF) before interpreting coefficients.

**Lasso performs implicit feature selection:** L1 regularization drives some coefficients to exactly zero. Surviving features retain the same marginal interpretation on the standardized scale.

```python
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)
selected = [(feat, coef) for feat, coef in zip(X.columns, lasso.coef_) if coef != 0]
```

### Decision tree path-based explanation

A decision tree is a flowchart. For any prediction, trace the path from root to leaf:

```
Is age > 30?
├── Yes: Is income > 50K?
│   ├── Yes: APPROVED (92%)
│   └── No: Is credit_score > 680?
│       ├── Yes: APPROVED (78%)
│       └── No: REJECTED (85%)
└── No: REJECTED (70%)
```

The path is the explanation: "Rejected because applicant is under 30 and credit score is below 680." This is natural language by construction.

```python
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

def tree_to_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.3f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.3f}"])
        else:
            class_idx = tree_.value[node].argmax()
            rules.append({
                'conditions': conditions,
                'prediction': class_names[class_idx],
                'samples': int(tree_.n_node_samples[node])
            })

    recurse(0, [])
    return rules
```

**What breaks:** Shallow trees are interpretable but inaccurate. Deep trees are accurate but cannot be followed by a human. Small changes in training data can produce completely different trees (high variance). This is why we use ensembles for accuracy and apply post-hoc methods to explain them.

---

## 4. Feature Importance

Feature importance answers: which features does this model rely on most?

### Mean Decrease in Impurity (MDI)

For tree ensembles, each split reduces impurity (Gini or entropy). Sum the reduction credited to each feature across all trees:

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**What breaks:** MDI is biased toward high-cardinality features — features with many unique values have more split thresholds to choose from. A random ID column would score high on MDI. Do not use MDI as a definitive importance measure.

### Permutation Importance

Shuffle one feature at a time. Measure how much model performance drops. If shuffling a feature destroys performance, that feature is important. Measure on the held-out test set:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)
```

**What breaks:** If features A and B are highly correlated, shuffling A does not hurt much because B still carries the same signal. Both appear unimportant even if the pair together is critical. Permutation importance underestimates correlated features.

### Drop-Column Importance

Retrain without each feature and measure performance loss. Most accurate; most expensive:

```python
from sklearn.metrics import accuracy_score

baseline = accuracy_score(y_test, rf.predict(X_test))
drop_col = {}

for col in X.columns:
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train.drop(columns=[col]), y_train)
    score = accuracy_score(y_test, rf_temp.predict(X_test.drop(columns=[col])))
    drop_col[col] = baseline - score
```

---

## 5. Partial Dependence Plots and ICE Plots

Feature importance tells you which features matter. PDPs tell you how — the shape of the relationship.

### Partial Dependence Plots (PDPs)

A PDP shows the marginal effect of one feature on predictions, averaged over all other features:

```
PD(x_s) = (1/n) Σᵢ f(x_s, xᵢ_c)
```

For each value of `x_s` in a grid, push it through the model holding other features at their observed values for each training instance, then average. The result shows the expected prediction as a function of `x_s` alone.

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    gb, X_train,
    features=[0, 1],
    feature_names=feature_names,
    ax=ax
)
plt.tight_layout()
```

**2D PDPs** show interactions between two features:

```python
PartialDependenceDisplay.from_estimator(
    gb, X_train,
    features=[(0, 1)],  # tuple = 2D PDP
    feature_names=feature_names
)
```

**What breaks:** PDPs average out heterogeneity. If the relationship between a feature and the outcome goes in opposite directions for different subgroups, the average is flat — and you miss the interaction entirely.

### Individual Conditional Expectation (ICE) Plots

ICE plots show one line per instance instead of the average. You can see heterogeneity that PDPs hide:

```python
# kind='individual' for ICE, 'average' for PDP, 'both' for both
PartialDependenceDisplay.from_estimator(
    gb, X_train,
    features=[0],
    feature_names=feature_names,
    kind='both'  # overlay ICE on PDP
)
```

**Centered ICE (c-ICE):** Subtract each line's value at a reference point so all lines start at zero. This separates the heterogeneity in slope from heterogeneity in level:

```python
PartialDependenceDisplay.from_estimator(
    gb, X_train,
    features=[0],
    feature_names=feature_names,
    kind='individual',
    centered=True  # c-ICE
)
```

**What breaks:** Both PDPs and ICE plots assume we can change one feature while holding others fixed. For correlated features this creates extrapolation into unrealistic regions of feature space — a PDP that varies income while holding job title constant will produce predictions for income/job combinations that don't exist in the real world.

---

## 6. LIME: Local Interpretable Model-Agnostic Explanations

### The problem LIME solves

A complex model's decision boundary is globally nonlinear and uninterpretable. But locally — in the neighborhood of any single prediction — it may be approximately linear. LIME exploits this.

### The core insight

The simplest true thing about any black-box model: **in a small neighborhood around a single point, the model's behavior can be approximated by a simple linear model.** That linear model is the explanation.

### The mechanics

For an input `x` to explain:
1. Generate `n` perturbed samples `z'` in `x`'s neighborhood
2. Get predictions `f(z')` from the black-box model for each sample
3. Weight samples by proximity to `x` using a kernel (closer = higher weight)
4. Fit a weighted sparse linear model `g` on the perturbed data: `g(z') ≈ f(z')`
5. The coefficients of `g` are the explanation for prediction `f(x)`

```python
import lime
import lime.lime_tabular
import numpy as np

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['class_0', 'class_1'],
    mode='classification',
    discretize_continuous=True
)

# Explain a single prediction
instance = X_test.iloc[0].values
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10,
    num_samples=5000
)

explanation.show_in_notebook()

# Get feature weights as dict
weights = dict(explanation.as_list())
print("Top positive features:", sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3])
print("Top negative features:", sorted(weights.items(), key=lambda x: x[1])[:3])
```

**LIME for text:** Perturbs by removing words (replacing with zeros), then fits a sparse linear model over word indicators:

```python
import lime.lime_text
from sklearn.pipeline import Pipeline

text_explainer = lime.lime_text.LimeTextExplainer(class_names=['negative', 'positive'])

text_explanation = text_explainer.explain_instance(
    text_instance="The product quality is excellent but delivery was slow",
    classifier_fn=text_pipeline.predict_proba,
    num_features=6
)
text_explanation.show_in_notebook()
```

### What breaks

**Instability.** LIME generates random perturbations. Run it twice on the same instance and you may get different explanations. The stochasticity in the neighborhood sampling propagates to the explanation.

**Neighborhood definition.** How far should the neighborhood extend? The kernel bandwidth hyperparameter controls this and is not obvious to set. Too small = too few samples to fit a stable model. Too large = the linear approximation is inaccurate.

**Fidelity-simplicity tradeoff.** LIME enforces sparsity — it returns a small number of features. If the true local explanation requires 20 features to be accurate, LIME's 10-feature model is both simpler and less faithful.

**Discretization artifacts.** LIME discretizes continuous features by default. "income > 50K" is the explanation even if the actual threshold is $52,384. This can misrepresent the model's actual boundary.

---

## 7. SHAP: Shapley Additive Explanations

### The problem SHAP solves

LIME is local and unstable. Feature importance is global but doesn't tell you direction. We want an explanation method that is: local (per-instance), consistent across runs, respects feature interactions, and has a grounding in a fairness axiom.

### The core insight

Game theory has a unique fair allocation scheme for cooperative games: the Shapley value. A coalition of players cooperates to earn a reward; the Shapley value is the unique allocation of that reward to players that satisfies fairness axioms. SHAP applies this to ML: the "players" are features, the "reward" is the prediction, and SHAP values are the unique fair allocation of the prediction to features.

### The Shapley axioms

The Shapley value is the unique value function satisfying all four:

1. **Efficiency:** Contributions sum to the prediction: `Σᵢ φᵢ = f(x) - E[f(x)]`
2. **Symmetry:** Features with identical contributions get equal values
3. **Dummy:** Features that never affect any prediction get zero
4. **Linearity:** SHAP values of a sum of models equal the sum of SHAP values

These four axioms together uniquely determine the Shapley value formula:

```
φᵢ = Σ_{S ⊆ F\{i}} [|S|!(|F|-|S|-1)!/|F|!] × [f(S∪{i}) - f(S)]
```

The contribution of feature `i` is its average marginal contribution across all possible orderings of features being added to the coalition.

### Computing SHAP values

**KernelSHAP (model-agnostic):** Frames SHAP as weighted linear regression over feature coalitions. Sample subsets of features, get predictions with and without each feature, fit weighted linear model. Exact Shapley values in expectation.

```python
import shap

# KernelSHAP — works with any model
background = shap.maskers.Independent(X_train, max_samples=100)
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_test[:50])

# For binary classification, shap_values is a list [class_0, class_1]
shap.summary_plot(shap_values[1], X_test[:50], feature_names=feature_names)
```

**TreeSHAP (exact, O(TLD²)):** Exploits tree structure to compute exact Shapley values by traversing the tree. Orders of magnitude faster than KernelSHAP. Works for random forests, gradient boosted trees (XGBoost, LightGBM, CatBoost):

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# TreeSHAP — model-specific, exact, fast
tree_explainer = shap.TreeExplainer(xgb_model)
shap_values = tree_explainer.shap_values(X_test)
expected_value = tree_explainer.expected_value
```

TreeSHAP complexity: O(TLD²) where T = number of trees, L = leaves per tree, D = max depth. Exact computation, not approximation.

**DeepSHAP:** Combines SHAP with DeepLIFT for neural networks. Propagates SHAP values from output back to inputs using DeepLIFT's backpropagation rules:

```python
deep_explainer = shap.DeepExplainer(neural_net_model, X_train[:100])
shap_values = deep_explainer.shap_values(X_test[:20])
```

### SHAP plots

**Force plot (local):** Visualize how each feature pushes the prediction above or below the expected value. Red features push prediction higher; blue push lower:

```python
# Single prediction force plot
shap.force_plot(
    base_value=expected_value,
    shap_values=shap_values[0],
    features=X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)
```

**Summary plot (global):** Each row is a feature, each dot is an instance. Color encodes feature value; x-axis position encodes SHAP value:

```python
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Dependence plot:** SHAP value of one feature vs its raw value. Color encodes another feature to reveal interactions:

```python
shap.dependence_plot(
    ind='age',
    shap_values=shap_values,
    features=X_test,
    interaction_index='income'  # color by income to show interaction
)
```

**Waterfall plot:** Single instance breakdown, showing exactly how each feature moves prediction from expected value to final value:

```python
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=expected_value,
    data=X_test.iloc[0].values,
    feature_names=feature_names
))
```

**Bar plot (global mean |SHAP|):**

```python
shap.bar_plot(shap_values.mean(0), feature_names=feature_names)
```

### What breaks

**Computational cost of KernelSHAP.** Exact Shapley computation is exponential in features. KernelSHAP approximates via sampling — slow for large datasets, and samples are needed per instance.

**Correlated features produce misleading SHAP values.** The Shapley value assumes features are independently maskable. When features are correlated, interventional SHAP (treating missing features as drawn from marginal distribution) differs from observational SHAP (conditioning on other features). The choice affects interpretation.

**SHAP does not identify causality.** A high SHAP value means the feature contributed to this prediction; it does not mean changing the feature would change the outcome. Income SHAP = +2.3 does not mean raising income by $10K would increase the predicted probability by any particular amount.

---

## 8. Attention Visualization

### Why attention seems like an explanation

Transformers compute attention weights: for each output token, a distribution over input tokens indicating "how much attention" each input received. It is tempting to interpret these as importance scores — high attention weight on a word means the model considered that word important for this prediction.

This interpretation is wrong.

### What breaks

**Attention is not explanation.** Four reasons:

1. **Gradient-attention disagreement.** Jain & Wallace (2019) showed that you can often permute or randomize attention weights with minimal effect on the output, because the value vectors (not just attention weights) determine what information is propagated. A different attention pattern over the same inputs can produce identical predictions.

2. **Multi-head mixing.** Output at each position is a weighted sum of value vectors from multiple heads, then projected. The attention weights in one head cannot be read off independently — the final representation mixes across all heads.

3. **Representational compression.** By layer 6 of a 12-layer transformer, input token representations have been thoroughly mixed. The attention pattern over original input tokens is not the same as the attention over meaningful input units.

4. **Rollout and flow methods underestimate mixing.** Attention rollout (multiplying attention matrices across layers) and attention flow both make independence assumptions that don't hold.

```python
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

text = "The patient shows signs of irregular heartbeat"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
# Shape: (12 layers,) each is (1, 12, seq_len, seq_len)
attentions = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Last layer, first head, first token's attention over all tokens
last_layer_first_head = attentions[-1][0, 0].numpy()
# This is NOT a reliable feature importance measure
```

**What to use instead:** Gradient-based saliency (Integrated Gradients), SHAP (DeepSHAP), or probing classifiers for understanding what attention heads encode.

---

## 9. Grad-CAM: Saliency Maps for CNNs

### The problem

For image classification, the question is: which pixels drove this prediction? We want a heatmap — same spatial resolution as the input — showing which regions the model attended to.

### The core insight

In a CNN, the final convolutional layer retains spatial information while encoding high-level semantics. The gradient of the predicted class score with respect to these feature maps tells you which channels were most important. Spatially average those gradients, use them as weights on the feature maps, and the result localizes class-discriminative regions.

### The mechanics

For class `c` and final convolutional feature maps `A^k` of spatial size H×W:

1. Compute gradient of class score w.r.t. each feature map: `∂y^c / ∂A^k_{ij}`
2. Global average pool to get channel weights: `α^c_k = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}`
3. Weighted combination of feature maps: `L^c_{Grad-CAM} = ReLU(Σ_k α^c_k A^k)`
4. Upsample to input resolution

ReLU is applied because we want regions that positively influence the class score.

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Global average pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

# Usage
model = models.resnet50(pretrained=True)
gradcam = GradCAM(model, target_layer=model.layer4[-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open("image.jpg").convert('RGB')
input_tensor = transform(img).unsqueeze(0).requires_grad_(True)

cam, predicted_class = gradcam.generate(input_tensor)

# Overlay on original image
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img)
axes[0].set_title('Original')
axes[1].imshow(img)
axes[1].imshow(cam, alpha=0.4, cmap='jet')
axes[1].set_title(f'Grad-CAM (class {predicted_class})')
plt.tight_layout()
```

**Variants:**

- **Grad-CAM++:** Uses higher-order gradients to better localize multiple occurrences of the same class in a single image. Weights positive partial derivatives of the score separately by spatial location.
- **Score-CAM:** Eliminates gradients entirely. Upsamples each activation map, uses it as a mask over the input, gets the score for each masked image, uses scores as weights. More robust to gradient noise.
- **Eigen-CAM:** Uses PCA of the feature maps instead of gradient weighting. No backward pass required.

### What breaks

**Grad-CAM only works on the chosen layer.** If you choose the wrong layer (e.g., too early in the network), the spatial resolution is too coarse and the semantic content is too low-level. The target layer should be the last convolutional layer before the classifier.

**Gradients can be noisy.** A single noisy backward pass produces a noisy heatmap. SmoothGrad addresses this: average saliency over many forward passes with Gaussian noise added to the input.

**Grad-CAM can highlight the right region for the wrong reason.** If the model identifies "dog" by background context (grass, yard) rather than the dog itself, Grad-CAM shows the background as important. The heatmap accurately reflects what the model is doing, not what it should be doing.

---

## 10. Integrated Gradients

### The problem

Vanilla gradient saliency (`∂output/∂input`) measures the local sensitivity of the output to each input dimension. For neural networks with saturated activations (e.g., an input dimension where the neuron is at the plateau of a sigmoid), the gradient is near zero even if that dimension has large absolute value. Gradient saliency says "unimportant" when the actual answer is "important, but the network is saturated."

### The core insight

Attribute not just the local gradient but the integral of gradients along a path from a baseline (a point with no information, typically a zero or blurred image) to the actual input. This captures the full cumulative effect of each input dimension on the output.

### The three axioms

Sundararajan et al. proved that Integrated Gradients is the unique attribution method satisfying all three:

1. **Sensitivity:** If a feature is the only difference between two inputs with different outputs, it gets nonzero attribution.
2. **Implementation Invariance:** Two networks computing the same function on all inputs get the same attributions, regardless of internal implementation.
3. **Completeness:** Attributions sum to the difference between the output at the input and the output at the baseline: `Σᵢ IG_i(x) = f(x) - f(x')`

Vanilla gradients violate sensitivity; LIME violates implementation invariance (two different models produce different LIME explanations even if they compute the same function).

### The mechanics

```
IG_i(x) = (x_i - x'_i) × ∫₀¹ [∂f(x' + α(x - x')) / ∂x_i] dα
```

Riemann sum approximation with `m` steps:

```
IG_i(x) ≈ (x_i - x'_i) × (1/m) Σₖ₌₁ᵐ [∂f(x' + (k/m)(x - x')) / ∂x_i]
```

```python
import torch
import numpy as np

def integrated_gradients(model, input_tensor, baseline=None, n_steps=50, target_class=None):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_steps + 1)
    interpolated = torch.stack([baseline + alpha * (input_tensor - baseline) for alpha in alphas])
    interpolated.requires_grad_(True)

    # Get gradients at each interpolation step
    outputs = model(interpolated)
    if target_class is None:
        target_class = outputs[-1].argmax().item()

    # Sum loss over interpolation steps
    target_outputs = outputs[:, target_class]
    target_outputs.sum().backward()

    gradients = interpolated.grad  # (n_steps+1, *input_shape)

    # Average gradients (trapezoidal rule)
    avg_gradients = gradients[:-1].mean(dim=0)

    # Element-wise multiply by (input - baseline)
    integrated_grads = (input_tensor - baseline) * avg_gradients
    return integrated_grads.detach()


# Captum implementation (cleaner)
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz

ig = IntegratedGradients(model)
attributions = ig.attribute(
    inputs=input_tensor,
    baselines=baseline,
    target=target_class,
    n_steps=50,
    method='gausslegendre'  # more accurate than Riemann sum
)

# Noise tunnel for smoother attributions (SmoothGrad + IG)
nt = NoiseTunnel(ig)
smooth_attributions = nt.attribute(
    inputs=input_tensor,
    baselines=baseline,
    target=target_class,
    n_steps=50,
    nt_type='smoothgrad',
    nt_samples=10,
    stdevs=0.1
)
```

**Baseline choice matters.** The baseline is the "no information" reference point. For images: black image (zeros) is standard. For text: all-padding or all-mask tokens. For tabular data: column means or zeros. Different baselines give different attributions — there is no single correct baseline. The baseline encodes what "absence of information" means.

### What breaks

**Completeness is not human-friendly.** The attributions sum to `f(x) - f(baseline)`. If the baseline predicts 0.5 and the input predicts 0.9, attributions sum to 0.4. But humans want to know which features drove the prediction to 0.9, not the difference from baseline.

**Computational cost.** `n_steps=50` requires 50 forward and backward passes per instance. For large models, this is expensive.

**Path dependence.** The straight-line path from baseline to input is conventional but not uniquely correct. Different paths give different attributions. IG uses the straight line as convention.

---

## 11. TCAV: Concept-Based Explanations

### The problem

SHAP and IG explain predictions in terms of input features (pixels, tokens, tabular columns). But humans think in concepts: "the model predicted malignant because it saw irregular borders," not "pixel at coordinates (234, 156) contributed +0.003."

TCAV (Testing with Concept Activation Vectors, Kim et al. 2018) bridges this gap.

### The core insight

If you have examples of a concept (images of "stripes," "irregular borders," "young patient") and non-examples, you can train a linear classifier on the model's internal activations to find the direction in activation space that separates concept from non-concept. That direction is the Concept Activation Vector (CAV). The TCAV score measures how often the model's predictions align with movement in the CAV direction.

### The mechanics

**Step 1:** Collect positive and negative concept examples. E.g., "stripe patterns" = 50 images of stripes; negatives = 50 random images.

**Step 2:** Run both sets through the model up to layer `l`. Collect activations.

**Step 3:** Train a linear classifier on the activations to separate concept from non-concept. The normal vector to the decision boundary is the CAV: `v_C^l`.

**Step 4:** For each training example of the target class, compute the directional derivative of the class score with respect to the CAV direction:

```
S_C,k,l(x) = ∇h_{l,k}(f_l(x)) · v_C^l
```

**Step 5:** TCAV score = fraction of class `k` examples with positive directional derivative:

```
TCAV_{Q,C,k,l} = |{x ∈ X_k : S_C,k,l(x) > 0}| / |X_k|
```

TCAV = 0.8 for "irregular borders" and "malignancy" means 80% of malignant images move in the irregular-borders direction in the model's activations.

```python
from captum.concept import TCAV, Concept
from captum.concept import Classifier
import torch

# Define concepts with example images
stripes_concept = Concept(
    id=0,
    name="stripes",
    data_iter=DataLoader(stripes_dataset, batch_size=64)
)
random_concept = Concept(
    id=1,
    name="random",
    data_iter=DataLoader(random_dataset, batch_size=64)
)

# Create TCAV object
tcav = TCAV(
    model=model,
    layers=['layer3', 'layer4'],
    model_id='resnet50_v1',
    classifier=Classifier()
)

# Compute TCAV scores
experimental_sets = [[stripes_concept, random_concept]]
tcav_scores = tcav.interpret(
    inputs=target_class_images,
    experimental_sets=experimental_sets,
    target=target_class_idx
)
```

**Statistical test.** A high TCAV score is only meaningful if the CAV is real — not an artifact of the specific random negative examples chosen. Run TCAV multiple times with different random negatives and check: does the TCAV score distribution differ significantly from 0.5? A concept that scores 0.82 consistently across 20 random negative splits is meaningful; one that scores 0.82 in one run and 0.53 in another is not.

### What breaks

**Concept quality depends on your examples.** Noisy or heterogeneous concept examples produce a weak CAV. The linear classifier trains on what you give it — garbage concept images produce a garbage CAV.

**Linear CAV assumption.** Concepts may not be linearly separable in activation space. A CAV trained with logistic regression is a linear boundary; nonlinear concepts are poorly captured.

**Layer sensitivity.** The TCAV score varies by layer. A concept that appears influential at layer 3 may be irrelevant at layer 7. You need to test across layers, which multiplies compute.

---

## 12. Model Cards and Documentation

### The problem

A model trained for one context is deployed in another. A skin lesion classifier trained on light-skinned patients is deployed on a diverse population. A hiring model trained on 2010 data is deployed in 2025. Without explicit documentation of training data, intended use cases, and known limitations, these failures are invisible until they cause harm.

### Model Cards (Mitchell et al. 2019)

A model card is a standardized one-to-two page document attached to a deployed model:

```markdown
## Model Details
- **Developer:** Johns Hopkins Medical AI Lab
- **Model date:** March 2024
- **Model version:** v2.3
- **Model type:** Fine-tuned ResNet-50 for binary classification
- **License:** Proprietary, internal use only
- **Citation:** [arXiv:2403.XXXXX]

## Intended Use
- **Primary use:** Assist radiologists in flagging chest X-rays for pneumonia
- **Intended users:** Radiologists in US hospital network
- **Out-of-scope uses:** Standalone diagnosis without radiologist review; 
                         pediatric patients under 12; non-chest imaging

## Factors
- **Relevant factors:** Patient age, BMI, scanner manufacturer, image quality
- **Evaluation factors:** Stratified by age group, sex, scanner type

## Metrics
- **Performance metrics:** AUROC, sensitivity @ 90% specificity
- **Thresholds:** 0.7 decision threshold for high-sensitivity deployment
- **Decision threshold:** Varies by deployment context

## Evaluation Data
- **Dataset:** CheXpert test set (n=234 patients) + internal holdout (n=1,847)
- **Demographics:** 47% female, 53% male; age 18-89; 3 scanner manufacturers

## Training Data
- **Source:** MIMIC-CXR (227,827 images) + CheXpert (224,316 images)
- **Known biases:** Underrepresented: patients >80 years, BMI >40, non-GE scanners

## Quantitative Analysis
| Demographic | AUROC | Sensitivity @ 90% spec |
|-------------|-------|------------------------|
| Age <40 | 0.91 | 82% |
| Age 40-70 | 0.94 | 87% |
| Age >70 | 0.88 | 79% |
| Female | 0.93 | 85% |
| Male | 0.92 | 84% |

## Ethical Considerations
- Model does not replace radiologist judgment
- Known performance gap in older patients

## Caveats and Recommendations
- Retrain or fine-tune before deployment with scanners not in training set
- Monitor performance drift quarterly against labeled production samples
```

**IBM AI FactSheets** extend this: structured metadata about data sources, preprocessing decisions, hyperparameter choices, and fairness audits. Intended for enterprise governance workflows.

**Datasheets for Datasets (Gebru et al. 2018):** Analogous documentation for datasets — motivation, composition, collection process, preprocessing, uses, distribution, maintenance.

---

## 13. Interpretability in LLMs

Language models present a different interpretability challenge. There are no spatial activations to visualize with Grad-CAM. The input is a sequence of tokens, not an image. And the model is predicting the next token, not a class — so "which input token was important" is a question with 50,277 possible outputs.

### Probing Classifiers

**The question:** Does layer `l` of a transformer encode property P (e.g., part-of-speech, syntactic role, semantic similarity)?

**The method:** Train a small classifier on top of the frozen activations at layer `l` to predict P. High accuracy = the representation at layer `l` encodes P.

```python
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

def get_layer_representations(texts, labels, layer_idx):
    all_reps = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # hidden_states: (n_layers+1, batch, seq_len, hidden_size)
        # Take [CLS] token representation at target layer
        rep = outputs.hidden_states[layer_idx][0, 0, :].numpy()
        all_reps.append(rep)
    return np.array(all_reps)

# Probe all 12 BERT layers for POS tagging
probing_results = []
for layer in range(13):  # 0 = embeddings, 1-12 = transformer layers
    reps = get_layer_representations(texts, pos_labels, layer)
    X_train, X_test, y_train, y_test = train_test_split(reps, pos_labels, test_size=0.2)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    accuracy = probe.score(X_test, y_test)
    probing_results.append({'layer': layer, 'accuracy': accuracy})
```

BERT finding: lower layers encode syntactic properties (POS, dependency arcs); higher layers encode semantic properties (coreference, NER). This is not a surprise — it recapitulates the analysis hierarchy.

**Amnesic probing caveat:** A high probing accuracy at layer `l` means the representation at layer `l` contains enough information to predict property P. It does not mean the model *uses* property P for its final predictions. The information might be present but causally irrelevant. Amnesic probing (Elazar et al. 2021) tests this: erase the probed property from activations and measure whether final task performance drops.

### Logit Lens

**The question:** What is the model "thinking" at intermediate layers?

**The method:** Apply the unembedding matrix `W_U` directly to the residual stream at each layer (before the final layer norm + unembedding). The resulting distribution over the vocabulary reveals what token the model "predicts" from that point forward:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, hidden)
W_U = model.lm_head.weight  # (vocab_size, hidden_size)

# Apply unembedding to each layer's last token representation
predictions_per_layer = []
for layer_idx, hidden in enumerate(hidden_states):
    last_token_hidden = hidden[0, -1, :]  # (hidden_size,)
    # Apply layer norm before unembedding
    normed = model.transformer.ln_f(last_token_hidden.unsqueeze(0))
    logits = (W_U @ normed.T).squeeze()
    top_token = tokenizer.decode([logits.argmax().item()])
    top_prob = torch.softmax(logits, dim=0).max().item()
    predictions_per_layer.append({
        'layer': layer_idx,
        'top_token': top_token,
        'probability': top_prob
    })
```

The logit lens reveals at which layer a model "converges" on the final answer. For factual recall in GPT-2, the correct answer often emerges in early-middle layers and stabilizes — the final layers refine rather than determine the answer.

### Mechanistic Interpretability

**The problem with probe classifiers and logit lens:** they tell you what information is present and when it appears, but not the algorithm the model implements. How does the model compute "The capital of France is Paris"?

Mechanistic interpretability attempts to reverse-engineer the circuits — the specific subgraph of attention heads and MLP neurons — that implement a particular computation.

**Superposition:** A network with `d` dimensions can represent more than `d` features if it accepts some interference. Toy model: 5-dimensional MLP representing 20 concepts. Features are represented as near-orthogonal directions in a lower-dimensional space. When features are sparse (most are zero for any given input), superposition works — interference is rare. This explains why neural networks pack far more information into their representations than their dimension count suggests.

**Induction heads:** A two-layer attention pattern that implements in-context copying. Head A at layer 1 is a "previous token head" — attends to the previous position. Head B at layer 2 is a "key-query composition head" — attends to positions whose previous token matches the current token. Together they implement: if [A][B] appeared earlier, predict [B] when [A] appears again. This is in-context learning in its simplest form and appears consistently across transformer architectures.

**The IOI circuit (Wang et al. 2022):** For "When Mary and John went to the store, John gave a drink to" → "Mary." A 26-head circuit implements: find the repeated name (John), suppress it, attend to the other name (Mary). Analyzed by ablating attention heads and measuring performance on the task.

**Activation patching (causal tracing):** Identify which model components are causally responsible for a specific fact:

```python
def activation_patching(model, clean_prompt, corrupted_prompt, target_token_id, layer_to_patch):
    """
    Run corrupted input. Patch activation at layer_to_patch from clean run.
    Measure recovery of target probability.
    """
    # Clean run - store activations
    clean_inputs = tokenizer(clean_prompt, return_tensors='pt')
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors='pt')

    with torch.no_grad():
        clean_outputs = model(**clean_inputs, output_hidden_states=True)
        clean_activations = clean_outputs.hidden_states[layer_to_patch]

    # Corrupted run with hook to patch specific layer
    patched_activations = {}

    def patch_hook(module, input, output):
        return clean_activations  # replace with clean activations

    hook = model.transformer.h[layer_to_patch].register_forward_hook(patch_hook)

    with torch.no_grad():
        patched_outputs = model(**corrupted_inputs)
        patched_logits = patched_outputs.logits[0, -1, :]
        patched_prob = torch.softmax(patched_logits, dim=0)[target_token_id].item()

    hook.remove()
    return patched_prob
```

Run across all layers to produce a "causal importance" map. Layers/heads where patching recovers the correct answer are causally involved in the computation.

### Sparse Autoencoders (SAEs)

**The problem:** Due to superposition, individual neurons are polysemantic — they activate for multiple unrelated concepts. Feature decomposition is impossible at the neuron level.

**The insight (Templeton et al. 2024, Anthropic):** Train a sparse autoencoder on model activations to learn a larger dictionary of monosemantic features. The SAE decomposes each activation into a sparse sum of learned features, each of which activates for semantically coherent concepts.

```
SAE: z = ReLU(W_enc(x - b_dec) + b_enc)
     x̂ = W_dec z + b_dec
Minimize: ||x - x̂||² + λ||z||₁
```

L1 penalty enforces sparsity. The features in `W_dec` are the learned dictionary. Templeton et al. (2024) scaled SAEs to Claude Sonnet and found interpretable features for: the Golden Gate Bridge, DNA replication, emotions, countries, specific people. The same model component that encodes "Golden Gate Bridge" also encodes structural/geographical concepts through superposition — the SAE decomposes this into distinct monosemantic features.

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.l1_coeff = l1_coeff

    def forward(self, x):
        # Pre-encoder bias
        x_centered = x - self.b_dec
        # Encode with ReLU for sparsity
        z = torch.relu(self.encoder(x_centered))
        # Decode
        x_hat = self.decoder(z) + self.b_dec
        # Compute losses
        reconstruction_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        sparsity_loss = self.l1_coeff * z.abs().sum(dim=-1).mean()
        return x_hat, z, reconstruction_loss + sparsity_loss

    def get_active_features(self, x, top_k=10):
        with torch.no_grad():
            x_centered = x - self.b_dec
            z = torch.relu(self.encoder(x_centered))
            values, indices = z.topk(top_k)
        return indices, values
```

---

## 14. Regulatory Context

### GDPR Article 22 and the Right to Explanation

Article 22 of GDPR restricts fully automated decisions that "significantly affect" EU individuals. Recital 71 specifies the right to "obtain an explanation of the decision reached."

Practically, this means:
- Credit denials, hiring decisions, insurance pricing by algorithm require a human review path
- The explanation must be provided in a "meaningful" form — not just "the model said so"
- The individual has the right to contest the decision

```python
class GDPRCompliantClassifier:
    def __init__(self, model, explainer, feature_names):
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names

    def predict_with_explanation(self, X, threshold=0.5):
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        shap_values = self.explainer.shap_values(X)

        explanations = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            instance_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values[1][i]
            # Top 3 features by absolute SHAP value
            top_indices = np.argsort(np.abs(instance_shap))[-3:][::-1]
            top_factors = [
                {
                    'feature': self.feature_names[j],
                    'shap_value': float(instance_shap[j]),
                    'impact': 'positive' if instance_shap[j] > 0 else 'negative'
                }
                for j in top_indices
            ]
            explanations.append({
                'prediction': int(pred),
                'probability': float(prob),
                'decision': 'approved' if pred == 1 else 'declined',
                'key_factors': top_factors,
                'explanation_text': self._generate_explanation_text(pred, top_factors)
            })
        return explanations

    def _generate_explanation_text(self, prediction, factors):
        if prediction == 0:
            neg_factors = [f for f in factors if f['impact'] == 'negative']
            if neg_factors:
                feature_list = ', '.join([f['feature'] for f in neg_factors[:3]])
                return f"Application declined primarily due to: {feature_list}"
        return "Application approved based on strong qualifying factors."
```

### Adverse Action Notices (ECOA/FCRA)

In the US, the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA) require lenders to provide applicants with the specific reasons for credit denial. The standard: list the top factors that negatively affected the decision.

Best practice with SHAP: sort features by their SHAP contribution to the negative outcome, provide the top 4 as the adverse action reasons:

```python
def generate_adverse_action_reasons(shap_values, feature_names, n_reasons=4):
    """
    ECOA/FCRA compliant adverse action notice generation.
    Returns top n negative-contributing features.
    """
    feature_shap_pairs = list(zip(feature_names, shap_values))
    # Sort by SHAP value (most negative first = hurt the application most)
    sorted_by_impact = sorted(feature_shap_pairs, key=lambda x: x[1])

    adverse_reasons = []
    for feature, shap_val in sorted_by_impact[:n_reasons]:
        if shap_val < 0:
            adverse_reasons.append({
                'reason': feature,
                'impact_score': abs(shap_val),
                'direction': 'decreased_score'
            })

    return adverse_reasons

# Generate adverse action notice
shap_vals = tree_explainer.shap_values(applicant_features)
reasons = generate_adverse_action_reasons(shap_vals[0], feature_names)

print("Adverse Action Notice")
print("We were unable to approve your application. Principal reasons:")
for i, reason in enumerate(reasons, 1):
    print(f"  {i}. {reason['reason']}")
```

### SR 11-7 (Federal Reserve, OCC, FDIC Model Risk Management)

Guidance issued 2011, expanded 2021. Key requirements for financial institutions:

1. **Model documentation:** Conceptual soundness, data sources, assumptions, known limitations
2. **Validation:** Independent validation by team not involved in development; ongoing monitoring
3. **Governance:** Model inventory, approval process, escalation path for model failures
4. **Explainability:** Models used for credit decisions must produce explainable outputs

SR 11-7 does not mandate specific XAI methods but requires that model outputs can be explained to regulators and to affected individuals.

### EU AI Act (2024)

Article 13 requires "high-risk AI systems" to be designed and developed with a level of transparency such that deployers can interpret the system's output. High-risk categories include: biometric identification, critical infrastructure, education, employment, essential services, law enforcement, migration, administration of justice.

### High-Stakes Domain Requirements

| Domain | Regulation | XAI Requirement |
|--------|-----------|-----------------|
| Credit | ECOA, FCRA, GDPR | Adverse action reasons, right to explanation |
| Healthcare | FDA (AI/ML guidance), GDPR | Clinical validation, failure mode documentation |
| Employment | EEOC guidance, GDPR | Disparate impact analysis, explainable screening |
| Criminal justice | Due process, COMPAS cases | Defendant's right to challenge algorithmic scores |
| Finance (models) | SR 11-7, Basel III | Model risk management, independent validation |

---

## 15. What Breaks

**LIME is unstable.** Run LIME twice on the same instance and get different top features. The stochasticity in neighborhood sampling propagates into the explanation. Use SHAP for production explanations where consistency matters.

**SHAP values require correlated-feature care.** KernelSHAP and TreeSHAP handle feature correlations differently. With highly correlated features (r > 0.8), SHAP values for individual features become difficult to interpret because the "contribution" is split arbitrarily between correlated features. Report SHAP values for correlated features together or acknowledge the limitation.

**Post-hoc explanations can be gamed.** Slack et al. (2020) showed that a model can produce fair-looking SHAP values in auditing mode while discriminating against protected groups in deployment. The model detects that it is being audited (e.g., via systematic input perturbations used by LIME/SHAP) and behaves differently. Explanations do not guarantee the model's actual behavior.

**Completeness axiom does not imply correctness.** SHAP values sum to the prediction; they are internally consistent. This does not mean they reflect the causal structure of the prediction. A model that happens to predict correctly using the wrong features will produce SHAP values that are internally consistent but directionally misleading.

**Attention is not explanation** — addressed in Section 8, but worth restating: high attention weights are not a reliable indicator of feature importance. Gradient-based methods (IG, Grad-CAM) are more reliable.

**Probing classifiers measure presence, not use.** High probing accuracy at layer `l` means the information is encoded there, not that the model uses it for its predictions. Amnesic probing is needed to establish causal relevance.

**Grad-CAM highlights the right region for wrong reasons.** If a model identifies "dog" by background context rather than the dog itself, the heatmap accurately shows the background as important — reflecting the model's actual (flawed) behavior. The heatmap is correct about what the model is doing, not about what it should be doing.

**Mechanistic interpretability does not scale yet.** Circuit analysis has been demonstrated on small transformers and specific tasks (IOI, indirect object identification, docstring completion). Scaling to full 70B+ parameter models for arbitrary tasks remains an open research problem.

**The Rashomon effect.** For many datasets, many models fit the data equally well but differ substantially in their feature importances. There is no unique "true" explanation — any of the equivalent models is equally valid, but each produces different SHAP values. High-stakes explanations should acknowledge that the model itself is one of many equally valid models.

---

## Key Interview Points

**Interpretability vs explainability:** Interpretability = model structure is directly readable (linear coefficients, decision tree paths). Explainability = post-hoc analysis of a black box (SHAP, LIME, Grad-CAM). Post-hoc explanations are approximations and can be wrong.

**LIME vs SHAP:** LIME fits a local weighted linear model to perturbed samples — fast but unstable (stochastic neighborhood sampling). SHAP computes Shapley values (the unique fair allocation of prediction to features satisfying efficiency, symmetry, dummy, linearity axioms) — slower but consistent and grounded in game theory.

**Negative SHAP value:** A negative SHAP value for feature `x_i` means that feature pushed the prediction below the baseline (expected) value. It does not mean the feature has a low raw value — it means its contribution to this specific prediction was negative relative to what would be predicted without knowing that feature's value.

**Attention weights are not explanation.** Jain & Wallace (2019): attention weights can be permuted or randomized with minimal effect on output. High attention weight does not imply causal importance. Use IG or Grad-CAM instead.

**Explaining a loan rejection under ECOA/FCRA:** Run TreeSHAP on the model's prediction for that applicant. Sort features by SHAP contribution to the rejection (most negative first). Report the top 4 as the adverse action reasons. These are the specific factors that most reduced the applicant's predicted creditworthiness.

**Validating that explanations reflect actual model behavior:** (1) Check consistency: same input → same explanation across runs. (2) Check fidelity: removing top SHAP features should degrade model performance proportionally. (3) Test adversarially: construct inputs where explanations look different but predictions are the same — if SHAP is consistent, this should be rare.

**PDP vs SHAP dependence plot:** PDP marginalizes over other features to show average relationship. SHAP dependence plot shows SHAP value vs feature value for each instance — preserves heterogeneity, can color by interaction feature to reveal nonlinearity and interactions. Use SHAP dependence when you suspect the average effect masks subgroup heterogeneity.

**TreeSHAP vs KernelSHAP:** TreeSHAP is model-specific (trees only), exact (not approximate), O(TLD²) — fast enough for real-time inference. KernelSHAP is model-agnostic (any model), approximate (weighted linear regression over sampled coalitions), O(2^d) or O(n_samples × n_features) — too slow for large feature spaces without approximation.

**GDPR right to explanation:** Article 22 restricts fully automated decisions significantly affecting EU individuals. Recital 71 specifies explanation of the logic involved. In practice: provide the adverse action reasons (top negative SHAP factors), allow human review, maintain a human-in-the-loop escalation path.

**Counterfactual explanations:** "What would have to change for the decision to be different?" Different from SHAP (which explains the current decision). DiCE (Diverse Counterfactual Explanations) generates multiple diverse counterfactuals respecting feature actionability constraints (e.g., age cannot decrease). Directly actionable for rejected applicants.

**SAEs and mechanistic interpretability:** Standard neurons are polysemantic due to superposition. Sparse autoencoders trained on model activations learn a larger dictionary of monosemantic features — each feature activates for a semantically coherent concept. Templeton et al. (2024) demonstrated this at scale on Claude Sonnet, finding interpretable features for specific entities, concepts, and emotions.

**TCAV score interpretation:** TCAV = 0.82 for (concept="stripes", class="zebra", layer="conv5") means 82% of zebra images have a positive directional derivative in the stripes CAV direction at conv5. Requires statistical testing across random negative splits to confirm the concept is genuinely informative.

---

## Quick Reference

| Task | Method |
|------|--------|
| Tree/forest explanation (local) | TreeSHAP force plot |
| Tree/forest explanation (global) | TreeSHAP summary plot or permutation importance |
| Neural network (image) | Grad-CAM or Integrated Gradients + Captum |
| Neural network (tabular) | DeepSHAP or KernelSHAP |
| Any model, fast | LIME (accept instability) |
| Any model, rigorous | KernelSHAP (slow but consistent) |
| LLM: what information is where | Probing classifiers by layer |
| LLM: when does answer form | Logit lens (W_U applied to residual stream) |
| LLM: causal circuit analysis | Activation patching |
| LLM: monosemantic features | Sparse autoencoders (SAEs) |
| Concept audit | TCAV |
| Regulatory compliance (credit) | TreeSHAP → adverse action reasons (top negative SHAP) |
| GDPR explanation | Force plot or waterfall plot, human review path |
