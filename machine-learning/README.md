# Machine Learning

## Algorithm Classification Framework

Machine learning algorithms can be categorized by their core mathematical approach. Understanding these categories helps in selecting the right algorithm for your problem.

---

## 1. Linear Methods (Hyperplane/Boundary-based)

These algorithms create decision boundaries or hyperplanes to separate or predict data.

### Algorithms Overview

| **Algorithm** | **Type** | **Time Complexity** | **Key Strength** | **When to Use** |
|--------------|---------|-------------------|-----------------|----------------|
| **Linear Regression** | Regression | O(n·p²) | Fast, interpretable | Linear relationships, baseline |
| **Logistic Regression** | Classification | O(n·p) | Probabilistic output | Binary/multiclass, need probabilities |
| **SVM/SVR** | Both | O(n²·p) to O(n³·p) | Handles high dimensions | Small-medium data, non-linear with kernels |
| **Perceptron** | Classification | O(n·p) | Simple, fast | Linearly separable data |
| **LDA** | Classification | O(n·p²) | Dimensionality reduction + classification | Gaussian features, reduce dimensions |
| **PCA** | Dimensionality Reduction | O(min(n²·p, n·p²)) | Variance preservation | Visualization, noise reduction |

*n = samples, p = features*

### Detailed Algorithms

#### Linear Regression
**Purpose:** Predict continuous output using linear relationship

**Formula:**
```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ
Loss: MSE = (1/n)Σ(y - ŷ)²
```

**Training:** Gradient Descent iteratively optimizes weights and bias
```python
# Pseudocode
Initialize: w, b randomly
For each iteration:
    ŷ = X @ w + b
    loss = MSE(y, ŷ)
    w -= learning_rate * ∂loss/∂w
    b -= learning_rate * ∂loss/∂b
```

**Assumptions:**
- Linear relationship between X and y
- Independent features (no multicollinearity)
- Homoscedasticity (constant variance of errors)
- Normal distribution of errors

**Interview Question:** *"What if features are correlated?"*
> Use Ridge/Lasso regularization or PCA to handle multicollinearity

---

#### Logistic Regression
**Purpose:** Binary or multiclass classification with probability outputs

**Formula:**
```
z = w·x + b
ŷ = σ(z) = 1/(1 + e⁻ᶻ)    [Sigmoid for binary]
ŷ = softmax(z) = e^zᵢ/Σe^zⱼ  [Softmax for multiclass]
Loss: Cross-Entropy = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Key Difference from Linear Regression:**
- Output: Probabilities [0,1] vs continuous values
- Loss: Cross-entropy vs MSE
- Decision boundary: Probabilistic threshold

---

#### Support Vector Machines (SVM/SVR)
**Purpose:** Find optimal hyperplane that maximizes margin

**Key Concepts:**
- **Margin:** Distance between hyperplane and nearest data points
- **Support Vectors:** Data points closest to the hyperplane
- **Kernel Trick:** Map data to higher dimensions for non-linear separation

**Kernel Types:**

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | K(x,y) = x·y | Linearly separable data |
| Polynomial | K(x,y) = (γx·y + r)^d | Polynomial relationships |
| RBF (Gaussian) | K(x,y) = exp(-γ\\|x-y\\|²) | General non-linear, most common |
| Sigmoid | K(x,y) = tanh(γx·y + r) | Neural network-like |

**For Regression (SVR):**
- Defines ε-tube (margin of tolerance) around prediction line
- Only penalizes errors outside the tube

**Interview Tip:** SVM is powerful but expensive; prefer for small-medium datasets (<10K samples)

---

#### Perceptron
**Purpose:** Simplest neural network, binary classification

**Formula:**
```
ŷ = sign(w·x + b)
Update rule: w += η·(y - ŷ)·x
```

**Limitations:**
- Only linearly separable data (XOR problem)
- Replaced by more robust methods (Logistic Regression, Neural Networks)

---

#### Linear Discriminant Analysis (LDA)
**Purpose:** Classification + dimensionality reduction

**How It Works:**
1. Maximize between-class variance
2. Minimize within-class variance
3. Project to lower dimensions that separate classes

**vs PCA:**
- LDA: Supervised (uses labels), maximizes class separation
- PCA: Unsupervised, maximizes variance

---

#### Principal Component Analysis (PCA)
**Purpose:** Reduce dimensionality while preserving variance

**Steps:**
1. Standardize data (mean=0, std=1)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top k components

**When to Use:**
- High-dimensional data (curse of dimensionality)
- Visualization (reduce to 2D/3D)
- Speed up training (fewer features)
- Remove noise/collinearity

**Explained Variance:**
```
Select k components that capture 95%+ variance
```

---

## 2. Tree/Graph-based Methods

Hierarchical decision-making using tree structures or graph operations.

### Algorithms Overview

| **Algorithm** | **Type** | **Time Complexity** | **Ensemble** | **When to Use** |
|--------------|---------|-------------------|-------------|----------------|
| **Decision Tree** | Both | O(n·log(n)·p) | No | Interpretability needed |
| **Random Forest** | Both | O(n·log(n)·p·t) | Bagging | General purpose, robust |
| **Gradient Boosting** | Both | O(n·log(n)·p·t) | Boosting | Highest accuracy on tabular |
| **AdaBoost** | Classification | O(n·log(n)·p·t) | Boosting | Binary classification |
| **XGBoost/LightGBM/CatBoost** | Both | Optimized GBM | Boosting | Production-grade performance |

*t = number of trees*

### Detailed Algorithms

#### Decision Tree
**Purpose:** Interpretable hierarchical decision-making

**Splitting Criteria:**
- **Classification:**
  - Gini Impurity: `1 - Σpᵢ²` (measures disorder)
  - Entropy: `-Σpᵢ·log(pᵢ)` (information gain)
- **Regression:**
  - MSE reduction

**Algorithm:**
```
1. For each feature:
   - Try all possible split points
   - Calculate information gain
2. Choose split with maximum gain
3. Recursively split child nodes
4. Stop when:
   - Max depth reached
   - Min samples per leaf
   - No further gain
```

**Pruning:**
- **Pre-pruning:** Early stopping (max_depth, min_samples_split)
- **Post-pruning:** Build full tree, then remove low-value branches

**Advantages:**
- Highly interpretable
- Handles non-linear relationships
- No feature scaling needed
- Handles mixed data types

**Disadvantages:**
- Prone to overfitting (high variance)
- Unstable (small data changes → different tree)
- Biased toward dominant classes

---

#### Random Forest
**Purpose:** Ensemble of decision trees to reduce variance

**How It Works:**
1. **Bootstrapping:** Create 100-1000 different datasets by sampling with replacement
2. **Random feature subset:** At each split, consider only √p random features
3. **Majority voting:** Aggregate predictions (vote for classification, average for regression)

**Advantages:**
- Reduces overfitting vs single tree
- Robust to outliers and noise
- Handles imbalanced data well
- Feature importance built-in

**Hyperparameters:**
- n_estimators: Number of trees (100-500)
- max_depth: Tree depth (None or 10-50)
- min_samples_split: Minimum samples to split (2-10)
- max_features: Features per split ('sqrt' for classification, 'log2' or 1/3 for regression)

**Interview Question:** *"Why random subset of features?"*
> Decorrelates trees, prevents all trees from splitting on the same strong features

---

#### Gradient Boosting Machines (GBM)
**Purpose:** Sequential ensemble that corrects previous errors

**How It Works:**
```
1. Start with weak initial prediction (mean for regression)
2. For each iteration:
   - Calculate residuals: r = y - ŷ_current
   - Fit new tree to predict residuals
   - Update: ŷ_new = ŷ_current + learning_rate × tree_prediction
3. Final model: Sum of all weak learners
```

**Key Insight:** Each tree focuses on mistakes of previous trees

**Advantages:**
- Often highest accuracy on tabular data
- Handles non-linear relationships
- Built-in feature selection

**Disadvantages:**
- Sequential training (slower than Random Forest)
- More prone to overfitting (needs careful tuning)
- Less interpretable

**Modern Variants:**

| **Variant** | **Key Innovation** | **Best For** |
|------------|-------------------|-------------|
| **XGBoost** | Regularization, parallel processing | General purpose, Kaggle winner |
| **LightGBM** | Leaf-wise growth, histogram binning | Large datasets (>10K samples) |
| **CatBoost** | Ordered target encoding for categoricals | Categorical features, less tuning |

---

#### AdaBoost (Adaptive Boosting)
**Purpose:** Boost weak learners by focusing on misclassified samples

**Algorithm:**
```
1. Initialize equal weights for all samples
2. For each weak learner:
   - Train on weighted data
   - Calculate error rate
   - Increase weights of misclassified samples
   - Decrease weights of correct samples
3. Final prediction: Weighted vote of all learners
```

**Difference from Gradient Boosting:**
- AdaBoost: Adjusts sample weights
- GBM: Fits to residuals

---

#### Graph-based Methods

**Graph Convolutional Networks (GCN):**
- Operate on graph structures (social networks, molecules)
- Use convolution on graph nodes
- Applications: Node classification, link prediction

**Hierarchical Clustering:**
- Tree structure (dendrogram) of data clusters
- Agglomerative (bottom-up) or Divisive (top-down)
- No need to specify k upfront

---

## 3. Probabilistic Methods

Algorithms based on Bayes' theorem and probability distributions.

### Algorithms Overview

| **Algorithm** | **Type** | **Assumption** | **When to Use** |
|--------------|---------|---------------|----------------|
| **Naive Bayes** | Classification | Feature independence | Text classification, fast baseline |
| **Gaussian Mixture Models** | Clustering | Gaussian distributions | Soft clustering, density estimation |
| **Hidden Markov Models** | Sequence | Markov property | Speech recognition, time series |
| **Bayesian Networks** | Various | Conditional independence | Causal reasoning, medical diagnosis |

### Detailed Algorithms

#### Naive Bayes
**Purpose:** Fast probabilistic classifier based on Bayes' theorem

**Bayes' Theorem:**
```
P(class|features) = P(features|class) × P(class) / P(features)
```

**"Naïve" Assumption:**
Features are conditionally independent given the class:
```
P(x₁,x₂,...,xₚ|class) = P(x₁|class) × P(x₂|class) × ... × P(xₚ|class)
```

**Variants:**

| **Type** | **Feature Distribution** | **Use Case** |
|----------|------------------------|-------------|
| **Gaussian NB** | Continuous (Gaussian) | Real-valued features |
| **Multinomial NB** | Discrete counts | Text classification (word counts) |
| **Bernoulli NB** | Binary | Binary features (word presence/absence) |

**Advantages:**
- Extremely fast training and prediction
- Works well with small data
- Handles high dimensions well
- Performs surprisingly well despite "naïve" assumption

**Applications:**
- Spam detection
- Sentiment analysis
- Document classification

---

#### Gaussian Mixture Models (GMM)
**Purpose:** Probabilistic clustering assuming data from mixture of Gaussians

**Key Concept:**
- Each cluster is a Gaussian distribution
- Data point has probability of belonging to each cluster (soft clustering)

**EM Algorithm:**
1. **E-step:** Estimate cluster membership probabilities
2. **M-step:** Update Gaussian parameters (mean, covariance)
3. Repeat until convergence

**vs K-Means:**
- GMM: Soft clustering (probabilities), elliptical clusters
- K-Means: Hard clustering, spherical clusters

---

#### Hidden Markov Models (HMM)
**Purpose:** Model sequential data with hidden states

**Components:**
- States: Hidden variables
- Observations: Visible outputs
- Transition probabilities: P(state_t | state_t-1)
- Emission probabilities: P(observation | state)

**Applications:**
- Speech recognition
- Part-of-speech tagging
- Gene sequence analysis

---

## Algorithm Selection Decision Tree

```
Start
  |
  ├─ Need interpretability? ──Yes─> Decision Tree, Linear Regression, Logistic Regression
  |                          
  ├─ Have labels?
  │   |
  │   ├─ Yes (Supervised)
  │   │   |
  │   │   ├─ Continuous output? ──Yes─> Regression
  │   │   │                           ├─ Linear relationship? ──Yes─> Linear Regression
  │   │   │                           ├─ No? ─> Random Forest → XGBoost
  │   │   │
  │   │   └─ Categorical output? ──Yes─> Classification
  │   │                               ├─ Need probabilities? ──Yes─> Logistic Regression
  │   │                               ├─ Text data? ──Yes─> Naive Bayes
  │   │                               ├─ <10K samples? ──Yes─> SVM
  │   │                               └─ Tabular data? ──Yes─> Random Forest → XGBoost
  │   │
  │   └─ No (Unsupervised)
  │       ├─ Find groups? ──Yes─> K-Means, GMM, Hierarchical
  │       ├─ Reduce dimensions? ──Yes─> PCA (unsupervised), LDA (supervised)
  │       └─ Find anomalies? ──Yes─> Isolation Forest, One-Class SVM
```

---

## Bias-Variance Trade-off (Enhanced)

One of the most critical concepts for ML interviews.

### Mathematical Decomposition

```
Expected Error = Bias² + Variance + Irreducible Error

E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²
```

### Definitions

**Bias:** Error from wrong assumptions
- High Bias = Underfitting (too simple)
- Example: Linear regression on polynomial data

**Variance:** Sensitivity to training data fluctuations  
- High Variance = Overfitting (too complex)
- Example: Deep decision tree memorizing noise

**Trade-off:**
- Complex models: ↓ Bias, ↑ Variance
- Simple models: ↑ Bias, ↓ Variance

### Algorithm Bias-Variance Characteristics

| **Algorithm** | **Bias** | **Variance** | **Fix Overfitting** | **Fix Underfitting** |
|--------------|---------|-------------|-------------------|---------------------|
| Linear Regression | High | Low | Add polynomial features | Use more complex model |
| Decision Tree (deep) | Low | High | Prune, limit depth | Already complex |
| Random Forest | Low | Low | More trees, max_depth | Increase tree depth |
| Boosting (GBM) | Low | Medium-High | Lower learning rate, early stopping | More iterations |
| KNN (small K) | Low | High | Increase K | Decrease K |
| SVM (RBF kernel) | Low | Medium | Increase C, decrease gamma | Decrease C, increase gamma |

### Practical Solutions

**Combat High Bias (Underfitting):**
1. Use more complex model
2. Add features (polynomial, interactions)
3. Reduce regularization (decrease λ)
4. Train longer (deep learning)
5. Remove noise from target variable

**Combat High Variance (Overfitting):**
1. Get more training data
2. Reduce model complexity
3. Feature selection / dimensionality reduction
4. Increase regularization (L1/L2)
5. Cross-validation
6. Ensemble methods (Random Forest)
7. Dropout (neural networks)
8. Early stopping

---

## Interview Quick Reference

### Common Questions

**1. "Explain overfitting vs underfitting"**
> Overfitting: Model learns training data too well, including noise. High variance, low bias. Good on train, poor on test.
> Underfitting: Model too simple to capture patterns. High bias, low variance. Poor on both train and test.

**2. "When would you use Random Forest vs XGBoost?"**
> Random Forest: Robust baseline, less tuning, parallel training, less overfitting risk
> XGBoost: Maximum accuracy on tabular data, sequential, more hyperparameters, can overfit

**3. "Why is Naive Bayes 'naive'?"**
> Assumes features are conditionally independent given the class, which is rarely true in practice. Despite this, it works surprisingly well for text classification.

**
