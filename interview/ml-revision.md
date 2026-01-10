# ML Quick Revision Cheat Sheet: 40+ Concepts

---

##  1. Regression Models

| Algorithm | Type | Loss | Key Insight |
|-----------|------|------|-------------|
| **Linear Regression** | Parametric | MSE | Assumes linearity, sensitive to outliers |
| **Ridge (L2)** | Regularized | MSE + λ||w||² | Shrinks weights, handles multicollinearity |
| **Lasso (L1)** | Regularized | MSE + λ||w||₁ | Creates sparse weights, feature selection |
| **Elastic Net** | Regularized | MSE + L1 + L2 | Best of both worlds |
| **Polynomial Reg** | Parametric | MSE | Add x², x³, etc. High degree = overfit |

**Key Metrics:**
- **MSE**: Mean Squared Error. Penalizes large errors.
- **MAE**: Mean Absolute Error. Robust to outliers.
- **R²**: Variance explained. Range: 0 to 1.

---

##  2. Classification Models

| Algorithm | Boundary | Key Insight |
|-----------|----------|-------------|
| **Logistic Regression** | Linear | Outputs probabilities via sigmoid |
| **Naive Bayes** | Non-linear | Assumes feature independence |
| **KNN** | Non-linear | Lazy learner, O(n) inference |
| **SVM** | Linear (or kernel) | Maximizes margin, kernel trick |
| **Decision Tree** | Non-linear | Interpretable, prone to overfit |

**Key Metrics:**
- **Accuracy**: (TP+TN) / Total
- **Precision**: TP / (TP+FP) — "Of predicted positive, how many correct?"
- **Recall**: TP / (TP+FN) — "Of actual positive, how many caught?"
- **F1**: Harmonic mean of P and R
- **AUC-ROC**: Area under ROC curve

---

##  3. Ensemble Methods

| Method | Technique | Goal |
|--------|-----------|------|
| **Random Forest** | Bagging | Reduce variance |
| **XGBoost/GBM** | Boosting | Reduce bias |
| **Stacking** | Meta-learning | Maximize accuracy |

**Key Concepts:**
- **Bagging**: Parallel, random samples, average predictions
- **Boosting**: Sequential, fix errors of previous models
- **Feature Subsampling**: Random Forest uses √p features per split

---

## 🧠 4. Neural Network Fundamentals

| Component | Purpose |
|-----------|---------|
| **Activation** | Introduce non-linearity |
| **Backprop** | Compute gradients via chain rule |
| **Optimizer** | Update weights (SGD, Adam) |
| **Regularization** | Prevent overfitting (Dropout, L2) |

**Activations:**
- **ReLU**: max(0, x). Default choice. Avoids vanishing gradient.
- **Sigmoid**: Maps to (0, 1). Output layer for binary.
- **Softmax**: Maps to probability distribution. Output for multi-class.

**Common Issues:**
- **Vanishing Gradient**: Use ReLU, BatchNorm, ResNets
- **Exploding Gradient**: Use gradient clipping

---

##  5. Key Formulas

| Concept | Formula |
|---------|---------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ |
| **Cross-Entropy** | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ |
| **Attention** | $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ |
| **Bias-Variance** | Error = Bias² + Variance + Noise |
| **Bayes** | $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ |

---

##  6. Unsupervised Learning

| Method | Goal | Key Metric |
|--------|------|------------|
| **K-Means** | Cluster by centroids | Inertia / Silhouette |
| **DBSCAN** | Density-based clusters | Handles noise |
| **PCA** | Reduce dimensions | Preserve variance |
| **t-SNE** | Visualization | Preserve local structure |

**PCA Key:**
- Find eigenvectors of covariance matrix
- Keep top k by eigenvalue
- Orthogonal components, maximize variance

---

##  7. Deep Learning Architectures

| Architecture | Input | Use Case |
|--------------|-------|----------|
| **CNN** | Grid (images) | Image classification, detection |
| **RNN/LSTM** | Sequence | Time series, legacy NLP |
| **Transformer** | Sequence | NLP, Vision, Multimodal |
| **GAN** | Noise | Image generation |
| **VAE** | Data | Generation, anomaly detection |
| **ResNet** | Images | Deep image classification |
| **U-Net** | Images | Segmentation |

---

##  8. Common Trade-offs

| Trade-off | Low | High |
|-----------|-----|------|
| **Bias-Variance** | Simple model (underfit) | Complex model (overfit) |
| **Precision-Recall** | High recall (catch all) | High precision (few errors) |
| **Latency-Accuracy** | Fast, simple model | Slow, accurate model |
| **Exploration-Exploitation** | Use known best | Try new options |

---

##  9. MLOps Quick Reference

| Concept | Definition |
|---------|------------|
| **Data Drift** | Input distribution changes |
| **Concept Drift** | Feature-target relationship changes |
| **Train-Serve Skew** | Training vs serving discrepancy |
| **Feature Store** | Centralized feature management |
| **A/B Testing** | Compare models on live traffic |
| **Canary Deployment** | Gradual rollout to subset |
| **Shadow Mode** | Test in parallel without affecting users |

---

##  10. Quick Fire Recall

**L1 vs L2?**
> L1 creates sparsity (zeros). L2 shrinks all weights.

**Generative vs Discriminative?**
> Generative: P(X, Y). Discriminative: P(Y|X).

**Parametric vs Non-parametric?**
> Parametric: Fixed form (LR). Non-parametric: Grows with data (KNN).

**Batch vs Layer Norm?**
> Batch: Across batch. Layer: Across features. Use Layer for Transformers.

**Encoder vs Decoder?**
> Encoder: Bidirectional, understanding. Decoder: Causal, generation.

**BERT vs GPT?**
> BERT: Encoder, masked LM. GPT: Decoder, autoregressive.

**Bagging vs Boosting?**
> Bagging: Parallel, reduce variance. Boosting: Sequential, reduce bias.

**Precision vs Recall?**
> Precision: Quality of positives. Recall: Completeness of positives.

**ROC vs PR curve?**
> ROC: Balanced data. PR: Imbalanced data.

**Why cross-validate?**
> More robust estimate. Uses all data for training and validation.
