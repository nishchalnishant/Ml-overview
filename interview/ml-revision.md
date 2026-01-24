# Quick Technical Revision

***

## 1. Regression Models

<table><thead><tr><th width="168.83160400390625">Algorithm</th><th width="123.0833740234375">Type</th><th width="133.040771484375">Loss</th><th>Key Insight</th></tr></thead><tbody><tr><td><strong>Linear Regression</strong></td><td>Parametric</td><td>MSE</td><td>Assumes linearity, sensitive to outliers</td></tr><tr><td><strong>Ridge (L2)</strong></td><td>Regularized</td><td>MSE + λ</td><td></td></tr><tr><td><strong>Lasso (L1)</strong></td><td>Regularized</td><td>MSE + λ</td><td></td></tr><tr><td><strong>Elastic Net</strong></td><td>Regularized</td><td>MSE + L1 + L2</td><td>Best of both worlds</td></tr><tr><td><strong>Polynomial Reg</strong></td><td>Parametric</td><td>MSE</td><td>Add x², x³, etc. High degree = overfit</td></tr></tbody></table>

**Key Metrics:**

* **MSE**: Mean Squared Error. Penalizes large errors.
* **MAE**: Mean Absolute Error. Robust to outliers.
* **R²**: Variance explained. Range: 0 to 1.

***

## 2. Classification Models

<table><thead><tr><th width="179.134521484375">Algorithm</th><th>Boundary</th><th>Key Insight</th></tr></thead><tbody><tr><td><strong>Logistic Regression</strong></td><td>Linear</td><td>Outputs probabilities via sigmoid</td></tr><tr><td><strong>Naive Bayes</strong></td><td>Non-linear</td><td>Assumes feature independence</td></tr><tr><td><strong>KNN</strong></td><td>Non-linear</td><td>Lazy learner, O(n) inference</td></tr><tr><td><strong>SVM</strong></td><td>Linear (or kernel)</td><td>Maximizes margin, kernel trick</td></tr><tr><td><strong>Decision Tree</strong></td><td>Non-linear</td><td>Interpretable, prone to overfit</td></tr></tbody></table>

**Key Metrics:**

* **Accuracy**: (TP+TN) / Total
* **Precision**: TP / (TP+FP) — "Of predicted positive, how many correct?"
* **Recall**: TP / (TP+FN) — "Of actual positive, how many caught?"
* **F1**: Harmonic mean of P and R
* **AUC-ROC**: Area under ROC curve

***

## 3. Ensemble Methods

| Method            | Technique     | Goal              |
| ----------------- | ------------- | ----------------- |
| **Random Forest** | Bagging       | Reduce variance   |
| **XGBoost/GBM**   | Boosting      | Reduce bias       |
| **Stacking**      | Meta-learning | Maximize accuracy |

**Key Concepts:**

* **Bagging**: Parallel, random samples, average predictions
* **Boosting**: Sequential, fix errors of previous models
* **Feature Subsampling**: Random Forest uses √p features per split

***

## 4. Neural Network Fundamentals

<table><thead><tr><th width="151.578125">Component</th><th>Purpose</th></tr></thead><tbody><tr><td><strong>Activation</strong></td><td>Introduce non-linearity</td></tr><tr><td><strong>Backprop</strong></td><td>Compute gradients via chain rule</td></tr><tr><td><strong>Optimizer</strong></td><td>Update weights (SGD, Adam)</td></tr><tr><td><strong>Regularization</strong></td><td>Prevent overfitting (Dropout, L2)</td></tr></tbody></table>

**Activations:**

* **ReLU**: max(0, x). Default choice. Avoids vanishing gradient.
* **Sigmoid**: Maps to (0, 1). Output layer for binary.
* **Softmax**: Maps to probability distribution. Output for multi-class.

**Common Issues:**

* **Vanishing Gradient**: Use ReLU, BatchNorm, ResNets
* **Exploding Gradient**: Use gradient clipping

***

## 5. Key Formulas

<table><thead><tr><th width="172.203125">Concept</th><th>Formula</th></tr></thead><tbody><tr><td><strong>Sigmoid</strong></td><td>$\sigma(z) = \frac{1}{1+e^{-z}}$</td></tr><tr><td><strong>Softmax</strong></td><td>$\frac{e^{z_i}}{\sum e^{z_j}}$</td></tr><tr><td><strong>Cross-Entropy</strong></td><td>$-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$</td></tr><tr><td><strong>MSE</strong></td><td>$\frac{1}{n}\sum(y-\hat{y})^2$</td></tr><tr><td><strong>Attention</strong></td><td>$Softmax(\frac{QK^T}{\sqrt{d_k}})V$</td></tr><tr><td><strong>Bias-Variance</strong></td><td>Error = Bias² + Variance + Noise</td></tr><tr><td><strong>Bayes</strong></td><td>$P(A</td></tr></tbody></table>

***

## 6. Unsupervised Learning

<table><thead><tr><th width="129.5989990234375">Method</th><th>Goal</th><th>Key Metric</th></tr></thead><tbody><tr><td><strong>K-Means</strong></td><td>Cluster by centroids</td><td>Inertia / Silhouette</td></tr><tr><td><strong>DBSCAN</strong></td><td>Density-based clusters</td><td>Handles noise</td></tr><tr><td><strong>PCA</strong></td><td>Reduce dimensions</td><td>Preserve variance</td></tr><tr><td><strong>t-SNE</strong></td><td>Visualization</td><td>Preserve local structure</td></tr></tbody></table>

**PCA Key:**

* Find eigenvectors of covariance matrix
* Keep top k by eigenvalue
* Orthogonal components, maximize variance

***

## 7. Deep Learning Architectures

<table><thead><tr><th width="136.10064697265625">Architecture</th><th width="164.51824951171875">Input</th><th>Use Case</th></tr></thead><tbody><tr><td><strong>CNN</strong></td><td>Grid (images)</td><td>Image classification, detection</td></tr><tr><td><strong>RNN/LSTM</strong></td><td>Sequence</td><td>Time series, legacy NLP</td></tr><tr><td><strong>Transformer</strong></td><td>Sequence</td><td>NLP, Vision, Multimodal</td></tr><tr><td><strong>GAN</strong></td><td>Noise</td><td>Image generation</td></tr><tr><td><strong>VAE</strong></td><td>Data</td><td>Generation, anomaly detection</td></tr><tr><td><strong>ResNet</strong></td><td>Images</td><td>Deep image classification</td></tr><tr><td><strong>U-Net</strong></td><td>Images</td><td>Segmentation</td></tr></tbody></table>

***

## 8. Common Trade-offs

| Trade-off                    | Low                     | High                        |
| ---------------------------- | ----------------------- | --------------------------- |
| **Bias-Variance**            | Simple model (underfit) | Complex model (overfit)     |
| **Precision-Recall**         | High recall (catch all) | High precision (few errors) |
| **Latency-Accuracy**         | Fast, simple model      | Slow, accurate model        |
| **Exploration-Exploitation** | Use known best          | Try new options             |

***

## 9. MLOps Quick Reference

<table><thead><tr><th width="199.4947509765625">Concept</th><th>Definition</th></tr></thead><tbody><tr><td><strong>Data Drift</strong></td><td>Input distribution changes</td></tr><tr><td><strong>Concept Drift</strong></td><td>Feature-target relationship changes</td></tr><tr><td><strong>Train-Serve Skew</strong></td><td>Training vs serving discrepancy</td></tr><tr><td><strong>Feature Store</strong></td><td>Centralized feature management</td></tr><tr><td><strong>A/B Testing</strong></td><td>Compare models on live traffic</td></tr><tr><td><strong>Canary Deployment</strong></td><td>Gradual rollout to subset</td></tr><tr><td><strong>Shadow Mode</strong></td><td>Test in parallel without affecting users</td></tr></tbody></table>

***

## 10. Quick Fire Recall

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
