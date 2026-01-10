# ML Glossary

Comprehensive A-Z glossary of machine learning terms with definitions, formulas, and examples.

---

## A

**Accuracy**
- **Definition:** Proportion of correct predictions out of total predictions
- **Formula:** `(TP + TN) / (TP + TN + FP + FN)`
- **Use Case:** Classification with balanced classes
- **Warning:** Misleading for imbalanced datasets

**Activation Function**
- **Definition:** Non-linear function applied to neuron outputs
- **Examples:** ReLU, Sigmoid, Tanh, GELU
- **Purpose:** Introduces non-linearity to learn complex patterns

**AdaBoost (Adaptive Boosting)**
- **Definition:** Ensemble method that combines weak learners sequentially
- **How:** Increases weights of misclassified samples
- **Use Case:** Binary classification

**Adam (Adaptive Moment Estimation)**
- **Definition:** Optimization algorithm combining momentum and RMSprop
- **Formula:** Adapts learning rate per parameter based on 1st and 2nd moments
- **Default Choice:** Most popular optimizer for deep learning

**AUC (Area Under Curve)**
- **Definition:** Area under the ROC curve
- **Range:** [0, 1], where 0.5 = random, 1 = perfect
- **Use Case:** Binary classification performance evaluation

**Autoencoder**
- **Definition:** Neural network that learns compressed representations
- **Structure:** Encoder (compress) + Decoder (reconstruct)
- **Use Case:** Dimensionality reduction, denoising, anomaly detection

---

## B

**Backpropagation**
- **Definition:** Algorithm to compute gradients using chain rule
- **Process:** Forward pass → compute loss → backward pass → update weights
- **Key:** Enables training of deep neural networks

**Bagging (Bootstrap Aggregating)**
- **Definition:** Ensemble method training models on random subsets
- **Example:** Random Forest
- **Effect:** Reduces variance

**Batch Normalization**
- **Definition:** Normalizes layer inputs across mini-batch
- **Benefits:** Faster training, higher learning rates, regularization
- **Formula:** `(x - μ) / √(σ² + ε)`

**Batch Size**
- **Definition:** Number of samples processed before updating weights
- **Trade-off:** Larger = faster training, more stable, but more memory
- **Common Values:** 32, 64, 128, 256

**Bias (Model)**
- **Definition:** Error from wrong assumptions in the learning algorithm
- **High Bias:** Underfitting (model too simple)
- **Low Bias:** Model captures true patterns

**Bias (Parameter)**
- **Definition:** Constant term in linear models (intercept)
- **Formula:** `y = wx + b` (b is bias)
- **Purpose:** Shifts decision boundary

**Bias-Variance Trade-off**
- **Definition:** Balance between bias and variance to minimize total error
- **Formula:** `Error = Bias² + Variance + Irreducible Error`
- **Goal:** Find sweet spot between underfitting and overfitting

**Binary Cross-Entropy**
- **Definition:** Loss function for binary classification
- **Formula:** `-[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- **Use Case:** Logistic regression, binary neural networks

**Boosting**
- **Definition:** Sequential ensemble where each model corrects previous errors
- **Examples:** AdaBoost, Gradient Boosting, XGBoost
- **Effect:** Reduces bias

---

## C

**Categorical Cross-Entropy**
- **Definition:** Loss function for multi-class classification
- **Formula:** `-Σ y_i · log(ŷ_i)`
- **Use Case:** Multi-class neural networks with softmax output

**CNN (Convolutional Neural Network)**
- **Definition:** Neural network using convolution operations
- **Components:** Conv layers, pooling, fully connected
- **Use Case:** Image classification, object detection

**Confusion Matrix**
- **Definition:** Table showing TP, FP, FN, TN
- **Purpose:** Detailed classification performance breakdown
- **Metrics Derived:** Precision, recall, F1, accuracy

**Convolution**
- **Definition:** Operation that slides a filter over input
- **Purpose:** Extract local features (edges, textures)
- **Output:** Feature maps

**Cross-Validation**
- **Definition:** Splitting data into K folds for robust evaluation
- **Types:** K-fold, stratified K-fold, time-series split
- **Purpose:** Better performance estimation, prevent overfitting

---

## D

**Data Augmentation**
- **Definition:** Artificially expanding dataset with transformations
- **Vision:** Rotation, flipping, cropping, color jitter
- **NLP:** Back-translation, synonym replacement
- **Benefit:** Reduces overfitting

**Decision Tree**
- **Definition:** Tree structure for classification/regression
- **Splitting:** Gini impurity, entropy, MSE
- **Pros:** Interpretable, no scaling needed
- **Cons:** Prone to overfitting

**Dimensionality Reduction**
- **Definition:** Reducing number of features
- **Methods:** PCA, t-SNE, UMAP, LDA
- **Purpose:** Visualization, speed, removing noise

**Dropout**
- **Definition:** Randomly deactivate neurons during training
- **Rate:** Typically 0.2-0.5
- **Effect:** Prevents overfitting, forces robust features

---

## E

**Early Stopping**
- **Definition:** Stop training when validation performance plateaus
- **Implementation:** Monitor validation loss, stop after N epochs without improvement
- **Benefit:** Prevents overfitting

**Embedding**
- **Definition:** Dense vector representation of discrete items
- **Example:** Word2Vec (words → vectors), entity embeddings (categories → vectors)
- **Dimension:** Typically 50-300 for words, 8-50 for categories

**Ensemble Methods**
- **Definition:** Combining multiple models for better performance
- **Types:** Bagging, boosting, stacking
- **Benefit:** Reduces overfitting, improves accuracy

**Epoch**
- **Definition:** One complete pass through entire training dataset
- **Training:** Typically requires 10-100 epochs
- **Stop When:** Validation loss stops improving

**Exploding Gradient**
- **Definition:** Gradients become very large during backprop
- **Causes:** Deep networks, especially RNNs
- **Solutions:** Gradient clipping, lower learning rate, batch norm

---

## F

**F1 Score**
- **Definition:** Harmonic mean of precision and recall
- **Formula:** `2 · (P · R) / (P + R)`
- **Use Case:** Imbalanced datasets, balance precision/recall
- **Range:** [0, 1], higher is better

**False Negative (FN)**
- **Definition:** Actual positive predicted as negative
- **Example:** Sick patient classified as healthy
- **Cost:** Depends on domain (very high in medical diagnosis)

**False Positive (FP)**
- **Definition:** Actual negative predicted as positive
- **Example:** Healthy patient classified as sick
- **Cost:** Depends on domain (high in spam filtering)

**Feature Engineering**
- **Definition:** Creating new features from existing data
- **Techniques:** Polynomial features, interactions, domain-specific transforms
- **Impact:** Often more important than algorithm choice

**Feature Scaling**
- **Definition:** Normalizing feature ranges
- **Methods:** Standardization (z-score), min-max normalization
- **When:** Required for gradient descent-based algorithms

**Fine-tuning**
- **Definition:** Training pretrained model on new task
- **Strategy:** Freeze base → train head → unfreeze → train end-to-end
- **Learning Rate:** Very low (1e-5 to 1e-6)

---

##G

**GAN (Generative Adversarial Network)**
- **Definition:** Two networks (generator vs discriminator) in adversarial training
- **Purpose:** Generate realistic data
- **Applications:** Image generation, style transfer

**Generalization**
- **Definition:** Model's ability to perform well on unseen data
- **Good:** Train and test performance similar
- **Poor:** High train accuracy, low test accuracy (overfitting)

**Gradient Descent**
- **Definition:** Optimization algorithm minimizing loss by following gradients
- **Variants:** SGD, mini-batch GD, batch GD
- **Formula:** `w = w - η · ∂L/∂w`

**Gradient Clipping**
- **Definition:** Limiting gradient magnitude to prevent exploding gradients
- **Methods:** Clip by value, clip by norm
- **Critical For:** RNN training

**Grid Search**
- **Definition:** Exhaustive hyperparameter search over defined grid
- **Alternative:** Random search (often better)
- **Use With:** Cross-validation

---

## H

**Hyperparameter**
- **Definition:** Parameters set before training (not learned)
- **Examples:** Learning rate, batch size, number of layers, regularization strength
- **Tuning:** Grid search, random search, Bayesian optimization

**Hyperplane**
- **Definition:** Decision boundary in n-dimensional space
- **Example:** Line (2D), plane (3D), hyperplane (n-D)
- **Used In:** SVM, linear/logistic regression

---

## I

**Imbalanced Data**
- **Definition:** Unequal class distribution (e.g., 95% negative, 5% positive)
- **Solutions:** SMOTE, class weights, undersampling
- **Metrics:** Precision, recall, F1, PR-AUC (not accuracy!)

**Interpretability**
- **Definition:** Understanding how model makes decisions
- **High:** Linear models, decision trees
- **Low:** Deep neural networks (black box)
- **Techniques:** SHAP, LIME, feature importance

---

## K

**K-Fold Cross-Validation**
- **Definition:** Split data into K parts, train K times
- **Process:** Each fold used as validation once
- **Common K:** 5 or 10
- **Variant:** Stratified K-fold (preserves class distribution)

**K-Means Clustering**
- **Definition:** Partitioning algorithm finding K centroids
- **Algorithm:** Initialize centroids → assign points → update centroids → repeat
- **Drawback:** Must specify K, assumes spherical clusters

**KNN (K-Nearest Neighbors)**
- **Definition:** Classify based on K closest training examples
- **Lazy Learning:** No training, stores all data
- **Drawback:** Slow prediction, curse of dimensionality

**Kernel Trick**
- **Definition:** Implicitly map data to higher dimension
- **Used In:** SVM for non-linear separation
- **Common Kernels:** RBF, polynomial, sigmoid

---

## L

**L1 Regularization (Lasso)**
- **Definition:** Adds sum of absolute weights to loss
- **Formula:** `Loss + λ Σ|w|`
- **Effect:** Sparse models (some weights → 0)
- **Use:** Feature selection

**L2 Regularization (Ridge)**
- **Definition:** Adds sum of squared weights to loss
- **Formula:** `Loss + λ Σw²`
- **Effect:** Shrinks weights, no zeros
- **Use:** Prevent overfitting with correlated features

**Learning Rate**
- **Definition:** Step size in gradient descent
- **Range:** Typically 0.001 to 0.1
- **Too High:** Unstable training, divergence
- **Too Low:** Slow convergence

**Logistic Regression**
- **Definition:** Linear model for classification using sigmoid
- **Output:** Probabilities [0,1]
- **Loss:** Binary cross-entropy
- **Despite Name:** CLASSIFICATION, not regression

**Loss Function**
- **Definition:** Measures error between predictions and truth
- **Examples:** MSE (regression), cross-entropy (classification)
- **Training:** Minimized via gradient descent

**LSTM (Long Short-Term Memory)**
- **Definition:** RNN variant with gates to control information flow
- **Components:** Forget gate, input gate, output gate
- **Purpose:** Capture long-range dependencies in sequences

---

## M

**MAE (Mean Absolute Error)**
- **Definition:** Average absolute difference between predictions and actuals
- **Formula:** `(1/n) Σ|y - ŷ|`
- **Robust to:** Outliers (vs MSE)

**Metric**
- **Definition:** Function measuring model performance
- **Classification:** Accuracy, precision, recall, F1, AUC
- **Regression:** MAE, MSE, RMSE, R²

**MSE (Mean Squared Error)**
- **Definition:** Average squared difference
- **Formula:** `(1/n) Σ(y - ŷ)²`
- **Sensitive to:** Outliers (quadratic penalty)

---

## N

**Naive Bayes**
- **Definition:** Probabilistic classifier based on Bayes' theorem
- **Assumption:** Feature independence (naïve)
- **Fast:** Excellent for text classification

**Normalization**
- **Definition:** Scaling features to standard range
- **Methods:** Min-max [0,1], z-score (mean=0, std=1)
- **When:** Gradient descent-based algorithms

---

## O

**Overfitting**
- **Definition:** Model learns training data too well, including noise
- **Signs:** High train accuracy, low test accuracy
- **Solutions:** More data, regularization, simpler model, cross-validation

**Optimizer**
- **Definition:** Algorithm updating weights to minimize loss
- **Examples:** SGD, Adam, RMSprop
- **Choice:** Adam for most DL, SGD+momentum for fine-tuning

---

## P

**PCA (Principal Component Analysis)**
- **Definition:** Dimensionality reduction preserving variance
- **Unsupervised:** Doesn't use labels
- **Use:** Visualization, speed up training, remove collinearity

**Precision**
- **Definition:** Of predicted positives, how many are correct?
- **Formula:** `TP / (TP + FP)`
- **High When:** Minimizing false positives is critical (spam filter)

**Pretrained Model**
- **Definition:** Model trained on large dataset, reusable for other tasks
- **Examples:** BERT (NLP), ResNet (vision)
 **Benefit:** Transfer learning, less data needed

**Pooling**
- **Definition:** Downsampling operation in CNNs
- **Types:** Max pooling, average pooling
- **Purpose:** Reduce spatial dimensions, translation invariance

---

## R

**R² (R-Squared)**
- **Definition:** Proportion of variance explained by model
- **Formula:** `1 - (SS_res / SS_tot)`
- **Range:** (-∞, 1], 1 = perfect fit
- **Can Be Negative:** If model worse than mean baseline

**Random Forest**
- **Definition:** Ensemble of decision trees using bagging
- **Process:** Bootstrap samples + random feature subsets
- **Robust:** Less overfitting than single tree

**Recall (Sensitivity)**
- **Definition:** Of actual positives, how many are caught?
- **Formula:** `TP / (TP + FN)`
- **High When:** Minimizing false negatives is critical (disease screening)

**Regularization**
- **Definition:** Technique to prevent overfitting by penalizing complexity
- **Types:** L1, L2, dropout, early stopping
- **Trade-off:** Bias-variance balance

**ReLU (Rectified Linear Unit)**
- **Definition:** Activation function f(x) = max(0, x)
- **Advantages:** Fast, no vanishing gradient
- **Disadvantage:** Dead ReLU problem

**ResNet (Residual Network)**
- **Definition:** CNN architecture with skip connections
- **Innovation:** Solves vanishing gradient, enables very deep networks (152 layers)
- **Formula:** `output = F(x) + x`

**RMSE (Root Mean Squared Error)**
- **Definition:** Square root of MSE
- **Formula:** `√[(1/n) Σ(y - ŷ)²]`
- **Benefit:** Same units as target variable

**ROC Curve (Receiver Operating Characteristic)**
- **Definition:** Plot of TPR vs FPR at various thresholds
- **AUC:** Area under ROC curve (performance metric)
- **Use:** Binary classification evaluation

---

## S

**Sigmoid Function**
- **Definition:** S-shaped activation function
- **Formula:** `σ(x) = 1 / (1 + e^(-x))`
- **Output:** (0, 1) - probabilities
- **Use:** Logistic regression, binary classification output

**Softmax Function**
- **Definition:** Converts logits to probability distribution
- **Formula:** `softmax(x_i) = e^(x_i) / Σe^(x_j)`
- **Output:** Sums to 1
- **Use:** Multi-class classification output

**Stochastic Gradient Descent (SGD)**
- **Definition:** Gradient descent using single or mini-batch samples
- **Variants:** SGD, SGD + momentum, SGD + Nesterov
- **Stochastic:** Introduces noise, helps escape local minima

**Stratification**
- **Definition:** Preserving class distribution in train/test split
- **When:** Imbalanced datasets
- **Implementation:** `stratify=y` in sklearn

**SVM (Support Vector Machine)**
- **Definition:** Finds optimal hyperplane maximizing margin
- **Kernel Trick:** Handles non-linear separation
- **Good For:** Small-medium datasets, high dimensions

---

## T

**Transfer Learning**
- **Definition:** Reusing pretrained model for new task
- **Process:** Load pretrained → freeze base → train head → fine-tune
- **Benefit:** Less data, faster training, better performance

**Transformer**
- **Definition:** Architecture using self-attention mechanism
- **Innovation:** Parallelizable, captures long-range dependencies
- **Examples:** BERT, GPT, T5

**True Negative (TN)**
- **Definition:** Actual negative correctly predicted as negative

**True Positive (TP)**
- **Definition:** Actual positive correctly predicted as positive

---

## U

**Underfitting**
- **Definition:** Model too simple to capture patterns
- **Signs:** Low train AND test accuracy
- **Solutions:** More complex model, more features, less regularization

---

## V

**Validation Set**
- **Definition:** Data used for hyperparameter tuning and model selection
- **Split:** Typically 10-20% of data
- **Critical:** Must be separate from test set

**Vanishing Gradient**
- **Definition:** Gradients become very small in early layers
- **Causes:** Deep networks with sigmoid/tanh
- **Solutions:** ReLU, batch norm, ResNet skip connections

**Variance (Model)**
- **Definition:** Error from sensitivity to training data fluctuations
- **High Variance:** Overfitting (model too complex)
- **Low Variance:** Consistent predictions across datasets

---

## W

**Weight Decay**
- **Definition:** L2 regularization in optimization
- **Implementation:** Gradually shrink weights during training
- **Effect:** Prevents overfitting

**Weight Initialization**
- **Definition:** Setting initial weights before training
- **Methods:** Xavier/Glorot (tanh), He (ReLU)
- **Important:** Affects convergence speed and stability

---

## X

**XGBoost (Extreme Gradient Boosting)**
- **Definition:** Optimized gradient boosting implementation
- **Features:** Regularization, parallel processing, custom objectives
- **Best For:** Tabular data, Kaggle competitions

---

## Z

**Z-score Normalization**
- **Definition:** Standardization to mean=0, std=1
- **Formula:** `z = (x - μ) / σ`
- **When:** Features on different scales

---

## Interview Tips

**Most Important Terms to Know Cold:**
1. Bias-Variance Trade-off
2. Overfitting vs Underfitting
3. Precision vs Recall
4. Cross-Validation
5. Regularization (L1/L2)
6. Gradient Descent
7. Backpropagation
8. Activation Functions
9. Loss Functions
10. Ensemble Methods

**Be Ready to:**
- Define any term concisely
- Provide formula (if applicable)
- Give practical example
- Explain when to use
- Compare with related concepts
