---
module: Foundations
topic: Flashcards
status: unread
tags: [foundations, flashcards, active-recall]
---
# Machine Learning Foundations Flashcards

Use these for active recall and spaced repetition (e.g., Anki, Obsidian).

## General ML & Architecture

- **Bias**: Error from wrong assumptions (underfitting). #flashcard
- **Variance**: Error from sensitivity to small fluctuations in training data (overfitting). #flashcard
- **L1 vs L2**: L1 (Lasso) creates sparsity (feature selection), L2 (Ridge) shrinks weights smoothly. #flashcard
- **Precision**: TP / (TP + FP). Out of all predicted positives, how many were real? #flashcard
- **Recall**: TP / (TP + FN). Out of all actual positives, how many did we catch? #flashcard
- **Bagging**: Parallel training on subsets to reduce variance (Random Forest). #flashcard
- **Boosting**: Sequential training correcting previous errors to reduce bias (XGBoost). #flashcard
- **Epoch**: One complete pass through the entire training dataset. #flashcard
- **Batch Size**: Number of samples processed before updating weights. #flashcard
- **Data Parallelism vs Tensor Parallelism**: DP splits the batch across GPUs (each gets full model); TP splits the model weights across GPUs. #flashcard
- **KV-Cache**: Caching past Key and Value tensors during autoregressive generation to avoid recalculating them. #flashcard

## Deep Learning & Optimization

- **Vanishing Gradients**: Gradients exponentially shrink as they backpropagate, preventing early layers from learning. Fix: ReLU, ResNet, BatchNorm. #flashcard
- **Exploding Gradients**: Gradients exponentially grow, destroying weights. Fix: Gradient Clipping. #flashcard
- **Adam vs SGD**: Adam adapts the learning rate per parameter using momentum and RMS; SGD applies a global learning rate. #flashcard
- **Batch Normalization**: Normalizes activations across the batch dimension. Best for CNNs. #flashcard
- **Layer Normalization**: Normalizes activations across the feature dimension. Best for Transformers. #flashcard
- **Transformer Scaling Factor**: Dividing the QK dot product by `√d_k` prevents large values from pushing the softmax into regions with zero gradient. #flashcard

## Math & Linear Algebra

- **Vector**: A 1D array of numbers, representing a point in space or magnitude/direction. #flashcard
- **Matrix**: A 2D array of numbers, representing a linear transformation. #flashcard
- **Eigenvector**: A vector that only scales (doesn't rotate) when a specific linear transformation is applied. #flashcard
- **Eigenvalue**: The scale factor by which an eigenvector is scaled. #flashcard
- **SVD**: Decomposing any matrix into U (left singular vectors), Sigma (singular values), and V (right singular vectors). #flashcard
- **Condition Number**: Ratio of largest to smallest singular value. A high condition number means the matrix is unstable/ill-conditioned. #flashcard
- **Hessian**: The matrix of second-order partial derivatives. Positive-definite means you are at a local minimum. #flashcard
- **KL Divergence**: A non-symmetric measure of the difference between two probability distributions. #flashcard
- **Maximum Likelihood Estimation (MLE)**: Finding parameters that make the observed data most probable. Equivalent to MSE (Gaussian) or Cross-Entropy (Categorical). #flashcard
- **Maximum A Posteriori (MAP)**: MLE but incorporating prior beliefs about the parameters. Equivalent to Regularization. #flashcard

## Activations & Losses

- **ReLU**: max(0, x). Computationally trivial, avoids saturation in the positive domain. Default for hidden layers. #flashcard
- **Sigmoid**: 1/(1+e⁻ˣ). Squashes to (0,1); used for binary output layers. Saturates at extremes, causing vanishing gradients in deep stacks. #flashcard
- **Softmax**: eˣⁱ/Σeˣʲ. Converts a score vector into probabilities summing to 1. Used at the final layer of multi-class classifiers. #flashcard
- **GELU**: x·Φ(x). Smooth approximation of ReLU; empirically stronger in transformers. #flashcard
- **MAE (Mean Absolute Error)**: (1/n)Σ|y-ŷ|. Linear penalty; outliers do not dominate. Not differentiable at 0 (but subdifferentiable). #flashcard
- **RMSE (Root Mean Squared Error)**: √MSE. Same units as the target variable; penalizes large errors like MSE. #flashcard
- **Elastic Net**: Loss + α₁·Σ|w| + α₂·Σw². Combines L1 and L2; handles correlated features better than pure L1. #flashcard

## Confusion Matrix & Validation

- **TP / TN / FP / FN**: Correct-positive (hit), correct-negative (correct rejection), false alarm (Type I error), miss (Type II error). #flashcard
- **Stratified K-Fold**: Each fold preserves the original class distribution. Required for classification on imbalanced data. #flashcard
- **Time Series Split (Walk-Forward)**: Validation folds are always in the future relative to training folds — never shuffle time series before splitting. #flashcard
- **Leave-One-Out (LOO)**: K = N. Maximum data usage; very expensive; high-variance estimates on noisy problems. #flashcard

## Data Leakage Red Flags

- **Target leakage**: Including the target variable or a near-synonym as a feature. #flashcard
- **Preprocessing leakage**: Fitting the scaler/imputer on the full dataset before splitting — validation statistics leak into preprocessing. #flashcard
- **Temporal leakage**: Using features computed with future timestamps (e.g. a "7-day rolling average" computed from the future). #flashcard
- **ID leakage**: Encoding an ID column that correlates with cohort or time. #flashcard

## Dimensionality Reduction

- **PCA**: Linear; finds axes of maximum variance, projects to lower dimensions preserving variance. Sensitive to scale — standardize first. #flashcard
- **t-SNE**: Non-linear; preserves local neighborhood structure, good for visualizing clusters. Does not preserve global distances — cluster positions are arbitrary. #flashcard
- **UMAP**: Non-linear; faster than t-SNE, better preserves both local and global structure. Better for downstream tasks. #flashcard
- **LDA**: Supervised; maximizes class separability rather than variance. Better for classification problems. #flashcard

## Fine-Tuning & Optimization Strategy

- **Full fine-tuning**: Update all weights. Expensive but maximally flexible. #flashcard
- **Head-only fine-tuning**: Freeze the pre-trained backbone, train only a new task-specific output layer. Cheap; works when the pre-training domain is close. #flashcard
- **PEFT / LoRA**: Add small trainable adapter layers while freezing the base model. Reduces trainable parameters by orders of magnitude. #flashcard
- **Batch GD vs SGD vs Mini-batch**: Batch GD computes gradient over the entire dataset (exact, slow); SGD uses one example (noisy, fast, escapes shallow minima); mini-batch (32–512) balances both. #flashcard
- **Grid vs Random vs Bayesian Search**: Grid is exhaustive but combinatorially expensive; random search wastes less budget on unimportant dimensions; Bayesian optimization builds a surrogate model to pick the next promising point. #flashcard
- **Warm-up / Cosine Decay / Linear Decay**: Warm-up ramps LR up from small to prevent early instability; cosine decay smoothly reduces LR on a cosine curve; linear decay is simpler and works well for LM fine-tuning. #flashcard

## Imbalanced Data

- **Resampling**: SMOTE synthesizes new minority examples by interpolation; undersampling randomly removes majority examples. #flashcard
- **Class weights**: `class_weight='balanced'` penalizes minority misclassifications proportionally more — mathematically equivalent to oversampling without duplicating data. #flashcard
- **Threshold adjustment**: Shift the decision threshold below 0.5 to increase recall for the minority class. #flashcard
- **Metric choice**: Use F1, PR-AUC, or MCC — not accuracy. #flashcard

## Clustering (K-Means Failure Modes)

- **Requires K in advance**: Use the Elbow Method or Silhouette Score to choose it. #flashcard
- **Assumes spherical, equal-size clusters**: Fails on elongated, concentric, or very different-sized clusters. #flashcard
- **Sensitive to outliers**: A single outlier can pull a centroid far from the true cluster. #flashcard
- **Random initialization**: Can trap the algorithm at poor local minima — use k-means++ initialization. #flashcard

## Gradient Health & Regularization Fixes

- **Fixes for vanishing/exploding gradients**: ReLU (gradient exactly 1 in positive domain, no attenuation), residual connections (gradients flow through addition, not multiplication), batch/layer normalization (keeps activations well-conditioned), careful initialization (Xavier for sigmoid/tanh, He for ReLU). #flashcard
- **Bias-variance fixes**: more training data (reduces variance directly), regularization (L1/L2, dropout, weight decay), simpler architecture, early stopping, data augmentation. #flashcard

## Causal Inference

- **RCT / A/B test**: Randomization breaks confounding. The gold standard when feasible. #flashcard
- **Difference-in-Differences (DiD)**: Compare treatment and control groups before and after treatment. Controls for time-invariant confounders. #flashcard
- **Propensity Score Matching**: Match treated and untreated units on observed covariates to approximate randomization. #flashcard
- **Instrumental Variables**: Use a variable that affects treatment but has no direct effect on outcome to isolate the causal effect. #flashcard

## RAG Evaluation

- **Faithfulness**: Does the answer only claim what the retrieved context supports? #flashcard
- **Answer Relevancy**: Does the answer actually address the question? #flashcard
- **Context Precision**: Are the retrieved documents relevant to the question? #flashcard
- **Context Recall**: Are the retrieved documents sufficient to answer the question? #flashcard
