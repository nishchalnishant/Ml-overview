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
