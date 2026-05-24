---
module: Foundations
topic: Ml Glossary
subtopic: ""
status: unread
tags: [foundations, ml, ml-glossary]
---
# Machine Learning Glossary

**How to use this:** Every entry starts with the problem the term was invented to solve. If you understand the problem, the definition follows logically. Skim the letters you're rusty on before a review session. Dense by design — definitions, formulas, production failure modes, and interview sound bites.

---

## A

**Accuracy**

*The problem:* You need one number that summarizes how often a classifier is correct. Accuracy is that number — but it only works when mistakes on each class are equally costly and equally common.

*Formula:* `(TP + TN) / (TP + TN + FP + FN)`

*When it lies:* Any time classes are imbalanced. A model that always predicts "not fraud" on a 1-in-10,000 fraud dataset achieves 99.99% accuracy while being worthless. Always inspect the confusion matrix before trusting accuracy. For imbalanced problems, use F1 or PR-AUC.

---

**Activation Function**

*The problem:* A stack of linear transformations is itself a single linear transformation. No matter how many layers you add, the model can only learn linear decision boundaries — useless for images, speech, or most real patterns.

*The fix:* Insert a non-linear function after each linear layer. The composition of many linear + non-linear operations can approximate any continuous function.

*Key examples:*
- **ReLU:** `max(0, x)` — computationally trivial, avoids saturation in the positive domain. Default for hidden layers.
- **Sigmoid:** `1/(1+e⁻ˣ)` — squashes to (0,1); used for binary output layers. Saturates at extremes, causing vanishing gradients in deep stacks.
- **Softmax:** `eˣⁱ/Σeˣʲ` — converts a score vector into probabilities summing to 1. Used at the final layer of multi-class classifiers.
- **GELU:** `x·Φ(x)` — smooth approximation of ReLU; empirically stronger in transformers.

*Interview trap:* "Why not use sigmoid in every layer?" Because deep sigmoid stacks starve backprop — the sigmoid gradient has a maximum of 0.25, so multiplying through 50 layers gives gradients near zero.

---

**AdaBoost (Adaptive Boosting)**

*The problem:* A single weak learner (e.g., a decision stump) barely beats random chance. How do you build a strong classifier from many weak ones?

*Core insight:* After each round, upweight the examples the current ensemble got wrong. The next learner is forced to focus on the hardest cases.

*What breaks:* AdaBoost is brittle to noisy labels. If an example is mislabeled, it will be upweighted repeatedly and the model will overfit around that noise. Unlike gradient boosting, AdaBoost cannot easily incorporate regularization.

---

**Adam (Adaptive Moment Estimation)**

*The problem:* SGD uses the same learning rate for every parameter. Parameters with large gradients (common features in NLP) get too large an update; sparse parameters get too small.

*Core insight:* Track a per-parameter running average of the gradient (first moment — like momentum) and the squared gradient (second moment — like variance). Divide the update by the square root of the variance. Parameters with historically large gradients get smaller effective learning rates; rarely updated parameters get larger ones.

*Formula:* `m = β₁m + (1-β₁)g` and `v = β₂v + (1-β₂)g²`, then `θ ← θ - α·m̂/√v̂`. Bias-corrected versions (m̂, v̂) prevent near-zero estimates at the start of training.

*Practical note:* Default starting learning rate `3e-4` (Andrej Karpathy's "safe bet") works across most architectures. For transformer fine-tuning, AdamW (Adam with decoupled weight decay) is preferred — standard Adam applies weight decay through the gradient, which interacts badly with adaptive scaling.

---

**AUC-ROC (Area Under the ROC Curve)**

*The problem:* A classifier produces a score, not a hard label. Any fixed threshold produces one (precision, recall) point. How do you evaluate the classifier independently of which threshold you choose?

*Core insight:* Plot the true positive rate against the false positive rate as the threshold is swept from 0 to 1. The area under this curve measures discriminative ability across all thresholds. A random classifier produces AUC = 0.5; a perfect classifier produces AUC = 1.0.

*When it lies:* On severely imbalanced problems, ROC-AUC can look high even when the model is terrible on the minority class, because the large number of true negatives inflates the TN rate. Use PR-AUC (Precision-Recall AUC) instead when the minority class is what you care about.

---

**Attention Mechanism**

*The problem:* RNNs compress an entire sequence into a fixed-size hidden state. Information from early in a long sequence may be lost by the time the decoder uses it. Translation quality degrades as sequence length increases.

*Core insight:* Let the decoder look directly at all encoder hidden states at each decoding step, with a learned relevance weight for each. The model learns which parts of the input to focus on when generating each output token.

*Scaled dot-product attention formula:* `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`

Q (Query) represents what the current position is looking for. K (Key) represents what each position has to offer. V (Value) is what gets passed forward if a position is attended to. Scaling by `√d_k` prevents large dot products in high dimensions from collapsing softmax into near-zero gradients.

*What breaks:* O(N²) memory and compute in sequence length. Flash Attention addresses this by restructuring the computation to stay in fast SRAM, avoiding materializing the full N×N matrix.

---

## B

**Backpropagation**

*The problem:* You have a loss that depends on millions of parameters through a chain of operations. You need the gradient of the loss with respect to every parameter. Computing each gradient independently would require millions of forward passes.

*Core insight:* Gradients compose via the chain rule. Compute the gradient of the loss with respect to each operation's output once, then multiply backward through the graph. Each gradient is computed exactly once regardless of network depth.

*Practical note:* Backprop is automatic differentiation applied to neural networks — modern frameworks (PyTorch, JAX) implement this as a general-purpose computation graph differentiation. The "credit assignment" framing: which weight contributed how much to the error?

*What breaks:* Gradients can vanish (products of numbers < 1 over many layers → 0) or explode (products of numbers > 1 → infinity). Solutions: ReLU activations, residual connections, gradient clipping, batch normalization.

---

**Bagging (Bootstrap Aggregating)**

*The problem:* A single high-variance model (e.g., a deep decision tree) memorizes the training set. Its predictions swing wildly depending on which specific examples it saw.

*Core insight:* If you train many independently randomized versions of the same model and average their predictions, the random errors cancel out. The average is lower variance than any individual model without meaningfully changing bias.

*Mechanics:* For each model, sample a new training set with replacement from the original (bootstrap sample). Train independently. Average predictions (regression) or majority vote (classification).

*Key example:* Random Forest — each tree is bagged and also uses a random feature subset at each split, further de-correlating the trees.

*What breaks:* Bagging helps variance, not bias. If the base model is consistently wrong (high bias), averaging many copies of it still gives the same wrong answer. For high-bias problems, use boosting.

---

**Batch Normalization**

*The problem:* As training progresses, the distribution of each layer's inputs shifts as earlier layers' weights change. Each layer must constantly adapt to a moving target distribution — this slows training and requires very low learning rates.

*Core insight:* Normalize each layer's pre-activations to zero mean and unit variance within each mini-batch. Introduce learnable scale (γ) and shift (β) parameters so the layer can still represent any distribution it needs — but start from a stable baseline.

*Formula:* `x̂ = (x - μ_B) / √(σ²_B + ε)`, then `y = γx̂ + β`

*Inference detail:* During inference, use a running average of μ and σ² accumulated during training, not the statistics of the current (potentially single-sample) batch.

*What breaks:* Small batches produce noisy estimates of μ and σ², making BatchNorm unstable. For transformers and small-batch settings, Layer Normalization (normalizes across the feature dimension, not the batch) is preferred.

---

**Bias-Variance Trade-off**

*The problem:* You train a model but it fails on new data. Is the failure because the model is too simple (it's wrong even on training data) or too complex (it memorized training noise)? The distinction matters because the fixes are opposite.

*Core insight:* Decompose expected prediction error into three parts: systematic error from wrong assumptions (bias), sensitivity to training data fluctuations (variance), and irreducible noise (σ²).

*Formula:* `E[(y - ŷ)²] = Bias² + Variance + σ²`

*High Bias = Underfitting:* The model is consistently wrong in the same direction (e.g., linear regression on quadratic data). Training error and validation error are both high. Fix: increase model capacity, reduce regularization, add features.

*High Variance = Overfitting:* The model fits the training set well but fails on validation. Training error is low, validation error is high. Fix: more data, regularization, dropout, simpler model.

*What breaks:* Ensemble methods (particularly bagging) can reduce variance without increasing bias — the trade-off is not universal, it describes single models. More training data reduces variance but doesn't change bias.

---

**Binary Cross-Entropy (Log Loss)**

*The problem:* MSE penalizes wrong predictions proportionally to how far off they are in absolute terms. For classification, what you care about is probability calibration — a confident wrong prediction should be penalized far more than an uncertain one.

*Formula:* `-1/N · Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]`

*Core insight:* The log penalty is asymmetric at the extremes. Predicting probability 0.001 for a true positive event incurs near-infinite loss. This forces the model to be calibrated, not just directionally correct.

*Practical use:* Standard loss for binary classifiers. The gradient of cross-entropy + softmax simplifies cleanly, making optimization numerically well-behaved.

---

## C

**Confusion Matrix**

*The problem:* A single accuracy number hides the structure of errors. You need to know: which classes are being confused with which? Are false positives or false negatives more common?

*The fix:* A table showing actual class (rows) vs. predicted class (columns). For binary classification, this gives four cells.

```
                 Predicted
                Pos     Neg
Actual   Pos    TP      FN
         Neg    FP      TN
```

- **TP (True Positive):** Correctly predicted positive. Hit.
- **TN (True Negative):** Correctly predicted negative. Correct rejection.
- **FP (False Positive):** Predicted positive, actually negative. False alarm. Type I error.
- **FN (False Negative):** Predicted negative, actually positive. Miss. Type II error.

All other classification metrics are derived from these four numbers. Always start here before reporting any metric.

---

**Cosine Similarity**

*The problem:* You want to compare two high-dimensional vectors (embeddings of text, images, users) but Euclidean distance is dominated by magnitude. Two documents about the same topic but of different lengths would seem dissimilar.

*Core insight:* Measure the angle between vectors, not the distance. Vectors pointing in the same direction are similar regardless of length.

*Formula:* `cos(A, B) = (A · B) / (||A|| · ||B||)`, range [-1, 1].

*Practical use:* Semantic search, retrieval-augmented generation (RAG), document deduplication, recommendation systems. In embedding spaces for text and images, the range is effectively [0, 1] because embeddings tend to be in the positive orthant.

*What breaks:* Cosine similarity cannot detect magnitude differences. Two vectors identical in direction but very different in magnitude will have cosine similarity of 1.0. For some tasks (e.g., detecting emphasis), magnitude matters.

---

**Cross-Validation (K-Fold)**

*The problem:* A single train/validation split gives a performance estimate that depends heavily on which specific examples ended up in each set. With limited data, this variance is large.

*Core insight:* Make K non-overlapping splits. Each example is used exactly once for validation and K-1 times for training. Average the K scores. The average is a lower-variance estimate of true generalization performance than any single split.

*Variants:*
- **Stratified K-Fold:** Each fold preserves the original class distribution. Required for classification on imbalanced data.
- **Time Series Split (Walk-Forward):** Validation folds are always in the future relative to training folds. Never shuffle time series before splitting — shuffling introduces future data into training.
- **Leave-One-Out (LOO):** K = N. Maximum data usage; very expensive; high-variance estimates on noisy problems.

*What breaks:* CV is K times more expensive. For hyperparameter tuning with CV, use nested CV — the outer loop estimates performance, the inner loop selects hyperparameters. Using the same loop for both produces optimistic bias.

---

## D

**Data Leakage**

*The problem:* You train a model that achieves suspiciously high performance on validation data — but it fails completely in production. The model learned something that is available during training but not at inference time.

*Core insight:* Any information that flows from the future (or from the label) into the features during training is leakage. The model learns a spurious shortcut rather than the true signal.

*Common forms:*
- Including the target variable or a near-synonym as a feature
- Fitting the scaler / imputer on the full dataset before splitting (the validation set statistics leak into the preprocessing)
- Using features computed with future timestamps (e.g., "7-day rolling average" computed from the future)
- Encoding an ID column that correlates with cohort or time

*Detection:* Suspiciously high accuracy (e.g., 99.9% on a hard problem) → suspect leakage first. Check feature importances for unexpected top features.

---

**Dimensionality Reduction**

*The problem:* High-dimensional data is sparse (all points are roughly equidistant — the curse of dimensionality), expensive to process, and impossible to visualize. Distance-based algorithms break in hundreds of dimensions.

*Core insight:* Most of the variation in high-dimensional data lies along a much lower-dimensional manifold. You can project onto that manifold while preserving the structure that matters (depending on your definition of "matters").

*Techniques:*
- **PCA (linear):** Finds the axes of maximum variance. Projects data to a lower-dimensional space that preserves as much variance as possible. Sensitive to scale — standardize first.
- **t-SNE (non-linear):** Preserves local neighborhood structure. Excellent for visualizing clusters in embeddings. Does not preserve global distances — cluster positions in t-SNE plots are arbitrary.
- **UMAP (non-linear):** Faster than t-SNE; better preserves both local and global structure. Better for downstream tasks.
- **LDA (supervised):** Maximizes class separability rather than variance. Better for classification problems.

---

**Dropout**

*The problem:* Deep networks develop co-adaptations — neurons that only work in concert with specific other neurons. If those partners change, the neuron fails. This makes the network brittle and prone to overfitting.

*Core insight:* During each training step, randomly zero a fraction of activations. No neuron can rely on any particular partner. Each neuron must learn a representation that is useful in isolation. The effect is similar to training an exponential ensemble of sub-networks.

*At inference:* All neurons are active but their outputs are scaled by (1 - dropout rate) to match the expected activation magnitude during training.

*What breaks:* Dropout does not work well with Batch Normalization — the random masking disturbs the batch statistics that BatchNorm relies on. Also ineffective in very small networks where capacity is already tight.

---

## E

**Eigenvalue / Eigenvector**

*The problem:* A matrix transformation generally rotates and scales vectors in complicated ways. Is there a set of directions that the transformation only scales — not rotates?

*Answer:* Eigenvectors are those directions. Eigenvalues are the scaling factors.

*Formula:* `Av = λv` — matrix A applied to eigenvector v produces the same vector scaled by eigenvalue λ.

*Practical use in ML:* PCA computes the eigenvectors of the data's covariance matrix. These are the directions of maximum variance in the data. Projecting onto the top-k eigenvectors (those with the largest eigenvalues) retains the most information in k dimensions.

---

**Embedding**

*The problem:* Categorical variables (words, user IDs, product IDs) are not numbers. One-hot encoding produces very high-dimensional, sparse vectors where all categories are equidistant from each other — no semantic structure.

*Core insight:* Map each category to a dense, low-dimensional real vector that is learned from data. The embedding space can encode similarity — semantically related categories will have similar vectors.

*Key property:* `"King" - "Man" + "Woman" ≈ "Queen"` (Word2Vec). Arithmetic operations on embeddings reflect semantic relationships.

*Use cases:* Word2Vec and GloVe embeddings for NLP, BERT contextual embeddings, user/item embeddings in recommender systems, entity embeddings for categorical features in tabular models.

---

==**Entropy (Shannon)**

*The problem:* You want a formal measure of uncertainty or disorder in a probability distribution. How "surprising" is a random variable?

*Formula:* `H(X) = -Σ p(x) log p(x)`

*Properties:* Maximum when all outcomes are equally likely (maximum uncertainty). Zero when one outcome has probability 1 (no uncertainty).

*Practical use in ML:* Decision trees use Information Gain — the reduction in entropy from splitting on a feature — to choose split points. Cross-entropy loss is the expected entropy of the true distribution under the model's predicted distribution.

---

## F

**F1 Score**

*The problem:* Precision and Recall are both important but pull in opposite directions — you can maximize one by destroying the other. You need a single number that penalizes extreme imbalance between them.

*Formula:* `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

*Why harmonic mean:* Arithmetic mean would give F1 = 0.5 when Precision=1 and Recall=0. Harmonic mean gives F1=0 in that case — correctly punishing the degenerate classifier that never predicts positive.

*F-beta:* `F_β = (1+β²)·P·R / (β²·P + R)`. β > 1 weights Recall more heavily (missing positives is worse). β < 1 weights Precision more (false alarms are worse).

*Practical use:* The standard metric for imbalanced classification. Report macro-F1 (unweighted average across classes) or weighted-F1 (weighted by class support) for multi-class problems.

---

**Fine-Tuning**

*The problem:* Training a large model from scratch requires enormous data and compute. Most applied tasks have neither. But a model trained on a huge general dataset has already learned useful representations — the question is how to adapt them cheaply.

*Core insight:* The pre-trained model's weights are a good initialization. Start from them and continue training on your specific task. The model retains general knowledge while adapting to the new domain.

*Types:*
- **Full fine-tuning:** Update all weights. Expensive but maximally flexible.
- **Head-only fine-tuning:** Freeze the pre-trained backbone, train only a new task-specific output layer. Cheap; works when the pre-training domain is close to the target domain.
- **PEFT / LoRA:** Add small trainable adapter layers while freezing the base model. Reduces trainable parameters by orders of magnitude.

*What breaks:* Fine-tuning with too high a learning rate causes catastrophic forgetting — the model overwrites previously learned representations with task-specific noise.

---

## G

**Gradient Descent**

*The problem:* You have a loss function L(θ) with millions of parameters. Evaluating L at every possible θ is impossible. You need to find a minimum without enumerating the space.

*Core insight:* The gradient ∇L(θ) points in the direction that increases L most steeply. Move in the opposite direction.

*Update rule:* `θ ← θ - α·∇L(θ)`. The learning rate α controls step size — too large and you overshoot minima; too small and convergence is impractically slow.

*Variants:*
- **Batch GD:** Compute gradient over entire dataset. Exact but slow per update.
- **SGD (Stochastic):** Compute gradient on one example (or a mini-batch). Noisy but fast per update; noise helps escape shallow local minima.
- **Mini-batch GD:** Typical practice — batches of 32–512 examples balance noise and computation.

*What breaks:* Non-convex loss landscapes have local minima, saddle points, and flat regions (plateaus). Gradients can vanish in deep networks (products of small numbers → 0) or explode (products of large numbers → ∞). Learning rate sensitivity is high — wrong learning rate means divergence or extremely slow convergence.

---

**GAN (Generative Adversarial Network)**

*The problem:* Modeling the data distribution directly (as a VAE does) often produces blurry, averaged outputs. You want crisp, realistic samples.

*Core insight:* Don't model the distribution explicitly. Train two networks in competition: a Generator that produces fake samples and a Discriminator that tries to distinguish fake from real. Competition forces the Generator to produce increasingly realistic outputs.

*What breaks:* Mode collapse — the Generator finds a small set of outputs that fool the Discriminator and stops exploring. Training instability — the Generator and Discriminator must improve at similar rates; if one dominates, the other's gradients become uninformative. Training GANs requires careful balancing of architecture, learning rates, and normalization choices.

---

## H

**Hyperparameter Tuning**

*The problem:* Model training optimizes weights (parameters) but not the learning rate, architecture choices, regularization strength, etc. (hyperparameters). These fundamentally determine what kind of function the model can learn, but gradient descent can't optimize them — they're not in the computation graph.

*Methods:*
- **Grid Search:** Exhaustively evaluate all combinations in a predefined grid. Correct but combinatorially expensive.
- **Random Search:** Sample hyperparameter combinations randomly. Surprisingly effective because many hyperparameters are not important — random search wastes less budget on unimportant dimensions.
- **Bayesian Optimization:** Build a surrogate model of the performance surface. Sample the next hyperparameter combination where the surrogate predicts high performance. Efficient for expensive evaluations.

*What breaks:* Hyperparameter optimization itself can overfit to the validation set if you run enough trials. Use a held-out test set for final evaluation, never for hyperparameter selection.

---

## I

**Imbalanced Data**

*The problem:* When one class is rare (1:100 or 1:10,000), a model that always predicts the majority class achieves high accuracy while providing zero value. The standard loss function treats all examples equally, so misclassifying 100 minority examples costs the same as misclassifying 1 majority example.

*Solutions:*
- **Resampling:** SMOTE synthesizes new minority examples by interpolating between existing ones. Undersampling randomly removes majority examples.
- **Class weights:** Pass `class_weight='balanced'` to make the loss function penalize minority misclassifications proportionally more. Mathematically equivalent to oversampling but without duplicating data.
- **Threshold adjustment:** Shift the decision threshold below 0.5 to increase recall for the minority class.
- **Metric choice:** Use F1, PR-AUC, or MCC — not accuracy.

---

**IoU (Intersection over Union)**

*The problem:* An object detector predicts a bounding box for each detected object. How do you measure whether the predicted box is close enough to the ground truth box to count as correct?

*Formula:* `IoU = Area(Predicted ∩ Ground Truth) / Area(Predicted ∪ Ground Truth)`. Range [0, 1].

*Standard threshold:* IoU > 0.5 is typically treated as a correct detection (True Positive). Used in YOLO, R-CNN, and other object detection systems to compute mAP (mean Average Precision).

---

## K

**K-Means Clustering**

*The problem:* You have unlabeled data and want to find natural groupings. You don't have labels to supervise learning, but you suspect the data has cluster structure.

*Core insight:* Assign each point to its nearest cluster center (centroid). Update each centroid to the mean of its assigned points. Repeat until convergence. This iteratively minimizes within-cluster variance.

*What breaks:*
- Requires specifying K in advance (use the Elbow Method or Silhouette Score to choose).
- Assumes spherical clusters of roughly equal size. Fails on elongated, concentric, or very different-sized clusters.
- Sensitive to outliers (a single outlier can pull a centroid far from the true cluster).
- Random initialization can trap the algorithm at poor local minima (use k-means++ initialization).

---

==**KL Divergence (Kullback-Leibler)**

*The problem:* You have two probability distributions P and Q. How different are they? Euclidean distance between probability vectors ignores the probabilistic structure.

*Formula:* `D_KL(P || Q) = Σ P(x) log (P(x)/Q(x))`

*Properties:* Always ≥ 0. Equals 0 only when P = Q. Not symmetric — `D_KL(P||Q) ≠ D_KL(Q||P)`.

*Practical use:* VAE loss (forces the latent distribution toward a standard Gaussian prior). t-SNE loss (measures how well the high-dimensional neighborhood structure is preserved in 2D). Used to measure data drift in production monitoring.

---

## L

**Learning Rate**

*The problem:* Gradient descent moves in the direction of the gradient, but how far? Too large a step and you overshoot the minimum and diverge. Too small and training takes impractically long, or gets stuck in flat regions.

*Core insight:* The learning rate α directly controls the trade-off between convergence speed and stability. There is no single correct value — the optimal rate depends on the loss landscape, which changes during training.

*Scheduling:*
- **Warm-up:** Start small, increase linearly to the target LR. Prevents instability from large random gradients at initialization.
- **Cosine Decay:** Smoothly reduce LR following a cosine curve. Good empirical performance.
- **Linear Decay:** Simpler; works well for language model fine-tuning.

*What breaks:* A learning rate that is too high causes loss to oscillate or diverge. A learning rate that is too low causes the model to get stuck in flat regions (saddle points). This is why learning rate is typically the most important hyperparameter to tune.

---

**LSTM (Long Short-Term Memory)**

*The problem:* Standard RNNs theoretically maintain a hidden state that carries information across sequence steps. In practice, gradients backpropagating through long sequences vanish — the model learns only short-range dependencies.

*Core insight:* Add explicit gating mechanisms. A Forget Gate decides what to discard from the cell state. An Input Gate decides what new information to add. An Output Gate decides what to pass to the hidden state. The cell state provides a gradient highway through time.

*Key mechanics:* Input Gate, Forget Gate, Output Gate, and a separate Cell State (`c_t`) that is updated additively rather than through repeated matrix multiplication — preserving gradients over many steps.

*What breaks:* LSTMs are still sequential (one step at a time), limiting parallelization. For most NLP tasks, Transformers have replaced LSTMs because they can parallelize across the full sequence.

---

## M

**MSE / MAE / RMSE**

*The problem:* You need a scalar loss function for regression that reflects how wrong your predictions are — but the right choice depends on how you want to weight different kinds of errors.

- **MSE (Mean Squared Error):** `(1/n)Σ(y-ŷ)²` — Squares the error, so large errors (outliers) are penalized disproportionately. Differentiable everywhere. Gradient is proportional to the error magnitude — large errors get larger gradient updates.
- **MAE (Mean Absolute Error):** `(1/n)Σ|y-ŷ|` — Linear penalty; outliers do not dominate. Not differentiable at 0 (but subdifferentiable; handled gracefully by modern optimizers).
- **RMSE (Root Mean Squared Error):** `√MSE` — Same units as the target variable. More interpretable than MSE. Penalizes large errors like MSE.

*Choosing between them:* Use MAE when outliers are data artifacts to ignore. Use MSE/RMSE when large errors are genuinely more important than small ones (e.g., predicting demand — a 10× miss is not just 10× worse than a 1× miss; it causes stockouts or overstock).

---

**Momentum**

*The problem:* Plain SGD oscillates in directions with high curvature and moves slowly in directions with low curvature. The gradient at each step is noisy and doesn't reflect the consistent direction of improvement.

*Core insight:* Accumulate a velocity vector — a running average of past gradients. Like a ball rolling down a hill, the optimizer builds up speed in consistent directions and smooths out noise. Update: `v = βv + (1-β)g`, then `θ ← θ - α·v`.

*What breaks:* Momentum can carry the optimizer past a minimum ("overshoot") if the learning rate is too high. Nesterov Momentum (evaluate gradient at the projected future position) partially corrects this.

---

## N

**Normalization vs. Standardization**

*The problem:* Many ML algorithms (SVMs, neural networks, KNN, PCA) are sensitive to feature scale. A feature with values in [0, 10000] will dominate the loss or distance calculation over a feature in [0, 1], regardless of relative importance.

- **Normalization (Min-Max Scaling):** Scale each feature to [0, 1]: `x' = (x - x_min) / (x_max - x_min)`. Preserves zero; sensitive to outliers (one large outlier compresses all other values).
- **Standardization (Z-Score):** Scale to `μ=0, σ=1`: `x' = (x - μ) / σ`. Robust to outliers; no bounded range.

*When to use what:* Standardization for algorithms assuming Gaussian inputs (logistic regression, SVM, linear regression, PCA). Normalization for image pixel values (naturally bounded, no outlier concern) or when the algorithm requires bounded inputs.

*Critical:* Fit the scaler on the training set only. Apply (transform without re-fitting) to validation and test sets. Fitting on all data is leakage — validation statistics flow into preprocessing.

---

## O

**Overfitting**

*The problem:* The model learns the training data so precisely that it memorizes noise and idiosyncrasies rather than the underlying pattern. It performs well on training data but fails on new examples.

*Core insight:* Overfitting is a symptom of model capacity exceeding what the training data can constrain. The model has too many degrees of freedom and uses them to fit noise.

*Diagnostic:* Low training error, high validation error. The gap between them is the overfit.

*Solutions:*
- More training data (reduces variance directly)
- Regularization: L1/L2, dropout, weight decay
- Simpler model architecture
- Early stopping: stop training when validation error stops decreasing
- Data augmentation: expand the effective training set

---

## P

**PCA (Principal Component Analysis)**

*The problem:* You have high-dimensional data and want to reduce its dimensionality while preserving as much information (variance) as possible, using a linear projection.

*Core insight:* Find the directions in the original space along which the data varies most. Project onto those directions. The first principal component explains the most variance; the second explains the most remaining variance, orthogonal to the first; and so on.

*Mechanics:* Compute the covariance matrix of the (standardized) data. Find its eigenvectors and eigenvalues. Sort eigenvectors by descending eigenvalue. Project data onto the top-k eigenvectors.

*What breaks:* PCA is linear — it cannot capture curved manifolds in the data. It is sensitive to scale (standardize before running PCA). PCA on data with outliers will produce components dominated by those outliers.

---

**Precision and Recall**

*The problem:* A classifier's confusion matrix has four cells. You need metrics that focus on the class that matters (usually the positive class) in two complementary ways: how trustworthy are its positive predictions, and how complete are they?

- **Precision:** `TP / (TP + FP)` — Of all the examples the model labeled positive, what fraction actually were? Measures trustworthiness of positive predictions. Maximize when false positives are costly (e.g., wrongly flagging legitimate emails as spam).
- **Recall (Sensitivity):** `TP / (TP + FN)` — Of all actual positives, what fraction did the model find? Measures completeness of detection. Maximize when false negatives are costly (e.g., missing a cancer diagnosis).

*Trade-off:* Lowering the decision threshold catches more positives (higher recall, lower precision). Raising it certifies predictions more carefully (higher precision, lower recall). The PR curve traces this trade-off.

---

## R

**RAG (Retrieval-Augmented Generation)**

*The problem:* A language model's knowledge is frozen at training time. It cannot answer questions about recent events, private documents, or proprietary data. And LLMs hallucinate — they generate plausible-sounding but incorrect statements.

*Core insight:* Separate memory from computation. At inference time, ==retrieve relevant documents from an external knowledge base and inject them into the context. The LLM then grounds its answer in the retrieved evidence.

*Components:* Embedding model (converts documents and queries to vectors) + Vector Database (stores and retrieves documents by semantic similarity) + LLM (generates an answer grounded in the retrieved context).

*What breaks:* Retrieval quality bottlenecks everything — the LLM cannot improve on bad context. Long retrieved documents may exceed the context window or cause "lost in the middle" effects where the model ignores content in the middle of long contexts. RAG doesn't fix factual errors baked into the LLM itself.

---

**Regularization**

*The problem:* A model with too many parameters relative to training examples will fit noise (overfit). You need to penalize complexity from within the loss function without changing the model architecture.

*Core insight:* Add a penalty for large weights. Large weights encode strong, specific patterns — penalizing them biases the model toward simpler explanations that are less likely to memorize noise.

- **L1 (Lasso):** `Loss + α·Σ|w|` — Absolute value penalty; non-differentiable at zero. Optimization produces exactly-zero weights — implicit feature selection. Choose when you suspect many features are irrelevant.
- **L2 (Ridge):** `Loss + α·Σw²` — Squared penalty; smooth gradient. Weights shrink toward zero but never exactly reach it. More stable with correlated features. Choose when most features likely contribute.
- **Elastic Net:** `Loss + α₁·Σ|w| + α₂·Σw²` — Combines both. Handles correlated features better than pure L1.

---

**ReLU (Rectified Linear Unit)**

*The problem:* Sigmoid and tanh activations saturate at the extremes — gradient is near zero when inputs are very large or very small. Deep stacks of saturating activations produce vanishing gradients; early layers stop learning.

*Solution:* `f(x) = max(0, x)`. In the positive domain, gradient is exactly 1 — no attenuation. Computationally trivial.

*What breaks — Dead ReLU:* If a neuron's weights are updated such that it always receives negative input, its output is always 0 and its gradient is always 0. It is dead and will never recover. Solutions: Leaky ReLU (small slope for negative inputs), careful weight initialization (He initialization), warm-up learning rate schedules.

---

## S

**Softmax**

*The problem:* A neural network classifier produces a vector of raw scores (logits), one per class. You need to convert them into a valid probability distribution — non-negative, summing to 1.

*Formula:* `σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ`

*Properties:* Exponential amplification means the highest logit dominates strongly. Dividing by the sum normalizes. Temperature scaling modifies the sharpness: `σ(z/T)` — lower T makes the distribution sharper (more confident); higher T makes it flatter (more uncertain).

*Numerical stability:* Subtract max(z) from all logits before exponentiating to prevent overflow. This doesn't change the output but prevents exp(large number) = inf.

---

**SGD (Stochastic Gradient Descent)**

*The problem:* Batch gradient descent computes the exact gradient over the full training set before taking a single update step. With millions of examples, this is prohibitively slow.

*Core insight:* The gradient computed from a single random example (or a small batch) is a noisy estimate of the true gradient, but it points in roughly the right direction. Take many cheap noisy steps instead of few exact steps — you make much more progress per unit of compute.

*Why "stochastic" helps:* Gradient noise provides implicit regularization — it prevents the optimizer from settling into sharp, narrow minima that don't generalize. It also helps escape saddle points, which have zero gradient and would trap batch GD indefinitely.

*What breaks:* High noise means high variance in the loss curve — it oscillates. Mini-batches of 32–256 balance noise and efficiency. Learning rate must typically be lower than for batch GD; learning rate warm-up is important.

---

==**SHAP (SHapley Additive exPlanations)**

*The problem:* A model predicted X. Which features drove that specific prediction, and by how much? LIME and gradient-based methods make approximations; you want a theoretically grounded attribution.

*Core insight:* Borrow Shapley values from cooperative game theory. The fair credit for a feature is its average marginal contribution across all possible orderings in which features could be added to the prediction.

*Properties:* Satisfies efficiency (attributions sum to the prediction), symmetry (identical features get identical attributions), and null player (irrelevant features get zero attribution).

*Types:* TreeSHAP (exact and fast for tree-based models), KernelSHAP (model-agnostic, slower), DeepSHAP (neural networks).

*What breaks:* Computing exact SHAP values is exponential in the number of features. All variants make approximations. SHAP values explain model behavior, not causal effects — a high SHAP value does not mean the feature causes the outcome.

---

==**SVM (Support Vector Machine)**

*The problem:* You want a classifier that is not just correct, but confident — one whose decision boundary is as far as possible from all training examples. A boundary that barely separates the classes will fail on slightly shifted test data.

*Core insight:* Find the hyperplane that maximizes the margin — the distance to the nearest examples from each class. Only the examples on the margin boundary (support vectors) determine the solution.

*Kernel Trick:* When data is not linearly separable, map it to a higher-dimensional space where it becomes linearly separable, then find the maximum margin boundary there. Crucially, you never compute the high-dimensional mapping explicitly — only the inner products (kernel values) are needed.

*What breaks:* SVMs are O(n² p) to O(n³ p) in training complexity — slow on large datasets. Choosing the right kernel requires domain knowledge. SVMs do not produce calibrated probabilities by default.

---

## T

**Transformer**

*The problem:* RNNs process sequences sequentially — step t+1 waits for step t. This prevents parallelization across time steps and makes training large models on long sequences impractically slow. The sequential bottleneck also struggles with very long-range dependencies.

*Core insight:* Replace sequential recurrence with parallel self-attention. Every position attends directly to every other position simultaneously. On a GPU, the entire sequence is processed in parallel.

*Architecture:* Encoder (stacks of self-attention + feedforward layers, for understanding) and Decoder (adds cross-attention to encoder outputs, for generation). Modern LLMs are decoder-only (GPT family) or encoder-only (BERT) variants.

*What breaks:* O(N²) memory and compute in sequence length. Requires positional encodings (attention is permutation-invariant without them). No inductive sequence bias — requires large data to learn sequence structure.

---

**Transfer Learning**

*The problem:* Training a competitive model from scratch requires enormous labeled data and compute. Most real tasks have neither. You need a way to leverage models trained on large general datasets.

*Core insight:* Features learned by models on large datasets (edges and textures for vision, syntax and world knowledge for language) are broadly useful. Start from a pre-trained model rather than random initialization — the model has already solved the general representation problem. Fine-tune on your specific task.

*What breaks:* Transfer works best when the source domain (pre-training) and target domain (fine-tuning) are similar. Transferring a model from natural images to medical X-rays may require updating deeper layers. Fine-tuning with too high a learning rate causes catastrophic forgetting.

---

## V

**Vanishing Gradient**

*The problem:* Backpropagation multiplies gradients through every layer via the chain rule. If each layer's Jacobian has values < 1, the product over 100 layers approaches zero exponentially. Early layers receive near-zero gradient signal and stop learning — the model learns only shallow patterns.

*Solutions:*
- **ReLU activations:** Gradient is exactly 1 in the positive domain — no attenuation.
- **Residual connections (ResNet):** Skip connections provide direct gradient paths that bypass layers — gradients flow through addition, not multiplication.
- **Batch/Layer Normalization:** Keeps activations in a well-conditioned range.
- **Careful initialization:** Xavier initialization for sigmoid/tanh; He initialization for ReLU — maintains gradient magnitude at initialization.

---

**Vector Database**

*The problem:* LLMs and embedding models produce high-dimensional dense vectors for text, images, and other data. Finding semantically similar items requires nearest-neighbor search in millions or billions of vectors — exact search is O(nd) and impractical at scale.

*Core insight:* Use approximate nearest-neighbor (ANN) algorithms (HNSW, IVF-PQ) that trade a small accuracy loss for orders-of-magnitude speed gain. Store these index structures in a system optimized for vector operations.

*Examples:* Pinecone, Milvus, Chroma, Weaviate, pgvector.

*Practical use:* The "long-term memory" for LLM agents and RAG systems — documents are embedded offline and stored; queries are embedded at runtime and retrieve the most similar stored vectors.

---

## Z

**Zero-Shot Learning**

*The problem:* Supervised models can only classify inputs into classes they saw during training. If a new class appears (or if you have no labeled examples), a traditional classifier fails entirely.

*Core insight:* If you have a rich enough representation of classes (descriptions, attributes, or embeddings), you can classify inputs into unseen classes by measuring similarity to the class representation rather than to labeled examples.

*LLM version:* Large language models encode broad world knowledge during pre-training. At inference, you describe a task and the model performs it without any task-specific training examples.

*Example:* "Classify this text as Happy/Sad/Neutral." GPT-4 classifies correctly without having been fine-tuned on this exact label set — it maps the task description to learned representations of sentiment.

---

## Advanced Terms (2024–2025)

==**LIME (Local Interpretable Model-agnostic Explanations)**

*The problem:* Black-box models give no insight into why a specific prediction was made. You need an explanation for a single prediction, not the whole model.

*Core insight:* Perturb the input, observe how the model's output changes, and fit a simple linear model to those changes in the neighborhood of the point. The local linear model is interpretable.

*What breaks:* The neighborhood definition is arbitrary. Small changes to how you define "local" can produce very different explanations. LIME explanations are unstable across runs on the same input.

---

**Graph Neural Network (GNN)**

*The problem:* Standard neural networks expect fixed-size vector inputs. Many real problems have data structured as graphs — social networks, molecules, knowledge graphs, road networks — where the number of neighbors varies and the structure itself carries information.

*Core insight:* Let each node aggregate information from its neighbors, update its own representation, and repeat for L layers. After L layers, each node's representation encodes information from its L-hop neighborhood.

*Variants:* GCN (spectral aggregation), GraphSAGE (inductive, samples neighbors), GAT (attention-weighted aggregation).

*What breaks — Over-smoothing:* Too many layers cause all nodes to converge to nearly identical representations, losing discriminative information.

---

**Markov Decision Process (MDP)**

*The problem:* You want a formal framework for sequential decision-making under uncertainty — how should an agent act to maximize long-term reward when actions have uncertain consequences?

*Formal definition:* (S, A, P, R, γ) — States, Actions, Transition probabilities, Reward function, Discount factor.

*Markov Property:* The next state depends only on the current state and action, not on the full history. This greatly simplifies the problem but is often only approximately true in practice.

---

**Q-Learning**

*The problem:* In RL, you want to learn which action is best in each state without explicitly modeling the environment dynamics.

*Core insight:* Learn Q(s, a) — the expected cumulative reward from taking action a in state s and following the optimal policy thereafter. The Bellman equation provides the recursive update: `Q(s,a) ← r + γ · max_a' Q(s', a')`.

*DQN:* Q-learning + neural network function approximator + experience replay (randomly sample past transitions to break temporal correlations) + target network (slow-moving copy of the Q-network to stabilize targets).

---

**PPO (Proximal Policy Optimization)**

*The problem:* Policy gradient methods update the policy by following the gradient of expected return. But large updates can collapse performance — the updated policy is so different that good states become unreachable and the old data is no longer representative.

*Core insight:* Clip the policy update ratio to stay within a "proximal" region of the old policy. This prevents destructively large updates while still making progress.

*Relevance:* The RL algorithm behind RLHF (Reinforcement Learning from Human Feedback) in InstructGPT and ChatGPT — optimizes a language model's policy to maximize a learned reward model representing human preferences.

---

**Two-Tower Model**

*The problem:* Recommender systems must retrieve from millions of items in milliseconds for each user. A model that scores all user-item pairs jointly cannot scale to this retrieval problem.

*Core insight:* Encode user and item separately into the same embedding space with independent neural networks (towers). At inference, compute user embedding once; retrieve top-k items by approximate nearest-neighbor search. Scoring and retrieval use dot products in the shared embedding space.

*Used by:* YouTube, Pinterest, Google Play.

*Cold start:* New users and items have no interaction history — the model has nothing to embed. Fallback: content-based features, popularity, or explicit preference elicitation.

---

**NDCG (Normalized Discounted Cumulative Gain)**

*The problem:* For ranking systems (search, recommendations), accuracy at the top of the list matters much more than accuracy at the bottom. A ranking metric must discount positions — being right at rank 1 is worth more than being right at rank 10.

*Mechanics:* DCG = Σ (relevance_i / log₂(i+1)). NDCG normalizes by the ideal DCG (best possible ranking). Range [0, 1]. Evaluated at cutoff K (NDCG@K) to focus on the top of the list.

---

**LoRA (Low-Rank Adaptation)**

*The problem:* Full fine-tuning of a 70B-parameter model requires 70B gradients — hundreds of GB of optimizer states. This is inaccessible to most practitioners.

*Core insight:* Weight update matrices during fine-tuning tend to be low-rank. Instead of updating W directly, add trainable low-rank matrices: W' = W + AB, where A ∈ R^(d×r) and B ∈ R^(r×k) with r << d, k. Only A and B are trained. Reduces trainable parameters by ~10,000×.

*At inference:* A and B can be merged into W for zero additional latency.

---

**Knowledge Distillation**

*The problem:* Large models are accurate but slow and expensive to serve. Small models are fast but less accurate. Can you get a small model to approach the accuracy of a large model?

*Core insight:* Train the small "student" model to match the large "teacher" model's full output probability distribution (soft labels), not just the ground-truth hard labels. The soft distribution reveals inter-class similarities (e.g., "5% cat, 3% dog" is more informative than "cat") that accelerate learning.

*Temperature scaling:* Raise the temperature during distillation to soften the distribution and expose these inter-class relationships.

---

**MoE (Mixture of Experts)**

*The problem:* Scaling a dense model requires proportionally more compute for every parameter added. You want to increase model capacity (parameters) without proportionally increasing compute.

*Core insight:* Partition the model's feedforward layers into E "expert" sub-networks. For each input token, a learned router selects K experts (typically K=2 out of E=8 or E=64). Only the selected experts compute for that token. Parameters scale with E; compute scales with K.

*What breaks — Load balancing:* Without explicit incentives, the router learns to always use the same few experts (expert collapse). An auxiliary load-balancing loss penalizes routing imbalance during training.

---

**Differential Privacy (DP)**

*The problem:* A model trained on sensitive data (medical records, private messages) may memorize and reproduce individual examples in its outputs, leaking private information.

*Core insight:* Add carefully calibrated noise to gradients during training (DP-SGD). The noise ensures that the presence or absence of any single training example has a bounded effect on the final model weights, controlled by ε (privacy budget).

*DP-SGD:* Clip each example's gradient to a maximum norm, then add Gaussian noise to the clipped gradients before averaging. This provides formal (ε, δ)-differential privacy guarantees.

---

**Federated Learning**

*The problem:* You want to train a model across data that cannot be centralized — due to privacy regulations, bandwidth, or data ownership. You cannot bring the data to the model.

*Core insight:* Bring the model to the data. Each client trains on its local data and sends only the model update (gradient or weight delta) to a central server. The server aggregates updates (FedAvg: weighted average by local dataset size) and sends the updated global model back.

*What breaks:* Non-IID data — each client's data distribution may be very different. Local optima on individual clients may conflict. Communication cost is still significant. Updates themselves can leak information about the training data (gradient inversion attacks).

---

**Speculative Decoding**

*The problem:* Autoregressive LLM inference generates one token at a time through the full model. A 70B model generates ~10 tokens/second — too slow for many applications.

*Core insight:* Use a small draft model (which is fast) to generate K candidate tokens, then verify all K tokens in parallel with one forward pass of the large model. The large model can accept or reject each token. Tokens are accepted at the rate the large model would have generated them, but many are processed per large-model forward pass.

*Result:* 2–4× speedup with no quality loss, because rejected tokens are re-sampled from the large model's distribution.

---

**Flash Attention**

*The problem:* Standard attention requires materializing the full N×N attention matrix in GPU memory. This is O(N²) memory, which makes long sequences impossible on a single GPU.

*Core insight:* Restructure the attention computation into tiles that fit within fast SRAM (on-chip cache). Process each tile without writing the intermediate N×N matrix to slow HBM (off-chip GPU memory). The computation is mathematically identical to standard attention but uses O(N) memory.

*Result:* Enables training on much longer sequences; also faster because SRAM access is orders of magnitude faster than HBM.

---

**Test-Time Scaling**

*The problem:* More training compute keeps improving model quality, but training large models is extremely expensive and slow. Can you get better outputs from an existing model by spending more compute at inference time?

*Core insight:* Allocate inference compute to "think longer" — generate intermediate reasoning steps (chain-of-thought), use search over multiple candidate answers (MCTS, beam search over reasoning chains), or verify and critique own outputs.

*Examples:* OpenAI o1, DeepSeek-R1 — models trained to generate explicit reasoning tokens before answering. Performance on hard reasoning benchmarks scales with inference compute budget.

---

**Constitutional AI (CAI)**

*The problem:* RLHF requires expensive human preference labels. Human raters are inconsistent and can be gamed. How do you align a model more efficiently and consistently?

*Core insight:* Give the model a set of principles (a "constitution"). Have the model critique its own outputs against these principles and revise them. Use the critiqued and revised outputs as training data. The model learns to align with the principles without requiring a human to evaluate every example.

---

**DPO (Direct Preference Optimization)**

*The problem:* RLHF requires training a separate reward model from human preferences, then running PPO (an unstable RL algorithm) against that reward model. The two-stage process is complex, slow, and sensitive to reward model quality.

*Core insight:* The optimal RLHF policy can be expressed analytically in terms of the preference data. This allows direct optimization on the preference pairs (chosen vs. rejected responses) using a simple classification-style loss — no reward model or PPO required.

*Result:* Simpler, more stable, and often comparably effective to RLHF.

---

**CLIP (Contrastive Language-Image Pretraining)**

*The problem:* Vision models trained on labeled image datasets can only classify into the training label set. Adapting to new categories requires new labeled data.

*Core insight:* Train image and text encoders jointly on 400M image-caption pairs using contrastive loss — matching images to their captions and pushing them apart from non-matching captions. The shared embedding space aligns visual and textual semantics.

*Zero-shot use:* At inference, embed class names as text ("a photo of a cat"). Classify an image by finding the class whose text embedding is most similar to the image embedding.

---

**Causal Inference**

*The problem:* Correlation between A and B in your data does not mean A causes B. A confounding variable C could cause both. Acting on a spurious correlation (e.g., increasing ice cream sales to prevent drowning) will fail.

*Core insight:* Correlation tells you what varies together. Causation tells you what happens when you intervene. These require different methodology.

*Tools:*
- **RCT / A/B test:** Randomization breaks confounding. The gold standard when feasible.
- **Difference-in-Differences (DiD):** Compare treatment and control groups before and after treatment. Controls for time-invariant confounders.
- **Propensity Score Matching:** Match treated and untreated units on observed covariates to approximate randomization.
- **Instrumental Variables:** Use a variable that affects treatment but has no direct effect on outcome to isolate causal effect.

*Peeking problem in A/B tests:* Stopping an experiment early when it looks significant (before the pre-specified sample size) inflates the false positive rate. Pre-register your sample size and significance threshold.

---

**Confounder**

*The problem:* You observe that A and B are correlated in your data. You want to know if A causes B. But the correlation might be entirely explained by a third variable C that independently causes both A and B.

*Classic example:* Ice cream sales correlate with drowning rates. The confounder is hot weather — it causes both more ice cream consumption and more swimming (which leads to more drowning). Restricting ice cream sales would not reduce drowning.

*In ML:* Confounders are why models trained on observational data fail when deployed as interventions. An ad click model trained on historical data may confound "user interest" with "user was already going to buy" — serving more ads to those users does not cause them to click.

---

**Mamba / SSM (Selective State Space Model)**

*The problem:* Transformer attention is O(N²) in sequence length. For sequences of hundreds of thousands of tokens, this is computationally prohibitive.

*Core insight:* State Space Models (SSMs) process sequences in O(N) time using a latent state that is updated recurrently. Selective SSMs (Mamba) introduce input-dependent state selection — the model learns which input tokens to incorporate into the state and which to forget.

*Result:* Linear-time sequence modeling with competitive performance on long sequences. An alternative to O(N²) attention for very long contexts.

---

**PagedAttention**

*The problem:* LLM inference maintains a KV (key-value) cache for each active request. With many concurrent requests of variable sequence length, naive memory allocation leads to high fragmentation — GPU memory appears available but is unusable.

*Core insight:* Manage the KV cache like an operating system manages virtual memory — in fixed-size pages, mapped non-contiguously. Pages are allocated only when needed and freed when a request completes.

*Result:* Near-zero memory fragmentation, enabling 24× higher throughput in vLLM vs. naive implementations.

---

**RAGAS (Retrieval Augmented Generation Assessment)**

*The problem:* RAG systems have two distinct failure modes: the retrieval component may return irrelevant context, or the generation component may produce answers inconsistent with the retrieved context. You need metrics that diagnose which component is failing.

*Metrics:*
- **Faithfulness:** Does the answer only claim what the retrieved context supports?
- **Answer Relevancy:** Does the answer actually address the question?
- **Context Precision:** Are the retrieved documents relevant to the question?
- **Context Recall:** Are the retrieved documents sufficient to answer the question?

*Use:* Automated evaluation of RAG pipelines without requiring expensive human annotation for every query.

## Flashcards

**ReLU: max(0, x)?** #flashcard
computationally trivial, avoids saturation in the positive domain. Default for hidden layers.

**Sigmoid: 1/(1+e⁻ˣ)?** #flashcard
squashes to (0,1); used for binary output layers. Saturates at extremes, causing vanishing gradients in deep stacks.

**Softmax: eˣⁱ/Σeˣʲ?** #flashcard
converts a score vector into probabilities summing to 1. Used at the final layer of multi-class classifiers.

**GELU: x·Φ(x)?** #flashcard
smooth approximation of ReLU; empirically stronger in transformers.

**TP (True Positive)?** #flashcard
Correctly predicted positive. Hit.

**TN (True Negative)?** #flashcard
Correctly predicted negative. Correct rejection.

**FP (False Positive)?** #flashcard
Predicted positive, actually negative. False alarm. Type I error.

**FN (False Negative)?** #flashcard
Predicted negative, actually positive. Miss. Type II error.

**Stratified K-Fold?** #flashcard
Each fold preserves the original class distribution. Required for classification on imbalanced data.

**Time Series Split (Walk-Forward): Validation folds are always in the future relative to training folds. Never shuffle time series before splitting?** #flashcard
shuffling introduces future data into training.

**Leave-One-Out (LOO)?** #flashcard
K = N. Maximum data usage; very expensive; high-variance estimates on noisy problems.

**Including the target variable or a near-synonym as a feature?** #flashcard
Including the target variable or a near-synonym as a feature

**Fitting the scaler / imputer on the full dataset before splitting (the validation set statistics leak into the preprocessing)?** #flashcard
Fitting the scaler / imputer on the full dataset before splitting (the validation set statistics leak into the preprocessing)

**Using features computed with future timestamps (e.g., "7-day rolling average" computed from the future)?** #flashcard
Using features computed with future timestamps (e.g., "7-day rolling average" computed from the future)

**Encoding an ID column that correlates with cohort or time?** #flashcard
Encoding an ID column that correlates with cohort or time

**PCA (linear): Finds the axes of maximum variance. Projects data to a lower-dimensional space that preserves as much variance as possible. Sensitive to scale?** #flashcard
standardize first.

**t-SNE (non-linear): Preserves local neighborhood structure. Excellent for visualizing clusters in embeddings. Does not preserve global distances?** #flashcard
cluster positions in t-SNE plots are arbitrary.

**UMAP (non-linear)?** #flashcard
Faster than t-SNE; better preserves both local and global structure. Better for downstream tasks.

**LDA (supervised)?** #flashcard
Maximizes class separability rather than variance. Better for classification problems.

**Full fine-tuning?** #flashcard
Update all weights. Expensive but maximally flexible.

**Head-only fine-tuning?** #flashcard
Freeze the pre-trained backbone, train only a new task-specific output layer. Cheap; works when the pre-training domain is close to the target domain.

**PEFT / LoRA?** #flashcard
Add small trainable adapter layers while freezing the base model. Reduces trainable parameters by orders of magnitude.

**Batch GD?** #flashcard
Compute gradient over entire dataset. Exact but slow per update.

**SGD (Stochastic)?** #flashcard
Compute gradient on one example (or a mini-batch). Noisy but fast per update; noise helps escape shallow local minima.

**Mini-batch GD: Typical practice?** #flashcard
batches of 32–512 examples balance noise and computation.

**Grid Search?** #flashcard
Exhaustively evaluate all combinations in a predefined grid. Correct but combinatorially expensive.

**Random Search: Sample hyperparameter combinations randomly. Surprisingly effective because many hyperparameters are not important?** #flashcard
random search wastes less budget on unimportant dimensions.

**Bayesian Optimization?** #flashcard
Build a surrogate model of the performance surface. Sample the next hyperparameter combination where the surrogate predicts high performance. Efficient for expensive evaluations.

**Resampling?** #flashcard
SMOTE synthesizes new minority examples by interpolating between existing ones. Undersampling randomly removes majority examples.

**Class weights?** #flashcard
Pass class_weight='balanced' to make the loss function penalize minority misclassifications proportionally more. Mathematically equivalent to oversampling but without duplicating data.

**Threshold adjustment?** #flashcard
Shift the decision threshold below 0.5 to increase recall for the minority class.

**Metric choice: Use F1, PR-AUC, or MCC?** #flashcard
not accuracy.

**Requires specifying K in advance (use the Elbow Method or Silhouette Score to choose).?** #flashcard
Requires specifying K in advance (use the Elbow Method or Silhouette Score to choose).

**Assumes spherical clusters of roughly equal size. Fails on elongated, concentric, or very different-sized clusters.?** #flashcard
Assumes spherical clusters of roughly equal size. Fails on elongated, concentric, or very different-sized clusters.

**Sensitive to outliers (a single outlier can pull a centroid far from the true cluster).?** #flashcard
Sensitive to outliers (a single outlier can pull a centroid far from the true cluster).

**Random initialization can trap the algorithm at poor local minima (use k-means++ initialization).?** #flashcard
Random initialization can trap the algorithm at poor local minima (use k-means++ initialization).

**Warm-up?** #flashcard
Start small, increase linearly to the target LR. Prevents instability from large random gradients at initialization.

**Cosine Decay?** #flashcard
Smoothly reduce LR following a cosine curve. Good empirical performance.

**Linear Decay?** #flashcard
Simpler; works well for language model fine-tuning.

**MSE (Mean Squared Error): (1/n)Σ(y-ŷ)²?** #flashcard
Squares the error, so large errors (outliers) are penalized disproportionately. Differentiable everywhere. Gradient is proportional to the error magnitude — large errors get larger gradient updates.

**MAE (Mean Absolute Error): (1/n)Σ|y-ŷ|?** #flashcard
Linear penalty; outliers do not dominate. Not differentiable at 0 (but subdifferentiable; handled gracefully by modern optimizers).

**RMSE (Root Mean Squared Error): √MSE?** #flashcard
Same units as the target variable. More interpretable than MSE. Penalizes large errors like MSE.

**Normalization (Min-Max Scaling)?** #flashcard
Scale each feature to [0, 1]: x' = (x - x_min) / (x_max - x_min). Preserves zero; sensitive to outliers (one large outlier compresses all other values).

**Standardization (Z-Score)?** #flashcard
Scale to μ=0, σ=1: x' = (x - μ) / σ. Robust to outliers; no bounded range.

**More training data (reduces variance directly)?** #flashcard
More training data (reduces variance directly)

**Regularization?** #flashcard
L1/L2, dropout, weight decay

**Simpler model architecture?** #flashcard
Simpler model architecture

**Early stopping?** #flashcard
stop training when validation error stops decreasing

**Data augmentation?** #flashcard
expand the effective training set

**Precision: TP / (TP + FP)?** #flashcard
Of all the examples the model labeled positive, what fraction actually were? Measures trustworthiness of positive predictions. Maximize when false positives are costly (e.g., wrongly flagging legitimate emails as spam).

**Recall (Sensitivity): TP / (TP + FN)?** #flashcard
Of all actual positives, what fraction did the model find? Measures completeness of detection. Maximize when false negatives are costly (e.g., missing a cancer diagnosis).

**L1 (Lasso): Loss + α·Σ|w|?** #flashcard
Absolute value penalty; non-differentiable at zero. Optimization produces exactly-zero weights — implicit feature selection. Choose when you suspect many features are irrelevant.

**L2 (Ridge): Loss + α·Σw²?** #flashcard
Squared penalty; smooth gradient. Weights shrink toward zero but never exactly reach it. More stable with correlated features. Choose when most features likely contribute.

**Elastic Net: Loss + α₁·Σ|w| + α₂·Σw²?** #flashcard
Combines both. Handles correlated features better than pure L1.

**ReLU activations: Gradient is exactly 1 in the positive domain?** #flashcard
no attenuation.

**Residual connections (ResNet): Skip connections provide direct gradient paths that bypass layers?** #flashcard
gradients flow through addition, not multiplication.

**Batch/Layer Normalization?** #flashcard
Keeps activations in a well-conditioned range.

**Careful initialization: Xavier initialization for sigmoid/tanh; He initialization for ReLU?** #flashcard
maintains gradient magnitude at initialization.

**RCT / A/B test?** #flashcard
Randomization breaks confounding. The gold standard when feasible.

**Difference-in-Differences (DiD)?** #flashcard
Compare treatment and control groups before and after treatment. Controls for time-invariant confounders.

**Propensity Score Matching?** #flashcard
Match treated and untreated units on observed covariates to approximate randomization.

**Instrumental Variables?** #flashcard
Use a variable that affects treatment but has no direct effect on outcome to isolate causal effect.

**Faithfulness?** #flashcard
Does the answer only claim what the retrieved context supports?

**Answer Relevancy?** #flashcard
Does the answer actually address the question?

**Context Precision?** #flashcard
Are the retrieved documents relevant to the question?

**Context Recall?** #flashcard
Are the retrieved documents sufficient to answer the question?
