---
module: Foundations
topic: Introduction To Ai
subtopic: ""
status: unread
tags: [foundations, ml, introduction-to-ai]
---
# Introduction to AI

---

## What Is AI, and Why Does the Question Matter?

**The problem:** Computers execute exact instructions. But most useful tasks — recognizing a face, translating a sentence, deciding whether a loan applicant will default — cannot be fully specified as rules. Every explicit rule system ever written for these tasks broke on cases the rule writer didn't anticipate.

**The core insight:** Instead of writing rules, write a *learning procedure* that infers rules from examples. The program's job shifts from "solve the problem" to "learn how to solve the problem from data."

**The mechanics:** An AI system is any program that uses this learned knowledge to act on new inputs. Machine learning is the dominant method for building such systems today.

**What breaks:** Learning from examples inherits all the biases and gaps in those examples. A system trained only on historical loan decisions learns the prejudices of the loan officers who made them. The problem of specifying rules is replaced by the harder problem of curating representative, unbiased data.

---

## Core Learning Paradigms

### Statistical Machine Learning

**The problem:** You have labeled examples (input → output pairs) and you want a function that generalizes — one that predicts correctly on inputs it has never seen.

**The core insight:** Structure your function family to be expressive enough to fit real patterns but constrained enough to avoid memorizing noise. A linear boundary, a splitting tree, or a maximum-margin hyperplane each embodies a different assumption about what "structure" looks like.

**The mechanics:** Algorithms like linear regression, logistic regression, SVMs, and decision trees learn explicit decision boundaries from labeled data. Random forests and gradient-boosted trees combine many such boundaries to reduce error.

**What breaks:** If the true pattern is outside the function family (e.g., you fit a line to curved data), the model will be wrong systematically — high bias. If the family is too flexible (e.g., a very deep tree), it memorizes training noise — high variance.

---

### Deep Learning

**The problem:** Hand-crafting features for images, audio, and text requires immense domain expertise and rarely generalizes across tasks. Even expert-designed features for face recognition fail on unusual lighting conditions no one thought to encode.

**The core insight:** Stack many simple parameterized transformations (layers). Each layer learns to detect patterns in the output of the previous layer. The hierarchy of features emerges from the data, not from a human designer.

**The mechanics:** Neural networks chain matrix multiplications and non-linear activations. Backpropagation computes how each weight contributed to the error; gradient descent updates them. With enough data and compute, hierarchical representations of images, words, and signals emerge automatically.

**What breaks:** Deep networks need enormous data and compute. They are hard to interpret. They fail silently on inputs that look statistically different from training data (distribution shift). Very deep networks suffer vanishing or exploding gradients unless special architecture choices (residual connections, normalization) are made.

---

### Generative AI

**The problem:** Discriminative models tell you which class an input belongs to, but they cannot synthesize new examples. You cannot ask a spam classifier to write an email.

**The core insight:** Model the data distribution P(x) itself — or a conditional version P(x | condition). Once you can sample from this distribution, you can generate new images, text, audio, or any other data type.

**The mechanics:** Different approaches model the distribution differently. GANs pit a generator against a discriminator; the generator improves until its samples are indistinguishable from real data. VAEs encode data into a structured latent space and decode samples from it. Diffusion models learn to iteratively denoise random noise back into real data. Autoregressive models (GPT-style) predict the next token given all previous ones, factoring P(x) into a product of conditional probabilities.

**What breaks:** GANs are notoriously unstable to train (mode collapse, training oscillation). VAEs produce blurry outputs because they average over the latent distribution. Diffusion models are slow at inference. Autoregressive models can hallucinate confidently — the decoding process has no mechanism to verify factual accuracy.

---

### Reinforcement Learning

**The problem:** For many tasks — game playing, robot locomotion, dialogue systems — there is no correct output to supervise against. The only feedback is a scalar reward signal that arrives late, after a sequence of actions.

**The core insight:** An agent should learn to choose actions that maximize cumulative future reward, not just immediate reward. This transforms the learning problem into a search for a *policy* — a mapping from states to actions.

**The mechanics:** The agent interacts with an environment, observes states, takes actions, and receives rewards. Algorithms like Q-learning learn to estimate the value of each (state, action) pair. Policy gradient methods (like PPO) directly optimize the policy by reinforcing actions that led to high returns. The Bellman equation provides the recursive backbone: the value of a state equals the immediate reward plus the discounted value of the next state.

**What breaks:** RL requires an enormous number of environment interactions to converge. Sparse rewards make credit assignment extremely hard (which of the 10,000 actions led to winning?). Simulated environments often produce policies that fail in the real world (sim-to-real gap). RLHF for language models inherits all these instabilities plus reward hacking — models learn to satisfy the reward model without satisfying the intent behind it.

---

## Machine Learning Fundamentals

### Bias-Variance Trade-off

**The problem:** You train a model and it performs well on training data but poorly on new data. Alternatively, it performs poorly on both. These are two distinct failure modes with different causes and different fixes — how do you tell them apart and what do you do?

**The core insight:** Prediction error has two controllable components. *Bias* is the systematic error from making wrong assumptions about the function shape — the model is consistently wrong in the same direction. *Variance* is the error from sensitivity to the specific training examples used — the model's predictions vary wildly across different training sets. A third component, irreducible noise, cannot be fixed regardless of model choice.

**The mechanics:**

```
E[(y - ŷ)²] = Bias² + Variance + σ²
```

Increasing model complexity (more features, deeper trees, more parameters) typically decreases bias but increases variance. Decreasing complexity does the reverse. The goal is to minimize total error, which sits at the sweet spot between the two.

| Factor | Effect on Bias | Effect on Variance |
|---|---|---|
| More training data | None | Decreases |
| More complex model | Decreases | Increases |
| Regularization (higher λ) | Increases | Decreases |
| Ensemble methods (bagging) | Decreases | Decreases |
| More features | Decreases | Increases |

**What breaks:** The trade-off is not symmetric. Adding more data always helps variance and never hurts bias — but you often can't get more data. Ensemble methods can simultaneously reduce both, which is why they dominate competition leaderboards; the cost is interpretability and training time.

---

### Regularization

**The problem:** A model with too many parameters relative to training examples will fit noise. You cannot always get more data or use a simpler model. You need a way to penalize complexity from within the loss function itself.

**The core insight:** Add a penalty for large weights to the thing you're minimizing. Large weights encode strong, specific patterns — penalizing them biases the model toward simpler explanations that are less likely to be coincidental noise in the training set.

**The mechanics:**

L1 (Lasso): `Loss = MSE + α × Σ|w|`
- The absolute-value penalty has a corner at zero; the gradient is discontinuous there. Optimization pushes weights exactly to zero — the model performs implicit feature selection.

L2 (Ridge): `Loss = MSE + α × Σw²`
- The squared penalty produces a smooth gradient everywhere. Weights shrink proportionally toward zero but never reach it exactly. Numerically stable; works well when all features contribute.

Elastic Net: `Loss = MSE + α₁ × Σ|w| + α₂ × Σw²`
- Combines both penalties. Handles correlated features better than pure L1 (which arbitrarily zeros one of a correlated pair).

**What breaks:** Regularization introduces a new hyperparameter (α / λ) that must itself be tuned. Too high and you underfit; the model is too constrained to learn real signal. Regularization also assumes all weights should be small — a bad assumption if a few features genuinely have large true effects.

---

### Train/Validation/Test Split

**The problem:** You need an honest estimate of how your model will perform on data it has never seen. If you use the same data to both train and evaluate, you will overestimate performance — the model has implicitly optimized for that data.

**The core insight:** Hold out data before training begins and never touch it until a final, irreversible evaluation. Every decision you make while looking at held-out performance (even just choosing a threshold) uses up some of that held-out data's independence.

**The mechanics:** Split data into three disjoint sets before any fitting occurs.

- **Training set (60–80%):** The model sees and learns from this.
- **Validation set (10–20%):** Used to compare models and tune hyperparameters. Each time you look at validation performance to make a decision, you are implicitly using it as a training signal.
- **Test set (10–20%):** Touched exactly once, after all decisions are final. This produces the only honest performance estimate.

For imbalanced classes, stratify each split so class proportions are preserved. For time series, always split chronologically — using future data to predict the past is leakage.

**What breaks:** The test set provides one sample from the distribution of possible evaluations. With small datasets, this estimate has high variance. Repeated use of the test set (across experiments, across papers on the same benchmark) is a form of slow leakage — the field eventually overfits the benchmark.

---

### Cross-Validation

**The problem:** With small datasets, a single train/validation split may assign you a lucky or unlucky split by chance. A model that performs well on one split might perform poorly on another. You need a more stable estimate of generalization performance.

**The core insight:** Instead of one split, make K different splits and average the results. Each data point acts as validation data exactly once. The average over K folds is a much lower-variance estimate of true generalization performance.

**The mechanics:** In K-Fold CV, partition data into K equal folds. For each fold: train on the remaining K-1 folds, evaluate on the held-out fold. Report the mean and standard deviation of the K scores.

- **Stratified K-Fold:** Preserves class distribution in each fold. Required for imbalanced classification.
- **Time Series Split:** Each validation fold is strictly in the future relative to its training fold. Never shuffle time series before splitting.
- **Leave-One-Out (LOO):** K = N. Maximally uses data but is computationally expensive and produces high-variance estimates on noisy problems.

**What breaks:** CV is K times more expensive than a single run. For large models or large datasets, this is prohibitive. Nested CV is required when tuning hyperparameters with CV — using the same CV loop for both model selection and performance estimation is a subtle form of optimistic bias.

---

### Ensemble Methods

**The problem:** A single model makes a single type of error. If you could combine models that make different errors, their mistakes might cancel out.

**The core insight:** Errors cancel only if the models are sufficiently independent — trained on different data, initialized differently, or using different algorithms. Combining dependent models does little.

**The mechanics:**

**Bagging (Bootstrap Aggregating):** Train multiple models in parallel, each on a random bootstrap sample (sampling with replacement) of the training data. Average predictions (regression) or take majority vote (classification). Each model sees a slightly different dataset, so their errors are partially independent. This reduces variance without changing bias. *Example: Random Forest.*

**Boosting:** Train models sequentially. Each new model is trained to correct the errors of the ensemble so far — either by upweighting misclassified examples (AdaBoost) or by fitting the residual errors directly (gradient boosting). Reduces bias. More powerful than bagging on most tabular problems but more sensitive to noisy data. *Examples: XGBoost, LightGBM, CatBoost.*

**Stacking:** Train a meta-model whose inputs are the predictions of base models on held-out data. The meta-model learns which base models to trust for which inputs.

**What breaks:** Boosting overfits noisy data because it is designed to reduce error on the training set — it aggressively chases every misclassification including noise. Ensembles are harder to interpret and slower to serve. The gains from stacking diminish as base models become more similar.

---

## Machine Learning Algorithms

### Supervised Learning

Supervised learning uses labeled data — (input, output) pairs — to learn a mapping function that generalizes to new inputs.

#### Regression Algorithms

| Algorithm | Time Complexity | Key Assumption | When to Use |
|---|---|---|---|
| Linear Regression | O(n·p²) | Linear relationship, independent features | Baseline, interpretability required |
| Ridge / Lasso | O(n·p²) | Same + many or irrelevant features | High-dimensional, feature selection |
| Decision Tree | O(n·log(n)·p) | None | Non-linear, interpretable |
| Random Forest | O(n·log(n)·p·t) | None | General purpose, non-linear |
| XGBoost / LightGBM | O(n·log(n)·p·t) | None | Highest accuracy on tabular data |
| SVR | O(n²·p)–O(n³·p) | Kernel-dependent | Small-medium data, non-linear |
| KNN | O(n·p) per prediction | Similar inputs cluster together | Small data, simple baseline |

#### Classification Algorithms

| Algorithm | Time Complexity | Strengths | Common Failure Modes |
|---|---|---|---|
| Logistic Regression | O(n·p) | Fast, interpretable, calibrated probabilities | Assumes linear decision boundary |
| Naive Bayes | O(n·p) | Works with tiny data, fast | Assumes feature independence |
| Decision Tree | O(n·log(n)·p) | Interpretable, non-linear | Prone to overfitting |
| Random Forest | O(n·log(n)·p·t) | Robust, handles imbalance | Less interpretable, slower |
| XGBoost | O(n·log(n)·p·t) | Best on tabular data | Overfits with poor tuning |
| LightGBM | O(n·p·t) | Fastest for large data | Overfits on small data |
| SVM | O(n²·p)–O(n³·p) | Effective in high dimensions | Slow on large datasets |
| KNN | O(n·p) | No training phase | Slow inference, curse of dimensionality |

*n = samples, p = features, t = trees/estimators*

#### Algorithm Selection

Start simple and escalate based on evidence:

1. **Regression:** Linear Regression → Random Forest → XGBoost
2. **Classification:** Logistic Regression → Random Forest → XGBoost

Choose based on constraints:
- Need interpretability: Linear/Logistic Regression, Decision Tree
- High-dimensional data: Lasso, Ridge, Random Forest
- Categorical features: CatBoost, LightGBM
- Limited data: Naive Bayes, regularized linear models
- Fast inference required: Linear models, Naive Bayes
- Maximum accuracy on tabular: XGBoost, LightGBM, CatBoost

---

### Unsupervised Learning

Unsupervised learning finds structure in data without labels.

#### Clustering

| Algorithm | Time Complexity | Best For | Limitations |
|---|---|---|---|
| K-Means | O(n·k·i·p) | Spherical clusters, large datasets | Must specify k; sensitive to outliers |
| DBSCAN | O(n·log(n)) | Arbitrary shapes, outlier detection | Struggles with varying density |
| Hierarchical | O(n²·log(n)) | Unknown k, dendrograms | Slow on large data |
| GMM | O(n·k·i·p) | Soft assignments, probabilistic clusters | Assumes Gaussian distributions |

*k = clusters, i = iterations*

#### Dimensionality Reduction

**The problem:** High-dimensional data is sparse (curse of dimensionality), slow to process, and hard to visualize. Distance metrics break down in hundreds of dimensions.

| Algorithm | Type | Preserves | Use Case |
|---|---|---|---|
| PCA | Linear | Global structure, variance | Noise reduction, preprocessing |
| t-SNE | Non-linear | Local structure | 2D/3D visualization |
| UMAP | Non-linear | Local + global structure | Faster visualization than t-SNE |
| LDA | Supervised linear | Class separability | Feature extraction for classification |
| Autoencoders | Non-linear | Learned features | Complex non-linear compression |

---

## Evaluation Metrics

### Regression Metrics

**The problem:** You need a single number that summarizes how wrong your predictions are across all examples — but "wrong" means different things in different contexts.

| Metric | Formula | Use When |
|---|---|---|
| MAE | (1/n) Σ\|y - ŷ\| | All errors matter equally; outliers should not dominate |
| MSE | (1/n) Σ(y - ŷ)² | Large errors are disproportionately costly |
| RMSE | √MSE | Large errors costly; want error in same units as target |
| R² | 1 - (SS_res / SS_tot) | Explaining variance; comparing models on same target |
| Adjusted R² | 1 - [(1-R²)(n-1)/(n-p-1)] | Comparing models with different numbers of features |
| MAPE | (100/n) Σ\|y - ŷ\|/y | Percentage errors; target values are never near zero |

Key distinctions: MAE has linear penalty so outliers do not dominate. MSE has quadratic penalty so one very wrong prediction can drive the entire metric. R² can be negative if your model is worse than predicting the mean. MAPE is undefined or explosive when targets are near zero.

---

### Classification Metrics

**The problem:** Accuracy is simple but misleading whenever classes are imbalanced. A classifier that always predicts "not fraud" achieves 99.9% accuracy on a fraud dataset. You need metrics that measure performance on the classes that matter.

**Confusion matrix:**

```
                  Predicted
                 Pos    Neg
    Actual Pos   TP     FN
           Neg   FP     TN
```

| Metric | Formula | What It Measures |
|---|---|---|
| Accuracy | (TP + TN) / Total | Overall correctness — only valid for balanced classes |
| Precision | TP / (TP + FP) | Of predicted positives, what fraction are real? |
| Recall (Sensitivity) | TP / (TP + FN) | Of all real positives, what fraction did we catch? |
| Specificity | TN / (TN + FP) | Of all real negatives, what fraction did we correctly reject? |
| F1 Score | 2·P·R / (P + R) | Harmonic mean — punishes extreme imbalances between P and R |
| F-beta | (1+β²)·P·R / (β²·P + R) | F1 weighted toward recall (β > 1) or precision (β < 1) |

**Advanced metrics:**

| Metric | Use When |
|---|---|
| ROC-AUC | Balanced classes; need a threshold-invariant summary |
| PR-AUC | Imbalanced classes; accuracy of the positive class matters most |
| Log Loss | You care about probability calibration, not just class membership |
| MCC | Balanced summary for imbalanced data across all four quadrants |

**Precision vs. Recall trade-off:** Increasing the decision threshold raises precision (fewer false alarms) but lowers recall (more misses). The right balance is determined by the cost of each error type. A cancer screening test should maximize recall (missing cancer is catastrophic). A spam filter should maximize precision (misclassifying real email is costly).

**Metric selection:**

| Scenario | Preferred Metric |
|---|---|
| Balanced classes | Accuracy, F1, ROC-AUC |
| Imbalanced classes | Precision, Recall, PR-AUC, F1 |
| False positives are costly | Precision |
| False negatives are costly | Recall |
| Need calibrated probabilities | Log Loss |
| Multi-class | Macro/Micro/Weighted F1 |

---

### Handling Imbalanced Data

**The problem:** Your model achieves high accuracy by ignoring the minority class entirely. The loss function treats all examples equally, so misclassifying 100 examples from the 1% minority class costs the same as misclassifying 1 example from the 99% majority class.

**The core insight:** Either change the data so minority examples are more common, or change the loss function so minority mistakes cost more. Both are equivalent mathematically — they differ in implementation convenience.

**Resampling approaches:**
- Oversample minority class: SMOTE synthesizes new minority examples by interpolating between existing ones. ADASYN focuses synthesis near decision boundaries.
- Undersample majority class: Randomly remove majority examples. Faster but discards potentially useful data.
- Combined: Oversample minority + undersample majority.

**Algorithmic approaches:**
- Class weights: Pass `class_weight='balanced'` or manual weights to your loss function. The model pays more for minority misclassifications without touching the data.
- Focal loss: Down-weights easy majority examples dynamically during training, forcing the model to focus on hard minority cases.

**Metric changes:** Switch from accuracy to precision, recall, F1, PR-AUC. Monitor the confusion matrix directly.

---

## Deep Learning

### Why Layers?

**The problem:** A single layer of linear transformations is itself a linear transformation, regardless of depth. No amount of stacking linear layers creates non-linearity.

**The core insight:** Insert a non-linear activation function after each linear transformation. Now the composition of layers is genuinely non-linear and can approximate any continuous function given enough parameters (Universal Approximation Theorem).

**What breaks:** This only guarantees approximation capacity. It says nothing about whether gradient descent will find the approximation in any reasonable number of steps, or whether the approximation will generalize.

---

### Core Components

#### Neural Network Layers

| Layer Type | Purpose | Common Use |
|---|---|---|
| Dense (Fully Connected) | Arbitrary learned transformations | MLPs, output layers |
| Convolutional | Extract spatial features with weight sharing | Images, CNNs |
| Recurrent (RNN, LSTM, GRU) | Maintain state across sequences | Time series, legacy NLP |
| Attention / Transformer | Compute weighted context from all positions | Modern NLP, vision |
| Pooling | Downsample feature maps | Dimensionality reduction in CNNs |
| Dropout | Randomly zero activations during training | Regularization |
| Batch Normalization | Normalize activations across the batch | Faster convergence, stability |
| Layer Normalization | Normalize activations across features | Transformers, small batches |
| Embedding | Map discrete tokens to dense vectors | NLP, categorical features |

---

#### Activation Functions

**The problem:** Without activation functions, a deep network is a single linear map. But not every non-linearity is equal — some kill gradients, some produce dead neurons.

| Function | Formula | Range | Advantage | Failure Mode |
|---|---|---|---|---|
| ReLU | max(0, x) | [0, ∞) | Fast, no vanishing gradient for positives | Dead neurons (always output 0 for negative inputs) |
| Leaky ReLU | max(αx, x) | (-∞, ∞) | Fixes dead ReLU | Requires tuning α |
| GELU | x·Φ(x) | (-∞, ∞) | Smooth, empirically strong | Slower computation |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Outputs probabilities | Saturates → vanishing gradient |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Zero-centered | Saturates → vanishing gradient |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class probabilities summing to 1 | Output layer only |

ReLU is preferred for hidden layers because it avoids saturation in the positive domain and is computationally trivial. The dead ReLU problem (neurons that always output 0 after a large negative gradient update) is addressed by Leaky ReLU or by careful initialization and learning rate choice.

---

#### Loss Functions

**The problem:** You need a differentiable measure of how wrong the model's outputs are — one that is small when predictions are good and large when they are bad, so gradient descent has something to minimize.

| Loss Function | Formula | Use Case |
|---|---|---|
| MSE | (1/n)Σ(y-ŷ)² | Regression |
| MAE | (1/n)Σ\|y-ŷ\| | Regression (outlier-robust) |
| Binary Cross-Entropy | -[y·log(ŷ) + (1-y)·log(1-ŷ)] | Binary classification |
| Categorical Cross-Entropy | -Σy·log(ŷ) | Multi-class classification |
| Sparse Categorical CE | Same with integer labels | Multi-class with many classes |
| Hinge Loss | max(0, 1-y·ŷ) | SVMs, margin-based learning |
| Huber Loss | MSE for small errors, MAE for large | Outlier-robust regression |

---

#### Optimizers

**The problem:** Gradient descent updates all parameters with the same learning rate. Some parameters have gradients that are consistently large; others are sparse and rarely updated. A fixed learning rate is simultaneously too large for some parameters and too small for others.

| Optimizer | Key Mechanism | When to Use |
|---|---|---|
| SGD | Raw gradient × learning rate | Simple problems; with momentum, often best final performance |
| SGD + Momentum | Accumulates gradient history (velocity) | Faster convergence, better escape from local minima |
| Adam | Adaptive per-parameter rates using first and second gradient moments | Default for most deep learning |
| AdamW | Adam with decoupled weight decay | Better regularization; preferred for transformers |
| RMSprop | Adaptive rates using recent squared gradient | RNNs, non-stationary problems |
| AdaGrad | Per-parameter rates that decay as parameters are updated | Sparse data, NLP |

Adam's adaptive rates are computed from moving averages of the gradient (first moment, momentum-like) and the squared gradient (second moment, variance-like). Bias corrections during early training prevent these estimates from being near zero.

---

### Training Instabilities

#### Vanishing and Exploding Gradients

**The problem:** Backpropagation multiplies gradients through every layer via the chain rule. If each layer's gradient has magnitude < 1, the product over 100 layers approaches zero — early layers stop learning. If each has magnitude > 1, the product explodes.

**Vanishing gradient solutions:**
- Use ReLU activations (gradient is 1 in the positive domain, not < 1)
- Batch normalization (keeps activations in a range where gradients are well-scaled)
- Residual connections (skip paths provide gradient highways that bypass layers)
- Careful weight initialization (Xavier for sigmoid/tanh, He for ReLU)

**Exploding gradient solutions:**
- Gradient clipping: cap the gradient norm before applying the update
- Lower learning rate
- Batch normalization

---

#### Batch vs. Layer Normalization

**The problem:** As training progresses, the distribution of each layer's inputs shifts as earlier layers' weights change. This "internal covariate shift" forces each layer to constantly readjust to a moving target.

**The core insight:** Normalize activations to a standard distribution before passing them to the next layer. Re-introduce learnable scale and shift parameters so the layer can learn any distribution it needs — but start from a stable, well-conditioned baseline.

| Aspect | Batch Norm | Layer Norm |
|---|---|---|
| Normalizes over | Batch dimension | Feature dimension |
| Best for | CNNs, large batches | Transformers, RNNs, small batches |
| Training vs. inference | Uses running statistics at inference | Same computation at both |
| Batch size dependency | Requires large batches | Works with batch size 1 |

---

### Modern Architectures

#### Computer Vision

| Architecture | Year | Key Innovation | Problem It Solved |
|---|---|---|---|
| LeNet | 1998 | First successful CNN | Digit recognition |
| AlexNet | 2012 | Deep CNN, ReLU, Dropout | ImageNet at scale |
| VGG | 2014 | Uniform 3×3 convolution stacks | Depth systematically helps |
| ResNet | 2015 | Skip connections | Training networks deeper than ~20 layers without gradient death |
| Inception | 2015 | Parallel multi-scale filters | Efficient capture of features at different scales |
| MobileNet | 2017 | Depthwise separable convolutions | ImageNet-quality models on mobile hardware |
| EfficientNet | 2019 | Compound scaling (depth + width + resolution) | Optimal efficiency frontier |
| ViT | 2020 | Transformers applied to image patches | SOTA on large-data vision tasks |

#### Natural Language Processing

| Architecture | Year | Key Innovation | Problem It Solved |
|---|---|---|---|
| Word2Vec | 2013 | Word embeddings from co-occurrence | Words had no continuous representation |
| GloVe | 2014 | Global co-occurrence matrix factorization | Richer embeddings than local window only |
| LSTM / GRU | 1997/2014 | Gated cells for long-range memory | RNNs forgot context after ~10 steps |
| Transformer | 2017 | Self-attention, fully parallelizable | RNNs were sequential; couldn't use modern GPU parallelism |
| BERT | 2018 | Bidirectional masked language model | Language understanding required full context, not left-to-right only |
| GPT | 2018 | Autoregressive transformer decoder | Coherent long-form text generation |
| T5 | 2019 | Every NLP task as text-to-text | Unification eliminated task-specific architectures |
| GPT-3/4 | 2020/2023 | Scale (175B+ parameters) | Few-shot generalization without task-specific fine-tuning |

#### Generative Models

| Model | Mechanism | Use Case | Key Failure Mode |
|---|---|---|---|
| GAN | Generator and discriminator trained adversarially | Image generation, style transfer | Mode collapse, training instability |
| VAE | Encoder maps to latent distribution; decoder samples from it | Structured generation, interpolation | Blurry outputs (averaging over posterior) |
| Diffusion Models | Iterative denoising from Gaussian noise | DALL-E, Stable Diffusion | Slow inference (many denoising steps) |
| Autoregressive | Predict next token given all previous | GPT, language generation | Hallucination; no global coherence constraint |

---

### Training Techniques

#### Preventing Overfitting

**Dropout:** Randomly zero a fraction of activations during each training step. Forces the network to learn redundant representations — no single neuron can be relied upon. Equivalent to training an exponential ensemble of sub-networks. At inference, scale activations by (1 - dropout rate) to match expected activation magnitude.

**Batch / Layer Normalization:** Normalizing activations acts as mild regularization because the normalization obscures the exact activation values of any single training example.

**Weight Decay / L2 Regularization:** Penalizes large weights, shrinking the effective capacity of the model.

**Early Stopping:** Monitor validation loss during training. Stop when validation loss stops improving (and starts increasing). The model at the stopping point has the best generalization, not the model at the end of training.

**Data Augmentation:** Expand the effective training set by applying transformations that preserve the label. Vision: random crop, horizontal flip, color jitter, mixup. NLP: back-translation, synonym substitution.

---

#### Learning Rate Scheduling

**The problem:** A fixed learning rate is too large early in training (diverges from the optimum) and too large late in training (oscillates around the minimum instead of converging into it).

**Scheduling strategies:**
- **Warm-up then decay:** Start with a small learning rate, linearly increase to the target, then decay. Prevents early instability from large random gradients.
- **Cosine annealing:** Smoothly reduce LR following a cosine curve. Can restart periodically to explore different loss landscape regions.
- **Step decay:** Multiply LR by a fixed factor at predetermined epochs.

**Gradient Clipping:** Cap the gradient norm before applying updates. Critical for RNNs and transformers where gradient magnitudes vary wildly. Prevents single large gradient steps from destroying previously learned representations.

**Mixed Precision Training:** Compute forward and backward passes in FP16 (faster, half the memory) but maintain FP32 master weights for accumulation. Loss scaling prevents FP16 underflow of small gradients.

---

#### Transfer Learning and Fine-tuning

**The problem:** Training a competitive vision or language model from scratch requires millions of labeled examples and weeks of GPU compute. Most real tasks have neither.

**The core insight:** The features learned by large models on large datasets (edges, textures, objects in vision; syntax, semantics, world knowledge in language) are broadly useful. The model has already solved the hard part. You only need to adapt the final mapping to your specific task.

**Fine-tuning strategy:**
1. Load a pre-trained model (e.g., ResNet-50 trained on ImageNet, BERT trained on Wikipedia).
2. Replace the final layer(s) with a new head matching your output space.
3. Train the head with frozen base — learn the task-specific mapping without disturbing general features.
4. Optionally unfreeze the top layers of the base and continue with a very low learning rate (full fine-tuning).

**What breaks:** If your domain is very different from the pre-training domain (medical images vs. natural photos, legal text vs. general web), early layers may need updating too. Fine-tuning with too high a learning rate causes catastrophic forgetting — the model overwrites previously learned representations.

---

## Model Development Best Practices

### Debugging Checklist

When a model underperforms, work through these layers in order:

**Data quality first:**
- Missing values: are they handled the same way in training and inference?
- Data leakage: do any features contain information from the future, or from the label itself?
- Feature scaling: applied before splitting? (Leakage.) Applied consistently to train, val, and test?
- Class imbalance: addressed before training, not after?

**Feature engineering:**
- Is domain knowledge incorporated?
- Are interaction terms or polynomial features needed?
- Does feature importance analysis reveal irrelevant columns consuming capacity?

**Model complexity:**
- High training error + high validation error → high bias → increase capacity, reduce regularization
- Low training error + high validation error → high variance → reduce capacity, increase regularization, collect more data

**Hyperparameters:**
- Learning rate: the most important hyperparameter; search it first
- Regularization strength
- Architecture depth / width

---

### Production Considerations

#### Serving

**Batch inference:** Process large datasets offline. No strict latency requirement. Can use complex, slow models. Examples: overnight recommendation re-ranking, weekly churn scoring.

**Real-time inference:** Low latency (single-digit to hundreds of milliseconds). Model optimization is critical. Examples: search ranking, fraud detection, content moderation.

**Optimization levers for inference:**
- **Quantization:** Reduce weights from FP32 to INT8 or INT4. Large memory and latency reduction; small accuracy cost.
- **Pruning:** Remove weights with small magnitude. Can be structured (entire neurons) or unstructured.
- **Knowledge distillation:** Train a small student model to match the output distribution of the large teacher. The student is faster; it learns the teacher's soft probabilities, which contain more information than hard labels.
- **ONNX / TensorRT:** Hardware-optimized inference runtimes.

---

#### Model Monitoring and Drift

**The problem:** A model trained on past data faces future data whose distribution may have shifted. Nothing breaks visibly — the model still runs, still returns scores. Only the scores stop meaning what they used to.

**Data drift:** The input feature distribution changes over time. Detection: Population Stability Index (PSI), KL divergence between reference and current feature distributions. Fix: retrain on recent data.

**Concept drift:** The relationship between inputs and the target changes over time. Detection: monitor model performance metrics directly (requires ground truth labels, which arrive with lag). Fix: retrain, potentially with new features.

**Monitoring metrics:**
- Model accuracy / precision / recall (lagged; requires ground truth)
- Prediction score distribution
- Feature value distributions
- Inference latency and throughput
- Error rates

---

#### A/B Testing

**The problem:** You have a new model and want to know if it is actually better in production, not just on a held-out test set. Offline metrics do not always predict online business outcomes.

**The core insight:** Randomize users between old and new models. Any difference in outcomes is then caused by the model choice, not by confounding factors.

**Setup:**
1. Split traffic (e.g., 90% control / 10% treatment).
2. Monitor both model metrics (accuracy, latency) and business metrics (conversion rate, engagement, revenue).
3. Run until statistical significance is reached at a pre-specified sample size — do not stop early when it looks significant (peeking inflates false positive rate).
4. Gradual rollout if treatment wins.

---

#### Model Versioning and Reproducibility

Essential practices:
- Version control for code (Git)
- Data versioning (DVC, Delta Lake)
- Experiment tracking: log hyperparameters, metrics, and model artifacts (MLflow, Weights & Biases)
- Containerize environments for reproducibility (Docker)
- Set random seeds for all sources of stochasticity

---

## Deep Learning Architecture Details

### How Attention Works

**The problem:** RNNs process sequences step by step. To use information from 500 steps ago, it must be carried through 500 intermediate states, compressing and potentially losing it at each step. And the computation is inherently sequential — step t+1 cannot begin until step t finishes.

**The core insight:** Skip the sequential bottleneck entirely. Let every position attend directly to every other position in the sequence, with attention weights that are learned from the data.

**The mechanics:** Each input is projected into three vectors: Query (Q), Key (K), and Value (V).

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

The dot product QKᵀ measures how relevant each position is to each other position. Division by √d_k prevents the dot products from becoming very large in high dimensions (which would push softmax into near-zero gradient regions). The softmax converts raw scores into attention weights that sum to 1. These weights are used to compute a weighted sum of the Value vectors.

**What breaks:** Attention has O(N²) complexity in sequence length — computing all pairwise scores is quadratic. This is manageable for sequences of hundreds of tokens but becomes prohibitive for sequences of hundreds of thousands of tokens. Flash Attention addresses this by reordering operations to stay within fast GPU SRAM, avoiding the O(N²) materialization of the full attention matrix.

---

### Why Transformers Replaced RNNs

**The problem:** RNNs cannot parallelize across time steps during training. On a sequence of length 1000, you must run 1000 sequential matrix multiplications. Modern GPUs are designed for parallel computation — sequential algorithms waste most of the hardware.

**The core insight:** If you replace sequential recurrence with parallel attention, the entire sequence can be processed simultaneously on a GPU. The parallelism is what made training modern large language models computationally feasible.

**What breaks:**
- Transformers require positional encodings to represent sequence order, since attention itself is permutation-invariant.
- The O(N²) attention cost makes very long sequences expensive.
- Transformers have no inductive bias for sequences the way RNNs do — they learn sequence structure from data, which requires more data.

## Flashcards

**The absolute-value penalty has a corner at zero; the gradient is discontinuous there. Optimization pushes weights exactly to zero?** #flashcard
the model performs implicit feature selection.

**The squared penalty produces a smooth gradient everywhere. Weights shrink proportionally toward zero but never reach it exactly. Numerically stable; works well when all features contribute.?** #flashcard
The squared penalty produces a smooth gradient everywhere. Weights shrink proportionally toward zero but never reach it exactly. Numerically stable; works well when all features contribute.

**Combines both penalties. Handles correlated features better than pure L1 (which arbitrarily zeros one of a correlated pair).?** #flashcard
Combines both penalties. Handles correlated features better than pure L1 (which arbitrarily zeros one of a correlated pair).

**Training set (60–80%)?** #flashcard
The model sees and learns from this.

**Validation set (10–20%)?** #flashcard
Used to compare models and tune hyperparameters. Each time you look at validation performance to make a decision, you are implicitly using it as a training signal.

**Test set (10–20%)?** #flashcard
Touched exactly once, after all decisions are final. This produces the only honest performance estimate.

**Stratified K-Fold?** #flashcard
Preserves class distribution in each fold. Required for imbalanced classification.

**Time Series Split?** #flashcard
Each validation fold is strictly in the future relative to its training fold. Never shuffle time series before splitting.

**Leave-One-Out (LOO)?** #flashcard
K = N. Maximally uses data but is computationally expensive and produces high-variance estimates on noisy problems.

**Need interpretability?** #flashcard
Linear/Logistic Regression, Decision Tree

**High-dimensional data?** #flashcard
Lasso, Ridge, Random Forest

**Categorical features?** #flashcard
CatBoost, LightGBM

**Limited data?** #flashcard
Naive Bayes, regularized linear models

**Fast inference required?** #flashcard
Linear models, Naive Bayes

**Maximum accuracy on tabular?** #flashcard
XGBoost, LightGBM, CatBoost

**Oversample minority class?** #flashcard
SMOTE synthesizes new minority examples by interpolating between existing ones. ADASYN focuses synthesis near decision boundaries.

**Undersample majority class?** #flashcard
Randomly remove majority examples. Faster but discards potentially useful data.

**Combined?** #flashcard
Oversample minority + undersample majority.

**Class weights?** #flashcard
Pass class_weight='balanced' or manual weights to your loss function. The model pays more for minority misclassifications without touching the data.

**Focal loss?** #flashcard
Down-weights easy majority examples dynamically during training, forcing the model to focus on hard minority cases.

**Use ReLU activations (gradient is 1 in the positive domain, not < 1)?** #flashcard
Use ReLU activations (gradient is 1 in the positive domain, not < 1)

**Batch normalization (keeps activations in a range where gradients are well-scaled)?** #flashcard
Batch normalization (keeps activations in a range where gradients are well-scaled)

**Residual connections (skip paths provide gradient highways that bypass layers)?** #flashcard
Residual connections (skip paths provide gradient highways that bypass layers)

**Careful weight initialization (Xavier for sigmoid/tanh, He for ReLU)?** #flashcard
Careful weight initialization (Xavier for sigmoid/tanh, He for ReLU)

**Gradient clipping?** #flashcard
cap the gradient norm before applying the update

**Lower learning rate?** #flashcard
Lower learning rate

**Batch normalization?** #flashcard
Batch normalization

**Warm-up then decay?** #flashcard
Start with a small learning rate, linearly increase to the target, then decay. Prevents early instability from large random gradients.

**Cosine annealing?** #flashcard
Smoothly reduce LR following a cosine curve. Can restart periodically to explore different loss landscape regions.

**Step decay?** #flashcard
Multiply LR by a fixed factor at predetermined epochs.

**Missing values?** #flashcard
are they handled the same way in training and inference?

**Data leakage?** #flashcard
do any features contain information from the future, or from the label itself?

**Feature scaling?** #flashcard
applied before splitting? (Leakage.) Applied consistently to train, val, and test?

**Class imbalance?** #flashcard
addressed before training, not after?

**Is domain knowledge incorporated?** #flashcard
Is domain knowledge incorporated?

**Are interaction terms or polynomial features needed?** #flashcard
Are interaction terms or polynomial features needed?

**Does feature importance analysis reveal irrelevant columns consuming capacity?** #flashcard
Does feature importance analysis reveal irrelevant columns consuming capacity?

**High training error + high validation error → high bias → increase capacity, reduce regularization?** #flashcard
High training error + high validation error → high bias → increase capacity, reduce regularization

**Low training error + high validation error → high variance → reduce capacity, increase regularization, collect more data?** #flashcard
Low training error + high validation error → high variance → reduce capacity, increase regularization, collect more data

**Learning rate?** #flashcard
the most important hyperparameter; search it first

**Regularization strength?** #flashcard
Regularization strength

**Architecture depth / width?** #flashcard
Architecture depth / width

**Quantization?** #flashcard
Reduce weights from FP32 to INT8 or INT4. Large memory and latency reduction; small accuracy cost.

**Pruning?** #flashcard
Remove weights with small magnitude. Can be structured (entire neurons) or unstructured.

**Knowledge distillation?** #flashcard
Train a small student model to match the output distribution of the large teacher. The student is faster; it learns the teacher's soft probabilities, which contain more information than hard labels.

**ONNX / TensorRT?** #flashcard
Hardware-optimized inference runtimes.

**Model accuracy / precision / recall (lagged; requires ground truth)?** #flashcard
Model accuracy / precision / recall (lagged; requires ground truth)

**Prediction score distribution?** #flashcard
Prediction score distribution

**Feature value distributions?** #flashcard
Feature value distributions

**Inference latency and throughput?** #flashcard
Inference latency and throughput

**Error rates?** #flashcard
Error rates

**Version control for code (Git)?** #flashcard
Version control for code (Git)

**Data versioning (DVC, Delta Lake)?** #flashcard
Data versioning (DVC, Delta Lake)

**Experiment tracking?** #flashcard
log hyperparameters, metrics, and model artifacts (MLflow, Weights & Biases)

**Containerize environments for reproducibility (Docker)?** #flashcard
Containerize environments for reproducibility (Docker)

**Set random seeds for all sources of stochasticity?** #flashcard
Set random seeds for all sources of stochasticity

**Transformers require positional encodings to represent sequence order, since attention itself is permutation-invariant.?** #flashcard
Transformers require positional encodings to represent sequence order, since attention itself is permutation-invariant.

**The O(N²) attention cost makes very long sequences expensive.?** #flashcard
The O(N²) attention cost makes very long sequences expensive.

**Transformers have no inductive bias for sequences the way RNNs do?** #flashcard
they learn sequence structure from data, which requires more data.
