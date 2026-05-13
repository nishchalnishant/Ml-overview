# Machine Learning Glossary: Quick Revision Guide

**How to use this:** Treat it like **liner notes** — skim the letters you’re rusty on before a call. Entries stay dense on purpose: definitions, formulas, **when a metric lies in production**, and interview sound bites.

**DevOps cheat:** If a term maps to something you already run, you’ve won half the battle — **A/B testing** is still A/B testing; **drift** is “silent dependency rot,” but for **data**.

Each term below still follows the cheat-sheet pattern: definitions, formulas, practical applications, and interview key points.

---

## A

**Accuracy**
- **Definition:** The ratio of correct predictions to total predictions.
- **Formula:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Practical Application:** Use mostly for **balanced** datasets (e.g., handwritten digit classification).
- **Interview Note:** **Never** use accuracy for fraud detection or medical diagnosis (imbalanced classes). Use F1 or PRC-AUC instead.

**Activation Function**
- **Definition:** Non-linear transformation applied to neuron outputs to enable learning complex patterns.
- **Key Examples:**
  - **ReLU:** $max(0, x)$ (Most usage).
  - **Sigmoid:** $\frac{1}{1+e^{-x}}$ (Binary output).
  - **Softmax:** Multi-class probabilities.
- **Practical Application:** ReLU for hidden layers (avoids vanishing gradient), Sigmoid/Softmax for output layers.

**AdaBoost (Adaptive Boosting)**
- **Definition:** An ensemble method that trains weak learners sequentially, correcting the mistakes of previous predictors.
- **Key Concept:** Assigns higher weights to misclassified training instances.
- **Practical Application:** Tabular classification tasks where data is clean but hard to separate.
- **Interview Note:** Sensitive to noisy data and outliers because it frantically tries to fit them.

**Adam (Adaptive Moment Estimation)**
- **Definition:** An adaptive learning rate optimization algorithm.
- **Key Concept:** Combines **Momentum** (keeps moving in average direction) and **RMSprop** (scales learning rate by variance).
- **Practical Application:** The default optimizer for training Deep Learning models (LLMs, CNNs).
- **Interview Note:** Often set learning rate $\alpha = 3e-4$ (Andrej Karpathy's "safe bet").

**AUC-ROC (Area Under Curve)**
- **Definition:** Performance metric measuring the ability to distinguish between classes at all threshold settings.
- **Key Range:** 0.5 (Random Guessing) to 1.0 (Perfect).
- **Practical Application:** Comparing models for credit scoring or ad-click prediction.
- **Interview Note:** Unlike Accuracy, AUC is **threshold-invariant** and **scale-invariant**.

**Attention Mechanism**
- **Definition:** Allows a model to focus on specific parts of the input sequence when generating output.
- **Formula:** $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
- **Practical Application:** The core of Transformers (ChatGPT, BERT). Enables long-range dependency modeling in translation.

---

## B

**Backpropagation**
- **Definition:** The algorithm for computing gradients of the loss function with respect to weights using the chain rule.
- **Key Concept:** "Credit assignment" — figuring out which weight contributed how much to the error.
- **Practical Application:** The engine of training Neural Networks.

**Bagging (Bootstrap Aggregating)**
- **Definition:** Training multiple models in parallel on random subsets (with replacement) of training data.
- **Key Example:** **Random Forest**.
- **Practical Application:** Reducing **Variance** (overfitting). Great for high-variance models like Decision Trees.

**Batch Normalization**
- **Definition:** Normalizing layer inputs to have mean 0 and variance 1 for each mini-batch.
- **Formula:** $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
- **Practical Application:** Accelerates training, allows higher learning rates, and acts as a weak regularizer.
- **Interview Note:** During **Inference**, use the moving average of mean/var calculated during training, not the batch statistics.

**Bias-Variance Tradeoff**
- **Definition:** The tension between a model's ability to minimize errors on training data (Bias) vs. unseen data (Variance).
- **Key Insight:**
  - **High Bias:** Underfitting (Linear Regression on nonlinear data).
  - **High Variance:** Overfitting (100-depth Decision Tree).
- **Goal:** Find the "Sweet Spot".

**Binary Cross-Entropy (Log Loss)**
- **Definition:** Loss function for binary classification.
- **Formula:** $-\frac{1}{N} \sum [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$
- **Practical Application:** Evaluating probabilities in spam detection or churn prediction.

---

## C

**Confusion Matrix**
- **Definition:** A table layout that visualizes the performance of a supervised learning algorithm.
- **Components:**
  - **TP:** Hit.
  - **TN:** Correct Rejection.
  - **FP:** False Alarm (Type I Error).
  - **FN:** Miss (Type II Error).

**Cosine Similarity**
- **Definition:** Measure of similarity between two non-zero vectors using the cosine of the angle between them.
- **Formula:** $\frac{A \cdot B}{||A|| ||B||}$
- **Practical Application:** Semantic Search, RAG (Retrieval Augmented Generation), Document Similarity.
- **Interview Note:** Range is [-1, 1]. In high-dimensional spaces (embeddings), usually [0, 1].

**Cross-Validation (K-Fold)**
- **Definition:** Resampling procedure used to evaluate ML models on a limited data sample.
- **Practical Application:** Validating that your model isn't just memorizing the specific train-test split.
- **Interview Note:** For Time Series, use **TimeSeriesSplit** (Walk-Forward validation), never random K-Fold.

---

## D

**Data Leakage**
- **Definition:** When information from outside the training dataset (or from the future) is used to create the model.
- **Examples:** Using 'Target' in feature engineering, scaling data before splitting, future timestamps.
- **Interview Note:** If you see 99.9% accuracy, suspect leakage immediately.

**Dimensionality Reduction**
- **Definition:** Transformation of data from a high-dimensional space into a low-dimensional space.
- **Techniques:**
  - **PCA (Linear):** Preserves variance.
  - **t-SNE / UMAP (Non-linear):** Preserves local structure/clusters.
- **Practical Application:** Visualizing embeddings, reducing noise, speeding up training.

**Dropout**
- **Definition:** Regularization technique where randomly selected neurons are ignored during training.
- **Key Concept:** Prevents neurons from co-adapting too much (relying on specific peers).
- **Practical Application:** Standard in almost all non-convolutional layers in Deep Learning.

---

## E

**Eigenvalue / Eigenvector**
- **Definition:** For a matrix $A$, $Av = \lambda v$. $v$ is the eigenvector (direction), $\lambda$ is eigenvalue (magnitude).
- **Practical Application:** PCA (Principal Component Analysis) projects data onto the eigenvectors with largest eigenvalues.

**Embedding**
- **Definition:** A relatively low-dimensional space into which high-dimensional vectors can be translated.
- **Key Concept:** Semantic meaning. "King" - "Man" + "Woman" $\approx$ "Queen".
- **Practical Application:** Word2Vec, BERT embeddings, Recommender Systems users/items.

**Entropy (Shannon)**
- **Definition:** Measure of uncertainty or impurity in a dataset.
- **Formula:** $H(X) = - \sum p(x) \log p(x)$
- **Practical Application:** Decision Trees use this (Information Gain) to decide split points.

---

## F

**F1 Score**
- **Definition:** The harmonic mean of Precision and Recall.
- **Formula:** $2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Practical Application:** The "single number" metric for imbalanced classification.
- **Interview Note:** Harmonic mean punishes extreme values more than arithmetic mean (if Recall=0, F1=0).

**Fine-Tuning**
- **Definition:** Taking a pre-trained model (e.g., Llama-2) and training it further on a specific dataset.
- **Types:**
  - **Full Fine-Tuning:** Update all weights.
  - **PEFT (LoRA):** Update only a small subset of adapters.
- **Practical Application:** Customizing an LLM for legal document analysis.

---

## G

**Gradient Descent**
- **Definition:** An iterative optimization algorithm for finding the local minimum of a function.
- **Formula:** $\theta_{new} = \theta_{old} - \alpha \nabla J(\theta)$ ($LearningRate \times Gradient$).
- **Practical Application:** The fundamental way nearly all metrics are minimized in ML.

**GAN (Generative Adversarial Network)**
- **Definition:** Two neural networks contested with each other in a game.
  - **Generator:** Creates fakes.
  - **Discriminator:** Detects fakes.
- **Practical Application:** DeepFakes, Image Super-resolution, Style Transfer.

---

## H

**Hyperparameter Tuning**
- **Definition:** Choosing the optimal set of parameters that govern the training process (not learned by the model).
- **Methods:**
  - **Grid Search:** Brute force.
  - **Random Search:** Surprisingly effective.
  - **Bayesian Optimization:** Smarter, probabilistic search.

---

## I

**Imbalanced Data**
- **Definition:** A dataset with a skewed class distribution (e.g., 1000 : 1).
- **Solutions:**
  - Resampling (SMOTE, Undersampling).
  - Class Weights (Change loss function).
  - Metric Choice (Use F1/AUC, not Accuracy).

**IOU (Intersection over Union)**
- **Definition:** Metric used to measure the accuracy of an object detector.
- **Formula:** $\frac{Area(Overlap)}{Area(Union)}$
- **Practical Application:** Evaluating bounding boxes in YOLO / R-CNN.

---

## K

**K-Means Clustering**
- **Definition:** Iterative algorithm that partitions data into $K$ clusters.
- **Key Concept:** Minimizes variance within clusters.
- **Interview Note:** Requires specifying $K$ beforehand (use Elbow Method to find optimal K).

**KL Divergence (Kullback-Leibler)**
- **Definition:** Measure of how one probability distribution distinguishes from a second, reference probability distribution.
- **Formula:** $D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$
- **Practical Application:** Loss function in VAEs (Variational Autoencoders) and t-SNE.

---

## L

**Learning Rate**
- **Definition:** Hyperparameter controlling how much we change the model in response to the estimated error each time the model weights are updated.
- **Practical Application:** Too high = diverge. Too low = slow convergence.
- **Tip:** Use a Scheduler (Cosine Decay, Warmup).

**LSTM (Long Short-Term Memory)**
- **Definition:** A type of RNN capable of learning long-term dependencies.
- **Key Mechanics:** Input Gate, Forget Gate, Output Gate.
- **Practical Application:** Time-series forecasting, older NLP translation models.

---

## M

**MSE / MAE / RMSE**
- **MSE (Mean Squared Error):** Penalizes large errors heavily. Differentiable.
- **MAE (Mean Absolute Error):** Robust to outliers. Not differentiable at 0.
- **RMSE:** Root of MSE. Interpretability same units as target.

**Momentum**
- **Definition:** Technique to accelerate Gradient Descent by accumulating a velocity vector in directions of persistent reduction in the objective function.
- **analogy:** Heavy ball rolling down a hill (it gains speed).

---

## N

**Normalization vs. Standardization**
- **Normalization (Min-Max):** Scales data to [0, 1]. Best for image data / bounded ranges.
- **Standardization (Z-Score):** Scales data to $\mu=0, \sigma=1$. Best for algorithms assuming Gaussian distribution (SVM, Logistic Regression).

---

## O

**Overfitting**
- **Definition:** When a model learns the detail and noise in the training data to the extent that it negatively impacts performance on new data.
- **The Fix:**
  - More data.
  - Regularization (L1/L2/Dropout).
  - Simpler Model.
  - Early Stopping.

---

## P

**PCA (Principal Component Analysis)**
- **Definition:** Linear dimensionality reduction.
- **Key Concept:** Finds axes (Principal Components) that maximize variance.
- **Interview Note:** Sensitive to scale (must standardise data first!).

**Precision & Recall**
- **Precision:** $\frac{TP}{TP + FP}$ (Quality). "Of all the spam we detected, how much was actually spam?"
- **Recall:** $\frac{TP}{TP + FN}$ (Quantity). "Of all the actual spam, how much did we find?"
- **Tradeoff:** Increasing threshold increases Precision but decreases Recall.

---

## R

**RAG (Retrieval-Augmented Generation)**
- **Definition:** Enhancing LLMs by retrieving relevant documents from an external knowledge base before generating an answer.
- **Components:** Vector DB + Embedding Model + LLM.
- **Practical Application:** "Chat with PDF", Enterprise Search.

**Regularization**
- **L1 (Lasso):** Adds absolute interactions. Can shrink coefficients to **zero** (Feature Selection).
- **L2 (Ridge):** Adds squared interactions. Shrinks coefficients towards zero but not *to* zero.

**ReLU (Rectified Linear Unit)**
- **Definition:** Activation function $f(x) = max(0, x)$.
- **Why?** Computationally cheap, solves vanishing gradient in positive domain.

---

## S

**Softmax**
- **Definition:** Function that turns a vector of $K$ real values into a vector of $K$ real values that sum to 1.
- **Formula:** $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$
- **Practical Application:** Final layer of Multi-Class Classification.

**SGD (Stochastic Gradient Descent)**
- **Definition:** Gradient descent using only a single sample (or mini-batch) to calculate the gradient.
- **Why?** Faster per step, adds noise which helps escape local minima.

**SVM (Support Vector Machine)**
- **Definition:** Finds the hyperplane that maximizes the **margin** between two classes.
- **Key Concept:** Kernel Trick (maps non-linear data to higher dimensions where it becomes linear).

---

## T

**Transformer**
- **Definition:** Deep learning architecture based entirely on the Attention mechanism.
- **Key Benefit:** Parallelizable (unlike RNNs) and captures long-term dependencies perfectly.

**Transfer Learning**
- **Definition:** Storing knowledge gained while solving one problem and applying it to a different but related problem.
- **Practical Application:** Using ResNet trained on ImageNet to classify medical X-rays.

---

## V

**Vanishing Gradient**
- **Definition:** In deep networks, gradients can shrink exponentially as they backpropagate, effectively stopping early layers from training.
- **Solution:** ResNet (Skip connections), ReLU, BatchNorm.

**Vector Database**
- **Definition:** A database optimized for storing and querying high-dimensional vectors.
- **Examples:** Pinecone, Milvus, Chroma.
- **Practical Application:** The "Long Term Memory" for LLM Agents.

---

## Z

**Zero-Shot Learning**
- **Definition:** The ability of a model to recognize objects or perform tasks it has not seen during training.
- **Example:** Asking GPT-4 to categorize text into "Happy/Sad" without giving it examples, simply by describing the task.

---

## New & Advanced Terms (2024-2025)

**SHAP (SHapley Additive exPlanations)**
- **Definition:** Feature attribution method from cooperative game theory — assigns each feature fair credit for a prediction.
- **Intuition:** Dividing prize money among teammates based on their marginal contribution to every possible coalition.
- **Types:** TreeSHAP (exact, fast for trees), KernelSHAP (model-agnostic), DeepSHAP (neural nets).

**LIME (Local Interpretable Model-agnostic Explanations)**
- **Definition:** Explains individual predictions by fitting a local linear model on perturbed samples around the point.
- **Limitation:** Unstable across runs; neighborhood definition is sensitive.

**Graph Neural Network (GNN)**
- **Definition:** Neural network operating on graph-structured data by aggregating information from neighboring nodes via message passing.
- **Variants:** GCN (spectral), GraphSAGE (inductive, samples neighbors), GAT (attention-weighted aggregation).

**Message Passing**
- **Definition:** Core GNN operation — each node aggregates features from neighbors, updates its own state, repeated L layers.
- **Over-smoothing:** Too many layers cause all nodes to converge to identical representations.

**Markov Decision Process (MDP)**
- **Definition:** Framework for sequential decision-making: (S, A, P, R, γ) — states, actions, transitions, rewards, discount.
- **Markov Property:** Next state depends only on current state and action, not history.

**Q-Learning**
- **Definition:** Model-free RL that learns Q(s,a) — expected return from action a in state s via Bellman updates.
- **DQN:** Q-learning + neural net + experience replay + target network.

**PPO (Proximal Policy Optimization)**
- **Definition:** Policy gradient RL with clipped objective to prevent destructively large policy updates.
- **Relevance:** The algorithm behind RLHF alignment in InstructGPT/ChatGPT.

**Two-Tower Model**
- **Definition:** Recommender architecture with separate user and item encoders; retrieval via dot product + ANN search.
- **Used by:** YouTube, Pinterest, Google Play.

**Collaborative Filtering**
- **Definition:** Recommendations based on user-item interaction patterns — "users like you also liked X."
- **Cold Start:** New users/items have no history — fallback to content features or popularity.

**NDCG (Normalized Discounted Cumulative Gain)**
- **Definition:** Ranking metric rewarding high-relevance items at top positions, discounted logarithmically by rank.

**Differential Privacy (DP)**
- **Definition:** Adding calibrated noise ensures any single person's data can't be detected in model outputs (controlled by ε).
- **DP-SGD:** Clips per-example gradients + adds Gaussian noise during training.

**Federated Learning**
- **Definition:** Train across decentralized devices without centralizing raw data. FedAvg: clients train locally, server averages updates.

**Mamba / SSM (Selective State Space Model)**
- **Definition:** Linear-time sequence model with input-dependent state selection — alternatives to O(N²) Transformer attention.

**Test-Time Scaling**
- **Definition:** Improving outputs by allocating more inference compute (reasoning tokens, MCTS) rather than training larger models.
- **Examples:** OpenAI o1, DeepSeek-R1.

**Knowledge Distillation**
- **Definition:** Training a small student model to match a large teacher's soft output probabilities, not just hard labels.
- **Temperature:** Raised during distillation to soften probability distributions and expose inter-class similarities.

**MoE (Mixture of Experts)**
- **Definition:** Sparse architecture routing each input token to a subset of "expert" sub-networks. Large parameter count, small active compute.
- **Challenge:** Load balancing — preventing expert collapse.

**Causal Inference**
- **Definition:** Estimating the causal effect of a variable on an outcome beyond correlation.
- **Tools:** RCTs (A/B tests), propensity matching, DiD, instrumental variables, causal graphs.

**Confounder**
- **Definition:** Variable affecting both treatment and outcome, creating spurious correlation.
- **Example:** Hot weather causes both ice cream sales and drowning — the two are not causally linked.

**PagedAttention**
- **Definition:** KV cache management treating GPU memory as virtual memory pages — enables vLLM's 24× throughput gain.

**LoRA (Low-Rank Adaptation)**
- **Definition:** Adds trainable low-rank matrices (A×B) alongside frozen weights. Reduces trainable params by ~10,000×.

**DPO (Direct Preference Optimization)**
- **Definition:** Trains on preference pairs (chosen vs rejected) without a separate reward model or PPO.

**CLIP (Contrastive Language-Image Pretraining)**
- **Definition:** Aligns image and text embeddings via contrastive loss on 400M pairs. Enables zero-shot vision classification.

**Speculative Decoding**
- **Definition:** Small draft model generates tokens; large model verifies many in parallel. 2-4× speedup with no quality loss.

**Flash Attention**
- **Definition:** Hardware-aware exact attention using tiling to keep computation in fast SRAM — avoids materializing N×N matrix.

**RAGAS**
- **Definition:** Evaluation framework for RAG: faithfulness, answer relevancy, context precision/recall.

**Constitutional AI (CAI)**
- **Definition:** Alignment technique using a "constitution" of principles — model critiques and revises its own outputs.

**DP-SGD**
- **Definition:** SGD with per-example gradient clipping + Gaussian noise — provides formal differential privacy guarantees during training.
