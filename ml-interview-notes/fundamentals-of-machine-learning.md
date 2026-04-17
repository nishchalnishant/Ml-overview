# Fundamentals of Machine Learning

These are core interview questions. A good answer should do three things quickly: define the concept, explain why it matters, and mention one practical implication or tradeoff.

---

# Q1: Explain Epoch, Batch, Batch Size, and Iteration.

**Interview-ready answer**

An epoch is one full pass through the entire training dataset. A batch is a subset of the data used for one parameter update, and batch size is how many examples that subset contains. An iteration is one optimizer update, so the number of iterations per epoch is roughly the number of batches needed to cover the dataset. The practical takeaway is that these terms describe how training work is organized and they directly affect memory use, gradient noise, and training speed.

---

# Q2: What are embeddings in Machine Learning?

**Interview-ready answer**

Embeddings are learned dense vector representations of discrete entities such as words, users, products, or categories. The goal is to place similar or behaviorally related entities near each other in a continuous space so the model can generalize across them. They are especially useful when the raw representation would be sparse or high-cardinality, such as one-hot encoded IDs.

**Good nuance**

Embeddings are not just compression; they are a learned representation that can encode semantic or behavioral similarity.

---

# Q3: What is Softmax Activation Function?

**Interview-ready answer**

Softmax converts a vector of logits into a probability distribution over classes by exponentiating and normalizing the scores so they sum to one. It is typically used at the output of multiclass classifiers. The important interview point is that softmax does not create information; it just turns relative scores into normalized probabilities, which is why it is usually paired with cross-entropy loss.

---

# Q4: What is Machine Learning?

**Interview-ready answer**

Machine learning is the process of learning patterns or decision rules from data rather than manually specifying those rules. More formally, we choose a model family and optimize its parameters so it performs well on a task such as prediction, classification, ranking, or control. The real challenge is not fitting the training data, but generalizing well to unseen data under realistic operating conditions.

---

# Q5: Differentiate between Supervised and Unsupervised Learning.

**Interview-ready answer**

Supervised learning uses labeled examples, so the model learns a mapping from inputs to known targets. Unsupervised learning uses only inputs and tries to discover structure such as clusters, latent factors, or compressed representations. In interviews, it helps to add that many real systems are hybrid, for example semi-supervised or self-supervised approaches that sit between the two extremes.

---

# Q6: What is Reinforcement Learning?

**Interview-ready answer**

Reinforcement learning is a framework in which an agent learns to take actions in an environment to maximize cumulative reward. Unlike supervised learning, the agent is not told the correct action directly; it has to learn from delayed and possibly sparse feedback. That makes credit assignment, exploration, and long-term planning central challenges.

---

# Q7: What is Bias?

**Interview-ready answer**

In the bias-variance context, bias is the error introduced by overly restrictive assumptions in the model. A high-bias model is too simple to capture the true pattern and tends to underfit. In interviews, it is useful to distinguish this from societal or fairness bias, because the word "bias" is used in both statistical and ethical contexts.

---

# Q8: What is the difference between Classification and Regression?

**Interview-ready answer**

Classification predicts a discrete label or class probability, while regression predicts a continuous value. The difference affects the loss function, evaluation metric, and interpretation of output. For example, logistic regression and cross-entropy are appropriate for binary classification, whereas squared error is a natural fit for continuous regression targets.

---

# Q9: Explain Overfitting and Underfitting. How can you prevent them?

**Interview-ready answer**

Overfitting happens when the model learns patterns that are too specific to the training data, including noise, so generalization suffers. Underfitting happens when the model is too simple or constrained to capture the main signal. I prevent overfitting through better validation, regularization, early stopping, more data, or simpler models, and I address underfitting by improving features, reducing excessive regularization, or using a more expressive model.

**Good nuance**

The same model can overfit some slices and underfit others, so aggregate metrics do not tell the full story.

---

# Q10: What Are L1 and L2 Loss Functions?

**Interview-ready answer**

L1 loss measures absolute error, while L2 loss measures squared error. L1 is more robust to outliers because large errors grow linearly. L2 penalizes large errors more strongly, which can be useful when big mistakes are especially costly. The choice depends on the error distribution and the business cost of large deviations.

---

# Q11: What is Regularization? Explain L1 (Lasso) and L2 (Ridge) regularization.

**Interview-ready answer**

Regularization adds a penalty or constraint that discourages overly complex models and improves generalization. L1 regularization penalizes the absolute value of coefficients and tends to push some of them to exactly zero, which can produce sparse solutions. L2 regularization penalizes squared coefficients and shrinks them smoothly toward zero, which often improves stability without feature elimination.

**Good nuance**

In deep learning, people often say "weight decay" rather than L2 regularization, and the implementation details matter for optimizers like AdamW.

---

# Q12: What are Loss Functions and Cost Functions? Explain the key difference between them.

**Interview-ready answer**

A loss function usually refers to the error on a single example, while a cost function refers to the aggregate objective over the dataset, often including regularization. In casual usage people often use the terms interchangeably, but the distinction matters conceptually: the optimizer minimizes the dataset-level objective, not a single-example error in isolation.

---

# Q13: What are dropouts?

**Interview-ready answer**

Dropout is a regularization technique where units are randomly zeroed during training so the network cannot rely too heavily on any single activation path. This encourages redundancy in the representation and reduces co-adaptation between neurons. At inference time dropout is turned off, so the model uses the full network. The main interview point is that dropout helps generalization, not just training noise.

---

# Q14: What is a Perceptron?

**Interview-ready answer**

A perceptron is the simplest linear binary classifier: it computes a weighted sum of features and applies a threshold to decide the class. Historically it is important because it introduced the idea of learning weights from data, but its expressive power is limited because it can only solve linearly separable problems. That limitation motivated multilayer neural networks.

---

# Q15: Explain Multilayer Perceptron (MLP).

**Interview-ready answer**

An MLP is a feedforward neural network made of fully connected layers and non-linear activations. The hidden layers allow it to learn non-linear decision boundaries, which makes it much more expressive than a single perceptron. In practice, MLPs are general-purpose function approximators, although for images, text, and sequences we usually prefer architectures with stronger inductive bias such as CNNs or transformers.

---

# Q16: What is Cross-Entropy?

**Interview-ready answer**

Cross-entropy measures how well a predicted probability distribution matches the target distribution. In standard classification with one-hot labels, it penalizes the model for assigning low probability to the correct class and heavily punishes confident wrong predictions. That is why it is the default loss for classification: it aligns naturally with likelihood and probability estimation.

---

# Q17: What are Logits?

**Interview-ready answer**

Logits are the raw, unnormalized scores produced by a model before a final activation such as sigmoid or softmax. They live on the real line and are convenient for numerical optimization because losses like cross-entropy are typically implemented directly from logits for stability. A strong answer should mention that probabilities are derived from logits, not the other way around.

---

# Q18: Explain Cross-Validation. Why is it used?

**Interview-ready answer**

Cross-validation evaluates a model across multiple train-validation splits so performance is not overly dependent on one arbitrary split. It is especially useful when data is limited and you want a more stable estimate of generalization. The best interview answer also mentions that the split method must match the problem, such as stratified folds for imbalance or time-based validation for temporal data.

---

# Q19: What are precision, recall, and F1-score?

**Interview-ready answer**

Precision tells you how trustworthy positive predictions are, recall tells you how much of the positive class you recover, and F1 summarizes their balance. I would use precision when false positives are costly, recall when false negatives are costly, and F1 when I need a single metric that does not let one of those collapse unnoticed. The important point is that these metrics reflect different business priorities.

---

# Q20: What is anomaly detection?

**Interview-ready answer**

Anomaly detection is the task of identifying unusual or rare patterns that differ significantly from normal behavior. It is often used when labeled anomalies are scarce, such as fraud, network intrusion, or equipment failure. The key challenge is that anomalies are rare, evolving, and context-dependent, so the problem is usually as much about defining normal behavior and monitoring drift as it is about the model itself.

---

# Q21: What is the difference between policy-based and value-based methods?

**Interview-ready answer**

In reinforcement learning, value-based methods learn how good states or state-action pairs are and derive a policy from those value estimates. Policy-based methods learn the policy directly by optimizing expected reward. Value-based methods can be more sample-efficient in discrete action spaces, while policy-based methods are often better for continuous actions and can model stochastic behavior more naturally.

---

# Q22: What is Q-Learning?

**Interview-ready answer**

Q-learning is an off-policy reinforcement learning algorithm that learns the action-value function `Q(s, a)`, which estimates the expected discounted reward of taking action `a` in state `s` and then following the best policy. Once you have Q-values, the policy is simply to choose the action with the highest estimated value. Its importance comes from showing that you can learn good control behavior from reward signals without directly learning a policy first.

---

# Q23: Explain the concept of exploration vs exploitation.

**Interview-ready answer**

Exploration means trying actions that may yield new information, while exploitation means choosing the action that currently looks best. The tension matters because if you only exploit, you may get stuck with a suboptimal policy, and if you explore too much, you waste reward. Good RL systems balance the two, often through epsilon-greedy policies, uncertainty estimates, or more advanced bandit and policy optimization methods.

---

# Q24: Explain the curse of dimensionality and how to address it.

**Interview-ready answer**

The curse of dimensionality refers to the fact that as feature dimension increases, data becomes sparse, distances become less informative, and many methods need exponentially more data to generalize well. This is why nearest-neighbor methods, density estimation, and clustering often degrade in high dimensions. Common fixes are feature selection, dimensionality reduction, regularization, stronger inductive bias, and collecting more relevant data rather than simply adding features.

---

# Q25: Explain Local Loss, Focal Loss, and Gradient Blending in Multi-Task Learning.

**Interview-ready answer**

In multi-task learning, different tasks can compete for capacity and produce gradients of very different scale or usefulness. Focal loss is used mainly in imbalanced classification to down-weight easy examples and focus learning on hard ones. Local losses attach supervision to intermediate layers so lower-level representations receive direct learning signals. Gradient blending refers to strategies that balance or reweight gradients across tasks so one task does not dominate the shared network. The deeper point is that multi-task learning is not just "sum the losses"; you often need to manage task interference explicitly.
