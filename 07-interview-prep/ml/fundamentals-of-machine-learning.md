---
module: Interview Prep
topic: Ml
subtopic: Fundamentals Of Machine Learning
status: unread
tags: [interviewprep, ml, ml-fundamentals-of-machine-lea]
---
# Fundamentals of Machine Learning

---

## 1. What Is Machine Learning?

**What the interviewer is testing**: whether your definition reveals the engineering problem being solved, not just a textbook sentence about "learning from data."

**The reasoning structure**: start with the question "what problem does ML solve that traditional software cannot?" Traditional software encodes logic explicitly: `if income > 50000 and credit_score > 700, approve loan`. This breaks the moment the real boundary involves 500 interacting features, is non-linear, or shifts over time. Writing those rules by hand is not a tractable engineering problem.

Machine learning inverts this: instead of writing rules, you specify a model family and an optimization objective, then let the optimizer discover the rules from examples. This is powerful and introduces one failure mode that traditional software does not have — the model can appear to work correctly (low training loss) while learning correlations that will not hold on new data. The difference between "memorizing training examples" and "learning the pattern" is the entire technical challenge.

**The pattern in action**: a spam filter written with hand-crafted keyword rules fails the moment spammers change their vocabulary. An ML spam filter adapts as new data arrives. But the ML version will also learn "emails mentioning Nigeria are spam" — accurate historically, but a proxy for the real pattern. Traditional software fails at scale; ML can fail subtly on unseen inputs.

**Common traps**:
- defining ML as "the model learns from data" without saying what generalization means — any system can memorize a training set; the hard part is performing on examples that were never seen during training
- not recognizing that the training objective and the true goal can diverge — a model that minimizes cross-entropy on training data is not necessarily doing what the business wants; these need to be audited separately

---

## 2. Supervised vs Unsupervised vs Reinforcement Learning

**What the interviewer is testing**: whether you understand the distinction as a consequence of what feedback is available during training, not just as a taxonomy to recite.

**The reasoning structure**: the three paradigms exist because different problems provide different types of feedback.

Supervised learning has labeled examples — you know the right answer for each training point. This is the richest feedback signal; you can directly measure how wrong each prediction is and adjust. The bottleneck is label acquisition, which is expensive and often introduces human error or bias.

Unsupervised learning has no labels. You only have inputs. The goal shifts from "predict a target" to "find structure in the input distribution." The problem is harder because there is no ground truth against which to measure correctness — "structure" is inherently underspecified by the data alone.

Reinforcement learning has neither fixed inputs nor fixed labels. An agent acts, an environment responds with reward (potentially delayed by many steps), and the agent tries to discover which actions in which states lead to high cumulative reward. The defining challenge is credit assignment: when you win a chess game on move 60, which of the 60 moves deserves credit?

**The pattern in action**:
- fraud detection with historical labeled transactions → supervised
- customer segmentation without any labels → unsupervised
- training a model to play a game by receiving the score → reinforcement learning

**Common traps**:
- treating these as mutually exclusive — many real systems blend them: self-supervised learning uses self-generated labels; semi-supervised uses a small labeled set plus a large unlabeled set; weakly supervised uses noisy labels from heuristics
- treating unsupervised as "unguided" — you still make explicit choices about distance metric, number of clusters, and architecture; these embed prior assumptions about the structure you expect to find

---

## 3. Bias, Variance, and the Bias-Variance Tradeoff

**What the interviewer is testing**: can you reason about model failures before seeing the data? Most candidates memorize the formula. The interesting question is "given this specific scenario, is the problem bias or variance, and how do you know?"

**The reasoning structure**: a high-bias model makes the same type of mistake regardless of which training set you use — it is systematically wrong in a predictable direction. The model is too simple to represent the true pattern.

A high-variance model makes very different mistakes depending on the training set — it memorizes noise. The model is too complex relative to the available data.

The diagnostic question: "if I retrained on a completely different sample from the same population, would the model make the same mistakes?" If yes: bias. If the mistakes would be completely different: variance.

**The pattern in action**:
- "95% training accuracy, 70% validation accuracy" → variance. The gap reveals memorization. Prescriptions: regularization, more data, simpler model, dropout.
- "70% training accuracy, 68% validation accuracy" → bias. The model is systematically wrong and does not have the capacity to learn the pattern. Prescriptions: more features, more expressive model, less regularization.

**Common traps**:
- applying regularization to a high-bias model — this restricts an already underpowered model and makes it worse
- assuming more data fixes variance — it helps, but a model with zero regularization will simply re-overfit to the larger training set; the fix requires a combination of more data and appropriate regularization
- treating the tradeoff as a fixed curve — with a better model architecture or better features, you can reduce both bias and variance simultaneously; the tradeoff only holds within a fixed model family

**The decomposition**:
$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

$\sigma^2$ is irreducible noise — it cannot be reduced regardless of model complexity. Bias is error from wrong assumptions in the model family. Variance is error from sensitivity to which specific training set was used.

---

## 4. Overfitting and Underfitting

**What the interviewer is testing**: whether these are real engineering diagnoses you can act on, not just vocabulary words.

**The reasoning structure**: underfitting means the model does not have enough capacity or the right inductive biases to capture the true pattern. It is wrong even on training data. Overfitting means the model has enough capacity to memorize training noise and did — it learned the training set's idiosyncratic noise, and that noise does not generalize.

The practical diagnosis uses the train/validation gap:
- Both high error: underfitting (bias problem)
- Training error low, validation error high: overfitting (variance problem)
- Both low with a small gap: the sweet spot

**The pattern in action**: "I fit a degree-1 polynomial to a clearly curved dataset. Training MSE is 45, validation MSE is 47 — both bad. The line cannot represent the curve. Underfitting. I try degree 10. Training MSE drops to 0.1, validation MSE is 120. The polynomial fits training points exactly then oscillates wildly between them. Overfitting. Degree 3 gives training MSE of 4 and validation MSE of 5 — well-fit."

**Common traps**:
- not having a validation set and only looking at training loss — you cannot diagnose overfitting without held-out data; training loss alone tells you nothing about generalization
- using validation performance to repeatedly select models without a final held-out test set — after 20 rounds of model selection based on the validation set, the validation estimate has effectively been "trained on" and is optimistically biased

---

## 5. Loss Functions vs Metrics

**What the interviewer is testing**: whether you understand why these are different things and when a model can look good on one while failing on the other.

**The reasoning structure**: a loss function must be differentiable (or subdifferentiable) and produce gradients that guide optimization. A metric must capture what the business or user actually cares about. These requirements often conflict.

Accuracy is the prototypical metric that cannot be a loss: it is discontinuous — a tiny weight change can flip a prediction, producing a discrete jump in accuracy that gives no gradient signal. Cross-entropy is differentiable and serves as a smooth proxy. But optimizing cross-entropy does not guarantee optimizing accuracy, especially under class imbalance.

The deeper issue: a model optimized for cross-entropy will minimize log-loss; whether the resulting predictions are useful depends on the downstream application. Loss and metric are aligned when the task is well-specified — and often are not.

**The pattern in action**: "I optimize cross-entropy loss on a fraud detection model and achieve 0.05 cross-entropy. But the business cares about recall — catching as much fraud as possible. When I compute recall on the validation set, it is only 40%. The model is well-calibrated but conservative — it hedges near 0.5 for borderline cases, and those cases are classified as not-fraud. The loss was reduced; the metric that matters was not."

**Common traps**:
- reporting training loss as model quality — loss magnitude is only meaningful relative to a baseline or when compared within the same loss function; an absolute cross-entropy number is not interpretable without context
- using accuracy as the metric on imbalanced data — if 99% of examples are negative, predict-all-negative achieves 99% accuracy with 0% recall on the class you actually care about
- not checking that the metric aligns with the business objective before training starts — discovering the metric mismatch after training is expensive

---

## 6. Epoch, Batch, Batch Size, Iteration

**What the interviewer is testing**: whether confusion here — a genuine red flag for a candidate — is present.

**The reasoning structure**: these terms describe how you partition data for gradient-based learning. You cannot process all data in one pass in practice (memory constraints), so you process it in chunks. Understanding these terms means understanding the tradeoffs in gradient estimation quality versus computational cost.

- **Epoch**: one full pass through the training dataset
- **Batch**: the subset of data used in one optimizer step
- **Batch size**: number of examples in one batch
- **Iteration**: one optimizer step (one batch processed)

If the dataset has $n = 10{,}000$ examples and batch size is 100, then 1 epoch = 100 iterations.

**The pattern in action**: "With batch size 32, each iteration gives a noisy estimate of the true gradient. After 312 iterations I have completed one epoch. If I increase batch size to 4096, my gradient estimates are more accurate per step but require more memory. I may also need to adjust the learning rate upward to compensate, and I risk converging to a sharper minimum because I have reduced the beneficial noise of small batches."

**Common traps**:
- confusing "faster training per epoch" with "better final performance" when increasing batch size — larger batches can converge faster per epoch but often generalize worse because noisy gradients from small batches act as implicit regularization
- not knowing that larger batch sizes typically require a proportionally larger learning rate (linear scaling rule) — using a large batch with the small-batch learning rate will underfit because each step is too small relative to the gradient quality

---

## 7. Classification vs Regression

**What the interviewer is testing**: whether you understand the output type as determining the appropriate loss function and output activation.

**The reasoning structure**: classification produces a discrete label; regression produces a continuous value. The distinction matters because:
- the loss function must match the output type (cross-entropy for classification, MSE/MAE for regression)
- the output layer activation must match (softmax for multiclass, sigmoid for binary, linear/identity for regression)
- calibration means different things in each case

The non-obvious case is ordinal regression — star ratings (1–5) are discrete but ordered. Treating as classification ignores the ordering. Treating as regression implies equal spacing between consecutive values, which may not hold. This requires specialized ordinal regression approaches.

**Common traps**:
- treating a regression problem as classification by binning the output — you lose ordinal information and add sensitivity to the choice of bin boundaries, which is an arbitrary hyperparameter
- using sigmoid output with MSE loss for binary classification — technically valid but creates poor gradient behavior near saturation; cross-entropy is the principled choice for classification outputs

---

## 8. L1 vs L2 Regularization

**What the interviewer is testing**: whether you understand the geometric reason L1 produces sparsity and L2 does not.

**The reasoning structure**: regularization adds a penalty to the loss that discourages large weights. The penalty's shape determines its effect on weights.

L2 (Ridge) adds $\lambda \|w\|_2^2$. The gradient of the penalty is $2\lambda w$ — it always pushes weights toward zero proportionally to their current size. Weights become small but rarely exactly zero, because the push weakens as the weight approaches zero.

L1 (Lasso) adds $\lambda \|w\|_1$. The gradient is $\lambda \cdot \text{sign}(w)$ — a constant push toward zero regardless of weight magnitude. Small weights, which have a small loss gradient keeping them away from zero, get overcome by the constant L1 push and reach exactly zero. Large weights, with a large loss gradient, survive.

Geometrically: L2 penalty contours are spheres (smooth, no corners). L1 penalty contours are diamonds with corners at the coordinate axes. The optimum of the constrained problem tends to land at a corner of the L1 diamond — which corresponds to zero in some coordinates — producing sparsity.

**The pattern in action**: "I have 500 features for a loan model. I suspect only ~20 are truly predictive. L1 regularization zeroes out the unimportant 480 and leaves a sparse, interpretable model. L2 gives me 500 small non-zero weights — the 20 important ones are larger, but the 480 noisy features still contribute, making the model harder to interpret and deploy."

**Common traps**:
- assuming L1 is always better for interpretability — when features are grouped and correlated, L1 arbitrarily picks one from the group and discards the rest; Elastic Net (L1 + L2) is more stable in this case
- applying strong regularization to a high-bias model — the model is already underpowered; adding regularization makes it worse by further restricting its capacity

---

## 9. Dropout

**What the interviewer is testing**: whether you understand dropout as a specific mechanism addressing co-adaptation, not just "randomly turns off neurons."

**The reasoning structure**: the problem dropout solves is co-adaptation. Without it, neurons develop complex interdependencies — neuron A's output is only useful when combined with neuron B's specific pattern. This makes the network brittle and tied to specific training examples. Co-adapted groups of neurons memorize, rather than generalize.

Dropout randomly deactivates a fraction of neurons during each training step (typically 20–50%). This prevents any neuron from relying on specific other neurons — each must learn to be useful with any random subset of its co-neurons. At inference time, dropout is disabled; weights are scaled to compensate for the change in expected activation magnitude.

An equivalent interpretation: dropout trains an implicit ensemble of $2^n$ thinned subnetworks and approximates their average at inference time through weight scaling.

**The pattern in action**: "My network has 98% training accuracy and 74% validation accuracy — severe overfitting. I add dropout with rate 0.5 after each hidden layer. Training accuracy drops to 88% (the network can no longer freely overfit) and validation accuracy rises to 83%. Neuron A can no longer rely on always receiving a signal from neuron B because B may be dropped. The network has been forced to learn more distributed representations."

**Common traps**:
- applying dropout to small, already-underpowered networks — if the model has high bias, dropout reduces capacity further and makes it worse
- forgetting to call `model.eval()` at inference time — active dropout at inference gives stochastic predictions with roughly half the neurons active each call, causing inconsistent outputs; this is a common production bug
- applying dropout after the final output layer — almost never correct

---

## 10. Embeddings

**What the interviewer is testing**: whether you understand embeddings as learned representations that capture semantic structure, not just a compression technique for categorical features.

**The reasoning structure**: raw categorical features (user IDs, product IDs, words) have no inherent numeric structure. One-hot encoding creates vectors where all categories are equally distant — "Paris" and "France" are as distant as "Paris" and "banana." This is wrong and wastes the opportunity to encode semantic relationships.

Embeddings learn a mapping from categories to dense vectors such that categories that behave similarly end up close in embedding space. The embedding is not designed by hand — it emerges from training on the task. The embedding space encodes semantic relationships as geometric relationships, which the model can then reason over.

**The pattern in action**: "In a word embedding trained on text, 'king' − 'man' + 'woman' ≈ 'queen' in vector arithmetic. This is not programmed — it emerged because words in similar contexts end up in similar regions of the embedding space. The geometry of the learned space reflects the structure of the language."

**Common traps**:
- treating pre-trained embeddings as permanently frozen when fine-tuning is available — if you have enough task data, fine-tuning the embeddings to your specific task often significantly improves performance; frozen embeddings may not encode the task-relevant distinctions
- using very high-dimensional embeddings for low-cardinality categories — wasted parameters; a practical rule of thumb is embedding dimension ≈ `min(50, category_count / 2)`

---

## 11. Cross-Entropy and Softmax

**What the interviewer is testing**: whether you understand the connection between the output representation (softmax probabilities) and the training objective (cross-entropy), and why this combination is the canonical choice.

**The reasoning structure**: after a classification network's final layer, you have raw logit scores — unbounded real numbers, one per class. You want probabilities that sum to 1. Softmax performs this conversion:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Cross-entropy then penalizes the model based on the probability assigned to the correct class:

$$\mathcal{L} = -\log p(\text{correct class})$$

The combination is elegant in two ways. First, the gradient of cross-entropy through softmax simplifies to $(p - y)$ — the predicted probability vector minus the one-hot label vector. This clean, bounded gradient makes training stable. Second, confident correct predictions incur near-zero loss; confident wrong predictions incur very high loss (approaching infinity), which creates strong gradient signal precisely where the model is most wrong.

**Common traps**:
- using sigmoid at the output of a multiclass classifier — sigmoid applied per class does not constrain probabilities to sum to 1; use softmax for multiclass problems
- computing softmax naively — exponentiating large logits causes numerical overflow; subtract the max logit first before exponentiating (mathematically equivalent but numerically stable)

---

## 12. Cross-Validation

**What the interviewer is testing**: whether you understand cross-validation as a variance-reduction technique for performance estimation, not just "a way to use more data for training."

**The reasoning structure**: a single 80/20 train/validation split gives one estimate of generalization performance. That estimate is noisy — you might have gotten a lucky or unlucky split. K-fold cross-validation rotates through $k$ different splits and averages the scores, giving a lower-variance estimate of true generalization performance.

The secondary diagnostic value: if performance varies a lot across folds (high standard deviation), the model's behavior is strongly sensitive to which specific examples end up in training vs validation. That sensitivity is itself a signal worth investigating.

**The pattern in action**: "My model gets 84% on one 80/20 split. I run 5-fold CV and get fold scores [84, 71, 89, 76, 82] — mean 80.4%, std 6.3%. The single split was optimistic. The high variance across folds tells me the model's performance depends heavily on which examples are in training — I should investigate whether certain subpopulations are causing this sensitivity."

**Common traps**:
- performing feature selection or preprocessing before cross-validation and then evaluating inside the CV loop — the preprocessing has already seen all the validation data; this is leakage. Preprocessing must be refit inside each fold.
- reporting the cross-validation score as the final generalization estimate after using it for model selection — the CV was used to select the model, so it is optimistically biased as a performance estimate. You need a held-out test set for the final number.

---

## 13. Precision, Recall, and F1

**What the interviewer is testing**: can you reason about metric choice from the cost structure of errors, not just recall the formulas?

**The reasoning structure**: every classification error is either a false positive (predicted positive, actually negative) or a false negative (predicted negative, actually positive). These errors have asymmetric costs in most real problems.

Ask: "what is worse — a false positive or a false negative?" In cancer screening, missing a real cancer (false negative) is far worse than a false alarm (false positive). In spam filtering, blocking a real email (false positive) is likely worse than letting spam through (false negative). The metric should reflect the actual cost structure.

Precision = $\frac{TP}{TP + FP}$: of everything predicted positive, how many were actually positive? Use when false positives are costly.

Recall = $\frac{TP}{TP + FN}$: of everything actually positive, how much did we catch? Use when false negatives are costly.

F1 = harmonic mean of precision and recall. The harmonic mean is the right choice because you want both to be high simultaneously — a model with 100% precision and 0% recall (predict nothing positive) gets F1 = 0, which is the correct score.

**The pattern in action**:
- Spam filter: optimize precision. Missing real email is worse than seeing spam.
- Cancer detection: optimize recall. Missing a real case is far worse than an unnecessary biopsy.
- Fraud detection where the investigation team has limited capacity: use F1 or set the threshold to match the team's daily review capacity from the precision-recall curve.

**Common traps**:
- using accuracy as the primary metric on imbalanced data — a model that predicts "not fraud" for every transaction achieves 99.9% accuracy on a 0.1% positive-rate dataset, with 0% recall on the class you care about
- fixing the threshold at 0.5 without reasoning about error costs — 0.5 assumes false positives and false negatives have equal cost; that is rarely true in practice

---

## 14. Anomaly Detection

**What the interviewer is testing**: whether you understand why standard supervised classification is often the wrong approach for anomaly detection, and what the alternatives are.

**The reasoning structure**: in fraud, industrial defect detection, and network intrusion, anomalies are rare and diverse. Class imbalance is severe (often 0.1% positive rate or less). More critically, future anomalies may look different from past anomalies — a novel fraud pattern will not appear in the training set.

A supervised classifier trained on known fraud patterns will have high recall on those specific patterns and zero recall on new ones. One-class and density estimation approaches instead model "what normal looks like" and flag anything that does not fit. This generalizes to unseen anomaly types because you are modeling the absence of normalcy, not the presence of specific known anomalies.

**The pattern in action**: "I build an anomaly detection system for API abuse. I train an autoencoder on normal traffic only. The autoencoder learns to reconstruct typical request patterns. An anomalous request — whether it matches a known attack pattern or not — will have high reconstruction error if it deviates from normal behavior. A completely novel attack that was never in training data will still be flagged if it is unlike normal traffic."

**Common traps**:
- treating all anomaly detection as supervised binary classification — you will miss novel anomaly types and the model degrades as adversaries learn your detection patterns
- setting the anomaly threshold without understanding the operational context — precision at very low recall is usually meaningless; set the threshold based on how many anomalies you can actually review per day

---

## 15. Curse of Dimensionality

**What the interviewer is testing**: whether you understand dimensionality as a source of concrete, specific failure modes rather than a vague warning.

**The reasoning structure**: as the number of features grows, the volume of the feature space grows exponentially. The same number of data points becomes exponentially sparser. Three specific consequences follow:

1. **Distance concentration**: in high dimensions, the ratio of the maximum to minimum pairwise distance between any two points approaches 1. "Nearest neighbor" and "farthest neighbor" become indistinguishable, breaking all distance-based methods.
2. **Volume concentration**: most of the volume of a high-dimensional hypercube is near its faces, not its center. Most of a hypersphere's volume is near its surface. Data distributions do not cover space as expected.
3. **Statistical sparsity**: to achieve the same density of coverage, you need exponentially more data as dimensions grow.

**The pattern in action**: "I have a KNN classifier with 5,000 training points in 500 dimensions. In 500 dimensions, all 5,000 points are roughly equidistant from any query point. The 5 'nearest' neighbors are not meaningfully closer than the 5,000th neighbor. KNN distances carry no information and performance collapses."

**Common traps**:
- thinking regularization alone addresses high-dimensional problems — dimensionality reduction (PCA, feature selection) or a model architecture with appropriate inductive bias is often more important than regularization
- treating all features as equally useful in high-dimensional settings — uninformative features drown out informative ones; feature selection becomes more critical, not less, as dimensionality grows

---

## 16. Policy-Based vs Value-Based Reinforcement Learning

**What the interviewer is testing**: whether you understand RL as a distinct framework with its own failure modes, not just supervised learning with delayed labels.

**The reasoning structure**: in supervised learning, you know the correct answer for each input and can directly measure and reduce prediction error. In RL, you only receive a scalar reward signal, possibly delayed by many timesteps, and must infer which actions were responsible for the eventual outcome — the credit assignment problem.

Value-based RL (Q-learning, DQN): learn a value function $Q(s, a)$ estimating expected cumulative reward of taking action $a$ in state $s$, then act greedily. Works well for discrete action spaces; struggles with large or continuous action spaces where you cannot enumerate all actions.

Policy-based RL (REINFORCE, PPO): directly parameterize and optimize the policy $\pi_\theta(a|s)$. Handles continuous action spaces naturally. The core challenge is high-variance gradient estimates — the policy gradient theorem gives you the right direction but the variance can be enormous, requiring baselines and other variance-reduction techniques.

**Common traps**:
- treating exploration as a minor implementation detail — in RL, insufficient exploration means the agent never discovers that better strategies exist; the explore-exploit tradeoff is a first-class problem, not an afterthought
- not recognizing that RL is dramatically harder to debug than supervised learning — the training signal is delayed, noisy, non-stationary, and the data distribution changes as the policy changes; standard supervised learning debugging intuitions do not transfer directly

## Flashcards

**defining ML as "the model learns from data" without saying what generalization means?** #flashcard
any system can memorize a training set; the hard part is performing on examples that were never seen during training

**not recognizing that the training objective and the true goal can diverge?** #flashcard
a model that minimizes cross-entropy on training data is not necessarily doing what the business wants; these need to be audited separately

**fraud detection with historical labeled transactions → supervised?** #flashcard
fraud detection with historical labeled transactions → supervised

**customer segmentation without any labels → unsupervised?** #flashcard
customer segmentation without any labels → unsupervised

**training a model to play a game by receiving the score → reinforcement learning?** #flashcard
training a model to play a game by receiving the score → reinforcement learning

**treating these as mutually exclusive?** #flashcard
many real systems blend them: self-supervised learning uses self-generated labels; semi-supervised uses a small labeled set plus a large unlabeled set; weakly supervised uses noisy labels from heuristics

**treating unsupervised as "unguided"?** #flashcard
you still make explicit choices about distance metric, number of clusters, and architecture; these embed prior assumptions about the structure you expect to find

**"95% training accuracy, 70% validation accuracy" → variance. The gap reveals memorization. Prescriptions?** #flashcard
regularization, more data, simpler model, dropout.

**"70% training accuracy, 68% validation accuracy" → bias. The model is systematically wrong and does not have the capacity to learn the pattern. Prescriptions?** #flashcard
more features, more expressive model, less regularization.

**applying regularization to a high-bias model?** #flashcard
this restricts an already underpowered model and makes it worse

**assuming more data fixes variance?** #flashcard
it helps, but a model with zero regularization will simply re-overfit to the larger training set; the fix requires a combination of more data and appropriate regularization

**treating the tradeoff as a fixed curve?** #flashcard
with a better model architecture or better features, you can reduce both bias and variance simultaneously; the tradeoff only holds within a fixed model family

**Both high error?** #flashcard
underfitting (bias problem)

**Training error low, validation error high?** #flashcard
overfitting (variance problem)

**Both low with a small gap?** #flashcard
the sweet spot

**not having a validation set and only looking at training loss?** #flashcard
you cannot diagnose overfitting without held-out data; training loss alone tells you nothing about generalization

**using validation performance to repeatedly select models without a final held-out test set?** #flashcard
after 20 rounds of model selection based on the validation set, the validation estimate has effectively been "trained on" and is optimistically biased

**reporting training loss as model quality?** #flashcard
loss magnitude is only meaningful relative to a baseline or when compared within the same loss function; an absolute cross-entropy number is not interpretable without context

**using accuracy as the metric on imbalanced data?** #flashcard
if 99% of examples are negative, predict-all-negative achieves 99% accuracy with 0% recall on the class you actually care about

**not checking that the metric aligns with the business objective before training starts?** #flashcard
discovering the metric mismatch after training is expensive

**Epoch?** #flashcard
one full pass through the training dataset

**Batch?** #flashcard
the subset of data used in one optimizer step

**Batch size?** #flashcard
number of examples in one batch

**Iteration?** #flashcard
one optimizer step (one batch processed)

**confusing "faster training per epoch" with "better final performance" when increasing batch size?** #flashcard
larger batches can converge faster per epoch but often generalize worse because noisy gradients from small batches act as implicit regularization

**not knowing that larger batch sizes typically require a proportionally larger learning rate (linear scaling rule)?** #flashcard
using a large batch with the small-batch learning rate will underfit because each step is too small relative to the gradient quality

**the loss function must match the output type (cross-entropy for classification, MSE/MAE for regression)?** #flashcard
the loss function must match the output type (cross-entropy for classification, MSE/MAE for regression)

**the output layer activation must match (softmax for multiclass, sigmoid for binary, linear/identity for regression)?** #flashcard
the output layer activation must match (softmax for multiclass, sigmoid for binary, linear/identity for regression)

**calibration means different things in each case?** #flashcard
calibration means different things in each case

**treating a regression problem as classification by binning the output?** #flashcard
you lose ordinal information and add sensitivity to the choice of bin boundaries, which is an arbitrary hyperparameter

**using sigmoid output with MSE loss for binary classification?** #flashcard
technically valid but creates poor gradient behavior near saturation; cross-entropy is the principled choice for classification outputs

**assuming L1 is always better for interpretability?** #flashcard
when features are grouped and correlated, L1 arbitrarily picks one from the group and discards the rest; Elastic Net (L1 + L2) is more stable in this case

**applying strong regularization to a high-bias model?** #flashcard
the model is already underpowered; adding regularization makes it worse by further restricting its capacity

**applying dropout to small, already-underpowered networks?** #flashcard
if the model has high bias, dropout reduces capacity further and makes it worse

**forgetting to call model.eval() at inference time?** #flashcard
active dropout at inference gives stochastic predictions with roughly half the neurons active each call, causing inconsistent outputs; this is a common production bug

**applying dropout after the final output layer?** #flashcard
almost never correct

**treating pre-trained embeddings as permanently frozen when fine-tuning is available?** #flashcard
if you have enough task data, fine-tuning the embeddings to your specific task often significantly improves performance; frozen embeddings may not encode the task-relevant distinctions

**using very high-dimensional embeddings for low-cardinality categories?** #flashcard
wasted parameters; a practical rule of thumb is embedding dimension ≈ min(50, category_count / 2)

**using sigmoid at the output of a multiclass classifier?** #flashcard
sigmoid applied per class does not constrain probabilities to sum to 1; use softmax for multiclass problems

**computing softmax naively?** #flashcard
exponentiating large logits causes numerical overflow; subtract the max logit first before exponentiating (mathematically equivalent but numerically stable)

**performing feature selection or preprocessing before cross-validation and then evaluating inside the CV loop?** #flashcard
the preprocessing has already seen all the validation data; this is leakage. Preprocessing must be refit inside each fold.

**reporting the cross-validation score as the final generalization estimate after using it for model selection?** #flashcard
the CV was used to select the model, so it is optimistically biased as a performance estimate. You need a held-out test set for the final number.

**Spam filter?** #flashcard
optimize precision. Missing real email is worse than seeing spam.

**Cancer detection?** #flashcard
optimize recall. Missing a real case is far worse than an unnecessary biopsy.

**Fraud detection where the investigation team has limited capacity?** #flashcard
use F1 or set the threshold to match the team's daily review capacity from the precision-recall curve.

**using accuracy as the primary metric on imbalanced data?** #flashcard
a model that predicts "not fraud" for every transaction achieves 99.9% accuracy on a 0.1% positive-rate dataset, with 0% recall on the class you care about

**fixing the threshold at 0.5 without reasoning about error costs?** #flashcard
0.5 assumes false positives and false negatives have equal cost; that is rarely true in practice

**treating all anomaly detection as supervised binary classification?** #flashcard
you will miss novel anomaly types and the model degrades as adversaries learn your detection patterns

**setting the anomaly threshold without understanding the operational context?** #flashcard
precision at very low recall is usually meaningless; set the threshold based on how many anomalies you can actually review per day

**thinking regularization alone addresses high-dimensional problems?** #flashcard
dimensionality reduction (PCA, feature selection) or a model architecture with appropriate inductive bias is often more important than regularization

**treating all features as equally useful in high-dimensional settings?** #flashcard
uninformative features drown out informative ones; feature selection becomes more critical, not less, as dimensionality grows

**treating exploration as a minor implementation detail?** #flashcard
in RL, insufficient exploration means the agent never discovers that better strategies exist; the explore-exploit tradeoff is a first-class problem, not an afterthought

**not recognizing that RL is dramatically harder to debug than supervised learning?** #flashcard
the training signal is delayed, noisy, non-stationary, and the data distribution changes as the policy changes; standard supervised learning debugging intuitions do not transfer directly
