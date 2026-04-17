# ML Algorithms

This file is your "know the players" section.

Not every algorithm needs a dramatic monologue.
But you should know:

- what problem it solves
- why it works
- when it fails
- what you would choose instead

That is how you stop sounding like a glossary and start sounding like an engineer.

---

# 1. Decision Trees

A decision tree keeps splitting the feature space into smaller and smaller regions until the data in each region becomes more pure.

In classification, purity usually means better separation by:

- Gini
- entropy

In regression, it usually means lower variance or MSE inside a node.

**Simple interview answer**

A decision tree recursively chooses the feature split that most reduces impurity and uses the resulting leaves to make predictions.

**Why people like trees**

- easy to explain
- little preprocessing needed
- naturally handle non-linearity
- capture feature interactions

**Why they can be messy**

- high variance
- easy to overfit
- unstable if data changes slightly

**Fashion analogy**

A tree is like deciding an outfit through a sequence of styling questions:

- Is it formal?
- Day or evening?
- Summer or monsoon?
- Structured or flowy?

Each answer narrows the path.

---

# 2. Gini vs Entropy

You do not need to perform theatre around this one.

Both measure impurity.

- **Gini** is a little simpler and faster
- **Entropy** has a more information-theory flavor

In practice, both often choose similar splits.

**Short senior-sounding line**

I care less about memorizing the exact formula differences and more about whether the split criterion improves generalization and stability for the use case.

That lands well.

---

# 3. Random Forest

Random Forest is a collection of decision trees trained with extra randomness.

It uses:

- bootstrap sampling of rows
- random feature subsets at splits

Then averages the trees.

Why that helps:

One tree is unstable.
Many decorrelated trees are much more reliable.

**Short answer**

Random Forest reduces variance by averaging many decorrelated decision trees.

**DevOps analogy**

One flaky health signal is risky.
Averaging multiple independent health checks gives a much more trustworthy picture.

That is Random Forest energy.

---

# 4. Bagging vs Boosting

This distinction matters a lot.

## Bagging

Train many models independently.
Then average them.

Main win:

- reduces variance

## Boosting

Train models sequentially.
Each new model focuses on earlier mistakes.

Main win:

- reduces bias

**Easy memory trick**

- bagging = committee vote
- boosting = strict tutor correcting you after every mistake

---

# 5. Gradient Boosting and XGBoost

Gradient boosting builds models stage by stage.

Each new tree tries to correct what the current ensemble is still getting wrong.

That is why boosting can be so powerful on tabular data.

## XGBoost

XGBoost is a highly optimized gradient boosting library with:

- regularization
- efficient split finding
- missing value handling
- excellent practical performance

**Why interviewers love asking about it**

Because it is a very common real-world baseline.

And because it teaches you the important tradeoff:

- excellent performance
- but can overfit if tuned carelessly

**Short interview answer**

XGBoost is a regularized, optimized implementation of gradient-boosted trees that performs especially well on structured tabular data.

---

# 6. Random Forest vs XGBoost

This is one of those classic comparison questions.

## Random Forest

- easier default
- more robust
- harder to overfit
- strong baseline

## XGBoost

- usually stronger when tuned well
- more sensitive to hyperparameters
- more likely to overfit noisy data if unchecked

**Cricket analogy**

Random Forest is the stable middle-order batter.
Rarely spectacular, rarely disastrous.

XGBoost is the aggressive match-winner.
Can take the game away.
Can also get caught at long-on if the shot selection is bad.

---

# 7. Logistic Regression

Despite the name, logistic regression is a classification model.

It models the log-odds of the positive class as a linear function of the features.

That means:

- decision boundary is linear
- output is probabilistic

Why it is still important:

- strong baseline
- fast
- interpretable
- easy to regularize

**Short answer**

Logistic regression predicts class probability by passing a linear score through a sigmoid function.

**When it shines**

- tabular data
- interpretable systems
- small to medium datasets
- baseline model comparisons

---

# 8. Linear Regression vs Logistic Regression

This gets asked a lot because people mix them up.

## Linear Regression

Predicts a continuous value.

## Logistic Regression

Predicts probability of a class.

**The subtle but important line**

Logistic regression is linear in the **log-odds**, not in the output probability itself.

That sentence alone makes your answer better than average.

---

# 9. SVM and the Kernel Trick

SVM tries to find the decision boundary with the largest margin between classes.

That margin idea is the main reason it can generalize well.

## Kernel Trick

When the data is not linearly separable, the kernel trick lets SVM behave as if the data had been mapped into a higher-dimensional space.

Without explicitly computing that mapping.

That is the clever part.

**Short answer**

The kernel trick allows SVMs to learn non-linear boundaries by replacing raw dot products with similarity functions in an implicit higher-dimensional space.

---

# 10. K-Nearest Neighbors

KNN predicts based on the closest examples in the training data.

For classification:

- majority vote

For regression:

- average nearby values

Why it is useful:

- simple
- intuitive
- strong teaching baseline

Why it is annoying:

- slow at inference
- very sensitive to scale
- weak in high dimensions

**Fashion analogy**

KNN is like saying:

> "Show me the five outfits most similar to this one, then guess the style based on those."

Works fine if your similarity notion is good.
Breaks if your features are messy.

---

# 11. K-Means

K-Means clusters data by repeatedly:

1. assigning points to the nearest centroid
2. recomputing centroids

It is fast and common.

But it assumes:

- roughly spherical clusters
- meaningful distance metric
- a known number of clusters `k`

**How to choose K**

Common methods:

- elbow method
- silhouette score

**Reality check**

Those are hints, not divine revelation.

Domain sense still matters.

---

# 12. Naive Bayes

Naive Bayes uses Bayes' theorem and assumes features are conditionally independent given the class.

That assumption is obviously unrealistic in many problems.

And yet:

it still works surprisingly well, especially in text classification.

Why?

Because a rough probabilistic model can still be very effective when:

- features are sparse
- speed matters
- data is limited

---

# 13. Decision Boundary

The decision boundary is the surface where a model switches from predicting one class to another.

Why it matters:

It tells you how flexible the model is.

- linear models = simple boundaries
- deep models / boosted trees = more complex boundaries

This is a good concept to mention when comparing model capacity.

---

# 14. Dimensionality Reduction

Dimensionality reduction means compressing data into fewer variables while preserving useful structure.

Why do it?

- faster training
- less noise
- easier visualization
- better behavior in high dimension

It is not always about performance.
Sometimes it is about sanity.

---

# 15. PCA

PCA finds the directions of maximum variance in the data and projects the data onto those directions.

These directions are called principal components.

Why it helps:

- compression
- denoising
- visualization
- decorrelation

**Important caveat**

PCA is unsupervised.

That means the directions of highest variance are not always the directions most useful for prediction.

**Short answer**

PCA reduces dimension by projecting data onto orthogonal directions that explain the most variance.

---

# 16. Gradient Descent and Variants

Even classical algorithms get wrapped around optimization eventually.

Important variants:

- batch gradient descent
- SGD
- mini-batch SGD
- momentum
- Adam

The practical lesson is:

Optimization choice changes:

- convergence speed
- stability
- sometimes generalization

---

# Quick Thought Experiment

You have medium-sized structured tabular data with:

- missing values
- mixed feature types
- not much feature engineering time

What is your first strong baseline?

Answer:

Usually a tree ensemble, especially boosted trees.

Not because it is trendy.
Because it is often the right hammer.

---

# Mini Pop Quiz

Which model is more likely to need feature scaling:

- KNN
- Random Forest

Answer:

KNN.

Because distance-based models care deeply about feature scale.
Tree splits usually do not.
