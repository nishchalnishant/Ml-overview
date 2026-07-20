---
module: Interview Prep
topic: Week 2 Algorithms
subtopic: Day 17 18 Hyperparameter Tuning
status: unread
tags: [studyplans, ml, week-2-algorithms-day-17-18-hy]
---
# Day 17-18: Hyperparameter Tuning

## Why This Topic Comes Here

You now know how to train models and how to evaluate them. Hyperparameter tuning is the bridge: given an evaluation metric you trust, how do you systematically improve model performance? This topic belongs after evaluation metrics because tuning without a reliable evaluation methodology is meaningless — if your metric is corrupted by leakage, the tuned hyperparameters are tuned for the wrong objective. It belongs before specialized techniques (NLP/CV) because the same search principles apply to any model in any domain. This session is also where overfitting and underfitting become operationally concrete: the hyperparameters that control model complexity are the direct levers on the bias-variance tradeoff.

---

## Executive Summary

| Method | Strategy | Pros | Cons |
|--------|----------|------|------|
| **Grid Search** | Exhaustive | Guarantees best in grid | Extremely slow |
| **Random Search** | Random samples | Faster, statistically sound | Might miss the narrow "best" |
| **Bayesian Opt** | Model-based | Samples promising areas | Complex to implement |
| **Successive Halving** | Resource-based | Discards bad models early | May kill "slow learners" |

---

## 1. Parameters vs. Hyperparameters

**Why the distinction matters:** Conflating the two is a common interview mistake. Getting this wrong in practice leads to real errors — for example, treating the number of layers as something that can be learned by backpropagation (it cannot).

**Parameters** are learned from data during training (e.g., weights $w$, bias $b$). **Hyperparameters** are set by the engineer *before* training to control the learning process (e.g., learning rate, $K$ in KNN, tree depth, regularization strength).

**Key insight:** Hyperparameters define the hypothesis space — the set of all possible models the algorithm can learn. Parameters are where the algorithm lands within that space, given the data. Choosing hyperparameters is therefore a higher-level decision: you are choosing what kind of model is even possible before training begins.

**How to verify understanding:** Is the number of neurons in a hidden layer a parameter or a hyperparameter? What about the weights connecting those neurons? Explain why they are categorically different.

**What trips people up:** Treating architecture choices (number of layers, width, which algorithm to use) as parameters that can be learned. These are discrete choices that define the search space — they are selected by the engineer, not learned by gradient descent.

---

## 2. Search Strategies

**Why you should understand all four strategies:** Each strategy makes a different assumption about the hyperparameter landscape, and the best choice depends on your compute budget and the structure of the problem.

### Grid Search

Defines a list of values for each hyperparameter and tries every single combination.
- **Search Space**: If you tune 3 parameters with 5 values each $\rightarrow 5^3 = 125$ training runs.

**Key insight:** Grid search only works well when all hyperparameters are roughly equally important. If one parameter dominates (e.g., learning rate) and the others matter little, grid search wastes most of its budget varying the unimportant ones at fixed, potentially bad values of the important one.

**How to verify understanding:** You have 4 hyperparameters and run a grid search with 5 values each (625 runs). You discover that 3 of the 4 parameters barely affect the metric. How many of your 625 runs were effectively wasted, and what search strategy should you have used?

**What trips people up:** Using grid search with wide value ranges. Grid search is most useful for final fine-grained search around a region you have already identified as promising — not for initial exploration of an unknown space.

### Random Search

Defines a distribution for each hyperparameter and samples $N$ random combinations.
- **Why it works**: In high-dimensional spaces, some parameters may not affect the outcome significantly. Random search explores more distinct values of the "important" parameters because it does not fix them to a grid.

**Key insight:** Random search is not just computationally cheaper than grid search — it is theoretically more efficient when only a few dimensions matter. A single "active" dimension gets $N$ distinct sampled values under random search, versus only $k$ values (where $k$ is the grid size) under grid search. This is why random search finds better results for the same compute budget in most practical cases.

**How to verify understanding:** Sketch the behavior of random search vs. grid search over a 2D hyperparameter space where performance depends strongly on dimension 1 and not at all on dimension 2. Which approach explores dimension 1 more thoroughly for the same number of evaluations?

**What trips people up:** Using random search with a uniform distribution over a range when the optimal values are known to be at the extremes (or in a log-scale regime). Always use log-uniform distributions for learning rates, regularization strengths, and other parameters that span multiple orders of magnitude.

### Bayesian Optimization

Builds a surrogate model (typically a Gaussian Process) of the objective function. Uses the surrogate to decide which hyperparameters to try next, balancing **exploration** (try uncertain regions) and **exploitation** (refine around the current best).

**Key insight:** Bayesian optimization is not just "smarter random search" — it builds a model of how hyperparameters relate to performance and uses that model to make informed choices about what to try next. It works best when each evaluation is expensive (training a large model), because the overhead of fitting the surrogate is justified by the savings in wasted training runs.

**How to verify understanding:** You have a budget of 20 hyperparameter evaluations for a model that takes 6 hours to train. Rank grid search, random search, and Bayesian optimization in terms of expected performance, and explain why.

**What trips people up:** Using Bayesian optimization when you have a large evaluation budget and cheap evaluations. In that regime, random search is competitive and simpler. Bayesian optimization's advantage is strongest in the small-budget regime.

---

## 3. Overfitting vs. Underfitting Control

**Why regularization hyperparameters connect back to the bias-variance tradeoff:** The regularization strength $\lambda$ is a direct dial on the bias-variance axis. Increasing $\lambda$ increases bias (simpler model) and decreases variance (less sensitive to training noise). Tuning $\lambda$ is how you move a model along the bias-variance curve.

### Regularization Tuning

- **L1 (Lasso)**: Adds $\lambda \sum |\theta|$ to the loss. Leads to sparse solutions (many weights go to exactly zero). Useful when you believe most features are irrelevant.
- **L2 (Ridge)**: Adds $\lambda \sum \theta^2$. Keeps all weights small but non-zero. Better when you believe all features contribute a little.
- **Elastic Net**: A linear combination of L1 and L2. Has two hyperparameters to tune ($\lambda$ and the L1/L2 mixing ratio).

**Key insight:** L1 regularization encourages sparsity because the penalty gradient is constant ($\pm\lambda$) regardless of weight magnitude — a small weight experiences the same pull toward zero as a large weight. L2's gradient is proportional to weight magnitude ($2\lambda\theta$), so large weights are penalized more than small ones, but small weights are never driven to exactly zero.

**How to verify understanding:** You train a Lasso regression on 100 features. After tuning $\lambda$, you find 85 weights are exactly zero. What does this tell you about the problem structure, and is this a sign of underfitting?

**What trips people up:** Applying L1/L2 regularization to tree-based models via scikit-learn and expecting the same effect as with linear models. Tree-based models have their own regularization mechanisms (max depth, min samples per leaf, min impurity decrease) that operate differently.

### The Learning Rate

The most sensitive hyperparameter for any gradient-based model.
- **Too High**: Oscillates or diverges.
- **Too Low**: Converges too slowly or gets stuck in local minima.
- **Learning Rate Schedule**: Start high (fast progress), decay over time (fine-grained convergence).

**Key insight:** The optimal learning rate is not a fixed number — it depends on the batch size, the model architecture, and the optimizer. Doubling the batch size and doubling the learning rate is a common heuristic (linear scaling rule) that works approximately in many regimes. When you change any of these, the learning rate may need to be re-tuned.

**How to verify understanding:** You increase batch size from 32 to 256 (8x) to speed up training. Your training loss curve becomes worse. What is the likely cause, and what is the standard adjustment?

**What trips people up:** Fixing the learning rate and assuming the model's training behavior is solely due to the architecture or data. The learning rate is often the primary cause of training instability, slow convergence, or poor generalization — before blaming the model, always check the learning rate.

---

## Interview Questions

**1. "What is the difference between a Model Parameter and a Hyperparameter?"**
> **Parameters** are learned from data during training (e.g., weights $w$, bias $b$). **Hyperparameters** are set by the engineer *before* training to control the learning process (e.g., learning rate, $K$ in KNN).

**2. "Why is Random Search often better than Grid Search?"**
> In a high-dimensional space, some parameters may not affect the outcome significantly. Grid search wastes time varying these unimportant parameters. Random search explores more distinct values for "important" parameters.

**3. "How does Bayesian Optimization work in simple terms?"**
> It builds a surrogate model (typically a Gaussian Process) of the objective function. It uses this model to decide which hyperparameters to try next by balancing **exploration** and **exploitation**.

---

## Implementation (Optuna Example)

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 10)
    l2 = trial.suggest_float("l2", 1e-4, 1.0, log=True)
    # Train model with these hyperparameters...
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
```
