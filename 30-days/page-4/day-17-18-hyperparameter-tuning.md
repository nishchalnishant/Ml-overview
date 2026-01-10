# Day 17-18: Hyperparameter Tuning

## 📋 Executive Summary
| Method | Strategy | Pros | Cons |
|--------|----------|------|------|
| **Grid Search** | Exhaustive | Guarantees best in grid | Extremely slow |
| **Random Search** | Random samples | Faster, statistically sound | Might miss the narrow "best" |
| **Bayesian Opt** | Model-based | Samples promising areas | Complex to implement |
| **Successive Halving** | Resource-based | Discards bad models early | May kill "slow learners" |

---

## 🧪 1. Search Strategies

### Grid Search
We define a list of values for each hyperparameter, and the algorithm tries every single combination.
- **Search Space**: If we tune 3 parameters with 5 values each $\rightarrow 5^3 = 125$ training runs.

### Random Search
Instead of a grid, we define a distribution. The algorithm picks $N$ random combinations.
- **Why it works**: High-dim spaces often have only a few "influential" parameters. Random search samples these parameters more effectively than a fixed grid.

---

## 📉 2. Overfitting vs. Underfitting Control

### Regularization Tuning
- **L1 (Lasso)**: Adds $\lambda \sum |\theta|$. Leads to sparsity.
- **L2 (Ridge)**: Adds $\lambda \sum \theta^2$. Keeps weights small.
- **Elastic Net**: A linear combination of L1 and L2.

### The Learning Rate ($\alpha$)
The most sensitive hyperparameter.
- **Too High**: Oscillates or diverges.
- **Too Low**: Converges too slowly or gets stuck in local minima.

---

## ❓ Interview Questions

**1. "What is the difference between a Model Parameter and a Hyperparameter?"**
> **Parameters** are learned from data during training (e.g., weights $w$, bias $b$). **Hyperparameters** are set by the engineer *before* training to control the learning process (e.g., learning rate, $K$ in KNN).

**2. "Why is Random Search often better than Grid Search?"**
> In a high-dimensional space, some parameters may not affect the outcome significantly. Grid search wastes time varying these unimportant parameters. Random search explores more distinct values for "important" parameters.

**3. "How does Bayesian Optimization work in simple terms?"**
> It builds a surrogate model (typically a Gaussian Process) of the objective function. It uses this model to decide which hyperparameters to try next by balancing **exploration** and **exploitation**.

---

## 💻 Implementation (Optuna Example)
```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 10)
    # Train model...
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```
