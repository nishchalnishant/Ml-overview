---
module: Classical Ml
topic: Hyperparameter Optimization
subtopic: ""
status: unread
tags: [classicalml, ml, hyperparameter-optimization]
---
# Hyperparameter Optimization

---

## The Problem HPO Solves

**The problem**: You have a model with a learning rate, a tree depth, a regularization strength, and a dropout rate. None of these can be learned from the training data — they control how the model learns, not what it learns. Setting them poorly produces a model that either underfits (too constrained) or overfits (too free). The question is: how do you find good values without manually trying every combination, which would take weeks?

**The core insight**: Hyperparameter optimization is a black-box function optimization problem. The objective function is "validation score given this configuration of hyperparameters." It is expensive to evaluate (requires training a model), noisy (randomness in training), and has no closed-form gradient. The best strategy depends on how expensive evaluations are and how many hyperparameters you have.

---

## Grid Search

**The problem**: You want to exhaustively evaluate every combination of a small, well-defined set of hyperparameter choices.

**The core insight**: If you know which hyperparameter values are worth trying and there are few of them, enumerate all combinations. Every combination is evaluated; you are guaranteed to find the best combination in your grid.

**The mechanics**: Define a grid of discrete values for each hyperparameter. Evaluate the cross-validated score for every combination. Return the configuration with the best score.

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators':  [100, 300, 500],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}
gs = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

Cost: $\prod_i |\text{grid}_i|$ evaluations. With 3 hyperparameters each having 3 values: 27 evaluations. With 5 hyperparameters each having 5 values: 3125 evaluations. Scales exponentially.

**What breaks**: Grid search evaluates the same values for each hyperparameter regardless of which hyperparameters actually matter. If only 1 of 5 hyperparameters affects the objective, grid search wastes evaluations on the 4 irrelevant ones. It also only covers discrete values you specify — the true optimum may lie between grid points.

---

## Random Search

**The problem**: Not all hyperparameters matter equally. If learning rate dominates and tree depth is irrelevant, grid search gives you only 3 distinct learning rate values regardless of how many depth values you tried. You want more exploration of the important dimensions.

**The core insight**: If you sample hyperparameter configurations randomly, each hyperparameter gets explored more independently. With a fixed budget of 50 evaluations, random search covers 50 distinct values of each hyperparameter. Grid search with the same 50 evaluations gives at most $\sqrt[k]{50}$ ≈ 3–4 values per hyperparameter for k=3.

**The mechanics**: Define a probability distribution over each hyperparameter's range (uniform, log-uniform, discrete). Sample N configurations i.i.d. Evaluate each configuration via cross-validation. Return the best.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'n_estimators':  randint(50, 500),
    'max_depth':     randint(3, 10),
    'learning_rate': loguniform(1e-3, 0.5),   # log-uniform because LR varies over orders of magnitude
    'subsample':     [0.6, 0.8, 1.0],
}
rs = RandomizedSearchCV(
    xgb.XGBClassifier(), param_dist,
    n_iter=50, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1
)
rs.fit(X_train, y_train)
```

Use log-uniform distributions for learning rates and regularization strengths — these parameters vary over orders of magnitude and you want equal probability mass per order of magnitude.

**What breaks**: Random search is unguided — it doesn't learn from past evaluations. Evaluation 50 is as blind as evaluation 1. With limited budget and many relevant hyperparameters, Bayesian optimization is much more efficient.

---

## Bayesian Optimization

**The problem**: You have a limited budget of expensive evaluations (training a deep model takes hours). Random search wastes budget on configurations that nearby evaluations have already shown are poor. You want to use information from past evaluations to guide future ones.

**The core insight**: Maintain a surrogate model of the objective function — a cheap-to-evaluate model that estimates how good each configuration will be. Use the surrogate to choose the next configuration that balances exploration (try uncertain regions) and exploitation (focus on promising regions). Update the surrogate after each evaluation.

**The mechanics**:
1. Evaluate a small set of initial configurations (random or grid).
2. Fit the surrogate model on {(configuration, score)} pairs seen so far.
3. Use an acquisition function to pick the next configuration — the one with the highest expected improvement over the current best.
4. Evaluate the objective at that configuration. Add to the history.
5. Repeat.

**Acquisition functions**:
- **Expected Improvement (EI)**: Expected gain over the current best, under the surrogate's uncertainty. Standard choice.
- **Upper Confidence Bound (UCB)**: $\mu(\lambda) + \kappa \cdot \sigma(\lambda)$. Optimistic under uncertainty. $\kappa$ controls exploration.
- **Thompson Sampling**: Sample a function from the surrogate's posterior, maximize it. Simple and parallelizable.

```python
import optuna

def objective(trial):
    lr    = trial.suggest_float('lr', 1e-4, 0.3, log=True)
    n_est = trial.suggest_int('n_estimators', 50, 500)
    depth = trial.suggest_int('max_depth', 3, 10)
    model = xgb.XGBClassifier(learning_rate=lr, n_estimators=n_est, max_depth=depth)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
print(study.best_params)
```

**Optuna's default sampler — TPE (Tree-structured Parzen Estimator)**: Rather than fitting a Gaussian Process surrogate over the full configuration space, TPE models two densities: $l(\lambda)$ = distribution of configurations where score > threshold, and $g(\lambda)$ = distribution where score ≤ threshold. The next configuration is chosen to maximize $l(\lambda) / g(\lambda)$ — prefer configurations that look like good ones rather than bad ones.

**What breaks**: Bayesian optimization is sequential — each evaluation informs the next. Parallelizing it (running multiple evaluations simultaneously) partially breaks this, since the surrogate hasn't seen results from in-progress evaluations. Optuna uses asynchronous TPE to mitigate this. GP-based Bayesian optimization scales as O(n³) in the number of evaluations — practical only up to ~500 evaluations.

---

## Hyperband and ASHA

**The problem**: Training a deep neural network to full convergence to evaluate one hyperparameter configuration takes hours. With a budget of 100 configurations, that's hundreds of hours. But most bad configurations are already obviously bad after a few epochs — you're wasting compute on them.

**The core insight**: Run many configurations with a small budget (few epochs). Discard the worst performers early. Give the survivors more budget. Repeat. The total compute is much less than training all configurations to completion, and you still find good configurations.

**The mechanics — Successive Halving**: Start with $n$ configurations, each given budget $b_{min}$. After evaluating, keep the top $1/\eta$ fraction and give them $\eta \times$ more budget. Repeat until one configuration has budget $b_{max}$.

**Hyperband**: Runs multiple brackets of Successive Halving with different starting budgets, parallelizing the tradeoff between many cheap evaluations and few expensive ones.

```
Bracket (81 configs, η=3):
  Round 0: 81 configs × 1 epoch  → keep 27 best
  Round 1: 27 configs × 3 epochs → keep 9 best
  Round 2:  9 configs × 9 epochs → keep 3 best
  Round 3:  3 configs × 27 epochs → keep 1 best
  Round 4:  1 config  × 81 epochs
```

**ASHA (Asynchronous Successive Halving)**: Removes synchronization barriers — workers promote configurations to the next round as soon as they are ready, without waiting for all peers to finish. More efficient in distributed settings.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(metric='val_loss', mode='min', max_t=100, grace_period=5)
analysis = tune.run(train_fn, config=search_space, scheduler=scheduler, num_samples=50)
```

**What breaks**: Early stopping only works when performance at early rounds correlates with final performance. For hyperparameters that affect training stability (learning rate schedule, warmup) or late-training behavior (annealing), a configuration that looks bad early may be the best at convergence. The `grace_period` parameter prevents eliminating configurations too early.

---

## BOHB — Bayesian Optimization with Hyperband

**The problem**: Hyperband's configuration selection is random — it explores many configurations cheaply but doesn't use past results to guide which configurations to try next. Bayesian optimization guides selection but doesn't use multi-fidelity early stopping.

**The core insight**: Combine them. Use a TPE surrogate to propose promising configurations (guided search, like Bayesian optimization), but evaluate them using Hyperband's early stopping schedule (multi-fidelity). Get the best of both worlds: informed search and cheap early rejection.

**What breaks**: BOHB requires a surrogate that can handle multi-fidelity observations (configurations evaluated at different budgets contribute to the same surrogate). Implementation complexity is higher than either method alone. BOHB is the standard for expensive deep learning HPO; for cheap-to-train models, Bayesian optimization without Hyperband is sufficient.

---

## Population-Based Training (PBT)

**The problem**: You fixed the hyperparameters before training started. But the optimal learning rate at epoch 1 may differ from the optimal learning rate at epoch 100. HPO methods that treat hyperparameters as fixed miss schedules and dynamic adaptations that improve training.

**The core insight**: Evolve a population of models simultaneously. Models with poor performance copy the weights and hyperparameters of top performers and then randomly perturb those hyperparameters. This creates an evolutionary search over *hyperparameter schedules*, not just fixed configurations.

**The mechanics**:
1. Initialize a population of models with different random configurations.
2. Train all models in parallel for N steps.
3. Rank models by validation score.
4. Bottom 20% copy weights from top 20% and perturb their hyperparameters (multiply by 0.8 or 1.2, or resample).
5. Repeat.

**What breaks**: PBT requires training the full population in parallel throughout training — compute cost is K× a single training run where K is population size. The copied weights from a top model may not be compatible with the perturbed hyperparameters of the bottom model — warmup periods after copying are often needed.

---

## Cross-Validation for HPO

For unbiased performance estimation after HP tuning, see [Nested Cross-Validation](06-cross-validation.md#nested-cross-validation) in the cross-validation module.

---

## Practical Guidelines

**Search scale for learning rate and regularization**: These vary over orders of magnitude. Always search in log space — a log-uniform distribution gives equal probability mass per decade.
```python
lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)    # Optuna
# or
lr_dist = loguniform(1e-4, 0.1)                          # scipy
```

**Identify which hyperparameters matter**: Before spending compute, estimate HP importance. Optuna provides this directly:
```python
optuna.importance.get_param_importances(study)
```

Focus the remaining budget on the top 2–3 most important hyperparameters. Fixing less important ones at reasonable defaults reduces the search space dramatically.

**Parallelism**:
- Grid and random search are trivially parallel — no communication needed between workers.
- Bayesian optimization has a sequential bottleneck — each new configuration depends on all previous results. Use asynchronous variants or batch acquisition (evaluate k configurations simultaneously using approximate acquisition).

**Time series HPO**: Use forward-chaining splits (TimeSeriesSplit) inside the cross-validation fold used to evaluate each configuration — same reasoning as in cross-validation more broadly.

| Method | Evaluations needed | Best when |
|---|---|---|
| Grid Search | O(∏ grid sizes) | Few HPs (< 3), discrete choices, fast training |
| Random Search | 50–200 | Many HPs, unclear which matter, moderate compute |
| Bayesian Optimization | 20–100 | Expensive models, want guided search |
| Hyperband / ASHA | Scales with n_configs | Deep learning, early stopping is meaningful |
| BOHB | 20–100 | Expensive deep models, best general-purpose |
| PBT | Population size × training cost | Hyperparameter schedules, RL-style training |

For active-recall drilling on these terms, see [classical-ml-flashcards.md](classical-ml-flashcards.md).
