# Hyperparameter Optimization

Hyperparameters (HPs) are set before training — learning rate, tree depth, regularization strength. HP optimization (HPO) finds values that maximize validation performance.

---

## Search Strategies

### Grid Search

Exhaustively evaluate every combination in a predefined grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}
gs = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

**Cost:** O(∏ |grid_i|) evaluations. Scales exponentially with number of HPs.  
**Use when:** Few HPs (<3), discrete choices, fast training.

---

### Random Search

Sample HP combinations at random from distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': loguniform(1e-3, 0.5),
    'subsample': [0.6, 0.8, 1.0],
}
rs = RandomizedSearchCV(XGBClassifier(), param_dist, n_iter=50, cv=5, 
                         scoring='roc_auc', random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
```

**Why better than grid:** If only a few HPs matter, random search explores more values for those HPs. With 5 HPs and 25 evaluations, grid gives 5 values per HP; random gives 25 distinct values per HP.

**Use when:** Many HPs, unclear which matter, moderate compute budget.

---

### Bayesian Optimization

Build a surrogate model (usually Gaussian Process) of the objective function. Use an acquisition function to pick the next HP combination that trades off exploration vs exploitation.

**Loop:**
1. Fit surrogate model on `{(λ_i, score_i)}`
2. Maximize acquisition function → `λ_next`
3. Evaluate objective at `λ_next`
4. Add `(λ_next, score_next)` to history

**Acquisition functions:**
- **Expected Improvement (EI):** Expected gain over current best — most common
- **UCB:** `μ(λ) + κ·σ(λ)` — optimistic under uncertainty
- **Thompson Sampling:** Sample from surrogate, maximize sample

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 0.3, log=True)
    n_est = trial.suggest_int('n_estimators', 50, 500)
    depth = trial.suggest_int('max_depth', 3, 10)
    model = XGBClassifier(learning_rate=lr, n_estimators=n_est, max_depth=depth)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
print(study.best_params)
```

**Optuna internals:** Uses TPE (Tree-structured Parzen Estimator) by default — a non-Gaussian surrogate that models `p(λ|score > threshold)` and `p(λ|score ≤ threshold)`, then picks λ maximizing their ratio.

**Use when:** Expensive evaluations (minutes to hours per trial), limited budget (20–200 trials).

---

### Hyperband / ASHA

Start many configurations with small budgets (few epochs), progressively eliminate poor performers, allocate full budget to survivors. Based on Successive Halving.

```
Bracket 0: 81 configs × 1 epoch → keep 27
Bracket 1: 27 configs × 3 epochs → keep 9
Bracket 2:  9 configs × 9 epochs → keep 3
Bracket 3:  3 configs × 27 epochs → keep 1
Final:      1 config  × 81 epochs
```

**ASHA (Async Successive Halving):** Async variant — workers don't block on synchronization barriers. More efficient in distributed settings.

**BOHB:** Combines Hyperband's early stopping with TPE surrogate. State of the art for expensive models.

```python
# With Ray Tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(metric='val_loss', mode='min', max_t=100, grace_period=5)
analysis = tune.run(train_fn, config=search_space, scheduler=scheduler, num_samples=50)
```

**Use when:** Deep learning HP search, many configs to try, early stopping is meaningful.

---

### Population-Based Training (PBT)

Evolve a population of models simultaneously. Every N steps, underperforming models "exploit" top performers (copy weights) and "explore" (perturb HPs).

```
Population of 10 models train in parallel
Every 1000 steps:
    Bottom 20% copy weights from top 20%
    Perturb HPs of copied models (±20% or resample)
```

**Advantage:** Adapts HPs during training (not just before). Can find schedules (e.g., increasing LR then decreasing) that fixed strategies miss.  
**Use when:** Very expensive models where HPs should change during training.

---

## Cross-Validation for HP Search

Never evaluate HPs on the test set. Use nested cross-validation to get unbiased performance estimate.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner CV: HP selection
# Outer CV: Performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=inner_cv)
nested_score = cross_val_score(clf, X, y, cv=outer_cv, scoring='roc_auc')
print(f"Nested CV AUC: {nested_score.mean():.3f} ± {nested_score.std():.3f}")
```

**Time series:** Use time-based splits (forward chaining) instead of random folds.

---

## Practical Guidelines

**Log-scale for LR and regularization:** Search `10^{-4}` to `10^{-1}` in log space — equal probability mass per order of magnitude.

**HP importance ranking:** Use fANOVA or Optuna's `optuna.importance.get_param_importances()` to identify which HPs matter most — focus budget there.

**Warm starting:** Initialize Bayesian search with knowledge from similar tasks (transfer learning for HPO).

**Parallelism:** Grid/random search trivially parallelizable. Bayesian search has sequential bottleneck at acquisition — use parallelization strategies (Kriging Believer, Local Penalization).

---

## Key Interview Points

- Random search outperforms grid search when only a few HPs matter — each HP gets more distinct values explored.
- Bayesian optimization (TPE/GP + EI) is standard for expensive models: deep nets, large ensembles.
- Hyperband eliminates poor runs early — same budget explores far more configs than fixed-budget methods.
- BOHB = Hyperband + TPE — best general-purpose HPO for deep learning.
- Never tune on test set. Use nested CV for unbiased performance estimates.
- Optuna is the most popular Python library; supports TPE, CMA-ES, and integration with PyTorch/scikit-learn.
