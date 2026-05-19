# Machine Learning Engineering

Based on Andriy Burkov's _Machine Learning Engineering_.

---

## What Machine Learning Engineering Is

**The problem**: ML research optimizes for accuracy on benchmark datasets. Production ML must be accurate, reliable, maintainable, debuggable, and economically justified — all simultaneously. These constraints are absent from academic ML and require a distinct engineering discipline.

**The core insight**: an ML engineer's job is to convert business problems into ML solutions that work reliably in production, not to achieve state-of-the-art accuracy on held-out test sets.

**The mechanics**: MLE project lifecycle — nine stages:

```
1. Goal definition
2. Data collection
3. Data preparation
4. Feature engineering
5. Model training
6. Model evaluation
7. Model deployment
8. Model serving
9. Model monitoring and maintenance
```

Each stage produces artifacts that the next stage consumes. Failure at any stage propagates forward. Most production failures originate in stages 1-4 (problem framing and data), not in model architecture.

**What breaks**: the most common MLE failure is excellent accuracy on a metric that doesn't matter. A model that achieves 97% precision at 30% recall passes offline evaluation but is commercially worthless — it misses 70% of opportunities. Goal definition failure propagates through every subsequent stage.

---

## Project Prioritization

**The problem**: ML projects are expensive (compute, engineering time, data labeling) and uncertain. Teams spend months on projects that should never have started — either because the problem doesn't require ML, or because the accuracy required makes it economically unsolvable.

**The core insight**: prioritize by (impact / cost). Impact is high when ML replaces complex hand-coded logic, or when imperfect-but-fast ML is more valuable than perfect-but-slow human judgment. Cost grows superlinearly with required accuracy.

**The mechanics**: the superlinear cost curve is the most important concept in project prioritization:

```
Required accuracy    Relative cost
80%                  1x
90%                  10x
95%                  30x
99%                  100x
99.9%                1000x
```

This is not a linear function. Going from 90% to 99% is not twice as hard — it's roughly 10x as hard. Data collection, labeling, and edge case handling multiply costs for each additional accuracy point.

```python
def estimate_project_viability(
    required_accuracy: float,
    current_best_accuracy: float,
    business_value_per_point: float,
    cost_per_point: float
) -> dict:
    """
    Rough viability estimate before committing to a project.
    """
    accuracy_gap = required_accuracy - current_best_accuracy

    # Cost grows superlinearly with gap
    relative_cost = (accuracy_gap / 0.1) ** 2  # rough heuristic

    expected_value = accuracy_gap * business_value_per_point
    expected_cost = relative_cost * cost_per_point
    roi = expected_value / (expected_cost + 1e-6)

    return {
        "accuracy_gap": accuracy_gap,
        "relative_cost": relative_cost,
        "roi": roi,
        "viable": roi > 1.0
    }
```

Use pilot projects to reduce uncertainty: implement a simplified version on a data slice to validate feasibility before full commitment.

**What breaks**: teams anchor to accuracy requirements set by non-technical stakeholders who don't understand the cost curve. "We need 99% accuracy" sounds reasonable until you realize it costs 10x more than "we need 95% accuracy." Push back with data: show the cost curve and let the business decide how much accuracy they can actually afford.

---

## Problem Framing

**The problem**: business objectives ("reduce churn," "improve customer satisfaction") are not directly optimizable. ML needs a precise mathematical definition: what input does the model receive, what does it output, what label does it train on, and what metric does it optimize.

**The core insight**: the translation from business goal to ML problem has three steps: understand the business objective, define what decision the model makes, and translate that decision into a prediction task.

**The mechanics**: the translation table:

```
Business objective          | ML task               | Label definition
---------------------------|----------------------|-------------------
Reduce customer churn       | Binary classification | churned in 30 days
Improve search relevance    | Ranking               | clicked / booked
Detect payment fraud        | Binary classification | confirmed fraud
Predict inventory demand    | Regression            | units sold next week
Personalize recommendations | Ranking               | engaged / purchased
```

Static vs dynamic predictions:
- Static: compute offline on a schedule (e.g., nightly churn scores)
- Dynamic: compute at request time (e.g., real-time fraud score)

Static is simpler, cheaper, and sufficient when freshness of prediction doesn't matter.

Baselines must be defined before training begins:

```python
baselines = {
    "majority_class": DummyClassifier(strategy='most_frequent'),
    "rule_based": RuleBasedChurnPredictor(threshold_days=90),
    "logistic_regression": LogisticRegression()
}

# If the ML model can't beat these, it has no business being deployed
for name, baseline in baselines.items():
    baseline.fit(X_train, y_train)
    score = evaluate(baseline, X_test, y_test)
    print(f"{name}: {score}")
```

**What breaks**: optimizing a proxy metric that diverges from the business objective. A fraud model optimizing for AUC might achieve great AUC while blocking legitimate transactions at a rate that drives customers away. Always track the actual business KPI (fraudulent dollars prevented, legitimate transactions blocked, net fraud savings) in parallel with ML metrics.

---

## Data Collection and Labeling

### Data Quality Dimensions

**The problem**: ML models learn from data. Garbage in, garbage out — but the garbage is often invisible until the model ships and fails in specific, confusing ways.

**The core insight**: data quality has six independent dimensions. Each can fail independently. A dataset can be valid but not representative, complete but not timely.

**The mechanics**:

```
Dimension          | Failure mode
-------------------|-------------------------------------------
Completeness       | Key fields missing for subpopulations
Consistency        | Same entity labeled differently over time
Validity           | Values outside expected range or format
Uniqueness         | Duplicate records inflate class frequencies
Timeliness         | Stale features: yesterday's price in today's prediction
Representativeness | Training data doesn't reflect serving distribution
```

Data validation before training:

```python
import great_expectations as ge

# Define expectations about data quality
context = ge.DataContext()
suite = context.create_expectation_suite("training_data")

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="training_data"
)

# Define what "valid" data looks like
validator.expect_column_values_to_not_be_null("user_id", mostly=0.99)
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_values_to_be_in_set("country", valid_country_codes)
validator.expect_column_proportion_of_unique_values_to_be_between(
    "transaction_id", min_value=0.99  # near-uniqueness
)

results = validator.validate()
if not results.success:
    raise DataValidationError(f"Training data failed validation: {results}")
```

**What breaks**: data quality issues are often correlated with target labels in unexpected ways. Missing age values might be more common for a specific demographic. Removing rows with missing values introduces sampling bias. Always investigate why data is missing, not just handle it mechanically.

---

### Labeling

**The problem**: supervised learning needs labels. Labels created by humans are inconsistent (different labelers disagree), expensive, slow, and can introduce systematic bias if the labeling protocol is flawed.

**The core insight**: inter-annotator agreement (IAA) is a prerequisite to trust labels. If two expert labelers disagree 30% of the time, the label noise floor is at least 30% — no amount of model sophistication can overcome it.

**The mechanics**: measure Cohen's kappa before training on any manually labeled dataset:

```python
from sklearn.metrics import cohen_kappa_score

# Compute kappa between two annotators on the same 500-example subset
kappa = cohen_kappa_score(annotator_a_labels, annotator_b_labels)

# Interpretation:
# kappa > 0.8: high agreement, labels are reliable
# kappa 0.6-0.8: moderate agreement, investigate disagreements
# kappa < 0.6: low agreement, labeling protocol needs revision

print(f"Inter-annotator agreement (kappa): {kappa:.3f}")
if kappa < 0.6:
    raise LabelerAgreementError("Labeling quality too low to proceed")
```

Programmatic labeling (weak supervision) to scale:

```python
# Use Snorkel: combine multiple weak labeling functions
def lf_keyword_positive(example):
    """Label positive if review contains positive keywords."""
    if any(word in example.text.lower() for word in ["excellent", "great", "love"]):
        return 1  # positive
    return -1  # abstain

def lf_keyword_negative(example):
    if any(word in example.text.lower() for word in ["terrible", "broken", "waste"]):
        return 0  # negative
    return -1  # abstain

def lf_short_review(example):
    """Short reviews with 5 stars are likely positive."""
    if example.stars == 5 and len(example.text) < 100:
        return 1
    return -1

from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel

lfs = [lf_keyword_positive, lf_keyword_negative, lf_short_review]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

label_model = LabelModel(cardinality=2)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100)
df_train['label'] = label_model.predict(L=L_train)
```

**What breaks**: programmatic labels are noisy approximations. A model trained entirely on Snorkel labels learns the combined noise of all labeling functions. Always validate on a manually labeled gold set. The acceptable noise level depends on the task — medical diagnosis can tolerate less label noise than product categorization.

---

### Data Leakage Prevention

**The problem**: features computed after the prediction time are included in the training data. The model learns to use information that would not be available at serving time — achieving high offline accuracy but failing completely in production.

**The core insight**: every feature in the training example must represent information that was available at the time the prediction would be made, not information that became available after the event occurred.

**The mechanics**:

```python
# WRONG: use_flag computed AFTER fraud is confirmed
# At prediction time, use_flag is not yet set
df['is_disputed_merchant'] = df['merchant_id'].isin(confirmed_fraud_merchants)
# This leaks future information into training

# RIGHT: only use information available at transaction time
df['merchant_risk_score'] = df['merchant_id'].map(
    merchant_risk_scores_as_of_transaction_time  # point-in-time snapshot
)

# For time-series data: strict temporal split
# Never shuffle before splitting
df = df.sort_values('timestamp')
n = len(df)
train = df.iloc[:int(0.6*n)]
val = df.iloc[int(0.6*n):int(0.8*n)]
test = df.iloc[int(0.8*n):]

# Fit scaler ONLY on training set — never on val/test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[feature_cols])
X_val_scaled = scaler.transform(val[feature_cols])   # use training stats
X_test_scaled = scaler.transform(test[feature_cols])  # use training stats
```

**What breaks**: leakage is often subtle. Average target encoding (replace category with its average label value) is a common leakage source — the target value is encoded into the feature. Compute target encoding only on training folds, never on the entire dataset before splitting.

---

## Data Preparation

### Missing Value Treatment

**The problem**: most ML algorithms cannot handle missing values. Dropping rows with missing data reduces dataset size and introduces selection bias. Imputing incorrectly misleads the model about what was observed.

**The core insight**: missing values are themselves a signal. Before imputing, ask why the value is missing. "Missing" might mean "never purchased," "test not administered," or "data pipeline error" — each requires different treatment.

**The mechanics**:

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Strategy selection based on missing mechanism:
# MCAR (Missing Completely At Random): impute with mean/median
# MAR (Missing At Random): impute with regression/KNN
# MNAR (Missing Not At Random): add indicator column

def handle_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    # Add missingness indicator before imputing (preserves the signal)
    for col in df.columns:
        if df[col].isnull().any():
            df[f'{col}_was_missing'] = df[col].isnull().astype(int)

    # Numerical: impute with median (robust to outliers)
    num_cols = df.select_dtypes(include='number').columns
    num_imputer = SimpleImputer(strategy=strategy)
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Categorical: impute with mode or "UNKNOWN" category
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")

    return df
```

**What breaks**: KNN imputation is accurate but O(n^2) — it's computationally infeasible on large datasets. Use it only for small datasets (<100K rows). For large datasets, use median imputation with missingness indicators.

---

### Outlier Treatment

**The problem**: extreme values pull regression coefficients, distort distance metrics, and cause numerical instability in gradient-based optimization. A single transaction of $10M in a dataset of $20 average transactions can dominate the MSE loss.

**The core insight**: distinguish outliers that are data errors (should be removed) from rare-but-valid extreme values (should be retained, possibly winsorized). Removing valid extreme values introduces bias; keeping erroneous ones introduces noise.

**The mechanics**:

```python
import numpy as np
from scipy import stats

def treat_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR  # 3x IQR for conservative removal
        upper = Q3 + 3 * IQR

    elif method == 'zscore':
        z = np.abs(stats.zscore(df[column].dropna()))
        lower = df[column].mean() - 3 * df[column].std()
        upper = df[column].mean() + 3 * df[column].std()

    # Winsorize: clip to bounds (retain the row, cap the value)
    # Better than removing when outliers might be valid
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df
```

Log transformation for skewed features:

```python
import numpy as np
# For right-skewed features (income, transaction amount, counts):
df['log_amount'] = np.log1p(df['amount'])  # log(1 + x) handles x=0
```

**What breaks**: outlier treatment during training must be applied identically during serving. If you winsorize at the training set's 99th percentile during training, the serving pipeline must winsorize at the same absolute value — not at the 99th percentile of serving data (which changes). Store the clip bounds as model metadata.

---

### Scaling and Encoding

**The problem**: gradient-based algorithms (neural networks, SVMs, KNN) are sensitive to feature scales. A feature ranging 0-1,000,000 dominates a feature ranging 0-1. The model effectively ignores the small-scale feature.

**The core insight**: features must be scaled to comparable ranges before gradient-based learning. The scaler must be fit on training data only and applied identically at serving time.

**The mechanics**:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: x' = (x - mean) / std
# Use when: feature is approximately Gaussian; need to handle outliers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NEVER fit on test data

# MinMaxScaler: x' = (x - min) / (max - min)
# Use when: need values in [0, 1]; neural networks with sigmoid/tanh output
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)

# Categorical encoding
from sklearn.preprocessing import OneHotEncoder

# One-hot: for low-cardinality categories (<50 unique values)
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_encoded = ohe.fit_transform(X_train[['category', 'region']])

# Ordinal encoding: for ordered categories
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
ordinal_encoded = ord_enc.fit_transform(X_train[['risk_level']])
```

Cyclical encoding for temporal features:

```python
import numpy as np

def cyclical_encode(value: np.ndarray, period: float) -> tuple:
    """
    Encode periodic features (hour, day, month) preserving cyclical structure.
    hour=23 and hour=0 should be close; linear encoding makes them far apart.
    """
    sin_enc = np.sin(2 * np.pi * value / period)
    cos_enc = np.cos(2 * np.pi * value / period)
    return sin_enc, cos_enc

# Hour of day (period = 24)
hour_sin, hour_cos = cyclical_encode(df['hour'].values, 24)
# Day of year (period = 365)
day_sin, day_cos = cyclical_encode(df['day_of_year'].values, 365)
```

**What breaks**: storing scaler metadata incorrectly. The StandardScaler fitted on training data stores `mean_` and `scale_` attributes. If these are not saved with the model artifact, the serving pipeline cannot apply the same transformation. Use scikit-learn Pipelines or TF preprocessing layers to bundle scaler and model into a single artifact.

---

## Feature Engineering

### Derived Features and Interactions

**The problem**: raw features often don't linearly predict the target. A linear model cannot learn that "loan amount / income" is more predictive than either feature alone, unless you provide the ratio explicitly.

**The core insight**: domain knowledge encodes signal that algorithms cannot discover from raw features alone. A domain expert who knows that "debt-to-income ratio" is the key credit risk factor can create this feature directly; a model trained on raw income and loan amount must learn the interaction implicitly, requiring exponentially more data.

**The mechanics**:

```python
import pandas as pd
import numpy as np

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ratio features: encode relationships directly
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['credit_utilization'] = df['current_balance'] / (df['credit_limit'] + 1)

    # Temporal features from timestamps
    df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek.isin([5, 6]).astype(int)
    df['days_since_last_login'] = (
        pd.Timestamp.now() - pd.to_datetime(df['last_login'])
    ).dt.days

    # Interaction features: capture combined effects
    df['age_x_income'] = df['age'] * np.log1p(df['income'])  # nonlinear interaction

    # Aggregation features: user-level behavioral context
    user_stats = df.groupby('user_id')['amount'].agg([
        ('user_avg_amount', 'mean'),
        ('user_std_amount', 'std'),
        ('user_transaction_count', 'count')
    ]).reset_index()
    df = df.merge(user_stats, on='user_id', how='left')

    # Anomaly features: deviation from personal baseline
    df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1)

    return df
```

**What breaks**: aggregation features computed after the target event introduce leakage. "User's average transaction amount" computed over the entire user history includes future transactions. Compute aggregations using only data available at the prediction time — point-in-time joins.

---

### Feature Selection

**The problem**: adding features increases model dimensionality, computation cost, and risk of overfitting. Many features are redundant (high correlation), irrelevant (no predictive signal), or noisy (add variance without reducing bias).

**The core insight**: fewer well-chosen features usually outperform many poorly-chosen features. Feature selection removes features that hurt more than they help.

**The mechanics**:

Filter methods (fast, model-agnostic):

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.stats import chi2

# Mutual information: works for any relationship (linear or nonlinear)
mi_selector = SelectKBest(mutual_info_classif, k=50)
X_train_selected = mi_selector.fit_transform(X_train, y_train)
selected_features = np.array(feature_names)[mi_selector.get_support()]

# Correlation-based: remove highly correlated features (|r| > 0.9)
corr_matrix = pd.DataFrame(X_train).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X_train = X_train.drop(columns=to_drop)
```

Wrapper methods (accurate, expensive):

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination: iteratively remove least important features
rf = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=rf, n_features_to_select=30, step=0.1)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
```

Embedded methods (training-time selection):

```python
# L1 regularization zeroes out irrelevant features automatically
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
important_features = np.array(feature_names)[lasso.coef_ != 0]

# Tree-based feature importance
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(50).index
```

SHAP for consistent, model-agnostic importance:

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
```

**What breaks**: feature importance from one model does not transfer to another. A feature with high importance in a random forest might be irrelevant to a neural network (which can discover the same signal from raw features). Always reevaluate feature importance when changing model architecture.

---

## Model Training

### Loss Function Selection

**The problem**: the loss function defines what the model optimizes. Using MSE for a classification problem, or cross-entropy for a regression problem, produces a model that optimizes something mathematically valid but practically wrong.

**The core insight**: match the loss function to the structure of the prediction task and the cost structure of errors.

**The mechanics**:

```python
import torch.nn.functional as F
import torch

# Regression
mse_loss = F.mse_loss(predictions, targets)          # penalizes large errors heavily
mae_loss = F.l1_loss(predictions, targets)            # robust to outliers
huber_loss = F.huber_loss(predictions, targets)       # combines MSE and MAE

# Binary classification
bce_loss = F.binary_cross_entropy(probabilities, targets)
bce_logit_loss = F.binary_cross_entropy_with_logits(logits, targets)  # more numerically stable

# Imbalanced binary: weighted cross-entropy
pos_weight = torch.tensor([99.0])  # 99:1 class ratio
weighted_bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

# Multiclass classification
ce_loss = F.cross_entropy(logits, class_labels)       # softmax + NLL

# Ranking: pairwise BPR loss
pos_scores = model(user, positive_items)
neg_scores = model(user, negative_items)
bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
```

**What breaks**: loss function and evaluation metric can diverge. Training with cross-entropy (which the optimizer can differentiate) but evaluating with F1 (which is non-differentiable) can produce models with good loss but poor F1. This is unavoidable — use surrogate losses that correlate with the desired metric, or use differentiable approximations of the metric.

---

### Hyperparameter Tuning

**The problem**: hyperparameters (learning rate, regularization strength, architecture size, tree depth) are not learned from data — they must be chosen before training. Wrong hyperparameters produce models that underfit, overfit, or fail to converge.

**The core insight**: hyperparameter tuning is a nested optimization problem. The outer loop searches hyperparameter space; the inner loop trains the model and evaluates it on a validation set.

**The mechanics**:

```python
import optuna

def objective(trial):
    # Optuna suggests hyperparameter values based on prior trials
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    model = build_model(n_layers, hidden_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader)
    return val_loss  # Optuna minimizes this

# Bayesian optimization: uses prior results to guide next trials
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour budget

best_params = study.best_params
print(f"Best hyperparameters: {best_params}")
```

Learning rate finding (Smith 2015):

```python
# Increase learning rate exponentially; find where loss stops decreasing
lrs = torch.logspace(-6, 0, steps=100)
losses = []

for lr in lrs:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr.item())
    loss = train_one_step(model, optimizer, batch)
    losses.append(loss.item())

# Plot: pick LR at steepest descent (just before loss diverges)
optimal_lr = lrs[np.argmin(np.gradient(losses))].item()
```

**What breaks**: hyperparameter tuning with a small validation set is unreliable. With 1000 validation examples and 100 trials, one trial will score well by chance — you're overfitting the hyperparameters to the validation set. Use k-fold cross-validation for hyperparameter tuning when data is limited, or a separate holdout set for final evaluation that was never used during tuning.

---

### Preventing Overfitting

**The problem**: a model with sufficient capacity memorizes training examples rather than learning generalizable patterns. It achieves near-zero training loss but high validation loss. Additional training data, regularization, or model simplification is needed.

**The core insight**: the total error is bias² + variance + irreducible noise. Overfitting is high variance — the model is too sensitive to the specific training examples. Reduce variance by increasing training data, reducing model capacity, or adding explicit regularization.

**The mechanics**:

```python
# Diagnosis: plot training vs validation loss over epochs
import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Training')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Overfitting signature: training loss still decreasing, val loss increasing

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'best_checkpoint.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(torch.load('best_checkpoint.pt'))
                return True  # stop training
        return False
```

**What breaks**: early stopping only works if validation set is truly held out. If you tune any hyperparameters based on val performance and then also use early stopping based on val performance, you've implicitly used val data twice — requiring a separate final test set for unbiased evaluation.

---

## Model Evaluation

### Reproducibility

**The problem**: two runs of the same training code produce different models because of random initialization, data ordering, and non-deterministic CUDA operations. Without reproducibility, you cannot debug regressions, compare experiments fairly, or audit model behavior.

**The core insight**: all sources of randomness must be seeded. Code, data, environment, and random seeds together define a completely reproducible experiment.

**The mechanics**:

```python
import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """Seed all randomness sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic CUDA operations (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Experiment tracking: log everything
import mlflow

with mlflow.start_run(run_name="experiment_v1"):
    mlflow.log_params({
        "seed": 42,
        "model_type": "random_forest",
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.01,
        "data_version": "v2.3.1"
    })
    mlflow.log_metrics({
        "val_auc": val_auc,
        "val_f1": val_f1,
        "val_pr_auc": val_pr_auc
    })
    mlflow.sklearn.log_model(model, "model")
```

DVC for data versioning:

```bash
# Track dataset version
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc
git commit -m "Training data v2.3.1"

# Reproduce exact experiment
dvc repro  # runs pipeline with tracked data and code
```

**What breaks**: experiment tracking without data versioning is incomplete. Two experiments can use the same code but different data (because the training pipeline re-ran overnight and added new rows). Track the exact data version hash alongside every experiment. MLflow + DVC together provide both.

---

### Model Comparison and Error Analysis

**The problem**: comparing models by a single aggregate metric hides whether the new model is genuinely better or just better on the majority of examples while being worse on important minority cases.

**The core insight**: a model upgrade is only an upgrade if it doesn't regress on important subgroups. Always do slice-level comparison and error analysis before shipping a new version.

**The mechanics**:

```python
def compare_models(model_a, model_b, test_df, slices: dict):
    """
    Compare two models across overall metrics and predefined slices.
    """
    results = []

    for slice_name, condition in [("overall", None)] + list(slices.items()):
        if condition:
            subset = test_df.query(condition)
        else:
            subset = test_df

        X = subset[feature_cols].values
        y = subset['label'].values

        for name, model in [("model_a", model_a), ("model_b", model_b)]:
            preds = model.predict_proba(X)[:, 1]
            results.append({
                "slice": slice_name,
                "model": name,
                "pr_auc": average_precision_score(y, preds),
                "n": len(subset)
            })

    return pd.DataFrame(results).pivot(index='slice', columns='model', values='pr_auc')

# Error analysis: examine misclassified examples
errors = test_df[test_df['pred'] != test_df['label']].copy()
errors['error_type'] = np.where(
    (errors['pred'] == 1) & (errors['label'] == 0), 'false_positive', 'false_negative'
)
# Examine patterns: what features distinguish errors from correct predictions?
errors.groupby('error_type')[feature_cols].mean()
```

**What breaks**: error analysis can identify patterns but not root causes without domain knowledge. A model that makes more errors on mobile users might reflect a mobile UI bug (wrong feature values), a different behavioral pattern (real model limitation), or sampling bias (mobile users are a different demographic). Distinguishing these requires human investigation, not just statistical analysis.

---

## Deployment

### Deployment Paradigms

**The problem**: different use cases have fundamentally different latency and throughput requirements. Fraud detection during payment needs a decision in <100ms. Nightly credit risk scoring can afford minutes per record. The wrong deployment mode wastes compute (batch when real-time is needed) or over-engineers infrastructure (real-time when batch suffices).

**The core insight**: latency requirement determines deployment mode. Batch for offline scoring where freshness doesn't matter; online API for synchronous real-time decisions; streaming for continuous event processing.

**The mechanics**:

```python
# Batch prediction: efficient, high-throughput, offline
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BatchScoring").getOrCreate()
df = spark.read.parquet("s3://data/users/features/")

# Broadcast model to all workers
import mlflow
model = mlflow.sklearn.load_model("models:/churn_model/production")
model_broadcast = spark.sparkContext.broadcast(model)

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import FloatType

@udf(FloatType())
def predict_udf(*features):
    import numpy as np
    x = np.array(features).reshape(1, -1)
    return float(model_broadcast.value.predict_proba(x)[0, 1])

feature_cols = ['age', 'days_inactive', 'purchase_count', 'support_tickets']
df_scored = df.withColumn('churn_score', predict_udf(*[df[c] for c in feature_cols]))
df_scored.write.parquet("s3://data/predictions/churn_scores/")
```

Online API serving:

```python
# FastAPI: real-time prediction endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI()
model = mlflow.sklearn.load_model("models:/fraud_detector/production")

class TransactionRequest(BaseModel):
    amount: float
    merchant_category: str
    user_id: str
    device_fingerprint: str
    hour_of_day: int

@app.post("/predict/fraud")
async def predict_fraud(request: TransactionRequest):
    try:
        features = extract_features(request)
        fraud_prob = model.predict_proba([features])[0, 1]
        return {
            "fraud_probability": float(fraud_prob),
            "decision": "block" if fraud_prob > 0.7 else "allow",
            "model_version": model.metadata.run_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**What breaks**: online serving fails when the feature pipeline has different latency characteristics than assumed. If a feature requires a database lookup that takes 50ms on average but 500ms at p99, the overall serving latency at p99 is dominated by that single feature lookup — not by model inference. Profile each feature source independently.

---

### Training-Serving Skew Prevention

**The problem**: the model was trained on features computed one way; the serving pipeline computes the same features differently. The model receives inputs at serving time that look systematically different from training inputs, degrading performance without any obvious error.

**The core insight**: the feature computation logic must be identical at training time and serving time. Any divergence — different libraries, different precision, different null handling — produces distribution shift the model cannot adapt to.

**The mechanics**:

```python
# Single feature transformation function used in both contexts
def compute_features(raw_data: dict, scaler=None, encoder=None) -> np.ndarray:
    """
    EXACTLY the same function called during training and serving.
    Version-controlled, tested, monitored.
    """
    age = float(raw_data.get('age', 0))
    income = float(raw_data.get('income', 0))
    loan_amount = float(raw_data.get('loan_amount', 0))

    # Derived features: same formula in both contexts
    loan_to_income = loan_amount / (income + 1e-6)

    # Scaling: use saved scaler from training
    features = np.array([[age, income, loan_amount, loan_to_income]])
    if scaler:
        features = scaler.transform(features)

    return features

# Save all artifacts needed for serving alongside model
import joblib
artifacts = {
    'model': trained_model,
    'scaler': fitted_scaler,
    'encoder': fitted_encoder,
    'feature_names': feature_names,
    'feature_compute_fn': compute_features,
    'training_data_schema': {
        'age': {'type': 'float', 'min': 0, 'max': 120},
        'income': {'type': 'float', 'min': 0, 'max': 1e7}
    }
}
joblib.dump(artifacts, 'model_package.joblib')
```

**What breaks**: functions in Python cannot be reliably serialized with joblib/pickle if they reference closures over mutable state. For production systems, the feature computation logic should be in a separate versioned library (a Python package), not inlined in a notebook.

---

## Monitoring and Maintenance

### Drift Detection

**The problem**: a model trained in January will encounter different data in July. User demographics shift, product inventory changes, economic conditions affect purchasing behavior. The model silently degrades because it was trained on a distribution that no longer reflects reality.

**The core insight**: monitor the input distribution continuously and alert when it diverges significantly from the training distribution. Do not wait for downstream metrics to decline — by then, users have already seen degraded predictions.

**The mechanics**:

```python
import numpy as np
from scipy import stats

def monitor_feature_drift(training_stats: dict, production_sample: np.ndarray,
                           feature_name: str, alpha: float = 0.05):
    """
    KS test for continuous features.
    Returns True if drift is detected.
    """
    reference = training_stats[feature_name]['sample']

    ks_statistic, p_value = stats.ks_2samp(reference, production_sample)

    is_drift = p_value < alpha
    if is_drift:
        print(f"DRIFT DETECTED: {feature_name}")
        print(f"  KS statistic: {ks_statistic:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Training mean: {np.mean(reference):.3f}")
        print(f"  Production mean: {np.mean(production_sample):.3f}")

    return is_drift

def compute_psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: stable
    PSI 0.1-0.2: moderate shift, investigate
    PSI > 0.2: major shift, retrain
    """
    ref_hist, bin_edges = np.histogram(reference, bins=n_bins)
    prod_hist, _ = np.histogram(production, bins=bin_edges)

    ref_pct = (ref_hist + 1e-6) / (ref_hist.sum() + 1e-6 * n_bins)
    prod_pct = (prod_hist + 1e-6) / (prod_hist.sum() + 1e-6 * n_bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi
```

**What breaks**: PSI and KS tests measure marginal feature distributions. Concept drift — where the relationship between features and labels changes without the feature distributions changing — is invisible to these tests. A fraud pattern can shift (fraudsters change tactics) while all individual feature distributions remain stable. Monitor prediction confidence distribution and actual outcomes (when available) in addition to feature statistics.

---

### Retraining Strategy

**The problem**: static models degrade over time. When and how to retrain is a decision with real costs — retraining too frequently wastes compute; retraining too infrequently means the model becomes stale.

**The core insight**: the right retraining frequency depends on how fast the data distribution changes. High-velocity domains (fraud, financial markets) need frequent retraining; stable domains (image classification of common objects) need it rarely.

**The mechanics**:

```python
class RetrainingManager:
    def __init__(self, psi_threshold=0.2, performance_threshold=0.05):
        self.psi_threshold = psi_threshold
        self.performance_threshold = performance_threshold

    def should_retrain(self, current_metrics: dict, baseline_metrics: dict,
                        feature_psi: dict) -> tuple:
        """
        Returns (should_retrain, reason).
        """
        # Performance-based trigger
        performance_drop = baseline_metrics['auc'] - current_metrics['auc']
        if performance_drop > self.performance_threshold:
            return True, f"Performance dropped {performance_drop:.3f} AUC"

        # Drift-based trigger
        drifted_features = [
            f for f, psi in feature_psi.items()
            if psi > self.psi_threshold
        ]
        if len(drifted_features) > 0:
            return True, f"Drift detected in features: {drifted_features}"

        return False, "No retraining needed"

    def select_training_data(self, all_data: pd.DataFrame,
                              strategy: str = 'rolling_window') -> pd.DataFrame:
        """
        Rolling window: use only recent data (good for fast-changing distributions)
        Full history: use all data (good for stable distributions with rare events)
        """
        if strategy == 'rolling_window':
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            return all_data[all_data['timestamp'] >= cutoff]
        elif strategy == 'full_history':
            return all_data
        elif strategy == 'weighted':
            # Exponential decay: recent examples weighted more
            all_data['sample_weight'] = np.exp(
                -0.01 * (pd.Timestamp.now() - pd.to_datetime(all_data['timestamp'])).dt.days
            )
            return all_data
```

**What breaks**: retraining with a rolling window discards historical rare events. If fraud patterns from 2 years ago reappear (old tactics recycled), a model trained on only the last 90 days has never seen them. Balance recency bias against the need to retain knowledge of rare but important patterns.

---

### Model Versioning and Rollback

**The problem**: a new model version is deployed and immediately degrades. Without versioned artifacts and a rollback mechanism, recovery requires retraining from scratch — unacceptable downtime for production systems.

**The core insight**: every model in production must have a versioned artifact in a model registry, with metrics logged and rollback tested before deployment.

**The mechanics**:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_to_production(model_name: str, candidate_run_id: str,
                           validation_metrics: dict, threshold: float = 0.02):
    """
    Promote a model to production only if it improves over current production.
    """
    # Get current production model metrics
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_run = prod_versions[0].run_id
        prod_metrics = client.get_run(prod_run).data.metrics
        current_auc = prod_metrics.get('val_auc', 0)
    else:
        current_auc = 0.0

    new_auc = validation_metrics['val_auc']
    improvement = new_auc - current_auc

    if improvement >= threshold:
        # Archive current production
        if prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_versions[0].version,
                stage="Archived"
            )

        # Promote new model
        new_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{candidate_run_id}/model",
            run_id=candidate_run_id
        )
        client.transition_model_version_stage(
            name=model_name, version=new_version.version, stage="Production"
        )
        return True, f"Promoted version {new_version.version} (AUC improvement: {improvement:.4f})"
    else:
        return False, f"Insufficient improvement: {improvement:.4f} < {threshold}"

def rollback_to_previous(model_name: str):
    """Revert to most recent Archived version."""
    archived = client.get_latest_versions(model_name, stages=["Archived"])
    prod = client.get_latest_versions(model_name, stages=["Production"])

    if not archived:
        raise ValueError("No archived version to rollback to")

    # Demote current production
    client.transition_model_version_stage(
        name=model_name, version=prod[0].version, stage="Staging"
    )
    # Restore archived version
    client.transition_model_version_stage(
        name=model_name, version=archived[0].version, stage="Production"
    )
```

**What breaks**: rollback without root cause analysis just defers the problem. If the new model failed because of a data pipeline bug, rolling back to the old model doesn't fix the bug — the old model is also receiving bad data. Rollback buys time; it is not a resolution.

---

## Ethics and Privacy

### Bias and Fairness

**The problem**: ML models trained on historical data inherit the biases in that data. A hiring model trained on past hires will perpetuate historical gender and racial discrimination because that pattern is encoded in the labels. Removing protected attributes (gender, race) doesn't fix it — correlated proxies (zip code, name, university) carry the same signal.

**The core insight**: fairness must be measured and enforced explicitly. It does not emerge automatically from removing protected attributes. Different fairness definitions (demographic parity, equal opportunity, equalized odds) are mathematically incompatible when base rates differ across groups — you must choose which to enforce.

**The mechanics**:

```python
import numpy as np

def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray,
                       sensitive_attr: np.ndarray) -> dict:
    """
    Evaluate model fairness across a binary sensitive attribute.
    """
    groups = {0: "group_A", 1: "group_B"}
    metrics = {}

    for g, g_name in groups.items():
        mask = sensitive_attr == g
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()

        metrics[g_name] = {
            "positive_rate": (tp + fp) / len(yt),           # Demographic parity
            "tpr": tp / (tp + fn + 1e-10),                  # Equal opportunity (TPR)
            "fpr": fp / (fp + tn + 1e-10),                  # Equalized odds (FPR)
            "n": mask.sum()
        }

    # Disparate impact ratio: must be >= 0.8 (80% rule, legal threshold)
    if "group_A" in metrics and "group_B" in metrics:
        di = (metrics["group_B"]["positive_rate"] /
              (metrics["group_A"]["positive_rate"] + 1e-10))
        metrics["disparate_impact_ratio"] = di
        metrics["passes_80_percent_rule"] = di >= 0.8

    return metrics
```

**What breaks**: the Impossibility Theorem (Chouldechova 2017) proves that demographic parity, equal opportunity, and calibration cannot all be satisfied simultaneously when group base rates differ. Any fairness intervention enforces one criterion at the cost of another. Document explicitly which fairness criterion was chosen and why.

---

### Privacy Preservation

**The problem**: ML models can memorize training data. A language model trained on medical records can be made to emit specific patient records through adversarial prompts. An image model trained on faces can reveal whether a specific person was in the training set through membership inference attacks.

**The core insight**: privacy-preserving techniques add noise or decentralize training to prevent information extraction, but they trade accuracy for privacy. The privacy-utility tradeoff must be explicitly calibrated.

**The mechanics**:

Differential privacy: add calibrated Gaussian noise to gradients during training.

```python
from opacus import PrivacyEngine

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data_loader = DataLoader(dataset, batch_size=64)

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    epochs=10,
    target_epsilon=5.0,   # privacy budget
    target_delta=1e-5,    # probability of privacy violation
    max_grad_norm=1.0     # gradient clipping bound
)
# epsilon=5.0: reasonable privacy; epsilon=1.0: very strong privacy (significant accuracy cost)
```

Federated learning: train on device, send only gradients.

```python
# Conceptual federated learning round
# Model lives on server; data never leaves device

def federated_round(global_model, client_datasets):
    local_updates = []

    for client_data in client_datasets:
        # Each client trains on local data
        local_model = copy.deepcopy(global_model)
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)

        for batch in DataLoader(client_data, batch_size=32):
            loss = compute_loss(local_model, batch)
            loss.backward()
            local_optimizer.step()
            local_optimizer.zero_grad()

        # Send weight delta (not raw data) to server
        delta = {k: local_model.state_dict()[k] - global_model.state_dict()[k]
                 for k in global_model.state_dict()}
        local_updates.append(delta)

    # Server aggregates: FedAvg
    new_weights = {k: global_model.state_dict()[k] +
                      torch.stack([u[k] for u in local_updates]).mean(0)
                   for k in global_model.state_dict()}
    global_model.load_state_dict(new_weights)
    return global_model
```

**What breaks**: differential privacy guarantees hold for the noise level specified but say nothing about model quality. With strong privacy (epsilon=0.1), gradient noise can be so large that the model fails to converge. With weak privacy (epsilon=100), the guarantee is nearly meaningless. Calibrate epsilon based on the sensitivity of the data, not a default value.
