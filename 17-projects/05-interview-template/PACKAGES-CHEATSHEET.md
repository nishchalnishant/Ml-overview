```python
# --- core ---
import numpy as np
import pandas as pd

# --- split / preprocess ---
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- baseline models ---
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# --- metrics ---
from sklearn.metrics import roc_auc_score, average_precision_score  # classification
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # regression

# --- deep learning ---
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --- hyperparameter search ---
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
import optuna
```

```python
# ML TEMPLATE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]

df = pd.read_csv("data.csv")
X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
]).fit(X_train)
X_train, X_test = preprocessor.transform(X_train), preprocessor.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

```python
# DL TEMPLATE
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]

df = pd.read_csv("data.csv")
X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
]).fit(X_train)
X_train, X_test = preprocessor.transform(X_train), preprocessor.transform(X_test)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
    nn.Linear(64, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

x_t = torch.tensor(X_train, dtype=torch.float32)
y_t = torch.tensor(y_train.values, dtype=torch.float32)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(x_t).squeeze(-1), y_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1)).numpy()
print(roc_auc_score(y_test, preds))
```
