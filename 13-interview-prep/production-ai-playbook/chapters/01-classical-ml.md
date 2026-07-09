# Chapter 1: Classical Machine Learning in Production

This chapter covers how to adapt the `01-tabular-ml-template` to solve 18 specific scenarios requested in SDE-2/Staff level interviews. The core architecture (FastAPI, Docker, MLflow, Pydantic) remains exactly the same. Only the `models/train.py`, `models/predict.py`, and `api/schemas.py` files need to be modified.

---

## 1. Binary Classification
**Core Concept:** Predicting 0 or 1 (e.g., Churn vs No Churn).
**Implementation in Template:**
The default template is already configured for this. Use `xgb.XGBClassifier(objective="binary:logistic")`.
**Tradeoff:**
- **XGBoost:** High accuracy, handles missing values natively, but prone to overfitting if `max_depth` isn't tuned.
- **Logistic Regression:** Easily interpretable (weights directly map to feature importance), fast, but fails on non-linear data.

## 2. Multi-class Classification
**Core Concept:** Predicting exactly one class from $N$ classes (e.g., User Segment: High, Medium, Low).
**Code Changes:**
```python
# In models/train.py
clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3)
```
```python
# In api/schemas.py
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Class 0, 1, or 2")
    probabilities: list[float] = Field(..., description="Prob for each class")
```
**Tradeoff:** `multi:softmax` only returns the argmax class. `multi:softprob` returns the probability distribution across all classes, which is mandatory in production for calculating prediction confidence thresholds.

## 3. Regression
**Core Concept:** Predicting a continuous float (e.g., LTV - Lifetime Value in dollars).
**Code Changes:**
```python
# In models/train.py
reg = xgb.XGBRegressor(objective="reg:squarederror")
```
```python
# In api/schemas.py
class PredictionResponse(BaseModel):
    predicted_value: float = Field(..., description="Predicted continuous value")
```
**Tradeoff:** Use `reg:squarederror` (MSE) for standard regression. If you have extreme outliers in LTV (e.g., whales spending $10k), use `reg:pseudohubererror` (Huber Loss), which is robust to outliers compared to MSE.

## 4. Time Series Forecasting
**Core Concept:** Predicting future values based on historical sequential data.
**Implementation:** Do not use XGBoost directly without heavy feature engineering (lag features). 
**Code Changes:**
- Use `Prophet` or `StatsForecast` for univariate.
- If using Tabular ML (LightGBM/XGBoost), engineer lag features:
```python
# In data/preprocessor.py
df['revenue_lag_1'] = df['revenue'].shift(1)
df['revenue_rolling_7d_mean'] = df['revenue'].rolling(window=7).mean()
```
**Tradeoff:** ARIMA/Prophet models are statistically sound but struggle to incorporate exogenous features (weather, marketing spend). Tree-based models (XGBoost) easily handle exogenous features but cannot extrapolate trends outside the bounds of the training data.

## 5. Recommendation Systems (Tabular approach)
**Core Concept:** Predicting if User U will click Item I.
**Implementation:** Frame as a binary classification problem (Click-Through Rate / CTR prediction).
**Code Changes:**
- Input schema must include `user_features` and `item_features`.
- Use a Factorization Machine (e.g., `xLearn`) or standard XGBoost with target encoding for high-cardinality categorical variables (like `item_id`).

## 6. Ranking (Learning to Rank)
**Core Concept:** Ordering a list of items for a specific user (Search Results).
**Implementation:** Use `XGBRanker`.
**Code Changes:**
```python
# In models/train.py
ranker = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg")
ranker.fit(X_train, y_train, group=train_groups) # group specifies the number of items per query
```
**Tradeoff:** Pointwise ranking (standard classification) looks at 1 item at a time. Listwise ranking (`rank:ndcg`) optimizes the actual order of the entire list simultaneously, maximizing metrics like NDCG.

## 7. Anomaly Detection
**Core Concept:** Finding outliers in unlabeled data (e.g., Fraud detection without labels).
**Implementation:** Use Isolation Forest.
**Code Changes:**
```python
# In models/train.py
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.01) # Assume 1% of data is anomalous
```
```python
# In models/predict.py
# Isolation Forest returns -1 for anomaly, 1 for normal
pred = self.model.predict(df)[0]
is_anomaly = bool(pred == -1)
```

## 8. Clustering
**Core Concept:** Grouping similar users together (e.g., Player Segmentation).
**Implementation:** K-Means or DBSCAN.
**Code Changes:**
```python
# In models/train.py
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_train)
```
**Tradeoff:** K-Means requires you to know `K` (the number of clusters) upfront and assumes spherical clusters. DBSCAN discovers `K` automatically and handles weird shapes, but fails in high dimensions due to the curse of dimensionality.

## 9. Feature Engineering
**Production Best Practice:** NEVER put `df['feature'] = df['A'] / df['B']` inside the FastAPI `predict()` function. It causes Training-Serving Skew.
**Implementation:** Use `sklearn.pipeline.Pipeline` with custom transformers or use a Feature Store (Feast).
```python
from sklearn.base import BaseEstimator, TransformerMixin
class RatioFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_new = X.copy()
        X_new['income_to_age'] = X_new['income'] / X_new['age']
        return X_new
```

## 10. Feature Selection
**Core Concept:** Dropping useless features to reduce latency and over-fitting.
**Implementation:** Add a `SelectFromModel` step to the pipeline.
```python
# In models/train.py
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('feature_selection', SelectFromModel(RandomForestClassifier())),
    ('classifier', xgb.XGBClassifier())
])
```

## 11. Sparse Data
**Core Concept:** Datasets where most values are zero (e.g., One-Hot Encoded text bags, TF-IDF).
**Implementation:** Storing sparse data in standard pandas DataFrames wastes massive amounts of RAM and crashes Docker containers.
**Code Changes:**
```python
import scipy.sparse
# Convert dense pandas to sparse CSR matrix before feeding to XGBoost
X_train_sparse = scipy.sparse.csr_matrix(X_train.values)
model.fit(X_train_sparse, y_train)
```
**Tradeoff:** CSR (Compressed Sparse Row) matrices use drastically less memory and speed up matrix multiplication, but you lose pandas' column names and slicing convenience.

## 12. Imbalanced Datasets
**Core Concept:** Fraud (0.1%) vs Not Fraud (99.9%).
**Implementation:** Do NOT use SMOTE in production unless absolutely necessary (it hallucinates fake data and ruins probability calibration). Instead, use algorithm-level weighting.
**Code Changes:**
```python
# In models/train.py
scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
clf = xgb.XGBClassifier(scale_pos_weight=scale_pos)
```

## 13. Large Datasets (Out of Core)
**Core Concept:** Dataset is 500GB, RAM is 16GB.
**Implementation:** You cannot use Pandas. You must use Dask or Polars with lazy evaluation, and use XGBoost's distributed/out-of-core training.
**Code Changes:**
```python
import dask.dataframe as dd
import xgboost as xgb
# Connect to Dask cluster
client = Client()
df = dd.read_parquet('s3://massive-dataset/')
# Dask-XGBoost handles distributed training natively
dtrain = xgb.dask.DaskDMatrix(client, df[features], df[target])
```

## 14. Streaming Datasets
**Core Concept:** Real-time data arriving constantly (Kafka).
**Implementation:** Use an incremental learning algorithm. Standard XGBoost cannot easily append new data without catastrophic forgetting (though recent versions support `xgb_model` continuation).
**Code Changes:** Use `sklearn.linear_model.SGDClassifier` or `River` (a library for online ML).
```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="log_loss")
# In production, loop over Kafka stream
for batch_X, batch_y in kafka_stream:
    clf.partial_fit(batch_X, batch_y, classes=[0, 1])
```

## 15. Missing Values
**Core Concept:** User leaves a form field blank.
**Implementation:** If using XGBoost, do *nothing*. XGBoost algorithmically routes missing values (NaNs) to the child node that minimizes loss.
If using Scikit-Learn (Logistic Regression, Random Forest), you must impute.
```python
from sklearn.impute import SimpleImputer
# Use 'median' for numerical (robust to outliers), 'constant' for categorical
imputer = SimpleImputer(strategy='median')
```

## 16. Outliers
**Core Concept:** Age = 999.
**Implementation:** Cap/Clip the values before passing to the model.
```python
# In data/preprocessor.py
class OutlierCapper(BaseEstimator, TransformerMixin):
    def transform(self, X):
        # Cap at 99th percentile
        p99 = X.quantile(0.99)
        return X.clip(upper=p99, axis=1)
```

## 17. Data Leakage Prevention
**Core Concept:** The model memorizes future information (e.g., using `time_on_site` to predict `will_click`, when `time_on_site` isn't known until *after* they click).
**Prevention:**
1. Only use `Point-In-Time` (As-Of) joins when pulling data from Snowflake/BigQuery.
2. Ensure `train_test_split` does not shuffle Time-Series data.
```python
# For temporal data, NEVER shuffle
# BAD: train_test_split(X, y, shuffle=True)
# GOOD: 
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
```

## 18. Concept Drift
**Core Concept:** The distribution of the real world changes over time (e.g., COVID hits, consumer behavior flips).
**Implementation:** You cannot fix drift inside the `predict.py` script. Drift requires **Monitoring**.
**Code changes:** Introduce `Evidently AI` or `Alibi Detect`.
```python
# In an offline Airflow DAG that runs nightly
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=training_data, current_data=yesterdays_production_logs)
if report.as_dict()['metrics'][0]['result']['dataset_drift']:
    trigger_retraining_pipeline()
```
