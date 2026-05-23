---
module: References
topic: Book Notes
subtopic: Machine Learning Machine Learning Pocket Reference
status: unread
tags: [references, ml, book-notes-machine-learning]
---
# Machine Learning Pocket Reference

## Chapter 1: Introduction — The Python ML Ecosystem

**The problem the book is addressing**
The Python ML ecosystem has dozens of libraries that overlap in scope. New practitioners don't know which library to reach for, what the conventions are, or how they interoperate. Without a mental map of the ecosystem, time is wasted on tooling rather than problems.

**The core insight**
Three libraries cover 90% of practical ML: scikit-learn (classical ML), pandas (data manipulation), XGBoost/LightGBM (gradient boosting for tabular data). Deep learning adds PyTorch/TensorFlow. The key convention: all scikit-learn estimators have `.fit()`, `.predict()`, and `.transform()` — this uniformity enables pipelines and composition.

**The mechanics**
- scikit-learn: classification, regression, clustering, preprocessing, pipelines; consistent API
- pandas: DataFrames for tabular data; read CSV/parquet, filter, group, join, transform
- XGBoost/LightGBM: gradient boosted trees; dominant for tabular data competitions and production
- Matplotlib/Seaborn/Plotly: visualization — exploratory analysis, evaluation plots
- Interop: scikit-learn pipelines can wrap XGBoost; pandas DataFrames feed scikit-learn transformers

**What the book gets right / what to watch out for**
The ecosystem overview is accurate and practically oriented. The implicit recommendation — use tree-based models for tabular data, deep learning for images/text/audio — is correct. The book predates the tabular deep learning resurgence (TabNet, FT-Transformer) but the gradient boosting recommendation remains the correct default for tabular data.

---

## Chapter 2: CRISP-DM — The ML Project Process

**The problem the book is addressing**
ML projects fail not because models are wrong but because the team is solving the wrong problem, using the wrong metric, or doesn't have a process for iterating. An end-to-end project methodology prevents getting lost in one phase at the expense of others.

**The core insight**
CRISP-DM (Cross-Industry Standard Process for Data Mining) provides six iterative phases: business understanding → data understanding → data preparation → modeling → evaluation → deployment. The critical insight is that these phases cycle — evaluation often reveals data preparation issues that require going back.

**The mechanics**
- Business understanding: define the problem in business terms; translate to an ML objective and metric
- Data understanding: explore data, assess quality, identify issues (missing values, class imbalance, outliers)
- Data preparation: clean, transform, feature engineer, split into train/val/test
- Modeling: select algorithm family, train, tune hyperparameters
- Evaluation: assess on held-out test set with appropriate metrics; check against business criteria
- Deployment: serve predictions via API or batch job; monitor for drift

**What the book gets right / what to watch out for**
The cyclic nature of CRISP-DM is its most important aspect — the phases don't execute once in order. The framework underspecifies modern concerns (data versioning, experiment tracking, continual learning) that are covered in MLOps-focused books. Use CRISP-DM as a checklist for what phases to think about, not as a rigid waterfall.

---

## Chapter 3: Classification Walkthrough — End-to-End Titanic Example

**The problem the book is addressing**
Reading about ML components in isolation doesn't prepare practitioners for the full pipeline. Seeing a complete example — from raw data to a deployed model — reveals the decisions made at each step and the debugging required between them.

**The core insight**
A complete classification pipeline: load data → explore → clean/impute → feature engineer → encode categoricals → split → train RandomForest → evaluate (confusion matrix, AUC) → plot learning curves → serialize with pickle → wrap in Flask API. Every step in this pipeline has a scikit-learn equivalent.

**The mechanics**
- Load: `pd.read_csv('titanic.csv')`
- Explore: `.describe()`, `.value_counts()`, `.isnull().sum()`
- Clean: handle missing values (median imputation for age, mode for embarked)
- Encode: `pd.get_dummies()` or `OrdinalEncoder` for categoricals
- Split: `train_test_split(X, y, test_size=0.2, stratify=y)`
- Train: `RandomForestClassifier(n_estimators=100).fit(X_train, y_train)`
- Evaluate: `classification_report`, `roc_auc_score`, confusion matrix
- Learning curves: plot train/val score vs training size — diagnoses bias vs variance
- Serialize: `pickle.dump(model, open('model.pkl', 'wb'))`
- Serve: Flask endpoint loads pickle, returns JSON prediction

**What the book gets right / what to watch out for**
The end-to-end walkthrough is the right teaching format. The Flask deployment is rudimentary — production deployment requires containerization (Docker), health checks, logging, and load balancing. Pickling the model directly couples the deployment artifact to the scikit-learn version — use ONNX or mlflow model format for more robust serialization.

---

## Chapter 4: Missing Data — Imputation Strategies

**The problem the book is addressing**
Real-world data always has missing values. Most ML algorithms can't handle NaN natively (scikit-learn raises an error). Dropping rows with missing values loses information and can introduce bias. The right imputation strategy depends on why data is missing.

**The core insight**
Missing data mechanisms: MCAR (Missing Completely At Random — safe to drop), MAR (Missing At Random — impute using other features), MNAR (Missing Not At Random — most dangerous, distribution of missingness depends on the missing value itself). Visualizing the missingness pattern before choosing a strategy is essential.

**The mechanics**
- Visualization: `missingno` library — matrix plot shows missingness patterns; correlation heatmap shows if columns are missing together
- Simple imputation: `SimpleImputer(strategy='mean/median/most_frequent')` — fast, loses variance
- Iterative imputation: `IterativeImputer` — fits a model to predict each feature from others; better for MAR
- Indicator columns: `SimpleImputer(add_indicator=True)` — adds binary column flagging if value was missing; lets model learn from missingness pattern
- XGBoost/LightGBM: handle NaN natively; learn the best split direction for missing values during training
- CatBoost: handles missing values for categorical features natively

**What the book gets right / what to watch out for**
Adding indicator columns for missingness is underused but often valuable — the fact that a value is missing is itself informative (e.g., a patient doesn't have a lab test because the doctor didn't think it was needed). MNAR is the hardest case — no imputation strategy is correct when missingness depends on the unobserved value; domain knowledge is required.

---

## Chapter 5: Cleaning Data — pandas and pyjanitor

**The problem the book is addressing**
Raw data has inconsistent formats, duplicate rows, incorrect types, and out-of-range values. Cleaning is the most time-consuming ML task but practitioners lack a systematic checklist for what to look for.

**The core insight**
Cleaning has two phases: structural cleaning (fix types, rename columns, deduplicate) and semantic cleaning (fix values that are syntactically valid but semantically wrong — negative ages, future dates in past records). Pyjanitor provides a method-chaining API that makes cleaning pipelines readable.

**The mechanics**
- Type fixing: `df.astype({'col': 'float'})`, `pd.to_datetime()`, `pd.to_numeric(errors='coerce')`
- Deduplication: `df.drop_duplicates()`, `df.drop_duplicates(subset=['id'])` for key-based
- Rename: `df.rename(columns={'old': 'new'})` or pyjanitor `clean_names()` (lowercase, replace spaces)
- Filter outliers: `df[df['age'].between(0, 120)]` — domain-driven bounds
- pyjanitor: `df.clean_names().remove_empty().rename_column('old', 'new').filter_column_isin('col', values)`
- Validation: after each step, assert shapes and value ranges to catch mistakes early

**What the book gets right / what to watch out for**
The pyjanitor method-chaining pattern produces readable, auditable cleaning code. The implicit advice — write cleaning as a reproducible script, not interactive notebook edits — is correct. Cleaning steps must be applied identically to training and serving data; encapsulate them in a function or Pipeline to avoid train/serve skew.

---

## Chapter 7: Feature Engineering

**The problem the book is addressing**
Raw features often don't capture the signal the model needs. A timestamp is not useful as-is; hour-of-day and day-of-week are. Categorical variables with thousands of levels can't be one-hot encoded efficiently. Feature engineering is the primary way domain knowledge improves model performance.

**The core insight**
The goal of feature engineering is to create features that make the true relationship between input and output more linear (easier for models to learn). When you can't articulate the transformation, use target encoding or embeddings; when you can, explicit features are more robust.

**The mechanics**
- Dummy variables: `pd.get_dummies(df['col'])` — one-hot encoding; introduces (C-1) columns for C categories
- Label encoding: `OrdinalEncoder()` — integer ID per category; suitable for tree models, not linear models
- Frequency encoding: replace category with its frequency in training set — handles high cardinality without dummy explosion
- Target encoding: replace category with mean target value per category; adds signal but risks data leakage — use cross-val target encoding (`category_encoders.TargetEncoder`)
- Date features: extract year, month, day, hour, day-of-week, is_weekend, days_since_event
- Interaction features: `df['a*b'] = df['a'] * df['b']`; or `PolynomialFeatures` from scikit-learn

**What the book gets right / what to watch out for**
Target encoding is powerful but dangerous — computing target means on the full training set leaks label information into features. Always use cross-validation target encoding (compute mean from out-of-fold data). Frequency encoding is an underused alternative that adds signal without leakage risk.

---

## Chapter 9: Model Evaluation

**The problem the book is addressing**
Accuracy is the wrong metric for most real-world problems. A model that predicts the majority class achieves 95% accuracy on a 5% minority class problem while being completely useless. Practitioners need to know which metric to use and why.

**The core insight**
Metric selection should be driven by the business objective. For imbalanced binary classification, use AUC-ROC (threshold-invariant) or precision/recall/F1 (threshold-dependent). For ranked predictions, use MAP or NDCG. For regression, use MAE (robust to outliers) or RMSE (penalizes large errors more). Always report multiple metrics.

**The mechanics**
- Confusion matrix: TP, FP, TN, FN → foundation for all classification metrics
- Precision = TP/(TP+FP): of predictions that are positive, how many are correct
- Recall = TP/(TP+FN): of actual positives, how many were found
- F1 = 2·P·R/(P+R): harmonic mean; balances precision and recall
- AUC-ROC: area under the TPR vs FPR curve as threshold varies; 0.5=random, 1.0=perfect; class-imbalance robust
- ROC curve: plot TPR (recall) vs FPR (1-specificity) across all thresholds
- Learning curves: plot train/val score vs training set size; gap → overfitting; both low → underfitting or need more data
- Validation curves: plot train/val score vs hyperparameter value — identifies optimal hyperparameter range
- Lift curve: plots how much better than random the model performs; useful for marketing campaigns
- Cumulative gains: what fraction of positives found vs fraction of population targeted
- Discrimination threshold: plot precision/recall/F1 vs threshold; choose threshold based on business cost of FP vs FN

**What the book gets right / what to watch out for**
The discrimination threshold plot is underused — most practitioners fix the threshold at 0.5, but the optimal threshold depends on the relative cost of false positives and false negatives, which is a business decision. AUC-ROC is the correct primary metric for imbalanced binary classification when threshold will be tuned separately.

---

## Chapter 10: Hyperparameter Tuning

**The problem the book is addressing**
Default hyperparameters rarely produce the best model. Grid search over all combinations is exponential in the number of parameters. Practitioners need a systematic approach that explores the hyperparameter space efficiently.

**The core insight**
Random search outperforms grid search for the same compute budget when some hyperparameters don't matter — random search explores each dimension independently, so it wastes no budget on unimportant parameters. Bayesian optimization (Optuna/Hyperopt) goes further by using past results to guide future search.

**The mechanics**
- Grid search: `GridSearchCV(estimator, param_grid, cv=5)` — exhaustive, O(k^d) evaluations
- Random search: `RandomizedSearchCV(estimator, param_distributions, n_iter=100, cv=5)` — sample from distributions; use log-uniform for LR, uniform for regularization strength
- Validation curves: `validation_curve(estimator, X, y, param_name, param_range)` — shows train/val score vs single hyperparameter
- Optuna: `study.optimize(objective, n_trials=100)` — TPE sampler (Tree-structured Parzen Estimator); suggests next hyperparameter based on past results
- Hyperopt: similar to Optuna; slightly lower-level API
- Cross-validation: use stratified k-fold for classification; time-based split for time series; never use test set for tuning
- AutoML/TPOT: automatically searches over algorithms and hyperparameters; useful baseline

**What the book gets right / what to watch out for**
Log-uniform sampling for learning rate (sample from log₁₀ uniform over [-5, -1]) is critical — most of the meaningful range is in the low end. Cross-validation with the pipeline (not just the model) is essential — fitting transformers on the training fold and evaluating on the validation fold prevents data leakage.

---

## Chapter 11: Pipelines — Preventing Data Leakage

**The problem the book is addressing**
Feature preprocessing steps (imputation, scaling, encoding) are often fit on the full training set and then applied to validation folds. This leaks information about the validation set into the model, producing optimistic performance estimates.

**The core insight**
A scikit-learn Pipeline chains preprocessing steps and a final estimator. When used with cross-validation, the entire pipeline (including preprocessing) is fit on the training fold and evaluated on the validation fold — preventing leakage. The pipeline also ensures identical preprocessing at train and serve time.

**The mechanics**
- Simple pipeline: `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])`
- Preprocessing + model: `make_pipeline(SimpleImputer(), StandardScaler(), RandomForestClassifier())`
- `ColumnTransformer`: apply different transformers to different columns; `remainder='passthrough'` keeps untransformed columns
- GridSearchCV with pipeline: `GridSearchCV(pipeline, {'clf__C': [0.1, 1, 10]})` — double underscore notation for nested params
- Classification pipeline: impute → encode → scale → classify
- Regression pipeline: impute → scale → regress
- PCA pipeline: impute → scale → PCA → classify

**What the book gets right / what to watch out for**
The data leakage prevention argument for pipelines is the most important concept in this chapter. Target encoding inside a pipeline requires careful implementation — `category_encoders.TargetEncoder` wrapped in a `FunctionTransformer` inside a `Pipeline` still leaks unless the target encoding uses cross-validation internally. Use `category_encoders.LeaveOneOutEncoder` for safe integration.

---

## Chapter 12: Model Interpretation

**The problem the book is addressing**
"Black box" model predictions are hard to trust, debug, or explain to stakeholders. A model that achieves good aggregate performance but behaves unexpectedly for specific subpopulations fails in deployment.

**The core insight**
Model interpretation operates at three levels: global (what features matter overall — feature importance), local (why this specific prediction — LIME, SHAP values), and policy (what should the model predict given a different threshold — discrimination threshold plots). Different stakeholders need different levels.

**The mechanics**
- Linear model coefficients: `model.coef_` — directly interpretable if features are scaled; sign and magnitude indicate direction and strength
- Tree feature importance: `model.feature_importances_` — based on impurity reduction; biased toward high-cardinality features
- Permutation importance: shuffle one feature at random; measure drop in performance — unbiased, works for any model
- LIME: fit a local linear model around a specific prediction using perturbed samples; explains individual predictions
- SHAP: Shapley values from game theory; computes each feature's contribution to each prediction; consistent and additive; `TreeExplainer` is fast for tree models
- Partial dependence plots (PDPs): plot predicted outcome vs one feature while averaging over all others — shows marginal relationship
- treeinterpreter: decomposes predictions of tree models into contributions per feature — fast for tree-based models

**What the book gets right / what to watch out for**
SHAP is the gold standard for model interpretation — it's the only method that satisfies all desirable mathematical properties (efficiency, symmetry, dummy, additivity). LIME is faster but less consistent. PDPs can be misleading when features are correlated (the averaging marginalizes over unrealistic feature combinations). Always interpret model predictions in the context of feature correlations.

---

## Chapter 14: Serialization and Deployment

**The problem the book is addressing**
A model trained in a Jupyter notebook is not a production artifact. Getting predictions to users requires serializing the model, loading it in a serving environment, and wrapping it in an API. These steps introduce subtle bugs when environments differ.

**The core insight**
The serialization format should preserve everything needed to reproduce the prediction: model weights, preprocessing steps, and feature names. Pickle couples the artifact to the Python version and scikit-learn version; joblib is better for large arrays; ONNX provides framework-agnostic portability.

**The mechanics**
- pickle: `pickle.dump(model, open('model.pkl', 'wb'))` / `pickle.load(open('model.pkl', 'rb'))` — simple but version-coupled
- joblib: `joblib.dump(model, 'model.joblib')` — faster for NumPy-heavy objects; preferred for scikit-learn
- Flask API: `@app.route('/predict', methods=['POST'])`; load model at startup; parse JSON input; return JSON prediction
- Save the entire pipeline, not just the model — preserves preprocessing
- Include the scikit-learn version in the artifact metadata

**What the book gets right / what to watch out for**
Saving the entire pipeline (not just the model) is the correct advice — it prevents the common bug of forgetting to apply the same preprocessing at inference. Flask is appropriate for low-traffic services; production systems need a proper WSGI server (Gunicorn), containerization, and a reverse proxy. MLflow provides a more robust artifact and registry system for production.

---

## Chapter 15: Working with Text

**The problem the book is addressing**
Text is unstructured and variable-length. Before applying standard ML algorithms, text must be converted to fixed-length numerical vectors. The choice of vectorization method determines how much signal is captured and what information is lost.

**The core insight**
TF-IDF is the right baseline for text classification when you have moderate amounts of data and interpretability matters. Word embeddings (Word2Vec, GloVe) capture semantic similarity that bag-of-words misses. For high-accuracy production text classification, fine-tuned BERT embeddings are the current standard.

**The mechanics**
- NLTK preprocessing: tokenize → lowercase → remove stopwords → stem/lemmatize
- Bag of words: `CountVectorizer()` — frequency matrix; each column is a vocabulary word
- TF-IDF: `TfidfVectorizer()` — weights word counts by inverse document frequency; downweights common words
- Word embeddings: `gensim.models.Word2Vec` — train skip-gram on corpus; use document mean embedding as feature
- Pipeline: `Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])`

**What the book gets right / what to watch out for**
TF-IDF + LogisticRegression is a strong baseline that is often competitive with more complex approaches for short text. For production text classification, use sentence-transformers (pretrained BERT-based embeddings) as features — they generalize better across domains. NLTK stemming/lemmatization is slow; consider spaCy for production preprocessing.

---

## Chapter 16: Time Series

**The problem the book is addressing**
Time series data violates the IID assumption of standard ML — observations are correlated across time. Standard train/test splits with random shuffling leak future information into training. Patterns in time series (trend, seasonality, cycles) require specialized models.

**The core insight**
Time series decomposition separates a series into trend (long-term direction), seasonality (periodic patterns), and residuals (random noise). Stationarity (constant mean and variance) is required for ARIMA. Random train/test splits on time series always leak — always use a temporal holdout (last N% as test set).

**The mechanics**
- Components: trend (long-term), seasonality (periodic), cycles (irregular), residuals
- Stationarity test: Augmented Dickey-Fuller (ADF) — p < 0.05 → stationary; differencing achieves stationarity
- ACF/PACF plots: autocorrelation and partial autocorrelation functions — identify AR and MA orders for ARIMA
- ARIMA(p,d,q): AutoRegressive Integrated Moving Average; p=AR order, d=differences, q=MA order; `statsmodels.tsa.ARIMA`
- Exponential smoothing: simple (level), double (level+trend), triple (level+trend+seasonality/Holt-Winters)
- Prophet: Facebook's model; additive decomposition; handles holidays, missing data, multiple seasonalities
- Temporal split: train on first 80% of time period; validate on next 10%; test on last 10% — never shuffle

**What the book gets right / what to watch out for**
The temporal split requirement is the most important rule in time series ML — violating it produces wildly optimistic performance estimates. ARIMA is appropriate for univariate short series; for multivariate or longer-horizon forecasting, LightGBM with lag features or N-BEATS/TiDE neural models outperform ARIMA significantly.

---

## Chapters 17–18: Deep Learning — Neural Networks, CNNs, RNNs, Transfer Learning

**The problem the book is addressing**
Classical ML fails on unstructured data (images, text, audio) where features are hard to engineer manually. Deep learning automates feature learning but introduces new complexity: architecture design, batch training, GPU management, and evaluation at scale.

**The core insight**
Neural networks are universal function approximators built from linear transformations and nonlinearities. CNNs exploit spatial structure in images via weight sharing. RNNs handle variable-length sequences via hidden state. Transfer learning reuses features learned on large datasets, enabling good performance on small datasets.

**The mechanics**
- MLP: Linear → ReLU → Linear → Softmax; cross-entropy loss; Adam optimizer; early stopping
- CNN: [Conv → BatchNorm → ReLU → MaxPool] × N → Flatten → FC; input normalized to [0,1], then ImageNet mean/std
- RNN/LSTM: hidden state hₜ updated at each timestep; cross-entropy loss on next-token prediction
- Transfer learning: load pretrained model (ResNet/BERT); freeze backbone; add task head; fine-tune head first, then optionally unfreeze backbone
- Keras API: `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`; `model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[EarlyStopping()])`

**What the book gets right / what to watch out for**
Transfer learning from pretrained ImageNet or BERT weights is the correct default for practical tasks — almost always outperforms training from scratch. The Keras API is easy to get started with but hides important details (batch normalization behavior in train vs eval mode, gradient accumulation). For production, understanding the underlying PyTorch mechanics prevents subtle training bugs.

---

## Chapter 19: Clustering

**The problem the book is addressing**
Not all problems have labels. Customer segmentation, anomaly detection, and exploratory analysis require finding structure in unlabeled data. Clustering algorithms vary in assumptions about cluster shape, number of clusters, and handling of noise.

**The core insight**
K-means assumes spherical, equal-sized clusters with known K. Hierarchical clustering builds a tree of merges (no K needed) but is O(n²). DBSCAN finds arbitrary-shape clusters and identifies noise points without specifying K. The right algorithm depends on the expected cluster geometry and whether noise is present.

**The mechanics**
- K-means: `KMeans(n_clusters=k).fit(X)`; initialize k centroids; assign each point to nearest centroid; recompute centroids; repeat
- Choosing K: elbow method (plot inertia vs K, look for elbow); silhouette score (higher is better)
- Hierarchical: `AgglomerativeClustering(n_clusters=k, linkage='ward')`; build dendrogram; cut at desired distance
- Dendrogram: `scipy.cluster.hierarchy.dendrogram(linkage(X, 'ward'))` — visualizes merge distances
- DBSCAN: `DBSCAN(eps=0.5, min_samples=5)`; finds core points (≥min_samples within eps), expands clusters; marks outliers as -1

**What the book gets right / what to watch out for**
The silhouette score is more principled than the elbow method — it measures both cohesion and separation. DBSCAN is particularly useful for geospatial clustering (naturally arbitrary shapes) and anomaly detection (outliers are explicitly identified). K-means requires feature scaling — features with large variance dominate distance calculations.

---

## Chapter 20: Big Data — Hadoop, Spark, Dask

**The problem the book is addressing**
Datasets that don't fit in RAM break pandas and scikit-learn. Traditional ML tools assume single-machine, in-memory computation. Big data tooling provides distributed computation but introduces new APIs, serialization overhead, and cluster management complexity.

**The core insight**
Spark processes data in distributed memory across a cluster, with lazy evaluation (operations build a DAG and execute only when results are needed). Dask provides pandas-compatible API on a single machine for datasets that fit on disk but not in RAM. The right tool depends on data size: pandas < RAM, Dask < disk, Spark for cluster.

**The mechanics**
- Hadoop: HDFS (distributed filesystem) + MapReduce (distributed compute); largely superseded by Spark for ML
- Spark: DataFrames with lazy evaluation; MLlib for distributed ML (logistic regression, decision trees, ALS)
- Spark ML pipeline: `Pipeline([('indexer', StringIndexer()), ('encoder', OneHotEncoder()), ('lr', LogisticRegression())])`
- Dask: `dask.dataframe.read_parquet('path/*.parquet')` — lazy reads; `.compute()` triggers execution
- Batch processing: process fixed data snapshots; high throughput, high latency
- Stream processing: process data as it arrives (Kafka + Spark Streaming); low latency, lower throughput

**What the book gets right / what to watch out for**
The Spark API is correct and the Hadoop-to-Spark transition is accurately described. Dask is often sufficient and much easier to set up than Spark for single-node large-data problems. The key constraint: Spark is optimized for data-parallel operations on DataFrames — irregular operations (complex graph processing, sequential algorithms) are poorly suited for Spark. Arrow/Parquet is the standard interchange format between Spark, pandas, and ML frameworks.

## Flashcards

**scikit-learn?** #flashcard
classification, regression, clustering, preprocessing, pipelines; consistent API

**pandas?** #flashcard
DataFrames for tabular data; read CSV/parquet, filter, group, join, transform

**XGBoost/LightGBM?** #flashcard
gradient boosted trees; dominant for tabular data competitions and production

**Matplotlib/Seaborn/Plotly: visualization?** #flashcard
exploratory analysis, evaluation plots

**Interop?** #flashcard
scikit-learn pipelines can wrap XGBoost; pandas DataFrames feed scikit-learn transformers

**Business understanding?** #flashcard
define the problem in business terms; translate to an ML objective and metric

**Data understanding?** #flashcard
explore data, assess quality, identify issues (missing values, class imbalance, outliers)

**Data preparation?** #flashcard
clean, transform, feature engineer, split into train/val/test

**Modeling?** #flashcard
select algorithm family, train, tune hyperparameters

**Evaluation?** #flashcard
assess on held-out test set with appropriate metrics; check against business criteria

**Deployment?** #flashcard
serve predictions via API or batch job; monitor for drift

**Load?** #flashcard
pd.read_csv('titanic.csv')

**Explore?** #flashcard
.describe(), .value_counts(), .isnull().sum()

**Clean?** #flashcard
handle missing values (median imputation for age, mode for embarked)

**Encode?** #flashcard
pd.get_dummies() or OrdinalEncoder for categoricals

**Split?** #flashcard
train_test_split(X, y, test_size=0.2, stratify=y)

**Train?** #flashcard
RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

**Evaluate?** #flashcard
classification_report, roc_auc_score, confusion matrix

**Learning curves: plot train/val score vs training size?** #flashcard
diagnoses bias vs variance

**Serialize?** #flashcard
pickle.dump(model, open('model.pkl', 'wb'))

**Serve?** #flashcard
Flask endpoint loads pickle, returns JSON prediction

**Visualization: missingno library?** #flashcard
matrix plot shows missingness patterns; correlation heatmap shows if columns are missing together

**Simple imputation: SimpleImputer(strategy='mean/median/most_frequent')?** #flashcard
fast, loses variance

**Iterative imputation: IterativeImputer?** #flashcard
fits a model to predict each feature from others; better for MAR

**Indicator columns: SimpleImputer(add_indicator=True)?** #flashcard
adds binary column flagging if value was missing; lets model learn from missingness pattern

**XGBoost/LightGBM?** #flashcard
handle NaN natively; learn the best split direction for missing values during training

**CatBoost?** #flashcard
handles missing values for categorical features natively

**Type fixing?** #flashcard
df.astype({'col': 'float'}), pd.to_datetime(), pd.to_numeric(errors='coerce')

**Deduplication?** #flashcard
df.drop_duplicates(), df.drop_duplicates(subset=['id']) for key-based

**Rename?** #flashcard
df.rename(columns={'old': 'new'}) or pyjanitor clean_names() (lowercase, replace spaces)

**Filter outliers: df[df['age'].between(0, 120)]?** #flashcard
domain-driven bounds

**pyjanitor?** #flashcard
df.clean_names().remove_empty().rename_column('old', 'new').filter_column_isin('col', values)

**Validation?** #flashcard
after each step, assert shapes and value ranges to catch mistakes early

**Dummy variables: pd.get_dummies(df['col'])?** #flashcard
one-hot encoding; introduces (C-1) columns for C categories

**Label encoding: OrdinalEncoder()?** #flashcard
integer ID per category; suitable for tree models, not linear models

**Frequency encoding: replace category with its frequency in training set?** #flashcard
handles high cardinality without dummy explosion

**Target encoding: replace category with mean target value per category; adds signal but risks data leakage?** #flashcard
use cross-val target encoding (category_encoders.TargetEncoder)

**Date features?** #flashcard
extract year, month, day, hour, day-of-week, is_weekend, days_since_event

**Interaction features?** #flashcard
df['ab'] = df['a']  df['b']; or PolynomialFeatures from scikit-learn

**Confusion matrix?** #flashcard
TP, FP, TN, FN → foundation for all classification metrics

**Precision = TP/(TP+FP)?** #flashcard
of predictions that are positive, how many are correct

**Recall = TP/(TP+FN)?** #flashcard
of actual positives, how many were found

**F1 = 2·P·R/(P+R)?** #flashcard
harmonic mean; balances precision and recall

**AUC-ROC?** #flashcard
area under the TPR vs FPR curve as threshold varies; 0.5=random, 1.0=perfect; class-imbalance robust

**ROC curve?** #flashcard
plot TPR (recall) vs FPR (1-specificity) across all thresholds

**Learning curves?** #flashcard
plot train/val score vs training set size; gap → overfitting; both low → underfitting or need more data

**Validation curves: plot train/val score vs hyperparameter value?** #flashcard
identifies optimal hyperparameter range

**Lift curve?** #flashcard
plots how much better than random the model performs; useful for marketing campaigns

**Cumulative gains?** #flashcard
what fraction of positives found vs fraction of population targeted

**Discrimination threshold?** #flashcard
plot precision/recall/F1 vs threshold; choose threshold based on business cost of FP vs FN

**Grid search: GridSearchCV(estimator, param_grid, cv=5)?** #flashcard
exhaustive, O(k^d) evaluations

**Random search: RandomizedSearchCV(estimator, param_distributions, n_iter=100, cv=5)?** #flashcard
sample from distributions; use log-uniform for LR, uniform for regularization strength

**Validation curves: validation_curve(estimator, X, y, param_name, param_range)?** #flashcard
shows train/val score vs single hyperparameter

**Optuna: study.optimize(objective, n_trials=100)?** #flashcard
TPE sampler (Tree-structured Parzen Estimator); suggests next hyperparameter based on past results

**Hyperopt?** #flashcard
similar to Optuna; slightly lower-level API

**Cross-validation?** #flashcard
use stratified k-fold for classification; time-based split for time series; never use test set for tuning

**AutoML/TPOT?** #flashcard
automatically searches over algorithms and hyperparameters; useful baseline

**Simple pipeline?** #flashcard
Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])

**Preprocessing + model?** #flashcard
make_pipeline(SimpleImputer(), StandardScaler(), RandomForestClassifier())

**ColumnTransformer?** #flashcard
apply different transformers to different columns; remainder='passthrough' keeps untransformed columns

**GridSearchCV with pipeline: GridSearchCV(pipeline, {'clf__C': [0.1, 1, 10]})?** #flashcard
double underscore notation for nested params

**Classification pipeline?** #flashcard
impute → encode → scale → classify

**Regression pipeline?** #flashcard
impute → scale → regress

**PCA pipeline?** #flashcard
impute → scale → PCA → classify

**Linear model coefficients: model.coef_?** #flashcard
directly interpretable if features are scaled; sign and magnitude indicate direction and strength

**Tree feature importance: model.feature_importances_?** #flashcard
based on impurity reduction; biased toward high-cardinality features

**Permutation importance: shuffle one feature at random; measure drop in performance?** #flashcard
unbiased, works for any model

**LIME?** #flashcard
fit a local linear model around a specific prediction using perturbed samples; explains individual predictions

**SHAP?** #flashcard
Shapley values from game theory; computes each feature's contribution to each prediction; consistent and additive; TreeExplainer is fast for tree models

**Partial dependence plots (PDPs): plot predicted outcome vs one feature while averaging over all others?** #flashcard
shows marginal relationship

**treeinterpreter: decomposes predictions of tree models into contributions per feature?** #flashcard
fast for tree-based models

**pickle: pickle.dump(model, open('model.pkl', 'wb')) / pickle.load(open('model.pkl', 'rb'))?** #flashcard
simple but version-coupled

**joblib: joblib.dump(model, 'model.joblib')?** #flashcard
faster for NumPy-heavy objects; preferred for scikit-learn

**Flask API?** #flashcard
@app.route('/predict', methods=['POST']); load model at startup; parse JSON input; return JSON prediction

**Save the entire pipeline, not just the model?** #flashcard
preserves preprocessing

**Include the scikit-learn version in the artifact metadata?** #flashcard
Include the scikit-learn version in the artifact metadata

**NLTK preprocessing?** #flashcard
tokenize → lowercase → remove stopwords → stem/lemmatize

**Bag of words: CountVectorizer()?** #flashcard
frequency matrix; each column is a vocabulary word

**TF-IDF: TfidfVectorizer()?** #flashcard
weights word counts by inverse document frequency; downweights common words

**Word embeddings: gensim.models.Word2Vec?** #flashcard
train skip-gram on corpus; use document mean embedding as feature

**Pipeline?** #flashcard
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])

**Components?** #flashcard
trend (long-term), seasonality (periodic), cycles (irregular), residuals

**Stationarity test: Augmented Dickey-Fuller (ADF)?** #flashcard
p < 0.05 → stationary; differencing achieves stationarity

**ACF/PACF plots: autocorrelation and partial autocorrelation functions?** #flashcard
identify AR and MA orders for ARIMA

**ARIMA(p,d,q)?** #flashcard
AutoRegressive Integrated Moving Average; p=AR order, d=differences, q=MA order; statsmodels.tsa.ARIMA

**Exponential smoothing?** #flashcard
simple (level), double (level+trend), triple (level+trend+seasonality/Holt-Winters)

**Prophet?** #flashcard
Facebook's model; additive decomposition; handles holidays, missing data, multiple seasonalities

**Temporal split: train on first 80% of time period; validate on next 10%; test on last 10%?** #flashcard
never shuffle

**MLP?** #flashcard
Linear → ReLU → Linear → Softmax; cross-entropy loss; Adam optimizer; early stopping

**CNN?** #flashcard
[Conv → BatchNorm → ReLU → MaxPool] × N → Flatten → FC; input normalized to [0,1], then ImageNet mean/std

**RNN/LSTM?** #flashcard
hidden state hₜ updated at each timestep; cross-entropy loss on next-token prediction

**Transfer learning?** #flashcard
load pretrained model (ResNet/BERT); freeze backbone; add task head; fine-tune head first, then optionally unfreeze backbone

**Keras API?** #flashcard
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[EarlyStopping()])

**K-means?** #flashcard
KMeans(n_clusters=k).fit(X); initialize k centroids; assign each point to nearest centroid; recompute centroids; repeat

**Choosing K?** #flashcard
elbow method (plot inertia vs K, look for elbow); silhouette score (higher is better)

**Hierarchical?** #flashcard
AgglomerativeClustering(n_clusters=k, linkage='ward'); build dendrogram; cut at desired distance

**Dendrogram: scipy.cluster.hierarchy.dendrogram(linkage(X, 'ward'))?** #flashcard
visualizes merge distances

**DBSCAN?** #flashcard
DBSCAN(eps=0.5, min_samples=5); finds core points (≥min_samples within eps), expands clusters; marks outliers as -1

**Hadoop?** #flashcard
HDFS (distributed filesystem) + MapReduce (distributed compute); largely superseded by Spark for ML

**Spark?** #flashcard
DataFrames with lazy evaluation; MLlib for distributed ML (logistic regression, decision trees, ALS)

**Spark ML pipeline?** #flashcard
Pipeline([('indexer', StringIndexer()), ('encoder', OneHotEncoder()), ('lr', LogisticRegression())])

**Dask: dask.dataframe.read_parquet('path/*.parquet')?** #flashcard
lazy reads; .compute() triggers execution

**Batch processing?** #flashcard
process fixed data snapshots; high throughput, high latency

**Stream processing?** #flashcard
process data as it arrives (Kafka + Spark Streaming); low latency, lower throughput
