# Machine Learning System Design: Master Framework

> [!IMPORTANT]
> **The ML System Design Interview Checklist**
>
> Use this 7-step framework to lead any ML design discussion. 
> 
> 1. **Clarification & Scoping:**
>    - [] What is the core business objective? (e.g., maximize clicks, minimize churn).
>    - [] Success Metrics: Online (CTR, revenue) vs. Offline (AUC, nDCG).
>    - [] Constraints: Latency (P99 < 100ms), Scale (1B users), fresh data frequency.
> 
> 2. **Data & Labels:**
>    - [] Label generation (Explicit vs. Implicit).
>    - [] Handling data imbalance (resampling, class weights).
>    - [] Sampling strategy (Negative sampling, Importance sampling).
> 
> 3. **Feature Engineering:**
>    - [] Categorical (One-hot, Hashing, Embeddings).
>    - [] Numerical (Scaling, Binning, Interaction features).
>    - [] Feature Store logic (consistent training/serving).
> 
> 4. **Model Architecture:**
>    - [] Baseline (Logistic Regression, GBDT).
>    - [] Candidate Generation (Two-tower, ANN search).
>    - [] Ranking (Deep Cross Networks, Transformers).
> 
> 5. **Training Pipeline:**
>    - [] Loss functions (Cross-entropy, Weighted MSE, Pairwise Hinge).
>    - [] Evaluation strategy (Time-series split, Rolling window).
> 
> 6. **Serving & Infrastructure:**
>    - [] Online vs. Batch inference.
>    - [] Model compression (Quantization, Distillation).
>    - [] Caching (Redis) for high-frequency features.
> 
> 7. **Post-Production:**
>    - [] Monitoring for Data/Concept Drift.
>    - [] A/B testing & Shadow deployments.
>    - [] Continual learning/retraining strategy.

---

## Chapter 1: Machine Learning Primer
This chapter covers the foundations required for ML system-design interviews.

1. Feature Selection & Engineering
   * One-Hot Encoding – definition, issues (tree models & sparsity), best practices, usage at Uber.
   * Mean Encoding – concept, leakage avoidance, smoothing, CV methods, Instacart example.
   * Feature Hashing – the hashing trick, collisions, hash functions (Murmur, Jenkins …), trade-offs, and usage at Booking, Facebook, Yahoo etc.
   * Cross Features – conjunctions, high cardinality, use of hashing, Uber example, and relation to Wide & Deep models.
   * Embedding – Word2Vec (CBOW / Skip-gram), co-trained embeddings, and case studies from Instagram, DoorDash, YouTube (two-tower retrieval), LinkedIn (reverse pyramid Hadamard model), Pinterest (visual search), Twitter and others.
   * Evaluation of Embeddings – downstream task performance, t-SNE / UMAP visualization, clustering, similarity metrics (cosine, dot, Euclidean) and related issues (norm bias, initialization).
   * Numeric Features – normalization, standardization, log transform, clipping outliers, and Netflix timestamp feature example.
2. Training Pipeline
   * Data formats (Parquet, Avro, ORC, TFRecord), partitioning, handling imbalanced classes (weights, down/up-sampling, SMOTE note).
   * Label generation strategies (cold-start handling, LinkedIn course recommendation case).
   * Train/test splitting for time-series (sliding and expanding windows).
   * Retraining levels (0–3: cold, near-line, warm-start) and scheduler patterns.
3. Loss Functions & Metrics
   * Regression losses (MSE, MAE, Huber, Quantile).
   * Facebook’s Normalized Cross Entropy example for CTR prediction.
   * Forecast metrics (MAPE, SMAPE).
   * Classification losses (Focal, Hinge) and their industry uses (Amazon focal loss experiments, Airbnb hinge loss library).
4. Model Evaluation
   * Offline metrics (AUC, MAP, MRR, nDCG).
   * Online metrics (CTR, time spent, satisfaction rates).
   * A/B testing overview and budget-splitting concept.
5. Sampling Techniques
   * Random, Rejection, Weighted, Importance, Stratified and Reservoir sampling —with examples and code snippets.
6. Common Deep Learning Architectures
   * Wide & Deep architecture (Google Play example).
   * Two-Tower architecture (YouTube retrieval).
   * Deep Cross Network (DCN V2) for automatic cross feature learning.

\
Here are the detailed notes of Chapter 2 – Common Recommendation System Components from _“Machine Learning Design Interview” by Khang Pham (2022)_:

***

### Chapter 2 – Common Recommendation System Components

1\. Overview

Modern recommendation systems have three main components:

1. Candidate Generation – From a massive set (billions of items), generate a smaller subset (hundreds/thousands).
   * Example: YouTube reduces billions of videos to a few hundred.
2. Ranking (Scoring) – A more complex ML model ranks the selected candidates for precision.
3. Re-ranking (Post-processing) – Adjusts for freshness, diversity, fairness, or removes disliked/undesirable items.

***

#### 2. Candidate Generation

**A. Content-Based Filtering**

* Uses item features and user preferences.
* Recommend similar items to what a user has liked before.
* Example: Google Play recommends apps similar to those installed by the user.

Process:

1. Represent each item by a feature vector (e.g., binary or numeric).
2. Compute similarity metrics — commonly dot product or cosine similarity.
3. Rank items by similarity score.

Pros:

* No need for data from other users.
* Easy to scale.
* Works well for new users with limited data.

Cons:

* Limited diversity (tends to show more of the same content).
* Requires rich, high-quality item metadata.

***

**B. Collaborative Filtering**

* Uses user–item interaction data (ratings, clicks, views).
* Predicts user preferences based on similar users’ behavior.

Two main types:

1. Memory-based – User–user or item–item similarity (e.g., cosine similarity, Pearson correlation).
2. Model-based – Matrix factorization, SVD, or deep learning embeddings.

Trade-offs:

* Pros: Learns from large-scale interactions automatically.
* Cons: Struggles with cold-start (new items/users), requires sufficient data.

***

**C. Industry Examples**

* Pinterest:
  * Uses _co-occurrence_ of items in sessions for candidate generation.
  * Example: If a user frequently views pins A, B, and C together, recommend D, which is co-viewed with A and B.
  * Uses random walks on item–item graphs for deeper relationships.
* YouTube:
  * Uses multistage retrieval stacks.
  * Stage 1: Candidate generation via embeddings and ANN (Approx. Nearest Neighbor).
  * Stage 2: Ranking using deep models considering user and video features.

***

#### 3. Ranking

Ranking models refine the top-N candidates using a more complex model (e.g., MLP or Gradient Boosted Trees).

**A. Learning-to-Rank Methods**

1. Pointwise – Predicts score for each item independently.
2. Pairwise – Optimizes relative ordering between pairs (e.g., RankNet).
3. Listwise – Considers the entire ranked list for optimization (e.g., LambdaRank).

RankNet (Microsoft Research)

* Uses a neural network to predict the probability that item A should be ranked higher than item B.
* Loss function: Cross-entropy between predicted and true order probabilities.

Example Workflow:

1. Input: (User, Item A, Item B) pairs.
2. Output: Probability that A > B.
3. Optimized using gradient descent to minimize misordered pairs.

***

#### 4. Re-Ranking

Final adjustments applied to improve user experience or business objectives.

**A. Freshness**

* Boosts newer or recently updated items.
* Example: Social media feeds prioritize newer posts.

**B. Diversity**

* Ensures variety among recommendations to avoid monotony.
* Example: Netflix mixes genres in top recommendations.

**C. Fairness**

* Prevents bias towards popular items or specific categories.
* Example: Ensuring smaller creators get exposure.

***

#### 5. Position Bias

Users are more likely to click on items appearing higher on the screen, which biases model training.

Mitigation Techniques:

* Use position as a feature in training.
* Inverse Propensity Scoring (IPS): Reweight training examples inversely to their display position probability.

LinkedIn Example – “People You May Know” (PYMK):

* Uses impression discounting to reduce bias from position effects in the ranking process.

***

#### 6. Calibration

Calibration ensures predicted probabilities reflect real-world likelihoods.

Example:

* If the model predicts p(click) = 0.2, then roughly 20% of such predictions should click.

Techniques:

* Platt scaling (logistic regression)
* Isotonic regression
* Temperature scaling
*   Facebook’s method:

    q = \frac{p}{p + \frac{1-p}{w\}}

    where _w_ = negative downsampling rate.

***

#### 7. Nonstationary Problem

Data distribution changes over time (a.k.a. concept drift).

Solution Approaches:

* Frequent retraining.
* Bayesian Logistic Regression – combines historical + near-time data.
* Lambda Learner (LinkedIn) – balances old and new data adaptively.

***

#### 8. Exploration vs. Exploitation

Trade-off between showing known high-performing content vs. exploring new items.

Common Approaches:

* Thompson Sampling
* ε-Greedy Exploration
* Example: Showing new ads despite uncertainty to find high performers.

***

#### 9. Case Study: Airbnb

Lesson 1: Deep Learning ≠ guaranteed improvement over GBDT.

Lesson 2: Dropout didn’t improve online metrics despite small offline NDCG gains.

Lesson 3: Use proper weight initialization (e.g., Xavier, random uniform for embeddings).

Lesson 4: Use optimizers like LazyAdam for large embeddings.

***

#### 10. Interview Exercises

Sample design questions:

1. Recommend “interesting places near me” on Facebook — how to sample positive/negative labels?
2. Predict attendance at Facebook events.
3. Detect if an image contains real or poster humans (use depth info).
4. Design feed ranking for Facebook.
5. Detect illegal items on Marketplace.
6. LinkedIn job recommendation system.
7. Feed recommendations for non-friends.
8. LinkedIn job seniority classifier.
9. Twitter smart notifications.
10. TikTok hashtag relevance filtering.
11. Instacart food category prediction.
12. DoorDash one-week demand forecasting.

***

#### Summary

* Recommendation systems = Candidate Generation + Ranking + Re-ranking
* Must balance accuracy, freshness, diversity, and fairness
* Address bias, drift, and exploration
* Industry examples from YouTube, Pinterest, LinkedIn, Airbnb, Facebook provide real-world context
* Interviewers expect end-to-end system understanding + practical trade-offs

***

Here are the detailed notes of Chapter 4 – Fraud Detection System from _“Machine Learning Design Interview” by Khang Pham (2022)_:

***

Here are the detailed notes of Chapter 3 – Search System Design from _Machine Learning Design Interview_ by Khang Pham (2022):

***

### Chapter 3 — Search System Design

<br>

#### 1⃣  Introduction

<br>

A search engine connects user queries to relevant items (documents, profiles, products, or listings).

Modern ML-based search systems combine information-retrieval (IR) techniques with machine-learning ranking to improve precision and personalization.

Typical use cases:

* Google Search → web documents
* LinkedIn Talent Search → member profiles
* Airbnb → rental listings
* Spotify → music tracks

<br>

High-level pipeline

1.  Retrieval / Candidate Generation

    Retrieve a few thousand possibly relevant results from millions or billions.
2.  Ranking

    Score those results with ML models using query, document, and user features.
3.  Re-ranking / Post-processing

    Adjust the ranked list for freshness, diversity, personalization, and fairness.

***

#### 2⃣  Keyword vs. Semantic Search

| Aspect         | Keyword (Lexical)                      | Semantic (Embedding-based)           |
| -------------- | -------------------------------------- | ------------------------------------ |
| Matching logic | Token overlap, TF-IDF, BM25            | Vector similarity in embedding space |
| Data index     | Inverted index (Elasticsearch, Lucene) | Vector index (FAISS, ScaNN, Milvus)  |
| Pros           | Fast, interpretable                    | Handles synonyms & paraphrases       |
| Cons           | Misses semantic matches                | Expensive, requires embeddings       |

Example – Onebar semantic search

* Uses Universal Sentence Encoder (USE) to embed queries and docs.
* Stores vectors in a FAISS index (cosine similarity).
* Serves results through a lightweight gRPC microservice.

***

#### 3⃣  Core ML Building Blocks

| Step                | Purpose                                                                 | Example                     |
| ------------------- | ----------------------------------------------------------------------- | --------------------------- |
| Label generation    | Clicks, dwell time, saves, purchases → implicit relevance labels        | YouTube “watch next” clicks |
| Feature sets        | User, item, and context features (time, device, language)               |                             |
| Architecture        | Two-tower or pyramid models for retrieval; MLP/GBDT for ranking         |                             |
| Loss functions      | Cross-entropy (pointwise), pairwise ranking loss, listwise (LambdaRank) |                             |
| Serving constraints | Low latency (<100 ms), high QPS                                         | LinkedIn Talent Search      |

***

#### 4⃣  Ad / Search Ranking Example

<br>

Ad Ranking System resembles search:

* Retrieve eligible ads via targeting rules.
* Compute pCTR and expected value = bid × pCTR × quality.
* Auction (GSP or VCG) decides placement and cost.

<br>

Why separate retrieval & ranking?

* Retrieval = fast, coarse; Ranking = slow, precise.
* Allows different update cadences and independent scaling.

<br>

Feature update issue

* Static features (geo, language) vs. dynamic ones (recent clicks).
* Cache frequently updated ones in Redis / DynamoDB.

***

#### 5⃣  Training–Serving Skew

<br>

Mismatch between offline preprocessing and online inference pipelines can cripple performance.

<br>

Solutions

1. Unified transformation graph (TF Transform).
2. Feature logging at serving → reuse for training.
3. TFDV / schema validation → detect drift.

<br>

_Spotify example:_ four-month bug fixed via feature logging + validation.

***

#### 6⃣  Scaling Retrieval Service

<br>

Scalability patterns

* Stateless retrieval pods behind load-balancer.
* Autoscale via Kubernetes (request-rate metrics).
* Graceful shutdown to avoid dropping inflight queries.
* Avoid single points of failure.

***

#### 7⃣  LinkedIn Talent Search Case Study

<br>

Goal: rank candidates most likely to accept recruiter messages (InMail).

| Component | Description                                                 |
| --------- | ----------------------------------------------------------- |
| Metrics   | Precision@5/25, nDCG (offline); acceptance rate (online)    |
| Features  | Recruiter context, query text, candidate features           |
| Models    | XGBoost baseline → Two-tower neural model                   |
| Indexing  | Lucene inverted + forward indices                           |
| Serving   | Broker service aggregates shards; ML layer re-ranks results |

Key improvements

* Tri-gram text embeddings for semantic matching.
* Context-aware ranking (industry, seniority).

***

#### 8⃣  Embedding Serving at LinkedIn (“Pensive”)

| Layer             | Function                                             |
| ----------------- | ---------------------------------------------------- |
| Offline training  | Train two-tower model → user/job embeddings          |
| Storage           | Key-value store of embeddings (Feature Marketplace)  |
| Nearline pipeline | Generate embeddings for new users/jobs via Kafka     |
| Retrieval         | Approximate NN (FAISS / ScaNN) for sub-100 ms search |

***

#### 9⃣  Airbnb Search Ranking Case

<br>

Objective: maximize booking likelihood for a given query (location + dates).

Challenges: large candidate pool, < 200 ms latency.

| Component  | Detail                                               |
| ---------- | ---------------------------------------------------- |
| Model type | Binary classifier → predicts booking probability     |
| Features   | Listing price, image quality, location, user history |
| Split      | Time-based (train on past, validate on future)       |
| Metrics    | DCG, nDCG (offline); Conversion rate (online)        |
| Latency    | ≤ 100 ms total inference                             |

***

#### Common Challenges & Remedies

| Challenge             | Mitigation                             |
| --------------------- | -------------------------------------- |
| Training–serving skew | Shared transforms, feature logging     |
| Feature freshness     | Nearline feature stores (Redis, Feast) |
| High QPS              | Sharding, ANN retrieval                |
| Latency               | Two-stage ranking, caching             |
| Personalization       | User embeddings + contextual features  |
| Evaluation            | A/B testing + nDCG + latency SLAs      |

***

#### 1⃣1⃣  Interview Angles

1. Design LinkedIn-like talent search ranking.
2. How to combine keyword and semantic retrieval results?
3. How to detect and correct training–serving skew?
4. Scale embedding search for billions of docs.
5. Handle cold start for new listings or users.
6. Choose metrics for evaluating a search system.

***

#### Key Takeaways

| Concept         | Summary                                     |
| --------------- | ------------------------------------------- |
| Search pipeline | Retrieval → Ranking → Re-ranking            |
| Modeling        | Two-tower / GBDT / pairwise rankers         |
| Infrastructure  | Lucene + FAISS + Redis + TensorFlow Serving |
| Latency target  | ≤ 100 ms end-to-end                         |
| Metrics         | Precision@K, nDCG, Conversion rate          |
| Examples        | LinkedIn, Airbnb, Spotify, Onebar           |

***

### Chapter 4 — Fraud Detection System

<br>

#### 1. Overview

<br>

Fraud detection systems aim to identify abnormal or malicious behaviors in transactions, signups, reviews, etc., while minimizing false positives that affect user experience.

Fraud detection = a real-time classification + graph-based pattern recognition problem.

<br>

Examples:

* Credit card fraud detection (Visa, Stripe, PayPal).
* Fake reviews on Amazon/Yelp.
* Fake listings on Airbnb.
* Fake job postings on LinkedIn.

***

#### 2. Common Fraud Scenarios

| Category           | Examples                                              |
| ------------------ | ----------------------------------------------------- |
| Payment Fraud      | Stolen credit card usage, chargebacks, refunds abuse. |
| Account Fraud      | Fake account creation, credential stuffing.           |
| Content Fraud      | Fake listings, spam posts, phishing links.            |
| Promotion Abuse    | Coupon misuse, referral manipulation.                 |
| Collusive Behavior | Groups coordinating fake transactions or reviews.     |

***

#### 3. Data Sources for Fraud Detection

| Data Type        | Examples                                                |
| ---------------- | ------------------------------------------------------- |
| User Data        | ID, account age, registration source, login pattern.    |
| Transaction Data | Amount, time, IP, device fingerprint.                   |
| Graph Features   | Shared IPs, shared devices, shared payment instruments. |
| Behavioral Data  | Clicks, session duration, typing speed.                 |
| External Data    | Blacklists, 3rd-party credit databases.                 |

Challenges

* Class imbalance (fraud cases << normal).
* Concept drift (fraud tactics evolve).
* Cold start (new users).
* Latency constraints for real-time decisions.

***

#### 4. ML Pipeline Overview

<br>

**Step 1.**&#x20;

**Label Generation**

* Labels derived from confirmed fraud cases, chargebacks, manual reviews.
* Delay between fraud occurrence and confirmation → label delay.
* To reduce delay: Use proxy labels (e.g., user banned within 7 days).

<br>

**Step 2.**&#x20;

**Feature Engineering**

* Device-level: OS, browser, fingerprint hash.
* Network-level: IP range, ASN, geolocation mismatch.
* Transaction-level: frequency, amount deviation, velocity features.
* User-level: account age, historical rejection rate.

<br>

**Step 3.**&#x20;

**Feature Storage**

* Real-time features in Redis or Feature Store.
* Offline historical features in BigQuery / Snowflake.

***

#### 5. Model Design

| Type                           | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| Supervised                     | Logistic Regression, Random Forest, XGBoost, LightGBM.           |
| Semi-Supervised / Unsupervised | Autoencoder, Isolation Forest, One-Class SVM (detect anomalies). |
| Graph-based                    | GNNs (Graph Neural Networks) using entity relationships.         |

Label Imbalance Handling

* Oversampling fraud cases (SMOTE).
* Cost-sensitive learning (higher penalty for false negatives).
* Ensemble learning (bagging/boosting).

***

#### 6. Graph-Based Fraud Detection

<br>

**A. Why Graphs?**

Fraudsters collaborate → form hidden entity clusters (shared IP, email, phone, or payment methods).

Represent system as heterogeneous graph with:

* Nodes: users, devices, IPs, credit cards.
* Edges: transactions, logins, ownership.

<br>

**B. GNN Models**

* GraphSAGE, R-GCN, or GAT (Graph Attention Networks).
* Learn embeddings that capture neighbor behavior and structural signals.

<br>

Industry Examples

* PayPal: DeepWalk + GNN hybrid for risk scoring.
* Alibaba: Graph embedding + XGBoost ensemble.
* Twitter: Detecting bot clusters with user-IP graph propagation.

***

#### 7. Real-Time Fraud Detection Architecture

<br>

**Typical Flow:**

1. Event ingestion → Kafka stream (e.g., payment, login, signup).
2. Feature fetching → query Redis/feature store for live features.
3. Model serving → compute fraud probability.
4. Decision Engine → rules + ML output to approve/block/review.
5. Feedback loop → store labeled outcomes for retraining.

<br>

**Tech Stack:**

* Data pipeline: Kafka, Spark Streaming.
* Feature Store: Feast, Redis, Cassandra.
* Model Serving: TensorFlow Serving / XGBoost on Triton.
* Monitoring: Prometheus, Grafana, Data Quality alerts.

***

#### 8. Hybrid Rules + ML System

<br>

Fraud detection systems combine static rules + ML models.

| Component      | Function                                                             |
| -------------- | -------------------------------------------------------------------- |
| Rules Engine   | Fast heuristic checks: blacklists, amount > threshold, geo mismatch. |
| ML Model       | Captures subtle non-linear patterns.                                 |
| Ensemble Logic | Weighted combination or layered approach.                            |

Example:

Final Fraud Score =

0.6 \* ML\_Model\_Score + 0.4 \* Rule\_Engine\_Score

<br>

Reason:

Rules offer interpretability and immediate blocking, while ML improves generalization and recall.

***

#### 9. Evaluation Metrics

| Metric             | Meaning                                                         |
| ------------------ | --------------------------------------------------------------- |
| Precision          | % of detected frauds that are truly fraudulent.                 |
| Recall             | % of total frauds correctly detected.                           |
| F1-score           | Trade-off between precision and recall.                         |
| AUC-ROC            | Model discrimination capability.                                |
| PR-AUC             | Preferred for highly imbalanced data.                           |
| Cost-based Metrics | Combines financial cost of false negatives vs. false positives. |

Example:

At PayPal, cost of false negatives >> cost of false positives → optimize for high recall with manual review buffer.

***

#### 10. Handling Concept Drift

<br>

Concept drift: Fraud patterns evolve (e.g., new devices, IP ranges).

<br>

Detection methods

* Monitor feature distributions (Kolmogorov–Smirnov test).
* Retraining triggers based on drift thresholds.
* Rolling window training (e.g., last 4 weeks).

<br>

Solution Strategies

* Online learning (incremental updates).
* Ensemble with model aging (new + old models).
* Shadow models for silent A/B testing.

***

#### 11. Explainability & Compliance

<br>

Why important: Financial & legal regulations require interpretability.

<br>

Techniques

* SHAP / LIME for per-transaction explanations.
* Rule-based post-hoc filtering: “Blocked because multiple devices used.”
* Dashboard for fraud analysts to override false positives.

<br>

Case Study:

Stripe Radar uses “risk reason codes” like _“email mismatch”_, _“velocity anomaly”_ to assist manual reviewers.

***

#### 12. Case Studies

<br>

**1.**&#x20;

**PayPal**

* Graph-based feature + GNN → XGBoost hybrid model.
* 80% reduction in false negatives, lower latency with Redis cache.

<br>

**2.**&#x20;

**Airbnb**

* Embedding-based account similarity detection.
* Detect fake listings using text similarity + image hash + GNN edges.

<br>

**3.**&#x20;

**Facebook**

* Detect fake accounts by friend-graph similarity + device overlap.

***

#### 13. Common Interview Angles

1. Design a real-time credit card fraud detection system.
2. How to handle concept drift in fraud data?
3. Why combine rules + ML in production?
4. How would you detect collusive fraud using graph structure?
5. What metrics would you use for imbalanced fraud detection data?
6. How would you log features for real-time fraud model serving?

***

#### 14. Key Takeaways

| Aspect            | Summary                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------- |
| Core Task         | Binary classification + anomaly detection.                                                |
| Main Challenges   | Imbalance, latency, evolving patterns, explainability.                                    |
| Architecture      | Kafka → Feature Store → ML Model → Decision Engine.                                       |
| Best Practices    | Use ensemble (rules + ML), retrain frequently, monitor drift, add interpretability layer. |
| Industry Examples | Stripe Radar, PayPal, Airbnb, Facebook.                                                   |

***

Here are the detailed notes of Chapter 5 – Feed Ranking System from _“Machine Learning Design Interview” by Khang Pham (2022)_:

***

### Chapter 5 — Feed Ranking System

<br>

#### 1. Overview

<br>

A feed ranking system determines the order of posts, photos, or videos on platforms like Facebook, LinkedIn, Twitter, or Instagram.

Its objective is to maximize user engagement (e.g., clicks, likes, shares, dwell time) while maintaining diversity and fairness.

<br>

Core steps:

1. Candidate Generation – Fetch relevant posts (from friends, pages, or creators).
2. Ranking – Predict engagement probability.
3. Re-ranking – Adjust for freshness, diversity, and fairness.
4. Feedback loop – Log engagement and user behavior for retraining.

***

#### 2. Candidate Generation

<br>

**A. Sources of Candidates**

* Friend Graph: Posts from user’s direct connections.
* Follow Graph: Pages or accounts followed.
* Interest Graph: Communities, topics, hashtags, etc.
* Content Pools: Popular, trending, or location-based posts.

<br>

Example – LinkedIn:

* Candidates from 1st and 2nd-degree connections.
* Additional ones from followed companies or influencers.

<br>

**B. Candidate Selection**

Use lightweight heuristics or ML scoring to shortlist thousands of posts.

Common techniques:

* Recency filter: Ignore stale posts.
* Quality filter: Minimum engagement threshold.
* ANN retrieval: Based on embedding similarity (user–post).

***

#### 3. Ranking Model

<br>

Ranking model scores each candidate to estimate engagement likelihood.

<br>

**A. Input Features**

1. User features – activity level, connection graph, dwell time.
2. Content features – text, image, video embeddings.
3. Context features – device type, session time, network speed.
4. Interaction features – relationship strength, past engagement.

<br>

**B. Model Architectures**

| Type                 | Description                            | Examples                   |
| -------------------- | -------------------------------------- | -------------------------- |
| Logistic Regression  | Simple, interpretable                  | Early LinkedIn feed models |
| GBDT (XGBoost)       | Handles non-linearity, robust          | Facebook early ranking     |
| Deep Neural Networks | Learns embeddings + feature crosses    | Modern Meta / TikTok       |
| Wide & Deep / DeepFM | Combines memorization + generalization | Google Play, Instagram     |

Common Output:

* Predicted probability for actions (e.g., click, like, comment).
* Composite score = weighted sum of multiple action probabilities.

<br>

Formula Example:

FinalScore = 0.4 \* p\_like + 0.3 \* p\_comment + 0.3 \* p\_share

***

#### 4. Multi-Objective Optimization

<br>

Real feed ranking often has multiple competing objectives:

* Engagement (CTR, likes, shares)
* Retention (dwell time, session length)
* Freshness
* Creator fairness
* Content quality / safety

<br>

**Techniques:**

1. Linear combination of objectives.
2. Pareto optimal optimization – no objective can be improved without worsening another.
3. Constraint-based ranking – e.g., maximize CTR under fairness constraints.

<br>

Facebook Example:

Optimize engagement while constraining user well-being and content diversity.

***

#### 5. Post-Processing / Re-ranking

<br>

**A. Freshness**

*   Boost recent posts, decay older ones exponentially.

    FreshnessBoost = e^(-λ \* post\_age\_hours)

<br>

**B. Diversity**

* Avoid showing many similar posts consecutively.
* Methods: Topic clustering + round-robin from each cluster.

<br>

**C. Fairness**

* Ensure visibility for new or small creators.
* Example: LinkedIn boosts visibility of less-connected users’ posts.

<br>

**D. Content Safety**

* Apply moderation filters to block sensitive or spammy content.

***

#### 6. Position Bias & Debiasing

<br>

Users tend to click higher-ranked items → causes training bias.

<br>

Solutions:

* Use inverse propensity weighting (IPW) to correct training samples.
* Introduce randomized experiments (e.g., shuffled results) to collect unbiased data.
* Add position feature explicitly in the model.

<br>

LinkedIn Example:

Position-based impression discounting for feed ranking.

***

#### 7. Data Pipeline

<br>

**A. Offline Training**

* Aggregate engagement logs (clicks, dwell time, reactions).
* Join user, content, and context features.
* Train DNN/GBDT with millions of samples.

<br>

**B. Online Serving**

* Real-time feature lookup (Redis, Feature Store).
* Model inference in <100 ms.
* Output ranked list for rendering in user’s feed.

<br>

Feature freshness is critical — stale features lead to lower engagement.

***

#### 8. Exploration vs. Exploitation

* Exploitation: Show known engaging posts.
* Exploration: Try new posts or creators to discover potential engagement.

<br>

Common strategies:

* ε-greedy exploration: Randomly show new content with probability ε.
* Thompson sampling: Bayesian exploration balancing known vs. new items.
* Bandit models (UCB, LinUCB): Learn reward confidence intervals.

<br>

TikTok Example:

Uses contextual bandits to explore new creators while optimizing watch time.

***

#### 9. Feedback Loop & Training Refresh

* Log impressions, clicks, reactions, shares.
* Aggregate daily or hourly for retraining.
* Detect concept drift (user interests change).
* Continuous retraining pipelines (Airflow/SageMaker).

<br>

Feature drift monitoring:

Use Chebyshev distance / KL divergence on feature distributions.

***

#### 10. Evaluation Metrics

| Category        | Metrics                                       |
| --------------- | --------------------------------------------- |
| Offline         | AUC, log-loss, Precision@K, nDCG              |
| Online          | CTR, dwell time, retention rate               |
| User Experience | Time spent, satisfaction surveys              |
| Fairness        | Distribution of impressions among user groups |

A/B Testing:

* 1–5% traffic for new models.
* Guardrail metrics: session time, complaint rate, ad revenue.

***

#### 11. Case Studies

<br>

**A.**&#x20;

**Facebook Feed**

* Multi-stage ranking: candidate generation → early-stage scoring → final ranking.
* Models include user embeddings and MLP ranking towers.
* Optimizes for engagement and long-term satisfaction (measured via surveys).

<br>

**B.**&#x20;

**LinkedIn Feed**

* Combines GBDT for fast scoring + DNN for personalization.
* Adds features: content type, recency, relationship strength, language match.
* Multi-objective: engagement + creator diversity.

<br>

**C.**&#x20;

**Twitter Timeline**

* Real-time ranking using GBDT + deep re-ranker.
* Heavy use of recency features + text embeddings from tweets.

<br>

**D.**&#x20;

**TikTok For You Feed**

* Fully deep neural architecture.
* Uses watch-time prediction, pairwise ranking, and continuous feedback loops.

***

#### 12. System Design Pattern

<br>

Architecture Overview:

```
Data Sources → Feature Store → Candidate Generator → Ranker → Re-ranker → Logging → Retraining
```

Key components:

* Kafka / PubSub for event ingestion.
* BigQuery / Hive for training data.
* Redis for real-time feature retrieval.
* TensorFlow Serving / PyTorch Serve for ranking models.
* Airflow for model retraining.

***

#### 13. Practical Considerations

| Issue         | Best Practice                                      |
| ------------- | -------------------------------------------------- |
| Latency       | Keep total inference under 200 ms                  |
| Cold Start    | Use content-based or popular-item bootstrapping    |
| Feature Drift | Monitor daily; retrain weekly                      |
| Fairness      | Add group-based constraints or penalties           |
| Logging       | Store engagement, dwell time, and non-click events |

***

#### 14. Common Interview Questions

1. Design Facebook or LinkedIn Feed ranking.
2. How do you handle multi-objective optimization in ranking?
3. What are strategies to ensure content diversity?
4. How to balance exploration vs. exploitation?
5. What features would you use for ranking posts?
6. How would you detect drift or bias in feed data?
7. How to design real-time feature retrieval for ranking models?
8. What’s your approach to measuring online performance?

***

#### Key Takeaways

| Concept          | Summary                                          |
| ---------------- | ------------------------------------------------ |
| Main Goal        | Show most relevant, engaging, and fair content   |
| Architecture     | Candidate generation → ranking → re-ranking      |
| Key Challenges   | Bias, drift, latency, multi-objective trade-offs |
| Evaluation       | CTR, dwell time, diversity, fairness             |
| Industry Leaders | Facebook, LinkedIn, Twitter, TikTok              |

***

Here are the detailed notes of Chapter 6 – Ads Ranking System from _“Machine Learning Design Interview” by Khang Pham (2022)_:

***

### Chapter 6 — Ads Ranking System

<br>

#### 1. Overview

<br>

Ads ranking systems are among the most complex and revenue-critical ML systems, used by companies like Google, Meta, LinkedIn, and TikTok.

They aim to:

* Maximize platform revenue (via CPC, CPM, CPA bidding).
* Maintain user satisfaction (avoid irrelevant or spammy ads).
* Ensure advertiser ROI (return on ad spend).

<br>

Thus, the ads ranking model optimizes for a multi-objective balance between these three forces.

***

#### 2. Core Architecture

| Stage                    | Purpose                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| 1. Candidate Generation  | Retrieve thousands of eligible ads using targeting filters (location, interest, budget). |
| 2. Preliminary Filtering | Filter by campaign status, budget, and policy checks.                                    |
| 3. Ranking / Scoring     | Predict expected user response (CTR, CVR).                                               |
| 4. Auction               | Combine prediction with advertiser bid → compute ad score.                               |
| 5. Re-ranking / Serving  | Final adjustments for fairness, diversity, pacing, and fatigue.                          |

***

#### 3. Ads Auction Basics

<br>

Most platforms use Generalized Second Price (GSP) auctions.

<br>

Key Terms:

* Bid (bᵢ): Max price advertiser willing to pay per click.
* Predicted CTR (pᵢ): Model-predicted click probability.
* Ad Quality (qᵢ): Content quality, landing page score.
* Expected Value (EV): EVᵢ = bᵢ × pᵢ × qᵢ
* Rank Score: Scoreᵢ = EVᵢ + adjustments

<br>

Payment Rule (GSP):

* Winner pays the minimum bid needed to beat the next advertiser’s rank score.

<br>

Example:

If A and B are top bidders:

```
A: bid = 2, pCTR = 0.1 → score = 0.2
B: bid = 3, pCTR = 0.05 → score = 0.15
```

 A wins, pays (0.15 / 0.1) = $1.5 per click.

***

#### 4. Modeling Components

<br>

**A.**&#x20;

**CTR Model (Click-Through Rate)**

Predicts probability of click.

* Features: user demographics, ad creative, context (time, device).
* Models: Logistic Regression → GBDT → DeepFM / DIN / DCN.
* Loss: Cross-entropy.
* Calibration: Platt scaling, isotonic regression.

<br>

**B.**&#x20;

**CVR Model (Conversion Rate)**

Predicts probability of conversion given click.

* Problem: Highly sparse labels (many clicks, few conversions).
* Solution: Delayed feedback modeling, two-stage models (click → conversion).
* Advanced: Joint CTR–CVR modeling using ESMM (Entire Space Multi-Task Model) by Alibaba.

<br>

**C.**&#x20;

**pCTR × pCVR (Expected Conversion)**

Compute expected ROI to advertisers.

<br>

**D.**&#x20;

**Ad Quality / Relevance Models**

* Language and image quality.
* Landing page load time.
* Policy compliance (NLP models for harmful text).

***

#### 5. Feature Engineering

| Feature Type | Examples                                                       |
| ------------ | -------------------------------------------------------------- |
| User         | Age, gender, geo, historical click rate.                       |
| Ad           | Text embedding, creative type, advertiser ID, campaign budget. |
| Context      | Device, network, time of day, session type.                    |
| Interaction  | Past user–advertiser engagement, recency of exposure.          |

Embedding & Cross Features

* Deep models use feature embeddings (user, ad, query).
* Combine via attention networks or outer-product transforms (as in DeepFM).

***

#### 6. Multi-Objective Optimization

<br>

Ads ranking optimizes several goals simultaneously:

* User engagement → maximize CTR, CVR.
* Advertiser ROI → maximize conversions per cost.
* Platform revenue → maximize bid × CTR.
* Fairness & pacing → avoid overexposure of a few ads.

<br>

Approaches

1. Weighted sum of objectives.
2. Constraint-based optimization (e.g., maintain ≥ X% user satisfaction).
3. Multi-task learning (shared tower with multiple output heads).

<br>

Meta Example: Multi-task learning for pCTR + pCVR + watch time → shared embedding backbone.

***

#### 7. Real-Time Serving Flow

<br>

Step-by-step:

1. User opens feed / search page.
2. Retrieve eligible ads (from billions).
3. Fetch user & ad features from Redis / Feature Store.
4. Model server computes CTR/CVR → ad score.
5. Auction logic ranks ads → winner selection.
6. Winning ad delivered; impression logged.
7. User clicks or converts → feedback loop updates labels.

<br>

Latency: < 50 ms per request.

<br>

Tech stack examples:

* Kafka → Redis → TensorFlow Serving / TorchServe → Thrift / gRPC serving.

***

#### 8. Calibration and Normalization

<br>

Because ads models are trained with sampled negatives, predicted probabilities are biased.

<br>

Fixes:

* Use weighted sampling and re-scaling.
* Calibration models (e.g., Platt scaling).
* Probability normalization: adjust pCTR by device type, geography.

<br>

Facebook Example:

Downsampling negative examples by 1:1000 → use reweighting factor _w_:

p’(click) = \frac{p}{p + \frac{1-p}{w\}}

***

#### 9. Online Learning & Drift Handling

<br>

Ad markets change hourly.

Solutions:

* Incremental model updates every few hours.
* Online learning using delayed-feedback correction.
* Replay buffer for click histories.

<br>

Monitoring:

* Feature drift detection (TFDV / KS test).
* A/B test performance metrics (CTR, CPM, ROI).

***

#### 10. Fairness, Fatigue, and Pacing

| Challenge           | Solution                                       |
| ------------------- | ---------------------------------------------- |
| Ad Fatigue          | Penalize ads shown too often to same user.     |
| Budget Pacing       | Spread impressions evenly through the day.     |
| Advertiser Fairness | Add per-advertiser caps, diversity re-ranking. |
| User Fatigue        | Blend organic + sponsored content in feed.     |

Meta Ads Example: Use _exposure decay_ → penalize over-shown creatives:

Penalty = e^{-λ × exposure\\\_count}

***

#### 11. Evaluation Metrics

| Category | Metrics                                                |
| -------- | ------------------------------------------------------ |
| Offline  | AUC, Log-loss, PR-AUC, Calibration error.              |
| Online   | CTR, CVR, eCPM, ROI.                                   |
| Business | Revenue lift, advertiser retention, user satisfaction. |

Key Formulas

*   eCPM (effective cost per mille):

    eCPM = bid × pCTR × 1000
* ROI: ROI = \frac{Revenue - Spend}{Spend}
* Lift over control: Compare test vs baseline campaigns.

***

#### 12. Case Studies

<br>

**A.**&#x20;

**Google Ads**

* Two-stage: pCTR → pCVR → bid × quality.
* Uses Wide & Deep networks for candidate scoring.
* Performs budget pacing + multi-objective optimization with constraints.

<br>

**B.**&#x20;

**Facebook Ads (Meta)**

* Architecture: Sparse embeddings + shared deep towers (multi-task).
* Targets multiple outcomes (click, view, conversion).
* Online learning pipeline refreshes every few hours.

<br>

**C.**&#x20;

**Alibaba**

* ESMM / ESM² (multi-task modeling for CTR–CVR–CTCVR).
* Jointly trained shared embeddings → better calibration on conversion predictions.

<br>

**D.**&#x20;

**LinkedIn Ads**

* Gradient boosted trees for explainability.
* Two-tower embeddings for matching advertisers and users.

***

#### 13. Common Interview Questions

1. Design an ads ranking system for Facebook/LinkedIn.
2. How would you combine bids and CTR predictions?
3. How to handle delayed feedback in CVR prediction?
4. How do you ensure calibration in pCTR/pCVR?
5. How would you detect and mitigate ad fatigue?
6. What is the difference between GSP and VCG auctions?
7. How would you design pacing logic for budget management?
8. What are your trade-offs in multi-objective optimization?

***

#### 14. Key Takeaways

| Aspect        | Summary                                                         |
| ------------- | --------------------------------------------------------------- |
| Goal          | Balance user satisfaction, advertiser ROI, and platform revenue |
| Model Types   | CTR, CVR, and joint multi-task networks                         |
| Auction Logic | GSP with quality-adjusted rank score                            |
| Latency       | <50 ms end-to-end                                               |
| Challenges    | Sparse labels, drift, calibration, fairness                     |
| Leaders       | Google Ads, Facebook Ads, Alibaba, LinkedIn                     |

***



