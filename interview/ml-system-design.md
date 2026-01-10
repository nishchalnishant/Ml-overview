# ML System Design: 30+ Questions & Case Studies

---

##  Framework Questions

**1. What is the typical ML System Design framework?**
> 1. Requirements & Scope, 2. Data & Features, 3. Model Selection, 4. Evaluation, 5. Deployment & Monitoring.

**2. What questions should you ask first in a system design interview?**
> What is the business goal? What are latency/throughput requirements? What data is available? What is the scale?

**3. How do you choose between batch and real-time inference?**
> **Batch**: High throughput, can use heavy models, daily/hourly predictions. **Real-time**: Low latency, needs optimized models, on-demand predictions.

**4. What is the Two-Stage pattern?**
> **Retrieval**: Fast filtering of millions to thousands (ANN search). **Ranking**: Precise scoring of candidates with heavy model.

**5. Why not just use one model for everything?**
> Scale. Can't run a heavy model on millions of items per request. Two-stage balances speed and accuracy.

---

##  Recommendation Systems

**6. Design a YouTube-like video recommendation system.**
> **Candidate Gen**: Two-Tower embeddings for user/video. **Ranking**: Deep model with user history, video features, context. **Objective**: Maximize watch time, not clicks.

**7. What features would you use for video recommendations?**
> User: watch history, demographics, time of day. Video: embeddings, duration, category, freshness. Context: device, location.

**8. How do you handle the cold-start problem?**
> For new users: content-based, popular items, explicit preferences. For new items: content features, creator popularity.

**9. What is Collaborative Filtering?**
> Recommend based on similar users (User-User) or co-consumed items (Item-Item). Matrix Factorization (ALS, SVD).

**10. What is Content-Based Filtering?**
> Recommend items similar to what user liked based on item attributes.

**11. What is a Two-Tower model?**
> Separate networks for user and item. Compute dot product of embeddings for similarity. Efficient for large catalogs.

**12. What is Approximate Nearest Neighbor (ANN)?**
> Algorithms (Faiss, HNSW) for fast similarity search. Trade small accuracy for huge speedup.

**13. How do you evaluate recommendations offline?**
> Recall@K, Precision@K, NDCG, Mean Reciprocal Rank (MRR).

**14. How do you evaluate recommendations online?**
> A/B testing with CTR, watch time, session length, user retention.

---

##  Search & Ranking

**15. Design a search ranking system.**
> **Query Understanding**: Spell check, query expansion. **Retrieval**: BM25 + Dense. **Ranking**: Cross-encoder or LTR model.

**16. What is BM25?**
> Probabilistic ranking function based on term frequency. Better than TF-IDF for retrieval.

**17. What is a Bi-Encoder vs Cross-Encoder?**
> **Bi-Encoder**: Encode query and doc separately. Fast. **Cross-Encoder**: Encode (query, doc) together. Accurate but slow.

**18. What is Learning to Rank (LTR)?**
> Train model to rank documents. Pointwise, Pairwise (RankNet), Listwise (LambdaMART).

**19. What is NDCG?**
> Normalized Discounted Cumulative Gain. Measures ranking quality with position-aware discounting.

**20. How do you handle Positional Bias?**
> Users click top results regardless of relevance. Include position as feature in training, set to fixed value at inference.

---

##  Fraud & Anomaly Detection

**21. Design a credit card fraud detection system.**
> **Real-time**: Fast model (XGBoost) for immediate decisioning. **Features**: Velocity, amount deviation, location. **Feedback**: Human review provides labels.

**22. Why is fraud detection challenging?**
> Extreme class imbalance (<0.1% fraud), adversarial (fraudsters adapt), high cost of false negatives.

**23. What metrics do you use for fraud?**
> Precision-Recall AUC. Optimize for recall (catch fraud) with acceptable precision (don't block legit users).

**24. What features are useful for fraud?**
> Velocity (transactions in last hour), amount vs history, device fingerprint, location anomaly, time patterns.

**25. How do you handle concept drift in fraud?**
> Fraudsters evolve tactics. Retrain frequently, monitor feature distributions, use online learning.

---

##  Content Moderation

**26. Design a content moderation system for social media.**
> **Text**: BERT classifier. **Image**: CLIP or ResNet. **Rules**: Hashlist for known bad content. **Human review**: Queue for edge cases.

**27. What's the recall vs precision trade-off here?**
> **High recall**: Catch more harmful content but more false positives (user frustration). Balance depends on severity (violence vs spam).

**28. How do you handle multi-modal content?**
> Separate models for text, image, video. Fuse predictions via rules or meta-classifier.

---

##  Production & MLOps

**29. What is Data Drift?**
> Input feature distributions change over time. Detect with PSI, K-S test.

**30. What is Concept Drift?**
> Relationship between features and target changes. Harder to detect—monitor model performance.

**31. What is Train-Serve Skew?**
> Differences between training and serving data/logic. Use Feature Store, same preprocessing code.

**32. How do you monitor models in production?**
> Prediction distribution, feature distributions, latency, error rates, business metrics decay.

**33. What is a Feature Store?**
> Centralized system for storing, versioning, and serving features consistently for training and inference.

**34. What is Canary Deployment?**
> Release new model to small % of traffic, monitor, gradually expand if healthy.

**35. What is Shadow Mode?**
> New model runs in parallel with production, logs predictions, but doesn't affect users. Compare before launch.

**36. What is Model A/B Testing?**
> Split traffic between models, measure business metrics, statistical significance.

**37. How long should you run an A/B test?**
> Until you reach required sample size for statistical power (usually 80%) and significance level (5%).

---

##  Advanced Design Questions

**38. How would you design a model to predict customer churn?**
> **Features**: Usage patterns, support tickets, payment history. **Model**: XGBoost or survival analysis. **Evaluation**: Precision at top K (focus on high-risk).

**39. How would you design a spam email filter?**
> **Features**: Sender reputation, text content, links, attachments. **Model**: Logistic Regression or GBM. **False positive cost**: High (don't miss important emails).

**40. How would you design a demand forecasting system?**
> **Time series**: ARIMA, Prophet, or TFT (Temporal Fusion Transformer). **Features**: Holidays, promotions, weather. **Evaluation**: MAPE, RMSE.

**41. How would you design an ad click prediction system?**
> **Latency**: <10ms. **Model**: Logistic Regression or light GBM. **Features**: User, ad, context embeddings. **Calibration**: Critical for bidding.

**42. How would you design a price prediction system for real estate?**
> **Features**: Location, size, amenities, market trends. **Model**: GBM or neural net. **Evaluation**: MAPE. **Explain**: Use SHAP for interpretability.
