# Interview 17 — Large-Scale Collaborative Filtering Recommender
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the EA Play subscription platform team. The platform has millions of users and thousands of games across PC and console. You want to build a "Recommended Games for You" section on the homepage.

Your task is to **design a large-scale Collaborative Filtering recommendation system** based purely on players' game installation and playtime history.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Implicit vs Explicit feedback (Do we have 5-star ratings, or just playtime?)
- Data scale (How many users? How many games? Can it fit in memory?)
- Serving latency (Real-time vs Batch pre-computation?)
- Sparsity (Most players only play 2-3 games out of thousands).
- Cold start (What happens when a brand new game is added to the platform?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What kind of data do we have? Ratings or just playtime?"**
   → *Answer: We only have implicit feedback: total hours played per game per user.*

2. **"How many users and games are we dealing with?"**
   → *Answer: 20 million users, 5,000 games.*

3. **"How do we evaluate if the system is working?"**
   → *Answer: Offline: NDCG or Hit Rate @ 10. Online: Click-through rate and subsequent game installation rate.*

4. **"How do we handle brand new games added to EA Play?"**
   → *Answer: Good catch. Pure collaborative filtering fails here. You'll need to explain how to mitigate this.*

---

## Part 4 — Expected Assumptions

- **Algorithm:** Alternating Least Squares (ALS) for implicit feedback (Matrix Factorization). Neural approaches (Two-Tower) are also acceptable if justified.
- **Sparsity:** The user-item matrix is extremely sparse (e.g., 99.9% zeros).
- **Architecture:** Batch training (PySpark) ➔ Export Embeddings ➔ Vector DB / Redis ➔ Real-time Serving API.

---

## Part 5 — High-Level Solution

```
  [Data Lake (S3)]
  Raw Playtime Logs ➔ Group by User, Game ➔ Implicit Rating Matrix
       │
       ▼
  [Batch Training (PySpark cluster)]
  1. Train ALS (Alternating Least Squares) Model on Spark.
  2. Output User Embeddings (Dense Vectors).
  3. Output Game Embeddings (Dense Vectors).
       │
       ▼
  [Serving Infrastructure]
  Game Embeddings ➔ Qdrant / FAISS (Vector DB)
  User Embeddings ➔ Redis (Key-Value)
       │
       ▼
  [Real-Time API (FastAPI)]
  1. Fetch User Embedding from Redis.
  2. Query Vector DB (Cosine Similarity / Dot Product).
  3. Filter out games the user already installed.
  4. Return Top 10 Games.
```

**Core ML Component:** Matrix Factorization for Implicit Feedback. We cannot use standard SVD because it treats missing values as zeros. Implicit ALS assigns a "confidence" score based on playtime (more hours = higher confidence they actually like it).

---

## Part 6 — Step-by-Step Implementation

### Step 1: Implicit Ratings Formulation
- Let $r_{ui}$ be the actual playtime.
- Binarize it: $p_{ui} = 1$ if $r_{ui} > 0$ else $0$ (Did they play it?)
- Confidence: $c_{ui} = 1 + \alpha \cdot r_{ui}$ (The more they play, the more confident we are that they like it).
- Loss function minimizes $(p_{ui} - \vec{u}^T \vec{i})^2$ weighted by $c_{ui}$.

### Step 2: Distributed Training
- 20M users × 5K games is a 100 Billion entry matrix.
- `pyspark.ml.recommendation.ALS` is the industry standard for distributing this matrix factorization across a cluster.

### Step 3: Fast Serving (ANN)
- Instead of computing dot products for 5K games on every API request, we use a Vector DB (or just a fast numpy matrix multiplication in memory if 5K is small enough). Since 5K is small, we can actually keep all game embeddings in the FastAPI memory and just fetch the User embedding from Redis.

---

## Part 7 — Complete Python Code

*We will write the PySpark ALS training script, which is the core of large-scale CF.*

```python
"""
als_recommender.py - Spark ALS for Implicit Feedback
"""
import logging
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import col, collect_list
import redis
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spark Training Job
# ---------------------------------------------------------------------------
def train_als_model(spark: SparkSession, data_path: str):
    logger.info("Loading playtime data...")
    # Schema: user_id (int), game_id (int), playtime_hours (float)
    df = spark.read.parquet(data_path)
    
    # Split for evaluation
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    logger.info("Configuring Implicit ALS...")
    als = ALS(
        maxIter=15,
        regParam=0.1,
        alpha=10.0, # Weight for implicit feedback confidence
        userCol="user_id",
        itemCol="game_id",
        ratingCol="playtime_hours",
        implicitPrefs=True, # CRITICAL for playtime data
        coldStartStrategy="drop"
    )
    
    logger.info("Fitting model on 20M users...")
    model = als.fit(train)
    
    # Get embeddings
    user_factors = model.userFactors
    item_factors = model.itemFactors
    
    return model, user_factors, item_factors, test

def push_embeddings_to_redis(user_factors_df):
    """Pushes user vectors to Redis for real-time serving."""
    logger.info("Pushing User Embeddings to Redis...")
    # In reality, use spark-redis connector. Doing it in python driver for demo.
    r = redis.Redis(host='redis-cluster', port=6379)
    
    # Collect to driver (Warning: Only do this in batches in prod!)
    rows = user_factors_df.collect()
    pipeline = r.pipeline()
    
    for row in rows:
        uid = row['id']
        vector = row['features'] # List of floats
        pipeline.set(f"user_emb:{uid}", json.dumps(vector))
        
        if len(pipeline) >= 1000:
            pipeline.execute()
    if len(pipeline) > 0:
        pipeline.execute()

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("EAPlay-ALS") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()
        
    # Mock execution
    # model, u_factors, i_factors, test = train_als_model(spark, "s3://data/playtime.parquet")
    # push_embeddings_to_redis(u_factors)
```

---

## Part 8 — Deployment

### ML Pipeline
- **Apache Airflow:** Triggers the PySpark EMR job every night.
- **Export:** PySpark exports the User Matrix to Redis and the Game Matrix to the Inference API.

### FastAPI Serving
Since $5,000$ games is very small, we do not need a heavy Vector DB (like Milvus). 
The FastAPI service loads the 5K game embeddings into a local NumPy matrix (shape: `5000 x 64`) on startup.
When a request comes in:
1. Fetch User Vector (1x64) from Redis.
2. `np.dot(user_vector, game_matrix.T)` -> returns 5000 scores in `< 1ms`.
3. Filter owned games, sort, return top 10.

---

## Part 9 — Unit Testing

```python
import numpy as np
from typing import List

def recommend_local(user_vector: List[float], game_matrix: np.ndarray, game_ids: List[int], owned_ids: List[int]) -> List[int]:
    """Mock the FastAPI core logic."""
    u = np.array(user_vector)
    # Calculate scores (dot product)
    scores = np.dot(u, game_matrix.T)
    
    # Create dict of game_id -> score
    results = {game_ids[i]: scores[i] for i in range(len(game_ids))}
    
    # Filter owned
    for oid in owned_ids:
        if oid in results:
            del results[oid]
            
    # Sort and return top 2
    sorted_games = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return [game_id for game_id, score in sorted_games[:2]]

def test_fastapi_dot_product_logic():
    # 3 games, 2-dimensional embeddings
    game_matrix = np.array([
        [1.0, 0.0], # Game 1
        [0.0, 1.0], # Game 2
        [1.0, 1.0]  # Game 3
    ])
    game_ids = [101, 102, 103]
    
    user_vector = [1.0, 0.0] # User likes dimension 0
    owned = [101] # User already owns Game 1
    
    # Scores: G1(1.0), G2(0.0), G3(1.0)
    # Owned: G1 is filtered.
    # Top 2 remaining: G3, G2
    recs = recommend_local(user_vector, game_matrix, game_ids, owned)
    
    assert recs == [103, 102]
```

---

## Part 10 — Integration Testing

- **Offline Metric Calculation:**
  - Hold out the last week of playtime for testing.
  - Generate Top K recommendations for users based on training data.
  - Calculate **Hit Rate @ 10**: What percentage of users actually played a game in the test set that was in our Top 10 recommendations?
  - Calculate **NDCG**: Evaluates if the *true* played games were ranked higher in the list (rank 1 is better than rank 10).

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **10x Users (200M)** | PySpark ALS handles this gracefully by sharding the matrix blocks across cluster nodes. |
| **10x Games (50,000)** | The in-memory numpy dot product in FastAPI starts to become a CPU bottleneck. We must move the game embeddings out of FastAPI and into an Approximate Nearest Neighbor index (e.g., FAISS or Redis Vector Search). |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Spark ALS vs Neural Two-Tower | ALS is highly optimized, mathematically guaranteed to converge, and very cheap to run on Spark. Two-Tower (Deep Learning) captures complex non-linear relationships and easily accepts side-features (like user age, game genre), but is much harder/more expensive to train. |
| Implicit vs Explicit | Explicit ratings (5-stars) are cleaner, but only 1% of players leave ratings. Implicit (playtime) captures 100% of the audience but requires the confidence weighting ($\alpha$ parameter) because someone might play a game for 10 hours and still hate it. |

---

## Part 13 — Alternative Approaches

1. **LightGCN (Graph Convolutional Networks):** Treat the user-game interactions as a bipartite graph. LightGCN propagates embeddings across the graph edges. Highly effective for collaborative filtering and captures higher-order connectivity (User A -> Game 1 -> User B -> Game 2).
2. **Item-Item Collaborative Filtering:** Instead of User-Item, pre-compute a similarity matrix between all 5,000 games. If a user played Game A, recommend the top 5 games most similar to Game A. (Extremely fast, pure cache lookups, no user embeddings needed).

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| The Harry Potter Problem | A wildly popular game is recommended to *everyone*, drowning out niche personalization. | This is "Popularity Bias". Down-weight popular items during training or inference. For example, penalize the dot product score by $\log(total\_players)$ of the game. |
| Redis Cluster Crash | API cannot fetch user embeddings | Degrade gracefully. Have the API return a static list of the "Top 10 Global Trending Games" cached in memory, ensuring the UI doesn't break. |

---

## Part 15 — Debugging

**Symptom:** A brand new game (released yesterday) is not being recommended to anyone, even though it's the biggest launch of the year.

**Debugging steps:**
1. This is the **Item Cold-Start Problem**. Matrix Factorization (ALS) relies entirely on historical interactions. A new game has 0 playtime, so it receives a random/zero embedding vector.
2. **Fix (Short-term):** Override the ML model with a UI rule. Force the "New Release" into Slot 1 for all users.
3. **Fix (Long-term):** Migrate from pure ALS to a Hybrid Model (like LightFM or Two-Tower). Use game metadata (Genre = Shooter, Developer = DICE) to generate a "warm" embedding for the new game before anyone plays it.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `offline_ndcg_score` | Checked nightly. Drop of > 5% means model retraining failed or data is corrupted. |
| `recommendation_diversity_index` | Monitor if the model is collapsing into only recommending the top 5 most popular games. |
| `fallback_trending_trigger_rate` | > 1% → Redis is failing to return user embeddings. |

---

## Part 17 — Production Improvements

1. **Session-based Context:** If a user usually plays RPGs, but they just played 3 hours of a Racing game, the nightly batch model won't know this until tomorrow. Add an online re-ranking layer (e.g., XGBoost) that takes the Top 50 ALS recommendations and re-scores them based on the user's *current 1-hour session context*.
2. **Explore/Exploit:** Reserve 1 slot in the Top 10 recommendations for a completely random or highly exploratory game to gather unbiased telemetry data and help new games break the cold-start problem.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"A user downloads a free game, opens it for 2 minutes, decides it's garbage, and uninstalls it. Your system logs this as `playtime > 0` and assumes they like it. How do you fix this?"**
2. **"ALS gives us a 64-dimensional vector for users and games. We have 5,000 games. Walk me through the exact mathematical complexity ($O(N)$) of the API scoring step for one user."**
3. **"We want to transition to a Two-Tower Deep Learning model so we can include user demographics. How does the deployment architecture change from your current ALS setup?"**

---

## Part 19 — Ideal Answers

**Q1 (Negative Implicit Feedback):**
> "We must refine the implicit rating definition. 2 minutes of playtime is actually a *negative* signal (bounce). We can define $r_{ui}$ as `playtime_hours / average_playtime_for_this_game`. If the ratio is < 0.1, we set $p_{ui} = 0$. Alternatively, we incorporate the 'Uninstall' event as an explicit negative feedback flag in a neural network."

**Q2 (Mathematical Complexity):**
> "For one user, we have a $1 \times 64$ vector. We have a game matrix of $5000 \times 64$. The dot product requires multiplying the user vector against every game vector. That's $5000 \times 64 = 320,000$ floating-point multiplications and additions per request. This is $O(M \cdot D)$ where $M$ is items and $D$ is dimensions. In modern CPUs using vectorized numpy (BLAS/LAPACK), 320k ops takes less than 1 millisecond, which is why an ANN index is unnecessary for only 5k items."

**Q3 (Two-Tower Transition):**
> "With ALS, the user embedding is fixed and pulled from Redis. With a Two-Tower model, we have two options. 
> Option 1 (Batch): Pass all users through the User Tower offline, cache vectors in Redis. This is identical to the ALS deployment.
> Option 2 (Real-time): We don't cache user vectors. The FastAPI service takes the user's raw features (demographics, recent clicks), passes them through the User Tower (PyTorch/ONNX) dynamically to generate the embedding on-the-fly, and then computes the dot product. Option 2 captures real-time intent but costs more CPU."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Immediately recognizes the Implicit Feedback nature of playtime data and chooses the correct algorithm (Implicit ALS / Weighted Matrix Factorization).
- Optimizes the API design by doing in-memory dot products (avoiding over-engineering with FAISS for only 5,000 items).
- Easily addresses the Item Cold-Start problem with metadata fallbacks.
- Calculates the Big-O complexity accurately.

### Hire
- Successfully sets up a Spark ML pipeline.
- Uses Redis for fast user state lookup.
- Understands how to calculate Hit Rate and NDCG.
- Handles the API filtering (removing owned games) natively.

### Lean Hire
- Suggests standard SVD (Singular Value Decomposition) which fails on implicit/sparse data. Interviewer must correct them.
- Defaults to a massively complex Deep Learning model when a standard matrix factorization would suffice for a baseline.

### Lean No Hire
- Proposes calculating recommendations in real-time by joining the 20-million row database during the API request (SQL `JOIN` query).
- Does not understand Matrix Factorization or Embeddings.

### No Hire
- Cannot write PySpark or Pandas code.
- Has no understanding of Recommender System architectures.
