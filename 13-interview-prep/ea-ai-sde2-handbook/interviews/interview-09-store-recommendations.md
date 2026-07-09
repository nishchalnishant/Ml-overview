# Interview 09 — In-Game Store Item Recommendation Engine
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the EA Central Tech team. You need to build a recommendation engine for the in-game cosmetic store (applicable to games like Apex Legends, The Sims, or Need for Speed). 

When a player opens the store, there is a "For You" carousel containing 6 items. There are thousands of items in the catalogue.

Your task is to **design and implement a system that selects the top 6 items to display for a specific player to maximize purchase conversion.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- **Cold Start Problem:** How do we handle brand new players with zero purchase history? How do we handle brand new items added today?
- **Latency:** Does this run in real-time on page load, or batch pre-computed?
- **Inventory rules:** Can we recommend items they already own?
- **Feedback Loop:** Do we track impressions (items they saw but didn't buy)?
- **Implicit vs Explicit feedback:** Do we only look at purchases, or also items they clicked/previewed?

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Are we allowed to recommend items the player already owns?"**
   → *Answer: No, cosmetics are unique. The system must filter out owned inventory.*

2. **"Does this need to run in real-time, or can we pre-compute recommendations nightly?"**
   → *Answer: Batch pre-computation is fine for the heavy ML, but we need real-time filtering (if they buy an item, it must disappear from the carousel instantly).*

3. **"Do we track negative signals, like items they saw but ignored?"**
   → *Answer: Yes, we have impression logs. We know exactly what they saw and didn't click.*

4. **"How do we handle the cold start problem for new items?"**
   → *Answer: We release new skins weekly. They have no historical purchase data. You need to tell me how you'd solve this.*

5. **"What is the scale?"**
   → *Answer: 50 million active players, 5,000 items in the catalogue.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Two-Tower / Dual-Encoder Neural Network (Retrieval) followed by a Ranking model, OR Matrix Factorization (ALS). Given 5,000 items, we could technically skip Retrieval and just Rank all 5,000, but a Two-Stage pipeline is standard industry practice.
- **Data:** User features (playtime, favorite character/weapon), Item features (color, rarity, character, price).
- **Serving:** Batch inference pushes Top 100 items per user to Redis. Game API fetches the 100, filters out owned items in real-time, and serves the Top 6.

---

## Part 5 — High-Level Solution

```
  [Nightly Batch Pipeline]
  1. Train / Update Model
  2. Batch Inference: Score 50M users × 5,000 items
  3. Push Top 100 Item IDs per user to Redis
       │
       ▼
  [Real-Time Serving API (FastAPI / Go)]
  ┌────────────────────────────────────────────────────────┐
  │ 1. Fetch Top 100 candidate items from Redis            │
  │ 2. Fetch User's currently owned items (from DB/Cache)  │
  │ 3. Filter out owned items                              │
  │ 4. Apply business rules (e.g., must include 1 sale item)│
  │ 5. Return Top 6 items                                  │
  └────────────────────────────────────────────────────────┘
```

**Core ML Component:** 
To handle the Cold Start problem for new items, pure Collaborative Filtering (ALS) fails. We must use a **Content-Based Model** or a **Two-Tower Neural Network** that uses item metadata (color, class, price) so it can recommend a new item based on its features.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Model Choice (Two-Tower Architecture)
- **User Tower:** Inputs user features (historical purchases, playtime per character). Outputs a dense vector (embedding) representing the user.
- **Item Tower:** Inputs item features (tags, rarity, price, hero). Outputs a dense vector representing the item.
- **Loss:** Contrastive loss. Push user and item embeddings closer if purchased, further if impressed-but-not-purchased.

### Step 2: Solving Cold Start
- When a new item drops, pass its metadata through the Item Tower to get its embedding. 
- It immediately lives in the vector space alongside similar older items, and will be recommended to users who like those features.

### Step 3: Batch Generation
- Calculate the dot product between every User embedding and every Item embedding.
- Fast matrix multiplication (FAISS) can do this in minutes for 50M x 5K.

### Step 4: Real-time API
- Look up pre-calculated list in Redis.
- Perform set difference with inventory.

---

## Part 7 — Complete Python Code

*Note: For the interview, writing a full Two-Tower PyTorch model is too long. We will mock the ML training and focus on the architecture and the Real-Time Serving API which handles the critical filtering.*

```python
"""
recsys_serving.py - API for Serving and Filtering Recommendations
"""
import logging
from typing import List
from fastapi import FastAPI, HTTPException
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Store Recommendation API")

# Connect to Redis
# db=0: precomputed recommendations, db=1: user inventory
redis_recs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
redis_inventory = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/store/recommendations/{player_id}")
async def get_recommendations(player_id: str, limit: int = 6):
    """
    Fetches personalized store items, filtering out owned inventory.
    """
    # 1. Fetch pre-computed Top N candidates (e.g., N=100)
    # Format stored in Redis: comma-separated string "item1,item2,item3..."
    candidates_raw = redis_recs.get(f"recs:{player_id}")
    
    if not candidates_raw:
        # Fallback to global popular items if cold-start user
        candidates_raw = redis_recs.get("recs:global_popular")
        if not candidates_raw:
            raise HTTPException(status_code=500, detail="Recommendation cache missing.")
            
    candidate_items = candidates_raw.split(',')
    
    # 2. Fetch User Inventory (O(1) set lookup if stored as Redis Set)
    # smembers returns a set of item_ids
    owned_items = redis_inventory.smembers(f"inv:{player_id}")
    
    # 3. Filter candidates
    final_recommendations = []
    for item in candidate_items:
        if item not in owned_items:
            final_recommendations.append(item)
            
        if len(final_recommendations) == limit:
            break
            
    # 4. Handle edge case: Whales who own almost everything
    if len(final_recommendations) < limit:
        logger.warning(f"Player {player_id} exhausted candidates. Filling with consumables.")
        consumables = ["xp_boost", "currency_pack_small", "currency_pack_large"]
        for item in consumables:
            if item not in final_recommendations:
                final_recommendations.append(item)
            if len(final_recommendations) == limit:
                break

    return {
        "player_id": player_id,
        "recommendations": final_recommendations
    }

# ---------------------------------------------------------------------------
# Mock Batch Job (Runs nightly)
# ---------------------------------------------------------------------------
def mock_batch_inference():
    """
    Simulates the nightly PyTorch / Spark job that calculates dot products
    and pushes to Redis.
    """
    # ... Matrix Multiplication: User_Embeddings @ Item_Embeddings.T ...
    
    dummy_top_100 = [f"skin_{i}" for i in range(100)]
    
    pipeline = redis_recs.pipeline()
    pipeline.set("recs:p123", ",".join(dummy_top_100))
    pipeline.set("recs:global_popular", ",".join(["skin_50", "skin_10", "skin_1"]))
    pipeline.execute()
    logger.info("Batch inference pushed to Redis.")

if __name__ == "__main__":
    mock_batch_inference()
```

---

## Part 8 — Deployment

### ML Training (Airflow + GPU instances)
- Run daily. Use TensorFlow Recommenders or PyTorch.
- For 50M users, dot product requires Spark or a highly optimized NumPy/FAISS script on a large memory node.
- Write output to S3, then a separate loader script pushes S3 -> Redis to avoid overloading the production cache during computation.

### Serving (Kubernetes)
- The FastAPI service scales horizontally. CPU only.
- Very high QPS (called every time a player enters the store lobby).
- Redis needs to be highly available (Redis Cluster or ElastiCache) as it holds state for 50M users.

---

## Part 9 — Unit Testing

```python
import pytest
import fakeredis
from fastapi.testclient import TestClient
from recsys_serving import app, redis_recs, redis_inventory

# Replace real redis with fakeredis for tests
app.dependency_overrides[redis_recs] = fakeredis.FakeStrictRedis(decode_responses=True)
app.dependency_overrides[redis_inventory] = fakeredis.FakeStrictRedis(decode_responses=True)
client = TestClient(app)

def test_filtering_owned_items():
    r_recs = fakeredis.FakeStrictRedis(decode_responses=True)
    r_inv = fakeredis.FakeStrictRedis(decode_responses=True)
    
    # Mock data injection (assuming global variables patched in actual test suite)
    r_recs.set("recs:p1", "item_a,item_b,item_c,item_d,item_e,item_f,item_g")
    
    # User owns item_a and item_c
    r_inv.sadd("inv:p1", "item_a", "item_c")
    
    # We request 4 items. It should skip A and C, returning B, D, E, F
    # ... assert logic here ...

def test_fallback_for_whales():
    # Test that if user owns ALL candidates, it appends consumables
    pass
```

---

## Part 10 — Integration Testing

- Shadow testing is critical for RecSys.
- Deploy the new model alongside the old model.
- The Game Client requests recommendations. The API queries *both* models, serves the Old Model to the player, but logs the New Model's output to the data warehouse.
- Offline, compare metrics: Did the New Model recommend items the user eventually bought organically?

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Catalogue Growth** | If the catalogue grows from 5K to 500K items (e.g., user-generated content), we cannot do a brute force dot product for all users. We must introduce an **Approximate Nearest Neighbor (ANN)** index like FAISS or HNSW. |
| **Real-time updates** | A user buys an item, we filter it locally. But what if their taste changes mid-session? Introduce an in-memory streaming update (e.g., Kafka -> Flink -> update User Embedding) to reflect real-time session context. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Batch Inference vs Real-Time Inference | Batch is cheap, stable, and has 1ms latency (Redis lookup), but recommendations are up to 24h stale. Real-time inference (passing user features through the Neural Net on request) captures immediate intent but costs massively in GPU inference and latency. |
| Matrix Factorization vs Two-Tower | MF is simpler and requires less data engineering, but suffers completely on the item cold-start problem. Two-Tower handles cold starts gracefully using metadata. |
| Implicit vs Explicit feedback | Explicit (Star ratings) is high quality but rare. Implicit (Purchases, Clicks, View time) is noisy but abundant. We use implicit (purchases = positive, viewed but ignored = negative). |

---

## Part 13 — Alternative Approaches

1. **Sequential/Session-Based Recommendations:** Use a Transformer (like BERT4Rec or SASRec) on the sequence of the user's recent clicks. Predict the next click. Excellent for real-time responsiveness.
2. **Bandits for Exploration:** Use multi-armed bandits to force the UI to explore showing new/unpopular items to users, preventing the "rich get richer" popularity bias inherent in Collaborative Filtering.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Training Pipeline OOM | Model doesn't update | Stale recommendations (yesterday's cache) remain in Redis. Set TTL on Redis keys to 48 hours to survive a 1-day pipeline failure. |
| Inventory Service down | API cannot filter owned items | We accidentally sell players items they already own. This is a severe bug causing support tickets. If Inventory DB times out, fail gracefully and serve *consumables only* (XP boosts) which can be bought infinitely. |
| Popularity Bias | Model only recommends the Top 10 best-selling items to everyone | Introduce a penalty term in the loss function or post-processing step to artificially boost long-tail items. |

---

## Part 15 — Debugging

**Symptom:** A new highly-anticipated cosmetic skin is released, but it is not appearing in *anybody's* recommendation carousel.

**Debugging steps:**
1. Check the item metadata. Did the designers forget to add tags (color, rarity, hero class) to the database? If features are missing, the Item Tower outputs a zero/garbage vector.
2. Check the candidate list size. If we only cache the Top 100 items per user, and the new item was ranked #105 because the model was conservative, it will never be seen.
3. Fix: Hardcode a business rule in the serving API: `Top 6 = [1 New Release Slot, 5 Personalized Slots]`. Never rely 100% on ML for new product launches.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `recommendation_ctr` (Click-Through Rate) | Monitor daily. Drop > 10% indicates model drift or bad catalog metadata. |
| `cache_hit_rate_redis` | < 95% → Warning (Batch job might be failing to score all active users). |
| `inventory_filter_latency_ms` | > 20ms → Critical (Inventory DB is slowing down the store). |

---

## Part 17 — Production Improvements

1. **Re-ranking Stage (Multi-Task Learning):** The Two-Tower model is the "Retrieval" stage. Add a heavier "Ranking" stage (e.g., XGBoost or DLRM) that takes the Top 100 items and re-ranks them in real-time based on immediate context (time of day, current device).
2. **Diversity Penalty:** If the Top 6 items are all Red Sniper Rifle skins, the user experience is bad. Implement Maximal Marginal Relevance (MMR) in the serving API to ensure visual/category diversity in the final 6 slots.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The Two-Tower model solves the item cold-start problem via metadata. But what about a *User* cold-start? A brand new player logs in for the first time. How do you handle them?"**
2. **"Our game has 'Bundles' (e.g., a Weapon Skin + a Character Skin). If a user owns the Character Skin, they can still buy the Bundle at a discount. How does your inventory filtering logic handle bundles?"**
3. **"We notice that if an item is placed in Slot 1 (far left) of the UI, it gets 5x more clicks than Slot 6 (far right), regardless of the item. How does this UI position bias affect your model training, and how do you fix it?"**

---

## Part 19 — Ideal Answers

**Q1 (User Cold Start):**
> "For a brand new user, the User Tower has no historical purchase data. We rely on onboarding features (e.g., platform, geographic region, time of day) or fallback to global popularity. Alternatively, we can prompt the user to select 3 favorite characters on their first login, and use those as the initial input features to generate a basic embedding."

**Q2 (Bundles):**
> "The simple set-difference filtering (`if item not in owned`) breaks for bundles. The Serving API needs business logic: it must break down the bundle into its component IDs. If the user owns *all* components, filter the bundle. If they own *some*, keep the bundle but flag it for dynamic pricing. The ML model just ranks the bundle ID; the API handles the complex graph of entitlements."

**Q3 (Position Bias):**
> "This is classical presentation bias. The model learns that Slot 1 items are 'better' just because they got clicked more. To fix this, we include the UI position as an input feature *during training*. The model learns the intrinsic value of the item + the value of the position. *During inference*, we set the position feature to a constant (e.g., Slot 1) for all items. This isolates the true relevance of the item."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Clearly distinguishes between Retrieval (Two-Tower) and Ranking.
- Solves the item cold-start problem using content/metadata features.
- Designs a robust real-time filtering API to handle inventory checks.
- Mentions diversity (MMR) and position bias without heavy prompting.

### Hire
- Understands Matrix Factorization and why it fails cold-start, pivoting to Neural approaches.
- Uses Redis effectively for the batch-to-realtime handoff.
- Can write the inventory filtering logic.
- Answers the fallback/whale question effectively.

### Lean Hire
- Suggests a pure real-time inference architecture for 50M users, but accepts batch processing when the cost is pointed out.
- Forgets to filter out already-owned items initially.
- Has a basic understanding of Recommender Systems but lacks production nuances (like diversity).

### Lean No Hire
- Suggests using a LLM (GPT-4) to recommend store items.
- Cannot explain how to convert user features into an embedding.
- Writes API code that does a SQL JOIN on 50M rows per user request.

### No Hire
- Fails to understand what a recommendation engine is (treats it as a simple classification problem).
- Cannot structure the architecture.
