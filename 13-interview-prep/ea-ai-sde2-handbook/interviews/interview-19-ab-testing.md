# Interview 19 — A/B Testing & Experimentation Platform
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Data Platform team. The matchmaking team has developed a new ML algorithm (Algorithm B) that supposedly reduces player wait times compared to the current system (Algorithm A). 

Your task is to **design a highly scalable A/B Testing & Experimentation Platform** that can route players to different algorithms, track their telemetry, and statistically determine if Algorithm B is actually better without introducing bias.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Network Effects / Interference (If Player A is in group A, and Player B is in group B, how can they play a match together?)
- Assignment logic (Deterministic hashing vs Random assignment).
- Statistical metric (What are we actually measuring? Wait time? Churn? Match fairness?)
- Sample size (How long do we run the test?)
- Novelty Effect (Do players just act differently because it's new?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"How do we handle multiplayer interference? If we bucket by user, they might end up in different matchmaking pools."**
   → *Answer: Excellent catch. For matchmaking, bucketing by user breaks the game. How would you solve this? (Candidate should suggest bucketing by Region/Time or Session).*

2. **"What is the Primary Evaluation Metric (OEC)?"**
   → *Answer: Primary is 'Average Wait Time'. Guardrail metric is 'Match Imbalance Score' (we don't want fast matches that are completely unfair).*

3. **"Is the assignment deterministic across sessions?"**
   → *Answer: If bucketed by user, yes. If they log out and log in, they must remain in the same bucket.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Assignment Service (FastAPI) ➔ Telemetry Ingestion (Kafka) ➔ Batch Analysis (Spark/Snowflake).
- **Bucketing Algorithm:** `hash(user_id + experiment_id) % 100` for deterministic routing without state lookup.
- **Statistics:** A/B testing isn't just routing; it's statistical inference (T-Test, Bootstrap, P-values).

---

## Part 5 — High-Level Solution

```
  [Game Client]
       │ 1. Request Matchmaking
       ▼
  [Experiment Assignment API]
  ┌────────────────────────────────────────────────────────┐
  │ Hash(player_id + "matchmaking_v2") % 100 = 42          │
  │ Config: Bucket A (0-49), Bucket B (50-99).             │
  │ Result: Route to Bucket A (Control).                   │
  └────────────────────────────────────────────────────────┘
       │
       ▼ 2. Route to Algorithm A or B
  [Matchmaking Service]
       │ 3. Match found. Emit Telemetry (wait_time, bucket)
       ▼
  [Kafka ➔ Snowflake]
       │
       ▼ 4. Nightly Analysis Job
  [Statistical Engine (Python / SciPy)]
  Calculates T-Test, P-Value, and Confidence Intervals.
```

**Core ML Component:** Designing a statistically rigorous platform that avoids Network Effects (SUTVA violations) and computes accurate statistical significance for massive datasets.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Deterministic Hashing
- Storing "User 1 = Bucket A" in a database for 50 million users is too expensive to look up in real-time.
- Use MD5 or MurmurHash: `murmur3(user_id + experiment_salt) % 1000`. This allows any microservice to independently determine the user's bucket in $<0.1ms$ with zero network calls.

### Step 2: Handling Multiplayer Interference
- **Problem:** If matchmaking is bucketed by user, Bucket A users can only play Bucket A users. This halves the player pool, artificially increasing wait times for both buckets (SUTVA violation).
- **Solution (Switchback Testing):** Instead of bucketing by User, bucket by **Time/Region**. 
  - Mon: US-East runs Alg A.
  - Tue: US-East runs Alg B.
  - Wed: US-East runs Alg A.
  This ensures the entire player pool is intact.

### Step 3: Statistical Evaluation
- Calculate the mean and variance of `wait_time` for Control and Treatment.
- Use a 2-sample T-Test (or Welch's T-Test for unequal variances) to calculate the p-value.
- If $p < 0.05$, reject the null hypothesis (Alg B is statistically different).

---

## Part 7 — Complete Python Code

```python
"""
ab_test_platform.py - Deterministic Routing & Statistical Analysis
"""
import logging
import hashlib
import numpy as np
from scipy import stats
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Assignment Service (High QPS, Zero State)
# ---------------------------------------------------------------------------
class ExperimentAssigner:
    def __init__(self, experiment_id: str, rollout_percentage: int = 50):
        self.experiment_id = experiment_id
        self.rollout_percentage = rollout_percentage # e.g., 50 means 50% Control, 50% Treatment

    def get_bucket(self, entity_id: str) -> str:
        """
        Deterministically assigns an entity to a bucket using hashing.
        entity_id could be user_id, or region_date (for switchback testing).
        """
        # Combine ID and salt to ensure a user isn't in Treatment for EVERY experiment
        hash_str = f"{entity_id}_{self.experiment_id}"
        
        # Get a number from 0 to 99
        hash_val = int(hashlib.md5(hash_str.encode('utf-8')).hexdigest(), 16) % 100
        
        if hash_val < self.rollout_percentage:
            return "treatment"
        return "control"

# ---------------------------------------------------------------------------
# 2. Statistical Engine (Nightly Batch)
# ---------------------------------------------------------------------------
def analyze_experiment(telemetry_df: pd.DataFrame, metric_col: str):
    """
    Evaluates the results of the A/B test.
    telemetry_df contains: [bucket, wait_time_seconds]
    """
    logger.info("Analyzing experiment results...")
    
    control_data = telemetry_df[telemetry_df['bucket'] == 'control'][metric_col]
    treatment_data = telemetry_df[telemetry_df['bucket'] == 'treatment'][metric_col]
    
    mean_c = np.mean(control_data)
    mean_t = np.mean(treatment_data)
    diff = mean_t - mean_c
    lift_pct = (diff / mean_c) * 100
    
    # Perform Welch's T-Test (does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
    
    alpha = 0.05
    is_significant = p_value < alpha
    
    result = {
        "metric": metric_col,
        "control_mean": mean_c,
        "treatment_mean": mean_t,
        "absolute_diff": diff,
        "relative_lift_pct": lift_pct,
        "p_value": p_value,
        "statistically_significant": is_significant
    }
    
    logger.info(f"Result: {result}")
    return result

# ---------------------------------------------------------------------------
# Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Routing
    assigner = ExperimentAssigner("matchmaking_v2", rollout_percentage=50)
    print(f"User 101 -> {assigner.get_bucket('user_101')}")
    print(f"User 102 -> {assigner.get_bucket('user_102')}")
    print(f"User 101 (check again) -> {assigner.get_bucket('user_101')}") # Must match
    
    # 2. Analysis
    # Mock data: Treatment (Alg B) is faster (lower wait time)
    np.random.seed(42)
    mock_df = pd.DataFrame({
        'bucket': ['control']*1000 + ['treatment']*1000,
        'wait_time_seconds': np.concatenate([
            np.random.normal(30.0, 5.0, 1000), # Control mean 30s
            np.random.normal(28.0, 5.0, 1000)  # Treatment mean 28s
        ])
    })
    
    analyze_experiment(mock_df, "wait_time_seconds")
```

---

## Part 8 — Deployment

### Assignment Caching
- Even though hashing is fast, we deploy the assignment logic as a library (SDK) embedded directly in the Game Server or Matchmaking microservice.
- This avoids making an HTTP call to an `ab-test-service` for every single action, reducing latency to zero.

### Experiment Config Delivery
- Experiments are defined in a UI. The config JSON (e.g., `{"experiment": "mm_v2", "rollout": 50, "status": "active"}`) is synced to a Redis cache or AWS AppConfig. The Game Server SDK polls this config every 5 minutes.

---

## Part 9 — Unit Testing

```python
from ab_test_platform import ExperimentAssigner

def test_deterministic_hashing():
    assigner = ExperimentAssigner("test_exp", 50)
    
    # Must be deterministic
    bucket_1 = assigner.get_bucket("user_alpha")
    bucket_2 = assigner.get_bucket("user_alpha")
    assert bucket_1 == bucket_2
    
    # Distribution test (Law of Large Numbers)
    treatment_count = 0
    for i in range(10000):
        if assigner.get_bucket(f"user_{i}") == "treatment":
            treatment_count += 1
            
    # Should be roughly 50%
    assert 4800 < treatment_count < 5200
```

---

## Part 10 — Integration Testing

- **A/A Test:**
  - Before running Alg A vs Alg B, run Alg A vs Alg A.
  - Route users to Bucket 1 and Bucket 2.
  - Run the statistical analysis.
  - Assert that $p > 0.05$ (Not significant). If it shows significance, your hashing algorithm is biased or your telemetry pipeline has a bug.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Data Volume** | Scipy `ttest_ind` requires loading arrays into Python memory. For 50 million matches, this OOMs. We must push the math down to the database using Spark SQL or Snowflake's native statistical functions (e.g., aggregating means, counts, and variances at the DB level, and only doing the final p-value calc in Python). |
| **Many Metrics** | Running a T-test on 50 different metrics (wait_time, latency, kills, chat_messages) increases the chance of a False Positive (Multiple Comparisons Problem). Must apply the **Bonferroni Correction** (divide the alpha threshold by the number of metrics). |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| User Bucketing vs Switchback Bucketing | User bucketing allows for tracking long-term retention per player. But it ruins matchmaking liquidity. Switchback (Time/Region) solves liquidity but makes it impossible to measure individual player retention over a month, because everyone experiences both algorithms. |
| Frequentist (T-Test) vs Bayesian | Frequentist is industry standard and easy to compute via SQL. Bayesian provides a much more intuitive output for product managers ("There is a 95% chance Alg B is better"), but requires complex MCMC sampling or conjugate priors. |

---

## Part 13 — Alternative Approaches

1. **Multi-Armed Bandit (MAB):** Instead of a fixed 50/50 split for 2 weeks, use a Bandit algorithm (e.g., Thompson Sampling). It constantly updates the probabilities. If Alg B is clearly much better on Day 2, it automatically routes 90% of traffic to Alg B, minimizing the "regret" of forcing players to suffer through Alg A for two weeks.
2. **Cluster Randomization:** Bucket by "Guild" or "Party_ID". If you bucket by User, 4 friends playing together might get split into different experiments, breaking the party system.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Simpson's Paradox | Alg B looks better overall, but when you split by Region, Alg A is better in *every single region*. | This happens when the ratio of users in buckets is uneven across regions. Always perform stratified sampling or check results segmented by major dimensions (Region, Platform). |
| Peeking Problem | PM checks the dashboard every hour. On hour 4, p < 0.05, so they stop the test and declare victory. | This inflates false positives massively. Force a fixed sample size. Hide the p-value on the dashboard until the 7-day minimum duration is reached. |

---

## Part 15 — Debugging

**Symptom:** Algorithm B shows a massive 40% reduction in wait time. You deploy it to 100% of users, but overall server wait times don't change at all.

**Debugging steps:**
1. This is a classic SUTVA violation (Interference). 
2. During the 50/50 test, Algorithm B (which prioritized speed over fairness) quickly scooped up all the players waiting in the pool, leaving none for Algorithm A. Algorithm A's wait time artificially skyrocketed because its pool was starved.
3. The 40% lift was an illusion caused by the algorithms cannibalizing each other in the same matchmaking pool.
4. **Fix:** Roll back. Re-run the experiment using Switchback (Time-slice) testing.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `Sample Ratio Mismatch (SRM)` | If bucket allocation is supposed to be 50/50, but actual traffic is 51/49, alert immediately. The hash is biased, or a bug is causing Treatment users to drop events. |
| `Guardrail Metrics` | e.g., Crash Rate. If Alg B reduces wait time but increases game crashes by 2%, auto-kill the experiment. |

---

## Part 17 — Production Improvements

1. **Variance Reduction (CUPED):** A/B tests can take weeks to reach significance if the metric is noisy (e.g., total spend). Use CUPED (Controlled-experiment Using Pre-Experiment Data). Adjust the test metric using historical data (e.g., `adjusted_spend = spend - theta * pre_experiment_spend`). This drastically reduces variance, allowing you to reach statistical significance in days instead of weeks.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"A product manager wants to run 10 different matchmaking experiments at the same time. How do you design the hashing architecture so experiments don't collide and bias each other?"**
2. **"During an experiment, a massive Twitch streamer starts playing, drawing 100,000 new users to the game. They all get hashed into Bucket B by chance. How do you detect and fix this outlier skew?"**
3. **"We want to measure 'Player Churn' (if they stop playing for 30 days). A T-Test doesn't work well for binary survival metrics. What statistical method should you use instead?"**

---

## Part 19 — Ideal Answers

**Q1 (Overlapping Experiments):**
> "We implement Orthogonal Hashing layers. We use the `experiment_id` as a salt in the hash function: `hash(user + exp_id)`. Because the salt is different for each experiment, User 1 might be in Treatment for Exp 1, but Control for Exp 2. Across 50 million users, the overlapping groups are perfectly distributed (orthogonal), preventing bias."

**Q2 (Outlier Handling):**
> "First, we detect it using the Sample Ratio Mismatch (SRM) alert. Second, we apply Outlier Capping (Winsorization) to the telemetry data before analysis, capping extreme values at the 99th percentile. Finally, we should use a non-parametric test (like the Mann-Whitney U test) which uses ranks instead of means, making it immune to massive outliers."

**Q3 (Survival Metrics / Churn):**
> "Player Churn is a time-to-event metric. We should use Survival Analysis, specifically the **Kaplan-Meier Estimator** to plot the survival curves of Bucket A vs Bucket B, and the **Log-Rank Test** to determine if the difference in churn rate over the 30 days is statistically significant."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Immediately identifies the SUTVA/Multiplayer Interference problem for matchmaking.
- Proposes Switchback or Cluster testing to solve interference.
- Articulates the Bonferroni Correction for multiple metrics.
- Understands CUPED or Variance Reduction techniques.

### Hire
- Writes a clean deterministic hashing function.
- Understands how to calculate a T-Test and P-value.
- Sets up the A/B testing pipeline correctly.

### Lean Hire
- Relies on "gut feeling" or percentage differences rather than statistical significance (Needs prompting to mention P-values).
- Doesn't know what to do when a data warehouse is too big for a Scipy function.

### Lean No Hire
- Suggests maintaining a massive Redis table storing `[user_id: bucket_a]` for 50 million users, ignoring the network/storage cost when deterministic hashing is standard.

### No Hire
- Has no understanding of A/B testing principles.
- Cannot explain how to evaluate if an ML model is better than a baseline in production.
