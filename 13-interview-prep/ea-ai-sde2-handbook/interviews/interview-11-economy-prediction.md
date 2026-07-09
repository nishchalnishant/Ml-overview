# Interview 11 — In-Game Economy Inflation Prediction
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on the MMORPG economy team (e.g., Star Wars: The Old Republic). The game has a complex virtual economy with millions of transactions daily (players trading items, earning gold from quests, spending gold on repairs).

Recently, the price of basic materials on the Auction House has spiked by 400%, ruining the experience for new players. The game designers suspect a bug is injecting free gold into the economy.

Your task is to **design an ML system that monitors the macro-economy and detects inflationary anomalies or exploits in real-time.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Data sources (Do we track every single trade, or just global aggregates?)
- Definitions of Faucets (gold creation) and Sinks (gold destruction).
- Granularity (Detecting a server-wide trend vs identifying the specific player exploiting the bug).
- Seasonality (Prices always spike on weekends or patch days).
- Latency (Do we need to freeze the auction house in real-time?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Are we trying to detect macro-inflation (server-wide) or micro-exploits (a single player duplicating items)?"**
   → *Answer: Both, but start with macro-inflation. We need to know if the total money supply is growing faster than expected.*

2. **"Do we have telemetry for 'Faucets' (gold generated) and 'Sinks' (gold destroyed)?"**
   → *Answer: Yes. Every time a monster drops gold, or a player buys an NPC item, we log a telemetry event.*

3. **"Is there strong seasonality in the data?"**
   → *Answer: Yes. Massive spikes in gold generation on weekends, and massive sinks when new content drops.*

4. **"Does this need to run in real-time?"**
   → *Answer: Hourly batch processing is fine for macro-trends. 5-minute windows for exploit detection.*

---

## Part 4 — Expected Assumptions

- **Time-Series Problem:** This is fundamentally a time-series anomaly detection problem.
- **Metrics:** Money Supply (M0), Velocity of Money, Auction House Price Index (CPI equivalent).
- **Algorithm:** Prophet, ARIMA, or an Autoencoder for multivariate time-series anomaly detection.

---

## Part 5 — High-Level Solution

```
  Game Servers (Telemetry)
       │ (JSON events: {type: 'loot', amount: 100})
       ▼
  Kafka ➔ Apache Flink (Aggregation)
  ┌────────────────────────────────────────────────────────┐
  │ Group by 1-hour tumbling windows                       │
  │ Output: total_gold_looted, total_gold_sunk, ah_volume  │
  └────────────────────────────────────────────────────────┘
       │
       ▼ (Time-Series DB: InfluxDB / Prometheus)
  Anomaly Detection Service (Cron Job / Python)
  ┌────────────────────────────────────────────────────────┐
  │ 1. Load 30 days of historical hourly aggregates        │
  │ 2. Predict expected bounds for the current hour        │
  │ 3. If actual > expected_upper_bound ➔ Trigger Alert    │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  Slack / PagerDuty ➔ Economy Designers
```

**Core ML Component:** A time-series forecasting model (e.g., Prophet) that explicitly models daily/weekly seasonality to generate dynamic thresholds, rather than using static thresholds (which fail on weekends).

---

## Part 6 — Step-by-Step Implementation

### Step 1: Feature Aggregation
- **M0 (Total Money Supply):** Previous M0 + Faucets - Sinks.
- **Faucet Volume:** Sum of gold generated from loot, quests, vendor sales.
- **Sink Volume:** Sum of gold destroyed via repairs, fast travel, auction house tax.
- **AH Price Index:** Volume-weighted average price of top 10 traded commodities (e.g., Iron Ore, Cloth).

### Step 2: Modeling Seasonality (Prophet)
- We use Facebook Prophet (or similar) because it handles strong daily/weekly seasonality out-of-the-box and is robust to missing data.
- Train the model on the last 30 days. Predict the next hour.
- Extract `yhat_upper` and `yhat_lower` (confidence intervals).

### Step 3: Anomaly Scoring
- If the actual observed gold generation exceeds `yhat_upper` by $X\%$, flag it.
- Multivariate check: If Faucets spike AND Auction House Prices spike, the probability of an exploit is extremely high.

---

## Part 7 — Complete Python Code

```python
"""
economy_monitor.py - Time-Series Anomaly Detection for Virtual Economy
"""
import logging
import pandas as pd
from prophet import Prophet
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SLACK_WEBHOOK = "https://hooks.slack.com/services/T0000/B0000/XXXX"
CONFIDENCE_INTERVAL = 0.99  # 99% prediction interval

def fetch_economy_data() -> pd.DataFrame:
    """
    Mock fetching hourly aggregated data from Snowflake/InfluxDB.
    Returns DF with columns: [ds, gold_generated, gold_destroyed, ah_index]
    """
    # Generating dummy seasonal data
    dates = pd.date_range(start="2023-01-01", end="2023-01-30", freq="H")
    df = pd.DataFrame({"ds": dates})
    
    # Baseline + daily seasonality + weekend spike
    df["gold_generated"] = 1000000 \
        + 500000 * pd.Series([i % 24 for i in range(len(df))]).apply(lambda x: 1 if 18 <= x <= 23 else 0) \
        + 1000000 * df["ds"].dt.dayofweek.isin([5, 6]).astype(int)
        
    return df

def train_and_predict(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Trains Prophet to establish seasonal baselines."""
    logger.info(f"Training forecasting model for {metric}...")
    
    # Prophet requires 'ds' (datetime) and 'y' (target)
    train_df = df[['ds', metric]].rename(columns={metric: 'y'})
    
    model = Prophet(
        interval_width=CONFIDENCE_INTERVAL,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    # Add holiday effect for game patch days
    # model.add_country_holidays(country_name='US') 
    
    model.fit(train_df)
    
    # Predict the same timeframe to get the confidence intervals (yhat_upper)
    forecast = model.predict(train_df)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def detect_anomalies(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, metric: str):
    """Compares actuals against predicted bounds."""
    logger.info("Evaluating anomalies...")
    
    merged = pd.merge(actual_df, forecast_df, on='ds')
    
    anomalies = []
    for _, row in merged.iterrows():
        actual = row[metric]
        upper = row['yhat_upper']
        
        if actual > upper * 1.10: # 10% tolerance above the 99% interval
            anomalies.append({
                "timestamp": str(row['ds']),
                "metric": metric,
                "actual": actual,
                "expected_max": upper,
                "severity": (actual - upper) / upper
            })
            
    return anomalies

def alert_designers(anomalies: list):
    """Sends Slack alerts."""
    if not anomalies:
        logger.info("Economy is stable. No anomalies.")
        return
        
    # Get the most recent anomaly
    latest = anomalies[-1]
    logger.warning(f"ANOMALY DETECTED: {latest}")
    
    msg = f"🚨 **Economy Alert** 🚨\nMetric: {latest['metric']} spiked!\nActual: {latest['actual']:,.0f}\nExpected Max: {latest['expected_max']:,.0f}"
    # requests.post(SLACK_WEBHOOK, json={"text": msg})
    print(msg)

def run_hourly_check():
    df = fetch_economy_data()
    
    # Introduce a fake exploit anomaly at the last hour
    df.loc[df.index[-1], 'gold_generated'] += 5_000_000 
    
    forecast = train_and_predict(df, "gold_generated")
    anomalies = detect_anomalies(df, forecast, "gold_generated")
    alert_designers(anomalies)

if __name__ == "__main__":
    run_hourly_check()
```

---

## Part 8 — Deployment

### Apache Flink + InfluxDB
- Telemetry events stream via Kafka. Flink runs an SQL query: `SELECT SUM(amount) FROM events WHERE type='gold_faucet' GROUP BY TUMBLE(time, INTERVAL '1' HOUR)`.
- Results are pushed to InfluxDB (optimized for time-series).

### Airflow / Cron Job
- The Python script runs at `MM:05` every hour.
- It pulls data for the last 30 days from InfluxDB, trains Prophet in memory (takes ~2 seconds), predicts the last hour, and alerts if anomalous.

---

## Part 9 — Unit Testing

```python
import pandas as pd
from economy_monitor import detect_anomalies

def test_anomaly_detection_logic():
    # Setup actuals
    actuals = pd.DataFrame({
        'ds': pd.date_range("2023-01-01", periods=3, freq="H"),
        'gold_generated': [100, 100, 500] # Spike at idx 2
    })
    
    # Setup forecast (yhat_upper is 150)
    forecast = pd.DataFrame({
        'ds': actuals['ds'],
        'yhat': [100, 100, 100],
        'yhat_lower': [50, 50, 50],
        'yhat_upper': [150, 150, 150]
    })
    
    anomalies = detect_anomalies(actuals, forecast, "gold_generated")
    
    # Only the 3rd row (500 > 150 * 1.10) should trigger
    assert len(anomalies) == 1
    assert anomalies[0]['actual'] == 500
```

---

## Part 10 — Integration Testing

- Replay historical telemetry from a known game patch where an exploit occurred (e.g., players found a way to duplicate items).
- Run the Flink aggregation and Prophet pipeline.
- Assert that the alert triggers exactly at the hour the exploit was discovered by the community, proving the system works.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Drill-down Analytics** | Macro-anomalies tell you *that* an exploit is happening. You need to know *who* is doing it. If Prophet triggers a macro alert, it should automatically trigger a secondary PySpark job that runs Isolation Forests on player-level transaction graphs to find the specific abusers. |
| **Model Retraining** | Prophet fits quickly. Retraining every hour on the last 30 days is computationally fine. No need for complex model registries for v1. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Prophet vs LSTMs | LSTMs are powerful for multivariate series, but require immense hyperparameter tuning, GPUs, and are opaque. Prophet is CPU-bound, interpretable (you can plot the seasonal components), and handles missing data gracefully. |
| Dynamic vs Static Thresholds | Static thresholds (Alert if Gold > 1M) fail on weekends. Dynamic models prevent "alert fatigue" where designers ignore alerts that fire every Saturday night. |

---

## Part 13 — Alternative Approaches

1. **Autoencoders (Multivariate):** Train an Autoencoder on a vector of `[faucets, sinks, ah_prices, active_players]`. A bug that gives free gold will break the correlation between `faucets` and `active_players` (massive gold, normal player count). The reconstruction error will spike, triggering an alert.
2. **Graph Analytics:** Represent the economy as a directed graph. Nodes = players/NPCs, Edges = transactions. Run PageRank or network flow algorithms to detect money laundering or bot rings pooling gold to a central mule account.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Patch Day Structural Break | New expansion drops, gold generation permanently doubles. Prophet flags an anomaly every hour for 30 days. | Implement a "Reset Baseline" feature. When a patch drops, the model should weight recent data heavily or truncate the training window to post-patch only. |
| Telemetry Pipeline Lag | Kafka backs up, hour 1 aggregates arrive at hour 3. | The model predicts low gold generation (because data is missing), which doesn't trigger a high-bound anomaly, but ruins the training set. Add a data-completeness check: do not run ML if `event_count < 90% expected`. |

---

## Part 15 — Debugging

**Symptom:** The system triggers a critical alert: "Auction House volume spiked by 500%." The designers panic, assuming a duplication exploit.

**Debugging steps:**
1. Plot the time-series. Is it a gradual slope or a vertical step function? (Exploits are usually vertical steps).
2. Check external metadata. Did a prominent streamer just tell 50,000 viewers to buy a specific item?
3. Check the micro-level distribution. Was the 500% spike caused by 10,000 players buying 1 item each, or 1 player buying 10,000 items? (If 1 player, it's an exploit or a bot).

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `economy_model_mape` (Mean Absolute Pct Error) | > 20% → The model no longer understands the baseline economy. |
| `data_freshness_delay_minutes` | > 15m → Critical (We are blind to live exploits). |

---

## Part 17 — Production Improvements

1. **Auto-Banning Mules:** Connect the macro-alert to a micro-level classifier. If macro-inflation spikes, automatically lower the ban-threshold for the player-level bot detection model.
2. **Predictive Economy Balancing:** Use the time-series model to simulate the future. "If we release a mount that costs 1M gold, what will happen to the M0 supply over the next 6 months?" Provide this tool to game designers.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The anomaly detector works perfectly. However, the game designers complain that it takes them 5 hours of SQL writing to figure out *which* item or quest is causing the gold spike. How do you automate the root-cause analysis?"**
2. **"During a major server outage, players couldn't log in for 12 hours. The gold generation was 0. How does this massive block of zeros affect Prophet's forecasting for the next week, and how do you handle it?"**
3. **"Hackers figure out they can inject 10% extra gold every hour, staying just underneath your `yhat_upper * 1.10` threshold. Over a month, this destroys the economy via 'boiling the frog'. How do you catch this?"**

---

## Part 19 — Ideal Answers

**Q1 (Root cause automation):**
> "We can implement a multidimensional contribution analysis (e.g., using a library like `Kats` or building an Apriori algorithm). When an anomaly triggers, the system automatically groups the gold generation by `quest_id`, `item_id`, and `zone_id`. It calculates the Kullback-Leibler (KL) divergence between the anomalous hour's distribution and the previous day's distribution. The Slack alert will say: *Anomaly Detected. Top Contributor: Quest_994 (Contribution increased by 800%).*"

**Q2 (Outage zeros):**
> "Outages create outliers that warp the model's learned seasonality. We must implement outlier masking before training. We define 'known outage windows' and set the values to `NaN`. Prophet handles missing data natively via interpolation, ensuring the 12-hour zero-block doesn't drag down the baseline for the following Tuesday."

**Q3 (Boiling the frog / Slow exploit):**
> "An hourly threshold won't catch a slow bleed. We need multi-resolution forecasting. We run the Prophet model concurrently on multiple window sizes: Hourly (catches sudden spikes), Daily (catches sustained 24h increases), and Weekly (catches the 10% bleed). If the 7-day rolling average exceeds the Weekly forecast, the alert fires even if no single hour crossed the hourly threshold."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Structures the time-series problem clearly (Faucets vs Sinks).
- Proposes Prophet or similar seasonal-aware models over static thresholds.
- Understands the difference between macro (server) and micro (player) detection.
- Flawlessly answers the root-cause (KL divergence/contribution) and multi-resolution (boiling the frog) questions.

### Hire
- Writes a solid forecasting script.
- Understands the need for an aggregation layer (Flink/SQL) before ML.
- Explains how to handle patches and missing data gracefully.

### Lean Hire
- Tries to build a player-level classification model instead of a macro time-series model (which is computationally unfeasible to run on every transaction in real-time without massive infra).
- Needs prompting to understand seasonality.

### Lean No Hire
- Suggests setting a static threshold: `if gold > 1,000,000: alert()`. Fails to recognize this breaks every weekend.
- Cannot write code to manipulate time-series data using Pandas.

### No Hire
- Fails to understand virtual economies.
- Suggests using LLMs for numerical time-series forecasting.
