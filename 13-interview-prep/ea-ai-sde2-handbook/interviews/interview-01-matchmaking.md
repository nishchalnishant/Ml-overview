# Interview 01 — Real-Time Skill-Based Matchmaking (FIFA Online)
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the FIFA Online team. The current matchmaking system uses a simple Elo rating with a fixed ±200 Elo search radius. Players are complaining about:
- Long queue times (>3 min) during off-peak hours
- Lopsided matches when the search radius widens too aggressively
- Smurfs (high-skill players on low-rank accounts) ruining beginner lobbies

Your task is to **design and implement a production-grade, real-time skill-based matchmaking (SBMM) system** that reduces match unfairness while keeping queue times acceptable.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Daily active players (DAU) and peak concurrent users
- Acceptable queue time SLA (p50, p95, p99)
- Match quality metric definition (what is "fair"?)
- Number of players per match (1v1? 5v5? 11v11?)
- Game modes (ranked vs. casual)
- Geographic regions and server topology
- Whether there is an existing matchmaking queue service
- Party/group play support
- Smurfing definition (how do you detect it?)
- Platform (PC, console, cross-play?)
- Latency constraints (network ping must be <80ms?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the peak concurrent player count globally?"**  
   → *Answer: ~800k peak CCU across all regions.*

2. **"What queue time SLAs are we targeting?"**  
   → *p50 < 30s, p95 < 90s, p99 < 3 min.*

3. **"How many players per match?"**  
   → *11v11 online, but also 1v1 (Ultimate Team). Start with 1v1.*

4. **"How do we define a 'fair' match?"**  
   → *Win probability for either side should be between 45–55% (near 50/50).*

5. **"Do we need to support pre-formed parties?"**  
   → *Not in v1. Solo queue only.*

6. **"What is our ping constraint?"**  
   → *Match server must be within 80ms RTT of both players.*

7. **"Is there a smurfing detection system already?"**  
   → *No. You may propose one as part of the solution.*

8. **"Are we cross-play (PC + Console)?"**  
   → *Yes, with an opt-out option.*

---

## Part 4 — Expected Assumptions

- 1v1 ranked mode, solo queue
- Target regions: NA, EU, APAC (3 matchmaking pools)
- ~267k peak CCU per region
- Match is fair if predicted win probability ∈ [0.45, 0.55]
- Max ping threshold: 80ms (used as a hard filter, not a soft signal)
- Smurf detection is a v2 feature; v1 will use behavioral signals as soft penalty
- Skill signal = Elo + 5 auxiliary features (recent form, playstyle cluster, device type, connection quality, historical head-to-head)

---

## Part 5 — High-Level Solution

```
Player enters queue
       │
       ▼
  Feature Extraction (real-time)
  ┌─────────────────────────────────────┐
  │  Elo · Recent win rate · Ping data  │
  │  Playstyle cluster · Time-in-queue  │
  └─────────────────────────────────────┘
       │
       ▼
  Matchmaking Service (stateful, in-memory)
  ┌─────────────────────────────────────┐
  │  Waiting queue (sorted by skill)    │
  │  Sliding search radius expander     │
  │  Win-probability predictor (model)  │
  │  Ping-matrix filter                 │
  └─────────────────────────────────────┘
       │
       ▼
  Match Proposal → Win-Prob Check → Accept/Reject
       │
       ▼
  Match Created → Lobby Service → Game Server Allocation
```

**Core ML component:** A lightweight win-probability model trained on historical match outcomes that takes two players' feature vectors and outputs P(player_A wins). A match is accepted if P ∈ [0.45, 0.55].

---

## Part 6 — Step-by-Step Implementation

### Step 1: Feature Engineering
- Elo rating (updated post-match via standard Elo formula)
- 7-day rolling win rate (from Redis time-series)
- Playstyle cluster (k-means on in-game action embeddings, recomputed weekly)
- Estimated ping between player pairs (from geolocation + latency lookup table)
- Time-in-queue (increases search radius dynamically)

### Step 2: Win-Probability Model
- **Model:** Siamese-style MLP or Gradient Boosted Trees (LightGBM)
- **Input:** Feature diff vector `|f_A - f_B|` + interaction terms
- **Output:** P(A wins) ∈ [0, 1]
- **Training data:** 100M+ historical 1v1 matches with outcomes
- **Label:** `1` if player A won, `0` otherwise
- **Calibration:** Platt scaling to ensure output = true probability

### Step 3: Queue Management
- Per-region priority queue sorted by wait time
- Every 500ms: try to match the longest-waiting player with the best candidate in their current search radius
- Expand search radius by ±5 Elo per 5 seconds of waiting (capped at ±200)
- Hard reject if ping > 80ms

### Step 4: Smurf Detection Signal (soft)
- Flag players whose Elo is in bottom 20% but recent win rate > 80%
- Apply a "hidden MMR" boost: use the behavioral model's estimate instead of displayed Elo
- Do not expose this to players

### Step 5: Post-Match Elo Update
- Standard Elo update with K-factor 32 for new players, 16 for established
- Also feed result back into win-probability model retraining pipeline (daily batch)

---

## Part 7 — Complete Python Code

```python
"""
matchmaking_service.py - Production SBMM core logic
"""
import heapq
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import lightgbm as lgb
import redis
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(order=True)
class QueuedPlayer:
    """Player waiting in the matchmaking queue."""
    entered_at: float = field(compare=True)
    player_id: str = field(compare=False)
    elo: float = field(compare=False)
    win_rate_7d: float = field(compare=False)
    playstyle_cluster: int = field(compare=False)
    region: str = field(compare=False)
    ping_zone: str = field(compare=False)  # e.g., "us-east-1"

    @property
    def time_in_queue(self) -> float:
        return time.time() - self.entered_at

    @property
    def search_radius(self) -> float:
        """Expand Elo search radius over time, capped at 200."""
        base_radius = 50.0
        expansion = (self.time_in_queue / 5.0) * 5.0  # +5 Elo per 5s
        return min(base_radius + expansion, 200.0)


class MatchRequest(BaseModel):
    player_id: str
    elo: float
    win_rate_7d: float
    playstyle_cluster: int
    region: str
    ping_zone: str


class MatchResult(BaseModel):
    matched: bool
    opponent_id: Optional[str] = None
    predicted_win_probability: Optional[float] = None
    queue_time_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Win-Probability Model
# ---------------------------------------------------------------------------

class WinProbabilityModel:
    """Wraps a trained LightGBM model that predicts P(player_A wins)."""

    def __init__(self, model_path: str):
        self.model = lgb.Booster(model_file=model_path)
        logger.info("Win probability model loaded from %s", model_path)

    def predict(self, player_a: QueuedPlayer, player_b: QueuedPlayer) -> float:
        """
        Returns P(A wins). Features: absolute diff + both raw values.
        """
        features = np.array([[
            player_a.elo - player_b.elo,           # Elo diff
            abs(player_a.elo - player_b.elo),       # Elo abs diff
            player_a.win_rate_7d,
            player_b.win_rate_7d,
            player_a.win_rate_7d - player_b.win_rate_7d,
            float(player_a.playstyle_cluster == player_b.playstyle_cluster),
        ]])
        prob = float(self.model.predict(features)[0])
        return prob


# ---------------------------------------------------------------------------
# Ping Filter
# ---------------------------------------------------------------------------

PING_TABLE: dict[tuple[str, str], float] = {
    # Precomputed average RTT between ping zones (ms)
    ("us-east-1", "us-west-2"): 65.0,
    ("us-east-1", "eu-west-1"): 95.0,
    ("eu-west-1", "ap-southeast-1"): 175.0,
    # ... (loaded from config in production)
}

def get_estimated_ping(zone_a: str, zone_b: str) -> float:
    key = tuple(sorted([zone_a, zone_b]))
    return PING_TABLE.get(key, 999.0)


# ---------------------------------------------------------------------------
# Matchmaking Queue (in-memory, per region)
# ---------------------------------------------------------------------------

class MatchmakingQueue:
    """Min-heap queue ordered by entry time (oldest first)."""

    def __init__(self, region: str, win_prob_model: WinProbabilityModel):
        self.region = region
        self.heap: list[QueuedPlayer] = []
        self.player_index: dict[str, QueuedPlayer] = {}
        self.model = win_prob_model
        self.FAIRNESS_LOWER = 0.45
        self.FAIRNESS_UPPER = 0.55
        self.MAX_PING_MS = 80.0

    def enqueue(self, player: QueuedPlayer) -> None:
        heapq.heappush(self.heap, player)
        self.player_index[player.player_id] = player
        logger.info("Player %s entered queue (region=%s, elo=%.0f)",
                    player.player_id, self.region, player.elo)

    def dequeue(self, player_id: str) -> None:
        player = self.player_index.pop(player_id, None)
        if player:
            # Mark as removed (lazy deletion)
            player.player_id = "__removed__"

    def find_match(self, candidate: QueuedPlayer) -> Optional[QueuedPlayer]:
        """
        Scan the queue for the best match for `candidate`.
        Criteria:
          1. Elo within candidate's current search radius
          2. Estimated ping ≤ 80ms
          3. Predicted win probability ∈ [0.45, 0.55]
        Returns the best opponent or None.
        """
        best: Optional[QueuedPlayer] = None
        best_fairness_score = float("inf")  # closer to 0 = closer to 50/50

        for opponent in self.heap:
            if opponent.player_id in ("__removed__", candidate.player_id):
                continue
            if opponent.player_id not in self.player_index:
                continue

            # Elo radius check
            if abs(candidate.elo - opponent.elo) > candidate.search_radius:
                continue

            # Ping check
            ping = get_estimated_ping(candidate.ping_zone, opponent.ping_zone)
            if ping > self.MAX_PING_MS:
                continue

            # Win-probability check
            win_prob = self.model.predict(candidate, opponent)
            if not (self.FAIRNESS_LOWER <= win_prob <= self.FAIRNESS_UPPER):
                continue

            # Pick the match closest to 50/50
            fairness_score = abs(win_prob - 0.5)
            if fairness_score < best_fairness_score:
                best_fairness_score = fairness_score
                best = opponent

        return best

    def tick(self) -> list[tuple[QueuedPlayer, QueuedPlayer, float]]:
        """
        Called every 500ms. Attempts to form matches.
        Returns list of (player_a, player_b, win_prob) tuples.
        """
        matches = []
        matched_ids: set[str] = set()

        # Rebuild heap to remove stale entries
        self.heap = [p for p in self.heap
                     if p.player_id in self.player_index]
        heapq.heapify(self.heap)

        for candidate in list(self.heap):
            if candidate.player_id in matched_ids:
                continue
            opponent = self.find_match(candidate)
            if opponent and opponent.player_id not in matched_ids:
                win_prob = self.model.predict(candidate, opponent)
                matches.append((candidate, opponent, win_prob))
                matched_ids.add(candidate.player_id)
                matched_ids.add(opponent.player_id)
                self.dequeue(candidate.player_id)
                self.dequeue(opponent.player_id)
                logger.info(
                    "Match formed: %s vs %s | P(A wins)=%.3f | "
                    "queue_time_A=%.1fs queue_time_B=%.1fs",
                    candidate.player_id, opponent.player_id, win_prob,
                    candidate.time_in_queue, opponent.time_in_queue,
                )

        return matches


# ---------------------------------------------------------------------------
# FastAPI Service
# ---------------------------------------------------------------------------

app = FastAPI(title="EA SBMM Service", version="1.0.0")
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

# One queue per region
_model = WinProbabilityModel(model_path="/models/win_prob_lgbm.txt")
queues: dict[str, MatchmakingQueue] = {
    "na": MatchmakingQueue("na", _model),
    "eu": MatchmakingQueue("eu", _model),
    "apac": MatchmakingQueue("apac", _model),
}


@app.post("/queue/join", response_model=MatchResult)
async def join_queue(req: MatchRequest, background_tasks: BackgroundTasks):
    region = req.region.lower()
    if region not in queues:
        return MatchResult(matched=False)

    player = QueuedPlayer(
        entered_at=time.time(),
        player_id=req.player_id,
        elo=req.elo,
        win_rate_7d=req.win_rate_7d,
        playstyle_cluster=req.playstyle_cluster,
        region=region,
        ping_zone=req.ping_zone,
    )
    queues[region].enqueue(player)
    redis_client.set(f"queue:{req.player_id}", "waiting", ex=300)
    return MatchResult(matched=False)


@app.delete("/queue/leave/{player_id}")
async def leave_queue(player_id: str):
    for q in queues.values():
        q.dequeue(player_id)
    redis_client.delete(f"queue:{player_id}")
    return {"status": "removed"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "queue_sizes": {region: len(q.heap) for region, q in queues.items()},
    }
```

---

## Part 8 — Deployment

### FastAPI Service
```yaml
# docker-compose.yml (local dev)
version: "3.9"
services:
  matchmaking:
    build: .
    ports: ["8080:8080"]
    environment:
      - MODEL_PATH=/models/win_prob_lgbm.txt
      - REDIS_HOST=redis
    volumes:
      - ./models:/models
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "matchmaking_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: matchmaking-service
  namespace: ea-sbmm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: matchmaking
  template:
    spec:
      containers:
      - name: matchmaking
        image: ea-registry/matchmaking:1.0.0
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: matchmaking-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: matchmaking-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

### Monitoring (Prometheus + Grafana)
Key metrics to expose:
- `sbmm_queue_size{region}` — real-time queue depth
- `sbmm_match_quality_histogram` — distribution of win_prob per match
- `sbmm_queue_time_seconds{quantile}` — p50/p95/p99 queue time
- `sbmm_tick_duration_seconds` — tick processing latency

### Logging
```python
import structlog
log = structlog.get_logger()
log.info("match_formed", player_a=..., player_b=..., win_prob=..., queue_time_a=..., queue_time_b=...)
```
→ Shipped to Datadog / Splunk via Fluentd sidecar.

### CI/CD (GitHub Actions)
```yaml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=matchmaking_service
  build-push:
    needs: test
    steps:
      - run: docker build -t ea-registry/matchmaking:${{ github.sha }} .
      - run: docker push ea-registry/matchmaking:${{ github.sha }}
  deploy:
    needs: build-push
    steps:
      - run: kubectl set image deployment/matchmaking-service matchmaking=ea-registry/matchmaking:${{ github.sha }}
```

---

## Part 9 — Unit Testing

```python
import pytest
from unittest.mock import MagicMock, patch
import time
from matchmaking_service import QueuedPlayer, MatchmakingQueue, WinProbabilityModel

@pytest.fixture
def mock_model():
    model = MagicMock(spec=WinProbabilityModel)
    model.predict.return_value = 0.50  # perfectly fair match
    return model

@pytest.fixture
def queue(mock_model):
    return MatchmakingQueue(region="na", win_prob_model=mock_model)

def make_player(pid, elo, zone="us-east-1"):
    return QueuedPlayer(
        entered_at=time.time(),
        player_id=pid,
        elo=elo,
        win_rate_7d=0.5,
        playstyle_cluster=0,
        region="na",
        ping_zone=zone,
    )

def test_enqueue_adds_player(queue):
    p = make_player("p1", 1500)
    queue.enqueue(p)
    assert "p1" in queue.player_index

def test_dequeue_removes_player(queue):
    p = make_player("p1", 1500)
    queue.enqueue(p)
    queue.dequeue("p1")
    assert "p1" not in queue.player_index

def test_match_formed_when_elo_close(queue, mock_model):
    p1 = make_player("p1", 1500)
    p2 = make_player("p2", 1510)
    queue.enqueue(p1)
    queue.enqueue(p2)
    matches = queue.tick()
    assert len(matches) == 1
    assert {matches[0][0].player_id, matches[0][1].player_id} == {"p1", "p2"}

def test_no_match_when_elo_far(queue, mock_model):
    mock_model.predict.return_value = 0.90  # unfair
    p1 = make_player("p1", 1500)
    p2 = make_player("p2", 2000)
    queue.enqueue(p1)
    queue.enqueue(p2)
    matches = queue.tick()
    assert len(matches) == 0

def test_search_radius_expands_with_time():
    p = make_player("p1", 1500)
    p.entered_at = time.time() - 30  # 30 seconds in queue
    # 30s / 5s * 5 Elo = +30 radius
    assert p.search_radius == min(50 + 30, 200)

def test_ping_filter_blocks_high_latency(queue, mock_model):
    p1 = make_player("p1", 1500, zone="us-east-1")
    p2 = make_player("p2", 1510, zone="eu-west-1")  # 95ms RTT > 80ms
    queue.enqueue(p1)
    queue.enqueue(p2)
    matches = queue.tick()
    assert len(matches) == 0
```

---

## Part 10 — Integration Testing

```python
# tests/integration/test_matchmaking_e2e.py
import pytest
import httpx
import asyncio

BASE_URL = "http://localhost:8080"

@pytest.mark.asyncio
async def test_two_similar_players_match():
    async with httpx.AsyncClient() as client:
        # Enqueue player A
        r1 = await client.post(f"{BASE_URL}/queue/join", json={
            "player_id": "int_p1", "elo": 1500, "win_rate_7d": 0.52,
            "playstyle_cluster": 0, "region": "na", "ping_zone": "us-east-1"
        })
        assert r1.status_code == 200

        # Enqueue player B (similar skill)
        r2 = await client.post(f"{BASE_URL}/queue/join", json={
            "player_id": "int_p2", "elo": 1520, "win_rate_7d": 0.50,
            "playstyle_cluster": 0, "region": "na", "ping_zone": "us-east-1"
        })
        assert r2.status_code == 200

        # Wait for matchmaking tick
        await asyncio.sleep(1.0)

        # Verify both are no longer in queue
        health = await client.get(f"{BASE_URL}/health")
        data = health.json()
        # Queue should be empty (matched)
        assert data["queue_sizes"]["na"] == 0
```

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **CCU growth** | Horizontal pod autoscaling; each pod owns a shard of the queue (partitioned by Elo bucket) |
| **Tick frequency** | Move from polling to event-driven: Kafka topic per region, player joins emit events |
| **Win-prob model** | Must be <1ms inference; LightGBM in-process satisfies this. Neural networks would require gRPC model server |
| **State** | Current in-memory state is per-pod. At scale, use Redis Sorted Set as shared queue (ZADD by entry time, ZRANGEBYSCORE by Elo) |
| **Global scale** | Separate Kubernetes clusters per region (NA, EU, APAC). Cross-region matchmaking only if wait > 2 min |
| **Model updates** | Blue/green model swap: load new model into `/models/candidate/`, promote after shadow testing |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| LightGBM vs. Neural Net for win-prob | LightGBM: <1ms, interpretable. Neural Net: potentially more accurate but 10-50ms, needs GPU server |
| In-memory queue vs. Redis queue | In-memory: lower latency, but loses state on pod restart. Redis: durable, horizontally scalable, ~1ms overhead |
| Elo expanding radius | Wider radius = faster queue but worse match quality. The expansion rate is the primary tuning knob |
| Hard ping filter (80ms) | Removes high-ping matches entirely; better experience but increases queue time in sparse regions |
| Calibrated probabilities | Required for the [0.45, 0.55] window to be meaningful. Without calibration, raw model scores are not true probabilities |

---

## Part 13 — Alternative Approaches

1. **TrueSkill (Microsoft):** Bayesian skill rating that models uncertainty. Better than Elo for new players. More complex update logic.
2. **Deep SBMM (meta-learning):** Train a policy gradient agent that optimizes long-term match quality and queue time jointly. EA research paper (2022) approach.
3. **Priority Queue with scoring function:** Replace win-probability model with a hand-crafted scoring function `quality(A,B) = f(elo_diff, ping, wait_time)`. Simpler, less accurate.
4. **Auction-based matching:** Players "bid" on match slots. Not suitable for games (creates pay-to-win perception).

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Win-prob model fails to load | All matches blocked | Fallback: pure Elo ±100 matching; alert PagerDuty |
| Redis unavailable | Queue state lost on pod restart | Circuit breaker; use in-memory with warning; restart recovery via player re-join |
| Smurf account exploits | Unfair matches for beginners | Behavioral anomaly detection; hidden MMR; manual review pipeline |
| Queue starvation (sparse region) | p99 > 5 min | Bot fill (PvAI) after 3 min; cross-region routing |
| Ping data stale | Players matched across high-latency zones | Refresh ping estimates every 30s from geolocation service |

---

## Part 15 — Debugging

**Symptom:** Match quality dropping (win_prob skewing toward 0.7+)

**Debugging steps:**
1. Check `sbmm_match_quality_histogram` in Grafana. Is the distribution shifting?
2. Query recent matches: `SELECT win_prob FROM matches WHERE created_at > NOW() - INTERVAL 1h`
3. Check if the win-probability model was recently updated. Roll back if needed.
4. Check if Elo distribution has shifted (e.g., inflation from a season reset)
5. Verify calibration: `from sklearn.calibration import calibration_curve` on recent holdout data

**Symptom:** Queue times spiking at p95

**Debugging steps:**
1. Check queue depth per region: `kubectl exec -it <pod> -- curl localhost:8080/health`
2. Check tick duration: `sbmm_tick_duration_seconds` — if >400ms, the tick is a bottleneck
3. Check if fairness window is too tight. Loosen temporarily to [0.42, 0.58].
4. Check ping filter: are many players being rejected due to latency?

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `sbmm_queue_time_p95_seconds` | > 90s → Warning |
| `sbmm_queue_time_p99_seconds` | > 180s → Critical |
| `sbmm_match_quality_p5_win_prob` | < 0.40 → Warning |
| `sbmm_queue_size{region}` | > 5000 → Scale-out trigger |
| `sbmm_tick_duration_p99_ms` | > 450ms → Alarm |
| `model_prediction_error_rate` | > 1% → Alert |

**Dashboards:**
- Real-time queue depth by region (line chart)
- Match quality distribution (histogram)
- Queue time percentiles (time series)
- Smurf detection flag rate (anomaly panel)

---

## Part 17 — Production Improvements

1. **Party/Group matchmaking:** Weighted average skill of party + variance penalty for mixed-skill parties
2. **Cross-play skill separation:** Separate MMR for controller vs. mouse/keyboard in cross-play
3. **TrueSkill 2 migration:** Incorporate playtime, team composition, and map as Bayesian factors
4. **Smurfing countermeasure:** Combine account age, match history variance, and decision tree classifier
5. **Dynamic fairness window:** Loosen [0.45, 0.55] to [0.40, 0.60] during off-peak; tighten during peak
6. **Match post-analysis feedback:** Collect actual win/loss and re-weight model features weekly via online learning

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"Your Elo rating only updates after each match. A player goes on a 20-game win streak mid-season. How does your system handle this lag?"**

2. **"The win-probability model was trained 6 months ago on a meta that has since changed (new player abilities patched). How do you detect this drift and what do you do about it?"**

3. **"You have 50 players in queue in the APAC region at 3am. Your fairness window means nobody gets matched. Walk me through exactly what happens."**

4. **"A player complains their predicted win probability is 0.51 against an opponent but they lost 8 of 10 games against similar opponents. Is the model wrong? How do you investigate?"**

5. **"Your Kubernetes cluster goes down mid-tick. 800 players are in queue. What happens to their queue state?"**

6. **"Product wants to A/B test a tighter fairness window [0.48, 0.52] vs. the current [0.45, 0.55]. How would you run this experiment without introducing survivorship bias?"**

---

## Part 19 — Ideal Answers

**Q1 (Elo lag):**  
> "This is the Elo lag problem. Two solutions: (a) supplement Elo with a fast-updating signal like 7-day rolling win rate which the system already uses, and (b) use a larger K-factor for players on a hot streak (detected as win rate > 70% over last 10 games), so Elo updates faster. TrueSkill naturally handles this better because it tracks uncertainty — a player on a win streak has lower uncertainty and gets bigger updates."

**Q2 (Model drift):**  
> "I would monitor the PSI (Population Stability Index) on key input features weekly — if PSI > 0.2, the feature distribution has shifted significantly. I'd also track calibration: bucket predicted probabilities and compare to actual win rates. If actual win rate in the [0.45-0.55] bucket drops to 0.61, the model is miscalibrated. Trigger retraining with recent 90-day data and shadow-test before promoting."

**Q3 (Sparse queue):**  
> "After 2 minutes, loosen to [0.40, 0.60]. After 3 minutes, loosen to [0.35, 0.65] and expand ping limit to 120ms. After 4 minutes, offer a PvAI match with disclosure to the player. Log all instances of bot fill for region capacity planning."

**Q4 (Model accuracy question):**  
> "0.51 means we expect it to be essentially a coin flip — 10 games is not statistically significant. With 10 games, a player could lose 8 of 10 even in truly 50/50 matchups with ~4.4% probability. I'd look at a sample of 100+ games against similar opponents. If actual win rate is 35% not 50%, there's a systematic bias and I'd investigate feature leakage, label skew, or meta-shift."

**Q5 (State loss):**  
> "With in-memory queues, all 800 players lose their queue state. Their clients should poll a `/queue/status/{player_id}` endpoint, get a 404 or timeout, and trigger an automatic re-join. Redis-backed queues would survive pod restarts since state is external. I'd implement this for v2."

**Q6 (A/B test):**  
> "Use player-level assignment (not match-level) to avoid contamination. Assign players deterministically by `hash(player_id) % 2`. Track metrics: p50/p95 queue time, win_prob distribution, player retention at 7-day and 30-day. The survivorship bias risk is real — if the tight-window group has longer queues, players may quit before being matched, making remaining players look more skilled. Measure dropout rates separately."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Asks ALL clarifying questions in Part 3 without prompting
- Proactively mentions Elo lag, calibration requirement, sparse-region fallback
- Designs Redis-backed state for HA without being asked
- Answers all follow-up questions with real-world nuance (PSI drift detection, survivorship bias in A/B)
- Proposes smurfing detection as v2 feature
- Writes clean, production-quality Python with proper dataclasses, logging, and type hints

### Hire
- Asks most clarifying questions with minor prompting
- Designs core queue + win-prob model correctly
- Mentions HPA, Redis, and monitoring with some gaps
- Answers follow-up questions partially (misses survivorship bias)
- Code is functional but lacks some error handling

### Lean Hire
- Gets the Elo + model approach right
- Misses calibration requirement for win probability
- Deployment section is generic (just "use Kubernetes")
- Cannot articulate the A/B test design properly
- Code works for the happy path but lacks edge cases

### Lean No Hire
- Designs a basic Elo system only
- Does not mention win-probability model
- Cannot discuss drift detection
- Deployment answer is "deploy with Docker"
- Cannot answer the Elo lag follow-up

### No Hire
- Cannot design a queue data structure
- Proposes neural network without discussing latency constraint
- No mention of fairness metrics
- Cannot discuss failure modes
- Code is pseudocode at best
