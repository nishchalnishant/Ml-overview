# PART 16: GAME AI DECISION FRAMEWORKS (Electronic Arts Focus)

## Goal
To teach candidates how AI engineering in gaming differs from generic enterprise AI, and how to apply domain-specific reasoning for EA interview scenarios.

## Mental Model
**Gaming AI operates at the intersection of real-time systems, massive scale, user psychology, and LiveOps economics.**
A recommendation system for a game is fundamentally different from Netflix: players have session states, game economy constraints, fairness requirements, and anti-cheat concerns. Always bring gaming context into your answers.

---

## 16.1 Player Churn Prediction

### Problem Definition
Predict which players are likely to stop playing in the next 7/14/30 days, enabling proactive retention interventions.

### Decision Framework
```text
What signals predict churn?
├── Engagement signals: Session frequency, session length, time since last login.
├── Behavioral signals: Match win rate, progression velocity, social connections.
├── Economic signals: Spending history, subscription status.
├── Sentiment signals: Chat sentiment, support ticket frequency, reviews.

Model selection:
├── Features: Primarily tabular → XGBoost / LightGBM.
├── Add behavioral sequences: LSTM or Transformer over session sequence.
└── Output: Churn probability + time-to-churn estimate.
```

### Gaming-Specific Considerations vs Enterprise
| Enterprise Churn | Game Churn |
| :--- | :--- |
| Measured in months | Can happen overnight ("rage quit") |
| Clear economic signals | Mixed signals (bad matchmaking ≠ churn) |
| User contacted via email | Intervention via in-game notification |
| Slow feedback loop | Fast feedback loop (player logs back in) |

### Intervention Design
```text
Churn probability buckets:
├── > 0.8 → High risk → Personalized "welcome back" reward.
├── 0.5-0.8 → Medium risk → Surface new content, social features.
└── < 0.5 → Low risk → No intervention (avoid notification fatigue).
```

### Evaluation Metrics
- **Offline:** Precision@K (top 10% at-risk users), Recall, AUC-PR.
- **Online:** Intervention CTR, 7-day reactivation rate, revenue retention.

---

## 16.2 Matchmaking (Skill-Based)

### Problem Definition
Create balanced matches that maximize player enjoyment (not just accuracy). **Fun > Precision.**

### Decision Framework
```text
What defines a "good" match?
├── Skill balance: Similar Elo/MMR across teams.
├── Wait time: Player will tolerate 30-60 seconds max.
├── Role/playstyle compatibility: Tank + Healer + DPS, not all snipers.
└── Social: Keep friends on the same team.

Tradeoff: Skill balance vs. wait time.
├── At peak hours → Stricter skill range (better matches).
└── Off-peak → Widen skill range (reduce wait time).
```

### Architecture
```text
[Player requests match]
        │
        ▼
[Candidate pool generation] ← Filter by region, mode, skill band.
        │
        ▼
[Matching algorithm] ← Hungarian algorithm or greedy Elo-based matching.
        │
        ▼
[Quality evaluation] ← Score: |Elo_team_A - Elo_team_B| < threshold.
        │
        ▼
[Dynamic expansion] ← If no match found in 15s, widen Elo band.
```

### Gaming-Specific Metrics
- **Match quality:** Elo standard deviation within a match.
- **Fun score:** Post-match survey response ("Was this match fair?").
- **Toxicity rate:** Toxic chats per match as proxy for match frustration.
- **Business:** Player retention after matched games.

---

## 16.3 Cheat Detection

### Problem Definition
Identify players using aimbots, wallhacks, speedhacks, or other cheating software in real-time or near-real-time.

### Decision Framework
```text
Cheat type?
├── AIMBOT (unnatural aim patterns) → Statistical analysis of aim behavior.
│   └── Features: Snap-to-target velocity, headshot rate, reaction time.
├── SPEEDHACK (impossible movement) → Physics violation detection.
│   └── Features: Packets showing position delta impossible given physics.
├── WALLHACK (sees through walls) → Information advantage analysis.
│   └── Features: Pre-fires targets that should be invisible.
└── ACCOUNT FARMING / BOOSTING → Graph analysis of suspicious win patterns.
```

### Architecture
```text
[Real-time game events (low latency, ~10ms)]
        │
        ▼
[Rules engine] ← Hard physics violations → Immediate flag.
        │
        ▼
[Statistical model] ← ML model on behavioral features → Risk score.
        │
        ▼
[Async review] ← High-risk scores → Human review or automatic ban.
```

### Critical Design Choice: False Positive Management
```text
False Positive (banning a legitimate player) is CATASTROPHIC to brand trust.

Architecture:
├── Tier 1: Automatic soft-ban (can't play ranked). Requires low threshold.
├── Tier 2: Manual review for permanent ban. Requires high threshold.
└── Appeal system: Players can appeal for human review.
```

---

## 16.4 Toxic Chat Moderation

### Decision Framework
```text
What is the latency requirement?
├── REAL-TIME (<50ms, block message before delivery) →
│   └── Lightweight model: FastText or distilled BERT via ONNX Runtime on CPU.
└── NEAR-REAL-TIME (1-5s delay acceptable) →
    └── Heavier model: Full BERT or multilingual model.

Language coverage?
├── English only → monolingual model (faster, cheaper).
└── Global game (EA titles are global) → Multilingual model (mBERT, XLM-R).
```

### Cascade Architecture (Production)
```text
Message arrives
    │
    ▼
[Tier 1: Regex + Keyword filter] ← 1ms, catches obvious slurs. ~70% of cases.
    │ (passes through)
    ▼
[Tier 2: FastText classifier] ← 5ms, handles paraphrasing. ~25% of cases.
    │ (uncertain probability 0.4-0.6)
    ▼
[Tier 3: BERT multilingual] ← 50ms, nuanced context. ~5% of cases.
    │
    ▼
[Action: Block / Mute / Report]
```

### Game-Specific Challenge
Players actively try to evade filters ("leetspeak", character substitution, multi-language mixing). The model must generalize, not memorize.

---

## 16.5 Dynamic Difficulty Adjustment (DDA)

### Problem Definition
Adjust game difficulty in real-time to keep players in the "flow state" (not bored, not frustrated).

### Decision Framework
```text
Player state assessment:
├── Win/loss ratio in last N matches → Struggling or coasting?
├── Time spent on a level before quitting → Frustration indicator.
├── Health remaining at level end → Too easy or too hard?
└── Engagement signals: Session length trend.

Adjustment levers:
├── Enemy AI difficulty parameters.
├── Enemy health/damage values.
├── Spawn rates and position.
└── Hint/reward frequency.
```

### Model Choice
- **Rule-based (baseline):** If player dies > 3 times on same checkpoint → reduce enemy damage by 20%.
- **Bandit algorithm (production):** Contextual bandit (player features → difficulty adjustment → reward: session continuation).
- **RL (advanced):** Deep RL agent that learns optimal difficulty policy maximizing long-term engagement.

---

## 16.6 Player Behavior Analytics & Telemetry

### Architecture for Game Telemetry at Scale
```text
[Game Client]
    │ (events: kills, deaths, purchases, clicks)
    ▼
[Kafka Topics] ← Partitioned by player_id, game session.
    │
    ├── [Stream Processor (Flink)]
    │       └── Real-time: Session analytics, live alerts.
    │
    └── [Batch (Spark)]
            └── Overnight: Feature engineering for ML models.
```

### Key Events to Track
| Event | Use In |
| :--- | :--- |
| Session start/end | Churn prediction, engagement |
| Match result | Matchmaking quality, ELO update |
| In-game purchases | Revenue forecasting, churn |
| Deaths/respawns | DDA, frustration detection |
| Chat messages | Toxicity detection |
| Feature usage | Game design decisions |

---

## 16.7 RAG for Game Knowledge (Player Support)

### Use Case
Player asks: "How do I unlock the Dragon Armor set in Battlefield 2042?"

### Gaming-Specific RAG Challenges
```text
Challenge 1: Knowledge freshness
→ Game patches release weekly. RAG knowledge must be updated within hours.
→ Solution: Automated ingestion pipeline triggered by patch notes publication.

Challenge 2: Version-specific answers
→ The answer for v2.1 may be wrong for v2.3.
→ Solution: Metadata filter on game_version field.

Challenge 3: Disambiguation
→ "How do I get the armor?" — which armor?
→ Solution: Conversational context window. Follow-up questions if ambiguous.

Challenge 4: Multi-language
→ EA's player base is global.
→ Solution: Multilingual embedding model. Translate queries to English at retrieval.
```

---

## 16.8 AI Developer Copilot (Internal Tool)

### Use Case
Helping EA game developers write game scripts, debug Lua/C++ game code, or query internal documentation.

### Design Decisions
```text
RAG Knowledge Base:
├── Internal codebase (indexed by file + function).
├── Internal wiki / design docs.
├── Engine documentation (Frostbite, Unreal).
└── Past bug reports and their solutions.

Security Requirements:
├── Source code is IP → Self-hosted LLM (Llama 3 / Code Llama on private infra).
├── No code sent to external APIs.
└── Role-based access: Junior devs can't query senior team's confidential roadmaps.

Model Choice:
├── Code generation: CodeLlama 34B or DeepSeekCoder, self-hosted.
└── Documentation Q&A: RAG over internal docs with local embedding model.
```

---

## 16.9 Fraud Detection (In-Game Economy)

### Gaming-Specific Fraud Types
| Fraud Type | Signal | ML Approach |
| :--- | :--- | :--- |
| **Currency duplication** | Impossible balance increases | Anomaly detection (Isolation Forest) |
| **Chargeback fraud** | Purchase then dispute charge | Graph analysis of purchase patterns |
| **Account takeover** | Sudden playstyle/location change | Session anomaly detection |
| **RMT (Real Money Trading)** | Trading high-value items to mules | Graph neural network on trade graph |
| **Bot accounts** | Superhuman efficiency metrics | Behavioral biometrics |

### Key Difference from Financial Fraud
Gaming fraud involves **game economy impacts** (inflation, unfair competitive advantage), not just monetary loss. The detection model must consider both financial and in-game economic signals.

---

## 16.10 Personalization & LiveOps

### Use Case
Surface the right in-game offers (cosmetics, battle passes) to the right players at the right moment.

### Architecture (Two-Stage)
```text
Stage 1: Candidate Generation
├── Collaborative filtering (which items do similar players buy?).
├── Content-based filtering (which items match this player's aesthetic preference?).
└── Eligibility filter (which items does the player NOT already own?).

Stage 2: Ranking
├── Features: Player spending history, engagement score, item popularity.
├── Model: LightGBM ranker trained on purchase conversion rates.
└── Business rules overlay: Never surface items below player's progression level.
```

### Gaming Ethical Constraint
**Avoid dark patterns.** Personalization must not exploit vulnerable spenders (whales) with predatory offers. Many jurisdictions regulate this. Add spending caps and cool-down periods as guardrails.

---

## Engineering Checklist (Game AI)

- [ ] Have I accounted for real-time latency constraints specific to gameplay?
- [ ] Have I designed for global scale (multilingual, multi-region)?
- [ ] Have I considered false positive costs (wrongly banning a legitimate player)?
- [ ] Is the game economy impact considered alongside financial impact?
- [ ] Is the knowledge base updated with patch notes automatically?
- [ ] Are there ethical guardrails against predatory personalization?

## Interview Follow-up Questions & Best Answers

**Q: "EA has 500 million players across all titles. How does this scale affect your AI architecture?"**
*Best Answer:* "At 500M players, scale dictates architecture at every level.
1. **Data:** We're talking petabyte-scale telemetry. I'd use a lakehouse architecture (Delta Lake on S3) with tiered storage — hot features in Redis, warm in Snowflake, cold in compressed Parquet.
2. **Training:** Distributed training is a must. Player embeddings for 500M users require federated computation across regional data centers, especially with GDPR constraints on EU player data.
3. **Serving:** No single model for all 500M users. I'd shard by game title, region, and player segment. Centralized models for cross-title recommendations (EA Play subscribers), localized models per title for in-game decisions.
4. **Cost:** At this scale, every wasted GPU cycle is expensive. I'd implement aggressive semantic caching for support chatbots (same questions repeat across millions of players) and batch inference for all non-real-time features."
