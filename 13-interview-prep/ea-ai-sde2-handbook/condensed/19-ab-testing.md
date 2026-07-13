# Interview 19 — A/B Testing & Experimentation Platform (Condensed)

Design a scalable platform to route players between matchmaking Algorithm A (current) and Algorithm B (new), track telemetry, and prove statistical significance without bias. Core challenge: this is a matchmaking system, so naive per-user bucketing breaks the player pool (interference/SUTVA).

---

## Core Architecture

```
Game Client → Experiment Assignment API (hash(player_id+exp_id) % 100)
           → Matchmaking Service (routed to Alg A / B)
           → Kafka → Snowflake (telemetry: bucket, wait_time)
           → Nightly Statistical Engine (T-test / Welch's, p-value, CI)
```

- **Assignment:** stateless deterministic hashing (MurmurHash/MD5 of `user_id + experiment_salt`), no DB lookup needed, embedded as SDK in game/matchmaking service for near-zero latency.
- **Interference fix:** switchback testing — bucket by Region+Time slice, not by user, to keep matchmaking pools intact.
- **Config delivery:** experiment config (rollout %, status) pushed via Redis/AppConfig, polled by SDK every few minutes.
- **Analysis engine:** batch job computing Welch's T-test, p-value, confidence intervals; pushed down to Spark/Snowflake SQL at scale (not raw Scipy in-memory).
- **Why this technique:** hashing gives O(1) stateless routing; switchback avoids SUTVA violations that would invalidate the whole experiment.

---

## Talking Points That Signal Seniority

- Proactively flags SUTVA/network-interference risk for matchmaking before being asked.
- Proposes switchback (time/region) or cluster (guild/party) randomization instead of naive per-user bucketing.
- Mentions Bonferroni correction when multiple metrics are evaluated simultaneously.
- Brings up CUPED for variance reduction to shorten time-to-significance.
- Flags the "peeking problem" and argues for a fixed sample size / hidden p-value until test completes.
- Notes Sample Ratio Mismatch (SRM) as the first health-check alert on any experiment.
- Suggests Multi-Armed Bandit (Thompson Sampling) as an alternative to fixed-split when minimizing regret matters.
- Recognizes that large-scale stats must be pushed into the data warehouse (Spark SQL/Snowflake), not pulled into Python memory.

---

## Top 3 Tradeoffs

- **User bucketing vs. switchback bucketing** — user bucketing enables long-term per-player retention tracking but destroys matchmaking liquidity; switchback preserves the pool but can't isolate individual long-term effects.
- **Frequentist (T-test) vs. Bayesian** — frequentist is SQL-friendly and industry standard; Bayesian gives more intuitive "95% chance B is better" output for PMs but needs MCMC/conjugate priors.
- **Fixed-horizon A/B vs. Multi-Armed Bandit** — fixed A/B gives clean statistical guarantees; bandits reduce regret by shifting traffic early but complicate clean significance testing.

---

## Biggest Pitfall

Missing the SUTVA/interference problem entirely (or proposing per-user bucketing for a matchmaking system without flagging it) — this single miss invalidates the whole experiment design and is the top reason candidates drop to No Hire.
