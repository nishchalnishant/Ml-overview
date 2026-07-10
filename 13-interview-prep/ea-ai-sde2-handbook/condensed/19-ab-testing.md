# Interview 19 — A/B Testing & Experimentation Platform (Condensed)

Design a scalable platform to route players between matchmaking Algorithm A (current) and Algorithm B (new), track telemetry, and prove statistical significance without bias. Core challenge: this is a matchmaking system, so naive per-user bucketing breaks the player pool (interference/SUTVA).

---

## Clarifying Questions to Ask

- **Multiplayer interference?** → Bucketing by user splits the matchmaking pool; must bucket by Region/Time instead (switchback).
- **What's the OEC (primary metric)?** → Avg wait time is primary; Match Imbalance Score is the guardrail.
- **Is assignment sticky across sessions?** → Yes — same user must stay in same bucket on re-login (deterministic hash).
- **Test duration / sample size fixed upfront?** → Yes, fixed horizon to avoid peeking bias.
- **Any novelty effect concern?** → Yes, new algorithm may get a temporary behavior bump — must run long enough to wash out.

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

## Toughest Follow-ups

**Q: 10 experiments run simultaneously — how do you prevent them from colliding/biasing each other?**
A: Use orthogonal hashing — salt the hash with `experiment_id` (`hash(user_id + exp_id)`). Different salts mean a user's bucket in one experiment is independent of their bucket in another, so across millions of users the overlaps are statistically orthogonal.

**Q: A huge Twitch streamer drives 100k new users who all land in Bucket B by chance — how do you detect/fix the skew?**
A: Detect via Sample Ratio Mismatch alerting (expected 50/50 vs actual). Fix telemetry with Winsorization (cap outliers at 99th percentile) and switch to a non-parametric test (Mann-Whitney U) so results aren't driven by extreme values.

**Q: T-tests don't work for churn (30-day binary survival metric) — what do you use instead?**
A: Treat churn as time-to-event data — use Kaplan-Meier estimator for survival curves per bucket, and the Log-Rank test to check if the difference in survival is statistically significant.

---

## Biggest Pitfall

Missing the SUTVA/interference problem entirely (or proposing per-user bucketing for a matchmaking system without flagging it) — this single miss invalidates the whole experiment design and is the top reason candidates drop to No Hire.
