# Dynamic In-Game Store Pricing & Promotions — Cheat Sheet

Ultimate Team wants personalized discount coupons instead of static pack prices, to lift revenue without players feeling cheated. Design an ML system that picks the optimal discount per player.

## Clarifying Questions to Ask
- Base price change or targeted coupons? → Coupons only; varying base price = legal/PR disaster.
- What currency? → Premium virtual currency (bought with real money).
- Optimize immediate conversion or LTV? → 30-day LTV — avoid discounts that cannibalize the rest of the month's spend.
- Classification or bandit/RL framing? → They want you to propose the optimal-discount (bandit) framing yourself.
- Historical data available? → Yes, 1-month random A/B test, discounts 0–50%.
- How often do prices refresh? → Daily batch, not real-time.

## Core Architecture
- Historical randomized A/B data → feature store (Snowflake).
- Train one regressor per discount arm (0%, 10%, 25%, 50%) predicting 30-day revenue — **XGBoost regressors**, chosen for tabular efficiency and easy per-arm isolation.
- Daily Airflow batch job: score every player against all arms, pick argmax (**contextual bandit**, epsilon-greedy exploration ~10%).
- Write `player_id → discount_tier` to Redis (TTL 24h).
- Game Store API reads Redis at login, <5ms; falls back to 0% discount if Redis is down/missing key.
- Log assignments (including exploration draws) back to warehouse to retrain next week — this closes the feedback loop and prevents bias collapse.

## Talking Points That Signal Seniority
- Proactively flags that varying base price across players is a legal/price-discrimination risk — pivots to coupon targeting or item targeting instead.
- Distinguishes predicting absolute LTV vs. incremental/uplift LTV — calls out that a "sure thing" whale should never get a discount just because their predicted revenue is high under every arm.
- Names Uplift Modeling (T-/X-learners, EconML/CausalML) as the principled production upgrade over naive per-arm regression.
- Explains why epsilon-greedy exploration is mandatory — without a random holdout, the model only ever sees data for the arm it already prefers and the training signal collapses.
- Proposes a surrogate/proxy metric (e.g., 3-day behavior → predicted 30-day LTV) to shorten the feedback loop instead of waiting a full month to retrain.
- Adds a business-rule guard rail before the Redis push (e.g., cap % of DAU receiving the max discount) to bound blast radius of a bad model.
- Mentions calibration monitoring — predicted vs. actual revenue — since a miscalibrated regressor silently erodes margin.
- Suggests Thompson Sampling as the natural evolution from epsilon-greedy for smarter exploration/exploitation tradeoff.

## Top 3 Tradeoffs
- **Contextual bandit vs. uplift modeling** — bandits are simple and explore continuously; uplift modeling directly targets causal incremental revenue but is harder to implement/evaluate. Matters because bandits alone can misfire on whales.
- **Per-arm models vs. single model with discount as a feature** — per-arm isolates signal cleanly but doesn't scale past a handful of arms; single model scales better but risks the discount feature getting drowned out by dominant features like playtime.
- **Epsilon-greedy vs. Thompson Sampling** — epsilon-greedy is trivial to implement but wastes a fixed fraction of traffic on pure randomness; Thompson Sampling explores more efficiently via posterior sampling but is materially harder to bolt onto XGBoost.

## Toughest Follow-ups
**"A whale looks high-LTV under every discount arm — how do you stop the model handing them a 50% discount anyway?"**
Predict incremental revenue, not absolute revenue: uplift = E[Rev|treatment] − E[Rev|control]. If uplift is negative or near zero, don't treat, even if absolute predicted revenue is high everywhere. This is exactly what uplift modeling is for.

**"Legal says no two players can see different prices for the same item — pivot the design."**
Keep item price identical for everyone; move personalization to *which item* gets featured in the "Just for You" slot. Same bandit machinery, but the arms become item choices instead of discount tiers.

**"30-day LTV means a 30-day wait for labels — how do you retrain weekly?"**
Train a surrogate model that maps a short window (e.g., 3 days) of post-promo behavior to predicted 30-day LTV, and use that prediction as the reward signal for the bandit, closing the loop in days instead of a month.

## Biggest Pitfall
Framing this as a binary "will they buy?" classifier instead of a bandit/RL revenue-optimization problem — it maximizes conversion while quietly destroying margin, and is the single fastest way to drop from Hire to No Hire.
