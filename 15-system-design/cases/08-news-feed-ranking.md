---
module: System Design
topic: System Design
subtopic: News Feed Ranking
status: unread
tags: [productionml, ml, system-design-news-feed-rankin]
---
# News Feed Ranking System Design

End-to-end ML system for ranking a personalized social feed. Canonical system design question at Meta, LinkedIn, Twitter/X, TikTok, YouTube.

**Scale:** 3B users, feed refreshed in <200ms end-to-end, 100B+ candidate posts/day, 10M+ QPS at peak.

---

## 1. Problem Framing

### Clarifying Questions

- **Push vs pull?** Facebook/Instagram are pull (feed built on-demand when app opens). Twitter historically used push (precompute fan-out to followers). Pull is harder to serve fast but more personalized; push is cheap at read time but expensive for celebrities with 50M followers (fan-out problem).
- **What signals define "good"?** Engagement (likes/comments/shares) vs time-spent vs meaningful interactions vs self-reported satisfaction. These diverge — outrage content maximizes engagement, wellness content maximizes satisfaction.
- **Ads integration?** Ranked inline with organic content, or separately budgeted? Inline creates a single auction but risks conflating ad and organic quality.
- **Objective hierarchy?** Is wellbeing a hard floor or a soft objective? Meta's 2018 "meaningful social interaction" pivot re-weighted creator/page content down and friends content up — a policy decision, not purely ML.
- **Creator vs social graph content?** LinkedIn is mostly creator content (articles, strangers' posts); Facebook is mostly social graph (friends, family). This shapes candidate retrieval.
- **Cold start threshold?** How many interactions before personalization kicks in? Fallback: trending content, demographic priors, onboarding interest selection.
- **Real-time vs near-real-time?** Must a post from 10 seconds ago appear immediately (real-time indexing), or is 1-5 min lag fine?

### Metric Trade-offs

**Business metrics:**

| Metric | Why it matters | Risk |
|---|---|---|
| DAU/MAU ratio | Stickiness | Can be gamed via outrage |
| Time-on-platform | Revenue proxy | Filter bubbles, addiction |
| Meaningful interactions | Comments/shares > passive likes | Hard to define "meaningful" |
| Creator retention | Healthy supply side | Suppressing small creators |

**ML metrics:**

| Metric | What it measures | Limitation |
|---|---|---|
| AUC-ROC per action type | Like/comment/share prediction quality | Ignores multi-objective tension |
| NDCG@k | Ranked list quality | Needs relevance labels, hard to define |
| Online CTR/engagement rate | Proxy for ranking quality | Optimizes for clickbait |
| Wellbeing surveys (holdout) | Self-reported satisfaction | Expensive, slow signal |

**The multi-objective tension:** optimizing engagement alone (likes, comments, shares) drives outrage content and addiction loops, at odds with user wellbeing and creator monetization.

Meta's 2018 insight: passive consumption of viral video raised engagement but lowered self-reported wellbeing. The MSI pivot re-weighted replies/comments over passive views, accepting a 5-10% drop in time-on-platform to improve wellbeing — a canonical example of constrained multi-objective optimization under policy pressure.

---

## 2. Scale

| Dimension | Number |
|---|---|
| Monthly active users | 3B |
| Daily active users | ~2B |
| Feed requests/day | ~10B (~5 app opens/user/day) |
| Peak QPS | ~10M |
| Posts published/day | 500M+ |
| Candidates per feed request | 1,000-10,000 after retrieval |
| Scored posts per request | 500 (light ranker) -> 100 (heavy ranker) |
| P99 latency budget | <200ms end-to-end |
| ML scoring budget | <50ms for heavy ranker |

---

## 3. System Architecture

```
User Opens App
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    CANDIDATE GENERATION                      │
│  Social Graph Retrieval | Interest Retrieval (ANN) | Ads     │
│                    ~5,000 candidates                          │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: LIGHTWEIGHT RANKER                                  │
│  Logistic regression on sparse features, <1ms/post            │
│  Scores all 5,000, selects top 500                            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: HEAVY RANKER                                        │
│  Deep multi-task NN (MLP + attention), full feature set       │
│  Predicts like/comment/share/time-spent/hide/unfollow         │
│  ~50ms for 500 posts (batched GPU), selects top 150            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: POLICY & DIVERSITY LAYER                             │
│  Wellbeing guardrails | Diversity injection (MMR/DPP) | Integrity filters │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FEED ASSEMBLY                                                 │
│  Interleave organic + ads, inject freshness slots, serialize   │
└─────────────────────────────────────────────────────────────┘
                               ▼
                        User's Feed (~50 posts)

Supporting Infrastructure:
- Feature Store (Redis/Memcached): user embeddings, post features, social context
- Offline Pipeline: Kafka log collection -> delayed label joins -> Spark feature
  computation -> training/eval -> model registry
```

---

## 4. Multi-Stage Ranking Pipeline

### Stage 1: Candidate Retrieval

**Goal:** Reduce 500M daily posts to ~5,000 candidates in <10ms.

**Social graph traversal:** fetch friends (avg ~300) + followed pages/groups, retrieve last N posts (5-10) per source. Implementation: graph database (TAO at Meta) with per-user adjacency list cached in RAM for O(1) lookup.

**Interest-based retrieval (ANN):** user has a learned interest embedding (128-dim, updated daily); posts are embedded at publish time (CLIP for images, BERT for text). FAISS/ScaNN ANN search returns top-K semantically similar posts from non-friends. This is how Instagram's "suggested posts" work.

**Recall target:** top-50 final posts should be in the top-5,000 candidates 95%+ of the time ("oracle recall," measured offline).

| Index type | Build time | QPS | Recall@100 |
|---|---|---|---|
| Exact | O(N×d) | Low | 100% |
| HNSW | Hours | Very high | 95-99% |
| IVF-PQ | Moderate | High | 85-95% |

### Stage 2: Lightweight Ranker

**Goal:** Score 5,000 candidates, select top 500, <5ms total.

- Logistic regression or shallow MLP, 10-20 features: post age (log-scaled), source affinity, engagement rate, media type, user's historical interaction rate with the source.
- Same labels as heavy ranker but must be fast — LR inference on 5,000 posts is ~1ms on CPU.

**EdgeRank (historical reference):** Meta's original pre-deep-learning formula:
```
EdgeRank = Σ (affinity_score × edge_weight × time_decay)
```
Replaced by learned models, but the feature groups (affinity, interaction type, recency) still exist.

### Stage 3: Heavy Ranker

**Goal:** Score top 500 posts with full features, multi-task predictions.

```
Input Features (post + user + social context)
         │
Shared Bottom: Dense(512)->BN->ReLU -> Dense(256)->BN->ReLU
         + Attention over user history embeddings
         │
  ┌──────┼──────┬─────────┬──────────┐
[Like] [Comment] [Share] [Time-spent] [Hide/Unfollow]
```

**Why multi-task:** separate models per action overfit sparse actions (shares are ~10x rarer than likes). Shared bottom layers share representations; task-specific heads capture differences.

```python
total_loss = (
    w_like * bce(pred_like, label_like) +
    w_comment * bce(pred_comment, label_comment) +
    w_share * bce(pred_share, label_share) +
    w_time * mse(pred_time, label_time) +
    w_hide * bce(pred_hide, label_hide) +
    w_unfollow * bce(pred_unfollow, label_unfollow)
)
```

**Composite score:**
```python
score = (
    +3.0 * p_comment + 2.0 * p_share + 1.0 * p_like
    +0.5 * p_time_spent_percentile
    -2.0 * p_hide - 3.0 * p_unfollow - 5.0 * p_report
)
```
Weights tuned via online experimentation. Negative signals (hide, unfollow) carry heavy penalties — they signal regret, not passive disengagement.

### Stage 4: Policy and Diversity Layer

Rule-based, non-ML adjustments applied post-ranking:

- **Wellbeing guardrails:** cap consecutive video posts (e.g., 3), demote "emotionally negative"-flagged posts.
- **Diversity injection:** no more than 2 consecutive posts from one creator, at least 3 topic clusters in top-10 (see Section 8).
- **Integrity filters:** misinformation under review -> penalty/removal; CSAM/hate speech -> hard remove; spam -> demotion.
- **Boost rules:** freshness boost for posts <2 hours old; occasional "on this day" nostalgia posts.

---

## 5. Feature Engineering

### Post Features

| Feature | Description | Update freq |
|---|---|---|
| `post_age_log` | log(seconds since publish) | Per request |
| `media_type` | Video/image/link/text | At publish |
| `engagement_rate_1h` | (likes+comments)/impressions | Streaming |
| `engagement_velocity` | rate of change of engagement | Streaming |
| `content_embedding` | 128-dim (CLIP/BERT) | At publish |
| `text_sentiment` | Positive/neutral/negative/outrage | At publish |
| `is_reshare` | Original vs reshared | At publish |
| `hashtag_trending` | Trending in last hour? | Hourly |
| `creator_follower_count_log` | log(followers) | Daily |
| `creator_avg_engagement_rate_7d` | Historical engagement | Daily |

Temporal decay is learned, not hardcoded:
```python
age_hours = (now - post_timestamp) / 3600
decay_feature = np.log1p(age_hours)  # model learns the weighting
```

### User Features

| Feature | Description | Update freq |
|---|---|---|
| `user_interest_embedding` | 128-dim learned vector | Daily |
| `user_engagement_rate_7d` | Avg engagement rate | Daily |
| `session_length_so_far` | Posts scrolled this session | Per request |
| `time_of_day_bucket` | Morning/afternoon/evening/night | Per request |
| `device_type` | Mobile/tablet/desktop | Per session |
| `social_graph_centrality` | Influence in social graph | Weekly |
| `content_format_preference` | Video/image/text engagement mix | Daily |
| `following_count_log` | log(following count) | Daily |

### Social Context Features

Among the strongest signals — encode implicit social proof:

| Feature | Description |
|---|---|
| `mutual_friends_engaged` | Count of mutual friends who engaged |
| `creator_affinity_score` | Learned user-creator pair score |
| `close_friend_engaged` | Did a close friend engage? |
| `group_membership_relevance` | Post from an active group? |
| `viewer_to_creator_distance` | Social graph hops |

**Creator affinity** is key: a collaborative-filtering-style embedding trained on user-creator interaction history, so users get high affinity even for a creator's brand-new, zero-engagement posts.

```python
affinity = np.dot(user_embedding, creator_embedding) / (
    np.linalg.norm(user_embedding) * np.linalg.norm(creator_embedding)
)
```

---

## 6. Multi-Objective Optimization

### Why Single-Objective Fails

Likes alone -> clickbait/outrage/low-effort memes win. Time-spent alone -> passive scroll-traps win. Comments alone -> controversial/political content wins.

### Combining Signals

**Weighted linear combination** (baseline): `score = Σ wᵢ × pᵢ`, weights set by product policy and tuned via A/B test on long-horizon metrics.

**Pareto optimization** (advanced): find the non-dominated frontier across objectives (engagement, wellbeing, creator reach); policy chooses the operating point.

**Constrained optimization with wellbeing floor:**
```
maximize: engagement_score(post)
subject to: wellbeing_score(feed) ≥ θ,  creator_diversity(feed) ≥ δ
```
Implemented as a post-ranking adjustment: if top-K posts push predicted wellbeing below θ, swap outrage posts for wellbeing-friendly ones.

### Meta's MSI Pivot (2018)

Prioritized "meaningful social interactions" over passive content: comments/shares re-weighted 5-10x vs likes, friends/family boosted over pages/creators, video specifically penalized, with a "meaningful" classifier trained on user surveys. Result: time-on-platform dropped ~5%, but engagement-per-minute and satisfaction improved.

### Inverse Signals (Negative Feedback)

| Signal | Weight | Indicates |
|---|---|---|
| Hide post | -2x | Unwanted |
| Snooze 30 days | -3x | Source fatigue |
| Unfollow | -5x | Permanent rejection |
| Report | -10x | Policy violation |
| Survey "not worth time" | -3x | Regret |

Negative signals are sparse but high-value: a post 1% of viewers hide should be treated very differently from one no one hides, even at equal like rate.

---

## 7. Creator Ecosystem

### Friends vs Creator Content

Two content types: friends/family (high personal relevance, low production quality) vs creator/page (low relevance, high quality). Friends content typically loses on raw engagement rate, so naive engagement ranking suppresses it. Meta uses separate ranking tracks blended at assembly time:

```
Feed composition target (example):
  60% social graph | 20% followed creators | 15% suggested (ANN) | 5% ads
```
These percentages are policy parameters, not ML outputs.

### Creator Reach Fairness

Without correction, top 0.1% of creators get 50%+ of impressions, creating a flywheel that entrenches their advantage.

**Mitigations:**
- Cap any single creator at N% of a user's daily feed slots
- Exploration budget: 10-15% of slots reserved for creators the user hasn't interacted with
- Small-creator boost for accounts <10K followers with strong relative engagement

### Virality Prediction

Surfacing a soon-to-be-viral post early is valuable. Signals: engagement velocity, cross-network spread, early commenter diversity (diverse engagers suggest organic virality vs. coordinated behavior).

```python
def virality_score(post_id, window_minutes=30):
    early = get_engagements(post_id, minutes=window_minutes)
    velocity = len(early) / window_minutes
    diversity = len(set(user_community[u] for u in early)) / len(early)
    return velocity * diversity
```

---

## 8. Diversity and Anti-Filter-Bubble

### Intra-List Diversity

A feed of 50 same-topic posts is worse than a varied one, even if each post is individually relevant: topic diversity (K clusters in top-N), creator diversity (no more than M consecutive from one creator), format diversity (video/image/text mix).

### DPP / Greedy MMR

DPPs trade off quality vs diversity: `P(S) ∝ det(L_S)`, where L encodes both quality and similarity — high-quality, dissimilar sets score highest.

Practical approximation, Greedy MMR:
```python
def greedy_mmr(candidates, lambda_param=0.5, k=50):
    # candidates: (post_id, score, embedding); lambda: relevance vs diversity tradeoff
    selected = [max(candidates, key=lambda x: x[1])]
    remaining = [c for c in candidates if c not in selected]
    while len(selected) < k and remaining:
        def mmr_score(c):
            max_sim = max(cosine_similarity(c[2], s[2]) for s in selected)
            return lambda_param * c[1] - (1 - lambda_param) * max_sim
        next_item = max(remaining, key=mmr_score)
        selected.append(next_item)
        remaining.remove(next_item)
    return selected
```

### Breaking Filter Bubbles

Pure relevance ranking reinforces engagement with existing interests, contributing to polarization.

**Interventions:** exploration budget (reserve slots outside historical interest clusters), cross-partisan exposure on political content, periodic interest-graph perturbation.

**Measurement tension:** breaking filter bubbles often costs short-term engagement. Standard A/B tests capture that cost but not the wellbeing/diversity benefit — need long-horizon surveys.

---

## 9. Feedback Loops

### Popularity Bias Amplification

Highly-ranked posts get more impressions -> more engagement -> higher future ranking, a rich-get-richer dynamic.

**Correction — inverse propensity scoring (IPS):** weight training examples by inverse show-probability, so a post shown to 1M users at 1% engagement isn't over-trusted relative to one shown to 1K users at the same rate.
```python
loss = bce(pred, label) / propensity_score  # propensity = P(item shown | features)
```

### Echo Chambers and Radicalization

Engagement optimization can push users toward progressively more extreme content (YouTube's 2019 changes addressed this directly).

**Detection:** track per-user "content trajectory" (is avg extremism score rising?); flag consistent drift.
**Mitigation:** hard score penalty above an extremism threshold; "rabbit hole" detection injects diverse content after N consecutive same-topic posts.

### Engagement-Wellbeing Divergence

Engagement metrics are instant; wellbeing metrics need multi-week surveys. Proxy signals: next-day return rate, session-end behavior, post-session "was this worth it?" surveys, passive-consumption ratio.

---

## 10. Online Experimentation

### A/B Testing Feed Ranking

**Randomization unit:** user (feed changes affect all of a user's sessions).

**Social network interference:** User A's ranking can be affected by User B's experiment arm through B's posts and engagement — violates SUTVA.

**Mitigation:** ego-cluster randomization (assign whole social neighborhoods to one arm — reduces power but reduces interference); for distribution experiments, randomize at the creator level instead.

### Long-Term vs Short-Term Effects

Short A/B tests (1-2 weeks) miss: novelty effects (fade over time), content ecosystem effects (creators post less if reach drops — takes months to show), and user behavior adaptation.

**Holdout groups:** keep ~1% of users on the old model for 3-6 months to catch slow-moving effects on diversity, wellbeing, and churn.

### Ad Interference in Organic Experiments

Organic ranking changes shift which ad slots are seen, confounding ad revenue. Mitigations: run organic/ads experiments on disjoint user sets, treat ad metrics as guardrails in organic experiments, use revenue-neutral ad insertion.

### Metric Hierarchy

```
Guardrails (invalidate experiment if degraded): feed latency P99, crash rate, ad revenue/DAU
Primary: meaningful interactions rate, 7-day return rate
Secondary: likes/session, time-on-platform, negative feedback rate
```

---

## 11. Failure Modes

### Misinformation Amplification

**Mechanism:** false/misleading posts often drive high short-term engagement (outrage, debate) and spread before fact-checkers act.
**Signature:** high engagement velocity + high report rate from a subset of users.
**Mitigation:** real-time misinformation classifier trained on fact-check labels; flagged posts get reduced distribution (not necessarily removal).

### Outrage Optimization

**Mechanism:** comments/shares (weighted as "meaningful") are disproportionately driven by anger and controversy.
**Signature:** top posts trend political/provocative over time; satisfaction declines while engagement rises.
**Mitigation:** sentiment-aware re-weighting — discount comment value by an anger-fraction classifier.

### Creator Cold Start

**Mechanism:** no engagement history -> low affinity -> low ranking -> no data -> perpetual cold start.
**Signature:** new creator cohorts get near-zero reach even with quality content.
**Mitigation:** exploration budget (show new-creator posts to a small random sample to gather signal); bootstrap creator embeddings from content embeddings.

### New User Cold Start

**Mechanism:** no history, no interest embedding -> defaults to generic trending content -> poor match, early churn.
**Signature:** Day-7 retention much lower for new users than tenured ones.
**Mitigation:** onboarding interest selection, demographic priors, near-real-time embedding updates from first-session interactions, a "new user" ranker that upweights diversity/quality over personalization until ~50+ interactions accumulate.

### Model Feedback Loop Collapse

**Mechanism:** model shapes what's shown -> engagement data shifts -> next model trains on biased data -> converges to a narrow content set.
**Signature:** served content diversity drops over time; feature distributions drift from training (covariate shift).
**Mitigation:** counterfactual logging + importance-weighted training; exploration policy serving some non-top items; periodic retraining including exploration data.

---

## Interview Angles

### Q: Engagement optimization often conflicts with user wellbeing — how do you handle it? [Hard]
A: Constrained multi-objective optimization: treat wellbeing as a floor constraint, not something to maximize. Compute a predicted wellbeing score (proxy signals: satisfaction surveys, hide/unfollow regret signals, long-term return rate) and require it stay above threshold θ while maximizing engagement within that constraint. Since wellbeing labels lag (survey-based, weeks) while engagement is instant, use long-horizon holdout groups (~1% permanently in control) to catch slow wellbeing effects short A/B tests miss. Meta's 2018 MSI pivot was effectively moving that constraint boundary — accepting lower engagement for a higher wellbeing floor.

**Cross-questions to expect:**

- *Your wellbeing proxy is survey-based and lags by weeks. Engagement is instant. Doesn't the model just learn to game engagement in the weeks before the constraint can react?* -> Yes -- that's the core danger. A fast objective under a slow constraint means the system can drift far past the floor before any survey catches it, and by then the damaged cohort has already churned or habituated. You mitigate with a permanent long-horizon holdout and with faster leading proxies (hide/unfollow/regret clicks) that correlate with the slow label, but you should be explicit that the constraint is always reacting to stale evidence.
- *Meta's MSI pivot moved the constraint boundary. How would you know, six months later, whether that trade was actually worth it?* -> You can't read it off short A/B tests -- the whole point is that the benefit is long-horizon retention and the cost is near-term engagement. You need the permanent holdout and a pre-registered long-term metric, otherwise every quarter's incentive is to quietly relax the floor back toward engagement.

**Trap:** framing wellbeing as another term in a weighted sum. A weighted objective lets a large engagement gain buy out a wellbeing loss; a floor constraint refuses the trade below the threshold. The distinction is the entire ethical content of the design.
**Q: How do you design candidate retrieval for a user with 5,000 friends and 2,000 followed pages in under 10ms?**
A: Parallelize retrieval across sources — graph traversal and ANN interest retrieval run concurrently. Keep adjacency lists and post indexes in memory (TAO/Redis) for fast fan-out. For very large graphs, only pull from "active" connections (posted in the last 7 days) instead of the full friend list. Pre-compute candidate sets off-peak and cache them, applying lightweight freshness updates on read. The 10ms budget covers cache hits; cache misses (new/cold-start users) use a separate slower path.

**Q: How do you detect and prevent popularity bias in training data?**
A: Items shown more often accumulate more engagement data, which makes the model favor them further — a feedback loop. Fix with inverse propensity scoring: weight training examples by the inverse probability of being shown, so frequently-shown items don't dominate the gradient. This requires logging the score/rank at impression time. Also maintain an exploration budget (ε-greedy or Thompson sampling) to gather unbiased signal on rarely-shown items.

### Q: A competitor launches and engagement drops 15%. Model problem or product problem? [Medium]
A: Decompose the drop. Is it uniform across segments, or concentrated (competitive threats usually hit specific demographics/power users first, model regressions are more uniform)? Check engagement rate per session (model quality) vs. session frequency/length (product/competitive). Replay the model on historical data to check for input feature drift. Check content supply (are creators posting less) vs. demand (same posts getting less engagement). Check negative signals: rising hide/unfollow/report suggests a model issue; flat negative signals with lower impressions suggests a retention/acquisition issue.

**Cross-questions to expect:**

- *You say model regressions are uniform and competitive threats are concentrated. Give a case where that heuristic misleads you.* -> A model regression tied to a feature that only exists for a subpopulation (e.g. a video-embedding bug) is *concentrated*, and a competitor that pulls broadly (a platform-wide TikTok-style shift) is *uniform*. The uniform/concentrated split is a useful prior, not a proof -- confirm with a model replay on historical data, which isolates model quality from the changed environment.
- *Negative signals (hide/unfollow/report) are flat but impressions are down. You concluded "retention, not model." What confound could break that?* -> If the competitor peeled off your most engaged power users first, the *remaining* population has structurally lower negative-signal rates, so flat negatives can coexist with a real quality problem for the users you kept. Segment the negative-signal trend by cohort tenure before trusting the aggregate.

**Trap:** running a single model replay and calling it settled. Replay tells you whether *inputs* drifted; it cannot see a demand-side shift where the same content simply gets less engagement because attention moved off-platform.
**Q: How does ads ranking interact with the organic feed ranker, and what can go wrong?**
A: Two models: unified auction (organic and ads get one quality score; combined score = predicted_value × bid, ad wins only if revenue exceeds organic opportunity cost) or fixed insertion (ads at every Nth slot regardless of organic quality). Unified is more efficient but complex. Failure modes: better organic ranking raises the opportunity cost of ad slots, so fewer ads clear the auction and revenue drops unexpectedly (track revenue as a guardrail in organic experiments); shared features between ad and organic models mean retraining one can silently shift the other; heavy ad density causes banner blindness that biases organic engagement labels too.

**Q: How would you measure filter bubble effects in the feed?**
A: Content-side: topic diversity index (avg pairwise embedding distance of weekly consumption), source diversity (distinct creators consumed), viewpoint diversity on political content. Trajectory: is a user's average content extremism/partisanship score drifting over time? Survey: periodic "did you see content that challenged your views this week?" Counterfactual: compare actual feed diversity to a diversity-maximizing baseline. Content diversity responds to MMR/DPP reranking (5-10% engagement cost); viewpoint diversity needs explicit cross-partisan injection with user consent.

### Q: Design fair ranking so small creators can get discovered, not just large ones. [Hard]
A: The core problem is a rich-get-richer loop: history -> affinity -> ranking -> impressions -> more history. Four levers: (1) exploration budget — reserve 10-15% of slots for creators the user hasn't interacted with, ranked within-bucket by engagement relative to peer creators of similar size; (2) propensity correction in training so a 10K-follower creator's 1% engagement counts equally to a 100M-follower creator's 1% engagement; (3) audience saturation penalty for repeatedly showing the same creator to the same user; (4) bootstrap new creators via content embeddings (CLIP/BERT) matched to users who engaged with similar content, sidestepping cold start entirely.

**Cross-questions to expect:**

- *You reserve 10-15% of slots for unexplored creators. What's the direct cost, and who pays it?* -> The cost is short-term engagement -- explored slots are, by construction, lower expected-value than exploited ones, so the user pays in feed quality now for creator-ecosystem health later. If you don't measure and cap that cost per user, the exploration budget silently becomes a tax the most engaged users resent, and they're the ones who'll notice.
- *Propensity correction makes a 10K-follower creator's 1% engagement count like a 100M creator's 1%. Doesn't that over-promote genuinely mediocre small creators?* -> It can. Equalizing for exposure removes the rich-get-richer bias but also removes a real signal -- large creators are sometimes large because they're good. The honest framing is that you're trading some precision for ecosystem diversity, and the right exploration rate is the one where the long-term supply-side gain (more creators staying) exceeds the demand-side engagement loss.

**Trap:** treating creator fairness as a pure win. Every lever here (exploration slots, propensity reweighting, saturation penalties) costs measurable near-term engagement -- pretending otherwise is how these systems get quietly rolled back the first bad quarter.
## Flashcards

**Why does optimizing pure engagement (likes/comments/shares) predictably degrade a feed over time?** #flashcard
Each single-metric objective has an exploit: likes alone favors low-effort/clickbait content, time-spent alone favors passive scroll-traps, and comments alone favors controversial/outrage content — none of these correlate with what users actually report as satisfying, which is why Meta's 2018 MSI pivot deliberately traded ~5% time-on-platform for higher self-reported wellbeing.

**Why does the ranking pipeline use a lightweight ranker before the heavy multi-task ranker instead of scoring all candidates with the full model?** #flashcard
Scoring 5,000 candidates with a deep multi-task network at full cost wouldn't fit the <200ms budget; a cheap logistic-regression pass (~1ms for all 5,000) filters down to ~500, so the expensive model only runs on posts likely to matter.

**Why is the heavy ranker multi-task (predicting like/comment/share/time/hide/unfollow jointly) instead of one model per action?** #flashcard
Rare actions like shares (~10x rarer than likes) don't have enough data to train a strong standalone model; a shared-bottom architecture lets common representations transfer across tasks while task-specific heads still capture per-action differences.

**Why do negative signals (hide, unfollow, report) get weighted far more heavily than positive signals in the composite score?** #flashcard
Negative signals are sparse but carry much higher information content — a post that 1% of viewers hide indicates active regret, not mere disinterest, and is a stronger quality signal than an equivalent like rate, so unfollow/report are weighted 5-10x a like.

**Why is creator affinity computed as a learned embedding dot product rather than raw historical engagement rate?** #flashcard
Raw engagement rate can't score a creator's brand-new post (zero historical engagement yet); a collaborative-filtering-style embedding trained on the user-creator interaction history generalizes to new content from a creator the user already likes, solving per-post cold start.

**Why does naive engagement-based ranking systematically suppress friends/family content relative to professional creator content?** #flashcard
Friends' posts are high in personal relevance but low in production quality, so they lose on raw engagement rate against polished creator content — platforms fix this by ranking friends and creator content on separate tracks and blending them at assembly time using policy-set percentages, not a single unified score.

**Why does A/B testing feed ranking changes violate the standard independence assumption (SUTVA)?** #flashcard
User A's feed experience depends on User B's engagement and posts, and if A and B are in different experiment arms, B's behavior under their arm leaks into A's outcomes through the shared social graph — mitigated by ego-cluster randomization (assigning whole social neighborhoods to one arm) at the cost of statistical power.

**Why do short 1-2 week A/B tests systematically miss the real effect of ranking changes?** #flashcard
Novelty effects fade only after weeks, and content ecosystem effects (creators posting less if their reach drops) take months to materialize — both invisible in a short window, requiring long-running holdout groups (~1% of users for 3-6 months) to catch slow-moving degradation.

**Why does popularity bias in training data create a self-reinforcing feedback loop, and how is it corrected?** #flashcard
Posts shown more often accumulate more engagement data purely from exposure volume, making the model favor them even more in the next training round regardless of true quality; inverse propensity scoring weights training examples by 1/P(shown), preventing frequently-shown posts from dominating the gradient.

**Why is an explicit exploration budget necessary even after diversity-aware reranking (MMR/DPP)?** #flashcard
MMR/DPP only diversify among already-retrieved, already-scored candidates — they can't surface a new creator or topic cluster that never made it into the candidate set in the first place. A dedicated exploration budget (reserving slots for unshown creators/topics) is required to break the cold-start and popularity-bias loop at its source.
