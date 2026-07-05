---
module: Production Ml
topic: System Design
subtopic: News Feed Ranking
status: unread
tags: [productionml, ml, system-design-news-feed-rankin]
---
# News Feed Ranking System Design

End-to-end ML system for ranking a personalized social media feed. Canonical system design question at Meta, LinkedIn, Twitter/X, TikTok, and YouTube.

**Scale:** 3B users, personalized feed refreshed in <200ms end-to-end, 100B+ candidate posts/day, 10M+ QPS at peak.

---

## 1. Problem Framing

### Clarifying Questions

- **Consumption model: push vs pull?** Facebook/Instagram are pull (user opens app → feed constructed on-demand). Twitter historically used push (precompute fan-out to followers). Pull is harder to serve fast but more personalized; push is cheaper at read time but expensive for celebrities with 50M followers (fan-out problem).
- **What signals define "good"?** Engagement (likes, comments, shares) vs time-spent vs meaningful interactions vs self-reported satisfaction. These diverge — outrage content maximizes engagement, wellness content maximizes satisfaction.
- **Ads integration?** Are ads ranked inline with organic content or separately budgeted? Inline ranking creates a single auction but risks ad-organic quality conflation.
- **Objective hierarchy?** Is wellbeing a hard constraint (floor) or a soft objective in the optimization? Meta's 2018 "meaningful social interaction" pivot was a policy decision to re-weight creator/page content down and friends content up — not purely ML.
- **Creator vs social graph content?** LinkedIn is mostly creator content (articles, posts from strangers); Facebook is mostly social graph (friends, family). This determines candidate retrieval strategy.
- **New user cold start threshold?** How many interactions before the personalization kicks in? What's the fallback: trending content, demographic-based priors, onboarding interest selection?
- **Real-time vs near-real-time?** Should posts published 10 seconds ago appear in the feed immediately (requires real-time indexing) or is 1–5 minute lag acceptable?

### Metric Trade-offs

**Business metrics (north star):**

| Metric | Why it matters | Risk |
|---|---|---|
| DAU / MAU ratio | Stickiness; people returning daily | Maximizing it via outrage |
| Time-on-platform | Revenue proxy (more ads shown) | Filter bubbles, addiction |
| Meaningful interactions | Comments and shares > passive likes | Hard to measure "meaningful" |
| Creator retention | Healthy supply side | Suppressing small creators |

**ML metrics:**

| Metric | What it measures | Limitation |
|---|---|---|
| AUC-ROC per action type | Like/comment/share prediction quality | Doesn't reflect multi-objective tension |
| NDCG@k | Ranked list quality | Requires relevance labels, hard to define |
| Online CTR / engagement rate | Proxy for ranking quality | Optimizes for clickbait |
| Hold-out group wellbeing surveys | User-reported satisfaction | Expensive, slow signal |

**The multi-objective tension:**

```
engagement (likes, comments, shares)
        ↑
        │  ← optimizing this alone → outrage content, addiction loops
        │
user wellbeing ──────────────────────── creator monetization
        ↑                                        ↑
 (minimizing regret,              (reach for pages, ad revenue)
  screen time, outrage)
```

Meta's 2018 insight: passive consumption of viral video content increased engagement metrics but decreased self-reported wellbeing. The MSI (meaningful social interaction) pivot re-weighted content that generated replies and comments over content that generated passive views — accepting a 5–10% reduction in time-on-platform to improve wellbeing metrics. This is a canonical example of constrained multi-objective optimization under policy pressure.

---

## 2. Scale

| Dimension | Number |
|---|---|
| Monthly active users | 3B |
| Daily active users | ~2B |
| Feed requests/day | ~10B (user opens app ~5x/day) |
| Peak QPS | ~10M |
| Posts published/day | 500M+ (friends posts, pages, ads) |
| Candidate posts per feed request | 1,000–10,000 after retrieval |
| Scored posts per request | 500 (lightweight) → 100 (heavy ranker) |
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
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Social Graph   │  │  Interest    │  │  Ads          │  │
│  │  Retrieval      │  │  Retrieval   │  │  Auction      │  │
│  │  (friends,      │  │  (ANN on     │  │  (targeting   │  │
│  │   groups,       │  │   user       │  │   + budget    │  │
│  │   pages)        │  │   embedding) │  │   pacing)     │  │
│  └────────┬────────┘  └──────┬───────┘  └───────┬───────┘  │
│           └──────────────────┴──────────────────┘           │
│                              │                              │
│                    ~5,000 candidates                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 2: LIGHTWEIGHT RANKER                    │
│                                                             │
│  Logistic regression / linear model on sparse features      │
│  <1ms per post, score all 5,000 candidates                  │
│  Select top 500 for heavy ranker                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                STAGE 3: HEAVY RANKER                         │
│                                                             │
│  Deep neural network (MLP + attention on embeddings)        │
│  Full feature set: post + user + social context features    │
│  Multi-task: predict like, comment, share, time-spent,      │
│              hide, unfollow, report simultaneously          │
│  ~50ms for 500 posts (batched GPU inference)                │
│  Select top 150 by composite score                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 4: POLICY & DIVERSITY LAYER             │
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  Wellbeing     │  │  Diversity     │  │  Integrity   │  │
│  │  Guardrails    │  │  Injection     │  │  Filters     │  │
│  │  (cap          │  │  (DPP or       │  │  (misinfo,   │  │
│  │   outrage,     │  │  greedy MMR)   │  │   CSAM,      │  │
│  │   hate speech  │  │               │  │   spam)      │  │
│  │   demotion)    │  │               │  │              │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FEED ASSEMBLY                           │
│                                                             │
│  Interleave organic + ads by insertion rules                │
│  Inject "new content" slots (posts < 2 hours old)           │
│  Apply user-specific settings (following-only mode, etc.)   │
│  Serialize to client response                               │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                        User's Feed (~50 posts)


Supporting Infrastructure:
┌──────────────────────────────────────────────────────────┐
│  Feature Store (Redis/Memcached)                          │
│  ├── User features: interest embeddings, engagement hist │
│  ├── Post features: engagement counts, age, embeddings   │
│  └── Social context: mutual engagement signals           │
├──────────────────────────────────────────────────────────┤
│  Offline Training Pipeline                                │
│  ├── Log collection (Kafka → data lake)                  │
│  ├── Label generation (delayed feedback joins)           │
│  ├── Feature computation (Spark batch + streaming)       │
│  └── Model training + evaluation → model registry        │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Multi-Stage Ranking Pipeline

### Stage 1: Candidate Retrieval

**Goal:** Reduce 500M daily posts to ~5,000 candidates in <10ms.

**Social graph traversal:**
- Fetch all friends (avg ~300 on Facebook) + followed pages/groups
- For each source, retrieve last N posts (N=5–10)
- Implementation: graph database (TAO at Meta) → per-user adjacency list cached in RAM → O(1) lookup

**Interest-based retrieval (ANN search):**
- User has a learned interest embedding (128-dim, updated daily)
- Posts are embedded at publish time (CLIP for images, BERT for text)
- FAISS / ScaNN index on post embeddings → ANN search returns top-K semantically similar posts from non-friends
- This is how Instagram's "suggested posts" and LinkedIn's "posts you might like" work

**Recall target:** The top-50 posts in the final feed should be in the top-5,000 candidates at least 95% of the time. Measure this with "oracle recall" using an offline evaluation set.

```
ANN Retrieval Tradeoff:
┌────────────┬──────────────┬──────────────┬──────────────┐
│ Index type │ Build time   │ QPS          │ Recall@100   │
├────────────┼──────────────┼──────────────┼──────────────┤
│ Exact      │ O(N×d)       │ low          │ 100%         │
│ HNSW       │ hours        │ very high    │ 95–99%       │
│ IVF-PQ     │ moderate     │ high         │ 85–95%       │
└────────────┴──────────────┴──────────────┴──────────────┘
```

### Stage 2: Lightweight Ranker

**Goal:** Score 5,000 candidates to select top 500 in <5ms total.

- Logistic regression or shallow MLP (10–20 features)
- Features: post age (log-scaled), source affinity score (precomputed), post engagement rate (likes+comments / impressions), media type (video > image > text in most feeds), user's historical interaction rate with this source
- Trained on same labels as heavy ranker but must be maximally fast
- At Meta scale: LR inference on 5,000 posts ≈ 1ms on CPU

**Facebook's EdgeRank (historical reference):**
EdgeRank was Meta's first generation feed ranking formula (pre-deep learning):
```
EdgeRank = Σ (affinity_score × edge_weight × time_decay)
```
- `affinity_score`: how often the user interacted with this edge (person/page)
- `edge_weight`: type of interaction (comment > like > view)
- `time_decay`: recency factor — older content ranked lower

Modern systems replaced EdgeRank with learned models, but the intuitions survive as feature groups.

### Stage 3: Heavy Ranker

**Goal:** Score top 500 posts with full feature set, multi-task predictions.

**Architecture: Multi-task neural network**

```
Input Features (post + user + social context)
         │
         ▼
┌────────────────────────────────────────────┐
│         Shared Bottom Layers               │
│   Dense(512) → BN → ReLU                  │
│   Dense(256) → BN → ReLU                  │
│   + Attention over user history embeddings │
└──────────────────┬─────────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
  [Like head] [Comment  ] [Share head]
              [head     ]
       ▼           ▼           ▼
  [Time-spent] [Hide head] [Unfollow ]
  [head      ]             [head     ]
```

**Multi-task learning motivation:** Training separate models per action leads to overfitting on sparse actions (shares are ~10× rarer than likes). Shared bottom layers allow the model to share representations across tasks; task-specific tower heads capture differences.

**Training:**
```python
total_loss = (
    w_like    * bce(pred_like,    label_like)    +
    w_comment * bce(pred_comment, label_comment) +
    w_share   * bce(pred_share,   label_share)   +
    w_time    * mse(pred_time,    label_time)     +
    w_hide    * bce(pred_hide,    label_hide)     +
    w_unfollow * bce(pred_unfollow, label_unfollow)
)
```

**Composite score:**
```python
score = (
    +3.0 * p_comment
    +2.0 * p_share
    +1.0 * p_like
    +0.5 * p_time_spent_percentile
    -2.0 * p_hide
    -3.0 * p_unfollow
    -5.0 * p_report
)
```

Weights are tuned via online experimentation; negative signals (hide, unfollow) carry heavy penalty because they signal regret rather than passive disengagement.

### Stage 4: Policy and Diversity Layer

Applied post-ranking, these are rule-based and non-ML adjustments:

- **Wellbeing guardrails:** Cap consecutive video posts at 3 (reduce passive scroll). Demote posts flagged as "emotionally negative" by a classifier. Enforce cross-session limits on content categories associated with low self-reported satisfaction.
- **Diversity injection:** No more than 2 consecutive posts from the same creator. Ensure at least 3 different topic clusters in top-10. See Section 8.
- **Integrity filters:** Posts under manual review for misinformation → score penalty or removal. CSAM, hate speech → hard remove. Spam accounts → demotion.
- **Boost rules:** "New content" freshness boost for posts under 2 hours (prevents feed staleness). Occasional "time capsule" post from 1 year ago (on this day features).

---

## 5. Feature Engineering

### Post Features

| Feature | Description | Update frequency |
|---|---|---|
| `post_age_log` | log(seconds since publish) | Per request |
| `media_type` | Video / image / link / text (1-hot) | At publish |
| `engagement_rate_1h` | (likes+comments) / impressions in 1h | Streaming |
| `engagement_velocity` | d(engagements)/dt — accelerating or slowing | Streaming |
| `content_embedding` | 128-dim embedding (CLIP for image, BERT for text) | At publish |
| `text_sentiment` | Positive / neutral / negative / outrage | At publish |
| `is_reshare` | Original vs reshared content | At publish |
| `hashtag_trending` | Is any hashtag trending in last hour? | Hourly |
| `creator_follower_count_log` | log(followers) | Daily |
| `creator_avg_engagement_rate_7d` | Historical engagement rate | Daily |

**Temporal decay** is critical. Facebook's original EdgeRank used exponential decay; modern systems learn the decay function:

```python
# Learned temporal decay (not hardcoded)
age_hours = (now - post_timestamp) / 3600
decay_feature = np.log1p(age_hours)  # feed into model as input, let model learn weighting
```

### User Features

| Feature | Description | Update frequency |
|---|---|---|
| `user_interest_embedding` | 128-dim learned interest vector | Daily |
| `user_engagement_rate_7d` | User's average engagement rate | Daily |
| `session_length_so_far` | Posts already scrolled in this session | Per request |
| `time_of_day_bucket` | Morning / afternoon / evening / night | Per request |
| `device_type` | Mobile / tablet / desktop | Per session |
| `social_graph_centrality` | User's influence in social graph | Weekly |
| `content_format_preference` | Fraction of video vs image vs text engaged | Daily |
| `following_count_log` | log(following count) — sparse graph vs dense | Daily |

### Social Context Features

These are among the strongest signals because they encode implicit social proof:

| Feature | Description |
|---|---|
| `mutual_friends_engaged` | Count of mutual friends who liked/commented |
| `creator_affinity_score` | Learned score for user-creator pair (historical interactions) |
| `close_friend_engaged` | Did a "close friend" (heavily interacted with) engage? |
| `group_membership_relevance` | Is this post from a group the user is active in? |
| `viewer_to_creator_distance` | Social graph hops between viewer and creator |

**Creator affinity score** is a key feature: a collaborative-filtering-style embedding trained on user-creator interaction history. Users who interact heavily with a creator get high affinity scores even for that creator's new posts with zero engagement.

```python
# Approximate creator affinity: dot product of user and creator embeddings
affinity = np.dot(user_embedding, creator_embedding) / (
    np.linalg.norm(user_embedding) * np.linalg.norm(creator_embedding)
)
```

---

## 6. Multi-Objective Optimization

### Why Single-Objective Fails

Training on likes alone: clickbait, outrage, and low-effort memes dominate because they maximize short-term engagement.
Training on time-spent alone: passive scroll-traps (infinite compilation videos) dominate.
Training on comments alone: controversial/political content dominates because it generates debate.

### Combining Signals

**Weighted linear combination** (baseline):
```
score = Σ wᵢ × pᵢ
```
Weights set by product policy, tuned by A/B test on long-horizon metrics.

**Pareto optimization** (advanced):
Find the Pareto frontier across objectives (engagement, wellbeing, creator reach). Any point on the frontier is non-dominated — improving one objective requires sacrificing another. The policy team chooses an operating point on this frontier.

**Constrained optimization with wellbeing floor:**
```
maximize: engagement_score(post)
subject to: wellbeing_score(feed) ≥ θ
            creator_diversity(feed) ≥ δ
```

This is implemented as a post-ranking adjustment: if the top-K posts push the predicted wellbeing score below threshold θ, swap out outrage posts for the next-best wellbeing-friendly posts.

### Meta's Meaningful Social Interaction (MSI) Pivot

In 2018, Zuckerberg announced the feed would prioritize "meaningful social interactions" — posts that spark genuine conversation — over passive content. Implementation:

- Comments and shares re-weighted 5–10× relative to likes
- Content from friends/family boosted relative to pages and creators
- Video content (associated with passive consumption) specifically penalized
- "Meaningful" classifier trained on user surveys about whether a post "was worth their time"

The tradeoff was explicit: time-on-platform dropped ~5%, but engagement-per-minute increased, and user satisfaction surveys improved.

### Inverse Signals (Negative Feedback)

| Signal | Weight | What it indicates |
|---|---|---|
| Hide post | -2× | Post was unwanted |
| Snooze user 30 days | -3× | User fatigue with source |
| Unfollow | -5× | Permanent rejection |
| Report | -10× | Policy violation or severe dislike |
| Survey: "not worth time" | -3× | Regret signal |

Negative signals are sparse but high-value. A post that 1% of viewers hide should be treated very differently from a post no one hides, even if both have the same like rate.

---

## 7. Creator Ecosystem

### Friends vs Creator Content Ranking Tension

Social feeds contain two fundamentally different content types:
1. **Friends/family posts:** High personal relevance, low production quality
2. **Creator/page posts:** Low personal relevance, high production quality

Friends content typically wins on relevance but loses on engagement rate (a friend's vacation photo gets fewer likes than a viral meme). Naively ranking by engagement rate systematically suppresses friends content. Meta addresses this with separate ranking tracks that are blended at feed assembly time.

```
Feed composition target (example):
  - 60% social graph content (friends, family)
  - 20% creator/page content user follows
  - 15% suggested content (interest-based, ANN retrieval)
  - 5% ads
```

These percentages are policy parameters, not ML outputs.

### Creator Reach Fairness

Without correction, reach is highly concentrated: top 0.1% of creators get 50%+ of impressions. This creates a flywheel where large creators get more data to optimize content, further increasing their reach advantage.

**Mitigation approaches:**
- Creator diversity constraint: cap any single creator at N% of a user's feed slots per day
- Exploration budget: reserve 10–15% of feed slots for creators the user hasn't interacted with (discovery)
- Small creator boost: apply a mild multiplicative boost to posts from creators with <10K followers who have strong engagement relative to their size

### Virality Prediction

A post going viral in the next 2 hours is highly valuable to surface early. Virality prediction uses:
- Engagement velocity: rate of change of engagement rate (2nd derivative)
- Cross-network spread: is this being shared across different social clusters?
- Early commenter diversity: high-diversity early engagers → organic virality vs. coordinated behavior

```python
def virality_score(post_id, window_minutes=30):
    early_engagements = get_engagements(post_id, minutes=window_minutes)
    velocity = len(early_engagements) / window_minutes
    
    # Engagement diversity: are engagers from different communities?
    community_ids = [user_community[u] for u in early_engagements]
    diversity = len(set(community_ids)) / len(community_ids)
    
    return velocity * diversity  # high velocity + high diversity = genuine viral
```

---

## 8. Diversity and Anti-Filter-Bubble

### Intra-List Diversity

A feed of 50 posts that are all about the same topic is worse than a feed with varied topics, even if each individual post is highly relevant. Diversity objectives:

**Topic diversity:** Ensure at least K distinct topic clusters in top-N posts.
**Creator diversity:** No more than M consecutive posts from the same creator.
**Format diversity:** Mix of video, images, and text in the feed.

### Determinantal Point Processes (DPP) for Diverse Selection

DPP provides a principled way to trade off quality vs diversity. The probability of selecting a subset S is:

```
P(S) ∝ det(L_S)
```

where L is the kernel matrix encoding both quality and similarity. High-quality, dissimilar items have high det(L_S).

**Practical approximation (Greedy MMR — Maximal Marginal Relevance):**
```python
def greedy_mmr(candidates, lambda_param=0.5, k=50):
    """
    candidates: list of (post_id, score, embedding)
    lambda_param: tradeoff between relevance and diversity (0=diversity, 1=relevance)
    """
    selected = []
    remaining = candidates.copy()
    
    # First item: highest score
    best = max(remaining, key=lambda x: x[1])
    selected.append(best)
    remaining.remove(best)
    
    while len(selected) < k and remaining:
        def mmr_score(candidate):
            relevance = candidate[1]
            max_sim = max(
                cosine_similarity(candidate[2], s[2])
                for s in selected
            )
            return lambda_param * relevance - (1 - lambda_param) * max_sim
        
        next_item = max(remaining, key=mmr_score)
        selected.append(next_item)
        remaining.remove(next_item)
    
    return selected
```

### Breaking Filter Bubbles

Pure relevance ranking creates filter bubbles: users who engage with content X get more X, which reinforces their engagement with X. Research shows this can drive political polarization and extremism pathways.

**Interventions:**
- **Exploration budget:** Reserve N% of feed slots for content outside the user's historical interest clusters. LinkedIn calls this "feed diversification."
- **Cross-partisan exposure:** On political content, deliberately mix content from different viewpoints (Meta's Civic Feed integrity work).
- **Interest graph perturbation:** Periodically add random walks from the user's interest graph to expose to adjacent topics.

**Measurement tension:** Breaking filter bubbles often reduces short-term engagement (users are less interested in content outside their bubble). You need long-horizon metrics (wellbeing surveys, diversity satisfaction surveys) to measure benefit — standard A/B tests show only the engagement cost, not the benefit.

---

## 9. Feedback Loops

### Popularity Bias Amplification

Highly-ranked posts get more impressions → more engagement → even higher ranking in future. This creates a rich-get-richer dynamic that concentrates engagement on a few posts and suppresses equal-quality content that received fewer early impressions.

**Correction:** Inverse propensity scoring (IPS) during training. A post shown to 1M users and getting 1% engagement should be treated differently from a post shown to 1K users getting 1% engagement — the second is a stronger signal because it wasn't boosted by the ranking system.

```python
# IPS-corrected training loss
loss = bce(pred, label) / propensity_score  # propensity = P(item shown | features)
```

### Echo Chambers and Radicalization Pathways

Engagement optimization can create radicalization pathways: users who engage with mildly extreme content are shown more extreme content, which they also engage with, leading to progressively more extreme content in the feed. YouTube's 2019 algorithm changes specifically addressed this.

**Detection:**
- Track "content trajectory" per user: is the average extremism score of consumed content increasing over time?
- Flag users whose content trajectory shows consistent drift toward high-extremism content

**Mitigation:**
- Hard cap on extremism score: posts above a threshold get a large score penalty regardless of engagement prediction
- "Rabbit hole" detection: if a user has consumed N consecutive posts on the same high-risk topic, inject diverse content

### Engagement-Wellbeing Divergence Measurement

The fundamental challenge: engagement metrics (clicks, likes) are available instantly, but wellbeing metrics (self-reported satisfaction, regret) require surveys with multi-week delays.

**Proxy signals for wellbeing:**
- Return rate after session: did the user come back tomorrow?
- Session end behavior: did the user close the app or did the app expire in background?
- Post-session survey (shown to random sample): "Was this time on [app] worth it?"
- Passive consumption ratio: high ratio of scroll-without-engagement may indicate addiction vs. satisfaction

---

## 10. Online Experimentation

### A/B Testing Feed Ranking

**Randomization unit:** User (not session or post). Feed ranking changes affect all sessions for a user, so user-level randomization is required to avoid cross-contamination.

**Key challenge: Social network interference.** User A and User B are in different experiment arms. User B creates a post. User A's ranking of that post may be affected by B's arm (e.g., if B's arm reduces engagement with posts, the post gets fewer early engagements, which feeds back into A's ranking). This violates the Stable Unit Treatment Value Assumption (SUTVA).

**Mitigation:**
- **Ego-cluster randomization:** Assign entire social neighborhoods to the same arm. Reduces power but reduces interference.
- **Two-sided marketplaces:** Treat feed separately from distribution — randomize at creator level for distribution experiments.

### Long-Term vs Short-Term Effects

Short-term A/B tests (1–2 weeks) capture engagement changes. They miss:
- **Novelty effects:** New features boost engagement initially, then fade
- **Content ecosystem effects:** If a ranking change reduces reach for creators, creators may post less (takes months to detect)
- **User behavior adaptation:** Users learn the new feed ordering and adapt their behavior

**Holdout groups (long-horizon measurement):**
Keep 1% of users in the old ranking model permanently. Compare long-term metrics (content diversity consumed, wellbeing surveys, churn rate) after 3–6 months. This is expensive but necessary for detecting slow-moving changes.

### Ad Interference in Organic Experiments

Ads are ranked inline with organic content. An organic ranking experiment changes which organic posts are shown → changes which ad slots are seen → changes ad revenue. This confounds the organic experiment.

**Solutions:**
- Run organic and ads experiments on disjoint user sets
- Measure ad metrics as secondary guardrail metrics in organic experiments
- Use revenue-neutral ad insertion (adjust ad count dynamically to maintain revenue parity across arms)

### Metric Hierarchy for Feed Experiments

```
Guardrail metrics (experiment invalid if these degrade):
  - Feed load latency P99
  - App crash rate
  - Ad revenue per DAU

Primary metrics (experiment decision):
  - Meaningful interactions rate (comments + shares per session)
  - 7-day return rate

Secondary metrics (directional signal):
  - Likes per session (lower weight)
  - Time-on-platform (controversial — can be good or bad)
  - Negative feedback rate (hide / unfollow / report)
```

---

## 11. Failure Modes

### Misinformation Amplification

**Mechanism:** False or misleading posts often generate high engagement (outrage, debate) in the short term. Without intervention, the ranking system surfaces them widely before fact-checkers can label them.

**Failure signature:** Viral posts with high engagement velocity but high report rate from a subset of users.

**Mitigation:** Real-time misinformation classifier (trained on fact-check labels); posts flagged as "likely false" receive a large score penalty. Third-party fact-checking labels (Meta's IFCN partners) feed back as strong negative labels within hours. "Reduced distribution" (not removed, but not amplified) is the policy lever.

### Outrage Optimization

**Mechanism:** Comments and shares (weighted heavily as "meaningful interactions") are disproportionately driven by angry reactions and controversy. A ranking system that maximizes comments is implicitly maximizing outrage.

**Failure signature:** Top posts in the feed are increasingly political, controversial, or emotionally provocative over time. User self-reported satisfaction declines while engagement metrics improve.

**Mitigation:** Sentiment-aware re-weighting: a "comment driven by anger" should be worth less than a "comment driven by interest." Train a comment sentiment classifier. Apply sentiment discount to engagement weights: comment_value = base_weight × (1 - anger_fraction).

### Creator Suppression (Cold Start for Creators)

**Mechanism:** New creators have no engagement history → low affinity scores → low ranking → low impressions → no engagement data → perpetual cold start. The system amplifies existing inequality.

**Failure signature:** New creator cohort show near-zero reach even for high-quality content.

**Mitigation:** Creator "exploration" budget: new posts from new creators are shown to a random 0.1% sample to gather initial engagement signal before ranking takes over. Bootstrap creator embeddings from content embeddings (even without engagement history, a creator's post can be matched to users who have interacted with similar content).

### Cold Start for New Users

**Mechanism:** New users have no engagement history, no interest embeddings, no affinity scores. Personalization defaults to trending/popular content, which may not match the user and drives early churn.

**Failure signature:** Day-7 retention for new users significantly lower than for users with 30+ days of history.

**Mitigation:** Onboarding interest selection (explicitly ask users what they care about on signup). Demographic priors (age group, country). Rapid cold start via first-session engagement: within the first session, each interaction updates the interest embedding in near-real-time. Use a "new user" ranker that upweights diversity and content quality signals and downweights personalization signals until sufficient history is accumulated (typically 50+ interactions).

### Model Feedback Loop Collapse

**Mechanism:** Model trained on logged engagement → model changes what's shown → engagement data shifts → next model trained on biased data. Over time, the model converges to a degenerate state (only ever showing a narrow set of content).

**Failure signature:** Content diversity in served feeds decreases over time. Feature distributions drift from training distribution (covariate shift).

**Mitigation:** Counterfactual logging (log model scores for items not shown); use importance-weighted training. Exploration policy: randomly serve non-top-ranked items to a small fraction of requests. Periodically retrain from scratch on a dataset that includes exploration data.

---

## Canonical Interview Q&As

**Q: How do you handle the fact that optimizing for engagement often conflicts with user wellbeing?**

A: This is the central product and ML problem in feed ranking. The technical approach is constrained multi-objective optimization: set wellbeing metrics as floor constraints rather than objectives to maximize. Concretely: compute a predicted wellbeing score for the feed (using proxy signals like self-reported satisfaction labels, regret signals like "hide" and "unfollow," and long-term return rate). Require this score to stay above a threshold θ. Within that constraint, maximize engagement. The challenge is that wellbeing labels are slow (survey-based, 2-4 week lag) while engagement labels are instant. You need long-horizon holdout groups (1% of users permanently in control) to detect slow wellbeing effects that short A/B tests miss. Meta's 2018 MSI pivot was essentially moving the constraint boundary: they accepted lower engagement to achieve higher wellbeing floor.

**Q: How would you design the candidate retrieval stage to handle a user with 5,000 friends and 2,000 followed pages — in under 10ms?**

A: Parallelize retrieval across sources. Social graph retrieval is a fan-out operation: given user U's adjacency list (friends + pages), fetch the latest N posts per source. For 5,000 friends × 5 posts each = 25,000 posts, this requires the adjacency list and post indexes to be in memory (TAO/Redis). Run ANN retrieval for interest-based candidates in parallel with graph traversal. Use approximate graph traversal: for users with very large social graphs (5K+ friends), only retrieve from "active" connections (those who posted in last 7 days) rather than all friends. Pre-compute per-user candidate sets during off-peak hours and cache them, then do lightweight freshness updates on top. The 10ms budget is for cache hits; cache misses (new users, cold start) have a separate slower path.

**Q: How do you detect and prevent popularity bias in training data?**

A: Popularity bias occurs because items shown more often get more engagement data, making them appear better to the model, leading to even more exposure. The fix is inverse propensity scoring (IPS): weight each training example by the inverse of its probability of being shown, so frequently-shown items don't dominate the gradient. Propensity scores are estimated from the logging policy (the ranking model that generated the training data). This requires careful logging: you must record the score and rank at the time of impression, not reconstruct them later. Additionally, maintain an exploration budget (ε-greedy or Thompson sampling) to gather unbiased training signal from items that wouldn't normally be shown. LinkedIn's feed ranking paper describes a similar debiasing approach using counterfactual estimators.

**Q: A new competitor launches. Suddenly users are spending less time on your platform. Your engagement metrics drop 15%. How do you investigate whether this is a model problem or a product problem?**

A: First, decompose the drop. Check segment-level metrics: is the drop uniform across age groups, geographies, and user tenure, or concentrated in a specific segment? A model regression would affect users more uniformly; a competitive threat would first affect specific demographics or power users. Second, check whether the engagement rate per session changed (model quality) vs. whether session frequency or session length changed (product/competitive). Third, run the ranking model on historical data and check if predicted engagement scores degraded (distribution shift in input features). Fourth, check content supply: did creators post less (supply problem) or did the same posts get lower engagement (demand problem)? Finally, look at negative signals: if hide/unfollow/report rates increased, it may be a model quality issue; if they're flat but impressions decreased, it's retention/acquisition.

**Q: How does the ads system interact with the organic feed ranker, and what could go wrong?**

A: Ads compete for the same feed slots as organic content. There are two models: (1) Unified auction: organic posts and ads receive a single quality score, then ads also have a bid, and the combined score = predicted_value × bid. An ad wins a slot only if its revenue value exceeds the organic opportunity cost. (2) Fixed insertion: ads are inserted at every Nth position regardless of organic ranking quality. Unified auction is more efficient but complex. What can go wrong: (a) If organic ranking improves (users see better posts), the opportunity cost of showing an ad increases, so fewer ads clear the auction → revenue falls unexpectedly. You need to run organic experiments with revenue as a guardrail metric. (b) If the ad quality classifier shares features with the organic ranker, a model retrain can unexpectedly shift ad performance. (c) Ads create position effects — users in ad-heavy feeds develop "banner blindness" and skip early organic posts too, biasing engagement labels for organic posts.

**Q: How would you measure and quantify filter bubble effects in the feed?**

A: Filter bubble intensity is multi-dimensional. Content-side metrics: topic diversity index of content consumed per user per week (using topic embeddings, compute average pairwise distance — higher = more diverse); source diversity (distinct creator count in weekly consumption); partisan/viewpoint diversity on political content. Trajectory metrics: track whether the average extremism or partisanship score of content consumed is drifting over time for users. Survey metrics: periodically ask "did you see content that challenged your views this week?" (self-reported perspective diversity). Counterfactual metrics: compare feed diversity to a diversity-maximizing baseline — what would the user have seen if the feed was diversified? These require different interventions: content diversity can be increased via MMR/DPP reranking (with 5–10% engagement cost), viewpoint diversity on political content requires explicit cross-partisan injection with user consent UI.

**Q: Walk me through how you would design a fair ranking system for creators — ensuring small creators can get discovered, not just large ones.**

A: The core problem is a self-reinforcing rich-get-richer dynamic: large creators have engagement history → high affinity scores → high ranking → more impressions → more engagement. Four interventions: (1) Exploration budget: reserve 10–15% of feed slots for content from creators the user has not interacted with (discovery bucket). Within this bucket, rank by engagement relative to peer creators (same follower tier, same content category) to give small creators fair competitive standing. (2) Propensity correction in training: a post from a creator with 100M followers that gets 1% engagement is shown to vastly more users than a post from a creator with 10K followers getting 1% engagement. Correct for this in training so both receive equal credit for the same relative engagement quality. (3) Audience saturation correction: a creator's posts reaching the same users repeatedly are less valuable than posts reaching new users. Penalize repeated delivery of the same creator to the same user. (4) Early signal bootstrapping: use content embeddings (CLIP, BERT) to match new creator content to users who have engaged with similar content, bypassing the cold start problem for new creators entirely. LinkedIn's "creator mode" and Instagram's "Reels ranking" both describe variants of this approach.

## Flashcards

**Consumption model?** #flashcard
push vs pull? Facebook/Instagram are pull (user opens app → feed constructed on-demand). Twitter historically used push (precompute fan-out to followers). Pull is harder to serve fast but more personalized; push is cheaper at read time but expensive for celebrities with 50M followers (fan-out problem).

**What signals define "good"? Engagement (likes, comments, shares) vs time-spent vs meaningful interactions vs self-reported satisfaction. These diverge?** #flashcard
outrage content maximizes engagement, wellness content maximizes satisfaction.

**Ads integration? Are ads ranked inline with organic content or separately budgeted? Inline ranking creates a single auction but risks ad-organic quality conflation.?** #flashcard
Ads integration? Are ads ranked inline with organic content or separately budgeted? Inline ranking creates a single auction but risks ad-organic quality conflation.

**Objective hierarchy? Is wellbeing a hard constraint (floor) or a soft objective in the optimization? Meta's 2018 "meaningful social interaction" pivot was a policy decision to re-weight creator/page content down and friends content up?** #flashcard
not purely ML.

**Creator vs social graph content? LinkedIn is mostly creator content (articles, posts from strangers); Facebook is mostly social graph (friends, family). This determines candidate retrieval strategy.?** #flashcard
Creator vs social graph content? LinkedIn is mostly creator content (articles, posts from strangers); Facebook is mostly social graph (friends, family). This determines candidate retrieval strategy.

**New user cold start threshold? How many interactions before the personalization kicks in? What's the fallback?** #flashcard
trending content, demographic-based priors, onboarding interest selection?

**Real-time vs near-real-time? Should posts published 10 seconds ago appear in the feed immediately (requires real-time indexing) or is 1–5 minute lag acceptable?** #flashcard
Real-time vs near-real-time? Should posts published 10 seconds ago appear in the feed immediately (requires real-time indexing) or is 1–5 minute lag acceptable?

**Fetch all friends (avg ~300 on Facebook) + followed pages/groups?** #flashcard
Fetch all friends (avg ~300 on Facebook) + followed pages/groups

**For each source, retrieve last N posts (N=5–10)?** #flashcard
For each source, retrieve last N posts (N=5–10)

**Implementation?** #flashcard
graph database (TAO at Meta) → per-user adjacency list cached in RAM → O(1) lookup

**User has a learned interest embedding (128-dim, updated daily)?** #flashcard
User has a learned interest embedding (128-dim, updated daily)

**Posts are embedded at publish time (CLIP for images, BERT for text)?** #flashcard
Posts are embedded at publish time (CLIP for images, BERT for text)

**FAISS / ScaNN index on post embeddings → ANN search returns top-K semantically similar posts from non-friends?** #flashcard
FAISS / ScaNN index on post embeddings → ANN search returns top-K semantically similar posts from non-friends

**This is how Instagram's "suggested posts" and LinkedIn's "posts you might like" work?** #flashcard
This is how Instagram's "suggested posts" and LinkedIn's "posts you might like" work

**Logistic regression or shallow MLP (10–20 features)?** #flashcard
Logistic regression or shallow MLP (10–20 features)

**Features?** #flashcard
post age (log-scaled), source affinity score (precomputed), post engagement rate (likes+comments / impressions), media type (video > image > text in most feeds), user's historical interaction rate with this source

**Trained on same labels as heavy ranker but must be maximally fast?** #flashcard
Trained on same labels as heavy ranker but must be maximally fast

**At Meta scale?** #flashcard
LR inference on 5,000 posts ≈ 1ms on CPU

**affinity_score?** #flashcard
how often the user interacted with this edge (person/page)

**edge_weight?** #flashcard
type of interaction (comment > like > view)

**time_decay: recency factor?** #flashcard
older content ranked lower

**Wellbeing guardrails?** #flashcard
Cap consecutive video posts at 3 (reduce passive scroll). Demote posts flagged as "emotionally negative" by a classifier. Enforce cross-session limits on content categories associated with low self-reported satisfaction.

**Diversity injection?** #flashcard
No more than 2 consecutive posts from the same creator. Ensure at least 3 different topic clusters in top-10. See Section 8.

**Integrity filters?** #flashcard
Posts under manual review for misinformation → score penalty or removal. CSAM, hate speech → hard remove. Spam accounts → demotion.

**Boost rules?** #flashcard
"New content" freshness boost for posts under 2 hours (prevents feed staleness). Occasional "time capsule" post from 1 year ago (on this day features).

**Comments and shares re-weighted 5–10× relative to likes?** #flashcard
Comments and shares re-weighted 5–10× relative to likes

**Content from friends/family boosted relative to pages and creators?** #flashcard
Content from friends/family boosted relative to pages and creators

**Video content (associated with passive consumption) specifically penalized?** #flashcard
Video content (associated with passive consumption) specifically penalized

**"Meaningful" classifier trained on user surveys about whether a post "was worth their time"?** #flashcard
"Meaningful" classifier trained on user surveys about whether a post "was worth their time"

**60% social graph content (friends, family)?** #flashcard
60% social graph content (friends, family)

**20% creator/page content user follows?** #flashcard
20% creator/page content user follows

**15% suggested content (interest-based, ANN retrieval)?** #flashcard
15% suggested content (interest-based, ANN retrieval)

**5% ads?** #flashcard
5% ads

**Creator diversity constraint?** #flashcard
cap any single creator at N% of a user's feed slots per day

**Exploration budget?** #flashcard
reserve 10–15% of feed slots for creators the user hasn't interacted with (discovery)

**Small creator boost?** #flashcard
apply a mild multiplicative boost to posts from creators with <10K followers who have strong engagement relative to their size

**Engagement velocity?** #flashcard
rate of change of engagement rate (2nd derivative)

**Cross-network spread?** #flashcard
is this being shared across different social clusters?

**Early commenter diversity?** #flashcard
high-diversity early engagers → organic virality vs. coordinated behavior

**Exploration budget?** #flashcard
Reserve N% of feed slots for content outside the user's historical interest clusters. LinkedIn calls this "feed diversification."

**Cross-partisan exposure?** #flashcard
On political content, deliberately mix content from different viewpoints (Meta's Civic Feed integrity work).

**Interest graph perturbation?** #flashcard
Periodically add random walks from the user's interest graph to expose to adjacent topics.

**Track "content trajectory" per user?** #flashcard
is the average extremism score of consumed content increasing over time?

**Flag users whose content trajectory shows consistent drift toward high-extremism content?** #flashcard
Flag users whose content trajectory shows consistent drift toward high-extremism content

**Hard cap on extremism score?** #flashcard
posts above a threshold get a large score penalty regardless of engagement prediction

**"Rabbit hole" detection?** #flashcard
if a user has consumed N consecutive posts on the same high-risk topic, inject diverse content

**Return rate after session?** #flashcard
did the user come back tomorrow?

**Session end behavior?** #flashcard
did the user close the app or did the app expire in background?

**Post-session survey (shown to random sample)?** #flashcard
"Was this time on [app] worth it?"

**Passive consumption ratio?** #flashcard
high ratio of scroll-without-engagement may indicate addiction vs. satisfaction

**Ego-cluster randomization?** #flashcard
Assign entire social neighborhoods to the same arm. Reduces power but reduces interference.

**Two-sided marketplaces: Treat feed separately from distribution?** #flashcard
randomize at creator level for distribution experiments.

**Novelty effects?** #flashcard
New features boost engagement initially, then fade

**Content ecosystem effects?** #flashcard
If a ranking change reduces reach for creators, creators may post less (takes months to detect)

**User behavior adaptation?** #flashcard
Users learn the new feed ordering and adapt their behavior

**Run organic and ads experiments on disjoint user sets?** #flashcard
Run organic and ads experiments on disjoint user sets

**Measure ad metrics as secondary guardrail metrics in organic experiments?** #flashcard
Measure ad metrics as secondary guardrail metrics in organic experiments

**Use revenue-neutral ad insertion (adjust ad count dynamically to maintain revenue parity across arms)?** #flashcard
Use revenue-neutral ad insertion (adjust ad count dynamically to maintain revenue parity across arms)

**Feed load latency P99?** #flashcard
Feed load latency P99

**App crash rate?** #flashcard
App crash rate

**Ad revenue per DAU?** #flashcard
Ad revenue per DAU

**Meaningful interactions rate (comments + shares per session)?** #flashcard
Meaningful interactions rate (comments + shares per session)

**7-day return rate?** #flashcard
7-day return rate

**Likes per session (lower weight)?** #flashcard
Likes per session (lower weight)

**Time-on-platform (controversial?** #flashcard
can be good or bad)

**Negative feedback rate (hide / unfollow / report)?** #flashcard
Negative feedback rate (hide / unfollow / report)
