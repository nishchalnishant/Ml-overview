---
module: System Design
topic: System Design
subtopic: Video Recommendation System
status: unread
tags: [productionml, ml, system-design-video-recommenda]
---
# Video Recommendation System Design

End-to-end ML system for personalized video recommendations at YouTube/TikTok scale. Canonical system design question at large-scale consumer tech companies (Google, Meta, ByteDance, Netflix).

**Scale:** 2B+ users, 500 hours of video uploaded per minute, personalized recommendations in <200ms, cold-start for new videos within minutes of upload.

---

## 1. Problem Framing

### Clarifying Questions

- **Which surface?** YouTube homepage (intent-driven, logged-in) vs Up Next (continuation of current session) vs TikTok For You Page (pure discovery, often anonymous)
- **Logged-in vs anonymous?** Anonymous users have no history — require session-based or demographic signals only
- **Optimization target?** Watch time, satisfaction (survey-based), creator ecosystem health, or composite objective
- **Content type?** Short-form (<60s), long-form (>10 min), live streams, podcasts — each has distinct engagement signals
- **Cold start scope?** New videos only, or also new users and new creators?
- **Policy constraints?** Age-gating, regional content restrictions, copyright enforcement, extremism pipeline avoidance
- **Creator monetization?** Are we responsible for creator reach/revenue or purely user engagement?

### Metric Trade-offs

**Primary business metrics:**

| Metric | Measures | Risk if over-optimized |
|---|---|---|
| Watch time (minutes) | Engagement depth | Clickbait, autoplay rabbit holes |
| DAU / 7-day retention | Platform stickiness | Addictive content loops |
| Click-through rate (CTR) | Relevance of recommendation surface | Thumbnail bait, sensationalist titles |
| Like / share / subscribe rate | Active satisfaction | Does not capture passive consumption |
| Creator monetization | Ecosystem health | Concentration on top creators |

**Why pure watch-time optimization fails:**

Watch time is a proxy for value, not value itself. A user who clicks a misleading thumbnail and watches 80% of a 10-minute video out of sunk-cost commitment generates high watch time — but when surveyed, reports dissatisfaction. Papernot et al. and YouTube's own research show that maximizing watch time correlates with recommending progressively more extreme content (radicalization pipeline) because extreme content generates longer, more emotionally activated sessions.

$$\text{watch\_time} \neq \text{satisfaction}$$

YouTube introduced satisfaction surveys (5-star) and explicit "not interested" / "don't recommend channel" signals after 2019 precisely because watch-time-only reward functions produced harmful emergent behaviors.

**Composite objective (multi-task framing):**

$$\text{score}(u, v) = w_1 \cdot \hat{p}_{watch} + w_2 \cdot \hat{p}_{like} + w_3 \cdot \hat{p}_{share} - w_4 \cdot \hat{p}_{skip} - w_5 \cdot \hat{p}_{dislike} - w_6 \cdot \hat{p}_{regret}$$

Weights are tuned via A/B experimentation against long-term retention metrics, not just session watch time.

---

## 2. Scale

| Dimension | Number |
|---|---|
| Monthly active users | 2B+ |
| Daily active users | 500M+ |
| Videos in corpus | 800M+ |
| New uploads per minute | 500 hours of content |
| Homepage recommendations per DAU | ~20 candidates shown |
| Total recommendations served per day | ~10B |
| Retrieval latency SLA | <50ms |
| End-to-end recommendation latency SLA | <200ms |
| New video cold-start time | <5 minutes from upload |
| Embedding index size (video tower) | ~10B float32 vectors → requires ANN |

At 500 hours uploaded per minute, the video corpus grows by ~720K videos per day. A brute-force nearest-neighbor search over 800M vectors is infeasible at serving time — this forces approximate nearest neighbor (ANN) retrieval as an architectural requirement, not an optimization.

---

## 3. System Architecture

```
User Request (homepage load / swipe)
            │
            ▼
  ┌─────────────────────┐
  │  Context Assembly   │  ← user_id, session signals, device,
  │                     │    recent watches, time of day
  └──────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────┐    ┌────────────┐
│  User   │    │  Video     │   TWO-TOWER MODEL
│  Tower  │    │  Tower     │
│(history,│    │(visual,    │
│ context)│    │ audio,     │
│         │    │ metadata)  │
└────┬────┘    └─────┬──────┘
     │               │
     │   user_emb    │  video_emb (pre-computed, ANN index)
     └───────┬───────┘
             │  inner product → top-K retrieval
             ▼
  ┌──────────────────────┐
  │  ANN Retrieval       │  ScaNN / FAISS over 800M video index
  │  (Candidate Gen)     │  → 1000–5000 candidates in <50ms
  └──────────┬───────────┘
             │
             │   also injected:
             │   ┌────────────────────────┐
             │   │  New Video Fast Lane   │  content-based candidates
             │   │  (<5 min old uploads)  │  from upload pipeline
             │   └────────────────────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Multi-Task Ranking  │  Wide & Deep / MMoE
  │  Model               │  predict: watch%, like, share,
  │                      │  skip, subscribe, regret
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Re-Ranking Layer    │
  │                      │  ← diversity (MMR, topic/creator dedup)
  │                      │  ← freshness boost
  │                      │  ← policy filters (copyright, age-gate,
  │                      │    extremism classifier score)
  │                      │  ← creator fairness (long-tail exposure)
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Final List Assembly │  20–50 items, ordered, with
  │  + Slate Optimizer   │  position-aware diversity
  └──────────────────────┘
             │
             ▼
        User's Feed

  ─────── Async Feedback Loop ───────

  User interactions (watch%, like, skip, survey)
             │
             ▼
  ┌──────────────────────┐
  │  Event Stream        │  Kafka / Pub-Sub
  │  (Kafka)             │
  └──────────┬───────────┘
             │
    ┌────────┴──────────┐
    │                   │
    ▼                   ▼
┌─────────┐      ┌────────────┐
│ Feature │      │ Training   │
│ Store   │      │ Pipeline   │
│ Update  │      │ (daily /   │
│(Flink)  │      │  weekly)   │
└─────────┘      └────────────┘
```

---

## 4. Two-Tower Retrieval

### Architecture Overview

The two-tower model (dual encoder) computes separate embeddings for users and videos. At serving time, the user tower runs online (given current context), while video tower embeddings are pre-computed and indexed for ANN search. Retrieval is a maximum inner product search (MIPS).

$$\text{score}(u, v) = \langle \mathbf{e}_u, \mathbf{e}_v \rangle$$

### User Tower

**Inputs:**

| Feature group | Features | Encoding |
|---|---|---|
| Watch history | Last 500 watched video IDs | Average pooled video embeddings |
| Search history | Recent queries | Subword token embeddings |
| Liked/disliked videos | IDs + timestamp | Recency-weighted average |
| Demographics | Age bucket, country, language | Learned embeddings |
| Real-time context | Time of day, device type, session length so far | Continuous + categorical |
| Negative signals | "Not interested" clicks, skips | Subtracted from positive history |

**Architecture:**

```python
class UserTower(nn.Module):
    def __init__(self, video_emb_dim=256, context_dim=64, output_dim=256):
        super().__init__()
        # Aggregate watch history
        self.history_attn = nn.MultiheadAttention(video_emb_dim, num_heads=4)
        # Encode real-time context
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(video_emb_dim + 128, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim)
        )

    def forward(self, history_embs, context_feats):
        # history_embs: [batch, seq_len, video_emb_dim]
        # Self-attention over watch history, recency-weighted
        h_agg, _ = self.history_attn(history_embs, history_embs, history_embs)
        h_pooled = h_agg.mean(dim=1)  # [batch, video_emb_dim]
        c = self.context_encoder(context_feats)  # [batch, 128]
        combined = torch.cat([h_pooled, c], dim=-1)
        return F.normalize(self.projection(combined), dim=-1)
```

**Key design decisions:**
- Normalize output to unit sphere → inner product = cosine similarity
- Attention over history (vs simple average) captures topical drift within a session
- Real-time context injected separately to capture "right now I want X" vs "historically I watch Y"

### Video Tower

**Inputs:**

| Feature group | Features | Encoding |
|---|---|---|
| Title + description | Text | BERT-style encoder, subword tokens |
| Thumbnails | 3 sampled frames | Vision transformer or ResNet-50 |
| Audio | Speech transcript + music/speech detection | Text encoder + audio spectrogram CNN |
| Engagement stats | Views, CTR, like ratio, avg watch % | Log-scaled continuous features |
| Metadata | Duration, category, channel, publish age | Categorical embeddings |
| Content topics | LDA topics from transcript | Topic distribution vector |

```python
class VideoTower(nn.Module):
    def __init__(self, text_dim=768, visual_dim=2048, audio_dim=512, output_dim=256):
        super().__init__()
        self.text_encoder = BertEncoder(output_dim=text_dim)
        self.visual_encoder = VisualEncoder(output_dim=visual_dim)  # ViT or ResNet
        self.audio_encoder = AudioCNN(output_dim=audio_dim)
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + visual_dim + audio_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim)
        )

    def forward(self, text_tokens, visual_frames, audio_features):
        t = self.text_encoder(text_tokens)
        v = self.visual_encoder(visual_frames)
        a = self.audio_encoder(audio_features)
        combined = torch.cat([t, v, a], dim=-1)
        return F.normalize(self.fusion(combined), dim=-1)
```

### Training the Two-Tower Model

**Loss function — in-batch softmax (sampled softmax):**

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\langle \mathbf{e}_{u_i}, \mathbf{e}_{v_i} \rangle / \tau)}{\sum_{j=1}^{B} \exp(\langle \mathbf{e}_{u_i}, \mathbf{e}_{v_j} \rangle / \tau)}$$

- Positive pair: (user, video they watched significantly)
- Negatives: other videos in the batch (random in-batch negatives)
- Temperature τ ≈ 0.07 (tuned)
- Hard negative mining: add videos with high predicted score but not watched

**Serving — ANN with ScaNN/FAISS:**

```
Video corpus (800M videos)
        │
        ▼ (offline, updated every 30 min)
  Video embeddings computed by video tower
        │
        ▼
  ScaNN index (quantized, HNSW or IVF-PQ)
        │  search with user_emb at query time
        ▼
  Top-1000 approximate nearest neighbors
  in <50ms (p99)
```

**Index update frequency for new videos:**
- New video embeddings computed within 2–3 minutes of upload (stream processing via Flink)
- Incremental index update (ScaNN supports online insertion without full rebuild)
- Full index rebuild nightly to remove deleted videos and re-quantize

---

## 5. Ranking Model

### Multi-Task Learning with MMoE

The ranker takes ~1000–5000 candidates from retrieval and outputs a scalar score per candidate. It must predict multiple labels simultaneously because optimizing a single label (e.g., watch time) leads to harmful behaviors.

**Tasks (output heads):**

| Task | Label | Why it matters |
|---|---|---|
| Watch percentage | Continuous [0, 1] | Engagement depth, not just click |
| Like | Binary | Explicit positive satisfaction |
| Share | Binary | Strong satisfaction signal |
| Subscribe | Binary | Long-term value signal |
| Skip (within 5s) | Binary | Negative satisfaction |
| Dislike / "Not interested" | Binary | Explicit negative signal |
| Regret proxy | Binary (short watch + dislike/not-interested) | Clickbait detection |

**Mixture of Experts (MMoE) architecture:**

```
Input Features
     │
     ▼
┌─────────────────────────────┐
│  Shared Expert Layers (8x)  │  each: 2-layer MLP, 256 units
│  + Task-specific Experts    │  2 experts per task
└────────────────┬────────────┘
                 │
     ┌───────────┼────────────────┐
     │  Gating networks per task  │  softmax over expert outputs
     └───────────┬────────────────┘
                 │
     ┌───────────┼──────────┬──────────┬──────────┬──────────┐
     ▼           ▼          ▼          ▼          ▼          ▼
  watch%      like       share    subscribe     skip      regret
  [sigmoid]  [sigmoid] [sigmoid]  [sigmoid]  [sigmoid] [sigmoid]
```

**Final composite ranking score:**

```python
def compute_ranking_score(predictions, weights):
    w_watch, w_like, w_share, w_sub, w_skip, w_regret = weights
    score = (
        w_watch * predictions['watch_pct'] +
        w_like  * predictions['like'] +
        w_share * predictions['share'] +
        w_sub   * predictions['subscribe'] -
        w_skip  * predictions['skip'] -
        w_regret * predictions['regret']
    )
    return score
```

Weights are held by a business logic layer — not learned — so policy changes (e.g., reduce regret weight during regulatory scrutiny) don't require retraining.

### Position Bias Correction

The training data is logged from previous recommendation surfaces. Videos shown in position 1 receive far more clicks than position 10 regardless of quality — this is position bias. Training without correction causes the model to learn "position 1 gets clicks" rather than "quality content gets clicks."

**Correction via Inverse Propensity Weighting (IPW):**

$$\mathcal{L}_{unbiased} = \sum_{(u,v,pos)} \frac{y_{u,v}}{\hat{p}_{pos}} \cdot \log \hat{p}_{click}(u,v)$$

where $\hat{p}_{pos}$ is the probability of being examined at position `pos` — estimated from a separate propensity model or from randomization experiments (randomly shuffle positions for a fraction of traffic).

**Simplified position-as-feature approach (YouTube's method):**
- Add `position` as a feature during training
- At inference time, set `position = 0` (or use a learned "position-free" inference)
- The model learns to separate position effect from relevance signal

### Wide & Deep Architecture (YouTube-style)

```
Wide component:                 Deep component:
Cross-product feature           DNN over dense embeddings
transformations                 of user + video features
(memorization)                  (generalization)
     │                                │
     └──────────────┬─────────────────┘
                    ▼
             Output layer
             (per-task heads)
```

---

## 6. Video Content Understanding

### Video Embeddings

Processing 500 hours/minute requires efficient frame sampling rather than per-frame encoding.

**Frame sampling strategy:**
- Uniform sampling: 1 frame per second for first 30s, 1 per 5s thereafter
- Scene-change detection: dense sampling around cuts
- Maximum 64 frames per video for embedding computation

**Visual encoder:**

```python
class VideoEmbeddingPipeline:
    def __init__(self):
        self.frame_sampler = AdaptiveFrameSampler(max_frames=64)
        self.visual_encoder = ViT_B_16(pretrained=True)  # or CLIP visual encoder
        self.temporal_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )

    def encode_video(self, video_path):
        frames = self.frame_sampler.sample(video_path)         # [T, H, W, C]
        frame_embs = self.visual_encoder(frames)               # [T, 768]
        video_emb = self.temporal_aggregator(frame_embs)       # [T, 768]
        return video_emb.mean(dim=0)                           # [768] pooled
```

### Audio Features

| Signal | Method | Use case |
|---|---|---|
| Speech transcript | Whisper ASR | Topic modeling, searchability |
| Music detection | Spectrogram CNN | Music video classification, copyright flag |
| Language detection | FastText LangID | Regional routing |
| Tone / sentiment | Prosody features | Emotional content flagging |
| Background noise | Audio event detection | Production quality signal |

### Thumbnail Quality Scoring

Thumbnails are the primary CTR driver. A thumbnail quality model predicts:
- Visual clarity (sharpness, contrast)
- Face presence and emotion (high CTR predictor)
- Text overlay (informative vs misleading)
- Clickbait indicator: compare thumbnail visual content vs video content similarity — high divergence = potential bait

```python
def thumbnail_clickbait_score(thumbnail_emb, video_content_emb):
    """High score = thumbnail and video content are dissimilar = potential bait."""
    similarity = F.cosine_similarity(thumbnail_emb, video_content_emb, dim=-1)
    return 1.0 - similarity  # [0, 1], higher = more bait
```

### Transcript-Based Topic Modeling

- Run Whisper or similar ASR on audio track
- Apply BERTopic or LDA over transcript
- Topics used in video tower as additional sparse features
- Enables cold-start recommendations based on topic before engagement data accumulates

---

## 7. Cold Start for New Videos

### The Problem

A new video uploaded 3 minutes ago has zero engagement data. The video tower must produce a useful embedding using only content features. Without intervention, new videos never enter the ANN index and are never recommended — the rich-get-richer feedback loop.

### Multi-Phase Cold Start Strategy

**Phase 0 (0–5 min post-upload): Content-only embedding**
- Extract visual, audio, transcript, title/description features immediately
- Compute video tower embedding from content features only (engagement stats = 0, treated as cold-start flag)
- Insert into ANN index within 5 minutes

**Phase 1 (5 min – 6 hours): Warm-up traffic allocation**
- Reserve 1–3% of recommendation slots for "exploration" — new videos from followed channels and topically similar channels
- Prioritize distribution to users with high exploration tolerance (identified from historical diversity preference)

**Phase 2 (6–48 hours): Bootstrap from engagement**
- Early engagement signals (CTR, 30-second retention rate) added to video features
- Video embedding updated every 30 minutes as engagement data accumulates
- Thompson sampling for impression allocation: treat each video as a Bernoulli arm, sample from Beta(α, β) where α = positive interactions, β = negative

```python
class ThompsonSamplingExplorer:
    def __init__(self):
        self.alpha = defaultdict(lambda: 1.0)  # successes + 1
        self.beta  = defaultdict(lambda: 1.0)  # failures + 1

    def sample_score(self, video_id):
        return np.random.beta(self.alpha[video_id], self.beta[video_id])

    def update(self, video_id, reward):
        if reward > 0:
            self.alpha[video_id] += reward
        else:
            self.beta[video_id] += 1
```

**Phase 3: Bootstrapping from similar established videos**

When a new video enters the index, find its K nearest neighbors among established videos (by content embedding). The new video inherits a weighted average of their engagement-adjusted embeddings as a prior:

$$\mathbf{e}_{v_{new}}^{(0)} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{e}_{v_k}$$

This gives the new video a reasonable starting position in the embedding space before any real engagement data.

### New Creator Cold Start

New creator (zero subscriber, zero video history) — even harder:
- No channel-level reputation signal
- No subscriber base for initial distribution
- Strategy: content-based routing to topically relevant audiences + capped exploration budget per creator per day
- Viral detection (see Section 9) to amplify breakout new creator videos quickly

---

## 8. Watch Time vs Satisfaction

### The Divergence Problem

Watch time and satisfaction are correlated for high-quality content but systematically diverge for:
- **Clickbait:** high CTR, high watch time (sunk cost), low satisfaction
- **Extreme/outrage content:** high watch time (emotional activation), low self-reported satisfaction, high regret
- **Autoplay rabbit holes:** cumulative watch time accrues, but user reports not meaning to watch that much

### Clickbait Detection

**Signal: click → short watch = dissatisfaction indicator**

```python
def compute_satisfaction_proxy(click_event, watch_event, video_duration_s):
    if watch_event is None:
        return -1.0  # clicked but didn't watch at all
    watch_fraction = watch_event.watch_seconds / video_duration_s
    # Short watch after click on a long video = potential dissatisfaction
    if watch_fraction < 0.1 and video_duration_s > 120:
        return -0.5  # likely clicked misleading thumbnail
    return watch_fraction
```

**Thumbnail-content divergence score** (Section 6) fed as a feature to the regret prediction head.

### Satisfaction Surveys

YouTube deploys periodic in-app surveys: "Did you enjoy this video?" (1–5 stars). These are:
- Sparse (~1% of sessions)
- Used to calibrate the satisfaction proxy signal
- Direct training label for the "like" and "satisfaction" prediction heads

Survey labels are weighted more heavily than implicit signals in training — explicit feedback is more reliable.

### Regret Minimization

"Were you satisfied with the time you spent on YouTube today?" — a longer-horizon question than per-video satisfaction. Training a regret predictor:

**Regret label construction:**
- User watches video → within 30 minutes explicitly dislikes / clicks "not interested" / exits app abruptly = regret label = 1
- User watches video → returns to app within 2 hours = regret label = 0 (satisfied session)

**Long-term user health metrics (offline evaluation):**
- 30-day retention rate: are users coming back month over month?
- Self-reported wellbeing survey (annual, sampled)
- Subscription cancellation rate by recommendation quality segment

### Rabbit Hole Detection and Intervention

A rabbit hole is a sequence of recommendations where the user keeps watching increasingly extreme or narrow content.

**Detection:**
- Track topic distribution of user's session: if last 5 recommendations are all in the same narrow subtopic cluster, flag as rabbit hole risk
- Monitor engagement pattern: autoplay-driven (passive) vs active click-driven (intentional)

**Intervention:**
- Inject diversity: insert a video from a different topic cluster after N consecutive similar videos
- "Take a break" prompt after 45 minutes continuous watch
- Reduce autoplay probability for content clusters flagged for rabbit hole potential

---

## 9. Creator Ecosystem

### Long-Tail Creator Support

Without intervention, the recommendation system concentrates watch time on the top 0.1% of creators (power law distribution). New and mid-tier creators receive insufficient exposure to grow.

**Creator reach fairness metric:**

$$\text{Gini}_{creator} = \frac{\sum_{i,j} |r_i - r_j|}{2n \sum_i r_i}$$

where $r_i$ is the total reach (impressions) of creator $i$. Track this metric over time; a rising Gini indicates concentration.

**Mitigation strategies:**
- Exploration budget: reserve X% of total impressions for non-top-1% creators
- Subscriber notification: prioritize new videos from subscribed channels in the first hour
- Search surface: less biased toward engagement history, easier entry for new creators

### Viral Video Detection and Amplification

**Signals indicating breakout potential:**
- CTR significantly above creator baseline (> 2 std deviations)
- Watch percentage > 70% on first 100 impressions
- Share rate > 3% (threshold tuned per category)
- Rapid social graph spread (being shared across disconnected user clusters)

**Response pipeline:**

```
Upload → standard cold start →
  [viral signals fire] →
    increase ANN index weight (boost retrieval score) →
    push to trending module →
    widen distribution beyond immediate subscriber base
```

Viral detection must be fast (<30 min) to capture organic spread before it plateaus.

### Copyright Detection Integration

- Content ID fingerprinting runs in parallel to recommendation pipeline
- Copyright-matched content flagged before entering ANN index
- Re-monetization decisions (split revenue vs block vs mute audio) resolved before first recommendation
- Copyright status change propagates to index as video metadata update

---

## 10. Diversity and Serendipity

### The Filter Bubble Problem

A purely accuracy-maximizing recommender shows users only content similar to what they've watched before. Over time:
- User's topic distribution narrows
- Cross-topic discovery ceases
- Platform becomes an echo chamber
- Long-term user satisfaction (as measured by retention) decreases despite short-term engagement increase

### Maximal Marginal Relevance (MMR)

After ranking, apply MMR to the final slate to balance relevance with diversity:

$$\text{MMR}(v_i, S) = \lambda \cdot \text{relevance}(v_i) - (1 - \lambda) \cdot \max_{v_j \in S} \text{sim}(v_i, v_j)$$

where $S$ is the already-selected set. At each step, add the video maximizing MMR. λ ∈ [0, 1] controls exploration-exploitation tradeoff.

```python
def mmr_rerank(candidates, relevance_scores, similarity_matrix, lambda_=0.7, k=20):
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(k):
        if not selected:
            # First item: pure relevance
            best = max(remaining, key=lambda i: relevance_scores[i])
        else:
            # Subsequent items: balance relevance vs similarity to selected
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * relevance_scores[i] -
                    (1 - lambda_) * max(similarity_matrix[i][j] for j in selected)
                )
            )
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]
```

### Topic and Creator Diversification Rules

Applied in re-ranking layer:

| Rule | Implementation |
|---|---|
| Max 2 consecutive videos from same creator | Hard constraint in slate optimizer |
| Max 3 videos from same topic cluster in first 10 | Soft constraint (penalty) |
| At least 1 video from a topic not in last 7 days history | "Serendipity slot" allocation |
| At least 1 video from non-subscribed creator | Discovery slot |

### Familiar vs Novel Content Balance

Users differ in exploration preference. Estimate per-user exploration tolerance:
- **Novelty seeker:** high historical click rate on out-of-distribution recommendations
- **Comfort viewer:** low historical click rate on novel content, high on familiar

Adjust λ in MMR per user: higher λ for comfort viewers (more relevance), lower λ for novelty seekers (more diversity).

---

## 11. Failure Modes

### Watch-Time Optimization → Extremism Pipeline

**How it happens:** Extreme content (conspiracy theories, outrage, political polarization) generates longer, more emotionally activated viewing sessions. A pure watch-time optimizer learns to recommend progressively more extreme content because each step slightly increases expected watch time.

**Detection:** Track topic cluster transitions in user sessions. Flag chains that move toward known harmful clusters (policy team maintains topic cluster risk scores).

**Mitigation:**
- Add regret prediction head — extreme content has high regret rate
- Policy-enforced score penalty for videos from flagged content clusters
- Hard cap on consecutive recommendations from high-risk clusters
- Separate safety classifier score subtracted from ranking score

### Creator Gaming: Thumbnail Bait

**How it happens:** Creators learn that click-through rate drives distribution. They optimize thumbnails for clicks (dramatic faces, false urgency, misleading text) independent of actual content quality.

**Detection:** Thumbnail-content divergence score (Section 6). Monitor correlation between CTR and short-session-exit rate per creator.

**Mitigation:**
- Penalize high thumbnail-content divergence in ranking score
- Track creator-level history of clickbait signals; adjust baseline score
- Rate limit distribution for creators with repeated clickbait violations

### Feedback Loops Amplifying Popular Content

**How it happens:** Popular videos get more impressions → more engagement → higher ranking score → even more impressions. The rich-get-richer dynamic means once a video becomes popular, it stays popular regardless of whether better content exists.

**Detection:** Monitor Gini coefficient of watch-time distribution across videos over time. Rising Gini = increasing concentration.

**Mitigation:**
- Temporal decay: reduce ranking score boost from historical engagement as video ages (unless still generating strong engagement)
- Exploration budget (Section 9): reserve impressions for non-top content
- Diversified retrieval: ensure ANN retrieval returns a diverse set, not just the globally most popular

### New User Cold Start

**How it happens:** New user has no watch history. User tower has no signal. Recommendations are generic, irrelevant, and the user churns before the system learns their preferences.

**Mitigation:**
- Onboarding interest selection (explicit preferences at signup)
- Geographic and demographic priors (what do similar users in this region watch?)
- Session-based warm-up: infer preferences from first 3–5 interactions within the session (real-time context window)
- Multi-armed bandit to efficiently explore user preferences with minimal impressions

### Adversarial Content Evasion

**How it happens:** Bad actors learn the ranking signals and craft content that scores well (high watch time, CTR) while containing policy-violating material (misinformation, spam, coordinated inauthentic behavior).

**Detection:**
- Anomaly detection on engagement pattern (bot-like: all watches exactly 31 seconds, all from same IP subnet)
- Content classifier ensemble (separate from recommendation model)
- Human review escalation pipeline for flagged content

**Mitigation:**
- Ranking score includes trust score from integrity systems
- New creator probation: limited reach until manual trust threshold
- Ensemble of content safety classifiers (text, visual, audio) — harder to evade all simultaneously

---

## 12. Interview Angles

**Q1: Why use a two-tower model for retrieval instead of a single model that scores (user, video) pairs directly?**

A cross-encoder model scoring each (user, video) pair would require running the model 800M times per query — infeasible at <200ms latency. The two-tower architecture decouples the computation: video embeddings are pre-computed offline, and at serving time only the user embedding needs to be computed. Retrieval then reduces to a single MIPS (maximum inner product search) query against the pre-built ANN index. The trade-off is expressive power: two-tower models cannot capture fine-grained user-video interactions (because the towers never see each other's inputs). That's why retrieval (two-tower) is followed by re-ranking (cross-encoder / interaction-heavy model) in a cascade.

---

**Q2: How do you handle position bias in training data?**

Training data comes from logged impressions, which are position-biased: items shown at position 1 have a far higher probability of being clicked regardless of quality. Naive training treats all clicks equally and learns to predict position rather than relevance. Solutions:

1. **Inverse Propensity Weighting (IPW):** estimate the probability of examination at each position (propensity score) using randomization experiments, then up-weight clicks on lower positions. Corrects the bias but requires knowing the propensity model.
2. **Position as a feature (YouTube's approach):** add position as an input feature during training. The model learns to "explain away" position effects. At inference, set position = 0 (serving position) or use a learned bias-free representation.
3. **Randomization experiments:** for a fraction of traffic, shuffle recommendation order randomly. This gives unbiased click data for calibration.

---

**Q3: How do you prevent the recommendation system from creating filter bubbles?**

Filter bubbles emerge when the recommender only shows users content similar to their history. Mitigation at multiple layers:

- **Retrieval:** ensure ANN retrieval doesn't over-index on globally popular or historically similar content. Use diverse retrieval strategies (topic-constrained retrieval, trending content, followed-channel content as separate pipelines that are merged).
- **Re-ranking:** apply MMR to diversify the final slate by topic, creator, and format.
- **Exploration allocation:** reserve slots for novel topics and creators.
- **Long-term metrics:** track user topic distribution breadth over time. A narrowing distribution is an early signal before user churn.
- **Per-user novelty tuning:** users with high exploration tolerance get more diverse recommendations; users who prefer familiar content get more familiar content (but with a minimum diversity floor).

---

### Q4: A new video goes viral but the system is slow to amplify it. How do you fix this? [Medium]

The root cause is that ANN index updates are batched (e.g., hourly), and the ranking model relies on historical engagement features that haven't accumulated yet.

**Short-term fix:**
- Reduce ANN index update frequency to every 5–10 minutes for videos crossing a viral-signal threshold.
- Add a "viral signal" feature (rate of change of engagement, not absolute level) to the ranking model.
- Maintain a separate fast-path "trending videos" retrieval pipeline that bypasses the ANN index and directly injects viral candidates.

**Structural fix:**
- Streaming feature pipeline (Flink): update engagement statistics in near-real-time (seconds) rather than batch.
- Decouple cold-start exploration (Section 7 Phase 1) from ANN retrieval so new videos can receive impressions even before full index insertion.

---

**Cross-questions to expect:**

- *You cut the ANN index refresh to every 5 minutes for viral candidates. What's the cost you just signed up for, and does it even scale?* -> Frequent partial index rebuilds are expensive and fragment the index; do it for everything and you melt the retrieval tier. So the trigger has to be a narrow viral-signal gate, which means you're now maintaining a second, hotter path with its own failure modes and its own false-positive problem -- a video that spikes for a minute and dies still forces a rebuild. The fast path buys latency by adding operational surface, not for free.
- *A "rate of change of engagement" feature amplifies whatever is spiking. How is that not just an engine for manufactured virality and brigading?* -> It is exactly that risk -- velocity features reward coordinated early bursts, which is precisely what bot rings and brigades manufacture. You need the velocity signal gated by authenticity/spam scores and by audience diversity (organic virality spreads across unrelated clusters; manufactured virality spikes within one). Rewarding raw velocity without a genuineness check turns your fast path into an attack surface.

**Trap:** treating "slow to amplify" as purely a freshness bug. Some of that latency is a deliberate safety buffer -- instant amplification of anything spiking is how harmful content goes viral before any human or classifier can look at it. The lag you're being asked to remove is partly load-bearing.
**Q5: How does TikTok's For You Page differ architecturally from YouTube's homepage recommendation?**

| Dimension | YouTube Homepage | TikTok For You Page |
|---|---|---|
| User intent | Browsing, looking for something specific | Pure serendipitous discovery |
| Logged-in rate | ~70% logged-in | Large anonymous/thin-identity fraction |
| Engagement signal | Multi-dimensional (watch%, like, subscribe) | Primarily watch-through rate + scroll-away |
| Content length | Heterogeneous (30s to 4 hours) | Homogeneous short-form (15s–3 min) |
| Session model | Discrete browsing sessions | Infinite scroll, autoplay |
| Cold start urgency | Less critical (users have search as fallback) | Critical (FYP is primary discovery surface) |

TikTok's Monolith system (2022) uses a large embedding model without embedding tables, enabling efficient online learning at inference time — the model updates user representations in real-time within a session, much faster than YouTube's batch retraining cycle. This is crucial for TikTok's "first 5 videos decide your FYP" behavior.

---

### Q6: How would you detect and mitigate the watch-time → extremism pipeline? [Hard]

**Detection:**
1. Map the video corpus onto a topic-risk graph: run community detection on the video-to-video recommendation graph; identify clusters associated with harmful content (validated by policy team).
2. Track user session trajectories through this graph. A trajectory that monotonically moves toward high-risk clusters across sessions is a radicalization signal.
3. Monitor per-cluster regret prediction scores — extreme content has systematically high regret.

**Mitigation:**
1. Policy-enforced score floor/ceiling: videos in flagged clusters have a hard cap on ranking score, regardless of predicted watch time.
2. Trajectory intervention: if a user's last N sessions are moving toward high-risk clusters, inject topic-diverse content (breaking the reinforcement cycle).
3. Multi-task optimization: include regret and safety classifier score as explicit negative terms in the composite ranking score, so the ranker is penalized for recommending high-risk content even if it generates watch time.
4. Audit trail: maintain a log of recommendations that led to harmful content consumption for post-hoc analysis and policy tuning.

---

**Cross-questions to expect:**

- *You add a safety-classifier score as a negative term in the composite ranking objective. A large watch-time gain can still buy out a small safety penalty. Isn't that the same weighted-sum failure as engagement-vs-wellbeing?* -> Yes -- a soft penalty in a weighted sum means enough predicted watch time always outbids the safety term, so borderline-harmful-but-magnetic content still surfaces. For content classes where the harm is categorical you need a hard score ceiling or an eligibility gate, not a term the objective can trade away. The weighted-sum framing is appropriate for taste, not for safety floors.
- *Your radicalization signal is "trajectory monotonically moving toward high-risk clusters." What's the base-rate problem that makes that alarm mostly wrong?* -> Almost everyone who watches one video in a risk-adjacent cluster does not radicalize, so even a fairly accurate trajectory detector fires overwhelmingly on false positives given how rare the real outcome is. Intervening on all of them (injecting off-topic content) degrades the experience for a huge benign majority to catch a tiny true-positive set -- you have to be explicit about that precision/harm trade, not just claim detection.

**Trap:** trusting the video-to-video recommendation graph to define "high-risk clusters" when that graph is itself produced by the recommender you're auditing. The clusters encode the system's own behavior, so you can end up measuring and correcting the recommender against its own biases -- the ground truth for what is harmful has to come from outside the engagement graph (policy/human labels), or the audit is circular.
### Q7: How would you evaluate recommendation quality offline, given that you can only observe engaged-with content, not all content the user would have liked? [Hard]

This is the **missing data / counterfactual evaluation problem** — the fundamental challenge in offline recommendation evaluation.

**Approaches:**

1. **Held-out watch data:** hold out the last K videos a user watched from their history, train on the rest, and evaluate whether the held-out videos appear in the top-N recommendations. This measures retrieval recall but suffers from selection bias (we only hold out watched content, not unwatched-but-would-have-enjoyed content).

2. **Inverse Propensity-Weighted (IPW) offline evaluation:** weight held-out interactions by the inverse of the probability they were shown in the original system. This corrects for the fact that lower-positioned items had lower examination probability.

3. **Interleaving experiments:** compare two models by interleaving their recommendations in real traffic and tracking which model's recommendations are clicked more often. More sensitive than A/B tests with the same traffic volume.

4. **Counterfactual evaluation with logged bandit feedback:** if a fraction of traffic uses a randomized policy, evaluate new models against this unbiased logged data using doubly-robust estimation.

5. **Long-term simulation:** train a user behavior simulator on historical data, then evaluate recommendations in simulation before deploying. Captures feedback loop effects that short A/B tests miss (but simulator fidelity is the bottleneck).

**Recommendation:** use held-out recall for fast offline iteration, IPW-corrected metrics for final offline evaluation, and interleaving for production gating — because interleaving is sensitive enough to detect small improvements with less traffic than a full A/B test.

---

**Cross-questions to expect:**

- *IPW corrects for exposure by dividing by propensity. What happens to the variance when some shown items had tiny propensities, and why does that quietly wreck your "unbiased" estimate?* -> Small propensities in the denominator produce enormous weights, so a handful of rarely-shown items dominate the estimate and the variance explodes -- the estimator is unbiased in expectation but so noisy per-run that it can't rank two models reliably. You clip or cap weights to control variance, which reintroduces bias. IPW trades the selection bias you can see for a variance problem you can't, and "unbiased" on the slide hides that.
- *You propose a user-behavior simulator to escape logged-data bias. Why is that often worse than the bias it replaces?* -> The simulator is trained on the same logged, biased data, so it learns the current system's blind spots and then rewards models that agree with it -- you get a confident number that measures resemblance to today's policy, not user value. Simulator fidelity is unfalsifiable offline (you can't check counterfactual watches), so a good simulation score can mask a model that's worse in production. It's most dangerous precisely because it looks like it solved the counterfactual problem.

**Trap:** reporting held-out recall as a quality metric. Held-out watched items are a biased sample of "would have liked" -- the content the user would have loved but was never shown is invisible to the metric, so a model can score high on recall by predicting what the old system already surfaced while being strictly worse at discovery, the thing you actually care about.
## References

- Covington, Adams, Sargin. *Deep Neural Networks for YouTube Recommendations*. RecSys 2016.
- Liu et al. *Monolith: Real Time Recommendation System With Collisionless Embedding Table*. ByteDance, 2022.
- Zhao et al. *Recommending What Video to Watch Next: A Multitask Ranking System*. RecSys 2019.
- Steck. *Calibrated Recommendations*. RecSys 2018. (diversity / filter bubble)
- Ribeiro et al. *Auditing Radicalization Pathways on YouTube*. FAT* 2020.
- Dean et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017.
- Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs (FAISS)*. IEEE T-BD 2021.

## Flashcards

**Why do two-tower user/video embeddings get L2-normalized before the dot product at retrieval time?** #flashcard
Normalizing to unit vectors makes the dot product equivalent to cosine similarity, so ANN search ranks by directional alignment (taste match) rather than raw magnitude; without it, videos with larger-norm embeddings — often just the ones seen more during training — would score higher regardless of actual relevance.

**Why does in-batch softmax training for the two-tower model need hard-negative mining on top of random in-batch negatives?** #flashcard
Random in-batch negatives are almost always "easy" — a random video is trivially distinguishable from the one a user actually watched — so the model quickly saturates on them and stops learning fine-grained distinctions; hard negatives (videos similar in topic/creator/embedding space to the positive but not watched) force the model to learn the subtler signals that actually separate relevant from near-relevant content.

**Why does YouTube's position-bias correction set position=0 at inference time after training position as an explicit feature?** #flashcard
Training with position as a feature lets the model learn how much of a click is attributable to slot placement versus true relevance, isolating that effect during training; at inference there's no real position yet to condition on, so fixing position=0 (the best slot) asks the model for its estimate of relevance independent of position bias, giving an unbiased ranking signal instead of just reproducing the current UI's positional click pattern.

**Why does new-video cold start use a multi-phase strategy instead of immediately serving on predicted engagement?** #flashcard
A brand-new video has no engagement history, so any engagement prediction is pure guesswork; the phased approach — content-only embedding, then a small guaranteed exploration budget, then Thompson-sampling-based traffic allocation as real signal arrives, then embedding refinement via KNN averaging from similar videos — lets the system safely gather enough real signal to trust before fully committing traffic, avoiding both starving good new content and over-investing in bad content.

**Why does Thompson sampling fit cold-start video exploration better than a fixed exploration percentage?** #flashcard
A fixed percentage (e.g. "always give new videos 5% of traffic") wastes impressions on videos that are quickly revealed to be poor and under-explores promising ones; Thompson sampling maintains a belief distribution over each video's true engagement rate and allocates traffic probabilistically in proportion to the odds it's actually good, so exploration budget naturally concentrates on videos worth learning more about.

**Why is watch-time alone a bad objective to directly optimize for recommendation ranking?** #flashcard
Watch-time can be maximized by content that's addictive rather than genuinely satisfying — outrage bait, autoplay-triggering cliffhangers, borderline extreme content that keeps users watching out of compulsion, not enjoyment; optimizing it directly creates a feedback loop toward this content, which is why production systems blend in explicit satisfaction proxies (surveys, "would recommend" signals) and regret signals rather than watch-time in isolation.

**Why is MMR-based re-ranking still needed after a multi-task ranking model has already scored every candidate?** #flashcard
The ranking model scores each video independently by predicted engagement, so the top-k by score alone can be near-duplicates (same creator, same topic) that individually look great but collectively provide no variety; MMR explicitly penalizes similarity to already-selected videos in the final list, trading a small amount of per-item predicted score for diversity that the independent per-item ranker structurally cannot represent.

**Why does the platform track the Gini coefficient of creator reach instead of just optimizing aggregate watch-time?** #flashcard
Aggregate watch-time can be maximized while impressions concentrate on a small set of already-viral creators, since popular content reliably keeps engagement high; the Gini coefficient exposes that concentration directly, letting the platform detect and correct for reach inequality (creator ecosystem health) that a pure engagement metric would never flag as a problem.

**Why does the video recommender need explicit feedback-loop detection rather than trusting that engagement-trained rankings are self-correcting?** #flashcard
Engagement-trained models reinforce whatever they've already surfaced — content shown more gets more clicks, which further raises its predicted score, regardless of whether it's actually the best content — so left unchecked, the system drifts toward funneling all traffic through a shrinking set of already-popular videos; explicit monitoring for rising concentration or extremism-pipeline patterns is needed because the training loop itself has no mechanism to notice or reverse this drift.

**Why is offline counterfactual/interleaving evaluation preferred over naive held-out watch prediction accuracy for comparing ranking models?** #flashcard
Held-out accuracy only measures how well a model predicts what users did under the *old* ranking policy, not how they'd behave under a new one — since what gets shown determines what can be watched at all, a model's held-out score says little about its real-world performance; interleaving/counterfactual methods compare policies by mixing their outputs for the same users or reweighting logged data, directly estimating how the new policy would perform instead of just how well it fits old, policy-biased logs.
