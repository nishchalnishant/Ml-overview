---
module: Production Ml
topic: System Design
subtopic: Content Moderation System
status: unread
tags: [productionml, ml, system-design-content-moderati]
---
# Content Moderation System Design

End-to-end ML system for detecting policy-violating content at platform scale. Canonical system design question at Meta, Google/YouTube, Twitter/X, TikTok, and any UGC platform.

**Scale:** 500M+ posts/day across text, image, video, and audio; <1s action on CSAM/imminent violence; global policy variation across 100+ markets; human review queues processing millions of items/week.

---

## 1. Problem Framing

### Clarifying Questions

- **What content types?** Text (posts, comments, DMs), images, video (live + recorded), audio, mixed-media posts
- **What violation categories?** CSAM, imminent violence/terrorism, hate speech, harassment, spam/scams, misinformation, self-harm, synthetic media/deepfakes
- **Proactive vs reactive?** Proactive (scan every upload) vs reactive (user reports only) vs hybrid
- **What action space?** Remove, label/interstitial, reduce distribution, suspend account, refer to law enforcement
- **Geographic scope?** Global → policy must reflect local laws (NetzDG in Germany, DSA in EU, FOSTA-SESTA in US)
- **Who gets harmed by errors?** False negative: victim not protected, violator remains; false positive: legitimate speaker silenced
- **Appeal mechanism?** Yes/no, SLA on appeals, who reviews appeals
- **Live content?** Live streaming requires real-time action with near-zero latency and no rewind

### Why Content Moderation is Harder Than Fraud Detection

| Dimension | Fraud Detection | Content Moderation |
|---|---|---|
| Label clarity | Binary (fraud / not fraud) | Ambiguous — same post is hate speech in one context, legitimate criticism in another |
| Context dependency | Transaction features are numeric | Meaning depends on cultural context, tone, platform norms, current events |
| Error asymmetry | Both FP/FN have monetary cost | FP silences legitimate speech; FN allows harm; both carry political cost |
| Policy evolution | Fraud patterns evolve slowly | Community standards updated frequently, especially around elections |
| Modality diversity | Primarily tabular | Text, image, video, audio, and their combinations |
| Adversarial creativity | Financial incentive to evade | Social/political incentive + highly creative adversaries |
| Human cost | Analyst reviews alerts | Moderators exposed to traumatic content at scale |
| Legal complexity | Regulatory but clear | Jurisdiction-specific, politically contentious |

### Business Metrics

**Prevalence** — fraction of content on platform violating policy:
$$\text{Prevalence} = \frac{\text{violating content views}}{\text{total content views}}$$
Measured by stratified random sampling + expert review (not ML output — ML output is biased).

**Proactive Rate (PR)** — fraction of removed content found by platform before user report:
$$\text{PR} = \frac{\text{actioned before first report}}{\text{total actioned}}$$
Target: >95% for CSAM, >90% for terrorism. Meta reports this in its Community Standards Enforcement Report.

**Over-removal Rate (ORR)** — fraction of removals that were incorrect:
$$\text{ORR} = \frac{\text{incorrectly removed pieces}}{\text{total removed}}$$
Measured through appeals data and random audit. High ORR signals political/PR risk.

**Metric trade-off:** Maximizing PR (proactive detection) tends to increase ORR (false positives). Threshold decision is a policy call, not a pure ML call.

---

## 2. Scale Requirements

| Dimension | Requirement |
|---|---|
| Throughput | 500M+ posts/day (~6,000/sec sustained, 30K/sec peak) |
| Text classification | <100ms P99 latency |
| Image classification | <200ms P99 latency |
| Video classification | <30s for 1-min video (async OK) |
| CSAM/imminent violence action | <1 second from upload to hold |
| Hash matching (PhotoDNA) | <10ms at ingestion |
| Human review queue SLA | 24h for high-severity, 72h for borderline |
| Live content | Near-real-time frame sampling, <5s to action |
| Global coverage | 100+ languages, including low-resource languages |

---

## 3. System Architecture

### Proactive Pipeline (scan on upload)

```
Content Upload
     │
     ▼
┌──────────────────┐
│  Ingestion &     │──── Extract: text, image frames, audio track
│  Media Parsing   │──── Generate content_id, store in object store
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Hash Matching   │──── PhotoDNA (CSAM), MD5/SHA for known-bad URLs
│  (Perceptual)    │──── Near-duplicate detection (pHash, SimHash)
│  <10ms           │──── Match → immediate HOLD + NCMEC report
└────────┬─────────┘
         │ (no hash match)
         ▼
┌──────────────────────────────────────────┐
│              ML Classifier Layer          │
│                                          │
│  ┌────────────┐  ┌────────────────────┐  │
│  │   Text     │  │  Image / Video     │  │
│  │ Classifier │  │    Classifier      │  │
│  │ (LLM-based)│  │  (CNN + ViT)       │  │
│  └─────┬──────┘  └────────┬───────────┘  │
│        │                  │              │
│        ▼                  ▼              │
│  ┌──────────────────────────────────┐   │
│  │    Multi-Modal Fusion Model      │   │
│  │    (cross-modal consistency)     │   │
│  └────────────────┬─────────────────┘   │
└───────────────────┼─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    Policy Engine    │──── Apply jurisdiction rules
         │                     │──── Account history lookup
         │                     │──── Severity tier assignment
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Decision Routing   │
         └──┬─────┬──────┬─────┘
            │     │      │
     AUTO   │  HOLD│  HUMAN│
     REMOVE │      │ REVIEW│
            ▼      ▼      ▼
        Remove  Queue  Distribute
        + Log   (w/    normally
                priority)
                    │
                    ▼
         ┌──────────────────┐
         │  Human Review    │──── Prioritized queue
         │  Interface       │──── Active learning sample selection
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Feedback Loop   │──── Reviewer decisions → training labels
         │                  │──── Appeals → corrective labels
         └──────────────────┘
```

### Reactive Pipeline (user reports)

```
User Report Submitted
        │
        ▼
┌─────────────────┐
│  Report         │──── Report type, reporter history, prior reports on item
│  Enrichment     │──── Is this content already in moderation queue?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Re-score with  │──── Re-run classifiers with report signal as feature
│  Report Context │──── Multiple reports → higher severity signal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Queue Priority │──── Merge with proactive queue
│  Update         │──── Boost priority by report volume + severity
└─────────────────┘
```

---

## 4. Multi-Modal Detection

### 4.1 Text Classification

**Architecture: Fine-tuned LLM**

Hate speech detection cannot be solved by keyword matching — context is everything ("I'm going to kill this presentation" vs. genuine threat). Modern approach: fine-tuned multilingual LLMs.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HateSpeechClassifier:
    """
    Fine-tuned XLM-RoBERTa for multilingual hate speech detection.
    Multi-label: a post can be hate speech + harassment simultaneously.
    """
    VIOLATION_LABELS = [
        'hate_speech', 'harassment', 'violence_threat',
        'self_harm', 'spam', 'misinformation', 'sexual_content'
    ]

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.VIOLATION_LABELS),
            problem_type='multi_label_classification'
        )
        self.model.eval()

    def score(self, text: str) -> dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors='pt',
            max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        return dict(zip(self.VIOLATION_LABELS, probs))
```

**Key design decisions:**
- XLM-RoBERTa over English-only BERT: covers 100 languages in one model
- Multi-label head: real content often violates multiple policies simultaneously
- Temperature calibration post-training: raw sigmoid scores are poorly calibrated
- Context window: include conversation thread for comments, not just isolated post

**Distilled inference model:**
Production uses a smaller distilled model (6-layer) for <50ms latency. Full model runs offline for hard cases and label generation.

### 4.2 Image Classification

**Two-stage pipeline:**

Stage 1: Fast binary classifiers (one per category, run in parallel)
- CNN backbone (ResNet-50 or EfficientNet-B3) fine-tuned on policy violation data
- Outputs: P(nudity), P(graphic_violence), P(hate_symbol), P(spam_visual)
- <50ms per image

Stage 2: Vision Transformer for nuanced decisions
- ViT-L/16 with policy-specific fine-tuning
- Used for borderline cases where Stage 1 score is 0.3–0.7
- Context: image + OCR text overlay + alt text

```python
class ImageModerationPipeline:
    def __init__(self):
        self.fast_cnn = load_efficientnet_classifier()   # stage 1
        self.vit_model = load_vit_classifier()            # stage 2
        self.ocr_engine = load_ocr()                      # text in image

    def classify(self, image: bytes) -> ModerationResult:
        img_tensor = preprocess(image)

        # Stage 1: fast classification
        fast_scores = self.fast_cnn(img_tensor)

        # Early exit: clear violation
        if max(fast_scores.values()) > 0.95:
            return ModerationResult(scores=fast_scores, tier='auto_action')

        # Early exit: clearly clean
        if max(fast_scores.values()) < 0.1:
            return ModerationResult(scores=fast_scores, tier='pass')

        # Stage 2: expensive model for borderline
        ocr_text = self.ocr_engine.extract(image)
        vit_scores = self.vit_model(img_tensor, text_context=ocr_text)
        return ModerationResult(scores=vit_scores, tier='borderline')
```

### 4.3 Video Classification

Video cannot be fully scanned frame-by-frame at 500M/day scale. Hierarchical sampling:

```
Video Upload
     │
     ▼
┌────────────────┐
│ Metadata check │──── Duration, format, upload history of account
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ I-frame sample │──── Sample 1 frame/second → image classifier
│ (fast pass)    │──── Flag if any frame scores >0.8
└────────┬───────┘
         │ (no flag)
         ▼
┌────────────────┐
│ Audio track    │──── Whisper transcription → text classifier
│ transcription  │──── Hate speech in audio not visible in frames
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Temporal model │──── LSTM/Transformer over frame sequence
│ (sequence)     │──── Catches content only visible in motion (e.g., violence)
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Multi-modal    │──── Fuse: visual score + audio/text score + metadata
│ fusion         │──── Cross-modal consistency check (see §4.4)
└────────────────┘
```

**Frame sampling strategy:**
- 1 fps for most content (90% of compute budget)
- 5 fps if initial score > 0.3
- All frames if score > 0.6 or account is flagged

### 4.4 Cross-Modal Consistency

Adversaries post benign images with violating text overlays, or pair hateful audio with innocent video. Cross-modal consistency catches these patterns.

```python
def cross_modal_consistency_score(
    text_score: float,
    image_score: float,
    audio_score: float
) -> float:
    """
    High inconsistency = one modality is clean while another is violating.
    This is a strong signal of adversarial evasion attempt.
    """
    scores = [text_score, image_score, audio_score]
    max_score = max(scores)
    min_score = min(scores)
    inconsistency = max_score - min_score

    # If max signal is high but others are low → suspicious
    if max_score > 0.7 and inconsistency > 0.5:
        # Boost overall score: likely evasion attempt
        return max_score * 1.2  # capped at 1.0
    return max_score
```

---

## 5. Severity Tiers and Routing

### Tier Definitions

| Tier | Content Type | Action | SLA |
|---|---|---|---|
| **Zero-tolerance** | CSAM, live terrorism, imminent credible violence threat | Immediate auto-remove + account suspend + legal referral | <1 second |
| **High-severity** | Hate speech (explicit), non-live terrorism glorification, graphic violence | Hold (restrict visibility) + expedited human review | 2 hours |
| **Medium-severity** | Borderline hate speech, harassment, adult content (where policy applies) | Full visibility + human review | 24 hours |
| **Low-severity / Spam** | Automated spam, scam links, coordinated inauthentic behavior | Auto-remove or throttle + no human review | Immediate |
| **Borderline / Context-dependent** | Political speech, satire, news reporting of violence | Distribute normally + low-priority queue | 72 hours |

### Routing Logic

```python
class PolicyRouter:
    ZERO_TOLERANCE_THRESHOLD = 0.85
    HIGH_SEVERITY_THRESHOLD = 0.65
    REVIEW_THRESHOLD = 0.30

    def route(self, scores: dict, account_history: AccountHistory) -> RoutingDecision:
        csam_score = scores.get('csam', 0)
        violence_score = scores.get('imminent_violence', 0)
        hate_score = scores.get('hate_speech', 0)
        spam_score = scores.get('spam', 0)

        # Zero-tolerance: no human review needed, act immediately
        if csam_score > self.ZERO_TOLERANCE_THRESHOLD:
            return RoutingDecision(
                action='REMOVE',
                severity='zero_tolerance',
                legal_referral=True,   # NCMEC for CSAM
                suspend_account=True
            )

        if violence_score > self.ZERO_TOLERANCE_THRESHOLD:
            return RoutingDecision(action='REMOVE', severity='zero_tolerance')

        # High severity: hold content, expedited review
        if hate_score > self.HIGH_SEVERITY_THRESHOLD:
            priority = self._compute_priority(scores, account_history)
            return RoutingDecision(
                action='HOLD',
                severity='high',
                queue_priority=priority
            )

        # Spam: auto-remove, no review needed
        if spam_score > self.HIGH_SEVERITY_THRESHOLD:
            return RoutingDecision(action='REMOVE', severity='spam', log_only=True)

        # Borderline: send to human review with low priority
        if any(s > self.REVIEW_THRESHOLD for s in scores.values()):
            return RoutingDecision(
                action='DISTRIBUTE',
                severity='borderline',
                queue_priority='low'
            )

        return RoutingDecision(action='DISTRIBUTE', severity='clean')

    def _compute_priority(self, scores, history):
        base_priority = max(scores.values())
        # Boost priority for repeat violators
        if history.prior_violations > 2:
            base_priority *= 1.3
        # Boost for high-reach accounts
        if history.follower_count > 100_000:
            base_priority *= 1.2
        return min(base_priority, 1.0)
```

---

## 6. Perceptual Hashing and Known-Bad Databases

### PhotoDNA for CSAM

PhotoDNA (developed by Microsoft, operated with NCMEC) converts an image into a "hash" that is robust to resizing, format conversion, and minor edits. Unlike cryptographic hashes (where one pixel change = completely different hash), PhotoDNA hashes of visually similar images are numerically close.

```
Image → Greyscale → Resize to 144×144
      → Divide into 36 cells of 24×24
      → DCT per cell
      → Quantize DCT coefficients → 144-byte hash

Distance(hash_A, hash_B) < threshold → visually similar → potential match
```

**At ingestion (before any ML scoring):**

```python
class PhotoDNAMatcher:
    def __init__(self, known_csam_hashes: np.ndarray, threshold: float = 75.0):
        # known_csam_hashes: [N, 144] array of known CSAM perceptual hashes
        self.known_hashes = known_csam_hashes
        self.threshold = threshold

    def check(self, image_bytes: bytes) -> MatchResult:
        query_hash = compute_photodna_hash(image_bytes)  # 144-byte hash
        # Vectorized L2 distance against full database
        distances = np.linalg.norm(self.known_hashes - query_hash, axis=1)
        min_distance = distances.min()
        match_idx = distances.argmin()

        if min_distance < self.threshold:
            return MatchResult(is_match=True, distance=min_distance, db_id=match_idx)
        return MatchResult(is_match=False)
```

**Scale:** At 500M images/day, brute-force L2 over millions of hashes is infeasible. Use approximate nearest-neighbor (FAISS with IVF index) for sub-millisecond lookup.

### Near-Duplicate Detection for Evasion

Adversaries add noise, rotate, crop, or overlay text to evade exact/near-exact matching. Defense layers:

| Evasion | Defense |
|---|---|
| Crop/resize | PhotoDNA is crop-robust (DCT over fixed grid) |
| Slight brightness/contrast change | PhotoDNA DCT quantization absorbs small changes |
| Adding text overlay | OCR extraction + separate text hashing |
| Screenshot of screenshot | Multi-level hash comparison; quality degrades but hash stays close |
| Video: extract single frame from CSAM video | Run PhotoDNA on I-frames independently |
| Mirroring/flipping | Generate augmented hashes (flip, rotate 90°) at indexing time |

**Adversarial robustness benchmark:** Measure true positive rate at 1% FPR under defined attack set (50 adversarial transformations). PhotoDNA degrades gracefully vs. cryptographic hashing which breaks completely.

---

## 7. Human-in-the-Loop

### Queue Architecture

Human reviewers are not a fallback — they are the ground truth source. The system exists to make human review efficient and safe.

```
High-volume ML output
         │
         ▼
┌────────────────────────────────────────┐
│         Priority Queue                  │
│                                        │
│  P0: Zero-tolerance (legal review)     │──── Specialist reviewers
│  P1: High-severity (2h SLA)           │──── Senior moderators
│  P2: Active learning samples           │──── Any reviewer
│  P3: Borderline (72h SLA)             │──── General review pool
│  P4: Appeals                           │──── Appeals specialists
└────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Reviewer UI     │──── Context: account history, prior reports
         │                  │──── Policy reference panel
         │                  │──── Translation support
         │                  │──── Psychological support prompt
         └──────────────────┘
```

### Active Learning for Queue Prioritization

Not all borderline content is equally informative for model improvement. Active learning selects items that will most improve the model:

```python
def active_learning_score(
    model_confidence: float,
    policy_category: str,
    language: str,
    last_policy_update: datetime
) -> float:
    """
    Score used to prioritize items for human review queue for label collection.
    High score = item is more useful for model training.
    """
    # Uncertainty sampling: items near decision boundary are most informative
    uncertainty = 1 - abs(2 * model_confidence - 1)  # 0 at 0/1, 1 at 0.5

    # Boost for low-resource languages (less training data)
    lang_boost = 2.0 if language in LOW_RESOURCE_LANGUAGES else 1.0

    # Boost for recently updated policies (existing labels may be stale)
    days_since_update = (datetime.now() - last_policy_update).days
    policy_freshness_boost = 1.5 if days_since_update < 30 else 1.0

    return uncertainty * lang_boost * policy_freshness_boost
```

### Inter-Annotator Agreement

Policy is ambiguous. Measure disagreement to find hard cases and policy gaps:

```python
from sklearn.metrics import cohen_kappa_score

def compute_iaa_report(annotations: list[Annotation]) -> IAAReport:
    """
    For each item reviewed by multiple annotators, compute pairwise agreement.
    Low kappa → policy is ambiguous for this content type → policy clarification needed.
    """
    # Items with kappa < 0.4 are "genuinely ambiguous" — flag for policy team
    kappas = []
    for item_id, item_annotations in group_by_item(annotations):
        if len(item_annotations) >= 2:
            for r1, r2 in combinations(item_annotations, 2):
                kappa = cohen_kappa_score(r1.labels, r2.labels)
                kappas.append({'item_id': item_id, 'kappa': kappa})

    low_agreement = [k for k in kappas if k['kappa'] < 0.4]
    return IAAReport(
        overall_kappa=np.mean([k['kappa'] for k in kappas]),
        ambiguous_items=low_agreement
    )
```

### Policy Drift

Community standards are updated (especially pre-election). When policy changes:
1. Existing model is wrong — it learned old policy
2. Historical labels are wrong — they reflect old policy
3. Both training data and model must be updated

**Handling policy drift:**
- Policy version tag on every training label
- When policy changes, relabel affected sample with new policy before retraining
- Use policy change date as a feature cutoff: don't mix pre/post-policy labels
- Fast retrain cycle for high-impact policy changes (days, not weeks)

### Annotator Trauma and Support

Moderators review the worst content on the internet at scale. This is a known harm (documented in lawsuits against major platforms).

**System-level mitigations:**
- Exposure limits: no moderator reviews CSAM/graphic violence >4h/day
- Greyscale/blurred preview mode: show blurred image unless reviewer confirms
- Queue assignment: new moderators not assigned to CSAM queue
- Mandatory breaks enforced by the review UI after N consecutive graphic items
- Escalation path: reviewer can escalate without making a decision (no decision pressure)

---

## 8. Adversarial Arms Race

### Common Evasion Techniques

| Technique | Description | Detection |
|---|---|---|
| Leetspeak | "h4t3 sp33ch" instead of "hate speech" | Normalization layer before classifier; character-level model |
| Homoglyph substitution | Cyrillic "а" instead of Latin "a" | Unicode normalization (NFKC) |
| Coded language | "skittles" for racial slur (coded languages evolve weekly) | Embedding similarity; human monitoring of emerging slang |
| Image text overlay | Hateful text rendered as image to avoid text classifier | OCR pipeline on all images |
| Steganography | Hidden messages in image LSBs | Statistical noise analysis (computationally expensive, used for high-risk accounts) |
| Video overlays | Violating frame spliced at timestamp unlikely to be sampled | Adaptive sampling based on motion/scene change detection |
| Dog whistles | "1488", "boogaloo" — known only to in-group | Maintain constantly-updated glossary; cluster-based detection of newly emerging terms |
| Context splitting | Split violating content across multiple posts (reassembled by followers) | Account-level analysis, not just post-level |

### Network-Level Detection: Coordinated Inauthentic Behavior (CIB)

Individual posts may be borderline, but coordinated networks of accounts amplifying the same content at scale is itself a policy violation (Meta's "coordinated inauthentic behavior" policy).

```python
class CIBDetector:
    """
    Detect networks of accounts coordinating to amplify borderline content.
    Signal: many accounts post similar content within short time window.
    """
    def detect_coordinated_campaign(
        self,
        posts: list[Post],
        window_hours: int = 24
    ) -> list[CoordinationCluster]:
        # Step 1: Cluster posts by text similarity (MinHash LSH)
        post_clusters = self.lsh_cluster(posts, similarity_threshold=0.8)

        suspicious_clusters = []
        for cluster in post_clusters:
            if len(cluster) < 10:  # too small to be coordinated
                continue

            # Step 2: Check if accounts are newly created, low-activity, similar behavior
            accounts = [p.account for p in cluster]
            account_features = self.extract_account_features(accounts)

            coordination_score = self.coordination_classifier.predict(account_features)
            if coordination_score > 0.7:
                suspicious_clusters.append(
                    CoordinationCluster(posts=cluster, score=coordination_score)
                )

        return suspicious_clusters
```

### Policy Update Velocity

Adversaries adapt to policy faster than the model can retrain. Mitigation:
- Maintain rules layer (fast to update, no retraining) for emerging evasions
- Weekly model updates for trending evasion patterns (not monthly)
- Human monitoring of adversarial forums (4chan, Telegram) to get ahead of new techniques
- Red team: internal team attempts to evade current model; findings drive rule/model updates

---

## 9. Evaluation

### Prevalence Estimation (the Hard Part)

ML classifier outputs cannot be used directly to estimate prevalence — the classifier is biased by its training distribution and threshold. Correct approach: stratified random sampling + expert human review.

```python
class PrevalenceEstimator:
    """
    Estimate true prevalence using stratified sampling.
    Stratify by ML score decile to get efficient sampling (over-sample high-score content).
    """
    def estimate(
        self,
        population_score_distribution: np.ndarray,  # ML scores for all content
        stratum_sizes: dict[str, int],               # {score_decile: N_sampled}
        stratum_labels: dict[str, list[int]]         # {score_decile: [0/1 labels from human review]}
    ) -> PrevalenceEstimate:
        total_content = len(population_score_distribution)
        weighted_prevalence = 0.0
        variance = 0.0

        for decile, labels in stratum_labels.items():
            # Fraction of population in this decile
            decile_fraction = np.mean(
                (population_score_distribution >= decile * 0.1) &
                (population_score_distribution < (decile + 1) * 0.1)
            )
            # Prevalence within this decile (from human review)
            decile_prevalence = np.mean(labels)
            decile_variance = decile_prevalence * (1 - decile_prevalence) / len(labels)

            weighted_prevalence += decile_fraction * decile_prevalence
            variance += (decile_fraction ** 2) * decile_variance

        ci_95 = 1.96 * np.sqrt(variance)
        return PrevalenceEstimate(
            prevalence=weighted_prevalence,
            ci_lower=weighted_prevalence - ci_95,
            ci_upper=weighted_prevalence + ci_95
        )
```

### Metrics Dashboard

| Metric | Target | Measurement Method |
|---|---|---|
| Prevalence (hate speech) | <0.03% of views | Stratified sampling + expert review (quarterly) |
| Proactive Rate (CSAM) | >99% | Fraction actioned before first report |
| Proactive Rate (terrorism) | >95% | Fraction actioned before first report |
| Over-removal Rate | <5% | Appeals granted / total removals |
| Time to action (P99, CSAM) | <1 second | System telemetry |
| Human review SLA (P1) | >95% within 2 hours | Queue metrics |
| Inter-annotator agreement | κ > 0.7 | Weekly IAA audit |
| Low-resource language coverage | <10% gap vs. English | Per-language prevalence audit |

### False Positive Measurement via Appeals

Appeals are a biased but useful signal. Not all false positives appeal (many users simply leave). Correction:

$$\text{estimated FPR} = \frac{\text{appeals granted}}{\text{appeals granted} + \text{removals that should have been removed}}$$

The denominator requires a random audit sample. Platform cannot rely on appeals rate alone as a proxy for FPR.

---

## 10. Failure Modes

### Over-Removal of Legitimate Political Speech

**Problem:** Hate speech models trained on Western English data over-remove political speech in other cultural/linguistic contexts. A model trained to detect anti-Black racism may flag counter-speech ("Black Lives Matter" + discussion of racial slurs in educational context).

**Real example:** Meta's Oversight Board repeatedly found that the platform over-removed Palestinian political speech during the 2021 conflict while under-removing incitement to violence in the other direction. Root cause: training data and policy guidelines written by teams in a single cultural context.

**Mitigation:**
- Separate models per cultural/political context (or heavy per-region fine-tuning)
- Policy teams in every major market, not just HQ
- Explicit "counter-speech" and "news reporting" exemptions in classifier training
- Regular regional audits of removal patterns

### Under-Removal in Low-Resource Languages

**Problem:** Swahili, Burmese, Amharic have <0.1% of English's training data for LLMs. Classifier performance on these languages is significantly worse.

**Real example:** UN investigators found that Facebook's failure to moderate hate speech in Burmese contributed to the Rohingya genocide. The platform had zero Burmese speakers on its trust & safety team until 2014.

**Mitigation:**
- Prioritize low-resource languages in human review (active learning score boost)
- Translation-based review: translate to high-resource language, review there (lossy but better than nothing)
- Cross-lingual transfer learning with targeted fine-tuning on 5K high-quality samples per language
- Track per-language coverage gap in metrics; make it a P1 engineering priority

### Coordinated Evasion at Scale

**Problem:** Well-funded adversaries (state actors, organized hate groups) reverse-engineer the classifier, identify the decision boundary, and craft content that reliably evades detection.

**Detection:** If many accounts are testing the same slightly-varied content (A/B testing evasions), this itself is a signal of coordinated evasion probing. Monitor for "probing clusters."

**Mitigation:**
- Do not publish precision/recall breakdown by evasion technique
- Model randomization: add noise to scores before exposing any API signal
- Multiple independent classifiers — evading one does not evade all
- Network-level detection catches the coordination even if individual posts evade

### Model Gaming via Feedback Loop

**Problem:** The review queue is prioritized by model score. Items that score below the review threshold never get human labels. This creates a feedback loop: the model never improves on content it confidently misclassifies as safe.

**Mitigation:**
- Random sample of low-scoring content sent to human review regardless of model score
- This is the "exploration budget" — typically 1–3% of queue capacity
- Treat as a bandit problem: balance exploitation (review high-score content) vs. exploration (review uncertain + low-score content)

### Cultural Context Failures

**Problem:** Same image means entirely different things in different cultures. Gesture that is offensive in one country is greeting in another. Platform-wide policy applied uniformly creates inconsistent enforcement.

**Mitigation:**
- Geo-routing: apply regional policy models based on content's target audience, not just poster's location
- Cultural context metadata as feature in classifier
- Regional policy teams with veto power over automated decisions on their region's content

---

## 11. Real-World System References

**Meta's WPIE (Whole Post Integrity Embeddings)**
Multi-modal embedding model that encodes text + images + video into a shared representation space. Enables cross-modal similarity search: find posts semantically similar to known policy violations without needing per-violation classifiers. Supports near-duplicate detection at 3B+ posts/day.

**YouTube's Three-Strikes System**
Severity-tiered enforcement: first violation → warning, second → content removal, third → channel termination. Forces explicit policy severity mapping into the engineering system. ML classifiers must output severity, not just binary violation.

**PhotoDNA (Microsoft/NCMEC)**
Industry standard for CSAM hash matching. Shared hash database across platforms (Facebook, Google, Twitter, Microsoft). Legal requirement in some jurisdictions to check against this database before distributing user content.

**Trust & Safety ML at Scale (published literature)**
- Facebook AI: "Hate Speech Detection in Hinglish" — code-switching between Hindi and English requires specialized models
- Google Jigsaw: Perspective API — public hate speech scoring API, trained on Wikipedia talk page data
- Twitter: "Abuse and Harassment" paper — account-level features matter as much as post-level features

---

## Canonical Interview Q&As

**Q: How do you measure whether your content moderation system is working if your ML classifier might be biased?**

A: Never use the classifier's own output as the measurement source — that produces circular metrics. Correct approach is stratified random sampling: sample content from the platform stratified by ML score decile, have expert human reviewers apply policy, then use Horvitz-Thompson estimation to compute unbiased prevalence from the stratified sample. This is how Meta computes its Community Standards Enforcement Report numbers. Separate from prevalence, measure proactive rate (fraction of removed content found before user report) and over-removal rate (fraction of removals reversed on appeal). Each requires independent ground truth — prevalence from random sampling, over-removal from appeals with a correction factor for non-appealing users.

**Q: Your hate speech classifier works well in English but performs poorly in Swahili. How do you close that gap without having millions of labeled Swahili examples?**

A: Several approaches in combination: (1) Cross-lingual transfer with XLM-RoBERTa — pre-trained representations transfer between languages; fine-tune on ~5K high-quality Swahili examples to get reasonable initial performance; (2) Translate-train-classify: translate Swahili content to English for classification (lossy, misses code-switching, but fast to deploy); (3) Active learning prioritization — route Swahili content to human review queue with high priority, use reviewer decisions as labels, retrain weekly; (4) Partner with local NGOs for labeled data creation — they have domain expertise in local context. Track per-language metrics explicitly; a platform that does not measure by language will not see the gap until a human rights crisis surfaces it.

**Q: How would you design the system to ensure <1 second action on CSAM while processing 500M uploads per day?**

A: Hash-matching (PhotoDNA) runs at ingestion, before any ML scoring, with <10ms latency. Implementation: maintain a FAISS IVF index of known CSAM hashes in memory, sharded across servers. At ingestion, compute perceptual hash in parallel with writing to object store; FAISS ANN lookup runs against the in-memory index with sub-millisecond query time. Match → HOLD immediately (block delivery), enqueue for specialist review + NCMEC reporting. The <1s SLA is met by hash matching alone — ML classifiers are not in the critical path for CSAM. For novel CSAM (not in hash database), the ML classifier may take up to 200ms for images; the content is held until either hash match or classifier score exceeds zero-tolerance threshold. Novel CSAM hash is added to the database within hours of human confirmation, so future variants are caught by hashing.

**Q: A viral post contains a video of a protest where police are using force. How does your system decide whether to remove it?**

A: This is a paradigmatic borderline case — graphic violence, but also political speech and news documentation. System design: (1) The video is not auto-removed (graphic violence alone ≠ zero-tolerance); (2) ML classifiers flag graphic_violence=0.75, news_documentation=0.60, political_speech=0.55; (3) Policy engine applies context rules: account is a news organization → reduce action weight; post includes news caption → reduce action weight; (4) Routes to high-priority human review queue with context: similar prior posts, account history, trending topic context; (5) Human reviewer applies "documentation exception" in policy: graphic violence by agents of state, documented for public interest, is typically policy-compliant with a sensitivity label; (6) Action: add interstitial label ("graphic content — tap to view") rather than remove. Key insight: the system should not auto-remove borderline content even if ML score is high — the cost of over-removal on political speech is asymmetric to the cost of a 2-hour review delay.

**Q: How do you prevent your training data pipeline from creating a feedback loop where the model never improves on content it consistently misclassifies?**

A: The feedback loop problem: classifier routes content below threshold away from human review, so those items never get labels, so the model never learns from its mistakes in that region. Three mitigations: (1) Exploration budget: send a random 2% sample of all content to human review regardless of classifier score — this ensures every score region gets labels; (2) Negative sampling for training: include confirmed-clean content from low-score region alongside violation samples — ensures model trains on easy negatives and does not drift toward marking everything as violation; (3) Calibration monitoring: if the model's score distribution shifts (e.g., more mass in 0.4–0.6 range) without a corresponding increase in human-reviewed violation rate, this indicates the model is becoming miscalibrated — trigger full audit. Additionally, every model update is validated against a holdout set that includes items previously below the review threshold (sampled via exploration budget).

**Q: How would you handle the case where an organized group discovers that adding a specific watermark to their images evades your visual classifier?**

A: Coordinated evasion of this type leaves multiple signals: (1) Many accounts posting images with the same watermark/artifact in a short window — this is a CIB (coordinated inauthentic behavior) signal even if individual images are borderline; (2) The accounts sharing these images likely form a detectable network (similar account age, same seeding accounts, similar engagement patterns); (3) The watermark itself, once identified by the red team or via network analysis, becomes a feature. Short-term: add watermark as a rule-based signal (within hours of detection, no retraining needed). Medium-term: retrain image classifier with augmented training data that includes the watermark. Long-term: train the classifier to be robust to adversarial perturbations using adversarial training. Publish no public information about the detection to avoid guiding the next iteration of evasion.

**Q: How do you handle policy changes — for example, the platform updates its definition of hate speech to include a new category. How do you update the system without creating inconsistency?**

A: Policy changes create three problems: (1) existing labels are wrong (they reflect old policy); (2) the deployed model encodes old policy; (3) content that was previously allowed may need retroactive removal. Process: (1) Policy team publishes policy change with an effective date and clear examples; (2) Relabeling sprint: sample ~10K items from the affected category, re-label under new policy, compute kappa between old and new labels to quantify the scope of change; (3) Retrain with new labels only from items labeled under the new policy (version tag policy on every label); (4) Retroactive enforcement: run new model in shadow mode over recently-allowed content, queue items newly flagged for expedited human review rather than auto-removing (retroactive auto-removal creates significant user trust risk); (5) For content already removed that may now be allowed under relaxed policy — run appeal-style review on a sample rather than mass reinstatement. The key constraint: never mix labels from before and after a policy change in the same training batch without explicit policy-version conditioning.

## Flashcards

**What content types? Text (posts, comments, DMs), images, video (live + recorded), audio, mixed-media posts?** #flashcard
What content types? Text (posts, comments, DMs), images, video (live + recorded), audio, mixed-media posts

**What violation categories? CSAM, imminent violence/terrorism, hate speech, harassment, spam/scams, misinformation, self-harm, synthetic media/deepfakes?** #flashcard
What violation categories? CSAM, imminent violence/terrorism, hate speech, harassment, spam/scams, misinformation, self-harm, synthetic media/deepfakes

**Proactive vs reactive? Proactive (scan every upload) vs reactive (user reports only) vs hybrid?** #flashcard
Proactive vs reactive? Proactive (scan every upload) vs reactive (user reports only) vs hybrid

**What action space? Remove, label/interstitial, reduce distribution, suspend account, refer to law enforcement?** #flashcard
What action space? Remove, label/interstitial, reduce distribution, suspend account, refer to law enforcement

**Geographic scope? Global → policy must reflect local laws (NetzDG in Germany, DSA in EU, FOSTA-SESTA in US)?** #flashcard
Geographic scope? Global → policy must reflect local laws (NetzDG in Germany, DSA in EU, FOSTA-SESTA in US)

**Who gets harmed by errors? False negative?** #flashcard
victim not protected, violator remains; false positive: legitimate speaker silenced

**Appeal mechanism? Yes/no, SLA on appeals, who reviews appeals?** #flashcard
Appeal mechanism? Yes/no, SLA on appeals, who reviews appeals

**Live content? Live streaming requires real-time action with near-zero latency and no rewind?** #flashcard
Live content? Live streaming requires real-time action with near-zero latency and no rewind

**XLM-RoBERTa over English-only BERT?** #flashcard
covers 100 languages in one model

**Multi-label head?** #flashcard
real content often violates multiple policies simultaneously

**Temperature calibration post-training?** #flashcard
raw sigmoid scores are poorly calibrated

**Context window?** #flashcard
include conversation thread for comments, not just isolated post

**CNN backbone (ResNet-50 or EfficientNet-B3) fine-tuned on policy violation data?** #flashcard
CNN backbone (ResNet-50 or EfficientNet-B3) fine-tuned on policy violation data

**Outputs?** #flashcard
P(nudity), P(graphic_violence), P(hate_symbol), P(spam_visual)

**<50ms per image?** #flashcard
<50ms per image

**ViT-L/16 with policy-specific fine-tuning?** #flashcard
ViT-L/16 with policy-specific fine-tuning

**Used for borderline cases where Stage 1 score is 0.3–0.7?** #flashcard
Used for borderline cases where Stage 1 score is 0.3–0.7

**Context?** #flashcard
image + OCR text overlay + alt text

**1 fps for most content (90% of compute budget)?** #flashcard
1 fps for most content (90% of compute budget)

**5 fps if initial score > 0.3?** #flashcard
5 fps if initial score > 0.3

**All frames if score > 0.6 or account is flagged?** #flashcard
All frames if score > 0.6 or account is flagged

**Policy version tag on every training label?** #flashcard
Policy version tag on every training label

**When policy changes, relabel affected sample with new policy before retraining?** #flashcard
When policy changes, relabel affected sample with new policy before retraining

**Use policy change date as a feature cutoff?** #flashcard
don't mix pre/post-policy labels

**Fast retrain cycle for high-impact policy changes (days, not weeks)?** #flashcard
Fast retrain cycle for high-impact policy changes (days, not weeks)

**Exposure limits?** #flashcard
no moderator reviews CSAM/graphic violence >4h/day

**Greyscale/blurred preview mode?** #flashcard
show blurred image unless reviewer confirms

**Queue assignment?** #flashcard
new moderators not assigned to CSAM queue

**Mandatory breaks enforced by the review UI after N consecutive graphic items?** #flashcard
Mandatory breaks enforced by the review UI after N consecutive graphic items

**Escalation path?** #flashcard
reviewer can escalate without making a decision (no decision pressure)

**Maintain rules layer (fast to update, no retraining) for emerging evasions?** #flashcard
Maintain rules layer (fast to update, no retraining) for emerging evasions

**Weekly model updates for trending evasion patterns (not monthly)?** #flashcard
Weekly model updates for trending evasion patterns (not monthly)

**Human monitoring of adversarial forums (4chan, Telegram) to get ahead of new techniques?** #flashcard
Human monitoring of adversarial forums (4chan, Telegram) to get ahead of new techniques

**Red team?** #flashcard
internal team attempts to evade current model; findings drive rule/model updates

**Separate models per cultural/political context (or heavy per-region fine-tuning)?** #flashcard
Separate models per cultural/political context (or heavy per-region fine-tuning)

**Policy teams in every major market, not just HQ?** #flashcard
Policy teams in every major market, not just HQ

**Explicit "counter-speech" and "news reporting" exemptions in classifier training?** #flashcard
Explicit "counter-speech" and "news reporting" exemptions in classifier training

**Regular regional audits of removal patterns?** #flashcard
Regular regional audits of removal patterns

**Prioritize low-resource languages in human review (active learning score boost)?** #flashcard
Prioritize low-resource languages in human review (active learning score boost)

**Translation-based review?** #flashcard
translate to high-resource language, review there (lossy but better than nothing)

**Cross-lingual transfer learning with targeted fine-tuning on 5K high-quality samples per language?** #flashcard
Cross-lingual transfer learning with targeted fine-tuning on 5K high-quality samples per language

**Track per-language coverage gap in metrics; make it a P1 engineering priority?** #flashcard
Track per-language coverage gap in metrics; make it a P1 engineering priority

**Do not publish precision/recall breakdown by evasion technique?** #flashcard
Do not publish precision/recall breakdown by evasion technique

**Model randomization?** #flashcard
add noise to scores before exposing any API signal

**Multiple independent classifiers?** #flashcard
evading one does not evade all

**Network-level detection catches the coordination even if individual posts evade?** #flashcard
Network-level detection catches the coordination even if individual posts evade

**Random sample of low-scoring content sent to human review regardless of model score?** #flashcard
Random sample of low-scoring content sent to human review regardless of model score

**This is the "exploration budget"?** #flashcard
typically 1–3% of queue capacity

**Treat as a bandit problem?** #flashcard
balance exploitation (review high-score content) vs. exploration (review uncertain + low-score content)

**Geo-routing?** #flashcard
apply regional policy models based on content's target audience, not just poster's location

**Cultural context metadata as feature in classifier?** #flashcard
Cultural context metadata as feature in classifier

**Regional policy teams with veto power over automated decisions on their region's content?** #flashcard
Regional policy teams with veto power over automated decisions on their region's content

**Facebook AI: "Hate Speech Detection in Hinglish"?** #flashcard
code-switching between Hindi and English requires specialized models

**Google Jigsaw: Perspective API?** #flashcard
public hate speech scoring API, trained on Wikipedia talk page data

**Twitter: "Abuse and Harassment" paper?** #flashcard
account-level features matter as much as post-level features
