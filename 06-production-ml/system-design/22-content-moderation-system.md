---
module: Production Ml
topic: System Design
subtopic: Content Moderation System
status: unread
tags: [productionml, ml, system-design-content-moderati]
---
# Content Moderation System Design

End-to-end ML system for detecting policy-violating content at platform scale. Canonical system design question at Meta, Google/YouTube, Twitter/X, TikTok, and any UGC platform.

**Scale:** 500M+ posts/day across text, image, video, audio; <1s action on CSAM/imminent violence; policy varies by market; human review queues process millions of items/week.

---

## 1. Problem Framing

### Clarifying Questions

- **Content types?** Text, images, video (live + recorded), audio, mixed-media posts
- **Violation categories?** CSAM, imminent violence/terrorism, hate speech, harassment, spam/scams, misinformation, self-harm, synthetic media/deepfakes
- **Proactive vs reactive?** Scan every upload vs. rely on user reports vs. hybrid
- **Action space?** Remove, label/interstitial, reduce distribution, suspend account, refer to law enforcement
- **Geographic scope?** Global policy must reflect local laws (e.g., EU DSA, Germany's NetzDG)
- **Error asymmetry?** False negative = victim unprotected; false positive = legitimate speaker silenced
- **Appeals?** SLA, who reviews
- **Live content?** Needs real-time action, no rewind

### Why This Is Harder Than Fraud Detection

| Dimension | Fraud Detection | Content Moderation |
|---|---|---|
| Label clarity | Binary | Ambiguous — context-dependent (same post can be hate speech or legitimate criticism) |
| Context | Numeric transaction features | Depends on culture, tone, platform norms, current events |
| Error asymmetry | Both FP/FN cost money | FP silences speech; FN allows harm; both carry political cost |
| Policy evolution | Slow | Frequent, especially around elections |
| Modality | Mostly tabular | Text, image, video, audio, and combinations |
| Human cost | Analyst reviews alerts | Moderators exposed to traumatic content at scale |

### Business Metrics

**Prevalence** — fraction of views that are violating content, measured via stratified random sampling + expert review (never from raw ML output, which is biased).

**Proactive Rate (PR)** — fraction of removed content caught before a user report. Target >95% for CSAM/terrorism.

**Over-removal Rate (ORR)** — fraction of removals that were incorrect, measured via appeals + random audit.

**Trade-off:** raising PR (more proactive catches) tends to raise ORR (more false positives). The threshold is a policy call, not a pure ML call.

---

## 2. Scale Requirements

| Dimension | Requirement |
|---|---|
| Throughput | 500M+ posts/day (~6K/sec sustained, 30K/sec peak) |
| Text classification | <100ms P99 |
| Image classification | <200ms P99 |
| Video classification | <30s per 1-min video (async OK) |
| CSAM/imminent violence action | <1 second |
| Hash matching (PhotoDNA) | <10ms at ingestion |
| Human review SLA | 24h high-severity, 72h borderline |
| Live content | <5s to action |
| Global coverage | 100+ languages |

---

## 3. System Architecture

### Proactive Pipeline (scan on upload)

1. **Ingestion & media parsing** — extract text/frames/audio, assign content_id, store in object store
2. **Hash matching** (<10ms) — PhotoDNA for CSAM, MD5/SHA for known-bad URLs, pHash for near-duplicates. Match → immediate hold + NCMEC report
3. **ML classifier layer** (if no hash match) — text classifier (LLM-based) and image/video classifier (CNN + ViT) run in parallel, fused by a multi-modal consistency model
4. **Policy engine** — applies jurisdiction rules, account history, severity tier
5. **Decision routing** — auto-remove / hold for review / distribute normally
6. **Human review** — prioritized queue, active-learning sample selection
7. **Feedback loop** — reviewer decisions and appeals become training labels

### Reactive Pipeline (user reports)

Report enrichment (reporter history, is item already queued) → re-score with report signal as a feature → merge into priority queue, boosted by report volume/severity.

---

## 4. Multi-Modal Detection

### 4.1 Text Classification

Keyword matching fails — context is everything ("I'm going to kill this presentation" vs. a real threat). Standard approach: fine-tuned multilingual LLM (e.g., XLM-RoBERTa) with a multi-label head, since one post can violate several policies at once.

```python
class HateSpeechClassifier:
    VIOLATION_LABELS = ['hate_speech', 'harassment', 'violence_threat',
                         'self_harm', 'spam', 'misinformation', 'sexual_content']

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(self.VIOLATION_LABELS),
            problem_type='multi_label_classification')

    def score(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        logits = self.model(**inputs).logits
        return dict(zip(self.VIOLATION_LABELS, torch.sigmoid(logits).squeeze().tolist()))
```

**Key decisions:**
- XLM-RoBERTa over English-only BERT: one model covers ~100 languages
- Multi-label, not multi-class: violations overlap
- Post-training temperature calibration: raw sigmoid scores are poorly calibrated
- Include conversation thread as context for comments, not just the isolated post
- Production runs a distilled 6-layer model for <50ms latency; the full model runs offline for hard cases and label generation

### 4.2 Image Classification

**Two-stage pipeline:**

1. Fast CNN (ResNet-50/EfficientNet-B3), <50ms, outputs P(nudity), P(violence), P(hate_symbol), P(spam) per image
2. ViT-L/16 for borderline cases (score 0.3–0.7), using image + OCR text overlay + alt text as context

```python
def classify(self, image):
    fast_scores = self.fast_cnn(preprocess(image))
    if max(fast_scores.values()) > 0.95:
        return ModerationResult(fast_scores, tier='auto_action')
    if max(fast_scores.values()) < 0.1:
        return ModerationResult(fast_scores, tier='pass')
    ocr_text = self.ocr_engine.extract(image)
    return ModerationResult(self.vit_model(image, text_context=ocr_text), tier='borderline')
```

### 4.3 Video Classification

Frame-by-frame scanning doesn't scale at 500M/day, so sampling is hierarchical:

1. Metadata check (duration, format, uploader history)
2. Sample 1 frame/sec → image classifier; flag if any frame >0.8
3. If no flag: transcribe audio (Whisper) → text classifier — hate speech can be audio-only
4. Temporal model (LSTM/Transformer over frame sequence) for content only visible in motion
5. Fuse visual + audio/text + metadata scores, apply cross-modal consistency check

**Sampling strategy:** 1 fps by default; 5 fps if score >0.3; all frames if score >0.6 or account is flagged.

### 4.4 Cross-Modal Consistency

Adversaries pair benign images with violating text overlays, or hateful audio with innocent video. A large gap between modality scores (one high, one low) is itself a suspicious signal and boosts the effective score:

```python
def cross_modal_consistency_score(text_score, image_score, audio_score):
    scores = [text_score, image_score, audio_score]
    inconsistency = max(scores) - min(scores)
    if max(scores) > 0.7 and inconsistency > 0.5:
        return max(scores) * 1.2  # capped at 1.0 — likely evasion attempt
    return max(scores)
```

---

## 5. Severity Tiers and Routing

| Tier | Content Type | Action | SLA |
|---|---|---|---|
| **Zero-tolerance** | CSAM, live terrorism, imminent violence threat | Auto-remove + suspend + legal referral | <1 second |
| **High-severity** | Explicit hate speech, terrorism glorification, graphic violence | Hold + expedited human review | 2 hours |
| **Medium-severity** | Borderline hate speech, harassment, adult content | Full visibility + human review | 24 hours |
| **Low-severity/spam** | Automated spam, scam links, coordinated inauthentic behavior | Auto-remove/throttle, no review | Immediate |
| **Borderline** | Political speech, satire, news reporting of violence | Distribute normally + low-priority queue | 72 hours |

```python
class PolicyRouter:
    ZERO_TOLERANCE_THRESHOLD = 0.85
    HIGH_SEVERITY_THRESHOLD = 0.65
    REVIEW_THRESHOLD = 0.30

    def route(self, scores, account_history):
        if scores.get('csam', 0) > self.ZERO_TOLERANCE_THRESHOLD:
            return RoutingDecision('REMOVE', 'zero_tolerance', legal_referral=True, suspend_account=True)
        if scores.get('imminent_violence', 0) > self.ZERO_TOLERANCE_THRESHOLD:
            return RoutingDecision('REMOVE', 'zero_tolerance')
        if scores.get('hate_speech', 0) > self.HIGH_SEVERITY_THRESHOLD:
            return RoutingDecision('HOLD', 'high', queue_priority=self._compute_priority(scores, account_history))
        if scores.get('spam', 0) > self.HIGH_SEVERITY_THRESHOLD:
            return RoutingDecision('REMOVE', 'spam', log_only=True)
        if any(s > self.REVIEW_THRESHOLD for s in scores.values()):
            return RoutingDecision('DISTRIBUTE', 'borderline', queue_priority='low')
        return RoutingDecision('DISTRIBUTE', 'clean')

    def _compute_priority(self, scores, history):
        p = max(scores.values())
        if history.prior_violations > 2: p *= 1.3   # repeat violators
        if history.follower_count > 100_000: p *= 1.2  # high-reach accounts
        return min(p, 1.0)
```

---

## 6. Perceptual Hashing and Known-Bad Databases

**PhotoDNA** (Microsoft/NCMEC) converts an image into a hash robust to resizing, format conversion, and minor edits — unlike cryptographic hashes, visually similar images produce numerically close hashes (greyscale → resize to 144×144 → DCT per cell → quantize → 144-byte hash). Distance below a threshold means a likely match.

Runs at ingestion, before any ML scoring. At 500M images/day, brute-force distance comparison against millions of hashes is infeasible — use an ANN index (FAISS IVF) for sub-millisecond lookup.

```python
class PhotoDNAMatcher:
    def check(self, image_bytes):
        query_hash = compute_photodna_hash(image_bytes)
        distances = np.linalg.norm(self.known_hashes - query_hash, axis=1)
        if distances.min() < self.threshold:
            return MatchResult(is_match=True, distance=distances.min(), db_id=distances.argmin())
        return MatchResult(is_match=False)
```

**Evasion defenses:** PhotoDNA is crop/resize/brightness robust by design. Text overlays need separate OCR + text hashing. Mirrored/rotated images need augmented hashes generated at indexing time. Video CSAM is caught by running PhotoDNA on extracted I-frames.

---

## 7. Human-in-the-Loop

Human reviewers are the ground-truth source, not a fallback — the system exists to make their review efficient and safe.

**Priority queue:** P0 zero-tolerance (specialist/legal review) → P1 high-severity (2h SLA, senior moderators) → P2 active-learning samples → P3 borderline (72h SLA) → P4 appeals.

### Active Learning for Queue Prioritization

Prioritize items most useful for model improvement: uncertainty near the decision boundary, boosted for low-resource languages and recently-changed policies (where old labels may be stale).

```python
def active_learning_score(model_confidence, language, last_policy_update):
    uncertainty = 1 - abs(2 * model_confidence - 1)
    lang_boost = 2.0 if language in LOW_RESOURCE_LANGUAGES else 1.0
    policy_boost = 1.5 if (datetime.now() - last_policy_update).days < 30 else 1.0
    return uncertainty * lang_boost * policy_boost
```

### Inter-Annotator Agreement

Policy is inherently ambiguous. Compute pairwise Cohen's kappa across annotators on the same item; items with kappa <0.4 are genuinely ambiguous and flagged for policy clarification, not treated as annotator error.

### Policy Drift

When policy changes: the deployed model still reflects old policy, and historical labels do too. Handling: tag every label with a policy version; relabel affected samples under the new policy before retraining; don't mix pre/post-policy labels in one training batch; fast-track retraining for high-impact changes (days, not weeks).

### Annotator Wellbeing

Moderators review the worst content on the internet at scale — a documented harm. Mitigations: exposure caps (e.g., no CSAM/graphic-violence review >4h/day), blurred preview by default, new moderators kept off the CSAM queue, mandatory breaks after N consecutive graphic items, and an escalation path that doesn't force a decision.

---

## 8. Adversarial Arms Race

| Technique | Detection |
|---|---|
| Leetspeak ("h4t3") | Normalization layer, character-level model |
| Homoglyph substitution (Cyrillic "а") | Unicode NFKC normalization |
| Coded language/slang | Embedding similarity + human monitoring of emerging terms |
| Text rendered as image | OCR on all images |
| Violating frame at unlikely sample point | Adaptive sampling on motion/scene change |
| Splitting violation across posts | Account-level analysis, not just post-level |

**Coordinated Inauthentic Behavior (CIB):** individually-borderline posts amplified by a coordinated network is itself a violation. Detect by clustering posts on text similarity (MinHash LSH), then checking whether the posting accounts show coordination signals (new, low-activity, similar behavior patterns).

```python
def detect_coordinated_campaign(self, posts, window_hours=24):
    clusters = self.lsh_cluster(posts, similarity_threshold=0.8)
    return [
        CoordinationCluster(posts=c, score=score)
        for c in clusters if len(c) >= 10
        for score in [self.coordination_classifier.predict(
            self.extract_account_features([p.account for p in c]))]
        if score > 0.7
    ]
```

**Keeping pace:** adversaries adapt faster than models retrain. Mitigate with a fast rules layer for emerging evasions (no retraining needed), weekly (not monthly) model updates, monitoring of adversarial forums, and an internal red team.

---

## 9. Evaluation

### Prevalence Estimation

ML output can't be used directly — it's biased by training distribution and threshold. Use stratified random sampling (over-sample high-score deciles) with expert human review, then compute a weighted estimate with confidence intervals:

```python
def estimate(self, population_score_distribution, stratum_labels):
    weighted_prevalence, variance = 0.0, 0.0
    for decile, labels in stratum_labels.items():
        decile_fraction = np.mean(
            (population_score_distribution >= decile * 0.1) &
            (population_score_distribution < (decile + 1) * 0.1))
        decile_prevalence = np.mean(labels)
        variance += (decile_fraction ** 2) * (decile_prevalence * (1 - decile_prevalence) / len(labels))
        weighted_prevalence += decile_fraction * decile_prevalence
    ci = 1.96 * np.sqrt(variance)
    return PrevalenceEstimate(weighted_prevalence, weighted_prevalence - ci, weighted_prevalence + ci)
```

### Metrics Dashboard

| Metric | Target | Method |
|---|---|---|
| Prevalence (hate speech) | <0.03% of views | Stratified sampling + expert review (quarterly) |
| Proactive rate (CSAM) | >99% | Actioned before first report |
| Proactive rate (terrorism) | >95% | Actioned before first report |
| Over-removal rate | <5% | Appeals granted / total removals |
| Time to action (P99, CSAM) | <1 second | System telemetry |
| Human review SLA (P1) | >95% within 2h | Queue metrics |
| Inter-annotator agreement | κ > 0.7 | Weekly audit |
| Low-resource language gap | <10% vs. English | Per-language prevalence audit |

**False positive rate via appeals:** appeals are biased (many false-positive victims never appeal). Estimated FPR = appeals granted / (appeals granted + removals confirmed correct), where the denominator needs a random audit — appeal rate alone isn't a reliable FPR proxy.

---

## 10. Failure Modes

**Over-removal of legitimate political speech.** Models trained on Western/English data over-remove political speech in other contexts (e.g., Meta's Oversight Board found over-removal of Palestinian political speech during the 2021 conflict). Mitigate with region-specific fine-tuning, policy teams in every major market, explicit counter-speech/news exemptions, and regular regional audits.

**Under-removal in low-resource languages.** Languages like Burmese have a fraction of English's training data; classifier quality suffers accordingly. UN investigators linked Facebook's Burmese-language moderation gaps to the Rohingya genocide. Mitigate with active-learning boosts for low-resource languages, translation-based review, targeted cross-lingual fine-tuning, and tracking per-language coverage as a P1 metric.

**Coordinated evasion at scale.** Well-funded adversaries reverse-engineer the decision boundary. Signal: many accounts probing slight content variations. Mitigate by not publishing precision/recall by evasion technique, adding score randomization, running multiple independent classifiers, and using network-level (not just post-level) detection.

**Feedback-loop gaming.** If review queue priority is purely score-driven, content confidently misclassified as safe never gets reviewed and the model never corrects. Fix with an exploration budget (1–3% of queue capacity reviewed randomly regardless of score) — treat it as a bandit problem balancing exploitation vs. exploration.

**Cultural context failures.** The same image or gesture can mean different things in different cultures; uniform global policy causes inconsistent enforcement. Mitigate with geo-routing based on target audience, cultural metadata as a feature, and regional policy teams with veto power.

---

## 11. Real-World References

- **Meta WPIE** — multi-modal embedding model (text+image+video in shared space) enabling cross-modal similarity search against known violations without per-category classifiers
- **YouTube three-strikes** — severity-tiered enforcement (warning → removal → termination); forces classifiers to output severity, not just binary violation
- **PhotoDNA** — industry-standard CSAM hash matching, shared across major platforms
- **Published literature** — Facebook's Hinglish hate-speech work (code-switching), Google Jigsaw's Perspective API, Twitter's account-level abuse features research

---

## Canonical Interview Q&As

**Q: How do you measure whether the system is working if the classifier itself might be biased?**
A: Never use classifier output as the measurement — that's circular. Sample content stratified by ML score decile, have human experts apply policy, then use weighted (Horvitz-Thompson-style) estimation for unbiased prevalence. Separately track proactive rate (caught before report) and over-removal rate (reversed on appeal) — each needs its own ground truth, since appeal rates alone are biased by non-appealing users.

**Q: Your classifier works well in English but poorly in Swahili — how do you close the gap without millions of labeled examples?**
A: Combine cross-lingual transfer (XLM-RoBERTa fine-tuned on ~5K quality Swahili examples), translate-then-classify as a fast stopgap, active-learning-prioritized human review to generate labels weekly, and local NGO partnerships for labeling expertise. Track per-language metrics explicitly — without that, the gap surfaces only after real-world harm.

**Q: How do you guarantee <1s action on CSAM at 500M uploads/day?**
A: PhotoDNA hash matching runs at ingestion before any ML scoring, against an in-memory FAISS IVF index, sub-millisecond per lookup. A match holds the content immediately and queues it for NCMEC reporting — the classifier isn't in the critical path. Novel CSAM not yet in the hash database goes through the image classifier (~200ms) and is held pending confirmation; once confirmed, its hash is added so future copies are caught instantly.

**Q: A viral video shows police using force at a protest — how does the system decide whether to remove it?**
A: This is a paradigm borderline case — graphic but also newsworthy. Not auto-removed (graphic violence alone isn't zero-tolerance). Classifiers flag graphic_violence and news/political signals; the policy engine down-weights action for verified news accounts and captions indicating documentation intent. Routes to a high-priority human queue with account history and context. Typical outcome: an interstitial sensitivity label rather than removal. The key principle: don't auto-remove high-score but context-dependent content — a review delay is cheaper than wrongly silencing news coverage.

**Q: How do you prevent a feedback loop where the model never improves on content it consistently misclassifies?**
A: Send a random 2% sample of all content to human review regardless of score, so every score region keeps getting labeled. Include confirmed-clean low-score content in training as negatives, so the model doesn't drift toward over-flagging. Monitor for score-distribution shifts unaccompanied by matching changes in reviewed violation rate — that signals miscalibration and should trigger an audit.

**Q: An organized group discovers a watermark that evades your visual classifier — how do you respond?**
A: Coordinated evasion leaves signals beyond the individual post: many accounts posting the same artifact in a short window (a CIB signal), and those accounts likely form a detectable network. Short-term, add the watermark as a rule-based signal (hours, no retraining). Medium-term, retrain with augmented data including it. Long-term, use adversarial training for robustness. Don't publish detection details — that just informs the next iteration.

**Q: Policy changes to add a new hate-speech category — how do you update without inconsistency?**
A: Three problems at once: old labels reflect old policy, the deployed model reflects old policy, and previously-allowed content may now need action. Process: policy team defines an effective date and examples; relabel a sample under the new policy and measure kappa against old labels to size the change; retrain using only new-policy-labeled data (version-tagged); run the new model in shadow mode over recently-allowed content and queue newly-flagged items for review rather than auto-removing retroactively (trust risk); for previously removed content that's now allowed, sample-review rather than mass-reinstate. Never mix pre/post-policy labels in one training batch without explicit version conditioning.

## Flashcards

**Content types?** #flashcard
Text, images, video (live + recorded), audio, mixed-media

**Violation categories?** #flashcard
CSAM, imminent violence/terrorism, hate speech, harassment, spam/scams, misinformation, self-harm, synthetic media

**Why measure prevalence via sampling, not ML output?** #flashcard
ML output is biased by training distribution and threshold — need stratified random sampling + expert review

**Proactive Rate vs Over-removal Rate?** #flashcard
PR = fraction caught before user report (target >95%); ORR = fraction of removals reversed on appeal (target <5%); raising PR tends to raise ORR

**Why XLM-RoBERTa over English-only BERT?** #flashcard
Covers ~100 languages in one model

**Why multi-label, not multi-class, for text classification?** #flashcard
Real content often violates multiple policies simultaneously

**Why is temperature calibration needed post-training?** #flashcard
Raw sigmoid scores are poorly calibrated

**Two-stage image pipeline?** #flashcard
Fast CNN (ResNet-50/EfficientNet) for clear cases <50ms; ViT for borderline (0.3-0.7) with OCR context

**Video sampling strategy?** #flashcard
1 fps default; 5 fps if score >0.3; all frames if score >0.6 or account flagged

**Cross-modal consistency signal?** #flashcard
Large gap between modality scores (one high, one low) suggests evasion — text/image/audio don't match

**PhotoDNA vs cryptographic hash?** #flashcard
PhotoDNA hashes of visually similar images are numerically close (robust to resize/crop/format); crypto hashes break on any pixel change

**Why hash-match before ML scoring?** #flashcard
<10ms latency, meets <1s CSAM SLA without the classifier in the critical path

**Active learning score components?** #flashcard
Uncertainty near decision boundary × low-resource-language boost × recent-policy-change boost

**How to handle policy drift in training data?** #flashcard
Version-tag every label; relabel affected samples under new policy; never mix pre/post-policy labels in one batch

**Annotator wellbeing mitigations?** #flashcard
Exposure caps, blurred previews, new moderators kept off CSAM queue, mandatory breaks, no-pressure escalation path

**Coordinated Inauthentic Behavior (CIB) detection?** #flashcard
Cluster posts by text similarity (MinHash LSH), then check if posting accounts show coordination signals (new/low-activity/similar behavior)

**Exploration budget / feedback loop fix?** #flashcard
Random 2-3% of all content reviewed regardless of score, so low-score regions still get labels — treat as exploitation/exploration bandit problem

**Root cause of Rohingya moderation failure?** #flashcard
Near-zero training data and no native speakers on trust & safety team for Burmese

**Why not auto-remove high-score but newsworthy graphic content?** #flashcard
Cost of over-removing political/news speech is asymmetric to the cost of a review delay

**Why not publish precision/recall by evasion technique?** #flashcard
Gives adversaries a roadmap to the next evasion iteration
