---
module: Study Plans
topic: Week 3 System Design
subtopic: Day 23 Case Studies
status: unread
tags: [studyplans, ml, week-3-system-design-day-23-ca]
---
# Day 23: Case Studies (Google Context)

## Why This Topic Comes Here

Day 22 gave you the framework for designing ML systems. Day 23 applies that framework to real-world systems at scale, using Google's products as reference implementations. These case studies are valuable not because you will copy them, but because they reveal the gap between textbook ML and production ML: the constraints (latency, scale, cold start, fairness) that change everything. Studying these systems trains the pattern of thinking that interviewers test — "given this constraint, what tradeoff does that force?"

---

## Executive Summary

| System | Core Problem | Key Technologies | Interview Anchor |
|--------|--------------|-------------------|------------------|
| **Search** | Ranking Relevance | BERT, RankBrain | NLP, Query Intent |
| **Photos** | Searchable Images | CNNs, Vision Transformers | Image Embedding |
| **YouTube** | Recommendations | Two-Tower Models | Retrieval & Ranking |
| **Gmail** | Spam/Smart Reply | RNNs, LSTMs, Transformers | Sequence Learning |

---

## 1. YouTube Recommendation (A Classic)

**Why this is the most commonly referenced recommendation system:** It is large enough to surface all real-world constraints (scale, cold start, diversity, exploitation vs. exploration) and the paper is public. Understanding this system gives you a template for any retrieval + ranking problem.

YouTube uses two main networks:
1. **Candidate Generation**: Filters millions of videos to hundreds using user history as the primary signal.
2. **Ranking**: Assigns a precise score to each candidate to produce a final list.
- **Metric**: Not just CTR (Click-Through Rate) but **Watch Time** (engagement quality).

**Key insight:** The metric shift from CTR to watch time was architecturally consequential. A model optimizing CTR would learn to recommend clickbait — content that gets clicked but abandoned quickly. Watch time as a metric forces the model to predict a behavior (continued engagement) that is more correlated with actual user value. The metric choice shapes what the model learns, not just how well it learns.

**How to verify understanding:** YouTube optimizes for watch time, not clicks. Name a third metric that would be even closer to "user satisfaction" than watch time, and describe what new data collection infrastructure you would need to optimize for it.

**What trips people up:** Assuming that a better model automatically means better user experience. The model is only as good as the metric it optimizes. A model that perfectly maximizes watch time could recommend addictive content that users regret watching. The business constraint (responsible recommendations) cannot be captured by any single metric — it requires auxiliary constraints and re-ranking rules.

---

## 2. Google Photos: Search by Semantic Content

Uses powerful visual embeddings.
- **Task**: Identify people/objects/scenes without explicit tags from users.
- **Tech**: Models like **CLIP** or contrastive learning to align images and text in a shared embedding space.

**Key insight:** The breakthrough in Google Photos search is not the image classifier — it is learning a shared embedding space where the image of a dog and the text "dog" end up near each other. This allows you to search images with arbitrary text queries without ever labeling those exact concepts in advance. The training signal is not "this image contains a dog" but "this image and this alt-text came from the same web page."

**How to verify understanding:** CLIP is trained on (image, text) pairs from the web. Explain why this training signal — even though it is noisy and not curated — produces a model that can zero-shot classify images into categories it was never explicitly trained on.

**What trips people up:** Thinking contrastive learning requires hand-labeled pairs. The key insight is using naturally occurring co-occurrence (image + its caption, image + its alt text) as weak supervision at scale. The noise is acceptable because the scale is enormous.

---

## 3. Google Search: Modular Architecture

**Why Google Search uses multiple models for one query:**

Complexity. One model handling all aspects of a query would be enormous, slow, and impossible to debug or improve in isolation. Google's pipeline separates concerns:
- One module handles spelling correction.
- One handles synonym expansion.
- One handles semantic intent (BERT-based).
- A ranking model combines all signals.

**Key insight:** The modular architecture is a form of system-level bias-variance tradeoff. Each specialized module has lower variance on its specific sub-task than a single monolithic model would. More importantly, it allows independent improvement — you can improve the semantic understanding module without risking regression in the spelling corrector.

**How to verify understanding:** You propose replacing the entire modular pipeline with a single large language model that handles all aspects of query understanding. Name two specific risks this creates and how you would mitigate them.

**What trips people up:** Thinking that end-to-end models are always superior to modular ones. End-to-end systems can optimize globally, which is an advantage. But they are harder to debug, harder to update partially, and any failure mode affects all functionality simultaneously. Modularity trades global optimization for local debuggability and reliability.

---

## 4. Fairness in Recommendation Systems

**Why fairness is a system design topic, not an afterthought:** A recommendation system trained on historical data learns historical biases. If certain creators received less distribution in the past (due to algorithmic or systemic reasons), a model trained to "optimize engagement" will perpetuate and amplify this disparity — not because of malicious intent, but because past under-representation is reflected in the data.

**Key insight:** Fairness constraints in recommendation systems are not binary — they are a spectrum. "Equal exposure" for all creators is one definition. "Proportional exposure relative to quality" is another. "Equal exposure conditional on user interest" is a third. These definitions conflict with each other and with the engagement optimization objective. Choosing a fairness definition is a product and ethical decision, not a technical one.

**How to verify understanding:** A recommendation system gives 10x more exposure to the top 1% of creators than the remaining 99% combined. The system was trained to maximize watch time. Is this a fairness problem, an optimization success, or both? What would you need to measure to answer this?

**What trips people up:** Treating debiasing as a post-processing step that can be added after the model is trained. Removing sensitive attributes from the feature set does not remove bias — the model can proxy those attributes through correlates (zip code as a proxy for race, for example). Fairness requires intervention at the data collection, labeling, and objective definition stages.

---

## Case Study Framework

When answering case study questions in interviews:
1. **Scope the problem** — Users, scale, goal, constraints.
2. **Handle Data** — Where do labels come from? How are they collected? What is the labeling lag?
3. **Draft Architecture** — Input → representation → model → output → serving.
4. **Discuss Edge Cases** — Cold start (new users/items), adversarial users, distribution shift.
5. **Define Monitoring** — What metric drift would trigger a retrain? How do you detect silent failures?

---

## Interview Questions (Strategic Thinking)

**1. "If you were designing a 'Related Articles' feature for Google News, what would be your primary metric?"**
> While CTR is easy to measure, **User Satisfaction** or **Retention** (do they come back tomorrow?) is the long-term goal. Metrics like "Time Spent on Article" or "Share Rate" are better proxies for quality than just clicks.

**2. "How would you handle 'Fairness' in a recommendation system?"**
> Discuss **Unbiased Data** (removing sensitive attributes is insufficient — proxies remain), **Equity in exposure** for minority creators, and monitoring for feedback loops and echo chambers.

**3. "Why does Google Search use multiple models for one query?"**
> Complexity. One model handles spelling, another handles synonyms, another handles semantic intent (BERT), and finally, a ranking model combines all signals. This is called a **Modular Architecture** and it enables independent improvement of each component.
