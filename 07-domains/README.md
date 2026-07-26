---
module: Domains
topic: Overview
subtopic: ""
status: unread
tags: [nlp, cv, recsys, time-series, index]
prerequisites: []
---
# Domains

Applied areas where general methods meet domain-specific constraints and vocabulary.

**What lives here:** NLP and time series, in depth; computer vision, at interview depth; recommender systems, routed elsewhere (see below). Speech and tabular are not separately covered — tabular is classical ML's default case ([`../03-classical-ml/`](../03-classical-ml/)), and speech-specific modeling isn't in scope for this repo's target roles.

## Core Modules

| File | What it covers |
| :--- | :--- |
| [01-nlp-fundamentals.md](01-nlp-fundamentals.md) | Classical and pre-transformer NLP: tokenization, embeddings, sequence models |
| [03-nlp-transformers.md](03-nlp-transformers.md) | Transformer-era NLP from first principles, interview-focused |
| [04-computer-vision.md](04-computer-vision.md) | CV fundamentals: CNNs, detection, segmentation |
| [05-computer-vision-interview.md](05-computer-vision-interview.md) | CV interview questions and drills, companion to file 04 |
| [06-time-series.md](06-time-series.md) | Time series ML: forecasting, decomposition, evaluation |

*Numbering skips `02-*`: no gap in coverage, the number was retired in an earlier reorganization.*

**Recsys — where it actually lives.** Recommender systems are covered, but under
`../15-system-design/`, because the interview question is nearly always a design question rather
than an algorithms question:

| For | Go to |
| :--- | :--- |
| MF/ALS and two-tower math, cold-start reasoning | [`../15-system-design/02-ml-system-design.md`](../15-system-design/02-ml-system-design.md) §retrieval |
| Two-tower architecture + code, hard-negative mining | [`../15-system-design/01-design-interview-framework.md`](../15-system-design/01-design-interview-framework.md) |
| Full case studies | [`cases/10-personalization.md`](../15-system-design/cases/10-personalization.md), [`11-recommendation-system.md`](../15-system-design/cases/11-recommendation-system.md), [`12-video-recommendation.md`](../15-system-design/cases/12-video-recommendation.md), [`08-news-feed-ranking.md`](../15-system-design/cases/08-news-feed-ranking.md) |

Earlier drafts of this README called recsys a "genuine gap." That was written from the target
structure rather than the files — four case studies and the retrieval math were already in place.
