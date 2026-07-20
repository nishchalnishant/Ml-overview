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

**What lives here:** NLP, computer vision, time series, speech, recommender systems, tabular.

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
