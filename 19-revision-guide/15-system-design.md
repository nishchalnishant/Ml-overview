---
module: System Design
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, system-design, ml-system-design, design-patterns, case-studies, mlops]
prerequisites: [15-system-design]
---
# 15 System Design — Revision Guide

Single-page revision for the entire `15-system-design/` folder: the interview framework, ML system design fundamentals, ML engineering lifecycle, reusable design patterns, building ML-powered applications, and 13 full case studies. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **15-system-design**
  - **Design Interview Framework**
    - Interview framework steps
    - Recsys: candidate gen + ranking
    - Position bias correction
    - Calibration + temperature
    - Explore/exploit
    - Search: keyword vs semantic
    - Training-serving skew
    - Fraud: threshold by SLA
    - Graph-based fraud
    - Feed: multi-objective ranking
    - Ads: GSP auction, ESMM
    - Budget pacing
    - Offline vs online metric alignment
  - **ML System Design Fundamentals**
    - Universal framework
    - Batch vs real-time serving
    - Two-stage retrieval + ranking
    - Recsys cold start
    - Fraud detection design
    - Search ranking: pairwise/listwise
    - LLM serving + KV cache
    - Monitoring layers: infra/model/business
    - Retraining triggers
    - Rollback plan
  - **ML Engineering & Production**
    - Project prioritization
    - Problem framing
    - Data collection + labeling
    - Data leakage prevention
    - Missing values / outliers
    - Scaling + encoding
    - Feature engineering + selection
    - Loss function selection
    - Hyperparameter tuning
    - Overfitting prevention
    - Reproducibility
    - Error analysis
    - Deployment paradigms
    - Drift detection
    - Retraining strategy
    - Model versioning + rollback
    - Bias/fairness + privacy
  - **ML Design Patterns**
    - Hashed feature
    - Embeddings
    - Feature cross
    - Multimodal input
    - Reframing
    - Multilabel classification
    - Ensembles: bagging/boosting/stacking
    - Cascade
    - Neutral class
    - Rebalancing / SMOTE
    - Transform
    - Multistage training
    - Transfer learning
    - Distillation
    - Regularization L1/L2
    - Evaluation metrics PR-AUC/NDCG
    - Slicing
    - Skew detection PSI
    - Baseline comparison
    - Prediction bias / adversarial debiasing
  - **Building ML-Powered Applications**
    - Don't start from the model
    - Dumbest end-to-end system first
    - Iterate on data before models
    - Write evals before the model
    - Debug failures systematically
    - Prototype to product transition
    - Testing ML systems
    - Communicating results
  - **Case Studies**
    - Ranking & Recommendation
      - Ad CTR prediction
      - News feed ranking
      - Search ranking
      - Personalization
      - Recommendation system
      - Video recommendation
    - Fraud & Risk & Trust
      - Fraud detection
      - Clinical ML system
      - Content moderation
    - Pricing & Value & Auctions
      - Customer LTV prediction
      - ETA + surge pricing
      - Real-time bidding
    - LLM Product
      - LLM product assistant / RAG
```

## Questions

### Easy

1. What are the main steps of the universal ML system design interview framework?
2. What clarifying questions should you ask before proposing any architecture?
3. Define batch serving vs real-time (online) serving. When is each appropriate?
4. Why do most large-scale recommendation and search systems use a two-stage (retrieval + ranking) architecture instead of scoring everything with one model?
5. What is candidate generation, and what are two common techniques for it?
6. What is training-serving skew and why does it happen?
7. What is calibration in the context of a classifier's output probabilities, and what does temperature scaling do?
8. Define PSI (Population Stability Index) and what it's used for.
9. What is the difference between covariate shift, label shift, and concept drift?
10. What is a hashed feature, and why hash categorical features instead of one-hot encoding them?
11. What is an embedding, and why is it preferred over one-hot encoding for high-cardinality categoricals?
12. What is a feature cross, and give a simple example.
13. What is the Cascade pattern, and what problem does it solve?
14. What is the Neutral Class pattern used for?
15. Name two techniques for handling severe class imbalance in training data.
16. What is knowledge distillation and why would you use it in production?
17. Why is PR-AUC often preferred over ROC-AUC for imbalanced classification problems?
18. What is NDCG and why does it weight by position?
19. What is data leakage? Give one example specific to a time-series problem.
20. What are the three monitoring layers described for production ML systems (infra, model, business)?
21. What is a feature store, and what problem does it solve?
22. Why should you build "the dumbest possible end-to-end system" before optimizing any model?
23. What does "iterating on data before iterating on models" mean in practice?
24. Why should you write evaluation code before you know what model you'll use?
25. What is the cold-start problem in recommendation systems? Name the two variants (user vs item).
26. What are RFM features in the context of customer LTV prediction?
27. What is the "10ms problem" in real-time bidding?
28. Why is fraud detection modeled as an ensemble of two models rather than a single model?
29. What does "explore/exploit" mean in the context of personalization or ads?
30. What is Learning to Rank (LTR), and how does it differ from standard binary classification?

### Medium

1. Walk through the universal framework end-to-end for a hypothetical prompt: "design a system that ranks products in a marketplace search."
2. Compare pointwise, pairwise, and listwise ranking losses. When would you choose LambdaMART/LambdaRank over pointwise cross-entropy?
3. How do you correct for position bias in ranking systems trained on click data? What's the role of propensity scores here?
4. You downsample negatives during fraud/CTR model training to speed up training. How do you recover a calibrated probability at inference time?
5. Explain the Two-Tower architecture for retrieval. Why can't you use a cross-encoder for first-stage candidate generation at scale?
6. What are "hard negatives" in two-tower training, and why does in-batch negative sampling alone often fail to produce a good model?
7. Compare hashed features vs learned embeddings as a way to handle high-cardinality categorical features — what are the tradeoffs (collisions, cold start, memory)?
8. Explain the Wide & Deep architecture: what does the wide component memorize vs what does the deep component generalize?
9. Walk through the Reframing pattern: how would you reframe a hard classification decision as a calibrated probability problem, and why would you want to?
10. Compare bagging, boosting, and stacking as ensemble strategies. Give a production example of each.
11. Describe the Multistage Training pattern (e.g., pretrain -> freeze -> fine-tune -> unfreeze). Why freeze early layers first?
12. What breaks when you deploy a model that was trained/evaluated on a random train/test split but the production data arrives as an ordered stream (e.g., fraud, ETA)? What's the fix?
13. How would you detect and diagnose feature skew between training and serving pipelines in production?
14. Contrast the retraining triggers you'd set for a fraud model vs a recommendation ranking model. Why might they differ?
15. What's different about designing for real-time bidding's ~10ms latency budget vs a news feed's few-hundred-ms budget? What architectural choices does that force?
16. Walk through the key design decisions for a fraud detection system: feature latency tiers, model ensemble, and label delay handling.
17. For customer LTV prediction, when would you use a probabilistic model (BG/NBD + Gamma-Gamma) instead of a gradient-boosted regressor?
18. Explain the "censoring problem" in non-contractual LTV estimation and why naive labeling introduces bias.
19. For ETA prediction, why would you use quantile/uncertainty-aware regression instead of a plain point estimate?
20. How does H3 hexagonal gridding help with ETA/surge pricing feature engineering compared to raw lat/lon?
21. What makes content moderation harder to design than fraud detection (per the case study's stated rationale)?
22. Why does news feed ranking use multi-objective optimization instead of a single engagement objective? What's a risk of getting this wrong (e.g., Meta's MSI pivot)?
23. Explain how a RAG-based LLM product assistant's retrieval, context assembly, and grounding stages interact — where does hallucination typically get introduced?
24. For a video recommendation system, why do you need separate signals (watch time, completion, explicit feedback) fused into a ranking score rather than optimizing watch time alone?
25. What's the difference in fairness/regulatory constraints between a clinical ML system and a typical consumer recsys? How does this change the architecture?
26. Compare BM25 lexical retrieval, dense two-tower retrieval, and hybrid fusion for search. When would you pick pure BM25?
27. Explain the adversarial debiasing approach to the Prediction Bias pattern — what is the discriminator predicting, and how does the loss function penalize the model?
28. How would you design a feature store to serve both real-time (<10ms) and batch features consistently to avoid skew?
29. What's the tradeoff between HNSW and IVF for billion-scale approximate nearest neighbor retrieval?
30. Compare MMoE (multi-gate mixture of experts) for multi-task ranking vs training separate models per task. What failure mode does MMoE address?

### Hard

1. Design an end-to-end ad CTR prediction system from clarifying questions through serving. Justify every architectural choice against a stated latency and scale budget.
2. Design a full fraud detection system: online path (<1ms features), batch/graph features, model ensemble, feedback loop with delayed labels, and concept drift handling. What breaks first at 10x transaction volume?
3. Design a real-time bidding pipeline under a strict 10ms end-to-end budget, including pCTR, pCVR, and bid-landscape models. Where do you cut corners first if latency creeps to 15ms?
4. Design an ETA prediction + dynamic surge pricing system. How do the ETA model, the road-network GNN, and the RL-based pricing policy interact, and what happens if the ETA model is systematically biased in one city?
5. Your offline AUC for a ranking model improved 3%, but online CTR dropped 1% after launch. Walk through a full debugging process covering training-serving skew, position bias, and metric misalignment.
6. Design a content moderation system covering proactive scanning, reactive user reports, multi-modal (text/image/video) classification, severity routing, and human-in-the-loop active learning. What happens when policy changes mid-flight (policy drift)?
7. Design a clinical ML system for a diagnostic-assist use case, addressing FDA SaMD classification, EHR data harmonization (OMOP CDM), selection bias, and asymmetric error costs. How does this change your evaluation strategy vs a typical consumer ML system?
8. Design a full end-to-end recommendation system: candidate retrieval, DCN-v2 ranking, re-ranking for diversity, cold start for both new users and new items, feature store, and A/B testing plan. What's your rollback plan if a new ranking model regresses a key business metric two days after full rollout?
9. Design a video recommendation system combining two-tower retrieval, MMoE multi-task ranking, and content-understanding signals (video embeddings, audio, thumbnails, transcripts). How would position bias correction interact with the multi-task heads?
10. Design a search ranking system that must serve well from click logs alone (no explicit relevance judgments). How do you train an unbiased LTR model given position bias and sparse clicks?
11. Design a personalization system that balances a long-term user profile with real-time session context. A user complains their feed is "too repetitive" — diagnose and fix without tanking relevance.
12. Design an LLM-powered product assistant (RAG) with a strict cost and latency budget. Retrieval returns the correct top-5 chunks but the LLM still answers incorrectly — enumerate every place in the pipeline you'd check, in order.
13. Design a customer LTV prediction system that must support both contractual and non-contractual business lines simultaneously. How do you unify probabilistic (BG/NBD) and ML-based approaches into one serving system?
14. You must migrate a fraud detection model from batch scoring to real-time scoring without any downtime or model drift discontinuity. Design the migration plan including shadow mode, feature parity validation, and rollback triggers.
15. Given a fixed inference cost budget, design a cascade across ad CTR, fraud detection, and content moderation — three cascades running on the same request pipeline. How do you order the stages and share compute?
16. A recommendation system's offline metrics (NDCG, recall@k) all improved after a model change, but a slice-level analysis reveals it regressed sharply for a specific user segment (e.g., new users, or a minority demographic). Diagnose using the Slicing and Prediction Bias patterns and propose a fix.
17. What breaks in a two-tower retrieval model if the item embedding table is updated nightly (batch) but user behavior shifts intraday (e.g., breaking news, flash sale)? Propose an architecture that handles both timescales.
18. Design monitoring and automated retraining triggers for three systems with very different drift dynamics: fraud (adversarial, fast-moving), news feed ranking (content velocity), and clinical ML (slow, regulated). Justify different retraining cadences and gating criteria for each.
19. Synthesize the Building ML-Powered Applications lifecycle (dumbest system first -> data iteration -> evals -> systematic debugging -> prototype-to-product -> testing -> communication) against one concrete case study of your choice — trace what could go wrong at each stage if skipped.
20. Design an experimentation (A/B testing) plan for a ranking system change that has strong network effects (e.g., a social feed), where standard independent-user randomization is invalid. What breaks with naive A/B testing here, and what's the fix?
21. A GBT fraud model and a sequence model for account takeover disagree strongly on a transaction. Design the arbitration/ensembling logic, and describe what monitoring would catch this ensemble silently degrading over time.
22. Estimate the QPS, feature-store read volume, and embedding-index size for a video recommendation system serving 500M DAU with a two-tower retrieval stage retrieving top-500 candidates per request. What's your capacity plan, and where's the first bottleneck?
23. Design a system that must serve both an ads ranking model (revenue-sensitive, needs budget pacing) and an organic feed ranking model (engagement-sensitive) from a shared candidate pool, blended into one feed. How do you prevent ads from cannibalizing organic engagement metrics, and how do you evaluate the blended result?
24. Take the Reframing, Neutral Class, and Cascade patterns together — design a content moderation pipeline that uses all three, and explain precisely where each pattern earns its place versus adding unnecessary complexity.
25. A model passes every offline evaluation gate (AUC, calibration, slicing) but causes a production incident within hours of full rollout. Using the deployment paradigms and rollback material from the ML Engineering guide, design a staged rollout process that would have caught this earlier, and specify the automatic rollback trigger conditions.
