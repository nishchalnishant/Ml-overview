---
module: Evaluation
topic: Revision Guide
subtopic: 04 Evaluation
status: unread
tags: [revision-guide, metrics, calibration, ab-testing, causal-inference, interpretability]
prerequisites: [probability, statistics]
---
# 04 Evaluation — Revision Guide

Single-page revision pass over `04-evaluation/`: metrics, model evaluation workflow, cross-validation, calibration/uncertainty, interpretability, A/B testing, and causal inference. Use the outline to check you can place every concept in its hierarchy, then self-test with the graded questions below.

## Map

```markdown
- **04 Evaluation**
  - **Evaluation Metrics**
    - Classification Metrics
      - Precision Recall F1 Fbeta
      - ROC-AUC vs PR-AUC
      - MCC
    - Calibration Metrics
      - ECE MCE
      - Brier Score
      - Proper Scoring Rules
    - Regression Metrics
      - MSE MAE Huber
      - MAPE Pitfalls
      - R2 Adjusted R2
      - Pinball Loss Quantile
    - Ranking Metrics
      - NDCG MAP MRR
      - Precision Recall at k
      - Diversity Coverage ILD Serendipity
    - Cost-Sensitive Evaluation
      - Cost Matrix
      - Optimal Threshold
    - Prediction Intervals
      - PICP MPIW
      - Conformal Prediction
    - Statistical Testing
      - McNemar Paired t Wilcoxon
      - Multiple Hypothesis Correction
    - Generative Model Eval
      - FID IS
      - BLEU ROUGE BERTScore
      - LLM as Judge
    - Offline vs Online Gap
      - Simpsons Paradox
      - Guardrail Metrics
  - **Model Evaluation Workflow**
    - Confusion Matrix First
    - Business Cost Framing
    - Threshold Selection Youden
    - Offline Shadow Canary Rollout
    - Feedback Loops Distribution Shift
  - **Cross-Validation**
    - Why Single Holdout Fails
    - K-Fold
    - Stratified K-Fold
    - Group K-Fold
    - Time Series CV
      - Expanding Window
      - Sliding Window
      - Gap Parameter
    - Repeated K-Fold
    - Nested CV Inner Outer
    - Pipeline Leakage Prevention
  - **Calibration and Uncertainty**
    - Reliability Diagrams
    - ECE MCE Thresholds
    - Platt Scaling
    - Isotonic Regression
    - Temperature Scaling
    - Aleatoric vs Epistemic
    - MC Dropout Deep Ensembles
    - Conformal Prediction Guarantee
  - **Model Interpretation**
    - Interpretability vs Explainability
    - SHAP
      - Shapley Value Formula
      - Efficiency Symmetry Dummy Linearity
      - Correlated Feature Failure
      - TreeSHAP KernelSHAP
    - LIME
      - Local Surrogate Algorithm
      - Instability
    - PDP vs ALE
    - Permutation Importance
    - Attention as Explanation Critique
    - Integrated Gradients
  - **A/B Testing**
    - Experiment Design Hypothesis Metrics
    - Power Analysis Sample Size
    - Deterministic Traffic Assignment
    - SRM Detection
    - CUPED Variance Reduction
    - Interference Network Effects
      - Cluster Randomization
      - Switchback Testing
    - Multi-Armed Bandits Thompson
    - Interleaving
    - Peeking Multiple Comparisons Novelty
  - **Causal Inference**
    - Confounding DAG Backdoor
    - ATE vs ATT
    - Propensity Scores
      - PSM
      - IPW
      - Doubly Robust
    - Difference in Differences
      - Parallel Trends
    - Regression Discontinuity
    - Instrumental Variables
      - Relevance Exclusion Independence
    - Method Choice Table
```

## Questions

### Easy

1. Define precision and recall in terms of the confusion matrix.
2. Why is accuracy a misleading metric on an imbalanced dataset?
3. What does the F1 score trade off, and why might F-beta be preferred over plain F1?
4. What does ROC-AUC measure? What does PR-AUC measure?
5. What is Matthews Correlation Coefficient (MCC) and why is it liked for imbalanced binary classification?
6. Define Expected Calibration Error (ECE) in words.
7. What is Brier score, and how does it relate to accuracy vs. calibration?
8. What is a "proper scoring rule"?
9. Name three regression metrics and one weakness of each.
10. Why can MAPE be misleading or undefined?
11. What does R² measure, and why do we sometimes use adjusted R² instead?
12. What is pinball loss used for?
13. What do NDCG, MAP, and MRR each measure in ranking?
14. What is Precision@k?
15. Why does a single train/test split give a "noisy" performance estimate?
16. What problem does K-Fold cross-validation solve versus a single holdout?
17. Why do we use Stratified K-Fold instead of plain K-Fold on imbalanced data?
18. What is Group K-Fold for, and what kind of leakage does it prevent?
19. Why can't you shuffle time series data before cross-validation?
20. What is a reliability diagram?
21. What does Platt scaling do?
22. What is isotonic regression calibration, at a high level?
23. What is temperature scaling used for, and where is it common (e.g., neural nets)?
24. What is the difference between aleatoric and epistemic uncertainty?
25. What is SHAP trying to compute, in one sentence?
26. Name the four Shapley value axioms.
27. What is LIME, and how does it differ from SHAP at a mechanical level?
28. What does a Partial Dependence Plot (PDP) show?
29. What is permutation importance, and how is it computed?
30. What is the primary metric vs. a guardrail metric in an A/B test?
31. What is Sample Ratio Mismatch (SRM) and how is it typically detected?
32. What is CUPED used for?
33. In causal inference, what is a confounder?
34. What is the difference between ATE and ATT?
35. What is a propensity score?

### Medium

1. Why is ROC-AUC often overly optimistic on highly imbalanced datasets compared to PR-AUC?
2. Walk through why optimizing accuracy on a 99%/1% fraud dataset can produce a useless model, and what metric(s) you'd propose instead.
3. How would you pick a classification threshold when false positives and false negatives have different business costs?
4. Explain the difference between calibration and discrimination (e.g., a model can have great AUC but be poorly calibrated — how?).
5. Why does Log Loss punish confident wrong predictions more harshly than Brier score?
6. Describe how you'd design a cost-sensitive evaluation for a loan default model where a missed default costs 10x a false alarm.
7. Explain PICP and MPIW — what tension exists between them when tuning a prediction interval method?
8. What guarantee does conformal prediction give you, and what assumption is that guarantee conditioned on?
9. Why is nested cross-validation necessary when you're both tuning hyperparameters and reporting a "final" performance number?
10. What specifically goes wrong if you fit a StandardScaler on the full dataset before running cross-validation?
11. Why might Group K-Fold and Stratified K-Fold conflict, and how does StratifiedGroupKFold resolve it?
12. In time-series CV, why is a "gap" parameter sometimes necessary even though folds already respect chronological order?
13. Explain why isotonic regression can overfit on small calibration sets while Platt scaling is more data-efficient.
14. Why does temperature scaling preserve accuracy while changing calibration? What does it NOT fix?
15. Give an example where SHAP values are misleading due to correlated features.
16. Compare TreeSHAP and KernelSHAP — what do you gain and lose in each?
17. Why is LIME's explanation unstable across repeated runs on the same instance?
18. Explain why permutation importance can misattribute importance among two highly correlated features.
19. Why does PDP assume feature independence, and how does ALE address that assumption's failure?
20. Why is attention weight a weak explanation of model behavior, despite being popular?
21. Explain Simpson's Paradox in the context of offline vs. online metrics.
22. Why do you need a power analysis before running an A/B test, and what four inputs does the sample size formula need?
23. Explain how CUPED reduces variance without biasing the treatment effect estimate.
24. Why does peeking at A/B test results early (optional stopping) inflate the false positive rate?
25. Describe a network-interference scenario in an A/B test and one method (e.g., cluster randomization or switchback) to address it.
26. Why can't you just "put all confounders into a regression" and read off the treatment coefficient safely?
27. What is the parallel trends assumption in Difference-in-Differences, and how would you build evidence for it?
28. What does Regression Discontinuity actually estimate, and why is that scope more limited than an RCT's ATE?
29. Why is inverse propensity weighting variance-sensitive, and what practical fix (trimming/clipping) is used?
30. Why is a doubly robust estimator preferred over plain IPW or plain outcome regression?

### Hard

1. Derive why the Brier score decomposes into calibration, refinement (resolution), and uncertainty terms — and explain what each term implies about model quality.
2. You have two models: Model A has higher ROC-AUC, Model B has higher PR-AUC and better calibration. Which do you ship for a fraud-alert system with a 0.5% base rate, and why might the "obviously better" AUC model still be the wrong choice?
3. Design an evaluation protocol for a recommender system that must be checked for both accuracy (NDCG) and fairness/diversity (exposure, ILD) — what happens when these objectives conflict, and how do you report it?
4. Explain precisely why K-fold cross-validation's variance does not shrink as 1/K, and why increasing K past a point stops helping (or hurts).
5. Walk through a leakage scenario where a Group K-Fold split still leaks information via engineered features (e.g., a rolling aggregate keyed on group), and how you'd fix it beyond just grouping the folds.
6. Explain why nested CV's outer-loop score cannot simply be reused as the deployed model's expected performance, even though it's "unbiased."
7. A model is well-calibrated in aggregate (ECE near 0) but is badly miscalibrated within a specific subgroup that's underrepresented. Explain how this happens mathematically and how you'd detect it.
8. Contrast MC Dropout and Deep Ensembles as uncertainty estimation methods: which uncertainty type does each actually capture well, and where does each fail silently?
9. Prove (or argue rigorously) why conformal prediction's coverage guarantee holds under exchangeability but breaks under distribution shift or non-i.i.d. time-series data.
10. Given a dataset with two perfectly correlated features, walk through what SHAP, permutation importance, and PDP each report for "importance" — and explain why they disagree.
11. A fairness audit finds SHAP attributes low importance to a "race" feature but high importance to "ZIP code." Explain why this doesn't mean the model is fair, using the proxy-variable argument.
12. Explain why LIME's local linear surrogate can produce an explanation that is locally faithful but globally misleading, and construct a scenario where this actively harms a business decision.
13. A metric improves offline (e.g., +2% NDCG) but the online guardrail metric (e.g., session length) regresses. Walk through the full diagnostic process you'd use to decide whether to ship anyway.
14. Derive the minimum detectable effect (MDE) relationship to sample size and explain why underpowered A/B tests systematically bias the literature of "successful" experiments (the "winner's curse").
15. Explain exactly what breaks in an A/B test under network interference (e.g., a social feature) even when SRM is clean and the test is properly powered.
16. Compare Thompson Sampling bandits to a fixed A/B test in terms of statistical validity of post-hoc analysis — why is "peeking" fine for one but toxic for the other?
17. A team ran a propensity-score-matched analysis and found a strong positive treatment effect. Construct the strongest case for why you should NOT believe this result, using DAG reasoning (mediators, colliders, unmeasured confounding).
18. Explain why conditioning on a collider variable in a causal analysis introduces bias rather than removing it — construct a concrete example (e.g., in a hiring or fraud pipeline).
19. Contrast what Difference-in-Differences, Instrumental Variables, and Regression Discontinuity each "buy" you in terms of the untestable assumption you must defend, and rank them by how defensible those assumptions typically are in a product analytics setting.
20. A feature rolled out to 100% of users with no control group and no threshold-based eligibility. Walk through the full decision tree of causal methods you'd attempt, in order, and what you'd say if none apply cleanly.
21. Explain why Instrumental Variables estimates a LATE (local average treatment effect) rather than an ATE, and describe a realistic scenario where the LATE is a misleading answer to the business question actually being asked.
22. A doubly robust estimator's outcome model and propensity model are both moderately misspecified. Does the "double robustness" property still help, and under what conditions does it fail entirely?
23. Explain how the Rosenbaum–Rubin propensity score theorem reduces a high-dimensional confounding-adjustment problem to a one-dimensional one, and identify the exact assumption that makes this collapse valid.
24. A model shows excellent held-out AUC and passes offline calibration checks, but online A/B testing shows no lift on the North Star metric. Enumerate every distinct failure mode across the metrics/evaluation stack (from 04-evaluation) that could explain this gap.
