---
module: Evaluation
topic: Overview
subtopic: ""
status: unread
tags: [metrics, calibration, ab-testing, index]
prerequisites: [probability, statistics]
---
# Evaluation

Metrics, calibration, error analysis, and experiment design.

This is a top-level section rather than a classical-ML subtopic on purpose: the same machinery governs LLM evals and online experiments. A model is only as trustworthy as the number you chose to judge it by.

**What lives here:** classification/regression metrics, calibration and uncertainty, model interpretation, A/B testing and experimentation, error analysis.

**Interviewers probe hardest at** the gap between an offline metric and the business outcome it proxies — and at whether you can say why accuracy is the wrong metric for an imbalanced problem.

## Core Modules

| File | What it covers |
| :--- | :--- |
| [01-ml-evaluation-metrics.md](01-ml-evaluation-metrics.md) | Precision/recall, ROC-AUC, calibration, ranking metrics |
| [02-model-evaluation.md](02-model-evaluation.md) | Model evaluation workflow — error analysis, slicing, offline/online gap |
| [03-cross-validation.md](03-cross-validation.md) | K-fold, stratified, time-series splits, leakage pitfalls |
| [04-calibration-and-uncertainty.md](04-calibration-and-uncertainty.md) | Platt scaling, isotonic regression, uncertainty quantification |
| [05-model-interpretation.md](05-model-interpretation.md) | SHAP, LIME, feature importance, partial dependence |
| [06-ab-testing-experimentation.md](06-ab-testing-experimentation.md) | A/B testing and experimentation infrastructure for ML systems |
| [07-causal-inference.md](07-causal-inference.md) | Causal inference when you can't randomize |
