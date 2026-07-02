---
module: Emerging Topics
topic: Interpretability and Explainable AI (XAI)
subtopic: ""
status: unread
tags: [emergingtopics, ml, interpretability-and-explainable-ai-xai]
---
# Interpretability and Explainable AI (XAI)

This folder covers how to explain what a model did and why — from intrinsically interpretable models to post-hoc explanation methods, LLM-specific interpretability, and the regulatory context that makes explainability a requirement rather than a nice-to-have.

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [interpretability-and-xai.md](interpretability-and-xai.md) | Intrinsic models (linear/tree), feature importance (MDI, permutation), PDP/ICE plots, LIME, SHAP, attention visualization, Grad-CAM, Integrated Gradients, TCAV, model cards, LLM interpretability (probing, logit lens, mechanistic interpretability, SAEs), and the regulatory context (GDPR Art. 22, ECOA/FCRA, SR 11-7, EU AI Act) |
| [mechanistic-interpretability.md](mechanistic-interpretability.md) | Deep dive on circuits, superposition, sparse autoencoders, and causal tracing inside transformer internals |

---

## How To Read It

If you need a specific explanation method (SHAP, LIME, Grad-CAM), jump directly to its section in [interpretability-and-xai.md](interpretability-and-xai.md) — each is self-contained with problem → insight → mechanics → what-breaks. For LLM-specific interpretability (probing, logit lens, circuits, SAEs), read §13 there first, then go deep in [mechanistic-interpretability.md](mechanistic-interpretability.md).

---

## Back to top

[Emerging Topics README](../README.md)
