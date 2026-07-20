---
module: Responsible AI
topic: Overview
subtopic: ""
status: unread
tags: [fairness, privacy, interpretability, security, index]
prerequisites: []
---
# Responsible AI

Fairness, interpretability, privacy, and security — the **model-agnostic** treatment, grounded
in the math.

| File | Covers |
| :--- | :--- |
| [01-privacy-and-fairness.md](01-privacy-and-fairness.md) | Privacy fundamentals, machine unlearning, differential privacy, DP-SGD, federated learning, fairness metrics, the impossibility theorems, bias mitigation, fairness in LLMs, membership inference |
| [02-adversarial-robustness.md](02-adversarial-robustness.md) | Threat models, FGSM, PGD, adversarial training min-max, certified defenses, gradient masking |

**Interpretability lives in [`../04-evaluation/05-model-interpretation.md`](../04-evaluation/05-model-interpretation.md)**
— SHAP (axioms, TreeSHAP/KernelSHAP), LIME, PDP/ALE, permutation importance, attention-as-explanation.
It sits under evaluation because it is used for debugging as much as for compliance; the
fairness half of the ZIP-code question is answered here in `01-`.

## Routing — this folder vs. LLM safety

Responsible-AI material is split across two homes **on purpose**:

| If the question is… | Go to |
| :--- | :--- |
| Mathematical or model-agnostic — ε-δ differential privacy, DP-SGD accounting, demographic parity vs. equalized odds, the impossibility results | here |
| LLM-specific behaviour — jailbreaks, prompt injection, guardrails, RLHF/DPO/Constitutional AI, sycophancy, reward hacking | [`../10-llms/interview-notes/`](../10-llms/interview-notes/) §10, §17, §18 |

Those LLM files stay where they are: they are part of a numbered interview arc, several have
paired `-snappy.md` modality files, and their framing is behavioural rather than mathematical.
The genuine overlap is narrow — bias measurement and PII handling — and is cross-linked rather
than duplicated.

**Increasingly its own interview round**, especially at larger companies. The trap is treating
fairness as a single metric — demographic parity and equalized odds are mathematically
incompatible except in degenerate cases, and knowing *that* is the answer.

**Known gap:** interpretability (SHAP/LIME mechanics, attention attribution) and adversarial
robustness have no deep-dive file; both are currently only touched in passing.
