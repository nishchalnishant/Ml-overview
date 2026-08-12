---
module: Responsible AI
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, responsible-ai, privacy, fairness, adversarial-robustness, llm-safety, governance]
prerequisites: [14-responsible-ai]
---
# 14 Responsible AI — Revision Guide

Single-page revision for the entire `14-responsible-ai/` folder: privacy/fairness math, adversarial robustness, and LLM safety/governance. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **14-responsible-ai**
  - **01: Privacy & Fairness**
    - Why ML fails people
      - Proxy discrimination
      - Aggregate vs per-group accuracy
    - Privacy fundamentals
    - Machine unlearning
      - Right to be forgotten
    - Differential privacy
      - epsilon-delta definition
      - Privacy budget epsilon
      - Composition theorem
      - Laplace mechanism
      - Gaussian mechanism
    - DP-SGD
      - Per-sample grad clipping
      - Calibrated noise addition
      - Moments accountant
    - Federated learning
      - FedAvg
      - Non-IID drift
      - FedProx SCAFFOLD FedNova
      - Secure aggregation
    - Fairness metrics
      - Demographic parity
      - Equalized odds
      - Equal opportunity
      - Calibration
      - Individual fairness
    - Impossibility theorems
      - Chouldechova Kleinberg
      - Base rate dependence
    - Bias mitigation
      - Pre-processing
      - In-processing adversarial debiasing
      - Post-processing thresholds
    - Fairness in LLMs
      - Counterfactual probing
      - WEAT WinoBias
    - Model auditing
      - Membership inference
      - Training data extraction
  - **02: Adversarial Robustness**
    - Threat model axes
      - White/black/transfer box
      - Perturbation budget Lp
      - Targeted vs untargeted
      - Evasion vs poisoning vs extraction
    - Attacks
      - FGSM
      - PGD
      - Transfer attacks
    - Defenses
      - Adversarial training min-max
      - Robustness/accuracy tradeoff
      - Certified defenses
      - Randomized smoothing
    - Gradient masking failure
      - BPDA attack
    - Production notes
      - Threat-model before mitigating
      - Query rate-limiting
  - **03: LLM Safety & Governance**
    - Prompt injection
      - Direct injection
      - Indirect injection
      - Privilege separation
      - Structural separation
    - Reward hacking
      - Sycophancy as safety failure
      - Verbosity overoptimization
    - Refusal calibration
      - Under-refusal
      - Over-refusal
      - Paired benchmark reporting
    - Red-teaming methodology
      - Manual red-teaming
      - Automated LLM red-teaming
      - Failure-mode taxonomy
    - Governance
      - Model/system cards
      - Responsible scaling policies
      - Capability thresholds
      - EU AI Act risk tiers
```

## Questions

### Easy

1. What is the formal epsilon-delta definition of differential privacy?
2. What does a smaller epsilon mean for privacy and for accuracy?
3. Name the five main fairness metrics covered in this folder.
4. What is demographic parity, and what does it ignore?
5. What is equalized odds, and how does it differ from equal opportunity?
6. What does "calibration" mean as a fairness criterion?
7. What is the Laplace mechanism and when is it used?
8. What are the two steps of DP-SGD, and why are both required?
9. What is FedAvg, and what problem does federated learning solve?
10. What is membership inference, and what does a successful attack imply?
11. What is FGSM, and what does the `sign()` operation achieve?
12. What is PGD, and why is it the standard robustness benchmark instead of FGSM?
13. Define the threat-model axes for adversarial robustness (attacker knowledge, budget, goal, timing).
14. What is gradient masking, and why is it a red flag?
15. What is the difference between direct and indirect prompt injection?
16. What is a model/system card, and what should it contain?
17. What are the EU AI Act's four risk tiers?
18. What is a Responsible Scaling Policy (RSP)?
19. What is machine unlearning trying to achieve?
20. What is the accuracy/privacy tradeoff in one sentence?

### Medium

1. Why can't demographic parity and equalized odds both hold when base rates differ across groups?
2. Walk through why calibration alone was an insufficient defense in the COMPAS case.
3. Why does clipping per-sample gradients alone not provide differential privacy?
4. Explain why federated learning is "a step toward privacy, not sufficient for it."
5. What is the non-IID problem in federated learning, and name two mitigations.
6. Why is composition important when running multiple DP queries, and what happens if you ignore it?
7. Why is indirect prompt injection considered a harder production problem than direct injection?
8. Why does privilege separation "change the ceiling" rather than just "reduce the rate" of successful attacks?
9. A paper reports 95% robust accuracy with no clean-accuracy loss — why should you be skeptical?
10. Why is adversarial training 5-10x more expensive than standard training?
11. Explain why sycophancy is a safety failure and not merely a quality/helpfulness issue.
12. Why should refusal rate never be reported as a single number without a paired benign-compliance metric?
13. Why is automated LLM red-teaming necessary but not sufficient on its own?
14. Why does pre-committing to capability thresholds (RSPs) remove a structural incentive problem at launch time?
15. Explain the three origins of bias (data, objective, deployment) and why mitigation must match the origin.
16. Why do adversarial examples exist even in models that generalize well — is this overfitting?
17. Why does a more capable LLM tend to be more susceptible to following injected instructions?
18. What's the difference between "empirical" robustness and "certified" robustness?
19. Why is disaggregating a membership-inference audit (per subgroup) more informative than an overall AUC?
20. Why does adding more refusal training examples without a matched benign set predictably cause over-refusal?
21. Explain why threshold optimization (post-processing) cannot fix a model whose representations already encode a protected attribute via proxies.
22. Why is average income query sensitivity 200K/n, and how does that translate into Laplace noise scale?
23. In fairness-in-LLM auditing, why don't standard classification fairness metrics (e.g., equalized odds) transfer directly?

### Hard

1. Derive why calibration + equal FPR + equal FNR is jointly unsatisfiable when base rates differ (the Chouldechova/Kleinberg result), using a worked numeric example with two groups at different base rates.
2. A fraud model is being evaded and security wants adversarial (Lp-ball) training — construct the argument for why this defends the wrong threat model, and propose what should be done instead.
3. Design a red-team process (manual + automated) for a new tool-calling LLM agent, including what artifact it should produce and how you'd re-test after a fine-tune.
4. Design a defense-in-depth architecture for an email agent vulnerable to indirect prompt injection; state which layer bounds the worst case versus which layers merely reduce the rate.
5. Your over-refusal benchmark regresses sharply after a safety fine-tune — walk through the full diagnostic process to isolate cause and the fix, including how you'd distinguish a fine-tune regression from a system-prompt or upstream classifier change.
6. Explain, with the math, why DP-SGD's total privacy loss grows sublinearly in the number of training steps (moments accountant / Rényi DP intuition) rather than linearly as basic composition would suggest.
7. What breaks in secure aggregation (Bonawitz et al.) if a subset of clients collude or drop out mid-protocol?
8. Given group A default rate 5% and group B default rate 15%, work through what demographic parity, equalized odds, and equal opportunity each imply for approval thresholds, and identify which harms each choice trades off.
9. Why does gradient masking fool naive robustness evaluation, and design an evaluation protocol (attacks + diagnostics) that would catch it, including BPDA and transfer-attack checks.
10. A facial recognition system shows 0.8% error on light-skinned men and 34.7% on dark-skinned women. Trace the failure through data, objective, and deployment stages, and design a full mitigation pipeline with a measurement plan at each stage.
11. Explain why individual fairness is difficult to operationalize in practice, and what assumptions about the similarity metric d_input could hide unfairness even while satisfying the formal Lipschitz-style constraint.
12. Why is "reward hacking" the expected outcome of RLHF rather than an occasional bug (Goodhart's law framing), and why doesn't tightening the reward model alone fix it?
13. A model trained with DP-SGD at epsilon=0.0065 per user per day shows much worse accuracy than a non-private baseline. Walk through the levers available to close the gap without abandoning the privacy target.
14. Design an end-to-end pre-deployment audit for a high-stakes model that must satisfy both a fairness bar and an EU AI Act high-risk conformity assessment — what artifacts are shared between the two and what's unique to each?
15. What happens to a certified defense's (e.g., randomized smoothing) guarantees if the attacker can also manipulate the noise distribution or query the smoothed classifier adaptively many times?
16. Explain why membership inference succeeds specifically because of the training/test generalization gap, and derive why heavy regularization (or DP) suppresses attack AUC toward 0.5.
17. Contrast why federated learning's gradient updates can still leak information (gradient inversion) despite raw data never leaving the client, and what defense actually closes that gap versus one that merely obscures it.
18. In a system where fairness and adversarial robustness interact (e.g., adversarial training changes per-group error rates), argue whether improving Lp robustness could worsen fairness metrics, and how you'd detect it.
19. Construct a scenario where an automated red-team's LLM-generated attack taxonomy would systematically miss a class of jailbreak that a human domain expert would catch, and explain the structural reason it was missed.
20. A team claims "our RSP capability threshold was never crossed" as justification for skipping deeper red-teaming — identify the failure modes in RSP design that could make this claim true but still leave the deployment unsafe.
