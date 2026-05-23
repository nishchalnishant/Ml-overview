---
module: Llms
topic: Interview Notes
subtopic: Ai Safety Ethics And Responsible Ai What Snappy
status: unread
tags: [llms, ml, interview-notes-ai-safety-ethi]
---
# AI safety, ethics & responsible AI — practical engineering notes

This is not a philosophy class. This is **how you avoid incidents** in production AI.

**DevOps translation:** safety = threat model + controls + audit logs + incident response.

---

# Q1: What are hallucinations in LLMs, and how do you mitigate them?
- **Direct answer:** Ground with RAG/citations, constrain outputs, lower randomness, add verifiers/evals.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q2: What is prompt injection, and what are the different types (direct, indirect)?
- **Direct answer:** Direct: user says ‘ignore rules’; indirect: malicious text inside retrieved content. Defend with delimiters + tool allow-lists.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q3: How do you implement input and output guardrails for AI systems?
- **Direct answer:** Validate inputs, constrain outputs (schemas), filter unsafe content, and add human approvals for risky actions.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q4: What is AI alignment, and why is it important?
- **Direct answer:** Align behavior to human values/policies so the system is helpful and safe, not just fluent.

---

# Q5: How do you detect and mitigate bias in AI systems?
- **Direct answer:** Measure per subgroup, test intersections, fix data/features/objectives, monitor drift.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q6: Key data privacy considerations (GDPR, CCPA)?
- **Direct answer:** Purpose limitation, minimization, consent, retention, access rights, auditability.

---

# Q7: How do you handle PII in LLM inputs and outputs?
- **Direct answer:** Redact, tokenize, isolate secrets, restrict logging, encrypt, and enforce retention policies.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q8: What is explainability in AI?
- **Direct answer:** Explain why a decision happened in a user/auditor-friendly way.

---

# Q9: Interpretability vs explainability?
- **Direct answer:** Interpretability: model is inherently understandable; explainability: post-hoc reasons/attributions.

---

# Q10: How do you build trust with users?
- **Direct answer:** Be transparent, cite sources, show uncertainty, allow appeals, and don’t overclaim.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q11: Adversarial attacks + defenses?
- **Direct answer:** Injection, evasion, backdoors; defend with validation, sandboxing, monitoring, red-teaming.

---

# Q12: What is data poisoning?
- **Direct answer:** Attacker corrupts training/finetune/RAG corpora to change behavior or insert backdoors.

---

# Q13: Content safety filters?
- **Direct answer:** Multi-stage: input filter, generation constraints, output filter, human review for edge cases.

---

# Q14: Responsible AI frameworks?
- **Direct answer:** Use Microsoft/NIST-style principles: fairness, reliability, privacy, transparency, accountability.

---

# Q15: Copyright/IP concerns?
- **Direct answer:** Respect licenses, avoid regurgitation, track provenance, add similarity checks and policies.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q16: EU AI Act impact?
- **Direct answer:** Risk-based obligations: documentation, logging, human oversight, testing, transparency for high-risk systems.

---

# Q17: Audit trails and logging?
- **Direct answer:** Trace every decision: prompts, retrieval, tools, model version, outputs—redacted and access-controlled.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q18: Model cards?
- **Direct answer:** Documentation of intended use, limitations, data, evals, safety notes.

---

# Q19: Misuse and abuse in production?
- **Direct answer:** Rate limits, abuse detection, user verification, policy enforcement, incident response.

---

# Q20: Differential privacy?
- **Direct answer:** Add noise to training/queries to bound individual data leakage.

---

# Q21: Design an AI incident response plan?
- **Direct answer:** Detection → triage → containment → comms → rollback → postmortem → prevention.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q22: NIST AI RMF?
- **Direct answer:** Framework for mapping/measure/managing AI risks across lifecycle.

---

# Q23: Healthcare chatbot gives diagnoses — add safety guardrails?
- **Direct answer:** Constrain scope, require disclaimers, route to clinician, crisis escalation, block diagnosis claims.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q24: Prevent copyrighted verbatim reproduction?
- **Direct answer:** Dedup data, similarity filters, retrieval grounding, refusal on near-match, output monitoring.

---

# Q25: Resume AI rejects more female candidates — fix gender bias?
- **Direct answer:** Audit features, remove proxies, rebalance data, adjust objective, human review, fairness constraints.

---

# Q26: Intersectional bias failures?
- **Direct answer:** Evaluate intersections, not only marginals; targeted data + constraints; document trade-offs.

---

# Q27: GDPR explanation for loan denial?
- **Direct answer:** Provide understandable factors, process transparency, and an appeals path; log decisions.

---

# Q28: Right to be forgotten but data in weights — comply?
- **Direct answer:** Prefer not training on personal data; for weights: retrain or use unlearning/filters + legal guidance.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q29: EU AI Act high-risk compliance?
- **Direct answer:** Governance, documentation, monitoring, human oversight, security, and quality management system.

---

# Q30: DP lost accuracy — balance privacy/utility?
- **Direct answer:** Tune epsilon, better features, larger datasets, privacy budgeting, and accept trade-offs.

---

# Q31: Federated learning poisoning defense?
- **Direct answer:** Robust aggregation, anomaly detection, participant reputation, secure enclaves.

---

# Q32: Proxy discrimination?
- **Direct answer:** Detect proxy features, remove/regularize, use fairness-aware learning, causality checks.

---

# Q33: Feedback loops of bias — break them?
- **Direct answer:** Randomization, human oversight, counterfactual evaluation, policy constraints.

---

# Q34: Watermarking for fake news images?
- **Direct answer:** Watermark outputs, provenance metadata, detection tools, policy + takedown pipelines.

---

# Q35: Appeals process?
- **Direct answer:** Human review channel, explanations, correction workflow, SLA for disputes.

---

# Q36: Build audit trails (no logs)?
- **Direct answer:** Start now: structured logs + immutable storage + retention + access controls + replay tools.
- **Azure/DevOps bridge:** RBAC + Key Vault + immutable logs + approval gates + rollbacks.

---

# Q37: Prevent re-identification?
- **Direct answer:** Stronger anonymization, DP, k-anonymity checks, reduce quasi-identifiers, limit release.

---

# Q38: Detect backdoored open-source models?
- **Direct answer:** Scan, test triggers, compare behavior, use trusted sources, sandbox, red-team.

---

# Q39: Respond to poisoned training data?
- **Direct answer:** Contain, rollback, identify source, retrain from clean snapshot, tighten supply chain.

---

# Q40: Mental health chatbot harm mitigation?
- **Direct answer:** Crisis detection, escalation, safe responses, limit scope, human support links.
- **Mini prompt:** What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

---

# Q41: Run blameless postmortem?
- **Direct answer:** Timeline, contributing factors, action items, no blame; improve guardrails and monitoring.

---

# Q42: Prevent human over-reliance (automation bias)?
- **Direct answer:** Calibrate confidence, show uncertainty, require second checks, train users, audit overrides.

---

# Q43: Cross-cultural moderation failures?
- **Direct answer:** Locale-specific policies, multilingual data, local reviewers, adaptive thresholds.

---

# Q44: Reduce environmental impact?
- **Direct answer:** Efficient training, smaller models, reuse, quantize, schedule on clean energy, measure emissions.

---

## Flashcards

**Direct answer?** #flashcard
Ground with RAG/citations, constrain outputs, lower randomness, add verifiers/evals.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Direct: user says ‘ignore rules’; indirect: malicious text inside retrieved content. Defend with delimiters + tool allow-lists.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Validate inputs, constrain outputs (schemas), filter unsafe content, and add human approvals for risky actions.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Align behavior to human values/policies so the system is helpful and safe, not just fluent.

**Direct answer?** #flashcard
Measure per subgroup, test intersections, fix data/features/objectives, monitor drift.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Purpose limitation, minimization, consent, retention, access rights, auditability.

**Direct answer?** #flashcard
Redact, tokenize, isolate secrets, restrict logging, encrypt, and enforce retention policies.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Explain why a decision happened in a user/auditor-friendly way.

**Direct answer?** #flashcard
Interpretability: model is inherently understandable; explainability: post-hoc reasons/attributions.

**Direct answer?** #flashcard
Be transparent, cite sources, show uncertainty, allow appeals, and don’t overclaim.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Injection, evasion, backdoors; defend with validation, sandboxing, monitoring, red-teaming.

**Direct answer?** #flashcard
Attacker corrupts training/finetune/RAG corpora to change behavior or insert backdoors.

**Direct answer?** #flashcard
Multi-stage: input filter, generation constraints, output filter, human review for edge cases.

**Direct answer?** #flashcard
Use Microsoft/NIST-style principles: fairness, reliability, privacy, transparency, accountability.

**Direct answer?** #flashcard
Respect licenses, avoid regurgitation, track provenance, add similarity checks and policies.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Risk-based obligations: documentation, logging, human oversight, testing, transparency for high-risk systems.

**Direct answer?** #flashcard
Trace every decision: prompts, retrieval, tools, model version, outputs—redacted and access-controlled.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Documentation of intended use, limitations, data, evals, safety notes.

**Direct answer?** #flashcard
Rate limits, abuse detection, user verification, policy enforcement, incident response.

**Direct answer?** #flashcard
Add noise to training/queries to bound individual data leakage.

**Direct answer?** #flashcard
Detection → triage → containment → comms → rollback → postmortem → prevention.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Framework for mapping/measure/managing AI risks across lifecycle.

**Direct answer?** #flashcard
Constrain scope, require disclaimers, route to clinician, crisis escalation, block diagnosis claims.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Dedup data, similarity filters, retrieval grounding, refusal on near-match, output monitoring.

**Direct answer?** #flashcard
Audit features, remove proxies, rebalance data, adjust objective, human review, fairness constraints.

**Direct answer?** #flashcard
Evaluate intersections, not only marginals; targeted data + constraints; document trade-offs.

**Direct answer?** #flashcard
Provide understandable factors, process transparency, and an appeals path; log decisions.

**Direct answer?** #flashcard
Prefer not training on personal data; for weights: retrain or use unlearning/filters + legal guidance.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Governance, documentation, monitoring, human oversight, security, and quality management system.

**Direct answer?** #flashcard
Tune epsilon, better features, larger datasets, privacy budgeting, and accept trade-offs.

**Direct answer?** #flashcard
Robust aggregation, anomaly detection, participant reputation, secure enclaves.

**Direct answer?** #flashcard
Detect proxy features, remove/regularize, use fairness-aware learning, causality checks.

**Direct answer?** #flashcard
Randomization, human oversight, counterfactual evaluation, policy constraints.

**Direct answer?** #flashcard
Watermark outputs, provenance metadata, detection tools, policy + takedown pipelines.

**Direct answer?** #flashcard
Human review channel, explanations, correction workflow, SLA for disputes.

**Direct answer?** #flashcard
Start now: structured logs + immutable storage + retention + access controls + replay tools.

**Azure/DevOps bridge?** #flashcard
RBAC + Key Vault + immutable logs + approval gates + rollbacks.

**Direct answer?** #flashcard
Stronger anonymization, DP, k-anonymity checks, reduce quasi-identifiers, limit release.

**Direct answer?** #flashcard
Scan, test triggers, compare behavior, use trusted sources, sandbox, red-team.

**Direct answer?** #flashcard
Contain, rollback, identify source, retrain from clean snapshot, tighten supply chain.

**Direct answer?** #flashcard
Crisis detection, escalation, safe responses, limit scope, human support links.

**Mini prompt?** #flashcard
What’s the first safe fallback? → refuse/redirect + cite policy + escalate when needed.

**Direct answer?** #flashcard
Timeline, contributing factors, action items, no blame; improve guardrails and monitoring.

**Direct answer?** #flashcard
Calibrate confidence, show uncertainty, require second checks, train users, audit overrides.

**Direct answer?** #flashcard
Locale-specific policies, multilingual data, local reviewers, adaptive thresholds.

**Direct answer?** #flashcard
Efficient training, smaller models, reuse, quantize, schedule on clean energy, measure emissions.
