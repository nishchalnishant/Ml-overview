# Behavioral & scenario questions — answers that sound like you’ve shipped

These aren’t “soft” questions. They’re **risk questions**.

**Best rhythm (STAR, but human):** Situation → Stakes → Action → Result → What you’d do differently.

---

# Q1: What is AI Engineering, and how does it differ from ML Engineering?
- **Direct answer:** AI engineering focuses on shipping AI features end-to-end (LLMs, RAG, agents, safety, cost). ML engineering often centers on training/serving predictive models.

---

# Q2: How do you decide AI vs traditional software?
- **Direct answer:** If rules are stable → code it. If language/ambiguity is core → AI. If failure cost is high → add constraints/humans.

---

# Q3: How do you measure ROI of an AI feature?
- **Direct answer:** Define a business metric (deflection, conversion, time saved) + unit cost (tokens/req) + quality guardrails.

---

# Q4: Hallucinations in production — what do you do?
- **Direct answer:** Triage scope → reproduce with traces → add grounding/constraints → ship fix behind flags → add eval to prevent regression.

---

# Q5: LLM API vs self-host open model?
- **Direct answer:** API for speed and capability; self-host for control/cost at scale/privacy. Decide by latency, data sensitivity, and ops maturity.

---

# Q6: Managing stakeholder expectations?
- **Direct answer:** Show demos + known failure modes early. Define “good enough” with metrics. Don’t promise magic.

---

# Q7: Debug a poor RAG system.
- **Direct answer:** Check retrieval first (chunking, top-k, hybrid, rerank). Then prompt/format. Then evals.
- **Mini prompt:** If answers are fluent but wrong, what broke first? → retrieval/grounding.

---

# Q8: Staying current in AI?
- **Direct answer:** Track a few reliable sources, run small experiments, and focus on principles (retrieval, evals, safety) over hype.

---

# Q9: Balance innovation with reliability?
- **Direct answer:** Ship experiments behind flags, use eval gates, canary releases, and rollback plans.

---

# Q10: Challenging AI project story (template)
- **Use this frame:**
  - **Stakes:** what broke / what mattered
  - **Constraints:** latency, privacy, budget
  - **Decision:** prompt vs RAG vs fine-tune
  - **Guardrails:** schema, filters, HITL
  - **Result:** metric lift + incident reduction

---

# Q11: Biased/harmful outputs in prod?
- **Direct answer:** Contain (disable feature/route to safer mode), investigate, fix data/policy, add monitoring + audits.

---

# Q12: Exceeding budget — cost optimization?
- **Direct answer:** Reduce tokens (context compression), cache, route to smaller models, reduce retries, batch.

---

# Q13: Accuracy vs latency trade-off?
- **Direct answer:** Tie to user value (p95 SLA) and business cost. Use routing: fast model default, slow model for hard cases.

---

# Q14: Quality degrades over time?
- **Direct answer:** Monitor drift (data + behavior), refresh retrieval corpus, retrain/fine-tune when needed, add continuous evals.

---

# Q15: Communicate AI limitations to non-technical stakeholders?
- **Direct answer:** Use analogies: “autocomplete with confidence.” Show examples + safety rails. Set expectations with metrics.

---

# Q16: Limited labeled data?
- **Direct answer:** Start with prompting + RAG, then synthetic data + PEFT; build a golden eval set early.

---

# Q17: Cross-functional teamwork?
- **Direct answer:** Align on definitions (success/failure), document contracts (schemas), and agree on escalation paths.

---

# Q18: Where is AI engineering heading (3–5 years)?
- **Direct answer:** More routing, more eval automation, more on-device, more governance—less “prompt vibes,” more engineering.

---

# Q19: Why this role?
- **Direct answer:** Tie to shipping + operating AI systems: reliability, cost control, and user value.

---

# Q20: PM wants to ship with 15% hallucination rate — how communicate?
- **Direct answer:** Translate to user harm + support load + legal risk. Offer mitigations: scope limits, RAG, refusals, HITL.

---

# Q21: Exec asks “why not 100% accurate?”
- **Direct answer:** Explain stochastic generation + incomplete info. Compare to humans: high skill, not perfect; build guardrails and appeal paths.

---

# Q22: Better benchmark agentic system vs simpler maintainable RAG — decide?
- **Direct answer:** Optimize for total cost of ownership: maintainability + safety + on-call burden. Use agents only where they add real value.
