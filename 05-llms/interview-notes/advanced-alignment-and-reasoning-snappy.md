---
module: Llms
topic: Interview Notes
subtopic: Advanced Alignment And Reasoning Snappy
status: unread
tags: [llms, ml, interview-notes-advanced-align]
---

> _Quick-recall companion. For the full deep-dive, see [advanced-alignment-and-reasoning.md](advanced-alignment-and-reasoning.md)._

# Advanced alignment & reasoning — the “senior vibes” set

This is where you stop saying “prompting” and start saying **policy, preferences, and verification**.

---

## Q1: What is RLHF (Reinforcement Learning from Human Feedback)?
- **Direct answer:** Align a base model to be helpful/safe using preference feedback (often SFT → reward signal → optimization like PPO historically).
- **DevOps bridge:** It’s like writing a policy + running continuous coaching so the system behaves under stress, not just in a unit test.

**Mini prompt:** Does RLHF teach new facts? → No. It mostly teaches behavior/style/safety.

---

## Q2: RLHF vs DPO (Direct Preference Optimization)
- **RLHF (PPO):** powerful but operationally finicky.
- **DPO:** trains directly on (chosen, rejected) pairs with a stable supervised-style objective.

**Analogy:** RLHF is a complicated training camp with a coach shouting every rep; DPO is “here’s the good answer, here’s the bad one—learn the difference.”

---

## Q3: What is Chain-of-Thought (CoT) prompting?
- **Direct answer:** Encourage intermediate reasoning steps to improve performance on multi-step tasks.
- **Production note:** Prefer hidden scratchpads + verifiers/tools over dumping long reasoning into user-visible output.

**Ghazal hook:** The meaning isn’t in the last word alone—it’s in the path that led there. CoT is that path (when you let it exist).

---

## Q4: Explain ReAct (Reasoning + Acting).
- **Direct answer:** Combine reasoning with tool use: think → act (tool) → observe → repeat.
- **Why it matters:** tools turn “guessing” into “checking.”

---

## Q5: What are AI agents and agentic workflows?
- **Direct answer:** Systems that plan, call tools, track state, and stop when goals are met (or budgets hit).
- **Guardrails:** max steps, budgets, allow-listed tools, HITL for irreversible actions.

**MI analogy:** Great captains don’t bowl one plan for 20 overs—they adapt each over, but with guardrails.
