---
module: Llms
topic: Interview Notes
subtopic: Prompt Engineering Snappy
status: unread
tags: [llms, ml, interview-notes-prompt-enginee]
---
# Prompt engineering — reliability patterns (Azure/DevOps edition)

Prompting isn’t cute wording. It’s **interface + policy design** for a stochastic service.

**DevOps rule:** treat prompts like pipelines: **precedence**, **contracts**, **validation**, **threat model**, **regression tests**.

---

## The prompt hierarchy (aka “pipeline precedence”)

```mermaid
graph TD
  subgraph P0["P0: Highest priority (like org policies)"]
    A[System instructions]
    B[Safety / policy guardrails]
  end

  subgraph P1["P1: Guidance (like tests + artifacts)"]
    C[Few-shot examples]
    D[Retrieved context / evidence]
  end

  subgraph P2["P2: Runtime inputs"]
    E[User request]
    F[Dynamic variables]
  end

  A --> G[Final constructed prompt]
  B --> G
  C --> G
  D --> G
  E --> G
  F --> G
```

**Mini pop quiz:** If user text conflicts with system rules, who wins? → System/policy.

---

# Q1: What is prompt engineering, and why is it critical for AI applications?
- **Direct answer:** Designing system/user messages + constraints so the model is **predictable**, safe, and cost-efficient.
- **Azure/DevOps bridge:** It’s your **API contract** + quality gates.
- **Fashion analogy:** Styling with a strict dress code—beautiful *and* compliant.

---

# Q2: Explain zero-shot, one-shot, and few-shot prompting.
- **Direct answer:** Zero-shot = no examples; one-shot = 1 example; few-shot = several examples.
- **Trade-off:** More shots = more tokens = more cost/latency.

---

# Q3: What is chain-of-thought (CoT) prompting, and when should you use it?
- **Direct answer:** Encourage step-by-step reasoning. Use for complex reasoning; avoid exposing sensitive reasoning; prefer verifiable tools.

---

# Q4: Explain self-consistency prompting.
- **Direct answer:** Sample multiple solutions and pick the most consistent (vote/score).

---

# Q5: What is tree-of-thought prompting?
- **Direct answer:** Search multiple reasoning branches and prune; powerful but expensive.

---

# Q6: What is ReAct prompting?
- **Direct answer:** Reason → act (tool) → observe → repeat.
- **DevOps bridge:** It’s orchestration with tools (like a pipeline that can run tasks).

---

# Q7: What is a system prompt?
- **Direct answer:** Highest-priority instructions: role, constraints, format, refusal rules.
- **Common mistake:** putting secrets in prompts.

---

# Q8: How do you structure prompts for consistent structured output (JSON/XML)?
- **Direct answer:** Explicit schema + “ONLY JSON” + few-shot + constrained decoding / structured outputs when available.

---

# Q9: What is prompt injection, and how do you defend against it?
- **Direct answer:** Untrusted text tries to override instructions.
- **Defense:** delimit untrusted content, least-privilege tools, validation, red-team evals.

---

# Q10: What is jailbreaking? Common techniques?
- **Direct answer:** Attempts to bypass safety with role-play, obfuscation, multi-turn manipulation.

---

# Q11: How do you optimize prompts for cost and latency?
- **Direct answer:** Reduce tokens and retries; keep top-k retrieval small; prefer schemas over repair loops.

---

# Q12: Prompt engineering vs prompt tuning?
- **Direct answer:** Engineering = text instructions; tuning = learned “soft prompts”/vectors.

---

# Q13: What is a prompt template?
- **Direct answer:** Parameterized prompt with placeholders; version it like code.

---

# Q14: How do you handle multi-turn conversations?
- **Direct answer:** Summarize + keep key facts; don’t paste endless chat logs.

---

# Q15: What is role prompting?
- **Direct answer:** Assign a role to bias style; constraints still matter.

---

# Q16: What is prompt chaining?
- **Direct answer:** Multi-step prompts: extract → validate → decide → format.

---

# Q17: How do you evaluate and iterate on prompt quality?
- **Direct answer:** Build an eval set + regression tests; measure task success and format validity.

---

# Q18: What are meta-prompts?
- **Direct answer:** Prompts that generate prompts (useful, but can reduce determinism).

---

# Q19: Common prompting failure modes and debugging?
- **Direct answer:** Hallucination, format drift, instruction failure, verbosity, tool misuse. Fix via tighter contracts + evals.

---

# Q20: Edge cases and adversarial inputs?
- **Direct answer:** Treat like API threat model: validation, allow-lists, monitoring.

---

# Q21: Lost-in-the-middle in long-context prompting?
- **Direct answer:** Middle content is used less reliably. Put best chunks at top/bottom; keep prompts short.

---

# Q22: What are output parsers?
- **Direct answer:** Validate/normalize output into typed structures; prevents stringly-typed chaos.

---

# Q23: Multi-language prompting effectively?
- **Direct answer:** Keep instructions consistent; use multilingual models/embeddings; budget for token tax.

---

# Q24: Few-shot inconsistent—how stabilize?
- **Fix:** normalize inputs; curate/reorder examples; reduce temperature; add schemas.

---

# Q25: Classification too sensitive to wording—reduce sensitivity.
- **Fix:** constrain label set; add counterexamples; structured output; eval-driven iteration.

---

# Q26: System prompt leaks—how prevent?
- **Direct answer:** Don’t store secrets in prompts; separate policy from data; use server-side rules.

---

# Q27: Agent prompt-injection defenses?
- **Direct answer:** treat tool inputs as untrusted; validate args; sandbox tools; log/audit.

---

# Q28: CoT not improving accuracy—what fix?
- **Fix:** decompose task; add verifiers/tools; ensure needed context via RAG.

---

# Q29: Works in English but fails elsewhere—multilingual support?
- **Fix:** locale-specific examples + evals; retrieval in target language; multilingual embeddings.

---

# Q30: Zero-shot cross-lingual transfer fails—how fix?
- **Fix:** translate, add few-shot per language, or fine-tune adapters.

## Flashcards

**Direct answer?** #flashcard
Designing system/user messages + constraints so the model is predictable, safe, and cost-efficient.

**Azure/DevOps bridge?** #flashcard
It’s your API contract + quality gates.

**Fashion analogy?** #flashcard
Styling with a strict dress code—beautiful and compliant.

**Direct answer?** #flashcard
Zero-shot = no examples; one-shot = 1 example; few-shot = several examples.

**Trade-off?** #flashcard
More shots = more tokens = more cost/latency.

**Direct answer?** #flashcard
Encourage step-by-step reasoning. Use for complex reasoning; avoid exposing sensitive reasoning; prefer verifiable tools.

**Direct answer?** #flashcard
Sample multiple solutions and pick the most consistent (vote/score).

**Direct answer?** #flashcard
Search multiple reasoning branches and prune; powerful but expensive.

**Direct answer?** #flashcard
Reason → act (tool) → observe → repeat.

**DevOps bridge?** #flashcard
It’s orchestration with tools (like a pipeline that can run tasks).

**Direct answer?** #flashcard
Highest-priority instructions: role, constraints, format, refusal rules.

**Common mistake?** #flashcard
putting secrets in prompts.

**Direct answer?** #flashcard
Explicit schema + “ONLY JSON” + few-shot + constrained decoding / structured outputs when available.

**Direct answer?** #flashcard
Untrusted text tries to override instructions.

**Defense?** #flashcard
delimit untrusted content, least-privilege tools, validation, red-team evals.

**Direct answer?** #flashcard
Attempts to bypass safety with role-play, obfuscation, multi-turn manipulation.

**Direct answer?** #flashcard
Reduce tokens and retries; keep top-k retrieval small; prefer schemas over repair loops.

**Direct answer?** #flashcard
Engineering = text instructions; tuning = learned “soft prompts”/vectors.

**Direct answer?** #flashcard
Parameterized prompt with placeholders; version it like code.

**Direct answer?** #flashcard
Summarize + keep key facts; don’t paste endless chat logs.

**Direct answer?** #flashcard
Assign a role to bias style; constraints still matter.

**Direct answer?** #flashcard
Multi-step prompts: extract → validate → decide → format.

**Direct answer?** #flashcard
Build an eval set + regression tests; measure task success and format validity.

**Direct answer?** #flashcard
Prompts that generate prompts (useful, but can reduce determinism).

**Direct answer?** #flashcard
Hallucination, format drift, instruction failure, verbosity, tool misuse. Fix via tighter contracts + evals.

**Direct answer?** #flashcard
Treat like API threat model: validation, allow-lists, monitoring.

**Direct answer?** #flashcard
Middle content is used less reliably. Put best chunks at top/bottom; keep prompts short.

**Direct answer?** #flashcard
Validate/normalize output into typed structures; prevents stringly-typed chaos.

**Direct answer?** #flashcard
Keep instructions consistent; use multilingual models/embeddings; budget for token tax.

**Fix?** #flashcard
normalize inputs; curate/reorder examples; reduce temperature; add schemas.

**Fix?** #flashcard
constrain label set; add counterexamples; structured output; eval-driven iteration.

**Direct answer?** #flashcard
Don’t store secrets in prompts; separate policy from data; use server-side rules.

**Direct answer?** #flashcard
treat tool inputs as untrusted; validate args; sandbox tools; log/audit.

**Fix?** #flashcard
decompose task; add verifiers/tools; ensure needed context via RAG.

**Fix?** #flashcard
locale-specific examples + evals; retrieval in target language; multilingual embeddings.

**Fix?** #flashcard
translate, add few-shot per language, or fine-tune adapters.
