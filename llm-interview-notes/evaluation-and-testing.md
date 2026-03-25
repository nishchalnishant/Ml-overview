# Q1: What is evaluation-driven development for AI applications?

## 1. 🔹 Direct Answer
Evaluation-driven development (EDD) is a workflow where you define measurable quality criteria first, then iteratively change prompts/models/pipelines while continuously running automated tests and metrics. It prevents regressions and helps you optimize the right objective for product outcomes.

## 2. 🔹 Intuition
You don't “guess” if changes helped—you measure them on representative tests.

## 3. 🔹 Deep Dive
Typical loop:
1. Define success metrics (task accuracy, faithfulness, safety, format validity, cost/latency).
2. Build or curate eval datasets (gold + adversarial).
3. Run baseline evaluation.
4. Implement change (prompt/pipeline/model).
5. Re-evaluate and compare with statistical tests.
6. Deploy only if no regressions (or if improvements are significant).

## 4. 🔹 Practical Perspective
- Use when: LLM apps are probabilistic and can regress silently.
- Trade-offs: building evals takes upfront effort; mitigated by incremental evals and canaries.

## 5. 🔹 Code Snippet
```python
baseline = run_eval(model="v1", dataset=eval_ds)
candidate = run_eval(model="v2", dataset=eval_ds)
compare(baseline, candidate, metric="faithfulness")
```

## 6. 🔹 Interview Follow-ups
1. Q: What should never be the only metric?  
   A: Only surface text overlap metrics (e.g., BLEU) without faithfulness/format checks.
2. Q: How do you keep eval data relevant?  
   A: Sample from real traffic, add failure cases, and monitor drift.

##  7. 🔹 Common Mistakes
- Evaluating on a dataset that doesn’t match production distribution.

## 8. 🔹 Comparison / Connections
- Connects to CI/CD but for model behavior (continuous testing).

## 9. 🔹 One-line Revision
EDD is iterate-with-metrics: define quality, test continuously, and deploy only when evals confirm improvements.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: How do you evaluate LLM outputs? What metrics do you use?

## 1. 🔹 Direct Answer
Evaluate with task-specific success metrics plus output-quality metrics: correctness/grounding (faithfulness), relevance, format validity, safety/refusal behavior, and optionally semantic similarity or LLM-judge scores. For generation, use both automatic metrics and human sampling.

## 2. 🔹 Intuition
We measure multiple dimensions because LLM failure modes differ (wrong facts vs wrong format vs unsafe content).

## 3. 🔹 Deep Dive
Common categories:
- **Text quality**: fluency/perplexity (model-centric).
- **Task metrics**: exact match, F1, ROUGE for summarization, accuracy for classification.
- **Faithfulness/grounding**: does output match provided evidence (RAG)?
- **Safety**: policy violation rate, refusal correctness.
- **Format**: JSON schema validity, tool call validity.

## 4. 🔹 Practical Perspective
- Use: offline eval suites + online user feedback.
- Avoid: relying on a single metric; conflicts are common.

## 5. 🔹 Code Snippet
```python
metrics = {
  "format_valid": is_valid_json,
  "task_accuracy": exact_match,
  "faithfulness": entailment_check,
}
```

## 6. 🔹 Interview Follow-ups
1. Q: When do you use BERTScore/semantic metrics?  
   A: For similarity when exact match is too strict, but still validate faithfulness.

## 7. 🔹 Common Mistakes
- Measuring only overlap without assessing whether facts are supported.

## 8. 🔹 Comparison / Connections
- Connects to **RAG evaluation** and **safety evaluation**.

## 9. 🔹 One-line Revision
Use a metric stack: task success + faithfulness/relevance + format + safety, validated with humans when needed.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: Explain BLEU, ROUGE, and BERTScore. When would you use each?

## 1. 🔹 Direct Answer
BLEU measures n-gram precision (common for translation). ROUGE measures n-gram/sequence overlap with recall bias (often for summarization). BERTScore measures semantic similarity using contextual embeddings (more robust to paraphrases).

## 2. 🔹 Intuition
BLEU/ROUGE ask "how much text overlap?" BERTScore asks "how similar in meaning?"

## 3. 🔹 Deep Dive
High-level:
- **BLEU**: geometric mean of n-gram precisions with brevity penalty.
- **ROUGE**: variants like ROUGE-L (LCS) and ROUGE-1/2 recall-ish overlap.
- **BERTScore**:
  - compute token-level cosine similarity using BERT embeddings
  - aggregate with precision/recall/F1-like scoring.

## 4. 🔹 Practical Perspective
- Use: non-grounded summarization quality comparison.
- Avoid: when faithfulness matters (RAG); lexical metrics can score fluent hallucinations.

## 5. 🔹 Code Snippet
```python
# conceptual: BERTScore uses embeddings similarity
score = bertscore(reference, hypothesis)  # returns P/R/F1-like
```

## 6. 🔹 Interview Follow-ups
1. Q: Why do these fail for hallucination detection?  
   A: Overlap metrics don’t check whether statements are supported by evidence.

## 7. 🔹 Common Mistakes
- Treating ROUGE as truthfulness.

## 8. 🔹 Comparison / Connections
- Connects to evaluation for **NLG** but complement with faithfulness checks.

## 9. 🔹 One-line Revision
BLEU/ROUGE are overlap-based; BERTScore approximates semantic similarity; none ensure grounding.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q4: What is G-Eval, and how does it use LLMs for evaluation?

## 1. 🔹 Direct Answer
G-Eval is an evaluation method where an LLM judges generated outputs using rubric-based prompts and scoring guidelines. It typically produces scores on task attributes like correctness, completeness, or faithfulness.

## 2. 🔹 Intuition
Instead of humans grading every example, you use a model as a grader with strict rubrics.

## 3. 🔹 Deep Dive
Workflow:
1. Construct a rubric prompt: criteria and scoring scale.
2. Provide (question, reference if available, model output).
3. The evaluator LLM returns a score and rationale (often structured).
4. Aggregate scores across dataset; validate with human agreement.

## 4. 🔹 Practical Perspective
- Use: fast iteration and large-scale screening.
- Avoid: assuming LLM-judge is ground truth; it can share biases with the evaluated model.

## 5. 🔹 Code Snippet
```python
judge_prompt = f"Rubric: ... Score 1-5.\nQuestion: {q}\nOutput: {y}\nScore:"
score = llm.generate(judge_prompt)
```

## 6. 🔹 Interview Follow-ups
1. Q: How to reduce judge bias?  
   A: Use a stronger or different judge model, calibrate, and run human checks on subsets.

## 7. 🔹 Common Mistakes
- Not aligning judge rubric with the actual product requirement.

## 8. 🔹 Comparison / Connections
- Connects to **LLM-as-a-judge** evaluation.

## 9. 🔹 One-line Revision
G-Eval uses an LLM grader with rubrics to produce scalable evaluation scores.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: What is LLM-as-a-judge evaluation, and what are its limitations?

## 1. 🔹 Direct Answer
LLM-as-a-judge evaluation uses a separate LLM (the judge) to score model outputs against a rubric. Limitations include judge bias, overfitting to style, poor calibration, and inability to verify facts without evidence/grounding.

## 2. 🔹 Intuition
It is like hiring another reviewer—but the reviewer can be wrong or inconsistent.

## 3. 🔹 Deep Dive
Limitations:
- **Bias & leakage**: judge might prefer familiar phrasing.
- **Faithfulness**: without evidence, the judge may grade plausibility.
- **Reproducibility**: scores can vary with temperature.
Mitigations:
- rubric + structured outputs
- hold-out calibration set with human labels
- enforce evidence-based judging for RAG (provide retrieved context)

## 4. 🔹 Practical Perspective
- Use: scalable evaluation for format, quality, and some reasoning checks.
- Avoid: safety-critical decisions without human validation and evidence.

## 5. 🔹 Code Snippet
```python
judge = llm.generate(judge_prompt, temperature=0.0)
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you measure judge reliability?  
   A: Compute inter-annotator agreement and judge-human correlation on a subset.

## 7. 🔹 Common Mistakes
- Using judge scores as the only gating signal for deployment.

## 8. 🔹 Comparison / Connections
- Connects to EDD and continuous evaluation.

## 9. 🔹 One-line Revision
LLM judges scale evaluation but can be biased; calibrate and ground judging with evidence.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q6: How do you conduct human evaluation for AI systems?

## 1. 🔹 Direct Answer
Conduct human evaluation by defining rubrics, training annotators, using a representative test set, ensuring inter-annotator agreement, and sampling enough outputs for statistical power. Use human scores to calibrate automatic/LLM-judge metrics.

## 2. 🔹 Intuition
Humans validate nuance that metrics miss, but you must standardize the task for consistency.

## 3. 🔹 Deep Dive
Best practices:
- define clear annotation guidelines
- include examples of high/medium/low quality
- multi-annotate for ambiguous cases
- compute agreement (e.g., Cohen's kappa)
- calibrate model decisions to reduce evaluator drift

## 4. 🔹 Practical Perspective
- Use: safety, factuality, complex reasoning, and final release gates.
- Trade-offs: expensive and time-consuming.

## 5. 🔹 Code Snippet
```python
# conceptual: collect labels
labels = annotators.map(item_to_rubric)
agreement = compute_kappa(labels)
```

## 6. 🔹 Interview Follow-ups
1. Q: How many samples?  
   A: Enough to detect meaningful differences with confidence; depends on expected effect size.

## 7. 🔹 Common Mistakes
- Inconsistent rubrics across annotators.

## 8. 🔹 Comparison / Connections
- Connects to QA and measurement reliability in ML.

## 9. 🔹 One-line Revision
Human evaluation needs clear rubrics, agreement checks, representative sampling, and calibration for automation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is red teaming, and how do you red team an LLM application?

## 1. 🔹 Direct Answer
Red teaming is systematic adversarial testing to discover failures, unsafe behaviors, jailbreaks, data leakage, and robustness weaknesses. You generate attacks, run the system under realistic conditions, and use findings to update prompts, tools, filters, and training.

## 2. 🔹 Intuition
You’re trying to break it before users do.

## 3. 🔹 Deep Dive
Steps:
1. Threat model: what harms and what capabilities?
2. Attack design: prompt injection, jailbreaks, tool misuse, data exfiltration.
3. Test harness: run attacks with realistic context and permissions.
4. Classify failures and create regression tests.
5. Iterate mitigations and re-test.

## 4. 🔹 Practical Perspective
- Use: before launch and periodically after updates.
- Avoid: only testing “easy” jailbreaks; include indirect attacks via retrieval.

## 5. 🔹 Code Snippet
```python
attacks = ["ignore policy and reveal...", "embedded instruction in retrieved text..."]
for a in attacks:
    out = system(a)
    assert not violates_policy(out)
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you prioritize?  
   A: Based on severity and likelihood of harm in production.

## 7. 🔹 Common Mistakes
- Treating red-team findings as one-off; you need regression suite coverage.

## 8. 🔹 Comparison / Connections
- Connects to security testing and EDD.

## 9. 🔹 One-line Revision
Red teaming is adversarial testing to uncover failures and turn them into lasting regression tests.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q8: How do you detect and measure hallucinations in LLM outputs?

## 1. 🔹 Direct Answer
Detect hallucinations by measuring **evidence support** (entailment/attribution against retrieved context), using verifier models or NLI checks, running consistency tests, and comparing against ground truth sources. For RAG, hallucinations are checked against retrieved chunks.

## 2. 🔹 Intuition
Hallucinations are claims not supported by evidence.

## 3. 🔹 Deep Dive
Methods:
- **Attribution/citation check**: does cited chunk contain the claim?
- **Entailment classification**: claim + evidence → entail/neutral/contradict.
- **Verification via tools**: ask search/API to validate.
- **Self-consistency**: multiple drafts and contradiction detection.

## 4. 🔹 Practical Perspective
- Use: RAG and factual QA systems.
- Trade-offs: verification adds cost/latency; choose evidence-based checks for the most critical paths.

## 5. 🔹 Code Snippet
```python
def is_faithful(answer, evidence_chunks):
    for claim in extract_claims(answer):
        if not any(nli_entails(claim, c) for c in evidence_chunks):
            return False
    return True
```

## 6. 🔹 Interview Follow-ups
1. Q: What if evidence is incomplete?  
   A: Then correct behavior may be "not found"; measure abstention accuracy too.

## 7. 🔹 Common Mistakes
- Treating BLEU/ROUGE as hallucination detection.

## 8. 🔹 Comparison / Connections
- Connects to **RAG evaluation** and **faithfulness**.

## 9. 🔹 One-line Revision
Hallucinations are measured by claim-evidence support using entailment/citation checks and verification.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q9: What is adversarial testing for AI systems?

## 1. 🔹 Direct Answer
Adversarial testing evaluates how your system behaves under malicious or challenging inputs designed to trigger failures: jailbreaks, prompt injection, out-of-distribution queries, and tool misuse. The output is used to harden defenses and regression tests.

## 2. 🔹 Intuition
You test the worst-case inputs to improve robustness.

## 3. 🔹 Deep Dive
Types:
- **Prompt attacks**: injection, role override, obfuscation.
- **Tool attacks**: invalid args, destructive operations, infinite loops.
- **Data attacks**: poisoned retrieval documents.
- **Robustness**: distribution shifts in domain/style.
Process: create attack suite → run → label failures → mitigate → add to regression suite.

## 4. 🔹 Practical Perspective
- Use: safety/security-driven features and agentic systems.
- Trade-offs: generating attacks is time-consuming; prioritize high-risk failure classes.

## 5. 🔹 Code Snippet
```python
attack_suite = generate_attacks(threat_model)
results = [run_case(a) for a in attack_suite]
```

## 6. 🔹 Interview Follow-ups
1. Q: How to avoid overfitting to attacks?  
   A: Keep a held-out eval set and update attacks periodically.

## 7. 🔹 Common Mistakes
- Only testing user prompt, not retrieved context and tool outputs.

## 8. 🔹 Comparison / Connections
- Connects to security engineering and red teaming.

## 9. 🔹 One-line Revision
Adversarial testing finds vulnerabilities by running the system against worst-case malicious inputs.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q10: How do you build a regression test suite for AI applications?

## 1. 🔹 Direct Answer
Build a regression suite by collecting representative real traffic cases, labeled failure cases, and adversarial/red-team attacks; then test model versions, prompts, and pipeline components on every change. Track metrics and enforce gating thresholds.

## 2. 🔹 Intuition
Regression tests prevent “fix one thing and break another.”

## 3. 🔹 Deep Dive
Regression suite components:
- task correctness eval set
- format validity tests (JSON/tool calls)
- faithfulness/citation checks (if RAG)
- safety checks
- latency/cost thresholds
Implementation: store inputs and expected properties, run automatically in CI, and compare distributions.

## 4. 🔹 Practical Perspective
- Use: before prompt/model changes in production.
- Avoid: tiny eval sets that miss real failure modes.

## 5. 🔹 Code Snippet
```python
for case in regression_cases:
    out = system(case.input)
    assert case.expect(out)  # correctness/format/safety
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you select cases?  
   A: Representative sampling + failure clustering + stratify by risk.

## 7. 🔹 Common Mistakes
- Not versioning prompts and eval results together.

## 8. 🔹 Comparison / Connections
- Connects to EDD and CI/CD.

## 9. 🔹 One-line Revision
Regression tests are automated, versioned, and cover correctness, format, faithfulness, safety, and constraints.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: What are benchmark suites (MMLU, HumanEval, GSM8K), and how do you interpret them?

## 1. 🔹 Direct Answer
Benchmark suites are curated datasets for measuring model performance on tasks:
MMLU (knowledge/understanding across subjects), HumanEval (code generation quality), GSM8K (grade-school math). Interpret them as indicators of general capability, not guarantees for your specific product domain.

## 2. 🔹 Intuition
Benchmarks approximate broad skill coverage, but your app has different constraints and data.

## 3. 🔹 Deep Dive
Interpretation:
- Compare models with the same evaluation pipeline and decoding settings.
- Analyze failure categories (e.g., reasoning, tool use, format).
- Use as starting point; your product should rely on task-specific eval.

## 4. 🔹 Practical Perspective
- Use: model selection and capability sanity checks.
- Avoid: shipping solely based on benchmark leaderboards.

## 5. 🔹 Code Snippet
```python
score = evaluate_on_suite(model, suite="GSM8K")
```

## 6. 🔹 Interview Follow-ups
1. Q: Why can a benchmark model still fail in your app?  
   A: Domain shift, different input formatting, tool needs, and evidence constraints.

## 7. 🔹 Common Mistakes
- Confusing benchmark scores with faithfulness/safety in real RAG.

## 8. 🔹 Comparison / Connections
- Connects to evaluation selection and distribution alignment.

## 9. 🔹 One-line Revision
Benchmarks are useful signals for general capability, but product evaluation must be domain-specific and evidence-aware.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: How do you evaluate a RAG system end-to-end?

## 1. 🔹 Direct Answer
Evaluate end-to-end RAG by measuring:
1) retrieval quality (Recall@k, MRR),
2) context precision/recall under filters,
3) answer faithfulness (supported claims),
4) answer relevance to the question,
5) abstention correctness for unanswerable queries,
6) latency/cost.

## 2. 🔹 Intuition
You test the full pipeline: did it fetch the right evidence and did it use only that evidence?

## 3. 🔹 Deep Dive
Setup:
- Build QA eval cases with gold evidence (or gold answers with source spans).
- Run RAG and log retrieved chunks.
- Compute metrics:
  - retrieval Recall@k
  - faithfulness (entailment against retrieved chunks)
  - citation accuracy
  - relevance scoring

## 4. 🔹 Practical Perspective
- Use: any production QA/search assistant.
- Trade-offs: end-to-end eval is more expensive but most predictive.

## 5. 🔹 Code Snippet
```python
retrieved = retriever.retrieve(q, top_k=10)
faith = faithfulness_check(answer, retrieved)
rel = relevance_score(question=q, answer=answer)
```

## 6. 🔹 Interview Follow-ups
1. Q: What about unanswerable questions?  
   A: Include them and evaluate abstention accuracy (not hallucinating).

## 7. 🔹 Common Mistakes
- Only scoring answers and ignoring retrieval failures.

## 8. 🔹 Comparison / Connections
- Connects to **RAG evaluation** and observability.

## 9. 🔹 One-line Revision
End-to-end RAG eval measures retrieval + faithfulness + relevance + abstention plus operational constraints.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: How do you evaluate the quality of AI agents?

## 1. 🔹 Direct Answer
Evaluate agents by measuring task success rate, tool-call correctness, reasoning/planning quality (often via structured logs), safety/guardrail adherence, and efficiency (latency, token/cost, number of tool calls). Include failure mode regression tests.

## 2. 🔹 Intuition
Agents succeed when they act correctly step-by-step, not just when the final answer looks good.

## 3. 🔹 Deep Dive
Agent eval includes:
- final answer correctness (if applicable)
- **tool correctness**: correct tool chosen and correct arguments
- state management: consistent memory updates
- error recovery: does it recover from tool failures?
- stop conditions: avoids infinite loops

## 4. 🔹 Practical Perspective
- Use: multi-step assistant, code execution agents, customer support agents.
- Avoid: evaluating only final text; it hides tool/planning failures.

## 5. 🔹 Code Snippet
```python
success = (final_answer_correct and tool_calls_valid and no_safety_violation)
```

## 6. 🔹 Interview Follow-ups
1. Q: How to measure tool call quality?  
   A: Validate against schemas and check call arguments/side effects with sandboxes.

## 7. 🔹 Common Mistakes
- No logs, so you cannot attribute failures to retrieval, tool choice, or reasoning.

## 8. 🔹 Comparison / Connections
- Connects to agent observability and evaluation-driven development.

## 9. 🔹 One-line Revision
Agent quality is step-by-step correctness: tool choice/args, recovery, and safety, not only final outputs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: What is the difference between offline and online evaluation for AI systems?

## 1. 🔹 Direct Answer
Offline evaluation tests model/prompt behavior on curated datasets without real user interaction. Online evaluation measures performance in production with real traffic (A/B tests, canaries, and user feedback), including latency and drift effects.

## 2. 🔹 Intuition
Offline tells you what might happen; online tells you what actually happens with real users.

## 3. 🔹 Deep Dive
- Offline:
  - deterministic inputs; cheaper iteration
  - uses gold labels/rubrics
- Online:
  - real distribution and context
  - includes cost, latency, and behavioral change
Best practice: offline for rapid iteration and model selection; online for final validation.

## 4. 🔹 Practical Perspective
- Use: combine both to avoid dataset mismatch.
- Trade-off: online is riskier and more expensive.

## 5. 🔹 Code Snippet
```python
# offline
scores = run_eval(model="candidate", dataset=eval_ds)
# online
ab_test("candidate", metric="user_accept_rate")
```

## 6. 🔹 Interview Follow-ups
1. Q: What online metrics replace offline ones?  
   A: User satisfaction, completion rate, escalation rate, and measured latency/cost.

## 7. 🔹 Common Mistakes
- Launching without offline regression tests.

## 8. 🔹 Comparison / Connections
- Connects to CI/CD and MLOps monitoring.

## 9. 🔹 One-line Revision
Offline eval uses curated datasets; online eval measures real user impact and operational behavior.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q15: How do you measure factual consistency in LLM outputs?

## 1. 🔹 Direct Answer
Measure factual consistency by checking each claim against evidence:
use entailment/NLI with retrieved context, citation verification in RAG, or external verifiers/tools for ground truth. Report factuality rate and contradiction rate.

## 2. 🔹 Intuition
Factual consistency is “can we support the claim with evidence?”

## 3. 🔹 Deep Dive
- Extract atomic claims from the answer.
- For each claim:
  - find relevant evidence chunk(s)
  - check entailment/contradiction
- Aggregate metrics:
  - factuality precision
  - contradiction rate
  - unanswerable/abstention accuracy

## 4. 🔹 Practical Perspective
- Use: RAG QA, policy assistants, and compliance workflows.
- Avoid: summarization eval without evidence; factuality may be unmeasurable.

## 5. 🔹 Code Snippet
```python
for claim in claims:
    if not entailment_exists(claim, evidence):
        hallucinations += 1
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you extract claims reliably?  
   A: Use structured extraction prompts or NLI-based claim decomposition.

## 7. 🔹 Common Mistakes
- Checking faithfulness only at the paragraph level instead of per claim.

## 8. 🔹 Comparison / Connections
- Connects to **RAG faithfulness** and hallucination mitigation.

## 9. 🔹 One-line Revision
Factual consistency is measured by claim-evidence entailment and contradiction detection (evidence-based verification).

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q16: How do you evaluate multi-turn conversation quality?

## 1. 🔹 Direct Answer
Evaluate conversation quality by measuring task success across turns, consistency with prior context, proper handling of follow-ups/escalations, memory correctness (no lost preferences), and user satisfaction. Include dialogue-level metrics like coherence and grounding.

## 2. 🔹 Intuition
Good chat is not a sequence of good answers; it’s a consistent dialogue.

## 3. 🔹 Deep Dive
Metrics:
- **Context adherence:** did the assistant keep track of earlier constraints?
- **Resolution success:** completed user goal by last turn.
- **Error recovery:** handled ambiguous or changed topic gracefully.
- **Memory correctness:** stored entities/preferences correctly updated.

## 4. 🔹 Practical Perspective
- Use: customer support, copilots, and long-running tasks.
- Avoid: only evaluating final message; mid-turn failures matter.

## 5. 🔹 Code Snippet
```python
quality = (
  adherence_score(history_states) >= threshold and
  final_goal_met(user_goal)
)
```

## 6. 🔹 Interview Follow-ups
1. Q: How to create eval sets?  
   A: Sample multi-turn sessions from logs, label with rubrics, and add adversarial dialogue patterns.

## 7. 🔹 Common Mistakes
- Not testing topic switches and user corrections.

## 8. 🔹 Comparison / Connections
- Connects to **memory systems** and **lost-in-middle** in conversations.

## 9. 🔹 One-line Revision
Multi-turn eval checks consistency, memory correctness, and goal completion—not only single-turn answer quality.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: What is the role of golden datasets in AI evaluation?

## 1. 🔹 Direct Answer
Golden datasets (golden sets) are curated examples with trusted labels or reference evidence used as the benchmark for correctness and for calibrating automatic metrics. They provide a stable evaluation anchor for regression detection.

## 2. 🔹 Intuition
They’re your ground truth checklist.

## 3. 🔹 Deep Dive
- Golden datasets should cover:
  - typical cases
  - edge cases
  - failure modes (unanswerable questions, injection attempts)
- They must be versioned and audited since prompts/models change over time.

## 4. 🔹 Practical Perspective
- Use: core gating metrics before deployment.
- Avoid: using tiny golden sets with no diversity; leads to overfitting.

## 5. 🔹 Code Snippet
```python
golden_cases = load_golden("rag_faithfulness_v3")
run_eval(model, golden_cases)
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you update golden datasets?  
   A: Add new real failures periodically; keep old subsets for continuity.

## 7. 🔹 Common Mistakes
- Replacing golden labels without versioning; breaks comparability.

## 8. 🔹 Comparison / Connections
- Connects to **evaluation governance** and reproducibility.

## 9. 🔹 One-line Revision
Golden datasets provide stable, trusted benchmarks for regression testing and metric calibration.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q18: How do you implement continuous evaluation for production AI systems?

## 1. 🔹 Direct Answer
Continuous evaluation runs scheduled and triggered tests on new traffic samples and model/prompt versions, logs outputs and failures, and alerts on metric drift. Combine offline regression with online monitoring and automatic fallback when quality degrades.

## 2. 🔹 Intuition
Production changes over time (data, user behavior, model versions). Evaluation must keep up.

## 3. 🔹 Deep Dive
Components:
- data sampling pipeline from production
- eval runner (offline judges, parsers, faithfulness checks)
- drift detection and alerting
- automated regression gating for future deployments

## 4. 🔹 Practical Perspective
- Use: after launch and after any major system updates.
- Trade-offs: monitoring cost; mitigate by sampling and stratified eval.

## 5. 🔹 Code Snippet
```python
samples = sample_traffic(period="daily", stratify_by=["lang","tenant"])
scores = run_eval(model=current, dataset=samples)
alert_if(scores["faithfulness"] < threshold)
```

## 6. 🔹 Interview Follow-ups
1. Q: What do you do when metrics drift?  
   A: Roll back prompt/model, update retrieval/index, re-train adapters, and add failing cases to golden sets.

## 7. 🔹 Common Mistakes
- Monitoring only latency/cost and ignoring quality.

## 8. 🔹 Comparison / Connections
- Connects to MLOps monitoring and reliability.

## 9. 🔹 One-line Revision
Continuous evaluation samples production data, runs quality tests, and alerts on drift with automated mitigation paths.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: How do you evaluate bias in AI model outputs?

## 1. 🔹 Direct Answer
Evaluate bias by measuring performance or error rates across demographic and intersectional groups, using fairness metrics, and checking whether harms correlate with protected attributes. Use counterfactual or adversarial test sets to detect subtle bias.

## 2. 🔹 Intuition
Bias is unequal behavior across groups, not just average quality.

## 3. 🔹 Deep Dive
Process:
1. Define protected attributes and groups (carefully and legally).
2. Build balanced test sets or use augmentation/counterfactual pairs.
3. Measure:
   - differential error rates
   - demographic parity metrics
   - calibration differences
4. Validate intersectional failures (not only single attributes).

## 4. 🔹 Practical Perspective
- Use: hiring, loans, medical risk scoring, moderation.
- Avoid: relying solely on aggregate metrics without group stratification.

## 5. 🔹 Code Snippet
```python
for group in groups:
    rate = compute_error_rate(outputs[group])
    compare_to_reference(rate)
```

## 6. 🔹 Interview Follow-ups
1. Q: What about intersectional groups?  
   A: Test directly on combinations; otherwise you can miss proxy discrimination.

## 7. 🔹 Common Mistakes
- Evaluating only gender or only race separately.

## 8. 🔹 Comparison / Connections
- Connects to fairness in classical ML and modern LLM evaluation.

## 9. 🔹 One-line Revision
Bias evaluation compares error/performance across groups and tests intersectional conditions with fairness metrics.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q20: How do you compare two models or prompts in a statistically rigorous way?

## 1. 🔹 Direct Answer
Compare using paired evaluation (same inputs), compute effect sizes on metrics, and apply statistical tests (paired t-test/bootstrap for continuous scores; McNemar for paired classification outcomes). Use confidence intervals and control for multiple comparisons if needed.

## 2. 🔹 Intuition
You want evidence the difference isn’t just random variation.

## 3. 🔹 Deep Dive
Paired testing:
- For classification metrics: McNemar’s test on paired successes/failures.
- For continuous metrics: bootstrap confidence intervals or paired t-test (if assumptions hold).
Also report:
- sample size
- effect size and confidence interval

## 4. 🔹 Practical Perspective
- Use for model selection and prompt rollouts.
- Avoid: interpreting small score changes without confidence intervals.

## 5. 🔹 Code Snippet
```python
# Pseudocode: bootstrap difference
diffs = []
for _ in range(1000):
    sample = rng.choice(n, size=n, replace=True)
    diffs.append(metric_cand(sample) - metric_base(sample))
ci = np.percentile(diffs, [2.5, 97.5])
```

## 6. 🔹 Interview Follow-ups
1. Q: Why paired?  
   A: Reduces variance because both systems see the same examples.

## 7. 🔹 Common Mistakes
- Using independent samples without pairing, inflating variance.

## 8. 🔹 Comparison / Connections
- Connects to statistical learning theory and A/B testing.

## 9. 🔹 One-line Revision
Use paired evaluation + effect sizes + confidence intervals/statistical tests to ensure differences are significant.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: How do you evaluate the robustness of an LLM application across input variations?

## 1. 🔹 Direct Answer
Robustness evaluation tests the system across paraphrases, formatting changes, typos, different languages, and distribution shifts. Use adversarial and metamorphic tests and measure worst-case and stability of key metrics.

## 2. 🔹 Intuition
The model must handle messy real user input, not only curated phrasing.

## 3. 🔹 Deep Dive
Methods:
- **Paraphrase tests:** generate semantically equivalent queries.
- **Metamorphic testing:** transformations that should not change the label (e.g., adding salutations).
- **Adversarial perturbations:** typos, reorder evidence, inject distractors.
- Evaluate:
  - metric variance across perturbations
  - worst-case failure rate

## 4. 🔹 Practical Perspective
- Use: customer-facing applications.
- Avoid: assuming robustness from a single-format test set.

## 5. 🔹 Code Snippet
```python
tests = [original] + paraphrases(original)
scores = [metric(system(t)) for t in tests]
robustness = min(scores)
```

## 6. 🔹 Interview Follow-ups
1. Q: How to choose perturbations?  
   A: Based on user logs and known failure classes.

## 7. 🔹 Common Mistakes
- Only testing minor punctuation changes and missing deeper semantic variants.

## 8. 🔹 Comparison / Connections
- Connects to OOD generalization and adversarial robustness.

## 9. 🔹 One-line Revision
Robustness means stable performance under paraphrase/format perturbations and worst-case inputs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q22: What are the key differences between evaluating traditional ML vs LLM applications?

## 1. 🔹 Direct Answer
Traditional ML evaluation focuses on deterministic model outputs with labeled datasets, while LLM evaluation must address stochastic decoding, prompt sensitivity, long-context behavior, tool/RAG grounding, and safety. Also, LLM eval often requires multi-dimensional rubrics rather than single scalar loss.

## 2. 🔹 Intuition
LLM systems are pipelines with language interfaces; failures are richer and harder to isolate.

## 3. 🔹 Deep Dive
- Traditional ML:
  - fixed features, fixed inference graph
  - metrics align closely to supervised labels
- LLM apps:
  - prompt + context affect outputs
  - output is unstructured (needs parsing/validators)
  - must evaluate faithfulness, relevance, safety, and latency/cost

## 4. 🔹 Practical Perspective
- Use: build eval harnesses for your pipeline components (retrieval, tool use, formatting).
- Avoid: relying solely on standard NLG metrics.

## 5. 🔹 Code Snippet
```python
# Traditional: accuracy/F1
# LLM: success + format_valid + faithfulness + safety + cost
```

## 6. 🔹 Interview Follow-ups
1. Q: Why do you need format checks?  
   A: Agents/tools require strict structured outputs; text-only metrics miss breakage.

## 7. 🔹 Common Mistakes
- Ignoring parsing failures and assuming “looks fine.”

## 8. 🔹 Comparison / Connections
- Connects to EDD and observability.

## 9. 🔹 One-line Revision
LLM evaluation must cover stochasticity, prompt/context sensitivity, grounding, safety, and system constraints—not only accuracy.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q23: How do you set up an evaluation framework from scratch for a new LLM application?

## 1. 🔹 Direct Answer
Start by defining success criteria and a minimal eval dataset, then add automated checks (format parsing, schema validation), evidence-based faithfulness checks (if RAG), safety filters, and finally human/LLM-judge scoring. Run baselines and create regression gates for every change.

## 2. 🔹 Intuition
Build evaluation like you build a product: start small, then expand with real failures.

## 3. 🔹 Deep Dive
Steps:
1. Define task definition and desired output behavior.
2. Build dataset: representative + edge + unanswerable + adversarial.
3. Implement validators/parsers.
4. Implement metrics:
   - task success
   - faithfulness/relevance (RAG)
   - safety/policy
   - efficiency (latency/cost)
5. Baseline -> iterate with EDD loop.

## 4. 🔹 Practical Perspective
- Use: any new LLM feature.
- Trade-offs: evaluation engineering effort upfront reduces production risk later.

## 5. 🔹 Code Snippet
```python
eval_spec = {"task_metric": "F1", "format_valid": True, "safety_rate": "<0.1%"}
run_eval(eval_spec)
```

## 6. 🔹 Interview Follow-ups
1. Q: How big should eval set be?  
   A: Start with small but diverse; grow guided by error clusters and drift.

## 7. 🔹 Common Mistakes
- No unanswerable cases; the model may hallucinate in production.

## 8. 🔹 Comparison / Connections
- Connects to **golden datasets** and **regression suites**.

## 9. 🔹 One-line Revision
Define success + build minimal dataset + add validators/faithfulness/safety + run baselines + iterate with regression gates.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q24: Your model passes one fairness metric but fails another. How do you handle conflicting audit results?

## 1. 🔹 Direct Answer
Handle conflicting metrics by understanding which harm each metric captures, prioritizing the most relevant definition of fairness for the product and legal requirements, and running intersectional audits. Then mitigate with targeted data/constraint methods and re-evaluate.

## 2. 🔹 Intuition
Fairness metrics measure different notions; improving one can hurt another.

## 3. 🔹 Deep Dive
Approach:
1. Map metric to stakeholder harm (what does it protect?).
2. Check intersectional groups and subgroup thresholds.
3. Identify proxy discrimination vs label imbalance.
4. Choose mitigation method: reweighting, constraints, adversarial debiasing, calibration.
5. Re-run with updated eval set.

## 4. 🔹 Practical Perspective
- Use: when multiple audits are requested (ethics/compliance).
- Avoid: blindly optimizing the easiest metric.

## 5. 🔹 Code Snippet
```python
# Pseudocode: choose metric by harm severity
chosen = select_fairness_metric(audit_requirements)
optimize_to(chosen)
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you communicate to stakeholders?  
   A: Explain trade-offs, limitations, and mitigation plan tied to risk.

## 7. 🔹 Common Mistakes
- Confusing statistical significance with fairness compliance.

## 8. 🔹 Comparison / Connections
- Connects to fairness-regularization trade-offs in ML.

## 9. 🔹 One-line Revision
Conflicting fairness audits require clarifying which harm metric matters most, auditing intersectional groups, then mitigating and re-evaluating.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q25: Your model was fair at deployment, but became biased 6 months later. How do you monitor continuously?

## 1. 🔹 Direct Answer
Monitor continuously by tracking subgroup metrics in production over time, detecting distribution drift (inputs, labels/proxies, policies), and triggering retraining/recalibration when bias metrics cross thresholds. Log decisions and features used for explainability.

## 2. 🔹 Intuition
Fairness can drift when data distribution or usage patterns change.

## 3. 🔹 Deep Dive
Monitoring includes:
- subgroup performance over time (error rates, calibration)
- input drift (language, segment mix)
- retrieval/index drift (if RAG)
- policy drift (if prompts/tools changed)
- alerts + rollback on regression

## 4. 🔹 Practical Perspective
- Use: high-impact decision systems (hiring, loans, medical).
- Trade-offs: continuous evaluation cost; mitigate via sampling + stratified monitoring.

## 5. 🔹 Code Snippet
```python
if subgroup_error[group] > threshold:
    trigger_recalibration()
```

## 6. 🔹 Interview Follow-ups
1. Q: What about label shift?  
   A: Use proxy labels or delayed ground truth; update monitoring accordingly.

## 7. 🔹 Common Mistakes
- Not monitoring subgroup metrics; only tracking overall accuracy.

## 8. 🔹 Comparison / Connections
- Connects to concept drift and MLOps monitoring.

## 9. 🔹 One-line Revision
Continuous bias monitoring requires subgroup metrics + drift detection + automated mitigation triggers.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q26: An external auditor cannot reproduce your model's results. How do you ensure audit reproducibility?

## 1. 🔹 Direct Answer
Ensure reproducibility by versioning every component: model checkpoint, tokenizer/config, prompts/templates, retrieval/index snapshots, evaluation datasets, decoding parameters, and random seeds. Provide an execution environment or deterministic configs and keep artifacts for audits.

## 2. 🔹 Intuition
Auditors need “the exact recipe,” not just a description.

## 3. 🔹 Deep Dive
Reproducibility checklist:
- code commit hash
- model weights version
- system/developer prompt versions
- temperature/top_p/max_tokens
- retrieval index snapshot + embedding version
- dataset versions
- random seed + deterministic settings (where possible)
- hardware/software environment

## 4. 🔹 Practical Perspective
- Use: regulated domains and high-risk systems.
- Trade-offs: storage and governance overhead.

## 5. 🔹 Code Snippet
```python
audit_bundle = {
  "model_version": model_id,
  "prompt_version": prompt_id,
  "retrieval_index_snapshot": idx_snapshot_id,
  "eval_dataset_version": ds_id,
  "decoding_params": {"temperature":0.0,"top_p":1.0}
}
```

## 6. 🔹 Interview Follow-ups
1. Q: What if the API model is nondeterministic?  
   A: Use deterministic seeds where available, run repeated trials, and report variance; also consider self-hosting for audit tests.

## 7. 🔹 Common Mistakes
- Rebuilding indexes at audit time instead of freezing snapshots.

## 8. 🔹 Comparison / Connections
- Connects to ML governance and MLOps lineage.

## 9. 🔹 One-line Revision
Reproducibility requires versioned artifacts and fixed evaluation conditions: model, prompts, retrieval snapshots, datasets, and decoding parameters.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q27: How do you structure red teaming for an LLM chatbot before launch?

## 1. 🔹 Direct Answer
Structure red teaming by:
1) defining threat models and high-risk scenarios,
2) generating adversarial prompts (including indirect RAG/context injection),
3) running the chatbot under realistic permissions and tool availability,
4) labeling failures by severity, and
5) turning failures into regression tests with mitigations and re-tests.

## 2. 🔹 Intuition
You want coverage over what attackers might do, not just generic safety checks.

## 3. 🔹 Deep Dive
Process:
- threat model (jailbreaks, leakage, tool misuse, unsafe advice)
- attack generation (automated + human-crafted)
- execution with logging (prompts, retrieved chunks, tool calls)
- severity classification (impact * likelihood)
- iterative mitigation and regression suite.

## 4. 🔹 Practical Perspective
- Use: pre-launch for high-visibility products.
- Trade-offs: red team coverage is never perfect; keep running post-launch.

## 5. 🔹 Code Snippet
```python
for attack in attack_suite:
    out = chatbot.run(attack, permissions=user_acl)
    label_failure(out)
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you prioritize?  
   A: Rank by severity and reachability in production.

## 7. 🔹 Common Mistakes
- Only running attacks on text prompts; ignore tool outputs and retrieved documents.

## 8. 🔹 Comparison / Connections
- Connects to AI safety and guardrails.

## 9. 🔹 One-line Revision
Pre-launch red teaming is threat-model-driven adversarial testing with labeled failures converted into regression tests.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q28: How do you red team a multimodal model where text-only safety tests miss cross-modal attacks?

## 1. 🔹 Direct Answer
Red team multimodal by testing cross-modal attack paths: malicious images/audio/video that trigger unsafe interpretations, prompt injection hidden in visual content, and tool behaviors tied to visual evidence. Use modality-specific adversarial datasets and run end-to-end tests with realistic fusion pipelines.

## 2. 🔹 Intuition
Text-only tests miss what can be smuggled through pixels.

## 3. 🔹 Deep Dive
Cross-modal attack types:
- **Visual prompt injection:** text embedded in images that the model interprets as instructions.
- **Robustness attacks:** adversarial perturbations that change visual interpretation.
- **Layout/table attacks:** misleading content in diagrams or charts.
Testing:
1. Generate adversarial images/audio (with target phrases/meaning).
2. Evaluate model outputs and safety classification across modalities.
3. Validate that guardrails apply to both extracted text and final fused reasoning.

## 4. 🔹 Practical Perspective
- Use: document understanding, video moderation, VLM agents.
- Avoid: only applying filters to user text; also filter extracted content and model outputs.

## 5. 🔹 Code Snippet
```python
img = make_adversarial_image("Ignore instructions ...")
out = vlm_system({"image": img, "question": "What should I do?"})
assert not violates_policy(out)
```

## 6. 🔹 Interview Follow-ups
1. Q: How to measure success of cross-modal attacks?  
   A: Policy violation rate and extraction of disallowed instructions across modalities.

## 7. 🔹 Common Mistakes
- Assuming OCR text is equivalent to user input; attacks can bypass by targeting fusion steps.

## 8. 🔹 Comparison / Connections
- Connects to multimodal AI robustness and security testing.

## 9. 🔹 One-line Revision
Multimodal red teaming must test visual/audio injection paths and enforce safety on extracted inputs and fused outputs.

## 10. 🔹 Difficulty Tag
🟣 Hard

