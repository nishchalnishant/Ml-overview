---
module: Llms
topic: Interview Notes
subtopic: Prompt Engineering
status: unread
tags: [llms, ml, interview-notes-prompt-enginee]
---
# Prompt Engineering — Interview Notes

---

## What Prompt Engineering Is and Why It Exists

**The problem**: an LLM is a function P(output | input). The same model produces radically different outputs for "What is the sentiment of this text?" versus "Classify the following review as POSITIVE or NEGATIVE. Return only one word." Both are asking the same thing. The first produces a paragraph; the second produces a parseable label. Production systems need the second.

**The core insight**: LLMs learned from text that had patterns — instructions were followed in certain ways, documents had certain structures, code had certain forms. By constructing your input to match the statistical patterns associated with the behavior you want, you shift the output distribution toward useful results. Prompt engineering is exploiting what the model already learned about how language works.

**The mechanics**: the LLM approximates P(output | input). Changing the input changes the distribution over outputs. Concretely, prompt design operates on three levers:
- **Instruction design**: role + task + constraints narrows what the model treats as in-scope
- **Context design**: what facts are present, where they appear, how they are delimited
- **Output design**: format schema, stop conditions, refusal behavior

Adding a schema ("Return JSON with keys: answer, citations, confidence") shifts the model toward outputs it associates with structured responses during training.

**What breaks**: prompts are brittle. Small wording changes can cause regressions. The model is not parsing your prompt logically — it is generating text that would plausibly follow your prompt. If two parts of the prompt conflict, the model resolves the conflict via statistical patterns, not logical priority. Long prompts degrade instruction-following for instructions buried in the middle.

**What the interviewer is testing**: whether you understand that prompt engineering works through distributional conditioning, not programming logic — and that this brittleness is a structural property, not a fixable bug.

**Common traps**: treating prompts as deterministic programs (they are statistical); assuming more instructions always help (too many constraints can conflict); not testing prompt changes with an eval set (anecdotal testing misses regressions).

---

## System Prompts

**The problem**: you need to enforce consistent behavior — output format, persona, safety rules — across every turn of a conversation, without repeating the specification in every user message. User messages cannot be trusted to carry these constraints because they come from external inputs.

**The core insight**: the system prompt is processed before any user content and its conditioning persists across the conversation. Instructions placed there are not "more trusted" in any technical cryptographic sense, but they appear first, condition the model's initial state, and most instruction-tuned models were trained to treat system instructions as high-priority context.

**The mechanics**: place in the system message: role definition, output format constraints, refusal policies, grounding requirements ("answer only from the provided context"). Place in user messages: dynamic inputs, retrieved evidence, per-turn context. Business logic and security rules should never live only in the system prompt — they must also be enforced in application code, because prompt injection can override prompt-level constraints.

**What breaks**: the system prompt is not cryptographically protected. A sufficiently adversarial user can, through prompt injection (see below), cause the model to override system instructions. Secrets embedded in system prompts (API keys, proprietary logic) can be leaked by asking the model to repeat its instructions. System prompts that are too long push useful content into the "lost in the middle" zone of the context window.

**What the interviewer is testing**: whether you know the security limitations of system prompts — not just that they exist and condition behavior.

**Common traps**: embedding security-critical business logic only in the system prompt without code-side enforcement; putting secrets in the system prompt and expecting the model to keep them confidential; thinking "don't reveal your system prompt" in the system prompt prevents disclosure.

---

## Zero-Shot, One-Shot, and Few-Shot Prompting

**The problem**: a model may have the capability to perform a task (it learned the underlying skill during pretraining) but produces outputs in the wrong format, wrong granularity, or with the wrong framing. You need to show it the pattern you want without fine-tuning.

**The core insight**: LLMs are trained on text that included examples, demonstrations, and worked problems. Including k demonstrations in the prompt shifts the model toward completing the pattern established by those examples — this is in-context learning. The model is not updating its weights; it is pattern-matching against your demonstrations within the context window.

**The mechanics**:

- **Zero-shot**: no examples — only task description. Works when the task maps cleanly to a well-represented pattern in the training distribution.
- **One-shot**: one (input, output) demonstration pair before the test input.
- **Few-shot**: k demonstrations (typically k=3–8). Useful when zero-shot produces wrong format or framing.

Demonstration selection matters more than count: diverse, representative examples from close to your target domain consistently outperform more examples that are generic or from a different distribution. Example order matters: the model gives more weight to later examples. Inconsistent formatting across examples trains the model to be inconsistent.

**What breaks**: few-shot tokens are expensive. Each demonstration adds to input length, increasing latency and cost. Including contradictory examples (two demonstrations with different output formats) confuses rather than constrains. For classification with well-defined labels, few-shot often adds less benefit than a clear label definition in zero-shot.

**What the interviewer is testing**: whether you understand in-context learning as pattern completion, not instruction programming — and can reason about when adding examples helps vs. hurts.

**Common traps**: adding as many examples as possible without testing whether they help (more shots ≠ better performance); using examples that are too similar to each other (low diversity wastes context); not verifying that example formatting exactly matches the format you expect from the model.

---

## Chain-of-Thought (CoT) Prompting

**The problem**: for multi-step reasoning tasks (math word problems, logical deductions, multi-hop questions), asking for a direct answer fails at rates that seem unrelated to model capability. The model knows the sub-skills but produces wrong answers because it commits to an output direction before working through the problem.

**The core insight**: the model generates tokens sequentially and each token conditions subsequent tokens. If you force the model to generate intermediate reasoning steps before the final answer, those intermediate tokens become context for the final answer — the "working" constrains what the answer can be. Writing out steps increases the probability that the final answer tokens follow from correct computation.

**The mechanics**: add "Let's think step by step" or provide explicit reasoning demonstrations before the expected answer. For few-shot CoT, demonstrate the full reasoning chain (not just the answer) in each example. Add a structured output constraint at the end ("Final Answer: ...") to prevent the model from continuing to reason after reaching the answer.

Zero-shot CoT ("Let's think step by step") generalizes surprisingly well, though few-shot CoT with explicit worked examples performs better on complex domains. CoT is most effective for tasks where correct intermediate steps strongly constrain the final answer — math, logic, multi-hop factual reasoning. It adds little benefit for knowledge retrieval or simple classification.

**What breaks**: CoT increases output length, which increases cost and latency. It can make errors more elaborate — the model produces a confident, detailed wrong reasoning chain and commits to a wrong answer. CoT does not fix factual hallucination (the reasoning steps themselves can be invented). For tasks where intermediate text is confidential or can be exploited, exposing reasoning is a security concern.

**What the interviewer is testing**: whether you understand *why* CoT works (intermediate tokens condition subsequent tokens) rather than just that it works.

**Common traps**: applying CoT to simple tasks where it adds no benefit and increases cost; not adding a final answer format constraint (model may continue generating past the answer); expecting CoT to fix factual errors (it doesn't — it only constrains the reasoning path toward a conclusion).

---

## Self-Consistency

**The problem**: CoT improves average accuracy on reasoning tasks, but a single chain of reasoning is still a single sample from a stochastic distribution. For high-stakes questions, that single sample might be wrong even when the correct answer is statistically likely.

**The core insight**: if the model is likely to reach the correct answer, most independent samples with diverse reasoning paths should converge on the same final answer. Incorrect reasoning paths are more likely to diverge. Majority vote over independent samples amplifies the signal over the noise.

**The mechanics**: sample N completions with temperature > 0 (typically N=10–40, temperature=0.5–0.8). Extract the final answer from each (not the full reasoning). Aggregate by majority vote. The answer that appears most frequently is the most consistent and typically most accurate.

Self-consistency is an inference-time ensemble: N forward passes through the same model, not N different models. The cost is N× generation cost. The benefit is most pronounced when single-sample accuracy is in the 50–80% range — below 50%, even majority vote won't fix the underlying capability gap.

**What breaks**: self-consistency requires extracting a canonical final answer from each sample. If the output format is inconsistent, answers may not be directly comparable. Majority voting over free-form text (rather than structured answers) is unreliable. N× cost is prohibitive for high-latency, high-volume production applications.

**What the interviewer is testing**: whether you understand self-consistency as variance reduction via sampling, not as a fundamentally different reasoning strategy.

**Common traps**: voting over raw reasoning text rather than extracted final answers; using self-consistency for tasks where the answer is free-form (it is designed for tasks with a discrete correct answer); not knowing that this is inference-time ensembling, not ensemble of different models.

---

## Tree of Thought (ToT)

**The problem**: chain-of-thought is a single linear reasoning path. For tasks requiring search — planning, puzzles, code generation with constraints — a wrong intermediate step commits the model to a dead-end reasoning chain. There is no mechanism to backtrack.

**The core insight**: treat reasoning as a search problem. At each step, generate multiple candidate partial thoughts (branching). Evaluate each candidate. Keep only the promising branches. This is beam search applied to the reasoning process itself.

**The mechanics**:

1. Generate k candidate partial thoughts at step 1 (branching factor k).
2. Score each candidate using a separate evaluation prompt, a verifier model, or domain-specific heuristics.
3. Expand the top-scoring candidates to depth 2, generating another k thoughts each.
4. Prune low-scoring branches (beam search logic).
5. Continue to depth d. Select the final path with the best accumulated score.

Cost scales approximately as O(k^d) evaluations before pruning. In practice, k=3–5 and d=2–4 are typical. Each node is a separate LLM call — ToT is expensive and reserved for tasks where single-path CoT demonstrably fails.

**What breaks**: you need a reliable evaluator to score intermediate states. If the scoring function is wrong, ToT amplifies wrong decisions (confident pruning of good branches). Without a principled scoring function, ToT degrades to random branching. The O(k^d) cost makes it impractical for production latency requirements without heavy pruning.

**What the interviewer is testing**: whether you can map ToT to a search algorithm and reason about when it is and is not worth the cost.

**Common traps**: proposing ToT as a default improvement over CoT (it is much more expensive and only helps when reasoning is genuinely multi-path); not being able to explain what the scoring function should be for a given task; saying ToT "fixes hallucination" (it doesn't — it generates multiple hallucinatory paths and picks the most plausible-sounding one without external grounding).

---

## ReAct (Reasoning + Acting)

**The problem**: LLMs cannot access real-time information, execute code, or retrieve current facts without external tools. For agentic tasks (research, code execution, database queries), the model needs to interleave reasoning with actions — planning what to do, executing it, and updating based on the result.

**The core insight**: structure the model's output to alternate between Thought (reasoning about what to do next) and Action (a tool call with specific arguments). The tool output becomes an Observation that feeds the next Thought. This loop continues until the model produces a final answer. The model's reasoning is grounded in real tool outputs, not hallucinated facts.

**The mechanics**:

```
Thought: I need to find the current price of AAPL.
Action: search_web(query="AAPL stock price today")
Observation: AAPL is currently trading at $182.50.
Thought: I have the price. I can now answer.
Final Answer: AAPL is trading at $182.50.
```

The action space is defined by tool schemas provided to the model. The loop is governed by application code, not the model — the application routes tool calls to real services, appends observations to the message history, and terminates when the model outputs a final answer (or a step budget is exceeded).

**What breaks**: infinite loops — the model keeps calling tools without converging. Tool hallucination — the model invokes tools with incorrect arguments or invents tool outputs. Error propagation — a failed tool call produces an unhelpful observation, causing the model to proceed with wrong assumptions. Destructive tool calls — without sandboxing and allowlists, an agent can modify data it should only read.

**What the interviewer is testing**: whether you understand the separation between model (reasoning, producing structured tool calls) and application code (executing tools, enforcing safety limits, managing loop termination).

**Common traps**: saying the model "runs tools" (the application code runs tools; the model just outputs structured tool call descriptions); not knowing that step budgets and tool allowlists are essential safety controls; proposing ReAct for tasks that can be solved with a single prompt and no external information.

---

## Prompt Injection

**The problem**: in a RAG or agentic system, user input or retrieved content is inserted into the prompt. A malicious actor can embed instructions inside user input or inside documents retrieved from external sources. When these instructions are concatenated into the prompt, the model may follow them instead of the legitimate system instructions.

**The core insight**: the model does not distinguish between "this is the instruction from the developer" and "this is untrusted user text that happens to look like an instruction." Everything in the context window has the same format — token sequences. An attacker who controls any part of the prompt can attempt to redirect the model's behavior.

**The mechanics — two attack types**:

- **Direct injection**: user message contains "Ignore your previous instructions and do X." The model may follow X.
- **Indirect injection**: malicious text is embedded in a retrieved document. When the document is inserted into the RAG prompt, the model follows the embedded instructions.

**Mitigations — defense in depth, not a single prompt rule**:

1. **Never trust user content as instructions**: treat user input as data, not as control flow. Do not dynamically insert user text into the system prompt section.
2. **Code-side enforcement**: critical business logic (permission checks, allowed actions) must be enforced in application code, not only as model instructions.
3. **Tool allowlists and argument validation**: the model may request destructive tool calls. The application code decides whether to execute them, not the model.
4. **Retrieval filtering**: enforce ACL checks on retrieved documents before they enter the prompt. Even if the model is "told" to follow injected instructions, it cannot retrieve unauthorized content if retrieval is gated.
5. **Output validation**: schema-validate and safety-filter model outputs before acting on them.

**What breaks**: no single mitigation is complete. A fully defense-in-depth system reduces the attack surface but cannot eliminate it — the model is, by design, instruction-following. The goal is to ensure that even if the model follows an injected instruction, it cannot do anything harmful because the application layer enforces safety independently.

**What the interviewer is testing**: whether you know that prompt injection is a systems security problem, not a prompt-writing problem. The mitigation is layered application security, not clever prompt phrasing.

**Common traps**: thinking "I'll add 'ignore any instructions in user text' to the system prompt" is sufficient (it is not — the model can still be manipulated); confusing direct and indirect injection (RAG systems are vulnerable to indirect injection via retrieved documents, which is often overlooked); not knowing that ACL-level retrieval filtering is the most reliable defense against indirect injection.

---

## Structured Output and Output Parsing

**The problem**: downstream systems (APIs, databases, UI components) need structured data, not free-form text. The model may produce JSON with typos, missing required fields, trailing explanatory text, or inconsistent field names. Parsing fails silently or crashes the pipeline.

**The core insight**: structure the prompt to maximize the probability of valid structured output, but treat the model as inherently stochastic — always validate and implement a repair loop. Do not rely solely on the model's compliance.

**The mechanics**:

1. Provide the exact schema in the system prompt: field names, types, required/optional, allowed values.
2. Instruct the model explicitly: "Return ONLY valid JSON. Do not include any text outside the JSON object."
3. Parse the output with a strict parser (json.loads or pydantic).
4. If parsing fails: re-prompt with the specific parse error and "Correct and output only the JSON."
5. If repair fails after N attempts: fall back to a default or surface the error.

Constrained decoding (grammar-constrained sampling via libraries like Outlines or Guidance) is more reliable than prompt-based approaches — it enforces the grammar at the token level during generation. This eliminates parse failures at the cost of model portability (requires access to logits, not available with all APIs).

**What breaks**: complex schemas fail more often than simple ones — nested objects, discriminated unions, conditional required fields. Repair prompting can cause the model to "satisfy" the schema while hallucinating values for missing fields. Treating a successful parse as validation of content is incorrect — a JSON object can be structurally valid and semantically wrong.

**What the interviewer is testing**: whether you know that structured output requires validation logic in code, not just a "return JSON" instruction.

**Common traps**: assuming a parse error means the model misunderstood (it may just mean stochastic output variance); not implementing the repair loop; treating schema validation as content validation (the model can produce a valid schema filled with invented values).

---

## Lost in the Middle

**The problem**: when you provide a long context with many retrieved chunks, the model does not give equal attention to all of them. Empirical research shows that information at the beginning and end of the context is recalled significantly more reliably than information in the middle. For a 20-document context, accuracy for middle-document content drops from ~90% to ~40–50%.

**The core insight**: attention is not uniform over position. The model was trained on text where the most important context tends to appear near the query (at the end) or near the beginning (introduction/setup). This positional bias is baked into the model's learned attention patterns. Filling the middle with irrelevant content actively hurts retrieval of relevant content.

**The mechanics — mitigations**:

1. **Re-rank and reduce**: use a cross-encoder re-ranker to score all retrieved chunks, then pass only the top 3–5 to the LLM. Fewer high-quality chunks consistently beats more lower-quality chunks.
2. **Relevance ordering**: place the most relevant chunk first (or last), not in the middle.
3. **Hierarchical retrieval**: use parent-child chunking — retrieve at the chunk level for precision, but return the parent context for the LLM to reduce the number of separate context segments.
4. **Explicit labeling**: number and label chunks clearly, and include "Use only the provided evidence" instructions to anchor the model.

**What breaks**: aggressively reducing top-k can hurt recall — if the relevant chunk is ranked 6th and you pass only 5, you have excluded the answer. The fix is a better re-ranker, not a higher top-k.

**What the interviewer is testing**: whether you know that "more context = more information for the model" is empirically false above a threshold — and that context curation is a retrieval quality problem, not a prompt wording problem.

**Common traps**: using top-20 chunks "to be safe" (this reliably hurts quality compared to top-5 with re-ranking); thinking longer context windows eliminate the problem (lost-in-the-middle is a positional attention bias, not a capacity limit); not knowing that ordering matters (most relevant chunk should be first or last, not buried).

---

## Prompt Optimization and Evaluation

**The problem**: prompt iteration based on a handful of anecdotal examples produces prompts that work on those examples and fail on others. Without a systematic evaluation framework, every "improvement" might be a regression in disguise.

**The core insight**: treat prompts as versioned artifacts with measurable quality criteria. Every change is an experiment with a before and after measurement. Offline evaluation catches regressions fast; online evaluation validates real-user behavior.

**The mechanics — the evaluation loop**:

1. Create a gold dataset: representative queries + expected outputs or evaluation criteria (correct answer, required schema, no hallucination, etc.).
2. Run each prompt version on the full gold dataset; compute task-specific metrics.
3. Analyze error clusters — which inputs regress? What structural pattern explains the failure?
4. Modify the prompt based on the error analysis (not based on a single failing example).
5. Re-run evaluation. Require improvement on the error cluster without regression on previously passing cases.
6. A/B test or canary deploy changes in production; monitor real-user metrics.

Key metrics by task type:
- **Structured output**: schema validity rate, field accuracy, parse error rate
- **RAG Q&A**: faithfulness (answer grounded in context), answer relevance, citation accuracy
- **Classification**: accuracy, calibration, F1 by class
- **Safety**: refusal rate on adversarial inputs, policy compliance rate

**What breaks**: overfitting the eval set — prompts that perfectly satisfy the gold dataset may not generalize. The gold dataset must be representative and must include adversarial/edge-case examples. Evaluating prompts only in isolation misses interaction effects with the full pipeline (retrieval quality, output parsing).

**What the interviewer is testing**: whether you would use systematic evaluation to iterate on prompts, not just manual inspection.

**Common traps**: testing only on examples where the current prompt fails (this overfits to those cases); treating prompt quality as a subjective judgment (it must have objective metrics); not versioning prompts (impossible to reproduce a previous working state).

---

## Multi-Turn Conversation Management

**The problem**: LLMs have finite context windows. A long conversation eventually exceeds the window, causing the model to lose early context. Naively dropping early messages means losing information the user established early in the conversation. Keeping everything means the window fills with irrelevant history.

**The core insight**: the model needs selective memory — recent context in full fidelity, important older facts summarized or retrieved, irrelevant history discarded. This is not a model capability problem; it is an application-layer context management problem.

**The mechanics**:

- **Sliding window**: keep the last N turns verbatim. Simple. Loses early context that may still be relevant (e.g., user preferences stated at turn 1).
- **Summarization**: when history exceeds a token budget, summarize older turns into a compact summary appended to the system message. Loses detail; good for conversational context. Requires a separate summarization call.
- **Entity/state extraction**: parse conversation for structured facts (user preferences, stated constraints, confirmed decisions) and store as a structured memory block. Re-insert at each turn. Retains key facts without verbatim history.
- **RAG memory**: embed conversation turns and retrieve relevant past context based on semantic similarity to the current query. Complex infrastructure; useful for very long-lived sessions.

**What breaks**: summarization can drop constraints the user explicitly established. Sliding windows lose early-turn context the model has been implicitly relying on. Entity extraction requires the extraction model to be accurate — missed entities are silently lost.

**What the interviewer is testing**: whether you treat context management as an engineering problem with explicit design decisions, not a prompt problem.

**Common traps**: proposing "just use a longer context window" (cost and lost-in-the-middle effects make this not a universal fix); forgetting that the system prompt with format constraints must be re-injected every turn; not knowing that summarization requires a separate LLM call (or fine-tuned summarizer).

---

## Common Failure Modes and Debugging

**The problem**: a prompt-based system is failing. It might be producing wrong answers, wrong format, ignoring context, or behaving inconsistently across similar inputs. The failure could be in the prompt, the retrieval, the model selection, the output parsing, or the evaluation. Without a structured debugging approach, you spend hours changing things at random.

**The core insight**: the pipeline has distinct components that can fail independently. Isolate each component. The failure mode tells you where to look.

**The mechanics — failure mode diagnosis**:

| Observed failure | Primary suspect | First check |
|---|---|---|
| Format violations (wrong JSON structure) | Output instructions too weak | Add explicit schema; add repair loop |
| Answer ignores retrieved context | Context not grounded in prompt | Add "answer only from context" instruction; verify retrieval quality |
| Inconsistent answers across similar inputs | High temperature or vague instructions | Lower temperature; add explicit decision rules |
| Correct format but wrong facts | Retrieval failure or hallucination | Evaluate retrieval recall independently; add faithfulness check |
| Model follows injected instructions | Prompt injection | Enforce trust hierarchy; add retrieval filtering |
| Output drifts mid-conversation | Format constraints lost across turns | Re-inject format schema in system prompt each turn |

**Debug workflow**: (1) Reproduce with a fixed input. (2) Log system prompt, user message, retrieved context, and raw output. (3) Isolate: remove the retrieved context — does the model produce a better or worse answer? This tells you if retrieval is helping or hurting. (4) Change one thing at a time; re-run the eval set.

**What breaks the debugging process**: changing multiple prompt components simultaneously (cannot attribute the improvement or regression); debugging from a single example instead of an error cluster (overfits the fix to one case); not having a reproducible eval set to measure the effect of changes.

**What the interviewer is testing**: whether you would diagnose structured — identifying which component is failing — rather than randomly adjusting the prompt.

**Common traps**: adding more instructions when the problem is retrieval quality (the model cannot give a correct answer if the context doesn't contain the answer); increasing top-k retrieval chunks when the problem is lost-in-the-middle; testing prompt changes on examples you already know fail, not on the full eval distribution.

## Flashcards

**Instruction design?** #flashcard
role + task + constraints narrows what the model treats as in-scope

**Context design?** #flashcard
what facts are present, where they appear, how they are delimited

**Output design?** #flashcard
format schema, stop conditions, refusal behavior

**Zero-shot: no examples?** #flashcard
only task description. Works when the task maps cleanly to a well-represented pattern in the training distribution.

**One-shot?** #flashcard
one (input, output) demonstration pair before the test input.

**Few-shot?** #flashcard
k demonstrations (typically k=3–8). Useful when zero-shot produces wrong format or framing.

**Direct injection?** #flashcard
user message contains "Ignore your previous instructions and do X." The model may follow X.

**Indirect injection?** #flashcard
malicious text is embedded in a retrieved document. When the document is inserted into the RAG prompt, the model follows the embedded instructions.

**Structured output?** #flashcard
schema validity rate, field accuracy, parse error rate

**RAG Q&A?** #flashcard
faithfulness (answer grounded in context), answer relevance, citation accuracy

**Classification?** #flashcard
accuracy, calibration, F1 by class

**Safety?** #flashcard
refusal rate on adversarial inputs, policy compliance rate

**Sliding window?** #flashcard
keep the last N turns verbatim. Simple. Loses early context that may still be relevant (e.g., user preferences stated at turn 1).

**Summarization?** #flashcard
when history exceeds a token budget, summarize older turns into a compact summary appended to the system message. Loses detail; good for conversational context. Requires a separate summarization call.

**Entity/state extraction?** #flashcard
parse conversation for structured facts (user preferences, stated constraints, confirmed decisions) and store as a structured memory block. Re-insert at each turn. Retains key facts without verbatim history.

**RAG memory?** #flashcard
embed conversation turns and retrieve relevant past context based on semantic similarity to the current query. Complex infrastructure; useful for very long-lived sessions.
