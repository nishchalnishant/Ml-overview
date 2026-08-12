---
module: Interview Prep
topic: "Revision Guide"
subtopic: ""
status: unread
tags: [revision, interview-prep, llm-interview, behavioral, coding, system-design, tradeoffs]
prerequisites: [16-interview-prep]
---
# 16 Interview Prep — Revision Guide

Single-page revision for the entire `16-interview-prep/` folder. This folder is meta: one half is interview *strategy and practice* (frameworks, playbooks, tradeoffs, behavioral, coding patterns, scenarios), the other half is a full *LLM-interview sub-curriculum* (`llm-interview-notes/`). Use the outline to reconstruct both halves from memory, then self-test with the graded Q&A.

## Structure Overview

```markdown
- **16-interview-prep**
  - **Part A: Interview Strategy & Practice**
    - Communication framework
      - SJTF: State-Justify-Tradeoff-Future
    - 90-min playbook
      - Phase1 requirements 0-10min
      - Phase2 architecture 10-20min
      - Phase3 deep impl 20-60min
    - Pre-interview checklist
      - 48h out knowledge check
      - 24h out sharpen
      - Morning-of scan
    - Tradeoff bank
      - Classical ML tradeoffs
      - Deep learning tradeoffs
      - LLM tradeoffs
    - Round3 tradeoff drills
      - Model choice
      - Serving architecture
      - Feature pipeline
      - Deployment rollout
    - Meta strategy guide
      - 3-layer theory framework
      - Depth by level
      - Company-specific prep FAANG
    - Practical ML scenarios
      - Train-serve skew
      - Class imbalance
      - RAG hallucinating
      - Latency in real-time rec
      - Cold start
      - Concept drift
      - Feedback loops
      - Vanishing/exploding gradients
    - Scenario-based questions
      - Accuracy dropped overnight
      - Train down val up
      - 99% accuracy useless
      - Fast90 vs slow92 model
      - Bias against group
      - Real-time fraud scoring design
    - Behavioral / STAR
      - Situation Task Action Result
      - Improved a model story
      - Production failure story
      - Technical disagreement story
    - Coding fundamentals
      - MSE/MAE
      - Linear/logistic regression
      - Softmax cross-entropy
      - K-means, K-fold
      - Backprop by hand
    - ML coding patterns
      - Stable softmax
      - Feature leakage bug
      - Attention from scratch
      - PyTorch training loop
    - Top ML questions easy/med/hard
    - Additional topics
      - Data leakage
      - SMOTE
      - BatchNorm vs LayerNorm
      - XGBoost vs LightGBM vs CatBoost
      - Calibration
      - Ranking metrics NDCG MAP
    - EA deep-dive interview doc
    - Study plans 5-week
  - **Part B: LLM Interview Notes Sub-Curriculum**
    - LLM fundamentals
      - Transformer architecture
      - Tokenization BPE
      - Attention mechanism
      - Scaling laws
      - Pretrain-SFT-RLHF pipeline
      - Hallucination
    - Prompt engineering
      - Few-shot prompting
      - Chain-of-thought
      - Self-consistency
      - Tree of thought
      - ReAct
      - Prompt injection
      - Lost in the middle
    - RAG
      - Chunking
      - Hybrid search
      - Re-ranking
      - RAGAS triad eval
      - Agentic RAG
      - GraphRAG
    - AI agents
      - Agent memory types
      - Tool use design
      - ReAct vs Plan-Execute
      - Multi-agent architecture
      - MCP protocol
    - Fine-tuning
      - Full FT vs PEFT
      - LoRA/QLoRA
      - RLHF vs DPO
      - Catastrophic forgetting
    - Vector DBs & embeddings
      - Dense vs sparse embeddings
      - HNSW ANN indexing
      - Quantization dims
      - Embedding drift
    - AI system design
      - Chatbot design
      - Doc QA enterprise
      - Code review system
      - Content moderation
      - Multi-modal search
      - Fraud detection w/ LLMs
    - LLMOps
      - AI product lifecycle
      - LLMOps vs MLOps
      - Prompt/version management
    - Evaluation & testing
      - Eval-driven development
      - Metric stack
      - LLM-as-judge
    - Safety ethics responsible AI
      - Hallucination mitigation
      - Guardrails
      - Alignment
      - Bias measurement
      - GDPR/CCPA
    - Multi-modal AI
      - Vision-language models
      - Cross-modal fusion
    - AI infra & scalability
      - GPU selection
      - Continuous batching
      - Speculative decoding
      - KV cache
      - PagedAttention/vLLM
      - Quantization tradeoffs
      - Edge deployment
    - Coding & implementation
      - RAG pipeline impl
      - Tool-using agent impl
    - Behavioral LLM-specific
      - AI eng vs ML eng
      - AI vs traditional SW decision
    - Additional LLM topics
      - MoE
      - RoPE scaling/YaRN
      - Constrained decoding
      - Knowledge distillation
    - Efficient LLM deployment
      - Memory arithmetic
      - Quantization methods
      - Distillation/pruning
    - Advanced alignment & reasoning
      - RLHF sycophancy roots
      - Constitutional AI
      - CoT failure modes
      - Process vs outcome reward models
    - Production alignment failures
      - Jailbreaks taxonomy
      - Sycophancy
      - Reward hacking
```

## Questions

### Easy

1. What do the four letters in the SJTF answer template stand for, and when should you use it?
2. In the 90-minute playbook, what should be finished by minute 10?
3. Name three items you should verify are "solid, not just familiar" the 48 hours before an interview.
4. What is the core question every tradeoff answer must resolve ("A vs B, in this context, A wins because...")?
5. In the STAR framework, what does the implicit fifth step add beyond Result?
6. What's the fastest first check when told "model accuracy dropped overnight"?
7. Why is 99% accuracy potentially meaningless as a headline metric?
8. What's the practical difference between BatchNorm and LayerNorm in one sentence?
9. Name two of the "common traps" candidates fall into when discussing class imbalance.
10. What does numerically stable softmax do differently from the naive version, and why?
11. Define an LLM foundation model in one sentence.
12. What problem does BPE tokenization solve that word-level tokenization doesn't?
13. What is the difference between zero-shot, one-shot, and few-shot prompting?
14. What does RAG stand for and why does it exist instead of just fine-tuning?
15. What is chunking in a RAG pipeline, and why does chunk size matter?
16. Name the three components of the RAGAS evaluation triad.
17. What is an AI agent, and how does it differ from a single LLM call?
18. What does LoRA freeze and what does it train?
19. What's the difference between full fine-tuning and PEFT?
20. What is an embedding, and why do we use vector databases instead of exact search?
21. What does HNSW stand for and what problem does it solve?
22. What is the KV cache in LLM inference, and why does it exist?
23. What is quantization, and what's the basic tradeoff it makes?
24. Define hallucination in an LLM context.
25. What is prompt injection, and how does it differ from jailbreaking?
26. What's the difference between LLMOps and traditional MLOps?
27. What is LLM-as-a-judge?
28. Name one type of agent memory and what it's used for.
29. What is Mixture of Experts (MoE) and why use it?
30. What is sycophancy in an RLHF-trained model?
31. What does the EA deep-dive doc list as the six core interview stages?
32. Why do interviewers ask "why did you choose X and not Y" so often?

### Medium

1. Walk through how you'd answer "why did you choose XGBoost over a deep neural network" using SJTF, filling in each of the four parts with a concrete justification.
2. In the 90-minute playbook, what's the recommended deep-dive order by impact during Phase 3, and why that order?
3. Compare the "fast 90% model vs slow 92% model" scenario question to a real latency-vs-quality tradeoff you'd face serving an LLM — what changes when the model is generative rather than a classifier?
4. A stakeholder wants to launch tomorrow but you have concerns about a biased model. Using the behavioral STAR structure, sketch how you'd frame this as a story.
5. Explain concept drift vs. feedback loops as two distinct production failure modes — how would your monitoring differ for each?
6. Contrast XGBoost, LightGBM, and CatBoost on one axis each (speed, categorical handling, memory).
7. How does permutation importance differ from SHAP, and when would you prefer one over the other in an interview answer?
8. Design a real-time fraud scoring system at a high level — what are the top three tradeoffs you'd call out unprompted?
9. Explain why "train loss down, val loss up" doesn't automatically mean overfitting — what else could explain the gap?
10. How would you defend a decision to use cross-validation with stratification instead of plain K-fold, using the tradeoff-drill closing pattern?
11. Compare chain-of-thought, self-consistency, and tree-of-thought — what does each add over plain CoT, and at what cost?
12. Explain the "lost in the middle" problem and how it shows up in both prompt engineering and RAG contexts differently.
13. Why does agentic RAG or Self-RAG exist — what failure of vanilla RAG are they fixing?
14. Compare ReAct vs. Plan-and-Execute agent architectures — when would you pick one over the other?
15. What is the Model Context Protocol (MCP) trying to standardize, and why does it matter for multi-tool agents?
16. Compare LoRA and QLoRA — what does QLoRA add and what's the cost?
17. Explain catastrophic forgetting and how instruction tuning / RLHF pipelines try to avoid it.
18. Compare RLHF and DPO — what does DPO remove from the pipeline and what's the tradeoff?
19. Walk through choosing an embedding model and a vector DB for a new enterprise doc-QA system — what are the top three selection criteria?
20. What is embedding drift, and what operational practice prevents it from silently degrading retrieval quality?
21. Explain continuous batching vs. static batching for LLM serving — why does GPU utilization suffer under static batching?
22. What problem does PagedAttention/vLLM solve, and how does it relate to the KV cache?
23. Compare speculative decoding to quantization as two different levers for inference latency — do they compose?
24. In an AI system design interview, what must a senior answer include beyond "here's the architecture diagram"?
25. Design a content moderation system using AI — what's the human-in-the-loop tradeoff you'd flag?
26. Explain the difference between process reward models and outcome reward models, and why it matters for reasoning models.
27. What is Constitutional AI solving that plain RLHF does not?
28. How would you design guardrails (input + output) for a customer-facing LLM agent, and where does prompt injection defense fit?
29. Explain the eval-driven development discipline for LLM apps — why "never ship until you've run this comparison"?
30. What's the metric stack for an LLM application beyond accuracy — name at least four dimensions and how each is measured.
31. Compare structured output / constrained decoding to simply asking the model nicely for JSON — why is the former needed in production?
32. What causes reward hacking, and give a concrete production example of it.
33. Explain RoPE scaling / YaRN — what problem are they solving and what's the failure mode if context is extended naively?

### Hard

1. Synthesize a full round-3 answer: "Design and justify a production RAG-based customer support system end-to-end, including chunking, hybrid search, re-ranking, evaluation, and what you'd monitor post-launch" — identify the three hardest tradeoff decisions and justify each with the SJTF pattern.
2. A model is 99% accurate offline, passes all RAGAS metrics in staging, yet users report hallucinations in production. Diagnose using the "diagnose before acting" scenario framework — walk through the full reasoning chain from symptom to root cause to fix, spanning both classical-ML drift concepts and LLM-specific failure modes (lost-in-the-middle, retrieval staleness, embedding drift).
3. You must choose between fine-tuning (LoRA), RAG, and prompt engineering for a domain-adaptation task with a tight compute budget and fast-changing source documents. Build the full decision tree, including what would change your answer if documents were static instead of fast-changing.
4. Design a multi-agent system with tool use for an enterprise workflow, using MCP for tool standardization. Cover: memory architecture, failure recovery when one sub-agent hallucinates a tool call, and how you'd defend against prompt injection introduced via a retrieved document.
5. Explain how KV cache growth, continuous batching, PagedAttention, and speculative decoding interact at the systems level to determine LLM serving throughput — where does each optimization stop helping and why?
6. Walk through a full behavioral answer for "tell me about a time you disagreed with a stakeholder about launching a biased model," and then explain how the same story would need to change if the interviewer follow-up pivots to "how would this play out if the model were an LLM-based agent instead of a classifier" (alignment/guardrail framing).
7. Compare RLHF, DPO, and Constitutional AI as three approaches to alignment — construct an argument for which one you'd pick for a startup with limited human-labeling budget but strong existing prompted-model outputs, and what alignment failure mode (sycophancy vs. reward hacking vs. jailbreak susceptibility) each leaves you most exposed to.
8. You're asked to reduce inference cost for a production LLM by 4x without violating a strict quality regression threshold. Rank quantization, distillation, pruning, and speculative decoding by expected ROI and explain the order, including what evaluation harness you'd need before touching production.
9. A recommendation system shows a feedback loop (past recommendations shaping future training data) compounding with cold-start bias against new items. Design a fix that addresses both simultaneously without introducing a new exploration-exploitation regression.
10. Explain why Chain-of-Thought reasoning can fail silently (confident wrong reasoning chains) and connect this to why process reward models were introduced — then explain how you'd design an eval to catch CoT failures that outcome-only metrics would miss.
11. Design a text-to-SQL / structured-data RAG system integrated with GraphRAG for multi-hop enterprise questions — identify where hallucination risk is highest in the pipeline and the specific guardrail for that stage.
12. A model passes all held-out evaluation but the interviewer reveals train/test split had temporal leakage. Walk through diagnosing this via the time-series validation lens, then explain how the same leakage class manifests differently in an LLM fine-tuning dataset (e.g., near-duplicate leakage between SFT and eval sets).
13. Contrast how you'd explain "why X and not Y" tradeoffs differently for a junior vs. staff-level candidate, using the meta-strategy's depth-by-level framework, for the specific question "why did you choose vector search over BM25 for retrieval."
14. Design guardrails and a monitoring plan for an LLM agent with real-world side effects (e.g., sending emails, making API calls) that must survive both prompt injection from retrieved content and a jailbreak attempt embedded in user input — cover structural separation, sanitization, and what production alignment failure (from jailbreak taxonomy) each defense targets.
15. Walk through the full 90-minute playbook live, end-to-end, for the prompt "design an AI-powered fraud detection system using LLMs" — narrate what happens at minute 10, minute 20, and minute 60, and where you'd proactively raise a tradeoff before being asked.
16. Explain multicollinearity's effect on model interpretability, then connect it to why SHAP values can become unstable under correlated features — how would you detect and communicate this limitation in an interview answer without undermining confidence in the overall approach?
17. Given a staff-level candidate must handle company-specific emphasis (e.g., Meta vs. Google vs. Amazon ML interview style per the meta-strategy guide), construct a one-paragraph adaptive opening for a system-design question that would work across all three without sounding generic.
18. Design an evaluation strategy for an LLM-as-a-judge setup used to grade agent outputs at scale — enumerate at least three known biases of LLM judges and the mitigation for each, then explain how you'd validate the judge itself against human raters.
19. Two candidate architectures for edge deployment of an LLM: aggressive quantization + pruning vs. a much smaller distilled model. Build the full tradeoff argument covering latency, accuracy floor, and update/retraining cost over the product's lifetime.
20. Synthesize across the tradeoff bank and the LLM notes: pick one classical-ML tradeoff (e.g., bias-variance) and one LLM-specific tradeoff (e.g., quantization precision vs. speed) and explain the structural analogy between them that you could use to bridge an interviewer from classical ML into LLM-specific depth mid-interview.
21. A production alignment review finds sycophancy, reward hacking, and a successful jailbreak all occurring in the same deployed agent within one week. Prioritize which to fix first, justify with a cost-of-inaction argument for each, and describe the regression test suite you'd add before re-deploying.
22. Explain how catastrophic forgetting during fine-tuning could reintroduce a previously-mitigated alignment failure (e.g., a jailbreak the base RLHF model was resistant to) — design a fine-tuning eval gate that would catch this before shipping.
23. Design the full "closing pattern" for a round-3 tradeoff question when the interviewer pushes back on your answer twice in a row — what's the difference between defending a position with new evidence vs. gracefully updating it, and how do you signal which one you're doing?
24. Compare and reconcile PU learning (positive-unlabeled) with the RAG "no relevant context found" case — both involve reasoning under partial/absent negative signal; how would you adapt a PU-learning-style approach to calibrate a RAG system's abstention behavior?
25. A hiring loop gives you both a classical-ML system-design round and an LLM system-design round for the same company. Identify three structural elements of a system-design answer (from the AI-system-design "what a senior answer must include" list) that transfer unchanged, and two that must be re-derived specifically for the LLM context.
