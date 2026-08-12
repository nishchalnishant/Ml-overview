---
module: LLM Applications
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, llm-applications, agentic-workflows, rag, prompt-optimization, synthetic-data, hallucination, inference-optimization, speculative-decoding, model-merging, multimodal, mcp]
prerequisites: [11-llm-applications]
---
# 11 LLM Applications — Revision Guide

Single-page revision for the entire `11-llm-applications/` folder: agentic workflows, RAG, prompt optimization/versioning, synthetic data, hallucination mitigation, inference optimization, speculative decoding, model merging, multimodal, tuning/optimization decisions, and MCP. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **11-llm-applications**
  - **Agentic Workflows**
    - CoT
    - ReAct
    - Reflection/Self-Critique
    - Tree of Thought
    - Tool Use
      - Hallucinated calls
      - Wrong args
      - Cascade errors
      - Infinite loops
    - Memory Systems
      - In-context
      - Short-term external
      - Long-term semantic
      - Structured state
    - Multi-Agent Patterns
      - Orchestrator+subagents
      - Supervisor/worker
      - Debate/critic
    - Routing
    - Guardrails/Safety
    - Observability
    - Production Failure Modes
  - **RAG**
    - Document Ingestion/Chunking
    - Embedding+Retrieval
      - Dense
      - Sparse BM25
      - Hybrid RRF
    - Vector DB/ANN
      - HNSW
      - IVF/IVFPQ
    - Query Optimization
      - Multi-Query
      - HyDE
      - Step-Back
    - Reranking
      - Bi-encoder
      - Cross-encoder
    - Context Assembly
      - Lost-in-the-middle
    - Advanced Patterns
      - Self-RAG
      - CRAG
      - GraphRAG
    - RAGAS Evaluation
    - RAG vs Fine-tuning
  - **Prompt Optimization/Versioning**
    - Prompt Registry/Versioning
    - LLM-as-Judge
      - Position bias
      - Verbosity bias
      - Self-preference
    - APE
    - DSPy
      - BootstrapFewShot
      - MIPRO
      - COPRO
    - Dynamic Few-Shot
    - Eval Framework
    - A/B Testing/Rollback
  - **Synthetic Data**
    - Self-Instruct
    - Distillation
      - Phi story
    - Synthetic Reasoning/CoT
    - LIMA/Quality over Quantity
    - Code Data/Exec Verification
    - Math Data/Rejection Sampling
    - Model Collapse
    - Data Flywheel
  - **Hallucination Mitigation**
    - Why LLMs Hallucinate
    - Taxonomy
      - Intrinsic/Extrinsic
      - Faithful vs Factual
    - Root Causes
      - Teacher forcing
      - Rare facts
      - RLHF sycophancy
    - Detection
      - Self-Consistency
      - SelfCheckGPT
      - NLI Checking
      - FactScore
    - Mitigation Stack
      - RAG
      - Prompting
      - Constrained Decoding
      - Calibration/ECE
      - Factuality FT
      - RLHF/RLAIF
    - Benchmarks
      - TruthfulQA/FActScore/HaluEval/FELM
  - **Inference Optimization**
    - KV Cache
    - PagedAttention
    - FlashAttention
    - Quantization
      - W4A16
      - W8A8
      - GPTQ/AWQ/SmoothQuant
    - Batching
      - Static
      - Dynamic
      - Continuous
    - Serving Frameworks
      - vLLM
      - TensorRT-LLM
      - llama.cpp
    - Memory Wall
    - Debugging Prefill vs Decode
  - **Speculative Decoding**
    - Autoregressive Bottleneck
    - Standard Spec Decoding
      - Draft/Verify/Rejection Sample
      - Acceptance rate math
    - Medusa
    - Eagle/Eagle-2
    - Self-Speculative
    - Multi-Token Prediction
    - When it Helps/Hurts
  - **Model Merging**
    - Weight Averaging
    - SLERP
    - Task Arithmetic
    - TIES
    - DARE
    - Model Soup
    - Frankenmerging
    - MergeKit
    - Failure Conditions
  - **Multimodal**
    - ViT
    - CLIP
    - Fusion Architectures
      - Linear Projection/LLaVA
      - Q-Former/BLIP-2
      - Perceiver Resampler/Flamingo
      - Early Fusion
    - Native Multimodal LLMs
    - Audio: Whisper
    - Audio: Neural Codecs/RVQ
    - Video Models
      - Sparse Sampling
      - Factorized Attention
      - Spacetime Patches/Sora
    - Production Latency
  - **Tuning and Optimization**
    - Diagnose Problem First
    - Prompt vs RAG vs FT vs RLHF
    - Knowledge Distillation
    - Evaluating Fine-Tuned Models
    - Production Versioning
  - **MCP**
    - Host/Client/Server
    - Transport: stdio/HTTP-SSE
    - Tools
    - Resources
    - Prompts
    - Tool Calling Lifecycle
    - MCP vs Function Calling vs LangChain
    - Security
      - Input validation
      - Prompt injection via results
      - Permission boundaries
      - Sandboxing
    - Multi-Agent MCP Coordination
```

## Questions

### Easy

1. What problem does Chain-of-Thought prompting address, and what is its core mechanism?
2. What is the difference between zero-shot and few-shot CoT?
3. Define ReAct. How does it differ from plain CoT?
4. What is the purpose of a "Thought → Action → Observation" loop?
5. Why is self-critique more reliable for code than for factual claims?
6. What is Tree of Thought, and why does it cost more than linear reflection?
7. Name three tool-use failure modes.
8. What are the four types of agent memory and their typical backends?
9. Why does routing queries save cost in an agentic system?
10. Name three guardrail mechanisms used in production agent systems.
11. Why is full-trace observability important for debugging multi-step agents?
12. What is the core problem RAG solves?
13. What is the precision-context tradeoff in chunking?
14. What is the difference between dense and sparse (BM25) retrieval?
15. What does Reciprocal Rank Fusion combine, and why?
16. What is HNSW, and what problem does it solve?
17. Define HyDE.
18. Why is reranking a separate stage from initial retrieval?
19. What is "lost-in-the-middle"?
20. Name the four RAGAS metrics.
21. What is prompt versioning, and why treat prompts like code?
22. What are the three main LLM-as-judge biases?
23. What is APE (Automatic Prompt Engineer)?
24. What does DSPy optimize that manual prompting does not?
25. Why is synthetic data attractive compared to human labeling?
26. What is Self-Instruct?
27. What is knowledge distillation in the context of LLMs?
28. What did the LIMA paper demonstrate?
29. What is model collapse?
30. Define intrinsic vs. extrinsic hallucination.
31. What is the difference between faithful and factual hallucination?
32. What does the KV cache store, and why does it help?
33. What problem does PagedAttention solve?
34. What is FlashAttention's core trick for avoiding O(n²) memory traffic?
35. What is the difference between W4A16 and W8A8 quantization?
36. What is continuous batching, and how does it differ from static batching?
37. What is the autoregressive bottleneck that motivates speculative decoding?
38. In speculative decoding, what roles do the draft and target models play?
39. What is a task vector in model merging?
40. What is SLERP, and why is it preferred over linear interpolation in some cases?
41. What is a Vision Transformer (ViT) patch, conceptually?
42. What is CLIP trained to do?
43. What is the Model Context Protocol (MCP) meant to standardize?
44. What are the three primitives in MCP (tools, resources, prompts)?
45. What is the difference between an MCP host, client, and server?

### Medium

1. Why does an agent looping on a single tool call happen, and what is the correct fix (vs. the tempting-but-wrong fix)?
2. Why do multi-agent systems multiply failure rates rather than just adding capability?
3. Compare orchestrator+subagents vs. supervisor/worker vs. debate/critic multi-agent patterns — when would you pick each?
4. Why can router misclassification be more dangerous than an outright error message?
5. Why does poor chunking cause more RAG failures than a bad embedding model choice, in practice?
6. Walk through the tradeoffs between fixed-size, semantic, and structure-aware chunking strategies.
7. Why combine dense and sparse retrieval instead of relying on embeddings alone, and what specific query types does BM25 rescue?
8. Explain why HyDE can help retrieval on some queries and hurt it on others.
9. Why is a bi-encoder used for retrieval and a cross-encoder for reranking, rather than cross-encoders throughout?
10. What is the debugging order for a RAG system producing wrong answers, and why that order?
11. Compare RAG and fine-tuning for injecting new knowledge — give the decision rule.
12. Why does "answer ONLY from provided context" sometimes make a RAG product feel more broken, not less?
13. How does position-bias mitigation for LLM-as-judge work mechanically?
14. When does chain-of-thought prompting help vs. actively hurt, and why?
15. What is the mechanism by which DSPy improves a multi-hop pipeline that manual prompt tuning cannot?
16. How do you avoid overfitting a prompt to its own eval set?
17. Why does distillation transfer more information than training on hard labels, mechanistically (temperature, soft targets)?
18. Why does execution-based verification make synthetic code data more trustworthy than synthetic prose data?
19. How does rejection sampling turn math's verifiability into a data-quality tool?
20. Why does model collapse occur, and what is the single most important mitigation?
21. Explain why teacher forcing during training contributes to hallucination at inference (exposure bias).
22. Why does RLHF often worsen calibration even as it improves helpfulness?
23. How do self-consistency and SelfCheckGPT detect hallucination without ground truth, and what class of error do both miss?
24. What does constrained decoding fix and not fix regarding hallucination?
25. Derive intuitively why the expected number of accepted tokens per speculative-decoding step is (1 − α^(K+1)) / (1 − α).
26. Why does continuous batching require PagedAttention to be practical?
27. Explain why LLM inference at small batch size is memory-bandwidth-bound rather than compute-bound, and why that fact determines which optimizations help.
28. Compare vLLM, TensorRT-LLM, and llama.cpp — what workload profile fits each?
29. Why does speculative decoding help code generation far more than creative writing?
30. Compare Medusa and Eagle — what does Eagle add, and at what cost?
31. Why does linear weight averaging fail across different base checkpoints but work within fine-tunes of the same base?
32. Explain the sign-conflict problem in task-vector merging and how TIES resolves it.
33. Why does DARE's "dropout on deltas" reduce merge interference without destroying task signal?
34. Why can't CLIP's global embedding support counting or OCR-style tasks well?
35. Compare linear projection (LLaVA), Q-Former (BLIP-2), and Perceiver Resampler (Flamingo) bridges — what does each trade off?
36. Why is early fusion (Gemini-style) harder to adopt than late-fusion adapter approaches?
37. Why is full video self-attention intractable, and how do sparse sampling and factorized attention each partially solve it?
38. In MCP, why must tool descriptions be written for the model rather than for humans?
39. Why is the separation between "tools" (model-initiated) and "resources" (application-initiated) important in MCP design?
40. Why must MCP servers remain stateless, and where does conversation state actually live?
41. How would you decide, given a broken model behavior, whether to fix it with prompting, RAG, fine-tuning, or RLHF?
42. Why is catastrophic forgetting a risk after narrow SFT, and how do you mitigate it?
43. Why is LLM-as-judge unreliable for absolute scoring but useful for relative comparison?

### Hard

1. Design a RAG system for a 10M-document enterprise knowledge base: justify chunking, embedding, vector store, hybrid retrieval, reranking, and evaluation choices end to end.
2. You add hybrid retrieval, query expansion, and aggressive reranking to fix RAG failure modes — explain how these three fixes can interact adversarially against each other (e.g., reranking depth vs. lost-in-the-middle, grounding strictness vs. retrieval recall).
3. HyDE embeds a generated hypothetical answer. On which queries does this make retrieval strictly worse, and why is that failure especially hard to detect?
4. A team fine-tunes on internal docs to "teach the model company knowledge" instead of using RAG. Explain precisely why this usually fails, including what fine-tuning does absorb from that corpus.
5. Design a coding agent that can execute arbitrary generated code safely — cover sandboxing, permissions, and pre-execution guardrails, and explain why prompting the model to "not write dangerous code" is insufficient.
6. A multi-agent system (Researcher/Writer/Reviewer) produces low-quality output. Describe a systematic (not eyeballing-traces) debugging approach.
7. What happens to quantization-based inference speedups (e.g., GPTQ calibration) when the serving-time input distribution diverges from the calibration distribution used to fit the quantized weights?
8. Under what serving conditions does speculative decoding go from a throughput win to a net loss, even with a favorable acceptance rate α?
9. Why does α (acceptance rate) drift within a single deployment based on traffic mix, and why does that break naive capacity planning for speculative decoding?
10. Derive the break-even condition for speculative decoding given draft speedup β and K draft tokens, and explain what it implies about when speculative decoding is/isn't worth deploying.
11. Compare standard speculative decoding, Medusa, Eagle, self-speculative decoding, and multi-token prediction along the axis of "where is the cost paid" (training time vs. inference time vs. deployment complexity).
12. Continuous batching admits new requests mid-flight to keep GPUs saturated — what fairness/latency cost does this impose on requests already streaming, and how would a production scheduler mitigate it?
13. PagedAttention claims "near-zero fragmentation" — what new failure mode (preemption) does paging introduce that contiguous allocation did not have?
14. You silently downgrade a model's precision to INT4 to cut serving cost, and a downstream fraud-detection pipeline's recall drops 5%. Explain why "cost per quality unit," not raw cost, is the correct framework, and how you'd have caught this pre-deployment.
15. Design a synthetic data pipeline (self-instruct + distillation + rejection sampling) that minimizes model collapse risk across multiple training generations — specify what real-data fraction, filtering, and verification steps you'd enforce and why.
16. Why is generating synthetic chain-of-thought traces more effective for training than distilling only final answers, in terms of what generalizes?
17. Design a hallucination-mitigation strategy for a production system that cannot use RAG (e.g., fully offline/on-device). Justify each layer of the mitigation stack you choose and its cost.
18. Self-consistency and SelfCheckGPT both fail to catch systematic hallucinations. Construct a concrete scenario where a model is both highly self-consistent and confidently wrong, and explain why sampling-based detection cannot, in principle, catch it.
19. A model is "faithful" to its retrieved context but the retrieved document itself contains a factual error. Walk through why NLI-based factuality checking will score this as correct, and what additional signal would be needed to catch it.
20. Design a model-merging pipeline (TIES + DARE via MergeKit) to combine a math-tuned, code-tuned, and instruct-tuned model from the same base — specify density/weight parameters and how you'd detect if λ scaling is eroding general capability.
21. Frankenmerging concatenates layers from different models with identical architecture but no guarantee of interface compatibility. Explain, in terms of what a transformer layer actually learns to expect as input, why this sometimes works and often produces garbage.
22. Design a production VLM serving pipeline for a document-QA product where images range from simple photos to dense multi-page scans — address resolution routing, KV cache sizing, and latency budget across preprocessing/encoding/prefill/decode.
23. Why did moving from a Whisper-ASR-to-text pipeline to a native audio-token model (GPT-4o style) change latency by an order of magnitude, and what capability was structurally impossible in the old pipeline regardless of how good the ASR model became?
24. Design the security boundary for an MCP-based agent with access to a filesystem server and a database server, where the host serves both an admin UI and a customer-facing chatbot. Specify exactly what the host filters vs. what each server independently enforces, and why duplicating that boundary in only one layer is insufficient.
25. A document read by an MCP tool contains an embedded prompt-injection payload instructing the model to call a destructive tool. Trace exactly where in the MCP request/response lifecycle this attack must be neutralized, and explain why "just tell the model to ignore instructions in tool results" is not a sufficient defense.
26. You must choose between MCP, raw provider function-calling, and a LangChain tool abstraction for a tool that must be shared across three independent internal teams' agent stacks. Justify the choice along reusability, deployment, and versioning axes — and identify the scenario where MCP is overkill.
27. A newly fine-tuned model excels at its target task but regresses on basic arithmetic the base model handled fine. Diagnose the mechanism (catastrophic forgetting) and design a mitigation that would show up in a CI eval pipeline before deployment.
28. Explain precisely why distillation, quantization, and fine-tuning are not substitutes for each other when the goal is deploying a smaller model to a resource-constrained environment — what does each fail to achieve that the others solve?
29. Construct an example where two task vectors from genuinely complementary (non-conflicting) capabilities still degrade a merged model's performance, despite no sign conflicts at any individual parameter.
30. You are asked to reduce end-to-end chatbot latency that spans retrieval, reranking, generation, and a safety guardrail pass. Given a fixed latency budget, propose an allocation across these stages and justify which stage you would cut first under further pressure, and which you would never cut.
