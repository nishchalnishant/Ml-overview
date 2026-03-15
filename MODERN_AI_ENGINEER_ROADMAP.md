# Modern AI Engineer Roadmap

A progression from **ML fundamentals** to **modern AI systems** (LLMs, RAG, agents, vector search), with recommended projects and exercises.

---

## Phase 1: Foundations (Classical ML)

- **Topics**: Linear and logistic regression, SVM, decision trees, ensemble methods (Random Forest, boosting), clustering (K-means), dimensionality reduction (PCA).
- **Skills**: Math (gradients, loss, metrics), Python (NumPy, scikit-learn), train/eval split, overfitting and regularization.
- **Resources**: This repo — [Machine Learning](machine-learning/README.md), [Supervised](machine-learning/supervised-learning.md), [Unsupervised](machine-learning/unsupervised-learning.md).
- **Project**: Tabular prediction (e.g. Kaggle); implement one algorithm from scratch (e.g. linear regression with gradient descent).

---

## Phase 2: Deep learning basics

- **Topics**: Neural networks, backpropagation, activation functions, loss functions, optimizers (SGD, Adam), regularization (dropout, weight decay).
- **Skills**: PyTorch (or TensorFlow): tensors, `nn.Module`, training loop, GPU.
- **Resources**: [Deep learning](deep-learning/README.md), [Parts of deep learning](deep-learning/parts-of-deep-learning/README.md), [PyTorch fundamentals](pytorch/pytorch-fundamentals.md).
- **Project**: Image or text classification with a small CNN or MLP; train on GPU and tune hyperparameters.

---

## Phase 3: Transformers and LLMs

- **Topics**: Attention, transformer architecture (encoder, decoder, decoder-only), tokenization, pretraining vs fine-tuning, instruction tuning, RLHF/DPO.
- **Resources**: [Attention](deep-learning/parts-of-deep-learning/attention.md), [Transformers](deep-learning/parts-of-deep-learning/transformers.md), [Pretraining / fine-tuning / RLHF](deep-learning/parts-of-deep-learning/pretraining-finetuning-rlhf.md), [How to train your Dragon (LLM)](llm-applications/how-to-train-your-dragon-llm.md).
- **Project**: Use an open LLM (e.g. Llama, Mistral) via API or local; prompt for a task (summarization, QA); try instruction-tuned vs base model.

---

## Phase 4: RAG and vector search

- **Topics**: Why RAG; chunking, embedding, vector store; retrieval (vector, keyword, hybrid); reranking; context injection; evaluation and latency.
- **Resources**: [RAG](llm-applications/rag.md), [Vector databases](llm-applications/vector-databases.md), [Build a RAG system](practical-guides/build-rag-system.md), [Build embedding search](practical-guides/build-embedding-search.md).
- **Project**: Build a RAG app over a doc set (e.g. your notes or a product manual); measure recall@k and answer quality; try chunk size and k.

---

## Phase 5: Agentic AI and tools

- **Topics**: Autonomous agents, reasoning (ReAct, plan-and-execute), tool use, multi-agent systems, memory, orchestration.
- **Resources**: [AGENTIC_AI](AGENTIC_AI/README.md), [Tool-using agents](AGENTIC_AI/tool-using-agents.md), [Build an AI agent](practical-guides/build-ai-agent.md).
- **Project**: Build an agent with 2–3 tools (e.g. search, calculator, RAG); use an orchestration framework (LangChain, LlamaIndex) or minimal loop with OpenAI function calling.

---

## Phase 6: Systems and production

- **Topics**: LLM system design (prompts, context, tokenization, batching, caching); application architectures (conversational, copilot, document search, agents); infrastructure (orchestration, vector DBs, model serving).
- **Resources**: [LLM system design](llm-applications/llm-system-design.md), [AI application architectures](llm-applications/ai-application-architectures.md), [Modern AI infrastructure](modern-ai-infrastructure/README.md).
- **Project**: Design and implement one full application (e.g. doc search with RAG, or coding assistant with tools); consider latency, cost, and observability.

---

## Phase 7: Post-2020 extensions (optional depth)

- **Topics**: Multimodal AI (CLIP, VLMs); generative models (diffusion, latent diffusion, GANs); self-supervised learning (SimCLR, MAE); mixture of experts; long-context (Flash Attention, SSM); architectures beyond transformers (Mamba); AI for science; interpretability; AI safety and alignment.
- **Resources**: [Generative models](deep-learning/deep-learning-methods/generative-models.md), [Computer vision](deep-learning/deep-learning-methods/computer-vision.md), [Multimodal AI](deep-learning/deep-learning-methods/multimodal-ai.md), [Self-supervised learning](deep-learning/parts-of-deep-learning/self-supervised-learning.md), [Mixture of experts](deep-learning/parts-of-deep-learning/mixture-of-experts.md), [Long-context models](llm-applications/long-context-models.md), [Architectures beyond transformers](deep-learning/parts-of-deep-learning/architectures-beyond-transformers.md), [AI for science](ai-for-science/README.md), [Interpretability](deep-learning/interpretability.md), [AI safety and alignment](llm-applications/ai-safety-and-alignment.md).
- **Project**: Pick one area (e.g. fine-tune a small diffusion model, or build a simple VLM pipeline) and go deep.

---

## Practical exercises (quick list)

1. **Linear regression from scratch** (NumPy): gradient descent, MSE, R².
2. **Train a classifier** (scikit-learn or PyTorch): try 2–3 algorithms; compare metrics and runtime.
3. **PyTorch training loop**: one small CNN or MLP; log loss, use validation set.
4. **Prompt an LLM**: few-shot and chain-of-thought on a reasoning task.
5. **RAG pipeline**: chunk → embed → FAISS (or Chroma) → retrieve → prompt → generate; tune k and chunk size.
6. **Agent with tools**: weather + search (or RAG) tool; run loop until answer.
7. **Evaluate RAG**: recall@k and LLM-as-judge or human score for faithfulness.

---

## Summary

| Phase | Focus | Outcome |
|-------|--------|--------|
| 1 | Classical ML | Solid base in regression, classification, ensembles, clustering |
| 2 | Deep learning | Neural nets, PyTorch, training loops, optimizers, regularization |
| 3 | Transformers / LLMs | Attention, tokenization, pretraining, alignment, in-context learning |
| 4 | RAG / vectors | Retrieval, embeddings, chunking, evaluation |
| 5 | Agents | Tools, reasoning loop, orchestration |
| 6 | Systems | End-to-end apps, latency, scale, infrastructure |
| 7 | Post-2020 | Multimodal, generative, SSL, MoE, long context, new arch, safety, interpretability |

Use this repo as your **knowledge base** and the projects/exercises as **hands-on checkpoints** on the path from classical ML to **complete modern AI engineering** (including all major developments after 2020).
