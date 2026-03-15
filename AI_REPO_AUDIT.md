# AI Repository Audit — Ml-overview

**Purpose:** Transform this repository into a **Complete Modern AI Engineering Knowledge Base** covering classical ML, deep learning, and modern AI systems (LLMs, agentic AI, RAG, vector databases).

---

## 1. Existing Topics (Current Coverage)

### Classical Machine Learning
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| Linear regression | `machine-learning/supervised-learning.md` | **Deep** | Maths, gradient descent, Python implementation, pros/cons, use cases |
| Logistic regression | `machine-learning/supervised-learning.md` | **Deep** | Formulas, metrics, code |
| SVM | `machine-learning/supervised-learning.md` | **Deep** | Kernel trick, margins, pros/cons |
| Decision trees | `machine-learning/supervised-learning.md` | **Deep** | Splits, impurity, pruning |
| Ensemble learning | `machine-learning/supervised-learning.md` | **Deep** | Bagging, boosting (AdaBoost, GBM, XGBoost) |
| Naive Bayes, KNN | `machine-learning/supervised-learning.md` | **Deep** | Explained with use cases |
| Regression/classification metrics | `machine-learning/supervised-learning.md` | **Deep** | MAE, MSE, RMSE, R², precision, recall, F1, ROC, confusion matrix |
| Clustering | `machine-learning/unsupervised-learning.md` | **Medium–Deep** | K-means, hierarchical, DBSCAN |
| Dimensionality reduction | `machine-learning/unsupervised-learning.md` | **Medium** | PCA, t-SNE |
| ML overview | `introduction-to-ai.md`, `machine_learning_overview.ipynb` | **Medium** | Bias–variance, algorithm list; notebook has placeholders |

### Deep Learning
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| Backpropagation | `deep-learning/parts-of-deep-learning/backpropagation.md` | **Stub** | Title only |
| Hidden layers | `deep-learning/parts-of-deep-learning/hidden-layers.md` | **Stub** | Title only |
| Activation functions | `deep-learning/parts-of-deep-learning/activation-functions.md` | **Stub** | Title only |
| Loss functions | `deep-learning/parts-of-deep-learning/loss-functions.md` | **Stub** | Title only |
| Optimizers | `deep-learning/parts-of-deep-learning/optimizers.md` | **Stub** | Title only |
| Regularization | `deep-learning/parts-of-deep-learning/regularization.md` | **Stub** | Title only |
| Attention | `deep-learning/parts-of-deep-learning/attention.md` | **Stub** | Title only |
| Transformers | `deep-learning/parts-of-deep-learning/transformers.md` | **Stub** | Title only |
| Autoencoders | `deep-learning/parts-of-deep-learning/autoencoders.md` | **Stub** | Title only |
| Computer vision | `deep-learning/deep-learning-methods/computer-vision.md` | **Stub** | Title only |
| NLP | `deep-learning/deep-learning-methods/nlp.md` | **Stub** | Title only |
| Time series | `deep-learning/deep-learning-methods/time-series.md` | **Stub** | Title only |
| Generative models | `deep-learning/deep-learning-methods/generative-models.md` | **Stub** | Title only |
| DL theory (detailed) | `book-notes/deep-learning/*` | **Deep** | D2L, Grokking, PyTorch, LLM from scratch, etc. |

### Reinforcement Learning
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| RL | `introduction-to-ai.md`, `30-days/page/day-1-2-*.md` | **Shallow** | Brief mention only; no dedicated section |

### NLP / LLM
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| LLM training | `llm-applications/how-to-train-your-dragon-llm.md` | **Medium** | Transformers, GPT vs BERT, training |
| RAG | `llm-applications/rag.md` | **Medium** | Indexing, retrieval, augmentation, generation; Traditional vs Graph RAG |
| Compound AI / DB Genie | `llm-applications/db-genie.md` | **Medium** | Tools, retrieval, compound systems |
| LLM research | `research-papers/deep-learning/llm.md` | **Deep** | Survey, benchmarks, agents |

### Computer Vision
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| CV | `deep-learning/deep-learning-methods/computer-vision.md`, `research-papers/deep-learning/computer-vision.md` | **Stub** | No substantive content |

### AI Systems / MLOps
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| MLOps | `mlops.md` | **Shallow** | Single external link |
| MLOps book notes | `book-notes/mlops/*` | **Deep** | Design patterns, designing ML systems, MLE, Keras→K8s |
| PyTorch | `pytorch/pytorch-fundamentals.md` | **Medium** | Tensors, GPU, reproducibility; no full training loop |

### Other
| Topic | Location | Depth | Notes |
|-------|----------|--------|--------|
| ML glossary | `ml-glossary.md` | **Empty** | Placeholder |
| Interview prep | `interview/machine-learning-interviews.md` | **Medium** | Book summary; no problem bank |
| 30-day plan | `30-days/*` | **Mixed** | Days 1–14 filled; many stub pages (e.g. page-15–32) |

---

## 2. Outdated Content

- **ml-glossary.md** — Empty; needs definitions for ML/DL/LLM/RAG/agents.
- **mlops.md** — Only a link; no in-repo pipelines, experiment tracking, or deployment.
- **Deep learning core** — All of `parts-of-deep-learning/` and `deep-learning-methods/` are stubs (no explanations).
- **Research-papers** — `research-papers/ml.md`, `research-papers/mlops.md`, `research-papers/deep-learning/computer-vision.md` are stubs.
- **30-days** — Many placeholder pages (e.g. page-13–32, page-15–20) with only "# Page N".
- **machine_learning_overview.ipynb** — Computer vision, NLP, generative, time series, activation functions, attention, autoencoders, backprop say "(Further details to be added)".
- **PyTorch** — Only tensor basics; no `nn.Module`, training loop, or end-to-end example.

---

## 3. Missing Modern AI Concepts

- **Reinforcement learning** — No dedicated section (MDP, Q-learning, policy gradient, DQN, etc.).
- **Computer vision** — No CNNs, ViT, detection, segmentation.
- **Modern LLM/GenAI** — No dedicated sections on:
  - Prompt engineering
  - Tool use / function calling
  - Agentic AI (ReAct, plan-and-execute, multi-agent)
  - Evaluation (LLM-as-judge, benchmarks)
  - Alignment (RLHF, DPO, instruction tuning)
  - Open vs closed models, scaling
- **Vector stores / embeddings** — Only in RAG context; no standalone treatment of vector DBs, ANN, HNSW, FAISS.
- **Agentic AI** — No section on autonomous agents, reasoning loops, tool-calling, memory, orchestration.
- **RAG (expanded)** — Missing: chunking strategies, reranking, hybrid search, evaluation, latency, scaling.
- **LLM system design** — Missing: context windows, tokenization, inference pipelines, batching, caching, model serving.
- **AI application architectures** — Missing: conversational AI, copilots, document search, recommendations, agent-with-tools end-to-end.
- **Modern AI infrastructure** — Missing: LLM orchestration (LangChain, etc.), vector DB systems, model serving platforms, GPU/distributed training.

---

## 4. Recommended New Sections

1. **AGENTIC_AI/** — New major section: autonomous agents, reasoning/planning, tool use, multi-agent systems, orchestration, memory; frameworks and diagrams.
2. **Vector Databases** — New section (under `llm-applications/` or new top-level): embeddings, similarity search, ANN, HNSW, FAISS, indexing, semantic search.
3. **RAG (expand)** — Expand `llm-applications/rag.md`: chunking, reranking, hybrid search, evaluation, latency, scaling.
4. **LLM System Design** — New section: prompt engineering, context windows, tokenization, inference, batching/caching, serving.
5. **AI Application Architectures** — New section: conversational AI, copilots, document search, recommendation systems, agents with tools; data flow, bottlenecks, scalability.
6. **Deep learning (fill stubs)** — Populate `parts-of-deep-learning/` and `deep-learning-methods/` with attention, transformers, encoder-decoder, pretraining vs fine-tuning, instruction tuning, RLHF; CV (CNNs, ViT); NLP; generative models.
7. **Reinforcement Learning** — New section (e.g. `machine-learning/reinforcement-learning.md` or new folder): MDP, Q-learning, policy gradient, DQN, PPO, brief connection to RLHF.
8. **Practical Implementation Guides** — Pseudocode and Python/PyTorch for: RAG pipeline, AI agent, embedding search.
9. **Modern AI Infrastructure** — Notes on LLM orchestration, vector DBs, model serving, GPU/distributed training.
10. **MODERN_AI_ENGINEER_ROADMAP.md** — Learning path from ML fundamentals to modern AI systems; recommended projects and exercises.
11. **Quick revision sections** — At end of each major topic: key concepts, diagrams, summaries, practical insights.
12. **ML glossary** — Populate with ML/DL/LLM/RAG/agents/vector DB terms.

---

## 5. Categorization Summary

| Category | Current State | Target State |
|----------|----------------|--------------|
| **Classical ML** | Strong (supervised, unsupervised) | Preserve and add quick revision + ensure all 7 topics have formulas, code, use cases |
| **Deep Learning** | Structure only; stubs | Full content: backprop, activations, loss, optimizers, attention, transformers, pretraining, RLHF, CV, NLP, generative |
| **Reinforcement Learning** | Mention only | Dedicated section with core algorithms |
| **NLP** | LLM applications + research | Keep; add LLM system design, prompt engineering, tokenization |
| **Computer Vision** | Stubs | CNNs, ViT, detection, segmentation |
| **AI Systems** | MLOps link + book notes | In-repo MLOps overview; add LLM serving, vector DBs, agent orchestration, infrastructure |

---

## 6. Next Steps (Transformation Plan)

1. Create **AI_REPO_AUDIT.md** (this file).
2. Preserve and improve classical ML (add quick revision blocks where missing).
3. Expand deep learning: fill stubs (attention, transformers, etc.) and add pretraining/fine-tuning/RLHF.
4. Add **AGENTIC_AI** section with agents, tools, multi-agent, memory, frameworks.
5. Expand **RAG** and add **Vector Databases** section.
6. Add **LLM System Design** and **AI Application Architectures**.
7. Add **Practical Implementation Guides** and **Modern AI Infrastructure**.
8. Create **MODERN_AI_ENGINEER_ROADMAP.md** and add quick revision sections to major topics.
9. Populate **ml-glossary.md** and update **SUMMARY.md** for new content.
