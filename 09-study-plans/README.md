# ML Study Plans

This folder contains structured learning paths for ML interview preparation and career development. Choose the track that matches your goal.

---

## How to Use This Folder

| Track | Use Case | Start Here |
|-------|----------|------------|
| [30-Day Interview Sprint](#30-day-interview-sprint) | Active interview prep | Week 1 foundations |
| [Career Path Tracks](#career-path-tracks) | Long-horizon skill building | Pick your role below |
| [Skill Level Tracks](#skill-level-tracks) | Self-assessed entry point | Beginner / Intermediate / Advanced |
| [Resource Lists](#resource-lists) | Books, courses, papers | See section below |

---

## 30-Day Interview Sprint

**Reality check:** Thirty days is a **training block**, not a magic spell. Show up daily, keep notes short, and tie each topic back to something you already ship (pipelines, rollbacks, observability).

**Pair with:** [AI_ML_REVISION_GUIDE.md](../01-foundations/AI_ML_REVISION_GUIDE.md) for the **night-before** sprint, and [mlops.md](../06-production-ml/mlops.md) when the conversation turns to **production**.

### Week 1: Foundations & Data Chemistry
*Focus: Understanding how data moves and how simple models fail.*

- **Day 1-2: ML Logic & Paradoxes**
    - Study: [Fundamentals of Machine Learning](../07-interview-prep/ml/fundamentals-of-machine-learning.md)
    - Key Concept: Bias-Variance Tradeoff, Occam's Razor, No Free Lunch.
- **Day 3-4: The Data Pipeline**
    - Study: [Data Preprocessing & Feature Engineering](../07-interview-prep/ml/data-preprocessing-and-feature-engineering.md)
    - Key Concept: Data Leakage prevention, Imputation strategies (MICE/Indicator).
- **Day 5-7: EDA & Statistical Mastery**
    - Study: [Probability & Statistics](../07-interview-prep/ml/probability-and-statistics.md)
    - Key Concept: P-values, CLT, Type I vs. Type II Errors, EDA patterns.

### Week 2: Algorithms & Theory (Whiteboard Ready)
*Focus: Derivations and the "why" behind the algorithms.*

- **Day 8-9: Classical Supervised Learning**
    - Study: [Algorithms & Theory](../07-interview-prep/ml/algorithms.md)
    - Key Concept: SVM Hinge Loss, k-NN curse of dimensionality, ensemble methods.
- **Day 10-11: Unsupervised Learning**
    - Study: [Unsupervised Learning](../02-classical-ml/unsupervised-learning.md)
    - Key Concept: K-Means convergence, PCA variance vs. predictive power.
- **Day 12-14: Neural Networks & Backprop**
    - Study: [Deep Learning Overview](../03-deep-learning/README.md)
    - Key Concept: Backprop Chain Rule, activation functions, Adam vs. SGD.
- **Day 15-16: Model Evaluation**
    - Study: [Model Evaluation & Metrics](../07-interview-prep/ml/model-evaluation.md)
    - Key Concept: ROC vs. PRC, Calibration Curves, F1-Score logic.
- **Day 17-18: Hyperparameter Tuning**
    - Study: Day 17-18 file in this folder.
    - Key Concept: Grid vs. random vs. Bayesian search, regularization levers.
- **Day 19-21: Specialized Techniques (NLP & CV)**
    - Study: Day 19-21 file in this folder.
    - Key Concept: Self-Attention, BERT vs. GPT, CNN inductive biases vs. ViT.

### Week 3: System Design & Production Polish
*Focus: RAG, Agentic Systems, and MLOps.*

- **Day 22-23: ML System Design**
    - Study: [System Design & MLOps](../07-interview-prep/ml/system-design-and-mlops.md)
    - Key Concept: Retrieval-Ranking, train-serve skew, feature stores.
- **Day 24: Case Studies**
    - Study: Day 23 case studies file in this folder.
    - Key Concept: YouTube two-tower, Google Search modularity, fairness in ranking.
- **Day 25: Behavioral & Soft Skills**
    - Study: Day 24-25 behavioral file in this folder.
    - Key Concept: STAR method, defending technical choices, cross-team collaboration.

### Week 4: Final Prep
*Focus: LLM-specific topics, rapid-fire Q&A, mock interviews.*

- **Day 26-27: LLM Specialist Training**
    - Study: [LLM Fundamentals](../05-llms/interview-notes/llm-fundamentals.md) & [RAG](../05-llms/interview-notes/retrieval-augmented-generation-rag.md)
    - Key Concept: Tokenization, RLHF, Prompt Engineering, LoRA, Vector DBs.
- **Day 28: Implementation Fluency**
    - Practice: Implement attention, backprop, k-means from scratch.
- **Day 29: Mock Interview Scenarios**
    - Practice: Full end-to-end timed sessions (25 min per round, spoken aloud).
- **Day 30: Night-Before Sprint**
    - Study: [AI & ML Cheat Sheet](../01-foundations/AI_ML_REVISION_GUIDE.md) only — no new concepts.

### Speed Revision Mode (The "Night Before")
If you have less than 48 hours, skip the deep dives and go straight to:
1. **[Master Revision Cheat Sheet](../01-foundations/AI_ML_REVISION_GUIDE.md)**
2. **[Master Q&A Bank](../07-interview-prep/llm/top-ml-interview-questions.md)**

---

## Career Path Tracks

Each role has a different technical center of gravity. Use the track matching your target role.

### ML Engineer (MLE)

**Core competency:** Shipping ML models into production reliably and at scale.

**Skills emphasis:** System design, inference optimization, MLOps, feature engineering, model serving.

**6-Month Curriculum:**

| Month | Focus | Key Topics |
|-------|-------|------------|
| 1 | ML Fundamentals | Supervised/unsupervised learning, evaluation metrics, bias-variance |
| 2 | Classical ML Depth | Ensembles, SVMs, feature engineering, cross-validation |
| 3 | Deep Learning | Neural networks, CNNs, Transformers, PyTorch fundamentals |
| 4 | Production ML | MLOps, feature stores, monitoring, A/B testing, CI/CD for models |
| 5 | System Design | Retrieval-ranking, real-time vs. batch serving, distributed training |
| 6 | Specialization | LLMs in production, RAG systems, inference optimization (quantization, distillation) |

**Project Milestones:**
- Month 2: End-to-end tabular ML pipeline (data cleaning → feature engineering → XGBoost → cross-validation report)
- Month 3: Image classifier with PyTorch, trained and evaluated with proper holdout strategy
- Month 4: Deploy a model as a REST API with latency monitoring and a retraining trigger
- Month 5: Design and implement a two-stage retrieval + ranking system for a small document corpus
- Month 6: RAG pipeline with chunking, vector DB, reranker, and an evaluation harness

**Milestone Checkpoints:**

After Month 2: Can you build a full feature pipeline without data leakage? Can you explain the bias-variance tradeoff using a concrete example from your project?

After Month 4: Can you deploy a model, monitor it, and write a rollback procedure? Have you debugged a train-serve skew issue?

After Month 6: Can you design a retrieval-ranking system on a whiteboard in 30 minutes? Can you explain 3 inference optimization techniques with tradeoffs?

---

### Research Scientist (RS)

**Core competency:** Advancing the state of the art through novel algorithms, architectures, or training methods.

**Skills emphasis:** Mathematical depth, paper reading, experimental rigor, novel contributions.

**6-Month Curriculum:**

| Month | Focus | Key Topics |
|-------|-------|------------|
| 1 | Mathematical Foundations | Linear algebra, probability theory, information theory, optimization |
| 2 | ML Theory | PAC learning, VC dimension, generalization bounds, kernel methods |
| 3 | Deep Learning Theory | Universal approximation, expressivity, implicit regularization of SGD |
| 4 | Research Areas | Transformers in depth, generative models (VAEs, diffusion), RLHF |
| 5 | Paper Reading & Replication | Reproduce 3 landmark papers; understand experimental setup fully |
| 6 | Original Research | Identify a gap in existing work; design and run a controlled experiment |

**Project Milestones:**
- Month 2: Implement logistic regression, PCA, and SVM from scratch (no sklearn) and verify against sklearn outputs
- Month 3: Implement a transformer from scratch (attention, positional encoding, multi-head), train on a toy task
- Month 5: Reproduce the results of one seminal paper (e.g., Attention Is All You Need, BERT, DDPM)
- Month 6: Write a 4-page report documenting an experiment that tests a hypothesis about model behavior

**Milestone Checkpoints:**

After Month 2: Can you derive the bias-variance decomposition from first principles? Can you prove the convergence conditions for gradient descent?

After Month 4: Can you read a new paper and, within 2 hours, identify the core contribution, the key experiment, and one limitation the authors do not acknowledge?

After Month 6: Have you run a controlled ablation study? Can you explain what a null result means and why it is not a failure?

---

### Data Scientist (DS)

**Core competency:** Extracting actionable insights from data and communicating them to decision-makers.

**Skills emphasis:** Statistics, experimentation, visualization, storytelling, business impact framing.

**6-Month Curriculum:**

| Month | Focus | Key Topics |
|-------|-------|------------|
| 1 | Statistics & Probability | Hypothesis testing, confidence intervals, Bayesian reasoning, A/B test design |
| 2 | Data Wrangling | SQL depth, pandas, data quality assessment, EDA methodology |
| 3 | Classical ML | Regression, classification, clustering, feature engineering for tabular data |
| 4 | Experimentation | A/B testing, causal inference, difference-in-differences, synthetic controls |
| 5 | Communication | Visualization principles, narrative structure, translating ML output to decisions |
| 6 | Domain Application | Pick one vertical (e.g., growth, finance, product) and apply all skills to a real dataset |

**Project Milestones:**
- Month 2: Full EDA report on a public dataset — distributions, missingness, correlations, and 3 actionable insights
- Month 3: Churn prediction model with a business-facing summary (not a model card — a decision memo)
- Month 4: Design, simulate, and analyze a mock A/B test from scratch, including power analysis
- Month 6: End-to-end analysis project: business question → data → model → recommendation → stakeholder presentation

**Milestone Checkpoints:**

After Month 2: Can you identify 5 data quality problems in a dataset and specify which ones affect downstream modeling?

After Month 4: Can you design an A/B test that avoids novelty effect, network effects, and survivorship bias? Can you calculate the required sample size?

After Month 6: Can you tell a business story from a model output without using technical jargon?

---

### MLOps Engineer

**Core competency:** Building the infrastructure that makes ML systems reliable, reproducible, and scalable in production.

**Skills emphasis:** CI/CD, containerization, distributed systems, monitoring, data pipelines, model registries.

**6-Month Curriculum:**

| Month | Focus | Key Topics |
|-------|-------|------------|
| 1 | ML Basics | Enough ML to understand what you are operationalizing (weeks 1-2 of this plan) |
| 2 | Software Engineering for ML | Docker, git workflows, testing ML code, configuration management |
| 3 | Training Pipelines | Orchestration (Airflow/Prefect), data versioning (DVC), experiment tracking (MLflow) |
| 4 | Model Serving | REST APIs (FastAPI), model registries, canary deployments, shadow mode |
| 5 | Monitoring & Observability | Data drift detection, model performance monitoring, alerting, feature freshness |
| 6 | Distributed & Advanced | Distributed training (Ray, Horovod), feature stores, infra-as-code, cost optimization |

**Project Milestones:**
- Month 2: Containerize an ML training and inference pipeline with Docker; write unit tests for data preprocessing
- Month 3: Build a reproducible training pipeline with MLflow experiment tracking and DVC data versioning
- Month 4: Deploy a model behind a FastAPI endpoint with canary rollout and a health check endpoint
- Month 5: Implement data drift detection that triggers a Slack alert and creates a retraining job
- Month 6: Full ML platform: training pipeline → model registry → serving → monitoring → automated retraining

**Milestone Checkpoints:**

After Month 2: Can you reproduce a training run exactly from a git hash and a DVC data version?

After Month 4: Can you route 10% of traffic to a new model version, monitor it, and roll back programmatically?

After Month 6: Can you describe the failure modes of your monitoring system — what would it miss and why?

---

## Skill Level Tracks

Use these to calibrate where to enter the 6-month career tracks above.

### Beginner Track (0-6 months ML exposure)

**Prerequisites:** Python programming, basic statistics (mean, variance, probability).

**Focus:** Build intuition before formalism. Work through concepts hands-on before reading derivations.

**Week-by-Week Plan (First 8 Weeks):**

| Week | Topic | Core Activity |
|------|-------|---------------|
| 1 | What is ML? Supervised vs. Unsupervised | Run a linear regression on a toy dataset; plot predictions vs. actuals |
| 2 | Data Cleaning & EDA | Take a messy Kaggle dataset, document 5 problems, fix them |
| 3 | Classification | Train a decision tree classifier; visualize the tree; change max_depth and observe |
| 4 | Model Evaluation | Compute precision, recall, F1 by hand for a small confusion matrix |
| 5 | Ensembles | Compare a single tree vs. random forest vs. XGBoost on the same dataset |
| 6 | Neural Networks Intro | Build a 2-layer network in PyTorch on MNIST; watch training and validation curves |
| 7 | Feature Engineering | Take a raw tabular dataset; engineer 5 new features; measure if they help |
| 8 | End-to-End Project | Kaggle competition: submit a baseline, then iterate 3 times |

**Key Resources:**
- *Hands-On Machine Learning* (Aurélien Géron) — Chapters 1-8 for this phase
- fast.ai Practical Deep Learning for Coders (Part 1)
- Kaggle Learn micro-courses (Python, Pandas, ML Intro, Intermediate ML)

**Milestone:** After 8 weeks, you should be able to take any tabular dataset, perform EDA, train at least 3 models, evaluate them properly, and explain why one outperforms the others.

---

### Intermediate Track (6 months – 2 years ML exposure)

**Prerequisites:** Comfortable with sklearn, can train and evaluate basic models, some Python for data analysis.

**Focus:** Go from "can run models" to "can explain every decision and debug failures."

**Week-by-Week Plan (8 Weeks):**

| Week | Topic | Core Activity |
|------|-------|---------------|
| 1 | Bias-Variance Deep Dive | Plot learning curves; diagnose over/underfitting; apply fix |
| 2 | Feature Engineering Advanced | Target encoding, interaction terms, temporal features |
| 3 | Deep Learning | Train a ResNet on CIFAR-10; understand every layer; add augmentation |
| 4 | NLP Fundamentals | TF-IDF classifier → fine-tune BERT on same task; compare |
| 5 | ML System Design Intro | Design a recommendation system on paper using the 5-step framework |
| 6 | MLOps Basics | Containerize a model; set up MLflow tracking; write a test for a data pipeline |
| 7 | LLMs & RAG | Build a basic RAG pipeline over a document set; evaluate retrieval quality |
| 8 | End-to-End Production Project | Build a model → serve as API → monitor with drift detection |

**Key Resources:**
- *Deep Learning* (Goodfellow, Bengio, Courville) — Chapters 5-9
- CS229 (Stanford) lecture notes — freely available
- *Designing Machine Learning Systems* (Chip Huyen) — Full book
- Papers: Attention Is All You Need, BERT, XGBoost

**Milestone:** After 8 weeks, you should be able to take a system design interview question, sketch a full architecture including monitoring, and defend tradeoffs at each stage.

---

### Advanced Track (2+ years ML experience)

**Prerequisites:** Has shipped ML to production, comfortable with deep learning, has read research papers.

**Focus:** Depth in theory, frontier research, architectural decisions at scale.

**8-Week Focus Areas:**

| Week | Topic | Core Activity |
|------|-------|---------------|
| 1 | ML Theory | Derive generalization bounds; understand the double descent phenomenon |
| 2 | Transformer Internals | Implement multi-head attention from scratch; analyze attention patterns |
| 3 | Scaling Laws | Read Chinchilla paper; implement compute-optimal training calculator |
| 4 | Alignment & RLHF | Understand PPO for RLHF; contrast with DPO; study reward hacking failure modes |
| 5 | Inference Optimization | Implement quantization (int8); benchmark speculative decoding; profile KV cache |
| 6 | Distributed Training | Set up data parallelism vs. tensor parallelism; understand ZeRO optimizer stages |
| 7 | Evaluation at Scale | Design an evaluation harness for an LLM; read HELM and BIG-Bench papers |
| 8 | Research Replication | Reproduce one paper from the last 12 months; document the delta from the paper |

**Key Resources:**
- *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — Full book
- Andrej Karpathy's minGPT/nanoGPT as code reference
- Papers: Chinchilla, InstructGPT, Flash Attention, LoRA, DPO
- Anthropic and OpenAI alignment research blog posts

**Milestone:** After 8 weeks, you should be able to read a new frontier paper, implement its core idea in code within a week, and identify one limitation the authors did not fully address.

---

## Resource Lists

### Books

**Foundations & Classical ML:**
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Géron) — Best practical intro
- *Pattern Recognition and Machine Learning* (Bishop) — Rigorous probabilistic treatment
- *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) — Free PDF; statistical ML depth
- *Introduction to Statistical Learning* (James et al.) — Gentler ESL; free PDF; R and Python labs

**Deep Learning:**
- *Deep Learning* (Goodfellow, Bengio, Courville) — Free online; standard reference
- *Dive into Deep Learning* (d2l.ai) — Free; interactive; code-first

**Production & MLOps:**
- *Designing Machine Learning Systems* (Chip Huyen) — Best production ML book currently available
- *Machine Learning Engineering* (Andriy Burkov) — Concise; practical; free draft online
- *Reliable Machine Learning* (Google SRE for ML) — Operations perspective

**Mathematics:**
- *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — Free PDF; linear algebra + probability + optimization
- *Linear Algebra Done Right* (Axler) — If you need to strengthen linear algebra foundations

**LLMs & Generative AI:**
- *Build a Large Language Model (from Scratch)* (Sebastian Raschka) — Implementation-focused
- *Natural Language Processing with Transformers* (Tunstall et al.) — Practical Hugging Face guide

---

### Courses

**Structured Fundamentals:**
- Andrew Ng's Machine Learning Specialization (Coursera) — Best structured intro to classical ML
- Andrew Ng's Deep Learning Specialization (Coursera) — Standard deep learning curriculum
- fast.ai Practical Deep Learning for Coders — Code-first; shows production patterns early
- CS229 (Stanford) — Rigorous mathematical ML; lecture notes free online

**Deep Learning & Research:**
- CS231n (Stanford) — Convolutional Neural Networks for Visual Recognition; notes and slides free
- CS224N (Stanford) — Natural Language Processing with Deep Learning
- MIT 6.S191 — Introduction to Deep Learning; yearly updated lectures on YouTube

**MLOps & Production:**
- Full Stack Deep Learning (FSDL) — Free course; covers the full production ML stack
- Made With ML (Goku Mohandas) — Free; end-to-end ML engineering curriculum
- MLOps Zoomcamp (DataTalks.Club) — Free; hands-on MLOps pipeline

**LLMs & GenAI:**
- Hugging Face NLP Course — Free; covers transformers, fine-tuning, deployment
- DeepLearning.AI Short Courses (LangChain, RAG, LLMOps) — 1-2 hour focused modules

---

### Foundational Papers

**Classical ML:**
- *A Training Algorithm for Optimal Margin Classifiers* (Boser, Guyon, Vapnik, 1992) — SVM origin
- *Random Forests* (Breiman, 2001) — The canonical ensemble paper
- *XGBoost: A Scalable Tree Boosting System* (Chen & Guestrin, 2016)

**Deep Learning:**
- *Learning Representations by Back-propagating Errors* (Rumelhart et al., 1986)
- *ImageNet Classification with Deep Convolutional Neural Networks* (Krizhevsky et al., 2012) — AlexNet
- *Batch Normalization* (Ioffe & Szegedy, 2015)
- *Deep Residual Learning for Image Recognition* (He et al., 2016) — ResNet
- *Attention Is All You Need* (Vaswani et al., 2017) — The Transformer

**NLP:**
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Language Models are Few-Shot Learners* (Brown et al., 2020) — GPT-3
- *Training language models to follow instructions with human feedback* (Ouyang et al., 2022) — InstructGPT/RLHF

**Scaling & Efficiency:**
- *Scaling Laws for Neural Language Models* (Kaplan et al., 2020)
- *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022) — Chinchilla
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
- *FlashAttention* (Dao et al., 2022)

**Production & Systems:**
- *Hidden Technical Debt in Machine Learning Systems* (Sculley et al., 2015) — Required reading for MLOps
- *Machine Learning: The High Interest Credit Card of Technical Debt* (Sculley et al., 2014)
- *Towards ML Engineering* (Paleyes et al., 2022) — Survey of production ML challenges

---

## Speed Revision Mode (The "Night Before")
If you have less than 48 hours, skip the deep dives and go straight to:
1. **[Master Revision Cheat Sheet](../01-foundations/AI_ML_REVISION_GUIDE.md)**
2. **[Master Q&A Bank](../07-interview-prep/llm/top-ml-interview-questions.md)**

---

> **Pedagogical note:** Don't just read the "Direct Answer" in any section. Spend time in the "Deep Dive" portions to understand mathematical tradeoffs. Real L5+ interviews are won in the "Practical Perspective" sections.
