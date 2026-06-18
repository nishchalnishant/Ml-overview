---
module: References
topic: Resources
status: unread
tags: [references, ml, datasets, benchmarks, tools, blogs, conferences]
---
# ML Resources Reference

Datasets, benchmarks, open-source tools, influential people, conference venues, and notation — in one place.

---

## Standard Datasets

### Computer Vision

| Dataset | Size | Task | Why It Matters |
|---------|------|------|----------------|
| [ImageNet (ILSVRC)](https://image-net.org/) | 1.2M images, 1000 classes | Classification | The benchmark that launched modern DL — AlexNet (2012) |
| [COCO](https://cocodataset.org/) | 200K images, 80 categories | Detection, segmentation, keypoints | Standard detection/segmentation benchmark |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | 9M images, 600 classes | Classification, detection | Google's largest annotated image dataset |
| [CIFAR-10 / CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | 60K images | Classification | Quick benchmark for new architectures |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | 200K face images | Face attribute recognition | Faces benchmark; fairness studies |
| [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | 20K images, 150 categories | Semantic segmentation | MIT semantic segmentation benchmark |

### NLP

| Dataset | Task | Why It Matters |
|---------|------|----------------|
| [GLUE](https://gluebenchmark.com/) | Multi-task NLP evaluation | BERT-era standard suite: sentiment, NLI, paraphrase |
| [SuperGLUE](https://super.gluebenchmark.com/) | Harder multi-task NLP | Replaced GLUE once models saturated it |
| [SQuAD 1.1 / 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Reading comprehension QA | Extractive QA benchmark |
| [WMT Translation](https://www.statmt.org/wmt24/) | Machine translation | Standard MT benchmark; En-De, En-Fr |
| [Common Crawl](https://commoncrawl.org/) | LLM pretraining data | ~petabyte-scale web crawl used in GPT, LLaMA |
| [The Pile (EleutherAI)](https://pile.eleuther.ai/) | LLM pretraining data | 825GB curated text from 22 diverse sources |

### Reasoning & Math

| Dataset | Task | Why It Matters |
|---------|------|----------------|
| [GSM8K](https://github.com/openai/grade-school-math) | Math word problems (grade school) | Standard CoT benchmark — 8.5K problems |
| [MATH](https://github.com/hendrycks/math) | Competition math (5 difficulty levels) | Harder math benchmark — AMC/AIME level |
| [MMLU](https://github.com/hendrycks/test) | 57-subject multiple choice | Broad knowledge evaluation: science, law, medicine |
| [HumanEval](https://github.com/openai/human-eval) | Python coding from docstrings | OpenAI coding benchmark — pass@k metric |
| [BIG-Bench](https://github.com/google/BIG-bench) | 204 diverse tasks | Google's suite for testing emergent capabilities |
| [HELM](https://crfm.stanford.edu/helm/) | Holistic LLM evaluation | Stanford's multi-dimensional model evaluation |

### Tabular / Classical ML

| Dataset | Why It Matters |
|---------|----------------|
| [UCI ML Repository](https://archive.ics.uci.edu/) | 600+ tabular datasets — the canonical classical ML benchmark source |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Real-world messy data across many domains |
| [OpenML](https://www.openml.org/) | Reproducible ML experiments with standardized datasets and tasks |

### Time Series

| Dataset | Task | Why It Matters |
|---------|------|----------------|
| [ETT (Electricity Transformer Temperature)](https://github.com/zhouhaoyi/ETDataset) | Long-horizon forecasting | Standard benchmark from the Informer paper |
| [M4 Competition Dataset](https://github.com/Mcompetitions/M4-methods) | Univariate forecasting | 100K series across 6 frequencies |
| [M5 Competition Dataset](https://www.kaggle.com/c/m5-forecasting-accuracy) | Hierarchical sales forecasting | Walmart sales — 42K series, interpretability challenge |
| [Electricity / Traffic / Weather](https://github.com/thuml/Autoformer) | Multivariate forecasting | Standard suite for Transformer forecasting papers |
| [PhysioNet](https://physionet.org/) | Clinical time series (ECG, ICU) | Medical ML benchmarks |

---

## Benchmark Leaderboards

| Leaderboard | Measures | URL |
|-------------|----------|-----|
| **Papers With Code** | SOTA across all tasks with reproducible code | https://paperswithcode.com/sota |
| **LMSYS Chatbot Arena** | LLM win-rate via human preference (Elo) | https://chat.lmsys.org/ |
| **Open LLM Leaderboard (HuggingFace)** | LLM evaluation: ARC, HellaSwag, MMLU, TruthfulQA | https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard |
| **BIG-Bench Hard** | Hard reasoning tasks LLMs struggle with | https://github.com/suzgunmirac/BIG-Bench-Hard |
| **HELM Leaderboard** | Multi-dimensional LLM evaluation (accuracy, robustness, fairness) | https://crfm.stanford.edu/helm/latest/ |
| **AlpacaEval** | Instruction-following quality via GPT-4 judge | https://tatsu-lab.github.io/alpaca_eval/ |
| **SWE-bench** | LLM ability to solve real GitHub issues | https://www.swebench.com/ |
| **MTEB** | Embedding model evaluation across 56 tasks | https://huggingface.co/spaces/mteb/leaderboard |

---

## Open-Source Tools & Libraries

### Core ML / Deep Learning

| Tool | Purpose | When to use |
|------|---------|-------------|
| [PyTorch](https://pytorch.org/) | Dynamic computation graph, research-friendly | Default for research and production training |
| [JAX](https://github.com/google/jax) | NumPy + autograd + XLA compilation | Google-ecosystem, TPU training, functional style |
| [scikit-learn](https://scikit-learn.org/) | Classical ML: preprocessing, models, pipelines | Tabular ML, feature engineering, evaluation |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosted trees | Best single model for most tabular competitions |
| [LightGBM](https://lightgbm.readthedocs.io/) | Fast gradient boosting | Faster than XGBoost for large datasets |
| [CatBoost](https://catboost.ai/) | Gradient boosting with native categoricals | When you have many categorical features |

### LLMs & NLP

| Tool | Purpose | When to use |
|------|---------|-------------|
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Pretrained models, fine-tuning, inference | Everything LLM-related |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention-based LLM serving | High-throughput LLM inference |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU/quantized LLM inference | Local inference on consumer hardware |
| [LangChain](https://www.langchain.com/) | LLM application framework: chains, agents, RAG | Rapid LLM app prototyping |
| [LlamaIndex](https://www.llamaindex.ai/) | Data framework for RAG | Document Q&A, retrieval-augmented generation |
| [Sentence Transformers](https://www.sbert.net/) | Dense text embeddings | Semantic search, RAG retrieval, similarity |
| [tiktoken](https://github.com/openai/tiktoken) | OpenAI tokenizer (BPE) | Token counting, cost estimation |
| [PEFT](https://github.com/huggingface/peft) | LoRA, QLoRA, prefix tuning | Parameter-efficient fine-tuning |
| [TRL](https://github.com/huggingface/trl) | RLHF, DPO, PPO training | Alignment fine-tuning pipelines |

### MLOps & Production

| Tool | Purpose | When to use |
|------|---------|-------------|
| [MLflow](https://mlflow.org/) | Experiment tracking, model registry | Logging metrics, params, artifacts |
| [Weights & Biases (W&B)](https://wandb.ai/) | Experiment tracking + sweeps | Rich visualization; hyperparameter sweeps |
| [DVC](https://dvc.org/) | Data and model versioning | Git for large data and model files |
| [Feast](https://feast.dev/) | Feature store | Train-serve consistency for online/offline features |
| [BentoML](https://www.bentoml.com/) | Model serving | Packaging and serving any ML model as an API |
| [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) | Distributed model serving | High-performance multi-model serving |
| [Evidently AI](https://www.evidentlyai.com/) | Data drift and model monitoring | Production drift detection and reporting |
| [Seldon Core](https://www.seldon.io/) | Kubernetes-native model serving | Enterprise-scale ML deployment |
| [ZenML](https://www.zenml.io/) | ML pipeline orchestration | Portable, reproducible ML pipelines |
| [Prefect](https://www.prefect.io/) | Workflow orchestration | Data and ML pipeline scheduling |
| [Apache Airflow](https://airflow.apache.org/) | Workflow scheduling | Large-scale DAG-based pipeline orchestration |

### Vector Databases (RAG / Embedding Search)

| Tool | When to use |
|------|-------------|
| [Pinecone](https://www.pinecone.io/) | Managed vector search at scale |
| [Weaviate](https://weaviate.io/) | Open-source vector DB with built-in ML modules |
| [Chroma](https://www.trychroma.com/) | Lightweight local vector DB for prototyping |
| [Qdrant](https://qdrant.tech/) | High-performance open-source vector search |
| [pgvector](https://github.com/pgvector/pgvector) | Vector search inside PostgreSQL |
| [FAISS](https://github.com/facebookresearch/faiss) | Meta's library for billion-scale ANN search |

### Evaluation & Interpretability

| Tool | Purpose |
|------|---------|
| [SHAP](https://shap.readthedocs.io/) | Shapley-value feature attribution for any model |
| [LIME](https://github.com/marcotcr/lime) | Local surrogate explanations |
| [Captum](https://captum.ai/) | PyTorch model interpretability |
| [Eleuther LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standardized LLM benchmark evaluation |
| [deepeval](https://github.com/confident-ai/deepeval) | LLM output evaluation framework |

---

## Influential Blogs & Researchers to Follow

### Blogs (regularly updated with research-quality content)

| Blog | Focus |
|------|-------|
| [Lilian Weng (lilianweng.github.io)](https://lilianweng.github.io/) | Deep dives on LLMs, RL, diffusion, agents — best technical blog in ML |
| [Sebastian Ruder (ruder.io)](https://ruder.io/) | NLP research progress reports |
| [Jay Alammar (jalammar.github.io)](https://jalammar.github.io/) | Visual explanations of transformers, BERT, GPT |
| [Andrej Karpathy (karpathy.github.io)](https://karpathy.github.io/) | Neural network intuitions, NN training tricks |
| [The Gradient](https://thegradient.pub/) | Research-accessible essays on ML trends |
| [Distill.pub](https://distill.pub/) | Interactive, rigorous ML explanations (archived but essential) |
| [Eugene Yan (eugeneyan.com)](https://eugeneyan.com/) | Applied ML, recommender systems, LLM evaluation |
| [Chip Huyen (huyenchip.com)](https://huyenchip.com/) | ML systems design, production ML, AI engineering |
| [Simon Willison's Weblog](https://simonwillison.net/) | LLM applications and ecosystem news |
| [Ahead of AI (Sebastian Raschka)](https://magazine.sebastianraschka.com/) | Research paper summaries, LLM training |

### Researchers to follow (Twitter/X, Google Scholar)

| Person | Known for |
|--------|----------|
| Andrej Karpathy | GPT, Tesla Autopilot, neural network pedagogy |
| Yann LeCun | CNNs, Meta AI Chief Scientist, JEPA |
| Geoffrey Hinton | Deep learning foundations, Boltzmann machines, capsules |
| Yoshua Bengio | RNNs, attention foundations, AI safety |
| Ilya Sutskever | GPT, scaling, OpenAI co-founder |
| Demis Hassabis | AlphaFold, AlphaGo, DeepMind CEO |
| Pieter Abbeel | Reinforcement learning, robotics |
| Chelsea Finn | Meta-learning, MAML |
| Percy Liang | HELM, CRFM, foundation model evaluation |
| Tim Dettmers | Quantization, QLoRA, bitsandbytes |
| Tri Dao | FlashAttention, Mamba |

---

## Conference Venues

### Machine Learning

| Conference | Focus | Acceptance Rate | Cadence |
|-----------|-------|-----------------|---------|
| [NeurIPS](https://neurips.cc/) | Broad ML/DL — algorithms, theory, systems | ~25% | Annual (December) |
| [ICML](https://icml.cc/) | Machine learning — theory and applications | ~28% | Annual (July) |
| [ICLR](https://iclr.cc/) | Representation learning, deep learning | ~32% | Annual (May) |
| [AISTATS](https://aistats.org/) | Probabilistic ML, statistics, Bayesian methods | ~25% | Annual (May) |
| [UAI](https://www.auai.org/) | Probabilistic and Bayesian reasoning | ~30% | Annual |

### Systems & Infrastructure

| Conference | Focus |
|-----------|-------|
| [MLSys](https://mlsys.org/) | ML systems, compilers, hardware, inference |
| [OSDI / SOSP](https://www.usenix.org/conferences/byname/179) | Systems papers for ML (vLLM, Orca published here) |
| [SysML / EuroSys](https://sysml.cc/) | ML systems and distributed computing |

### Computer Vision

| Conference | Focus |
|-----------|-------|
| [CVPR](https://cvpr.thecvf.com/) | Computer vision — largest CV conference |
| [ICCV](https://iccv2025.thecvf.com/) | International computer vision (odd years) |
| [ECCV](https://eccv.ecva.net/) | European computer vision (even years) |

### NLP

| Conference | Focus |
|-----------|-------|
| [ACL](https://www.aclweb.org/portal/acl) | Computational linguistics and NLP |
| [EMNLP](https://2024.emnlp.org/) | Empirical methods in NLP |
| [NAACL](https://naacl.org/) | North American chapter of ACL |
| [EACL](https://eacl.org/) | European chapter of ACL |

### Specialized

| Conference | Focus |
|-----------|-------|
| [KDD](https://www.kdd.org/) | Knowledge discovery, data mining, applied ML |
| [AAAI](https://aaai.org/) | Broad AI including symbolic methods |
| [SIGIR](https://sigir.org/) | Information retrieval, search, recommendation |
| [RecSys](https://recsys.acm.org/) | Recommender systems |
| [FAccT](https://facctconference.org/) | Fairness, accountability, transparency in ML |

---

## Notation Glossary

Standard notation used across ML papers and textbooks.

### Scalars, Vectors, Matrices

| Symbol | Meaning |
|--------|---------|
| $x$ | Scalar (lowercase italic) |
| $\mathbf{x}$ | Vector (lowercase bold): $\mathbf{x} \in \mathbb{R}^n$ |
| $\mathbf{W}$ | Matrix (uppercase bold): $\mathbf{W} \in \mathbb{R}^{m \times n}$ |
| $\mathcal{X}$ | Set or space (calligraphic) |
| $x_i$ | $i$-th element of vector $\mathbf{x}$ |
| $W_{ij}$ | Element in row $i$, column $j$ of $\mathbf{W}$ |
| $\mathbf{x}^\top$ | Transpose of vector $\mathbf{x}$ |
| $\|\mathbf{x}\|_2$ | L2 (Euclidean) norm: $\sqrt{\sum_i x_i^2}$ |
| $\|\mathbf{x}\|_1$ | L1 norm: $\sum_i |x_i|$ |
| $\mathbf{x} \odot \mathbf{y}$ | Hadamard (element-wise) product |

### Probability

| Symbol | Meaning |
|--------|---------|
| $P(X)$ | Probability of random variable $X$ |
| $p(x)$ | Probability density (continuous) or mass (discrete) |
| $P(Y \mid X)$ | Conditional probability of $Y$ given $X$ |
| $\mathbb{E}[X]$ | Expectation: $\int x\, p(x)\, dx$ |
| $\text{Var}(X)$ | Variance: $\mathbb{E}[(X - \mu)^2]$ |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |
| $\text{KL}(P \| Q)$ | KL divergence: $\int p \log(p/q)\, dx$ |
| $\mathcal{H}(P)$ | Shannon entropy: $-\sum_x p(x) \log p(x)$ |

### Optimization

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Model parameters |
| $\hat{\theta}$ | Estimated / learned parameters |
| $\nabla_\theta \mathcal{L}$ | Gradient of loss w.r.t. $\theta$ |
| $\eta$ or $\alpha$ | Learning rate |
| $\lambda$ | Regularization coefficient |
| $\mathcal{L}$ | Loss function |
| $\mathcal{R}$ | Risk or regularization term |
| $\beta_1, \beta_2$ | Adam optimizer momentum coefficients |
| $\epsilon$ | Small constant (numerical stability) |

### Neural Networks

| Symbol | Meaning |
|--------|---------|
| $\sigma$ | Sigmoid activation: $1/(1 + e^{-z})$ |
| $\text{ReLU}(z)$ | $\max(0, z)$ |
| $\text{GELU}(z)$ | $z \cdot \Phi(z)$ where $\Phi$ is the Gaussian CDF |
| $\text{softmax}(\mathbf{z})_i$ | $e^{z_i} / \sum_j e^{z_j}$ |
| $L$ | Number of layers |
| $d$ or $d_\text{model}$ | Hidden dimension / model dimension |
| $d_k$ | Key/query dimension in attention |
| $H$ | Number of attention heads |
| $B$ | Batch size |
| $T$ or $n$ | Sequence length |
| $V$ | Vocabulary size |

### Transformers

| Symbol | Meaning |
|--------|---------|
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | Query, Key, Value matrices |
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ | Projection matrices for Q, K, V |
| $\mathbf{W}_O$ | Output projection in multi-head attention |
| $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ | $\text{softmax}(\mathbf{QK}^\top / \sqrt{d_k})\mathbf{V}$ |
| $\text{FFN}(\mathbf{x})$ | Feed-forward network: $\text{Linear} \to \text{GELU} \to \text{Linear}$ |
| $\text{LN}$ | Layer normalization |
| $r$ | Rank in LoRA decomposition |
| $\Delta \mathbf{W}$ | Weight update: $\mathbf{B}\mathbf{A}$ in LoRA |

### Evaluation Metrics

| Symbol / Metric | Meaning |
|----------------|---------|
| $\text{MSE}$ | Mean squared error: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| $\text{MAE}$ | Mean absolute error: $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |
| $R^2$ | Coefficient of determination |
| $\text{AUC-ROC}$ | Area under ROC curve |
| $F_1$ | $2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$ |
| $\text{BLEU}$ | Translation quality (n-gram precision) |
| $\text{ROUGE-L}$ | Summarization quality (longest common subsequence) |
| $\text{Perplexity}$ | $\exp(\text{cross-entropy loss})$ — lower is better |
| $\text{pass@}k$ | Coding benchmark: probability any of $k$ samples passes tests |

---

## Cheatsheets

### Choosing a Model

```
Problem type → Model family
──────────────────────────────────────────────────────────────
Tabular, <10M rows             → XGBoost / LightGBM
Tabular, many categoricals     → CatBoost
Images (limited data)          → Pretrained CNN / EfficientNet (fine-tune)
Images (large data)            → ViT / CLIP backbone
Text classification            → BERT / RoBERTa (fine-tune)
Text generation                → LLaMA / Mistral (fine-tune with LoRA)
Time series forecasting        → TFT, N-BEATS, or DeepAR
Graph data                     → GCN / GAT / GraphSAGE
Structured → Probabilistic     → Gaussian Process / BNN
```

### Loss Functions Quick Reference

| Task | Loss | Notes |
|------|------|-------|
| Binary classification | Binary cross-entropy | Use with sigmoid output |
| Multiclass classification | Categorical cross-entropy | Use raw logits (PyTorch `nn.CrossEntropyLoss`) |
| Regression | MSE / MAE | MAE more robust to outliers |
| Ordinal regression | Ordinal cross-entropy | |
| Ranking | Pairwise / ListNet | Learning-to-rank |
| Object detection | Focal loss + smooth L1 | Focal loss handles class imbalance |
| Segmentation | BCE + Dice | Dice handles small-region imbalance |
| Language modeling | Cross-entropy | Next-token prediction |
| Contrastive learning | InfoNCE / NT-Xent | SimCLR, CLIP |
| RLHF reward model | Pairwise ranking loss | Bradley-Terry model |

### Regularization Decision Tree

```
Overfitting?
├── Too few data       → Data augmentation, collect more
├── Model too complex  → Reduce capacity, pruning
├── Features           → L1 (Lasso) for sparsity; L2 (Ridge) for small weights
├── Deep networks      → Dropout (FC layers), DropBlock (conv), weight decay
└── Ensembles          → Bagging (reduce variance); stacking (reduce bias+variance)
```

### Evaluation Metric Selection

| Metric | Use when |
|--------|----------|
| Accuracy | Balanced classes, cost of errors is equal |
| F1 / PR-AUC | Imbalanced classes |
| AUC-ROC | Ranking quality across thresholds |
| Log-loss | Need calibrated probability scores |
| MSE | Regression; penalize large errors heavily |
| MAE | Regression; outlier-robust |
| MAPE | Forecasting with percentage-scale interpretation |
| sMAPE | Forecasting; symmetric, handles near-zero actuals |
| BLEU / ROUGE | NLP generation quality |
