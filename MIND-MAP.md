---
module: ml-overview
topic: Mind Map
subtopic: ""
status: unread
tags: [mloverviewroot, ml, mind-map]
---
# ML Overview — Mind Map

```
MACHINE LEARNING
│
├── 01 FOUNDATIONS
│   ├── AI Paradigms (narrow AI, AGI, symbolic vs sub-symbolic)
│   ├── ML vs DL vs AI taxonomy
│   ├── Bias–Variance Tradeoff
│   ├── No Free Lunch Theorem
│   ├── Linear Algebra & Numerical Methods
│   └── Glossary (key terms)
│
├── 02 CLASSICAL ML
│   ├── SUPERVISED LEARNING
│   │   ├── Regression
│   │   │   ├── Linear Regression (OLS, Ridge, Lasso, ElasticNet)
│   │   │   └── Polynomial Regression
│   │   ├── Classification
│   │   │   ├── Logistic Regression
│   │   │   ├── SVM (kernel trick, margin, slack)
│   │   │   ├── Naive Bayes
│   │   │   └── k-NN
│   │   └── Tree Methods
│   │       ├── Decision Trees (CART, ID3, C4.5)
│   │       ├── Random Forest
│   │       ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│   │       └── AdaBoost
│   │
│   ├── UNSUPERVISED LEARNING
│   │   ├── Clustering
│   │   │   ├── K-Means, K-Medoids
│   │   │   ├── DBSCAN, HDBSCAN
│   │   │   ├── Hierarchical (Ward, complete, average)
│   │   │   └── Gaussian Mixture Models (EM algorithm)
│   │   ├── Dimensionality Reduction
│   │   │   ├── PCA, Kernel PCA
│   │   │   ├── t-SNE, UMAP
│   │   │   ├── LDA (Linear Discriminant Analysis)
│   │   │   └── Autoencoders
│   │   └── Density Estimation
│   │       └── KDE, Parzen Windows
│   │
│   ├── SEMI-SUPERVISED LEARNING
│   │   ├── Label Propagation
│   │   ├── Self-Training
│   │   └── Co-Training
│   │
│   ├── DATA PREPROCESSING
│   │   ├── Feature Engineering
│   │   ├── Scaling (standard, min-max, robust)
│   │   ├── Imputation (mean, median, KNN, MICE)
│   │   └── Encoding (one-hot, target, ordinal)
│   │
│   ├── CLASS IMBALANCE
│   │   ├── SMOTE, ADASYN, Tomek Links
│   │   ├── Focal Loss, Cost-Sensitive Learning
│   │   └── Threshold Moving, PR curves
│   │
│   ├── FEATURE SELECTION
│   │   ├── Filter (MI, chi², ANOVA F)
│   │   ├── Wrapper (RFE, sequential)
│   │   └── Embedded (Lasso, tree importance, SHAP)
│   │
│   ├── ENSEMBLE METHODS
│   │   ├── Bagging, Boosting, Voting
│   │   └── Stacking (OOF predictions)
│   │
│   ├── CROSS-VALIDATION
│   │   ├── Stratified, Group, Time Series CV
│   │   └── Nested CV (unbiased HP + perf estimation)
│   │
│   ├── HYPERPARAMETER OPTIMIZATION
│   │   ├── Bayesian Optimization (Optuna, TPE)
│   │   └── Hyperband, BOHB, PBT
│   │
│   ├── CALIBRATION & UNCERTAINTY
│   │   ├── Platt Scaling
│   │   ├── Isotonic Regression
│   │   ├── Temperature Scaling
│   │   └── Conformal Prediction
│   │
│   ├── ANOMALY DETECTION
│   │   ├── Isolation Forest
│   │   ├── One-Class SVM
│   │   ├── Local Outlier Factor
│   │   ├── Elliptic Envelope
│   │   └── Autoencoders for anomaly detection
│   │
│   ├── ACTIVE LEARNING
│   │   ├── Uncertainty Sampling
│   │   ├── Query by Committee
│   │   ├── Expected Model Change
│   │   └── Core-Set Methods
│   │
│   ├── BAYESIAN METHODS
│   │   ├── Bayes' Theorem in ML
│   │   ├── MAP vs MLE
│   │   ├── Gaussian Processes
│   │   ├── Bayesian Neural Networks
│   │   ├── Variational Inference
│   │   └── Probabilistic Programming (Pyro, NumPyro)
│   │
│   ├── ONLINE LEARNING
│   │   ├── Perceptron, Passive-Aggressive
│   │   ├── Online Gradient Descent
│   │   └── Streaming algorithms
│   │
│   ├── TIME SERIES ANALYSIS
│   │   ├── ARIMA, SARIMA, Holt-Winters
│   │   ├── Stationarity, ACF/PACF
│   │   └── Granger Causality
│   │
│   ├── MODEL INTERPRETATION
│   │   ├── SHAP, LIME
│   │   ├── Partial Dependence Plots
│   │   └── Global vs Local explanations
│   │
│   └── WHEN CLASSICAL ML WINS
│       ├── Tabular data heuristics
│       └── Compute / data constraints
│
├── 03 DEEP LEARNING
│   ├── COMPONENTS (building blocks)
│   │   ├── Activation Functions (ReLU, GELU, Swish, sigmoid, tanh)
│   │   ├── Loss Functions (MSE, cross-entropy, focal, triplet, ranking)
│   │   ├── Optimizers (SGD, Adam, AdaGrad, LAMB, Lion)
│   │   ├── Regularization (Dropout, BatchNorm, LayerNorm, weight decay)
│   │   ├── Normalization (BN/LN/GN/RMSNorm/SpectralNorm)
│   │   ├── Weight Initialization (Xavier, He, orthogonal)
│   │   ├── Backpropagation (chain rule, vanishing/exploding gradients)
│   │   ├── Hidden Layers (depth, width, capacity)
│   │   ├── Attention (scaled dot-product, multi-head, cross-attention)
│   │   ├── Model Compression (quantization, distillation, pruning)
│   │   ├── Quantization & Pruning (PTQ, QAT, structured/unstructured)
│   │   ├── Scaling Laws & Chinchilla
│   │   ├── Distributed Training & Parallelism (DDP, FSDP, tensor/pipeline)
│   │   ├── Instruction Tuning & Alignment
│   │   └── RNN / LSTM / GRU (sequence modeling)
│   │
│   ├── ARCHITECTURES
│   │   ├── MLPs / Feedforward Networks
│   │   ├── CNNs (LeNet → ResNet → EfficientNet → ConvNeXt)
│   │   ├── RNNs, LSTMs, GRUs
│   │   ├── Transformers (full architecture)
│   │   ├── Autoencoders (AE, VAE, VQVAE)
│   │   ├── U-Net (encoder-decoder with skip connections)
│   │   └── Diffusion Models (DDPM, DDIM, Score Matching)
│   │
│   ├── COMPUTER VISION
│   │   ├── Image Classification (ResNet, ViT, EfficientNet)
│   │   ├── Object Detection (YOLO, Faster R-CNN, DETR)
│   │   ├── Segmentation
│   │   │   ├── Semantic (FCN, DeepLab, SegFormer)
│   │   │   ├── Instance (Mask R-CNN, SOLOv2, Mask2Former)
│   │   │   └── Panoptic (PQ metric, Panoptic FPN)
│   │   ├── Pose Estimation
│   │   │   ├── OpenPose (bottom-up, PAFs)
│   │   │   ├── HRNet (top-down, high-res representations)
│   │   │   └── ViTPose (Transformer-based)
│   │   ├── Metric Learning & Retrieval
│   │   │   ├── Contrastive / Triplet / ArcFace Loss
│   │   │   ├── FAISS indexing, Recall@K
│   │   │   └── Person Re-ID
│   │   ├── Generative (GANs, Diffusion, FLUX)
│   │   ├── Self-Supervised (SimCLR, DINO, MAE)
│   │   ├── Video Understanding
│   │   │   ├── 3D CNNs (C3D, I3D)
│   │   │   ├── Video Transformers (TimeSformer, VideoMAE)
│   │   │   └── Optical Flow
│   │   └── 3D Vision
│   │       ├── Point Clouds (PointNet, PointNet++)
│   │       ├── Voxels
│   │       └── NeRF / 3D Gaussian Splatting
│   │
│   ├── NLP (Natural Language Processing)
│   │   ├── NLP Fundamentals
│   │   │   ├── Tokenization (BPE, WordPiece, SentencePiece)
│   │   │   ├── Embeddings (Word2Vec, GloVe, FastText, ELMo, BERT)
│   │   │   ├── Seq2Seq + Attention, Beam Search
│   │   │   ├── NER (BIO tagging, BERT token classification)
│   │   │   └── MT Metrics (BLEU, ROUGE, BERTScore)
│   │   ├── NLP Advanced
│   │   │   ├── Summarization (BART, TextRank, ROUGE)
│   │   │   ├── Semantic Similarity (SBERT)
│   │   │   ├── NLI / Zero-Shot Classification
│   │   │   ├── Coreference Resolution
│   │   │   └── Dependency Parsing, Relation Extraction
│   │   ├── Transformers (BERT, GPT, T5, RoBERTa)
│   │   └── Machine Translation
│   │
│   ├── GENERATIVE MODELS
│   │   ├── VAEs (reparameterization trick, ELBO)
│   │   ├── GANs (vanilla, WGAN, StyleGAN, CycleGAN)
│   │   ├── Diffusion Models (DDPM, DDIM, LDM)
│   │   ├── Flow Models (Normalizing Flows, FLUX)
│   │   └── Score-Based Models
│   │
│   ├── TIME SERIES (deep learning)
│   │   ├── LSTM, TCN, N-BEATS, PatchTST
│   │   ├── Anomaly Detection (iForest, OCSVM, VAE)
│   │   └── Forecasting Systems Design
│   │
│   ├── TRANSFER LEARNING & DOMAIN ADAPTATION
│   │   ├── Pre-training → Fine-tuning paradigm
│   │   ├── Feature Extraction vs Full Fine-tuning
│   │   ├── Domain Adaptation (DANN, CORAL)
│   │   ├── Few-Shot Learning (Prototypical Networks, MAML)
│   │   ├── Meta-Learning
│   │   └── Zero-Shot Learning
│   │
│   ├── MCP (Model Context Protocol)
│   │
│   └── PYTORCH
│       ├── Foundations (tensors, autograd)
│       ├── Training Loops and DataLoaders
│       ├── Distributed Training (DDP, FSDP)
│       ├── Mixed Precision (AMP)
│       └── Model Serialization and Export
│
├── 04 SPECIALIZED DOMAINS
│   ├── REINFORCEMENT LEARNING
│   │   ├── MDPs (states, actions, rewards, transitions)
│   │   ├── Bellman Equations
│   │   ├── Model-Free (Q-learning, SARSA, DQN, DDQN)
│   │   ├── Policy Gradient (REINFORCE, PPO, A3C, SAC)
│   │   ├── Model-Based RL
│   │   ├── Advanced RL (MARL, imitation learning, sim-to-real)
│   │   ├── Multi-Armed Bandits
│   │   ├── Exploration Strategies (ε-greedy, UCB, Thompson)
│   │   └── RLHF → LLM alignment
│   │
│   ├── RECOMMENDER SYSTEMS
│   │   ├── Collaborative Filtering (user-based, item-based, MF)
│   │   ├── Content-Based Filtering
│   │   ├── Hybrid Systems
│   │   ├── Two-Tower Models
│   │   ├── Learning-to-Rank (pointwise, pairwise, listwise)
│   │   ├── Cold-Start Problem
│   │   ├── Session-Based Models (GRU4Rec, BERT4Rec)
│   │   └── GNN-Based RecSys (PinSage, LightGCN)
│   │
│   └── GRAPH NEURAL NETWORKS
│       ├── Graph Basics (nodes, edges, adjacency)
│       ├── Spectral Methods (GCN, ChebNet)
│       ├── Spatial Methods (GraphSAGE, GAT, GIN)
│       ├── Message Passing Framework
│       ├── Knowledge Graphs (TransE, RotatE)
│       ├── Heterogeneous Graphs
│       ├── Scalability (mini-batch sampling, GraphSAINT)
│       └── Dynamic Graphs & Graph Generation
│           ├── Temporal GNNs (TGN, TGAT, EvolveGCN)
│           ├── VGAE, GraphRNN, GRAN
│           └── Molecule Generation (Junction Tree VAE)
│
├── 05 LARGE LANGUAGE MODELS
│   ├── ARCHITECTURE
│   │   ├── Transformer (attention, FFN, positional encoding)
│   │   ├── Positional Encodings (RoPE, ALiBi, learned)
│   │   ├── Attention Variants (MHA, MQA, GQA, FlashAttention)
│   │   ├── KV-Cache & MQA/GQA (memory/compute tradeoffs)
│   │   ├── Mixture of Experts (routing, load balancing)
│   │   └── Long Context (ring attention, streaming)
│   │
│   ├── TRAINING
│   │   ├── Pre-training (next-token prediction, masked LM)
│   │   ├── Supervised Fine-Tuning (SFT)
│   │   ├── RLHF (PPO-based alignment)
│   │   ├── DPO / ORPO / SimPO (offline preference optimization)
│   │   ├── Scaling Laws (Chinchilla optimal)
│   │   ├── Data Quality and Synthetic Data
│   │   └── Training Stability (loss spikes, mixed precision)
│   │
│   ├── TOKENIZATION
│   │   ├── BPE, WordPiece, SentencePiece
│   │   └── Vocabulary design tradeoffs
│   │
│   ├── EFFICIENT FINE-TUNING
│   │   ├── LoRA / QLoRA / DoRA
│   │   ├── PEFT Adapters
│   │   ├── Prompt Tuning / Prefix Tuning
│   │   └── Full Fine-tuning at Scale (FSDP)
│   │
│   ├── INFERENCE OPTIMIZATION
│   │   ├── Quantization (INT8, INT4, GPTQ, AWQ)
│   │   ├── KV-Cache Management
│   │   ├── Continuous Batching (vLLM, TGI)
│   │   ├── Speculative Decoding (Medusa, Eagle, EAGLE-2)
│   │   └── Model Parallelism (tensor, pipeline, sequence)
│   │
│   ├── APPLICATIONS
│   │   ├── RAG (retrieval-augmented generation)
│   │   ├── Agentic Workflows (tool use, planning, memory)
│   │   ├── Prompt Engineering (CoT, few-shot, ReAct)
│   │   ├── Prompt Optimization & Versioning
│   │   ├── Multimodal (CLIP, LLaVA, Flamingo, GPT-4V)
│   │   ├── Model Merging (SLERP, TIES, DARE, MergeKit)
│   │   ├── Hallucination Mitigation
│   │   │   ├── SelfCheckGPT, FactScore, NLI-checking
│   │   │   └── Calibration, Constrained Decoding
│   │   └── Context Window Extension
│   │       ├── RoPE Scaling (YaRN, NTK), ALiBi
│   │       ├── Sparse Attention (Longformer, BigBird)
│   │       └── FlashAttention, Mamba/SSM alternatives
│   │
│   ├── EVALUATION
│   │   ├── Benchmarks (MMLU, GSM8K, HumanEval, HELM)
│   │   ├── LLM-as-Judge
│   │   ├── Human Evaluation
│   │   └── Red-Teaming and Safety Evals
│   │
│   ├── MoE ADVANCED & ROUTING
│   │   ├── Expert routing strategies
│   │   └── DeepSeek-MoE, Mixtral internals
│   │
│   ├── ARCHITECTURE DEEP DIVE
│   │   └── Layer-by-layer dissection, attention patterns
│   │
│   ├── SCALING & DATA
│   │   ├── Data pipelines, deduplication, quality filters
│   │   └── Synthetic data generation
│   │
│   ├── BOOKS
│   │   ├── Hand-On Large Language Models (Alammar)
│   │   └── Agentic Design Patterns
│   │
│   └── INTERVIEW NOTES
│       ├── LLM Fundamentals
│       ├── Fine-Tuning & Model Adaptation
│       ├── Efficient LLM Deployment
│       ├── RAG (detailed)
│       ├── Prompt Engineering
│       ├── Vector Databases & Embeddings
│       ├── AI Agents & Agentic Systems
│       ├── Multi-Modal AI
│       ├── AI System Design
│       ├── AI Infrastructure & Scalability
│       ├── LLMOps & Production AI
│       ├── Evaluation & Testing
│       ├── Advanced Alignment & Reasoning
│       ├── AI Safety, Ethics & Responsible AI
│       ├── Behavioral & Scenario-Based Questions
│       ├── Coding & Practical Implementation
│       ├── Additional LLM Interview Topics
│       └── Production Alignment Failures
│
├── 06 PRODUCTION ML
│   ├── MLOPS PIPELINE
│   │   ├── Data Versioning (DVC, Delta Lake)
│   │   ├── Feature Stores (Feast, Tecton)
│   │   ├── Experiment Tracking (MLflow, W&B)
│   │   ├── CI/CD for ML (retraining triggers, model promotion)
│   │   ├── Model Registry & Versioning
│   │   └── A/B Testing Infrastructure
│   │
│   ├── SERVING & DEPLOYMENT
│   │   ├── Serving Frameworks (TorchServe, BentoML, KServe)
│   │   ├── REST vs gRPC APIs
│   │   ├── Batching (offline, online, micro-batch)
│   │   ├── Edge Deployment (TFLite, CoreML, ONNX)
│   │   └── Containerization (Docker, Kubernetes for ML)
│   │
│   ├── MONITORING
│   │   ├── Data Drift (covariate shift, concept drift)
│   │   ├── Model Drift (performance degradation)
│   │   ├── Metrics Logging and Dashboards
│   │   └── Alerting and Retraining Triggers
│   │
│   ├── DEPLOYMENT PATTERNS
│   │   ├── Canary / Blue-Green / Shadow Mode
│   │   ├── Latency Optimization (quantization, ONNX, TensorRT)
│   │   ├── Feature Store Design (online/offline consistency)
│   │   ├── Training-Serving Skew Detection
│   │   └── Model Rollback and Versioning
│   │
│   ├── MODEL GOVERNANCE
│   │   ├── Model Cards and Documentation
│   │   ├── Audit Trails
│   │   ├── GDPR / Compliance
│   │   └── Champion-Challenger Testing
│   │
│   └── SYSTEM DESIGN CASE STUDIES
│       ├── Ad CTR Prediction
│       ├── End-to-End Recommendation System
│       ├── Video Recommendation System
│       ├── News Feed Ranking
│       ├── Search Ranking System
│       ├── Personalization System
│       ├── Fraud Detection (full system)
│       ├── Content Moderation System
│       ├── Real-Time Bidding System
│       ├── Real-Time ML Systems
│       ├── Streaming ML Pipeline
│       ├── Embedding Pipeline Design
│       ├── Feature Store Architecture & Advanced
│       ├── Distributed Training
│       ├── LLM Inference Ops
│       ├── ETA & Surge Pricing
│       ├── Customer LTV Prediction
│       ├── Clinical ML System
│       ├── A/B Testing & Experimentation
│       ├── Cost Optimization
│       ├── Data Engineering for ML
│       ├── Production Diagnosis Flowchart
│       ├── Model Registry & Versioning
│       ├── ML Design Patterns
│       ├── ML Design Interview (framework)
│       └── Books (ML Engineering, Building ML-Powered Apps)
│
├── 07 INTERVIEW PREP
│   ├── ML INTERVIEWS
│   │   ├── Algorithms
│   │   ├── Fundamentals of ML
│   │   ├── Deep Learning
│   │   ├── NLP
│   │   ├── Computer Vision
│   │   ├── Large Language Models
│   │   ├── Optimization
│   │   ├── Optimization Theory
│   │   ├── Probabilistic Graphical Models
│   │   ├── Model Evaluation
│   │   ├── Data Preprocessing & Feature Engineering
│   │   ├── Probability & Statistics
│   │   ├── Maths & Math Derivations
│   │   ├── Coding
│   │   ├── Practical ML Scenarios
│   │   ├── Privacy & Fairness
│   │   ├── System Design & MLOps
│   │   ├── Behavioral & Scenario-Based Questions
│   │   └── Additional ML Interview Topics
│   │
│   └── LLM INTERVIEWS
│       ├── ML Revision (rapid review)
│       ├── DL Architectures
│       ├── NLP & Transformers
│       ├── ML System Design
│       ├── ML Coding Patterns
│       ├── Math Derivations
│       ├── Statistics & Probability
│       ├── Scenario-Based Questions
│       ├── Top ML Interview Questions
│       └── Machine Learning Interviews (book notes)
│
├── 08 EMERGING TOPICS
│   ├── STATE SPACE MODELS
│   │   ├── S4, S6 (Structured SSMs)
│   │   ├── Mamba (selective SSMs)
│   │   └── Hybrid Mamba-Transformer
│   │
│   ├── LARGE REASONING MODELS
│   │   ├── Chain-of-Thought generation
│   │   ├── Best-of-N sampling
│   │   ├── Monte Carlo Tree Search for LLMs
│   │   └── Process Reward Models (PRMs)
│   │
│   ├── MIXTURE OF EXPERTS (emerging)
│   │   ├── Top-K routing
│   │   ├── Load balancing (aux loss)
│   │   ├── DeepSeek-MoE, Mixtral
│   │   └── Expert specialization
│   │
│   ├── MULTIMODAL ARCHITECTURES
│   │   ├── Vision-Language Models
│   │   ├── Audio-Text fusion
│   │   └── Unified multimodal training
│   │
│   ├── AGENTIC AI SYSTEMS
│   │   ├── Tool use, planning, memory
│   │   ├── Multi-agent coordination
│   │   └── Autonomous agent evaluation
│   │
│   ├── ADVANCED RAG & MEMORY
│   │   ├── GraphRAG, HyDE, FLARE
│   │   ├── Long-term memory systems
│   │   └── Retrieval evaluation
│   │
│   ├── POST-TRAINING & ALIGNMENT
│   │   ├── DPO, GRPO, Constitutional AI
│   │   ├── Reward modeling
│   │   └── Alignment tax
│   │
│   ├── 2025 FRONTIER MODELS
│   │   ├── GPT-4o, Claude 3.x, Gemini 1.5/2.0
│   │   ├── DeepSeek-R1, Llama 3
│   │   └── Frontier AI Developments 2025
│   │
│   ├── SMALL LANGUAGE MODELS & EDGE
│   │   ├── Phi-3, Gemma, Mistral-7B
│   │   └── On-device inference
│   │
│   ├── VECTOR DATABASES
│   │   ├── FAISS, Pinecone, Weaviate, Qdrant
│   │   └── ANN algorithms (HNSW, IVF)
│   │
│   ├── AGI & ASI
│   │   ├── Definitions and timelines debate
│   │   └── Safety implications
│   │
│   ├── INTERPRETABILITY & XAI
│   │   ├── Feature Importance (SHAP, LIME, permutation)
│   │   ├── Saliency Maps (GradCAM, Integrated Gradients)
│   │   ├── Mechanistic Interpretability (circuits, features)
│   │   ├── Concept-Based Explanations (TCAV)
│   │   └── Explainability in Production
│   │
│   ├── CAUSAL INFERENCE & EXPERIMENTATION
│   │   ├── A/B Testing (design, power, pitfalls)
│   │   ├── Causal Graphs (DAGs, d-separation)
│   │   ├── Potential Outcomes Framework
│   │   ├── Difference-in-Differences
│   │   ├── Instrumental Variables
│   │   └── Uplift Modeling
│   │
│   ├── PRIVACY-PRESERVING ML
│   │   ├── Differential Privacy (ε-δ DP, DP-SGD)
│   │   ├── Federated Learning (FedAvg, FedProx)
│   │   ├── Secure Multi-Party Computation
│   │   ├── Homomorphic Encryption for ML
│   │   └── Membership Inference Attacks
│   │
│   ├── CONTINUAL / ONLINE LEARNING
│   │   ├── Catastrophic Forgetting
│   │   ├── Elastic Weight Consolidation (EWC)
│   │   ├── Progressive Neural Networks
│   │   ├── Replay Buffers / Experience Replay
│   │   └── Neural Architecture Search (NAS)
│   │
│   ├── ADVERSARIAL ROBUSTNESS
│   │   ├── FGSM, PGD, C&W, AutoAttack
│   │   ├── Adversarial Training (Madry, TRADES)
│   │   └── Certified Defenses (Randomized Smoothing)
│   │
│   ├── RED TEAMING & ALIGNMENT FAILURES
│   │   ├── Jailbreaks, prompt injection
│   │   └── Production failure case studies
│   │
│   └── FAIRNESS & BIAS
│       ├── Fairness Metrics (demographic parity, equalized odds)
│       ├── Impossibility Theorem
│       ├── Pre/In/Post-processing Mitigations
│       └── LLM Bias (WinoBias, RLHF amplification)
│
├── 09 STUDY PLANS
│   ├── WEEK 1 — FOUNDATIONS
│   │   ├── Day 1–2: Introduction to Machine Learning
│   │   ├── Day 3–4: Data Preprocessing Techniques
│   │   └── Day 5–7: Exploratory Data Analysis (EDA)
│   │
│   ├── WEEK 2 — ALGORITHMS
│   │   ├── Day 8–9: Supervised Learning Algorithms
│   │   ├── Day 10–11: Unsupervised Learning Algorithms
│   │   ├── Day 12–14: Neural Networks
│   │   ├── Day 15–16: Evaluation Metrics
│   │   ├── Day 17–18: Hyperparameter Tuning
│   │   └── Day 19–21: Specialized Techniques (NLP, CV)
│   │
│   ├── WEEK 3 — SYSTEM DESIGN
│   │   ├── Day 22: ML System Design
│   │   ├── Day 23: Case Studies
│   │   └── Day 24–25: Behavioral / Soft Skills
│   │
│   └── WEEK 4 — FINAL PREP
│       └── Day 26–30: Final Prep & Mock Interviews
│
├── 10 REFERENCES
│   ├── BOOK NOTES
│   │   ├── Deep Learning
│   │   │   ├── Alice in Differentiable Wonderland
│   │   │   ├── Build a Large Language Model from Scratch
│   │   │   ├── Deep Learning: A Practitioner's Approach
│   │   │   ├── Deep Learning with PyTorch
│   │   │   ├── Dive into Deep Learning
│   │   │   └── Grokking Deep Learning
│   │   ├── Machine Learning
│   │   │   └── Machine Learning Pocket Reference
│   │   └── MLOps
│   │       ├── Designing Machine Learning Systems
│   │       ├── Keras to Kubernetes
│   │       ├── Machine Learning Design Patterns
│   │       └── Machine Learning Engineering
│   │
│   └── RESEARCH PAPERS
│       ├── ML (classical)
│       ├── MLOps
│       └── Deep Learning
│           ├── Computer Vision
│           ├── LLMs
│           └── Time Series
│
└── 11 DATA SCIENTIST
    ├── Statistics & Probability
    ├── Experiment Design & A/B Testing
    ├── Causal Inference
    ├── EDA & Data Quality
    ├── Metrics & Business Analytics
    ├── SQL & Data Manipulation
    └── Data Scientist Interview Prep
```

---

## Navigation Guide

**By skill level:**
- Beginner → `01-foundations` → `09-study-plans/week-1`
- Intermediate → `02-classical-ml` → `03-deep-learning/components` → `07-interview-prep/ml`
- Advanced → `04-specialized-domains` → `05-llms` → `06-production-ml/system-design`
- Expert → `08-emerging-topics` → `05-llms/applications` → `10-references`

**By interview type:**
- Classical ML coding → `07-interview-prep/ml/08-algorithms.md` + `coding.md`
- ML system design → `06-production-ml/system-design/03-machine-learning-design-interview.md`
- LLM theory → `05-llms/03-architecture-deep-dive.md` + `05-llms/interview-notes/01-llm-fundamentals.md`
- LLM system design → `05-llms/interview-notes/07-ai-system-design.md`
- Data scientist → `11-data-scientist/01-data-scientist-interview-prep.md`
- Behavioral → `07-interview-prep/ml/17-behavioral-and-scenario-based-questions.md`

**By domain:**
- Vision → `03-deep-learning/methods/03-computer-vision.md`
- NLP → `03-deep-learning/methods/nlp.md` + `05-llms/`
- Time Series → `02-classical-ml/20-time-series-analysis.md` + `03-deep-learning/methods/09-time-series.md`
- RL → `04-specialized-domains/reinforcement-learning/`
- RecSys → `04-specialized-domains/recommender-systems/`
- GNNs → `04-specialized-domains/graph-neural-networks/`
- Emerging → `08-emerging-topics/emerging-trends/`

## Flashcards

**Beginner → 01-foundations → 09-study-plans/week-1?** #flashcard
Beginner → 01-foundations → 09-study-plans/week-1

**Intermediate → 02-classical-ml → 03-deep-learning/components → 07-interview-prep/ml?** #flashcard
Intermediate → 02-classical-ml → 03-deep-learning/components → 07-interview-prep/ml

**Advanced → 04-specialized-domains → 05-llms → 06-production-ml/system-design?** #flashcard
Advanced → 04-specialized-domains → 05-llms → 06-production-ml/system-design

**Expert → 08-emerging-topics → 05-llms/applications → 10-references?** #flashcard
Expert → 08-emerging-topics → 05-llms/applications → 10-references

**Classical ML coding → 07-interview-prep/ml/08-algorithms.md + coding.md?** #flashcard
Classical ML coding → 07-interview-prep/ml/08-algorithms.md + coding.md

**ML system design → 06-production-ml/system-design/03-machine-learning-design-interview.md?** #flashcard
ML system design → 06-production-ml/system-design/03-machine-learning-design-interview.md

**LLM theory → 05-llms/03-architecture-deep-dive.md + 05-llms/interview-notes/01-llm-fundamentals.md?** #flashcard
LLM theory → 05-llms/03-architecture-deep-dive.md + 05-llms/interview-notes/01-llm-fundamentals.md

**LLM system design → 05-llms/interview-notes/07-ai-system-design.md?** #flashcard
LLM system design → 05-llms/interview-notes/07-ai-system-design.md

**Data scientist → 11-data-scientist/01-data-scientist-interview-prep.md?** #flashcard
Data scientist → 11-data-scientist/01-data-scientist-interview-prep.md

**Behavioral → 07-interview-prep/ml/17-behavioral-and-scenario-based-questions.md?** #flashcard
Behavioral → 07-interview-prep/ml/17-behavioral-and-scenario-based-questions.md

**Vision → 03-deep-learning/methods/03-computer-vision.md?** #flashcard
Vision → 03-deep-learning/methods/03-computer-vision.md

**NLP → 03-deep-learning/methods/nlp.md + 05-llms/?** #flashcard
NLP → 03-deep-learning/methods/nlp.md + 05-llms/

**Time Series → 02-classical-ml/20-time-series-analysis.md + 03-deep-learning/methods/09-time-series.md?** #flashcard
Time Series → 02-classical-ml/20-time-series-analysis.md + 03-deep-learning/methods/09-time-series.md

**RL → 04-specialized-domains/reinforcement-learning/?** #flashcard
RL → 04-specialized-domains/reinforcement-learning/

**RecSys → 04-specialized-domains/recommender-systems/?** #flashcard
RecSys → 04-specialized-domains/recommender-systems/

**GNNs → 04-specialized-domains/graph-neural-networks/?** #flashcard
GNNs → 04-specialized-domains/graph-neural-networks/

**Emerging → 08-emerging-topics/emerging-trends/?** #flashcard
Emerging → 08-emerging-topics/emerging-trends/
