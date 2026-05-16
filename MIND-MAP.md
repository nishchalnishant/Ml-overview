# ML Overview — Mind Map

```
MACHINE LEARNING
│
├── 01 FOUNDATIONS
│   ├── AI Paradigms (narrow AI, AGI, symbolic vs sub-symbolic)
│   ├── ML vs DL vs AI taxonomy
│   ├── Bias–Variance Tradeoff
│   ├── No Free Lunch Theorem
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
│   │   ├── Encoding (one-hot, target, ordinal)
│   │   ├── Class Imbalance ← [NEW: 02-classical-ml/imbalanced-data.md]
│   │   │   ├── SMOTE, ADASYN, Tomek Links
│   │   │   ├── Focal Loss, Cost-Sensitive Learning
│   │   │   └── Threshold Moving, PR curves
│   │   ├── Feature Selection ← [NEW: 02-classical-ml/feature-selection.md]
│   │   │   ├── Filter (MI, chi², ANOVA F)
│   │   │   ├── Wrapper (RFE, sequential)
│   │   │   └── Embedded (Lasso, tree importance, SHAP)
│   │   ├── Ensemble Methods ← [NEW: 02-classical-ml/ensemble-methods.md]
│   │   │   ├── Bagging, Boosting, Voting
│   │   │   └── Stacking (OOF predictions)
│   │   ├── Cross-Validation ← [NEW: 02-classical-ml/cross-validation.md]
│   │   │   ├── Stratified, Group, Time Series CV
│   │   │   └── Nested CV (unbiased HP + perf estimation)
│   │   └── Hyperparameter Optimization ← [NEW: 02-classical-ml/hyperparameter-optimization.md]
│   │       ├── Bayesian Optimization (Optuna, TPE)
│   │       └── Hyperband, BOHB, PBT
│   │
│   ├── CALIBRATION & UNCERTAINTY
│   │   ├── Platt Scaling
│   │   ├── Isotonic Regression
│   │   ├── Temperature Scaling
│   │   └── Conformal Prediction ← [NEW: 02-classical-ml/conformal-prediction.md]
│   │
│   ├── ANOMALY DETECTION ← [NEW: 02-classical-ml/anomaly-detection.md]
│   │   ├── Isolation Forest
│   │   ├── One-Class SVM
│   │   ├── Local Outlier Factor
│   │   ├── Elliptic Envelope
│   │   └── Autoencoders for anomaly detection
│   │
│   ├── ACTIVE LEARNING ← [NEW: 02-classical-ml/active-learning.md]
│   │   ├── Uncertainty Sampling
│   │   ├── Query by Committee
│   │   ├── Expected Model Change
│   │   └── Core-Set Methods
│   │
│   └── BAYESIAN METHODS ← [NEW: 02-classical-ml/bayesian-methods.md]
│       ├── Bayes' Theorem in ML
│       ├── MAP vs MLE
│       ├── Gaussian Processes
│       ├── Bayesian Neural Networks
│       ├── Variational Inference
│       └── Probabilistic Programming (Pyro, NumPyro)
│
├── 03 DEEP LEARNING
│   ├── COMPONENTS (building blocks)
│   │   ├── Activation Functions (ReLU, GELU, Swish, sigmoid, tanh)
│   │   ├── Loss Functions (MSE, cross-entropy, focal, triplet, ranking)
│   │   ├── Optimizers (SGD, Adam, AdaGrad, LAMB, Lion)
│   │   ├── Regularization (Dropout, BatchNorm, LayerNorm, weight decay)
│   │   ├── Backpropagation (chain rule, vanishing/exploding gradients)
│   │   ├── Attention (scaled dot-product, multi-head, cross-attention)
│   │   └── Model Compression (quantization, distillation, pruning)
│   │
│   ├── ARCHITECTURES
│   │   ├── MLPs / Feedforward Networks
│   │   ├── CNNs (LeNet → ResNet → EfficientNet → ConvNeXt)
│   │   ├── RNNs, LSTMs, GRUs (sequence modeling)
│   │   ├── Transformers (full architecture)
│   │   ├── Autoencoders (AE, VAE, VQVAE)
│   │   ├── U-Net (encoder-decoder with skip connections)
│   │   └── Diffusion Models (DDPM, DDIM, Score Matching)
│   │
│   ├── COMPUTER VISION
│   │   ├── Image Classification (ResNet, ViT, EfficientNet)
│   │   ├── Object Detection (YOLO, Faster R-CNN, DETR)
│   │   ├── Segmentation ← [NEW: 03-deep-learning/methods/segmentation.md]
│   │   │   ├── Semantic (FCN, DeepLab, SegFormer)
│   │   │   ├── Instance (Mask R-CNN, SOLOv2, Mask2Former)
│   │   │   └── Panoptic (PQ metric, Panoptic FPN)
│   │   ├── Pose Estimation ← [NEW: 03-deep-learning/methods/segmentation.md]
│   │   │   ├── OpenPose (bottom-up, PAFs)
│   │   │   ├── HRNet (top-down, high-res representations)
│   │   │   └── ViTPose (Transformer-based)
│   │   ├── Metric Learning & Retrieval ← [NEW: 03-deep-learning/methods/metric-learning.md]
│   │   │   ├── Contrastive / Triplet / ArcFace Loss
│   │   │   ├── FAISS indexing, Recall@K
│   │   │   └── Person Re-ID
│   │   ├── Generative (GANs, Diffusion, FLUX)
│   │   ├── Self-Supervised (SimCLR, DINO, MAE)
│   │   ├── Video Understanding ← [NEW: 03-deep-learning/methods/video-understanding.md]
│   │   │   ├── 3D CNNs (C3D, I3D)
│   │   │   ├── Video Transformers (TimeSformer, VideoMAE)
│   │   │   └── Optical Flow
│   │   └── 3D Vision ← [NEW: 03-deep-learning/methods/3d-vision.md]
│   │       ├── Point Clouds (PointNet, PointNet++)
│   │       ├── Voxels
│   │       └── NeRF / 3D Gaussian Splatting
│   │
│   ├── NLP (Natural Language Processing)
│   │   ├── NLP Fundamentals ← [NEW: 03-deep-learning/methods/nlp-fundamentals.md]
│   │   │   ├── Tokenization (BPE, WordPiece, SentencePiece)
│   │   │   ├── Embeddings (Word2Vec, GloVe, FastText, ELMo, BERT)
│   │   │   ├── Seq2Seq + Attention, Beam Search
│   │   │   ├── NER (BIO tagging, BERT token classification)
│   │   │   └── MT Metrics (BLEU, ROUGE, BERTScore)
│   │   ├── NLP Advanced ← [NEW: 03-deep-learning/methods/nlp-advanced.md]
│   │   │   ├── Summarization (BART, TextRank, ROUGE)
│   │   │   ├── Semantic Similarity (SBERT)
│   │   │   ├── NLI / Zero-Shot Classification
│   │   │   ├── Coreference Resolution
│   │   │   └── Dependency Parsing, Relation Extraction
│   │   ├── Sequence Models (RNN, LSTM, GRU)
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
│   ├── TIME SERIES
│   │   ├── Classical (ARIMA, SARIMA, Holt-Winters)
│   │   ├── ML Models (Prophet, LightGBM on lag features)
│   │   ├── Deep Learning (LSTM, TCN, N-BEATS, PatchTST)
│   │   ├── Anomaly Detection (iForest, OCSVM, VAE)
│   │   └── Forecasting Systems Design
│   │
│   ├── TRANSFER LEARNING & DOMAIN ADAPTATION ← [NEW: 03-deep-learning/transfer-learning.md]
│   │   ├── Pre-training → Fine-tuning paradigm
│   │   ├── Feature Extraction vs Full Fine-tuning
│   │   ├── Domain Adaptation (DANN, CORAL)
│   │   ├── Few-Shot Learning (Prototypical Networks, MAML)
│   │   ├── Meta-Learning
│   │   └── Zero-Shot Learning
│   │
│   └── PYTORCH
│       ├── Tensors and Autograd
│       ├── Training Loops and DataLoaders
│       ├── Distributed Training (DDP, FSDP) ← [NEW coverage]
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
│   │   ├── Multi-Armed Bandits
│   │   ├── Exploration Strategies (ε-greedy, UCB, Thompson)
│   │   └── RLHF → LLM alignment
│   │
│   ├── RECOMMENDER SYSTEMS
│   │   ├── Collaborative Filtering (user-based, item-based, matrix factorization)
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
│       └── Dynamic Graphs & Graph Generation ← [NEW: 03-deep-learning/methods/dynamic-graphs.md]
│           ├── Temporal GNNs (TGN, TGAT, EvolveGCN)
│           ├── VGAE, GraphRNN, GRAN
│           └── Molecule Generation (Junction Tree VAE)
│
├── 05 LARGE LANGUAGE MODELS
│   ├── ARCHITECTURE
│   │   ├── Transformer (attention, FFN, positional encoding)
│   │   ├── Positional Encodings (RoPE, ALiBi, learned)
│   │   ├── Attention Variants (MHA, MQA, GQA, FlashAttention)
│   │   ├── Mixture of Experts (routing, load balancing)
│   │   ├── KV-Cache (memory/compute tradeoffs)
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
│   ├── EFFICIENT FINE-TUNING
│   │   ├── LoRA / QLoRA / DoRA
│   │   ├── PEFT Adapters
│   │   ├── Prompt Tuning / Prefix Tuning
│   │   └── Full Fine-tuning with FSDP
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
│   │   ├── Multimodal (CLIP, LLaVA, Flamingo, GPT-4V)
│   │   ├── Model Merging ← [NEW: 05-llms/applications/model-merging.md]
│   │   │   ├── Task Arithmetic, SLERP, TIES, DARE
│   │   │   └── Model Soup, MergeKit
│   │   ├── Hallucination Mitigation ← [NEW: 05-llms/applications/hallucination-mitigation.md]
│   │   │   ├── SelfCheckGPT, FactScore, NLI-checking
│   │   │   └── Calibration, Constrained Decoding
│   │   └── Context Window Extension ← [NEW: 05-llms/applications/context-window-extension.md]
│   │       ├── RoPE Scaling (YaRN, NTK), ALiBi
│   │       ├── Sparse Attention (Longformer, BigBird)
│   │       └── FlashAttention, Mamba/SSM alternatives
│   │
│   └── EVALUATION
│       ├── Benchmarks (MMLU, GSM8K, HumanEval, HELM)
│       ├── LLM-as-Judge
│       ├── Human Evaluation
│       └── Red-Teaming and Safety Evals
│
├── 06 PRODUCTION ML
│   ├── MLOPS PIPELINE
│   │   ├── Data Versioning (DVC, Delta Lake)
│   │   ├── Feature Stores (Feast, Tecton) ← [NEW coverage]
│   │   ├── Experiment Tracking (MLflow, W&B)
│   │   ├── CI/CD for ML (retraining triggers, model promotion)
│   │   ├── Model Registry
│   │   └── A/B Testing Infrastructure
│   │
│   ├── SERVING & DEPLOYMENT
│   │   ├── Serving Frameworks (TorchServe, BentoML, KServe) ← [NEW coverage]
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
│   ├── SYSTEM DESIGN
│   │   ├── Recommendation Systems
│   │   ├── Search Ranking
│   │   ├── Fraud Detection
│   │   ├── Content Moderation
│   │   └── NLP Services
│   │
│   ├── DEPLOYMENT PATTERNS ← [NEW: 06-production-ml/deployment-patterns.md]
│   │   ├── Canary / Blue-Green / Shadow Mode
│   │   ├── Latency Optimization (quantization, ONNX, TensorRT)
│   │   ├── Feature Store Design (Feast, online/offline consistency)
│   │   ├── Training-Serving Skew Detection
│   │   └── Model Rollback and Versioning
│   │
│   └── MODEL GOVERNANCE
│       ├── Model Cards and Documentation
│       ├── Audit Trails
│       ├── GDPR / Compliance
│       └── Champion-Challenger Testing
│
├── 07 MATHEMATICS & STATISTICS
│   ├── LINEAR ALGEBRA
│   │   ├── Vectors, matrices, tensors
│   │   ├── Eigenvalues/eigenvectors (PCA connection)
│   │   ├── SVD
│   │   └── Matrix calculus
│   │
│   ├── PROBABILITY & STATISTICS
│   │   ├── Distributions (Gaussian, Bernoulli, Categorical, Dirichlet)
│   │   ├── Bayes' Theorem
│   │   ├── MLE and MAP estimation
│   │   ├── Hypothesis Testing
│   │   ├── Confidence Intervals
│   │   └── A/B Testing (power, p-values, MDE)
│   │
│   ├── CALCULUS & OPTIMIZATION
│   │   ├── Gradient Descent (batch, SGD, mini-batch)
│   │   ├── Convexity and Local Minima
│   │   ├── Second-order Methods (Newton, L-BFGS)
│   │   ├── Constrained Optimization (Lagrangians, KKT)
│   │   └── Learning Rate Schedules (warmup, cosine, cyclic)
│   │
│   ├── INFORMATION THEORY
│   │   ├── Entropy, KL Divergence
│   │   ├── Mutual Information
│   │   ├── Cross-Entropy Loss connection
│   │   └── Rate-Distortion Theory
│   │
│   ├── PROBABILISTIC GRAPHICAL MODELS ← [NEW: 07-interview-prep/ml/probabilistic-graphical-models.md]
│   │   ├── Bayesian Networks, HMMs (Viterbi, Baum-Welch)
│   │   ├── MRFs, Belief Propagation (sum-product)
│   │   ├── Variational Inference (ELBO, mean-field, CAVI)
│   │   ├── MCMC (Gibbs, HMC, NUTS)
│   │   ├── LDA (topic modeling)
│   │   └── CRFs (sequence labeling)
│   │
│   └── OPTIMIZATION THEORY ← [NEW: 07-interview-prep/ml/optimization-theory.md]
│       ├── Variance Reduction (SVRG, SAGA)
│       ├── Natural Gradient & Information Geometry
│       ├── SAM (Sharpness-Aware Minimization)
│       └── Loss Landscape & NTK
│
├── 08 EMERGING TOPICS
│   ├── STATE SPACE MODELS
│   │   ├── S4, S6 (Structured SSMs)
│   │   ├── Mamba (selective SSMs)
│   │   └── Hybrid Mamba-Transformer
│   │
│   ├── TEST-TIME COMPUTE
│   │   ├── Chain-of-Thought generation
│   │   ├── Best-of-N sampling
│   │   ├── Monte Carlo Tree Search for LLMs
│   │   └── Process Reward Models (PRMs)
│   │
│   ├── MIXTURE OF EXPERTS
│   │   ├── Top-K routing
│   │   ├── Load balancing (aux loss)
│   │   ├── DeepSeek-MoE, Mixtral
│   │   └── Expert specialization
│   │
│   ├── INTERPRETABILITY & XAI
│   │   ├── Feature Importance (SHAP, LIME, permutation)
│   │   ├── Saliency Maps (GradCAM, Integrated Gradients)
│   │   ├── Mechanistic Interpretability (circuits, features)
│   │   ├── Concept-Based Explanations (TCAV)
│   │   └── Explainability in Production
│   │
│   ├── CAUSAL INFERENCE
│   │   ├── A/B Testing (design, power, pitfalls)
│   │   ├── Causal Graphs (DAGs, d-separation)
│   │   ├── Potential Outcomes Framework
│   │   ├── Difference-in-Differences
│   │   ├── Instrumental Variables
│   │   └── Uplift Modeling
│   │
│   ├── PRIVACY-PRESERVING ML ← [NEW: 08-emerging-topics/privacy-preserving-ml.md]
│   │   ├── Differential Privacy (ε-δ DP, DP-SGD)
│   │   ├── Federated Learning (FedAvg, FedProx)
│   │   ├── Secure Multi-Party Computation
│   │   ├── Homomorphic Encryption for ML
│   │   └── Membership Inference Attacks
│   │
│   ├── CONTINUOUS / ONLINE LEARNING ← [NEW: 08-emerging-topics/continual-learning.md]
│   │   ├── Catastrophic Forgetting
│   │   ├── Elastic Weight Consolidation (EWC)
│   │   ├── Progressive Neural Networks
│   │   ├── Replay Buffers / Experience Replay
│   │   └── Neural Architecture Search (NAS)
│   │
│   ├── ADVERSARIAL ROBUSTNESS ← [NEW: 08-emerging-topics/adversarial-robustness.md]
│   │   ├── FGSM, PGD, C&W, AutoAttack
│   │   ├── Adversarial Training (Madry, TRADES)
│   │   └── Certified Defenses (Randomized Smoothing)
│   │
│   └── FAIRNESS & BIAS ← [NEW: 08-emerging-topics/fairness-and-bias.md]
│       ├── Fairness Metrics (demographic parity, equalized odds)
│       ├── Impossibility Theorem
│       ├── Pre/In/Post-processing Mitigations
│       └── LLM Bias (WinoBias, RLHF amplification)
│
├── 09 EVALUATION & METRICS
│   ├── CLASSIFICATION
│   │   ├── Accuracy, Precision, Recall, F1
│   │   ├── ROC-AUC, PR-AUC
│   │   ├── Confusion Matrix analysis
│   │   └── Multi-class (macro, micro, weighted)
│   │
│   ├── REGRESSION
│   │   ├── MAE, MSE, RMSE, MAPE
│   │   └── R², Adjusted R²
│   │
│   ├── RANKING
│   │   ├── NDCG, MAP, MRR
│   │   └── Hit Rate, Recall@K
│   │
│   ├── GENERATION
│   │   ├── BLEU, ROUGE, METEOR (NLP)
│   │   ├── FID, IS (images)
│   │   └── Perplexity (LMs)
│   │
│   └── OFFLINE vs ONLINE EVALUATION
│       ├── Hold-out sets, K-fold CV
│       ├── Temporal splits for time series
│       ├── Shadow mode testing
│       └── Canary deployments
│
└── 10 AI SAFETY & ETHICS
    ├── ALIGNMENT
    │   ├── RLHF, DPO, Constitutional AI
    │   ├── Reward Hacking
    │   └── Mesa-Optimization
    ├── FAIRNESS
    │   ├── Group Fairness (demographic parity, equalized odds)
    │   ├── Individual Fairness
    │   └── Fairness-Accuracy tradeoff
    ├── ROBUSTNESS
    │   ├── Adversarial Examples (FGSM, PGD)
    │   ├── Distribution Shift
    │   └── Certified Defenses
    └── GOVERNANCE
        ├── EU AI Act
        ├── Model Cards
        └── Risk Classification
```

---

## Gap Analysis — Topics Added or Expanded

| # | Gap | New File Created |
|---|-----|------------------|
| 1 | Anomaly Detection (classical) | `02-classical-ml/anomaly-detection.md` |
| 2 | Active Learning | `02-classical-ml/active-learning.md` |
| 3 | Bayesian Methods | `02-classical-ml/bayesian-methods.md` |
| 4 | Conformal Prediction | `02-classical-ml/conformal-prediction.md` |
| 5 | Imbalanced Data (SMOTE, focal loss, threshold) | `02-classical-ml/imbalanced-data.md` |
| 6 | Feature Selection (filter/wrapper/embedded/SHAP) | `02-classical-ml/feature-selection.md` |
| 7 | Ensemble Methods (stacking, OOF, blending) | `02-classical-ml/ensemble-methods.md` |
| 8 | Cross-Validation (grouped, time-series, nested) | `02-classical-ml/cross-validation.md` |
| 9 | Hyperparameter Optimization (Optuna, Hyperband) | `02-classical-ml/hyperparameter-optimization.md` |
| 10 | Weight Initialization (Xavier, He, orthogonal) | `03-deep-learning/components/weight-initialization.md` |
| 11 | Normalization (BN/LN/GN/RMSNorm/SpectralNorm) | `03-deep-learning/components/normalization.md` |
| 12 | Transfer Learning & Domain Adaptation | `03-deep-learning/transfer-learning.md` |
| 13 | NLP Fundamentals (tokenization, embeddings, NER) | `03-deep-learning/methods/nlp-fundamentals.md` |
| 14 | NLP Advanced (summarization, SBERT, NLI, coref) | `03-deep-learning/methods/nlp-advanced.md` |
| 15 | Segmentation + Pose Estimation | `03-deep-learning/methods/segmentation.md` |
| 16 | Metric Learning & Image Retrieval | `03-deep-learning/methods/metric-learning.md` |
| 17 | Video Understanding | `03-deep-learning/methods/video-understanding.md` |
| 18 | 3D Vision & Point Clouds | `03-deep-learning/methods/3d-vision.md` |
| 19 | Dynamic Graphs & Graph Generation | `03-deep-learning/methods/dynamic-graphs.md` |
| 20 | Advanced RL (MARL, imitation, sim-to-real) | `04-specialized-domains/reinforcement-learning/advanced-rl.md` |
| 21 | Model Merging (SLERP, TIES, DARE, MergeKit) | `05-llms/applications/model-merging.md` |
| 22 | Hallucination Mitigation | `05-llms/applications/hallucination-mitigation.md` |
| 23 | Context Window Extension (YaRN, FlashAttention) | `05-llms/applications/context-window-extension.md` |
| 24 | Deployment Patterns (canary, shadow, feature store) | `06-production-ml/deployment-patterns.md` |
| 25 | Probabilistic Graphical Models (HMM, BP, LDA, CRF) | `07-interview-prep/ml/probabilistic-graphical-models.md` |
| 26 | Optimization Theory (SVRG, natural gradient, SAM) | `07-interview-prep/ml/optimization-theory.md` |
| 27 | Privacy-Preserving ML | `08-emerging-topics/privacy-preserving-ml.md` |
| 28 | Continual / Online Learning + NAS | `08-emerging-topics/continual-learning.md` |
| 29 | Adversarial Robustness (FGSM, PGD, certified) | `08-emerging-topics/adversarial-robustness.md` |
| 30 | Fairness & Bias (metrics, mitigations, LLM bias) | `08-emerging-topics/fairness-and-bias.md` |

---

## Navigation Guide

**By skill level:**
- Beginner → `01-foundations` → `09-study-plans/week-1`
- Intermediate → `02-classical-ml` → `03-deep-learning/components` → `07-interview-prep/ml`
- Advanced → `04-specialized-domains` → `05-llms` → `06-production-ml/system-design`
- Expert → `08-emerging-topics` → `05-llms/applications` → `10-references`

**By interview type:**
- Classical ML coding → `07-interview-prep/ml/algorithms.md` + `coding.md`
- ML system design → `06-production-ml/system-design/machine-learning-design-interview.md`
- LLM theory → `05-llms/architecture-deep-dive.md` + `interview-notes/llm-fundamentals.md`
- LLM system design → `05-llms/interview-notes/ai-system-design.md`
- Behavioral → `07-interview-prep/ml/behavioral-and-scenario-based-questions.md`

**By domain:**
- Vision → `03-deep-learning/methods/computer-vision.md`
- NLP → `03-deep-learning/methods/nlp.md` + `05-llms/`
- Time Series → `03-deep-learning/methods/time-series.md`
- RL → `04-specialized-domains/reinforcement-learning/`
- RecSys → `04-specialized-domains/recommender-systems/`
