# Specialized Domains

This folder covers eight specialized areas that appear in ML interviews — reinforcement learning, recommender systems, graph neural networks, computer vision, NLP, speech/audio, time series, and tabular deep learning. Each has its own interview circuit with domain-specific vocabulary and design questions. If you're preparing for a generalist ML role, skim the overview sections. If you're targeting a specialist role, read the full file.

---

## Subdomains

### Computer Vision

[computer-vision/README.md](computer-vision/README.md)

CV covers the full arc from CNNs to modern vision transformers: convolutional networks (AlexNet, VGG, ResNet residual connections, EfficientNet), object detection (Faster R-CNN, YOLO, FPN, IoU, NMS, focal loss), image segmentation (U-Net, DeepLab, Mask R-CNN, panoptic segmentation), Vision Transformers (ViT patch embeddings, positional encoding, DeiT, DINO, MAE), Swin Transformer (shifted window attention), CLIP (contrastive image-text pretraining), GANs, and diffusion models. 7 interview Q&As included.

**When it comes up:** Any team working on visual content — image search, video understanding, autonomous vehicles, medical imaging, content moderation with visual signals. ViT and CLIP are increasingly relevant for multimodal LLM roles.

---

### NLP

[nlp/README.md](nlp/README.md)

NLP covers the full progression from classical text representations to modern LLMs: pre-neural methods (Bag-of-Words, TF-IDF, n-gram language models, CRF), word embeddings (Word2Vec CBOW/skip-gram/negative sampling, GloVe, FastText), sequence models (RNN, LSTM, GRU, BiRNN, seq2seq), attention mechanisms (Bahdanau attention, ELMo), Transformer architecture (self-attention, multi-head attention, positional encoding, RoPE), BERT (MLM, NSP, fine-tuning variants), GPT (causal LM, in-context learning, InstructGPT), T5 and BART, and alignment methods (RLHF, DPO, Flash Attention). 7 interview Q&As included.

**When it comes up:** Any LLM or NLP role. TF-IDF and word embeddings appear in RecSys and search interviews. BERT fine-tuning is standard for classification/NER/QA roles. Transformer architecture and attention mechanics are tested at virtually every company doing applied NLP.

---

### Reinforcement Learning

[reinforcement-learning/README.md](reinforcement-learning/README.md)

RL covers sequential decision-making under uncertainty: agents, environments, policies, rewards, and value functions. The main file goes from first principles (MDPs, Bellman equations) through classic algorithms (Q-Learning, SARSA, Monte Carlo, TD) to modern deep RL (DQN, PPO, Actor-Critic) and lands on RLHF — the mechanism behind ChatGPT-style alignment. Multi-armed bandits cover exploration-exploitation tradeoffs. 20 interview Q&As and a full equation reference included.

Additional files:
- [advanced-rl.md](reinforcement-learning/advanced-rl.md): Imitation learning (BC, DAgger, GAIL), Inverse RL (MaxEntropy IRL), Multi-agent RL (QMIX, MAPPO, self-play), Hierarchical RL (Options, HER, Feudal RL), Meta-RL (MAML, RL²), Curriculum learning, Sim-to-real transfer, Offline RL (CQL, IQL, Decision Transformer).
- [model-based-rl.md](reinforcement-learning/model-based-rl.md): Model-free vs model-based tradeoffs, learned world models (Dyna, MCTS, DreamerV3, MuZero), connection to LLM reasoning (o1-style MCTS, Process Reward Models).

**When it comes up:** Alignment/RLHF roles, LLM fine-tuning positions, any team working on model alignment or instruction following. PPO and RLHF are table stakes for applied LLM roles. Also appears in robotics, game AI, and resource scheduling interviews.

---

### Speech and Audio

[speech-audio/README.md](speech-audio/README.md)

Speech/audio covers the signal processing foundations and end-to-end deep learning systems: audio signals and waveforms, STFT, mel spectrograms, MFCCs, classical HMM-DNN ASR, the alignment problem, CTC (Connectionist Temporal Classification), wav2vec 2.0, attention-based ASR, Conformer architecture, Whisper, speaker verification (d-vectors, x-vectors, ArcFace, EER), speaker diarization (pipeline and EEND), and text-to-speech (Tacotron 2, FastSpeech 2, WaveNet/HiFi-GAN vocoders, VALL-E). 5 interview Q&As included.

**When it comes up:** Voice assistant teams (ASR, TTS), speech analytics, any product processing audio content, accessibility tooling, and increasingly in multimodal LLM roles.

---

### Time Series

[time-series/README.md](time-series/README.md)

Time series covers both classical statistical methods and modern deep learning approaches: AR/MA/ARMA/ARIMA/SARIMA (with equations), exponential smoothing/Holt-Winters/ETS, Prophet (trend + seasonality + holidays), stationarity testing (ADF, KPSS), STL decomposition, LSTM forecasting, TCN (dilated causal convolutions), N-BEATS, Informer (ProbSparse attention), PatchTST (patch-based transformer), iTransformer, anomaly detection (z-score, Isolation Forest, CUSUM, autoencoder, Deep SVDD), evaluation metrics (MAE/RMSE/MAPE/sMAPE/MASE/CRPS), and multi-step forecasting strategies (DIRECT, RECURSIVE, MIMO). 5 interview Q&As included.

**When it comes up:** Demand forecasting, financial modeling, infrastructure capacity planning, IoT sensor data, anomaly detection in operations, any team whose data has a temporal index.

---

### Recommender Systems

[recommender-systems/README.md](recommender-systems/README.md)

RecSys covers the full production recommendation pipeline: collaborative filtering (user-based, item-based, matrix factorization, ALS, iALS for implicit feedback), content-based filtering, neural collaborative filtering (NCF), two-tower retrieval models, learning to rank (pointwise/pairwise/listwise, BPR, LambdaMART), the cold start problem, diversity and filter bubbles, offline/online evaluation metrics (NDCG, MAP, MRR, Precision/Recall@K), and production patterns including feature stores, caching, A/B testing, and monitoring. GNN-based recommenders (LightGCN, PinSage, KG-augmented) and session-based models (GRU4Rec, SASRec, BERT4Rec, SR-GNN) are also covered. 12 interview Q&As included.

**When it comes up:** Feed ranking roles at social/media companies, search ranking, any product with a discovery surface (e-commerce, streaming, news). The two-stage retrieval + ranking architecture is a near-universal design pattern — expect system design questions around it. Also appears in ads/monetization team interviews.

---

### Graph Neural Networks

[graph-neural-networks/README.md](graph-neural-networks/README.md)

GNNs cover the theory and practice of learning on graph-structured data: graph basics (adjacency matrix, degree, Laplacian), GCN (spectral derivation, message passing, normalization), GraphSAGE (inductive learning, neighborhood sampling, aggregators), Graph Attention Networks (GAT, GATv2), MPNN framework (messages, aggregation, update, readout), GIN (1-WL expressiveness), graph pooling (DiffPool, TopK, SAGPooling), knowledge graph embeddings (TransE, RotatE, DistMult, ComplEx), GNNs in LLM pipelines (GraphRAG, G-Retriever), and scalability at billion-node scale (PinSage, Cluster-GCN, SIGN, over-smoothing, over-squashing). 12 interview Q&As included.

**When it comes up:** Fraud detection (transaction graphs, device fingerprinting), drug discovery (molecular property prediction), knowledge graph reasoning, social network analysis. Also increasingly relevant for LLM retrieval and structured reasoning. Any team working on graph-structured data or entity relationship modeling.

---

### Tabular Data and Deep Learning

[tabular-data/README.md](tabular-data/README.md)

Tabular deep learning covers approaches for structured/tabular data: feature engineering fundamentals (encoding, normalization, missing values, target encoding), entity embeddings for categorical variables, gradient boosting (XGBoost, LightGBM, CatBoost — the production default for most tabular tasks), deep learning approaches (TabNet, FT-Transformer/SAINT, NODE), when neural networks beat tree ensembles (and when they don't), and production patterns for tabular ML pipelines.

**When it comes up:** Most real-world production ML uses tabular data — fraud detection, credit scoring, churn prediction, click-through rate, demand forecasting. Any data science or applied ML role will involve tabular data. Knowing when to use gradient boosting vs. neural networks is a standard interview question.

---

## Interview Priority by Role

| Topic | Generalist ML | Specialist LLM / Alignment | MLOps | CV / Multimodal | RecSys / Search |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RL basics** (MDP, Q-Learning, TD) | Low–Medium | High | Low | Low | Low |
| **PPO + RLHF** | Low | Critical | Low | Low | Low |
| **CV** (CNNs, ResNet, detection) | Low | Low | Low | Critical | Low |
| **ViT / CLIP** | Low | Medium | Low | Critical | Low |
| **NLP** (BERT, GPT, attention) | Medium | Critical | Low | Medium | Low |
| **Word embeddings** (Word2Vec, GloVe) | Medium | Medium | Low | Low | Medium |
| **Speech/Audio** (ASR, TTS) | Low | Low | Low | Low | Low |
| **Time Series** (ARIMA, LSTM, TCN) | Medium | Low | Medium | Low | Low |
| **RecSys (CF, MF)** | Medium | Low | Low | Low | Critical |
| **Two-tower / ranking** | Medium | Low | Low | Low | Critical |
| **GNN basics** (GCN, message passing) | Low | Low | Low | Low | Low |
| **Tabular ML** (XGBoost, embeddings) | High | Low | High | Low | Medium |
| **Exploration-exploitation / bandits** | Medium | Medium | Low | Low | Medium |
| **A/B testing + offline eval** | Medium | Low | High | Low | High |

**Generalist ML:** Focus on tabular ML (XGBoost/LightGBM dominates production), RecSys system design (two-stage pipeline, cold start, NDCG), RL fundamentals (Bellman equation, Q-Learning), and NLP basics (BERT fine-tuning, attention). Skip GNNs and Speech unless the JD mentions them.

**Specialist LLM / Alignment:** PPO and RLHF are mandatory — know the three-stage pipeline (SFT → reward model → PPO), the KL penalty, and DPO as the RL-free alternative. NLP depth is critical: transformer architecture, BERT vs GPT design choices, attention mechanics. Understand reward hacking and how KL mitigates it.

**MLOps:** Prioritize tabular ML pipelines (feature stores, training-serving skew, drift detection), RecSys production patterns (latency budgets, caching, A/B testing pitfalls), and time series monitoring. RL and GNNs are rarely tested in MLOps interviews.

**CV / Multimodal:** ResNet architecture and skip connections, object detection pipeline (anchors, NMS, IoU, focal loss), ViT vs CNN tradeoffs, CLIP contrastive training. Diffusion models are increasingly relevant.

**RecSys / Search:** Full two-stage pipeline design, matrix factorization and ALS, BPR and LambdaMART, cold start strategies, NDCG and offline-to-online metric correlation, feature stores. LightGCN and SASRec for graph/session-based variants.
