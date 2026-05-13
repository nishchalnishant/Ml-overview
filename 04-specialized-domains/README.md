# Specialized Domains

This folder covers three areas that appear in ML interviews for specific roles — reinforcement learning, recommender systems, and graph neural networks. Each has its own interview circuit with domain-specific vocabulary and design questions. If you're preparing for a generalist ML role, skim the overview sections. If you're targeting a specialist role, read the full file.

---

## Subdomains

### Reinforcement Learning

[reinforcement-learning/README.md](reinforcement-learning/README.md)

RL covers sequential decision-making under uncertainty: agents, environments, policies, rewards, and value functions. The file goes from first principles (MDPs, Bellman equations) through classic algorithms (Q-Learning, SARSA, Monte Carlo, TD) to modern deep RL (DQN, PPO, Actor-Critic) and lands on RLHF — the mechanism behind ChatGPT-style alignment. Multi-armed bandits are included as a self-contained unit covering exploration-exploitation tradeoffs that appear in A/B testing and ad serving contexts. 20 interview Q&As and a full equation reference are included.

**When it comes up:** Alignment/RLHF roles, LLM fine-tuning positions, any team working on model alignment or instruction following. PPO and RLHF are table stakes for applied LLM roles. Also appears in robotics, game AI, and resource scheduling interviews.

---

### Recommender Systems

[recommender-systems/README.md](recommender-systems/README.md)

RecSys covers the full production recommendation pipeline: collaborative filtering (user-based, item-based, matrix factorization, ALS), content-based filtering, neural collaborative filtering (NCF), two-tower retrieval models, learning to rank (pointwise/pairwise/listwise, BPR, LambdaMART), the cold start problem, diversity and filter bubbles, offline/online evaluation metrics (NDCG, MAP, MRR, Precision/Recall@K), and production patterns including feature stores, caching, A/B testing, and monitoring. GNN-based recommenders (LightGCN, PinSage) and session-based models (GRU4Rec, SASRec, BERT4Rec) are also covered.

**When it comes up:** Feed ranking roles at social/media companies, search ranking, any product with a discovery surface (e-commerce, streaming, news). The two-stage retrieval + ranking architecture is a near-universal design pattern — expect system design questions around it. Also appears in ads/monetization team interviews.

---

### Graph Neural Networks

[graph-neural-networks/README.md](graph-neural-networks/README.md)

GNNs cover the theory and practice of learning on graph-structured data. Topics include graph basics (adjacency matrix, degree, Laplacian), GCN (spectral and spatial), GraphSAGE (inductive learning for unseen nodes), Graph Attention Networks (GAT), message passing neural networks (MPNN), graph pooling, knowledge graph embeddings (TransE, RotatE), scalability at billion-node scale (PinSage, mini-batch training), and how GNNs plug into LLM pipelines.

**When it comes up:** Fraud detection (transaction graphs, device fingerprinting), drug discovery (molecular property prediction), knowledge graph reasoning, social network analysis. Also increasingly relevant for LLM retrieval and structured reasoning. Any team working on graph-structured data or entity relationship modeling.

---

## Interview Priority by Role

| Topic | Generalist ML | Specialist LLM / Alignment | MLOps |
| :--- | :--- | :--- | :--- |
| **RL basics** (MDP, Q-Learning, TD) | Low–Medium | High | Low |
| **PPO + RLHF** | Low | Critical | Low |
| **RecSys (CF, MF)** | Medium | Low | Low |
| **Two-tower / ranking** | Medium | Low | Low |
| **GNN basics** (GCN, message passing) | Low | Low | Low |
| **GNN applications** (fraud, mol.) | Low | Low | Low |
| **Exploration-exploitation / bandits** | Medium | Medium | Low |
| **A/B testing + offline eval** | Medium | Low | High |

**Generalist ML:** Focus on RecSys system design (two-stage pipeline, cold start, NDCG) and RL fundamentals (Bellman equation, Q-Learning, bias-variance in RL). Skip GNNs unless the JD mentions graphs.

**Specialist LLM / Alignment:** PPO and RLHF are mandatory — know the three-stage pipeline (SFT → reward model → PPO), the KL penalty, and DPO as the RL-free alternative. Understand reward hacking and how the KL term mitigates it.

**MLOps:** Prioritize RecSys production patterns — feature stores, training-serving skew, A/B testing pitfalls, drift detection, latency budgets. RL and GNNs are rarely tested in MLOps interviews.
