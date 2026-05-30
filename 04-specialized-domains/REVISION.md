---
module: Specialized Domains
topic: Revision Card
subtopic: ""
status: unread
tags: [rl, recsys, gnn, cv, nlp, speech, timeseries, tabular, revision, cheatsheet]
---
# Specialized Domains — 10-Minute Revision Card

Eight domains. One card. CV → NLP → Speech → Time Series → RL → RecSys → GNN → Tabular.

---

## Part 1: Computer Vision

### Mental Model

CNNs: translation-equivariant feature extractors via local receptive fields + weight sharing. Stack conv→pool → grow receptive field while reducing spatial size. Residual connections (ResNet) let gradients skip layers, enabling 100+ layer networks. Vision Transformers (ViT) abandon convolutions entirely: split image into patches, project to embeddings, run transformer encoder.

**When CNNs beat ViT:** small datasets, limited compute. **When ViT beats CNNs:** large-scale pretraining, long-range spatial dependencies, CLIP-style multimodal tasks.

---

### Architecture Selector

| Situation | Use |
|-----------|-----|
| Image classification, limited data | ResNet-50 / EfficientNet-B4 |
| Object detection, speed-critical | YOLOv8 |
| Object detection, accuracy-critical | Faster R-CNN + FPN |
| Instance segmentation | Mask R-CNN |
| Image-text matching / zero-shot | CLIP |
| Self-supervised pretraining | MAE or DINO |
| Dense prediction (medical imaging) | U-Net |

---

### ResNet Residual Connection

$$y = F(x, \{W_i\}) + x$$

Skip connection adds the identity shortcut. Why it works: gradients flow directly back through the shortcut, avoiding vanishing gradient even in 150-layer networks. The shortcut also initializes the block as identity (F→0), making very deep nets easier to train.

---

### Object Detection Fast Reference

**Two-stage (Faster R-CNN):** Region Proposal Network → RoI Pooling → classification + regression. Accurate, slower (~5 FPS).

**One-stage (YOLO):** Single forward pass predicts all boxes. Faster (≥30 FPS), slightly less accurate on small objects.

**FPN (Feature Pyramid Network):** Multi-scale feature maps — detect large objects at coarse scale, small objects at fine scale. Standard component in all modern detectors.

**IoU** = intersection / union of predicted and ground-truth boxes. Threshold for positive: ≥0.5.

**NMS (Non-Maximum Suppression):** After detection, many overlapping boxes for same object. NMS keeps the highest-confidence box and suppresses overlapping boxes with IoU > threshold.

**Focal Loss** (RetinaNet): addresses foreground/background class imbalance. Downweights easy negatives:
$$FL(p_t) = -(1-p_t)^\gamma \log(p_t)$$
γ=2 standard. Easy backgrounds get ≈0 loss; hard foregrounds get full loss.

---

### ViT in 30 Seconds

Split 224×224 image into 16×16 patches → 196 patches. Project each patch: (H×W×C) → d_model. Prepend [CLS] token. Add learnable positional embeddings. Feed to standard transformer encoder. Use [CLS] output for classification.

**DeiT:** ViT training trick — knowledge distillation from CNN teacher. Makes ViT competitive without massive datasets.

**DINO:** Self-supervised ViT with momentum encoder. Learns semantic features without labels.

**MAE (Masked Autoencoder):** Mask 75% of patches, reconstruct. Learns rich visual representations at low compute cost.

---

### CLIP in 30 Seconds

Train image encoder and text encoder jointly: maximize cosine similarity for (image, caption) pairs, minimize for all other pairs in the batch (contrastive loss / InfoNCE).

At inference: embed image, embed text query "a photo of a cat", compute similarity → zero-shot classification without any class-specific training.

$$\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$$

---

### CV Interview Quick-Draws

**"Why does ResNet work so much better than plain deep nets?"**
→ Skip connections let gradients bypass layers entirely. Initializing F≈0 means the block starts as identity, making depth free to add without degrading training dynamics.

**"YOLO vs Faster R-CNN — when to use each?"**
→ YOLO: real-time inference, edge deployment, speed matters. Faster R-CNN: accuracy matters, small objects, offline processing.

**"What is focal loss and why does it matter for detection?"**
→ Class imbalance: 99% of anchor boxes are background. Focal loss downweights the easy negatives so the model focuses gradient on hard foreground/background ambiguities.

**"ViT vs CNN for your task?"**
→ CNN: small dataset, limited GPU. ViT: large dataset or pretrained, need global context, multimodal. CLIP: zero-shot or image-text matching.

---

## Part 2: NLP

### Mental Model

Progression: words → fixed bag-of-words → dense embeddings (Word2Vec) → contextual embeddings (ELMo, LSTM) → attention (seq2seq) → full self-attention (Transformer) → pretrained large models (BERT, GPT).

The key insight at each step: richer representation of word meaning in context.

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Text classification, small dataset | TF-IDF + logistic regression |
| Semantic similarity, sentence-level | SBERT (Sentence-BERT) |
| Classification/NER/QA (fine-tuning) | BERT or RoBERTa |
| Text generation | GPT-2/3/4 |
| Translation / summarization | T5 or BART |
| In-context learning | GPT-4 / Claude |
| Long documents | Longformer or BigBird |

---

### Word Embeddings Fast Reference

**TF-IDF:** Weight words by frequency in document × rarity across corpus. No semantic understanding; but strong baseline for keyword-heavy tasks.

**Word2Vec (skip-gram):** Predict context words from center word. Train with negative sampling (SGNS). Words with similar contexts → nearby vectors. "king − man + woman ≈ queen."

**GloVe:** Factorize word co-occurrence matrix. GloVe = global corpus statistics; Word2Vec = local context windows. Performance similar in practice.

**FastText:** Word2Vec over character n-grams. Handles OOV words and morphological variants.

---

### Transformer Attention in 30 Seconds

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Q, K, V are linear projections of the input. The √d_k scaling prevents softmax saturation as d_k grows.

**Multi-head:** Run h attention heads in parallel with different projections; concatenate outputs. Different heads learn different relationship types (syntax, coreference, etc.).

**Positional encoding:** Transformer has no recurrence → inject position via sinusoidal encoding (absolute) or RoPE (rotary, relative, used in LLaMA/GPT-4).

---

### BERT vs GPT

| | BERT | GPT |
|--|------|-----|
| Objective | MLM (masked language modeling) | CLM (causal/next-token prediction) |
| Direction | Bidirectional | Left-to-right (causal) |
| Pretraining task | Predict masked tokens | Predict next token |
| Fine-tuning | Classification, NER, QA | Generation, few-shot |
| Architecture | Encoder-only | Decoder-only |
| Key use case | Understanding tasks | Generation tasks |

**BERT pretraining:** Mask 15% of tokens; predict them. Also Next Sentence Prediction (NSP). Fine-tune by adding a task-specific head (linear layer) and training the full model.

**GPT pretraining:** Left-to-right language modeling. No masking. At inference, generate autoregressively. In-context learning: the prompt itself is the "fine-tuning."

---

### NLP Interview Quick-Draws

**"Why does self-attention capture long-range dependencies better than LSTMs?"**
→ LSTM: information must pass through every hidden state between two positions — O(n) steps, vanishing gradient. Self-attention: direct path from any token to any other token in one layer — O(1) steps. But O(n²) memory.

**"What is the attention bottleneck?"**
→ Standard attention is O(n²) in sequence length. Flash Attention computes the same result with O(n) memory via tiling/recomputation. Allows 32K+ context lengths.

**"BERT or GPT for classification?"**
→ BERT: bidirectional context → better for understanding tasks (classification, NER, QA). GPT: if you also need generation, or if the problem can be framed as few-shot prompting.

**"What is RoPE and why is it better than sinusoidal PE?"**
→ RoPE encodes relative position by rotating query and key vectors. Benefits: (1) dot products naturally capture relative positions, (2) generalizes better to lengths beyond training, (3) no learned position parameters.

---

## Part 3: Speech and Audio

### Mental Model

Audio = 1D waveform sampled at 16kHz. Raw waveform → STFT → mel spectrogram → log-mel → MFCCs (optional compression). The mel spectrogram is the "image" that deep learning models operate on.

ASR pipeline: audio → features → acoustic model → language model → text. End-to-end models (CTC, attention-based) replaced the classical three-component pipeline.

---

### Audio Feature Ladder

| Feature | Description | Use |
|---------|-------------|-----|
| Waveform | Raw samples at 16kHz | Wav2Vec 2.0, WaveNet |
| STFT | Frequency × time, complex | Intermediate |
| Mel spectrogram | Log-frequency bins (mel scale) | ASR, classification |
| Log-mel | Log of mel spectrogram | Standard ASR input |
| MFCCs | DCT of log-mel, 13-80 coefficients | Classical ASR, speaker ID |

---

### CTC in 30 Seconds

Alignment problem: audio is 100 frames, transcript is 10 characters — no explicit alignment known. CTC marginalizes over all possible alignments.

CTC introduces a blank token `_`. Any path that collapses to the right sequence (by removing blanks and duplicate consecutive chars) is valid. Loss = −log of the sum over all valid paths (computed efficiently with forward-backward algorithm).

At decoding: greedy (take argmax at each frame, then collapse) or beam search with language model.

**Key property:** CTC output is conditionally independent — probability of each output token depends only on the input, not other outputs. Limits modeling of inter-token dependencies.

---

### ASR Architecture Selector

| Situation | Use |
|-----------|-----|
| Offline transcription, accuracy | Whisper (large-v3) |
| Real-time streaming | CTC-based (e.g., wav2vec 2.0 + CTC) |
| Low-resource language | wav2vec 2.0 fine-tuning |
| Speaker verification | x-vectors + PLDA or ArcFace |
| TTS, natural prosody | FastSpeech 2 + HiFi-GAN |

---

### Whisper in 30 Seconds

Encoder-decoder transformer. Input: 80-channel log-mel spectrogram, 30-second chunks. Encoder: CNN stem + transformer encoder. Decoder: autoregressive transformer decoder.

Multitask: transcription, translation, language ID, timestamp prediction — all via special tokens in the decoder prompt. Training: 680k hours of weakly-supervised web data.

Key insight: scale + diverse supervised data beats self-supervised pretraining for ASR. Zero-shot multilingual ASR with no per-language fine-tuning.

---

### Speech Interview Quick-Draws

**"Why mel scale instead of linear frequency?"**
→ Human hearing is approximately logarithmic in frequency (mel scale). The mel scale compresses high frequencies where humans have less resolution, matching perceptual sensitivity. Models learn features aligned with what matters perceptually.

**"CTC vs attention-based decoder — tradeoffs?"**
→ CTC: conditionally independent outputs → fast streaming, simple decoding. Attention: models inter-token dependencies → better accuracy, full sequence required. Conformer + hybrid CTC/attention is the modern standard (CTC for fast prefix search, attention for reranking).

**"What is speaker diarization?"**
→ "Who spoke when" — segment audio by speaker identity without knowing who the speakers are. Pipeline: VAD (detect speech vs silence) → segmentation → speaker embedding per segment → clustering. EEND (End-to-End Neural Diarization) does this jointly.

---

## Part 4: Time Series

### Mental Model

Time series data has three special properties: temporal ordering (cannot shuffle), temporal dependencies (autocorrelation), and non-stationarity (statistics change over time). These require different data handling (no random train/test split — use temporal split), different models, and different evaluation metrics than standard supervised learning.

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Single series, linear trends/seasonality | SARIMA or Prophet |
| Many series, short history | Prophet or LightGBM with lag features |
| Many series, enough data | PatchTST or N-BEATS |
| Long-range dependencies | Informer or PatchTST |
| Few-shot / zero-shot | Chronos (pretrained TS transformer) |
| Anomaly detection | Isolation Forest (point) or LSTM autoencoder (contextual) |

---

### ARIMA in 30 Seconds

$$\text{ARIMA}(p, d, q): \quad \phi(B)(1-B)^d y_t = c + \theta(B)\epsilon_t$$

- **AR(p):** regress on past p values
- **I(d):** difference d times to achieve stationarity
- **MA(q):** regress on past q forecast errors

**Seasonality → SARIMA(p,d,q)(P,D,Q)[s]:** adds seasonal AR, I, MA components.

**Stationarity check:** ADF test (H₀: unit root exists = non-stationary). If p > 0.05, difference the series.

---

### Forecasting Model Fast Reference

| Model | Key Idea | Best When |
|-------|----------|-----------|
| ARIMA/SARIMA | Linear AR + MA + differencing | Univariate, linear trends |
| Prophet | Trend + Fourier seasonality + holidays | Business time series with trend changes |
| LSTM | Sequence-to-sequence with memory | Complex non-linear, long history |
| TCN | Dilated causal convolutions | Fast training, parallelizable |
| N-BEATS | Doubly residual stacks, trend/seasonality blocks | M4/M5 competition winner |
| PatchTST | Patch-based ViT for time series | Long-range, multivariate |
| iTransformer | Transposes attention: attends across variates | Multivariate forecasting |

---

### Evaluation Metrics Fast Reference

| Metric | Formula | Note |
|--------|---------|------|
| MAE | mean|y - ŷ| | Scale-dependent |
| MAPE | mean|y - ŷ|/|y| × 100 | Blows up near zero |
| sMAPE | mean 2|y-ŷ|/(|y|+|ŷ|) | Symmetric, bounded |
| MASE | MAE / MAE_naive | Scale-free, best for cross-series comparison |
| CRPS | Probabilistic — proper scoring rule | Use for distributional forecasts |

**Gotcha:** Always use temporal train/val/test split — never random. Walk-forward validation for hyperparameter tuning.

---

### Time Series Interview Quick-Draws

**"ARIMA vs Prophet — when to use which?"**
→ ARIMA: single series, linear, no missing data, need interpretable AR/MA decomposition. Prophet: business context with trend changepoints, multiple seasonalities, holidays, robust to missing data, handles non-linear trends.

**"Why can't you shuffle time series data for cross-validation?"**
→ Temporal leakage: shuffling uses future information to predict the past. The model sees future values during training, inflating validation performance. Use walk-forward validation or a strict temporal split.

**"How do you handle non-stationarity?"**
→ Check with ADF/KPSS tests. If non-stationary: (1) differencing (removes trends), (2) log transform (stabilizes variance), (3) seasonal differencing (removes seasonality). Neural models can sometimes handle non-stationarity with instance normalization.

**"Point anomaly vs contextual anomaly?"**
→ Point: a single value is far from the global distribution (z-score, Isolation Forest). Contextual: a value is normal globally but anomalous in its local context — e.g., 20°C in December in Oslo (LSTM autoencoder detects this).

---

## Part 5: Reinforcement Learning

### Mental Model

Agent takes actions → environment returns reward → agent updates policy to maximize cumulative reward. No labeled examples; only outcome signals.

**When RL fits:** game playing, robotics, LLM alignment (RLHF). **When it doesn't:** you have labels → use supervised learning.

---

### MDP in 30 Seconds

$$\text{MDP} = (S, A, P, R, \gamma) \qquad \text{Return } G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

**Markov property:** future depends only on current state, not history.

**Bellman optimality:**
$$Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s,a\right]$$

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Discrete actions, large state | DQN |
| Continuous actions, stability matters | PPO |
| Continuous actions, sample efficiency | SAC |
| LLM alignment | PPO + RLHF or DPO |
| Small tabular MDP | Q-learning |
| Bandit (no state transitions) | UCB / Thompson Sampling |

---

### DQN Essentials

**Two innovations that make it work:**

1. **Experience replay** — random mini-batches from buffer break temporal correlation
2. **Target network** — frozen copy $\theta^-$ updated every N steps; prevents chasing a moving target

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]$$

**Gotcha:** DQN only works for discrete actions. For continuous → PPO or SAC.

---

### Policy Gradient Essentials

**The theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]$$

Increase log-prob of actions with positive advantage; decrease for negative.

**Advantage:** $A(s,a) = Q(s,a) - V(s)$ — how much better than average?

**GAE:** $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. At $\lambda=0$: 1-step TD. At $\lambda=1$: MC. Standard PPO: $\lambda=0.95$.

---

### PPO Clipped Objective

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

$r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ — importance ratio.

**Intuition:** if action was good, increase its prob — but not by more than $1+\epsilon$. Prevents catastrophic large updates.

**Why PPO is default for RLHF:** stable, on-policy, handles large token action spaces.

---

### RLHF vs DPO

| | RLHF | DPO |
|--|------|-----|
| Reward model | Explicit (trained separately) | Implicit (in policy ratio) |
| RL loop | Yes (PPO) | No |
| Complexity | High | Low |
| Failure mode | Reward hacking | Distribution shift from offline data |

**DPO loss:**
$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

---

### RL Interview Quick-Draws

**"On-policy vs off-policy?"**
→ On-policy (PPO, A2C): learn about the current policy — data gets stale when policy changes. Off-policy (DQN, SAC): can reuse old data — enables experience replay.

**"Why does DQN need a target network?"**
→ Without it, both prediction and target move simultaneously — chasing a moving target → training oscillates. Frozen $\theta^-$ stabilizes the target for N steps.

**"What is the credit assignment problem?"**
→ With delayed sparse rewards, which of the 40 moves in a chess game caused the win? Solutions: discounting, eligibility traces, advantage estimation.

**"Reward hacking?"**
→ Agent finds unintended behaviors that score high on the reward proxy but not on true intent. RLHF example: verbose but unhelpful responses score high. Fix: KL penalty vs reference policy.

---

## Part 6: Recommender Systems

### Mental Model

Interaction matrix $R$ (users × items) is 99%+ sparse. Goal: fill in blanks. Two signals: **behavioral** (CF — who liked what) and **content** (CB — what items are like).

**Production pipeline:**
```
All items (100M) → [Two-Tower ANN] → 500 candidates
  → [Ranking model (DNN/LightGBM)] → Top 50
  → [Re-ranking: diversity, rules] → Final 10-20
```

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Dense interaction data | Matrix Factorization (ALS) |
| Large-scale retrieval | Two-Tower + ANN (FAISS) |
| Ranking over candidates | LambdaMART / DNN ranker |
| Cold new item | Content-based (feature tower) |
| Sequential / session | SASRec or GRU4Rec |
| Graph-structured interactions | LightGCN |

---

### Matrix Factorization

$$\hat{r}(u,i) = \mu + b_u + b_i + p_u^\top q_i$$

Minimize over observed entries: $\sum_{(u,i)\text{ obs.}} (\hat{r}_{ui} - r_{ui})^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$

**Implicit feedback (iALS):** confidence $c_{ui} = 1 + \alpha f_{ui}$. All (user, item) pairs used; zero-interactions get low confidence.

**ALS:** fix $Q$ → closed-form solve $P$; fix $P$ → solve $Q$. Parallelizable, no learning rate.

---

### Two-Tower Model

```
User features → User Tower → user_emb (d-dim)
                                    ↘ dot product → score
Item features → Item Tower → item_emb (d-dim)
```

**Why it scales:** item embeddings precomputed offline. At query time: one user forward pass + ANN lookup = milliseconds over 100M items.

**Training:** in-batch negatives — other items in the batch serve as negatives.

$$\mathcal{L} = \text{CrossEntropy}\!\left(\frac{\text{logits}}{\tau},\ \text{diagonal labels}\right)$$

**Sampling bias fix:** subtract $\log(p_i)$ from item scores where $p_i$ = sampling probability of item $i$.

---

### Ranking Paradigms

| Approach | Loss | Optimizes |
|----------|------|-----------|
| Pointwise | MSE/BCE per item | Absolute relevance |
| Pairwise (BPR) | $-\log\sigma(\hat{r}_{ui} - \hat{r}_{uj})$ | Relative order |
| Listwise (LambdaMART) | $\lambda_{ij} \propto \|\Delta\text{NDCG}\|$ | Full list quality |

**BPR gotcha:** treats all pairs equally — swapping ranks 1↔2 same weight as 50↔51. LambdaRank fixes this with $|\Delta\text{NDCG}|$ weighting.

---

### Evaluation Metrics Fast Reference

| Metric | Formula | Gotcha |
|--------|---------|--------|
| Precision@K | $\|\text{rel} \cap \text{top-K}\| / K$ | Doesn't penalize missing relevant items |
| Recall@K | $\|\text{rel} \cap \text{top-K}\| / \|\text{rel}\|$ | Treats all positions equally |
| NDCG@K | $\text{DCG} / \text{IDCG}$, DCG $= \sum (2^{r_i}-1)/\log_2(i+1)$ | Best: position-weighted |
| MAP | Mean AP over users | Penalizes both rank and recall |
| HR@K | Fraction of users with ≥1 hit | Simple, good for sparse data |

**Gotcha:** offline NDCG improvements don't reliably translate to online CTR. Always A/B test.

---

### Cold Start

| Problem | Fix |
|---------|-----|
| New user | Onboarding survey, popular fallback, exploration slate |
| New item | Content-based feature tower (works day-1), warm-up injection |

**Feature-based item tower:** item tower takes features (not IDs) → new item gets embedding immediately from its content.

---

### RecSys Interview Quick-Draws

**"CF vs content-based?"**
→ CF: behavioral patterns, needs history, enables serendipity. CB: item features, works for cold items, overspecializes. Production: always hybrid.

**"How do two-tower models scale to 100M items?"**
→ User and item only interact at the dot product. Precompute all item embeddings offline. At serving: 1 forward pass + ANN lookup.

**"Popularity bias?"**
→ Popular items get more training signal → recommended more → even more popular. Fix: sampling correction $(-\log p_i)$, IPS weighting, diversity constraints, exploration slots.

**"How do you detect recommendation degradation?"**
→ Monitor CTR trend, NDCG on held-out set, feature drift (PSI), catalog coverage dropping.

---

## Part 7: Graph Neural Networks

### Mental Model

Node embeddings = function of node features + neighborhood features. Stack $k$ layers → see $k$-hop neighborhood. Constraint: must be **permutation invariant** (reordering neighbors doesn't change result).

**Core message passing:**
```
For each layer k, for each node v:
  messages  = aggregate({h_u^(k-1) : u ∈ N(v)})
  h_v^(k)   = update(h_v^(k-1), messages)
```

---

### Architecture Selector

| Situation | Use |
|-----------|-----|
| Node classification (transductive) | GCN |
| Node classification (inductive, new nodes) | GraphSAGE |
| Heterogeneous neighbor importance | GAT |
| Maximum expressiveness (1-WL) | GIN |
| Recommendation at scale | LightGCN |
| Billion-node graphs | PinSage (random walk sampling) |
| Molecular property prediction | MPNN / NNConv (edge features) |
| Knowledge graph completion | TransE / RotatE |

---

### GCN

$$H^{(k+1)} = \sigma\!\left(\tilde{A} H^{(k)} W^{(k)}\right)$$

$\tilde{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$ — symmetric normalized adjacency with self-loops.

**Self-loops** (+I): include node's own features in update.  
**Normalization** ($D^{-1/2}$): prevent high-degree hubs from dominating.

**Gotcha:** GCN is transductive — can't generalize to new nodes.

---

### GraphSAGE vs GCN vs GAT

| | GCN | GraphSAGE | GAT |
|--|-----|-----------|-----|
| Inductive | No | Yes | Yes |
| Aggregation | Fixed normalized mean | Learned (mean/max/LSTM) | Learned attention |
| New nodes | Can't | Can | Can |
| Edge features | No | No | Partial |
| Memory | $O(|E|d)$ | $O(\text{sample} \times d)$ | $O(|E| \times \text{heads} \times d)$ |

**GAT attention:**
$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(a^\top [Wh_v \| Wh_u]))}{\sum_{w \in N(v)} \exp(\ldots)}$$

GAT = transformer attention restricted to the adjacency structure.

---

### GIN — Maximum Expressiveness

GIN achieves the maximum discriminative power of any message-passing GNN (= 1-WL test):

$$h_v^{(k)} = \text{MLP}^{(k)}\!\left((1+\epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\right)$$

**Why sum > mean > max:** sum preserves neighborhood size information — mean conflates a node with 2 neighbors [A,B] with a node with 100 neighbors having the same mean.

---

### Over-Smoothing — The Core Failure Mode

After too many layers, all node representations converge to the same vector (random walk mixing).

**Symptom:** accuracy peaks at 2–3 layers, drops with more.

**Fixes:**
- Residual connections: $h_v^{(k)} = h_v^{(k-1)} + \text{AGG}(\ldots)$
- Initial residual (APPNP): $h_v^{(k)} = \alpha h_v^{(0)} + (1-\alpha)\text{AGG}(\ldots)$
- JK-Net: aggregate across all layers, not just the last

---

### Knowledge Graph Embeddings

**TransE:** $h + r \approx t$, score $= -\|h+r-t\|$. Fails on 1-to-N relations and symmetric relations.

**RotatE:** $t = h \odot r$ in complex space, $|r_i|=1$. Handles symmetry, antisymmetry, inversion, composition.

| Relation pattern | TransE | RotatE |
|-----------------|--------|--------|
| Symmetric (A↔B) | ❌ | ✓ (rotate by π) |
| 1-to-N | ❌ | ✓ |
| Composition | Partial | ✓ |

---

### Scalability

| Technique | Idea | Tradeoff |
|-----------|------|---------|
| Neighbor sampling (GraphSAGE) | Fix $k$ neighbors per layer | Gradient variance |
| Cluster-GCN | Mini-batch by graph partition | Cross-cluster info lost |
| SIGN | Precompute $A^1X, A^2X$... offline; train MLP | No online graph traversal |
| PinSage | Random walk neighbor importance | Approximate, memory-efficient |

**Neighbor explosion:** $k$ layers, avg degree $d$ → $d^k$ nodes per training example. Sampling is mandatory at $k \geq 2$.

---

### GNN Interview Quick-Draws

**"GCN vs GraphSAGE?"**
→ GCN is transductive (bakes adjacency into a fixed matrix, can't handle new nodes). GraphSAGE learns an aggregation function that generalizes — sample neighbors, concatenate own state + aggregated neighbors, transform. Works on unseen nodes.

**"What is over-smoothing?"**
→ Repeated neighborhood averaging blends all representations toward the same vector. Fix: 2–3 layers max, residual connections, JK-Net.

**"Why is sum better than mean in GIN?"**
→ Mean loses neighborhood size — a node with 2 neighbors [A,B] and one with 100 neighbors with the same average look identical. Sum preserves size → GIN achieves 1-WL expressiveness.

**"How do GNNs handle billion-node graphs?"**
→ Neighbor sampling (GraphSAGE): fix neighborhood size. Cluster-GCN: mini-batch by partition. SIGN: precompute multi-hop features offline, train MLP.

**"GAT vs transformer attention?"**
→ Both compute query-key attention weights. GAT restricts attention to the graph adjacency (sparse neighbors only). Transformer attends all-to-all (dense). Same mechanism, different mask.

---

## Part 8: Tabular Data and Deep Learning

### Mental Model

Most production ML uses tabular data. The default answer is almost always **gradient boosting** (XGBoost / LightGBM / CatBoost) — these beat neural networks on tabular tasks in the vast majority of benchmarks. Neural networks win when: (1) the dataset is very large (>1M rows), (2) there are many high-cardinality categorical features where entity embeddings help, (3) the task requires shared representations across multiple objectives, or (4) the model must be part of a larger differentiable pipeline.

---

### Algorithm Selector

| Situation | Use |
|-----------|-----|
| Standard classification/regression | LightGBM or XGBoost |
| Mixed features, many categoricals | CatBoost |
| Need probability calibration | XGBoost + isotonic regression |
| Very large dataset, GPU available | LightGBM on GPU |
| High-cardinality embeddings critical | TabNet or FT-Transformer |
| Multi-task / differentiable pipeline | FT-Transformer or SAINT |
| Interpretability required | LightGBM + SHAP |

---

### Gradient Boosting Fast Reference

**XGBoost / LightGBM / CatBoost** all build an ensemble of decision trees where each tree corrects the errors of the previous trees.

**Key hyperparameters:**
```
n_estimators:     100–2000 (use early stopping)
learning_rate:    0.01–0.3 (lower + more trees = better)
max_depth:        3–8 (LightGBM: use num_leaves instead)
subsample:        0.7–0.9 (row sampling, reduces overfitting)
colsample_bytree: 0.7–0.9 (feature sampling per tree)
reg_alpha/lambda: L1/L2 regularization
```

**LightGBM advantages:** leaf-wise tree growth (vs level-wise in XGBoost), histogram-based binning, GOSS (Gradient-based One-Side Sampling), DART dropout. Typically 10–100× faster than XGBoost.

**CatBoost advantages:** native ordered target encoding for categorical features — eliminates target leakage from standard mean encoding.

---

### Entity Embeddings for Categoricals

**The problem:** One-hot encoding of a categorical with 10,000 values creates a 10,000-dim sparse vector. Tree models handle this via splits; neural networks need dense representations.

**The core insight:** Learn a dense embedding for each category ID, similar to word embeddings. The embedding captures similarity structure: "Monday" and "Tuesday" end up near each other; "January" near "February."

```python
import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, cat_dims, embed_dims, n_cont, out_dim):
        super().__init__()
        # One embedding table per categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, n_emb) 
            for n_cat, n_emb in zip(cat_dims, embed_dims)
        ])
        total_emb = sum(embed_dims)
        self.fc = nn.Sequential(
            nn.Linear(total_emb + n_cont, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x_cat, x_cont):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.fc(x)
```

**Embedding dimension rule of thumb:** min(50, (n_categories + 1) // 2)

---

### Feature Engineering Essentials

**Numerical features:**
- StandardScaler (zero mean, unit variance) — required for neural networks; irrelevant for tree models
- Log transform for right-skewed distributions (income, counts)
- Quantile transform (rank-based normalization) — robust to outliers

**Categorical features:**
- **Ordinal encoding:** for tree models with natural order
- **One-hot encoding:** for low-cardinality (<20 values), tree models
- **Target encoding:** mean of target per category — powerful but leaky; use leave-one-out or k-fold encoding
- **Entity embeddings:** for neural networks with high-cardinality features

**Missing values:**
- Tree models (LightGBM, XGBoost): handle natively — just pass NaN
- Neural networks: impute + add binary "was_missing" indicator feature

**Interaction features:** `feature_A × feature_B`, `feature_A / feature_B` — tree models learn these automatically; can help linear models and small NNs.

---

### TabNet in 30 Seconds

TabNet uses sequential attention to select features at each step. At each of N steps:
1. A learned attention mask selects which features to use (sparse, like feature importance)
2. A feature transformer processes the selected features
3. The output contributes to the final prediction

**Key property:** Built-in feature selection + interpretable attention weights per sample. No need for external SHAP/LIME — TabNet shows which features mattered for each prediction.

**When it helps:** Datasets where only a subset of features is relevant per sample (not all features are always informative), and when interpretability is required alongside neural-network-level performance.

---

### FT-Transformer (Feature Tokenizer + Transformer)

**The core insight:** Convert each tabular feature (numerical or categorical) into a token embedding. Run a standard transformer over these tokens. The [CLS] token produces the final prediction.

```
Numerical feature x_i → linear projection → token embedding
Categorical feature → learned embedding → token embedding
[CLS] + all feature tokens → Transformer → [CLS] output → prediction
```

**Why this works:** Self-attention across features can learn arbitrary feature interactions. A numerical feature "income" attending to categorical "occupation" can learn "high income doctor" as a compound feature.

**FT-Transformer vs TabNet:**
- FT-Transformer: better accuracy on larger datasets, standard architecture, easy to pretrain
- TabNet: better interpretability, built-in feature selection, faster inference

---

### When Neural Networks Beat Gradient Boosting

| Condition | Advantage |
|-----------|-----------|
| Dataset size > 1M rows | NN scales better with data |
| High-cardinality categoricals | Entity embeddings generalize better than one-hot |
| Multi-task learning | Shared backbone across tasks |
| Integration with other NNs | End-to-end differentiability |
| Semi-supervised learning | Pretraining on unlabeled tabular data (SCARF) |

**Most of the time:** gradient boosting wins. The 2022 "Why do tree-based models still outperform deep learning on tabular data?" paper (Grinsztajn et al.) found that for datasets with <10K rows and mixed feature types, LightGBM beats all neural approaches on average.

---

### Tabular Interview Quick-Draws

**"XGBoost vs LightGBM — which would you use?"**
→ LightGBM by default: faster training (histogram binning, leaf-wise growth), lower memory, handles large datasets. XGBoost when exact split finding matters or when you need better GPU support for very wide sparse data.

**"How do you encode high-cardinality categorical features?"**
→ For tree models: target encoding (mean target per category, with k-fold to avoid leakage). For neural networks: entity embeddings. Avoid one-hot for cardinality >50 — too many dimensions, too sparse.

**"When would you use a neural network instead of LightGBM for tabular data?"**
→ Very large dataset (>1M rows), many high-cardinality categoricals, need to share representations across multiple objectives, or need to plug into an end-to-end differentiable pipeline. Otherwise, LightGBM is almost always faster to train and competitive or better.

**"How do you handle missing values in production?"**
→ LightGBM/XGBoost: just pass NaN — they handle it internally via learned "missing goes left/right" splits. Neural networks: mean/median imputation + binary "was_missing" flag as additional feature. Always impute with training-set statistics, stored in the pipeline.
