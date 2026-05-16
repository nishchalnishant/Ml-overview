# ML Overview — Topic Flowcharts for Recall

> One flowchart per topic area. Each node = concept. Arrow = "leads to" or "causes" or "is part of".
> Use for: daily 5-min recall drills, pre-study revision, interview prep.

---

## HOW TO USE THESE FLOWCHARTS

```
Before reading a topic  →  glance at its flowchart  →  predict the structure
After reading a topic   →  close the file            →  redraw from memory
Before an interview     →  only redraw, don't read   →  if you can draw it, you can explain it
```

---

## PART 1 — FOUNDATIONS

### 1.1 AI/ML Taxonomy

```
AI
├── Narrow AI → task-specific, no generalization
├── AGI → general reasoning across domains
├── Symbolic AI → logic rules, explicit knowledge
└── Sub-symbolic AI → learned representations, data-driven
    │
    └── ML
        ├── Supervised → labeled input-output pairs
        ├── Unsupervised → structure from unlabeled data
        ├── Semi-supervised → few labels + many unlabeled
        ├── Self-supervised → labels from data itself
        └── Reinforcement Learning → reward signal, policy
            │
            └── Deep Learning → multi-layer representations
                │
                └── LLMs → transformer + massive pretraining
                    ├── GPT family → autoregressive, decoder-only
                    ├── BERT family → masked LM, encoder-only
                    └── T5/Seq2Seq → encoder-decoder, text-to-text

No Free Lunch Theorem
└── no universally best algorithm → must match model to domain

Bias-Variance Tradeoff
├── High Bias → underfitting, model too simple
├── High Variance → overfitting, memorizes training noise
└── Sweet Spot → low bias + low variance → generalization
    ├── Underfitting signals → high train + test error
    └── Overfitting signals → low train, high test error
```

---

## PART 2 — CLASSICAL ML

### 2.1 Supervised Learning

```
Supervised Learning
│
├── Regression
│   ├── Linear Regression
│   │   ├── OLS → minimize sum of squared residuals
│   │   └── Regularization → penalize large weights
│   │       ├── Ridge (L2) → weight shrinkage, no zeroing
│   │       ├── Lasso (L1) → sparse weights, feature selection
│   │       └── ElasticNet → L1+L2 hybrid, grouped features
│   └── Polynomial Regression → fit nonlinear via feature expansion
│
└── Classification
    ├── Logistic Regression
    │   ├── sigmoid → maps logit to [0,1] probability
    │   └── log-loss → penalizes confident wrong predictions
    │
    ├── SVM (Support Vector Machine)
    │   ├── max-margin → maximize gap between classes
    │   ├── kernel trick → non-linear boundary via implicit mapping
    │   │   ├── RBF → infinite-dim Gaussian feature space
    │   │   └── Polynomial → degree-d decision boundary
    │   └── slack variable → soft margin, allows misclassification
    │
    ├── Naive Bayes
    │   ├── P(Y|X) ∝ P(Y)·∏P(xᵢ|Y)
    │   └── conditional independence → features independent given class
    │
    └── k-NN
        ├── lazy learner → no training, stores all data
        ├── distance metric → Euclidean / cosine / Manhattan
        └── curse of dimensionality → distances converge in high-D
```

### 2.2 Tree Methods

```
Tree Methods
│
├── Decision Trees (CART)
│   ├── split criterion
│   │   ├── Gini impurity → classification purity measure
│   │   └── Information Gain → entropy reduction at split
│   ├── pruning → reduce overfitting, remove low-info leaves
│   └── greedy splits → locally optimal, not globally
│
├── Random Forest
│   ├── bagging → parallel trees on bootstrap samples
│   ├── feature subsampling → √p features per split
│   └── ensemble vote → low variance, robust to noise
│
├── Boosting (Gradient)
│   ├── XGBoost → regularized gradient boosting, column sampling
│   ├── LightGBM → leaf-wise growth → faster, less memory
│   └── CatBoost → ordered boosting → handles categoricals natively
│       └── mechanism: fit residuals sequentially → additive model
│
└── AdaBoost
    ├── reweight misclassified samples each round
    └── final model → weighted sum of weak learners
```

### 2.3 Unsupervised Learning

```
Unsupervised Learning
│
├── Clustering
│   ├── K-Means
│   │   ├── assign → nearest centroid
│   │   ├── update → recompute centroid means
│   │   ├── elbow method → find K by inertia drop
│   │   └── silhouette score → cluster cohesion vs separation
│   │
│   ├── DBSCAN
│   │   ├── density reachability → core, border, noise points
│   │   ├── no K needed → discovers arbitrary shapes
│   │   └── handles noise → labels outliers explicitly
│   │
│   ├── HDBSCAN
│   │   ├── hierarchical density → builds condensed cluster tree
│   │   └── variable density → better than DBSCAN on real data
│   │
│   └── GMM (Gaussian Mixture Model)
│       ├── EM algorithm → iterative E and M steps
│       ├── E-step → compute soft cluster assignments
│       └── M-step → update Gaussian params per component
│
└── Dimensionality Reduction
    ├── PCA
    │   ├── max variance directions → principal components
    │   ├── SVD decomposition → X = UΣVᵀ
    │   └── linear, global → fast but misses nonlinear structure
    │
    ├── t-SNE
    │   ├── KL divergence → match neighbor distributions
    │   ├── perplexity → controls neighborhood size
    │   └── visualization only → distances not meaningful globally
    │
    └── UMAP
        ├── topology preserving → maintains local + global structure
        ├── faster than t-SNE → scalable to large datasets
        └── fuzzy simplicial sets → Riemannian manifold approx
```

### 2.4 Data Handling

```
Data Handling
│
├── Imbalanced Data
│   ├── SMOTE → interpolate between minority neighbors
│   ├── Focal Loss → down-weight easy negatives in loss
│   └── threshold moving → tune cutoff, use PR curve not accuracy
│
├── Feature Selection
│   ├── Filter methods → score independently of model
│   │   ├── Mutual Information → nonlinear feature-label dependence
│   │   ├── Chi² → categorical feature vs class association
│   │   └── ANOVA F-test → continuous feature class separation
│   │
│   ├── Wrapper methods
│   │   └── RFE → recursively eliminate weakest features
│   │
│   └── Embedded methods
│       ├── Lasso → L1 zeroes out irrelevant weights
│       └── SHAP → model-agnostic feature importance
│
├── Cross-Validation
│   ├── Stratified K-Fold → preserve class balance in folds
│   ├── Group K-Fold → prevent leakage across related samples
│   ├── Time-Series CV → forward chaining only, no future leak
│   └── Nested CV
│       ├── outer loop → unbiased performance estimate
│       └── inner loop → hyperparameter selection
│
└── Anomaly Detection
    ├── Isolation Forest → random splits isolate outliers fast
    ├── LOF (Local Outlier Factor) → local density ratio vs neighbors
    └── One-Class SVM → learns boundary around normal data
```

### 2.5 Model Building Strategies

```
Model Building Strategies
│
├── Ensemble Methods
│   ├── Bagging → parallel base learners, variance reduction
│   ├── Boosting → sequential learners, bias reduction
│   └── Stacking
│       ├── OOF predictions → out-of-fold base learner outputs
│       └── meta-learner → learns from base model outputs
│
├── Hyperparameter Optimization (HPO)
│   ├── Grid Search → exhaustive, expensive, scales poorly
│   ├── Random Search → often better than grid, faster
│   ├── Bayesian Opt (Optuna/TPE)
│   │   ├── probabilistic surrogate model → approximates objective
│   │   └── acquisition function → balance explore vs exploit
│   └── Hyperband
│       ├── successive halving → kill bad configs early
│       └── bracket scheduling → multi-fidelity budget allocation
│
└── Conformal Prediction
    ├── distribution-free → no distributional assumptions
    ├── coverage guarantee → P(Y ∈ C(X)) ≥ 1 - α
    └── nonconformity scores → rank calibration set residuals
```

---

## PART 3 — DEEP LEARNING COMPONENTS

### 3.1 Building Blocks

```
Deep Learning Building Blocks
│
├── Activation Functions
│   ├── ReLU → max(0,x), dying neuron if all negative inputs
│   ├── GELU → smooth ReLU, used in Transformers/BERT/GPT
│   ├── Swish → x·σ(x), self-gated, non-monotonic
│   └── Sigmoid/Tanh → saturate → vanishing gradient risk
│
├── Weight Initialization
│   ├── Xavier/Glorot → for tanh/sigmoid → 2/(n_in + n_out)
│   ├── He/Kaiming → for ReLU → 2/n_in, prevents dying neurons
│   ├── Orthogonal → for RNNs → preserves gradient norms
│   └── Zero init → anti-pattern → symmetry never breaks
│
├── Normalization Layers
│   ├── BatchNorm
│   │   ├── normalize over batch dimension
│   │   └── train ≠ eval → uses running stats at inference
│   ├── LayerNorm
│   │   ├── normalize over feature dimension
│   │   └── batch-independent → used in Transformers
│   ├── RMSNorm
│   │   ├── LayerNorm without mean subtraction
│   │   └── used in LLaMA / Gemma → faster
│   ├── GroupNorm → normalize within groups → small batches
│   └── InstanceNorm → per-sample per-channel → style transfer
│
└── Pre-Norm vs Post-Norm
    ├── Post-Norm → normalize after residual add → original Transformer
    └── Pre-Norm → normalize before sublayer → more stable deep gradients
```

### 3.2 Optimization & Regularization

```
Optimization & Regularization
│
├── Optimizers
│   ├── SGD → noisy gradients → implicit regularization effect
│   ├── Adam → adaptive LR per param via m/v moment estimates
│   ├── AdamW → Adam + decoupled weight decay → better generalization
│   └── Schedule-Free → no LR schedule needed, online averaging
│
├── Gradient Management
│   ├── Gradient clipping → clip by norm → prevents RNN explosion
│   └── Gradient accumulation → simulate large batch, low memory
│
├── Regularization Techniques
│   ├── Dropout → stochastic zero-out → reduces co-adaptation
│   ├── Weight decay (L2) → shrinks weights toward zero
│   ├── BatchNorm as regularizer → noise via batch statistics
│   └── Data augmentation → expand training distribution
│
└── Loss Functions
    ├── MSE → regression, penalizes large residuals heavily
    ├── Cross-Entropy → classification, measures distribution match
    ├── Focal Loss → imbalanced, down-weights easy examples
    ├── Triplet Loss → metric learning, anchor-pos-neg margin
    └── Ranking Losses → NDCG optimization, information retrieval
```

### 3.3 Backpropagation

```
Backpropagation
│
├── Forward Pass
│   ├── compute activations layer by layer
│   └── compute scalar loss at output
│
├── Backward Pass
│   ├── chain rule → ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
│   ├── accumulate gradients per parameter
│   └── optimizer → update weights via gradient step
│
├── Vanishing Gradient
│   ├── cause → tanh/sigmoid saturate, small derivatives chain
│   └── fix → ReLU activations + residual connections
│
├── Exploding Gradient
│   ├── cause → deep RNNs multiply large Jacobians
│   └── fix → gradient clipping by global norm
│
└── Residual Connections
    ├── output = F(x) + x → gradient highway through identity
    ├── enables very deep networks → ResNet / Transformers
    └── gradient flows directly → bypasses vanishing problem
```

### 3.4 Attention Mechanism

```
Attention Mechanism
│
├── Scaled Dot-Product Attention
│   ├── Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
│   ├── scale by √d_k → prevent softmax saturation
│   └── queries match keys → retrieve weighted values
│
├── Multi-Head Attention (MHA)
│   ├── h parallel attention heads → different subspaces
│   ├── concat + project → combine head outputs
│   └── captures diverse relationship patterns
│
├── Cross-Attention
│   ├── Q from decoder, K/V from encoder
│   └── decoder attends to encoder representations
│
├── Efficient Attention Variants
│   ├── MQA (Multi-Query) → one K/V head, many Q heads
│   ├── GQA (Grouped-Query) → grouped K/V sharing → inference speedup
│   └── FlashAttention
│       ├── tiled SRAM computation → avoids HBM roundtrips
│       └── memory O(N) not O(N²) → longer context feasible
│
└── KV-Cache
    ├── store past K/V tensors → skip recomputation
    ├── autoregressive inference speedup → each token cheaper
    └── memory grows with sequence length → eviction strategies needed
```

---

## PART 4 — DL ARCHITECTURES

### 4.1 Core Architectures

```
DL CORE ARCHITECTURES
│
├── CNNs
│   ├── Conv Layer → local receptive field + weight sharing
│   ├── Pooling → spatial downsampling, translation invariance
│   ├── ResNet → skip connections prevent vanishing gradients
│   ├── EfficientNet → compound scaling (depth + width + resolution)
│   └── ConvNeXt → CNN redesigned with ViT principles
│
├── RNNs
│   ├── LSTM
│   │   ├── Forget Gate → discard irrelevant past state
│   │   ├── Input Gate → write new info to cell
│   │   ├── Output Gate → expose cell state selectively
│   │   └── Cell State → long-range dependency carrier
│   └── GRU
│       ├── Reset Gate → how much past to forget
│       └── Update Gate → simplified LSTM, fewer params
│
├── Transformer
│   ├── Self-Attention → query-key-value dot-product scores
│   ├── Multi-Head → attend multiple representation subspaces
│   ├── FFN → position-wise two-layer projection
│   └── Parallelizable → no sequential bottleneck (vs RNNs)
│
└── ViT (Vision Transformer)
    ├── Patch Embedding → image split into fixed-size tokens
    ├── [CLS] Token → global classification representation
    └── Pure Transformer → no CNN inductive bias
```

### 4.2 Generative Models

```
GENERATIVE MODELS
│
├── VAE (Variational Autoencoder)
│   ├── Encoder → outputs μ, σ (distribution params)
│   ├── Reparameterization Trick → z = μ + σ·ε (differentiable sampling)
│   ├── Decoder → reconstruct x from z
│   └── ELBO Loss → reconstruction loss − KL divergence
│
├── GAN (Generative Adversarial Network)
│   ├── Generator → produce samples to fool discriminator
│   ├── Discriminator → distinguish real vs fake
│   ├── Minimax Game → adversarial training objective
│   ├── Mode Collapse → generator ignores latent diversity
│   └── Training Instability → vanishing gradients for generator
│
├── WGAN (Wasserstein GAN)
│   ├── Wasserstein Distance → smoother gradient landscape
│   ├── Lipschitz Constraint → enforced via gradient penalty
│   └── Stable Training → no mode collapse / log-saturation
│
├── StyleGAN
│   ├── Mapping Network → z → w (disentangled latent space)
│   ├── AdaIN → style injected at each resolution layer
│   └── Progressive Growing → coarse-to-fine training
│
└── Diffusion Models
    ├── Forward Process (q) → gradually add Gaussian noise
    ├── Reverse Process (p_θ) → U-Net denoise step-by-step
    ├── DDPM → discrete T steps, slower sampling
    ├── DDIM → deterministic, fewer steps, faster inference
    └── LDM (Latent Diffusion) → operate in latent space → cheaper compute
```

### 4.3 Transfer Learning

```
TRANSFER LEARNING
│
├── Pre-Training → learn general features on large dataset
│
├── Feature Extraction
│   ├── Freeze Backbone → no gradient through base model
│   └── Train Head Only → fast, less data required
│
├── Fine-Tuning
│   ├── Unfreeze All Layers → adapt to target domain
│   └── Discriminative LR → lower LR for early layers, higher for head
│
├── Domain Adaptation
│   ├── DANN (Domain-Adversarial NN)
│   │   ├── Gradient Reversal Layer → flip gradient for domain classifier
│   │   └── Feature Alignment → domain-invariant representations
│   └── CORAL → match second-order statistics across domains
│
└── Few-Shot / Zero-Shot
    ├── ProtoNets → class prototype = mean embedding → nearest prototype
    ├── MAML → learn initialization → fast adapt via few gradient steps
    └── Zero-Shot (CLIP) → align image + text embeddings → no target labels
```

---

## PART 5 — NLP

### 5.1 Text Representations

```
TEXT REPRESENTATIONS
│
├── Tokenization
│   ├── BPE (Byte-Pair Encoding)
│   │   ├── Merge most frequent byte pairs iteratively
│   │   └── Used in → GPT-2, LLaMA
│   ├── WordPiece
│   │   ├── Merge by likelihood maximization
│   │   ├── ## prefix → continuation subword token
│   │   └── Used in → BERT
│   └── SentencePiece
│       ├── Operates on raw text, no pre-tokenization
│       └── Used in → T5, mT5 (language-agnostic)
│
├── Static Embeddings
│   ├── Word2Vec
│   │   ├── Skip-Gram → predict context from word
│   │   └── CBOW → predict word from context window
│   ├── GloVe → factorize global co-occurrence matrix
│   └── FastText → character n-grams → handles OOV words
│
└── Contextual Embeddings
    ├── ELMo → BiLSTM hidden states, context-dependent
    └── BERT → bidirectional Transformer, MLM pre-training
```

### 5.2 Sequence Modeling

```
SEQUENCE MODELING
│
├── Seq2Seq (RNN-based)
│   ├── Encoder → compress input to fixed context vector
│   └── Decoder → generate output (bottleneck problem!)
│
├── Bahdanau Attention
│   ├── Attend over all encoder hidden states per decode step
│   └── Precursor to Transformer self-attention mechanism
│
├── Beam Search
│   ├── Maintain top-K hypotheses at each decode step
│   ├── Better coverage than greedy decoding
│   └── Length Penalty → avoid bias toward short sequences
│
└── NER (Named Entity Recognition)
    ├── BIO Tagging → B=begin, I=inside, O=outside
    └── BERT + Token Classification Head → state-of-the-art
```

### 5.3 NLP Tasks

```
NLP TASKS
│
├── Text Classification
│   ├── TF-IDF + Logistic Regression → fast interpretable baseline
│   └── BERT Fine-Tune → 2e-5 LR, ~3 epochs, strong accuracy
│
├── Summarization
│   ├── BART → denoising pre-training → abstractive summaries
│   └── TextRank → PageRank on sentence similarity graph → extractive
│
├── Evaluation Metrics
│   ├── BLEU → precision n-gram overlap (machine translation)
│   ├── ROUGE → recall n-gram overlap (summarization)
│   └── BERTScore → contextual cosine similarity, semantic aware
│
├── Semantic Similarity
│   ├── SBERT → siamese BERT + mean pooling → cosine similarity
│   └── NLI → entailment / neutral / contradiction → zero-shot templates
│
└── Coreference Resolution
    ├── Span Detection → identify candidate mention spans
    └── Clustering → group coreferring mentions together
```

### 5.4 Advanced NLP

```
ADVANCED NLP
│
├── Summarization Faithfulness
│   ├── FactCC → NLI-based factual consistency check
│   └── SummaC → segment-level consistency scoring
│
├── Dependency Parsing
│   ├── Head-Dependent syntactic relations in sentence
│   └── spaCy → fast production parser, arc-eager transition
│
├── Relation Extraction
│   └── Identify semantic relations between entity pairs
│
└── Dialogue Systems
    ├── Pipeline → NLU → DST → Policy → NLG
    │   ├── NLU → intent + slot detection
    │   ├── DST → belief state tracker over turns
    │   ├── Policy → select system action
    │   └── NLG → surface action as natural language
    └── End-to-End → fine-tune LLM on conversation data
```

---

## PART 6 — COMPUTER VISION

### 6.1 Detection & Segmentation

```
DETECTION & SEGMENTATION
│
├── Object Detection
│   ├── YOLO → single-shot anchor-based → real-time inference
│   ├── Faster R-CNN → RPN + RoI Pooling → high accuracy
│   └── DETR → Transformer + bipartite matching → no anchors/NMS
│
├── Semantic Segmentation
│   ├── FCN → replace FC layers with conv → dense predictions
│   ├── U-Net → skip connections between encoder and decoder
│   ├── DeepLab → atrous conv + ASPP → multi-scale context
│   └── SegFormer → hierarchical Transformer + lightweight MLP decoder
│
├── Instance Segmentation
│   ├── Mask R-CNN → adds mask head to Faster R-CNN + RoIAlign
│   │   └── RoIAlign → no quantization error (vs RoI Pooling)
│   └── SOLOv2 → grid-based, no region proposals needed
│
└── Panoptic Segmentation
    ├── Stuff → amorphous regions (semantic)
    ├── Things → countable objects (instance)
    └── PQ (Panoptic Quality) = SQ × RQ
```

### 6.2 Pose Estimation

```
POSE ESTIMATION
│
├── OpenPose (Bottom-Up)
│   ├── Part Affinity Fields → encode limb orientations
│   └── Assemble keypoints → no person detector needed
│
├── HRNet (Top-Down)
│   ├── Person Detector → crop each person first
│   ├── High-Resolution Maintained → no upsampling artifacts
│   └── Multi-Scale Fusion → parallel resolution streams
│
├── ViTPose
│   ├── Transformer Backbone → scalable feature extraction
│   └── Multi-Task Capable → unified pose estimation framework
│
└── Metrics
    ├── PCKh → % keypoints within threshold of head size
    └── OKS → per-keypoint weighted distance similarity
```

### 6.3 Metric Learning & Retrieval

```
METRIC LEARNING & RETRIEVAL
│
├── Loss Functions
│   ├── Contrastive Loss → positives close, negatives beyond margin
│   ├── Triplet Loss → anchor-positive close, anchor-negative far
│   │   └── Semi-Hard Mining → critical for training stability
│   └── ArcFace → additive angular margin → more discriminative than softmax
│
├── Approximate Nearest Neighbor
│   └── FAISS → IVF index → billion-scale ANN at speed
│
├── Retrieval Metrics
│   ├── Recall@K → fraction of true matches in top-K
│   └── mAP → mean average precision across queries
│
└── Re-ID (Re-Identification)
    ├── Cross-Camera Identity Matching
    └── Challenges → viewpoint change, occlusion, illumination
```

### 6.4 Video & 3D Vision

```
VIDEO & 3D VISION
│
├── Video Understanding
│   ├── Two-Stream → RGB stream + optical flow stream → late fusion
│   ├── I3D → inflate 2D conv kernels to 3D → ImageNet init possible
│   ├── SlowFast → slow pathway (semantics) + fast pathway (motion)
│   ├── TimeSformer → divided space-time attention → efficient Transformer
│   └── VideoMAE → masked autoencoder, 90% masking ratio → strong pretraining
│
└── 3D Vision
    ├── PointNet
    │   ├── Per-Point MLP → process each point independently
    │   └── Global Max Pool → permutation invariant aggregation
    ├── PointNet++
    │   ├── Hierarchical Local Grouping → neighborhood context
    │   └── Multi-Scale Features → robust to sampling density
    ├── NeRF (Neural Radiance Field)
    │   ├── Implicit Neural Scene Representation → MLP maps (x,y,z,θ,φ) → (RGB, σ)
    │   └── Novel View Synthesis → volume rendering integral
    └── 3D Gaussian Splatting
        ├── Explicit 3D Gaussian Primitives → scene representation
        └── Real-Time Rendering → rasterization-based, no ray marching
```

---

## PART 7 — LARGE LANGUAGE MODELS

### 7.1 LLM Architecture

```
LLM Architecture
│
├── Transformer Decoder
│   ├── causal masking → attend only to past tokens
│   └── autoregressive → one token at a time
│
├── Positional Encodings
│   ├── RoPE → rotate Q/K by position index
│   │   └── used in LLaMA → relative position via rotation
│   └── ALiBi → linear bias added to attention score
│       └── no vector modification → long context friendly
│
├── Attention Variants
│   ├── MHA → multi-head, separate K/V per head
│   ├── MQA → all heads share single K/V
│   └── GQA → grouped heads share K/V sets
│       └── inference speedup → less KV memory
│
├── Mixture of Experts (MoE)
│   ├── top-K routing → sparse expert activation
│   ├── scale params without proportional compute
│   └── examples: DeepSeek, Mixtral
│
└── KV-Cache
    ├── store past K/V tensors across steps
    └── O(1) per new token at inference
```

### 7.2 LLM Training

```
LLM Training
│
├── Pre-training
│   ├── next-token prediction → cross-entropy loss
│   └── Chinchilla scaling law → 20 tokens per param optimal
│
├── Supervised Fine-Tuning (SFT)
│   └── train on human demonstrations → labeled (prompt, response)
│
├── RLHF
│   ├── reward model → scores responses from human prefs
│   ├── PPO → policy gradient with clipped objective
│   └── KL penalty → stay close to SFT baseline
│
├── DPO (Direct Preference Optimization)
│   ├── no RL loop → closed-form preference loss
│   └── simpler, stable → replaces reward model + PPO
│
└── Efficient Fine-Tuning
    ├── LoRA → ΔW = A·B (low-rank), <1% trainable params
    │   └── freeze base → only train A, B matrices
    └── QLoRA → 4-bit quantized base + LoRA adapters
        └── fits 65B model on 48GB GPU
```

### 7.3 LLM Inference & Applications

```
LLM Inference & Applications
│
├── Quantization
│   ├── INT8 / INT4 → weight-only or activation quant
│   ├── GPTQ → post-training, layer-wise reconstruction
│   └── AWQ → activation-aware weight quantization
│
├── Speculative Decoding
│   ├── draft model proposes multiple tokens
│   └── main model verifies in parallel → net speedup
│
├── Continuous Batching (vLLM)
│   ├── PagedAttention → KV cache in paged blocks
│   └── high GPU utilization → no wasted memory
│
├── RAG (Retrieval-Augmented Generation)
│   ├── retrieve relevant docs at query time
│   ├── ground generation with retrieved context
│   └── reduces hallucination → factual grounding
│
├── Agents
│   ├── tool use → call APIs, code, search
│   ├── planning: ReAct → reason + act interleaved
│   └── CoT → chain-of-thought → multi-step reasoning
│
└── Context Extension
    ├── YaRN → NTK-aware RoPE frequency scaling
    ├── FlashAttention → tiled O(N) memory attention
    └── Longformer → sliding window + global tokens
```

### 7.4 LLM Challenges

```
LLM Challenges
│
├── Hallucination
│   ├── model generates plausible but false content
│   │
│   ├── Detection
│   │   ├── SelfCheckGPT → sample multiple → check consistency
│   │   └── FactScore → decompose → retrieve → verify per claim
│   │
│   └── Mitigation
│       ├── RAG → ground in retrieved facts
│       ├── constrained decoding → Outlines → structured output
│       └── calibration → align confidence to accuracy
│
└── Model Merging
    ├── task vector = fine-tuned weights − base weights
    │
    ├── Task Arithmetic → add/subtract task vectors directly
    ├── TIES → trim small deltas + elect sign + merge
    ├── DARE → randomly drop weight deltas before merge
    └── SLERP → spherical interpolation between checkpoints
```

---

## PART 8 — REINFORCEMENT LEARNING

### 8.1 Core RL

```
Core RL
│
├── MDP → (S, A, P, R, γ) — formal environment model
│
├── Bellman Equations
│   └── V(s) = R + γ · max_a Q(s,a) → recursive value
│
├── Value-Based Methods
│   ├── Q-Learning → off-policy TD, tabular or approx
│   ├── DQN → experience replay + target network → stable
│   └── DDQN → decouple action selection from evaluation
│
├── Policy Gradient Methods
│   ├── REINFORCE → MC returns → high variance baseline
│   ├── PPO → clipped surrogate objective → stable updates
│   └── SAC → entropy regularization → exploration bonus
│
└── Model-Based RL
    ├── learn transition model of environment
    └── Dyna / MBPO → plan with model → data efficient
```

### 8.2 Advanced RL

```
Advanced RL
│
├── Imitation Learning
│   ├── BC (Behavioral Cloning) → supervised on (s,a) pairs
│   │   └── compounding error → small mistakes amplify
│   ├── DAgger → iteratively query expert on visited states
│   └── GAIL → discriminator: expert vs policy rollout
│       └── PPO as generator → adversarial imitation
│
├── Inverse RL (IRL)
│   └── MaxEntropy IRL → infer reward matching expert occupancy
│
├── Multi-Agent RL (MARL)
│   ├── QMIX → monotone mixing of individual Q-functions
│   ├── MAPPO → centralized critic + decentralized actors
│   └── self-play → emergent strategies (AlphaGo)
│
├── Hierarchical RL
│   ├── Options framework → (I, π_ω, β) — subgoal policies
│   └── HER → relabel failed trajectory with achieved goal
│
└── Meta-RL
    ├── MAML → learn init for fast few-shot adaptation
    └── RL² → RNN hidden state encodes task identity
```

### 8.3 Sim-to-Real & Offline RL

```
Sim-to-Real & Offline RL
│
├── Sim-to-Real Transfer
│   ├── domain randomization → vary physics params → robust policy
│   ├── system identification → fit sim to real trajectories
│   └── RMA → privileged sim policy + real-time adapt module
│
└── Offline RL
    ├── CQL → penalize out-of-distribution action Q-values
    ├── IQL → expectile regression on V(s) → no OOD queries
    └── Decision Transformer → RL as sequence modeling
        └── return-conditioned autoregressive generation
```

---

## PART 9 — RECOMMENDER SYSTEMS & GNNs

### 9.1 Recommender Systems

```
Recommender Systems
│
├── Collaborative Filtering
│   ├── user-user / item-item → cosine similarity on ratings
│   └── Matrix Factorization → UV^T ≈ R → latent factors
│       └── optimized via ALS or SGD
│
├── Content-Based
│   └── item features → user profile → cosine similarity match
│
├── Two-Tower Model
│   ├── query tower + item tower → separate encoders
│   └── dot product score → scalable ANN retrieval
│
├── Learning-to-Rank
│   ├── pointwise → MSE on relevance scores
│   ├── pairwise → RankNet, BPR → prefer relevant over not
│   └── listwise → LambdaRank → NDCG-weighted gradient
│
├── Cold-Start
│   ├── feature-based → use item/user attributes
│   ├── popularity fallback → recommend trending items
│   └── meta-learning → fast adapt to new user/item
│
├── Session-Based
│   ├── GRU4Rec → RNN on click sequence → next item
│   └── BERT4Rec → masked item prediction → bidirectional
│
└── GNN-Based
    ├── PinSage → GraphSAGE on item-item graph
    └── LightGCN → simplified GCN, no feature transform
```

### 9.2 Graph Neural Networks

```
Graph Neural Networks
│
├── Message Passing Framework
│   ├── aggregate neighbor messages → update node embedding
│   └── stack L layers → captures L-hop neighborhood
│
├── GCN → spectral: Aˆ·H·W (normalized adjacency)
│   └── transductive → fixed graph at train time
│
├── GraphSAGE → inductive: sample + aggregate neighbors
│   ├── aggregate: mean / LSTM / pooling
│   └── generalizes to unseen nodes at test time
│
├── GAT → attention weights per neighbor edge
│   └── different importance to different neighbors
│
├── GIN → sum aggregation → max expressiveness
│   └── as powerful as WL graph isomorphism test
│
├── Knowledge Graphs
│   ├── TransE → h + r ≈ t in embedding space
│   └── RotatE → relation as rotation in complex space
│
├── Dynamic Graphs
│   ├── TGN → memory module + temporal message passing
│   └── TGAT → time encoding via random Fourier features
│
└── Graph Generation
    ├── VGAE → encode → sample z → decode adjacency
    └── GraphRNN → sequential node/edge autoregressive
```

---

## PART 10 — PRODUCTION ML

### 10.1 MLOps Pipeline

```
MLOps Pipeline
│
├── Stages
│   ├── Data → feature engineering → training
│   └── evaluation → deployment → monitoring
│
├── Data Versioning
│   └── DVC → Git-like versioning for datasets/models
│
├── Feature Store
│   ├── offline → batch features for training
│   ├── online → low-latency features for serving
│   └── Feast → unified offline + online feature store
│
├── Experiment Tracking
│   └── MLflow → log params + metrics + artifacts
│
├── CI/CD for ML
│   └── retrain trigger → data drift or schedule
│
└── Model Registry
    └── versioning + stage transitions → staging → production
```

### 10.2 Deployment Patterns

```
Deployment Patterns
│
├── Blue/Green → swap 100% traffic instantly → zero downtime
│
├── Canary → route % traffic to new model → gradual rollout
│
├── Shadow Mode → mirror traffic → compare → no user impact
│
├── A/B Test
│   ├── split traffic → control vs treatment
│   ├── t-test / z-test → statistical significance
│   └── MDE → min detectable effect → sample size planning
│
└── Rollback
    └── automated trigger → latency spike or metric drop
```

### 10.3 Serving & Latency

```
Serving & Latency Optimization
│
├── Quantization
│   └── INT8 / FP16 → 2-4× inference speedup
│
├── ONNX Export
│   └── hardware-optimized runtime → cross-platform
│
├── TensorRT
│   └── GPU kernel fusion → minimize memory transfers
│
├── Dynamic Batching
│   └── amortize per-request overhead → higher throughput
│
├── Serving Frameworks
│   ├── TorchServe → PyTorch multi-model serving
│   └── Triton → multi-framework, concurrent execution
│
└── Training-Serving Skew
    └── use same feature pipeline → feature store solves this
```

### 10.4 Monitoring

```
Monitoring
│
├── Data Drift Detection
│   ├── PSI (Population Stability Index) → numerical shift
│   ├── KS test → distribution shift → continuous features
│   └── chi-squared → categorical feature distribution shift
│
├── Concept Drift
│   ├── model performance degrades over time
│   └── trigger → scheduled or threshold-based retraining
│
├── Logging
│   └── log predictions + features → compare to training dist
│
└── Slice-Based Evaluation
    ├── disaggregate metrics by subgroup
    └── catch hidden failures → fairness, edge cases
```

---

## PART 11 — EMERGING TOPICS

### 11.1 Privacy, Safety & Ethics

```
Privacy, Safety & Ethics
│
├── Differential Privacy
│   ├── (ε,δ)-DP → bound info leaked per individual
│   ├── add calibrated Gaussian or Laplace noise
│   └── DP-SGD → clip gradients + noise → Opacus library
│
├── Federated Learning
│   ├── FedAvg → local SGD + aggregate weights centrally
│   │   └── data stays on device → privacy preserving
│   └── FedProx → proximal term → handles heterogeneous clients
│
├── Adversarial Robustness
│   ├── FGSM → single step: x + ε·sign(∇L)
│   ├── PGD → multi-step FGSM with projection → stronger
│   ├── adversarial training → train on PGD examples
│   └── randomized smoothing → certifiable L2 robustness
│
└── Fairness
    ├── demographic parity → equal positive prediction rates
    ├── equalized odds → equal TPR and FPR across groups
    ├── impossibility theorem → can't satisfy all simultaneously
    └── bias mitigation
        ├── pre-process → reweight or resample data
        ├── in-process → adversarial debiasing during training
        └── post-process → threshold calibration per group
```

### 11.2 Continual & Meta Learning

```
Continual & Meta Learning
│
├── Catastrophic Forgetting
│   └── new task training overwrites old task weights
│
├── Mitigation Strategies
│   ├── EWC → penalize changes to Fisher-important weights
│   ├── Experience Replay → store + interleave old samples
│   └── Progressive Nets → new columns per task + lateral links
│
└── Neural Architecture Search (NAS)
    ├── DARTS → differentiable search → continuous relaxation
    └── Hyperband → successive halving → early stopping budget
```

### 11.3 Probabilistic & Optimization Theory

```
Probabilistic & Optimization Theory
│
├── Probabilistic Graphical Models
│   ├── Bayesian Networks → DAG + CPT → factored joint dist
│   ├── HMM
│   │   ├── Viterbi → decode most likely state sequence
│   │   ├── Baum-Welch → learn emission + transition params
│   │   └── forward algorithm → evaluate observation prob
│   ├── CRF → discriminative MRF → NER, sequence labeling
│   │   └── globally optimal decoding via Viterbi
│   └── LDA → plate model → collapsed Gibbs sampling
│
├── Natural Gradient
│   └── steepest descent in distribution space → F^{-1}·∇L
│
├── SVRG (Stochastic Variance Reduced Gradient)
│   ├── periodic full gradient snapshot → reduce variance
│   └── O(1/T) convergence → faster than SGD
│
└── SAM (Sharpness-Aware Minimization)
    ├── find worst perturbation in weight neighborhood
    ├── minimize loss at perturbed point → flat minima
    └── flat minima → better generalization on test set
```
