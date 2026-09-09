---
module: ML Overview
topic: Flowcharts
subtopic: ""
status: unread
tags: [mloverviewroot, ml, flowcharts]
---
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

**Coverage map — which repo section each part backs onto**

| Part | Topic | Repo section |
|---|---|---|
| 1 | Foundations | `00-meta/`, `01-math-foundations/` |
| 2 | Classical ML | `02-data/`, `03-classical-ml/`, `04-evaluation/` |
| 3 | DL components | `05-deep-learning-core/` |
| 4 | DL architectures | `06-architectures/` |
| 5–6 | NLP, Computer Vision | `07-domains/` |
| 7 | Large language models | `10-llms/` |
| 8 | Reinforcement learning | `09-reinforcement-learning/` |
| 9 | RecSys & GNNs | `06-architectures/05-gnn.md`, `15-system-design/cases/` |
| 10 | Production ML | `13-production-ml/` |
| 11–12 | Emerging topics & trends | `08-generative/`, `11-llm-applications/`, `14-responsible-ai/` |
| 13 | Multimodal | `11-llm-applications/09-multimodal.md` |
| 14 | Data scientist | `02-data/03-sql-for-interviews.md`, `04-evaluation/06-ab-testing-experimentation.md`, `04-evaluation/07-causal-inference.md` |
| 15 | Production deployment | `12-systems-and-scale/`, `13-production-ml/` |
| 16 | AI infrastructure / inference data plane | `20-ai-infrastructure/` |

---

## PART 1 — FOUNDATIONS

### 1.1 AI/ML Taxonomy

```
AI
├── Narrow AI → task-specific, no generalization across domains
├── AGI → general reasoning across domains, no specialized training
├── Symbolic AI → logic rules, explicit knowledge, brittle to noise
└── Sub-symbolic AI → learned representations, data-driven, tolerant to noise
    │
    └── ML
        ├── Supervised → labeled input-output pairs → minimize empirical loss
        ├── Unsupervised → structure from unlabeled data → density, clusters, latents
        ├── Semi-supervised → few labels + many unlabeled → pseudo-labeling or consistency
        ├── Self-supervised → labels from data itself → MAE, BERT, contrastive pretraining
        └── Reinforcement Learning → reward signal, policy → no i.i.d. assumption
            │
            └── Deep Learning → multi-layer representations → end-to-end gradient flow
                │
                └── LLMs → transformer + massive pretraining → emergent capabilities
                    ├── GPT family → autoregressive, decoder-only → next-token loss
                    ├── BERT family → masked LM, encoder-only → bidirectional context
                    └── T5/Seq2Seq → encoder-decoder, text-to-text → unified format

No Free Lunch Theorem
└── no universally best algorithm → must match model to domain → inductive bias matters

Bias-Variance Tradeoff
├── High Bias → underfitting, model too simple → cannot capture true function
├── High Variance → overfitting, memorizes training noise → poor generalization
└── Sweet Spot → low bias + low variance → generalization
    ├── Underfitting signals → high train + test error → increase model capacity
    └── Overfitting signals → low train, high test error → regularize or get more data
```

---

## PART 2 — CLASSICAL ML

### 2.1 Supervised Learning

```
Supervised Learning
│
├── Regression
│   ├── Linear Regression
│   │   ├── OLS → minimize sum of squared residuals → closed form w = (XᵀX)⁻¹Xᵀy
│   │   └── Regularization → penalize large weights → prevent overfitting
│   │       ├── Ridge (L2) → weight shrinkage, no zeroing → λ‖w‖² added to loss
│   │       ├── Lasso (L1) → sparse weights, feature selection → λ‖w‖₁ induces zeros
│   │       └── ElasticNet → L1+L2 hybrid, grouped features → α controls mix
│   └── Polynomial Regression → fit nonlinear via feature expansion → risk: high-degree overfits
│
└── Classification
    ├── Logistic Regression
    │   ├── sigmoid → maps logit to [0,1] probability → σ(z) = 1/(1+e⁻ᶻ)
    │   └── log-loss → penalizes confident wrong predictions → -y·log(p) - (1-y)·log(1-p)
    │
    ├── SVM (Support Vector Machine)
    │   ├── max-margin → maximize gap between classes → only support vectors matter
    │   ├── kernel trick → non-linear boundary via implicit mapping → K(x,x') = φ(x)·φ(x')
    │   │   ├── RBF → infinite-dim Gaussian feature space → K=exp(-γ‖x-x'‖²)
    │   │   └── Polynomial → degree-d decision boundary → K=(xᵀx'+c)^d
    │   └── slack variable → soft margin, allows misclassification → C trades margin vs error
    │
    ├── Naive Bayes
    │   ├── P(Y|X) ∝ P(Y)·∏P(xᵢ|Y) → log-space for numerical stability
    │   └── conditional independence → features independent given class → strong but often works
    │
    └── k-NN
        ├── lazy learner → no training, stores all data → O(N) query cost
        ├── distance metric → Euclidean / cosine / Manhattan → choice is domain-dependent
        └── curse of dimensionality → distances converge in high-D → need dim reduction first
```

### 2.2 Tree Methods

```
Tree Methods
│
├── Decision Trees (CART)
│   ├── split criterion
│   │   ├── Gini impurity → classification purity measure → 1 - Σpᵢ² → faster to compute
│   │   └── Information Gain → entropy reduction at split → -Σp·log(p), higher = better split
│   ├── pruning → reduce overfitting, remove low-info leaves → cost-complexity parameter α
│   └── greedy splits → locally optimal, not globally → no backtracking
│
├── Random Forest
│   ├── bagging → parallel trees on bootstrap samples → each tree sees 63% unique samples
│   ├── feature subsampling → √p features per split → decorrelates trees
│   └── ensemble vote → low variance, robust to noise → avg of uncorrelated estimators
│
├── Boosting (Gradient)
│   ├── XGBoost → regularized gradient boosting, column sampling → second-order Taylor approx
│   ├── LightGBM → leaf-wise growth → faster, less memory → histogram-based splits
│   └── CatBoost → ordered boosting → handles categoricals natively → target stats leak-free
│       └── mechanism: fit residuals sequentially → additive model → fₜ(x) = fₜ₋₁(x) + η·hₜ(x)
│
└── AdaBoost
    ├── reweight misclassified samples each round → exponential weight increase on errors
    └── final model → weighted sum of weak learners → αₜ = 0.5·ln((1-εₜ)/εₜ)
```

### 2.3 Unsupervised Learning

```
Unsupervised Learning
│
├── Clustering
│   ├── K-Means
│   │   ├── assign → nearest centroid → O(N·K·d) per iteration
│   │   ├── update → recompute centroid means → convergence guaranteed, not global optimum
│   │   ├── elbow method → find K by inertia drop → diminishing returns in WCSS
│   │   └── silhouette score → cluster cohesion vs separation → s ∈ [-1,1], higher is better
│   │
│   ├── DBSCAN
│   │   ├── density reachability → core, border, noise points → minPts within ε radius
│   │   ├── no K needed → discovers arbitrary shapes → not convex-cluster assumption
│   │   └── handles noise → labels outliers explicitly → robust to non-spherical clusters
│   │
│   ├── HDBSCAN
│   │   ├── hierarchical density → builds condensed cluster tree → soft cluster assignments
│   │   └── variable density → better than DBSCAN on real data → extracts flat from hierarchy
│   │
│   └── GMM (Gaussian Mixture Model)
│       ├── EM algorithm → iterative E and M steps → converges to local maximum of likelihood
│       ├── E-step → compute soft cluster assignments → γₙₖ = πₖ·N(xₙ|μₖ,Σₖ) / Σⱼ(...)
│       └── M-step → update Gaussian params per component → MLE given responsibilities
│
└── Dimensionality Reduction
    ├── PCA
    │   ├── max variance directions → principal components → eigenvectors of covariance matrix
    │   ├── SVD decomposition → X = UΣVᵀ → columns of V are principal components
    │   └── linear, global → fast but misses nonlinear structure → use kernel PCA for nonlinear
    │
    ├── t-SNE
    │   ├── KL divergence → match neighbor distributions → KL(P‖Q) not symmetric
    │   ├── perplexity → controls neighborhood size → effective number of neighbors ~5-50
    │   └── visualization only → distances not meaningful globally → no inverse transform
    │
    └── UMAP
        ├── topology preserving → maintains local + global structure → better than t-SNE globally
        ├── faster than t-SNE → scalable to large datasets → O(N log N) vs O(N²)
        └── fuzzy simplicial sets → Riemannian manifold approx → theoretically grounded
```

### 2.4 Data Handling

```
Data Handling
│
├── Imbalanced Data
│   ├── SMOTE → interpolate between minority neighbors → synthetic samples in feature space
│   ├── Focal Loss → down-weight easy negatives in loss → FL = -α(1-pₜ)^γ log(pₜ)
│   └── threshold moving → tune cutoff, use PR curve not accuracy → AUC-PR over AUC-ROC
│
├── Missing Data & Imputation
│   ├── Mean/Median imputation → fast, distorts distribution → breaks correlations, inflates N
│   ├── KNN imputation → fill with weighted neighbor values → preserves local feature structure
│   └── MICE (Multiple Imputation by Chained Equations) → iterative multivariate regression per
│       feature → gold standard → captures inter-feature dependencies, handles MCAR/MAR
│
├── Feature Scaling
│   ├── StandardScaler → zero mean, unit variance → z = (x - μ)/σ → assumes Gaussian-like
│   ├── MinMaxScaler → bounds features to [0,1] → x' = (x-min)/(max-min) → sensitive to outliers
│   └── RobustScaler → median/IQR centering → x' = (x-median)/IQR → outlier-resistant
│
├── Categorical Encoding
│   ├── One-hot encoding → sparse binary columns, no ordinal assumption → high cardinality → curse
│   ├── Label encoding → ordinal integers → only valid for truly ordered categories
│   ├── Target encoding → replace category with mean(target) → leakage risk → must use CV folds
│   └── Embeddings → learned dense vectors for high-cardinality → Entity Embeddings (NN-based)
│
├── Feature Selection
│   ├── Filter methods → score independently of model
│   │   ├── Mutual Information → nonlinear feature-label dependence → I(X;Y) = H(X) - H(X|Y)
│   │   ├── Chi² → categorical feature vs class association → requires non-negative features
│   │   └── ANOVA F-test → continuous feature class separation → assumes Gaussian within class
│   │
│   ├── Wrapper methods
│   │   └── RFE → recursively eliminate weakest features → expensive, model-dependent
│   │
│   └── Embedded methods
│       ├── Lasso → L1 zeroes out irrelevant weights → sparsity is automatic
│       └── SHAP → model-agnostic feature importance → Shapley values from game theory
│
├── Cross-Validation
│   ├── Stratified K-Fold → preserve class balance in folds → critical for imbalanced datasets
│   ├── Group K-Fold → prevent leakage across related samples → e.g., same patient in both sets
│   ├── Time-Series CV → forward chaining only, no future leak → expanding or sliding window
│   └── Nested CV
│       ├── outer loop → unbiased performance estimate → test score on held-out fold
│       └── inner loop → hyperparameter selection → prevents overfitting to val set
│
└── Anomaly Detection
    ├── Isolation Forest → random splits isolate outliers fast → anomalies need fewer splits
    ├── LOF (Local Outlier Factor) → local density ratio vs neighbors → k-distance reachability
    └── One-Class SVM → learns boundary around normal data → RBF kernel in practice
```

### 2.5 Model Building Strategies

```
Model Building Strategies
│
├── Ensemble Methods
│   ├── Bagging → parallel base learners, variance reduction → each on bootstrap sample
│   ├── Boosting → sequential learners, bias reduction → each corrects prior's residuals
│   └── Stacking
│       ├── OOF predictions → out-of-fold base learner outputs → avoids leakage in meta-features
│       └── meta-learner → learns from base model outputs → often simple logistic regression
│
├── Hyperparameter Optimization (HPO)
│   ├── Grid Search → exhaustive, expensive, scales poorly → d dimensions = exponential trials
│   ├── Random Search → often better than grid, faster → covers space uniformly in expectation
│   ├── Bayesian Opt (Optuna/TPE)
│   │   ├── probabilistic surrogate model → approximates objective → Gaussian Process or TPE
│   │   └── acquisition function → balance explore vs exploit → EI, UCB, PI
│   └── Hyperband
│       ├── successive halving → kill bad configs early → multi-fidelity resource allocation
│       └── bracket scheduling → multi-fidelity budget allocation → ASHA variant is async
│
└── Conformal Prediction
    ├── distribution-free → no distributional assumptions → valid under exchangeability only
    ├── coverage guarantee → P(Y ∈ C(X)) ≥ 1 - α → marginal, not conditional coverage
    └── nonconformity scores → rank calibration set residuals → qhat = quantile(1-α) of scores
```

---

## PART 3 — DEEP LEARNING COMPONENTS

### 3.1 Building Blocks

```
Deep Learning Building Blocks
│
├── Activation Functions
│   ├── ReLU → max(0,x), dying neuron if all negative inputs → gradient = 0 for x<0
│   ├── GELU → smooth ReLU, used in Transformers/BERT/GPT → x·Φ(x), Φ is Gaussian CDF
│   ├── Swish → x·σ(x), self-gated, non-monotonic → outperforms ReLU in deep nets
│   └── Sigmoid/Tanh → saturate → vanishing gradient risk → σ' ≤ 0.25, tanh' ≤ 1
│
├── Weight Initialization
│   ├── Xavier/Glorot → for tanh/sigmoid → 2/(n_in + n_out) → preserves variance both ways
│   ├── He/Kaiming → for ReLU → 2/n_in, prevents dying neurons → accounts for half-zero mask
│   ├── Orthogonal → for RNNs → preserves gradient norms → W·Wᵀ = I
│   └── Zero init → anti-pattern → symmetry never breaks → all neurons learn same thing
│
├── Normalization Layers
│   ├── BatchNorm
│   │   ├── normalize over batch dimension → μ,σ from current mini-batch
│   │   └── train ≠ eval → uses running stats at inference → BN fails with small batches
│   ├── LayerNorm
│   │   ├── normalize over feature dimension → per-sample, batch-independent
│   │   └── batch-independent → used in Transformers → works for seq2seq and variable lengths
│   ├── RMSNorm
│   │   ├── LayerNorm without mean subtraction → RMS(x) = √(1/n·Σxᵢ²)
│   │   └── used in LLaMA / Gemma → faster → no mean recentering overhead
│   ├── GroupNorm → normalize within groups → small batches or detection tasks
│   └── InstanceNorm → per-sample per-channel → style transfer, not classification
│
└── Pre-Norm vs Post-Norm
    ├── Post-Norm → normalize after residual add → original Transformer → harder to train deep
    └── Pre-Norm → normalize before sublayer → more stable deep gradients → LLaMA uses this
```

### 3.2 Optimization & Regularization

```
Optimization & Regularization
│
├── Optimizers
│   ├── SGD → noisy gradients → implicit regularization effect → good generalization at scale
│   ├── Adam → adaptive LR per param via m/v moment estimates → mₜ/(√vₜ + ε) step
│   ├── AdamW → Adam + decoupled weight decay → better generalization → L2 ≠ weight decay in Adam
│   └── Schedule-Free → no LR schedule needed, online averaging → Primal averaging trick
│
├── Gradient Management
│   ├── Gradient clipping → clip by norm → prevents RNN explosion → g ← g·clip_val/‖g‖
│   └── Gradient accumulation → simulate large batch, low memory → avg over N micro-batches
│
├── Regularization Techniques
│   ├── Dropout → stochastic zero-out → reduces co-adaptation → scale by 1/(1-p) at train
│   ├── Weight decay (L2) → shrinks weights toward zero → penalizes large weight magnitudes
│   ├── BatchNorm as regularizer → noise via batch statistics → reduces need for dropout
│   └── Data augmentation → expand training distribution → flips, crops, color jitter, Mixup
│
└── Loss Functions
    ├── MSE → regression, penalizes large residuals heavily → L = (y - ŷ)²
    ├── Cross-Entropy → classification, measures distribution match → -Σyᵢ·log(pᵢ)
    ├── Focal Loss → imbalanced, down-weights easy examples → -(1-pₜ)^γ·log(pₜ)
    ├── Triplet Loss → metric learning, anchor-pos-neg margin → max(d(a,p)-d(a,n)+m, 0)
    └── Ranking Losses → NDCG optimization, information retrieval → LambdaRank approx
```

### 3.3 Backpropagation

```
Backpropagation
│
├── Forward Pass
│   ├── compute activations layer by layer → cache intermediate values for backward
│   └── compute scalar loss at output → single number for gradient to flow from
│
├── Backward Pass
│   ├── chain rule → ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w → product of local Jacobians
│   ├── accumulate gradients per parameter → sum across batch
│   └── optimizer → update weights via gradient step → w ← w - η·∂L/∂w
│
├── Vanishing Gradient
│   ├── cause → tanh/sigmoid saturate, small derivatives chain → product < 1 collapses
│   └── fix → ReLU activations + residual connections → gradient highway around layers
│
├── Exploding Gradient
│   ├── cause → deep RNNs multiply large Jacobians → product > 1 diverges
│   └── fix → gradient clipping by global norm → ‖g‖ > threshold → rescale
│
└── Residual Connections
    ├── output = F(x) + x → gradient highway through identity → ∂L/∂x = ∂L/∂(F+x) · (∂F/∂x + I)
    ├── enables very deep networks → ResNet / Transformers → 100+ layers viable
    └── gradient flows directly → bypasses vanishing problem → I term always passes gradient
```

### 3.4 Attention Mechanism

```
Attention Mechanism
│
├── Scaled Dot-Product Attention
│   ├── Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V → O(N²·d) time and space
│   ├── scale by √d_k → prevent softmax saturation → large d_k pushes dot products to extremes
│   └── queries match keys → retrieve weighted values → soft dictionary lookup
│
├── Multi-Head Attention (MHA)
│   ├── h parallel attention heads → different subspaces → each head dim = d_model/h
│   ├── concat + project → combine head outputs → W_O·concat(head₁,...,headₕ)
│   └── captures diverse relationship patterns → syntactic in one head, semantic in another
│
├── Cross-Attention
│   ├── Q from decoder, K/V from encoder → decoder queries encoder memory
│   └── decoder attends to encoder representations → core of seq2seq translation
│
├── Efficient Attention Variants
│   ├── MQA (Multi-Query) → one K/V head, many Q heads → reduces KV cache by factor h
│   ├── GQA (Grouped-Query) → grouped K/V sharing → inference speedup → quality between MHA/MQA
│   └── FlashAttention
│       ├── tiled SRAM computation → avoids HBM roundtrips → IO-aware algorithm, not sparse
│       └── memory O(N) not O(N²) → longer context feasible → same exact output as standard
│
├── Sparse Attention Patterns
│   ├── Sparse Attention → attend only to subset of tokens → O(N√N) complexity
│   │   └── BigBird, Longformer → combine local + global + random attention patterns
│   └── Sliding Window Attention → each token attends to fixed local window w
│       ├── O(N·w) complexity → w ≪ N → used in Mistral
│       └── NOTE: distinct from FlashAttention — FlashAttention is IO-aware full attention impl
│
└── KV-Cache
    ├── store past K/V tensors → skip recomputation → linear memory growth with seq length
    ├── autoregressive inference speedup → each token cheaper → O(1) per step vs O(N)
    └── memory grows with sequence length → eviction strategies needed → sliding window eviction
```

---

## PART 4 — DL ARCHITECTURES

### 4.1 Core Architectures

```
DL CORE ARCHITECTURES
│
├── CNNs
│   ├── Conv Layer → local receptive field + weight sharing → translation equivariance
│   ├── Pooling → spatial downsampling, translation invariance → max pool or avg pool
│   ├── ResNet → skip connections prevent vanishing gradients → identity shortcut F(x)+x
│   ├── EfficientNet → compound scaling (depth + width + resolution) → NAS-derived φ coefficients
│   └── ConvNeXt → CNN redesigned with ViT principles → depthwise conv + GELU + LayerNorm
│
├── RNNs
│   ├── LSTM
│   │   ├── Forget Gate → discard irrelevant past state → fₜ = σ(Wf·[hₜ₋₁,xₜ]+b)
│   │   ├── Input Gate → write new info to cell → iₜ·c̃ₜ added to cell state
│   │   ├── Output Gate → expose cell state selectively → oₜ = σ(Wo·[hₜ₋₁,xₜ]+b)
│   │   └── Cell State → long-range dependency carrier → additive updates avoid vanishing
│   └── GRU
│       ├── Reset Gate → how much past to forget → r = σ(Wr·[h,x])
│       └── Update Gate → simplified LSTM, fewer params → z controls interpolation h vs h̃
│
├── Transformer
│   ├── Self-Attention → query-key-value dot-product scores → O(N²) but parallelizable
│   ├── Multi-Head → attend multiple representation subspaces → h=8 or 16 typical
│   ├── FFN → position-wise two-layer projection → expand 4× then contract → ReLU/GELU
│   └── Parallelizable → no sequential bottleneck (vs RNNs) → train on full sequence at once
│
└── ViT (Vision Transformer)
    ├── Patch Embedding → image split into fixed-size tokens → 16×16 patches, linear proj
    ├── [CLS] Token → global classification representation → prepended, attended by all patches
    └── Pure Transformer → no CNN inductive bias → needs more data or strong augmentation
```

### 4.2 Generative Models

```
GENERATIVE MODELS
│
├── VAE (Variational Autoencoder)
│   ├── Encoder → outputs μ, σ (distribution params) → q_φ(z|x) approximates true posterior
│   ├── Reparameterization Trick → z = μ + σ·ε (differentiable sampling) → ε~N(0,I)
│   ├── Decoder → reconstruct x from z → p_θ(x|z)
│   └── ELBO Loss → reconstruction loss − KL divergence → KL(q‖p) keeps z near prior
│
├── GAN (Generative Adversarial Network)
│   ├── Generator → produce samples to fool discriminator → min log(1-D(G(z)))
│   ├── Discriminator → distinguish real vs fake → max log D(x) + log(1-D(G(z)))
│   ├── Minimax Game → adversarial training objective → Nash equilibrium at p_g = p_data
│   ├── Mode Collapse → generator ignores latent diversity → produces subset of modes
│   └── Training Instability → vanishing gradients for generator → D too strong → no signal
│
├── WGAN (Wasserstein GAN)
│   ├── Wasserstein Distance → smoother gradient landscape → meaningful even when supports disjoint
│   ├── Lipschitz Constraint → enforced via gradient penalty → ‖∇D(x̂)‖ ≈ 1 at interpolated x̂
│   └── Stable Training → no mode collapse / log-saturation → gradient always informative
│
├── StyleGAN
│   ├── Mapping Network → z → w (disentangled latent space) → 8-layer MLP
│   ├── AdaIN → style injected at each resolution layer → normalize then scale+shift by w
│   └── Progressive Growing → coarse-to-fine training → 4×4 → 8×8 → ... → 1024×1024
│
└── Diffusion Models
    ├── Forward Process (q) → gradually add Gaussian noise → q(xₜ|xₜ₋₁) = N(√(1-β)xₜ₋₁, βI)
    ├── Reverse Process (p_θ) → U-Net denoise step-by-step → predict noise ε_θ(xₜ,t)
    ├── DDPM → discrete T steps, slower sampling → T=1000, ~seconds per image
    ├── DDIM → deterministic, fewer steps, faster inference → 50 steps vs 1000, no randomness
    └── LDM (Latent Diffusion) → operate in latent space → cheaper compute → Stable Diffusion
```

### 4.3 Transfer Learning

```
TRANSFER LEARNING
│
├── Pre-Training → learn general features on large dataset → ImageNet / C4 / The Pile
│
├── Feature Extraction
│   ├── Freeze Backbone → no gradient through base model → only compute forward pass
│   └── Train Head Only → fast, less data required → avoids catastrophic forgetting
│
├── Fine-Tuning
│   ├── Unfreeze All Layers → adapt to target domain → needs enough target data
│   └── Discriminative LR → lower LR for early layers, higher for head → ULMFiT trick
│
├── Domain Adaptation
│   ├── DANN (Domain-Adversarial NN)
│   │   ├── Gradient Reversal Layer → flip gradient for domain classifier → -λ·∂L_domain/∂θ
│   │   └── Feature Alignment → domain-invariant representations → fool domain discriminator
│   └── CORAL → match second-order statistics across domains → minimize Frobenius ‖Cs - Ct‖²
│
└── Few-Shot / Zero-Shot
    ├── ProtoNets → class prototype = mean embedding → nearest prototype → inductive bias works
    ├── MAML → learn initialization → fast adapt via few gradient steps → second-order gradients
    └── Zero-Shot (CLIP) → align image + text embeddings → no target labels → cosine sim at test
```

---

## PART 5 — NLP

### 5.1 Text Representations

```
TEXT REPRESENTATIONS
│
├── Tokenization
│   ├── BPE (Byte-Pair Encoding)
│   │   ├── Merge most frequent byte pairs iteratively → greedy compression
│   │   └── Used in → GPT-2, LLaMA → 32K–128K vocab size typical
│   ├── WordPiece
│   │   ├── Merge by likelihood maximization → P(ab)/P(a)P(b)
│   │   ├── ## prefix → continuation subword token → marks non-initial subword
│   │   └── Used in → BERT → 30K vocab
│   └── SentencePiece
│       ├── Operates on raw text, no pre-tokenization → language-agnostic
│       └── Used in → T5, mT5 (language-agnostic) → handles whitespace as token
│
├── Static Embeddings
│   ├── Word2Vec
│   │   ├── Skip-Gram → predict context from word → better for rare words
│   │   └── CBOW → predict word from context window → faster training
│   ├── GloVe → factorize global co-occurrence matrix → log(Pᵢⱼ) = wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ
│   └── FastText → character n-grams → handles OOV words → sum of n-gram vectors
│
└── Contextual Embeddings
    ├── ELMo → BiLSTM hidden states, context-dependent → task-weighted layer combination
    └── BERT → bidirectional Transformer, MLM pre-training → [MASK] token prediction
```

### 5.2 Sequence Modeling

```
SEQUENCE MODELING
│
├── Seq2Seq (RNN-based)
│   ├── Encoder → compress input to fixed context vector → bottleneck loses long-range info
│   └── Decoder → generate output (bottleneck problem!) → attention solves this
│
├── Bahdanau Attention
│   ├── Attend over all encoder hidden states per decode step → context = Σαᵢhᵢ
│   └── Precursor to Transformer self-attention mechanism → score = vᵀtanh(W1h + W2s)
│
├── Beam Search
│   ├── Maintain top-K hypotheses at each decode step → K=4 or 5 common
│   ├── Better coverage than greedy decoding → avoids locally optimal token choices
│   └── Length Penalty → avoid bias toward short sequences → divide by |y|^α, α~0.6
│
└── NER (Named Entity Recognition)
    ├── BIO Tagging → B=begin, I=inside, O=outside → IOB2 standard
    └── BERT + Token Classification Head → state-of-the-art → per-token softmax over tag set
```

### 5.3 NLP Tasks

```
NLP TASKS
│
├── Text Classification
│   ├── TF-IDF + Logistic Regression → fast interpretable baseline → good for short docs
│   └── BERT Fine-Tune → 2e-5 LR, ~3 epochs, strong accuracy → [CLS] token as sequence repr
│
├── Summarization
│   ├── BART → denoising pre-training → abstractive summaries → span infilling + token deletion
│   └── TextRank → PageRank on sentence similarity graph → extractive → no training needed
│
├── Evaluation Metrics
│   ├── BLEU → precision n-gram overlap (machine translation) → brevity penalty for short output
│   ├── ROUGE → recall n-gram overlap (summarization) → ROUGE-L uses LCS
│   └── BERTScore → contextual cosine similarity, semantic aware → matches tokens via BERT embeds
│
├── Semantic Similarity
│   ├── SBERT → siamese BERT + mean pooling → cosine similarity → 100× faster than cross-encoder
│   └── NLI → entailment / neutral / contradiction → zero-shot templates → MNLI fine-tune
│
└── Coreference Resolution
    ├── Span Detection → identify candidate mention spans → all spans up to max length
    └── Clustering → group coreferring mentions together → higher-order inference
```

### 5.4 Advanced NLP

```
ADVANCED NLP
│
├── Summarization Faithfulness
│   ├── FactCC → NLI-based factual consistency check → entailment between doc and summary claim
│   └── SummaC → segment-level consistency scoring → chunk source, check each claim
│
├── Dependency Parsing
│   ├── Head-Dependent syntactic relations in sentence → directed tree structure
│   └── spaCy → fast production parser, arc-eager transition → O(N) parsing
│
├── Relation Extraction
│   └── Identify semantic relations between entity pairs → entity marker tokens + classifier
│
└── Dialogue Systems
    ├── Pipeline → NLU → DST → Policy → NLG
    │   ├── NLU → intent + slot detection → semantic frame parsing
    │   ├── DST → belief state tracker over turns → accumulates slot values
    │   ├── Policy → select system action → dialogue act selection
    │   └── NLG → surface action as natural language → template or neural
    └── End-to-End → fine-tune LLM on conversation data → TOD-BERT, dialogue fine-tuning
```

---

## PART 6 — COMPUTER VISION

### 6.1 Detection & Segmentation

```
DETECTION & SEGMENTATION
│
├── Object Detection
│   ├── YOLO → single-shot anchor-based → real-time inference → 30+ FPS, trades recall for speed
│   ├── Faster R-CNN → RPN + RoI Pooling → high accuracy → two-stage, slower but precise
│   └── DETR → Transformer + bipartite matching → no anchors/NMS → end-to-end trainable
│
├── Semantic Segmentation
│   ├── FCN → replace FC layers with conv → dense predictions → upsample via transposed conv
│   ├── U-Net → skip connections between encoder and decoder → preserves spatial detail
│   ├── DeepLab → atrous conv + ASPP → multi-scale context → dilation rate r expands RF
│   └── SegFormer → hierarchical Transformer + lightweight MLP decoder → no positional encoding
│
├── Instance Segmentation
│   ├── Mask R-CNN → adds mask head to Faster R-CNN + RoIAlign → per-instance binary masks
│   │   └── RoIAlign → no quantization error (vs RoI Pooling) → bilinear interpolation at coords
│   └── SOLOv2 → grid-based, no region proposals needed → faster than Mask R-CNN
│
└── Panoptic Segmentation
    ├── Stuff → amorphous regions (semantic) → sky, grass, road
    ├── Things → countable objects (instance) → person, car, dog
    └── PQ (Panoptic Quality) = SQ × RQ → SQ=seg quality, RQ=recognition quality
```

### 6.2 Pose Estimation

```
POSE ESTIMATION
│
├── OpenPose (Bottom-Up)
│   ├── Part Affinity Fields → encode limb orientations → vector field per limb type
│   └── Assemble keypoints → no person detector needed → Hungarian matching
│
├── HRNet (Top-Down)
│   ├── Person Detector → crop each person first → bounding box from detector
│   ├── High-Resolution Maintained → no upsampling artifacts → parallel branches at all scales
│   └── Multi-Scale Fusion → parallel resolution streams → repeated across stages
│
├── ViTPose
│   ├── Transformer Backbone → scalable feature extraction → ViT-B/L/H variants
│   └── Multi-Task Capable → unified pose estimation framework → single model, multiple datasets
│
└── Metrics
    ├── PCKh → % keypoints within threshold of head size → threshold = 0.5 × head diameter
    └── OKS → per-keypoint weighted distance similarity → COCO standard, sigmas per joint
```

### 6.3 Metric Learning & Retrieval

```
METRIC LEARNING & RETRIEVAL
│
├── Loss Functions
│   ├── Contrastive Loss → positives close, negatives beyond margin → L = (1-y)d² + y·max(m-d,0)²
│   ├── Triplet Loss → anchor-positive close, anchor-negative far → max(d(a,p)-d(a,n)+m, 0)
│   │   └── Semi-Hard Mining → critical for training stability → negatives inside margin+anchor
│   └── ArcFace → additive angular margin → more discriminative than softmax → m added to θy
│
├── Approximate Nearest Neighbor
│   └── FAISS → IVF index → billion-scale ANN at speed → GPU support, quantization options
│
├── Retrieval Metrics
│   ├── Recall@K → fraction of true matches in top-K → primary metric for retrieval tasks
│   └── mAP → mean average precision across queries → area under precision-recall per query
│
└── Re-ID (Re-Identification)
    ├── Cross-Camera Identity Matching → same identity across disjoint camera views
    └── Challenges → viewpoint change, occlusion, illumination → domain gap between cameras
```

### 6.4 Video & 3D Vision

```
VIDEO & 3D VISION
│
├── Video Understanding
│   ├── Two-Stream → RGB stream + optical flow stream → late fusion → Simonyan & Zisserman
│   ├── I3D → inflate 2D conv kernels to 3D → ImageNet init possible → k²×k spatiotemporal
│   ├── SlowFast → slow pathway (semantics) + fast pathway (motion) → α=8 frame rate ratio
│   ├── TimeSformer → divided space-time attention → efficient Transformer → factored attention
│   └── VideoMAE → masked autoencoder, 90% masking ratio → strong pretraining → temporal redundancy
│
└── 3D Vision
    ├── PointNet
    │   ├── Per-Point MLP → process each point independently → shared weights across points
    │   └── Global Max Pool → permutation invariant aggregation → invariant to point order
    ├── PointNet++
    │   ├── Hierarchical Local Grouping → neighborhood context → ball query radius
    │   └── Multi-Scale Features → robust to sampling density → MSG/MRG grouping
    ├── NeRF (Neural Radiance Field)
    │   ├── Implicit Neural Scene Representation → MLP maps (x,y,z,θ,φ) → (RGB, σ)
    │   └── Novel View Synthesis → volume rendering integral → C = ∫T(t)·σ(t)·c(t)dt
    └── 3D Gaussian Splatting
        ├── Explicit 3D Gaussian Primitives → scene representation → position, cov, opacity, color
        └── Real-Time Rendering → rasterization-based, no ray marching → 100+ FPS novel views
```

---

## PART 7 — LARGE LANGUAGE MODELS

### 7.1 LLM Architecture

```
LLM Architecture
│
├── Transformer Decoder
│   ├── causal masking → attend only to past tokens → lower-triangular attention mask
│   └── autoregressive → one token at a time → p(xₜ|x₁,...,xₜ₋₁)
│
├── Positional Encodings
│   ├── RoPE → rotate Q/K by position index → RoPE(q,m) = q·e^(imθ) in complex space
│   │   └── used in LLaMA → relative position via rotation → extrapolates beyond training length
│   └── ALiBi → linear bias added to attention score → -|i-j|·m per head
│       └── no vector modification → long context friendly → zero cost at train time
│
├── Attention Variants
│   ├── MHA → multi-head, separate K/V per head → d_kv per head = d_model/h
│   ├── MQA → all heads share single K/V → reduces KV cache by factor h
│   └── GQA → grouped heads share K/V sets → G groups, h/G heads share K/V
│       └── inference speedup → less KV memory → Llama-3, Mistral use this
│
├── Mixture of Experts (MoE)
│   ├── top-K routing → sparse expert activation → only K of N FFN experts run per token
│   ├── scale params without proportional compute → 8×7B Mixtral has 46.7B params, 12.9B active
│   └── examples: DeepSeek, Mixtral → load balancing loss prevents expert collapse
│
└── KV-Cache
    ├── store past K/V tensors across steps → avoid recomputing all past positions
    └── O(1) per new token at inference → total mem = 2·L·h·d_k·N bytes for N tokens
```

### 7.2 LLM Training

```
LLM Training
│
├── Pre-training
│   ├── next-token prediction → cross-entropy loss → teacher forcing at train time
│   └── Chinchilla scaling law → 20 tokens per param optimal → N_tokens ≈ 20·N_params
│
├── Supervised Fine-Tuning (SFT)
│   └── train on human demonstrations → labeled (prompt, response) → same CE loss, curated data
│
├── RLHF
│   ├── reward model → scores responses from human prefs → Bradley-Terry pairwise model
│   ├── PPO → policy gradient with clipped objective → clip(r, 1-ε, 1+ε)·Â
│   └── KL penalty → stay close to SFT baseline → β·KL(π‖π_sft) added to reward
│
├── DPO (Direct Preference Optimization)
│   ├── no RL loop → closed-form preference loss → rearranges RLHF objective analytically
│   └── simpler, stable → replaces reward model + PPO → L = -log σ(β(log π/π_ref)_w - (...)_l)
│
├── GRPO (Group Relative Policy Optimization)
│   ├── used in DeepSeek-R1 → samples group of G responses per prompt
│   ├── normalize rewards within group → Aᵢ = (rᵢ - mean(r)) / std(r) → no separate value model
│   └── eliminates critic network → reduces memory and compute vs PPO
│
├── Constitutional AI / RLAIF
│   ├── critique-revise loop → model critiques own outputs using a constitution
│   ├── AI generates preference labels → no human annotation needed at scale
│   └── scalable oversight → model quality improves without bottlenecking on human feedback
│
├── SimPO (Simple Preference Optimization)
│   ├── reference-free DPO variant → no reference model needed at inference
│   ├── uses average log-prob as implicit reward → r = (1/|y|)·log π(y|x) → length-normalized
│   └── adds target reward margin γ → L = -log σ(β(r_w - r_l) - γ)
│
└── Efficient Fine-Tuning
    ├── LoRA → ΔW = A·B (low-rank), <1% trainable params → rank r ≪ d
    │   └── freeze base → only train A, B matrices → merge at inference: W' = W + αAB
    └── QLoRA → 4-bit quantized base + LoRA adapters → NF4 quantization + double quant
        └── fits 65B model on 48GB GPU → paged optimizers for memory spikes
```

### 7.3 LLM Inference & Applications

```
LLM Inference & Applications
│
├── Quantization
│   ├── INT8 / INT4 → weight-only or activation quant → 2-4× memory reduction
│   ├── GPTQ → post-training, layer-wise reconstruction → minimize ‖WX - Ŵx‖² per layer
│   └── AWQ → activation-aware weight quantization → protect salient weights per channel
│
├── Speculative Decoding
│   ├── draft model proposes multiple tokens → small cheap model generates candidates
│   └── main model verifies in parallel → net speedup → 2-3× without quality loss
│
├── Continuous Batching (vLLM)
│   ├── PagedAttention → KV cache in paged blocks → non-contiguous memory like OS paging
│   └── high GPU utilization → no wasted memory → iteration-level scheduling
│
├── RAG (Retrieval-Augmented Generation)
│   ├── retrieve relevant docs at query time → dense retrieval via FAISS or BM25
│   ├── ground generation with retrieved context → prepend to prompt
│   └── reduces hallucination → factual grounding → still needs faithfulness check
│
├── Agents
│   ├── tool use → call APIs, code, search → function calling / tool schemas
│   ├── planning: ReAct → reason + act interleaved → Thought: ... Act: ... Obs: ...
│   └── CoT → chain-of-thought → multi-step reasoning → few-shot or zero-shot prompting
│
└── Context Extension
    ├── YaRN → NTK-aware RoPE frequency scaling → interpolate position frequencies
    ├── FlashAttention → tiled O(N) memory attention → IO-aware, not sparse
    └── Longformer → sliding window + global tokens → O(N·w) complexity
```

### 7.4 LLM Challenges

```
LLM Challenges
│
├── Hallucination
│   ├── model generates plausible but false content → high perplexity facts most risky
│   │
│   ├── Detection
│   │   ├── SelfCheckGPT → sample multiple → check consistency → stochastic sampling variance
│   │   └── FactScore → decompose → retrieve → verify per claim → Wikipedia-grounded
│   │
│   └── Mitigation
│       ├── RAG → ground in retrieved facts → reduces parametric reliance
│       ├── constrained decoding → Outlines → structured output → JSON grammar enforcement
│       └── calibration → align confidence to accuracy → temperature scaling post-hoc
│
└── Model Merging
    ├── task vector = fine-tuned weights − base weights → τ = θ_ft - θ_base
    │
    ├── Task Arithmetic → add/subtract task vectors directly → θ_new = θ_base + λ·Στᵢ
    ├── TIES → trim small deltas + elect sign + merge → reduces interference
    ├── DARE → randomly drop weight deltas before merge → sparsify task vectors
    └── SLERP → spherical interpolation between checkpoints → s(t) = sin((1-t)Ω)/sinΩ·θ₁ + ...
```

---

## PART 8 — REINFORCEMENT LEARNING

### 8.1 Core RL

```
Core RL
│
├── MDP → (S, A, P, R, γ) — formal environment model → Markov property: future ⊥ past | present
│
├── Bellman Equations
│   ├── V(s) = R + γ · max_a Q(s,a) → recursive value → basis for all DP/TD methods
│   ├── Q(s,a) = R(s,a) + γ · Σ P(s'|s,a) · max_a' Q(s',a') → action-value version
│   ├── TD(λ) → bridges TD(0) and MC via eligibility traces → λ=0 → TD(0), λ=1 → MC returns
│   │   └── eₜ(s) = γλeₜ₋₁(s) + 1[Sₜ=s] → credit assignment decays with λ and γ
│   └── GAE (Generalized Advantage Estimation) → λ-weighted sum of TD residuals
│       └── Â^GAE = Σ(γλ)ᵏ·δₜ₊ₖ → λ=0 → one-step TD advantage, λ=1 → MC advantage
│
├── Value-Based Methods
│   ├── Q-Learning → off-policy TD, tabular or approx → Q ← Q + α(r + γ·maxQ' - Q)
│   ├── DQN → experience replay + target network → stable → breaks correlation and non-stationarity
│   └── DDQN → decouple action selection from evaluation → reduces overestimation bias
│
├── Policy Gradient Methods
│   ├── REINFORCE → MC returns → high variance baseline → ∇J = E[G_t·∇log π(a|s)]
│   ├── PPO → clipped surrogate objective → stable updates → clip(r_t, 1-ε, 1+ε)·Â_t
│   └── SAC → entropy regularization → exploration bonus → maximize E[r] + α·H[π]
│
├── Actor-Critic
│   ├── separate value network (critic) + policy network (actor) → reduces variance vs REINFORCE
│   ├── critic computes Â or V(s) → baseline reduces gradient variance without bias
│   └── A3C/A2C → async or sync multi-worker → critic updates online, not MC
│
└── Model-Based RL
    ├── learn transition model of environment → p(s'|s,a) from collected data
    └── Dyna / MBPO → plan with model → data efficient → generate synthetic rollouts
```

### 8.2 Advanced RL

```
Advanced RL
│
├── Imitation Learning
│   ├── BC (Behavioral Cloning) → supervised on (s,a) pairs → simple but fragile
│   │   └── compounding error → small mistakes amplify → distribution shift from training
│   ├── DAgger → iteratively query expert on visited states → reduces covariate shift
│   └── GAIL → discriminator: expert vs policy rollout → matches occupancy measure
│       └── PPO as generator → adversarial imitation → imitates without explicit reward
│
├── Inverse RL (IRL)
│   └── MaxEntropy IRL → infer reward matching expert occupancy → max H[π] s.t. feature match
│
├── Multi-Agent RL (MARL)
│   ├── QMIX → monotone mixing of individual Q-functions → centralized training, decentralized exec
│   ├── MAPPO → centralized critic + decentralized actors → shared global state for critic
│   └── self-play → emergent strategies (AlphaGo) → curriculum of increasingly strong opponents
│
├── Hierarchical RL
│   ├── Options framework → (I, π_ω, β) — subgoal policies → β is termination condition
│   └── HER → relabel failed trajectory with achieved goal → sparse reward problem solver
│
└── Meta-RL
    ├── MAML → learn init for fast few-shot adaptation → θ* = argmin Σ L_τ(θ - α∇L_τ(θ))
    └── RL² → RNN hidden state encodes task identity → fast adaptation via recurrent memory
```

### 8.3 Sim-to-Real & Offline RL

```
Sim-to-Real & Offline RL
│
├── Sim-to-Real Transfer
│   ├── domain randomization → vary physics params → robust policy → friction, mass, latency
│   ├── system identification → fit sim to real trajectories → MLE on observed transitions
│   └── RMA → privileged sim policy + real-time adapt module → distill into real-deployable net
│
└── Offline RL
    ├── CQL → penalize out-of-distribution action Q-values → Q(s,a) ← Q(s,a) - α·E[Q(s,·)]
    ├── IQL → expectile regression on V(s) → no OOD queries → τ-expectile of Q distribution
    └── Decision Transformer → RL as sequence modeling → token: (R̂, s, a) triples
        └── return-conditioned autoregressive generation → specify desired return at inference
```

---

## PART 9 — RECOMMENDER SYSTEMS & GNNs

### 9.1 Recommender Systems

```
Recommender Systems
│
├── Collaborative Filtering
│   ├── user-user / item-item → cosine similarity on ratings → memory-based, no latents
│   └── Matrix Factorization → UV^T ≈ R → latent factors → r̂ᵤᵢ = uᵤᵀvᵢ + bᵤ + bᵢ
│       └── optimized via ALS or SGD → ALS closed-form, SGD more scalable
│
├── Content-Based
│   └── item features → user profile → cosine similarity match → cold-start friendly
│
├── Two-Tower Model
│   ├── query tower + item tower → separate encoders → trained with in-batch negatives
│   └── dot product score → scalable ANN retrieval → FAISS at serving time
│
├── Learning-to-Rank
│   ├── pointwise → MSE on relevance scores → ignores inter-item ordering
│   ├── pairwise → RankNet, BPR → prefer relevant over not → σ(sᵢ - sⱼ) > 0.5
│   └── listwise → LambdaRank → NDCG-weighted gradient → ΔNDCGᵢⱼ scales gradient
│
├── Cold-Start
│   ├── feature-based → use item/user attributes → content features as proxy for interactions
│   ├── popularity fallback → recommend trending items → exploration via bandit
│   └── meta-learning → fast adapt to new user/item → few-shot collaborative filtering
│
├── Session-Based
│   ├── GRU4Rec → RNN on click sequence → next item → hidden state = session representation
│   └── BERT4Rec → masked item prediction → bidirectional → captures future context in session
│
└── GNN-Based
    ├── PinSage → GraphSAGE on item-item graph → Pinterest production system
    └── LightGCN → simplified GCN, no feature transform → only neighborhood aggregation
```

### 9.2 Graph Neural Networks

```
Graph Neural Networks
│
├── Message Passing Framework
│   ├── aggregate neighbor messages → update node embedding → hᵥ ← AGG({hᵤ : u ∈ N(v)})
│   └── stack L layers → captures L-hop neighborhood → deeper = larger receptive field
│
├── GCN → spectral: Aˆ·H·W (normalized adjacency) → D^{-1/2}AD^{-1/2}
│   └── transductive → fixed graph at train time → new nodes require retrain
│
├── GraphSAGE → inductive: sample + aggregate neighbors → fixed-size neighborhood sampling
│   ├── aggregate: mean / LSTM / pooling → each type has different expressiveness
│   └── generalizes to unseen nodes at test time → inductive capability
│
├── GAT → attention weights per neighbor edge → αᵢⱼ = softmax(a([Whᵢ‖Whⱼ]))
│   └── different importance to different neighbors → multi-head attention on graph
│
├── GIN → sum aggregation → max expressiveness → hᵥ = MLP((1+ε)·hᵥ + Σhᵤ)
│   └── as powerful as WL graph isomorphism test → provably most expressive MPNN
│
├── Knowledge Graphs
│   ├── TransE → h + r ≈ t in embedding space → L1/L2 distance scoring
│   └── RotatE → relation as rotation in complex space → h ∘ r = t, |rᵢ|=1
│
├── Dynamic Graphs
│   ├── TGN → memory module + temporal message passing → compressed interaction history
│   └── TGAT → time encoding via random Fourier features → cos/sin time embedding
│
└── Graph Generation
    ├── VGAE → encode → sample z → decode adjacency → A_hat = sigmoid(ZZᵀ)
    └── GraphRNN → sequential node/edge autoregressive → generates adjacency row by row
```

---

## PART 10 — PRODUCTION ML

### 10.1 MLOps Pipeline

```
MLOps Pipeline
│
├── Stages
│   ├── Data → feature engineering → training → no shortcuts in data quality
│   └── evaluation → deployment → monitoring → feedback loop back to data
│
├── Data Versioning
│   └── DVC → Git-like versioning for datasets/models → .dvc files track remote storage
│
├── Feature Store
│   ├── offline → batch features for training → consistent with online at serving
│   ├── online → low-latency features for serving → Redis/DynamoDB backed
│   └── Feast → unified offline + online feature store → prevents training-serving skew
│
├── Experiment Tracking
│   └── MLflow → log params + metrics + artifacts → compare runs, register best model
│
├── CI/CD for ML
│   └── retrain trigger → data drift or schedule → automated pipeline with eval gate
│
└── Model Registry
    └── versioning + stage transitions → staging → production → rollback by version tag
```

### 10.2 Deployment Patterns

```
Deployment Patterns
│
├── Blue/Green → swap 100% traffic instantly → zero downtime → requires 2× resources
│
├── Canary → route % traffic to new model → gradual rollout → 5% → 25% → 100%
│
├── Shadow Mode → mirror traffic → compare → no user impact → validate before promoting
│
├── A/B Test
│   ├── split traffic → control vs treatment → randomize at user level
│   ├── t-test / z-test → statistical significance → p < 0.05 threshold
│   └── MDE → min detectable effect → sample size planning → n ∝ σ²/(δ²)
│
└── Rollback
    └── automated trigger → latency spike or metric drop → p99 latency or F1 threshold
```

### 10.3 Serving & Latency

```
Serving & Latency Optimization
│
├── Quantization
│   └── INT8 / FP16 → 2-4× inference speedup → PTQ or QAT → accuracy tradeoff to measure
│
├── ONNX Export
│   └── hardware-optimized runtime → cross-platform → graph optimizations: fusion, pruning
│
├── TensorRT
│   └── GPU kernel fusion → minimize memory transfers → layer fusion + precision calibration
│
├── Dynamic Batching
│   └── amortize per-request overhead → higher throughput → queue requests for N ms window
│
├── Serving Frameworks
│   ├── TorchServe → PyTorch multi-model serving → handler API, model archiver
│   └── Triton → multi-framework, concurrent execution → gRPC + HTTP2, ensemble pipelines
│
└── Training-Serving Skew
    └── use same feature pipeline → feature store solves this → same transform code path
```

### 10.4 Monitoring

```
Monitoring
│
├── Data Drift Detection
│   ├── PSI (Population Stability Index) → numerical shift → PSI = Σ(A-E)·ln(A/E), >0.2 = shift
│   ├── KS test → distribution shift → continuous features → max CDF difference statistic
│   └── chi-squared → categorical feature distribution shift → observed vs expected frequencies
│
├── Concept Drift
│   ├── model performance degrades over time → target relationship changes, not just input dist
│   └── trigger → scheduled or threshold-based retraining → monitor upstream label proxy
│
├── Logging
│   └── log predictions + features → compare to training dist → basis for all drift detection
│
├── Shadow Scoring
│   ├── run new model in parallel, log outputs, no user impact → safe pre-promotion validation
│   └── compare output distributions before promoting → KL divergence or PSI on score dist
│
├── SHAP-based Explanation Logging
│   ├── log per-prediction feature attributions → SHAP values per feature per request
│   └── detect which features drift in importance → feature attribution shift = early drift signal
│
├── Model Lineage / Model Cards
│   ├── track training data, eval metrics, intended use, limitations per version
│   └── enables reproducibility + auditability → required for regulated industries
│
└── Slice-Based Evaluation
    ├── disaggregate metrics by subgroup → age, gender, geography cohorts
    └── catch hidden failures → fairness, edge cases → overall metric can mask subgroup regression
```

---

## PART 11 — EMERGING TOPICS

### 11.1 Privacy, Safety & Ethics

```
Privacy, Safety & Ethics
│
├── Differential Privacy
│   ├── (ε,δ)-DP → bound info leaked per individual → P[M(D)∈S] ≤ eᵉ·P[M(D')∈S] + δ
│   ├── add calibrated Gaussian or Laplace noise → σ ∝ sensitivity/ε
│   └── DP-SGD → clip gradients + noise → Opacus library → privacy budget accounting
│
├── Federated Learning
│   ├── FedAvg → local SGD + aggregate weights centrally → weighted by dataset size
│   │   └── data stays on device → privacy preserving → no raw data leaves client
│   └── FedProx → proximal term → handles heterogeneous clients → μ/2·‖w-wᵍ‖² penalty
│
├── Adversarial Robustness
│   ├── FGSM → single step: x + ε·sign(∇L) → fast but weak attack
│   ├── PGD → multi-step FGSM with projection → stronger → k steps inside ε-ball
│   ├── adversarial training → train on PGD examples → best empirical defense, costly
│   └── randomized smoothing → certifiable L2 robustness → majority vote over Gaussian noise
│
├── Red-Teaming
│   ├── structured adversarial probing → find harmful outputs before deployment
│   └── human red-teamers + automated methods → coverage across harm categories
│
├── RLAIF Safety Alignment
│   ├── Constitutional AI pipeline → AI self-critique using written principles
│   └── AI generates preference labels → scalable oversight without human bottleneck
│
├── Jailbreak Taxonomy
│   ├── prompt injection → embed instructions in user/context data → override system prompt
│   ├── role-play exploits → "pretend you are DAN" → bypass safety via persona framing
│   ├── encoding tricks → base64, rot13, l33tspeak → bypass token-level filters
│   └── multi-turn manipulation → gradually escalate context → build compliant precedent
│
└── Fairness
    ├── demographic parity → equal positive prediction rates → P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    ├── equalized odds → equal TPR and FPR across groups → P(Ŷ=1|Y=y,A=a) same for all a
    ├── impossibility theorem → can't satisfy all simultaneously → Chouldechova 2017
    └── bias mitigation
        ├── pre-process → reweight or resample data → reweighing algorithm
        ├── in-process → adversarial debiasing during training → gradient reversal on group
        └── post-process → threshold calibration per group → equalize FPR/TPR post hoc
```

### 11.2 Continual & Meta Learning

```
Continual & Meta Learning
│
├── Catastrophic Forgetting
│   └── new task training overwrites old task weights → gradient descent destroys prior optima
│
├── Mitigation Strategies
│   ├── EWC → penalize changes to Fisher-important weights → Ω = diag of Fisher Information
│   ├── Experience Replay → store + interleave old samples → episodic memory buffer
│   └── Progressive Nets → new columns per task + lateral links → old columns frozen
│
└── Neural Architecture Search (NAS)
    ├── DARTS → differentiable search → continuous relaxation → α weights over operations
    └── Hyperband → successive halving → early stopping budget → ASHA for async version
```

### 11.3 Probabilistic & Optimization Theory

```
Probabilistic & Optimization Theory
│
├── Probabilistic Graphical Models
│   ├── Bayesian Networks → DAG + CPT → factored joint dist → P(X) = ΠP(Xᵢ|parents(Xᵢ))
│   ├── HMM
│   │   ├── Viterbi → decode most likely state sequence → dynamic programming O(T·K²)
│   │   ├── Baum-Welch → learn emission + transition params → forward-backward EM
│   │   └── forward algorithm → evaluate observation prob → α_t(i) = P(o₁...oₜ, qₜ=i)
│   ├── CRF → discriminative MRF → NER, sequence labeling → P(Y|X) not P(X,Y)
│   │   └── globally optimal decoding via Viterbi → no label-bias problem
│   └── LDA → plate model → collapsed Gibbs sampling → topic coherence as eval
│
├── Natural Gradient
│   └── steepest descent in distribution space → F^{-1}·∇L → F = Fisher Information Matrix
│
├── SVRG (Stochastic Variance Reduced Gradient)
│   ├── periodic full gradient snapshot → reduce variance → μ̃ = ∇f(w̃) every m steps
│   └── O(1/T) convergence → faster than SGD → linear convergence for strongly convex
│
└── SAM (Sharpness-Aware Minimization)
    ├── find worst perturbation in weight neighborhood → ε̂ = ρ·∇L/‖∇L‖
    ├── minimize loss at perturbed point → flat minima → L(w + ε̂) minimized
    └── flat minima → better generalization on test set → loss landscape geometry matters
```

---

## PART 12 — EMERGING TRENDS

### 12.1 State Space Models

```
STATE SPACE MODELS
│
├── S4 (Structured State Space)
│   ├── continuous-time SSM → xₜ' = Axₜ + Buₜ, yₜ = Cxₜ + Duₜ → state space ODE
│   ├── HiPPO matrix → initializes A for long-range memory → polynomial projection
│   └── O(N log N) via convolution → convolve input with learned kernel → parallelizable
│
├── Mamba (Selective SSM)
│   ├── input-dependent SSM → A, B, C, Δ depend on input xₜ → selectivity
│   ├── hardware-aware algorithm → parallel scan in SRAM → avoids materializing state
│   ├── O(N) recurrent inference → O(N log N) parallel training → best of both worlds
│   └── outperforms Transformer on language at equal params up to ~1B scale
│
└── Hybrid Mamba-Transformer
    ├── interleave SSM layers + attention layers → capture local + global patterns
    └── examples: Jamba, Zamba → better efficiency at long context than pure Transformer
```

### 12.2 Large Reasoning Models

```
LARGE REASONING MODELS
│
├── Test-Time Compute Scaling
│   ├── chain-of-thought (CoT) → generate intermediate reasoning steps → improves accuracy
│   ├── Best-of-N → sample N responses → pick best per reward model → exploits stochasticity
│   ├── MCTS for LLMs → tree search over reasoning paths → process reward at each node
│   └── key insight: more compute at inference ↔ better accuracy (orthogonal to model size)
│
├── Process Reward Models (PRMs)
│   ├── score each reasoning step → step-level reward → not just final answer
│   └── ORM (outcome) vs PRM → PRM better at catching early errors → more credit assignment
│
├── DeepSeek-R1
│   ├── trained via GRPO with verifiable rewards → math + code → no human preference labels
│   ├── emergent self-reflection → "aha moments" → model learns to revise reasoning chains
│   └── long CoT at inference → extended thinking budget → o1-comparable benchmark scores
│
└── OpenAI o1/o3 Style
    ├── extended thinking → hidden reasoning chain before final answer
    └── MCTS + PRM → search over reasoning tree → select best path
```

### 12.3 Agentic AI Systems

```
AGENTIC AI SYSTEMS
│
├── Core Loop
│   └── observe → think → act → observe → repeat → ReAct pattern
│
├── Tool Use
│   ├── function calling → structured JSON tool invocation → model outputs schema-valid call
│   ├── code interpreter → execute Python, handle outputs → sandboxed runtime
│   └── web search, calculator, APIs → extend LLM with real-world grounding
│
├── Planning Paradigms
│   ├── ReAct → interleaved Thought + Action + Observation → trace visible to model
│   ├── Plan-and-Execute → separate planner + executor → replanning on failure
│   └── Tree of Thoughts → explore multiple reasoning branches → BFS/DFS over thought space
│
├── Memory Systems
│   ├── in-context → conversation history, retrieved docs → limited by context window
│   ├── external → vector store, key-value → persistent across sessions → RAG-based recall
│   └── parametric → fine-tuned into weights → implicit, hard to update
│
├── Multi-Agent Coordination
│   ├── role specialization → planner, coder, critic, executor agents → division of labor
│   ├── debate / critic → agents critique each other → improve answer quality
│   └── evaluation challenges → no ground truth for open-ended tasks → LLM-as-Judge
│
└── Interop Protocols
    ├── MCP (Model Context Protocol) → agent ↔ TOOLS/data sources → server exposes tools/resources/prompts
    │   └── standardizes the tool layer → one integration works across hosts
    └── A2A (Agent-to-Agent) → agent ↔ AGENT across org/vendor boundaries
        ├── Agent Card → capability discovery → who can do what
        └── task lifecycle + streaming updates → long-running delegated work
    → MCP is vertical (down to tools), A2A is horizontal (across peers) → complementary, not competing
```

### 12.4 Advanced RAG & Memory

```
ADVANCED RAG
│
├── Retrieval Improvements
│   ├── HyDE → hypothetical document embeddings → generate fake answer, embed, retrieve similar
│   ├── FLARE → forward-looking active retrieval → retrieve only when uncertain mid-generation
│   └── GraphRAG → build knowledge graph from corpus → entity + community-level retrieval
│
├── Reranking
│   ├── two-stage → fast ANN retrieval (bi-encoder) → rerank with cross-encoder → quality/speed tradeoff
│   └── Cohere Rerank / ColBERT → per-token interaction → more precise than bi-encoder similarity
│
├── Chunking Strategies
│   ├── fixed-size → simple, ignores semantics → overlapping windows reduce boundary artifacts
│   ├── semantic → split at sentence/paragraph boundaries → better coherence
│   └── hierarchical → summary + chunk tree → retrieve at multiple granularities
│
└── Evaluation
    ├── RAGAS → faithfulness + answer relevancy + context precision + recall → automated LLM eval
    └── hallucination rate → fraction of claims not grounded in retrieved context
```

### 12.5 Frontier Models (2025)

```
FRONTIER MODELS 2025
│
├── OpenAI
│   ├── GPT-4o → native multimodal (text+image+audio+video) → unified token stream
│   └── o1/o3 → extended thinking via hidden CoT → MCTS + PRM search
│
├── Anthropic
│   ├── Claude 3.x (Haiku/Sonnet/Opus) → constitutional AI, extended context
│   └── Claude 4.x (Haiku 4.5, Sonnet 4.6, Opus 4.7) → latest generation, improved reasoning
│
├── Google DeepMind
│   ├── Gemini 1.5/2.0 → 1M+ token context, native multimodal → mixture of experts backbone
│   └── Gemma → open-weight efficient models → 2B/7B/27B variants
│
├── Meta
│   └── Llama 3 → open-weight → 8B/70B/405B → GQA, extended vocab, long context
│
├── Mistral / DeepSeek
│   ├── Mistral-7B/Mixtral-8×7B → sliding window + MoE → efficient inference
│   └── DeepSeek-R1 → open reasoning model → GRPO training → o1-competitive
│
└── Key 2025 Trends
    ├── reasoning models (o1/R1 style) → test-time compute scaling → new capability axis
    ├── native multimodality → not bolted-on → joint pretraining across modalities
    ├── long context (1M+ tokens) → full document / codebase in context → retrieval less critical
    └── small + capable (SLMs) → Phi-3, Gemma → on-device, edge deployment
```

---

## PART 13 — MULTIMODAL MODELS

### 13.1 Vision-Language Models

```
VISION-LANGUAGE MODELS
│
├── CLIP (Contrastive Language-Image Pretraining)
│   ├── dual encoder → image encoder (ViT/ResNet) + text encoder (Transformer) → separate towers
│   ├── contrastive loss → align matching pairs, repel non-matching → InfoNCE on NxN matrix
│   ├── zero-shot transfer → class names as text prompts → no task-specific training needed
│   └── ALIGN → similar but noisier 1.8B pairs → scale compensates for noise → Google, 2021
│
├── Flamingo
│   ├── frozen vision encoder + frozen LLM + cross-attention bridges → modular fusion
│   ├── Perceiver Resampler → compress variable image tokens to fixed count → 64 latents
│   ├── few-shot in-context learning → interleaved image-text sequences → (img,txt,img,txt,...)
│   └── gated cross-attention → controls how much vision info flows to LLM → tanh gate
│
├── LLaVA (Large Language and Vision Assistant)
│   ├── CLIP visual encoder → extract image features → ViT-L/14 at 336px resolution
│   ├── linear projection / MLP → map visual features to LLM token space → W·z_v
│   ├── LLaMA/Vicuna backbone → process combined vision-language tokens → full attention
│   └── instruction tuning → GPT-4 generated visual instruction data → 150K conversations
│
├── GPT-4V / Claude Vision
│   ├── proprietary vision encoder fused into LLM → architecture undisclosed
│   ├── supports interleaved image-text inputs → multiple images per context
│   └── OCR, diagram reasoning, spatial understanding capabilities → strong emergent behaviors
│
└── PaLM-E / Gemini
    ├── embodied multimodal → robotics + vision + language → sensor fusion in embedding space
    ├── Gemini → natively multimodal from pretraining (not post-hoc fusion) → joint token stream
    └── token types: text, image patches, audio, video → unified sequence → single Transformer
```

### 13.2 Multimodal Training Paradigms

```
MULTIMODAL TRAINING PARADIGMS
│
├── Contrastive Pretraining → align modalities in shared embedding space → cosine sim objective
│   └── InfoNCE loss → log(exp(sim(i,t)/τ) / Σ exp(sim(i,tⱼ)/τ)) → τ=temperature, N negatives
│
├── Generative Pretraining → predict tokens across modalities → unified autoregressive loss
│   └── next-token on interleaved image+text → Chameleon, Unified-IO → discrete image tokens
│
├── Masked Multimodal Modeling
│   ├── MAE → mask 75% image patches → reconstruct pixel values → high mask ratio forces semantics
│   ├── BEiT → predict discrete visual tokens (dVAE codebook) → visual token IDs not pixels
│   └── data2vec → predict contextualized teacher representations → EMA teacher, student learns
│
├── Instruction Tuning (Multimodal)
│   ├── visual instruction data → (image, question, answer) triples → GPT-4 generated at scale
│   └── LLaVA-1.5 → MLP projector + stronger data mix → SOTA on benchmarks → ShareGPT4V data
│
└── Evaluation Benchmarks
    ├── VQAv2 → visual question answering, balanced answer distribution → 1.1M QA pairs
    ├── MMMU → multi-discipline university-level multimodal reasoning → 11.5K questions
    ├── MMBench → structured capability evaluation across 20 dimensions → GPT-4 judged
    └── SeedBench → 19K QA pairs across image + video understanding → 12 capability dimensions
```

---

## PART 14 — DATA SCIENTIST

### 14.1 Statistics & Probability

```
STATISTICS & PROBABILITY
│
├── Distributions
│   ├── Normal → CLT basis → mean + variance fully characterize → N(μ, σ²)
│   ├── Binomial → n Bernoulli trials → P(X=k) = C(n,k)pᵏ(1-p)^(n-k)
│   ├── Poisson → rare events per interval → P(X=k) = λᵏe^{-λ}/k! → mean = variance = λ
│   └── t-distribution → heavy tails, small samples → degrees of freedom ν → converges to N as ν→∞
│
├── Hypothesis Testing
│   ├── null hypothesis H₀ → assume no effect → reject if p-value < α
│   ├── p-value → P(data ≥ observed | H₀) → not probability H₀ is true
│   ├── Type I error (α) → false positive → reject H₀ when true → controlled by significance level
│   ├── Type II error (β) → false negative → fail to reject H₀ when false → 1-β = power
│   └── t-test → compare means → one-sample, two-sample, paired → assumes Normality (or large N)
│
├── Confidence Intervals
│   ├── x̄ ± z·(σ/√n) → 95% CI → z=1.96 for two-tailed → repeated sampling interpretation
│   └── NOT: 95% probability true mean is in this interval → frequentist construction
│
└── Effect Size
    ├── Cohen's d → (μ₁ - μ₂) / σ_pooled → 0.2 small, 0.5 medium, 0.8 large
    └── separates practical from statistical significance → large N → tiny Δ is significant
```

### 14.2 A/B Testing & Experimentation

```
A/B TESTING
│
├── Design
│   ├── randomization unit → user, session, device → choose to avoid network effects
│   ├── sample size → n = 2·(z_α/2 + z_β)²·σ²/δ² → MDE δ determines cost
│   └── pre-experiment analysis → check balance, set guardrail metrics
│
├── Running
│   ├── avoid peeking → inflates Type I error → use sequential testing or fixed horizon
│   ├── novelty effect → initial engagement spike → run ≥ 1-2 weeks → wait for washout
│   └── network effects → SUTVA violation → switched randomization or cluster assignment
│
├── Analysis
│   ├── z-test / t-test → metric is mean → assumes CLT for large N
│   ├── bootstrap → non-parametric CI → resample with replacement → 10K iterations
│   ├── ratio metrics → delta method → Var(Y/X) ≈ Var(Y)/μ_X² - 2·Cov/μ_X³ + ...
│   └── multiple testing → Bonferroni / BH correction → FWER vs FDR control
│
├── Common Pitfalls
│   ├── peeking at results early → inflated false positives → pre-commit to n
│   ├── survivor bias → only analyze active users at end → include all exposed at start
│   ├── Simpson's paradox → aggregate reversal → stratify by segment
│   └── carryover effects → user remembers treatment → washout period between experiments
│
└── Advanced
    ├── CUPED → reduce variance via pre-experiment covariate → θ̂ = ȳ - θ·(x̄ - E[x])
    ├── stratified sampling → reduce variance by pre-stratification → Neyman allocation
    └── interleaving → position-debiased ranking comparison → implicit comparison > A/B for rankers
```

### 14.3 Causal Inference

```
CAUSAL INFERENCE
│
├── Potential Outcomes (Rubin Framework)
│   ├── Y(1), Y(0) → treated / untreated potential outcomes → only one observed per unit
│   ├── ATE → E[Y(1) - Y(0)] → average treatment effect → population-level causal estimate
│   └── SUTVA → no interference between units, single treatment version → often violated
│
├── DAGs (Structural Causal Models)
│   ├── nodes = variables, edges = causal direction → d-separation → conditional independence
│   ├── backdoor criterion → block all confounding paths → select valid adjustment set
│   └── do-calculus → P(Y|do(X)) ≠ P(Y|X) → intervention vs observation
│
├── Estimation Methods
│   ├── OLS with controls → unbiased if no unmeasured confounders → conditional ignorability
│   ├── Propensity Score Matching → balance covariates across groups → e(X) = P(T=1|X)
│   │   ├── IPW → reweight by 1/e(X) → creates pseudo-population → stabilized weights
│   │   └── matching → nearest neighbor on e(X) → reduces covariate imbalance
│   ├── DiD (Difference-in-Differences)
│   │   ├── parallel trends assumption → pre-trends test → event study plot
│   │   └── DiD = (Ȳ_T,post - Ȳ_T,pre) - (Ȳ_C,post - Ȳ_C,pre)
│   └── Instrumental Variables (IV)
│       ├── instrument Z → affects treatment but not outcome directly → exclusion restriction
│       └── 2SLS → first stage: T̂ = α + γZ → second stage: Y = β + δT̂ → LATE estimate
│
└── Uplift Modeling
    ├── estimate CATE → τ(x) = E[Y(1)-Y(0)|X=x] → individual causal effect
    ├── T-learner → separate model per treatment group → subtract predictions
    └── X-learner → iterative CATE estimation → better for unbalanced treatment
```

### 14.4 EDA & Data Quality

```
EDA & DATA QUALITY
│
├── Univariate Analysis
│   ├── numeric → histogram, boxplot, KDE → check skewness, kurtosis, outliers
│   └── categorical → bar chart, frequency table → cardinality, imbalance
│
├── Bivariate Analysis
│   ├── numeric-numeric → scatter + Pearson/Spearman correlation → linear vs monotone
│   ├── numeric-categorical → violin/box per group → ANOVA F-test for differences
│   └── categorical-categorical → heatmap of crosstab proportions → chi² test of independence
│
├── Multivariate
│   ├── pairplot → all variable pairs simultaneously → color by target → identify separability
│   └── VIF (Variance Inflation Factor) → detect multicollinearity → VIF > 10 is problematic
│
└── Data Quality Checks
    ├── missingness → MCAR / MAR / MNAR → missingness pattern matters for imputation choice
    ├── outliers → Z-score > 3, IQR rule, isolation forest → domain context required
    ├── duplicates → exact + near-duplicates → Levenshtein for text, hash for rows
    └── distribution shift → compare train vs prod → PSI per feature → flag before retraining
```

### 14.5 Metrics & Business Analytics

```
METRICS & BUSINESS ANALYTICS
│
├── North Star Metric
│   └── single metric capturing core product value → DAU, GMV, NPS → aligns teams
│
├── Metric Decomposition
│   ├── revenue = users × conversion × ARPU → isolate which lever is moving
│   └── funnel → impressions → clicks → signups → activations → retention
│
├── Cohort Analysis
│   ├── group users by join date → track retention over time → triangular retention matrix
│   └── identify vintage effects → newer cohorts performing differently → product changes
│
├── Customer Metrics
│   ├── LTV → Σ(margin × retention^t) → discount rate → LTV/CAC > 3 rule of thumb
│   ├── churn rate → 1 - retention → monthly vs annual → compounding matters
│   └── NPS → % promoters − % detractors → scale -100 to +100
│
└── Metric Pitfalls
    ├── Goodhart's Law → measure becomes the target → optimize proxy, miss true goal
    ├── Simpson's Paradox → aggregate trend reversed when disaggregated → stratify always
    └── survivorship bias → only see successful examples → include churned users in analysis
```

### 14.6 SQL & Data Manipulation

```
SQL PATTERNS FOR DATA SCIENCE
│
├── Window Functions
│   ├── ROW_NUMBER() / RANK() / DENSE_RANK() → ranking within partition
│   ├── LAG() / LEAD() → access previous/next row → session gap detection
│   └── SUM() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN ...) → running totals
│
├── Cohort Queries
│   └── JOIN users on first_event date → group by cohort_month → pivot on period offset
│
├── Funnel Analysis
│   └── COUNT(DISTINCT user_id) at each step → LEFT JOIN steps → calculate drop-off %
│
├── Session Construction
│   └── LAG(event_time) → gap > 30min → new session → CONDITIONAL_COUNT pattern
│
└── Performance
    ├── avoid SELECT * → project only needed columns → reduce IO
    ├── predicate pushdown → filter early → use WHERE not HAVING when possible
    └── partition pruning → filter on partition column → avoid full table scan
```

---

## PART 15 — ML/DL/LLM PRODUCTION DEPLOYMENT

### 15.1 Real-Time Serving Architecture

```
Real-Time Model Serving (System Design)
│
├── Request Path
│   ├── Client → API Gateway/LB → Model Service → Response → p99 SLA governs every hop
│   ├── Feature fetch → online feature store (Redis/DynamoDB) → must match offline training features
│   └── Model inference → in-process (embedded) or RPC (gRPC/HTTP to model server) → RPC adds network hop
│
├── Sync vs Async Serving
│   ├── Sync (request-response) → user-facing, low-latency → recommendations, fraud, search ranking
│   └── Async (queue-based) → batch scoring, non-blocking → Kafka/SQS → consumer pool scores + writes result
│
├── Model Server Choice
│   ├── Custom FastAPI/Flask → simple, full control → no dynamic batching, DIY scaling
│   ├── Triton Inference Server → multi-framework, dynamic batching, concurrent model execution → GPU-efficient
│   ├── TorchServe → PyTorch-native, handler API → good default for single-framework shops
│   └── vLLM / TGI (LLM-specific) → PagedAttention, continuous batching → built for autoregressive generation
│
├── Latency Budget Decomposition
│   ├── network (client↔gateway↔service) → 5-20ms typical → unavoidable floor
│   ├── feature fetch → 1-10ms if cached, 50ms+ if cold → cache aggressively
│   ├── inference → model-size dependent → GBDT <1ms, small NN 5-20ms, LLM 100ms-several sec
│   └── budget determines model class → sub-10ms SLA rules out large NNs/LLMs entirely
│
└── Scaling the Service
    ├── horizontal pod autoscaling → CPU/QPS/custom metric (GPU util via Prometheus Adapter)
    ├── load shedding → drop/degrade at overload → return cached/fallback rather than fail
    └── multi-region deployment → route by geo → reduces cross-region network latency
```

### 15.2 Classical ML / DL Deployment Pipeline

```
Classical ML/DL Deployment Pipeline
│
├── Packaging
│   ├── serialize model → pickle/joblib (sklearn), TorchScript/ONNX (DL) → framework-portable format
│   ├── containerize → Dockerfile bundles model + deps + serving code → immutable artifact
│   └── model registry → versioned, stage-tagged (staging/prod) → MLflow/SageMaker Model Registry
│
├── CI/CD for Models
│   ├── train → eval gate (metric threshold) → package → push to registry → auto-fails bad models
│   ├── build image → run smoke tests (load model, sample predict) → push to container registry
│   └── deploy → triggered by registry promotion, not just code merge → decouples model & code release
│
├── Kubernetes Deployment
│   ├── Deployment (replicas, resource requests/limits) + Service (stable endpoint) + HPA (autoscale)
│   ├── readiness probe gates traffic until model loaded into memory → avoids routing to cold pod
│   └── rolling update → old pods drain via terminationGracePeriodSeconds → no dropped in-flight requests
│
├── Rollout Strategy
│   ├── Shadow → mirror prod traffic to new model, log only, zero user impact → validate before any exposure
│   ├── Canary → 5% → 25% → 100% traffic ramp → auto-rollback on metric/latency regression
│   └── Blue/Green → instant full swap, old env kept warm → fast rollback, 2× resource cost during switch
│
└── Rollback
    ├── automated trigger → error rate spike, p99 latency breach, prediction-distribution shift
    └── revert to last-known-good registry version → same deployment pipeline in reverse
```

### 15.3 LLM Production Deployment

```
LLM Production Deployment
│
├── Serving Stack
│   ├── vLLM/TGI on GPU nodes → PagedAttention + continuous batching → high throughput per GPU
│   ├── model doesn't fit one GPU → quantize (INT8/AWQ/GPTQ) → tensor parallel (single node, multi-GPU)
│   │   → multi-node model parallel (Ray/KubeRay + DeepSpeed/Megatron) → CPU offload as last resort
│   └── KV-cache growth with context length → drives GPU memory ceiling → sliding window/eviction to bound it
│
├── Kubernetes for LLMs
│   ├── nvidia.com/gpu resource request → whole-GPU scheduling by default → MPS/MIG needed to share
│   ├── nodeSelector/tolerations → route pods to GPU node pool → separate from CPU workloads
│   └── KubeRay → orchestrates multi-node Ray cluster for models spanning >1 node → plain Deployments can't coordinate this
│
├── Request Handling
│   ├── streaming responses (SSE/WebSocket) → user sees tokens as generated → masks high total latency
│   ├── prompt caching → reuse KV-cache for repeated prefixes (system prompt) → cuts prefill cost
│   └── request queuing/prioritization → interactive vs batch traffic classes → prevent batch starving chat
│
├── RAG in the Serving Path
│   ├── retrieve (vector DB/BM25) → rerank → construct prompt → generate → adds retrieval latency before LLM call
│   └── semantic cache → skip regeneration for near-duplicate queries → embed query, check cache similarity
│
├── Safety & Guardrails at Serve Time
│   ├── input moderation → block jailbreak/injection before hitting the model → fast, cheap classifier first
│   ├── output moderation/constrained decoding → grammar-constrained JSON, toxicity filter → post-generation gate
│   └── human-in-the-loop → approval gate before any irreversible action an agent proposes
│
└── Cost & Latency Levers
    ├── model size → smaller/distilled model where task allows → biggest single cost lever
    ├── speculative decoding → small draft model + big model verifies → 2-3× speedup, no quality loss
    └── batching aggressiveness → higher batch = higher throughput, higher per-request latency → tune to SLA
```

### 15.4 Monitoring & Feedback Loop in Production

```
Production Monitoring & Feedback Loop
│
├── Online Metrics
│   ├── latency (p50/p95/p99) + throughput (QPS) + error rate → the operational trio
│   └── prediction distribution → compare live scores to training distribution → early proxy for drift
│
├── Model Quality Signals
│   ├── data drift → PSI on input features → >0.2 = investigate
│   ├── concept drift → live accuracy/label-based metric decay → often structural (patch/season), not gradual
│   └── calibration drift → predicted probability vs realized outcome rate diverges → re-calibrate or retrain
│
├── LLM-Specific Monitoring
│   ├── hallucination rate → sampled human eval or automated (SelfCheckGPT/FactScore/RAGAS)
│   ├── token cost & latency per request → budget guardrails, alert on spend anomalies
│   └── refusal rate / jailbreak attempts → safety-team dashboard, feeds back into guardrail tuning
│
├── Feedback Loop
│   ├── log predictions + features + (eventually) outcomes → forms next retrain dataset
│   └── retrain trigger → scheduled or drift-threshold-based → gated behind data-quality check
│
└── Incident Response
    ├── automated rollback on metric/latency breach → same registry-version revert as CI/CD
    └── fallback to heuristic/cached/previous-model response → never let ML outage break the product
```

---

## PART 16 — AI INFRASTRUCTURE / INFERENCE DATA PLANE

### 16.1 Inference Engine Internals

```
LLM Inference Engine (the four subsystems)
│
├── Memory Manager  ← the real differentiator
│   ├── PagedAttention (vLLM) → KV cache in fixed blocks → no contiguous allocation → ~0 fragmentation
│   ├── copy-on-write block sharing → parallel samples/beams share prefix blocks → n× memory saving
│   ├── automatic prefix caching → exact shared prefix reused across requests → cuts prefill
│   └── RadixAttention (SGLang) → radix tree keyed on token prefix → longest-common-prefix match
│       → beats flat cache on: multi-turn chat, agent loops, tree search, few-shot prompts
│
├── Scheduler  ← the other real differentiator
│   ├── continuous batching → finished seq leaves, new seq joins mid-flight → no wait for slowest
│   ├── chunked prefill → split long prefill into chunks interleaved with decode → bounds HOL blocking
│   └── preemption/swap → evict low-priority KV to CPU or recompute → admission under memory pressure
│
├── Kernels
│   ├── FlashAttention → tiling in SRAM → IO-aware → no N² matrix materialized
│   └── fused ops, quantized GEMM (INT8/FP8/AWQ/GPTQ) → arithmetic intensity ↑
│
└── Executor
    ├── CUDA graphs → capture decode step → kill per-token launch overhead
    └── tensor/pipeline parallel dispatch → see 16.3

Engine choice
├── vLLM        → default; broadest model support, strong ecosystem, PagedAttention
├── SGLang      → prefix-heavy workloads (agents, multi-turn), structured output via compressed FSM
├── TensorRT-LLM→ max NVIDIA perf; AOT compile step → fastest, least flexible, slow model turnaround
└── Modular MAX → Mojo kernels, portability thesis → newest, smallest ecosystem
```

### 16.2 Control Plane vs Data Plane

```
Serving a fleet (not a single engine)
│
├── THE INVARIANT
│   └── data plane must keep serving when control plane is down
│       → anti-pattern: synchronous controller lookup on every request
│       → correct: controller pushes state; router caches it; stale-but-serving > correct-but-down
│
├── Control Plane (slow path, low QPS)
│   ├── model registry / weights distribution → which model versions exist, where
│   ├── placement & autoscaling → what runs on which GPUs
│   ├── health aggregation → mark replicas in/out of the pool
│   └── config & rollout → canary %, rate limits, tenant quotas
│
├── Data Plane (fast path, every token)
│   ├── router → picks replica (see 16.4)
│   ├── engine → prefill + decode
│   └── streaming transport → gRPC/SSE back to client
│
└── Frameworks
    ├── llm-d          → K8s-native, Gateway API Inference Extension, vLLM-centric
    ├── NVIDIA Dynamo  → disaggregation-first, KV-aware routing, NIXL KV transfer, engine-agnostic
    └── Ray Serve      → heterogeneous pipelines (retriever→reranker→LLM), Python-native composition
```

### 16.3 GPU Hardware, Interconnects & Parallelism

```
Bandwidth hierarchy — ~1 order of magnitude per step
SRAM → HBM → NVLink → PCIe → RoCE/IB network → storage
│
├── Interconnect
│   ├── NVLink/NVSwitch → intra-node GPU↔GPU → all-to-all inside the node
│   ├── XGMI / Infinity Fabric → AMD MI300X equivalent
│   ├── RoCE vs InfiniBand → RoCE cheaper/Ethernet-native but needs PFC/ECN tuned → else congestion collapse
│   └── nvidia-smi topo -m → NV#=NVLink, PIX/PXB=PCIe switch, SYS=cross-socket (worst)
│
├── Collectives (NCCL)
│   ├── all-reduce ring cost = 2S(N−1)/N → ≈2S, independent of N → why TP scales at all
│   ├── all-gather / reduce-scatter → building blocks of TP + FSDP
│   └── all-to-all → MoE expert dispatch → cost is load-dependent → imbalance = stall
│
├── Memory ceiling
│   ├── weights: params × bytes/param
│   ├── KV: 2 × L × H_kv × d_head × seq × batch × bytes  ← GQA shrinks H_kv, this is the lever
│   └── leftover = concurrency budget → KV occupancy is the true saturation metric
│
└── PARALLELISM DECISION RULE (in order)
    ├── 1. fits on one GPU? → data parallel only → best throughput/$, no collectives
    ├── 2. needs >1 GPU? → tensor parallel, but only WITHIN an NVLink domain
    │      → constraint: TP degree must divide KV heads (GQA)
    ├── 3. spans nodes? → TP inside node, pipeline parallel across nodes
    │      → PP bubble = (S−1)/(M+S−1) → serving can't hide it (small M) → avoid if possible
    └── 4. MoE? → expert parallel inside the fast domain → watch expert load imbalance
    Sizing example: 70B on 8×80GB → TP=4 with 2 replicas often beats TP=8 with 1
      → less collective overhead, independent failure domains, better tail latency
```

### 16.4 Smart Routing & Scheduling

```
Why round-robin / least-connections fails for LLMs
├── false assumption 1: requests cost the same → prompt lengths vary 100×
└── false assumption 2: replicas are interchangeable → they aren't, KV cache state differs
│
├── Routing signals
│   ├── queue depth, KV cache occupancy, running batch size
│   ├── prefix-cache overlap with this replica  ← the LLM-specific one
│   └── UNKNOWABLE: output length → all scheduling is under uncertainty
│
├── Prefix-affinity routing
│   └── score = w₁·prefix_match_length − w₂·load_factor
│       → pure affinity → hotspots;  pure balance → cache thrash;  tune between
│
├── Admission control & queue discipline
│   ├── FCFS → fair, bad tail;  SJF → good mean, starves long prompts
│   ├── priority classes (interactive vs batch) → prevent batch starving chat
│   └── reject early (429) rather than queue forever → queueing latency is invisible failure
│
├── Head-of-line blocking → one 100K-token prefill stalls everyone
│   └── chunked prefill is a FAIRNESS feature, not just a throughput one
│
├── Load shedding order → batch first → low-priority tenants → degrade (smaller model) → 429
│
└── Disaggregated routing → prefill pool and decode pool routed separately
    → KV must transfer between them (NIXL) → only pays off when prefill/decode ratio is skewed
```

### 16.5 Multi-Tenancy & Isolation

```
Isolation ↔ Utilization tension
├── separate GPUs per tenant → perfect isolation, terrible utilization
├── MIG partitions → hard isolation, fixed slices → good for MANY SMALL models, not large ones
├── shared engine + quotas → best utilization, softest isolation  ← usual answer
└── multi-LoRA on one base model → many tenants, one set of weights → the density play
│
├── FIVE noisy-neighbour mechanisms (each needs its own fix)
│   ├── KV exhaustion        → per-tenant KV budget
│   ├── batch-slot monopoly  → cap concurrent seqs per tenant
│   ├── prefill interference → chunked prefill + prefill quota
│   ├── prefix-cache eviction→ per-tenant cache partition or scoped LRU
│   └── LoRA adapter thrash  → adapter pinning / affinity routing
│
├── Fairness → weighted fair queueing over TOKENS, not requests
│   → but output length unknown → charge as you go, not up front
│
├── Rate limiting → TPM *and* concurrency (either alone is gameable)
│
└── Data / fault / rollout isolation
    └── cross-tenant prefix cache = TTFT timing side channel → scope caches per tenant for sensitive data
```

### 16.6 SLOs & Observability for Inference

```
Why generic web SLOs fail
├── one "request" = a stream of tokens, not one response
├── cost varies 100× with prompt length → a single p99 latency SLO is meaningless
└── degradation is often silent (quality, not errors)
│
├── SLIs that matter
│   ├── TTFT (prefill-bound) and TPOT/ITL (decode-bound) → measure SEPARATELY
│   ├── GOODPUT = SLO-conformant tokens/sec  ← optimize this, not raw throughput
│   ├── KV cache utilization → the saturation signal (GPU util lies: 100% while starved)
│   ├── queue wait time, preemption rate, prefix cache hit rate
│   └── shed rate (429s) → its own SLI with its own budget, else you "meet" latency by rejecting
│
├── Scope SLOs by prompt-length band → e.g. TTFT p95 < 500ms for <2K-token prompts
│
├── Alerting → multi-window burn-rate on error budget, not static thresholds
│
├── Telemetry sources → DCGM (GPU: SM util, HBM, power, ECC, throttle) + engine metrics + traces
│   └── trace attribute cardinality: tenant/model OK, request-id as label = metric explosion
│
└── Silent degradation → functional health check (canary prompt, verify output) not just /healthz
```

### 16.7 Service Architecture — gRPC & Go

```
Inference service in Go
│
├── Why gRPC → HTTP/2 streaming, binary protobuf, deadlines + cancellation as first-class
│
├── FOUR streaming concerns
│   ├── cancellation MUST reach the GPU → client disconnect → free KV immediately, else leak
│   ├── deadlines, not timeouts → propagate through every hop
│   ├── backpressure → stream.Send blocks under flow control → that IS your signal
│   └── mid-stream errors → status after partial output → client needs a defined contract
│
├── Go patterns
│   ├── ctx := stream.Context() → select on ctx.Done() and the token channel
│   ├── bounded worker pool = admission control (unbounded goroutines = OOM under load)
│   ├── channels for ownership transfer, mutexes for shared state
│   └── classic bugs: leaked goroutine on early return, loop-variable capture, unbuffered channel deadlock
│
├── Service decomposition (boundary rule: split on failure domain + scaling axis, not on nouns)
│   └── gateway (auth/quota) │ router │ engine pool │ controller │ metrics-telemetry
│
└── Resilience with an inference twist
    ├── retries → only on idempotent/pre-token failures; never replay a half-streamed generation
    ├── circuit breaker → per-replica, tripped on TTFT not just errors
    └── graceful drain → stop admitting, let in-flight generations finish, then exit
```

### 16.8 Where the Layers Meet (recall drill)

```
A token's journey — name what happens at each hop
client → gateway (authn, quota, TPM check)
       → router (prefix-affinity + load score → replica)
       → engine admission (KV budget available? else queue or 429)
       → prefill (chunked, cache hit on shared prefix?)
       → decode loop (continuous batching, spec decoding, CUDA graph)
       → tokens stream back (gRPC flow control = backpressure)
       → client disconnects → ctx cancelled → KV freed
Every arrow has an SLI, a failure mode, and a noisy-neighbour risk.
```
