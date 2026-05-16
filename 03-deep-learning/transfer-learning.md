# Transfer Learning & Domain Adaptation

Transfer learning leverages knowledge from a source task/domain to improve learning on a target task/domain. The core insight: features learned on large datasets (ImageNet, large text corpora) generalize broadly.

---

## Taxonomy

```
Transfer Learning
├── Domain Adaptation     (same task, different distribution)
├── Fine-Tuning           (adapt pre-trained model to new task)
├── Few-Shot Learning     (learn from very few labeled examples)
├── Zero-Shot Learning    (generalize to unseen classes/tasks)
└── Meta-Learning         (learn to learn quickly)
```

---

## Pre-Training → Fine-Tuning

The dominant paradigm in modern ML:

1. Pre-train large model on large dataset (source domain)
2. Fine-tune on small dataset (target domain)

```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained model
backbone = models.resnet50(weights='IMAGENET1K_V2')

# Replace final layer for new task (binary classification)
backbone.fc = nn.Linear(backbone.fc.in_features, 2)

# Option A: Fine-tune all layers
optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)

# Option B: Feature extraction — freeze backbone, train only head
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc.requires_grad_(True)
optimizer = torch.optim.AdamW(backbone.fc.parameters(), lr=1e-3)
```

### When to fine-tune all layers vs freeze

| Scenario | Strategy |
|----------|----------|
| Target domain similar to source, small dataset | Feature extraction (freeze backbone) |
| Target domain similar to source, large dataset | Fine-tune all (low LR) |
| Target domain very different, small dataset | Fine-tune top layers only |
| Target domain very different, large dataset | Fine-tune all (may diverge — careful) |

### Learning Rate Strategies

**Discriminative fine-tuning (ULMFiT):** Apply different learning rates per layer — lower for early layers (general features), higher for later layers (task-specific).

```python
param_groups = [
    {'params': backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': backbone.layer2.parameters(), 'lr': 3e-5},
    {'params': backbone.layer3.parameters(), 'lr': 1e-4},
    {'params': backbone.fc.parameters(),     'lr': 3e-4},
]
optimizer = torch.optim.AdamW(param_groups)
```

**Gradual unfreezing:** Start with only head, progressively unfreeze layers from top to bottom across epochs. Prevents catastrophic forgetting of lower-level features.

---

## Domain Adaptation

Source domain `D_S = (X_S, P_S(x))`, target domain `D_T = (X_T, P_T(x))`. Same task but different distributions.

### Types of Shift

| Type | Definition | Example |
|------|-----------|---------|
| Covariate shift | `P_S(x) ≠ P_T(x)`, same `P(y|x)` | Train on daytime images, test on nighttime |
| Label shift | `P_S(y) ≠ P_T(y)`, same `P(x|y)` | Different class frequencies |
| Concept drift | `P_S(y|x) ≠ P_T(y|x)` | User behavior changes over time |

### DANN — Domain-Adversarial Neural Networks

Learn features that are discriminative for the task but invariant to the domain.

```
Feature extractor → Task classifier (minimize task loss)
               ↘  Domain classifier (maximize domain confusion via gradient reversal)
```

**Gradient Reversal Layer (GRL):** During forward pass, identity. During backward pass, multiply gradient by -λ. Forces feature extractor to make features indistinguishable across domains.

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None
```

### CORAL — Correlation Alignment

Minimize distance between second-order statistics (covariances) of source and target feature distributions. Simple, no adversarial training.

`L_CORAL = (1/4d²) ‖C_S - C_T‖²_F`

### Instance Reweighting

If we can estimate density ratio `w(x) = P_T(x)/P_S(x)`, reweight source samples during training.

**KLIEP, uLSIF** — algorithms to estimate `w(x)` without estimating densities separately.

### Test-Time Adaptation (TTA)

Adapt model at inference time using only the test batch (no labels). Update batch norm statistics or use entropy minimization.

```python
# Simple TTA: average predictions over augmented views
augmented_preds = [model(augment(x)) for _ in range(10)]
final_pred = torch.stack(augmented_preds).mean(0)
```

---

## Few-Shot Learning

Learn to classify from K examples per class (K-shot learning). Standard benchmarks: miniImageNet, tieredImageNet, Omniglot.

### Prototypical Networks

Represent each class by the mean embedding of its K support examples (prototype). Classify query by nearest prototype.

```python
def prototypical_forward(support_embeddings, query_embeddings, n_way, k_shot):
    # support_embeddings: (n_way * k_shot, embed_dim)
    prototypes = support_embeddings.reshape(n_way, k_shot, -1).mean(dim=1)
    # query_embeddings: (n_queries, embed_dim)
    dists = torch.cdist(query_embeddings, prototypes)   # (n_queries, n_way)
    return -dists  # logits (negative distance = similarity)
```

### Matching Networks

Use attention-weighted sum of support labels as the prediction. Similar to nearest-neighbor but with learned attention.

### MAML — Model-Agnostic Meta-Learning

Learn an initialization θ that can be fine-tuned with few gradient steps to any task.

```
For each task T_i:
    Sample support set D_i^train, query set D_i^test
    Adapted params: θ'_i = θ - α ∇_θ L(f_θ, D_i^train)
    Meta-loss: L_meta = Σ_i L(f_{θ'_i}, D_i^test)

Meta-update: θ ← θ - β ∇_θ L_meta
```

**Inner loop:** task-specific adaptation (K gradient steps)  
**Outer loop:** meta-optimization across tasks

**Problem:** Expensive second-order gradients. **FOMAML** approximates with first-order, nearly same performance.

### Relation Networks

Instead of a fixed distance metric (Euclidean for ProtoNets), learn a comparison function.

```
Concat(query_embed, prototype_embed) → relation_score
```

---

## Zero-Shot Learning

Classify instances from classes **not seen during training**, using semantic class descriptions (attributes, word embeddings, text descriptions).

### Attribute-Based ZSL

Each class is described by a binary attribute vector (e.g., "has stripes", "is aquatic").

`P(y | x) ∝ compatibility(f(x), a(y))`

where `f(x)` is image embedding, `a(y)` is class attribute vector.

### CLIP-Based Zero-Shot

CLIP jointly trains image and text encoders with contrastive loss. At test time, compute similarity between image embedding and text embeddings of class names.

```python
import clip
model, preprocess = clip.load("ViT-B/32")

image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
```

**Generalized ZSL:** The harder and more realistic setting — test classes include both seen and unseen classes.

---

## Meta-Learning (Learning to Learn)

The model learns a learning algorithm from a distribution of tasks, not just one task.

### Categories

| Category | Examples | Mechanism |
|----------|---------|-----------|
| Optimization-based | MAML, Reptile | Learn good initialization |
| Metric-based | ProtoNets, Matching Nets | Learn good embedding space |
| Model-based | SNAIL, Memory-Augmented Networks | Learned update rule in model |

### Reptile (simpler than MAML)

```
For each iteration:
    Sample task T_i
    Train θ on T_i for k steps → θ'_i
    Update: θ ← θ + ε(θ'_i - θ)
```

No second-order gradients — just move initialization toward task-adapted parameters.

---

## Self-Supervised Pre-Training

Learning representations without labels, then fine-tuning supervised.

| Method | Pretext Task | Domain |
|--------|-------------|--------|
| SimCLR | Contrastive (augmented views) | Vision |
| DINO / DINOv2 | Self-distillation | Vision |
| MAE | Masked patch reconstruction | Vision |
| BERT | Masked language modeling | NLP |
| GPT | Next-token prediction | NLP |

These learned representations transfer better than supervised ImageNet features for many downstream tasks.

---

## Key Interview Points

- Feature extraction (frozen backbone) is preferred when target data is scarce and source/target domains are similar.
- Fine-tune all layers with a small LR when you have enough target data; use discriminative LRs for layer-wise tuning.
- Domain adaptation is needed when test distribution ≠ train distribution — check with covariate shift tests.
- DANN learns domain-invariant features via gradient reversal — widely used in NLP and CV domain adaptation.
- ProtoNets are the go-to simple baseline for few-shot classification.
- MAML learns an initialization for fast adaptation; computationally expensive but powerful.
- CLIP enables zero-shot classification by aligning image and text embedding spaces.
