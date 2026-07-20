---
module: Deep Learning Core
topic: Transfer Learning
subtopic: ""
status: unread
tags: [deeplearning, ml, transfer-learning]
---
# Transfer Learning & Domain Adaptation

---

## The foundational question

Why does a network trained on ImageNet help with medical images?

ImageNet contains 1.2 million photos of everyday objects — dogs, cars, chairs. A radiology scan contains none of those things. The pixel statistics are completely different. The classes are unrelated. Yet fine-tuning from an ImageNet checkpoint consistently outperforms training from scratch on medical imaging tasks, often by large margins.

The answer reveals what neural networks actually learn: not patterns specific to the source task's classes, but a hierarchy of general visual abstractions. Early layers learn edge detectors, color gradients, and texture patches. Middle layers compose these into object parts — curves, grids, blobs. Late layers combine parts into high-level representations. These intermediate representations are general — they describe structure in any natural image, including radiological ones.

The medical images task needs all of those intermediate computations. Training from scratch forces the network to re-derive them from a small medical dataset. Fine-tuning from ImageNet starts with those representations already in place and only needs to adapt the final layers to the target task.

This is the core insight: **deep networks factor visual perception into a reusable feature hierarchy. The expensive part — learning that hierarchy — is done on large data and reused everywhere.**

---

## Taxonomy

```
Transfer Learning
├── Fine-Tuning           (pretrained → small target dataset, same or different task)
├── Domain Adaptation     (same task, different distribution)
├── Few-Shot Learning     (K examples per class, learn new classes quickly)
├── Zero-Shot Learning    (generalize to classes never seen during training)
└── Meta-Learning         (learn to learn quickly across tasks)
```

---

## 1. Pre-Training → Fine-Tuning

### The problem

You have a small labeled dataset for your target task. Training a deep network from scratch requires thousands of examples per class to learn general low-level features before learning task-specific ones. With limited data, the network overfits.

### The core insight

The pretrained network has already learned the expensive part: general feature representations. You only need to adapt the task-specific head (and possibly the upper feature layers) to your target task. The earlier layers contain knowledge that transfers freely — you can either freeze them or update them with a very small learning rate.

### The mechanics

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained backbone
backbone = models.resnet50(weights='IMAGENET1K_V2')

# Replace the classification head for your task
backbone.fc = nn.Linear(backbone.fc.in_features, num_target_classes)

# Option A: Feature extraction — freeze backbone, train only head
# Use when: small target dataset and source/target domains are similar
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc.requires_grad_(True)
optimizer = torch.optim.AdamW(backbone.fc.parameters(), lr=1e-3)

# Option B: Full fine-tuning — update all parameters
# Use when: large target dataset, or source/target domains differ substantially
optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)
# Lower LR than from-scratch training — weights are already close to a good solution
```

### When to freeze vs fine-tune

| Scenario | Strategy | Reasoning |
|----------|----------|-----------|
| Small target dataset, similar domain | Freeze backbone, train head | Limited data will overfit if full model is updated; pretrained features are already appropriate |
| Large target dataset, similar domain | Fine-tune all layers with low LR | More data supports full adaptation; low LR preserves knowledge while improving fit |
| Small target dataset, different domain | Fine-tune upper layers only | Lower layers still learn general features; upper layers need to adapt to domain shift |
| Large target dataset, different domain | Fine-tune all (may need careful tuning) | Enough data to safely update everything; risk of overwriting useful low-level features |

### Discriminative fine-tuning

**The problem**: different layers have different optimal learning rates. Early layers have already converged to good general features — they need small updates. The task-specific head starts from random initialization and needs large updates.

**The core insight**: each layer occupies a different position in the feature hierarchy and therefore requires a different adaptation magnitude. Earlier layers should change slowly; later layers can change more aggressively.

```python
param_groups = [
    {'params': backbone.layer1.parameters(), 'lr': 1e-5},   # general edges/textures
    {'params': backbone.layer2.parameters(), 'lr': 3e-5},
    {'params': backbone.layer3.parameters(), 'lr': 1e-4},
    {'params': backbone.layer4.parameters(), 'lr': 3e-4},
    {'params': backbone.fc.parameters(),     'lr': 1e-3},   # task-specific head
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

### Gradual unfreezing

**The problem**: fine-tuning all layers simultaneously means the randomly initialized head generates large, noisy gradients early in training. These gradients propagate through the backbone and can corrupt the pretrained features — this is called catastrophic forgetting.

**The core insight**: the head should stabilize first. Once it has learned a reasonable mapping from pretrained features to target classes, you can safely unfreeze the backbone layers, starting from the top (most task-specific) and working down (most general).

```python
# Epoch 1-3: train only the head
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc.requires_grad_(True)

# Epoch 4-6: unfreeze last feature block
for param in backbone.layer4.parameters():
    param.requires_grad = True

# Epoch 7+: unfreeze everything
for param in backbone.parameters():
    param.requires_grad = True
```

**What breaks**: starting with full fine-tuning when you have very little target data. The head's large initial gradients can overwrite the backbone's pretrained features within a few batches, and then the model has neither good pretrained features nor sufficient target data to relearn them. Training effectively collapses.

---

## 2. Domain Adaptation

### The problem

Your model was trained on one distribution (source) and deployed into a different distribution (target). The task is the same — you still want to classify, detect, or predict — but the input data looks different. Performance degrades even though the model is "correct" for the source domain.

**Formalizing the shift**: let `D_S = (X_S, P_S(x))` and `D_T = (X_T, P_T(x))` be source and target domains.

| Type | Condition | Example |
|------|-----------|---------|
| Covariate shift | `P_S(x) ≠ P_T(x)`, same `P(y|x)` | Train on daylight images, test on night |
| Label shift | `P_S(y) ≠ P_T(y)`, same `P(x|y)` | Training data skews toward common classes, test doesn't |
| Concept drift | `P_S(y|x) ≠ P_T(y|x)` | The meaning of a label changes over time |

### DANN — Domain-Adversarial Neural Networks

**The problem**: a classifier trained on source features will encode source-domain artifacts. If you could force the feature extractor to produce representations that are indistinguishable between source and target, those features would be domain-agnostic — a classifier trained on them would generalize to the target domain.

**The core insight**: set up an adversarial game. A domain classifier tries to tell source features from target features. The feature extractor tries to fool it — producing features that the domain classifier cannot distinguish. When the domain classifier is maximally confused (performance near chance), the features are domain-invariant. The task classifier is trained simultaneously to ensure domain-invariant features still carry task-relevant information.

**The mechanics**: the gradient reversal layer (GRL) is the key implementation trick. During the forward pass it is the identity. During the backward pass it multiplies the gradient by -λ. This single trick makes the feature extractor minimize domain classification accuracy without needing separate update passes.

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()   # identity forward

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None   # reversal backward

# Full DANN training loop logic:
# features = feature_extractor(x)
# task_loss = task_criterion(task_head(features[source]), labels[source])
# domain_loss = domain_criterion(domain_head(GRL(features, alpha)), domain_labels)
# total_loss = task_loss + domain_loss
# total_loss.backward()
```

**What breaks**: DANN requires paired source and target data in each batch. If the domain shift is very large, the feature extractor may not be able to find a useful domain-invariant representation — there may be no feature space where domains overlap.

### CORAL — Correlation Alignment

**The problem**: DANN requires adversarial training, which is unstable and sensitive to hyperparameters.

**The core insight**: if source and target features have the same second-order statistics (covariance structure), they're aligned in the directions that matter most. You don't need adversarial training — just minimize the distance between covariance matrices.

`L_CORAL = (1 / 4d²) ‖C_S - C_T‖²_F`

Where `C_S` and `C_T` are the feature covariance matrices for source and target. Add this term to the task loss. Simpler and more stable than DANN, though less powerful for large shifts.

### Instance reweighting

**The problem**: under covariate shift, the source distribution `P_S(x)` over-represents some inputs and under-represents others relative to the target distribution `P_T(x)`.

**The core insight**: if we weight each source sample by `w(x) = P_T(x) / P_S(x)`, the reweighted source distribution matches the target distribution. A model trained with these weights will behave as if trained on target data.

**The mechanics**: estimate the density ratio `w(x)` without separately estimating each density — algorithms like KLIEP and uLSIF directly estimate the ratio by solving a convex optimization problem.

**What breaks**: density ratio estimation fails when the support of `P_S(x)` doesn't cover `P_T(x)` — you cannot reweight source samples to cover target regions where no source data exists.

### Test-Time Adaptation (TTA)

**The problem**: at deployment, you have test batches but no labels. You can't train, but you can adapt.

**The core insight**: even without labels, the model should be consistent — similar inputs should produce similar predictions. Entropy minimization pushes the model toward confident, low-entropy predictions on target data, which tends to improve accuracy even without supervision.

```python
# Simplest TTA: average predictions over augmented views
def predict_tta(model, x, n_augments=10):
    augmented_preds = [model(augment(x)) for _ in range(n_augments)]
    return torch.stack(augmented_preds).mean(0)
```

---

## 3. Few-Shot Learning

### The problem

Standard supervised learning needs hundreds to thousands of examples per class. Many real-world tasks cannot provide this: rare diseases, low-resource languages, newly emerging object categories. You need a model that can generalize from K examples (K = 1 or 5 typically).

### The core insight

The bottleneck is not the classifier — it's the feature space. If the feature extractor maps semantically similar inputs close together, a simple nearest-neighbor classifier needs only one or a few examples per class. The goal of few-shot learning is to learn a feature space where this proximity reflects semantic similarity — a space that generalizes to new classes not seen during training.

### Prototypical Networks

**The problem**: how do you classify a query image into one of N novel classes given only K examples per class?

**The core insight**: represent each class by the centroid (prototype) of its K support examples in feature space. Classify by nearest prototype. The model needs to learn an embedding space where this simple rule works across arbitrary new classes.

```python
def prototypical_forward(support_embeddings, query_embeddings, n_way, k_shot):
    # support_embeddings: (n_way * k_shot, embed_dim)
    prototypes = support_embeddings.reshape(n_way, k_shot, -1).mean(dim=1)
    # prototypes: (n_way, embed_dim) — one centroid per class

    # query_embeddings: (n_queries, embed_dim)
    dists = torch.cdist(query_embeddings, prototypes)   # (n_queries, n_way)
    return -dists   # negative distance as logits — closer = higher score
```

Training uses episodic training: sample many N-way K-shot episodes, compute prototypes, classify, backpropagate. The network never trains on fixed classes — it trains on the *task structure* of few-shot classification.

**What breaks**: prototypical networks assume the mean of support embeddings is a good class representative. For classes with multi-modal distributions in embedding space (e.g., "dog" covers many breeds), the prototype is a poor representative and nearest-neighbor fails.

### Matching Networks

**The problem**: a fixed prototype (mean) discards information about the distribution of support examples.

**The core insight**: instead of computing a single representative, use attention-weighted sum over all support examples. The attention weights are learned, not fixed as uniform average.

Classification of query q: `sum_i( a(q, s_i) * y_i )` where `a(q, s_i)` is attention between query and support example i.

### MAML — Model-Agnostic Meta-Learning

**The problem**: few-shot learning via ProtoNets or Matching Networks assumes the right embedding space is fixed after meta-training. But some tasks require the model to *adapt its parameters* to the task, not just find nearest neighbors.

**The core insight**: learn an initialization θ such that after just a few gradient steps on any new task, the model performs well. The initialization encodes a "readiness to adapt" — it sits at a point in parameter space where small moves in any task-relevant direction produce good task-specific models.

```
For each task T_i drawn from task distribution p(T):
    Sample support set D_i^train, query set D_i^test
    Adapt: θ'_i = θ - α * ∇_θ L(f_θ, D_i^train)       ← inner loop
    Evaluate adapted model: L(f_{θ'_i}, D_i^test)

Meta-update: θ ← θ - β * ∇_θ Σ_i L(f_{θ'_i}, D_i^test)  ← outer loop
```

The outer gradient requires differentiating through the inner gradient step — second-order derivatives. This is expensive.

**FOMAML** approximates by ignoring second-order terms: `θ ← θ - β * ∇_{θ'} Σ_i L(f_{θ'_i}, D_i^test)`. Empirically almost as good as full MAML, much cheaper.

**Reptile** is even simpler: `θ ← θ + ε * (θ'_i - θ)`. Just move the initialization toward where the task-adapted models end up. No second-order gradients, no inner/outer loop formalism.

**What breaks**: MAML's bi-level optimization is sensitive to hyperparameters. The inner loop learning rate α, the number of inner steps, and the outer loop LR β all interact. If the inner loop over-adapts, the meta-gradient becomes meaningless. MAML also assumes all tasks have the same structure, which fails when tasks are too heterogeneous.

### Relation Networks

**The problem**: prototypical networks use Euclidean distance, which is a fixed, potentially wrong similarity metric.

**The core insight**: learn the comparison function. Concatenate a query embedding with a prototype, pass through a learned network, and output a similarity score. The comparison function can capture complex non-Euclidean structure.

```
relation_score = relation_net(concat(query_embed, prototype_embed))
```

---

## 4. Zero-Shot Learning

### The problem

Training a classifier requires labeled examples of every class. But new classes constantly appear — new diseases, new product categories, new animal species. You cannot retrain the model every time.

### The core insight

Every class has semantic properties — attributes, descriptions, relationships to other classes. If you learn a mapping from visual features to semantic representations, you can classify images into classes you've never seen by reasoning about their semantic descriptions.

### Attribute-Based ZSL

Each class is described by a vector of semantic attributes. "Zebra" = [has_stripes=1, is_aquatic=0, is_predator=0, ...]. "Shark" = [has_stripes=0, is_aquatic=1, is_predator=1, ...].

`P(y | x) ∝ compatibility(f(x), a(y))`

where `f(x)` maps image to visual feature space, `a(y)` maps class to attribute space, and compatibility is a learned or fixed function measuring how well they match.

At test time, seen or unseen classes are distinguished by their attribute vectors. The model never sees a zebra image but knows what stripe-detection activations should look like for a striped animal.

**What breaks**: the semantic attribute space must be expressive enough to distinguish all classes, including unseen ones. If two classes are described by identical or very similar attribute vectors, zero-shot classification cannot separate them.

### CLIP-Based Zero-Shot

**The problem**: manually defining attributes for every class is labor-intensive and brittle.

**The core insight**: train a model to align visual and textual representations of the same concept in a shared embedding space. Natural language is a free-form attribute system that already describes every concept you care about.

CLIP trains image and text encoders jointly with contrastive loss: push the image embedding of "a photo of a dog" close to the text embedding of "a photo of a dog", and far from text embeddings of other captions.

At test time, zero-shot classification is just nearest neighbor in this shared space:

```python
import clip

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a cat", "a photo of a dog", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    # similarity[0] — probabilities for each class
```

No fine-tuning needed. Any class describable in natural language is classifiable.

**Generalized ZSL**: the harder and more realistic setting — at test time, both seen classes (from training) and unseen classes are candidates. Models tend to overpredict seen classes since they were present during training. Calibrating confidence between seen and unseen classes is an open problem.

---

## 5. Meta-Learning (Learning to Learn)

### The problem

A standard model learns one task. When that task ends, the knowledge is locked in its parameters — it cannot transfer to a new task without retraining from scratch (or forgetting what it knew). You want a model that gets better at *learning new tasks* as it sees more tasks.

### The core insight

Instead of training on one fixed dataset, train on a distribution of tasks. The model's job is not to perform well on any specific task, but to learn how to adapt quickly to new ones. The model learns a learning algorithm — an inductive bias for the task structure — rather than a solution to a specific problem.

### Categories of approaches

| Category | Examples | What is learned |
|----------|----------|-----------------|
| Optimization-based | MAML, Reptile, FOMAML | Good initialization for fast adaptation |
| Metric-based | ProtoNets, Matching Nets, Relation Nets | Embedding space where task structure is simple |
| Model-based | SNAIL, Memory-Augmented Networks | Learned update rule encoded in model architecture |

**Optimization-based** meta-learning trains for fast gradient-based adaptation. The model can change its weights for a new task.

**Metric-based** meta-learning trains a fixed embedding such that task-specific classification requires no weight updates — just nearest-neighbor queries. Faster at test time but less flexible.

**Model-based** meta-learning builds the adaptation mechanism into the architecture (e.g., LSTM hidden state as working memory, external memory bank for key-value lookup). Can adapt in a single forward pass without any gradient computation.

---

## 6. Self-Supervised Pre-Training

### The problem

Labeled data is expensive and scarce. But unlabeled data is abundant. If you could learn useful representations from unlabeled data, you'd have access to orders of magnitude more training signal.

### The core insight

Design a pretext task — a self-supervised objective that can be computed without human labels — such that solving it requires learning representations that generalize to downstream supervised tasks. The key insight is that certain prediction tasks (what's hidden in an image, what comes next in a sequence) force the model to understand structure in the data.

| Method | Pretext Task | Domain |
|--------|-------------|--------|
| SimCLR | Contrastive: representations of two augmented views of the same image should be similar; representations of different images should differ | Vision |
| DINO / DINOv2 | Self-distillation: student representation of a local crop should match teacher representation of global crop | Vision |
| MAE | Masked patch reconstruction: reconstruct 75%+ of masked patches from visible ones | Vision |
| BERT | Masked token prediction: predict masked tokens from surrounding context | NLP |
| GPT | Next-token prediction: predict the next token given all previous ones | NLP |

**Why these representations transfer**: the pretext task requires modeling the statistical structure of data — edges, textures, and shapes for vision; syntax, semantics, and world knowledge for language. These learned structures are exactly what downstream tasks need.

**What breaks**: the quality of transfer depends on alignment between the pretext task and the downstream task. MAE representations transfer well to dense prediction tasks (segmentation) because reconstructing patches requires spatial understanding. Contrastive methods transfer well to retrieval tasks because they explicitly learn a similarity structure. Misalignment reduces transfer quality.

---

## Key Interview Points

**Why does ImageNet pretraining help on unrelated domains?**
Deep networks learn a hierarchy of features — edges, textures, parts, objects. The early and middle layers are general-purpose visual feature detectors useful for any visual task, not ImageNet-specific patterns.

**Feature extraction vs fine-tuning — when to use each?**
Feature extraction (frozen backbone) when target data is scarce and source/target domains are similar. Full fine-tuning when target data is abundant, using discriminative LRs (lower for early layers) to preserve general features.

**What is catastrophic forgetting and how do you prevent it?**
The phenomenon where fine-tuning on a new task overwrites knowledge from pretraining. Prevent it with: gradual unfreezing, low learning rates for pretrained layers, discriminative LRs, regularization that penalizes large deviations from pretrained weights (EWC).

**What is DANN and how does the gradient reversal layer work?**
DANN learns domain-invariant features by adversarially training a domain classifier against the feature extractor. The GRL is identity during forward pass and multiplies gradients by -λ during backward pass — forcing the feature extractor to produce features that confuse the domain classifier, without a separate adversarial update loop.

**How do ProtoNets work?**
Represent each class by the centroid (mean embedding) of its support examples. Classify queries by nearest prototype. Train episodically on N-way K-shot tasks sampled from training classes. At meta-test time, the learned embedding space generalizes to new classes not seen during meta-training.

**What does MAML learn?**
An initialization θ that can be adapted to any new task in a few gradient steps. The outer loop learns the initialization; the inner loop simulates task-specific adaptation. The initialization encodes a prior over what makes a good starting point for the task distribution.

**How does CLIP enable zero-shot classification?**
CLIP trains image and text encoders with contrastive loss to align visual and linguistic representations of the same concept. Zero-shot classification is nearest neighbor: embed the image, embed text descriptions of each class, classify by which text embedding is closest.
