---
module: Emerging Topics
topic: Continual Learning
subtopic: ""
status: unread
tags: [emergingtopics, ml, continual-learning]
---
# Continual Learning (Lifelong Learning)

---

## 1. The Problem

You have a ResNet trained on ImageNet — 1.28 million images, 1000 classes, 76% top-1 accuracy. A hospital wants to deploy it to classify chest X-rays. You fine-tune on their 50,000 labeled radiographs. After training, the model achieves 91% on the held-out chest X-ray set.

Then you check its ImageNet accuracy: 12%.

The model has almost entirely forgotten how to classify cars, dogs, and furniture. All the computation and data that produced the original model is gone. This is **catastrophic forgetting** — the central problem in continual learning.

Why does it happen? Stochastic gradient descent finds a region of weight space where the loss is low for the current task. That region is not the same region where the loss was low for the previous task. In a high-dimensional parameter space, these low-loss regions are typically non-overlapping, and SGD has no mechanism to avoid moving between them. The loss landscape for task B does not reference task A at all.

This failure mode matters every time a model must adapt to new data in deployment: a language model updated with new events, a recommendation system adapting to new user behavior, a medical model updated as protocols change.

---

## 2. Three Different Responses

Catastrophic forgetting has three distinct causes, and each gives rise to a different family of solutions:

1. **The wrong weights changed.** Some weights are critical for old tasks; others are irrelevant. SGD treats all weights equally. Fix: identify critical weights and penalize changing them. This is regularization.

2. **The old data is gone.** SGD on task B never sees task A examples. The model can only optimize what it sees. Fix: keep old examples and mix them into current training. This is replay.

3. **There's no separate space for new tasks.** The new task overwrites the same weights the old task used. Fix: give each task dedicated capacity so old task weights are never modified. This is architecture.

Each response has a different cost — computation, memory, or model size — and a different failure mode.

---

## 3. The Mechanics

### Settings

| Setting | What changes between tasks | Task ID at test? |
|---------|--------------------------|-----------------|
| **Task-Incremental** | Different tasks (distinct label spaces) | Yes (given) |
| **Domain-Incremental** | Same labels, different distributions | No |
| **Class-Incremental** | New classes added each task | No |

Class-incremental is hardest: the model must distinguish new and old classes at test time without knowing which task a sample belongs to.

---

### Regularization-Based: EWC

The problem with naive L2 regularization toward old weights is that all parameters are treated equally. But some weights matter enormously for old tasks while others are nearly irrelevant.

**The insight:** the curvature of the loss surface around the optimal weights for task A tells you which parameters matter. Parameters at a sharp optimum — where small changes cause large loss increases — are critical. Parameters in a flat region can be moved freely.

**Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017)** uses the diagonal of the Fisher information matrix as a curvature estimate:

```
L_total = L_B(θ) + Σ_i (λ/2) F_i (θ_i - θ*_{A,i})²
```

- `F_i`: Fisher information for parameter i, estimated from task A gradients
- `θ*_A`: optimal weights after task A
- `λ`: regularization strength

```python
def compute_fisher(model, dataset, device):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x, y in DataLoader(dataset, batch_size=64):
        model.zero_grad()
        output = model(x.to(device))
        loss = F.nll_loss(F.log_softmax(output, dim=1), y.to(device))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2 / len(dataset)
    return fisher

def ewc_loss(model, fisher, optima, lamda=5000):
    loss = 0
    for n, p in model.named_parameters():
        if n in fisher:
            loss += (fisher[n] * (p - optima[n]) ** 2).sum()
    return (lamda / 2) * loss
```

**Synaptic Intelligence (SI):** online version of EWC — accumulate importance during training rather than computing it post-hoc from the final checkpoint:

```
Ω_k ≈ Σ_t (∂L/∂θ_k) * Δθ_k / (Δθ_k² + ξ)
```

**What breaks:** The Fisher approximation is diagonal — it ignores correlations between parameters. For many sequential tasks, the EWC penalty becomes so restrictive that the model cannot learn new tasks at all (intransigence). Also requires storing O(|θ|) Fisher values per task.

---

### Replay-Based: Experience Replay

The direct solution: keep a buffer of old examples and mix them into every training batch. The model cannot forget what it keeps seeing.

```python
class ReplayBuffer:
    def __init__(self, capacity, strategy='random'):
        self.capacity = capacity
        self.buffer = []
        self.strategy = strategy

    def add(self, x, y):
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            idx = np.random.randint(len(self.buffer))
            self.buffer[idx] = (x, y)

    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))

for (x_new, y_new) in new_task_loader:
    replay_samples = buffer.sample(batch_size)
    x_old, y_old = zip(*replay_samples)
    x_combined = torch.cat([x_new, torch.stack(x_old)])
    y_combined = torch.cat([y_new, torch.stack(y_old)])
    loss = criterion(model(x_combined), y_combined)
    loss.backward()
    optimizer.step()
```

**Buffer management strategies:**
- **Random replacement:** simplest; biased toward recent data
- **Reservoir sampling:** maintains an unbiased random sample of the full data stream — each example from the past has equal probability of remaining in the buffer
- **Greedy coreset (herding):** select samples that best represent the class distribution in feature space
- **Ring buffer:** FIFO per class

**iCaRL** (Incremental Classifier and Representation Learning) for class-incremental CL:
1. Learn features with distillation (keep old class representations stable)
2. Store representative exemplars per class via herding
3. Classify by nearest-mean-of-exemplars in feature space

**Gradient Episodic Memory (GEM):** instead of mixing data, constrain the optimization. The gradient for new task training must not increase loss on any replay sample:

```
minimize L_new(θ)   s.t.  ⟨g_new, g_task_i⟩ ≥ 0  ∀ previous tasks i
```

Project the new-task gradient if it violates any constraint. Guarantees non-increasing loss on old tasks.

**A-GEM:** single constraint using the averaged gradient from the replay buffer. Much cheaper than GEM at some performance cost.

**What breaks:** Buffer size is a hard constraint. With 10 tasks and a 1000-sample buffer, each task gets 100 examples — far too few for complex tasks. Privacy regulations (GDPR, HIPAA) may prohibit storing historical examples.

---

### Architecture-Based: Dedicated Capacity Per Task

The purest solution: old weights are never modified. Each new task gets its own parameters. Zero forgetting by construction.

**Progressive Neural Networks (Rusu et al., 2016):** add a new column of neurons for each task. Connect to all previous columns via lateral connections (frozen).

```
Task 1:  [Column 1 (frozen)]
Task 2:  [Column 1 → Column 2]  (lateral: C1 features → C2)
Task 3:  [Column 1 → Column 3]  (lateral: C1, C2 features → C3)
          [Column 2 → Column 3]
```

**What breaks:** linear growth in model size with number of tasks.

**PackNet (Mallya & Lazebnik, 2018):** fixed model size. After each task, iteratively prune weights and reassign freed capacity. Binary masks define which weights serve which task.

**HAT (Hard Attention to the Task):** learn a binary mask per task via trainable attention units. Masks protect task-specific parameters from gradient updates on new tasks.

**LoRA-based continual learning:** train separate LoRA adapters per task. Switch adapters at inference. Memory-efficient, no forgetting, but requires knowing the task ID at inference time.

```python
from peft import PeftModel

model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model_task1 = PeftModel.from_pretrained(model, "lora_task1/")
model_task2 = PeftModel.from_pretrained(model, "lora_task2/")
```

---

### Generative Replay (Pseudo-Rehearsal)

The buffer privacy problem has a workaround: instead of storing real examples, train a generative model (VAE or GAN) on each task's distribution and generate synthetic replays.

```
After Task A:
    Train G_A (generator) on Task A data

When learning Task B:
    Generate synthetic Task A samples: x_fake ~ G_A
    Train classifier on Task B data + x_fake
```

**What breaks:** the generator itself forgets across tasks (recursive catastrophic forgetting). Generator quality degrades, especially on complex distributions. This is an active research problem with no clean solution.

---

### Neural Architecture Search (NAS)

When the task sequence is not fixed in advance, the capacity allocation problem becomes architectural. NAS automates architecture search.

**Search spaces:**
- Cell-based: search for repeating cell structures (macro-architecture fixed)
- Layer-wise: search operation types per layer
- Global: full architecture search

**Search strategies:**

| Strategy | Description | Cost |
|----------|------------|------|
| **Grid/Random Search** | Enumerate architectures | O(n) training runs |
| **Reinforcement Learning** | Controller predicts architecture | Many training runs |
| **Evolutionary Algorithms** | Mutate and select architectures | Many training runs |
| **DARTS (Gradient-based)** | Relax discrete choices to continuous, optimize jointly | Single training run |
| **One-Shot / SuperNet** | Train a supernetwork containing all architectures | Single training run |

**DARTS — Differentiable Architecture Search:** relax the discrete choice of operation at each edge to a softmax mixture:

```
ō(x) = Σ_o (exp(α_o) / Σ_{o'} exp(α_{o'})) o(x)
```

Learn architecture parameters α and network weights w jointly via bi-level optimization:

```
minimize_{α} L_val(w*(α), α)
subject to  w*(α) = argmin_w L_train(w, α)
```

After training, discretize by taking argmax over α at each edge.

**Once-For-All (OFA):** train a single supernetwork supporting many sub-networks with elastic width, depth, and kernel size. At deployment, extract a specialized sub-network for any hardware constraint without retraining.

---

### Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Average Accuracy** | `Acc = (1/T) Σ_i a_{T,i}` | Performance across all tasks after final task |
| **Backward Transfer** | `BWT = (1/T-1) Σ_{i<T} (a_{T,i} - a_{i,i})` | Negative = forgetting; positive = improvement |
| **Forward Transfer** | `FWT = (1/T-1) Σ_{i>1} (a_{i-1,i} - b_i)` | How much old learning helps new tasks |
| **Intransigence** | — | Inability to learn new tasks (from over-regularization) |

---

### Practical CL in Production

**Concept drift:** distribution changes gradually. Retrain periodically or on trigger (monitor validation performance).

**Rolling window:** train on recent data only. Simple; forgets old patterns by design.

**Continual pre-training for LLMs:** updated incrementally with new data (e.g., monthly checkpoints). Mix new data with old via replay to prevent forgetting:

```
L_total = L_task_new + λ · KL(f_old(x) || f_new(x))
```

---

## 4. What Breaks

**Regularization becomes too restrictive at scale.** EWC with many tasks accumulates constraints that prevent learning anything new. The diagonal Fisher approximation also underestimates true importance when parameters are correlated.

**Replay fails under privacy constraints.** GDPR and HIPAA may prohibit storing historical training data. Generative replay degrades in quality and itself suffers from forgetting.

**Architecture methods hit capacity limits.** Progressive networks grow linearly. PackNet and HAT require task IDs at inference, which are often unavailable in domain-incremental and class-incremental settings.

**Class-incremental is fundamentally hard.** Without task ID at test time, the model must distinguish all old and new classes simultaneously. The output head must grow, the decision boundary must not collapse old classes, and replay is the only consistently effective method — but it requires stored examples.

**NAS does not solve the CL problem.** It finds efficient architectures; it does not address how those architectures should be updated when new tasks arrive. Treating NAS as a CL solution conflates two distinct problems.

---

## Key Interview Points

- Catastrophic forgetting: SGD for task B finds a low-loss region that is high-loss for task A. No mechanism in standard training prevents this.
- EWC, replay, and progressive networks each address a different aspect of the forgetting problem: EWC protects the weights that matter (via Fisher-weighted regularization); replay keeps old data visible; progressive networks give each task separate capacity.
- EWC adds L2 penalty weighted by Fisher information (diagonal curvature estimate). Importance-weighted regularization, not flat.
- Experience replay is empirically the most effective general approach. Reservoir sampling gives an unbiased buffer sample from the full data stream.
- Progressive networks have zero forgetting by design but linear parameter growth.
- Class-incremental is the hardest setting: no task ID at test time, must distinguish all classes, output head must grow.
- In production LLMs, continual pre-training with data replay (old + new documents mixed) is standard practice.
- DARTS enables gradient-based NAS in a single training run via a softmax relaxation over operation choices; Once-For-All trains one supernetwork deployable to any hardware constraint.

---

## Canonical Interview Q&As

**Q1: Derive the EWC objective from first principles. Why does the diagonal Fisher approximation work, and where does it fail?**

EWC goal: after training on task A, find parameters for task B that perform well on both. Formally, we want to constrain the posterior `p(θ | D_A, D_B)`.

Bayesian derivation:
```
log p(θ | D_A, D_B) = log p(D_B | θ) + log p(θ | D_A) - log p(D_B)
                    ≈ log p(D_B | θ) + log p(θ | D_A)
```

The key is approximating `log p(θ | D_A)`. Using a Laplace approximation around the MAP estimate `θ_A*`:
```
log p(θ | D_A) ≈ log p(θ_A* | D_A) - (1/2)(θ - θ_A*)ᵀ F_A (θ - θ_A*)
```

where `F_A` is the Fisher information matrix:
```
F_A = E_{x~D_A}[ ∇_θ log p(y|x,θ) · ∇_θ log p(y|x,θ)ᵀ ]
    ≈ (1/N) Σ_i [ g_i · g_iᵀ ]  (empirical Fisher)
```

This gives the EWC objective:
```
L_EWC(θ) = L_B(θ) + (λ/2) Σ_i F_{A,ii} (θ_i - θ*_{A,i})²
```

The **diagonal approximation** (`F_{A,ii}` instead of full `F_A`) assumes parameters are independent — their joint importance equals the product of marginal importances. This is computationally necessary (full F_A is d×d, unaffordable) but wrong when:
- Parameter correlations are high (early layers with shared feature detectors)
- Multiple tasks share parameters that are important in correlated directions
- Network is over-parameterized with degenerate directions in Fisher

In practice: diagonal EWC works well for sequential task pairs and simple architectures. With >10 tasks, accumulated constraints become over-restrictive (the constraint set intersection shrinks), and the diagonal approximation causes EWC to over-protect parameters that appear important marginally but are not important jointly.

---

**Q2: Compare experience replay, EWC, and progressive networks on the Class-Incremental benchmark. Why is class-incremental the hardest CL setting?**

Three CL settings defined by what the model knows at test time:
```
Task-Incremental:     Task ID known at test → separate head per task → easy
Domain-Incremental:   No task ID, same output space → harder
Class-Incremental:    No task ID, output space grows → hardest
```

In **Class-Incremental**, the model must:
1. Correctly classify all previously seen classes (no forgetting)
2. Correctly classify new classes (plasticity)
3. Do both without knowing which task the test example comes from
4. Grow the output head without disturbing old class representations

Performance comparison on Split-CIFAR-100 (5 tasks × 20 classes):

| Method | Class-Inc Accuracy | Notes |
|--------|-------------------|-------|
| Fine-tuning (baseline) | ~15% | Catastrophic forgetting |
| EWC | ~25-30% | Constraints help but head collapse remains |
| Progressive Networks | ~65-70% | Zero forgetting; linear growth; task ID implicitly needed for routing |
| Experience Replay (ER-ACE) | ~55-60% | 200 exemplars/class |
| DER++ | ~60-65% | Replay + knowledge distillation on stored logits |
| MEMO | ~68-72% | SOTA: memory + model expansion |

Why EWC struggles in Class-Inc: even if weights are protected, the output head must grow. Adding new class nodes disturbs the softmax distribution over old classes. Old class probabilities are renormalized by the new partition function. EWC protects weights but not the classification boundary in the output space.

Why replay is the dominant approach: it directly solves the fundamental problem — old examples remain in the gradient signal, so the loss landscape for old tasks is not abandoned. The challenge is buffer size (realistic constraint: 1-5% of original data) and class imbalance (new task has full data, old tasks have tiny replay buffer → imbalanced gradient updates).

---

**Q3: You're building a recommendation system that must continuously learn from new user interactions without forgetting long-tail user preferences. Design the architecture.**

Long-tail preferences are the hardest case: few training examples, high variance, low replay frequency.

**Architecture:**

```
1. Embedding layer: user/item embeddings (most sensitive to forgetting)
2. Interaction model: MLP tower (standard continual learning problem)
3. Output: ranking scores
```

**Forgetting failure modes specific to rec systems:**
- Long-tail users appear rarely in replay buffers → under-represented → over-forgotten
- New items have no embeddings → cold-start problem entangled with forgetting
- Popularity shift: yesterday's trending items are over-represented in memory

**Solution stack:**

```python
# 1. Reservoir sampling with stratification
class StratifiedBuffer:
    def __init__(self, capacity, strata_fn):
        self.buckets = defaultdict(list)  # strata_fn assigns bucket (e.g., activity percentile)
        self.capacity_per_bucket = capacity // NUM_STRATA
    
    def update(self, interaction):
        bucket = strata_fn(interaction.user_id)
        if len(self.buckets[bucket]) < self.capacity_per_bucket:
            self.buckets[bucket].append(interaction)
        else:
            # Reservoir replacement
            idx = random.randint(0, self.stream_count[bucket])
            if idx < self.capacity_per_bucket:
                self.buckets[bucket][idx] = interaction

# 2. EWC on embedding layer (protect long-tail user embeddings specifically)
fisher_weights = compute_fisher(model, long_tail_users, data_loader)
ewc_loss = sum(f * (p - p_old)**2 for f, p, p_old in zip(fisher_weights, params, old_params))

# 3. Distillation loss on old predictions to stabilize ranking order
distill_loss = KL(old_model(replay_batch), new_model(replay_batch))

# Total loss
L = L_new + λ_replay * L_replay + λ_ewc * ewc_loss + λ_distill * distill_loss
```

**Infrastructure:** maintain a sliding "importance score" per user (inverse interaction frequency). This drives both replay sampling priority and EWC Fisher weighting — rare users get higher protection. Checkpoint every 24 hours; evaluate BWT on long-tail held-out cohort specifically.

---

**Q4: What is the relationship between NAS and continual learning? When is OFA relevant to production ML?**

NAS and CL solve orthogonal problems that compose naturally in production:

**NAS** answers: given a fixed task and hardware constraint, what architecture maximizes accuracy/latency tradeoff?

**CL** answers: given a fixed architecture, how do we update weights over time without forgetting?

They interact in **hardware-heterogeneous deployments**: a recommendation system must serve users on phones (2GB RAM), tablets (6GB RAM), and servers (40GB GPU). Once-For-All (OFA) trains a single supernetwork from which multiple sub-networks are extracted:

```
Supernetwork training:
  1. Train full network normally (establish supernet weights)
  2. Progressive shrinking: train with randomly sampled sub-networks at each step
     - Elastic depth: {2,3,4} blocks active
     - Elastic width: {0.25, 0.35, 0.5} × full width
     - Elastic kernel: {3,5,7} conv kernels

Deployment:
  - Given hardware constraint (latency budget T on device D):
    - Run accuracy predictor + latency predictor on 1000 sub-network configs
    - Select Pareto-optimal sub-network: max accuracy s.t. latency ≤ T
    - No retraining — extract weights directly from supernetwork
```

**Where CL applies to OFA in production:** when training data evolves (e.g., new product categories), the supernetwork itself must be updated without re-extracting and re-validating all previously deployed sub-networks. This is a CL problem at the supernetwork level — update the supernetwork weights on new data while preserving the accuracy of all already-deployed sub-network configurations.

Practical relevance: OFA is widely used in on-device ML (Apple Neural Engine search, TensorFlow Model Optimization Toolkit). The continual update problem for deployed OFA networks is largely unsolved in the literature and is an active research area.

## Flashcards

**F_i?** #flashcard
Fisher information for parameter i, estimated from task A gradients

**θ*_A?** #flashcard
optimal weights after task A

**λ?** #flashcard
regularization strength

**Random replacement?** #flashcard
simplest; biased toward recent data

**Reservoir sampling: maintains an unbiased random sample of the full data stream?** #flashcard
each example from the past has equal probability of remaining in the buffer

**Greedy coreset (herding)?** #flashcard
select samples that best represent the class distribution in feature space

**Ring buffer?** #flashcard
FIFO per class

**Cell-based?** #flashcard
search for repeating cell structures (macro-architecture fixed)

**Layer-wise?** #flashcard
search operation types per layer

**Global?** #flashcard
full architecture search

**Catastrophic forgetting?** #flashcard
SGD for task B finds a low-loss region that is high-loss for task A. No mechanism in standard training prevents this.

**EWC, replay, and progressive networks each address a different aspect of the forgetting problem?** #flashcard
EWC protects the weights that matter (via Fisher-weighted regularization); replay keeps old data visible; progressive networks give each task separate capacity.

**EWC adds L2 penalty weighted by Fisher information (diagonal curvature estimate). Importance-weighted regularization, not flat.?** #flashcard
EWC adds L2 penalty weighted by Fisher information (diagonal curvature estimate). Importance-weighted regularization, not flat.

**Experience replay is empirically the most effective general approach. Reservoir sampling gives an unbiased buffer sample from the full data stream.?** #flashcard
Experience replay is empirically the most effective general approach. Reservoir sampling gives an unbiased buffer sample from the full data stream.

**Progressive networks have zero forgetting by design but linear parameter growth.?** #flashcard
Progressive networks have zero forgetting by design but linear parameter growth.

**Class-incremental is the hardest setting?** #flashcard
no task ID at test time, must distinguish all classes, output head must grow.

**In production LLMs, continual pre-training with data replay (old + new documents mixed) is standard practice.?** #flashcard
In production LLMs, continual pre-training with data replay (old + new documents mixed) is standard practice.

**DARTS enables gradient-based NAS in a single training run via a softmax relaxation over operation choices; Once-For-All trains one supernetwork deployable to any hardware constraint.?** #flashcard
DARTS enables gradient-based NAS in a single training run via a softmax relaxation over operation choices; Once-For-All trains one supernetwork deployable to any hardware constraint.

**Parameter correlations are high (early layers with shared feature detectors)?** #flashcard
Parameter correlations are high (early layers with shared feature detectors)

**Multiple tasks share parameters that are important in correlated directions?** #flashcard
Multiple tasks share parameters that are important in correlated directions

**Network is over-parameterized with degenerate directions in Fisher?** #flashcard
Network is over-parameterized with degenerate directions in Fisher

**Long-tail users appear rarely in replay buffers → under-represented → over-forgotten?** #flashcard
Long-tail users appear rarely in replay buffers → under-represented → over-forgotten

**New items have no embeddings → cold-start problem entangled with forgetting?** #flashcard
New items have no embeddings → cold-start problem entangled with forgetting

**Popularity shift?** #flashcard
yesterday's trending items are over-represented in memory

**Elastic depth?** #flashcard
{2,3,4} blocks active

**Elastic width?** #flashcard
{0.25, 0.35, 0.5} × full width

**Elastic kernel?** #flashcard
{3,5,7} conv kernels

**Given hardware constraint (latency budget T on device D):?** #flashcard
Given hardware constraint (latency budget T on device D):

**Run accuracy predictor + latency predictor on 1000 sub-network configs?** #flashcard
Run accuracy predictor + latency predictor on 1000 sub-network configs

**Select Pareto-optimal sub-network?** #flashcard
max accuracy s.t. latency ≤ T

**No retraining?** #flashcard
extract weights directly from supernetwork
