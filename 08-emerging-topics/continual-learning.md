# Continual Learning (Lifelong Learning)

Continual learning (CL) enables models to learn a sequence of tasks without forgetting previously acquired knowledge. A fundamental requirement for real-world AI systems that must adapt to new data over time.

---

## The Problem: Catastrophic Forgetting

When a neural network is fine-tuned on a new task, gradient updates overwrite weights important for old tasks. Performance on old tasks collapses — this is **catastrophic forgetting** (catastrophic interference).

```
Train on Task A → accuracy 90% on A
Train on Task B → accuracy 92% on B, but only 10% on A (catastrophic forgetting)
```

**Why it happens:** SGD finds a low-loss region for Task B, but this region has high loss for Task A. The loss landscape is shared.

---

## Continual Learning Settings

| Setting | What changes between tasks | Task ID at test? |
|---------|--------------------------|-----------------|
| **Task-Incremental** | Different tasks (distinct label spaces) | Yes (given) |
| **Domain-Incremental** | Same labels, different distributions | No |
| **Class-Incremental** | New classes added each task | No |

Class-incremental is hardest: must distinguish new and old classes without task labels at test time.

---

## Approaches

### 1. Regularization-Based

Protect important weights from changing too much when learning new tasks.

#### EWC — Elastic Weight Consolidation (Kirkpatrick et al., 2017)

Add a penalty for changing weights important to previous tasks. Importance = Fisher information (curvature of loss with respect to each parameter).

`L_total = L_B(θ) + Σ_i (λ/2) F_i (θ_i - θ*_{A,i})²`

- `F_i`: Fisher information for parameter i (estimated from task A data)
- `θ*_A`: optimal weights for task A
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

**Limitation:** Fisher approximation is diagonal (ignores parameter correlations). For many tasks, the penalty becomes too restrictive.

#### SI — Synaptic Intelligence

Online version of EWC: accumulate importance during training (not post-hoc).

`Ω_k ≈ Σ_t (∂L/∂θ_k) * Δθ_k / (Δθ_k² + ξ)`

#### Online EWC / L2 Regularization

Simplified version: regularize towards previous task's optimal weights with L2 penalty. Weaker but computationally cheap.

---

### 2. Replay-Based

Store or generate samples from previous tasks and interleave them with new task data.

#### Experience Replay

Maintain a memory buffer `M` of previous task examples. Train on mix of new data + buffer.

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
            # Random replacement
            idx = np.random.randint(len(self.buffer))
            self.buffer[idx] = (x, y)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))

# Training with replay
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
- **Random:** Simple uniform sampling
- **Reservoir sampling:** Maintains unbiased random sample of the stream
- **Greedy coreset (herding):** Select samples that best represent class distribution
- **Ring buffer:** FIFO per class

#### iCaRL — Incremental Classifier and Representation Learning

For class-incremental CL:
1. Learn features with distillation (keep old class features stable)
2. Store representative exemplars (herding selection) per class
3. Classify by nearest-mean-of-exemplars in feature space

#### Gradient Episodic Memory (GEM)

Solve a constrained optimization: new task gradient should not increase loss on replay samples.

`minimize L_new(θ)   s.t.  ⟨g_new, g_task_i⟩ ≥ 0  ∀ previous tasks i`

Project the new task gradient to satisfy constraints. Guarantees no increase in old-task loss.

#### A-GEM (Averaged GEM)

Simplification: single constraint using averaged gradient from replay buffer. Much cheaper than GEM.

---

### 3. Architecture-Based

Allocate different model capacity to different tasks.

#### Progressive Neural Networks (Rusu et al., 2016)

Add a new column of neurons for each new task. Connect to all previous columns via lateral connections (frozen). Old weights never modified.

```
Task 1:  [Column 1 (frozen)]
Task 2:  [Column 1 → Column 2]  (lateral: C1 features → C2)
Task 3:  [Column 1 → Column 3]  (lateral: C1, C2 features → C3)
          [Column 2 → Column 3]
```

**Pro:** Zero forgetting by design.  
**Con:** Linear growth in model size with number of tasks.

#### PackNet (Mallya & Lazebnik, 2018)

Iteratively prune weights after each task; reassign freed capacity to new tasks. Fixed model size, masks define which weights serve which task.

#### HAT — Hard Attention to the Task

Learn a binary mask per task via trainable attention units. Masks protect task-specific parameters from overwriting.

#### LoRA-Based Approaches (for LLMs)

Train separate LoRA adapters per task. Switch adapters at inference. Memory-efficient continual learning without forgetting.

```python
# Load task-specific LoRA weights at inference
from peft import PeftModel

model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model_task1 = PeftModel.from_pretrained(model, "lora_task1/")
model_task2 = PeftModel.from_pretrained(model, "lora_task2/")
```

---

### 4. Generative Replay (Pseudo-Rehearsal)

Instead of storing real samples, train a generative model (VAE or GAN) on each task. For new tasks, generate synthetic old-task samples for replay.

```
After Task A:
    Train G_A (generator) on Task A data
    
When learning Task B:
    Generate synthetic Task A samples: x_fake ~ G_A
    Train classifier on Task B data + x_fake
```

**Limitation:** Generator quality degrades across many tasks; generator itself forgets (recursion problem).

---

## Neural Architecture Search (NAS)

NAS automates the design of neural network architectures, searching for the best architecture for a given task.

### Search Spaces

- **Cell-based:** Search for repeating cells (macro-architecture fixed)
- **Layer-wise:** Search operation types per layer
- **Global:** Full architecture search

### Search Strategies

| Strategy | Description | Cost |
|----------|------------|------|
| **Grid/Random Search** | Enumerate architectures | O(n) training runs |
| **Reinforcement Learning** | Controller predicts architecture | Many training runs |
| **Evolutionary Algorithms** | Mutate and select architectures | Many training runs |
| **DARTS (Gradient-based)** | Relax discrete choices to continuous, optimize jointly | Single training run |
| **One-Shot / SuperNet** | Train a supernetwork containing all architectures | Single training run |

### DARTS — Differentiable Architecture Search

Relax the discrete choice of operation at each edge (e.g., 3×3 conv, 5×5 conv, skip, max-pool) to a mixture:

`ō(x) = Σ_o (exp(α_o) / Σ_{o'} exp(α_{o'})) o(x)`

Learn architecture parameters α and network weights w jointly via bi-level optimization.

```
minimize_{α} L_val(w*(α), α)
subject to  w*(α) = argmin_w L_train(w, α)
```

After training, discretize by taking argmax over α at each edge.

### Once-For-All (OFA)

Train a single supernetwork supporting many sub-networks (elastic width, depth, kernel size). At deployment, extract a specialized sub-network for any hardware constraint without retraining.

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Average Accuracy** | `Acc = (1/T) Σ_i a_{T,i}` | Performance across all tasks after final task |
| **Backward Transfer** | `BWT = (1/T-1) Σ_{i<T} (a_{T,i} - a_{i,i})` | Negative = forgetting; positive = improvement |
| **Forward Transfer** | `FWT = (1/T-1) Σ_{i>1} (a_{i-1,i} - b_i)` | How much old learning helps new tasks |
| **Intransigence** | Inability to learn new tasks (often from over-regularization) | — |

---

## Practical CL in Production

**Concept drift:** Distribution changes gradually. Retrain periodically or trigger-based (monitor performance on held-out validation).

**Rolling window:** Train on recent data only. Simple but forgets old patterns.

**Continual pre-training:** LLMs updated incrementally with new data (e.g., monthly checkpoint releases). Use data replay (mix new data with old) to prevent forgetting.

**Distillation-based update:** Use old model as teacher when training on new data:
`L_total = L_task_new + λ · KL(f_old(x) || f_new(x))`

---

## Key Interview Points

- Catastrophic forgetting is the core problem: SGD for new tasks overwrites old task representations.
- EWC adds L2 penalty weighted by Fisher information — protects important weights.
- Experience replay is empirically the most effective approach; reservoir sampling gives unbiased buffer.
- Progressive networks avoid forgetting entirely but grow linearly with task count.
- Class-incremental learning is hardest: model must distinguish old and new classes without task ID.
- In production LLMs, continual pre-training with data replay (old + new documents) is standard.
- NAS automates architecture search; DARTS enables gradient-based search in a single training run; OFA enables once-train, many-deploy.
