# PyTorch Foundations

---

## 1. Tensors

### The problem

Neural networks operate on batches of multi-dimensional numeric data: images are `[batch, channels, height, width]`, sequences are `[batch, time, features]`, weight matrices live somewhere in between. You need a data structure that can represent any of these shapes, run efficiently on GPU, and participate in automatic differentiation. NumPy arrays solve the first two but not the third. Raw GPU arrays solve the second but not the first or third. Tensors solve all three.

### The core insight

A tensor is a multi-dimensional array with two extra properties: it knows which device it lives on (CPU or GPU), and it can optionally record every operation done to it so that gradients can be computed later. Everything else is convenience.

### The mechanics

```python
import torch

# Creation
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from data
x = torch.zeros(3, 4)
x = torch.randn(2, 3, 4)                       # standard normal
x = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, steps=5)              # [0, 0.25, 0.5, 0.75, 1.0]

# Shape inspection
x = torch.randn(2, 3, 4)
x.shape        # torch.Size([2, 3, 4])
x.ndim         # 3
x.numel()      # 24

# Reshape operations — all preserve total element count
x = torch.randn(6, 4)
x.view(24)              # requires contiguous memory layout
x.reshape(24)           # handles non-contiguous memory — prefer this
x.unsqueeze(0)          # [1, 6, 4] — insert a new dim
x.squeeze()             # remove all size-1 dims
x.permute(1, 0)         # [4, 6] — reorder axes
x.transpose(0, 1)       # swap exactly two axes
```

**Why `.view()` vs `.reshape()`**: tensors in PyTorch may not be contiguous in memory after operations like `.permute()` or `.transpose()`. `.view()` requires contiguous layout and errors otherwise; `.reshape()` calls `.contiguous()` first if needed. Use `.reshape()` unless you explicitly want the error (to catch unintended copies).

### Dtype and device

The dtype controls memory footprint and precision. Getting it wrong is a common source of subtle bugs.

```python
x = torch.randn(3, 3)
x.dtype          # torch.float32 — default for model weights and activations

x = x.to(torch.float16)   # explicit cast
x = x.half()              # shorthand for float16
x = x.bfloat16()          # preferred for training on A100/H100 — same exponent range as float32
x = x.long()              # int64 — standard for integer indices and labels

# Device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
x = x.cuda()        # shorthand
x = x.cpu()         # move to CPU — required before .numpy()
```

**What breaks**: mixing devices in any operation raises `RuntimeError`. Mixing dtypes in operations usually raises an error rather than silently casting — intentional, to force you to be explicit.

```python
a = torch.randn(3).cuda()
b = torch.randn(3).cpu()
a + b   # RuntimeError: Expected all tensors to be on device cuda
```

### Indexing and math

```python
x = torch.randn(4, 8)
x[0]              # first row — shape [8]
x[:, 2]           # third column — shape [4]
x[1:3, 4:6]       # slice — shape [2, 2]

mask = x > 0
x[mask]           # 1D tensor of positive elements

# Matrix operations
x = torch.randn(3, 4)
w = torch.randn(4, 5)
x @ w             # matmul — [3, 5]

# Batched matmul — broadcasts over the first dim
A = torch.randn(8, 3, 4)
B = torch.randn(8, 4, 5)
A @ B             # [8, 3, 5]

# Reductions
a = torch.randn(3, 4)
a.sum()                        # scalar
a.sum(dim=0)                   # sum along rows → shape [4]
a.sum(dim=1, keepdim=True)     # shape [3, 1] — keepdim preserves rank for broadcasting
a.argmax(dim=1)                # index of max per row
```

---

## 2. Autograd

### The problem

To update a neural network's weights, you need the gradient of the loss with respect to every parameter. A network with millions of parameters, composed of dozens of operations, cannot have its gradients derived by hand. You need a system that watches what operations you perform and automatically derives the gradient computation.

### The core insight

Every differentiable operation has a known local derivative. If you record the sequence of operations in a graph, you can apply the chain rule by traversing that graph backward from the loss to each parameter. You never have to write a single gradient formula by hand — only the forward computation.

### The mechanics

PyTorch builds a dynamic directed acyclic graph (DAG) during every forward pass. Each tensor produced by an operation carries a `grad_fn` that knows how to compute the gradient of that operation's output with respect to its inputs. `.backward()` traverses this graph from the loss, multiplying local gradients together via the chain rule, and accumulates the results into `.grad` on each leaf tensor.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1    # y = x² + 3x + 1, dy/dx = 2x + 3

y.backward()
print(x.grad)              # tensor(7.)  — 2(2) + 3 = 7

# Multi-variable
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
loss = (a * b).sum()

loss.backward()
print(a.grad)   # tensor([4., 5., 6.]) — d(sum(a*b))/da = b
print(b.grad)   # tensor([1., 2., 3.]) — d(sum(a*b))/db = a

# Inspecting the graph
z = (x ** 2).sum()
print(z.grad_fn)                    # <SumBackward0>
print(z.grad_fn.next_functions)     # [(PowBackward0, 0)]
```

**Why model parameters have `requires_grad=True` by default**: `nn.Parameter` wraps a tensor and sets `requires_grad=True` automatically. Input data tensors don't need it — you're not optimizing inputs.

### `optimizer.zero_grad()`

**The problem**: PyTorch accumulates gradients — every `.backward()` call *adds* to whatever `.grad` is already stored. For a standard training loop where you compute one loss per batch, you don't want accumulation. You want fresh gradients each step.

**Why accumulation is the default**: some training patterns deliberately accumulate. Gradient accumulation (running multiple small batches before one optimizer step) requires it. Multi-task loss terms that are computed separately also require it. Making accumulation the default makes those patterns trivial — you just call `.backward()` multiple times and then step.

**The mechanics**: `optimizer.zero_grad()` sets `.grad` to `None` (with `set_to_none=True`, the default since PyTorch 2.0) or to zero for all tracked parameters. Call it before `loss.backward()` each step, or after `optimizer.step()` — either is fine as long as you call it before the *next* backward pass.

```python
# Standard placement — zero before backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Also valid — zero after step
optimizer.step()
optimizer.zero_grad()
# (gradients from this step are cleared before the next batch begins)

# set_to_none=True is slightly faster — avoids writing zeros, just deallocates
optimizer.zero_grad(set_to_none=True)
```

**What breaks**: forgetting to call it causes gradients to accumulate across steps. The effective gradient on step N is the sum of gradients from steps 1 through N. Training appears to work but with chaotically scaled updates — often the loss explodes within a few steps, or converges to a worse solution.

### `.detach()`

**The problem**: sometimes you need the *value* of a tensor computed during the forward pass, but you don't want gradients to flow back through it. Examples: using a frozen target network in RL, computing a reward signal that should not update the policy network, converting a tensor to NumPy for logging.

**The core insight**: the gradient graph is just a chain of references. If you sever the chain at a specific tensor, no gradient flows further back through that path. You still have the value — you just have no `grad_fn`.

**The mechanics**: `.detach()` returns a new tensor sharing the same storage but with no `grad_fn` and `requires_grad=False`.

```python
# RL: target network value used to compute loss, but target network itself
# should NOT be updated by this gradient
target_q = target_net(next_states).max(dim=1).values.detach()
loss = F.mse_loss(current_q, target_q)
loss.backward()   # gradient flows to current_net, not target_net

# NumPy conversion requires no grad
arr = tensor.detach().cpu().numpy()
```

**What breaks**: calling `.numpy()` on a tensor that has `requires_grad=True` raises an error — NumPy cannot participate in the gradient graph. Always `.detach()` first.

### Disabling gradient tracking entirely

```python
# torch.no_grad() — all operations inside create no graph, saves memory
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=-1)

# Decorator form
@torch.no_grad()
def evaluate(model, loader): ...

# Permanently freeze parameters — backprop stops here
for p in model.encoder.parameters():
    p.requires_grad_(False)
```

**Why `no_grad()` matters for inference**: even if you never call `.backward()`, PyTorch still builds the graph during the forward pass if `requires_grad=True` tensors are involved. That graph sits in memory until the tensors are garbage collected. On long inference runs or large models, this bloats memory substantially. `no_grad()` prevents the graph from being built at all.

---

## 3. Building Models with `nn.Module`

### The problem

A neural network is a collection of parameters organized into operations. You need a way to: define the operations, automatically track all parameters (so you can pass them to an optimizer), switch between training and inference modes (Dropout and BatchNorm behave differently), and save/load state. Doing this manually for each model is tedious and error-prone.

### The core insight

`nn.Module` is a parameter container with a hook system. When you assign another `nn.Module` as an attribute, its parameters are automatically registered. When you call `model(x)`, it runs `model.forward(x)` plus any registered hooks. The entire parameter tree is accessible via `model.parameters()`.

### The mechanics

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Assigning nn.Module subclasses as attributes registers their parameters
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MLP(784, 256, 10)
logits = model(torch.randn(32, 784))   # [32, 10]
```

**Why you never call `forward()` directly**: `model(x)` calls `__call__`, which runs pre-forward hooks, then `forward()`, then post-forward hooks. If you call `model.forward(x)` directly, hooks are skipped — this breaks features like gradient checkpointing and certain debuggers.

### Parameter inspection and freezing

```python
# Inspect all parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Freeze the encoder — backprop will not reach it
for p in model.encoder.parameters():
    p.requires_grad_(False)
# Only the unfrozen head parameters will appear in optimizer updates
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
```

**What breaks if you forget to filter frozen params from the optimizer**: AdamW will maintain optimizer state (momentum, variance) for frozen parameters but never update them — wasted memory, no correctness issue. Filtering them out is the right practice.

### `train()` and `eval()` modes

**The problem**: Dropout and BatchNorm behave differently during training versus inference. Dropout should randomly zero neurons during training (regularization) but be disabled at inference (you want deterministic outputs). BatchNorm should use batch statistics during training but running statistics at inference (test batches may be small or size-1).

**The mechanics**: `model.train()` and `model.eval()` recursively set a flag on every sub-module. Individual layers check this flag in their `forward()`.

```python
model.train()   # enables Dropout, BatchNorm uses batch stats
model.eval()    # disables Dropout, BatchNorm uses running stats
```

**What breaks**: forgetting `model.eval()` during validation makes validation metrics noisy and pessimistic — Dropout is randomly zeroing activations, so each forward pass produces a different result. BatchNorm's batch statistics become unreliable with small validation batches. The result is validation loss that looks artificially high and noisier than training loss.

### `Sequential` and `ModuleList`

```python
# Sequential: good for simple stacks with no branching
encoder = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
)

# ModuleList: good when you need indexed access (e.g., transformer layers
# where you might apply a different function to each layer, or skip certain layers)
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Why `ModuleList` and not a plain Python list**: parameters inside a plain Python list are invisible to `model.parameters()` — they won't be returned by the iterator and won't be passed to the optimizer. `ModuleList` registers them properly.

### Weight initialization

**The problem**: randomly initialized weights with wrong scale cause the forward pass activations to either explode or vanish, especially in deep networks. A network that starts in this state can take much longer to converge or may not converge at all.

**The core insight**: the scale of the initialization should be chosen so that the variance of activations is roughly preserved from layer to layer. Xavier (Glorot) initialization achieves this for linear layers with symmetric activations; Kaiming (He) initialization accounts for the dead-side of ReLU.

```python
def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)   # good for tanh, sigmoid
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)  # applies recursively to all submodules
```

---

## 4. Loss Functions

### The problem

The loss function converts the model's output and the ground-truth label into a single scalar that summarizes how wrong the prediction is. That scalar must be differentiable with respect to the model's parameters — otherwise you cannot compute gradients.

```python
# Classification — expects raw logits (before softmax) and integer labels
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)   # logits [B, C], labels [B] as integers

# Why raw logits and not probabilities: CrossEntropyLoss combines LogSoftmax + NLLLoss
# in a single numerically stable operation. Computing softmax first then taking log
# introduces unnecessary floating-point error.

# Imbalanced classes — upweight rare classes
weights = torch.tensor([1.0, 2.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=weights)

# Binary classification
criterion = nn.BCEWithLogitsLoss()   # numerically stable sigmoid + BCE fused
loss = criterion(logits.squeeze(), labels.float())
# Do not use nn.BCELoss on probabilities (sigmoid output) if you can avoid it —
# it is less numerically stable and requires the sigmoid to already be applied.

# Regression
criterion = nn.MSELoss()
criterion = nn.L1Loss()              # MAE — less sensitive to outliers
criterion = nn.HuberLoss(delta=1.0)  # L2 close to zero, L1 further out — best of both

# Sequence models: ignore padding positions in loss
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Label smoothing: prevents the model from becoming overconfident (assigning ~1.0
# probability to the correct class). Forces it to maintain a small probability mass
# on wrong answers, which improves calibration and often generalization.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 5. Optimizers

### The problem

Gradient descent says: update each parameter by subtracting a small multiple (the learning rate) of its gradient. This works but has problems. With a fixed learning rate, too large overshoots minima, too small takes forever. Different parameters may have very different gradient magnitudes. Noisy gradients from stochastic mini-batches cause erratic updates.

**SGD with momentum** solves the oscillation problem by accumulating a velocity vector — the effective update is a moving average of past gradients. This smooths out noise and accelerates progress in consistent directions.

**Adam** solves the per-parameter scale problem: it maintains a running estimate of each parameter's gradient mean (first moment) and variance (second moment), and scales each update inversely by the approximate gradient scale. Parameters with large gradients get small updates; parameters with small gradients get large updates.

**AdamW** fixes a mathematical error in Adam's weight decay. In Adam, L2 regularization is applied by adding `wd * param` to the gradient before the adaptive scaling — so weight decay gets scaled by the same adaptive factor and effectively applies differently to each parameter. AdamW applies weight decay directly to the weights, independent of the adaptive gradient scaling. This is the theoretically correct formulation and is empirically better, especially for transformers.

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Different learning rates for different parameter groups
# Early layers: small LR (general features, should change slowly)
# Head: large LR (task-specific, needs to adapt quickly)
optimizer = optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(),    "lr": 1e-3},
], weight_decay=0.01)
```

### Learning rate schedulers

**The problem**: a constant learning rate is a poor choice throughout training. Early on, you want a larger LR to explore the loss landscape quickly. Late in training, you want a smaller LR to settle into a minimum without overshooting.

```python
# Cosine annealing: decays from max to 0 following a cosine curve
# Good default for most tasks
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# OneCycleLR: linear warmup to max_lr, then cosine decay
# Very effective for vision models; can train in fewer epochs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=total_steps
)

# Linear warmup: start at start_factor * lr, ramp to full lr over total_iters steps
# Commonly used before cosine decay for transformers
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)

# ReduceLROnPlateau: reduce LR when a metric stops improving
# Good when you don't know when to schedule in advance
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

**When to call `scheduler.step()`**: for epoch-based schedulers (`CosineAnnealingLR`, `StepLR`), call once per epoch after validation. For step-based schedulers (`OneCycleLR`, `LinearLR`), call once per batch inside the training loop. Calling at the wrong cadence produces the wrong LR curve.

---

## 6. The Training Loop

### The problem

Training a neural network is an iterative loop: forward pass, compute loss, backward pass, update weights. The loop has a fixed structure, but the details — device placement, gradient clipping, mixed precision, validation mode switching — must all be correct or training silently fails.

### Complete training + validation loop

```python
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                                      # clear stale gradients
        outputs = model(inputs)                                    # forward pass
        loss = criterion(outputs, targets)                         # compute loss
        loss.backward()                                            # accumulate gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # prevent explosion
        optimizer.step()                                           # update weights
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()   # .item() pulls scalar value out of graph

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total
```

### Gradient clipping

**The problem**: during early training or with certain architectures (RNNs, very deep transformers), gradients can grow to enormous magnitudes — a single bad batch sends all parameters far from their current values. This is called gradient explosion, and once it happens the model is usually broken.

**The core insight**: the *direction* of the gradient is almost always informative even when the *magnitude* is pathological. Clipping the magnitude while preserving the direction prevents explosion without discarding gradient direction information.

**The mechanics**: `clip_grad_norm_` computes the global L2 norm of all parameter gradients concatenated, and if it exceeds `max_norm`, scales all gradients down so the total norm equals `max_norm`.

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Call after loss.backward() and before optimizer.step()
```

**What breaks**: calling before `loss.backward()` clips all-zero gradients (no effect but wasted compute). Not calling it at all when training RNNs or transformer language models often results in NaN weights within the first few hundred steps.

### Gradient accumulation

**The problem**: you want to train with an effective batch size of 512 but your GPU can only fit 64 examples. Reducing batch size degrades gradient signal quality and, for some models, changes training dynamics.

**The core insight**: because PyTorch accumulates gradients by default, you can run multiple small forward-backward passes and then take a single optimizer step. The accumulated gradient is mathematically equivalent to a single pass over the combined batch.

```python
accumulation_steps = 8    # effective batch = batch_size * 8
optimizer.zero_grad()

for step, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    # Divide loss by accumulation_steps so the final accumulated gradient has the
    # same scale as if the full batch had been processed in one pass
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## 7. DataLoader and Datasets

### The problem

GPU training is compute-bound — the GPU can process a batch in milliseconds. But if data loading (disk I/O, decoding, augmentation) takes longer, the GPU starves and sits idle. Efficient data loading is the difference between 90% GPU utilization and 20%.

### The core insight

Data loading and model computation can overlap: while the GPU processes batch N, worker processes on the CPU should already be preparing batch N+1. `DataLoader` manages this overlap automatically through a pool of worker processes and prefetch buffers.

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = load_image(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

**Why the three-method contract**: `DataLoader` needs to know the total size (`__len__`) to construct indices for shuffling, and then retrieve individual samples (`__getitem__`) by those indices. The contract is simple enough to implement in minutes but powerful enough to represent any dataset — in-memory arrays, files on disk, database queries, or streamed data.

### DataLoader parameters

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # spawn 4 worker processes for parallel loading
    pin_memory=True,         # allocate batches in pinned (page-locked) CPU memory
    drop_last=True,          # discard the last incomplete batch
    prefetch_factor=2,       # each worker prefetches 2 batches ahead
    persistent_workers=True, # keep workers alive between epochs
)
```

**Why `pin_memory=True`**: the GPU DMA engine can transfer data from pinned (page-locked) CPU memory directly, without an intermediate copy through pageable memory. The transfer is noticeably faster on large batches.

**Why `drop_last=True` during training**: BatchNorm statistics become unreliable for very small batches. If the last batch is size 1 and BatchNorm is enabled, training will error. `drop_last=True` prevents this.

**Why `persistent_workers=True`**: without it, PyTorch destroys and re-spawns the worker processes at the end of each epoch. On large datasets the worker startup overhead per epoch is measurable. Keeping workers alive eliminates it.

| Setting | Default | Recommended | Why |
| :--- | :--- | :--- | :--- |
| `num_workers` | 0 (main thread) | 4–8 | Parallel preprocessing |
| `pin_memory` | False | True on CUDA | Faster CPU→GPU transfer |
| `persistent_workers` | False | True | Avoids worker restart each epoch |
| `drop_last` | False | True (training) | Consistent batch size for BatchNorm |
| `prefetch_factor` | 2 | 2–4 | Overlap loading with compute |

---

## 8. Mixed Precision Training

### The problem

FP32 (32-bit float) uses more memory than necessary for most neural network computations. A typical large model's activations, gradients, and optimizer states in FP32 can exceed GPU memory. Modern GPUs also have specialized hardware (Tensor Cores) that runs FP16/BF16 operations much faster than FP32.

### The core insight

Not all parts of training require FP32 precision. The forward pass and gradient computation can run in lower precision — activations and gradients don't need to represent tiny differences precisely. The optimizer step, however, makes very small updates to weights and requires FP32 to avoid those updates being rounded to zero.

The distinction between FP16 and BF16: FP16 has 5 exponent bits (max value ~65504), BF16 has 8 exponent bits (same range as FP32). FP16 is prone to overflow during the forward pass of large models; BF16 is not. On A100/H100, use BF16. On older hardware, use FP16 with loss scaling.

### The mechanics

```python
from torch.cuda.amp import GradScaler, autocast

# BF16 on A100/H100 — no scaling needed
def train_step_bf16(model, inputs, targets, optimizer, criterion):
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# FP16 on older hardware — requires loss scaling to prevent underflow
scaler = GradScaler()

def train_step_fp16(model, inputs, targets, optimizer, criterion):
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()       # scale loss to prevent gradient underflow
    scaler.unscale_(optimizer)          # unscale before clipping
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)              # optimizer step (skipped if inf/nan)
    scaler.update()                     # adjust scale for next step
```

**Why loss scaling for FP16**: FP16 gradients can underflow to zero for small values (less than ~6e-5). Multiplying the loss by a large scale factor (e.g., 2^16) before backward shifts gradient magnitudes into the representable FP16 range. Before the optimizer step, `scaler.unscale_()` reverses this scaling. If any gradient is still inf/nan after unscaling, the optimizer step is skipped.

| Tensor | Dtype | Why |
| :--- | :--- | :--- |
| Model weights (forward/backward) | BF16 | Half memory, same exponent range as FP32 |
| Activations | BF16 | Large memory savings |
| Optimizer states (Adam m, v) | FP32 | Small updates need precision |
| Gradient accumulation buffer | FP32 | Avoid precision loss across steps |
| Master weights | FP32 | Source of truth for weight updates |

---

## 9. Checkpointing and Reproducibility

### Saving and loading

```python
# Save everything needed to resume training
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": val_loss,
}
torch.save(checkpoint, "checkpoint.pt")

# Resume
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

**Why save `state_dict()` and not the full model**: `torch.save(model, ...)` serializes the Python class definition alongside the weights. If you rename the class, move the file, or change the module structure, loading breaks. `state_dict()` is just a dictionary of tensors — it loads into any model whose architecture matches, regardless of how the code was organized when it was saved.

**Why `map_location`**: the checkpoint records which device each tensor was on when saved. Loading on a machine with no GPU without `map_location="cpu"` raises an error. Always use `map_location=device` when loading.

### Reproducibility

```python
import random, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Why `cudnn.benchmark = False`**: by default, cuDNN benchmarks several CUDA kernel implementations for each new input shape and picks the fastest. This introduces randomness (the selected kernel may vary) and adds latency on the first batch. Disabling it forces consistent kernel selection — deterministic but slightly slower.

---

## 10. Debugging Patterns

### Shape debugging

Shape mismatches are the most common PyTorch error, especially when adding new model components or changing input resolution.

```python
# Insert a debug layer between modules to trace shapes
class DebugLayer(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"[{self.name}] shape={x.shape}, dtype={x.dtype}, device={x.device}")
        return x

# Or register forward hooks temporarily — removes cleanly
hooks = []
for name, layer in model.named_modules():
    h = layer.register_forward_hook(
        lambda m, inp, out, n=name: print(f"{n}: {out.shape if hasattr(out, 'shape') else type(out)}")
    )
    hooks.append(h)

model(dummy_input)

for h in hooks:
    h.remove()
```

### Gradient flow inspection

```python
def check_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm():.6f}, param_norm={param.norm():.6f}")
        else:
            print(f"{name}: NO GRADIENT")
```

A parameter with no gradient means either: (1) it has `requires_grad=False`, (2) it's not connected to the loss in the computation graph (dead code path), or (3) there's a `.detach()` somewhere in the graph that severs the connection.

### NaN detection

```python
# Anomaly detection: slower, gives exact location of first NaN
with torch.autograd.detect_anomaly():
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

# Register hooks to catch NaN gradients immediately
def nan_hook(grad):
    if torch.isnan(grad).any():
        raise RuntimeError(f"NaN gradient detected: {grad}")

for param in model.parameters():
    param.register_hook(nan_hook)
```

### Common errors

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `RuntimeError: Expected all tensors on device cuda:0` | Input on CPU, model on GPU | `.to(device)` on inputs |
| `RuntimeError: view size is not compatible` | Non-contiguous tensor | Use `.reshape()` or `.contiguous().view()` |
| `nan` loss after first step | LR too high, bad init, overflow | Reduce LR, check input normalization, gradient clipping |
| `CUDA out of memory` | Batch too large, no `no_grad` in eval | Reduce batch, use `@torch.no_grad()` for inference |
| Loss not decreasing | Gradients not flowing, dead ReLUs | Run `check_gradients()`, check `model.train()` |
| `RuntimeError: one of the variables needed for gradient computation has been modified in-place` | In-place op on tensor in graph | Replace `x += y` with `x = x + y` |

---

## 11. Key Patterns Reference

```python
# Standard training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
criterion = nn.CrossEntropyLoss()

# Training step
model.train()
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = criterion(model(x.to(device)), y.to(device))
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()

# Inference
model.eval()
with torch.no_grad():
    preds = model(x.to(device)).argmax(dim=-1)
```
