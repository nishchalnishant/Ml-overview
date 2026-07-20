---
module: Deep Learning Core
topic: Pytorch Fundamentals
subtopic: ""
status: unread
tags: [deeplearning, ml, pytorch-fundamentals]
---
# PyTorch Fundamentals

---

## 1. Tensors

### What a tensor actually is — and why it exists

**The problem**: a neural network is nothing but a sequence of matrix multiplications, nonlinearities, and reductions applied to batches of data. You need one data structure that can represent all of these — scalars, vectors, matrices, batches of images, batches of sequences — live on either CPU or GPU, and integrate with automatic differentiation. NumPy arrays cover the first two points but not the third. Raw GPU arrays cover the second but not the first or third.

**The core insight**: a tensor is a multi-dimensional array with two extra properties: it knows which device it lives on, and it can optionally track the operations performed on it for gradient computation. Everything else follows from those two additions.

```python
import torch

# A scalar — 0-dimensional tensor
s = torch.tensor(7)
s.ndim   # 0
s.shape  # torch.Size([])

# A vector — 1-dimensional
v = torch.tensor([1.0, 2.0, 3.0])
v.ndim   # 1
v.shape  # torch.Size([3])

# A matrix — 2-dimensional
m = torch.tensor([[1, 2], [3, 4]])
m.ndim   # 2
m.shape  # torch.Size([2, 2])

# Common creation patterns
torch.zeros(3, 4)
torch.ones(3, 4)
torch.rand(3, 4)          # uniform [0, 1)
torch.randn(3, 4)         # standard normal
torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.eye(3)              # 3×3 identity matrix
```

### Dtypes

**The problem**: different stages of a model have different precision requirements. Weights and activations can tolerate lower precision (saving memory and running faster on GPU Tensor Cores). Optimizer states and gradient accumulators need higher precision to avoid rounding small updates to zero.

**The core insight**: PyTorch refuses to silently cast between dtypes in operations. If you mix `float32` and `float16`, you get an error. This is intentional — silent casting in numerical code is a source of hard-to-find bugs.

```python
torch.float32   # default for model weights and activations
torch.float16   # half precision, used in mixed-precision training on older GPUs
torch.bfloat16  # brain float — same exponent range as float32, preferred on A100/H100
torch.int64     # default for integer indices and labels
torch.bool

# Casting
x = torch.randn(3)
x.to(torch.float16)
x.half()      # shorthand for float16
x.bfloat16()
x.long()      # shorthand for int64
x.float()     # shorthand for float32
```

### Device: CPU vs GPU

**The problem**: matrix multiplications at the scale required by neural networks are thousands of times slower on CPU than on a modern GPU. But the GPU has its own memory, separate from RAM. Moving data between them is expensive if done carelessly.

**The mechanics**: every tensor has a `device` attribute. All tensors in any operation must be on the same device — mixing raises a `RuntimeError`. Moving data to GPU is a one-time cost per batch; once there, all operations happen on-device.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.rand(3, 4).to(device)        # move existing tensor
x = torch.rand(3, 4, device=device)    # create directly on device

# Must be on CPU before converting to NumPy
arr = x.cpu().numpy()
```

**What breaks**: creating a tensor on GPU then passing it to code that expects CPU (e.g., NumPy), or having the model on GPU and inputs on CPU.

### Shapes and reshaping

**The problem**: operations between tensors require compatible shapes. As data flows through a model — convolution, flatten, linear, reshape for attention — shape errors are the most common source of failures. You need to be able to manipulate shapes precisely.

```python
x = torch.rand(4, 6)

# Reshaping — total element count must be preserved
x.view(24)        # requires contiguous memory; errors if non-contiguous
x.reshape(2, 12)  # handles non-contiguous by copying if needed — prefer this
x.flatten()       # same as reshape(-1)

# Adding / removing size-1 dimensions
x.unsqueeze(0)    # (4, 6) → (1, 4, 6) — adds batch dimension
x.unsqueeze(-1)   # (4, 6) → (4, 6, 1)
x.squeeze()       # removes all size-1 dims
x.squeeze(0)      # removes only dim 0 if it is size 1

# Reordering axes
t = torch.rand(2, 3, 4)
t.permute(0, 2, 1)   # (2, 3, 4) → (2, 4, 3)
t.transpose(1, 2)    # swap two specific dims — same result here

# Broadcasting without copying memory
x = torch.rand(1, 4)
x.expand(3, 4)    # (1, 4) → (3, 4), no copy — only valid if dim is size 1
x.repeat(3, 1)    # (1, 4) → (3, 4), actual copy
```

**Why `.view()` fails on non-contiguous tensors**: after `.permute()` or `.transpose()`, the tensor's logical order doesn't match its memory order. `.view()` requires them to match (contiguous). `.reshape()` transparently calls `.contiguous()` if needed — use `.reshape()` unless you specifically want to catch accidental copies.

### Common operations

```python
a = torch.rand(3, 4)
b = torch.rand(3, 4)

# Element-wise
a + b;  a - b;  a * b;  a / b
a ** 2;  torch.sqrt(a);  torch.exp(a);  torch.log(a)

# Matrix multiplication
a = torch.rand(3, 4)
b = torch.rand(4, 5)
a @ b              # (3, 5)
torch.matmul(a, b) # identical

# Reductions
a.sum()
a.sum(dim=0)                   # sum across rows → shape [4]
a.mean(dim=1, keepdim=True)    # shape [3, 1] — keepdim preserves rank for broadcasting
a.argmax(dim=1)                # index of max per row

# Combining tensors
torch.stack([a, a], dim=0)     # new dimension: (3,4) → (2,3,4)
torch.cat([a, b], dim=0)       # concat along existing dim
```

### NumPy interop

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)   # shares memory — mutating arr changes t and vice versa

arr2 = t.numpy()            # also shares memory
arr3 = t.detach().numpy()   # safe when t has requires_grad=True

# To copy without shared memory
t_copy = torch.tensor(arr)
```

**What breaks**: modifying the NumPy array in-place after `from_numpy` changes the tensor. This is the kind of subtle mutation bug that only appears in production. When in doubt, use `torch.tensor(arr)` to copy.

---

## 2. Autograd

### The computational graph

**The problem**: a neural network with millions of parameters needs gradients of a scalar loss with respect to every parameter. Deriving these by hand is impossible. The computation is also dynamic — the network structure can change per batch (e.g., variable-length sequences).

**The core insight**: the chain rule says that the derivative of a composite function can be computed by multiplying local derivatives along the chain. If you record every operation in a graph, you can apply the chain rule automatically by traversing that graph backward. You write only the forward computation; PyTorch derives the backward pass.

**The mechanics**: when you perform operations on tensors with `requires_grad=True`, PyTorch builds a directed acyclic graph. Each node stores a reference to its `grad_fn` — a function that knows how to compute the gradient of that operation with respect to its inputs. Calling `.backward()` on a scalar traverses this graph from loss to leaves, multiplying local gradients (chain rule), and writes results into `.grad`.

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.0, 0.5], requires_grad=True)

y = (x * w).sum()    # scalar
y.backward()

print(x.grad)   # dy/dx = w = [1.0, 0.5]
print(w.grad)   # dy/dw = x = [2.0, 3.0]
```

Key facts about `requires_grad`:
- `nn.Parameter` sets it `True` automatically — model weights are always tracked
- Input data tensors should be `False` — you don't optimize inputs
- Intermediate tensors (non-leaves) don't retain `.grad` by default — they're computed on demand

### Gradient accumulation across `.backward()` calls

**The problem**: PyTorch *adds* new gradients to whatever `.grad` already holds. Call `.backward()` three times, and `.grad` holds the sum of three backward passes.

```python
x = torch.tensor([1.0], requires_grad=True)
for _ in range(3):
    loss = x ** 2
    loss.backward()
print(x.grad)   # tensor([6.0]) — 2 + 2 + 2, NOT 2.0
```

**Why this is the default**: gradient accumulation for large effective batch sizes and multi-task losses both rely on this behavior. Making accumulation the default means those patterns need no special API — just don't zero before you're ready to step.

**The consequence**: you must call `optimizer.zero_grad()` before each new set of backward passes. Forget it and gradients from previous batches corrupt the current update.

### `.detach()` and `no_grad()`

**The problem**: sometimes you need a tensor's numerical value but don't want gradients to flow through it. Examples: computing a target value in RL that should not update the target network; logging metrics without building graph; converting to NumPy.

**`.detach()`**: returns a tensor that shares storage with the original but has no `grad_fn` — it's outside the graph.

```python
# In actor-critic RL: critic target should not affect the target network's parameters
target_value = target_network(next_states).detach()
loss = F.mse_loss(current_value, target_value)
loss.backward()   # gradient flows only to current_network
```

**`torch.no_grad()`**: disables graph construction for an entire block. Saves both memory (no graph stored) and compute (no grad bookkeeping overhead). Use it in every evaluation and inference loop.

```python
with torch.no_grad():
    output = model(x)

@torch.no_grad()
def predict(model, x):
    return model(x)
```

**`torch.inference_mode()`**: a stricter version of `no_grad()` with slightly less overhead. Use for pure inference where you will never need gradients.

**What breaks**: calling `.numpy()` on a tensor with `requires_grad=True` raises `RuntimeError`. Always `.detach()` first or wrap in `torch.no_grad()`.

### Gradient hooks for inspection

```python
def print_grad(grad):
    print("gradient:", grad)

x = torch.rand(3, requires_grad=True)
y = x ** 2
y.register_hook(print_grad)   # fires when y's gradient is computed
y.sum().backward()

# Check leaf and graph attributes
print(x.is_leaf)      # True
print(x.grad_fn)      # None for leaf nodes
y = x * 2
print(y.grad_fn)      # <MulBackward0>
```

---

## 3. `nn.Module`

### Why `nn.Module` exists

**The problem**: a model is a tree of parameterized operations. You need to: register all parameters (so the optimizer knows what to update), switch between training/inference modes (Dropout and BatchNorm behave differently in each), save and restore state, and compose sub-modules arbitrarily. Doing this manually for every model is error-prone boilerplate.

**The core insight**: `nn.Module` is a parameter registry with a hook system. Assign any `nn.Module` as an attribute and its parameters are automatically registered into the parent's parameter tree. Call `model.parameters()` and you get every leaf parameter, recursively.

### Defining a model

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = SimpleNet(10, 64, 2)
output = model(torch.rand(32, 10))   # calls __call__ → forward + hooks
```

**Why `model(x)` not `model.forward(x)`**: calling `model.forward(x)` directly skips the `__call__` machinery — pre-hooks, post-hooks, gradient checkpointing, etc. Always call the module as a function.

**Why layers assigned as attributes auto-register**: `nn.Module.__setattr__` overrides Python's default attribute assignment. When it sees an `nn.Module` value, it registers it in an internal `_modules` dict. `model.parameters()` then recursively yields parameters from all registered modules.

**What breaks if you use a plain Python list instead of `nn.ModuleList`**: the list is just a regular Python attribute — `nn.Module.__setattr__` doesn't intercept it. Parameters inside are invisible to `model.parameters()`, won't appear in `model.state_dict()`, and won't be passed to the optimizer.

### Parameters and state dict

```python
# Iterate parameters
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

# Count
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# state_dict — ordered dict of name → tensor
# Includes parameters AND registered buffers (e.g., BatchNorm running mean)
state = model.state_dict()
model.load_state_dict(state)
```

**The difference between `parameters()` and `state_dict()`**: `parameters()` returns only learnable weights (tensors with `requires_grad=True`). `state_dict()` returns everything needed to reconstruct the model's state — including non-learnable buffers like BatchNorm's `running_mean` and `running_var`. Use `state_dict()` for saving; use `parameters()` for the optimizer.

### Built-in layers

```python
nn.Linear(128, 64)
nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
nn.Conv1d(128, 64, kernel_size=3)

# Normalizations — each solves a different statistics problem
nn.BatchNorm2d(64)       # normalizes over (batch, H, W) for each channel
nn.LayerNorm(64)         # normalizes over last dim per sample — batch-size agnostic
nn.GroupNorm(8, 64)      # intermediate: groups of channels, no batch dependency

nn.ReLU()
nn.GELU()                # common in transformers — smooth, no dead-neuron problem
nn.Dropout(p=0.5)

nn.LSTM(64, 128, num_layers=2, batch_first=True)
nn.MultiheadAttention(512, num_heads=8, batch_first=True)
nn.Embedding(vocab_size=10000, embedding_dim=256)
```

**Why BatchNorm vs LayerNorm**: BatchNorm normalizes each feature across the batch — it needs a reasonably large batch to estimate stable statistics, and its behavior changes at inference (uses running stats). LayerNorm normalizes across features within each sample independently — behavior is identical at train and inference time, batch size doesn't matter. Transformers use LayerNorm because they often process small or variable batches, and you don't want normalization behavior to change at inference.

### `train()` and `eval()` modes

**The problem**: Dropout randomly zeros activations during training to prevent co-adaptation of neurons. At inference you want deterministic, full-strength predictions. BatchNorm uses batch statistics during training, but at inference the batch may be size 1 — meaningless statistics.

**The mechanics**: `model.train()` and `model.eval()` set a boolean flag on every sub-module recursively. Dropout and BatchNorm check this flag in their `forward()`.

```python
model.train()   # dropout active, BatchNorm uses batch stats
model.eval()    # dropout disabled, BatchNorm uses running stats
```

**What breaks**: forgetting `model.eval()` during validation leaves Dropout active. Each call to `model(x)` randomly zeros different neurons, making validation loss noisy and artificially high. A validation curve that looks oddly noisy (not smooth) is a common symptom of this mistake.

---

## 4. The Training Loop

### Dataset and DataLoader

**The problem**: a dataset can be too large to fit in memory. Preprocessing (augmentation, tokenization) is expensive and should be parallelized. The GPU should never be waiting for data.

**The core insight**: `Dataset` defines how to get one sample; `DataLoader` handles batching, shuffling, and parallel prefetching. The three-method contract (`__len__`, `__getitem__`) is intentionally minimal — it works for in-memory data, on-disk files, databases, and streamed data.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(
    MyDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    num_workers=4,      # 4 worker processes prepare batches in parallel
    pin_memory=True,    # allocate in pinned memory for faster GPU transfer
)
```

**Why `num_workers > 0`**: with `num_workers=0`, data loading runs in the main process, blocking GPU training. Each batch takes: disk I/O + decoding + augmentation + transfer. With `num_workers=4`, four workers prepare the next batches while the GPU is processing the current one. The GPU never sits idle waiting for data.

**Why `pin_memory=True`**: GPU DMA transfers are faster from pinned (page-locked) CPU memory because the transfer can bypass the CPU entirely. With `pin_memory=False`, there's an intermediate copy through pageable memory first.

### The training loop

```python
model = SimpleNet(10, 64, 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()              # 1. clear stale gradients
        logits = model(batch_X)            # 2. forward pass
        loss = loss_fn(logits, batch_y)    # 3. compute loss
        loss.backward()                    # 4. compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. clip
        optimizer.step()                   # 6. update weights
```

Each step exists for a reason:
1. **`zero_grad()`**: PyTorch accumulates gradients. Without this, gradients from every previous batch pile up. See autograd section.
2. **`model(batch_X)`**: runs the forward pass, builds the computation graph.
3. **`loss_fn(...)`**: converts predictions + labels into a scalar. Must be scalar for step 4.
4. **`loss.backward()`**: traverses the graph, computes and accumulates gradients.
5. **`clip_grad_norm_`**: if the total gradient norm exceeds `max_norm`, scale all gradients down proportionally. Prevents a single bad batch from making catastrophically large weight updates.
6. **`optimizer.step()`**: reads `.grad` from each parameter and updates weights according to the optimizer's rule.

### The validation loop

```python
model.eval()
val_loss, correct = 0.0, 0

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        val_loss += loss_fn(logits, batch_y).item()
        correct += (logits.argmax(dim=1) == batch_y).sum().item()

accuracy = correct / len(val_dataset)
```

Three things must happen for validation to be correct: `model.eval()` (disable Dropout, use running stats for BatchNorm), `torch.no_grad()` (no graph built, no memory for activations), `.item()` (detach scalar from graph before accumulating).

### Optimizers

**SGD**: follows the negative gradient directly, optionally with momentum. Momentum maintains a velocity vector — a running average of past gradients. Reduces oscillation in narrow loss valleys, accelerates progress in consistent gradient directions.

**Adam**: maintains per-parameter running estimates of gradient mean (first moment, m) and squared gradient (second moment, v). Updates are scaled as `m / sqrt(v)`. Parameters with large past gradients get small updates; parameters with small gradients get large updates. Adapts to the curvature of each parameter's loss surface individually.

**AdamW**: fixes Adam's weight decay. In Adam, weight decay is implemented by adding `wd * param` to the gradient, which means it gets scaled by `1 / sqrt(v)` — large parameters get proportionally less decay than small ones. AdamW applies weight decay directly to the parameter after the gradient update, bypassing the adaptive scaling. Theoretically correct and empirically better for transformers.

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
torch.optim.Adam(model.parameters(), lr=1e-3)
torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## 5. Learning Rate Schedulers

**The problem**: a fixed learning rate is rarely optimal. Too large early on overshoots; too small throughout means slow convergence. The ideal schedule is high initially (explore broadly), then lower (settle into a minimum).

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cosine annealing: smooth decay from max_lr to 0 following a cosine curve
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# OneCycleLR: linear warmup to max_lr, then cosine decay
# Very effective for vision models, enables aggressive max_lr
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=30
)

# ReduceLROnPlateau: reduce by factor when metric stops improving
# Useful when you can't predict in advance when to reduce
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

**When to step**:
- Most schedulers (`CosineAnnealingLR`, `StepLR`, `ReduceLROnPlateau`): call `scheduler.step()` once per epoch, after validation.
- Batch-level schedulers (`OneCycleLR`): call `scheduler.step()` once per batch, inside the batch loop.
- Calling at the wrong cadence produces a wrong learning rate curve with no error message.

```python
print(optimizer.param_groups[0]['lr'])   # inspect current lr
```

---

## 6. Common Architectures in PyTorch

### MLP

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### CNN

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # halves spatial dimensions

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))               # fixed output size regardless of input
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                         # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded)            # output: (batch, seq_len, hidden*2)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(last_hidden)
```

### Transformer encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=0.1,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, padding_mask=None):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.head(x[:, 0, :])   # use [CLS] token
```

---

## 7. Custom Loss Functions

**The problem**: the standard loss functions cover the common cases, but many applications need specialized objectives — focal loss for imbalanced detection, contrastive loss for embedding learning, custom regularization terms.

**The core insight**: a loss function is just a differentiable function from predictions and targets to a scalar. If you use only PyTorch tensor operations, autograd handles the gradient automatically.

```python
# Function style — sufficient for stateless losses
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

# Class style — needed when the loss has learnable parameters or state
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label = 0 if similar, 1 if dissimilar
        dist = F.pairwise_distance(output1, output2)
        loss = (1 - label) * dist ** 2 + label * torch.clamp(self.margin - dist, min=0) ** 2
        return loss.mean()
```

---

## 8. Model Saving and Loading

**The problem**: training large models takes hours or days. You need to save state to resume after interruption, revert to best checkpoint, or deploy separately from training code.

**Why save `state_dict()` and not the full model**: `torch.save(model, path)` serializes the Python class alongside the weights. If you rename a class or refactor the module structure, loading fails. `state_dict()` is a plain dictionary of tensor names to tensors — it loads into any model whose architecture matches, regardless of code organization.

```python
# Save only weights — for inference or fine-tuning from a fixed checkpoint
torch.save(model.state_dict(), 'weights.pth')

# Save full training checkpoint — to resume training
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': val_loss,
}, 'checkpoint.pth')
```

```python
# Load weights
model.load_state_dict(torch.load('weights.pth', map_location=device))
model.eval()

# Resume training
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Partial loading for fine-tuning — load only matching keys
pretrained = torch.load('pretrained.pth', map_location=device)
current = model.state_dict()
matching = {k: v for k, v in pretrained.items() if k in current and v.shape == current[k].shape}
current.update(matching)
model.load_state_dict(current)
```

**Why `map_location`**: checkpoints record the device of each tensor. Without `map_location`, loading a GPU checkpoint on a CPU machine raises `RuntimeError`. Always specify `map_location=device` when loading.

---

## 9. Debugging

### Shape issues

Shape errors are the most common failure mode. Build the habit of printing shapes at every stage of a new model.

```python
# Option 1: print inside forward
class DebugNet(nn.Module):
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.conv(x)
        print(f"After conv: {x.shape}")
        x = x.flatten(1)
        print(f"After flatten: {x.shape}")
        return self.fc(x)

# Option 2: forward hooks — cleaner, easy to remove
hooks = []
for name, layer in model.named_modules():
    h = layer.register_forward_hook(
        lambda m, inp, out, n=name: print(f"{n}: {out.shape if hasattr(out, 'shape') else '?'}")
    )
    hooks.append(h)

model(dummy_input)
for h in hooks:
    h.remove()
```

### Gradient inspection

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

A parameter with `NO GRADIENT` means: (1) `requires_grad=False`, (2) disconnected from loss in the graph, or (3) `.detach()` somewhere severs the chain.

### NaN and Inf

NaN gradients are silent killers — the model appears to be training, loss is a number, but weights are all NaN.

```python
# Anomaly detection: gives exact location but is slow
with torch.autograd.detect_anomaly():
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

# Hooks to catch NaN gradients immediately
def nan_hook(grad):
    if torch.isnan(grad).any():
        raise RuntimeError("NaN gradient detected")

for param in model.parameters():
    param.register_hook(nan_hook)
```

Common causes of NaN loss:
- `log(0)` — add epsilon: `torch.log(x + 1e-8)`
- Division by zero — guard denominators
- Exploding gradients — use gradient clipping
- Bad initialization — use Xavier/Kaiming

### Mixed precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_X, batch_y in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch_X)
        loss = loss_fn(output, batch_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Why `GradScaler`**: FP16 gradients can underflow to zero for small values. The scaler multiplies the loss by a large factor before backward (shifting gradients into FP16's representable range), then unscales before the optimizer step. If any gradient is `inf` after unscaling, the optimizer step is skipped for that iteration.

---

## 10. Common Interview Questions

**Q: What's the difference between `.view()` and `.reshape()`?**

`.view()` requires contiguous memory and returns a view (no copy). `.reshape()` handles non-contiguous tensors by making a copy if needed. After `.permute()` or `.transpose()`, the tensor may not be contiguous — `.view()` will error, `.reshape()` will silently copy. Use `.reshape()` unless you explicitly need a view guarantee.

**Q: Why call `optimizer.zero_grad()` before `loss.backward()`?**

PyTorch accumulates gradients by default — each `.backward()` call adds to `.grad`. Without zeroing, gradients from previous batches add to current ones. The rare legitimate exception is gradient accumulation (multiple backward passes per optimizer step to simulate a larger batch).

**Q: What's the difference between `model.parameters()` and `model.state_dict()`?**

`parameters()` yields only learnable weight tensors (those with `requires_grad=True`). `state_dict()` is an ordered dict mapping names to tensors — it includes parameters and non-learnable buffers (BatchNorm running stats). Use `state_dict()` for saving; `parameters()` for the optimizer.

**Q: When would you use `requires_grad=False` on layers?**

When fine-tuning a pretrained model: freeze early layers (general features shouldn't change), train only the head or upper layers. Also saves compute — no graph is built through frozen parameters.

**Q: What's the difference between BatchNorm and LayerNorm?**

BatchNorm normalizes across the batch for each feature — needs a large batch for stable statistics, uses running stats at inference. LayerNorm normalizes across features within each sample — batch-size agnostic, identical behavior at train and test time. Transformers use LayerNorm because they often process small or variable batches.

**Q: What does `.detach()` do?**

Returns a new tensor sharing data with the original but with no `grad_fn`. Gradients don't flow through it. Use it when you need a tensor's value but don't want that path included in backpropagation (target networks in RL, metrics logging, NumPy conversion).

**Q: What's the difference between `CrossEntropyLoss` and `NLLLoss`?**

`CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` in one numerically stable operation. Pass raw logits. `NLLLoss` expects log-probabilities (output of `F.log_softmax`). Use `CrossEntropyLoss` for classification — applying softmax externally then passing to `NLLLoss` introduces unnecessary floating-point error.

**Q: How does gradient clipping work?**

Computes the global L2 norm of all parameter gradients concatenated. If this norm exceeds `max_norm`, scales all gradients down so the total norm equals `max_norm`. Preserves gradient direction, clips magnitude. Call after `loss.backward()`, before `optimizer.step()`.

**Q: What's the advantage of AdamW over Adam?**

In Adam, weight decay is applied by adding `wd * param` to the gradient before the adaptive scaling — so the effective decay is scaled by `1/sqrt(v)`, differently for each parameter. AdamW decouples weight decay from the gradient update, applying it directly to weights. Theoretically correct; empirically better especially for transformers.

**Q: What happens if you forget `model.eval()` during validation?**

Dropout stays active — each forward pass randomly zeros different activations, producing different outputs for the same input. BatchNorm uses batch statistics instead of running statistics — unreliable with small validation batches. Validation loss is artificially high and noisy. The model may appear to be overfitting when it isn't.
