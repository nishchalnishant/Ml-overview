# PyTorch Foundations

PyTorch is an eager-execution framework: operations run immediately, and a dynamic computation graph is built on-the-fly during the forward pass. This makes debugging straightforward but requires understanding a few core mechanics — tensors, autograd, and the training loop — before everything else makes sense.

---

## 1. Tensors

### Creation and Shape Manipulation

```python
import torch

# Creation
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # from data
x = torch.zeros(3, 4)                                  # zeros
x = torch.ones(3, 4)
x = torch.randn(2, 3, 4)                               # normal dist
x = torch.arange(0, 10, 2)                             # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, steps=5)                      # [0, 0.25, 0.5, 0.75, 1.0]

# Shape
x = torch.randn(2, 3, 4)
print(x.shape)          # torch.Size([2, 3, 4])
print(x.ndim)           # 3
print(x.numel())        # 24

# Reshape operations
x = torch.randn(6, 4)
x.view(24)              # flatten — requires contiguous memory
x.reshape(24)           # same but handles non-contiguous
x.unsqueeze(0)          # [1, 6, 4] — insert dim
x.squeeze()             # remove all size-1 dims
x.permute(1, 0)         # [4, 6] — reorder axes
x.transpose(0, 1)       # swap two axes

# Typical image tensor: [batch, channels, height, width]
images = torch.randn(32, 3, 224, 224)
# Flatten spatial dims for an MLP:
flat = images.view(images.shape[0], -1)  # [32, 150528]
```

### Indexing and Slicing

```python
x = torch.randn(4, 8)

x[0]           # first row — shape [8]
x[:, 2]        # third column — shape [4]
x[1:3, 4:6]   # slice — shape [2, 2]

# Boolean indexing
mask = x > 0
x[mask]        # 1D tensor of elements where condition is True

# Gather along dimension
idx = torch.tensor([[0, 1], [2, 3]])
torch.gather(x, dim=1, index=idx)   # selects specific column per row
```

### Dtype and Device

```python
x = torch.randn(3, 3)
x.dtype        # torch.float32 (default)

x = x.to(torch.float16)   # cast
x = x.half()              # same
x = x.bfloat16()          # BF16 — preferred for training on A100/H100
x = x.long()              # int64, common for indices/labels

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
x = x.cuda()              # same
x = x.cpu()               # move back to CPU for numpy conversion

# Never mix devices — this raises RuntimeError:
# a = torch.randn(3).cuda()
# b = torch.randn(3).cpu()
# a + b  # ERROR
```

### Math Operations

```python
a, b = torch.randn(3, 4), torch.randn(3, 4)

# Elementwise
a + b
a * b
torch.exp(a)
torch.log(a.abs())

# Reductions
a.sum()                  # scalar
a.sum(dim=0)             # sum along rows → shape [4]
a.sum(dim=1, keepdim=True)  # shape [3, 1] — preserves dimension

a.mean(), a.std(), a.max(), a.min()
a.argmax(dim=1)          # index of max per row

# Matrix operations
x = torch.randn(3, 4)
w = torch.randn(4, 5)
x @ w                    # matmul → [3, 5]
torch.matmul(x, w)       # same

# Batched matmul
A = torch.randn(8, 3, 4)   # batch of 8 matrices
B = torch.randn(8, 4, 5)
A @ B                       # [8, 3, 5] — broadcasts over batch dim
```

---

## 2. Autograd

### How the Computation Graph Works

PyTorch builds a dynamic DAG of operations during the forward pass. Each node knows how to compute the gradient of its output with respect to its inputs. `backward()` traverses this graph in reverse, accumulating gradients into `.grad`.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1

y.backward()              # dy/dx = 2x + 3 = 7
print(x.grad)             # tensor(7.)

# Multi-variable
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
loss = (a * b).sum()      # sum(a_i * b_i)

loss.backward()
print(a.grad)             # tensor([4., 5., 6.]) — gradient w.r.t. a = b
print(b.grad)             # tensor([1., 2., 3.]) — gradient w.r.t. b = a
```

### Gradient Accumulation and Zeroing

Gradients accumulate — they are added to `.grad` on each `.backward()` call. This is intentional for gradient accumulation (simulating larger batches), but requires explicit zeroing between steps.

```python
for step, (x, y) in enumerate(dataloader):
    loss = model(x)
    loss.backward()               # accumulates into .grad

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()     # clear accumulated gradients
```

### Disabling Gradient Tracking

```python
# Inference — no graph needed, saves memory and compute
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=-1)

# Detach a tensor from the graph (for metrics, reward model, etc.)
reward = reward_model(x).detach()   # returns tensor with no grad_fn

# Permanently mark a tensor as not requiring grad
frozen_params = model.encoder.parameters()
for p in frozen_params:
    p.requires_grad_(False)
```

### Inspecting the Graph

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2
z = y.sum()

print(z.grad_fn)           # <SumBackward0>
print(z.grad_fn.next_functions)  # [(PowBackward0, 0)]
```

---

## 3. Building Models with nn.Module

### Minimal Custom Module

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MLP(784, 256, 10)
x = torch.randn(32, 784)    # batch of 32
logits = model(x)            # [32, 10]
```

### Parameter Inspection

```python
# All learnable parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,} | Trainable: {trainable:,}")

# Access specific layers
model.fc1.weight.shape    # [256, 784]
model.fc1.bias.shape      # [256]
```

### Sequential and ModuleList

```python
# Sequential for simple stacks
encoder = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
)

# ModuleList when you need indexed access (e.g., transformer layers)
class TransformerModel(nn.Module):
    def __init__(self, num_layers: int, d_model: int):
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

### Weight Initialization

```python
def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)
```

---

## 4. Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()          # expects raw logits + integer labels
loss = criterion(logits, labels)           # logits [B, C], labels [B]

# With class weights (for imbalanced data)
weights = torch.tensor([1.0, 2.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=weights)

# Binary classification
criterion = nn.BCEWithLogitsLoss()         # numerically stable sigmoid + BCE
loss = criterion(logits.squeeze(), labels.float())

# Regression
criterion = nn.MSELoss()
criterion = nn.L1Loss()                    # MAE — more robust to outliers
criterion = nn.HuberLoss(delta=1.0)        # L1 far, L2 near — best of both

# Sequence models: ignore padding
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Label smoothing (prevents overconfident predictions)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 5. Optimizers

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

# AdamW — Adam with decoupled weight decay (standard for LLMs)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Different LRs for different parameter groups
optimizer = optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3},
], weight_decay=0.01)

# Learning rate schedulers
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=total_steps
)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
```

---

## 6. The Training Loop

### Complete Training + Validation Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

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
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# Main training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(784, 256, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
```

### Gradient Accumulation

Simulates larger batch sizes when GPU memory is limited:

```python
accumulation_steps = 8    # effective batch = batch_size * 8
optimizer.zero_grad()

for step, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps  # scale loss
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## 7. DataLoader and Datasets

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
        image = load_image(self.image_paths[idx])   # returns PIL or numpy
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,         # parallel data loading
    pin_memory=True,       # faster CPU→GPU transfer
    drop_last=True,        # avoid partial batches messing up BatchNorm
    prefetch_factor=2,     # prefetch 2 batches per worker
)
```

### DataLoader Performance

| Setting | Default | Recommended | Why |
| :--- | :--- | :--- | :--- |
| `num_workers` | 0 (main thread) | 4–8 | Parallel preprocessing |
| `pin_memory` | False | True on CUDA | Faster host→device transfer |
| `prefetch_factor` | 2 | 2–4 | Overlap data loading with compute |
| `persistent_workers` | False | True | Avoids worker restart per epoch |
| `drop_last` | False | True (training) | Consistent batch size |

---

## 8. Mixed Precision Training

BF16/FP16 reduces memory usage ~50% and speeds up compute on modern GPUs. Use `torch.cuda.amp.autocast` for automatic casting.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()    # only needed for FP16, not BF16

def train_step_amp(model, inputs, targets, optimizer, criterion):
    optimizer.zero_grad()

    # Autocast: runs forward pass in BF16/FP16
    with autocast(dtype=torch.bfloat16):    # BF16 on A100/H100
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # BF16: no scaling needed (same exponent range as FP32)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # FP16 version (older hardware):
    # scaler.scale(loss).backward()
    # scaler.unscale_(optimizer)
    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # scaler.step(optimizer)
    # scaler.update()
```

### Memory Layout of Mixed Precision Training

| Tensor type | Dtype | Why |
| :--- | :--- | :--- |
| Model weights (forward/backward) | BF16 | Half memory, same exp range as FP32 |
| Activations | BF16 | Large memory savings |
| Optimizer states (Adam m, v) | FP32 | Precision needed for small updates |
| Gradient accumulation buffer | FP32 | Avoid precision loss across steps |
| Master weights | FP32 | Source of truth for weight updates |

---

## 9. Checkpointing and Reproducibility

### Saving and Loading Models

```python
# Save everything needed to resume training
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": val_loss,
}
torch.save(checkpoint, "checkpoint.pt")

# Load and resume
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1

# Inference-only: save just weights
torch.save(model.state_dict(), "weights.pt")
model.load_state_dict(torch.load("weights.pt", map_location=device))

# HuggingFace-style: save full model
model.save_pretrained("./saved_model")
model = ModelClass.from_pretrained("./saved_model")
```

### Reproducibility

```python
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

`cudnn.benchmark = True` is the default and improves speed by auto-tuning CUDA kernels. Disable it only when exact reproducibility matters.

---

## 10. Debugging Patterns

### Shape Debugging

```python
class DebugLayer(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"[{self.name}] shape={x.shape}, dtype={x.dtype}, device={x.device}")
        return x

# Insert between layers to trace shapes through a model
model = nn.Sequential(
    nn.Linear(784, 256),
    DebugLayer("after fc1"),
    nn.ReLU(),
    DebugLayer("after relu"),
    nn.Linear(256, 10),
)
```

### Gradient Flow Inspection

```python
def check_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param.norm().item():.6f}")
        else:
            print(f"{name}: NO GRADIENT")

# Call after loss.backward() and before optimizer.step()
check_gradients(model)
```

### Common Errors and Fixes

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `RuntimeError: Expected all tensors on device cuda:0` | Tensor on CPU, model on GPU | `.to(device)` on inputs |
| `RuntimeError: view size is not compatible` | Non-contiguous tensor | Use `.reshape()` or `.contiguous().view()` |
| `nan` loss after first step | LR too high, bad init, overflow | Reduce LR, check input normalization, use gradient clipping |
| `CUDA out of memory` | Batch too large, KV cache, no `no_grad` | Reduce batch, use `torch.no_grad()` for inference |
| Loss not decreasing | Gradients not flowing (dead ReLU, vanishing) | Check `check_gradients()`, try different activation |
| `RuntimeError: one of the variables needed for gradient computation has been modified in-place` | In-place op on tensor that needs grad | Replace `x += y` with `x = x + y` |

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
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()

# Inference
model.eval()
with torch.no_grad():
    preds = model(x.to(device)).argmax(dim=-1)
```

> [!TIP]
> **Interview structure:** PyTorch = tensors (shaped data) + autograd (dynamic graph, backward() traversal) + nn.Module (parameter containers) + training loop (forward → loss → zero_grad → backward → clip → step). The subtleties that distinguish strong engineers: `zero_grad` placement for gradient accumulation, `no_grad` for inference, `pin_memory` + `num_workers` for throughput, BF16 autocast for memory, and gradient clipping to prevent instability.
