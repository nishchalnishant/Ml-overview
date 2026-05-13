# PyTorch Fundamentals

PyTorch is the framework that turned deep learning from academic math into something you can actually debug at 2am.

If you already think in terms of arrays, pipelines, and execution graphs, PyTorch will feel natural quickly. If you come from TensorFlow's early static-graph days, PyTorch's dynamic execution model will feel like a breath of fresh air.

This guide is a dense, practical reference. It's organized the way interviews usually probe the topic: tensors first, then autograd, then model definition, then the training loop, then architecture patterns, then the production and debugging stuff most people skip.

---

# 1. Tensors

## What a Tensor Actually Is

A tensor is a multi-dimensional array. That's it. A scalar is a 0-d tensor, a vector is 1-d, a matrix is 2-d, and from there you stack dimensions as needed.

PyTorch tensors are like NumPy arrays with two critical upgrades:
- they can live on a GPU
- they integrate with autograd (automatic differentiation)

Everything in a neural network is tensors being shuffled, multiplied, and differentiated.

## Creation

```python
import torch

# Scalar (0-d)
s = torch.tensor(7)
print(s.ndim, s.shape)  # 0, torch.Size([])

# Vector (1-d)
v = torch.tensor([1.0, 2.0, 3.0])
print(v.ndim, v.shape)  # 1, torch.Size([3])

# Matrix (2-d)
m = torch.tensor([[1, 2], [3, 4]])
print(m.ndim, m.shape)  # 2, torch.Size([2, 2])

# 3-d tensor (e.g. a batch of sequences)
t = torch.zeros(4, 10, 64)  # batch=4, seq_len=10, hidden=64
print(t.shape)  # torch.Size([4, 10, 64])

# Common factory functions
torch.zeros(3, 4)           # all zeros
torch.ones(3, 4)            # all ones
torch.rand(3, 4)            # uniform [0, 1)
torch.randn(3, 4)           # standard normal
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.eye(3)                # 3x3 identity matrix
torch.full((2, 3), fill_value=7)  # all 7s
```

## Dtypes

The dtype controls memory footprint and numerical precision. Getting this wrong is a frequent source of bugs.

```python
# Common dtypes
torch.float32   # default for most model weights and activations
torch.float16   # half precision, used in mixed-precision training
torch.bfloat16  # brain float, popular for transformer training (better range)
torch.float64   # double precision, rarely needed in deep learning
torch.int32
torch.int64     # default for indices
torch.bool

# Creating with explicit dtype
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
y = torch.tensor([1, 2], dtype=torch.int64)

# Casting
x_half = x.to(torch.float16)
x_back = x_half.float()       # shortcut for .to(torch.float32)
x_int = x.long()              # shortcut for .to(torch.int64)
```

Key rule: when you mix dtypes in an operation, PyTorch will usually raise an error rather than silently cast. Explicit is better.

## Device: CPU vs GPU

This is where PyTorch really earns its keep.

```python
# Check what's available
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())  # Apple Silicon

# Standard device setup pattern
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move tensors to device
x = torch.rand(3, 4)
x = x.to(device)

# Create directly on device
x = torch.rand(3, 4, device=device)

# Move back to CPU (required before converting to NumPy)
x_cpu = x.cpu()
x_numpy = x_cpu.numpy()
```

The key rule: all tensors in an operation must be on the same device. Mixing CPU and GPU tensors raises an error.

## Shapes and Reshaping

Shape mismatches are the most common PyTorch error. Get comfortable with these.

```python
x = torch.rand(4, 6)

# Reshaping
x.view(24)          # flatten to 1-d (requires contiguous memory)
x.view(2, 12)
x.reshape(2, 12)    # like view but handles non-contiguous tensors
x.flatten()         # same as view(-1)
x.flatten(start_dim=0)

# Adding/removing dimensions
x.unsqueeze(0)      # (4, 6) -> (1, 4, 6)
x.unsqueeze(-1)     # (4, 6) -> (4, 6, 1)
x.squeeze()         # removes all size-1 dims
x.squeeze(0)        # removes dim 0 only if it's size 1

# Permuting dimensions (not transposing — permuting)
t = torch.rand(2, 3, 4)
t.permute(0, 2, 1)  # (2, 3, 4) -> (2, 4, 3)
t.transpose(1, 2)   # swap two specific dims

# Expanding without copying
x = torch.rand(1, 4)
x.expand(3, 4)      # (1, 4) -> (3, 4), no memory copy
x.repeat(3, 1)      # (1, 4) -> (3, 4), actual copy
```

## Common Operations

```python
a = torch.rand(3, 4)
b = torch.rand(3, 4)

# Element-wise
a + b
a - b
a * b          # NOT matrix multiplication
a / b
a ** 2
torch.sqrt(a)
torch.exp(a)
torch.log(a)

# Matrix multiplication
a = torch.rand(3, 4)
b = torch.rand(4, 5)
c = a @ b              # (3, 5)
c = torch.matmul(a, b) # same thing

# Reduction
a.sum()
a.sum(dim=0)           # sum along dim 0
a.mean(dim=1, keepdim=True)
a.max()
a.min()
a.argmax(dim=1)        # index of max along dim 1
a.std()
a.var()

# Stacking and concatenating
torch.stack([a, a], dim=0)   # new dim: (3,4) -> (2,3,4)
torch.cat([a, b], dim=0)     # existing dim: concat along dim 0

# Comparison
a > 0.5                # boolean tensor
(a > 0.5).float()      # convert bool to float for masking
```

## Indexing

```python
x = torch.rand(4, 5)

x[0]           # first row
x[-1]          # last row
x[1:3]         # rows 1 and 2
x[:, 2]        # column 2
x[0, 2]        # element at (0, 2)
x[0:2, 1:3]    # 2x2 submatrix

# Boolean indexing
mask = x > 0.5
x[mask]        # 1-d tensor of values where mask is True

# Advanced indexing
indices = torch.tensor([0, 2, 3])
x[indices]     # rows 0, 2, 3
```

## Mixing with NumPy

```python
import numpy as np

# NumPy -> PyTorch
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)   # shares memory — changes to one affect the other

# PyTorch -> NumPy (must be on CPU)
arr2 = t.numpy()            # still shares memory
arr3 = t.detach().numpy()   # safe version if tensor has grad

# To copy without shared memory
t_copy = torch.tensor(arr)  # new memory, does not share
```

The shared-memory behavior of `from_numpy` catches people in production. If you modify the NumPy array in-place, the tensor changes too.

---

# 2. Autograd

## The Computational Graph

When you do math on tensors with `requires_grad=True`, PyTorch builds a computational graph in the background. Every operation creates a node. The graph records how each output depends on each input.

When you call `.backward()`, PyTorch walks this graph in reverse and computes gradients using the chain rule.

```
forward:  input -> op1 -> op2 -> loss
backward: loss -> d(loss)/d(op2) -> d(loss)/d(op1) -> d(loss)/d(input)
```

That's automatic differentiation. You write the forward pass, PyTorch figures out the backward pass.

## requires_grad

```python
# Leaf tensors with requires_grad=True track gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.0, 0.5], requires_grad=True)

# Simple computation
y = (x * w).sum()

# Compute gradients
y.backward()

print(x.grad)   # dy/dx = w = [1.0, 0.5]
print(w.grad)   # dy/dw = x = [2.0, 3.0]
```

Key facts:
- Model parameters automatically have `requires_grad=True`
- Input data tensors usually don't — you don't optimize your inputs
- Non-leaf tensors (intermediate results) don't retain gradients by default

## .backward() in Detail

```python
# Scalar output: backward with no args
loss = torch.tensor(5.0, requires_grad=True)
loss.backward()

# Non-scalar output: need to pass gradient
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
z = y ** 2
z.backward(torch.ones_like(z))   # gradient w.r.t. each element
print(y.grad)   # [2.0, 4.0, 6.0]

# Accumulation: gradients add up across calls
x = torch.tensor([1.0], requires_grad=True)
for _ in range(3):
    loss = x ** 2
    loss.backward()
print(x.grad)  # 6.0 (= 2 + 2 + 2), NOT 2.0

# This is why you need optimizer.zero_grad() in the training loop
```

## .detach() and no_grad

```python
# .detach() — create a tensor that doesn't participate in the graph
x = torch.rand(3, requires_grad=True)
y = x * 2
y_detached = y.detach()   # same values, no grad tracking

# torch.no_grad() — disable gradient computation entirely
# Use for inference and validation — saves memory and computation
with torch.no_grad():
    output = model(x)

# Equivalent decorator form
@torch.no_grad()
def predict(model, x):
    return model(x)

# torch.inference_mode() — even faster than no_grad for pure inference
with torch.inference_mode():
    output = model(x)
```

Use `no_grad()` in every evaluation loop. It's not optional if you care about memory.

## Gradient Inspection

```python
# Registering hooks to inspect gradients mid-graph
def print_grad(grad):
    print("Gradient:", grad)

x = torch.rand(3, requires_grad=True)
y = x ** 2
y.register_hook(print_grad)   # fires when y.grad is computed
y.sum().backward()

# Checking if gradient exists
print(x.grad)            # tensor after backward
print(x.grad_fn)         # None for leaf nodes
print(x.is_leaf)         # True for x
```

---

# 3. nn.Module

## Defining a Model

`nn.Module` is the base class for everything in PyTorch. Every layer, every model, every block you build inherits from it.

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        # Define layers as attributes
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(in_features=10, hidden=64, out_features=2)
```

When you define layers as attributes of an `nn.Module`, PyTorch automatically registers their parameters. That's the magic — you don't have to manually track weights.

## forward()

`forward()` defines the computation. You never call it directly — you call the model as a function, which triggers `__call__`, which runs `forward()` plus hooks.

```python
x = torch.rand(32, 10)    # batch of 32 inputs
output = model(x)          # calls model.__call__(x) -> model.forward(x)
print(output.shape)        # (32, 2)
```

## Parameters and State Dict

```python
# Iterate over all parameters
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")

# State dict — dictionary mapping layer names to tensors
state = model.state_dict()
print(state.keys())  # odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

# Load state dict
model.load_state_dict(state)
```

## Built-in Layers Reference

```python
# Linear
nn.Linear(in_features=128, out_features=64, bias=True)

# Convolutions
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)

# Normalizations
nn.BatchNorm2d(num_features=64)   # normalizes over batch, H, W
nn.LayerNorm(normalized_shape=64) # normalizes over last dim (used in transformers)
nn.GroupNorm(num_groups=8, num_channels=64)

# Activations
nn.ReLU()
nn.GELU()          # common in transformers
nn.Sigmoid()
nn.Tanh()
nn.LeakyReLU(negative_slope=0.01)

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))  # pools to fixed output size

# Dropout
nn.Dropout(p=0.5)       # kills random neurons during training
nn.Dropout2d(p=0.5)     # kills entire channels

# Recurrent
nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
nn.GRU(input_size=64, hidden_size=128, batch_first=True)

# Attention
nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

# Embedding
nn.Embedding(num_embeddings=10000, embedding_dim=256)

# Sequential — for stacking without subclassing
block = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32)
)
```

## train() and eval() Modes

```python
model.train()   # enables dropout, uses batch statistics in batchnorm
model.eval()    # disables dropout, uses running stats in batchnorm

# Always switch modes explicitly
# train() during training, eval() during validation/inference
```

Forgetting `model.eval()` during validation is a classic bug that makes validation loss look oddly noisy.

---

# 4. The Training Loop

## Dataset and DataLoader

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

dataset = MyDataset(X_train, y_train)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,       # parallel data loading
    pin_memory=True,     # faster CPU -> GPU transfer
    drop_last=False      # whether to drop last incomplete batch
)
```

## The Training Loop

This is the rhythm you should be able to write from memory.

```python
import torch
import torch.nn as nn

# Setup
model = SimpleNet(10, 64, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 1. Forward pass
        logits = model(batch_X)

        # 2. Compute loss
        loss = loss_fn(logits, batch_y)

        # 3. Zero gradients (must come before backward)
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. (Optional) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

Why `zero_grad()` before `backward()`? Because PyTorch accumulates gradients by default. If you don't zero them, gradients from the previous batch add onto the current one. That's rarely what you want.

Some people call `zero_grad(set_to_none=True)` which is slightly faster — it sets gradients to `None` instead of zero, saving a memory write.

## Validation Loop

```python
model.eval()
val_loss = 0.0
correct = 0

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        val_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()

val_loss /= len(val_loader)
accuracy = correct / len(val_dataset)
print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
```

## Common Optimizers

```python
# SGD with momentum — classic, works well with learning rate schedules
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — adaptive, great default choice
torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

# AdamW — Adam with decoupled weight decay, preferred for transformers
torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# RMSprop — useful for RNNs
torch.optim.RMSprop(model.parameters(), lr=1e-3)
```

---

# 5. Learning Rate Schedulers

The learning rate is often the most important hyperparameter. Schedulers adjust it during training to help convergence.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# StepLR — decay by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# CosineAnnealingLR — smooth cosine decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau — reduce when metric stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# OneCycleLR — warm up then decay, very effective for vision models
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=30
)

# Warmup + cosine (common in transformers, often done manually or via transformers lib)
```

How to use it in the training loop:

```python
for epoch in range(num_epochs):
    # ... train loop ...
    
    # Step after each epoch (most schedulers)
    scheduler.step()
    
    # For ReduceLROnPlateau, pass the monitored metric
    scheduler.step(val_loss)
    
    # For OneCycleLR, step after each batch (not epoch)
    # inside the batch loop: scheduler.step()

print(optimizer.param_groups[0]['lr'])  # inspect current lr
```

---

# 6. Common Architectures in PyTorch

## MLP (Multi-Layer Perceptron)

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

model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
```

## CNN (Convolutional Neural Network)

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))                # flexible sizing
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

## RNN / LSTM

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
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidir

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)             # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded) # output: (batch, seq_len, hidden*2)
        
        # Use last hidden state from both directions
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(last_hidden)
```

## Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = lambda x: x[:, 0, :]   # use [CLS] token
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.pool(x)
        return self.head(x)
```

---

# 7. Custom Loss Functions

The simplest way is just a function. Use a class when you need persistent state or learnable parameters.

```python
# Function style
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

# Class style — better for complex losses
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label = 0 if similar, 1 if dissimilar
        euclidean_dist = torch.nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * euclidean_dist ** 2 \
             + label * torch.clamp(self.margin - euclidean_dist, min=0) ** 2
        return loss.mean()

# Weighted cross-entropy (handle class imbalance)
class_weights = torch.tensor([1.0, 5.0, 2.0], device=device)  # higher weight for rare class
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Label smoothing (helps with overconfident predictions)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

# 8. Model Saving and Loading

## Saving

```python
# Recommended: save only the state dict
torch.save(model.state_dict(), 'model_weights.pth')

# Save full checkpoint with training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': val_loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

## Loading

```python
# Load weights only
model = SimpleNet(10, 64, 2)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

# Load full checkpoint and resume training
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# Handle partial loading (e.g., fine-tuning with new head)
pretrained = torch.load('pretrained.pth')
model_dict = model.state_dict()
# Filter out keys that don't match
pretrained_filtered = {k: v for k, v in pretrained.items() if k in model_dict}
model_dict.update(pretrained_filtered)
model.load_state_dict(model_dict)
```

Always save the state dict, not the full model object. Saving the full model with `torch.save(model, ...)` serializes the Python class definition along with the weights, making it brittle across code changes.

---

# 9. Debugging Tips

## Tensor Shape Issues

Shape errors are the most common problem. Build the habit of printing shapes throughout the forward pass.

```python
class DebugNet(nn.Module):
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.conv(x)
        print(f"After conv: {x.shape}")
        x = x.flatten(1)
        print(f"After flatten: {x.shape}")
        x = self.fc(x)
        print(f"Output: {x.shape}")
        return x
```

A cleaner approach is to register forward hooks during debugging:

```python
hooks = []
for name, layer in model.named_modules():
    hook = layer.register_forward_hook(
        lambda m, inp, out, n=name: print(f"{n}: {out.shape}")
    )
    hooks.append(hook)

output = model(dummy_input)

# Remove hooks after debugging
for h in hooks:
    h.remove()
```

## Gradient Inspection

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT")

# Check for gradient vanishing/explosion
grad_norms = {name: param.grad.norm().item()
              for name, param in model.named_parameters()
              if param.grad is not None}
print(max(grad_norms.values()), min(grad_norms.values()))
```

## NaN and Inf Detection

NaN gradients are the silent killer — the model trains, the loss looks like a number, but your weights are all NaN.

```python
# Check tensor for NaN/Inf
def check_tensor(t, name=""):
    if torch.isnan(t).any():
        print(f"NaN detected in {name}")
    if torch.isinf(t).any():
        print(f"Inf detected in {name}")

# Register backward hook to catch NaN gradients
def nan_hook(grad):
    if torch.isnan(grad).any():
        raise RuntimeError("NaN gradient detected")

for param in model.parameters():
    param.register_hook(nan_hook)

# anomaly detection mode — slower but gives exact location
with torch.autograd.detect_anomaly():
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
```

Common causes of NaN:
- log of zero (add epsilon: `torch.log(x + 1e-8)`)
- division by zero
- exploding gradients (fix: gradient clipping)
- bad weight initialization

## Mixed Precision Training

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

Mixed precision (float16 for forward/backward, float32 for optimizer) typically gives 2-3x speedup on modern GPUs with minimal accuracy loss.

---

# 10. PyTorch Lightning Quick Overview

PyTorch Lightning takes the raw PyTorch training loop and wraps it in a structured class. The same logic, less boilerplate.

```python
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleNet(10, 64, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]

# Training
trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',       # mixed precision
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    enable_checkpointing=True,
)
trainer.fit(model, train_loader, val_loader)
```

Lightning handles: device placement, gradient accumulation, multi-GPU, logging, checkpointing, mixed precision. You focus on the model and loss.

---

# 11. Common Interview Questions

**Q: What's the difference between `.view()` and `.reshape()`?**

Both reshape a tensor. `.view()` requires the tensor to be contiguous in memory and returns a view (no copy). `.reshape()` handles non-contiguous tensors by copying if needed. Safe rule: use `.reshape()` unless you specifically need a view guarantee.

**Q: Why do we call `optimizer.zero_grad()` before `loss.backward()`?**

PyTorch accumulates gradients by default — it adds new gradients to existing ones. If you don't zero them between batches, gradients from previous batches contaminate the current update. There are rare legitimate use cases for accumulation (e.g., gradient accumulation over multiple mini-batches to simulate larger batch size), but in the standard loop you always zero first.

**Q: What's the difference between `model.parameters()` and `model.state_dict()`?**

`parameters()` returns an iterator of parameter tensors (the actual weight and bias values). `state_dict()` returns an ordered dictionary mapping layer names to tensors — it includes both parameters and registered buffers (like BatchNorm running stats). Use `state_dict()` for saving/loading, use `parameters()` for the optimizer.

**Q: When would you use `requires_grad=False` on model layers?**

When fine-tuning a pretrained model, you often freeze the early layers (the feature extractor) and only train the later layers (the head). Freezing means setting `requires_grad=False` on those parameters so backprop stops there, reducing compute and preventing overfitting.

```python
# Freeze all layers except the final classifier
for param in model.features.parameters():
    param.requires_grad = False
# Only model.classifier parameters will be updated
```

**Q: What's the difference between BatchNorm and LayerNorm?**

BatchNorm normalizes across the batch dimension (over all examples for each feature). It needs a reasonably large batch size to get stable statistics, and it uses running mean/variance at inference. LayerNorm normalizes across the feature dimension for each example independently, making it batch-size agnostic. That's why transformers use LayerNorm — they process variable-length sequences and often have small batches.

**Q: What does `detach()` do and when would you use it?**

`.detach()` creates a new tensor that shares data with the original but is removed from the computational graph. Use it when: you want to use a tensor value but don't want gradients to flow through it (e.g., in actor-critic RL, the critic's target is detached to avoid updating the critic network during the actor update), or when converting to NumPy.

**Q: What's the difference between `nn.CrossEntropyLoss` and `nn.NLLLoss`?**

`CrossEntropyLoss` combines `LogSoftmax` and `NLLLoss` in one step. You pass raw logits directly. `NLLLoss` expects log-probabilities as input (output of `F.log_softmax`). Always use `CrossEntropyLoss` unless you have a specific reason to separate the steps — it's numerically more stable.

**Q: How does gradient clipping work and why is it used?**

Gradient clipping caps the global gradient norm before the optimizer step. If the total gradient norm exceeds a threshold (commonly 1.0), all gradients are scaled down proportionally.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

It's most important for RNNs and transformers, which are prone to gradient explosion. The clip prevents a single bad batch from sending weights to infinity.

**Q: What's the advantage of AdamW over Adam?**

Adam applies L2 regularization (weight decay) in a mathematically inconsistent way — the weight decay gets scaled by the adaptive learning rate and effectively doesn't apply uniformly. AdamW decouples weight decay from the gradient update, applying it directly to the weights. This is theoretically correct and empirically better, especially for transformers.

**Q: What happens if you forget `model.eval()` during validation?**

Dropout stays active (randomly zeroing activations), making validation results non-deterministic and generally worse than they should be. BatchNorm uses batch statistics instead of running statistics, which adds noise when the validation batch is small. Your validation metrics will be artificially pessimistic and noisy.

**Q: How would you debug a model that's producing NaN loss?**

Start with `torch.autograd.detect_anomaly()` to get the exact location. Then check: input data for NaN/Inf (log transforms on zero values are common culprits), weight initialization (Xavier/Kaiming is safer than random), learning rate (too high causes explosion), log operations (add epsilon), division operations (guard denominators). Gradient clipping often fixes the symptom without fixing the root cause, so dig deeper.
