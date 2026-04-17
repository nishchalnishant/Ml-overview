# PyTorch Foundations (Deep-Dive)

PyTorch is the industry-standard framework for deep learning research and production. It is built on the concept of **Tensors** and **Dynamic Computational Graphs**.

---

# 1. 🔹 Tensors: The Primitive of AI

## Q1: What is a Tensor, and how does it differ from a NumPy array?

### 🔹 Direct Answer
A **Tensor** is a multi-dimensional array that can be run on a **GPU** or **TPU**. While it shares an API with NumPy, PyTorch Tensors automatically track gradients via **Autograd**, which is essential for backpropagation.

### 🔹 Tensor Dimensions
- **Scalar (0-dim):** `torch.tensor(7)`
- **Vector (1-dim):** `torch.tensor([1, 2, 3])`
- **Matrix (2-dim):** `torch.tensor([[1, 2], [3, 4]])`
- **Tensor (n-dim):** Representing an image as `[Batch, Channels, Height, Width]`.

---

# 2. 🔹 Automatic Differentiation (Autograd)

## Q2: How does PyTorch handle backpropagation?

### 🔹 Direct Answer
PyTorch uses a **Dynamic Computational Graph** (Define-by-Run). Every operation on a tensor with `requires_grad=True` is recorded. When `.backward()` is called on the loss, PyTorch traverses this graph in reverse to calculate the partial derivatives for every weight.

### 🔹 Practical Snippet
```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward() # Computes gradients
print(x.grad)  # Access the calculated gradients
```

---

# 3. 🔹 Training Workflow: The Big Four

To train a model in PyTorch, you always follow these 4 steps:

1. **The Model:** Define a class inheriting from `nn.Module`.
2. **The Loss:** Define a criterion (e.g., `nn.CrossEntropyLoss()`).
3. **The Optimizer:** Define how weights update (e.g., `torch.optim.AdamW()`).
4. **The Loop:**
    - `optimizer.zero_grad()`: Reset gradients from the previous step.
    - `loss.backward()`: Compute new gradients.
    - `optimizer.step()`: Update the weights.

---

# 4. 🔹 Hardware Acceleration

## Q3: How do you move tensors to GPU?

### 🔹 Best Practice
```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
X, y = X.to(device), y.to(device)
```

---

> [!TIP]
> **Production Recommendation:** For production-grade PyTorch code, use **PyTorch Lightning** to separate the boilerplate (loops, device management) from the research logic (architecture, data).
