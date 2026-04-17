# PyTorch Foundations

PyTorch is where deep learning stops being theory and starts becoming code that either trains beautifully or ruins your afternoon.

If Azure DevOps is your comfort zone, think of PyTorch as the runtime and training framework where:

- tensors are the payload
- autograd is the dependency tracker
- the training loop is your build pipeline

---

# 1. Tensors

Tensors are the basic data structure in PyTorch.

They are like NumPy arrays, but with two huge superpowers:

- they work cleanly with GPUs
- they integrate with autograd for backprop

You should be comfortable with:

- scalar
- vector
- matrix
- higher-dimensional tensors

Especially shapes like:

- `[batch, channels, height, width]`

for images.

---

# 2. Autograd

Autograd automatically tracks operations so gradients can be computed during backprop.

If a tensor has:

- `requires_grad=True`

then PyTorch keeps enough graph information to differentiate later.

That means:

- forward pass builds the graph
- backward pass computes gradients

Very convenient.
Very important.

---

# 3. The Training Loop

This is the PyTorch rhythm you should know cold:

1. forward pass
2. compute loss
3. zero gradients
4. backward pass
5. optimizer step

If you can explain that cleanly, you already sound solid in most implementation discussions.

---

# 4. Model, Loss, Optimizer

These are the big three pieces:

- `nn.Module` for model definition
- loss function for training objective
- optimizer for parameter updates

That trio appears in nearly every PyTorch workflow.

---

# 5. Devices: CPU, CUDA, MPS

You move both:

- model
- data

onto the same device.

That sounds trivial.
It is also one of the most common practical mistakes.

**Short rule**

If tensors and model are on different devices, pain arrives quickly.

---

# 6. Good Practical Instinct

In real PyTorch work, people care about:

- shape sanity
- device placement
- gradient flow
- reproducibility
- data loader behavior

So if you want to sound strong, do not only talk about layers.
Talk about the whole workflow.
