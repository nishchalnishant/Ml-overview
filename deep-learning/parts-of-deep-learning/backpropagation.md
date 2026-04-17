# Backpropagation

Backpropagation is how a neural network learns which weights deserve blame.

That is the cleanest mental model.

Forward pass:

- make prediction

Backward pass:

- assign responsibility for error

---

# 1. The Core Idea

Backpropagation uses the chain rule to compute how the final loss changes with respect to each weight.

Instead of differentiating everything from scratch for every parameter, it reuses intermediate derivatives efficiently.

That efficiency is the reason deep learning is practical at all.

---

# 2. Why the Chain Rule Matters

If a network is a stack of functions, then the effect of an early weight on final loss depends on all the functions after it.

So the gradient is a chained product of local sensitivities.

That is why:

- deep networks can suffer vanishing gradients
- exploding gradients can happen
- architecture choices affect trainability

---

# 3. Vanishing vs Exploding Gradients

## Vanishing

Gradients shrink too much.

Result:

- early layers barely learn

## Exploding

Gradients grow too much.

Result:

- unstable updates
- `NaN` sadness

Common fixes:

- ReLU-style activations
- residual connections
- normalization
- gradient clipping

---

# 4. Autograd in Practice

In frameworks like PyTorch:

- the forward pass builds the computation graph
- the backward pass walks it in reverse

So you usually do not hand-code derivatives in production.

But understanding backprop still matters because it helps you debug:

- stalled learning
- unstable training
- wrong loss/activation pairing

---

# 5. Whiteboard Answer Formula

If asked to explain backprop in an interview:

1. define the forward equations
2. name the loss
3. apply the chain rule backward
4. explain the gradient meaning
5. mention the update step

That structure sounds clean and confident.

---

# Quick Thought Experiment

If an early layer never learns, what are you suspicious of?

- vanishing gradients
- bad activation choice
- poor initialization

Usually some combination of those.
