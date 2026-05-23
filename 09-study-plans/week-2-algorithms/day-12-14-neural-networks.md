---
module: Study Plans
topic: Week 2 Algorithms
subtopic: Day 12 14 Neural Networks
status: unread
tags: [studyplans, ml, week-2-algorithms-day-12-14-ne]
---
# Day 12-14: Neural Networks & Deep Learning

## Why This Topic Comes Here

You have now studied the main classical algorithms: linear models, ensembles, SVMs, and clustering. Neural networks come next because they are best understood as the logical extension of what came before, not as a separate paradigm. A neural network with no hidden layers is logistic regression. A network with one hidden layer can approximate any function — whereas all of the earlier models are constrained to specific functional forms. The jump from day 11 to day 12 is the jump from "here are specific algorithms for specific problem shapes" to "here is one architecture that can approximate all of them, given sufficient data and compute."

---

## Executive Summary

| Concept | Role | Analogy |
|---------|------|---------|
| **Neuron** | Weighted sum + Activation | A decision node |
| **Backprop** | Error signal distribution | Teaching from mistakes |
| **Optimizer** | Weight update strategy | The way we walk downhill |
| **Activation** | Introducing Non-linearity | A logical switch |

---

## 1. The Perceptron & Beyond

**Why start with the single neuron:** The mechanics of one neuron — linear combination, nonlinear activation, update via gradient — is the template for the entire field. A network is just this template repeated and stacked. If you understand one neuron deeply, the rest is architecture.

The building block: $z = w^T x + b$, then apply $\sigma(z)$ to produce the activation.
- **MLP (Multi-Layer Perceptron)**: A stack of fully connected layers.
- **Universal Approximation Theorem**: A network with one hidden layer can approximate any continuous function given enough neurons.

**Key insight:** The Universal Approximation Theorem proves that a sufficiently wide shallow network *can* approximate any function — but says nothing about whether gradient descent will *find* that approximation, how much data you need to learn it reliably, or how efficiently the function is represented. Depth (many layers) often provides the same expressive power with exponentially fewer neurons, which is the practical reason deep networks work better than wide shallow ones.

**How to verify understanding:** A single-hidden-layer network with 1 million neurons vs. a 10-layer network with 1000 neurons per layer. Both can approximate the same functions in theory. Give two practical reasons why you would prefer the deep version.

**What trips people up:** Citing the Universal Approximation Theorem to justify that "a big enough network will always work." The theorem is an existence result, not a learnability result. It says nothing about the size of the dataset, the optimization landscape, or generalization. A network can be big enough to represent the target function while gradient descent fails to find it.

---

## 2. Activation Functions

**Why activation functions are not just implementation details:** Without non-linear activations, the entire network collapses to a linear transformation regardless of depth. But the choice of activation function also shapes the gradient signal — and a poor choice (sigmoid in deep networks) can kill learning entirely.

- **Sigmoid/Tanh**: Saturate (output approaches ±1/0 at extreme inputs), causing vanishing gradients in deep networks.
- **ReLU**: $\max(0, z)$. Non-saturating, fast to compute, but can "die" (neurons stuck at 0 if bias pushes pre-activation permanently negative).
- **Leaky ReLU / GELU**: Fix ReLU's dying neuron problem.

**Key insight:** ReLU is not universally better than sigmoid — it depends on the architecture and depth. In shallow networks, sigmoid is fine. In networks deeper than ~5 layers, vanishing gradients with sigmoid/tanh make training prohibitively slow. ReLU's non-saturating gradient was one of the key practical discoveries that made deep learning work.

**How to verify understanding:** A 20-layer network using sigmoid activations trains very slowly and the early layers show near-zero weight updates after 100 epochs. Explain the mechanism causing this. What architectural change (not just switching to ReLU) would also address it?

**What trips people up:** Thinking ReLU is always the right default. For output layers, the activation depends on the task: sigmoid for binary classification, softmax for multi-class, linear (no activation) for regression. Applying ReLU at the output of a regression network will clip all negative predictions to zero.

---

## 3. Backpropagation & Optimization

**Why backpropagation connects directly to day 1's gradient descent:** On day 1, you computed gradients of a simple MSE loss over one weight. Backpropagation is the same operation on a computation graph with millions of weights — it applies the chain rule recursively from the output layer back to the input layer. The math is the same; only the bookkeeping scales.

### The Chain Rule

The engine of learning. We calculate the gradient of the loss $L$ with respect to each weight $w$ by moving backward from output to input:
$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_k} \times \frac{\partial a_k}{\partial z_k} \times \dots$$

**Key insight:** Backpropagation does not "understand" the model — it is pure automatic differentiation. The gradient at each layer tells you how much that layer's weights contributed to the final error. Crucially, if the gradient is tiny at a layer (vanishing) or enormous (exploding), the update to that layer's weights is proportionally useless or destructive.

**How to verify understanding:** In a 10-layer network, which layers receive the smallest gradient updates during early training (assuming sigmoid activations)? Why, and what does this mean for how those layers learn relative to layers near the output?

**What trips people up:** Thinking that backpropagation guarantees the network will learn the right features. Backpropagation guarantees the network will minimize the training loss — which is not the same thing. If the loss function does not properly capture the task, the network will faithfully learn to minimize the wrong thing.

### Common Optimizers

- **SGD**: Simple stochastic updates. Requires careful learning rate tuning. Can escape local minima due to noise.
- **Momentum SGD**: Adds a velocity term to smooth out gradient noise.
- **Adam**: Combines **Momentum** (don't stop) and **RMSProp** (adjust learning rate per parameter). Usually the default starting point.

**Key insight:** Adam is not strictly better than SGD — there is evidence that well-tuned SGD with momentum generalizes better on some tasks (particularly image classification). Adam is popular because it is less sensitive to the initial learning rate choice. When you need to get something working quickly, use Adam. When you need to squeeze out the last 1% of performance, consider trying SGD with a learning rate schedule.

**How to verify understanding:** Adam adapts learning rates per parameter. What does this mean for a feature that rarely appears in the training data (sparse feature)? Why is Adam better than vanilla SGD for sparse features?

**What trips people up:** Setting the learning rate once and leaving it constant throughout training. Learning rate schedules (warmup, cosine decay, step decay) often significantly improve convergence and final performance. The "default" Adam learning rate of 1e-3 is a starting point, not a universal answer.

---

## 4. Regularization in Neural Networks

**Why regularization connects back to the bias-variance tradeoff from day 1:** Neural networks are high-variance models by nature — they have enough capacity to memorize any dataset. Regularization is the set of techniques that reduce this variance. Each technique has a different mechanism and appropriate use case.

- **L1/L2 Weight Decay**: Penalizes large weights in the loss function.
- **Dropout**: Randomly sets activations to zero during training. At test time, scales activations by the keep probability.
- **Batch Normalization**: Normalizes activations within each mini-batch, stabilizing training and acting as a mild regularizer.
- **Early Stopping**: Stop training when validation loss starts increasing.

**Key insight:** Dropout can be interpreted as approximate Bayesian inference over an ensemble of $2^N$ sub-networks (where $N$ is the number of neurons). Each training step trains a different subnetwork. At test time, using all neurons with scaled weights approximates averaging those subnetworks. This is why dropout sometimes generalizes better than explicit regularization.

**How to verify understanding:** You add dropout with p=0.5 to every layer, and your training loss becomes much noisier and slower to decrease. Is this a problem? Explain what dropout is doing to the loss landscape.

**What trips people up:** Applying dropout to the output layer during inference. Dropout must be turned off at inference time (`model.eval()` in PyTorch). Leaving it on during inference introduces random variation in predictions — a subtle but severe bug.

---

## Interview Questions

**1. "Why do we need non-linear activation functions (like ReLU)?"**
> If we used linear activations, the entire network would just be a linear combination of the inputs, making it equivalent to a single-layer linear model regardless of depth.

**2. "What is the Vanishing Gradient problem?"**
> In deep networks using sigmoid/tanh, gradients become very small as they propagate backward. Weights in early layers stop updating, "freezing" the network. **Solution**: Use **ReLU** and **Batch Normalization**.

**3. "Explain the 'Reparameterization Trick' (common for advanced DL interviews)."**
> To propagate gradients through a stochastic node (like in VAEs), we move the randomness to an external variable: $z = \mu + \sigma \epsilon$ where $\epsilon \sim N(0, 1)$. This makes the sampling process differentiable.

---

## PyTorch Scaffold

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout(self.bn(self.relu(self.fc(x))))
        return self.out(x)
```

## Flashcards

**MLP (Multi-Layer Perceptron)?** #flashcard
A stack of fully connected layers.

**Universal Approximation Theorem?** #flashcard
A network with one hidden layer can approximate any continuous function given enough neurons.

**Sigmoid/Tanh?** #flashcard
Saturate (output approaches ±1/0 at extreme inputs), causing vanishing gradients in deep networks.

**ReLU?** #flashcard
$\max(0, z)$. Non-saturating, fast to compute, but can "die" (neurons stuck at 0 if bias pushes pre-activation permanently negative).

**Leaky ReLU / GELU?** #flashcard
Fix ReLU's dying neuron problem.

**SGD?** #flashcard
Simple stochastic updates. Requires careful learning rate tuning. Can escape local minima due to noise.

**Momentum SGD?** #flashcard
Adds a velocity term to smooth out gradient noise.

**Adam?** #flashcard
Combines Momentum (don't stop) and RMSProp (adjust learning rate per parameter). Usually the default starting point.

**L1/L2 Weight Decay?** #flashcard
Penalizes large weights in the loss function.

**Dropout?** #flashcard
Randomly sets activations to zero during training. At test time, scales activations by the keep probability.

**Batch Normalization?** #flashcard
Normalizes activations within each mini-batch, stabilizing training and acting as a mild regularizer.

**Early Stopping?** #flashcard
Stop training when validation loss starts increasing.
