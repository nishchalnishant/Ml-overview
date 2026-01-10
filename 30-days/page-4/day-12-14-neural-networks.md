# Day 12-14: Neural Networks & Deep Learning

##  Executive Summary
| Concept | Role | Analogy |
|---------|------|---------|
| **Neuron** | Weighted sum + Activation | A decision node |
| **Backprop** | Error signal distribution | Teaching from mistakes |
| **Optimizer** | Weight update strategy | The way we walk downhill |
| **Activation** | Introducing Non-linearity | A logical switch |

---

## 🧠 1. The Perceptron & Beyond
The building block of NN: $z = w^T x + b$, then apply $\sigma(z)$.
- **MLP (Multi-Layer Perceptron)**: A stack of fully connected layers.
- **Universal Approximation Theorem**: A network with one hidden layer can approximate any continuous function given enough neurons.

---

##  2. Backpropagation & Optimization

### The Chain Rule
The engine of learning. We calculate the gradient of the loss $L$ with respect to each weight $w$ by moving backward from output to input.
$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_k} \times \frac{\partial a_k}{\partial z_k} \times \dots$$

### Common Optimizers
- **SGD**: Simple, stochastic updates.
- **Adam**: Combines **Momentum** (don't stop) and **RMSProp** (adjust learning rate per parameter). Usually the default.

---

##  Interview Questions

**1. "Why do we need non-linear activation functions (like ReLU)?"**
> If we used linear activations, the entire network would just be a linear combination of the inputs, making it equivalent to a single-layer linear model regardless of depth.

**2. "What is the Vanishing Gradient problem?"**
> In deep networks using sigmoid/tanh, gradients become very small as they propagate backward. Weights in early layers stop updating, "freezing" the network. **Solution**: Use **ReLU** and **Batch Normalization**.

**3. "Explain the 'Reparameterization Trick' (common for advanced DL interviews)."**
> To propagate gradients through a stochastic node (like in VAEs), we move the randomness to an external variable: $z = \mu + \sigma \epsilon$ where $\epsilon \sim N(0, 1)$. This makes the sampling process differentiable.

---

##  PyTorch Scaffold
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        return self.out(self.relu(self.fc(x)))
```
