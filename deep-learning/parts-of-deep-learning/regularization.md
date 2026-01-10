# Regularization in Deep Learning

Regularization is a set of techniques used to prevent **overfitting** by discouraging the model from learning the noise in the training data, thereby improving its ability to generalize to unseen data.

---

## 🏗️ Core Techniques

### 1. L1 and L2 Regularization (Weight Decay)
Adds a penalty term to the loss function based on the magnitude of the weights.

- **L1 (Lasso):** Loss = Original Loss + $\lambda \sum |\theta_i|$. Leads to **sparse weights** (feature selection).
- **L2 (Ridge):** Loss = Original Loss + $\lambda \sum \theta_i^2$. Shrinks weights toward zero; commonly used in weight decay.

### 2. Dropout
During training, randomly "drop out" (set to zero) a percentage of neurons in a layer for each iteration.
- **Why?** Forces the network to learn redundant representations and prevents cross-dependence between neurons.
- **Note:** Dropout is only applied during **training**, not during inference. During inference, weights are scaled by the dropout probability.

### 3. Batch Normalization
While primarily an optimization technique, it has a regularizing effect.
- **Mechanism:** Normalizes the activations of each layer to have zero mean and unit variance.
- **Benefit:** Reduces internal covariate shift and allows for higher learning rates.

### 4. Early Stopping
Monitor the model's performance on a validation set and stop training when the validation loss starts to increase, even if the training loss is still decreasing.

### 5. Data Augmentation
Artificially increases the size of the training set by applying transformations to the existing data (e.g., flips, rotations, noise in images; synonym replacement in text).

---

## ❓ Interview Questions

**1. "Why does L1 lead to sparsity while L2 doesn't?"**
> L1 has a constant gradient (except at zero), so it keeps pushing weights all the way to zero. L2's gradient decreases as the weight gets smaller, so it effectively "slows down" the shrinkage as it nears zero.

**2. "How do you handle Dropout during inference?"**
> During inference, we don't drop any neurons. Instead, we multiply the output of the layer by $(1-p)$, where $p$ is the dropout probability used during training, to maintain the expected value of the activations.

**3. "Can Batch Norm replace Dropout?"**
> Often, yes. In many modern architectures (like ResNet), Batch Norm provides enough regularization that Dropout is not strictly necessary.
