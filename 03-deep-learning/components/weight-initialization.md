# Weight Initialization

Weight initialization determines the starting point of optimization. Poor initialization causes vanishing or exploding gradients, slowing or breaking training entirely.

---

## Why Initialization Matters

At the start of training, the loss landscape is navigated from an initial point. If:
- **Weights too small:** Activations and gradients shrink layer by layer → vanishing gradients
- **Weights too large:** Activations saturate or explode → exploding gradients / saturation
- **All weights equal:** All neurons compute identical gradients → symmetry breaking fails (network equivalent to a single neuron)

**Goal:** Keep activations and gradients at similar scales across all layers throughout the early phase of training.

---

## Zero / Constant Initialization

```python
nn.init.zeros_(layer.weight)
nn.init.constant_(layer.weight, 0.01)
```

**Never initialize weights to zero.** All neurons compute identical outputs and gradients — symmetry is never broken. Biases can be initialized to zero (or small constants).

---

## Random Normal / Uniform

```python
nn.init.normal_(layer.weight, mean=0, std=0.01)
nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
```

**Problem:** Choosing std arbitrarily leads to vanishing/exploding signals as networks deepen. Variance not calibrated to layer size.

---

## Xavier / Glorot Initialization

Designed for **tanh and sigmoid** activations (approximately linear near zero).

**Idea:** Set variance so signal neither shrinks nor explodes. For a layer with `n_in` inputs and `n_out` outputs:

**Uniform:** `W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]`

**Normal:** `W ~ N(0, 2/(n_in + n_out))`

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

**Derivation:** Requires `Var[W] × n_in = 1` (forward pass) and `Var[W] × n_out = 1` (backward pass). Compromise: `Var[W] = 2 / (n_in + n_out)`.

---

## He / Kaiming Initialization

Designed for **ReLU** activations (which zero out half of inputs, effectively halving variance).

**Normal:** `W ~ N(0, 2/n_in)`  
**Uniform:** `W ~ U[-√(6/n_in), √(6/n_in)]`

```python
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

- `mode='fan_in'`: Preserves variance in forward pass
- `mode='fan_out'`: Preserves variance in backward pass

**For LeakyReLU:** Pass `nonlinearity='leaky_relu'` and `a=slope`.

**In practice:** PyTorch defaults to Kaiming Uniform for `nn.Linear` and `nn.Conv2d`. Default is correct for ReLU networks.

---

## Orthogonal Initialization

Initialize weight matrix as a random orthogonal matrix. Preserves gradient norms exactly at initialization for linear networks.

```python
nn.init.orthogonal_(layer.weight, gain=1.0)
```

**Use for:** RNNs (prevents vanishing/exploding gradients in recurrent connections), deep linear networks. Less common for standard CNNs/MLPs.

---

## LSTM / RNN Initialization

```python
for name, param in lstm.named_parameters():
    if 'weight_ih' in name:
        nn.init.kaiming_uniform_(param)     # input-hidden weights
    elif 'weight_hh' in name:
        nn.init.orthogonal_(param)          # hidden-hidden weights
    elif 'bias' in name:
        nn.init.zeros_(param)
        # Initialize forget gate bias to 1 — helps remember at start
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)
```

---

## Transformer Initialization

GPT-style transformers use a scaled initialization for residual connections:

```python
# Scale residual branch output by 1/√(2 * n_layers)
# Prevents variance explosion through many residual blocks
std = 0.02 / math.sqrt(2 * config.n_layers)
nn.init.normal_(layer.weight, mean=0, std=std)
```

**Embedding initialization:** Small normal `N(0, 0.02)` standard in GPT.

---

## Bias Initialization

- **Linear / Conv layers:** Zero — standard
- **BatchNorm γ (scale):** 1; β (shift): 0
- **Output layer for classification:** Zero (balanced starting logits)
- **Output layer for regression:** Set to dataset mean (faster convergence)
- **LSTM forget gate:** 1 (encourage remembering at start)

---

## Gain / Nonlinearity Scaling

Xavier and He initializations have a `gain` parameter for different activation functions:

| Activation | Recommended gain |
|-----------|-----------------|
| Linear | 1.0 |
| Sigmoid | 1.0 |
| Tanh | 5/3 ≈ 1.667 |
| ReLU | √2 ≈ 1.414 |
| LeakyReLU(0.01) | √(2/1.01²) |

```python
gain = nn.init.calculate_gain('relu')
std = gain / math.sqrt(n_in)
nn.init.normal_(weight, std=std)
```

---

## Layer-Specific Best Practices

```python
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

model.apply(init_weights)
```

---

## Key Interview Points

- Never initialize all weights to the same value — symmetry breaking is essential.
- Xavier: for tanh/sigmoid. He/Kaiming: for ReLU. Key difference: ReLU zeroes half the activations, needing 2× the variance.
- PyTorch's default `nn.Linear` uses Kaiming Uniform — correct for ReLU.
- Orthogonal initialization is preferred for RNN hidden-hidden weights.
- LSTM forget gate bias = 1 helps the network remember by default at the start of training.
- Residual networks scale the residual branch by `1/√(2L)` to prevent variance accumulation.
