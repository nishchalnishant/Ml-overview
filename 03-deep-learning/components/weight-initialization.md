# Weight Initialization

---

## The Signal Propagation Problem

**The problem**: before any training begins, you choose the starting values for all weights. If these values are wrong, the first forward pass produces activations that are either near zero (all neurons saturated or silent) or astronomically large (activations overflow). Either way, the gradients computed in the first backward pass are useless — near zero because activations are saturated, or NaN because values exploded. Training never recovers from a bad start.

**The core insight**: at initialization, you want the signal through the network to stay at roughly the same scale at every layer. If layer $l$ receives a vector of variance 1 and outputs a vector of variance $\gg 1$ or $\ll 1$, those discrepancies compound multiplicatively through $L$ layers. The goal is a starting configuration where forward and backward passes both propagate signal without shrinkage or explosion.

---

## Zero Initialization

**The problem**: the simplest initialization — set all weights to zero.

**What breaks**: every neuron in the same layer receives identical inputs and computes an identical output. Every neuron computes an identical gradient. After the first update, all neurons in each layer remain identical — the network is equivalent to a single neuron per layer, regardless of declared width. Symmetry is never broken; the network fails to learn diverse representations.

Biases can be initialized to zero (or small constants). Weights must not be.

---

## Naive Random Initialization

**The problem**: initialize weights from a small fixed normal distribution (e.g., $\mathcal{N}(0, 0.01)$) without considering layer size.

**What breaks**: consider a layer with 1000 inputs. The output of neuron $j$ is $a_j = \sum_{i=1}^{1000} w_i x_i$. If each $w_i \sim \mathcal{N}(0, 0.01)$ and each $x_i$ has unit variance, then $\text{Var}[a_j] = 1000 \times 0.01^2 \times 1 = 0.1$ — small, activations are near zero, gradients vanish. If instead $\mathcal{N}(0, 0.1)$, then $\text{Var}[a_j] = 1000 \times 0.01 = 10$ — large, activations saturate, gradients vanish.

A fixed standard deviation without layer-size correction cannot work for arbitrary network widths.

---

## Xavier / Glorot Initialization

**The problem**: you need initialization designed for symmetric activations (tanh, sigmoid) that are approximately linear near zero. How do you set weight variance so signal neither shrinks nor explodes through the layer?

**The core insight**: require that the variance of the output equals the variance of the input (forward pass), and that the gradient variance is preserved backward. For a layer with $n_\text{in}$ inputs and $n_\text{out}$ outputs, the forward constraint gives $\text{Var}[W] = 1/n_\text{in}$ and the backward constraint gives $\text{Var}[W] = 1/n_\text{out}$. Compromise: use the harmonic mean.

**The mechanics**:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_\text{in} + n_\text{out}}\right) \quad \text{or} \quad W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right]$$

```python
nn.init.xavier_uniform_(layer.weight)   # uniform variant
nn.init.xavier_normal_(layer.weight)    # normal variant
```

**What breaks**: Xavier's derivation assumes the activation is linear (or approximately linear). Tanh and sigmoid are approximately linear near zero, so Xavier works for them. ReLU is not: it zeroes half the inputs unconditionally, cutting the effective variance in half at every layer. Applying Xavier to a ReLU network causes activations to shrink by $\sqrt{2}$ per layer — in a 10-layer network, activations shrink by $(\sqrt{2})^{10} = 32\times$.

---

## He / Kaiming Initialization

**The problem**: Xavier assumes the activation is linear. ReLU zeroes all negative pre-activations — effectively, half the neurons contribute nothing. If variance is calibrated for a linear unit, a ReLU network has half as much variance at each layer. This halving compounds through layers.

**The core insight**: ReLU halves the variance at each layer. Compensate by doubling the initial weight variance. The factor of 2 in the variance exactly offsets the factor of 2 lost to the zeroing of negative activations.

**The mechanics**:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_\text{in}}}\right)$$

The variance is $2/n_\text{in}$ instead of Xavier's $1/n_\text{in}$. The "2" is the compensation for ReLU's expected zeroing of half the inputs.

```python
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
```

`mode='fan_in'` preserves variance in the forward pass. `mode='fan_out'` preserves variance in the backward pass. For most networks, `fan_in` is preferred.

PyTorch's default initialization for `nn.Linear` and `nn.Conv2d` is Kaiming Uniform — correct for ReLU networks.

**What breaks**: He initialization assumes exactly half of inputs are zeroed (the expected behavior for zero-mean inputs). If the input distribution is shifted (e.g., after BatchNorm that doesn't zero-center), fewer or more inputs may be zeroed, and the factor of 2 is inexact. In practice, this is a minor issue — He initialization is robust enough.

For Leaky ReLU with slope $\alpha$: the effective variance multiplier is $(1 + \alpha^2)/2$ instead of $1/2$. Pass `nonlinearity='leaky_relu', a=alpha` to get the corrected initialization.

---

## Orthogonal Initialization

**The problem**: in recurrent networks (RNNs, LSTMs), the same weight matrix $W_h$ is applied repeatedly at every timestep: $h_t = f(W_h h_{t-1} + \ldots)$. The gradient of the loss with respect to $h_0$ involves $W_h^T$ raised to the power of the sequence length. If $\|W_h\| < 1$, gradients vanish. If $\|W_h\| > 1$, gradients explode. You need the gradient norm to be preserved exactly.

**The core insight**: an orthogonal matrix satisfies $W^T W = I$ — it is an isometry. Multiplying a vector by an orthogonal matrix preserves its norm exactly. Initializing $W_h$ as an orthogonal matrix means the gradient norm is preserved at every recurrent step.

**The mechanics**:

```python
nn.init.orthogonal_(layer.weight, gain=1.0)
```

**What breaks**: the orthogonal initialization only preserves norms for a linear map. Once a non-linear activation is applied, the norm guarantee breaks. Still, orthogonal initialization provides a much better starting point for recurrent networks than random normal initialization.

---

## Transformer-Specific Initialization

**The problem**: residual connections add the sublayer output to the input directly. If each sublayer adds a vector with non-trivial variance, the variance of the residual stream grows with depth: after $L$ layers, the stream variance is $L$ times the variance of a single sublayer's output. For deep Transformers (GPT-3 has 96 layers), this causes the output logit magnitudes to grow with depth, destabilizing training.

**The core insight**: scale the output of each residual sublayer by $1/\sqrt{2L}$ where $L$ is the number of layers. Each sublayer contributes variance $1/(2L)$ instead of 1. After $L$ additive layers, total variance is $L \times 1/(2L) = 1/2$ — bounded regardless of depth.

**The mechanics** (GPT-style):

```python
import math

std = 0.02 / math.sqrt(2 * n_layers)
nn.init.normal_(output_projection.weight, mean=0, std=std)
```

Applied to the output projection of each attention block and each FFN block. All other weights initialized from $\mathcal{N}(0, 0.02)$.

**What breaks**: if this scaling is omitted, deeper Transformers require more careful learning rate tuning and are prone to training instability in the first few thousand steps. The issue is particularly visible at large model scale.

---

## LSTM Initialization

**The problem**: at the start of LSTM training, the forget gate determines how much of the previous cell state to retain. If the forget gate activates near zero (due to random initialization), the LSTM immediately forgets everything — it cannot learn long-range dependencies from the first batch.

**The core insight**: bias the forget gate toward remembering at initialization. Set the forget gate bias to 1 (in the range where sigmoid $\approx 0.73$). The LSTM starts in a state that preserves context; it can learn to forget when the data requires it.

**The mechanics**:

```python
for name, param in lstm.named_parameters():
    if 'weight_ih' in name:
        nn.init.kaiming_uniform_(param)       # input-to-hidden: treat as feedforward
    elif 'weight_hh' in name:
        nn.init.orthogonal_(param)            # hidden-to-hidden: orthogonal for stability
    elif 'bias' in name:
        nn.init.zeros_(param)
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)     # forget gate bias = 1
```

**What breaks**: if the forget gate bias is set too high (e.g., bias = 5), the LSTM is too slow to learn to forget — it memorizes everything and fails to segment sequences. The initialization to 1 is a soft prior, not a hard constraint.

---

## Bias Initialization Reference

| Layer type | Weight init | Bias init | Reason |
| :--- | :--- | :--- | :--- |
| Linear / Conv | Kaiming (ReLU) or Xavier (tanh) | Zero | Standard |
| BatchNorm $\gamma$ | One | — | Start as identity scaling |
| BatchNorm $\beta$ | — | Zero | Start as identity shift |
| Output (classification) | Xavier / Kaiming | Zero | Balanced starting logits |
| Output (regression) | Xavier / Kaiming | Dataset mean | Faster early convergence |
| LSTM forget gate | — | One | Encourage remembering at start |
| Embedding | $\mathcal{N}(0, 0.02)$ | — | GPT standard |

---

## Initialization Summary

| Method | Activation | Formula | Key idea |
| :--- | :--- | :--- | :--- |
| **Zero** | Never use for weights | $w = 0$ | Breaks symmetry |
| **Xavier** | Tanh, Sigmoid | $\mathcal{N}(0, 2/(n_\text{in}+n_\text{out}))$ | Preserve signal for linear-ish activations |
| **He / Kaiming** | ReLU, Leaky ReLU | $\mathcal{N}(0, 2/n_\text{in})$ | Compensate for ReLU zeroing half inputs |
| **Orthogonal** | RNN hidden-to-hidden | $W^T W = I$ | Preserve gradient norms in recurrence |
| **GPT residual scaling** | Any | $\mathcal{N}(0, 0.02/\sqrt{2L})$ | Prevent variance growth through depth |
