---
module: Deep Learning Core
topic: Dl
subtopic: Deep Learning Fundamentals
status: unread
tags: [interviewprep, dl, dl-deep-learning-fundamentals, interview-framing]
---
# Deep Learning

---

## 1. What Is a Neural Network?

**What the interviewer is testing**: whether you understand a neural network as a learned representation hierarchy, not just as a stack of matrix multiplications with nonlinearities.

**The reasoning structure**: the fundamental problem neural networks solve is feature engineering at scale. For images, text, and audio, useful features are not obvious and require composition — an edge detector is useful, but what you really need is a curve detector built from edge detectors, then a shape detector built from curves, then a concept detector built from shapes. Building this hierarchy by hand is infeasible for anything complex. Neural networks learn the hierarchy from data.

Each layer transforms its input into a new representation. What form that representation takes is determined by what helps the loss decrease. By the final layers, the representation is no longer raw pixels or text tokens — it is a task-relevant abstraction. The same mechanism that makes neural networks hard to interpret is what makes them powerful: the learned representations are not human-engineered categories.

**The pattern in action**: a convolutional network trained on ImageNet learns edge detectors in layer 1, texture detectors in layer 2, part detectors (eyes, wheels) in layer 4, and object detectors in later layers — without anyone specifying that "wheels" should be a useful intermediate concept. The hierarchy emerged because it was useful for reducing classification loss.

**Common traps**:
- claiming neural networks can learn any function — technically true (universal approximation theorem) but practically misleading; the specific inductive biases of the architecture, the optimization landscape, and the amount of data all determine what is actually learned; "universal approximator" is not a guarantee of practical performance
- treating depth as always better than width — depth enables hierarchical abstractions; width enables more features at each level; the right balance depends on the structure of the problem

---

## 2. Backpropagation

**What the interviewer is testing**: whether you understand backpropagation as efficient application of the chain rule, and specifically what makes it efficient compared to the naive alternative.

**The reasoning structure**: the naive way to compute gradients is finite differences — perturb each parameter slightly, measure the change in loss. With 100 million parameters, this requires 100 million forward passes. Backpropagation computes all gradients in a single backward pass using the chain rule.

The chain rule says: the gradient of the loss with respect to an early layer's parameters equals the gradient of the loss with respect to the next layer's output, multiplied by the gradient of that output with respect to the current layer's parameters. Backpropagation applies this recursively from output layer to input layer.

What makes it efficient: you cache the activations from the forward pass, then reuse them in the backward pass to compute gradients. The backward pass costs roughly the same as the forward pass, regardless of the number of parameters.

**The pattern in action**: for a dense layer $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$ with activation $a^{(l)} = f(z^{(l)})$:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot (a^{(l-1)})^T$$

The gradient with respect to the weights is the outer product of the upstream gradient and the cached activations from the forward pass. Without caching $a^{(l-1)}$, you would need to recompute the entire forward pass for each layer's gradient — turning a single backward pass into $L$ forward passes.

**Common traps**:
- confusing backpropagation with gradient descent — backprop computes the gradients; the optimizer uses them to update parameters; these are separate algorithms
- not knowing that ReLU's backward pass is just masking: $\frac{d}{dx}\text{ReLU}(x) = \mathbb{1}[x > 0]$ — the gradient is blocked wherever the pre-activation was negative

---

## 3. Vanishing and Exploding Gradients

**What the interviewer is testing**: whether you understand gradient instability as a consequence of the chain rule applied to deep networks, not just as "a known problem with deep networks."

**The reasoning structure**: backpropagation multiplies gradients layer by layer. For a network with $L$ layers, the gradient at layer $l$ involves a product of $L - l$ Jacobian matrices:

$$\frac{\partial L}{\partial \theta^{(l)}} \propto \prod_{k=l}^{L-1} \frac{\partial a^{(k+1)}}{\partial a^{(k)}}$$

If each Jacobian has maximum singular value less than 1 — which happens with sigmoid or tanh activations that saturate — the product decays exponentially with depth. Vanishing gradients: early layers receive near-zero gradient signal and barely update.

If each Jacobian has maximum singular value greater than 1 — common in RNNs unrolled over long sequences — the product grows exponentially. Exploding gradients: training diverges.

**The pattern in action**: "I train a 20-layer network with sigmoid activations. After 100 epochs, I check gradient norms layer by layer. Layer 20: norm 0.5. Layer 10: norm 0.001. Layer 1: norm $10^{-9}$. The first few layers have essentially learned nothing — they received no gradient signal. The fix: replace sigmoids with ReLU, add residual connections, use batch normalization to keep activations in a stable range."

**Common traps**:
- thinking gradient clipping fixes vanishing gradients — clipping only addresses exploding gradients (by capping magnitude); it cannot increase gradient magnitude for signals that are already near zero; vanishing gradients require architectural solutions
- using sigmoid activations in hidden layers of a deep network — $\sigma'(z) \leq 0.25$ everywhere, meaning each sigmoid layer multiplies the gradient by at most 0.25; after 20 layers, the gradient is below $10^{-12}$

---

## 4. Activation Functions

**What the interviewer is testing**: whether you understand activation function choice as a consequence of the gradient behavior you need, not as a menu you memorize.

**The reasoning structure**: without activation functions, a stack of linear layers is mathematically equivalent to a single linear layer. Activations introduce the nonlinearity that gives deep networks their expressive power. The specific choice determines gradient behavior, which determines whether deep networks train stably.

The key property: does the activation saturate? Saturating activations (sigmoid, tanh) have near-zero gradients in large portions of their domain. Fine for shallow networks or output layers. For deep hidden layers, saturation kills gradients via the vanishing gradient problem.

ReLU fixes saturation for positive inputs (gradient is always 1 for positive inputs) but introduces dying neurons: if a neuron's pre-activation is consistently negative, it receives zero gradient and never updates — it is effectively dead. Leaky ReLU prevents this with a small negative slope for negative inputs.

**The pattern in action**:

| Activation | Where to use | Why | Problem |
| :--- | :--- | :--- | :--- |
| ReLU | Hidden layers (CNNs, MLPs) | Non-saturating, cheap | Dying neurons |
| Leaky ReLU | When dying neurons are observed | Prevents dead neurons | Slightly more compute |
| GELU | Transformer hidden layers | Smooth, strong empirical performance | More expensive than ReLU |
| Sigmoid | Binary classification output only | Maps to (0, 1) probability | Saturates in hidden layers |
| Tanh | RNN hidden states | Zero-centered, bounded | Saturates |
| Softmax | Multiclass classification output | Probabilities summing to 1 | Numerical overflow without max subtraction |

**Common traps**:
- using sigmoid in hidden layers of a deep network — sigmoid is appropriate only as the final activation for binary classification output
- not implementing numerically stable softmax — subtract the maximum logit before exponentiating: `exp(z - max(z))`; mathematically equivalent but prevents overflow
- treating GELU as just a smooth ReLU — GELU gates the input by the probability that a standard Gaussian exceeds that value: $\text{GELU}(x) = x \cdot \Phi(x)$; this soft gating has consistently outperformed ReLU in attention-based architectures

---

## 5. Batch Normalization

**What the interviewer is testing**: whether you understand BatchNorm as a solution to internal covariate shift, with specific and critical behavioral differences during training vs inference.

**The reasoning structure**: during training, as parameters update in one layer, the distribution of that layer's outputs changes. The next layer must continually adapt to a shifting input distribution — internal covariate shift. This slows training because each layer's optimal weights keep becoming suboptimal as upstream layers change.

BatchNorm normalizes each layer's activations to zero mean and unit variance across the batch, then applies learnable scale $\gamma$ and shift $\beta$. This reduces dependence on initialization and makes the optimization landscape smoother, allowing higher learning rates.

Critical behavioral difference: during training, BatchNorm uses batch mean and variance computed on the current batch. During inference, it uses running statistics accumulated during training. Forgetting to switch modes is one of the most common deep learning production bugs.

**The pattern in action**: "I train a ResNet without BatchNorm and need careful learning rate scheduling and warmup to avoid instability. I add BatchNorm after each convolution and can use a higher learning rate, training faster and reaching better performance. At inference, I call `model.eval()` — otherwise BatchNorm computes statistics from the single inference batch, which is noisy and wrong for models that process one example at a time."

**Common traps**:
- using BatchNorm with very small batch sizes (< 8) — batch statistics from 2–4 images are too noisy to be useful; switch to LayerNorm (used in Transformers, operates per example independently of batch size) or GroupNorm
- placing BatchNorm before dropout in the same block — BatchNorm's normalization changes when dropout is applied first, because the dropped activations alter the batch statistics; typically apply BatchNorm before dropout if both are used

---

## 6. Dropout

**What the interviewer is testing**: whether you understand dropout's regularization mechanism and the inference-time behavior change.

**The reasoning structure**: the problem dropout solves is co-adaptation — neurons develop complex interdependencies where A's output is only useful when combined with B's specific output. Together they memorize training examples. Randomly disabling neurons during each training step breaks these dependencies; each neuron must learn to be useful on its own with any random subset of co-neurons.

An equivalent view: dropout trains an implicit ensemble of $2^n$ thinned networks and approximates the ensemble average at inference time through weight scaling.

**The pattern in action**: "I add dropout with rate 0.5 after the second hidden layer. During each training step, roughly half the neurons in that layer are randomly disabled. Neuron A cannot rely on always receiving a signal from neuron B because B might be dropped. The gradient update for A is computed in contexts where B is absent, forcing A to be independently useful. At inference, I call `model.eval()` to disable dropout and get consistent, deterministic predictions."

**Common traps**:
- applying dropout to small networks with high bias — reduces capacity further when the model is already underpowered; makes things worse
- forgetting `model.eval()` at inference time — active dropout halves expected neuron activity and gives stochastic, inconsistent predictions; this is a silent production bug
- using high dropout rates (> 0.5) in early layers — you risk discarding too much input signal before the network can extract anything useful from it

---

## 7. Convolutional Neural Networks

**What the interviewer is testing**: whether you understand CNNs as encoding two specific inductive biases — locality and translation equivariance — that are appropriate for spatial data.

**The reasoning structure**: two assumptions are built into the CNN architecture by design:

1. **Locality**: useful features in images are local — an edge, a texture, a corner can be detected from a small region, not the entire image.
2. **Translation equivariance**: if a cat is in the top-left corner vs the bottom-right corner, the same feature detector should fire; you should not need separate detectors for each position.

Convolutions implement both: a small filter (locality) slides across the entire image (translation equivariance). Parameter sharing means the same filter is applied at every position — this is why CNNs have orders of magnitude fewer parameters than a fully connected network.

**The pattern in action**: "A fully connected network applied to a 224×224×3 image with one 1024-neuron hidden layer has 224×224×3×1024 ≈ 154 million parameters in just the first layer. A CNN with a 3×3 convolutional layer with 64 filters has 3×3×3×64 = 1,728 parameters in the first layer — 90,000× fewer — and can detect edges everywhere in the image, not just in the position it happened to see during training."

**Common traps**:
- applying CNNs to non-spatial data without justification — CNNs encode a locality assumption; tabular data has no spatial structure and this inductive bias is wrong; the spatial prior is a feature for images, a liability for arbitrary feature vectors
- forgetting that pooling layers sacrifice translation equivariance for approximate translation invariance — max pooling over a 2×2 region makes the network slightly insensitive to small position shifts, but at the cost of spatial precision

---

## 8. RNNs, LSTMs, and GRUs

**What the interviewer is testing**: whether you understand the specific failure of vanilla RNNs (vanishing gradients over time) and how gated units address it architecturally.

**The reasoning structure**: vanilla RNNs process sequences one step at a time. The hidden state $h_t = f(W_h h_{t-1} + W_x x_t)$ carries information across timesteps. But the gradient of $h_1$ with respect to $h_t$ is a product of $t-1$ copies of $W_h$ — exactly the vanishing/exploding gradient problem applied across time rather than depth.

LSTMs solve this with a cell state $c_t$ that flows through time with only additive updates by default. Gates control what to add, what to forget, and what to expose. The cell state path is the key — gradients can flow backward through it without being multiplied by many Jacobians. The forget gate can simply set its value to 1 to pass gradients through unchanged.

GRUs simplify to two gates (reset and update), with similar gradient flow benefits at lower computational cost.

**The pattern in action**: "I classify sentiment of product reviews, some of which are 500 words long. A vanilla RNN loses information about the beginning of the review by the time it reaches the end — the gradient signal from the final timestep barely reaches the early timesteps. An LSTM can carry sentiment-relevant information in its cell state across all 500 steps, and the forget gate can decide which information to retain or discard at each step."

**Common traps**:
- using LSTMs for most sequence tasks in 2025 when Transformers are available — for language tasks, attention-based models outperform RNNs, especially on long sequences, and are easily parallelizable; RNNs are inherently sequential
- assuming LSTMs eliminate vanishing gradients entirely — they mitigate but do not eliminate the problem; for very long sequences (> 1000 steps), LSTMs still struggle; attention handles arbitrary-length context more reliably

---

## 9. Transformers and Self-Attention

**What the interviewer is testing**: whether you understand self-attention as a specific mechanism for computing context-dependent representations, not just "Transformers are better."

**The reasoning structure**: RNNs process sequences step by step and must compress all past context into a fixed-size hidden state. For long sequences, early information gets lost. Self-attention solves this differently: for each position, compute a weighted combination of all other positions' representations. The weights (attention scores) are determined by the content of the representations themselves, not their position.

In Transformer self-attention, every token plays three roles simultaneously: a query (what am I looking for?), a key (what do I have to offer?), and a value (what information do I provide?). The attention score between query $q_i$ and key $k_j$ is $\frac{q_i \cdot k_j}{\sqrt{d_k}}$. The output for position $i$ is the attention-weighted sum of all values.

Why this is powerful: every token can directly attend to every other token in a single layer — no distance penalty for long-range dependencies. The model is also fully parallelizable unlike RNNs, where step $t$ depends on step $t-1$.

**The pattern in action**: "In 'The trophy didn't fit in the suitcase because it was too big,' the word 'it' is ambiguous. A sequential model must carry the ambiguity across many steps. Self-attention lets 'it' simultaneously attend to both 'trophy' and 'suitcase' and compute a representation that resolves the coreference based on the full sentence context — in a single forward pass."

**Common traps**:
- thinking Transformers are always better than CNNs for images — Vision Transformers work well at large scale but require significantly more data than CNNs to match performance; CNNs' locality inductive bias is a real advantage on smaller image datasets
- not recognizing the quadratic cost of attention: $O(n^2)$ in sequence length, because every token attends to every other token — this is the bottleneck for long-context modeling and the primary motivation for efficient attention variants (sparse attention, linear attention, sliding window attention)

**Scaled dot-product attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## 10. Generative vs Discriminative Models

**What the interviewer is testing**: whether you understand the modeling objective difference and what it implies about what each model type learns.

**The reasoning structure**: discriminative models learn $P(y|x)$ — the conditional distribution of label given input. They answer: "given this input, what is the most likely class?" They need not model the input distribution $P(x)$ at all and can focus all capacity on the decision boundary.

Generative models learn the joint distribution $P(x, y)$ or the marginal $P(x)$. They model the data-generating process and can generate new samples, compute likelihoods, and perform anomaly detection by identifying low-probability inputs.

The practical tradeoff: discriminative models typically achieve higher classification accuracy because they focus entirely on the decision boundary. Generative models are more flexible but require more parameters and data to model the full input distribution.

**The pattern in action**: "I need to classify emails as spam AND generate synthetic spam for testing. Logistic regression gives better classification accuracy. A VAE or language model lets me generate realistic spam but may have lower classification accuracy. For the dual requirement, a generative model applied discriminatively via Bayes' rule is one option."

**Common traps**:
- treating all generative models as equivalent — GANs, VAEs, diffusion models, and autoregressive models make very different tradeoffs in generation quality, diversity, training stability, and computational cost; they are not interchangeable
- not recognizing that large language models are generative models used discriminatively — GPT predicts $P(\text{next token}|\text{context})$; using it for classification is a downstream adaptation of a generative objective

---

## 11. Autoencoders and VAEs

**What the interviewer is testing**: whether you understand what problem each solves and why VAEs are strictly more capable for generation.

**The reasoning structure**: a standard autoencoder encodes input into a compressed latent code and decodes it back. It learns a compressed representation. The problem for generation: the latent space has "holes" — regions that decode to garbage because no training point mapped there. You cannot sample new latent codes reliably.

A VAE forces the encoder to produce a distribution over latent codes (mean and variance), not a single point. The decoder is trained to reconstruct from samples drawn from this distribution. The KL divergence term pushes all distributions toward a standard Gaussian, which fills the latent space smoothly and makes sampling tractable: sample from $\mathcal{N}(0, I)$ and the decoder produces a coherent output.

**The pattern in action**: "I train an autoencoder on handwritten digits. If I encode a '3' and a '7' and try to decode a linear interpolation between their latent codes, I may get garbage halfway through — the interpolated code was never seen during training. With a VAE, the regularized latent space ensures smooth interpolation: both latent codes are in a region that decodes coherently, and the interpolation produces plausible intermediate digit images."

**Common traps**:
- using autoencoders for generation without acknowledging the latent space discontinuity problem — standard autoencoders are for compression and denoising, not generation; the latent space is not designed to be smooth
- mishandling the reconstruction vs KL tradeoff in VAEs — too much weight on KL and the model ignores the input entirely (posterior collapse); too little and the latent space becomes as discontinuous as a standard autoencoder

---

## 12. GANs

**What the interviewer is testing**: whether you understand GANs as a min-max game with specific failure modes, and why training is inherently unstable.

**The reasoning structure**: a GAN trains two networks adversarially. The generator $G$ produces fake samples from random noise. The discriminator $D$ tries to distinguish real from fake. $G$'s objective is to fool $D$; $D$'s objective is not to be fooled. At Nash equilibrium, $G$ produces perfect samples and $D$ cannot do better than random guessing.

Training is unstable because it is a two-player game, not a single objective minimization. Gradient descent does not converge to Nash equilibria in general — it can orbit around them or diverge. Mode collapse is the pathological failure: the generator finds a small subset of outputs that successfully fool the discriminator and produces nothing else, ignoring the full diversity of the real data distribution.

**The pattern in action**: "My GAN for face generation produces increasingly realistic samples until iteration 10,000, when all generated faces become slight variations of the same person — mode collapse. The generator found one mode that the discriminator cannot distinguish from real, and stopped exploring. Fixes include spectral normalization, gradient penalties, or architectures like StyleGAN that explicitly promote diversity."

**Common traps**:
- treating GAN training as standard supervised learning — GANs require careful monitoring of both generator and discriminator loss; watching only one is misleading
- not measuring generation quality appropriately — FID and IS are standard but imperfect proxies; human evaluation is often necessary for final quality assessment
- defaulting to GANs when diffusion models are available and quality matters — diffusion models are now generally preferred for image synthesis due to greater training stability and better quality

---

## 13. Transfer Learning and Fine-Tuning

**What the interviewer is testing**: whether you understand when pretraining helps and what the right fine-tuning strategy is.

**The reasoning structure**: training a large model from scratch requires enormous data and compute. Pretrained models encode general-purpose representations from diverse data. Transfer learning reuses these representations because features useful for one task (detecting edges, parsing grammar) are often useful for related tasks.

Fine-tuning adapts the pretrained model to your task. The two extremes: feature extraction (freeze all pretrained weights, train only the final layer) and full fine-tuning (update all weights). Feature extraction works when your data is similar to pretraining data and the task is simple. Full fine-tuning works when you have sufficient labeled data and the task requires the representations to meaningfully adapt.

**The pattern in action**: "I have 500 labeled medical images and a pretrained ResNet-50 from ImageNet. Training from scratch with 500 images will severely overfit. Freezing all layers and training only the classification head gives poor performance because ImageNet representations are not specialized for medical images. The sweet spot: freeze early layers (which learned universal features like edges and textures) and fine-tune later layers (which learned ImageNet-specific concepts that need to adapt to medical context)."

**Common traps**:
- using too-large a learning rate when fine-tuning — the pretrained weights encode useful representations; a large learning rate destroys them rapidly; use ~10–100× smaller than typical pretraining learning rates
- not using a pretrained model when one is available — "I'll train from scratch to understand the problem better" is almost always the wrong choice when pretrained models exist; the data and compute cost is enormous with no benefit
- forgetting that the inference pipeline must match the pretrained model's expected inputs — same normalization statistics, same input size, same tokenizer; mismatches cause silent performance degradation

---

## 14. Training vs Inference Mode

**What the interviewer is testing**: whether you understand the behavioral differences between these modes and the production implications of getting it wrong.

**The reasoning structure**: training and inference differ in more than computational cost. Several model behaviors are explicitly different by design:

- **Dropout**: active during training (randomly drops neurons), disabled during inference (all neurons active with scaled weights)
- **Batch Normalization**: uses batch statistics during training, uses running statistics accumulated across all training batches during inference
- **Gradient computation**: required during training, unnecessary during inference (disable with `torch.no_grad()` to save memory)

Forgetting to switch modes is a class of production bugs specific to deep learning that can be invisible in local testing (where batch sizes are large) but catastrophic in production (where you may process one example at a time).

**The pattern in action**: "My model achieves 91% accuracy in offline evaluation and 73% in production. I check the production code and find I forgot to call `model.eval()` before serving. Dropout is still active — each prediction uses a random subset of neurons. The model is behaving as a high-variance random ensemble at inference time rather than a deterministic trained model."

**Common traps**:
- using `model.train()` mode during inference — always call `model.eval()` before any evaluation or serving
- not using `torch.no_grad()` during inference — unnecessary gradient computation wastes memory proportional to the depth of the model; on large models this can cause out-of-memory errors during serving
- ignoring precision differences between training and inference — FP32 training with INT8 inference can introduce precision errors; always validate the production inference pipeline against the training evaluation pipeline

---

## 15. Exploding and Vanishing Gradients: Practical Fixes

**What the interviewer is testing**: whether you know the diagnostic signals and the specific fixes for each problem, and critically — whether you know that these require different solutions.

**The reasoning structure**: the two problems have different causes and different fixes. Getting them confused and applying the wrong fix is a common mistake.

Vanishing gradients: gradients become negligibly small in early layers; those layers barely learn; training loss plateaus early; more training does not help. Cause: saturating activations (sigmoid, tanh) or deep architectures without residual connections. Fixes: ReLU-family activations, residual connections, LayerNorm/BatchNorm, better weight initialization (Xavier/He).

Exploding gradients: gradients grow exponentially; parameters update wildly; loss spikes or becomes NaN. Cause: large weight matrices, long sequences in RNNs. Fix: gradient clipping — scale down the gradient vector if its norm exceeds a threshold.

**The pattern in action**: "My training loss is stable for 100 steps then suddenly goes NaN. I add gradient norm logging and see norms jump from ~1.0 to 10,000 right before the spike. Exploding gradients. I add gradient clipping with `max_norm=1.0` and training stabilizes.

Second scenario: my 50-layer network's first 10 layers show near-zero gradient norms throughout all of training. Vanishing gradients. I add residual connections and the early layers start receiving usable gradient signal."

**Common traps**:
- applying gradient clipping to a vanishing gradient problem — clipping restricts gradient magnitude; it cannot increase gradient magnitude for signals that are already near zero; you need an architectural fix
- using a fixed `max_norm` without monitoring whether clipping is actually firing — if gradients never reach `max_norm`, clipping is doing nothing; if they always hit `max_norm`, your learning rate is likely too high

---

## 16. Residual Connections

**What the interviewer is testing**: whether you understand why residual connections were transformative for deep network training, and why depth was useless (or harmful) before them.

**The reasoning structure**: before ResNets, adding more than ~20 layers often hurt performance — not due to overfitting, but due to optimization failure. Gradients vanished over 20+ layers and early layers barely trained.

Residual connections add the input of a layer directly to its output: $y = F(x) + x$. The gradient through a residual connection is: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(1 + \frac{\partial F}{\partial x}\right)$. Even if $\frac{\partial F}{\partial x}$ vanishes, the gradient still has a direct path back through the identity term (+1). This "gradient highway" makes training 100+ layer networks feasible.

A secondary benefit: residual networks learn residual functions $F(x) = \text{output} - x$ rather than direct mappings. Learning a small correction to an existing good representation is easier than learning the complete mapping from scratch.

**The pattern in action**: "ResNet-152 outperforms ResNet-34 on ImageNet, whereas a plain 152-layer network degrades compared to a plain 34-layer network. The residual connections are the only architectural difference — they are what makes additional depth actually beneficial rather than harmful."

**Common traps**:
- thinking residual connections are only for ResNets — skip connections appear in U-Net (semantic segmentation), DenseNet, every Transformer block (residual around attention and FFN), and virtually every modern deep architecture
- not matching dimensions when adding $x$ to $F(x)$ — if $F(x)$ changes the dimensionality, you need a projection (usually a 1×1 convolution or a linear layer) on the skip connection

---

## 17. Parameter Sharing

**What the interviewer is testing**: whether you understand parameter sharing as an inductive bias that encodes a structural assumption about the problem.

**The reasoning structure**: sharing parameters is equivalent to asserting that the same computation is useful in multiple places. For CNNs: an edge detector is useful at every spatial location, so share filter weights across positions. For RNNs: the same transition dynamics apply at each timestep, so share weights across time. For Transformers: the same attention mechanism applies at every sequence position.

Parameter sharing reduces the number of learnable parameters (regularization) and allows the model to generalize across the shared axis — a CNN that learned to detect edges at one location automatically detects them everywhere else, without needing to see training examples at each position.

**Common traps**:
- applying parameter sharing when the assumption is wrong — if features at different positions genuinely have different statistics (e.g., the first token in a sentence behaves very differently from middle tokens), position-specific parameters may be warranted
- confusing weight sharing with weight tying — in transformer language models, the input embedding matrix and output projection matrix are often tied (same weights); this is a specific design choice that reduces parameters and improves performance, separate from the general parameter sharing in convolutions

---

## 18. Normalization: Batch, Layer, Group, Instance

**What the interviewer is testing**: whether you can select the appropriate normalization based on architecture constraints and batch size.

**The reasoning structure**: all normalization methods normalize activations to zero mean and unit variance, then apply learnable scale $\gamma$ and shift $\beta$. They differ in which dimensions are averaged over — and that determines compatibility with different architectures and batch sizes.

- **BatchNorm**: normalizes across the batch dimension. Requires large batches (≥ 8–16) for stable statistics. Does not work for variable-length sequences or single-example inference.
- **LayerNorm**: normalizes across the feature dimension per example, independent of batch size. Works for any batch size including 1. Standard in Transformers.
- **GroupNorm**: normalizes across groups of channels per example. Works with small batches. Standard in object detection where large images force small batch sizes.
- **InstanceNorm**: normalizes across spatial dimensions per channel per example. Used in style transfer, where you want to normalize away content while preserving style structure.

**The pattern in action**: "I train object detection with batch size 2. BatchNorm with batch size 2 is unstable — the batch statistics from 2 images are too noisy. I switch to GroupNorm and training stabilizes. For a language model Transformer, I use LayerNorm because sequences have variable length and BatchNorm would require fixed-length padding or more complex handling."

**Common traps**:
- using BatchNorm in a Transformer — virtually no modern Transformer uses BatchNorm; LayerNorm is the universal choice and is better suited to sequential, variable-length data
- forgetting that Pre-LN (normalization before attention/FFN) is more stable to train than Post-LN (the original Transformer configuration) — Post-LN requires more careful learning rate warmup to avoid instability; most modern implementations use Pre-LN by default
