# Deep Learning

---

# Q1: What are neural networks?

## 1. 🔹 Direct Answer
**Neural networks** are **composable** layers of **affine transforms + nonlinearities** that **approximate** functions from data. **Universal approximators** with sufficient width/depth.

## 2. 🔹 Intuition
Stacked **learned** feature transforms—**hierarchical** patterns.

## 3. 🔹 Deep Dive
**Parameters** θ optimized by **gradient-based** methods on **loss**.

## 4. 🔹 Practical Perspective
**Data** and **compute** hungry; **regularization** essential.

## 5. 🔹 Code Snippet
```python
import torch.nn as nn
nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs kernel methods? **A:** NNs learn representations; kernels fixed.

## 7. 🔹 Common Mistakes
Assuming depth always helps without data.

## 8. 🔹 Comparison / Connections
GLMs as one-layer nets.

## 9. 🔹 One-line Revision
Neural nets learn hierarchical nonlinear mappings via stacked layers and gradients.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: Explain the Feedforward Neural Network.

## 1. 🔹 Direct Answer
**FFN** flows **acyclic** from input→output—no **recurrence**. Each layer **h^{l} = σ(W^l h^{l-1} + b^l)**.

## 2. 🔹 Intuition
Information moves **one way**—used in **MLPs**, **CNN** backbones before heads.

## 3. 🔹 Deep Dive
**Inference** is deterministic forward pass.

## 4. 🔹 Practical Perspective
**Batch** processing on GPU.

## 5. 🔹 Code Snippet
```text
x -> L1 -> relu -> L2 -> ... -> logits
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs RNN? **A:** No temporal state in FFN alone—needs sequence models for order.

## 7. 🔹 Common Mistakes
Confusing **feedforward** with **forward pass** terminology only.

## 8. 🔹 Comparison / Connections
DAG computation graphs.

## 9. 🔹 One-line Revision
Feedforward nets are acyclic layer stacks—basis for most deep architectures’ core.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q3: What are forward propagation and backward propagation?

## 1. 🔹 Direct Answer
**Forward**: compute **outputs** and **loss** from inputs. **Backward**: apply **chain rule** to propagate **∂L/∂** activations and **weights** (**backprop**).

## 2. 🔹 Intuition
Forward = **predict**; backward = **blame** assignment to parameters.

## 3. 🔹 Deep Dive
**Computational graph** with **automatic differentiation**.

## 4. 🔹 Practical Perspective
**Mixed precision**, **gradient checkpointing** trade memory.

## 5. 🔹 Code Snippet
```python
loss.backward()  # PyTorch
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Vanishing? **A:** See activations, residuals, norms.

## 7. 🔹 Common Mistakes
Thinking backward pass is **O(1)**—same order as forward typically.

## 8. 🔹 Comparison / Connections
Adjoint methods, manual derivatives.

## 9. 🔹 One-line Revision
Forward computes loss; backward propagates gradients via chain rule—autodiff automates.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What is backpropagation?

## 1. 🔹 Direct Answer
**Backpropagation** efficiently computes **gradients** ∂L/∂w using **chain rule** on the **computational graph**—**reverse-mode** autodiff. **Core** of deep learning training.

## 2. 🔹 Intuition
**Credit assignment** through layers.

## 3. 🔹 Deep Dive
**Topological** order reverse traversal; **reuse** stored activations.

## 4. 🔹 Practical Perspective
Frameworks hide complexity—**know** **vanishing/exploding** mitigations.

## 5. 🔹 Code Snippet
```text
∂L/∂w = ∂L/∂h * ∂h/∂w (chain)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Second-order? **A:** Hessian expensive—approximations (K-FAC, SAM).

## 7. 🔹 Common Mistakes
Confusing with **optimizer** (SGD vs Adam).

## 8. 🔹 Comparison / Connections
Forward-mode AD for Jacobian-vector products.

## 9. 🔹 One-line Revision
Backprop is reverse-mode differentiation computing ∂L/∂weights efficiently.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Name and explain hyperparameters for training neural networks.

## 1. 🔹 Direct Answer
**Architecture**: depth, width, activation. **Optimization**: **learning rate**, batch size, **momentum**, **weight decay**, **scheduler**. **Regularization**: **dropout**, **label smoothing**. **Training length**: epochs, **early stopping**.

## 2. 🔹 Intuition
**Knobs** not learned by gradient on minibatch objective—**search** space.

## 3. 🔹 Deep Dive
**Batch size** interacts with **LR** (linear scaling rule caution).

## 4. 🔹 Practical Perspective
**Log** scale search for LR; **one-cycle** policies.

## 5. 🔹 Code Snippet
```python
optimizer = AdamW(lr=3e-4, weight_decay=0.01)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Batch norm as hyper? **A:** Momentum in BN running stats—small effect.

## 7. 🔹 Common Mistakes
Tuning **everything** at once—**ablation** order matters.

## 8. 🔹 Comparison / Connections
AutoML, HPO.

## 9. 🔹 One-line Revision
Key NN hyperparameters: LR, batch, wd, depth/width, dropout, schedule—tune with validation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What is the advantage of deep learning over traditional machine learning?

## 1. 🔹 Direct Answer
**Automatic feature learning** from raw inputs (images, text), **scalable** with **data/compute**, **state-of-art** on **perception** and **sequence** tasks. **Flexible** function approximators.

## 2. 🔹 Intuition
**Hand features** bottleneck removed—**representations** emerge.

## 3. 🔹 Deep Dive
**Inductive biases** via architecture (conv, attention).

## 4. 🔹 Practical Perspective
**Cost**: data, GPU, **debugging** harder than linear models.

## 5. 🔹 Code Snippet
```text
end-to-end vs feature engineering pipeline
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When traditional ML? **A:** Small tabular, interpretability, fast iteration.

## 7. 🔹 Common Mistakes
DL for **100-row** tables without strong regularization.

## 8. 🔹 Comparison / Connections
Kernel methods, boosting on tabular.

## 9. 🔹 One-line Revision
DL learns representations and scales with data—best for unstructured data when compute exists.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What are activation functions, and why are they used?

## 1. 🔹 Direct Answer
**Nonlinearities** between linear layers—**without** them, deep stack **collapses** to single linear map. **Introduce** **expressivity** and **sparsity** (ReLU).

## 2. 🔹 Intuition
**Break** linearity so network can **curve** decision boundaries.

## 3. 🔹 Deep Dive
**Universal approximation** needs nonlinearity.

## 4. 🔹 Practical Perspective
**ReLU/GELU** default in Transformers; **sigmoid** at binary output.

## 5. 🔹 Code Snippet
```python
nn.ReLU(); nn.GELU()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Dead ReLU? **A:** Neurons stuck at 0—Leaky/GELU mitigate.

## 7. 🔹 Common Mistakes
Linear activation in hidden layers by mistake.

## 8. 🔹 Comparison / Connections
See Q8 for specific activations.

## 9. 🔹 One-line Revision
Activations inject nonlinearity—essential depth; ReLU family common in hidden layers.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q8: Sigmoid, Tanh, ReLU, LeakyReLU, Softmax—pros and cons.

## 1. 🔹 Direct Answer
**Sigmoid**: (0,1) **squash**—**vanishing** gradients in deep nets; good **binary** output. **Tanh**: zero-centered, still saturates. **ReLU**: fast, sparse, **dying** neurons. **LeakyReLU**: small slope <0 fixes dying. **Softmax**: **multiclass** probs—**not** typically hidden activation.

## 2. 🔹 Intuition
Saturation **kills** gradients; ReLU **linear** in positive region.

## 3. 🔹 Deep Dive
**GELU/Swish** smooth ReLU variants in Transformers.

## 4. 🔹 Practical Perspective
**Output layer** choice matches loss (sigmoid+BCE, softmax+CE).

## 5. 🔹 Code Snippet
```python
nn.Sigmoid(); nn.Tanh(); nn.ReLU(); nn.LeakyReLU(0.01); nn.Softmax(dim=-1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Softmax stability? **A:** Subtract max before exp.

## 7. 🔹 Common Mistakes
Softmax in **hidden** layers unnecessarily.

## 8. 🔹 Comparison / Connections
Mish, ELU.

## 9. 🔹 One-line Revision
ReLU family for hidden layers; sigmoid/softmax for outputs; avoid saturated sigmoid/tanh deep stacks.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: Why are Sigmoid and Tanh not preferred in hidden layers?

## 1. 🔹 Direct Answer
**Saturation** near 0/1 or ±1 → **tiny derivatives** → **vanishing gradients** in deep networks—**slow** learning. **ReLU** avoids full saturation on positive side (**faster**).

## 2. 🔹 Intuition
Flat regions mean **no error signal** flows.

## 3. 🔹 Deep Dive
**Xavier** init helped but **ReLU** + **He** init standard now.

## 4. 🔹 Practical Perspective
**GELU** in Transformers—not sigmoid.

## 5. 🔹 Code Snippet
```text
d sigmoid/dx = σ(1-σ) small when σ≈0 or 1
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Still use sigmoid where? **A:** Binary output, gates in LSTM.

## 7. 🔹 Common Mistakes
Banning sigmoid entirely—output layers OK.

## 8. 🔹 Comparison / Connections
Batch norm interaction historically.

## 9. 🔹 One-line Revision
Sigmoid/tanh saturate and starve deep networks of gradients—ReLU/GELU preferred hidden.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What is dropout, and why is it effective?

## 1. 🔹 Direct Answer
**Random** neuron removal during training—**prevents co-adaptation**, acts as **ensemble** of subnetworks, **strong regularizer**.

## 2. 🔹 Intuition
Can’t rely on **one** feature detector—**redundancy**.

## 3. 🔹 Deep Dive
**Inverted dropout** scales during training so inference **unchanged**.

## 4. 🔹 Practical Perspective
Typical **0.1–0.5**; **attention dropout** optional.

## 5. 🔹 Code Snippet
```python
nn.Dropout(0.3)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Test time? **A:** `eval()` disables.

## 7. 🔹 Common Mistakes
Dropout on **small** data without enough width.

## 8. 🔹 Comparison / Connections
Stochastic depth, L2.

## 9. 🔹 One-line Revision
Dropout regularizes by training random subnetwork ensembles—disable at inference.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: Effect of dropout on training and inference speed.

## 1. 🔹 Direct Answer
**Training**: slight **overhead** from masking; **inference**: **faster** (no mask) and **deterministic**. **Throughput** impact usually **small** vs conv/matmul.

## 2. 🔹 Intuition
Less compute when dropping? Implementation often **still** dense ops with zeroing.

## 3. 🔹 Deep Dive
**Inference** can **fuse** layers without dropout branches.

## 4. 🔹 Practical Perspective
**MC dropout** intentionally runs multiple forward passes—**slower** for uncertainty.

## 5. 🔹 Code Snippet
```python
model.train()  # dropout on
model.eval()   # dropout off
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Speed vs batch norm? **A:** BN adds stats—different trade-off.

## 7. 🔹 Common Mistakes
Leaving **train** mode in production—wrong behavior.

## 8. 🔹 Comparison / Connections
Batch norm train/eval shift.

## 9. 🔹 One-line Revision
Dropout mainly affects training stochasticity; inference drops mask for speed and determinism.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: L1/L2 regularization in neural networks.

## 1. 🔹 Direct Answer
Add **λ||w||₁** or **λ||w||₂²** to loss—**weight decay** encourages **small/sparse** weights, **reducing overfitting**. **L2** most common (**AdamW** decoupled decay).

## 2. 🔹 Intuition
Penalize **large** weights unless data demands.

## 3. 🔹 Deep Dive
**L1** can **sparsify** for interpretability/feature selection in **linear** layers.

## 4. 🔹 Practical Perspective
Tune **λ** on validation; **batch norm** params often **excluded** or treated carefully.

## 5. 🔹 Code Snippet
```python
optim.AdamW(params, weight_decay=0.01)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Implicit reg of SGD? **A:** Noise in updates—related to flat minima stories.

## 7. 🔹 Common Mistakes
Confusing **L2 reg** with **L2 weight norm** clipping.

## 8. 🔹 Comparison / Connections
Dropout, early stopping.

## 9. 🔹 One-line Revision
L2 weight decay shrinks weights; L1 can sparsify—standard explicit regularization for NNs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: What is batch normalization, and why is it used?

## 1. 🔹 Direct Answer
**BatchNorm** normalizes **activations** per feature across **batch** (then **scale/shift** **γ,β**)—**stabilizes** training, allows **higher LR**, **reduces** internal covariate shift (informal).

## 2. 🔹 Intuition
Keep layer inputs **well-scaled** as weights change.

## 3. 🔹 Deep Dive
**Running stats** at inference; **small batch** noise as regularizer.

## 4. 🔹 Practical Perspective
**LayerNorm** in Transformers (per sample across features)—different axis.

## 5. 🔹 Code Snippet
```python
nn.BatchNorm2d(channels)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Where place BN? **A:** After conv, before ReLU (debated order).

## 7. 🔹 Common Mistakes
BN on **batch_size=1** training issues.

## 8. 🔹 Comparison / Connections
GroupNorm, InstanceNorm.

## 9. 🔹 One-line Revision
BatchNorm standardizes activations across batch for stable deep CNN training—use LayerNorm in Transformers.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: Batch normalization hyperparameters to optimize.

## 1. 🔹 Direct Answer
**Momentum** for running mean/variance, **ε** stability, **affine** **γ,β** learned. Usually **defaults** OK—**tune** rarely vs **LR/architecture**. **Batch size** affects BN noise—large batch can reduce regularization effect.

## 2. 🔹 Intuition
Most impact from **batch size** and **training mode** correctness.

## 3. 🔹 Deep Dive
**SyncBatchNorm** across GPUs for consistency.

## 4. 🔹 Practical Perspective
If **small batch**, consider **GroupNorm**.

## 5. 🔹 Code Snippet
```python
nn.BatchNorm2d(64, momentum=0.1, eps=1e-5)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Freeze BN in transfer? **A:** Sometimes for small data—careful with running stats.

## 7. 🔹 Common Mistakes
**Eval** mode with **bad** running stats from mismatched batch stats.

## 8. 🔹 Comparison / Connections
Weight standardization.

## 9. 🔹 One-line Revision
BN has momentum/eps; practical tuning focuses on batch size and sync—not usually primary HPO axis.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: What is parameter sharing in deep learning?

## 1. 🔹 Direct Answer
**Same weights** applied across **positions**—**CNNs** share filters across spatial locations; **RNNs** share across **time**; **massive** **parameter reduction** and **translation equivariance** (CNN).

## 2. 🔹 Intuition
Reuse pattern detectors everywhere—don’t learn separate edge detector per pixel.

## 3. 🔹 Deep Dive
**Tied weights** in some autoencoder decoders.

## 4. 🔹 Practical Perspective
**Data efficiency** vs fully connected on images.

## 5. 🔹 Code Snippet
```python
nn.Conv2d(in_ch, out_ch, kernel_size=3)  # one filter reused
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs attention all unique? **A:** Attention weights **dynamic** per pair—different inductive bias.

## 7. 🔹 Common Mistakes
Confusing **weight tying** with **parameter freezing**.

## 8. 🔹 Comparison / Connections
Equivariance, symmetries.

## 9. 🔹 One-line Revision
Parameter sharing reuses weights across space/time—core to CNNs and RNNs for efficiency and structure.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q16: What is representation learning, and why is it useful?

## 1. 🔹 Direct Answer
Learn **features** **automatically** from data (embeddings, CNN feature maps)—**reuse** across tasks (**transfer learning**), **compress** raw inputs to **semantic** space.

## 2. 🔹 Intuition
**Raw pixels** too high-dim—learn **manifold**.

## 3. 🔹 Deep Dive
**Self-supervised** pretraining builds **general** representations.

## 4. 🔹 Practical Perspective
**Fine-tune** smaller labeled sets.

## 5. 🔹 Code Snippet
```text
encoder: x -> z compact
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Disentanglement? **A:** Separate latent factors—β-VAE etc.

## 7. 🔹 Common Mistakes
Assuming learned features always **interpretable**.

## 8. 🔹 Comparison / Connections
Manifold hypothesis.

## 9. 🔹 One-line Revision
Representation learning maps raw data to useful latent features—enables transfer and downstream efficiency.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: Generative vs discriminative models.

## 1. 🔹 Direct Answer
**Discriminative**: model **P(y|x)** directly (classifiers). **Generative**: model **P(x|y)** or **P(x)** joint—can **generate** samples, handle **missing** data, but **density** estimation harder.

## 2. 🔹 Intuition
Generative **explains** how data produced; discriminative **draws boundaries**.

## 3. 🔹 Deep Dive
**Naive Bayes** generative; **logistic** discriminative.

## 4. 🔹 Practical Perspective
**GANs/VAE/Diffusion** generative; **BERT classifier head** discriminative.

## 5. 🔹 Code Snippet
```text
Gen: maximize log p(x) ; Disc: maximize log p(y|x)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which for imbalanced? **A:** Either with care—generative can synthesize data.

## 7. 🔹 Common Mistakes
Thinking generative always better for classification—often discriminative wins with same data.

## 8. 🔹 Comparison / Connections
Energy-based models.

## 9. 🔹 One-line Revision
Discriminative models boundaries; generative models data distributions—trade-offs in flexibility and training difficulty.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: How does a generative model work?

## 1. 🔹 Direct Answer
Learns **distribution** over data **Pθ(x)** (or conditional)—**sample** by **running** **forward** process: **VAE** sample **z** then **decode**; **GAN** **noise→generator**; **diffusion** **iterative denoise**.

## 2. 🔹 Intuition
**Simulate** the **data factory** that produced training points.

## 3. 🔹 Deep Dive
**Training** via **MLE**, **adversarial** objective, or **ELBO**.

## 4. 🔹 Practical Perspective
**Evaluation**: FID, Inception score, **human** eval.

## 5. 🔹 Code Snippet
```text
z ~ N(0,I); x = G(z)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Tractable density? **A:** Flows, autoregressive models.

## 7. 🔹 Common Mistakes
Confusing **implicit** GAN generator with **explicit** density.

## 8. 🔹 Comparison / Connections
Normalizing flows.

## 9. 🔹 One-line Revision
Generative models learn to sample from learned data distribution—objectives vary by family.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: Encoder–Decoder Architecture.

## 1. 🔹 Direct Answer
**Encoder** maps input to **latent** **z** (compression); **Decoder** maps **z** back to **output** domain—**bottleneck** forces **representation**. Used in **seq2seq**, **VAE**, **UNet** (skip connections).

## 2. 🔹 Intuition
**Compress** then **expand**—like zip/unzip with learning.

## 3. 🔹 Deep Dive
**Skip** connections in U-Net preserve **detail**.

## 4. 🔹 Practical Perspective
**seq2seq** for translation; **ASR** encoder-decoder.

## 5. 🔹 Code Snippet
```text
z = Enc(x); y_hat = Dec(z)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Info bottleneck? **A:** Trade compression vs reconstruction quality.

## 7. 🔹 Common Mistakes
Confusing with **autoencoder** only—broader pattern.

## 8. 🔹 Comparison / Connections
T5 text-to-text.

## 9. 🔹 One-line Revision
Encoder-decoder learns compact latents then reconstructs or generates outputs—foundation for seq2seq and VAEs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q20: What is latent space?

## 1. 🔹 Direct Answer
**Latent space** is **lower-dimensional** **z** capturing **factors of variation**—**smooth** interpolation often corresponds to **semantic** changes (in idealized generative models).

## 2. 🔹 Intuition
**Hidden** coordinates describing data—not directly observed.

## 3. 🔹 Deep Dive
**VAE** prior **N(0,I)**; **GAN** latent **noise** vector.

## 4. 🔹 Practical Perspective
**Walk** latent vectors to **edit** attributes (with disentanglement).

## 5. 🔹 Code Snippet
```text
z in R^k, k << input_dim
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Manifold learning? **A:** Latent space approximates data manifold.

## 7. 🔹 Common Mistakes
Assuming **linear** semantics always hold.

## 8. 🔹 Comparison / Connections
Embeddings in metric learning.

## 9. 🔹 One-line Revision
Latent space is low-d hidden representation where similar points often mean similar data—key in generative models.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21–Q22: Autoencoders and VAE.

## Q21 Autoencoders
**Reconstruction** **x̂ ≈ x** with bottleneck **z = Enc(x)**, **x̂ = Dec(z)**—**denoise**, **compression**, **pretraining**. **Layers**: encoder downsampling, decoder upsampling.

## Q22 VAE
**Probabilistic** encoder **q(z|x)**, decoder **p(x|z)**—optimize **ELBO** = reconstruction + **KL** to prior—**generates** by **sampling z**.

### One-line Revision
Autoencoders compress-reconstruct; VAE adds stochastic latents and KL for generative probabilistic model.

### Difficulty Tag
🟡 Medium

---

# Q23: VAE probabilistic latent structure—why important?

## 1. 🔹 Direct Answer
**Regularizes** latent space to **match prior**—**smooth**, **interpolatable**; **enables** **sampling** **new** points. **Without** stochasticity, holes in latent space break generation.

## 2. 🔹 Intuition
Force **organized** **Gaussian-like** cloud rather than **arbitrary** twisting.

## 3. 🔹 Deep Dive
**Reparameterization** trick **z = μ + σε** for backprop through **stochastic** node.

## 4. 🔹 Practical Perspective
**β-VAE** trades reconstruction vs disentanglement.

## 5. 🔹 Code Snippet
```text
L = E_q[log p(x|z)] - KL(q(z|x)||p(z))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Posterior collapse? **A:** Decoder ignores z—mitigate with annealing, architecture.

## 7. 🔹 Common Mistakes
Confusing VAE **loss** with plain MSE autoencoder.

## 8. 🔹 Comparison / Connections
Wake-sleep, diffusion as alternative generative path.

## 9. 🔹 One-line Revision
VAE’s stochastic latents + KL align approximate posterior to prior for tractable sampling and smooth space.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q24–Q27: GANs (architecture, roles, mode collapse, applications)

## Q24 Architecture
**Generator G(z)** maps noise to data; **Discriminator D(x)** scores real vs fake—**minimax** game.

## Q25 Roles
**G** fools **D**; **D** distinguishes—**adversarial** signal trains **G**.

## Q26 Mode collapse
**G** outputs **few** modes—**diversity** loss, **unrolled** GAN, **WGAN-GP**, **minibatch** discrimination.

## Q27 Applications
**Image synthesis**, **super-resolution**, **style transfer**, **data augmentation**.

### One-line Revision
GANs adversarially train generator and discriminator—watch mode collapse and training instability.

### Difficulty Tag
🟣 Hard

---

# Q28–Q33: CNNs (overview, filters, stride, padding, pooling, FC layers)

## Q28 CNN
**Convolutional** layers apply **shared filters** over spatial input—**local connectivity**, **translation equivariance**, **hierarchical** features.

## Q29 Filters
**Learned** **kernels** detecting edges/textures—**depth** = number of filters.

## Q30 Stride
**Step size** of filter—**downsamples** when &gt;1, **reduces** resolution.

## Q31 Padding
**Same** padding keeps spatial size; **valid** shrinks—**control** **border** effects.

## Q32 Pooling
**Max/Avg** pooling **downsamples**, adds **local translation invariance**—less common now (strided conv preferred).

## Q33 FC in CNN
**Flattens** feature maps to **vector** for **classification** logits—**global average pooling** reduces params.

### Code
```python
nn.Conv2d(3, 64, 3, stride=1, padding=1)
nn.MaxPool2d(2)
```

### One-line Revision
CNNs use shared filters, stride/padding control shape, pooling/strides reduce spatial size—FC head for classification.

### Difficulty Tag
🟡 Medium

---

# Q34–Q38: RNNs, limitations, LSTM/GRU, gates, exploding gradients

## Q34 RNN
**Hidden state h_t** updated **recursively** **h_t = f(h_{t-1}, x_t)**—sequence modeling.

## Q35 Limitations
**Vanishing/exploding** gradients, **slow** parallelization, **long-range** deps hard—**LSTM/GRU/Attention** address.

## Q36 LSTM/GRU
**Gated** **memory** cells—**forget/input/output** gates (LSTM), **update/reset** (GRU)—**better** long deps.

## Q37 LSTM gates
**Forget** what to erase from cell; **input** what to write; **output** what to expose—**cell state** highway.

## Q38 Exploding gradients
**Clip** gradients **norm**; **proper init**; **residual** RNN variants.

### One-line Revision
RNNs model sequences with recurrence; LSTM/GRU gates stabilize memory; gradient clipping handles explosion.

### Difficulty Tag
🟣 Hard

---

# Q39–Q43: Transformers vs CNN/RNN, Attention, LSTM vs Transformer, Diffusion vs AR, Transfer learning

## Q39 Transformer vs CNN/RNN
**Global attention** in **O(1)** depth connections vs **local** conv or **sequential** RNN—**parallel** training, **long-range** deps, **quadratic** cost.

## Q40 Attention significance
**Dynamic**, **content-based** mixing—**SOTA** language and increasingly vision (ViT).

## Q41 LSTM vs Transformer
**LSTM** sequential, **O(T)** steps; **Transformer** parallel, **O(T²)** attention—Transformers **scale** better with compute.

## Q42 Diffusion vs Autoregressive
**Diffusion** iterative **denoise** **parallelizable** training; **AR** **token-by-token**—diffusion **quality** images; **AR** dominant text historically.

## Q43 Transfer learning
**Pretrain** encoder on **large** task, **fine-tune** on **small** target—**faster**, **better** than scratch.

### One-line Revision
Transformers replaced RNNs at scale; diffusion excels at image gen quality; transfer learning reuses pretrained representations.

### Difficulty Tag
🟣 Hard

---
