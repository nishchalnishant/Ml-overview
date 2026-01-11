# Deep Learning Architectures: 30+ Questions

---

## CNNs (Convolutional Neural Networks)

**1. What is a Convolution operation?**
> A filter slides over the input, computing element-wise multiplication and sum at each position. Captures local patterns.

**2. What is the purpose of Pooling?**
> Downsamples feature maps, reducing spatial dimensions. Provides translation invariance. Max pooling is most common.

**3. What is the formula for output size after convolution?**
> $O = \frac{W - K + 2P}{S} + 1$ (W=input, K=kernel, P=padding, S=stride).

**4. Why use 3x3 kernels instead of larger ones?**
> Two 3x3 layers = same receptive field as one 5x5, but fewer parameters and more non-linearities.

**5. What is Stride?**
> Step size of the filter. Stride=2 halves the output size.

**6. What is Padding?**
> Adding zeros around the input. "Same" padding preserves spatial dimensions. "Valid" = no padding.

**7. Explain AlexNet's key innovations.**
> ReLU activation, Dropout, GPU training, data augmentation. Won ImageNet 2012.

**8. What made ResNet revolutionary?**
> Skip connections: $y = F(x) + x$. Enabled training of 100+ layer networks by allowing gradient flow.

**9. What is the Bottleneck architecture in ResNet?**
> 1x1 conv (reduce channels) → 3x3 conv → 1x1 conv (expand channels). Reduces compute.

**10. What is Depthwise Separable Convolution?**
> Depthwise (one filter per channel) + Pointwise (1x1). Used in MobileNet. Much fewer parameters.

---

## RNNs & LSTMs

**11. What is the hidden state in RNN?**
> $h_t = f(W_h h_{t-1} + W_x x_t)$. Carries information across time steps.

**12. What is the Vanishing Gradient problem in RNNs?**
> Gradients shrink exponentially over long sequences, preventing learning of long-range dependencies.

**13. How does LSTM solve vanishing gradients?**
> Cell state acts as a highway. Gates control what to forget, add, and output.

**14. What are the three gates in LSTM?**
> **Forget**: what to discard. **Input**: what new info to store. **Output**: what to output.

**15. What is GRU?**
> Simplified LSTM with 2 gates (Reset, Update). Fewer parameters, similar performance.

**16. Bidirectional RNN vs Unidirectional?**
> **Bi**: Processes sequence both forward and backward. Better for tasks where full context is available.

**17. When would you use RNN over Transformer?**
> Streaming data, very long sequences that don't fit in memory, edge devices with limited compute.

---

## Transformers

**18. What is Self-Attention?**
> Each token attends to all other tokens in the sequence. Computes weighted sum based on relevance.

**19. Write the Attention formula.**
> $Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V$

**20. What are Q, K, V?**
> **Query**: What I'm looking for. **Key**: What I contain. **Value**: What I contribute.

**21. Why scale by $\sqrt{d_k}$?**
> Dot products grow with dimension. Scaling prevents softmax saturation and gradient vanishing.

**22. What is Multi-Head Attention?**
> Run attention h times with different learned projections, then concatenate. Captures different relationships.

**23. What is Positional Encoding?**
> Added to embeddings to give sequence order information. Sinusoidal or learned.

**24. What is Masked (Causal) Attention?**
> Tokens can only attend to previous tokens. Used in decoders for autoregressive generation.

**25. What is Cross-Attention?**
> Query from decoder, Key/Value from encoder. Used in encoder-decoder models (T5, translation).

**26. BERT vs GPT architecture?**
> **BERT**: Encoder-only, bidirectional. **GPT**: Decoder-only, causal.

**27. What is the computational complexity of self-attention?**
> $O(L^2 \cdot d)$ where L is sequence length. Quadratic in length.

**28. How do you handle long sequences in Transformers?**
> Truncation, sliding window, sparse attention (Longformer), hierarchical models.

---

## Generative Models

**29. How does a GAN work?**
> Generator creates fakes from noise. Discriminator distinguishes real vs fake. Both improve via adversarial training.

**30. What is the GAN loss function?**
> $\min_G \max_D E[\log D(x)] + E[\log(1-D(G(z)))]$

**31. What is Mode Collapse?**
> Generator produces limited variety. Discriminator exploits this. Fix: WGAN, diversity penalties.

**32. What is a VAE?**
> Encoder maps to latent distribution, Decoder reconstructs. Loss = Reconstruction + KL divergence.

**33. What is the Reparameterization Trick?**
> $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim N(0,1)$. Makes sampling differentiable for backprop.

**34. GAN vs VAE: when to use which?**
> **GAN**: Sharper images, harder to train. **VAE**: Stable training, blurrier outputs, good latent space.

**35. What are Diffusion Models?**
> Gradually add noise to data, learn to reverse the process. State-of-the-art for image generation.

---

## Object Detection & Segmentation

**36. Two-stage vs One-stage detectors?**
> **Two-stage**: Region proposal then classification (Faster R-CNN). More accurate. **One-stage**: Direct prediction (YOLO). Faster.

**37. What is a Region Proposal Network (RPN)?**
> Proposes candidate bounding boxes. Part of Faster R-CNN.

**38. How does YOLO work?**
> Divides image into grid. Each cell predicts bounding boxes and class probabilities. Single pass.

**39. What is Non-Maximum Suppression (NMS)?**
> Removes redundant overlapping boxes. Keeps highest confidence, removes high IoU duplicates.

**40. What is IoU (Intersection over Union)?**
> $\frac{Area of Overlap}{Area of Union}$. Measures bounding box accuracy.

**41. What is Feature Pyramid Network (FPN)?**
> Multi-scale feature maps for detecting objects of different sizes. Top-down pathway with lateral connections.

**42. What is Semantic vs Instance Segmentation?**
> **Semantic**: Classify each pixel (all cars = one class). **Instance**: Distinguish individual objects (car1, car2).

**43. What is U-Net?**
> Encoder-decoder with skip connections. Standard for medical image segmentation.

---

## Optimization & Regularization

**44. What is Weight Initialization? Why does it matter?**
> Initial weights affect gradient flow. **Xavier**: For tanh/sigmoid. **He**: For ReLU.

**45. What is Learning Rate Scheduling?**
> Decrease LR over time. Common: step decay, cosine annealing, warmup.

**46. What is Gradient Clipping?**
> Cap gradient magnitude to prevent exploding gradients. Common in RNNs.

**47. What is Label Smoothing?**
> Instead of hard labels (0,1), use soft labels (0.1, 0.9). Reduces overconfidence.

**48. What is Data Augmentation?**
> Artificially increase dataset with transformations (flips, rotations, crops). Reduces overfitting.

**49. What is Mixup?**
> Blend two images and their labels: $x' = \lambda x_i + (1-\lambda)x_j$. Regularization technique.

**50. What is the difference between BatchNorm and LayerNorm?**
> **BatchNorm**: Normalize across batch dimension. **LayerNorm**: Normalize across feature dimension. LayerNorm for Transformers.
