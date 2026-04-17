# Deep Learning

The best deep learning answers are layered: definition, intuition, then the training or architecture tradeoff that actually matters in practice.

---

# Q1: What are neural networks?

**Interview-ready answer**

Neural networks are parameterized function approximators built by composing linear transformations with non-linear activations. Their power comes from representation learning: instead of hand-engineering every feature, the network learns intermediate features that make the final task easier. In practice, the strength of a neural network is not just "it is non-linear," but that deeper layers can learn increasingly abstract representations from data.

---

# Q2: Explain the Feedforward Neural Network.

**Interview-ready answer**

A feedforward neural network processes information in one direction, from input through hidden layers to output, without recurrent loops. Each layer applies an affine transformation followed by a non-linearity, allowing the model to learn increasingly rich transformations of the input. This is the basic template behind multilayer perceptrons and also a building block inside larger architectures like transformers.

---

# Q3: What are forward propagation and backward propagation?

**Interview-ready answer**

Forward propagation computes the activations, predictions, and final loss from the input. Backward propagation computes the gradients of that loss with respect to every parameter by applying the chain rule in reverse through the computation graph. The intuition is simple: the forward pass makes the prediction, and the backward pass assigns responsibility for the error.

---

# Q4: What is backpropagation?

**Interview-ready answer**

Backpropagation is the efficient algorithm that computes gradients for all model parameters in a layered network. Instead of differentiating each parameter independently from scratch, it reuses intermediate derivatives and propagates gradients backward. This is why deep networks with millions or billions of parameters can be trained at all. A strong answer should also note that backprop computes gradients; the optimizer then uses those gradients to update parameters.

---

# Q5: Name and explain hyperparameters for training neural networks.

**Interview-ready answer**

The most important training hyperparameters are learning rate, batch size, optimizer choice, weight decay, number of epochs, architecture size, dropout, normalization settings, and the learning-rate schedule. In practice, learning rate is usually the first thing I tune because it has the largest effect on whether training is stable. After that I focus on capacity and regularization. Strong candidates mention that these hyperparameters interact, especially batch size, optimizer, and learning rate.

---

# Q6: What is the advantage of deep learning over traditional machine learning?

**Interview-ready answer**

Deep learning is most valuable when the input is high-dimensional and unstructured, such as images, audio, text, or multimodal data, because it can learn hierarchical representations directly from raw or lightly processed inputs. Traditional ML often depends more heavily on manual feature engineering but can be better on small tabular data, when interpretability matters, or when compute budgets are tight. So the real advantage of deep learning is not that it is universally better; it is that it scales feature learning with data and compute.

---

# Q7: What are activation functions, and why are they used?

**Interview-ready answer**

Activation functions introduce non-linearity into the network. Without them, stacking multiple linear layers would still collapse to a single linear transformation, which would severely limit expressive power. Activations also affect optimization because their shape changes gradient flow, sparsity, and stability. That is why activation choice is about both model expressiveness and trainability.

---

# Q8: Sigmoid, Tanh, ReLU, LeakyReLU, Softmax - pros and cons.

**Interview-ready answer**

Sigmoid maps to `(0, 1)` and is useful for binary outputs, but it saturates and causes vanishing gradients in hidden layers. Tanh is zero-centered and often slightly better behaved than sigmoid, but it still saturates. ReLU is simple, efficient, and works well in hidden layers because it avoids saturation on the positive side, though it can suffer from dead neurons. LeakyReLU reduces that dead-neuron issue by allowing a small negative slope. Softmax is different from the others because it is typically an output transformation for multiclass probabilities rather than a hidden-layer activation.

---

# Q9: Why are Sigmoid and Tanh not preferred in hidden layers?

**Interview-ready answer**

The main problem is saturation. When sigmoid or tanh outputs move into their flat regions, the gradients become very small, so earlier layers learn slowly or stop learning altogether. Sigmoid also is not zero-centered, which can make optimization less efficient. That is why modern deep networks usually prefer ReLU-family activations in hidden layers.

---

# Q10: What is dropout, and why is it effective?

**Interview-ready answer**

Dropout randomly zeroes a subset of activations during training, which prevents the network from relying too heavily on specific neurons or paths. This acts as a regularizer by encouraging the model to learn more robust and distributed representations. It is effective especially when a network has enough capacity to overfit, although in some modern architectures normalization, data scale, and other regularization methods reduce the need for heavy dropout.

---

# Q11: Effect of dropout on training and inference speed.

**Interview-ready answer**

During training, dropout adds stochastic masking and usually makes optimization noisier, which can require more epochs to converge even if each individual step is not much more expensive. During inference, dropout is disabled, so prediction typically uses the full network without added randomness. The main point is that dropout is a generalization tool, not a speed optimization.

---

# Q12: L1/L2 regularization in neural networks.

**Interview-ready answer**

L1 regularization encourages sparsity by pushing some weights exactly to zero, while L2 regularization shrinks weights smoothly and discourages overly large parameters. In neural networks, L2-style weight decay is much more common because it stabilizes training and generalization without forcing hard sparsity. The strong interview nuance is that in modern optimizers, especially AdamW, decoupled weight decay is usually preferred over naive L2 added directly into the adaptive update.

---

# Q13: What is batch normalization, and why is it used?

**Interview-ready answer**

Batch normalization normalizes layer activations using batch statistics and then learns a scale and shift. It helps training by making optimization more stable, allowing higher learning rates, and often adding a mild regularizing effect. While the historical explanation was "internal covariate shift," the more practical interview answer is that BatchNorm smooths optimization and improves training dynamics.

---

# Q14: Batch normalization hyperparameters to optimize.

**Interview-ready answer**

The main BatchNorm-related knobs are whether to use it at all, where to place it relative to activation and linear layers, the momentum used for running statistics, and sometimes epsilon for numerical stability. In practice, though, the biggest interactions are indirect: BatchNorm changes the learning-rate regime, batch-size sensitivity, and the need for other regularization like dropout.

---

# Q15: What is parameter sharing in deep learning?

**Interview-ready answer**

Parameter sharing means using the same weights in multiple places rather than learning separate parameters for each position or location. This is powerful because it reduces the number of parameters and injects useful inductive bias. Convolutions share filters across spatial locations, and recurrent networks share weights across time steps. The benefit is not only efficiency; it is that the model assumes similar patterns can appear in different positions.

---

# Q16: What is representation learning, and why is it useful?

**Interview-ready answer**

Representation learning means learning intermediate features automatically from data instead of relying entirely on hand-crafted features. Good representations make downstream tasks easier by separating important factors of variation, compressing noise, and aligning similar inputs in meaningful ways. This is one of the core reasons deep learning works well on unstructured data.

---

# Q17: Generative vs discriminative models.

**Interview-ready answer**

Discriminative models learn `p(y | x)` or a direct decision boundary, while generative models try to model how the data is generated, such as `p(x)` or the joint distribution `p(x, y)`. Discriminative models are often stronger for pure prediction tasks because they focus directly on the target. Generative models are useful when you want sampling, density estimation, missing-data handling, or latent structure.

---

# Q18: How does a generative model work?

**Interview-ready answer**

A generative model tries to capture the distribution of data so it can sample new examples or reason about how likely observed examples are. Different generative models do this differently: VAEs learn latent-variable distributions, GANs learn through adversarial training, autoregressive models factorize the data distribution sequentially, and diffusion models learn to reverse a noise process. The unifying idea is that the model is learning structure in the input space itself, not only a label boundary.

---

# Q19: Encoder-Decoder Architecture.

**Interview-ready answer**

An encoder-decoder architecture separates representation building from output generation. The encoder compresses or contextualizes the input into a latent representation, and the decoder uses that representation to generate the desired output. This pattern is central to translation, summarization, image segmentation, autoencoders, and many multimodal systems because it naturally supports input-output mappings where the input and output differ in length or form.

---

# Q20: What is latent space?

**Interview-ready answer**

Latent space is the internal representation space in which a model encodes the underlying factors of variation in the data. The idea is that high-dimensional observations, like images or sentences, can often be described by a smaller set of hidden variables. A good latent space organizes similar examples nearby and separates meaningful factors so interpolation, retrieval, or generation becomes easier.

---

# Q21-Q22: Autoencoders and VAE.

**Interview-ready answer**

An autoencoder learns to compress input data into a latent representation and then reconstruct it, so it is useful for denoising, compression, and representation learning. A variational autoencoder adds a probabilistic latent structure: instead of mapping each input to a single point, it learns a distribution in latent space and regularizes that space toward a prior. That extra structure makes VAEs better suited for controlled generation and smooth latent interpolation, though reconstructions are often blurrier than those of deterministic autoencoders or GANs.

---

# Q23: VAE probabilistic latent structure - why important?

**Interview-ready answer**

The probabilistic latent structure is important because it makes the latent space continuous, regularized, and sampleable. Nearby latent points tend to decode to semantically related outputs, which is why VAEs support interpolation and generation better than plain autoencoders. The KL term in the objective is what enforces that structured latent space, trading some reconstruction sharpness for better generative behavior.

---

# Q24-Q27: GANs (architecture, roles, mode collapse, applications)

**Interview-ready answer**

GANs consist of a generator that produces synthetic samples and a discriminator that tries to distinguish real from fake samples. The two models train adversarially, so the generator improves by learning to fool the discriminator. GANs can produce very sharp and realistic outputs, which made them influential in image generation, super-resolution, style transfer, and data synthesis. Their main challenge is training instability, including mode collapse, where the generator produces only a narrow subset of outputs instead of covering the full data distribution.

**What to mention if pushed**

- The generator learns the data manifold implicitly through adversarial feedback.
- The discriminator supplies a learned loss rather than a simple per-pixel objective.
- GANs can look visually strong while still missing diversity.

---

# Q28-Q33: CNNs (overview, filters, stride, padding, pooling, FC layers)

**Interview-ready answer**

CNNs use local filters that slide across an input to detect spatial patterns such as edges, textures, and higher-level structures. Stride controls how far the filter moves each step, padding controls how borders are handled, and pooling reduces spatial resolution while retaining strong responses. Fully connected layers, when used, combine the learned spatial features into a final prediction. The reason CNNs work so well for vision is that locality and parameter sharing are strong inductive biases for images.

---

# Q34-Q38: RNNs, limitations, LSTM/GRU, gates, exploding gradients

**Interview-ready answer**

RNNs process sequences one step at a time while carrying hidden state forward, which makes them a natural way to model order and temporal dependence. Their weakness is that long-range dependencies are hard to learn because gradients can vanish or explode over many steps, and the sequential nature limits parallelism. LSTMs and GRUs address this with gating mechanisms that control what information to keep, forget, and expose. They improved sequence modeling significantly before transformers became dominant.

**Good nuance**

Exploding gradients are usually handled with gradient clipping, while vanishing gradients require architectural changes or better inductive bias.

---

# Q39-Q43: Transformers vs CNN/RNN, Attention, LSTM vs Transformer, Diffusion vs AR, Transfer learning

**Interview-ready answer**

Transformers rely on attention to model dependencies between all tokens directly, which makes them highly parallelizable and very effective for long-range context. Compared with RNNs, they train more efficiently at scale and handle long dependencies better. Compared with CNNs, they have weaker locality bias but stronger global context modeling. LSTMs remain useful when data is limited or sequence lengths are modest, but transformers dominate many large-scale text and multimodal tasks. For generation, autoregressive models produce outputs token by token, while diffusion models generate by gradually denoising and often excel in image synthesis. Transfer learning ties all of this together: modern practice is usually to start from a pretrained foundation model and adapt it rather than training from scratch.
