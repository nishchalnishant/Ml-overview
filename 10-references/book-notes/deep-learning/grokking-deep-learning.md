# Grokking Deep Learning

## Chapter 1: The Automation of Intelligence

**The problem the book is addressing**
Deep learning is often taught through APIs — call `model.fit()` and it works. When it doesn't work, practitioners have no model for why. The goal is to build intuition from scratch so that debugging and design feel natural rather than magical.

**The core insight**
Deep learning is pattern recognition learned from data. Rather than specifying rules explicitly, you specify a network architecture and loss, then optimization discovers the rules. The book's approach — implement everything from NumPy — forces understanding of the mechanics at every level.

**The mechanics**
- Automation of intelligence = learning input-output mappings from labeled examples
- The three components: network (transforms input to prediction), loss (measures error), optimizer (updates network to reduce loss)
- Every deep learning system reduces to these three components, regardless of complexity

**What the book gets right / what to watch out for**
Building from NumPy builds real intuition. The tradeoff is that the implementations are pedagogical, not production-quality — they're 10–100× slower than PyTorch and miss important numerical stability tricks. Use this book to understand why things work; use PyTorch for everything else.

---

## Chapter 2: Supervised vs Unsupervised Learning, Parametric vs Nonparametric

**The problem the book is addressing**
Practitioners often reach for the most complex model available. Understanding the fundamental distinction between learning paradigms and model families helps select the right tool for the problem rather than defaulting to the most fashionable approach.

**The core insight**
Supervised learning requires labels. Unsupervised learning finds structure in inputs alone. Parametric models compress training data into fixed parameters — fast at inference, but limited by their functional form. Nonparametric models retain training data and grow with dataset size — flexible but slow for large datasets.

**The mechanics**
- Supervised: learn f: X → Y from labeled pairs {(xᵢ, yᵢ)}; neural networks are parametric
- Unsupervised: learn structure in {xᵢ} without labels; clustering, autoencoders, language models
- Parametric: fixed number of parameters regardless of training set size; neural networks, linear models
- Nonparametric: parameters grow with data; k-nearest neighbors, kernel methods, Gaussian processes

**What the book gets right / what to watch out for**
The parametric/nonparametric distinction is useful but breaks down for modern LLMs — a 70B-parameter model is technically parametric but effectively memorizes training data in a distributed way. Unsupervised learning has largely been displaced by self-supervised learning (masked language modeling, contrastive learning) which uses unlabeled data to construct supervision signals.

---

## Chapter 3: Forward Propagation — Dot Products and Single/Multiple Neurons

**The problem the book is addressing**
A neuron as a biological metaphor is misleading. The mathematical operation — a weighted sum of inputs followed by a nonlinearity — is what matters. Understanding this from first principles explains why neural networks can represent complex functions.

**The core insight**
A single neuron computes a dot product of input and weights, adds a bias, and applies an activation: y = activation(w·x + b). Multiple neurons in a layer compute multiple such dot products simultaneously — equivalent to a matrix multiplication: Y = activation(XW + b). Stacking layers composes these transformations.

**The mechanics**
- Single neuron: y = σ(wᵀx + b); w is (d,), b is scalar
- Layer of n neurons: Y = activation(XW + b); X is (batch, d_in), W is (d_in, d_out), b is (d_out,)
- Forward pass: repeatedly apply layer transformations; final layer produces prediction
- NumPy: `np.dot(X, W) + b` is the core operation

**What the book gets right / what to watch out for**
The dot product framing is the right mental model — once you see matrix multiplication as batched dot products, architecture design becomes geometric intuition. The biological neuron analogy should be abandoned — artificial neurons are linear combiners with a fixed nonlinearity, not the complex electrochemical systems in brains.

---

## Chapter 4: Gradient Descent — MSE, Hot/Cold Learning, Predict/Compare/Learn

**The problem the book is addressing**
A neural network initialized randomly makes poor predictions. You need a principled method to adjust parameters in the direction that reduces prediction error. Understanding gradient descent from first principles is necessary before understanding backpropagation.

**The core insight**
The learning loop is: predict (forward pass), compare (compute loss), learn (update parameters to reduce loss). For a single parameter, the derivative of the loss with respect to that parameter tells you which direction to move it and how much. Gradient descent follows the steepest descent direction.

**The mechanics**
- MSE loss: L = (1/n) Σ(ŷᵢ - yᵢ)²; smooth, differentiable, squared penalizes large errors more
- Hot/cold learning (conceptual): nudge parameter up and down; keep the direction that reduces loss — expensive but intuitive
- Analytical gradient: ∂L/∂w = (2/n) Σ(ŷᵢ - yᵢ) · ∂ŷ/∂w; exact, free (same cost as forward pass)
- Update: w ← w - η · ∂L/∂w; η (learning rate) controls step size
- Learning rate too high: overshoots, oscillates; too low: slow convergence

**What the book gets right / what to watch out for**
The hot/cold analogy is pedagogically effective for building intuition before introducing derivatives. MSE is not the right loss for classification — use cross-entropy. Learning rate is the most important hyperparameter in practice — start with 3e-4 (Adam) or 0.1 (SGD) and tune from there.

---

## Chapter 5: Gradient Descent with Multiple Inputs and Outputs

**The problem the book is addressing**
Single-input gradient descent generalizes to multiple inputs and outputs, but the bookkeeping is non-trivial. Understanding how weight deltas are computed for every parameter in a layer prepares the foundation for backpropagation.

**The core insight**
For multiple inputs, each weight's gradient is the product of the output error signal and the input that weight processed. Weight deltas form a matrix: ΔW = ηxᵀδ, where x is the input vector and δ is the output error. This outer product structure is the core of the backpropagation weight update.

**The mechanics**
- Multiple inputs, single output: ∂L/∂wᵢ = (ŷ-y) · xᵢ; gradient is error weighted by input
- Multiple outputs: ΔW is an outer product; ΔWᵢⱼ = δⱼ · xᵢ; δ is output error vector
- Matrix form: ΔW = xᵀδ (outer product); shapes: (d_in,) × (d_out,) → (d_in, d_out)
- Weight sensitivity: inputs with large magnitude produce large weight deltas — normalize inputs to prevent instability

**What the book gets right / what to watch out for**
The outer product form of the weight update is the correct mental model that unifies single-neuron and multi-layer backpropagation. Large input values producing large gradients is the motivation for input normalization — always normalize features to zero mean unit variance before training.

---

## Chapter 6: Backpropagation — Error Attribution Across Layers

**The problem the book is addressing**
The streetlight problem: how do you assign credit/blame to parameters in early layers when you can only observe error at the output? This is the core challenge backpropagation solves. Without it, only the output layer would learn.

**The core insight**
Error attribution works by propagating the output error backwards through the network using the chain rule. Each layer's error signal is the downstream error multiplied by the layer's Jacobian (local derivative). The same weight that contributed to a prediction contributes proportionally to that prediction's error signal.

**The mechanics**
- Forward pass: store intermediate activations (needed for backward pass)
- Output error: δ_out = ŷ - y (for MSE); δ_out = ŷ - y_onehot (for softmax + cross-entropy)
- Hidden layer error: δ_h = (Wᵀ δ_out) ⊙ σ'(z_h); propagate error through weights, apply activation derivative
- Weight update: ΔW = η · aᵢₙᵀ δ_out; where aᵢₙ is input activations to the layer
- SGD vs mini-batch vs batch: SGD updates after each example (noisy, fast); mini-batch averages gradient over B examples (standard); batch uses full dataset (slow, exact)

**What the book gets right / what to watch out for**
The streetlight problem framing is one of the clearest motivations for backprop in any textbook. Storing all intermediate activations for the backward pass is why memory grows linearly with network depth — gradient checkpointing trades compute for memory by recomputing activations rather than storing them.

---

## Chapter 7: Visualizing Neural Networks — Layer Notation and Algebraic Expressions

**The problem the book is addressing**
Neural network diagrams can look like complex graphs that are hard to map to code. A clear notation system — with consistent layer labels, shape tracking, and algebraic expressions — helps practitioners read papers, debug shapes, and translate architectures to code.

**The core insight**
A network can be described fully by the sequence of matrix multiplications and activation functions. Layer dimensions tell you parameter count, memory requirements, and where shape mismatches occur. Reading a network algebraically — tracking (batch, features) shapes at each layer — is more useful than the neuron diagram.

**The mechanics**
- Letter notation: x = input, h = hidden, ŷ = output, W = weights, b = bias, σ = activation
- Layer: h = σ(Wₓh x + bₓh); subscripts indicate from/to layer
- Shape tracking: (batch, d_in) @ (d_in, d_out) → (batch, d_out)
- Parameter count per layer: d_in × d_out + d_out (weights + biases)
- Total parameters: Σ over all layers; useful for sanity checking architecture size

**What the book gets right / what to watch out for**
Shape tracking is the most practically useful habit in deep learning — the majority of runtime bugs are shape mismatches. The habit of annotating each tensor's shape at each step catches errors before running expensive training. In PyTorch, `tensor.shape` and `print(model)` are the primary debugging tools.

---

## Chapter 8: Regularization and Batching

**The problem the book is addressing**
A network with millions of parameters trained on thousands of examples will memorize the training data. It achieves high training accuracy but poor test accuracy. You need techniques that prevent memorization while allowing the model to learn genuine patterns.

**The core insight**
Overfitting happens when the model has higher capacity than needed for the genuine signal. The three main interventions: early stopping (stop before the model memorizes), dropout (randomly disable neurons to prevent co-adaptation), mini-batch training (gradient noise acts as implicit regularization).

**The mechanics**
- Early stopping: track validation loss; stop when it increases for N consecutive epochs; save checkpoint at best validation performance
- Dropout: Bernoulli mask M ~ Bernoulli(1-p); h_dropped = h ⊙ M / (1-p); applied during training, disabled at inference
- p=0.5 for FC layers; p=0.1–0.2 for conv layers; inverted dropout (divide by 1-p) keeps expected activations consistent between train and test
- Mini-batch GD: gradient over B examples; B=32–256 typical; noise in gradient prevents overfitting to individual examples; enables GPU parallelism

**What the book gets right / what to watch out for**
The Bernoulli mask explanation of dropout is exact and correct. Forgetting to disable dropout at inference (`model.eval()`) is a common bug that causes predictions to be stochastic and inconsistent. In modern deep networks, early stopping and weight decay are often sufficient — dropout is less commonly used in transformer architectures.

---

## Chapter 9: Activation Functions — Sigmoid, Tanh, Softmax, Output Layer Choices

**The problem the book is addressing**
The output layer activation determines what the model can predict. Using the wrong activation (sigmoid for multiclass, softmax for binary) produces incorrect probability estimates that mislead the loss function and metrics.

**The core insight**
The output activation and loss function must match the task: sigmoid + BCE for binary classification, softmax + cross-entropy for multiclass, linear for regression. Hidden layer activations should be non-saturating (ReLU and variants) to prevent vanishing gradients.

**The mechanics**
- Sigmoid: σ(z) = 1/(1+e^-z) ∈ (0,1); output layer for binary probability; saturates → vanishing gradients in hidden layers
- Tanh: tanh(z) = (e^z-e^-z)/(e^z+e^-z) ∈ (-1,1); zero-centered (better than sigmoid for hidden layers), still saturates
- Softmax: ŷₖ = exp(oₖ) / Σⱼ exp(oⱼ); converts logits to probability distribution over K classes; sum = 1
- ReLU: max(0,z); non-saturating; default for hidden layers; dead neurons possible if LR too high
- Output layer choices: binary → sigmoid; multiclass → softmax; regression → linear (no activation); multilabel → independent sigmoids

**What the book gets right / what to watch out for**
The output-activation-to-task mapping is the most important table in this chapter. GELU (smooth ReLU approximation) is now preferred over ReLU for transformer architectures. Numerical stability: compute softmax in log-space (`log_softmax`) to avoid exp overflow; PyTorch's `nn.CrossEntropyLoss` takes raw logits and handles this internally.

---

## Chapter 10: CNNs — Weight Reuse, Kernels, Pooling

**The problem the book is addressing**
Images have spatial structure — nearby pixels are correlated and the same pattern (edge, texture, face) can appear at any location. A fully connected network ignores this structure, requiring millions of parameters and failing to generalize to different positions.

**The core insight**
Convolution applies the same small filter at every position (weight sharing), dramatically reducing parameters while encoding spatial locality and translation equivariance. Stacking convolutions with pooling builds a hierarchy of features: early layers detect edges, later layers detect complex shapes.

**The mechanics**
- 2D convolution: output[i,j] = Σ_{k,l} filter[k,l] × input[i+k, j+l]
- Weight sharing: same filter applied at all positions → O(k²) parameters per filter vs O(H×W) for FC
- Max pooling: takes max over 2×2 window; halves spatial dimensions; provides translation invariance
- Average pooling: averages over window; smoother downsampling
- CNN architecture: [Conv → ReLU → MaxPool] × N → Flatten → FC
- NumPy implementation: triple nested loop; educational but O(H×W×K²×C_in×C_out) — much slower than optimized implementations

**What the book gets right / what to watch out for**
The NumPy convolution implementation is excellent for understanding what's happening inside the black box. The weight-sharing insight (parameter reduction from O(H×W) to O(k²)) is what makes CNNs practical for images. Global average pooling (replacing final FC layer with GAP) is now standard in ResNet — reduces parameters and overfitting while making the model input-size agnostic.

---

## Chapter 11: NLP — Bag of Words, Word Embeddings, Analogies

**The problem the book is addressing**
Text data is symbolic — words are labels with no inherent numerical structure. One-hot encoding is valid but treats "king" and "queen" as orthogonal vectors with no relationship. Bag-of-words ignores word order and treats documents as frequency histograms.

**The core insight**
Word embeddings map each word to a dense d-dimensional vector (d=100–300 typical) in a continuous space where similar words are nearby. Skip-gram training (predict context words from a target word) discovers that semantic relationships correspond to geometric relationships: king - man + woman ≈ queen.

**The mechanics**
- Bag-of-words: document → vector of word counts; vocabulary size is vector dimension; ignores order
- TF-IDF: weight each word count by inverse document frequency; downweights common words
- Word embedding: lookup table V × d; each token ID maps to a d-dimensional vector; trained end-to-end
- Skip-gram: maximize P(context|target) over a window of ±k words
- Analogy: word_a - word_b + word_c → find nearest word in embedding space

**What the book gets right / what to watch out for**
The word embedding intuition is pedagogically important. In production, static word embeddings (word2vec, GloVe) have been superseded by contextual embeddings from BERT/GPT — the same word gets different vectors in different contexts. The analogy test is fragile — it works ~30% of the time on standard benchmarks, not reliably enough for production use.

---

## Chapter 12: RNNs — Character Language Models, Truncated Backprop

**The problem the book is addressing**
Language sequences have arbitrary length and long-range dependencies. A feedforward network requires fixed-size input and processes tokens independently. How do you build a model that reads text sequentially and maintains context across characters or words?

**The core insight**
An RNN maintains a hidden state vector that is updated at each timestep: hₜ = σ(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ). The hidden state "remembers" previous inputs. A character language model predicts the next character given all previous characters, trained on text by backpropagating through the sequence.

**The mechanics**
- RNN forward: for each timestep t: hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ); ŷₜ = softmax(Wₕᵧhₜ + bᵧ)
- Training objective: cross-entropy on next-character prediction at each step
- Truncated BPTT: backpropagate through the last T=16–64 steps only; prevents gradient explosion and memory overflow on long sequences; treats hₜ₋ₜ as a constant (detach from graph)
- Identity initialization of Wₕₕ: helps gradients flow at initialization; used by iRNN (Identity RNN)
- Transition matrices: Wₕₕ encodes the "memory" of the network; its eigenvalues determine whether gradients vanish or explode

**What the book gets right / what to watch out for**
Truncated BPTT is the correct treatment — without it, training sequences longer than ~50 timesteps becomes intractable. The identity initialization insight is a practical trick for reducing vanishing gradients without LSTMs. For most practical sequence tasks, LSTMs or transformers should be used rather than vanilla RNNs.

---

## Chapter 13: Building a Deep Learning Framework — Autograd from Scratch

**The problem the book is addressing**
Using PyTorch without understanding how autograd works makes debugging gradient flow nearly impossible. Implementing a minimal autodiff system reveals exactly what happens during `.backward()` and why `zero_grad()` is required.

**The core insight**
Automatic differentiation records operations in a computation graph (DAG). Each tensor node stores the operation that created it and references to input tensors. `.backward()` traverses this graph in reverse (topological order), applying the chain rule at each node to accumulate gradients. Gradients accumulate by default — `zero_grad()` must be called explicitly.

**The mechanics**
- Tensor class: stores data (NumPy array), grad (accumulated gradient), grad_fn (how this tensor was created)
- Computation graph: each operation creates a new tensor and records its inputs
- Backward: traverse graph in reverse topological order; at each node: grad_inputs = backward_fn(grad_output); accumulate into input grads
- SGD optimizer: loop over parameters; w.data -= lr * w.grad; w.grad = None (zero grad)
- Layer base class: holds parameters; `forward()` defines computation; backward propagates gradients

**What the book gets right / what to watch out for**
This chapter is the most important in the book — understanding the computation graph removes most gradient debugging mystery. The key insight: gradients accumulate by default because gradient accumulation (summing gradients over multiple mini-batches before stepping) is a useful feature for large batch training when GPU memory is limited. Always call `zero_grad()` unless you intentionally want to accumulate.

---

## Chapter 14: LSTMs — Gates, Cell State, Long-Range Memory

**The problem the book is addressing**
Vanilla RNNs lose information over long sequences because gradients vanish through many tanh operations. The hidden state gets overwritten at each step with no mechanism for selectively preserving or discarding information over long ranges.

**The core insight**
LSTMs add a cell state cₜ — a "conveyor belt" that runs the length of the sequence with only additive interactions (no repeated multiplications). Three gates (forget, input, output) are sigmoid units that learn to open and close like valves, controlling what the cell state remembers, what new information is added, and what is exposed as the hidden state.

**The mechanics**
- Forget gate: fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf) — decides what to erase from cell state
- Input gate: iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi); c̃ₜ = tanh(Wc·[hₜ₋₁,xₜ]+bc) — decides what new info to write
- Cell update: cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ — additive update preserves gradients
- Output gate: oₜ = σ(Wo·[hₜ₋₁,xₜ]+bo); hₜ = oₜ⊙tanh(cₜ) — exposes filtered cell state
- Truncated BPTT: still needed for LSTMs; standard range T=16–64 steps
- Text generation: sample next character from softmax distribution; feed back as next input

**What the book gets right / what to watch out for**
The additive cell update as the key to gradient preservation is the correct explanation — the highway for gradients. For sequence tasks where LSTMs currently work, transformers now match or exceed performance while parallelizing training. LSTMs retain a niche for online/streaming tasks where you can't wait for the full sequence before predicting.

---

## Chapter 15: Federated Learning — Privacy-Preserving Distributed Training

**The problem the book is addressing**
Centralizing sensitive data (medical records, personal messages) for training creates privacy risks and legal constraints. How do you train a model when data cannot leave user devices?

**The core insight**
Federated learning trains the model locally on each device and shares only parameter updates (gradients), not raw data. The server aggregates updates using federated averaging. Differential privacy and secure aggregation prevent the server from inferring individual user data from gradient updates.

**The mechanics**
- Federated averaging: server sends model; each client trains for K steps on local data; clients send gradient updates; server averages and updates global model
- Secure aggregation: clients add random noise (that cancels out in aggregate) before sending updates — server sees only the sum, not individual updates
- Differential privacy: add calibrated Gaussian noise to gradients before sending; bounds the amount of individual information that can be inferred
- Homomorphic encryption (PHE): clients encrypt gradients; server aggregates encrypted values; result decrypts to the correct aggregate — computationally expensive

**What the book gets right / what to watch out for**
Federated averaging is the correct foundation — this is what Google uses for Gboard. Secure aggregation via randomized responses is a practical approximation. Homomorphic encryption provides the strongest privacy guarantee but is 100–1000× slower than plaintext computation — practical only for small models or high-stakes applications.

---

## Chapter 16: Self-Supervised and Contrastive Learning

**The problem the book is addressing**
Labeled data is expensive to collect. Most data in the world is unlabeled. How do you train useful representations without manual labels?

**The core insight**
Self-supervised learning creates supervision from the data itself via pretext tasks (predict masked tokens, predict rotation, predict context). Contrastive learning enforces a geometric constraint: representations of augmentations of the same input should be close (positive pairs); representations of different inputs should be far (negative pairs). The resulting representations transfer to downstream tasks.

**The mechanics**
- Pretext tasks: predict randomly masked tokens (BERT), predict next sentence (NSP), predict image rotation angle, colorization, context prediction
- Triplet loss: L = max(d(anchor, positive) - d(anchor, negative) + margin, 0); anchor and positive are augmentations of same input; negative is a different input
- Contrastive loss (SimCLR): for each input, create two augmented views; maximize similarity within pairs, minimize across pairs; InfoNCE loss
- Positive pairs: different augmentations (crop, color jitter, blur) of the same image
- Negative pairs: other images in the mini-batch (easy to obtain without explicit labeling)

**What the book gets right / what to watch out for**
Self-supervised pretraining on unlabeled data followed by fine-tuning on labeled data is now the dominant paradigm for vision, NLP, and audio — the same principle BERT, GPT, and CLIP use. Triplet loss requires careful negative mining — easy negatives don't provide gradient signal; hard negatives (similar but different class) improve representations. SimCLR and BYOL have largely superseded triplet loss for image representations.
