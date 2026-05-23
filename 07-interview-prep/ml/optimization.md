---
module: Interview Prep
topic: Ml
subtopic: Optimization
status: unread
tags: [interviewprep, ml, ml-optimization]
---
# Optimization for ML Interviews

---

## 1. Gradient Descent

**What the interviewer is testing**: whether you understand why gradient descent works at all — the geometric argument for why it reliably reduces the loss — not just the mechanics of the update rule.

**The reasoning structure**: training a model means finding parameter values that minimize the loss function. The loss surface is a high-dimensional function over millions or billions of parameters. You cannot search it exhaustively. Gradient descent works by exploiting one fact: the gradient of the loss points in the direction of steepest increase. Moving in the opposite direction (steepest decrease) reduces the loss locally.

The word "locally" is critical. The gradient is a first-order approximation — it tells you the slope at the current point, not the global structure. Gradient descent is a local search algorithm. For convex losses (logistic regression, linear regression, SVM), every local minimum is the global minimum, so this suffices — convergence to the global optimum is guaranteed given a sufficiently small learning rate. For neural networks (non-convex), gradient descent finds a local minimum. In practice for overparameterized networks, this works empirically because the loss landscape has a dense manifold of approximately equivalent global minima (see optimization-theory.md for the overparameterization argument), and SGD noise biases the search toward flat, well-generalizing solutions.

$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

The learning rate $\eta$ determines how far to move in the gradient direction. It is not "how fast the model learns" — it is the trust radius of the linear approximation. The gradient is only accurate locally; if you move too far, the actual loss might increase even though the gradient said to go down.

Backpropagation is not the same as gradient descent. Backpropagation is the algorithm that efficiently computes $\nabla_\theta L$ by applying the chain rule layer by layer. Gradient descent is the algorithm that uses $\nabla_\theta L$ to update $\theta$. They are distinct and separable.

**The pattern in action**: "Training logistic regression: the loss is convex (a sum of log-sigmoid functions), so there is one global minimum. With $\eta = 0.01$, gradient descent decreases loss monotonically and converges. With $\eta = 10$, the step overshoots the minimum and the loss oscillates or diverges. The correct $\eta$ is below $1/L$ where $L$ is the Lipschitz constant of the gradient (the smoothness of the loss surface). For neural networks, non-convexity means gradient descent might stop at a local minimum. For a 100-layer transformer with millions of parameters, the landscape is so high-dimensional that gradient descent reliably finds a solution good enough for the task — the theory of why is the overparameterization + implicit regularization story."

**Common traps**:
- Conflating gradient descent (the update rule) with backpropagation (the gradient computation algorithm). Backpropagation computes $\nabla_\theta L$; gradient descent uses it. A model trained with automatic differentiation (PyTorch autograd) uses backprop for gradient computation, but could use any optimizer (SGD, Adam, L-BFGS) for the update step.
- Treating gradient descent as an exact optimization algorithm. The update $\theta - \eta \nabla_\theta L$ exactly minimizes the linearized loss around $\theta$. But the actual loss is not linear, so the true minimum is not at $\theta - \eta \nabla_\theta L$ except infinitesimally. Each step is an approximation.

---

## 2. Batch Size and Gradient Quality

**What the interviewer is testing**: whether you understand the variance-compute tradeoff in gradient estimation, and the counterintuitive finding that more accurate gradients can produce worse generalization.

**The reasoning structure**: the true gradient is $\nabla_\theta L(\theta) = \frac{1}{n}\sum_{i=1}^n \nabla_\theta \ell_i(\theta)$, computed over all $n$ training examples. For large $n$, this is prohibitively expensive. Stochastic gradient descent approximates the true gradient using a random minibatch of size $m$:

$$g_B = \frac{1}{m}\sum_{i \in B} \nabla_\theta \ell_i(\theta)$$

This estimator is unbiased: $\mathbb{E}[g_B] = \nabla_\theta L(\theta)$. Its variance is $\sigma^2/m$ — larger batches give lower-variance estimates. This is the variance-compute tradeoff: larger batches give more accurate gradient directions but cost more compute per step.

There is a second, less obvious tradeoff: generalization. Keskar et al. (2017) showed empirically that large-batch training consistently converges to sharp minima (high $\lambda_{\max}$ of the Hessian at the solution), which generalize worse than flat minima. Small batches have high gradient noise. This noise acts as an implicit perturbation — the optimizer can only converge to a minimum that remains stable despite the noise, which means a flat minimum. Large batches have low noise, so gradient descent follows the steepest path into the nearest minimum, which tends to be the sharp one. Accurate gradients are not always beneficial for generalization.

| Variant | Batch size | Gradient variance | Generalization tendency |
| :--- | :--- | :--- | :--- |
| Full batch GD | All $n$ | Exact | Sharp minima |
| SGD | 1 | Very high | Strongest flat-minimum bias |
| Mini-batch SGD | 32–512 | Moderate | Standard choice |

**The pattern in action**: "I increase batch size from 64 to 4096 to take advantage of 16 GPUs. Training is ~20x faster per epoch. But validation accuracy drops 1.5% compared to small-batch training. The large-batch model converged to a sharper minimum. I partially compensate with linear learning rate scaling ($\eta_{\text{new}} = \eta_{\text{old}} \times 4096/64$) and linear warmup (Goyal et al., 2017). The accuracy gap closes to 0.6%. Residual gap: I add SAM (sharpness-aware minimization) which explicitly biases the optimizer toward flat minima regardless of batch size. Gap closes to 0.2%."

**Common traps**:
- Assuming larger batches are strictly better because they estimate the gradient more accurately. Gradient accuracy and solution quality are not the same thing. The gradient noise in small-batch training is not a bug — it is an implicit regularizer.
- Not scaling the learning rate when increasing batch size. The standard heuristic: multiply learning rate by the factor by which you scale the batch size. (With warmup: first warm up from the original LR to the scaled LR over ~5 epochs, then apply the scheduled LR.) Without this scaling, large-batch training undershoots the optimal LR for the landscape.

---

## 3. Learning Rate

**What the interviewer is testing**: whether you understand learning rate as the control parameter for optimization stability — the thing that determines whether you converge, oscillate, or diverge — not just "how fast the model learns."

**The reasoning structure**: the learning rate $\eta$ scales the gradient step. Its consequences come from the relationship between the step size and the curvature of the loss surface. In a direction with curvature $\kappa$ (second derivative), the optimal step size is $1/\kappa$ — larger than that overshoots, smaller makes slow progress. Different directions of the loss surface have different curvatures, and a single scalar $\eta$ is always a compromise across all of them.

Too large: the update overshoots the minimum in high-curvature directions. The loss oscillates — goes down in some directions but up in the high-curvature directions — or diverges. The critical learning rate for convergence is $2/\lambda_{\max}(H)$ where $\lambda_{\max}$ is the largest eigenvalue of the Hessian (maximum curvature). Exceeding this causes divergence.

Too small: you are making progress, but the ratio of step size to distance to minimum is so small that convergence takes impractically many steps.

The loss landscape's curvature varies through training: early on you are far from any minimum, moving quickly through relatively flat regions; later you are in a curved valley near a minimum. A fixed $\eta$ is optimal for neither phase. Learning rate scheduling addresses this.

**The pattern in action**: "I start with $\eta = 0.1$: loss oscillates and training is unstable. At $\eta = 0.01$: loss decreases but slowly, and after 100 epochs the model is still not converged. At $\eta = 0.03$: rapid decrease, no oscillation, converges in 50 epochs. The 'right' learning rate is just below the instability threshold. I find it by the LR range test: sweep $\eta$ from $10^{-7}$ to 1 over 300 steps and plot loss vs. LR. Choose the LR in the steep-decrease region, just before it starts increasing."

**Practical starting points**: AdamW at $3 \times 10^{-4}$ (Karpathy constant — works reliably for many architectures). SGD + momentum at 0.01–0.1. When fine-tuning pretrained models, use discriminative learning rates: 10–100x lower LR for pretrained layers than newly added heads.

**Common traps**:
- Setting learning rate once and leaving it constant. Learning rate scheduling — warmup + cosine decay — typically delivers more benefit than any other single hyperparameter choice. The specific LR value matters less than the schedule shape.
- Using the same learning rate for all layers when fine-tuning. The first few layers of a pretrained network have learned generic low-level representations; they need minimal updating. Later layers need more adaptation. Using a single LR either destroys the generic early-layer representations or fails to adapt the task-specific later layers.

---

## 4. Momentum

**What the interviewer is testing**: whether you understand momentum as exponential smoothing of gradient history and can derive from that description why it helps in narrow valleys and what its failure modes are.

**The reasoning structure**: plain gradient descent has two problems in narrow loss valleys — regions where the loss is curved steeply in one direction and gently in another. Gradients in the narrow direction alternate sign (zigzag), making small progress toward the minimum along the valley floor. Momentum addresses this by accumulating gradient history:

$$v_t = \beta v_{t-1} + (1-\beta) g_t, \quad \theta_t = \theta_{t-1} - \eta v_t$$

With $\beta = 0.9$, the velocity $v_t$ is an exponentially weighted average of the last ~10 gradient steps ($1/(1-\beta) = 10$). Gradients that alternate sign (oscillating across the narrow valley) cancel in the average. Gradients that consistently point the same direction (along the valley floor) accumulate, producing larger updates in that direction. The result: faster progress along the valley, reduced oscillation across it.

Nesterov momentum computes the gradient at the anticipated next position rather than the current position:

$$v_t = \beta v_{t-1} + \eta \nabla_\theta L(\theta_{t-1} - \beta v_{t-1})$$

This "look-ahead" makes Nesterov slightly more responsive — if the gradient at the next position says to slow down, Nesterov corrects before overshooting, while standard momentum corrects only after. For convex problems, Nesterov has provably better convergence rates.

**The pattern in action**: "Training a linear regression on poorly-scaled features (one feature spans [0, 1000], another spans [0, 1]). Without momentum, SGD oscillates violently in the dimension of the large-scale feature (the gradient is large, the update overshoots, the gradient reverses, overshoots the other way...) while making barely any progress in the small-scale feature dimension. The loss decreases painfully slowly. With momentum $\beta = 0.9$, the oscillations average out. Progress along the small-scale dimension accumulates. Convergence is ~5x faster. The real fix is to scale features, but momentum masks the problem and works."

**Common traps**:
- Setting $\beta$ too high (e.g., 0.99 or higher). The effective smoothing window is $1/(1-\beta)$ steps. At $\beta = 0.99$, the window is 100 steps — the velocity accumulates so much history that it overshoots minima and causes instability. Standard is $\beta = 0.9$ for SGD + momentum.
- Not resetting the velocity when making large changes to the learning rate or task. The accumulated velocity encodes gradient information from the old learning rate or task. After a major LR reduction or a task switch in continual learning, reset momentum to zero.

---

## 5. Adam

**What the interviewer is testing**: whether you understand what specific problem Adam solves that SGD + momentum does not, and what the bias correction mechanism is and why it is necessary.

**The reasoning structure**: different parameters in a neural network receive gradients of vastly different scales. Parameters in frequently-activated paths (dense layers in later layers, attention projection matrices) receive large gradients. Parameters in rarely-activated paths (sparse embeddings, lower layers in deep networks) receive small gradients. A single global learning rate forces a tradeoff: large enough to update the rarely-updated parameters, it will cause destructively large updates for the frequently-updated ones.

Adam (Adaptive Moment Estimation) maintains per-parameter estimates of gradient scale:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment: gradient moving average)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment: gradient scale estimate)}$$

The effective learning rate for parameter $i$ is $\eta / \sqrt{\hat{v}_i + \epsilon}$. When the gradient for parameter $i$ is consistently large, $\hat{v}_i$ is large and the effective LR is small. When the gradient is consistently small, $\hat{v}_i$ is small and the effective LR is large. Adam automatically applies smaller updates where gradients are large and larger updates where gradients are small — independent of the global learning rate.

**Bias correction**: at $t = 0$, $m_0 = 0$ and $v_0 = 0$. After one step, $m_1 = (1-\beta_1) g_1 = 0.1 g_1$. The estimate is only 10% of the actual gradient — severely underestimated. Without correction, the first ~10 steps would have tiny effective learning rates regardless of gradient magnitude. The bias correction divides by the cumulative weight:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

After step 1: $\hat{m}_1 = (1-\beta_1)g_1 / (1-\beta_1) = g_1$. After step 1000 with $\beta_1 = 0.9$: $1 - 0.9^{1000} \approx 1$, so the correction is negligible. Bias correction only matters in the first ~$1/(1-\beta)$ steps — thereafter the exponential averages stabilize and the denominator $\to 1$.

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3 \times 10^{-4}$.

**Common traps**:
- Using `torch.optim.Adam` with weight decay for deep learning. Standard Adam applies weight decay incorrectly — the weight decay term is adaptively scaled, so the regularization is weaker for high-gradient parameters. Use AdamW (see section 6).
- Assuming Adam always converges faster than SGD. Adam converges in fewer steps for most problems and requires less hyperparameter tuning. But for vision models where SGD is carefully tuned (cosine schedule, right LR), SGD often achieves better final accuracy because its gradient noise produces flatter, more generalizable minima. Adam's adaptivity reduces noise and can cause convergence to sharper solutions.

---

## 6. AdamW — Why Adam Gets Weight Decay Wrong

**What the interviewer is testing**: whether you understand the subtle but consequential bug in standard Adam's weight decay implementation — specifically why coupling weight decay to the adaptive scaling changes its character — and why AdamW fixes it.

**The reasoning structure**: weight decay (L2 regularization) is supposed to apply a uniform shrinkage to all parameters: at each step, reduce every parameter by a fraction $\lambda\theta$. This implements Bayesian regularization with a Gaussian prior on weights and is intended to prevent any weight from growing excessively large.

The conventional implementation adds the regularization term to the gradient before the optimizer step: $g \leftarrow g + \lambda\theta$, then applies the optimizer to the modified gradient. In SGD, this works correctly: $\theta \leftarrow \theta - \eta(g + \lambda\theta) = \theta(1 - \eta\lambda) - \eta g$. The weight shrinkage is $\eta\lambda$ for every parameter, uniformly.

In Adam, the modified gradient $(g + \lambda\theta)$ is passed through the adaptive scaling. The update becomes:

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t + \lambda\theta_{t-1}}{\sqrt{\hat{v}_t} + \epsilon}$$

The weight decay term $\lambda\theta$ is divided by $\sqrt{\hat{v}_t}$. For parameters with large gradient history ($\hat{v}_t$ large), the effective weight decay is $\lambda\theta / \sqrt{\hat{v}_t}$ — much weaker than $\lambda\theta$. For parameters with small gradient history, the effective weight decay is stronger. Weight decay is supposed to be uniform; in Adam it is inversely proportional to the historical gradient scale. This corrupts the regularization semantics.

AdamW decouples weight decay from the gradient update:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

The weight decay term $\lambda\theta_{t-1}$ is subtracted independently of the adaptive scaling. Every parameter is shrunk by the same absolute rate $\eta\lambda$ per step, regardless of its gradient history. This is the correct L2 regularization behavior.

**The pattern in action**: "I train a Transformer language model with `torch.optim.Adam(weight_decay=0.01)`. The attention projection matrices receive large gradients; the embedding layers receive smaller gradients. The effective weight decay on attention matrices is ~0.001 (diluted by the large adaptive scaling); the effective weight decay on embeddings is ~0.02 (amplified by the small adaptive scaling). The regularization is strongest on the parameters I trust most (those with large gradient signal) and weakest on the parameters most in need of regularization (those with small gradient signal). Switching to `torch.optim.AdamW(weight_decay=0.01)` gives uniform 1% shrinkage per step across all parameters."

**Common traps**:
- Using `torch.optim.Adam` for any modern deep learning task. AdamW should be the default. Standard Adam with weight_decay is misleading because it does not implement L2 regularization correctly.
- Setting `weight_decay > 0.1`. AdamW with weight_decay=0.3 combined with dropout is excessive regularization that hurts performance. Typical effective range: 0.01–0.1 for most tasks. Start at 0.01.

---

## 7. Vanishing and Exploding Gradients

**What the interviewer is testing**: whether you can diagnose each problem from training symptoms and select the appropriate targeted fix — understanding that the two problems have different causes and different solutions.

**The reasoning structure**: backpropagation computes gradients by applying the chain rule layer by layer, multiplying Jacobians from the output back to the input. In a network with $L$ layers, the gradient at the first layer involves a product of $L-1$ matrices:

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial \theta_1}$$

Each Jacobian $\frac{\partial h_{k+1}}{\partial h_k}$ has singular values. If typical singular values are < 1, the product's magnitude decays exponentially with $L$ — vanishing gradients. If typical singular values are > 1, the product's magnitude grows exponentially — exploding gradients. Both are exponential in network depth.

**Vanishing gradient symptoms**: early layers have near-zero gradient norms (monitor with `torch.nn.utils.clip_grad_norm_` or per-layer gradient logging). Training loss plateaus early. Performance does not improve with more training time. Later layers keep training; earlier layers are effectively frozen.

**Exploding gradient symptoms**: loss is NaN or Inf. Training is unstable — loss spikes suddenly. Gradient norm monitoring shows values of 100, 1000, growing unboundedly. Weights eventually become NaN.

**Fixes for vanishing gradients**: residual connections (create additive bypasses that carry gradient unchanged through the shortcut path, avoiding the multiplicative chain); ReLU/GELU activations (non-saturating in the positive half, gradient = 1 for positive inputs unlike sigmoid's maximum gradient of 0.25); LayerNorm/BatchNorm (normalize activations into a range where gradients flow); Xavier/He initialization (set initial weight scales to preserve gradient variance through the network at initialization).

**Fixes for exploding gradients**: gradient clipping — scale down the gradient vector if its global norm exceeds threshold $\tau$:
$$g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|_2}\right)$$
Clip by norm (not by value) — clipping by value changes the update direction; clipping by norm preserves direction and only reduces magnitude.

**The pattern in action**: "My LSTM sequence model produces NaN loss after exactly 200 training steps. I add gradient norm logging: norms are 1.2, 1.5, 2.0, 5.0, 50, 500, NaN over the first 200 steps. This is exploding gradients — the LSTM's recurrent weight matrix has eigenvalues slightly above 1, and the repeated matrix multiplications through 50 time steps amplify the gradient exponentially. I add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between the backward pass and optimizer step. Training stabilizes. Norms stay around 0.8–1.2, occasionally clipping at 1.0."

**Common traps**:
- Applying gradient clipping to a vanishing gradient problem. Clipping restricts gradient magnitude downward — it cannot increase magnitude. A vanishing gradient needs architectural solutions: residual connections, better initialization, normalization, non-saturating activations.
- Clipping by value instead of by norm. Clipping $g_1 = 1000$ and $g_2 = 0.001$ to $c = 1$ produces $(1, 0.001)$, which points almost entirely in the $g_1$ direction — the gradient direction has changed from (1, 0) to approximately (1, 0). Clipping by norm instead scales both components proportionally, preserving the direction exactly.

---

## 8. Learning Rate Scheduling

**What the interviewer is testing**: whether you can derive why a fixed learning rate is suboptimal from the structure of the optimization problem, and mechanistically explain why warmup is necessary for Adam specifically.

**The reasoning structure**: the optimal learning rate changes through training because the optimization landscape changes. Early in training, you are far from any minimum — the loss surface is roughly smooth and a large step makes fast progress. Late in training, you are near a minimum — the landscape is curved and a large step overshoots, causing oscillation. A fixed learning rate is a compromise that is too large late in training and too small early.

**Why Adam needs warmup specifically**: Adam's second moment estimate $v_t$ starts at zero and accumulates with $\beta_2 = 0.999$. After $t$ steps, the effective weight on the true second moment is $1 - \beta_2^t$. After step 1: weight = 0.001 — essentially only a single gradient's worth of information. After step 100: weight ≈ 0.095 — still very unstable. After step 1000: weight ≈ 0.63 — reasonable. The bias correction $\hat{v}_t = v_t / (1-\beta_2^t)$ adjusts for this, but it amplifies noise: at step 1, $\hat{v}_1 = v_1 / 0.001 = 1000 \times v_1$ — the denominator of the Adam update is very small and erratic. Some parameters will receive enormous updates; others will receive tiny ones. The effective learning rate per parameter is unstable for ~1000 steps. Destructive early updates during this window can push weights into bad regions from which recovery is slow.

Warmup solution: hold the learning rate very small (e.g., $10^{-6}$) for the first 1000 steps while $v_t$ stabilizes, then ramp up to the full learning rate. Once $v_t$ is stable, the Adam updates are reliable and the full LR is safe.

**Warmup + cosine decay** (standard for transformers):
$$\eta_t = \begin{cases} \eta_{\max} \cdot t / T_w & t < T_w \text{ (linear warmup)} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi(t - T_w)}{T - T_w}\right) & t \geq T_w \end{cases}$$

**Why cosine rather than step decay**: cosine shape has slow decay at the start (high LR, fast progress in smooth regions) and gradual flattening at the end (low LR, fine-grained convergence near the minimum). Step decay creates abrupt transitions that can destabilize training if the step happens when the model is in a sensitive region. Cosine is smoother and consistently outperforms step decay in practice.

**ReduceLROnPlateau**: reduce LR by a factor when validation metric stops improving. Adaptive to actual training dynamics but adds a `patience` hyperparameter and can get stuck reducing LR repeatedly without real progress.

**The pattern in action**: "I train a 6-layer transformer with constant LR = $3 \times 10^{-4}$. Loss is unstable for the first 500 steps — spikes of 30-50% above the trend, then recoveries. I add 1000-step linear warmup from $10^{-6}$ to $3 \times 10^{-4}$. Instability disappears completely. The small LR during warmup keeps updates small enough that Adam's unreliable early moment estimates cause only small errors. After warmup, I apply cosine decay to $3 \times 10^{-5}$. Final validation loss: 10% lower than with constant LR."

**Common traps**:
- Using the same warmup step count regardless of total training steps. Warmup should be approximately 5–10% of total training steps, not a fixed number. A model trained for 100K steps needs ~5K-10K warmup steps; a model trained for 10K steps needs 500–1K steps.
- Applying warmup to SGD for the same reason as Adam. SGD does not have the moment stabilization problem — it has no second moment estimate. Warmup for SGD has a different motivation (gradual adaptation from random initialization) and different optimal duration.

---

## 9. Hyperparameter Search

**What the interviewer is testing**: whether you understand hyperparameter search as an optimization problem with a budget constraint, and why the choice of search strategy should be proportional to the cost per evaluation.

**The reasoning structure**: a model has $k$ hyperparameters. Each evaluation requires training a model and measuring performance — potentially hours of compute. You need to find good hyperparameter values within a fixed budget of $N$ evaluations.

Grid search: enumerate a finite grid of combinations. With 5 hyperparameters and 4 values each, that is $4^5 = 1024$ evaluations. Most hyperparameter spaces have low effective dimensionality — a few hyperparameters (typically learning rate and model capacity) drive most of the variation in performance, while others matter little. Grid search wastes most evaluations varying the unimportant dimensions while the important dimensions are held fixed across 256 evaluations at each value combination.

Random search: sample hyperparameter values independently from specified distributions. With the same evaluation budget, random search projects onto each dimension independently — the marginal distribution of each hyperparameter is approximately uniform over the specified range, regardless of what other hyperparameters look like. Bergstra & Bengio (2012) showed empirically and theoretically that random search is better than grid search when the number of relevant dimensions is small relative to the total number of hyperparameters.

Bayesian optimization: fit a surrogate model (Gaussian Process or tree-based model like random forest) to the mapping from hyperparameter values to validation performance, using all previous evaluations. Use the surrogate to select the next point via an acquisition function (expected improvement, upper confidence bound) that balances exploration (trying uncertain regions) with exploitation (trying regions the surrogate predicts are good). With each evaluation, the surrogate becomes a better model of the objective, and subsequent evaluations become more targeted. Best when each evaluation is expensive.

**Practical decision rule**: if each evaluation costs < 5 minutes, random search over a wide range is fast and adequate. If evaluations cost hours, Bayesian optimization pays off after ~20–30 evaluations. Grid search is rarely better than random search for any budget.

**The pattern in action**: "I have 5 hyperparameters: learning rate (log-scale $10^{-5}$ to $10^{-1}$), batch size (64 to 2048), dropout rate (0 to 0.5), weight decay ($10^{-5}$ to $10^{-1}$), and number of layers (2 to 8). Experience suggests learning rate is the most critical, followed by model capacity. Grid search with 4 values each = 1024 runs at 2 hours each = 2048 GPU-hours. Random search over 50 samples provides good coverage of the 2D important subspace (LR × layers) while varying the others. Bayesian optimization with 50 runs focuses after 15 exploration runs. For a 2-hour-per-evaluation problem, I use Optuna with 50 trials."

**Common traps**:
- Tuning hyperparameters against the test set. Hyperparameter search is model selection. Use a separate validation set or CV. Every hyperparameter selection based on test set performance inflates the reported final metric.
- Tuning hyperparameters one at a time. Learning rate and batch size interact (optimal LR scales with batch size). Model capacity and regularization interact (larger models need stronger regularization). Sequential single-hyperparameter tuning misses these interactions. Tune jointly.

---

## 10. Optimizer Selection

**What the interviewer is testing**: whether you can select an optimizer based on problem characteristics and training constraints, not just default to Adam or AdamW for everything.

**The reasoning structure**: optimizer choice involves three considerations: the architecture (does it have pathological gradient behavior?), the available compute budget (do you have time to tune SGD carefully?), and the training regime (pretraining, fine-tuning, full training?).

| Optimizer | When to use | Why |
| :--- | :--- | :--- |
| SGD + Nesterov momentum | Vision models (ResNets, ConvNets) with careful tuning | Better final accuracy than Adam when LR schedule is right; flat minima via noise |
| AdamW | Transformers, NLP, multimodal models; any case with variable gradient scales | Correct weight decay; reliable convergence; good default |
| Adam | Fast prototyping; tabular ML | Good default, reliable convergence; replace with AdamW for final runs |
| RMSprop | RNNs, reinforcement learning | No momentum accumulation issues; handles non-stationary objectives |
| Adagrad | Sparse features (NLP bag-of-words, sparse embeddings) | Per-parameter LR decay gives large updates to rarely-updated parameters |
| L-BFGS | Full-batch or near-full-batch training; fine-tuning with small datasets | Curvature information; very fast convergence when gradient noise is low |

The Adam vs. SGD tradeoff: Adam requires minimal tuning and reliably converges, but converges to sharper minima due to reduced gradient noise. SGD with momentum and a carefully tuned cosine LR schedule achieves better generalization on vision benchmarks. The tradeoff is engineering time vs. final accuracy.

**The pattern in action**: "Training a ResNet-50 on ImageNet. Fast path: AdamW with $3 \times 10^{-4}$, cosine schedule, gets to 76.2% top-1 accuracy in 100 epochs. Tuned path: SGD + Nesterov (LR 0.1, momentum 0.9, cosine schedule with warmup 5 epochs, weight decay $10^{-4}$), 90 epochs → 77.0% top-1. The 0.8% gap matters for production; the 10% extra engineering time does too. For a research prototype, Adam. For a production deployment where 0.8% matters, SGD."

**Common traps**:
- Using Adagrad for deep learning. Adagrad accumulates squared gradients monotonically and never forgets — the effective learning rate decays toward zero over time, making long training runs impossible. For deep networks with long training, use RMSprop (which uses an exponential moving average of squared gradients) or Adam.
- Forgetting to switch from Adam to AdamW when using weight decay. If `weight_decay > 0` in `torch.optim.Adam`, the regularization is incorrect. Always use `torch.optim.AdamW` when applying weight decay.

---

## 11. Practical Debugging Checklist

**What the interviewer is testing**: whether you have a systematic, symptom-based approach to training failures rather than randomly changing hyperparameters.

**The reasoning structure**: training failures have specific causes that map to observable symptoms. Each symptom narrows the diagnosis space dramatically. The first response to any training failure should be diagnosis, not random hyperparameter changes.

**Symptom → most likely cause → fix**:

| Symptom | Most likely cause | Fix |
| :--- | :--- | :--- |
| Loss is NaN from step 1 | LR too high; log(0) or division by zero in loss | Reduce LR by 10×; check loss implementation for numerical stability |
| Loss is NaN after $N$ steps | Exploding gradients | Add gradient clipping (max_norm=1.0); log gradient norms |
| Loss not decreasing | LR too low; wrong loss function; data pipeline bug | Increase LR; verify loss function matches task; check label encoding; overfit single batch first |
| Train loss ↓, val loss plateau | Overfitting | More regularization (dropout, weight decay); more data; simpler architecture |
| Train loss ↓, val loss eventually ↑ | Overfitting; LR too high late in training | Early stopping; add LR decay; checkpoint best val model |
| Gradient norms near zero | Vanishing gradients; saturated activations | Check activation functions; add residual connections; verify initialization |
| Loss decreasing then suddenly spikes | Saddle point neighborhood; LR too high | Gradient clipping; reduce LR; increase warmup |
| Validation loss bouncing | LR too high (oscillating); bad batch normalization statistics | Reduce LR; verify batch statistics; check for train/eval mode mismatch |

**The single most useful debugging technique**: intentionally overfit one batch. Take 1–8 examples from your training set and train until the model memorizes them perfectly (training loss → 0). If the model cannot memorize 8 examples, the architecture or loss function is broken — this has nothing to do with data quality, regularization, or generalization. It confirms the basic forward pass, loss computation, and gradient flow are correct before worrying about anything else.

**The second most useful technique**: log gradient norms per layer. Plot `torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))` for the whole model and `p.grad.norm()` for individual layers. This reveals whether gradients are flowing to all layers, which layers have pathological gradient magnitudes, and whether clipping is active more than expected.

**Common traps**:
- Changing multiple hyperparameters simultaneously when debugging. If you change LR, batch size, and architecture at once and performance improves, you do not know which change helped. Change one thing at a time. Keep a log.
- Not logging gradient norms. Gradient norm monitoring is cheap (one extra line per training step) and catches vanishing/exploding gradient issues before they become catastrophic. Add it from the start.

---

## 12. Quantization

**What the interviewer is testing**: whether you understand quantization as a precision-performance tradeoff with specific implications for inference latency, memory, and accuracy degradation — not just "making the model smaller."

**The reasoning structure**: neural network weights and activations are stored as 32-bit floats (FP32) during training. This provides high numerical precision to represent gradients and accumulate small updates stably. At inference, this precision is often unnecessary — the forward pass does not require gradient precision, and 8-bit integers or 16-bit floats often preserve accuracy while delivering 2–4x memory reduction and latency improvement on hardware with native INT8/FP16 support.

Quantization replaces full-precision values with lower-precision approximations. For INT8 quantization of a weight tensor with range $[w_{\min}, w_{\max}]$:

$$\hat{w} = \text{round}\left(\frac{w - z}{s}\right), \quad s = \frac{w_{\max} - w_{\min}}{2^8 - 1}, \quad z = -\frac{w_{\min}}{s}$$

The scale $s$ and zero-point $z$ are stored per tensor (or per channel for higher accuracy). Dequantization recovers the approximate FP32 value: $w \approx \hat{w} \cdot s + z$. Quantization error is $O(s)$ — proportional to the range of the weight tensor.

**Post-training quantization (PTQ)**: take a trained FP32 model and quantize it without any retraining. Calibrate the scale/zero-point parameters using a small representative dataset (calibration set, typically 100–1000 samples). Fast and simple, but can lose accuracy when the weight or activation distributions have wide range or outliers.

**Quantization-aware training (QAT)**: simulate quantization during training by inserting "fake quantization" operations — quantize and dequantize in the forward pass, but maintain FP32 weights for gradient updates. The model learns to be robust to quantization by experiencing it during training. Recovers most of the accuracy that PTQ loses. Required for aggressive quantization (INT4, ternary).

| Format | Memory (1B params) | Training | Typical accuracy impact |
| :--- | :--- | :--- | :--- |
| FP32 | 4 GB | Standard | Baseline |
| BF16 / FP16 | 2 GB | Mixed-precision standard | Negligible (< 0.1%) |
| INT8 | 1 GB | PTQ or QAT | Small (< 1% for most tasks) |
| INT4 | 0.5 GB | QAT or GPTQ | Moderate (1–5%, task-dependent) |

**Mixed-precision training** is different from quantization: FP16 for forward/backward passes (faster on modern GPUs with tensor cores), FP32 for optimizer state and gradient accumulation (prevents underflow). Requires gradient scaling to avoid FP16 underflow in small gradients.

**The pattern in action**: "I deploy a 7B-parameter language model. FP32 would require 28 GB of GPU memory — does not fit on a single 24 GB GPU. BF16 (2 bytes per param): 14 GB — fits. INT8 (1 byte per param): 7 GB — easily fits, and benchmark perplexity increases by 0.2%. INT4 (GPTQ): 3.5 GB, perplexity increases by 1.5% on standard benchmarks but on my specific task (code generation) degrades more noticeably. I deploy INT8 via bitsandbytes for 2x memory reduction with negligible quality loss."

**Common traps**:
- Quantizing without benchmarking on the production task. Accuracy degradation from quantization is task- and distribution-dependent. Aggregate benchmarks may show 0.5% degradation while the specific input distribution the model will serve shows 5% degradation. Always benchmark on representative production data.
- Assuming FP16 is always safe without gradient scaling. Naively running training in FP16 without loss scaling causes underflow: gradients with magnitude < $\sim 6 \times 10^{-8}$ (FP16 minimum positive) are rounded to zero. Framework mixed-precision implementations (`torch.cuda.amp.autocast`) handle this automatically via dynamic loss scaling — but do not implement manual FP16 without understanding this.

## Flashcards

**Conflating gradient descent (the update rule) with backpropagation (the gradient computation algorithm). Backpropagation computes $\nabla_\theta L$; gradient descent uses it. A model trained with automatic differentiation (PyTorch autograd) uses backprop for gradient computation, but could use any optimizer (SGD, Adam, L-BFGS) for the update step.?** #flashcard
Conflating gradient descent (the update rule) with backpropagation (the gradient computation algorithm). Backpropagation computes $\nabla_\theta L$; gradient descent uses it. A model trained with automatic differentiation (PyTorch autograd) uses backprop for gradient computation, but could use any optimizer (SGD, Adam, L-BFGS) for the update step.

**Treating gradient descent as an exact optimization algorithm. The update $\theta - \eta \nabla_\theta L$ exactly minimizes the linearized loss around $\theta$. But the actual loss is not linear, so the true minimum is not at $\theta - \eta \nabla_\theta L$ except infinitesimally. Each step is an approximation.?** #flashcard
Treating gradient descent as an exact optimization algorithm. The update $\theta - \eta \nabla_\theta L$ exactly minimizes the linearized loss around $\theta$. But the actual loss is not linear, so the true minimum is not at $\theta - \eta \nabla_\theta L$ except infinitesimally. Each step is an approximation.

**Assuming larger batches are strictly better because they estimate the gradient more accurately. Gradient accuracy and solution quality are not the same thing. The gradient noise in small-batch training is not a bug?** #flashcard
it is an implicit regularizer.

**Not scaling the learning rate when increasing batch size. The standard heuristic?** #flashcard
multiply learning rate by the factor by which you scale the batch size. (With warmup: first warm up from the original LR to the scaled LR over ~5 epochs, then apply the scheduled LR.) Without this scaling, large-batch training undershoots the optimal LR for the landscape.

**Setting learning rate once and leaving it constant. Learning rate scheduling?** #flashcard
warmup + cosine decay — typically delivers more benefit than any other single hyperparameter choice. The specific LR value matters less than the schedule shape.

**Using the same learning rate for all layers when fine-tuning. The first few layers of a pretrained network have learned generic low-level representations; they need minimal updating. Later layers need more adaptation. Using a single LR either destroys the generic early-layer representations or fails to adapt the task-specific later layers.?** #flashcard
Using the same learning rate for all layers when fine-tuning. The first few layers of a pretrained network have learned generic low-level representations; they need minimal updating. Later layers need more adaptation. Using a single LR either destroys the generic early-layer representations or fails to adapt the task-specific later layers.

**Setting $\beta$ too high (e.g., 0.99 or higher). The effective smoothing window is $1/(1-\beta)$ steps. At $\beta = 0.99$, the window is 100 steps?** #flashcard
the velocity accumulates so much history that it overshoots minima and causes instability. Standard is $\beta = 0.9$ for SGD + momentum.

**Not resetting the velocity when making large changes to the learning rate or task. The accumulated velocity encodes gradient information from the old learning rate or task. After a major LR reduction or a task switch in continual learning, reset momentum to zero.?** #flashcard
Not resetting the velocity when making large changes to the learning rate or task. The accumulated velocity encodes gradient information from the old learning rate or task. After a major LR reduction or a task switch in continual learning, reset momentum to zero.

**Using torch.optim.Adam with weight decay for deep learning. Standard Adam applies weight decay incorrectly?** #flashcard
the weight decay term is adaptively scaled, so the regularization is weaker for high-gradient parameters. Use AdamW (see section 6).

**Assuming Adam always converges faster than SGD. Adam converges in fewer steps for most problems and requires less hyperparameter tuning. But for vision models where SGD is carefully tuned (cosine schedule, right LR), SGD often achieves better final accuracy because its gradient noise produces flatter, more generalizable minima. Adam's adaptivity reduces noise and can cause convergence to sharper solutions.?** #flashcard
Assuming Adam always converges faster than SGD. Adam converges in fewer steps for most problems and requires less hyperparameter tuning. But for vision models where SGD is carefully tuned (cosine schedule, right LR), SGD often achieves better final accuracy because its gradient noise produces flatter, more generalizable minima. Adam's adaptivity reduces noise and can cause convergence to sharper solutions.

**Using torch.optim.Adam for any modern deep learning task. AdamW should be the default. Standard Adam with weight_decay is misleading because it does not implement L2 regularization correctly.?** #flashcard
Using torch.optim.Adam for any modern deep learning task. AdamW should be the default. Standard Adam with weight_decay is misleading because it does not implement L2 regularization correctly.

**Setting weight_decay > 0.1. AdamW with weight_decay=0.3 combined with dropout is excessive regularization that hurts performance. Typical effective range?** #flashcard
0.01–0.1 for most tasks. Start at 0.01.

**Applying gradient clipping to a vanishing gradient problem. Clipping restricts gradient magnitude downward?** #flashcard
it cannot increase magnitude. A vanishing gradient needs architectural solutions: residual connections, better initialization, normalization, non-saturating activations.

**Clipping by value instead of by norm. Clipping $g_1 = 1000$ and $g_2 = 0.001$ to $c = 1$ produces $(1, 0.001)$, which points almost entirely in the $g_1$ direction?** #flashcard
the gradient direction has changed from (1, 0) to approximately (1, 0). Clipping by norm instead scales both components proportionally, preserving the direction exactly.

**Using the same warmup step count regardless of total training steps. Warmup should be approximately 5–10% of total training steps, not a fixed number. A model trained for 100K steps needs ~5K-10K warmup steps; a model trained for 10K steps needs 500–1K steps.?** #flashcard
Using the same warmup step count regardless of total training steps. Warmup should be approximately 5–10% of total training steps, not a fixed number. A model trained for 100K steps needs ~5K-10K warmup steps; a model trained for 10K steps needs 500–1K steps.

**Applying warmup to SGD for the same reason as Adam. SGD does not have the moment stabilization problem?** #flashcard
it has no second moment estimate. Warmup for SGD has a different motivation (gradual adaptation from random initialization) and different optimal duration.

**Tuning hyperparameters against the test set. Hyperparameter search is model selection. Use a separate validation set or CV. Every hyperparameter selection based on test set performance inflates the reported final metric.?** #flashcard
Tuning hyperparameters against the test set. Hyperparameter search is model selection. Use a separate validation set or CV. Every hyperparameter selection based on test set performance inflates the reported final metric.

**Tuning hyperparameters one at a time. Learning rate and batch size interact (optimal LR scales with batch size). Model capacity and regularization interact (larger models need stronger regularization). Sequential single-hyperparameter tuning misses these interactions. Tune jointly.?** #flashcard
Tuning hyperparameters one at a time. Learning rate and batch size interact (optimal LR scales with batch size). Model capacity and regularization interact (larger models need stronger regularization). Sequential single-hyperparameter tuning misses these interactions. Tune jointly.

**Using Adagrad for deep learning. Adagrad accumulates squared gradients monotonically and never forgets?** #flashcard
the effective learning rate decays toward zero over time, making long training runs impossible. For deep networks with long training, use RMSprop (which uses an exponential moving average of squared gradients) or Adam.

**Forgetting to switch from Adam to AdamW when using weight decay. If weight_decay > 0 in torch.optim.Adam, the regularization is incorrect. Always use torch.optim.AdamW when applying weight decay.?** #flashcard
Forgetting to switch from Adam to AdamW when using weight decay. If weight_decay > 0 in torch.optim.Adam, the regularization is incorrect. Always use torch.optim.AdamW when applying weight decay.

**Changing multiple hyperparameters simultaneously when debugging. If you change LR, batch size, and architecture at once and performance improves, you do not know which change helped. Change one thing at a time. Keep a log.?** #flashcard
Changing multiple hyperparameters simultaneously when debugging. If you change LR, batch size, and architecture at once and performance improves, you do not know which change helped. Change one thing at a time. Keep a log.

**Not logging gradient norms. Gradient norm monitoring is cheap (one extra line per training step) and catches vanishing/exploding gradient issues before they become catastrophic. Add it from the start.?** #flashcard
Not logging gradient norms. Gradient norm monitoring is cheap (one extra line per training step) and catches vanishing/exploding gradient issues before they become catastrophic. Add it from the start.

**Quantizing without benchmarking on the production task. Accuracy degradation from quantization is task- and distribution-dependent. Aggregate benchmarks may show 0.5% degradation while the specific input distribution the model will serve shows 5% degradation. Always benchmark on representative production data.?** #flashcard
Quantizing without benchmarking on the production task. Accuracy degradation from quantization is task- and distribution-dependent. Aggregate benchmarks may show 0.5% degradation while the specific input distribution the model will serve shows 5% degradation. Always benchmark on representative production data.

**Assuming FP16 is always safe without gradient scaling. Naively running training in FP16 without loss scaling causes underflow: gradients with magnitude < $\sim 6 \times 10^{-8}$ (FP16 minimum positive) are rounded to zero. Framework mixed-precision implementations (torch.cuda.amp.autocast) handle this automatically via dynamic loss scaling?** #flashcard
but do not implement manual FP16 without understanding this.
