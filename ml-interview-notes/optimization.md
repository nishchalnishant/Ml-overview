# Optimization

---

# Q1: What is gradient descent? How does it work?

## 1. 🔹 Direct Answer
**Gradient descent** minimizes a loss **L(θ)** by iteratively moving parameters **opposite** the gradient: **θ ← θ − η ∇L**. **η** is the learning rate.

## 2. 🔹 Intuition
The gradient points **uphill** in loss; stepping downhill finds local minima (hopefully good enough).

## 3. 🔹 Deep Dive
- **Batch GD**: gradient on full data—accurate, slow.
- **Convex** problems: global minimum (e.g., linear regression with MSE).
- **Non-convex** (deep nets): many local minima/saddle points—**momentum**, **schedules** help.

## 4. 🔹 Practical Perspective
Monitor **loss curve**; use **learning rate finder**, **warmup**, **clip** gradients.

## 5. 🔹 Code Snippet
```python
for _ in range(steps):
    grad = compute_grad(theta, X, y)
    theta -= lr * grad
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Saddle points? **A:** Hessian has + and − eigenvalues; noise/momentum helps escape plateaus.

## 7. 🔹 Common Mistakes
Confusing **global** batch GD with **mini-batch** SGD used in practice.

## 8. 🔹 Comparison / Connections
SGD, Adam, second-order methods (Newton—expensive).

## 9. 🔹 One-line Revision
Gradient descent steps opposite ∇L with learning rate—batch vs minibatch trades noise for speed.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: What is stochastic gradient descent (SGD)?

## 1. 🔹 Direct Answer
**SGD** uses **one** (or a **minibatch**) sample’s gradient as a **noisy** estimate of full gradient—**faster** updates, **lower** memory, **regularizing** noise can help generalization.

## 2. 🔹 Intuition
Full-batch gradient is expensive; noisy steps **explore** and can escape sharp minima (flat minima often generalize better—informal).

## 3. 🔹 Deep Dive
- **Minibatch** balances variance and throughput (GPU efficiency).
- **With replacement** sampling assumptions for theory.

## 4. 🔹 Practical Perspective
Default in deep learning: **batch size** is key hyperparameter (often 32–256).

## 5. 🔹 Code Snippet
```python
for batch in loader:  # mini-batch SGD
    grad = backward(loss(model(batch.x), batch.y))
    opt.step()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Large batch issues? **A:** Sharp minima, needs higher LR scaling rules (linear scaling rule—careful).

## 7. 🔹 Common Mistakes
Calling minibatch GD “SGD” imprecisely—clarify batch size 1 vs small batch.

## 8. 🔹 Comparison / Connections
Adam, LARS, large-batch training tricks.

## 9. 🔹 One-line Revision
SGD/minibatch uses stochastic gradients for scalable noisy optimization—tune LR and batch jointly.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What are vanishing gradients?

## 1. 🔹 Direct Answer
In deep **sigmoid/tanh** nets, backprop **multiplies** small derivatives layer-wise → gradients **shrink** exponentially—early layers learn **slowly** or not at all.

## 2. 🔹 Intuition
Multiplying many numbers < 1 drives product toward **zero**—signals die.

## 3. 🔹 Deep Dive
- **Mitigations**: **ReLU** family, **residual** connections, **Batch/Layer Norm**, **better init** (He/Xavier), **gating** (LSTM), **shortcut** paths.

## 4. 🔹 Practical Perspective
Less of an issue with modern architectures; still watch **depth** and **activation**.

## 5. 🔹 Code Snippet
```text
∂L/∂h1 = ∂L/∂hL ∏ ∂h_{k+1}/∂h_k  →  product of many small terms
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Exploding gradients? **A:** Opposite—clip, init, residual.

## 7. 🔹 Common Mistakes
Blaming only depth without mentioning activation choice.

## 8. 🔹 Comparison / Connections
RNN long sequences, ResNet.

## 9. 🔹 One-line Revision
Vanishing gradients come from chained small derivatives—ReLU, norms, and residuals preserve signal.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What is a learning rate? How to choose a good one?

## 1. 🔹 Direct Answer
**Learning rate η** scales gradient steps. Too **large**: divergence/oscillation; too **small**: slow training / stuck in plateaus. Tune via **grid/random search**, **LR range test**, **schedules** (cosine, warmup).

## 2. 🔹 Intuition
Step size in downhill walk—big steps overshoot the valley floor.

## 3. 🔹 Deep Dive
- **Warmup**: small η early for stability (Transformers).
- **Cosine decay**, **step decay**, **ReduceLROnPlateau**.

## 4. 🔹 Practical Perspective
**1cycle** policy, **Adam** with default LR often OK for many tasks—still tune.

## 5. 🔹 Code Snippet
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Adam vs SGD LR? **A:** Different scales; Adam less sensitive but not always best generalization.

## 7. 🔹 Common Mistakes
Same LR across orders-of-magnitude different batch sizes without scaling heuristics.

## 8. 🔹 Comparison / Connections
Line search, second-order methods.

## 9. 🔹 One-line Revision
Learning rate governs step size—use schedules, warmup, and empirical search; pair with batch size.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: How does the learning rate affect model training?

## 1. 🔹 Direct Answer
**Higher** LR → faster convergence risk **instability** (loss spikes, NaNs). **Lower** LR → stable but **slow**; may stop in **suboptimal** regions if too low. Interacts with **batch size**, **optimizer**, **initialization**.

## 2. 🔹 Intuition
Controls how aggressively you trust each gradient estimate.

## 3. 🔹 Deep Dive
- **Convex**: exists optimal η range.
- **Non-convex**: η affects **basin** of attraction—**SGD noise** + LR affects implicit regularization.

## 4. 🔹 Practical Perspective
Plot **train loss vs step**; if noisy, reduce LR or increase batch (trade-offs).

## 5. 🔹 Code Snippet
```text
if loss is NaN: lower lr, check grads, mixed precision
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Cyclical LR? **A:** Explore range; can escape local minima—Smith’s 1cycle.

## 7. 🔹 Common Mistakes
Tuning everything else but leaving default LR inappropriate for dataset scale.

## 8. 🔹 Comparison / Connections
Batch norm interaction (effective LR), weight decay.

## 9. 🔹 One-line Revision
LR trades speed vs stability and determines optimization trajectory—monitor loss and align with schedule.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: How do you approach hyperparameter tuning?

## 1. 🔹 Direct Answer
**Define** budget and **search space** (log scale for LR, batch discrete). Start **coarse** random search, then **narrow**. Use **CV** or **holdout**; **nested** CV if reporting unbiased performance. **Track** experiments (MLflow/W&B).

## 2. 🔹 Intuition
Grid over high-dim space wastes runs—**random** often finds good regions faster (Bergstra & Bengio).

## 3. 🔹 Deep Dive
- **Bayesian optimization** / **TPE** for expensive evals.
- **Early stopping** bad trials (Hyperband).

## 4. 🔹 Practical Perspective
Tune **big rocks** first: LR, wd, architecture depth—before tiny augment knobs.

## 5. 🔹 Code Snippet
```python
import random
for _ in range(20):
    lr = 10 ** random.uniform(-4, -2)
    wd = 10 ** random.uniform(-5, -3)
    # train_eval(lr, wd)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** HPO on test? **A:** Never—validation only.

## 7. 🔹 Common Mistakes
**Overfitting** the validation set by too many manual peeks—use final test once.

## 8. 🔹 Comparison / Connections
AutoML, neural architecture search.

## 9. 🔹 One-line Revision
Random/Bayesian search with CV, logging, and clear val/test discipline—budget-aware.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is model quantization, and when would you use it?

## 1. 🔹 Direct Answer
**Quantization** reduces numeric precision (**FP32→FP16/BF16/INT8/INT4**) of **weights** and/or **activations** to **shrink memory** and **accelerate** inference. Use for **latency**, **edge**, **cost**—accept some **accuracy** loss; often **PTQ** first, **QAT** if needed.

## 2. 🔹 Intuition
Fewer bits = coarser numbers—OK when signal tolerates rounding.

## 3. 🔹 Deep Dive
- **Symmetric/asymmetric** scales, **per-channel** weights.
- **Calibration** batches for activation ranges (percentile clipping).

## 4. 🔹 Practical Perspective
LLMs: **INT4** weights + FP16 activations common; verify **perplexity** / task metrics.

## 5. 🔹 Code Snippet
```python
import torch.quantization as quant
# torch.ao.quantization — PyTorch FX graph mode example in docs
```

## 6. 🔹 Interview Follow-ups
1. **Q:** QAT vs PTQ? **A:** QAT trains with fake quant—better accuracy at INT8.

## 7. 🔹 Common Mistakes
Quantizing without measuring **outlier** layers (attention sometimes sensitive).

## 8. 🔹 Comparison / Connections
Pruning, distillation, TensorRT.

## 9. 🔹 One-line Revision
Quantization trades precision for speed/size—calibrate, validate task metrics, use QAT when PTQ fails.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q8: How do you ensure fairness and reduce bias in ML models?

## 1. 🔹 Direct Answer
**Define** fairness metrics (**equalized odds**, **demographic parity**, **calibration** across groups). **Audit** data for **historical bias**, **representation** gaps. **Mitigate**: reweighting, **constrained** optimization, **post-hoc** calibration, **human** review for high-stakes. **Document** trade-offs—often **impossibility** results between criteria.

## 2. 🔹 Intuition
“Fair” isn’t one number—**stakeholders** choose constraints.

## 3. 🔹 Deep Dive
- **Proxy** variables can reintroduce discrimination.
- **Intersectional** subgroups, not only single-axis.

## 4. 🔹 Practical Perspective
**Monitoring** in prod for **drift** across cohorts; **appeals** process.

## 5. 🔹 Code Snippet
```text
report metrics by slice: precision, recall, FNR per group
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Equalized odds vs calibration? **A:** Can conflict—know which error type is costlier.

## 7. 🔹 Common Mistakes
Removing protected attribute from features while correlated proxies remain.

## 8. 🔹 Comparison / Connections
Ethics, robustness, causal fairness.

## 9. 🔹 One-line Revision
Fairness needs explicit metrics, subgroup evaluation, mitigation, and governance—not blind debiasing.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q9: Explain Grid Search vs Random Search vs Bayesian Optimization.

## 1. 🔹 Direct Answer
- **Grid**: exhaustive on discrete grid—**curse of dimensionality**.
- **Random**: samples uniformly—often **more efficient** in high-dim (many dims irrelevant).
- **Bayesian** (e.g., **GP**, **TPE**): builds **surrogate** for objective, picks **promising** points—best when each eval is **expensive**.

## 2. 🔹 Intuition
Random explores widely; Bayes **exploits** structure of observed scores.

## 3. 🔹 Deep Dive
- BO: acquisition function (**EI**, **UCB**) balances explore/exploit.
- **Parallel** suggestions for clusters.

## 4. 🔹 Practical Perspective
Start random; switch to **Optuna**/**Ray Tune** BO for big models.

## 5. 🔹 Code Snippet
```python
import optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    return val_score(lr)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Categorical hparams? **A:** TPE handles mixed spaces well.

## 7. 🔹 Common Mistakes
Grid search LR on linear grid instead of **log** scale.

## 8. 🔹 Comparison / Connections
Hyperband, population-based training.

## 9. 🔹 One-line Revision
Use random for cheap exploration; Bayesian optimization when trials are costly and structured.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: Explain TPE hyperparameter optimization.

## 1. 🔹 Direct Answer
**Tree-structured Parzen Estimator (TPE)** models **p(x|y)** as two densities: **l(x)** from good trials (y<threshold) and **g(x)** from bad—proposes **x** maximizing **l(x)/g(x)** (expected improvement flavor). Works in **mixed** categorical/continuous spaces.

## 2. 🔹 Intuition
Learn **where good configs live** vs **where failures cluster**—sample from promising region.

## 3. 🔹 Deep Dive
- Used in **Hyperopt**, **Optuna** default sampler.
- Not GP-based—scales better to **high-dim** categorical than naive GP.

## 4. 🔹 Practical Perspective
Great default for deep learning HPO with **pruning** (Optuna).

## 5. 🔹 Code Snippet
```text
Optuna: study.optimize(..., sampler=TPESampler())
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs Gaussian Process? **A:** GP smooth surrogate; TPE handles non-smooth + categorical more flexibly.

## 7. 🔹 Common Mistakes
Assuming TPE guarantees global optimum—still heuristic.

## 8. 🔹 Comparison / Connections
SMAC, Bayesian optimization family.

## 9. 🔹 One-line Revision
TPE splits trials into good/bad and models densities to propose high-likelihood-improvement configs.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q11: Explain Bayesian Optimization.

## 1. 🔹 Direct Answer
**Bayesian optimization** maintains a **probabilistic surrogate** (often **GP**) for **f(x)** (validation metric), updates with each observation, and picks next **x** by an **acquisition** balancing exploration/exploitation—sample-efficient for **black-box** expensive **f**.

## 2. 🔹 Intuition
Instead of random guessing, you **reason** about uncertainty—try points where **mean is high** or **variance is large**.

## 3. 🔹 Deep Dive
- Acquisition: **EI**, **PI**, **UCB**.
- GPs scale **O(n³)**—use sparse approximations or TPE for large n.

## 4. 🔹 Practical Perspective
Use for **AutoML** HPO, neural arch search with costly training.

## 5. 🔹 Code Snippet
```text
surrogate: GP(mean=0, kernel=RBF); maximize EI(x) over search space
```

## 6. 🔹 Interview Follow-ups
1. **Q:** High-dimensional? **A:** GPs struggle—random embeddings, trust regions, or TPE.

## 7. 🔹 Common Mistakes
Confusing BO with Bayesian neural networks.

## 8. 🔹 Comparison / Connections
Gaussian processes, bandits, TPE.

## 9. 🔹 One-line Revision
Bayesian optimization uses a surrogate + acquisition to optimize expensive black-box functions sample-efficiently.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q12: Explain Adam Optimizer.

## 1. 🔹 Direct Answer
**Adam** = **Adaptive moments**: maintains **exponential moving averages** of gradients (**m**) and squared gradients (**v**) with bias correction; per-parameter **adaptive** learning rates. Default **β1=0.9, β2=0.999, ε=1e-8**.

## 2. 🔹 Intuition
Like momentum + RMSprop—scale step by **typical gradient magnitude** per parameter.

## 3. 🔹 Deep Dive
Updates: **m_t, v_t** decay; **θ -= η m̂ / (√v̂ + ε)**.

## 4. 🔹 Practical Perspective
Works well out-of-box; some **generalization** studies prefer **SGD+Momentum** with tuning for vision.

## 5. 🔹 Code Snippet
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Weight decay in Adam? **A:** Use **AdamW** (decoupled WD)—fixes bad regularization interaction.

## 7. 🔹 Common Mistakes
Thinking Adam never needs LR tuning—it does for best results.

## 8. 🔹 Comparison / Connections
AdamW, Lion, LAMB.

## 9. 🔹 One-line Revision
Adam adapts per-parameter steps via momentum and second-moment estimates—use AdamW for proper weight decay.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: Explain the RMSprop Optimizer.

## 1. 🔹 Direct Answer
**RMSprop** divides gradient by **root mean square** of recent squared gradients (**adaptive** scaling per parameter). Helps with **non-stationary** objectives (RNNs)—**α** decay for moving average of **g²**.

## 2. 🔹 Intuition
Shrinks steps in directions with **large historical gradients**—faster progress in flat directions.

## 3. 🔹 Deep Dive
**cache**: **E[g²] = α E[g²] + (1-α) g²**; update **θ -= η g / (√E[g²]+ε)**.

## 4. 🔹 Practical Perspective
Largely superseded by **Adam** (adds momentum) but concept core to Adam’s **v** term.

## 5. 🔹 Code Snippet
```python
torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs Adagrad? **A:** Adagrad accumulates all past squares—learning slows; RMSprop uses **moving** window.

## 7. 🔹 Common Mistakes
Confusing with Adam—Adam combines RMSprop-like scaling with momentum.

## 8. 🔹 Comparison / Connections
Adagrad, Adam, AdaDelta.

## 9. 🔹 One-line Revision
RMSprop uses exponentially weighted squared-gradient norm for adaptive per-dimension scaling—good for RNN-style landscapes.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: What is Adagrad Optimizer?

## 1. 🔹 Direct Answer
**Adagrad** accumulates **sum of squares** of all past gradients per parameter and scales update inversely: **larger** cumulative grad² → **smaller** step. Great for **sparse** features (big steps for infrequent dims)—but learning rate may **decay to zero** too aggressively.

## 2. 🔹 Intuition
Frequent features get **tamed** steps; rare features keep larger effective LR early.

## 3. 🔹 Deep Dive
**G_t = G_{t-1} + g_t²** (elementwise); **θ -= η g / (√G + ε)**.

## 4. 🔹 Practical Perspective
Less common in deep nets now—**RMSprop/Adam** fix monotonic shrinkage.

## 5. 🔹 Code Snippet
```python
torch.optim.Adagrad(model.parameters(), lr=0.01)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When still useful? **A:** Sparse linear models, some online learning.

## 7. 🔹 Common Mistakes
Forgetting that accumulated denominator can **stop** learning.

## 8. 🔹 Comparison / Connections
Adadelta (limits window), Adam.

## 9. 🔹 One-line Revision
Adagrad adapts per-parameter LR by full history of squared grads—great sparsity handling but aggressive decay.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
