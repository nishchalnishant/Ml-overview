# Optimization

Optimization questions are rarely about memorizing formulas alone. Interviewers usually want to know whether you understand convergence, stability, compute tradeoffs, and what you would tune first in practice.

---

# Q1: What is gradient descent? How does it work?

**Interview-ready answer**

Gradient descent is an iterative optimization method that updates parameters in the direction that most decreases the loss locally, which is the negative gradient. At each step, you compute the gradient of the loss with respect to the parameters and move by a step size called the learning rate. In ML, we use it because most modern models do not have a closed-form solution, so we need a scalable way to improve the objective step by step.

**Good depth to add**

- The gradient tells you the local slope, not the global best direction.
- In convex problems, gradient descent has stronger guarantees; in deep learning it finds useful local minima or flat basins empirically.
- Too small a step makes training slow, while too large a step can cause divergence or oscillation.

---

# Q2: What is stochastic gradient descent (SGD)?

**Interview-ready answer**

SGD estimates the gradient using one example or a mini-batch rather than the full dataset. That makes each update cheaper and much more scalable, which is why it is the default training style for deep learning. The tradeoff is that the gradient estimate is noisy, but that noise is often helpful because it can act as an implicit regularizer and help the optimizer escape sharp or poor regions of the loss surface.

**What to mention**

- Full-batch gradient descent is usually too expensive for large datasets.
- In practice, "SGD" often means mini-batch SGD rather than batch size 1.
- Momentum is often added because plain SGD can zig-zag badly in ill-conditioned directions.

---

# Q3: What are vanishing gradients?

**Interview-ready answer**

Vanishing gradients happen when gradients shrink as they are propagated backward through many layers or time steps, so early layers receive almost no learning signal. This is common when repeated derivatives are smaller than one, as with saturating activations like sigmoid or tanh in deep or recurrent networks. The result is slow learning, especially for long-range dependencies.

**How we address them**

- Use activations like ReLU variants
- Add residual or skip connections
- Use normalization carefully
- Initialize weights well
- In sequence models, prefer architectures like LSTM, GRU, or transformers over plain RNNs

---

# Q4: What is a learning rate? How to choose a good one?

**Interview-ready answer**

The learning rate controls how large each parameter update is. It is often the single most important hyperparameter because it directly determines whether training converges, stalls, or diverges. In practice, I pick it empirically by starting from known good defaults for the optimizer and architecture, then checking the training curve, possibly using a learning-rate range test or a coarse log-scale sweep.

**What strong candidates say**

- Search on a log scale, not a linear scale.
- The right value depends on batch size, optimizer, normalization, and model scale.
- A learning rate schedule is often as important as the initial learning rate.

---

# Q5: How does the learning rate affect model training?

**Interview-ready answer**

If the learning rate is too high, the optimizer can overshoot good regions, oscillate, or diverge. If it is too low, training becomes painfully slow and may get stuck in poor basins or appear to plateau before reaching a good solution. In deep learning, the learning rate also affects the kind of solution found: very aggressive updates can be unstable, while well-tuned schedules often improve both convergence speed and final generalization.

**Nice nuance**

Do not answer this only in terms of speed. The learning rate also changes training stability and sometimes the quality of the minimum you end up in.

---

# Q6: How do you approach hyperparameter tuning?

**Interview-ready answer**

I start with a strong baseline and tune in an order that reflects likely impact. Usually that means first verifying data splits and evaluation, then tuning the learning rate or the most important capacity-control parameters, and only later exploring a wider search. I prefer a disciplined process over tuning everything at once: define the objective, choose a small set of influential hyperparameters, use a reproducible search space, and analyze results rather than just picking the single best trial.

**What to emphasize**

- Fix the evaluation protocol before tuning
- Use domain-informed ranges
- Track experiments and seeds
- Look for stable configurations, not only the absolute best score

---

# Q7: What is model quantization, and when would you use it?

**Interview-ready answer**

Quantization reduces the numerical precision of model weights and sometimes activations, for example from 32-bit floating point to 8-bit integers. The main goal is to reduce memory footprint and improve inference speed, especially on edge devices or latency-sensitive systems. The tradeoff is that aggressive quantization can hurt accuracy, particularly for small models, sensitive layers, or tasks requiring fine-grained numerical precision.

**Good nuance**

- Post-training quantization is simpler but may lose more accuracy.
- Quantization-aware training usually preserves accuracy better because the model learns under quantization effects.
- Quantization is especially valuable when memory bandwidth or serving cost is the bottleneck.

---

# Q8: How do you ensure fairness and reduce bias in ML models?

**Interview-ready answer**

I treat fairness as a full lifecycle issue rather than a last-step metric check. That means examining data collection, labeling, feature choice, objective design, thresholding, and monitoring by subgroup. In practice, I start by clarifying which fairness notion matters in the product context because different notions, such as equal opportunity and demographic parity, can conflict. Then I audit performance by relevant groups, look for proxy variables and coverage gaps, and decide whether the fix belongs in the data, the model, the threshold policy, or the surrounding workflow.

**Strong interview framing**

- Fairness is not only about protected attributes; proxies can also introduce harm.
- Aggregate metrics can hide severe subgroup failures.
- Sometimes the correct solution is policy and product design, not only reweighting the model.

---

# Q9: Explain Grid Search vs Random Search vs Bayesian Optimization.

**Interview-ready answer**

Grid search evaluates every combination in a fixed grid, random search samples configurations from a search space, and Bayesian optimization uses previous results to choose promising new trials. Grid search is simple but wasteful in high dimensions because most hyperparameters are not equally sensitive. Random search is often a stronger default because it explores more of the space efficiently. Bayesian optimization becomes useful when evaluations are expensive and you want a smarter search strategy that balances exploration and exploitation.

**Rule of thumb**

- Small search space and cheap training: grid can be fine
- Moderate space and practical workflows: random search is often best
- Expensive trials and few evaluation opportunities: Bayesian optimization is attractive

---

# Q10: Explain TPE hyperparameter optimization.

**Interview-ready answer**

TPE, or Tree-structured Parzen Estimator, is a form of Bayesian optimization that models good and bad regions of the hyperparameter space separately. Instead of modeling the objective directly, it estimates the density of configurations associated with strong outcomes and compares that against the density of weaker ones. It is especially useful for mixed search spaces with conditional parameters, which is why it is popular in practical HPO tools.

**Why it is useful**

- Handles discrete, continuous, and conditional spaces well
- Works better than naive search when trials are expensive
- Often easier to apply than Gaussian-process Bayesian optimization in messy real search spaces

---

# Q11: Explain Bayesian Optimization.

**Interview-ready answer**

Bayesian optimization is a strategy for optimizing expensive black-box functions, such as validation score as a function of hyperparameters. It builds a surrogate model of the objective and uses an acquisition function to decide where to evaluate next. The value is sample efficiency: instead of spending hundreds of random trials, it tries to learn where promising regions are and focus evaluation budget there.

**Good things to mention**

- The surrogate can be a Gaussian process, TPE-style density model, or another probabilistic model.
- The acquisition function balances exploring uncertain regions and exploiting promising ones.
- It is most helpful when each model run is expensive enough that smarter search is worth the overhead.

---

# Q12: Explain Adam Optimizer.

**Interview-ready answer**

Adam combines momentum with adaptive per-parameter learning rates. It keeps moving averages of the first moment of the gradient and the second moment of the squared gradient, then uses bias-corrected estimates to scale updates. In practice, Adam converges quickly and is very robust, which is why it is widely used in deep learning. However, it does not always generalize as well as tuned SGD with momentum, especially in some vision settings.

**Good interview nuance**

- Adam is often excellent for fast iteration and sparse gradients.
- Decoupled weight decay in AdamW is usually preferable to naive L2 inside Adam.
- "Converges faster" does not always mean "gives the best final model."

---

# Q13: Explain the RMSprop Optimizer.

**Interview-ready answer**

RMSprop adapts the learning rate of each parameter by dividing the gradient by a running average of recent squared gradients. That helps stabilize training when gradient magnitudes vary a lot across dimensions, which is common in deep networks and recurrent models. You can think of it as reducing the step size in directions that have consistently large gradients and allowing relatively larger steps where gradients are smaller.

**Where it fits**

- Historically strong for RNNs and non-stationary objectives
- Simpler than Adam because it does not include the first-moment momentum term in the same way
- Less common as a default today, but still important conceptually

---

# Q14: What is Adagrad Optimizer?

**Interview-ready answer**

Adagrad adapts the learning rate for each parameter based on the accumulated sum of past squared gradients. Parameters that receive frequent updates get smaller effective learning rates over time, while infrequent parameters can still receive relatively larger updates. That makes Adagrad attractive for sparse features, such as text or recommendation problems, but its main weakness is that the learning rate can decay too aggressively and eventually become too small to keep learning effectively.

**Nice comparison**

Adagrad explains the intuition behind later optimizers: adaptive scaling is useful, but you need to control the decay behavior, which is why methods like RMSprop and Adam became more popular.
