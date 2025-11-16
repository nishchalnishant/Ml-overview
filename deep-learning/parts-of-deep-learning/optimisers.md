# Optimisers

Here are detailed notes on the most common optimizers used in deep learning, complete with their pros, cons, and typical use cases.

#### 📜 What is an Optimizer?

In deep learning, an optimizer is an algorithm used to change the attributes of your neural network, such as weights and learning rates, to minimize the loss function.

The loss function measures how far your model's prediction is from the actual target. The optimizer's job is to "steer" the model's weights in the direction that makes this error as small as possible. Think of it as a hiker in a foggy valley (the loss landscape) trying to find the lowest point. The optimizer is their strategy for taking steps to get to the bottom.

***

#### 1. Stochastic Gradient Descent (SGD)

This is the most fundamental optimization algorithm. It's the basis for many others.

* How it Works: It calculates the gradient (the "slope" of the loss) for a small batch of training data and updates the weights by taking a small step in the opposite direction of the gradient.
* Pros:
  * Simple: Easy to understand and implement.
  * Memory-light: Requires minimal memory.
  * Can find better minima: The "noisy" updates from small batches can help it escape shallow local minima and find a more "flat" (and thus more generalizable) minimum.
* Cons:
  * Slow convergence: Can take a long time to find the minimum.
  * Oscillation: Tends to "bounce around" noisily, especially in steep ravines of the loss landscape.
  * Sensitive to learning rate: Picking the right learning rate is critical and difficult. A rate that's too high will cause it to diverge; too low, and it will take forever to train.
* When to Use:
  * For learning purposes.
  * If you have a lot of time and are skilled at tuning learning rate schedules.
  * Some state-of-the-art computer vision models still use it (with momentum) as it can offer better generalization.

***

#### 2. SGD with Momentum

This is an improvement on standard SGD that helps it converge faster and more reliably.

* How it Works: It adds a "momentum" term, which is an exponentially decaying average of past gradients. This helps the optimizer build up "speed" in the correct direction.
* Analogy: Imagine a ball rolling down a hill. It doesn't just stop at every small bump; its momentum carries it over them and helps it accelerate down the consistent slope.
* Pros:
  * Faster convergence than standard SGD.
  * Dampens oscillations: The momentum term smooths out the "bumpy" updates, leading to a more stable path.
  * Helps escape local minima and navigate ravines more effectively.
* Cons:
  * One extra hyperparameter: You now have to tune the momentum term (usually denoted as $$ $\beta$ $$, with a common default of 0.9).
  * Still requires careful tuning of the learning rate.
* When to Use:
  * A solid baseline for many problems, especially in Computer Vision (CV).
  * When you want better performance than standard SGD without the complexity of adaptive methods.

***

#### 3. Adagrad (Adaptive Gradient Algorithm)

Adagrad is an "adaptive" optimizer, meaning it adapts the learning rate for _each parameter_ individually.

* How it Works: It gives _smaller_ learning rates to parameters that have received _large_ gradients in the past, and _larger_ learning rates to parameters that have received _small_ gradients. It does this by accumulating the _sum of squared past gradients_ for each parameter.
* Pros:
  * Excellent for sparse data: It's highly effective in tasks like Natural Language Processing (NLP) or recommendation systems, where some features (e.g., rare words) are seen infrequently.
  * No manual learning rate tuning: The initial learning rate is often left at a default (like 0.01), as the algorithm handles the rest.
* Cons:
  * Dying learning rate: The main weakness. Because it _accumulates_ all past squared gradients, the sum in the denominator keeps growing. This causes the learning rate to eventually become infinitely small, and the model stops learning entirely.
* When to Use:
  * NLP tasks (like training word embeddings) or any problem with very sparse features.
  * Rarely used for deep vision models.

***

#### 4. RMSprop (Root Mean Square Propagation)

RMSprop was developed to solve Adagrad's "dying learning rate" problem.

* How it Works: Like Adagrad, it uses a per-parameter learning rate. However, instead of accumulating _all_ past squared gradients, it uses an exponentially decaying average of them. This means it "forgets" the distant past and only focuses on the recent gradient history.
* Pros:
  * Solves Adagrad's main problem: The learning rate no longer monotonically decreases to zero.
  * Converges quickly: Often much faster than SGD.
  * Works well on "non-stationary" problems (where the data distribution changes).
* Cons:
  * Still requires tuning: You need to set a global learning rate and a decay parameter ($$ $\rho$ $$, usually \~0.9).
* When to Use:
  * A great general-purpose optimizer, especially for Recurrent Neural Networks (RNNs).
  * A good alternative to Adam if you want a solid adaptive optimizer.

***

#### 5. Adam (Adaptive Moment Estimation)

Adam is the most common and popular optimizer today. It essentially combines the best parts of RMSprop and Momentum.

* How it Works:
  1. It keeps an exponentially decaying average of past gradients (like Momentum, the "first moment").
  2. It keeps an exponentially decaying average of past _squared_ gradients (like RMSprop, the "second moment").
* Pros:
  * Combines best of both worlds: Has the fast convergence of momentum and the adaptive learning rates of RMSprop.
  * Fast and efficient: Works very well on a huge variety of problems "out of the box."
  * Relatively low-maintenance: The default hyperparameter values (learning rate=0.001, $$ $\beta_1$ $$=0.9, $$ $\beta_2$ $$=0.999) work well for most tasks.
* Cons:
  * Can converge to a worse minimum: Some research suggests that Adam can sometimes find a "sharper" local minimum, which may not generalize as well as the "flatter" minimum found by SGD+Momentum.
  * More memory: Requires storing two moving averages (first and second moments) for _every single parameter_ in the model.
* When to Use:
  * The default, go-to optimizer for almost any problem.
  * An excellent starting point for any new project (CV, NLP, etc.).

***

#### 6. AdamW (Adam with Weight Decay)

AdamW is a _correction_ to the way Adam handles L2 regularization (also known as "weight decay").

* How it Works: Standard Adam implements L2 regularization by adding it to the gradient. This couples it with the "moment" calculations, which is incorrect. AdamW decouples the weight decay, applying it directly to the weights _after_ the optimizer step.
* Pros:
  * Better generalization: This "fix" often leads to models that perform significantly better on test data.
  * Fixes a fundamental flaw in Adam's implementation of weight decay.
  * Improved performance with a better-tuned `weight_decay` parameter.
* Cons:
  * Adds another hyperparameter (`weight_decay`) that needs to be tuned correctly.
* When to Use:
  * Whenever you would use Adam _and_ L2 regularization.
  * It has become the new standard for training large models like Transformers (e.g., BERT, GPT-style models).

***

#### 🚀 Summary: Which Optimizer Should I Choose?

* Start with Adam or AdamW: For most problems, Adam (or AdamW if you are using weight decay) will give you the fastest results and is the easiest to tune.
* For Computer Vision: If you're not getting the generalization you want from Adam, try SGD with Momentum (or Nesterov Momentum). It often requires a lot more tuning of the learning rate and a "warm-up" schedule, but it can lead to a final model with better test accuracy.
* For Sparse Data (NLP): While Adam works well, Adagrad or RMSprop are also excellent choices specifically designed for this type of problem.

Would you like a deeper dive into the math behind any of these, or perhaps to see how they are implemented in a library like PyTorch or TensorFlow?
