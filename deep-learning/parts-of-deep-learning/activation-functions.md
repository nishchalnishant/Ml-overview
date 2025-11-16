# Activation functions

Here are detailed notes on the activation functions used in deep learning, including what they do, when to use them, and their pros and cons.

#### 📜 What is an Activation Function?

An activation function is a mathematical "gate" applied to the output of a neuron (or a layer of neurons). Its primary job is to introduce non-linearity into the network.

Why is this critical? Without non-linear activation functions, a deep neural network, no matter how many layers it has, would just be a simple linear function. It would be no more powerful than a single layer (like basic linear regression) and could never learn complex patterns like recognizing speech, understanding language, or identifying objects in images.

Think of it as a "dimmer switch" or "gatekeeper" that decides how much of a neuron's signal should be passed on to the next layer.

![Image of a neural network layer with activation](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcRBCQ_MUB6OUs1fx02lnV6Tc8DsHE3K-Kjvc6fqidEb1KsNqQimrV8WrH6IdbPnT83_luhD0C15-oYbM6Gppbs9wGdbl79fzD8u9ZLKtXqz9qYLbVc)Shutterstock

***

#### 1. Functions for Hidden Layers

These are the functions you use _between_ the input and output layers.

**### ReLU (Rectified Linear Unit)**

This is the most popular and widely used activation function today.

* How it Works: It's a very simple "max" function.
  * If the input is positive, it passes the value through. $$ $f(x) = x$ $$
  * If the input is negative, it outputs zero. $$ $f(x) = 0$ $$
  * Formula: $$ $f(x) = \max(0, x)$ $$
* Pros:
  * Computationally very fast: It's just a simple comparison to zero.
  * Solves the Vanishing Gradient Problem: For positive inputs, the gradient (slope) is a constant 1. This allows the error signal to flow back through deep networks without dying out.
  * Induces sparsity: By outputting 0 for many inputs, it can make the network sparse (some neurons are "off"), which can be more efficient.
* Cons:
  * The "Dying ReLU" Problem: If a neuron's inputs consistently make it output 0, its gradient will always be 0. This means the neuron's weights will never get updated, and the neuron effectively "dies," never to learn again.
  * Not zero-centered: The outputs are always non-negative (0 or higher). This can slightly slow down the convergence of the optimizer.
* When to Use:
  * This is the default, go-to activation for all hidden layers. Start with ReLU.

***

**### Leaky ReLU**

This is a popular variant of ReLU designed to fix the "Dying ReLU" problem.

* How it Works:
  * If the input is positive, it passes the value through. $$ $f(x) = x$ $$
  * If the input is negative, it outputs a small, non-zero value. $$ $f(x) = \alpha \cdot x$ $$
  * (Where $$ $\alpha$ $$ is a small number like 0.01)
* Pros:
  * Fixes the Dying ReLU problem: Because the gradient for negative inputs is a small, non-zero $$ $\alpha$ $$ (e.g., 0.01), neurons can't get "stuck" at zero. They can always recover.
  * Keeps the fast computation and good convergence properties of ReLU.
* Cons:
  * Another hyperparameter: You now have to choose the value of $$ $\alpha$ $$ (though 0.01 is a common default).
  * _Slightly_ more complex to compute than ReLU.
* When to Use:
  * A common drop-in replacement for ReLU, especially if you suspect your model is suffering from dying neurons.

***

**### ELU (Exponential Linear Unit)**

This is another ReLU variant that aims to be a "best of all worlds" function.

* How it Works:
  * It's like ReLU for positive inputs.
  * For negative inputs, it becomes a smooth exponential curve that saturates at -1.
* Pros:
  * Fixes the Dying ReLU problem.
  * Zero-centered outputs: Unlike ReLU, its outputs can be negative. This (like Tanh) can help speed up training by pushing the average output of the layer closer to zero.
  * Smooth curve for negative inputs (unlike Leaky ReLU's sharp "corner").
* Cons:
  * Computationally slower: It involves calculating an exponential ($$ $e^x$ $$), which is much more demanding than ReLU's simple `max` operation.
* When to Use:
  * When you want the benefits of ReLU but also need zero-centered outputs. It often provides a good boost in performance (especially in Computer Vision) if you are willing to accept the slower training time.

***

**### Tanh (Hyperbolic Tangent)**

This was a popular "classic" activation function before ReLU.

* How it Works: It squashes any real-valued input into a range between -1 and 1.
* Pros:
  * Zero-centered: Its outputs are centered around 0. This is a key advantage over Sigmoid and can help the optimizer converge faster.
* Cons:
  * Vanishing Gradient Problem: Like Sigmoid, for very large positive or negative inputs, the function's slope becomes almost zero. This stops the gradient from flowing, making it very hard to train deep networks.
* When to Use:
  * Rarely used in hidden layers of modern feed-forward networks (like CNNs or MLPs). It has been almost completely replaced by the ReLU family.
  * It is still commonly used in Recurrent Neural Networks (RNNs) like LSTMs and GRUs, often as a gate.

***

**### Sigmoid (or Logistic)**

This is the _other_ "classic" function, famous for its "S" shape.

* How it Works: It squashes any real-valued input into a range between 0 and 1.
* Pros:
  * Interpretable output: The (0, 1) range is useful for representing a probability.
* Cons:
  * Vanishing Gradient Problem: Its main weakness. Like Tanh, the gradient is tiny for most inputs, making deep networks untrainable.
  * Not zero-centered: Outputs are always positive (0 to 1), which can slow training.
* When to Use:
  * ALMOST NEVER in hidden layers.
  * Its main use is in the output layer (see below).

***

#### 2. Functions for the Output Layer

These are special-purpose functions you use _only_ in the final layer of your network. The one you choose depends entirely on your problem.

**### Linear**

* How it Works: It's just $$ $f(x) = x$ $$. It does nothing.
* Pros:
  * Allows the network to output any real number (positive, negative, greater than 1, etc.).
* Cons:
  * None, for its intended purpose.
* When to Use:
  * For all Regression problems: When you are predicting a continuous value, like the price of a house, the age of a person, or the temperature.

***

**### Sigmoid**

* How it Works: Squashes the output to be between 0 and 1.
* When to Use:
  1. Binary Classification: Use a _single_ output neuron with a Sigmoid. The output (e.g., 0.8) can be interpreted as the probability of the "positive" class (e.g., 80% chance this email is "Spam").
  2. Multi-Label Classification: Use _N_ output neurons (one for each class) with a Sigmoid on _each one_. This allows the model to predict the probability for each class independently (e.g., a photo can be 90% "Cat", 85% "Dog", and 10% "Car").

***

**### Softmax**

* How it Works: Takes a vector of _N_ numbers (one for each class) and converts it into a probability distribution. This means each number will be between 0 and 1, and all _N_ numbers will sum up to 1.
* Pros:
  * Perfect for representing the probability of a single outcome among several possibilities.
* Cons:
  * Only suitable when classes are mutually exclusive (e.g., an image can be a "Cat" _or_ a "Dog," but not both at the same time).
* When to Use:
  * For all Multi-Class Classification problems: When you are predicting a single label from 3 or more classes (e.g., MNIST digit recognition, 0-9).

***

#### 🚀 Summary: How to Choose

Here is a simple rule-of-thumb:

1. For all your Hidden Layers:
   * Start with ReLU. It's fast, simple, and usually works very well.
   * If your model is slow to train or you suspect "Dying ReLU," switch to Leaky ReLU.
   * If you have extra compute time and want to try for a performance boost, try ELU.
   * _Never_ use Sigmoid or Tanh in hidden layers (unless you're using RNNs).
2. For your Output Layer:
   * Regression (predicting a number): Linear
   * Binary Classification (A or B): Sigmoid (1 neuron)
   * Multi-Class Classification (A or B or C...): Softmax (N neurons)
   * Multi-Label Classification (A and C, but not B...): Sigmoid (N neurons)

Would you like to see the math behind the vanishing gradient problem and how ReLU specifically solves it?
