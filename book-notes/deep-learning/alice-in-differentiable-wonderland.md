# Alice in differentiable wonderland

## Chapter 1&#x20;

of _Alice's Adventures in a Differentiable Wonderland_ offers an introduction to neural networks and discusses the power of scaling in modern artificial intelligence. Below are detailed study notes based on the content of this chapter:

#### 1. Introduction to Neural Networks

* Neural networks are now central to various technological advances, from language models to molecular design.
* Modern neural networks rely on a few fundamental principles that, when scaled (in terms of data, compute power, and model size), lead to significant performance improvements.

#### 2. Neural Scaling Laws

* **Scaling Laws**: As you increase data, compute power, and model size, the performance of neural networks improves predictably.
* This idea has driven massive investments in AI, as it allows researchers to focus on scaling models to achieve better results.
* The cost of training AI models has dramatically increased over time, with larger models demanding vast computational resources.

#### 3. Neural Networks as General-Purpose Tools

* Neural networks approximate probability distributions given data, and with enough data and compute, they generalize well even to tasks not explicitly in the training set.
* **Foundation Models**: Large, pre-trained models (like LLMs) can be fine-tuned for specific tasks, significantly reducing the need to build models from scratch.

#### 4. Open-Source AI and Democratization

* While large-scale models require significant resources to build, open-source communities like Hugging Face provide opportunities for customization, model merging, and experimentation on smaller scales.
* With the right tools, even consumer-grade hardware can be leveraged to explore differentiable models.

#### 5. Understanding Differentiable Models

* Differentiable models are essentially compositions of differentiable primitives (basic mathematical functions), which makes them suitable for optimization techniques like gradient descent.
* These models work primarily with tensors (multidimensional arrays of numbers), which are ideal for representing complex data like images, text, and audio.

#### 6. Designing Neural Networks

* The book emphasizes the importance of understanding what types of data can be handled by neural networks and what types of operations (primitives) can be applied to these data structures.
* Modern neural networks, despite their name, are very different from biological neurons and have evolved into purely mathematical constructs.
* The term "differentiable models" is preferred as it highlights the true nature of these systems, focusing on their mathematical foundations rather than biological metaphors.

#### 7. Overview of the Book's Structure

* **Part I: Compass and Needle**: This section covers the fundamentals of linear algebra, gradients, and how to optimize neural networks through gradient descent.
* **Part II: A Strange Land**: Here, the book dives into more advanced concepts like convolutional layers and techniques used in image, audio, and text processing.
* **Part III: Down the Rabbit-Hole**: This section explores modern innovations in neural networks, such as transformers, graph models, and recurrent networks.

#### 8. Historical Context and Future Outlook

* Neural networks have a long history dating back to the 20th century, with several waves of interest and breakthroughs.
* The rise of convolutional networks for vision tasks and transformers for language models has significantly shaped AI research.
* The book also touches on the history of terms like "deep learning" and "connectionism," emphasizing how these concepts have evolved.

#### 9. Key Takeaways

* Neural networks are powerful tools that can be scaled for a variety of tasks.
* Differentiable models rely on a small set of principles, and understanding their structure allows for creative experimentation and problem-solving.
* While the field is rapidly evolving, a strong foundation in the underlying mathematics (differentiation, gradient descent, etc.) is essential for navigating modern AI literature.

These notes encapsulate the main points of Chapter 1 and provide an understanding of the key concepts that will be expanded upon in subsequent chapters.



## Chapter 2: Mathematical Preliminaries (Detailed Study Notes)

Chapter 2 of _Alice's Adventures in a Differentiable Wonderland_ focuses on key mathematical concepts required for understanding differentiable models. It is divided into three sections: **Linear Algebra**, **Gradients and Jacobians**, and **Optimization**. Here's a detailed breakdown:

***

#### 1. **Linear Algebra**

This section establishes the foundational concepts necessary for deep learning, focusing on tensors and matrix operations.

* **Tensors (Definition D.2.1)**: Tensors are n-dimensional arrays, where:
  * Scalars are 0-dimensional tensors.
  * Vectors are 1-dimensional tensors.
  * Matrices are 2-dimensional tensors.
  * Tensors of higher dimensions are used in complex models.
  * Tensors play a pivotal role in deep learning as they facilitate parallel computations, especially on GPUs/TPUs.
* **Matrix Operations**: A matrix ( X ) can be seen as a stack of row vectors, and matrix multiplication allows for linear transformations. For example:
  * **Matrix-vector multiplication**: ( z = W \cdot x ) produces a new vector by applying the transformation ( W ).
  * **Matrix-matrix multiplication**: ( Z = X \cdot Y ) combines two matrices.
* **Other Operations**:
  * **Hadamard (Element-wise) Multiplication** (Definition D.2.4): Element-wise multiplication, used in various operations like masking in neural networks.
  * **Exponentials and Reductions**: Operations such as summation, exponentiation, and matrix exponentials are fundamental for manipulating tensors.
* **Computational Complexity**: An essential concept in optimizing matrix operations, where matrix multiplication has time complexity ( O(abc) ) for two matrices of dimensions ( (a, b) ) and ( (b, c) ).

***

#### 2. **Gradients and Jacobians**

This section introduces key concepts for differentiation, which is central to training machine learning models.

* **Derivatives and Chain Rule**:
  * Derivatives of basic functions (e.g., polynomials, logarithms, sine) are introduced geometrically as slopes of tangents at points on the function's graph.
  * **Chain Rule**: The derivative of a composite function is the product of the derivatives of the individual functions: \[ \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x) ]
* **Gradients (Definition D.2.6)**: The gradient is the vector of partial derivatives for multi-variable functions. It points in the direction of the steepest ascent of the function.
* **Directional Derivatives**: The rate of change of a function in any given direction can be computed as the dot product between the gradient and the direction vector.
* **Jacobians (Definition D.2.7)**: For vector-valued functions, the Jacobian matrix contains partial derivatives of each component function. It generalizes the gradient for functions with multiple outputs.

***

#### 3. **Optimization**

The final section covers methods to find the minimum of a function, which is essential for model training.

* **Gradient Descent**: A basic algorithm used to optimize functions by iteratively moving in the direction opposite to the gradient (the steepest descent):
  * ( x\_{n+1} = x\_n - \eta \nabla f(x\_n) )
  * Here, ( \eta ) is the learning rate, controlling the step size.
* **Gradient Descent Exercise**: Implementing this algorithm in NumPy, JAX, and PyTorch is suggested as practice. The exercise covers:
  1. Vectorizing the function to compute over a batch of inputs.
  2. Manually coding the gradient computation.
  3. Visualizing the gradient descent paths.
  4. Adding momentum to the gradient descent algorithm.

***

#### Key Takeaways:

* Understanding tensors, matrix operations, and their computational complexities is crucial for working with deep learning models.
* Gradients and Jacobians are fundamental for optimization, which is the core of model training.
* Implementing gradient descent in various frameworks helps in solidifying these concepts.

These notes provide a concise overview of Chapter 2's mathematical background, setting up the tools and techniques needed for more complex topics in neural networks and differentiable programming.



## Chapter 3: Datasets and Losses (Detailed Study Notes)

Chapter 3 of _Alice's Adventures in a Differentiable Wonderland_ introduces foundational concepts in supervised learning, focusing on datasets, loss functions, empirical risk minimization, and a probabilistic approach to learning. Below are detailed study notes:

***

#### 1. **Datasets in Supervised Learning**

* **Definition of Dataset (D.3.1)**: A dataset ( S\_n ) in supervised learning consists of ( n ) pairs ( (x\_i, y\_i) ), where ( x\_i ) is an input, and ( y\_i ) is the corresponding output.
  * These pairs are assumed to be independent and identically distributed (i.i.d.) samples from an unknown probability distribution ( p(x, y) ).
  * The i.i.d. assumption is crucial for supervised learning models to generalize well. However, in real-world data collection, ensuring that data are i.i.d. is not trivial, as factors such as temporal shifts in data distribution (e.g., evolving car models in image recognition tasks) can invalidate this assumption.

#### 2. **Loss Functions**

* **Definition of Loss Function (D.3.2)**: A loss function ( l(y, \hat{y}) ) measures the difference between the true label ( y ) and the model's prediction ( \hat{y} ). It is a differentiable scalar function designed to quantify prediction errors.
  * **Example**: A smaller loss value indicates better model performance.
  * **Types of Losses**: Squared loss, absolute loss, and hinge loss are examples of different loss functions, each suited for different tasks such as regression or classification.

**2.1. Empirical Risk Minimization (ERM)**

* The goal of ERM is to minimize the average loss over a dataset. The optimization objective is expressed as: \[ f^\* = \text{argmin}_f \frac{1}{n} \sum_{i=1}^n l(y\_i, f(x\_i)) ]
  * This is referred to as **empirical risk** because it is the loss calculated on the available training data. The true goal, however, is to minimize the loss over unseen future data, which leads to the concept of **expected risk**.

#### 3. **Expected Risk and Generalization**

* **Definition of Expected Risk (D.3.3)**: The expected risk is the theoretical average loss over all possible input-output pairs, not just those in the training dataset: \[ ER\[f] = \mathbb{E}\_{p(x,y)}\[l(y, f(x))] ]
  * Since calculating the expected risk is generally impractical (requiring knowledge of the entire data distribution), empirical risk serves as an approximation.
  * **Generalization Gap**: The difference between the empirical risk and the expected risk is known as the **generalization gap**. A model that minimizes empirical risk but performs poorly on new data is said to **overfit** the training data. This is often tested by evaluating model performance on a separate test set.

#### 4. **Probabilistic Formulation and Maximum Likelihood**

* A more general view of supervised learning comes from a probabilistic standpoint. In this view:
  * Each output ( y ) is seen as a sample from a probability distribution ( p(y | x) ) conditioned on the input.
  * **Maximum Likelihood Estimation (MLE)**: The model is trained to maximize the likelihood of observing the dataset ( S\_n ): \[ f^\* = \text{argmax}_f \prod_{i=1}^n p(y\_i | f(x\_i)) ]
    * Taking the log of the likelihood, this can be reframed as minimizing the negative log-likelihood, which can be interpreted as a "pseudo-loss" function: \[ f^\* = \text{argmin}_f \sum_{i=1}^n -\log p(y\_i | f(x\_i)) ]

#### 5. **Bayesian Learning**

* In **Bayesian Neural Networks (BNNs)**, we go beyond maximum likelihood by assigning a probability distribution ( p(f) ) over the model functions themselves, known as the **prior**. Once data is observed, the distribution is updated to a **posterior** distribution over functions via Bayes’ theorem: \[ p(f | S\_n) = \frac{p(S\_n | f) p(f)}{p(S\_n)} ]
  * Bayesian methods allow the model to capture uncertainty by averaging over many possible functions, rather than selecting a single best function. This is useful when multiple models fit the data well, helping to make more robust predictions.
  * **Maximum A Posteriori (MAP)**: This combines prior information and likelihood into a regularized solution. The MAP estimate is: \[ f^\* = \text{argmax}\_f \left( \log p(S\_n | f) + \log p(f) \right) ]
    * This introduces a regularization term, encouraging simpler or more stable models depending on the choice of the prior.

***

#### Key Takeaways:

* Datasets in supervised learning are formed by pairs of inputs and outputs, with assumptions about their distribution (i.i.d.) crucial for model training.
* Loss functions are essential for quantifying prediction errors and guiding the optimization of machine learning models.
* Empirical risk minimization is the primary objective in training models, though care must be taken to ensure that models generalize well to unseen data.
* The probabilistic formulation of supervised learning, particularly Bayesian methods, provides a more flexible approach to model training by handling uncertainties and regularizing solutions to prevent overfitting.

These concepts serve as the foundational backbone for the subsequent chapters, laying out the essential principles of supervised learning and model optimization .



## Chapter 4: Linear Models (Detailed Study Notes)

Chapter 4 of _Alice's Adventures in a Differentiable Wonderland_ delves into linear models, starting with regression and progressing to classification using logistic regression. It introduces key concepts like least-squares regression, logistic regression, and loss functions. Below are detailed study notes based on this chapter:

***

#### 1. **Least-Squares Regression**

This is the foundational linear model introduced in the chapter, where the task is to predict a continuous output.

* **Linear Models (Definition D.4.1)**: A linear model on an input ( x ) is defined as: \[ f(x) = w^\top x + b ] where ( w ) is the weight vector, and ( b ) is the bias term.
  * **Geometrical Interpretation**: For one feature, the model represents a line, for two features a plane, and for more than two, a hyperplane.
* **Least-Squares Problem (Definition D.4.2)**: The least-squares optimization problem minimizes the squared error between the model's predictions and the true outputs. It can be expressed as: \[ \min\_{w, b} \frac{1}{n} \sum\_{i=1}^{n} \left( y\_i - w^\top x\_i - b \right)^2 ]
* **Gradient Descent**: Gradient descent is used to solve the least-squares problem. The gradient of the least-squares loss is linear in the model's parameters, which simplifies optimization.
* **Closed-form Solution**: In certain cases, least-squares regression has a closed-form solution. The weights can be computed as: \[ w^\* = \left( X^\top X \right)^{-1} X^\top y ] This is a more direct solution than iterative gradient descent.

**Regularization:**

* **Ridge Regression (Regularized Least-Squares)**: When the matrix ( X^\top X ) is close to singular, adding a regularization term helps stabilize the solution. This gives rise to ridge regression, where the optimization problem becomes: \[ \min\_{w} \frac{1}{n} \sum\_{i=1}^{n} \left( y\_i - w^\top x\_i \right)^2 + \lambda |w|^2 ]

***

#### 2. **Loss Functions**

Loss functions are essential to training machine learning models by guiding optimization.

* **Squared Loss (E.4.1)**: The squared loss is used in regression problems to measure the difference between the predicted and actual values: \[ l(\hat{y}, y) = (\hat{y} - y)^2 ]
* **Other Loss Functions**:
  * **Absolute Loss**: More robust to outliers compared to squared loss: \[ l(\hat{y}, y) = |\hat{y} - y| ]
  * **Huber Loss**: A combination of squared and absolute loss: \[ L(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq 1 \ |y - \hat{y}| - \frac{1}{2} & \text{otherwise} \end{cases} ]

***

#### 3. **Classification: Logistic Regression**

The chapter moves on to classification, where the output is a class label rather than a continuous value.

* **Logistic Regression**:
  * The model predicts a probability between 0 and 1 by passing a linear combination of inputs through the **sigmoid function**: \[ f(x) = \sigma(w^\top x + b) = \frac{1}{1 + \exp(-w^\top x - b)} ]
  * **Binary Classification**: In binary classification, the model outputs a probability ( \hat{y} \in \[0, 1] ), and the predicted class is: \[ \hat{y} = \begin{cases} 1 & \text{if } f(x) > 0.5 \ 0 & \text{otherwise} \end{cases} ]

**Loss Function for Logistic Regression:**

* **Cross-Entropy Loss (E.4.17)**: For binary classification, cross-entropy is used to measure the discrepancy between predicted and actual class probabilities: \[ CE(\hat{y}, y) = - y \log(\hat{y}) - (1 - y) \log(1 - \hat{y}) ] Minimizing this loss ensures the model outputs probabilities close to the true class labels.

**Softmax for Multiclass Classification:**

* For multi-class classification, the **softmax function** generalizes logistic regression. It converts raw scores (logits) into probabilities: \[ \text{softmax}(x)_i = \frac{\exp(x\_i)}{\sum_{j} \exp(x\_j)} ]
  * The softmax output represents a probability distribution over all possible classes, and the predicted class is the one with the highest probability.

***

#### Key Takeaways:

* **Linear models** form the backbone of many machine learning methods, with least-squares regression being the simplest case for continuous outcomes.
* **Loss functions** like squared, absolute, and cross-entropy losses help in model training by quantifying errors.
* **Logistic regression** extends linear models to classification tasks, with the sigmoid function used for binary outcomes and the softmax function for multi-class problems.
* **Regularization** techniques, like ridge regression, help avoid overfitting and stabilize model training in cases of multicollinearity.

These concepts provide a strong mathematical foundation for understanding more complex models covered in later chapters.



## Chapter 5: Fully-Connected Models (Detailed Study Notes)

Chapter 5 of _Alice's Adventures in a Differentiable Wonderland_ explores fully-connected models, also known as multilayer perceptrons (MLPs). It builds upon linear models and introduces non-linearities through activation functions, hidden layers, and stochastic optimization. Below are the key topics covered in this chapter:

***

#### 1. **Limitations of Linear Models**

* Linear models, by definition, can only capture linear relationships in data. This limitation prevents them from solving problems like the **XOR problem**, where the input data is not linearly separable.
* For example, changing one feature (e.g., income) of an input vector ( x ) results only in a proportional change in the output in a linear model. Non-linear models are required to handle more complex relationships between features, such as interactions or thresholds (e.g., income being low unless age < 30).

***

#### 2. **Fully-Connected (FC) Models**

* **Multilayer Perceptrons (MLPs)**: MLPs overcome the limitations of linear models by composing multiple layers of transformations. The core idea is to add **hidden layers**, where each layer applies a transformation to the input before passing it to the next layer.
* **Composition of Functions**: The model ( f(x) ) is the composition of several functions, where each function introduces its own set of parameters. For example: \[ f(x) = (f\_l \circ f\_{l-1} \circ \cdots \circ f\_1)(x) ]
  * In each layer, a matrix multiplication ( W\_ix + b\_i ) is followed by a non-linearity, preventing the collapse of layers into a single linear transformation. Without this non-linearity, the stacked layers would reduce to a single linear model.

***

#### 3. **Activation Functions**

* Non-linearities (activation functions) are introduced between the linear transformations to ensure that the model can learn non-linear relationships.

**Common Activation Functions:**

* **ReLU (Rectified Linear Unit)**: Defined as ( \text{ReLU}(s) = \max(0, s) ), it sets negative inputs to zero, introducing sparsity and computational efficiency.
* **Leaky ReLU**: A variation of ReLU that allows a small gradient for negative inputs to avoid "dead neurons."
* **Softplus**: A smooth approximation of ReLU, defined as ( \text{Softplus}(s) = \log(1 + \exp(s)) ), which prevents discontinuities at zero.
* **ELU (Exponential Linear Unit)**: Similar to ReLU but with a smooth transition for negative inputs, helping to avoid dead neurons and improve gradient flow.
* **Universal Approximation Theorem**: A seminal result that states any continuous function can be approximated to arbitrary precision by an MLP with a single hidden layer and non-linear activation functions like sigmoid. However, the width of the hidden layer may need to be large.

***

#### 4. **Stochastic Optimization**

* Training deep networks with large datasets can be computationally expensive. Gradient descent on the full dataset may be slow and memory-intensive.
* **Stochastic Gradient Descent (SGD)**: A common optimization technique where, instead of computing the gradient on the entire dataset, gradients are estimated using smaller subsets called **mini-batches**. This reduces memory usage and increases the frequency of updates.
* **Mini-batch SGD**: The dataset is divided into mini-batches, each containing a subset of examples. The loss is computed for each mini-batch, and the weights are updated after every mini-batch instead of after the entire dataset.
  * **Epoch**: One complete pass through the dataset is called an epoch. Multiple epochs are typically required to train the model fully.

***

#### 5. **Training with GPUs**

* Modern frameworks like PyTorch and TensorFlow allow for parallelism, enabling multiple mini-batches to be processed across different GPUs. The gradients computed on each device are aggregated, making the training process faster for large models and datasets.

***

#### Key Takeaways:

* **Fully-connected models** introduce hidden layers and non-linear activation functions to overcome the limitations of linear models.
* **Activation functions** like ReLU and its variants play a critical role in the ability of neural networks to model non-linear relationships.
* **Stochastic optimization** techniques like mini-batch SGD make it feasible to train large models efficiently on big datasets.
* **Parallelism** across GPUs or machines helps accelerate the training process by distributing the computation of gradients across multiple devices.

These notes summarize the theoretical foundation and practical considerations of training fully-connected models, setting the stage for more advanced neural network architectures discussed in later chapters.



## Chapter 6: Automatic Differentiation (Detailed Study Notes)

Chapter 6 of _Alice's Adventures in a Differentiable Wonderland_ introduces the essential concept of **automatic differentiation (AD)**, a core technique in training differentiable models. This chapter provides a comprehensive explanation of forward-mode and reverse-mode AD, explains their implementation, and compares them to other differentiation methods.

***

#### 1. **Problem Setup and Evaluation Trace**

* **Evaluation Trace**: In any differentiable program, an evaluation trace captures the sequence of operations. For example, in neural networks, it involves applying layers of transformations, followed by a loss function, which produces a scalar output. This scalar is critical for optimization through AD.
* **Definition of AD (D.6.1)**: AD is the process of computing derivatives (gradients) of the output concerning each model parameter. The key is to simultaneously and efficiently compute the Jacobians (matrices of partial derivatives) using the structure of the computational graph, i.e., how inputs, weights, and operations are interconnected.

***

#### 2. **Types of Differentiation**

* **Numerical Differentiation**: This method directly approximates the derivative using finite differences. It is simple but inefficient because it requires multiple function calls for each derivative, making it impractical for large models.
* **Symbolic Differentiation**: Symbolic engines like SymPy can pre-compute exact derivatives of functions symbolically. However, this method can generate unnecessarily complex expressions (e.g., redundant computations), which makes it inefficient for large-scale models.
* **Automatic Differentiation (AD)**: Unlike the above, AD exploits the structure of the program to compute gradients efficiently. The two main modes of AD are **forward-mode** and **reverse-mode**, both of which apply the chain rule of differentiation but in different orders.

***

#### 3. **Forward-Mode Automatic Differentiation**

* **Forward-Mode (F-AD)**: This mode computes the gradient by updating the Jacobians as the program progresses from the input to the output.
* **Process**:
  1. Each primitive function in the program initializes its Jacobian (called the **tangent**).
  2. As each new function is applied, the Jacobians are updated.
  3. The result is a gradient of the output concerning all inputs.
* **Limitations**: Forward-mode is efficient when the number of inputs (parameters) is small but becomes computationally expensive as the number of outputs grows, making it suboptimal for neural networks.

***

#### 4. **Reverse-Mode Automatic Differentiation (Backpropagation)**

* **Reverse-Mode (R-AD)**: Commonly referred to as **backpropagation** in neural networks, reverse-mode computes the gradient by starting at the output and working backward through the computational graph.
* **Process**:
  1. The program is first executed in its entirety, and all intermediate values are stored.
  2. A backward pass starts from the output, recursively applying the chain rule in reverse order to compute the gradients concerning each parameter.
  3. The result is an efficient computation of gradients for models with large numbers of parameters but only a single scalar output (like in neural networks).
* **Advantages**: Reverse-mode is much more efficient than forward-mode for neural networks because it scales well with the number of parameters.

***

#### 5. **Key Considerations**

* **Memory Trade-offs**: Reverse-mode AD requires storing all intermediate values from the forward pass, leading to high memory usage. Techniques like **gradient checkpointing** can help by reducing memory consumption at the cost of recomputing some intermediate values during the backward pass.
* **Vector-Jacobian Products (VJPs)**: Reverse-mode AD relies on vector-Jacobian products, where a row vector is multiplied by a Jacobian matrix. This operation scales efficiently and is key to the backpropagation process in neural networks.

***

#### 6. **Practical Implementations**

* **PyTorch and JAX**: Both frameworks implement AD by augmenting tensors with information about the operations that generated them. This allows the automatic computation of gradients using the reverse-mode algorithm.
* **Extensibility**: Developers can implement custom primitives in frameworks like PyTorch, specifying both their forward pass and the corresponding backward pass to ensure compatibility with AD.

***

#### 7. **Comparison of Differentiation Techniques**

* **Forward-mode AD** is best suited for models with few inputs and many outputs, such as in certain scientific computations.
* **Reverse-mode AD** is highly efficient for models like neural networks, which have many parameters (inputs) but a single scalar output.

***

#### Key Takeaways:

* Automatic differentiation is essential for training modern machine learning models, especially deep networks.
* Reverse-mode AD (backpropagation) is the most efficient technique for large-scale models like neural networks.
* Practical implementations of AD in frameworks like PyTorch and JAX simplify the process of building and training differentiable models.

These study notes summarize the key ideas from Chapter 6, which lay the groundwork for understanding backpropagation and its role in modern deep learning systems.



## Chapter 7: Convolutional Layers (Detailed Study Notes)

In _Alice's Adventures in a Differentiable Wonderland_, Chapter 7 introduces **convolutional layers**, a key component in many neural networks, particularly those designed for image processing. Below are the detailed notes based on this chapter:

***

#### 1. **Introduction to Convolutional Layers**

* **Purpose**: Convolutional layers are designed to process image data (or any sequential data) by utilizing the concepts of **locality** and **parameter sharing**.
  * **Locality**: The idea that pixels in an image close to each other are likely to share meaningful information.
  * **Parameter Sharing**: A single filter (set of weights) is used across different parts of the input image, enabling efficient learning by reducing the number of unique parameters.
* **Input Representation**: An image can be represented as a tensor ( X ) of shape ( (h, w, c) ), where ( h ) and ( w ) represent height and width, and ( c ) represents the number of channels (e.g., 3 for RGB images). When working with batches of images, an additional dimension ( b ) is added for the batch size, resulting in a tensor of shape ( (b, h, w, c) ).

***

#### 2. **Convolution Operation**

* **Filter Application**: A filter (or kernel) is a small matrix that moves across the input image, applying element-wise multiplication to local regions of the image (called patches) and summing the results. This produces an output feature map that highlights certain patterns (like edges) in the image.
* **Mathematical Representation**: \[ H\_{i,j,z} = \sum\_{i', j', d} W\_{i', j', z, d} \cdot X\_{i' + t(i), j' + t(j), d} ] Here, ( W ) is the weight tensor representing the filter, and ( t(i) ) accounts for the shifting of the filter over the image. This equation describes how a filter with weights ( W ) is applied to a patch of the input ( X ).

***

#### 3. **Receptive Field**

* **Definition**: The receptive field ( R(i, j) ) of a pixel in the output feature map is the portion of the input image that contributed to the computation of that output. For a single convolutional layer, the receptive field corresponds to a small patch of the input image.
  * For deeper networks with multiple layers, the receptive field increases linearly as more layers are added, eventually covering larger regions of the image.
  * This is particularly useful for capturing broader and more complex features as we stack more convolutional layers.

***

#### 4. **Max-Pooling**

* **Purpose**: Max-pooling is used to downsample the input by reducing its spatial dimensions while retaining the most important features.
  * A **max-pooling layer** takes small windows of the input and returns the maximum value within each window.
  * **Example**: For an input tensor ( X ) of shape ( (h, w, c) ), a max-pooling operation with a 2x2 window reduces the shape to ( (h/2, w/2, c) ).
* **Advantages**:
  * Max-pooling introduces **translation invariance**, meaning that small shifts in the input image do not significantly affect the output.
  * It also helps reduce computational complexity by shrinking the input size for subsequent layers.

***

#### 5. **Designing Convolutional Networks (CNNs)**

* **Building Blocks**: A **convolutional block** typically consists of multiple convolutional layers followed by activation functions (like ReLU) and pooling operations. These blocks can be stacked to form deep CNN architectures.
* **Complete Model Design**: A typical CNN for image classification consists of:
  1. **Convolutional Blocks**: These extract features from the input image.
  2. **Global Pooling**: After the convolutional layers, a global pooling layer (e.g., global average pooling) aggregates the spatial information into a single feature vector.
  3. **Classification Head**: A fully-connected layer (or multiple) is used at the end to map the feature vector to class probabilities, typically using softmax.

***

#### 6. **Invariance and Equivariance**

* **Translation Equivariance**: Convolutional layers are **equivariant** to translations, meaning that shifting the input results in a corresponding shift in the output. This is a key advantage in image recognition tasks, where objects may appear in different locations.
* **Translation Invariance**: Operations like max-pooling introduce **translation invariance**, meaning that small translations in the input do not affect the output classification.

***

#### 7. **Advanced Types of Convolutions**

* **1x1 Convolutions**: These involve using a filter of size ( 1 \times 1 ), which operates on the channel dimension alone, without considering neighboring pixels. This is useful for adjusting the number of channels without affecting the spatial dimensions.
* **Depthwise Separable Convolutions**: These convolutions separate spatial and channel-wise operations, significantly reducing the number of parameters while retaining model performance. They are commonly used in lightweight models designed for mobile devices.

***

#### Key Takeaways:

* Convolutional layers are the cornerstone of deep learning models for image processing, with their ability to capture local patterns and generalize across spatial locations.
* Max-pooling reduces the spatial dimensions, making the model more computationally efficient while introducing translation invariance.
* Designing CNNs involves stacking convolutional blocks, pooling operations, and fully-connected layers for classification.

These concepts are fundamental in understanding how modern convolutional neural networks (CNNs) process visual data efficiently, providing a foundation for more advanced architectures covered in later chapters.



## Chapter 8: Convolutions Beyond Images (Detailed Study Notes)

Chapter 8 of _Alice's Adventures in a Differentiable Wonderland_ focuses on extending convolutional models beyond image processing to handle other types of data like 1D and 3D sequences. This chapter highlights how convolutional layers can be applied to time series, audio, text, and video, and it introduces concepts like forecasting, causal models, and generative models.

***

#### 1. **Convolutions for 1D and 3D Data**

* **1D Convolutions**: These are particularly useful for time-series data (e.g., stock prices, sensor readings) and audio signals, which are represented as a 1D sequence. A 1D convolution applies a filter along the time dimension to capture temporal patterns.
  * **Input Representation**: A 1D input sequence is represented as a matrix (X \sim (t, c)), where (t) is the length of the sequence and (c) is the number of channels (e.g., sensor measurements).
* **3D Convolutions**: These are used for processing volumetric data like videos or 3D medical scans. The convolutional operation is extended to three dimensions: height, width, and depth (or time for videos).
  * **Input Representation**: A video is represented as a rank-4 tensor (X \sim (t, h, w, c)), where (t) is the number of frames, (h) and (w) are the height and width of each frame, and (c) is the number of channels (e.g., RGB channels).

***

#### 2. **Handling Variable-Length Inputs**

* **Zero Padding**: When working with variable-length sequences (e.g., different audio files of varying duration), a common strategy is to pad the shorter sequences with zeros to match the longest sequence in the mini-batch. This allows for efficient mini-batch processing and vectorization.
* **Global Average Pooling**: To handle the varying lengths of the sequences, convolutional models often use global average pooling, which aggregates the information across the entire sequence, making the length of the sequence irrelevant after the convolutional layers.

***

#### 3. **Dilated Convolutions**

* **Dilated Convolutions**: Introduced to extend the receptive field without increasing the number of parameters, dilated convolutions are useful for processing long sequences. By skipping input elements during the convolution, dilated convolutions exponentially increase the receptive field with each layer.
  * **Application**: This method is popular in models like WaveNet for audio generation, where long-range dependencies are essential.

***

#### 4. **Forecasting and Causal Models**

* **Forecasting**: Predicting the future values of a sequence based on its history (e.g., energy prices, traffic flow) is a key application of time-series models. The goal is to train a model to predict the next element in a sequence using past elements as input. This task is known as **autoregressive forecasting**.
  * **Self-supervised Learning**: Forecasting can be seen as a form of self-supervised learning since the target values are derived directly from the input sequence without requiring explicit labels.
* **Causal Models**:
  * **Causal Layers**: A layer is considered **causal** if the output at each position depends only on the current and past input values, not future ones. This constraint is essential for time-series forecasting and language modeling, where future information should not influence current predictions.
  * **Masking in Convolutions**: To make a convolutional layer causal, the filter is masked to ensure that future input values are not used in the computation of the output. This masking technique is critical for autoregressive models.

***

#### 5. **Generative Models**

* **Autoregressive Models**: These models predict each token (or sequence element) one step at a time, using previous outputs as inputs for the next step. For example, in text generation, a model might start with a single token and generate the rest of the sentence step by step.
  * **Teacher Forcing**: During training, models are often trained using the true values as inputs (instead of the predicted values), a method known as **teacher forcing**. As training progresses, predicted values can gradually replace true values to make the model more robust to errors in its own predictions.
* **Applications**: Autoregressive models are widely used in natural language processing (e.g., language models) and audio generation (e.g., WaveNet).

***

#### Key Takeaways:

* **1D and 3D Convolutions** extend the application of convolutional models to various data types beyond images, such as audio, time series, and videos.
* **Dilated Convolutions** allow for processing longer sequences by expanding the receptive field while maintaining a manageable number of parameters.
* **Causal Models** are crucial for tasks like time-series forecasting and autoregressive text generation, ensuring that future inputs do not affect the current predictions.
* **Generative Models** leverage autoregressive structures to generate new sequences by predicting each element step-by-step.

These notes provide a comprehensive overview of Chapter 8, emphasizing how convolutional models can be adapted for diverse data types and tasks beyond image processing.



## Chapter 9: Scaling Up the Models (Detailed Study Notes)

Chapter 9 of _Alice's Adventures in a Differentiable Wonderland_ explores techniques used to stabilize and improve the training of very large models. This chapter highlights methods that are fundamental in modern machine learning, particularly for deep neural networks. Below are the key concepts covered in this chapter:

***

#### 1. **The ImageNet Challenge**

* The chapter begins by revisiting the **ImageNet Large Scale Visual Recognition Challenge**, which played a pivotal role in the development of deep convolutional models. Early models (pre-2012) used traditional methods like **linear kernels** combined with image descriptors, achieving error rates above 25%.
* **AlexNet (2012)**, with its deep convolutional layers trained using **gradient descent**, revolutionized the field, reducing top-5 error to 15.3%. This sparked a shift toward convolutional models, driving advancements in neural network architectures up to 2017【38:10†source】.

***

#### 2. **Techniques to Stabilize Training**

Scaling models beyond a few layers introduces several challenges, such as **slow optimization**, **gradient issues**, and **numerical instabilities**. Various techniques developed during 2012-2017 help mitigate these issues:

**2.1. Weight Regularization**

* **L2 Regularization**: Also known as weight decay, L2 regularization penalizes large weights, effectively controlling overfitting by adding a regularization term to the loss function: \[ L\_{reg} = L(w, S\_n) + \lambda |w|^2 ] Here, ( \lambda ) controls the regularization strength【38:16†source】.

**2.2. Data Augmentation**

* **Data Augmentation** involves transforming the dataset to generate new training examples, increasing model robustness. Examples include **rotations**, **cutmix** (stitching images), and **mixup** (linear interpolation between two images)【38:19†source】.

***

#### 3. **Dropout and Normalization**

**3.1. Dropout**

* Dropout is a regularization technique that randomly "drops out" units (neurons) during training by setting them to zero. This forces the network to learn redundant representations, which enhances robustness and reduces overfitting.
  * **Training**: A binary mask is sampled from a Bernoulli distribution to determine which units to drop.
  * **Inference**: The output is rescaled by multiplying by the drop probability ( p ) to account for the dropped units during training. This technique is called **inverted dropout**【38:2†source】【38:16†source】.

**3.2. Batch Normalization (BN)**

* BN normalizes the activations of each layer across the mini-batch to have zero mean and unit variance. This reduces internal covariate shift, leading to more stable gradients and faster training.
  * **Training**: BN normalizes the outputs and then re-scales them with two trainable parameters (mean ( \beta ) and variance ( \alpha )).
  * **Inference**: The mean and variance computed during training are used for normalization【38:12†source】【38:13†source】【38:14†source】.
* **Layer Normalization (LN)** and **Group Normalization** are variants of BN that operate on different subsets of dimensions. **Root Mean Square Normalization (RMSNorm)** simplifies layer normalization by removing mean centering【38:14†source】【38:17†source】.

***

#### 4. **Residual Connections**

* **Residual Blocks**: In deep networks, increasing the depth often leads to optimization issues, where deeper models perform worse than shallower ones due to gradient vanishing or exploding. **Residual connections** (also known as skip connections) mitigate this by allowing the input to bypass one or more layers: \[ r(x) = f(x) + x ]
  * This forces the model to learn residuals (deviations from the identity) instead of the entire transformation, making it easier for the model to learn the identity function when needed【38:7†source】【38:0†source】.
  * **ResNet (Residual Networks)**: A deep neural network composed of residual blocks. ResNets have been shown to scale up to hundreds of layers and perform well on a wide range of tasks【38:7†source】.

***

#### 5. **Residual Block Design**

* The residual block can include multiple operations, such as batch normalization, convolution, and activation (ReLU). The design has evolved, with modern architectures using **bottleneck layers** (as in ResNeXt), which reduce the number of parameters while increasing the model's receptive field【38:17†source】【38:7†source】.

***

#### 6. **Scaling Challenges**

* While scaling models generally improves accuracy, doing so without addressing optimization and stability challenges can lead to diminishing returns. Therefore, a combination of techniques, including dropout, normalization, and residual connections, is essential for building scalable and robust deep learning models【38:18†source】.

***

#### Key Takeaways:

* **Dropout** and **batch normalization** are widely used to regularize deep models and stabilize training.
* **Residual connections** allow for deep architectures like ResNets, making them easier to train by mitigating vanishing/exploding gradient problems.
* Scaling up models requires balancing depth with proper regularization and stabilization techniques to maintain efficiency and accuracy.

These techniques are now foundational in training very deep neural networks and continue to influence modern architectures such as transformers.



## Chapter 10, _Transformer Models_

#### Overview of Transformer Models

* Transformers are introduced as a class of models overcoming limitations of convolutional neural networks (CNNs) and recurrent neural networks (RNNs), especially in handling long sequences and non-local dependencies.
* This chapter highlights how transformers can process sequences like text, images, and other data due to their data-agnostic architecture, which allows them to handle various data types using appropriate tokenizers .

#### 10.1 Long Convolutions and Non-Local Models

* Traditional convolutional models struggle with capturing long-range dependencies because of limited receptive fields. Transformers address this issue using attention mechanisms.
* The transformer model was developed in response to challenges in handling sequences efficiently, particularly for natural language processing (NLP). It extends beyond text to images, time series, and graphs, largely due to its scalability with large datasets .
* A detailed explanation of attention mechanisms is provided, emphasizing how tokens in sequences (e.g., words in sentences) relate to each other semantically, even when far apart.

#### 10.2 Positional Embeddings

* Since transformers do not inherently understand sequence order, positional embeddings are essential. These embeddings allow the model to learn the position of each token in a sequence.
* Several types of embeddings are discussed:
  * **Absolute embeddings** (both trainable and sinusoidal) that encode specific positions.
  * **Relative positional embeddings** are described for very long sequences, modifying the attention mechanism based on token offsets .

#### 10.3 Building the Transformer Model

* A typical transformer model is composed of a series of multi-head attention (MHA) and feedforward layers, with the MHA mechanism allowing the model to weigh the relevance of other tokens to each target token.
* The model components include:
  1. **Tokenization and Embedding**: Input sequences are embedded into vectors.
  2. **Positional Embeddings**: These are added to token embeddings to indicate their positions in the sequence.
  3. **Transformer Blocks**: Each block consists of MHA and a multi-layer perceptron (MLP) layer, which can be implemented with either pre-normalization or post-normalization approaches. Pre-normalization is noted to improve training stability and efficiency .

#### Class Tokens and Register Tokens

* Transformers allow the addition of auxiliary tokens like the **class token** for classification tasks, where information can be compressed.
* **Register tokens** provide additional storage for information that may not be linked to specific input tokens but is helpful for downstream tasks, improving model flexibility .

#### Key Takeaways

* The transformer model revolutionized sequence modeling by allowing parallel computation through attention mechanisms, unlike RNNs, which are sequential.
* Transformers are not restricted by sequence length due to their attention-based design, which can handle dependencies across long input sequences efficiently.
* The architecture’s scalability and data-agnostic properties have made transformers the go-to model for a wide range of tasks beyond NLP, including computer vision and multimodal tasks .

These notes encapsulate the core ideas presented in Chapter 10, which provides a foundational understanding of how transformer models work, their architectural components, and their applications.



## Chapter 11, _Transformers in Practice_

#### Overview of Transformer Applications

Chapter 11 discusses practical applications of transformers beyond the standard model, covering:

1. Encoder-decoder architectures.
2. Causal multi-head attention (MHA).
3. Cross-attention.
4. Variants used in different domains, such as images and audio.

#### 11.1 Encoder-Decoder Transformers

* **Sequence-to-Sequence Tasks**: Transformers are highly effective for tasks where both input and output are sequences, like machine translation.
* **Encoder-Decoder Structure**: The encoder processes the input sequence into a representation, and the decoder uses this representation to generate an output sequence. The decoder’s autoregressive nature allows it to predict tokens one at a time, conditioning on previous ones.

**11.1.1 Causal Multi-Head Attention**

* **Causality in Attention**: To make MHA causal, the interaction between tokens must follow a strict temporal order. A masking matrix is applied during softmax to prevent tokens from attending to future tokens.
* **Masking Implementation**: The upper triangular matrix used in masking has negative infinite values in positions where future tokens should be ignored, ensuring proper autoregressive properties【8:0†source】.

**11.1.2 Cross-Attention**

* **Cross-attention**: An important component in encoder-decoder models, cross-attention allows the decoder to focus on specific parts of the encoder’s output. This is done by calculating attention using keys and values from the encoder and queries from the decoder.
* **Mechanism**: The cross-attention mechanism is formally written as \[ CA(X, Z) = softmax(\frac{XW\_q (Z W\_k)^T}{\sqrt{d\_k\}}) Z W\_v ], where ( Z ) contains the encoder outputs, while ( X ) is the decoder’s input sequence【8:15†source】.

#### 11.2 Computational Considerations

* **Time and Memory Complexity**: Attention’s quadratic complexity, in both time and memory, has prompted the development of various techniques to optimize large-scale transformer models.
  * **Linear Attention**: Some transformer models employ linear or sub-quadratic time complexity attention mechanisms by using fixed sets of latent tokens or chunking the attention computation【8:16†source】.
  * **FlashAttention**: This efficient implementation of MHA significantly reduces memory usage, especially useful when dealing with long sequences, by processing the attention matrix in chunks and reusing computations where possible【8:5†source】.
* **KV Caching**: In autoregressive generation, transformers store previously calculated key-value pairs in a KV cache, reducing the number of computations needed for generating subsequent tokens【8:5†source】.

#### 11.3 Transformers for Images and Audio

**Vision Transformers (ViTs)**

* **Patch-Based Tokenization**: Instead of pixels, images are tokenized into patches, which are then flattened and linearly projected to a fixed embedding size. This is much more computationally feasible for high-resolution images.
* **Positional Embeddings**: Since transformers do not inherently process sequence order, positional embeddings (trainable or sinusoidal) are added to these patches to retain spatial information.
* **Image Applications**: ViTs can handle both image classification by using class tokens and image generation by training the model to predict patches in sequence【8:1†source】【8:13†source】.

**Audio Transformers**

* **Audio Tokenization**: Audio data is typically tokenized with 1D convolutional layers to reduce the sequence length. For example, the Wav2Vec model processes raw audio to create embeddings that align with text transcriptions.
* **Applications**: Encoder-only models like Wav2Vec focus on generating embeddings, while encoder-decoder models, like Whisper, are optimized for transcription, where the model is trained to generate text output from audio input【8:12†source】.

#### Transformer Variants and Alternatives

* **Mixer Models**: In situations where attention may not be needed, transformer variants like mixer models replace MHA with MLP layers, applied alternately to mix tokens and channels, which can reduce computation.
* **MetaFormers**: A generalized class of transformers that employ simple MLPs, convolutions, or Fourier transforms instead of attention for token mixing, useful for certain vision and audio tasks【8:8†source】.

#### Key Takeaways

* Transformer architectures have versatile applications, and chapter 11 delves into how to adapt these architectures for specific tasks by incorporating various configurations, such as causal MHA, cross-attention, and efficient computation techniques.
* The chapter covers adaptations of transformers for different data types, such as images and audio, showing how the tokenization process and positional embeddings are adapted to new domains, making transformers data-agnostic and highly adaptable.

These notes summarize Chapter 11's exploration of how transformers are implemented in practice, optimized, and adapted to diverse data types beyond text【8:0†source】【8:10†source】.



## Chapter 12, _Graph Models_

#### Overview of Graph-Based Data

* Graphs model complex relationships between data points. They are described by nodes (representing entities) and edges (representing relationships).
* This chapter discusses specialized layers for graph-structured data, emphasizing two main approaches: **message-passing layers** and **graph transformers**【12:0†source】.

#### 12.1 Learning on Graph-Based Data

**Graph Structure and Adjacency Matrix**

* A graph ( G = (V, E) ) consists of nodes ( V ) and edges ( E ).
* The adjacency matrix ( A ) represents connectivity: ( A\_{ij} = 1 ) if nodes ( i ) and ( j ) are connected; otherwise, ( A\_{ij} = 0 ). In undirected graphs, ( A ) is symmetric.
* Features of nodes are often stored in a feature matrix ( X ), where each row represents a node's features【12:1†source】【12:5†source】.

**Types of Graph Features**

* **Node features**: Attributes associated with individual nodes (e.g., user profiles in a social network).
* **Edge features**: Attributes associated with relationships (e.g., the number of messages exchanged).
* **Graph-level features**: Global properties that describe the entire graph (e.g., connectivity metrics).
* Degree matrices and normalization help represent node connectivity and scale features accordingly【12:6†source】.

#### 12.2 Graph Convolutional Layers (GCL)

**Graph Convolution**

* A Graph Convolutional (GC) layer aggregates features from each node's local neighborhood.
* The process involves two steps:
  1. **Node-wise update**: Applying a linear transformation to node features.
  2. **Neighborhood aggregation**: Combining transformed features from neighboring nodes.
* The GC layer can be defined as: \[ f(X, A) = \phi(A(XW + b)) ] where ( W ) is the weight matrix, ( b ) is a bias term, and ( \phi ) is an activation function【12:7†source】【12:12†source】.

**Properties of GC Layers**

* **Permutation Equivariance**: The output remains unchanged under node reordering.
* **Locality**: Information propagation in GCNs is limited by the neighborhood of each node.
* **Higher-Order Neighborhoods**: Stacking GC layers extends the receptive field, allowing nodes to aggregate information from more distant neighbors【12:13†source】.

**Building a Graph Convolutional Network**

* GC networks (GCNs) are built by stacking multiple GC layers, allowing for deeper aggregation of features.
* Layers can incorporate normalization and residual connections, similar to CNNs, to improve training stability【12:18†source】.

#### 12.3 Extensions Beyond GC Layers

**Graph Attention Networks (GAT)**

* **GAT Layers**: Utilize attention mechanisms to weigh neighbors based on feature similarity. This allows GATs to adaptively aggregate neighbors’ features, addressing scenarios where some neighbors may be more relevant than others.
* Attention scores are calculated via: \[ \alpha(x\_i, x\_j) = \text{LeakyReLU}(a^\top \[Wx\_i \parallel Wx\_j]) ] where ( a ) and ( W ) are trainable parameters【12:8†source】【12:9†source】.

**Message-Passing Neural Networks (MPNN)**

* In MPNNs, nodes update their features based on aggregated messages from neighbors, making it a flexible framework for graph-based tasks.
* MPNNs generalize GCNs by allowing different ways to define messages and update functions based on the graph structure【12:19†source】.

**Graph Transformers**

* Graph transformers embed graph connectivity into structural embeddings, which are summed with node features and processed through standard transformer layers. These are useful for larger graphs where simple convolutional structures may be insufficient【12:3†source】.

#### Applications in Graph-Based Tasks

1. **Node Classification**: Predicts labels for individual nodes based on the surrounding structure. Each node’s representation is updated through graph layers, and predictions are made using a final classification layer.
2. **Edge Prediction**: Infers the presence or type of relationships between nodes by using node embeddings to score edges.
3. **Graph Classification**: Aggregates node embeddings to create a single embedding for the entire graph, which is then used for classification【12:15†source】.

#### Implementation Considerations

* Efficiently handling large graphs is essential. Sparse matrix representations reduce memory usage and computation, and libraries such as PyTorch Geometric facilitate implementation.
* **Mini-Batch Processing**: Multiple graphs are processed by building a larger, sparse adjacency matrix with block diagonal structure, where each block represents a graph. This enables mini-batching for graph-based datasets【12:10†source】.

These notes outline the foundational concepts of graph-based data, describe the structure and function of graph neural networks, and delve into specific models and applications within this domain【12:0†source】【12:11†source】.



## Chapter 13, _Recurrent Models_

#### Overview of Recurrent Models

* Chapter 13 explores recurrent models, which are efficient alternatives to transformers for sequence processing. Unlike transformers, recurrent models process sequences in a linear, element-wise manner, making them more computationally efficient, especially for long sequences.
* The chapter covers various types of recurrent models, including classical RNNs, gated RNN variants, structured state space models (SSMs), and modern linearized attention-based models.

#### 13.1 Linearized Attention Models

* **Linearized Attention**: This approach generalizes the attention layer in a recurrent form, replacing the standard dot product with a similarity function (\alpha).
  * The attention mechanism in linearized form allows for recurrent computation, where only past elements contribute to the current token, useful in autoregressive models.
  * Attention memory (S\_i) and normalizer memory (z\_i) can be calculated recursively: \[ S\_i = S\_{i-1} + \phi(k\_i)v\_i^\top \quad \text{and} \quad z\_i = z\_{i-1} + \phi(k\_i) ] where (\phi) is a feature expansion function【16:9†source】.

#### 13.2 Classical Recurrent Layers

**General Recurrent Layer**

* A basic recurrent layer can be formulated as: \[ s\_i = f(s\_{i-1}, x\_i) \quad \text{and} \quad h\_i = g(s\_i, x\_i) ] where (s\_i) represents the state vector, (f) is the transition function, and (g) is the readout function【16:8†source】.

**Vanilla RNNs**

* Vanilla RNNs use a simple state transition function: \[ f(s\_{i-1}, x\_i) = \phi(As\_{i-1} + Bx\_i) ] where (\phi) is an activation function, and (A) and (B) are weight matrices. However, these layers are prone to issues like vanishing and exploding gradients, which can cause training difficulties on long sequences【16:6†source】.

**Gated Recurrent Networks (GRUs and LSTMs)**

* **GRUs** and **LSTMs** are variants that use gates to control information flow, addressing the limitations of vanilla RNNs.
  * **GRUs** incorporate a reset gate and an update gate, allowing for selective updating of the state.
  * **LSTMs** include an additional forget gate to manage long-term dependencies better by controlling what information should be stored or discarded【16:7†source】【16:11†source】.

#### 13.3 Structured State Space Models (SSMs)

**Linear Recurrent Layers**

* SSMs operate by simplifying the transition function to a linear form: \[ f(s\_{i-1}, x\_i) = As\_{i-1} + Bx\_i ] This layer is less expressive but can be enhanced by adding non-linearities in subsequent layers or by interleaving with MLPs.
* Recent research has introduced variants like HiPPO-based SSM layers, which compress input sequences while retaining long-term dependencies by applying structured matrices that efficiently handle large sequences【16:4†source】【16:18†source】.

**Parallel Scans**

* SSMs benefit from parallel algorithms such as associative scans, which enable faster computations. These algorithms aggregate elements in parallel, reducing the time complexity in sequence processing.
* Associative scans can significantly improve the efficiency of linear SSM layers, making them competitive with transformers in terms of sequence length scalability【16:18†source】.

#### 13.4 Additional Variants

**Attention-Free Transformers (ATF)**

* ATF models reduce computational complexity by replacing standard attention mechanisms with element-wise multiplications, avoiding the quadratic complexity associated with sequence length.
* In these models, each channel’s attention is computed independently, using relative embeddings and simplified gating functions【16:16†source】.

**Receptance Weighted Key Value (RWKV) Model**

* The RWKV model is an alternative to the transformer, using recurrent layers to emulate attention-like behavior.
* It includes architectural features like selective gating and token-wise projections, making it suitable for large-scale sequence modeling tasks. RWKV is one of the few RNN models that can match transformers' performance on large datasets【16:16†source】.

**Selective State Space Models**

* Selective SSMs adjust the transition matrix over time, allowing it to vary according to input tokens. This approach enables time-varying dynamics, which are useful for certain sequence modeling tasks.
* These models, such as the Mamba layer, can achieve transformer-level performance on large contexts by allowing input-dependent state updates【16:10†source】.

#### Key Takeaways

* Recurrent models provide a more memory-efficient alternative to transformers for sequential data processing, particularly for long sequences.
* While classic RNNs and LSTMs struggle with gradient issues on long sequences, gated and structured state space models offer improved performance.
* Newer models like linearized attention, ATF, and RWKV provide competitive alternatives to transformers, balancing efficiency and expressiveness, especially useful in resource-constrained environments.

These notes provide an overview of the recurrent models, addressing their architectures, variants, and practical applications【16:12†source】.



Here are the detailed study notes on the _Probability Theory_ section from the appendix of your document:

## Overview of Probability Theory

* Probability theory is essential in machine learning for handling uncertainties in data and model predictions. The appendix provides a foundation in the basic rules and distributions that form the basis of probabilistic modeling.

#### A.1 Basic Laws of Probability

* **Random Variables**: Represent outcomes with assigned probabilities. For example, a random variable ( w ) can denote different ticket outcomes in a lottery.
*   **Probability Distributions**: The distribution ( p(w) ) defines the likelihood of each possible outcome, ensuring that probabilities are non-negative and sum up to one.

    **Joint Probability**: Represents the likelihood of two random events occurring together. For example, ( p(r, w) ) might represent the probability of a ticket being real or fake and its outcome.
* **Conditional Probability**: Given as ( p(r | w) ), which is the probability of one event happening given that another event has occurred.
  * **Product Rule**: Used to calculate joint probabilities: \[ p(r, w) = p(r | w)p(w) ]
  * **Sum Rule**: Used for marginalizing over variables: \[ p(w) = \sum\_r p(w, r) = \sum\_r p(w | r)p(r) ]
* **Bayes' Theorem**: A fundamental theorem for "reversing" conditional probabilities: \[ p(r | w) = \frac{p(w | r)p(r)}{p(w)} ] This theorem is often used in classification tasks to compute the probability of certain outcomes based on observed data .

#### A.2 Real-Valued Distributions

* When dealing with continuous random variables, probability densities are used rather than discrete probabilities.
  * **Cumulative Density Function (CDF)**: Represents the probability that a random variable ( X ) is less than or equal to a certain value ( x ): \[ P(x) = \int\_{-\infty}^{x} p(t) , dt ]
  * **Probability Density Function (PDF)**: The derivative of the CDF, representing the density of the probability distribution.
  * **Integral-Based Sum and Product Rules**: For continuous distributions, sums in probability rules are replaced with integrals. For example: \[ p(x, y) = p(x | y)p(y) \quad \text{and} \quad p(x) = \int\_y p(x | y)p(y) , dy ] .

#### A.3 Common Distributions

* **Categorical Distribution**: Applicable for discrete variables that can take on a finite number of values. Represented as: \[ p(x) = \prod\_i p\_i^{x\_i} ] where ( x ) is a one-hot encoded vector of observed outcomes, and ( p\_i ) are probabilities for each category.
* **Bernoulli Distribution**: A specific case of the categorical distribution with only two possible outcomes.
* **Gaussian Distribution**: Used for continuous data, defined by mean ( \mu ) and variance ( \sigma^2 ): \[ p(x) = \frac{1}{\sqrt{2 \pi \sigma^2\}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) ] For multiple variables, the Gaussian distribution can be extended to the multivariate case with mean vector ( \mu ) and covariance matrix ( \Sigma ) .

#### A.4 Moments and Expected Values

* **Expected Value (Mean)**: The average or mean of a function ( f(x) ) over a probability distribution: \[ \mathbb{E}\[f(x)] = \sum\_x f(x)p(x) \quad \text{or} \quad \mathbb{E}\[f(x)] = \int\_x f(x)p(x) , dx ]
* **Moments**: Represent specific expected values that summarize the distribution's characteristics, such as the mean (first moment) and variance (second moment).
* **Monte Carlo Estimation**: An approximation method for expected values when the distribution is unknown, relying on random sampling: \[ \mathbb{E}\[f(x)] \approx \frac{1}{n} \sum\_{i=1}^n f(x\_i) ] where ( x\_i ) are samples from the distribution .

#### A.5 Distance between Distributions

* **Kullback-Leibler (KL) Divergence**: Measures the difference between two distributions ( p(x) ) and ( q(x) ): \[ KL(p | q) = \int p(x) \log \frac{p(x)}{q(x)} , dx ] This is commonly used in optimization tasks and provides a sense of how well ( q(x) ) approximates ( p(x) ) .

#### A.6 Maximum Likelihood Estimation (MLE)

* **Maximum Likelihood Estimation**: A method for estimating parameters that maximizes the probability of observed data under a given model.
  * Given a parametric distribution ( p(x; \theta) ) and a dataset ( {x\_i} ), the log-likelihood ( L(\theta) ) is maximized: \[ L(\theta) = \sum\_{i=1}^n \log p(x\_i; \theta) ] where ( \theta ) are the parameters to be optimized.
* **Applications**: MLE is used to estimate parameters for various distributions, such as ( p ) in the Bernoulli distribution or ( \mu ) and ( \sigma^2 ) in the Gaussian distribution .

These notes outline the principles and techniques of probability theory, emphasizing its importance for understanding machine learning models and their behavior in uncertain environments.



Here are detailed study notes on the _1D Universal Approximation_ section from your document:

## Overview of 1D Universal Approximation

The 1D universal approximation section provides a visual and intuitive proof for the universal approximation theorem, primarily for functions with a single input and output. This theorem indicates that a neural network with one hidden layer is capable of approximating any continuous function on a compact domain to any desired level of accuracy.

#### B.1 Approximating a Step Function

* The section begins by showing how a single neuron in a hidden layer can approximate a step function, a basic building block in function approximation.
* The function of a single neuron can be represented as: \[ f(x) = a \sigma(w(x - s)) ] where:
  * ( a ) controls the amplitude,
  * ( w ) controls the slope,
  * ( s ) shifts the function.
* By adjusting ( w ), the slope becomes steeper, making the function closely resemble a step function. This is fundamental for constructing more complex approximations【32:3†source】.

#### B.2 Approximating Constant Functions Over Intervals

* Adding a second neuron allows the network to approximate functions that are constant over small intervals (referred to as "bin" functions).
* The function is formulated with two hidden layer neurons: \[ f(x) = a \sigma\left(w\left(x - s - \frac{\Delta}{2}\right)\right) - a \sigma\left(w\left(x - s + \frac{\Delta}{2}\right)\right) ] where ( \Delta ) controls the width of the bin.
* This formulation enables the creation of localized functions, effectively zero outside of a specific interval and constant within the interval, which is key to approximating more complex continuous functions【32:4†source】.

#### B.3 Approximating a Generic Function

* By combining multiple bin functions, it is possible to approximate any continuous function over a specified interval.
* To approximate a function ( g(x) ) over the interval (\[0, 1]):
  1. Divide the input domain into ( m ) equally spaced intervals, where ( m ) determines the approximation accuracy.
  2. For each interval ( B\_i ), compute the average value ( g\_i ) of the function ( g(x) ).
  3. Use a neural network with ( 2m ) neurons, placing two neurons in each interval to approximate the function.
* The output function can be represented as: \[ f(x) = \sum\_{i=1}^{m} f\left(x; g\_i, \frac{i}{m}, \Delta\right) ] where ( g\_i ) represents the constant approximation over the ( i )-th bin.
* As ( m ) increases, the approximation error reduces, allowing the network to closely match the original function ( g(x) ). The mean squared error (MSE) decreases exponentially with the number of bins, which enhances the approximation’s precision【32:0†source】【32:1†source】.

#### Key Takeaways

* The universal approximation theorem suggests that with a sufficient number of neurons, a neural network with a single hidden layer can approximate any continuous function on a compact domain to arbitrary accuracy.
* This is achieved by combining step and bin functions to approximate functions that are constant over small intervals and piecing them together.
* While this section focuses on 1D functions, the concept can extend to multiple dimensions, though the required number of neurons grows exponentially with the number of input dimensions.

These notes encapsulate the essentials of the 1D universal approximation theorem, explaining how single-layer networks can serve as universal function approximators by using step and localized constant functions as building blocks【32:10†source】.

