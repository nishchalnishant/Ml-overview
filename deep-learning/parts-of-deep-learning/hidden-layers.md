# Hidden layers

Hidden layers are the layers between the input and output layers in a neural network. They are responsible for learning the representational "features" of the data.

***

## <mark style="color:red;">Types of Hidden Layers</mark>

### 1. Dense (Fully Connected) Layers

Every neuron in the layer is connected to every neuron in the previous layer.

* **Best for:** Tabular data and final classification/regression heads.
* **Complexity:** $O(n \times m)$ weights, where $n$ and $m$ are the layer sizes.

### 2. Convolutional Layers (Conv)

Applies a set of learnable filters to the input to extract spatial features (edges, textures, shapes).

* **Key Concepts:** Kernels, Stride, Padding, Dilated Convolutions.
* **Best for:** Images and spatial data.

### 3. Recurrent Layers (RNN/LSTM/GRU)

Maintains a "hidden state" that captures information from previous steps in a sequence.

* **Best for:** Sequential data (text, time-series, audio).

### 4. Attention Layers

Allows the model to focus on specific parts of the input sequence regardless of their distance.

* **Mechanism:** Query, Key, Value vectors.
* **Best for:** Long-range dependencies in NLP (Transformers).

***

## Intuition: Hierarchical Representation

The beauty of deep learning lies in how hidden layers decompose complex problems:

1. **Early Layers:** Learn simple features (e.g., edges, blobs).
2. **Mid Layers:** Combine simple features into parts (e.g., eyes, wheels).
3. **Late Layers:** Combine parts into objects (e.g., faces, cars).

***

## Interview Questions

**1. "What happens if you have no hidden layers?"**

> The model becomes a simple linear classifier (like Logistic Regression) and cannot learn non-linear decision boundaries.

**2. "How do you choose the number of hidden layers and neurons?"**

> Generally via hyperparameter tuning. However, the modern trend is to use established architectures (e.g., ResNet, BERT) and fine-tune them rather than designing from scratch.

**3. "Explain the role of Activation Functions in hidden layers."**

> They introduce non-linearity. Without them, the entire network would collapse into a single linear transformation, regardless of the number of layers.
