# Hidden Layers

Hidden layers are the intermediate layers between input and output in a neural network. They learn hierarchical representations that make complex functions learnable.

---

## Role

- **Input layer:** Receives raw features. **Output layer:** Produces predictions (e.g. class probabilities or regression value). **Hidden layers:** Transform the input through successive nonlinear mappings.
- Each hidden unit computes a weighted sum of its inputs plus a bias, then applies an **activation function** (ReLU, GELU, etc.). Stacking many such layers allows the network to approximate complex, hierarchical patterns.

---

## Depth and width

- **Depth:** Number of hidden layers. Deeper networks can represent more complex functions but are harder to train (vanishing/exploding gradients); skip connections (ResNet) and normalization help.
- **Width:** Number of units per layer. Wider layers increase capacity and are often used in transformers (e.g. large FFN dimension).

---

## Representation learning

- Early layers often learn low-level features (edges, local patterns); later layers learn higher-level abstractions (objects, semantics). This hierarchy is especially clear in CNNs and in encoder stacks of transformers.
- **Universal approximation:** A single hidden layer with enough width can approximate any continuous function on a compact set; in practice, depth improves efficiency and generalization.

---

## Quick revision

- **Hidden layers** sit between input and output; each applies linear transform + activation. **Depth** adds representational capacity; **width** adds capacity per layer. **Skip connections** and **normalization** enable training of very deep networks.
