# Activation Functions

Activation functions are what stop neural networks from becoming glorified spreadsheet formulas.

Without them, stacking layers would still collapse into something linear and boring.

So yes, these matter.

---

# 1. Why Activations Exist

Activations introduce **non-linearity**.

That is what lets a network learn:

- curves
- boundaries
- interactions
- richer structure

Without activation functions, depth would look impressive but behave disappointingly.

---

# 2. ReLU

ReLU means:

- keep positive values
- zero out negative ones

Why it became dominant:

- simple
- cheap
- avoids the worst saturation problems of sigmoid/tanh on the positive side

**Short answer**

ReLU became the default hidden-layer activation because it is simple, fast, and usually much easier to optimize in deep networks than sigmoid or tanh.

---

# 3. Why Sigmoid and Tanh Fell Behind

They saturate.

That means:

- very large positive or negative inputs produce tiny gradients
- learning slows down in deep networks

That is the vanishing-gradient pain point.

**Quick memory trick**

- sigmoid/tanh = elegant but fragile
- ReLU = less poetic, more practical

---

# 4. Leaky ReLU

Leaky ReLU keeps a small slope for negative inputs instead of making them exactly zero.

Why use it:

- helps reduce dead neurons
- can improve training stability in some cases

Not always necessary.
But useful to know.

---

# 5. GELU

GELU is common in modern Transformers.

Why people like it:

- smoother than ReLU
- works well in large language-model style architectures

**Short answer**

GELU is a smoother activation that became popular in Transformers because it works well at scale and fits modern deep-learning architectures nicely.

---

# 6. Sigmoid vs Softmax

These usually belong at the output layer.

## Sigmoid

Use for:

- binary classification
- multi-label classification

Why:

Each output can be treated independently.

## Softmax

Use for:

- multiclass classification

Why:

Outputs become a probability distribution summing to 1.

---

# 7. Which Activation to Use Where

## Hidden layers

- ReLU is a strong default
- GELU often appears in Transformers

## Output layer

- regression = no activation / identity
- binary classification = sigmoid
- multiclass classification = softmax

That decision tree alone is enough for many interview questions.

---

# Mini Pop Quiz

If multiple classes can all be true at the same time, do you want:

- softmax
- sigmoid

Answer:

Sigmoid.

Because the outputs should be independent, not forced to compete.
