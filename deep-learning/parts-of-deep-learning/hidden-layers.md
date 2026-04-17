# Hidden Layers

Hidden layers are where a network stops being a simple input-output calculator and starts building internal representations.

That is the entire point of depth.

Without hidden layers, the model has very limited expressive power.

With them, it can learn progressively richer structure.

---

# 1. What Hidden Layers Do

Each hidden layer transforms the representation it receives.

Early layers often learn:

- simple patterns

Deeper layers learn:

- more abstract structure

That is why deep learning often feels like stacked feature engineering, except the model is doing the engineering for you.

---

# 2. Why More Layers Help

Depth lets the model build complex functions out of simpler ones.

That means it can capture:

- interactions
- hierarchy
- abstraction

But more layers also mean:

- harder optimization
- more overfitting risk
- higher compute cost

So deeper is not automatically better.

It is just more powerful when used well.
