# Deep Learning

This folder is the deep-learning version of a good Azure DevOps project:

- clear building blocks
- reusable components
- strong deployment instincts
- fewer mysterious things happening in production

If you already come from **Azure + DevOps**, here is the easiest bridge:

- **model architecture** = service design
- **training loop** = build pipeline
- **validation** = release gates
- **inference** = deployed runtime
- **monitoring** = observability plus business outcomes

Deep learning is just software delivery with:

- tensors
- gradients
- GPUs
- and a little more emotional volatility

---

## Start Here

If you want the highest-value path first:

1. `parts-of-deep-learning/activation-functions.md`
2. `parts-of-deep-learning/backpropagation.md`
3. `parts-of-deep-learning/optimisers.md`
4. `parts-of-deep-learning/regularization.md`
5. `parts-of-deep-learning/attention.md`
6. `parts-of-deep-learning/transformers.md`
7. `pytorch-foundations.md`

That gives you the fundamentals before the fancier methods.

---

## What This Folder Covers

- `parts-of-deep-learning/`
  The building blocks: activations, loss, backprop, optimizers, attention, regularization, transformers

- `deep-learning-methods/`
  Applications and model families: NLP, CV, generative models, time series

- `pytorch-foundations.md`
  The practical implementation layer

- `mcp.md`
  Tooling and protocol notes for model-connected workflows

---

## Quick Thought Experiment

If a deep model performs beautifully in training but fails in deployment, what do you inspect first?

- the architecture?
- the optimizer?
- the data path?
- the serving mismatch?

If your DevOps brain answered:

> "data path and serving mismatch first"

excellent.

That instinct will save you a lot of drama.
