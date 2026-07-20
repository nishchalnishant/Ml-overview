---
module: Meta
topic: Orientation
subtopic: What Is AI
status: unread
tags: [orientation, paradigms, ml-vs-dl, entry-point]
prerequisites: []
---
# Introduction to AI

## Artificial Intelligence

**Definition:** Artificial Intelligence (AI) is the field of computer science focused on creating systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, problem-solving, perception, and language understanding.

### Core Learning Paradigms

AI systems learn patterns and make decisions through various approaches:

* **Statistical Machine Learning** - Uses mathematical and probabilistic methods to find patterns in data:
  * Creates decision boundaries, hyperplanes, or hierarchical splits to divide data
  * Examples: Linear/logistic regression, SVMs, decision trees, random forests
* **Deep Learning** - Mimics human neural networks to learn complex patterns through iterative optimization:
  * Learns hierarchical feature representations automatically
  * Excels at unstructured data (images, text, audio)
  * Examples: CNNs, RNNs, Transformers
* **Generative AI** - Creates new data samples after learning from existing data:
  * Examples: GANs, VAEs, Diffusion Models (DALL-E, Stable Diffusion)
* **Reinforcement Learning** - Learns through interaction with an environment via rewards and penalties:
  * Agent learns optimal actions through trial and error
  * Examples: Game playing (AlphaGo), robotics, recommendation systems

### Key Challenges in Machine Learning

* **Data Representation** - How to encode and feed data to ML models (feature engineering, embeddings)
* **Performance Monitoring** - Tracking model progress during training (loss curves, validation metrics)
* **Generalization** - Ensuring models learn the right patterns and perform well on unseen data
* **Interpretability** - Understanding how models make decisions
* **Scalability** - Handling large datasets and deploying models in production

### Machine Learning vs Deep Learning

| **Aspect**              | **Machine Learning**                               | **Deep Learning**                           |
| ----------------------- | -------------------------------------------------- | ------------------------------------------- |
| **Approach**            | Statistical and probabilistic methods              | Neural networks with multiple hidden layers |
| **Data Requirements**   | Works well with smaller datasets (1K-100K samples) | Requires large datasets (100K-1M+ samples)  |
| **Computation**         | Lower computational cost, can run on CPUs          | High computational cost, requires GPUs/TPUs |
| **Feature Engineering** | Manual feature engineering required                | Automatic feature learning                  |
| **Training Time**       | Minutes to hours                                   | Hours to days/weeks                         |
| **Interpretability**    | High (e.g., decision trees, linear models)         | Low (black-box models)                      |
| **Use Cases**           | Tabular data, structured problems, limited data    | Images, text, audio, unstructured data      |
| **Performance Scaling** | Plateaus with more data                            | Improves with more data and model size      |
| **Hardware Needs**      | Standard CPU sufficient                            | GPUs/TPUs often necessary                   |

**When to Use Machine Learning:**

* Small to medium-sized datasets
* Structured/tabular data
* Need for model interpretability
* Limited computational resources
* Quick iteration and deployment needed

**When to Use Deep Learning:**

* Large datasets available
* Unstructured data (images, text, audio, video)
* Complex pattern recognition required
* Performance is priority over interpretability
* Sufficient computational resources available


---

## Where to go next

This page is the entry point — definitions and the map of the field. Each paradigm above is
developed properly elsewhere:

| Paradigm | Depth |
| :--- | :--- |
| Statistical ML | [../03-classical-ml/](../03-classical-ml/README.md) |
| Deep Learning | [../05-deep-learning-core/](../05-deep-learning-core/README.md), [../06-architectures/](../06-architectures/README.md) |
| Generative AI | [../08-generative/](../08-generative/README.md) |
| Reinforcement Learning | [../09-reinforcement-learning/](../09-reinforcement-learning/README.md) |

The challenges listed above are each a section in their own right — data representation in
[../02-data/](../02-data/README.md), generalization and interpretability in
[../04-evaluation/](../04-evaluation/README.md), scalability in
[../12-systems-and-scale/](../12-systems-and-scale/README.md).

**Provenance:** restored from `introduction-to-ai.md`, which the repo restructure deleted. Only
the orientation material was kept; the remainder of that file (algorithms, evaluation metrics,
transfer learning, interview Q&A) is covered at greater depth in the sections above and was
deliberately not carried forward.
