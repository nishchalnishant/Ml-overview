# Fundamentals of Machine Learning

This file is the **must-know shortlist**.

If ML interviews were a Mumbai Indians match, these are your **powerplay overs**. You do not freestyle here. You score cleanly, quickly, and with confidence.

---

# 1. The Core ML Lifecycle

If you already know Azure DevOps, here is the clean mapping:

- **Dataset** = source repo + artifact input
- **Training** = build stage
- **Validation** = test stage
- **Model registry** = artifact store
- **Inference service** = deployed app
- **Monitoring** = App Insights + dashboards + alerts

ML is basically **software delivery with uncertainty baked in**.

Code can be correct and the model can still fail.

That is the whole game.

---

# 2. What is Machine Learning?

**Short answer**

Machine Learning is about teaching systems to learn patterns from data instead of hard-coding every rule manually.

**DevOps parallel**

In normal software, you write the logic.
In ML, you write the pipeline, the objective, and the constraints, then let the model learn the logic from data.

So instead of:

- `if x > 10, do A`

you say:

- "Here are 10 million examples. Learn the mapping."

**Why this matters**

The tricky part is not getting low training error.
The tricky part is getting a model that works on **new, messy, real-world data**.

---

# 3. Supervised vs Unsupervised vs Reinforcement Learning

## Supervised Learning

You have inputs and labels.

Examples:

- spam vs not spam
- house price prediction
- churn prediction

**Azure-style mental model:** supervised learning is like training from a ticket history where the "correct resolution" is already attached.

## Unsupervised Learning

You only have inputs. No labels.

The goal is to discover structure:

- clusters
- patterns
- compressed representations

**Think:** grouping outfits by style, fabric, silhouette, and season even if nobody labeled them "minimalist", "bridal", or "streetwear."

## Reinforcement Learning

An agent learns from **rewards** over time.

Not "here is the right answer."
More like:

- take action
- get feedback
- adjust

**Cricket analogy:** a captain tweaks field placement over after over. You do not get a clean label saying "this exact field is correct." You infer from outcome, pressure, batter behavior, and scoreboard context.

---

# 4. Epoch, Batch, Batch Size, Iteration

These four terms are simple, but interviewers ask them because confusion here is a red flag.

## Epoch

One full pass over the entire dataset.

## Batch

A smaller chunk of the dataset used for one update.

## Batch Size

How many examples are inside that chunk.

## Iteration

One optimizer step.

So if your dataset has 10,000 rows and batch size is 100:

- 1 epoch = all 10,000 rows seen once
- 100 iterations = one epoch

**Azure/CI analogy**

Think of:

- dataset = full codebase
- batch = one pipeline job chunk
- iteration = one build/test execution
- epoch = the whole pipeline finishing once end-to-end

**Why batch size matters**

- Bigger batch = smoother gradients, more memory
- Smaller batch = noisier gradients, less memory, sometimes better generalization

**Mini Pop Quiz**

If a model trains faster after increasing batch size, does that automatically mean it will generalize better?

Answer: **No.**
Speed and generalization are not the same thing.

---

# 5. Classification vs Regression

## Classification

Predicts a category.

Examples:

- fraud or not fraud
- cat or dog
- yes or no

## Regression

Predicts a number.

Examples:

- revenue
- delivery time
- temperature

**Quick memory trick**

- If the answer looks like a **bucket**, it is classification.
- If it looks like a **dial**, it is regression.

---

# 6. Bias, Variance, Underfitting, Overfitting

This is one of the most important ideas in ML.

## High Bias = Underfitting

The model is too simple.
It misses the signal.

It performs badly on:

- training data
- validation data

## High Variance = Overfitting

The model memorizes noise or quirks in training data.

It performs:

- very well on training
- badly on validation or production

**Fashion analogy**

Underfitting is like wearing one plain black t-shirt to every event.
Safe, simple, but not right for the occasion.

Overfitting is like wearing a wildly over-styled runway look with six layers, feathers, sequins, and dramatic sleeves to a quick coffee run.
Impressive, but deeply impractical.

**The goal**

You want a model that captures enough structure to be useful without becoming obsessed with irrelevant detail.

---

# 7. Loss Functions vs Metrics

People mix these up. Do not.

## Loss Function

Used during training.
The optimizer tries to minimize it.

Examples:

- cross-entropy
- MSE
- MAE

## Metric

Used to judge model quality.
Often used during evaluation or reporting.

Examples:

- accuracy
- precision
- recall
- F1
- RMSE

**DevOps analogy**

Loss is like the internal signal your pipeline uses to optimize a build.
Metrics are the release checks the business actually cares about.

You can reduce loss and still have a poor business outcome if the wrong metric is optimized.

---

# 8. L1 vs L2 Loss and L1 vs L2 Regularization

Two related topics. Different use.

## L1 Loss

Uses absolute error.

- more robust to outliers
- less harsh on big misses

## L2 Loss

Uses squared error.

- penalizes large errors more strongly
- common in regression

Now the regularization version:

## L1 Regularization

Pushes some weights to exactly zero.

Useful when you want sparsity.

## L2 Regularization

Shrinks weights smoothly.

Useful when you want stability without aggressively zeroing features.

**Short memory hook**

- **L1** = likes cleanup
- **L2** = likes control

---

# 9. What is Regularization?

Regularization is how you stop a model from becoming too clever in all the wrong ways.

It adds pressure to keep the model simpler, smoother, or less reliant on specific signals.

Common forms:

- L1 / L2 regularization
- dropout
- early stopping
- data augmentation
- smaller models

**Kishore/Lata remaster analogy**

Think of fine model training like remastering a classic romantic track.
You want it cleaner, richer, sharper.
Not overprocessed until all warmth disappears.

Regularization is what stops you from "polishing" the signal into artificial nonsense.

---

# 10. Dropout

Dropout randomly turns off some neurons during training.

Why do that?

Because it prevents the network from becoming overly dependent on a few specific paths.

It forces the model to learn more distributed, robust representations.

**Simple way to say it in an interview**

Dropout is like forcing the team to perform even when one star player is unavailable.
If the system collapses without one neuron, it was too dependent to begin with.

---

# 11. Embeddings

Embeddings are learned vector representations for things like:

- words
- users
- products
- IDs

Instead of representing a product or word as a giant sparse one-hot vector, you map it into a dense numerical space.

That space captures similarity.

**Fashion analogy**

Imagine turning outfits into style coordinates:

- fabric
- cut
- color mood
- silhouette
- occasion

Now two outfits that "feel similar" sit close together in this style space even if they are not literally identical.

That is what embeddings do for tokens, products, users, and more.

---

# 12. Softmax and Logits

## Logits

Raw model scores before they are converted into probabilities.

## Softmax

Turns those logits into probabilities that sum to 1.

Used mostly in multiclass classification.

**Why this matters**

A model usually thinks in scores first.
Softmax is the translation layer that makes those scores human-friendly.

---

# 13. Cross-Entropy

Cross-entropy is a very common loss function for classification.

It rewards the model when it puts high probability on the correct class.
It punishes the model heavily when it is confidently wrong.

That last part matters.

A model saying:

- "I'm 51% sure"

and being wrong is one thing.

A model saying:

- "I'm 99.9% sure"

and being wrong is far worse.

Cross-entropy captures that difference.

---

# 14. Cross-Validation

Cross-validation is how you reduce the risk of trusting one lucky train/validation split.

You split the data into multiple folds and rotate which fold is used for validation.

This gives you a more stable estimate of how the model behaves.

**DevOps analogy**

It is like testing your deployment across multiple environments instead of saying:

> "Worked once on my machine. Ship it."

Absolutely not.

---

# 15. Precision, Recall, and F1

These are interview favorites.

## Precision

Of the things the model predicted as positive, how many were actually positive?

Use when false positives are expensive.

Example:

- spam filter
- fraud alert

## Recall

Of the things that were actually positive, how many did the model catch?

Use when false negatives are expensive.

Example:

- cancer screening
- fraud detection

## F1

Balanced harmonic mean of precision and recall.

Useful when both matter.

**Cricket analogy**

Precision is like saying:

> "When I go for a risky shot, how often do I actually get boundary value?"

Recall is:

> "Of all scoring opportunities, how many did I capitalize on?"

F1 is the balanced batter who is neither reckless nor timid.

---

# 16. Anomaly Detection

Anomaly detection is about finding patterns that look unusual compared to the normal baseline.

Examples:

- fraudulent transactions
- broken sensors
- strange user behavior
- traffic spikes

It is useful when anomalies are rare and labels are limited.

**Azure angle**

Think of it like monitoring unusual deployment behavior:

- sudden latency spike
- odd CPU pattern
- weird error burst

You may not have labeled every future failure mode, but you know what "normal" looks like.

---

# 17. Policy-Based vs Value-Based RL

This matters if RL comes up.

## Value-Based

Estimate how good an action is.

Then choose the best one.

Example: Q-learning.

## Policy-Based

Learn the action strategy directly.

Instead of scoring every move first, you directly learn:

> "Given this state, what should I do?"

**Easy way to remember**

- value-based = learn the scoreboard
- policy-based = learn the playbook

---

# 18. Exploration vs Exploitation

This is the eternal RL tension.

- **Exploration** = try something new
- **Exploitation** = use what already works

**Mumbai Indians analogy**

Do you keep bowling the field plan that worked last over?
Or do you change because the batter has adapted?

That tradeoff is exploration vs exploitation in one line.

---

# 19. Curse of Dimensionality

As feature count grows, data becomes sparse and distance-based reasoning becomes less reliable.

That means:

- nearest-neighbor methods struggle
- clustering gets noisy
- you need more data

**Fashion analogy**

If you compare outfits on just:

- color
- fit
- fabric

similarity is manageable.

If you compare on 500 tiny style attributes, suddenly everything looks oddly far apart.

That is the curse of dimensionality.

---

# Quick Thought Experiment

You are asked to build a churn model in Azure.

Before choosing the algorithm, what do you lock down first?

- data split
- metric
- feature availability at inference time
- rollback plan

If your brain said:

> "metric + split + feature availability"

excellent.

That means you are already thinking like someone who ships models, not just slides.
