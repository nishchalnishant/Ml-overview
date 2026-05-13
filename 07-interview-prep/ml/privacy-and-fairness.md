# Privacy and Fairness in Machine Learning

ML models are not just mathematical objects.
They are systems that affect real people.

This file covers two questions that come up constantly in senior ML interviews:

- How do you build models that do not expose private information?
- How do you build models that treat people fairly?

Both have deep technical answers. Both matter more as models get deployed at scale.

---

# 1. Data Privacy Fundamentals

Before talking about techniques, you need to understand what you are protecting and why.

---

## 1.1 Personally Identifiable Information (PII)

PII is any data that can be used to identify an individual.

**Direct PII:**
- Name
- Email address
- Phone number
- Social security number
- Passport number

**Indirect PII (quasi-identifiers):**
- ZIP code
- Date of birth
- Gender

The famous re-identification study (Sweeney, 2000) showed that 87% of Americans could be uniquely identified by just three fields: ZIP code, date of birth, and gender.

Even "anonymized" data is often not truly anonymous.

**Why it matters for ML:**

Your training data probably contains PII.
Your model may memorize parts of it.
Large language models have been shown to reproduce training data, including personal information.

---

## 1.2 GDPR and Privacy Regulations

The General Data Protection Regulation (EU, 2018) is the most comprehensive privacy law affecting ML systems.

**Core principles that affect ML:**

**Lawful basis for processing**
You need explicit consent or a legitimate interest to train on personal data.

**Data minimization**
Collect only what you need.
This is in tension with the ML instinct to collect everything.

**Purpose limitation**
Data collected for one purpose cannot be freely used for another.
You cannot train a recommendation model on data collected for fraud detection.

**Storage limitation**
You cannot store personal data indefinitely.
This creates engineering challenges for models trained on historical data.

---

## 1.3 Right to Erasure (Right to Be Forgotten)

Under GDPR, individuals can request that their personal data be deleted.

**The ML problem**

If your model was trained on someone's data, and they request erasure, what do you do?

Simply deleting the training row is not enough.
The model has potentially learned from it.

**Machine Unlearning**

An active research area.
The goal: efficiently update a trained model to "forget" specific training examples without full retraining.

**Approaches:**

1. **Exact unlearning**: remove data and retrain from scratch. Correct but expensive.
2. **Approximate unlearning**: apply targeted gradient updates to undo the influence of specific data points. Faster but harder to verify.
3. **Data partitioning (SISA training)**: train on shards. When erasure is requested, only retrain the relevant shard.

**Interview angle**

This is an area where ML engineering meets legal compliance.
Being able to articulate the challenge and the current solutions puts you ahead.

---

# 2. Differential Privacy

Differential privacy is the gold standard for privacy-preserving computation.

It gives a mathematical guarantee: an observer cannot determine whether any specific individual's data was used in a computation.

---

## 2.1 The Intuition

**The census analogy**

Imagine a national census.
You want population statistics (average income, disease rates) without revealing any individual's answer.

Naive approach: collect all answers, compute statistics.
Problem: the statistics can leak individual information.

Differential privacy approach: each respondent adds random noise to their answer before submitting.
Individually, each response is plausibly deniable.
In aggregate, the noise cancels out and the statistics are still accurate.

The individual has plausible deniability: their noisy answer could have come from many different true values.

**The core guarantee**

A mechanism M is ε-differentially private if:

For any two datasets D and D' that differ in exactly one record, and for any output set S:

```
P[M(D) ∈ S] ≤ e^ε * P[M(D') ∈ S]
```

**Plain English interpretation**

No matter what you observe from the output, you cannot determine with confidence whether any specific individual was in the dataset.

The output distribution barely changes when you include or exclude any single person.

---

## 2.2 The Privacy Budget ε

ε (epsilon) is the privacy loss parameter.

**Small ε (e.g., 0.1):** very strong privacy, but more noise, more accuracy loss.
**Large ε (e.g., 10):** weaker privacy guarantee, but less noise.

**Typical values in practice:**

- Academic papers: ε = 1 (strong)
- Industry deployments: ε = 10–50 (pragmatic)
- Apple and Google deploy differential privacy with ε around 1–8 for certain use cases

**Composition**

If you run multiple differentially private queries, the total privacy loss accumulates.

k mechanisms each with ε-DP → total privacy loss is at most k*ε (basic composition).

Advanced composition gives tighter bounds for many queries.

This is why you cannot just run unlimited private queries: the budget runs out.

---

## 2.3 Laplace Mechanism

For numeric queries, add Laplace noise calibrated to the query's sensitivity.

**Sensitivity**

The maximum change in query output that can result from adding or removing one individual's record.

For average income with values in [0, 200K] and n people:
- sensitivity = 200K / n

**The mechanism**

```
M(D) = f(D) + Lap(sensitivity / ε)
```

Where Lap(b) is noise from a Laplace distribution with scale b.

As ε increases, the scale decreases, and the noise is smaller.

**When to use it**

Laplace is optimal for queries with bounded L1 sensitivity.
Works well for numerical statistics: counts, averages, histograms.

---

## 2.4 Gaussian Mechanism

For vector-valued or high-dimensional queries, the Gaussian mechanism is often preferred.

```
M(D) = f(D) + N(0, σ^2 * I)
```

Where σ is calibrated to the L2 sensitivity and ε, δ parameters.

**ε, δ differential privacy**

The Gaussian mechanism provides (ε, δ)-differential privacy, a slightly weaker guarantee:

```
P[M(D) ∈ S] ≤ e^ε * P[M(D') ∈ S] + δ
```

δ is a small failure probability (e.g., 10^-5).
This allows the Gaussian mechanism while maintaining strong practical privacy.

---

## 2.5 DP-SGD: Private Machine Learning

The above mechanisms work for one-shot queries.
Training a neural network involves many gradient updates — each of which could leak information.

**DP-SGD (Abadi et al., 2016)** makes gradient descent differentially private.

**Algorithm**

1. Sample a minibatch of training examples.
2. Compute the gradient for each example individually (per-sample gradients).
3. Clip each per-sample gradient to a maximum L2 norm C (this bounds sensitivity).
4. Add Gaussian noise calibrated to C and the privacy budget.
5. Average the noisy clipped gradients.
6. Update the model.

**Why clip gradients?**

Without clipping, a single example with large gradients can dominate the update.
Clipping bounds the influence of any one example, which is required for bounded sensitivity.

**The noise-accuracy tradeoff**

More noise = stronger privacy = worse model.
Less noise = weaker privacy = better model.

Finding the sweet spot requires tuning ε based on the sensitivity of the data and the acceptable accuracy degradation.

**Practical note**

DP-SGD requires computing per-sample gradients, which is more expensive than standard minibatch gradients.
Libraries like Opacus (PyTorch) make this efficient with vectorized per-sample gradient computation.

**Real-world use**

Google uses DP-SGD for training language models on keyboard data.
Apple uses local differential privacy for usage statistics.

---

# 3. Federated Learning

Differential privacy keeps outputs private.
Federated learning keeps the data itself private by never centralizing it.

---

## 3.1 The Core Problem

**Standard ML:**

Collect all data in one place → train a model → deploy.

**Problem:** data may be sensitive (medical records, financial transactions, private messages).
Centralizing it creates privacy risk, legal liability, and trust issues.

**Federated learning:**

Keep data on the devices or institutions where it was created.
Train the model by sending it to the data, not the other way around.

---

## 3.2 FedAvg Algorithm

The foundational federated learning algorithm (McMahan et al., 2017).

**Round t:**

1. Server selects a random subset of K clients.
2. Server sends the current global model weights to each selected client.
3. Each client runs local SGD on their local data for E epochs.
4. Each client sends their updated weights back to the server.
5. Server aggregates the weights by weighted averaging:
```
w_{t+1} = Σ_k (n_k / n) * w_k
```
Where n_k is client k's data size and n is the total.

**Why this works**

Gradients/weights are shared, not raw data.
The server learns model updates, not individual training examples.

---

## 3.3 The Non-IID Problem

**IID** = Independent and Identically Distributed.

Standard ML assumes your training data is IID from some underlying distribution.

Federated data is almost never IID.

**Why:** each client has data generated by their own behavior.

- A hospital's patient data reflects their patient population, not all patients globally.
- A user's typing data reflects their vocabulary, not all vocabulary.
- A city's traffic data reflects local patterns.

**What goes wrong**

Local updates on non-IID data push the model toward each client's local distribution.
Averaging these updates is noisy and can diverge.

Convergence is slower.
Accuracy is lower than centralized training on the same data.

**Mitigation approaches**

- **FedProx**: add a proximal term to penalize local updates that deviate too far from the global model.
- **SCAFFOLD**: use control variates to correct for client drift.
- **FedNova**: normalize local updates by the number of local steps.
- **More rounds with fewer local steps**: reduces client drift but increases communication cost.

---

## 3.4 Communication Efficiency

Sending full model weights every round is expensive.

**Gradient compression**

Instead of sending full gradients, send:
- Top-k gradients (only the largest k updates)
- Quantized gradients (fewer bits per value)
- Random sparse updates with error feedback

**Model compression for communication**

Use structured updates: only learn low-rank weight updates and send those.

**Why this matters at scale**

If you have 10 million mobile clients and a 100MB model, sending weights naively is 1 petabyte per round.

Practical federated learning requires aggressive communication optimization.

---

## 3.5 Stragglers

Not all clients are equally fast.

Some clients have slow connections.
Some are on low-battery.
Some get interrupted mid-training.

**The straggler problem**

If you wait for all selected clients, slow clients block progress.
If you do not wait, some clients' updates are lost.

**Approaches**

- **Asynchronous federated learning**: accept updates as they arrive. Risk: stale updates.
- **Partial participation**: proceed when a sufficient fraction of selected clients respond.
- **Client selection**: prefer fast, high-capacity clients. Risk: model skews toward certain client populations.

---

## 3.6 Secure Aggregation

A deeper privacy concern with FedAvg: the server sees individual client updates.

Even without raw data, updates can reveal information about a client's data.

**Secure aggregation (Bonawitz et al., 2017)**

Clients mask their updates with pairwise random seeds.
The masks cancel out in the aggregate.
The server only sees the sum of updates, not individual contributions.

**Properties**

- The server learns only the aggregate.
- Even if the server is semi-honest or colluding with some clients, individual updates remain hidden.
- Works even with client dropout.

---

# 4. ML Fairness

A model that is accurate on average can still systematically harm specific groups.

---

## 4.1 Types of Bias

**Historical bias**

The world has historical inequalities.
Data collected from that world reflects those inequalities.

Training a hiring model on historical hiring decisions encodes past discrimination.
Even if the data is "accurate," the patterns it captures are unfair.

**Representation bias**

If certain groups are underrepresented in your training data, the model performs worse for them.

A face recognition system trained primarily on light-skinned faces will underperform on dark-skinned faces.
This is not because of prejudice in the algorithm; it is because of who was in the dataset.

**Measurement bias**

The features used may be imperfect proxies that systematically disadvantage certain groups.

Credit score as a proxy for financial reliability has been criticized for encoding historical access to credit, which was not equally distributed.

**Aggregation bias**

A model trained on a heterogeneous population applies uniform predictions to a diverse group.

A diabetes risk model trained on a mixed population may underperform for specific ethnic groups that have different disease patterns.

**Label bias**

Human annotators have biases.
Labels inherit those biases.

Sentiment analysis models trained on text labeled by annotators from one culture may mislabel text from another.

---

## 4.2 Fairness Metrics

Different definitions of fairness capture different intuitions.
They are not all compatible with each other.

**Demographic Parity (Statistical Parity)**

The model's predictions should be independent of the protected attribute.

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

Equal positive prediction rates across groups.

**Problem:** if the base rates differ between groups, demographic parity forces the model to ignore real differences.

**Equalized Odds**

Both true positive rate and false positive rate should be equal across groups.

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)   [TPR equality]
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)   [FPR equality]
```

This captures: given the same true outcome, the model should treat groups equally.

Used in criminal recidivism risk assessment debates (COMPAS controversy).

**Equal Opportunity**

A relaxed version: only require equal TPR (true positive rate).

The model should identify positive cases at the same rate across groups.

**Calibration**

If the model says the probability of default is 0.7, that should be true at the same rate for all groups.

Calibration is important for high-stakes decisions where the predicted probability is used directly.

**Individual Fairness**

Similar individuals should receive similar predictions.

```
d_prediction(Ŷ(x), Ŷ(x')) ≤ L * d_input(x, x')
```

Hard to operationalize: what does "similar" mean?

---

## 4.3 The Impossibility Theorems

This is one of the most important results in fairness.

**Chouldechova (2017) and Kleinberg et al. (2016) independently showed:**

Except in degenerate cases (equal base rates between groups, or a perfect classifier), you cannot simultaneously satisfy:
- Calibration
- Equal false positive rates
- Equal false negative rates

**What this means**

There is no single "fair" definition.
Choosing a fairness criterion is a value judgment, not a technical question.

In criminal justice: should we equalize false positive rates (prevent falsely detaining innocent people) or false negative rates (prevent releasing people who will reoffend)?

These goals are in tension. The choice is ethical, not mathematical.

**Interview implication**

If someone asks "how do you make a fair model," the correct answer starts with: "which fairness criterion, and for whom?"

---

## 4.4 Bias Mitigation Techniques

**Pre-processing: fix the data before training**

- **Resampling**: oversample underrepresented groups, undersample overrepresented ones.
- **Reweighting**: give higher loss weights to examples from protected groups.
- **Data augmentation**: generate synthetic examples for underrepresented groups.
- **Label cleaning**: audit and correct biased labels.

Advantage: model-agnostic, any model trained on the cleaned data benefits.

Disadvantage: does not address all bias sources; cannot fix features that are themselves biased proxies.

**In-processing: modify training**

- **Adversarial debiasing**: add a fairness adversary during training. The model tries to be accurate; the adversary tries to predict the protected attribute from the model's representations. The model is penalized if the adversary succeeds.

- **Fairness constraints in the objective**: add a Lagrangian penalty for fairness violations.
```
L = L_accuracy + λ * L_fairness_violation
```

- **Regularization toward demographic parity**: penalize difference in group-level prediction rates.

Advantage: directly controls the accuracy-fairness tradeoff.

Disadvantage: more complex to implement; can be sensitive to hyperparameter tuning.

**Post-processing: adjust predictions after training**

- **Threshold optimization**: set different decision thresholds per group to equalize a fairness metric.
- **Calibrated equalized odds**: adjust the classifier's output probabilities to satisfy equalized odds post-hoc.
- **Reject-option classification**: for predictions near the decision boundary (uncertain cases), defer to alternative rules.

Advantage: can be applied to any existing model.

Disadvantage: treating groups differently can itself be controversial; individual fairness violations possible.

---

## 4.5 Fairness in LLMs and RLHF

Large language models introduce new fairness challenges.

**Bias in pretrained models**

LLMs trained on internet text absorb the biases in that text.
Associations between gender and profession, race and criminality, religion and violence.

These manifest as:
- Different quality responses to equivalent prompts about different groups
- Stereotyped completions
- Disparate performance on tasks related to certain cultures or languages

**RLHF and value alignment**

Reinforcement Learning from Human Feedback (RLHF) trains models to produce outputs that human raters prefer.

**The fairness problem with RLHF:**

Human raters have biases.
If raters systematically prefer responses about certain groups, the model learns those preferences.
The trained model can amplify annotator biases.

**Mitigations:**

- Diverse annotator pools across geography, culture, language
- Explicit fairness criteria in annotation guidelines
- Separate reward models for helpfulness vs harmlessness
- Red-teaming to surface disparate performance

**Constitutional AI (Anthropic)**

Rather than relying solely on human feedback, use AI feedback grounded in explicit principles.
This can reduce inconsistency in annotator biases.

**Ongoing challenge**

There is no consensus on what "fair" LLM behavior looks like.
Different cultures have different values.
A model aligned to Western liberal values may be perceived as biased by users from other contexts.

---

# 5. Model Auditing and Red-Teaming

Knowing the theory is not enough.
You need processes to find problems before deployment.

---

## 5.1 Model Auditing

A structured evaluation of a deployed model for fairness, privacy, and safety issues.

**Stages**

**Pre-deployment audit:**
- Test performance metrics disaggregated by demographic group
- Measure fairness metrics (demographic parity, equalized odds)
- Check for memorization of training data (privacy)
- Evaluate on adversarial and edge-case inputs

**Post-deployment audit:**
- Monitor for distribution shift in inputs across demographic groups
- Track outcome disparities over time
- Look for feedback loops: does the model's recommendations affect future data in ways that amplify bias?

**What to measure:**
- Accuracy, precision, recall, F1 per group
- False positive and false negative rates per group
- Coverage: does the model disproportionately abstain for certain groups?
- Calibration per group

---

## 5.2 Red-Teaming

Red-teaming is adversarial testing: trying to find failure modes and harms before deployment.

**For fairness:**
- Prompt the model with equivalent requests about different demographic groups and compare outputs.
- Probe for stereotyped associations.
- Test edge cases involving sensitive attributes.

**For privacy:**
- Membership inference attacks: can an attacker determine whether a specific example was in the training set?
- Training data extraction: can you prompt the model to reproduce training data verbatim?
- Attribute inference: can you infer private attributes from model outputs?

**Membership inference**

Train a "shadow model" on data of known membership.
Use it to distinguish training examples from non-training examples based on model confidence.

Well-calibrated models trained with DP have lower membership inference vulnerability.

**Training data extraction**

Studies on GPT-2 and GPT-3 showed that careful prompting can extract verbatim training data including email addresses, phone numbers, and copyrighted text.

**Mitigations:**

- Differential privacy during training
- Output filtering for known PII patterns
- Limiting memorization via data deduplication before training
- Temperature-based inference limits (deterministic outputs are easier to extract from)

---

# 6. Common Interview Questions

---

**Q: What is differential privacy and what does ε mean?**

Differential privacy guarantees that an observer cannot determine whether any specific individual's data was used in a computation.

Formally: for any two datasets differing in one record, the output distributions differ by at most a factor of e^ε.

Small ε means strong privacy (more noise added).
Large ε means weaker privacy (less noise).

In practice, ε is chosen based on the sensitivity of the data and the acceptable accuracy tradeoff.

---

**Q: How does DP-SGD work?**

DP-SGD makes gradient descent differentially private.

For each training step:
1. Compute per-sample gradients (not minibatch average).
2. Clip each per-sample gradient to L2 norm C.
3. Add Gaussian noise calibrated to C and privacy budget.
4. Average the noisy gradients and update the model.

The clipping bounds sensitivity; the noise provides the privacy guarantee.

---

**Q: What is federated learning and when would you use it?**

Federated learning trains a model across distributed clients without centralizing their data.

Each client trains locally on their data and sends model updates (not raw data) to a server, which aggregates them.

Use it when:
- Data is too sensitive to centralize (medical, financial, personal)
- Data cannot leave the device for regulatory reasons
- Centralizing data is too expensive or slow

---

**Q: What are the main challenges of federated learning?**

- **Non-IID data**: clients have heterogeneous data distributions. Local training can push models toward local optima, causing divergence when aggregated.
- **Communication cost**: sending model updates per round is expensive at scale.
- **Stragglers**: slow clients slow down rounds or are dropped, causing update bias.
- **Privacy amplification**: even sending gradients can leak information. Secure aggregation or DP is needed for strong privacy.

---

**Q: What is the fairness impossibility theorem?**

You cannot simultaneously achieve calibration, equal false positive rates, and equal false negative rates across groups (except in degenerate cases).

This means: every fairness criterion involves tradeoffs. There is no definition of "fair" that satisfies all intuitions simultaneously.

Practically: fairness criteria must be chosen based on the specific harms you are trying to prevent, which is an ethical decision, not a technical one.

---

**Q: What are demographic parity and equalized odds? When would you use each?**

**Demographic parity**: equal positive prediction rates across groups.
Use when: you want representation parity in outputs (e.g., loan approvals across demographics).
Problem: ignores real differences in base rates.

**Equalized odds**: equal true positive rates and false positive rates across groups.
Use when: the harms from false positives and false negatives should be equally distributed.
Example: a medical screening tool should miss cancers at equal rates across groups.

The choice depends on the specific harm you are mitigating.

---

**Q: What are the three approaches to bias mitigation?**

**Pre-processing**: modify training data before training.
- Resampling, reweighting, data augmentation.
- Model-agnostic but cannot fix all bias sources.

**In-processing**: modify training.
- Adversarial debiasing, fairness constraints in loss function.
- Direct tradeoff control but more complex.

**Post-processing**: adjust predictions after training.
- Threshold optimization per group.
- Can be applied to any existing model but may violate individual fairness.

---

**Q: What is the right to erasure and how does it affect ML?**

Under GDPR, individuals can request deletion of their personal data.

For ML: simply deleting training data is insufficient — the model has learned from it.

Machine unlearning addresses this:
- Exact unlearning: retrain from scratch without the data (correct but expensive).
- Approximate unlearning: targeted gradient updates to reduce the data's influence.
- SISA training: partition data into shards; only retrain affected shards.

---

**Q: What is membership inference and how do you defend against it?**

Membership inference: an attacker trains a shadow classifier to determine whether a given example was in the training set.

Models with high training accuracy and low generalization are most vulnerable: they behave differently on training data vs new data.

Defenses:
- Differential privacy during training (by construction limits membership inference)
- Regularization to reduce overfitting
- Output calibration (confident outputs on all inputs reduce the signal)
- Limiting access to logits (only returning hard predictions)

---

**Q: How do LLMs memorize training data and why is this a privacy risk?**

LLMs trained on large corpora can memorize rare or repeated sequences verbatim.

Memorization is more likely when:
- A sequence is repeated many times in training data
- The model is large (more capacity to memorize)
- The sequence is unique and the model cannot generalize away from it

Risk: if PII (emails, phone numbers, addresses) appears in training data, the model can be prompted to reproduce it.

Mitigations:
- Data deduplication before training (remove repeated sequences)
- Differential privacy during training
- Post-hoc output filters for PII patterns
- Audit with extraction attacks before deployment

---

**Q: How would you design a fairness-aware hiring model?**

1. **Define the problem**: what is "fair" here? Equal interview rates across demographic groups? Equal hiring rates? Equal false rejection rates?

2. **Audit training data**: check if historical hiring data reflects biased past decisions. If so, pre-processing is needed.

3. **Feature audit**: remove features that are proxies for protected attributes (zip code, name, graduation year as age proxy).

4. **Measure performance disaggregated**: precision and recall per demographic group.

5. **Choose a fairness criterion**: equalized odds is often appropriate (equal false rejection rates across groups).

6. **Apply mitigation**: threshold optimization per group or adversarial debiasing during training.

7. **Human-in-the-loop**: any automated tool in hiring decisions should have human oversight, especially for borderline cases.

8. **Ongoing auditing**: measure outcome disparities post-deployment. Feedback loops can amplify bias over time.
