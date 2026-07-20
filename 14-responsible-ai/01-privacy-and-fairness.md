---
module: Responsible AI
topic: Ml
subtopic: Privacy And Fairness
status: unread
tags: [interviewprep, ml, ml-privacy-and-fairness, interview-framing]
---
# Privacy and Fairness in Machine Learning

## What This File Is For

Two things happened that should not have happened.

**Case 1.** A bank's loan approval model denied loans to applicants in certain zip codes at higher rates. ZIP code is not race. But in US cities shaped by redlining, ZIP code predicts race. The model never saw race in its features. It encoded it through a proxy. The harm was real and measurable.

**Case 2.** A facial recognition system deployed for law enforcement had a 0.8% error rate on light-skinned men and a 34.7% error rate on dark-skinned women (Buolamwini and Gebru, 2018). The system was not "biased" in any naive sense — it optimized overall accuracy. Overall accuracy was fine. Per-group accuracy was not.

These cases are not failures of individual practitioners. They are failures that emerge predictably from standard ML practice when applied to real-world data about real-world people. Every concept in this file exists because of failures like these.

The structure for each topic:
1. What the interviewer is actually testing — the underlying competency
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

# 1. Why ML Systems Fail People

## What the interviewer is actually testing

Whether you can reason about failure modes before they occur — not just fix them after. Senior practitioners anticipate where standard ML practice generates harms. Junior practitioners react after harm is visible.

## The reasoning structure

Three failure mechanisms account for most privacy and fairness harms:

**Proxy encoding.** Protected attributes (race, gender, disability) were legally excluded from decisions in many domains. But the world correlates protected attributes with ZIP codes, names, educational institutions, browsing behavior. A model trained to predict an outcome will use whatever features are predictive. If ZIP code is predictive because of historical segregation, the model learns and amplifies that correlation.

**Representation failure.** A model trained on data that underrepresents a group will underperform for that group. The facial recognition case is the canonical example. Training data from stock photo databases skewed toward light-skinned faces. The model minimized average error. Average error does not expose per-group failure.

**Feedback loops.** A predictive policing model sends more officers to flagged neighborhoods. More officers generate more arrests in those neighborhoods. The next model trains on that data and flags the same neighborhoods more strongly. The loop amplifies an initial skew until it is structurally self-reinforcing.

## The pattern in action

The COMPAS recidivism risk tool, used across US courts, assigned risk scores predicting re-offense. ProPublica's 2016 analysis found:
- Black defendants were nearly twice as likely to be falsely flagged as high-risk (false positive rate: 44.9% vs 23.5%)
- White defendants were more likely to be falsely classified as low-risk (false negative rate: 47.7% vs 28.0%)

Northpointe (the COMPAS vendor) responded: the tool is calibrated — for a given risk score, the actual re-offense rate is the same across races.

Both claims were true. They are measuring different things. They cannot both be satisfied simultaneously when base rates differ between groups.

## Common traps

**"We didn't include protected attributes."** Proxy features do the work anyway. The fix is not excluding race — it is testing whether outcomes correlate with race.

**"Our model is accurate."** Accurate on average does not mean accurate for your most vulnerable users. Always disaggregate metrics by relevant groups.

**"Fairness is a post-deployment concern."** Fairness problems are data problems. They exist in training data before a model is ever trained. Auditing post-deployment is too late to prevent harm from the initial deployment.

---

# 2. Data Privacy Fundamentals

## What the interviewer is actually testing

Whether you understand that "removing PII" is not privacy. Anonymization is a claim that requires verification, not a property that removing a name column confers.

## The reasoning structure

**The re-identification problem.** Latanya Sweeney (2000) showed that 87% of Americans can be uniquely identified by three fields: ZIP code, date of birth, and sex. None of these are direct PII. Each is a quasi-identifier. Their combination is identifying.

**Direct PII vs quasi-identifiers:**
- Direct: name, email, phone number, SSN, passport number
- Quasi-identifiers: ZIP code, date of birth, gender, job title, employer, medical condition, browser fingerprint

A dataset can contain no direct PII and still be re-identifiable by linking quasi-identifiers against public records.

**Why ML makes this worse.** Training a model introduces a second re-identification surface: the model itself. Large language models have been shown to reproduce training data verbatim, including email addresses and phone numbers. The model is not just a summary of the data — it can contain the data.

**GDPR principles that constrain ML practice:**
- Lawful basis for processing: need consent or legitimate interest to train on personal data
- Data minimization: collect only what you need (in tension with "collect everything" ML instinct)
- Purpose limitation: data collected for fraud detection cannot be repurposed to train a recommendation model
- Storage limitation: personal data cannot be retained indefinitely, creating challenges for models trained on historical records
- Right to erasure: individuals can request deletion of their data

## The pattern in action

Netflix released an "anonymized" dataset of user movie ratings for a prize competition. Researchers linked it to public IMDb ratings and re-identified 99% of records, exposing viewing history including politically sensitive films.

The dataset contained no names. It contained a unique behavioral fingerprint.

**What this means for ML practice:**
- Anonymization requires k-anonymity (each record indistinguishable from at least k-1 others), l-diversity, t-closeness, or formal differential privacy guarantees
- Simply removing identifying columns is not anonymization
- Every dataset released from a company should be treated as potentially re-identifiable

## Common traps

**"We removed the name column."** Re-identification via quasi-identifier combinations. You need formal privacy guarantees, not column deletion.

**"Our data is internal — not released publicly."** Internal datasets can be accessed by engineers, vendors, or breach attackers. Privacy-preserving techniques protect against internal exposure too.

**"The model doesn't output individual records."** Models memorize training data. LLMs reproduce text verbatim. Privacy attacks against the model are privacy attacks against the training data.

---

# 3. Machine Unlearning and the Right to Be Forgotten

> This covers the unlearning algorithms (exact retraining, influence functions, SISA). For how an erasure request flows through a production system end-to-end (registry lookup, influence thresholding, audit-log pseudonymization, GDPR Article 17 vs. financial retention conflicts) see [model-governance.md §5](../13-production-ml/03-model-governance.md#5-regulatory-compliance).

## What the interviewer is actually testing

Whether you can translate a legal requirement (GDPR right to erasure) into a technical problem and reason about its difficulty honestly. This is a signal that you have thought about ML systems in production legal environments.

## The reasoning structure

**The problem.** If Person X's data was in your training set and they request erasure, deleting their database record does not erase the model's learned parameters that were shaped by their data. Retraining from scratch works but costs O(entire training run). Neither option is practical at scale.

**Machine unlearning** is the research area that aims to efficiently update a trained model to "forget" specific training examples.

**Three approaches, ordered by correctness vs. cost:**

**1. Exact unlearning — retrain from scratch.** Remove the data, retrain the model from initialization. Correct by definition. Cost: O(full training run per erasure request). Impractical for large models or frequent requests.

**2. Approximate unlearning — gradient-based.** Apply targeted gradient updates to reduce the influence of specific data points. Based on influence functions (Koh and Liang, 2017): compute how the model's parameters would change if training example i were removed. Use that to construct a targeted update. Faster than retraining. Harder to verify — the guarantee is approximate.

**3. SISA training (Sharded, Isolated, Sliced, and Aggregated).** Partition the training set into shards. Train a sub-model on each shard. Final predictions are aggregated from sub-models. When erasure is requested for a data point in shard k, only retrain the sub-model on shard k. Cost: O(1/num_shards × full training run) per erasure. Correct for the affected shard. Requires architectural forethought before training.

## The pattern in action

A healthcare platform trains a diagnostic model on patient records. A patient exercises GDPR right to erasure.

**What they cannot do:** delete the database row and declare compliance. The model's weights are shaped by that patient's records.

**What they can do (SISA):**
1. Before training: partition patient records into 100 shards of ~1% each
2. Train 100 sub-models, one per shard
3. Final prediction = majority vote or average across sub-models
4. Erasure request: identify which shard contains the patient, retrain that sub-model only
5. Cost per erasure: approximately 1% of full training run

**Verifying unlearning:** run membership inference attacks against the updated model on the erased examples. If they are indistinguishable from non-training examples, unlearning succeeded.

## Common traps

**"We deleted the data so we're compliant."** The model still contains that data's influence. Regulators are increasingly aware of this.

**"We'll just retrain when someone asks."** This works for small models. For a 70B parameter model trained for months, retraining on each erasure request is not a viable operational plan.

**"Approximate unlearning is fine."** It may not satisfy regulators who require a verifiable guarantee. The verification gap between approximate unlearning and actual erasure is an open problem.

---

# 4. Differential Privacy

## What the interviewer is actually testing

Whether you can state the formal guarantee, explain what epsilon means intuitively, and know when DP is the right tool vs. when it is security theater. Many candidates can define DP; fewer can explain why the formal definition is the right one.

## The reasoning structure

**The motivating question.** You want to learn statistics about a population without learning anything about individuals. Is this possible?

The naive answer is no — any statistic reveals something about the individuals who produced it. DP's answer: you can bound exactly how much an individual's participation reveals, and you can make that bound arbitrarily small.

**The formal definition.** A randomized mechanism M is ε-differentially private if for any two datasets D and D' that differ in exactly one record, and for any output set S:

```
P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S]
```

**What this says.** The output distribution barely changes when one person's data is included or excluded. An observer cannot determine from the output whether any specific person participated.

**The privacy budget ε.**
- ε = 0: perfectly private, completely useless (output is constant)
- ε small (0.1–1): strong privacy, significant accuracy loss
- ε = 1: industry academic standard for strong DP
- ε = 10: Apple/Google range for telemetry; pragmatically private
- ε large: weak guarantee, barely more than publishing raw data

**Composition.** Running k ε-differentially private mechanisms gives total privacy loss at most k·ε (basic composition). Advanced composition gives √(2k ln(1/δ)) · ε for (ε,δ)-DP. The budget depletes with each query. You cannot run unlimited private queries.

## The pattern in action

**Laplace mechanism for numeric queries:**

```
M(D) = f(D) + Lap(sensitivity / ε)
```

Sensitivity = maximum change in f(D) from adding/removing one person.

Average income query, values in [0, 200K], n = 10,000 people:
- sensitivity = 200K / 10,000 = 20
- ε = 1 → scale = 20/1 = 20 → standard deviation ≈ 28K
- ε = 10 → scale = 2 → standard deviation ≈ 2.8K

With n=10,000, the mean is robust enough that ε=10 adds imperceptible noise. Privacy cost: an observer cannot determine with confidence whether any specific person's income was in the dataset.

**Gaussian mechanism for (ε,δ)-DP:**

```
M(D) = f(D) + N(0, σ² · I)
```

Provides (ε,δ)-DP: the guarantee fails with probability δ (e.g., 10⁻⁵). Useful for vector-valued queries (gradient updates in DP-SGD).

## Common traps

**"We used ε=100 so it's differentially private."** Technically true, practically meaningless. e^100 ≈ 2.7 × 10^43. The bound allows the output distribution to change by an astronomical factor. DP with large ε is privacy theater.

**"DP protects against all privacy attacks."** DP bounds information leakage about individuals. It does not prevent membership inference (you can still learn statistical information), re-identification via external data, or model inversion. It bounds, it does not eliminate.

**Forgetting composition.** Running 100 ε=1 mechanisms gives total privacy loss up to ε=100. You must track the privacy budget.

---

# 5. DP-SGD: Private Machine Learning

## What the interviewer is actually testing

Whether you can extend DP from simple statistics to iterative optimization, and whether you understand why each step of DP-SGD is necessary — not just what the steps are.

## The reasoning structure

**The problem with standard SGD.** Each gradient step uses a minibatch of training examples. The gradient aggregates information from that minibatch. A single example with extreme features can dominate the gradient update, and an observer watching the gradient updates can infer information about individual examples.

**The solution has two parts:**
1. **Bound the influence of any single example** — clip per-sample gradients to a maximum L2 norm C
2. **Add calibrated noise** — add Gaussian noise to the clipped gradient, providing the DP guarantee

**DP-SGD algorithm (Abadi et al., 2016):**

1. Sample a minibatch of training examples
2. Compute the gradient for each example individually (per-sample gradients, not minibatch average)
3. Clip each per-sample gradient: g̃_i = g_i / max(1, ‖g_i‖₂/C)
4. Add Gaussian noise: g̃ = (1/B) · Σ g̃_i + N(0, σ²C²I)
5. Update the model: θ = θ - η · g̃

**Why clip?** Without clipping, one example's gradient can be arbitrarily large. Clipping bounds the sensitivity — the maximum change in the gradient from adding or removing one example is 2C/B. The noise must be calibrated to this sensitivity.

**The privacy accountant.** Over T steps of DP-SGD, the total privacy loss is tracked using the moments accountant (Abadi et al.) or Rényi DP. The result: training for T steps with batch fraction q, noise multiplier σ, gives (ε, δ)-DP with ε that grows sublinearly in T.

**The accuracy cost.** Noise added per step degrades gradient quality. Effect is worse for:
- Large models (more dimensions → more total noise)
- Small batch sizes (noise-to-signal ratio is higher)
- Tight ε requirements (more noise needed)

## The pattern in action

Google trained a language model on Gboard keyboard data using DP-SGD. The data is sensitive (what people type). The deployed model needed to predict next words without any individual user's typing being recoverable.

Setting: ε=0.0065 per user per day (very tight, strong DP). The noise made training significantly noisier than standard SGD. But for federated keyboard prediction, a slightly less accurate model with strong privacy guarantees is the correct engineering tradeoff.

Libraries: **Opacus** (PyTorch) and **TF Privacy** compute per-sample gradients efficiently using vectorized Jacobian computation. Without these, DP-SGD requires batch_size × forward passes per step, making it unusable.

## Common traps

**"Per-sample gradients are just gradients on a batch of size 1."** Conceptually yes, but implementing it as a loop of batch-size-1 forward passes is 32× slower than a vectorized computation. Opacus uses hooks to extract per-sample gradients without this overhead.

**"We clipped gradients so we have DP."** Clipping alone bounds sensitivity but provides no privacy. You also need calibrated noise. Both steps are required.

**Ignoring the privacy cost of hyperparameter tuning.** Every training run you do while selecting ε, batch size, and noise level consumes privacy budget. Tuning hyperparameters burns DP budget even before the final model is trained.

---

# 6. Federated Learning

## What the interviewer is actually testing

Whether you understand when the architecture of federated learning provides value, and whether you can reason about its failure modes under real-world conditions (non-IID data, stragglers, communication cost).

## The reasoning structure

**The motivating problem.** Standard ML collects all data centrally, trains, then deploys. This works when data can be collected. It fails when:
- Data is legally or contractually impossible to centralize (HIPAA, GDPR, inter-hospital data sharing)
- Data is physically too large to move (satellite imagery, medical imaging across 10,000 hospitals)
- Users do not consent to sharing their raw data (mobile keyboard input, health sensors)

**Federated learning inverts the data flow.** Instead of moving data to the model, move the model to the data. Each participant trains locally. Only model updates are transmitted.

**FedAvg (McMahan et al., 2017):**

Each round t:
1. Server selects K clients
2. Server sends current global weights w_t to each client
3. Each client runs E epochs of SGD on local data: w_k = LocalUpdate(w_t, D_k)
4. Clients return w_k to server
5. Server aggregates: w_{t+1} = Σ_k (n_k / n) · w_k

The key insight: averaging weight updates from local SGD approximates SGD on the union of all data, without ever centralizing that data.

## The pattern in action

**The non-IID problem is the central challenge.** Standard ML assumes training data is IID — drawn independently from the same distribution. Federated data violates this by construction.

Hospital A's patients are sicker (referral center). Hospital B's patients reflect a regional population. Hospital C sees mostly pediatric cases. Local models trained on these datasets will drift toward their local distributions. Averaging locally drifted models produces a global model that may be worse than centralizing the data.

**Mitigations:**
- **FedProx**: add proximal term to local objective: minimize L_local(w) + (μ/2)‖w - w_t‖² — penalizes diverging from the global model
- **SCAFFOLD**: control variates that correct for client drift — each client tracks a local correction term
- **FedNova**: normalizes updates by number of local steps before aggregation — removes scale differences between clients

**Communication cost at scale.** A 100M parameter model sends 400MB of float32 weights per round. With 10,000 clients per round and 1,000 rounds, total communication is 4 petabytes. **Gradient compression** (top-k sparse gradients, quantization to 8-bit or lower) reduces this by 10–100×.

**Stragglers.** If round t waits for all K selected clients, one slow client blocks the round. Solutions: partial participation (proceed when 80% respond), asynchronous FL (accept updates as they arrive — but stale updates from slow clients can hurt convergence).

**Secure aggregation.** The server sees individual client updates in standard FedAvg. Bonawitz et al. (2017) developed secure aggregation using pairwise random masks: clients mask their updates, masks cancel in the sum, server learns only the aggregate. Even the server cannot recover individual updates.

## Common traps

**"Federated learning is private because the server never sees raw data."** The server sees gradient updates, which can leak information about local data. Gradient inversion attacks can approximately reconstruct training data from gradients. Federated learning is a step toward privacy, not sufficient for it.

**"Non-IID is a minor issue."** In practice, non-IID data causes FedAvg to diverge or converge to a poor optimum on many tasks. It is the main accuracy challenge in federated learning.

**Ignoring communication cost in design.** At 10M mobile clients, naive FedAvg is infeasible. Communication budget must be a first-class constraint in the system design.

---

# 7. Fairness Metrics

## What the interviewer is actually testing

Whether you know the formal definitions AND can reason about which metric is appropriate for a given harm. Anyone can recite demographic parity. The test is whether you can explain why the COMPAS case required equalized odds rather than demographic parity, or why calibration alone was insufficient.

## The reasoning structure

**Start from the harm, not the metric.**

The loan denial case: the harm is that creditworthy Black applicants are denied at higher rates than equally creditworthy white applicants. The relevant metric is equal true negative rates — denial of qualified applicants should be equal across groups.

The facial recognition case: the harm is that dark-skinned faces are misidentified at far higher rates. The harm is unequal error rates. The metric is equal false positive rates and false negative rates — equalized odds.

**The five main metrics:**

**Demographic parity (statistical parity):**
```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```
Equal positive prediction rates across groups. Does not condition on the true label. Forces equal representation in predicted positives regardless of ground truth differences.

**Equalized odds:**
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)   [equal TPR]
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)   [equal FPR]
```
Conditions on the true label. Equal error rates for people who truly qualify and for people who truly do not. The COMPAS requirement.

**Equal opportunity:**
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```
Equal TPR only. The model should identify positive cases at the same rate across groups. Use this when false positives are relatively harmless but false negatives are costly (medical screening).

**Calibration:**
```
P(Y = 1 | Ŷ = p, A = 0) = P(Y = 1 | Ŷ = p, A = 1) = p
```
If the model says 70% probability, it should be correct 70% of the time for all groups. Essential when predicted probabilities are used directly in decisions.

**Individual fairness:**
```
d_prediction(Ŷ(x), Ŷ(x')) ≤ L · d_input(x, x')
```
Similar individuals should receive similar predictions. Hard to operationalize — requires defining a task-specific similarity metric.

## The pattern in action

**Loan approval model.** Group A: default rate = 5%. Group B: default rate = 15%.

**Demographic parity** forces equal approval rates. But approving the same fraction of both groups means either approving high-risk B applicants or rejecting low-risk A applicants. It ignores the base rate difference.

**Equalized odds** requires: of creditworthy applicants (Y=0, will not default), approve at equal rates across groups; of non-creditworthy applicants (Y=1, will default), reject at equal rates. This is what equal treatment actually means given base rate differences.

**Calibration** alone (the COMPAS vendor's defense) means: when the model says 70% risk, it is right 70% of the time for both groups. True but insufficient — if A's true risk is lower, calibration does not prevent the model from assigning higher scores to A members through proxy features.

## Common traps

**"We'll just check if group approval rates are equal."** That is demographic parity, which ignores base rates. Equal rates with different base rates either over-approves high-risk applicants or under-approves low-risk ones.

**Treating fairness as a post-hoc threshold adjustment.** If the model's scores are biased by proxy features, adjusting thresholds compensates but does not fix the root cause. Features that encode protected attributes via proxies corrupt the entire score.

**Picking a fairness metric before defining the harm.** The metric should follow from the harm. Define who is harmed, how, and then choose the metric that captures that harm. Starting from the metric produces solutions to the wrong problem.

---

# 8. The Impossibility Theorems

## What the interviewer is actually testing

Whether you can explain why fairness is inherently a value choice, not a technical optimization. This distinguishes candidates who understand fairness from those who have memorized fairness definitions.

## The reasoning structure

**Chouldechova (2017) and Kleinberg et al. (2016)** independently proved:

Except when base rates are equal across groups OR the classifier is perfect, you cannot simultaneously satisfy:
- Calibration (equal accuracy of predicted probabilities)
- Equal false positive rates
- Equal false negative rates

**The COMPAS illustration.** Group A re-offends at rate 40%. Group B re-offends at rate 30%.

A calibrated model: if it assigns risk score 60%, 60% of those actually re-offend, for both groups.

For calibration to hold with different base rates, the model must assign different score distributions to the two groups. The group with higher base rates gets higher average scores.

Equal FPR: among those who do not re-offend, the fraction flagged as high-risk should be equal. But because Group A has more true positives at any threshold (higher base rate), matching FPR requires different thresholds. Different thresholds with equal FPR means unequal FNR.

You cannot equalize FPR and FNR simultaneously when base rates differ. Calibration + equal FPR + equal FNR requires equal base rates. When base rates differ, you must choose.

## The pattern in action

**Criminal justice (COMPAS).** Different people weight the harms differently:
- Prioritize equal FPR: false detention is the primary harm. Reduce false positives equally across groups.
- Prioritize equal FNR: public safety harm is the primary concern. Reduce missed re-offenders equally.
- Prioritize calibration: if the model says 70% risk, that should be true for both groups so decision-makers get accurate information.

These are ethical stances, not technical positions. The data cannot resolve them.

**Medical screening.** A cancer screening model. False negative = missed cancer (patient dies). False positive = unnecessary biopsy (patient inconvenienced). For most cancers, minimizing FNR is ethically dominant. Choose equal opportunity (equal TPR) over demographic parity or equal FPR.

## Common traps

**"We just need better data."** The impossibility result holds for any dataset where base rates differ. Better data does not change the mathematical constraint.

**"We'll satisfy all fairness criteria."** This is the core impossibility — you cannot. Every fairness criterion involves a choice about whose errors to prioritize. Pretending otherwise is either confusion or dishonesty.

**Not naming the ethical choice explicitly.** A model deployed in production implies a fairness criterion, even if no one chose it deliberately. The implicit choice is usually "maximize overall accuracy," which typically harms minority groups with lower representation. Making the choice explicit is a requirement for responsible deployment.

---

# 9. Bias Mitigation

## What the interviewer is actually testing

Whether you understand which stage of the pipeline each technique affects, and why attacking bias at the right stage matters. Many candidates jump to adversarial debiasing without checking if the problem is in the data.

## The reasoning structure

Bias has three origins — and the mitigation should match:

1. **Biased data.** Training labels reflect historical discrimination. Fix: pre-processing.
2. **Biased objective.** Training optimizes aggregate accuracy, which under-weights minority groups. Fix: in-processing.
3. **Biased deployment.** A fair model is used with an unfair threshold. Fix: post-processing.

Applying in-processing to a biased-labels problem is attacking the wrong stage.

**Pre-processing: fix the data before training.**
- **Resampling**: oversample underrepresented groups; undersample overrepresented ones. Improves representation but does not fix label bias.
- **Reweighting**: weight training examples by group membership or proxy of fairness violation. Penalizes errors on underrepresented groups more.
- **Data augmentation**: generate synthetic examples for underrepresented groups (for CV: augment dark-skinned faces in facial recognition training data).
- **Label cleaning**: audit and correct labels that reflect annotator bias (e.g., sentiment labels applied differently to text from different cultural contexts).

Advantage: model-agnostic. Any model trained on cleaned data benefits.
Disadvantage: cannot fix proxy-feature encoding without feature engineering.

**In-processing: modify training.**
- **Adversarial debiasing**: add a fairness adversary during training. Predictor minimizes task loss. Adversary tries to predict the protected attribute from the predictor's representation. Predictor is penalized when adversary succeeds. Result: representations that are predictive of the target but not of the protected attribute.
- **Fairness-constrained objective**: Lagrangian relaxation with fairness constraint:
```
minimize L_accuracy + λ · L_fairness_violation
```
- **Equalized odds post-processing via regularization**: add a gradient-based penalty for TPR or FPR disparities across groups.

Advantage: directly controls the accuracy-fairness tradeoff during training.
Disadvantage: more complex; sensitive to λ; adversarial training can be unstable.

**Post-processing: adjust predictions after training.**
- **Threshold optimization per group**: find separate decision thresholds for each group that minimize the chosen fairness criterion. Simple and effective when the model's predicted probabilities are reliable.
- **Calibrated equalized odds (Pleiss et al., 2017)**: adjust output probabilities post-hoc to satisfy equalized odds.
- **Reject-option classification**: for predictions near the decision boundary, defer to a human reviewer or secondary rule. Avoids automated decisions in the high-uncertainty region where proxy bias is most harmful.

Advantage: can be applied to any existing model without retraining.
Disadvantage: separate thresholds for groups requires knowing group membership at decision time. Can violate individual fairness (two similar individuals from different groups get different treatment).

## The pattern in action

Facial recognition with 35% error on dark-skinned women. Tracing the failure:

1. Training data: 86.2% of images in benchmark datasets were light-skinned (IJB-A, Adience — actual measured figures from Buolamwini/Gebru).
2. Model: minimized average error. Average error was excellent. Per-group error was not measured.
3. Deployment: used in high-stakes contexts (law enforcement matching) without per-group evaluation.

Mitigation sequence:
1. Pre-processing: augment training data with dark-skinned face images (both collected and synthetic)
2. In-processing: weight loss function by inverse group frequency during training
3. Evaluation: mandate disaggregated metrics (accuracy per demographic subgroup) as a deployment gate — do not deploy until per-group accuracy is acceptable
4. Post-processing: raise the match confidence threshold (lower FPR at the cost of FNR) for high-stakes identification use cases

## Common traps

**Threshold optimization as the universal fix.** Changing thresholds does not fix a model that encodes protected attributes in its representations. The scores themselves are biased. Threshold changes reduce one type of error for one group while potentially worsening it for another.

**Adversarial debiasing as the silver bullet.** Adversarial training removes the linear ability of an adversary to predict the protected attribute from the representation. Nonlinear relationships may persist. Downstream accuracy loss can be significant. Run careful ablations before committing to this approach.

**Not measuring what you fixed.** After applying any mitigation, measure the relevant fairness metric on a held-out test set. And measure the accuracy cost. The goal is explicit knowledge of the tradeoff, not hoping the mitigation worked.

---

# 10. Fairness in LLMs

> **See also:** [`../10-llms/interview-notes/10-ai-safety-ethics-and-responsible-ai-what.md`](../10-llms/interview-notes/10-ai-safety-ethics-and-responsible-ai-what.md)
> covers the behavioural side — guardrails, prompt injection, content-moderation calibration —
> and [`../10-llms/interview-notes/17-advanced-alignment-and-reasoning.md`](../10-llms/interview-notes/17-advanced-alignment-and-reasoning.md)
> covers RLHF/DPO/Constitutional AI. This section stays with the fairness mathematics.


## What the interviewer is actually testing

Whether you can extend fairness reasoning beyond classification models to generative systems, where the failure modes are different and the measurement is harder.

## The reasoning structure

**Standard fairness metrics do not transfer directly.** Equalized odds requires a label Y. LLMs generate free text — what is Y? The failure modes of generative models are:
- **Stereotyped completions**: "The engineer walked to her..." → model completes differently based on whether name suggests male/female
- **Disparate quality**: responses about underrepresented groups are lower quality, less factually accurate
- **Disparate representation**: some groups are overrepresented as negative examples in associations (race-crime co-occurrence)
- **Cultural miscalibration**: model aligned to Western liberal values may fail users from other cultural contexts

**Bias in pretraining.** LLMs trained on internet text absorb internet biases. The model learns the statistical regularities of the training corpus, including biased ones.

**RLHF amplifies annotator biases.** RLHF trains the model to produce outputs that human raters prefer. If raters systematically prefer responses about certain groups (cultural familiarity, language quality), the model learns those preferences. The model can amplify annotator biases rather than correct them.

## The pattern in action

**Measuring LLM bias (practical approach):**

1. **Counterfactual probing**: construct pairs of prompts that are identical except for a protected attribute reference. Compare outputs. The WinoBias benchmark does this for gender and profession.
2. **Embedding bias tests**: WEAT (Word Embedding Association Test) measures whether word embeddings associate certain concepts with certain groups beyond ground truth rates.
3. **Toxicity probing**: prompt with text about different groups; measure toxicity of completion using a classifier.

**Mitigations:**
- Diverse annotator pools: geographic, cultural, linguistic diversity in RLHF annotation reduces single-culture skew
- Explicit fairness criteria in annotation guidelines: annotators are instructed on what constitutes biased output
- Constitutional AI: use AI feedback grounded in explicit principles rather than relying on potentially biased human preferences alone
- Red-teaming: systematic adversarial testing before deployment to surface differential quality across groups
- Output filtering: detect and filter stereotyped or biased outputs post-generation (last resort; does not fix the model)

## Common traps

**"The LLM is just reflecting reality."** Statistical regularities in training data reflect historical biases, not ground truth distributions. The model learns what was written about groups, not what is true about groups. These differ substantially for historically marginalized groups.

**"We added diversity to our RLHF pool."** Annotator diversity is necessary but not sufficient. The annotation rubric also needs to explicitly address fairness. Without that, a diverse pool may still exhibit consistent biases on certain questions.

**No systematic evaluation.** LLM bias cannot be assessed by manual inspection of a few outputs. It requires systematic probing across hundreds of test cases per group, using validated benchmarks and adversarial prompts.

---

# 11. Model Auditing and Membership Inference

## What the interviewer is actually testing

Whether you understand how to operationalize privacy and fairness concerns into concrete measurement processes, and whether you know the specific attacks that reveal privacy violations.

## The reasoning structure

**Auditing is not just metrics computation.** A fairness audit answers: what harms could this model cause, for whom, and at what rate? It requires disaggregated evaluation, adversarial testing, and reasoning about deployment context.

**Membership inference as a privacy attack.** An attacker observes model outputs (probabilities, logits, or just the prediction) for a target example. The attacker wants to determine whether that example was in the training set.

**Why it works:** models that overfit produce different outputs on training examples vs. test examples. Training examples get higher confidence scores. An attacker trains a shadow classifier to distinguish training from non-training examples based on these output signatures.

**Attack success indicates a privacy violation.** If training membership is detectable from outputs, then model outputs reveal information about individual training examples.

**Defenses:**
- Differential privacy during training: by construction limits membership inference — the output distribution is bounded to differ by at most e^ε between including and excluding any example
- Regularization: reduces overfitting, which reduces the training/test distribution gap
- Output calibration: well-calibrated models produce consistent confidence regardless of training membership
- Limiting output precision: returning only the top class (not probabilities) reduces the signal available to attackers

## The pattern in action

**Training data extraction from LLMs (Carlini et al., 2021).** GPT-2 was prompted with the beginning of memorized sequences. With careful prompting (using prefix text that appeared in training), researchers extracted:
- Full names and email addresses of individuals
- Copyrighted passages
- Phone numbers

Models memorize when sequences appear repeatedly in training data. Deduplication of training corpora before training significantly reduces memorization. The study found that 1,000× longer generation attempts extracted significantly more memorized content.

**Pre-deployment audit checklist:**
- Performance metrics disaggregated by demographic group (accuracy, FPR, FNR per group)
- Fairness metric measurement (demographic parity, equalized odds — chosen based on the harm model)
- Membership inference attack audit: measure attack AUC on a held-out test set
- Data extraction probing for LLMs: attempt extraction of known PII from training data
- Calibration per group: confirm predicted probabilities are equally accurate across groups

**Post-deployment monitoring:**
- Input distribution shift per group (are certain groups' inputs shifting out of training distribution?)
- Outcome disparity tracking (are error rates changing over time for any group?)
- Feedback loop detection (do the model's predictions affect the distribution of future inputs?)

## Common traps

**Auditing at deployment and then never again.** Distribution shift can introduce new fairness violations after deployment, especially in systems with feedback loops. Auditing must be continuous.

**Only measuring overall membership inference AUC.** If overall AUC is 0.55 (slightly better than chance) but AUC for a specific subgroup is 0.80, that subgroup has a serious privacy exposure. Disaggregate the audit.

**Not red-teaming for data extraction.** For LLMs especially, membership inference is less informative than direct extraction probing. Attempting to extract known PII sequences from the deployed model is the most direct test of whether the model poses a data privacy risk.

---

# Quick Diagnostics

**If asked to audit a model for fairness before deployment:**

Start by defining the harm: who could be harmed, how? This determines which fairness metric to measure. Compute disaggregated accuracy, FPR, and FNR per demographic group. Apply the chosen fairness metric. If violated, trace back to whether the violation is in the data (biased labels, underrepresentation), the objective (aggregate optimization), or the threshold (deployment setting). Mitigation strategy follows from the diagnosis.

**If asked how differential privacy protects against membership inference:**

DP bounds the change in output distribution when any single training example is added or removed. By construction, this limits how much the model's behavior on a specific example can reveal whether that example was in training. The membership inference attack depends on detecting a difference in model behavior between training and non-training examples. DP limits that difference to a factor of e^ε.

**If asked why removing protected attributes is insufficient for fairness:**

Proxy features encode protected attributes through correlations present in the data. ZIP code predicts race in US cities due to historical segregation. Name frequency distributions predict gender and ethnicity. College attended predicts socioeconomic background. The model learns these correlations whether or not the protected attribute is included. The solution is to test for disparate outcomes by group, not to remove attributes.
