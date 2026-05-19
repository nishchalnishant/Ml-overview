# Privacy-Preserving Machine Learning

---

## 1. The Problem

In 2021, Nicholas Carlini and colleagues at Google took a fine-tuned GPT-2 and queried it with the prompt "My social security number is". The model completed it with a real social security number — one that appeared verbatim in the training data.

They extracted 604 memorized training examples from GPT-2 using only 1,800 API queries. Phone numbers, names, email addresses, code snippets, poems — all lifted directly from the training set. The model had not been trained to reproduce them. It had simply memorized them as a consequence of being trained to minimize next-token loss on a large web corpus.

This is the foundational privacy problem in ML: **training a model does not destroy the training data. It encodes it.** And that encoding can be interrogated.

The consequences scale with model size and capability. Larger models memorize more. Fine-tuned models memorize their fine-tuning data more readily than pre-training data, because fine-tuning datasets are smaller and each example is seen more times. A model fine-tuned on electronic health records, legal documents, or financial data is a potential vector for data reconstruction.

Memorization enables three concrete attacks:

- **Membership inference:** determine whether a specific record was in the training set (privacy violation even without reconstruction)
- **Model inversion:** reconstruct approximate training inputs from model outputs
- **Training data extraction:** directly reproduce verbatim training examples

The existence of these attacks is not theoretical. They have been demonstrated at production scale on deployed models.

---

## 2. Two Answers to the Same Problem

The Carlini attack reveals two distinct things the training process does that enable it:

1. **The data reaches the training machine.** The data was centralized on Google's servers to train GPT-2. Centralization is itself a privacy risk — it creates a single point of exposure.

2. **The model encodes individual records.** Even if the data reached the server legitimately, the resulting model "remembers" specific individuals. The encoding can be queried.

These are independent problems, and they have independent solutions:

- **Differential privacy** addresses the second problem: bound how much any individual's data can affect the model, so the encoding is provably limited.
- **Federated learning** addresses the first problem: train without centralizing the raw data at all.

Both are responses to the same root failure. They operate at different points in the pipeline.

---

## 3. Differential Privacy (DP)

### The Guarantee

Algorithm `M` is `(ε, δ)`-differentially private if, for all datasets D and D' differing by one individual's record, and all possible outputs S:

```
P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ
```

Adding or removing any single person's data changes the output distribution by at most a factor of `e^ε` (with failure probability `δ`). This bounds the privacy risk to any individual in the training set.

- `ε` (epsilon): privacy budget. Smaller = stronger privacy = more noise. ε = 1 is strong; ε > 10 is weak.
- `δ` (delta): failure probability. Typically 1/n² or smaller.

The guarantee is not "your data cannot be extracted." It is "your data's presence or absence cannot be detected beyond a bounded probability." That bound directly limits membership inference attack success.

### Mechanisms

**Gaussian Mechanism:** add `N(0, σ²Δf²)` noise where `Δf` is the L2 sensitivity of the query.

**Laplace Mechanism:** add `Lap(Δf/ε)` noise. Used for pure DP (δ=0) with L1 sensitivity.

**Randomized Response:** for boolean queries — answer truthfully with probability p, randomly otherwise. Classic mechanism for survey data collection.

---

### DP-SGD: Making Deep Learning Differentially Private

Standard SGD computes the gradient of the loss over a minibatch. That gradient aggregates individual training examples' contributions. A single individual's unusual example can shift the gradient arbitrarily far — which is exactly how memorization happens.

DP-SGD makes each training step private:

1. Compute per-sample gradients (not the minibatch average)
2. Clip each per-sample gradient to L2 norm ≤ C
3. Sum the clipped gradients and add Gaussian noise
4. Divide by batch size to get the noisy gradient estimate

```
g̃ = (1/B)(Σ_i clip(g_i, C) + N(0, σ²C²I))
```

The clipping step bounds each individual's influence. The Gaussian noise hides whether any particular individual was in the batch.

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

model = ModuleValidator.fix(model)  # Replace incompatible layers (e.g., BatchNorm → GroupNorm)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=10,
    target_epsilon=5.0,
    target_delta=1e-5,
    max_grad_norm=1.0,   # per-sample gradient clipping bound C
)

for batch in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"ε = {epsilon:.2f}")
```

**Privacy accounting:** each training step consumes privacy budget. The Moments Accountant or PRV (Privacy Random Variables) tracks cumulative `(ε, δ)` over all steps across the entire training run.

**Privacy-utility tradeoff:** more noise = lower accuracy. The gap shrinks with larger datasets (more data per ε budget), better initialization (pre-trained model requires fewer fine-tuning steps), and higher ε.

---

### DP Fine-tuning for LLMs

The most practical deployment: pre-training data is large and public (Internet text, books), contributing little privacy risk. The fine-tuning dataset is small and sensitive (clinical notes, financial records). Apply DP-SGD only during fine-tuning.

LoRA reduces the number of trainable parameters dramatically. With fewer parameters being updated, clipping and noise addition are cheaper and more precise:

```python
from peft import get_peft_model, LoraConfig

lora_model = get_peft_model(base_model, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
# Apply Opacus to lora_model — only LoRA params are trained and noised
```

**Ghost Clipping:** computing per-sample gradients requires individual backward passes, which is expensive. Ghost Clipping computes per-sample gradient norms without materializing the gradients explicitly, reducing memory overhead significantly.

---

## 4. Federated Learning (FL)

### The Problem It Solves

DP-SGD requires centralized training: all data must reach one machine so gradients can be computed. For many applications, this centralization is itself the primary threat — the data must never reach the server at all.

Hospitals cannot share patient records across institutions. Mobile phones cannot upload keyboard inputs to a server. Banks cannot pool transaction records. Federated learning enables model training without centralizing raw data.

### FedAvg (McMahan et al., 2017)

Each client trains locally on their data and sends weight updates — not raw data — to a central server. The server aggregates the updates and broadcasts the improved global model.

```
Server initializes global model θ_0
For each round t = 1, 2, ...:
    Server sends θ_t to subset of clients
    Each client trains locally:
        θ_i^{t+1} = LocalSGD(θ_t, D_i, E epochs)
    Clients send Δθ_i = θ_i^{t+1} - θ_t to server
    Server aggregates:
        θ_{t+1} = θ_t + η Σ_i (n_i/N) Δθ_i  (weighted average)
```

```python
def fedavg_aggregate(global_model, client_updates, client_sizes):
    total = sum(client_sizes)
    with torch.no_grad():
        for key in global_model.state_dict():
            weighted = sum(
                (n/total) * delta[key]
                for delta, n in zip(client_updates, client_sizes)
            )
            global_model.state_dict()[key] += weighted
```

### Challenges

| Challenge | Description | Mitigation |
|-----------|------------|------------|
| **Non-IID data** | Clients have heterogeneous data distributions | FedProx, FedNova, SCAFFOLD |
| **Client drift** | Local updates diverge from global optimum | SCAFFOLD (control variates) |
| **Communication cost** | Sending full gradients per round is expensive | Gradient compression, quantization |
| **Stragglers** | Slow clients delay aggregation | Asynchronous FL, client selection |
| **Client dropout** | Clients disconnect during training | Fault-tolerant aggregation |

### FedProx

Adds a proximal term to each client's local objective to limit how far local updates drift from the global model:

```
F_i(w) = L_i(w) + (μ/2)‖w - w_global‖²
```

### Secure Aggregation

FL protects raw data but still transmits gradients, which can leak information via gradient inversion attacks. Secure aggregation ensures the server only sees the sum of client updates, not individual gradients. Uses additive secret sharing or homomorphic encryption.

### FL + DP

Combining FL and DP: each client runs DP-SGD locally before sending updates. This provides **local DP** (client trusts no one, including the server) or **central DP** (clients trust the server, which adds noise before publication). Local DP is stronger but requires more noise for the same utility.

Real-world deployments: Gboard (mobile keyboard prediction), medical imaging across hospitals, financial fraud detection across banks.

---

## 5. Membership Inference Attacks

An adversary queries the model and infers whether a specific record was in the training set.

**Attack mechanism:** models trained with gradient descent are more confident (lower loss) on training data than held-out data. This gap is detectable. Shokri et al. (2017): train shadow models that mimic the target model's behavior. Use the confidence gap between training and non-training examples as a classifier for membership.

**Defense:**
- DP training quantifiably bounds MIA success — the DP guarantee directly limits how much the presence of any record can affect the output distribution
- Regularization reduces overfitting, shrinking the confidence gap
- Output perturbation adds noise to probabilities
- Confidence score restriction: only return top-k classes

### Model Inversion

Reconstruct approximate training inputs from model outputs. More severe for generative models. Defense: DP training, output perturbation, restricting query access.

---

## 6. Secure Multi-Party Computation (SMPC)

Three hospitals want to jointly train a model on patient data without any hospital seeing another's records. Federated learning sends gradients to a server; SMPC avoids even this.

### Secret Sharing

Split secret `s` into shares `s_1, s_2, ..., s_n` such that any subset of size k can reconstruct `s`, but any subset smaller than k reveals nothing.

**Additive sharing:** `s = s_1 + s_2 + s_3 (mod p)`. Each party holds one share. Addition is free; multiplication requires communication.

### Oblivious Transfer and Garbled Circuits

Cryptographic primitives enabling secure evaluation of arbitrary functions. Practically too slow for neural network training at scale. Used for specific subproblems: private set intersection, secure model inference where a user queries a private model without the server seeing the input.

---

## 7. Homomorphic Encryption (HE)

Compute on encrypted data without decrypting. The server receives encrypted input, computes, returns encrypted output; only the client can decrypt.

```
Enc(x₁) + Enc(x₂) = Enc(x₁ + x₂)
Enc(x₁) × Enc(x₂) = Enc(x₁ × x₂)   (in FHE)
```

**Types:**
- **PHE** (Partial): supports only one operation (+ or ×)
- **SHE** (Somewhat): limited depth of operations
- **FHE** (Fully): arbitrary operations — 100–10,000× slower than plaintext

**Libraries:** SEAL (Microsoft), HElib, TFHE, OpenFHE

Practical use: private inference for small models. Not yet feasible for training large networks.

---

## 8. Privacy-Preserving Techniques Summary

| Technique | What it Protects | Overhead | Maturity |
|-----------|----------------|---------|---------|
| DP-SGD | Training data memorization | Medium (accuracy cost) | Production-ready (Opacus) |
| Federated Learning | Raw data stays local | High (communication) | Production (Google, Apple) |
| SMPC | Input data during computation | Very high | Research/specialized |
| Homomorphic Encryption | Input data during inference | Extremely high | Research/specialized |
| Differential Privacy (output) | Query responses | Low | Standard practice |

---

## 9. What Breaks

**DP-SGD degrades utility, especially on small datasets.** The noise necessary for small ε on a 10,000-sample fine-tuning dataset may make the model useless. The privacy-utility tradeoff is not a nuisance — it is fundamental.

**Federated learning does not prevent gradient inversion.** Raw data stays local, but gradients can be inverted to reconstruct inputs, especially for structured data. FL must be combined with DP or secure aggregation for meaningful privacy.

**Non-IID data degrades FL convergence.** When each hospital has a different patient population, local SGD updates diverge. FedProx and SCAFFOLD help but do not eliminate the gap.

**DP guarantees are per-training-run, not per-deployment.** If the same sensitive data is used to train multiple models or multiple fine-tuning runs, the composition theorem means cumulative ε grows. Unlimited reuse of sensitive data invalidates DP guarantees.

**HE is too slow for neural networks at scale.** 100–10,000× overhead is not a gap hardware improvements will close in the near term. HE for ML remains a research problem except for small models and specific inference-only cases.

**Membership inference is still possible under DP.** DP bounds MIA success probabilistically; it does not make MIA impossible. At ε = 10 (weak privacy), attack success rates may be nearly unchanged from unprotected training.

---

## Key Interview Points

- Carlini et al. demonstrated verbatim training data extraction from GPT-2 — models memorize training data, and that memorization can be queried. This is the foundational motivation for privacy-preserving ML.
- DP and federated learning address two different aspects of the same root problem: DP limits how much any individual's data can influence the model; FL prevents raw data from reaching a central server at all.
- `(ε, δ)`-DP: smaller ε = stronger privacy. ε = 1 is strong, ε = 10 is weak. The guarantee bounds how much the presence of any record changes the output distribution — which directly limits membership inference.
- DP-SGD: clip per-sample gradients to norm ≤ C, add Gaussian noise `N(0, σ²C²I)`, divide by batch size. Clipping bounds individual influence; noise hides presence. This is what makes DP-SGD expensive vs. standard SGD.
- FL keeps raw data local — only weight deltas are shared. Gradients can still leak information via gradient inversion. Combine with DP or secure aggregation for full protection.
- Best practice for LLMs: pre-train on large public data (no DP needed), fine-tune on small sensitive data with DP-SGD. Use LoRA to reduce trainable parameters and improve utility at the same privacy budget.
- HE is theoretically powerful but 100–10,000× slower than plaintext — not feasible for training large networks.
