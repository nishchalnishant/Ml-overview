# Privacy-Preserving Machine Learning

Privacy-preserving ML allows models to be trained and used while protecting sensitive data. Critical for healthcare, finance, and any system handling personal information.

---

## Why Privacy Matters in ML

**Data leakage risks:**
- Models can memorize training data and reproduce it
- Membership inference: can determine if a record was in the training set
- Model inversion: can reconstruct training samples from the model

**Regulations:** GDPR (EU), CCPA (California), HIPAA (US healthcare), India PDPB — all restrict how personal data is used for ML.

---

## Differential Privacy (DP)

**Core guarantee:** Adding or removing any single individual's data changes the output distribution by at most a factor of `e^ε` (with probability at least `1 - δ`).

**Formal definition:** Algorithm `M` is `(ε, δ)`-DP if for all datasets D, D' differing by one record, and all outputs S:
`P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ`

- `ε` (epsilon): privacy budget — smaller = stronger privacy = more noise
- `δ` (delta): failure probability — typically `1/n²` or smaller
- `ε = 1` is strong privacy; `ε > 10` is weak (little privacy guarantee)

### Mechanisms

**Gaussian Mechanism:** Add `N(0, σ²Δf²)` noise where `Δf` is the L2 sensitivity.

**Laplace Mechanism:** Add `Lap(Δf/ε)` noise; used for pure DP (δ=0), L1 sensitivity.

**Randomized Response:** For boolean queries — answer truthfully with probability p, randomly with probability (1-p). Classic mechanism for survey data.

### DP-SGD (Differentiable Privacy for Deep Learning)

Apply DP during gradient descent by clipping and noising per-sample gradients.

```python
# Using Opacus (PyTorch DP library)
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
    max_grad_norm=1.0,   # per-sample gradient clipping
)

# Training loop is unchanged
for batch in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"ε = {epsilon:.2f}")
```

**DP-SGD Steps:**
1. Compute per-sample gradients (not mini-batch average)
2. Clip each per-sample gradient to L2 norm ≤ C
3. Add Gaussian noise: `g̃ = (1/B)(Σ clip(g_i, C) + N(0, σ²C²I))`
4. Update parameters with noisy gradient

**Privacy accounting:** Use the Moments Accountant or PRV (Privacy Random Variables) to track cumulative `(ε, δ)` over all steps.

**Privacy-utility tradeoff:** More noise = lower accuracy. This gap shrinks with:
- Larger datasets (more data per ε budget)
- Better pre-training (start from a good initialization)
- Higher `ε` (weaker privacy)

### DP for LLMs

**DP fine-tuning:** Fine-tune a pre-trained LLM with DP-SGD. Pre-training's large-dataset advantage helps — the fine-tuning dataset is usually small and sensitive.

```python
# DP fine-tuning with LoRA + Opacus
from peft import get_peft_model, LoraConfig

lora_model = get_peft_model(base_model, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
# Apply Opacus to lora_model — only LoRA params are trained and noised
```

**Challenge:** Per-sample gradient computation is expensive. Use DP-FLOP-efficient methods like Ghost Clipping.

---

## Federated Learning (FL)

Train a model across decentralized devices (clients) without sharing raw data. Only model updates (gradients or weights) are shared with the server.

### FedAvg (McMahan et al., 2017)

```
Server initializes global model θ_0
For each round t = 1, 2, ...:
    Server sends θ_t to subset of clients
    Each client trains locally on their data:
        θ_i^{t+1} = LocalSGD(θ_t, D_i, E epochs)
    Clients send Δθ_i = θ_i^{t+1} - θ_t to server
    Server aggregates:
        θ_{t+1} = θ_t + η Σ_i (n_i/N) Δθ_i  (weighted average)
```

```python
# Simplified FedAvg server aggregation
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
| **Communication cost** | Sending gradients per round is expensive | Gradient compression, quantization |
| **Stragglers** | Slow clients delay aggregation | Asynchronous FL, client selection |
| **Client dropout** | Clients disconnect during training | Fault-tolerant aggregation |

### FedProx

Adds a proximal term to client objective to limit how far clients drift from global model:
`F_i(w) = L_i(w) + (μ/2)‖w - w_global‖²`

### Secure Aggregation

Clients encrypt updates such that the server only sees the sum, not individual gradients. Uses additive secret sharing or homomorphic encryption.

### FL + DP

Combine DP-SGD on each client with FL. Provides **local DP** (client trusts no one) or **central DP** (clients trust the server, which adds noise before publication).

**Use cases:** Mobile keyboard prediction (Gboard), medical imaging across hospitals, financial fraud detection across banks.

---

## Secure Multi-Party Computation (SMPC)

Multiple parties jointly compute a function without revealing their inputs to each other.

**Example:** Three hospitals jointly train a model on patient data without any hospital seeing another's data.

### Secret Sharing

Split secret `s` into shares `s_1, s_2, ..., s_n` such that:
- Any subset of size k can reconstruct `s`
- Any subset of size < k reveals nothing

**Additive sharing:** `s = s_1 + s_2 + s_3 (mod p)`. Each party holds one share. Additions are free; multiplications require communication.

### Oblivious Transfer (OT) and Garbled Circuits

Cryptographic primitives enabling secure evaluation of arbitrary functions. Practically too slow for neural network training on large models.

### Use cases

- Private set intersection (ad conversion measurement without sharing user IDs)
- Secure model inference (user queries private model without server seeing input)

---

## Homomorphic Encryption (HE)

Perform computations directly on encrypted data. Server receives encrypted input, computes on it, returns encrypted output; only the client can decrypt.

`Enc(x₁) + Enc(x₂) = Enc(x₁ + x₂)`  
`Enc(x₁) × Enc(x₂) = Enc(x₁ × x₂)` (in FHE)

**Types:**
- **PHE** (Partial): Only supports one operation (+ or ×)
- **SHE** (Somewhat): Limited depth of operations
- **FHE** (Fully): Arbitrary operations — 100–10,000× slower than plaintext

**Libraries:** SEAL (Microsoft), HElib, TFHE, OpenFHE

**Practical use:** Private inference for small models; not yet feasible for training large networks.

---

## Membership Inference Attacks

An adversary queries the model and infers whether a specific record was in the training set.

**Attack:** Train shadow models that mimic the target model. Observe that models tend to be more confident (lower loss) on training data.

**Defense:**
- DP training (quantifiably bounds MIA success)
- Regularization (reduces overfitting = less confidence gap)
- Output perturbation (add noise to probabilities)
- Confidence score restriction (only return top-k classes)

### Model Inversion

Reconstruct approximate training inputs from model outputs. More severe for generative models.

**Defense:** DP training, output perturbation, restricting query access.

---

## Privacy-Preserving Techniques Summary

| Technique | What it Protects | Overhead | Maturity |
|-----------|----------------|---------|---------|
| DP-SGD | Training data privacy | Medium (accuracy cost) | Production-ready (Opacus) |
| Federated Learning | Raw data stays local | High (communication) | Production (Google, Apple) |
| SMPC | Input data during computation | Very high | Research/specialized |
| Homomorphic Encryption | Input data during inference | Extremely high | Research/specialized |
| Differential Privacy (output) | Query responses | Low | Standard practice |

---

## Key Interview Points

- `(ε, δ)`-DP: smaller ε = stronger privacy. ε=1 is strong, ε=10 is weak.
- DP-SGD clips per-sample gradients and adds Gaussian noise. The clipping step is what makes DP-SGD expensive vs standard SGD.
- Federated Learning keeps raw data local — gradients can still leak information (gradient inversion attacks). Combine with DP or SMPC for full protection.
- Membership inference attacks are the most practical threat to ML models — DP training is the principled defense.
- HE is theoretically powerful but too slow for practical neural network training at scale.
- DP pre-training then fine-tuning: the pre-training dataset is large/public, fine-tuning dataset is small/sensitive — apply DP only during fine-tuning for best utility.
