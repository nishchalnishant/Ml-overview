# Model Merging

Combine the weights of multiple fine-tuned models into a single model — no retraining, no data required.

---

## Table of Contents

1. [What is Model Merging](#1-what-is-model-merging)
2. [Simple Weight Averaging](#2-simple-weight-averaging)
3. [SLERP](#3-slerp-spherical-linear-interpolation)
4. [Task Arithmetic](#4-task-arithmetic)
5. [TIES Merging](#5-ties-merging)
6. [DARE](#6-dare)
7. [Model Soup](#7-model-soup)
8. [Frankenmerging](#8-frankenmerging)
9. [MergeKit](#9-mergekit)
10. [When Merging Fails](#10-when-merging-fails)
11. [Key Interview Points](#11-key-interview-points)

---

## 1. What is Model Merging

Model merging combines the weight tensors of two or more fine-tuned models into a single model that inherits capabilities from all sources — without any additional training or labeled data.

**Core idea:** fine-tuning from a shared pretrained base moves weights into nearby regions of parameter space. Those regions can often be interpolated or combined without catastrophic loss of either capability.

**Why it matters:**
- Zero training cost — no GPU time, no datasets
- Additive capabilities — a coding model + a reasoning model can yield a model with both skills
- Democratizes specialization — merge community fine-tunes rather than running inference on many models
- Avoids catastrophic forgetting that re-fine-tuning on a mix would cause

**Primary applications:**

| Use case | Example |
|---|---|
| Combine task-specific models | code + math + instruction-following |
| Ensemble without runtime cost | average multiple checkpoints |
| Add/remove capabilities | subtract a "toxicity" task vector |
| Creative model blending | Frankenmerge layers from different families |

**Prerequisite:** models must share the same architecture and, for most methods, the same base model.

---

## 2. Simple Weight Averaging

The simplest merge: take the arithmetic mean of each parameter tensor across all models.

```
θ_merged = (1/n) Σ θ_i
```

Or with explicit weights (weighted average):

```
θ_merged = Σ αᵢ θᵢ,   where Σ αᵢ = 1
```

**Why it works at all:** fine-tuned models that share the same pretrained initialization tend to lie in the same convex basin of the loss landscape. Their linear interpolation stays roughly inside that basin.

**Loss barrier problem:** between two independently fine-tuned models (or two models with different bases), the loss landscape typically has a barrier — the midpoint has higher loss than either endpoint. This is called a **linear mode connectivity failure**.

```
Loss
 |          *  ← barrier at midpoint
 |        /   \
 |      /       \
 |    *           *
 |  model A     model B
 +-------------------→ interpolation α
```

Simple averaging works best when:
- Models share the same base and fine-tuning procedure
- Tasks are not too dissimilar
- The number of models being averaged is small

---

## 3. SLERP (Spherical Linear Interpolation)

Rather than interpolating along a straight line in weight space, SLERP interpolates along the **geodesic on the unit hypersphere** — the shortest arc connecting two points on the sphere's surface.

**Why this matters:** weight vectors have a magnitude (norm) and a direction. Linear interpolation shrinks the norm at the midpoint (by up to 1/√2 for orthogonal vectors). SLERP keeps the magnitude constant throughout the interpolation, which better preserves the scale of activations.

**Formula:**

```
SLERP(θ₀, θ₁, t) = sin((1-t)Ω)/sin(Ω) · θ₀  +  sin(tΩ)/sin(Ω) · θ₁

where Ω = arccos( θ₀·θ₁ / (||θ₀|| · ||θ₁||) )
```

- `t = 0` returns `θ₀`, `t = 1` returns `θ₁`
- `Ω` is the angle between the two weight vectors

When `Ω ≈ 0` (vectors nearly parallel), SLERP degenerates gracefully to linear interpolation.

**Implementation:**

```python
import numpy as np

def slerp(v0: np.ndarray, v1: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
    """
    Spherical linear interpolation between two weight vectors.

    Args:
        v0: weight vector from model A (flattened)
        v1: weight vector from model B (flattened)
        t:  interpolation factor in [0, 1]; 0 => v0, 1 => v1
        eps: threshold below which we fall back to linear interpolation

    Returns:
        Interpolated weight vector of same shape as v0 / v1
    """
    # Normalize to unit sphere
    v0_norm = v0 / (np.linalg.norm(v0) + eps)
    v1_norm = v1 / (np.linalg.norm(v1) + eps)

    # Angle between the two vectors
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if abs(omega) < eps:
        # Nearly parallel — linear interpolation is numerically safe
        return (1.0 - t) * v0 + t * v1

    sin_omega = np.sin(omega)
    coeff0 = np.sin((1.0 - t) * omega) / sin_omega
    coeff1 = np.sin(t * omega) / sin_omega

    return coeff0 * v0 + coeff1 * v1


def slerp_model_weights(
    weights_a: dict,
    weights_b: dict,
    t: float = 0.5,
) -> dict:
    """
    Apply SLERP layer-by-layer across two state dicts.

    Args:
        weights_a: state_dict of model A
        weights_b: state_dict of model B
        t:         blend factor; 0 => model A, 1 => model B

    Returns:
        Merged state_dict
    """
    merged = {}
    for key in weights_a:
        w0 = weights_a[key].numpy().flatten().astype(np.float64)
        w1 = weights_b[key].numpy().flatten().astype(np.float64)
        merged_flat = slerp(w0, w1, t)
        merged[key] = merged_flat.reshape(weights_a[key].shape)
    return merged


# --- Example usage ---
# import torch
# from transformers import AutoModelForCausalLM
#
# model_a = AutoModelForCausalLM.from_pretrained("model-A")
# model_b = AutoModelForCausalLM.from_pretrained("model-B")
#
# merged_weights = slerp_model_weights(
#     {k: v.cpu() for k, v in model_a.state_dict().items()},
#     {k: v.cpu() for k, v in model_b.state_dict().items()},
#     t=0.5,
# )
```

SLERP is most useful for merging **two** models. For more than two models it must be applied sequentially, and the order matters.

---

## 4. Task Arithmetic

**Paper:** Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023)

The central abstraction is the **task vector**: the difference between a fine-tuned model's weights and the pretrained base weights.

```
τ_task = θ_finetuned − θ_pretrained
```

Task vectors can be **added, subtracted, and scaled** just like word vectors in word2vec — hence the name.

**Operations:**

| Operation | Formula | Effect |
|---|---|---|
| Add capability | θ_new = θ_pre + λ · τ_task | inject a skill |
| Remove capability | θ_new = θ_pre − λ · τ_task | suppress a skill |
| Combine capabilities | θ_new = θ_pre + λ₁τ₁ + λ₂τ₂ | multi-task model |
| Transfer analogy | τ_A→B analogous to τ_C→? | cross-domain transfer |

**Analogy to word2vec:**
- `king − man + woman ≈ queen`
- `θ_math_ft − θ_base + θ_code_base ≈ θ_code_with_math`

**Python implementation (numpy):**

```python
import numpy as np
from typing import Dict

# Type alias: parameter name → numpy array
StateDict = Dict[str, np.ndarray]


def compute_task_vector(
    finetuned: StateDict,
    pretrained: StateDict,
) -> StateDict:
    """Compute τ = θ_finetuned − θ_pretrained for every parameter."""
    return {k: finetuned[k] - pretrained[k] for k in pretrained}


def apply_task_vectors(
    pretrained: StateDict,
    task_vectors: list[StateDict],
    scales: list[float] | None = None,
) -> StateDict:
    """
    Compose multiple task vectors onto a pretrained base.

    Args:
        pretrained:   base model weights
        task_vectors: list of task vectors (each a StateDict of deltas)
        scales:       per-vector scaling λ; defaults to 1.0 for each

    Returns:
        Merged StateDict: θ_pre + Σ λᵢ τᵢ
    """
    if scales is None:
        scales = [1.0] * len(task_vectors)

    merged = {k: v.copy() for k, v in pretrained.items()}
    for tau, lam in zip(task_vectors, scales):
        for k in merged:
            merged[k] = merged[k] + lam * tau[k]
    return merged


def negate_task_vector(tau: StateDict) -> StateDict:
    """Negate a task vector to remove a capability."""
    return {k: -v for k, v in tau.items()}


# --- Demo ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    D = 1000  # fake parameter size

    pretrained   = {"W": rng.standard_normal(D)}
    math_model   = {"W": pretrained["W"] + 0.3 * rng.standard_normal(D)}
    code_model   = {"W": pretrained["W"] + 0.3 * rng.standard_normal(D)}
    safety_model = {"W": pretrained["W"] + 0.3 * rng.standard_normal(D)}

    tau_math   = compute_task_vector(math_model,   pretrained)
    tau_code   = compute_task_vector(code_model,   pretrained)
    tau_safety = compute_task_vector(safety_model, pretrained)

    # Combine math + code, suppress safety (just as a demonstration)
    merged = apply_task_vectors(
        pretrained,
        task_vectors=[tau_math, tau_code, negate_task_vector(tau_safety)],
        scales=[0.8, 0.8, 0.5],
    )
    print("Merged W norm:", np.linalg.norm(merged["W"]))
```

**Scaling factor λ** is a hyperparameter: too large causes forgetting of the base; too small yields no benefit. Typical range: 0.3 – 1.0.

---

## 5. TIES Merging

**Paper:** Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (NeurIPS 2023)

Simple task arithmetic ignores **interference** — when different task vectors have conflicting signs for the same parameter, they partially cancel. TIES adds three explicit resolution steps.

**Three steps (TrIm, Elect sign, diScard & merge):**

```
Step 1 — TrIm (prune small deltas)
   For each parameter p in each task vector τᵢ:
       if |τᵢ[p]| < threshold:
           τᵢ[p] = 0          ← discard low-magnitude changes (they are noise)

Step 2 — Elect sign (resolve sign conflicts by majority vote)
   For each parameter p:
       γ[p] = sign( Σᵢ τᵢ[p] )    ← aggregate sign across all models

Step 3 — diScard & merge (keep only agreeing task vectors)
   For each model i and parameter p:
       if sign(τᵢ[p]) ≠ γ[p]:
           τᵢ[p] = 0          ← discard parameters that lost the vote

   τ_merged[p] = mean of { τᵢ[p] | τᵢ[p] ≠ 0 }
   θ_merged    = θ_pretrained + λ · τ_merged
```

**Intuition:**
- **Trim** removes fine-tuning noise — parameters that barely changed probably encode nothing task-specific.
- **Elect** gives each parameter a consensus direction — prevents positive and negative updates from annihilating each other.
- **Discard** enforces the consensus — only models that agree on the direction contribute to the final value.

**Why TIES outperforms simple averaging:** averaging sign-conflicting updates pushes parameters toward zero, erasing both capabilities. TIES preserves the dominant direction instead.

---

## 6. DARE

**Paper:** Yu et al., "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (2023)

DARE (**D**rop **A**nd **RE**scale) is a preprocessing step applied to each task vector before any merge method (averaging, TIES, etc.).

**Algorithm:**

```
For each parameter p in task vector τ:
    with probability drop_rate:
        τ[p] = 0                    ← randomly zero out the delta
    else:
        τ[p] = τ[p] / (1 − drop_rate)   ← rescale to preserve expectation
```

This is identical in form to **dropout** applied to the delta weights rather than activations.

**Why it helps:**
- Most delta parameters are small and carry redundant or conflicting information
- Randomly dropping them reduces interference between models
- Rescaling preserves the expected magnitude of the task vector
- Empirically effective even at high drop rates (60 – 90 % of deltas dropped)

**Combined pipeline (DARE + TIES):**

```
pretrained + [model_1, model_2, ..., model_n]
    → compute task vectors τᵢ = θᵢ − θ_pre
    → apply DARE to each τᵢ   (drop + rescale)
    → apply TIES              (trim, elect sign, discard)
    → θ_merged = θ_pre + λ · τ_merged
```

```python
import numpy as np

def dare(tau: np.ndarray, drop_rate: float = 0.9, rng=None) -> np.ndarray:
    """
    Apply DARE to a task vector.

    Args:
        tau:       task vector (any shape)
        drop_rate: fraction of parameters to zero out (e.g. 0.9 = 90%)
        rng:       numpy Generator for reproducibility

    Returns:
        Sparsified and rescaled task vector
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(tau.shape) >= drop_rate   # True where we KEEP
    rescale = 1.0 / (1.0 - drop_rate)
    return tau * mask * rescale
```

---

## 7. Model Soup

**Paper:** Wortsman et al., "Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy and Robustness" (ICML 2022)

Model Soup exploits the fact that multiple hyperparameter configurations fine-tuned from the **same pretrained base** often lie in the same loss basin — their average is often better than any individual model.

**Two variants:**

**Uniform soup** — simple average of all models:
```
θ_soup = (1/n) Σ θᵢ
```

**Greedy soup** — add models one at a time, keeping a model only if it improves performance on a held-out validation set:

```
Algorithm GreedySoup(models sorted by val accuracy, val_set):
    soup ← weights of best individual model
    for each remaining model m (in order of val accuracy):
        candidate ← average(soup, m)
        if val_accuracy(candidate) > val_accuracy(soup):
            soup ← candidate
    return soup
```

**Key insight:** standard hyperparameter search discards all but the best configuration. Greedy soup recycles them for free — the models are already trained.

**When it works:**
- All models must share the same pretrained initialization (same checkpoint)
- Fine-tuning hyperparameters (LR, epochs, augmentation) differ, but the task is the same
- Works best with large pretrained models (CLIP, ViT-G) where the loss basin is wide and flat

**Why greedy outperforms uniform:** models that happen to lie at the edge of or outside the basin degrade the average. Greedy filtering excludes them.

---

## 8. Frankenmerging

Frankenmerging (also called **layer stacking** or **passthrough merging**) assembles a new model by **concatenating layers** from different models rather than interpolating weights within each layer.

**How it works:**

```
Base model:   [L0, L1, L2, L3, L4, L5, L6, L7]  (8 layers)
Donor model:  [L0, L1, L2, L3, L4, L5, L6, L7]  (8 layers)

Frankenmerge: [L0_base, L1_base, L2_donor, L3_donor,
               L4_base, L5_donor, L6_base, L7_base]
```

The merged model has the **same number of layers** (or more, if layers are duplicated). Only the embedding and LM head typically stay from one model to maintain vocabulary consistency.

**Why early and late layers vs. middle layers:**
- Early layers handle tokenization, syntax, low-level features — tend to be base-model-specific
- Middle layers encode task representations — good candidates for mixing
- Late layers (LM head, unembedding) handle output distribution — usually kept from one model

**Practical use:** the Mistral + Llama-derived "Goliath" and "Solar" models used layer duplication and cross-model layer insertion to create 120B+ parameter models from two 70B models without training.

**Limitations:**
- No theoretical guarantee — results are empirical and often require many attempts
- Models must have identical hidden dimensions (same architecture family)
- Perplexity can spike unless layers are carefully ordered

**In MergeKit** this is the `passthrough` merge type — see section 9.

---

## 9. MergeKit

[MergeKit](https://github.com/arcee-ai/mergekit) is an open-source Python library (Arcee AI) that implements all major merge methods via a declarative YAML configuration. It handles sharded models, bfloat16 precision, and lazy loading for large models that don't fit in RAM.

**Installation:**
```bash
pip install mergekit
```

**SLERP merge config:**

```yaml
# slerp_merge.yaml
# Blend model-A and model-B using SLERP at t=0.5
merge_method: slerp
base_model: path/to/model-A

models:
  - model: path/to/model-A
  - model: path/to/model-B

parameters:
  t: 0.5          # interpolation factor; 0 => model-A, 1 => model-B

dtype: bfloat16
```

**TIES merge config:**

```yaml
# ties_merge.yaml
# Merge three task-specific models onto a shared base using TIES + DARE
merge_method: ties
base_model: path/to/pretrained-base

models:
  - model: path/to/math-model
    parameters:
      weight: 1.0
      density: 0.5      # DARE drop rate complement (keep 50% of deltas)

  - model: path/to/code-model
    parameters:
      weight: 1.0
      density: 0.5

  - model: path/to/instruct-model
    parameters:
      weight: 1.0
      density: 0.7      # keep 70% of instruct deltas

parameters:
  normalize: true       # normalize task vectors before merging
  int8_mask: true       # use int8 for sign election (memory efficient)

dtype: bfloat16
```

**Frankenmerge (passthrough) config:**

```yaml
# franken_merge.yaml
# Stack layers: first 16 from base, next 16 from donor
merge_method: passthrough

slices:
  - sources:
      - model: path/to/base-model
        layer_range: [0, 16]
  - sources:
      - model: path/to/donor-model
        layer_range: [16, 32]

dtype: bfloat16
```

**Running a merge:**
```bash
mergekit-yaml ties_merge.yaml ./output-model \
    --cuda \
    --lazy-unpickle \
    --allow-crimes      # required for cross-architecture merges
```

**Supported merge methods in MergeKit:**

| Method | `merge_method` value |
|---|---|
| Linear / weighted average | `linear` |
| SLERP | `slerp` |
| Task Arithmetic | `task_arithmetic` |
| TIES | `ties` |
| DARE-TIES | `dare_ties` |
| DARE-linear | `dare_linear` |
| Passthrough (Frankenmerge) | `passthrough` |

---

## 10. When Merging Fails

Understanding failure modes prevents wasted effort and misattributed results.

**Different base models**

The most common failure. Models fine-tuned from different base checkpoints (e.g., Llama-3 vs. Mistral-v0.3) are initialized from different points in weight space and their fine-tuned weights lie in different basins. Averaging them produces incoherent outputs.

```
Base A ──→ FT-A   \
                    ✗  merging these is undefined / harmful
Base B ──→ FT-B   /
```

**Very different task domains**

Even with the same base, if task vectors point in nearly orthogonal directions (e.g., image captioning adapter vs. code generation), their sum may degrade both. TIES mitigates but does not eliminate this.

**Architecture mismatch**

Models must have identical layer counts and hidden dimensions for weight-level merging. Different vocabulary sizes break embedding layers. Different attention head counts break attention weight shapes.

**Excessive scaling (λ too large)**

Scaling task vectors too aggressively (λ >> 1) causes the merged model to forget the base model's general capabilities — it "overrides" rather than augments.

**Merging too many models at once**

Each additional model adds noise and potential interference. Merging 10+ models rarely outperforms merging 3–4 selected models. Greedy selection (as in Model Soup) helps.

**Summary of conditions required for successful merging:**

| Condition | Required? |
|---|---|
| Same architecture | Yes (always) |
| Same vocabulary / tokenizer | Yes (for weight merging) |
| Same base model checkpoint | Strongly recommended |
| Same training precision | Recommended (bfloat16 throughout) |
| Similar task domains | Helpful but not always required |

---

## 11. Key Interview Points

**Q: What is a task vector and why is it useful?**
A task vector `τ = θ_ft − θ_pre` encodes only what fine-tuning added to the base. Because it is zero-centered around the pretrained weights, multiple task vectors can be added or subtracted independently, enabling compositional capability editing without retraining.

**Q: Why does simple weight averaging work for models from the same base but fail for models from different bases?**
Models fine-tuned from the same checkpoint lie in the same convex basin of the loss landscape; their linear interpolation stays within that basin (linear mode connectivity). Models from different checkpoints lie in separate basins with a loss barrier between them — the midpoint has higher loss than either model.

**Q: What problem does TIES solve that Task Arithmetic does not?**
Task Arithmetic ignores sign conflicts: if model A pushes parameter `p` positive and model B pushes it negative, averaging cancels both updates. TIES resolves conflicts via majority-vote sign election, then discards parameters that disagree, preventing cancellation.

**Q: What is DARE and why does it work?**
DARE randomly zeros a large fraction (60–90%) of delta parameters before merging, then rescales to preserve expected magnitude. The intuition is that most small deltas are noise or redundant; sparsifying them reduces inter-model interference without substantially hurting task performance.

**Q: When would you choose SLERP over linear interpolation?**
When merging exactly two models and you want to preserve the magnitude of weight vectors throughout the interpolation. Linear interpolation shrinks vector norms at the midpoint (worst case: ~29% shrinkage for orthogonal vectors). SLERP travels along the sphere's surface, keeping norms constant, which better preserves the scale of network activations.

**Q: What is Frankenmerging and what risk does it carry?**
Frankenmerging assembles a model by concatenating layers from different models. It can combine capabilities without weight interference, but there is no theoretical guarantee the layer interfaces are compatible. Perplexity often spikes and results require empirical validation.

**Q: What are the minimum requirements for any merge to work?**
Identical model architecture (layer count, hidden size, attention heads) and the same tokenizer/vocabulary. Shared pretrained base is strongly recommended. Fine-tuned tasks should not be so dissimilar that their parameter updates are almost entirely orthogonal.

**Q: How does Model Soup differ from ensembling?**
Ensembling runs multiple models at inference time and combines their output distributions — multiplicative cost. Model Soup averages weights before deployment, producing a single model with no inference overhead. The accuracy gain is smaller than a true ensemble but the cost is zero.

**Q: What hyperparameter controls the strength of task vector application, and how do you choose it?**
The scaling factor `λ` (lambda). It is typically searched in [0.3, 1.0] using a small validation set. `λ = 1.0` fully applies the task vector; values below 1 blend more conservatively with the base. Values above 1.0 are possible but risk overriding the base model's general capabilities.

**Q: Name the merge methods from simplest to most sophisticated.**
Uniform weight average → SLERP → Task Arithmetic → Model Soup (greedy) → TIES → DARE-TIES → Frankenmerging.
