---
module: LLMs
topic: Applications
subtopic: Model Merging
status: unread
tags: [llms, ml, applications-model-merging]
---
# Model Merging

*(niche — mostly relevant to model-lab/fine-tuning specialist roles; rarely central in a general applied ML/AI-engineer interview, but "what is a task vector / TIES / SLERP" can come up as a trivia check. Know the core idea and one or two methods by name; the full mechanics below are reference depth.)*

---

## The Core Problem

**The problem:** fine-tuning a model for a new task changes its weights, but each separate fine-tune creates a separate model. A system needing math, coding, and safety simultaneously requires either one expensive multi-task fine-tune on all three datasets at once, or three separate models running at inference. Neither is cheap.

**The core insight:** fine-tuned models that share the same pretrained base tend to occupy nearby regions of the loss landscape. Their weight differences from the base are small relative to the base itself. This proximity makes arithmetic on weights meaningful — the interpolated midpoint is also likely to be in a low-loss region.

---

## Simple Weight Averaging

**The problem:** the most naive combination of two models is to average their weights. This only works if the two models are close enough in weight space that their midpoint is also in a low-loss region.

**The core insight:** fine-tuned models sharing the same pretrained initialization tend to lie in the same convex basin of the loss landscape. Their linear interpolation stays roughly inside that basin — a property called linear mode connectivity. Models from different base checkpoints do not share a basin; their midpoint has higher loss than either endpoint.

**The mechanics:**
```
θ_merged = Σ αᵢ θᵢ,   where Σ αᵢ = 1
```
Simplest case: uniform average over n models, αᵢ = 1/n.

**What breaks:** fails when models were fine-tuned from different base checkpoints. Even with the same base, tasks that push weights in opposite directions produce a midpoint worse than either model. The "loss barrier" at the midpoint can be checked empirically by evaluating the interpolated model at t = 0.5 — if performance collapses, the basin assumption does not hold.

---

## SLERP (Spherical Linear Interpolation)

**The problem:** linear interpolation of weight vectors shrinks their magnitude at the midpoint. For two orthogonal unit vectors, the midpoint has magnitude 1/√2 — a 29% reduction. This scale change alters activation distributions throughout the model, degrading performance in ways unrelated to the task content being merged.

**The core insight:** weight vectors have both a direction and a magnitude. The direction encodes what the model has learned; the magnitude encodes the scale of that learning. A meaningful interpolation should travel along the shortest arc connecting the two vectors on the unit hypersphere — preserving magnitude throughout.

**The mechanics:**
```
SLERP(θ₀, θ₁, t) = sin((1-t)Ω)/sin(Ω) · θ₀  +  sin(tΩ)/sin(Ω) · θ₁

where Ω = arccos( θ₀·θ₁ / (||θ₀|| · ||θ₁||) )
```
At t=0: returns θ₀. At t=1: returns θ₁. At any intermediate t: the interpolated vector has the same magnitude as the endpoints. When the two vectors are nearly parallel (Ω ≈ 0), SLERP degenerates gracefully to linear interpolation.

**What breaks:** SLERP is defined for exactly two vectors. Merging three or more models requires sequential SLERP applications, and the result depends on the order. The magnitude-preservation advantage applies per tensor; it does not address task interference when models have learned conflicting representations.

---

## Task Arithmetic

**The problem:** weight averaging treats full model weights as the unit of combination. But two fine-tuned models share most of their weights with the pretrained base — the meaningful difference is small. Averaging full weights blurs the distinction between what was in the base and what each fine-tune contributed.

**The core insight:** the difference between a fine-tuned model and its pretrained base — the task vector — encodes exactly what fine-tuning added. Task vectors from different fine-tunes can be added, subtracted, and scaled independently, because each is centered at zero relative to the base. This is directly analogous to word vectors: just as `king − man + woman ≈ queen`, `θ_base + τ_math + τ_code` yields a model with both capabilities.

**The mechanics:**
```
Task vector:     τ_task = θ_finetuned − θ_pretrained

Add capability:  θ_new = θ_pre + λ · τ_task
Remove:          θ_new = θ_pre − λ · τ_task
Combine:         θ_new = θ_pre + λ₁τ₁ + λ₂τ₂ + ...
```

The scaling factor λ controls the strength of each task vector's contribution. Typical range: 0.3–1.0. Values above 1.0 risk overriding the base model's general capabilities.

**What breaks:** task vectors from very different domains may point in nearly orthogonal directions — their sum introduces noise rather than combining capabilities. When two task vectors have conflicting signs for the same parameter (one pushes a weight positive, another negative), they partially cancel. This sign conflict problem is what TIES addresses.

---

## TIES Merging

**The problem:** when multiple task vectors are added naively, parameters with conflicting signs partially cancel each other out. A parameter that model A learned to increase and model B learned to decrease averages near zero — erasing both contributions. Additionally, not every parameter change from fine-tuning represents learned task knowledge; many are small noise fluctuations that add interference without benefit.

**The core insight:** sign conflicts are the primary source of interference. Resolving them by majority vote — keeping only updates that agree with the dominant direction — prevents cancellation and preserves the actual learned signal.

**The mechanics — three steps (TrIm, Elect sign, diScard):**

Step 1 — TrIm: for each parameter in each task vector, zero out small-magnitude changes.
```
if |τᵢ[p]| < threshold:
    τᵢ[p] = 0     ← discard fine-tuning noise
```

Step 2 — Elect sign: compute the aggregate sign across all task vectors for each parameter.
```
γ[p] = sign( Σᵢ τᵢ[p] )    ← majority vote
```

Step 3 — diScard and merge: discard task vector entries that disagree with the elected sign, then average the remaining.
```
if sign(τᵢ[p]) ≠ γ[p]:
    τᵢ[p] = 0

τ_merged[p] = mean of { τᵢ[p] | τᵢ[p] ≠ 0 }
θ_merged    = θ_pretrained + λ · τ_merged
```

**What breaks:** TIES requires choosing a trim threshold. Too aggressive and task-specific knowledge is discarded along with noise; too lenient and interference remains. The majority-vote sign election can be wrong when a useful minority update is overruled. TIES reduces interference; it does not eliminate it when tasks are genuinely contradictory.

---

## DARE

**The problem:** even after TIES, merged models can carry redundant or conflicting delta parameters that degrade performance. Interference grows with the number of models being merged.

**The core insight:** most delta parameters are either small noise or redundant with other deltas. Randomly dropping a large fraction (60–90%) reduces inter-model interference without substantially hurting task performance, because the remaining deltas carry sufficient task signal. This is dropout applied to delta weights rather than activations.

**The mechanics:**
```
For each parameter p in task vector τ:
    with probability drop_rate:
        τ[p] = 0                          ← randomly zero out
    else:
        τ[p] = τ[p] / (1 − drop_rate)    ← rescale to preserve expectation
```

The rescaling step preserves the expected magnitude of the task vector — the same logic as dropout rescaling during training. DARE is a preprocessing step applied to each task vector before any merge method (averaging, TIES, etc.).

**What breaks:** DARE is stochastic — merges are not reproducible across runs without a fixed seed. At very high drop rates (>90%), useful task signal can be lost along with noise. DARE reduces interference probabilistically; it does not guarantee that dropped parameters were uninformative.

---

## Model Soup

**The problem:** hyperparameter search over fine-tuning produces many model checkpoints. Standard practice keeps only the best one and discards the rest. But these discarded models — each trained from the same pretrained base — occupy nearby points in the same loss basin. Their average is often better than any individual.

**The core insight:** multiple fine-tuning runs from the same base checkpoint differ only in hyperparameters (learning rate, epochs, augmentation). They lie in the same convex basin. Averaging them is cheap and often improves both in-distribution accuracy and out-of-distribution robustness — the average sits closer to the basin center than any individual run.

**The mechanics — two variants:**

Uniform soup: average all models regardless of individual quality.
```
θ_soup = (1/n) Σ θᵢ
```

Greedy soup: add models one at a time, keeping a model only if including it improves validation performance.
```
soup ← best individual model
for each remaining model m (sorted by val accuracy):
    candidate ← average(soup, m)
    if val_accuracy(candidate) > val_accuracy(soup):
        soup ← candidate
```

**What breaks:** all models must share the same pretrained initialization. Models fine-tuned from different checkpoints violate the basin assumption. Greedy soup filters outliers but requires a held-out validation set. The soup's accuracy ceiling is bounded by the diversity of hyperparameter configurations — if all runs converged to nearly the same weights, averaging adds nothing.

---

## Frankenmerging

**The problem:** weight interpolation requires merged models to have learned compatible representations in each layer. Sometimes you want capabilities from models that represent them at different depths — early layers handle syntax, middle layers encode semantic representations, late layers handle task-specific output. Layer-level combination is more flexible than weight-level interpolation.

**The core insight:** assemble a new model by concatenating layers from different models rather than blending weight tensors. The resulting model inherits the input processing of one model and the output processing of another, sidestepping weight interference entirely for layers taken intact from different sources.

**The mechanics:**
```
Base model:   [L0, L1, L2, L3, L4, L5, L6, L7]
Donor model:  [L0, L1, L2, L3, L4, L5, L6, L7]

Frankenmerge: [L0_base, L1_base, L2_donor, L3_donor,
               L4_base, L5_donor, L6_base, L7_base]
```

The embedding and LM head typically stay from one model to maintain vocabulary consistency. Layers can also be duplicated — some 120B+ models have been created from two 70B models by layer duplication without additional training.

**What breaks:** there is no theoretical guarantee that layer interfaces are compatible across models, even with identical architecture. The KV representations produced by layer i in model A may not be appropriate inputs for layer i+1 in model B. Perplexity often spikes at mixed-source layer boundaries. Most combinations fail; results require empirical validation. Models must have identical hidden dimensions and attention head counts.

---

## MergeKit

**The problem:** implementing the above methods from scratch for large sharded models (70B+) requires careful handling of memory constraints, sharding, precision, and tensor alignment.

**The core insight:** model merging is a declarative operation — specify which models, which method, and which hyperparameters, and the framework handles loading, computation, and saving.

**The mechanics:** MergeKit (Arcee AI) accepts a YAML config specifying the merge method and models, then executes the merge with lazy loading for large models.

```yaml
# TIES + DARE merge of three task-specific models
merge_method: ties
base_model: path/to/pretrained-base

models:
  - model: path/to/math-model
    parameters:
      weight: 1.0
      density: 0.5      # DARE: keep 50% of deltas
  - model: path/to/code-model
    parameters:
      weight: 1.0
      density: 0.5
  - model: path/to/instruct-model
    parameters:
      weight: 1.0
      density: 0.7

parameters:
  normalize: true
dtype: bfloat16
```

Supported methods: `linear`, `slerp`, `task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `passthrough` (Frankenmerging).

**What breaks:** cross-architecture merges require `--allow-crimes` and usually produce garbage. Vocabulary mismatches break embedding layers silently — the merge completes but the model outputs nonsense. MergeKit does not validate whether the semantic assumptions of each method are satisfied for the given models.

---

## When Merging Fails

**Different base models:** the most common failure. Models fine-tuned from different base checkpoints (e.g., Llama-3 vs. Mistral) lie in separate loss basins. Averaging them produces incoherent outputs. There is no fix — the prerequisite is the same pretrained base.

**Architecture mismatch:** weight-level merging requires identical layer counts, hidden dimensions, and attention head counts. Different vocabulary sizes break embedding layers. The merge may complete without error but produce a model that cannot generate coherent text.

**Excessive λ (task vector scaling):** scaling task vectors above 1.0 causes the merged model to forget the base model's general capabilities. The task vector overrides rather than augments. Evaluate on a general benchmark (MMLU, HellaSwag) alongside task-specific benchmarks to detect this.

**Too many models merged simultaneously:** each additional model adds interference. Merging 10+ models rarely outperforms merging 3–4 carefully selected ones. Greedy selection helps identify which models improve the aggregate.

**Conditions required for successful merging:**

| Condition | Required? |
|:---|:---|
| Same architecture | Yes |
| Same vocabulary and tokenizer | Yes |
| Same pretrained base checkpoint | Strongly recommended |
| Similar task domains | Helpful |
| Same training precision | Recommended |

*Related: [Tuning and Optimization](10-tuning-optimization.md) | [Inference Optimization](06-inference-optimization.md)*

## Flashcards

**model?** #flashcard
path/to/math-model

**model?** #flashcard
path/to/code-model

**model?** #flashcard
path/to/instruct-model
