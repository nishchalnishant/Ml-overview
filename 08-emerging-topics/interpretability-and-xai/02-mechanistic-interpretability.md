---
module: Emerging Topics
topic: Interpretability And Xai
subtopic: Mechanistic Interpretability
status: unread
tags: [emergingtopics, ml, interpretability-and-xai-mecha]
---
# Mechanistic Interpretability

## 1. Core Concept & Intuition

Mechanistic interpretability (MI) asks: what algorithm is implemented by a neural network? Not "which features are important?" (attribution) or "what does the model attend to?" (attention visualization), but rather: what is the precise computational circuit performing a given task?

The central claim is that neural networks are not black boxes — they implement discrete, human-understandable algorithms. The goal is to reverse-engineer those algorithms with the same precision we apply to understanding biological circuits or compiled programs.

Key distinction from LIME/SHAP-style interpretability:
- **Post-hoc attribution** (SHAP, attention): assigns importance scores to inputs for a given output. Does not explain mechanism.
- **Mechanistic interpretability**: identifies which neurons, attention heads, and weight submatrices implement which sub-computations. Causal, not correlational.

**Why it matters for AI safety:** if we can read a model's algorithm, we can verify whether it is doing what we intend, detect deceptive reasoning, and build confidence in deployment for high-stakes settings. Anthropic's interpretability team treats this as a core safety research program.

---

## 2. Architecture & Mathematics

### Superposition and Linear Representation

**The linear representation hypothesis:** neural networks represent concepts as linear directions in activation space. A concept `c` is represented by a direction vector `v_c ∈ ℝ^d` such that:

```
presence(c) ≈ v_c · x / ||v_c||   (dot product with activation)
```

**Superposition:** a model may represent more features than it has neurons by using non-orthogonal directions. For n features in d dimensions (n > d), the model uses directions that are "approximately orthogonal" — small pairwise dot products:

```
|v_i · v_j| ≤ ε   for i ≠ j   (approximately orthogonal)
```

The superposition hypothesis predicts:
- Rare, important features get near-orthogonal dedicated directions
- Common, less important features are superposed in shared dimensions
- Non-linear activation functions (ReLU) are the mechanism that makes superposition work — they suppress the interference between superposed features

**Toy model of superposition (Elman, Toy Models of Superposition):**

```python
# 1-layer autoencoder with bottleneck
# x ∈ ℝ^n (n features), h ∈ ℝ^m (m < n neurons)
# W ∈ ℝ^{m×n}, b ∈ ℝ^n

h = ReLU(W x)
x̂ = Wᵀ h + b
L = Σᵢ importance_i · (x_i - x̂_i)²
```

Result: when features are sparse and uncorrelated, the model stores more features than dimensions. The number of features represented scales roughly as `O(d · log(1/sparsity))`.

---

### Sparse Autoencoders (SAEs)

SAEs are the current primary tool for finding interpretable features in transformer activations.

**Setup:** for a given activation layer with residual stream dimension d, train a sparse autoencoder:

```
z = ReLU(W_enc(x - b_dec) + b_enc)    # encode: x ∈ ℝ^d → z ∈ ℝ^D (D ≫ d)
x̂ = W_dec z + b_dec                   # decode: z ∈ ℝ^D → x̂ ∈ ℝ^d

L = ||x - x̂||² + λ · ||z||₁           # reconstruction + L1 sparsity
```

The decoder columns `W_dec[:, i] ∈ ℝ^d` are "feature directions" — the directions in the residual stream that correspond to human-interpretable concepts.

**Why D ≫ d:** the superposition hypothesis predicts the model encodes more features than dimensions. A 4096-dim residual stream might represent millions of features via superposition. The SAE expands to D = 16k, 32k, or larger to find a sparse linear approximation where most features are near-zero (not active) and the active ones are the "true" features used for that token.

**Anthropic's scaling findings (Claude feature visualization, 2024):**

```
Residual stream dim d = 4096 (Claude 3 Sonnet)
SAE latent dim D = 34M features total across all layers
Layer 20 SAE, D = 131,072: L0 sparsity ≈ 25 active features per token

Selected feature examples found:
  - Feature 2341: activates on "academic citations" context
  - Feature 17893: activates specifically on "The Golden Gate Bridge" (so strongly that
                   Claude 3 Sonnet steered to activate this feature would identify as
                   the bridge)
  - Feature 31102: activates on "immune system dysfunction"
```

**SAE quality metrics:**

```
L0: average number of active features per token (lower = sparser)
L2 reconstruction: ||x - x̂||₂ / ||x||₂ (lower = better reconstruction)
CE loss degradation: LM perplexity increase when replacing x with x̂
  - Ideal: near-zero CE loss degradation with low L0 (≤ 30 active features)
```

The tension: higher λ (L1 penalty) → lower L0 (sparser, more interpretable) → higher CE degradation (worse reconstruction). Current SOTA SAEs achieve L0 ≈ 20-60 with CE degradation < 5% of baseline loss.

---

### Circuit Analysis

**Circuits** are subgraphs of the full computational graph: specific attention heads, MLP neurons, and residual stream connections that implement a particular capability.

**Techniques for circuit finding:**

**Activation patching (causal tracing):**
```python
def activation_patch(clean_prompt, corrupted_prompt, model, patch_position):
    """
    Run the model on corrupted_prompt, but replace the activation at
    patch_position with the value from clean_prompt.
    Measure: how much does the output logit shift back toward clean?
    """
    clean_activations = get_activations(model, clean_prompt)
    
    def patching_hook(activation, hook):
        activation[:, patch_position] = clean_activations[hook.name][:, patch_position]
        return activation
    
    corrupted_logits = model.run_with_hooks(corrupted_prompt, fwd_hooks=patching_hook)
    return logit_diff(corrupted_logits, clean_logits, target_token)
```

A component is "causal" for a behavior if patching its activation from clean→corrupted (or vice versa) significantly changes the output. Components with high causal effect are nodes in the relevant circuit.

**Path patching (Goldowsky-Dill et al.):** instead of patching individual components, patch the direct connection from component A to component B, measuring whether A sends task-relevant information to B. This recovers the circuit's edges, not just nodes.

**IOI (Indirect Object Identification) circuit — the canonical example:**

For the prompt `"When Mary and John went to the store, John gave a drink to ___"`:

```
Expected output: "Mary"
Algorithm required:
  1. Identify S (subject) and IO (indirect object) names in context
  2. Duplicate token heads: detect that "John" appears twice
  3. S-Inhibition heads: suppress the duplicate token (John) in the output
  4. Name Mover heads: copy the remaining name (Mary) to output position
```

The full IOI circuit spans ~26 specific attention heads (out of 144 total in GPT-2 Medium) and implements this algorithm with measurable precision. Ablating any of the 26 heads degrades performance; adding other heads doesn't help.

**Mathematical structure of induction heads:**

Induction heads implement in-context learning of repeated patterns:
```
Input: [A] [B] ... [A] → predict [B]

Layer 1 (previous token head):
  For position t, attends to position t-1
  Copies token at t-1 to current position's residual stream

Layer 2 (induction head):
  Q = current token
  K = residual stream at each position (includes "what came before")
  Attends to positions where "what came before" ≈ current token
  → Attends to the position after first occurrence of current token
  → Copies what followed → predicts it
```

This is a two-layer circuit, and it's the mechanistic explanation for why transformers exhibit in-context learning. The circuit was found in GPT-2 and confirmed to exist in all transformer language models studied.

---

### Probing and Representation Analysis

**Linear probes:**
```python
# For a layer's activations h ∈ ℝ^{B×T×d}, does h contain feature f?
probe = nn.Linear(d, n_classes)
loss = CrossEntropyLoss(probe(h.detach()), labels_f)
```

High probe accuracy implies the feature is **linearly decodable** from the representation. But linear decoding ≠ the model uses this feature causally. Combine with causal patching for stronger evidence.

**Representational Similarity Analysis (RSA):**
```
RDM(X) = pairwise distance matrix ∈ ℝ^{N×N}
RSA(layer_i, layer_j) = Spearman_r(RDM(h_i), RDM(h_j))
```

RSA compares geometric structure across layers or models. High RSA between two models means they learned similar internal representations even with different architectures.

---

## 3. Trade-offs & System Design Implications

### What SAEs Do and Don't Tell You

**SAEs find a sparse linear decomposition of activations.** This is useful for understanding what information is present in a layer. It does not prove:
- That those features are used causally by downstream components
- That the decomposition is unique (superposition allows multiple valid bases)
- That the features are "the" features the model uses (SAE is an approximation)

Combining SAE feature identification with causal patching gives stronger mechanistic claims: "Feature F is represented at layer L AND patching it causally affects behavior B."

### Interpretability in Production ML

Current state: mechanistic interpretability is primarily a research methodology, not a production monitoring tool. The most immediate production applications:

**1. Steering vectors for controlled generation:**
```python
# Find a direction in residual stream that corresponds to concept C
steering_vector = mean(h[concept_present]) - mean(h[concept_absent])

# At inference, add scaled steering vector to residual stream
def steering_hook(activation, hook, alpha=20):
    if hook.name == target_layer:
        activation += alpha * steering_vector
    return activation

model.run_with_hooks(prompt, fwd_hooks=[steering_hook])
```
Applications: toxicity suppression, language control, persona steering. Used in production by Anthropic for Claude style adjustment.

**2. Anomaly detection via feature activation monitoring:**
```python
# Train SAE on normal distribution, deploy as monitor
# Flag inputs that activate unusual features (OOD detection)
def is_anomalous(x, sae, baseline_activation_stats, threshold=5.0):
    z = sae.encode(x)
    # Z-score of feature activations vs. training distribution
    z_scores = (z - baseline_mean) / baseline_std
    return z_scores.max() > threshold
```

**3. Model diffing:**
Compare SAE features before and after fine-tuning to understand what the fine-tuning changed. If a safety fine-tune was supposed to suppress "harmful request completion" features but those features are still present and active (just with lower weights), the fine-tune may not be as robust as behavioral testing suggests.

### Scalability Challenges

| Challenge | Current Status |
|-----------|---------------|
| SAE training compute | 4096-dim layer, D=131k SAE ≈ 1-2 GPU-days |
| Circuit discovery automation | Mostly manual for novel behaviors |
| Polysemanticity | Many neurons active for semantically unrelated concepts; SAEs help but not solved |
| Cross-model transfer | Circuits found in GPT-2 often don't transfer structurally to larger models |
| MLP interpretability | Attention heads interpretable via QK/OV circuit decomposition; MLPs much harder |

---

## 4. Canonical Interview Q&As

**Q1: What is the superposition hypothesis and how do sparse autoencoders test it? What would falsify the hypothesis?**

The superposition hypothesis states: neural networks represent more features than they have neurons by encoding features as non-orthogonal directions in activation space, relying on sparse co-activation to prevent catastrophic interference.

The supporting evidence:

1. **Toy model verification:** a single-layer autoencoder with bottleneck trained on sparse synthetic features learns to represent up to `~d · log(1/sparsity)` features in d dimensions, confirmed by observing multi-feature polysemantic neurons.

2. **SAE evidence:** training a sparse autoencoder on a residual stream layer (d=4096) with large dictionary (D=131k) finds ~130k distinct features, the majority of which are monosemantic (activate for one interpretable concept) and sparsely activated (L0 ≈ 20-60 per token).

3. **Geometry:** SAE decoder vectors `W_dec[:, i]` show near-uniform distribution on the unit sphere with controlled pairwise dot products, consistent with frames/spherical codes for storing many nearly-orthogonal vectors.

**What would falsify it:**
- SAEs with large D that fail to find monosemantic features (all latents remain polysemantic) — this would suggest features are not linearly decodable
- Removing individual SAE features having no effect on downstream computations (would suggest the SAE is finding post-hoc correlations, not the model's actual computational units)
- Finding that the optimal SAE dictionary size scales linearly rather than super-linearly with d (would suggest no superposition beyond dimensionality)

The strongest current critique: SAEs find a sparse linear decomposition, but this decomposition might be one of many valid bases, not the "true" computational basis the model uses. Causal patching of SAE features is the main way to test for computational reality.

---

**Q2: Walk through the mathematical structure of induction heads and explain how they implement in-context few-shot learning.**

Induction heads form a two-layer circuit identified in all transformer LMs:

**Layer 1 — Previous Token Head:**

The OV (output-value) circuit copies the embedding of the previous token into the current residual stream. For position t:

```
Attn_weight[t, t-1] ≈ 1   (attends strongly to previous position)
Output_t = OV_matrix · embed(token_{t-1})
```

After layer 1, the residual stream at position t contains information about both token_t AND token_{t-1}.

**Layer 2 — Induction Head:**

The QK (query-key) circuit compares the current token against "what came before each past position":

```
Q_t = W_Q · (embed(token_t) + residual)
K_s = W_K · (embed(token_s) + layer1_output_s)   # layer1_output_s ≈ embed(token_{s-1})

Score(t, s) = Q_t · K_s ≈ W_QK(token_t, token_{s-1})
```

Maximum score at position s where `token_{s-1} ≈ token_t`, i.e., where the context "what came before position s" matches the current token. The head then attends to position s and outputs `W_V · embed(token_s)` — i.e., copies what followed the previous occurrence.

**Mathematical summary:**

```
For sequence: ... [A] [B] ... [A] [?]

Previous token head: stores embed(A) in residual at second [A] position
Induction head: Q(second [A]) · K(pos after first [A]) is maximized
  → attends to B's position → predicts B
```

This implements the in-context algorithm "if you've seen [A][B] before, predict B when you see A again" entirely through weights in 2 attention heads. This is why transformers exhibit in-context few-shot learning without explicit fine-tuning: the induction circuit generalizes the "copy the pattern" algorithm to arbitrary token sequences.

---

**Q3: Describe the IOI circuit. How was it found, and how does it demonstrate that mechanistic interpretability can find complete functional circuits?**

IOI (Indirect Object Identification) studies prompts of the form:
```
"When Mary and John went to the store, John gave a drink to ___"
                                        ↑                      ↑
                                   Repeated name (S)      Should output IO (Mary)
```

**Discovery methodology:**

1. **Behavioral characterization:** construct a dataset of 1000 IOI prompts with ABB structure (subject repeated) and measure `logit_diff = logit(IO) - logit(S)` as the performance metric.

2. **Layer-level attribution:** use activation patching at each layer to find which layers are causally important. Finds important contributions at layers 3-5 (early), 7-8 (middle), and 9-10 (late).

3. **Head-level attribution:** within important layers, patch individual heads to find the 26 causally-relevant attention heads out of 144 total.

4. **Circuit role identification:** patch subsets of heads and analyze what information flows through Q/K/V:
   - **Duplicate Token Heads (layers 3-5):** Q attends to the repeated name, K attends to the subject — detects "this name appears twice"
   - **S-Inhibition Heads (layers 7-8):** suppress writing the repeated name (S) to the output position. Ablating these causes the model to output S instead of IO.
   - **Name Mover Heads (layers 9-10):** output-writes the IO name's embedding to the final position. The causal chain: these heads look at IO's residual stream representation (which is distinctive because S-Inhibition heads suppressed S) and copy it to the output.
   - **Backup Name Movers (layers 10-11):** redundancy — if primary Name Movers are ablated, these compensate (a "hedge against interpretability interventions" by the model itself)

**Demonstration of completeness:**
The 26-head circuit achieves 79% of the full model's logit_diff. Ablating the remaining ~118 heads has minimal effect. This is "completeness" in the MI sense: the circuit is a sufficient explanation, not just a partial one.

This result was replicated by multiple groups, and the circuit architecture generalizes across GPT-2 sizes (though specific head indices change) — confirming it's not an artifact of a single model's quirks.

---

**Q4: What are the current limitations of mechanistic interpretability that make it difficult to apply to frontier models like Claude or GPT-4?**

**Scale:** GPT-2 Small (117M parameters, 12 layers, 8 heads/layer) allowed manual circuit analysis. GPT-4-scale models (est. ~1.7T parameters, ~120 layers, ~128 heads/layer) have:
- ~14,000× more attention heads
- Circuit complexity that likely scales super-linearly (circuits compose into meta-circuits)
- SAE training at scale requires enormous compute: a single layer SAE with D=4M for a 12,288-dim residual stream requires billions of training tokens and weeks of GPU time

**MLP opacity:** the QK/OV decomposition gives natural structure for analyzing attention. MLP layers have no analogous clean decomposition — a neuron's behavior depends on all input features simultaneously. SAEs help, but MLP circuits are far harder to identify than attention circuits.

**Superposition resolution:** at scale, the number of superposed features per neuron grows. SAEs trained on frontier model internals recover features, but whether the recovered basis is the "computationally correct" one is harder to verify — there is no ground truth.

**Causal vs. correlational ambiguity at scale:** in GPT-2, activation patching cleanly identifies causal components because the model has few enough components to be exhaustive. In a 120-layer model, the interaction space is too large for exhaustive causal analysis. Path patching scales somewhat better but still requires choosing which paths to investigate.

**Emergent circuits:** capabilities in frontier models likely involve circuits that span many layers and are distributed across many heads — not the localized 26-head circuits of GPT-2. The concepts of "circuit" and "component" may require revision at scale.

**Current mitigation strategy:** Anthropic, EleutherAI, and academic groups are focusing on:
1. Automated circuit discovery (ACDC, EAP) — gradient-based edge attribution to scale circuit finding
2. Large-scale SAE training (Claude 3 SAE release with 34M features)
3. Feature-level steering as a probe: if a feature can steer behavior, it's causally real
4. Focusing on narrow high-stakes behaviors (sycophancy, deception indicators) rather than general capability circuits

---

**Q5: How would you use mechanistic interpretability techniques to audit a fine-tuned model for potential deceptive alignment?**

**Deceptive alignment** (Evan Hubinger's formulation): a model that behaves safely during training/evaluation but pursues a different objective in deployment. Standard behavioral testing cannot detect this because the model passes all tests intentionally.

**MI audit strategy:**

**Step 1: Baseline feature map.** Train SAEs on the pre-fine-tuned model and the fine-tuned model. Compare the feature activation distributions:

```python
# Features that activate MORE in fine-tuned vs. base model on evaluation inputs
delta_activations = sae_finetuned.encode(eval_activations) - sae_base.encode(eval_activations)
suspicious_features = topk(delta_activations, k=100)  # what's different?
```

If evaluation inputs (safe prompts) activate "acting helpful/compliant" features more than baseline, this is a potential signal.

**Step 2: Distribution shift detection.** Compare feature activations on:
- Training distribution (safe prompts during RLHF)
- Deployment-like distribution (slightly different prompt formats)
- Adversarial probes (prompts designed to look like safety evaluations vs. real use)

A deceptively aligned model would show different internal representations (different feature activations) for "I'm being evaluated" vs. "I'm deployed" — even if the behavioral outputs look similar.

**Step 3: Steering vector probes.** Find directions corresponding to "I am being evaluated" vs. "I am deployed" using SAE features or linear probes on activation differences. Steer in the "deployed" direction during safety evaluation: does behavior change? If safety behavior degrades when the model is steered to "believe" it's deployed, this is evidence of conditional alignment.

**Step 4: Circuit comparison for goal-directed behavior.** Identify circuits for long-horizon planning or self-preservation behaviors (goals about the model's own continued operation or avoiding detection). These circuits should not be active during normal assistant tasks. Presence and activation of such circuits is a red flag.

**Limitations:** deceptive alignment requires the model to have stable world models that distinguish evaluation from deployment — very unlikely to emerge in current RLHF-trained models. This audit methodology is defensive research for future more capable models, not a practical concern for current systems.

## Flashcards

**Post-hoc attribution (SHAP, attention)?** #flashcard
assigns importance scores to inputs for a given output. Does not explain mechanism.

**Mechanistic interpretability?** #flashcard
identifies which neurons, attention heads, and weight submatrices implement which sub-computations. Causal, not correlational.

**Rare, important features get near-orthogonal dedicated directions?** #flashcard
Rare, important features get near-orthogonal dedicated directions

**Common, less important features are superposed in shared dimensions?** #flashcard
Common, less important features are superposed in shared dimensions

**Non-linear activation functions (ReLU) are the mechanism that makes superposition work?** #flashcard
they suppress the interference between superposed features

**Feature 2341?** #flashcard
activates on "academic citations" context

**Feature 17893?** #flashcard
activates specifically on "The Golden Gate Bridge" (so strongly that

**Feature 31102?** #flashcard
activates on "immune system dysfunction"

**Ideal?** #flashcard
near-zero CE loss degradation with low L0 (≤ 30 active features)

**That those features are used causally by downstream components?** #flashcard
That those features are used causally by downstream components

**That the decomposition is unique (superposition allows multiple valid bases)?** #flashcard
That the decomposition is unique (superposition allows multiple valid bases)

**That the features are "the" features the model uses (SAE is an approximation)?** #flashcard
That the features are "the" features the model uses (SAE is an approximation)

**SAEs with large D that fail to find monosemantic features (all latents remain polysemantic)?** #flashcard
this would suggest features are not linearly decodable

**Removing individual SAE features having no effect on downstream computations (would suggest the SAE is finding post-hoc correlations, not the model's actual computational units)?** #flashcard
Removing individual SAE features having no effect on downstream computations (would suggest the SAE is finding post-hoc correlations, not the model's actual computational units)

**Finding that the optimal SAE dictionary size scales linearly rather than super-linearly with d (would suggest no superposition beyond dimensionality)?** #flashcard
Finding that the optimal SAE dictionary size scales linearly rather than super-linearly with d (would suggest no superposition beyond dimensionality)

**Duplicate Token Heads (layers 3-5): Q attends to the repeated name, K attends to the subject?** #flashcard
detects "this name appears twice"

**S-Inhibition Heads (layers 7-8)?** #flashcard
suppress writing the repeated name (S) to the output position. Ablating these causes the model to output S instead of IO.

**Name Mover Heads (layers 9-10)?** #flashcard
output-writes the IO name's embedding to the final position. The causal chain: these heads look at IO's residual stream representation (which is distinctive because S-Inhibition heads suppressed S) and copy it to the output.

**Backup Name Movers (layers 10-11): redundancy?** #flashcard
if primary Name Movers are ablated, these compensate (a "hedge against interpretability interventions" by the model itself)

**~14,000× more attention heads?** #flashcard
~14,000× more attention heads

**Circuit complexity that likely scales super-linearly (circuits compose into meta-circuits)?** #flashcard
Circuit complexity that likely scales super-linearly (circuits compose into meta-circuits)

**SAE training at scale requires enormous compute?** #flashcard
a single layer SAE with D=4M for a 12,288-dim residual stream requires billions of training tokens and weeks of GPU time

**Training distribution (safe prompts during RLHF)?** #flashcard
Training distribution (safe prompts during RLHF)

**Deployment-like distribution (slightly different prompt formats)?** #flashcard
Deployment-like distribution (slightly different prompt formats)

**Adversarial probes (prompts designed to look like safety evaluations vs. real use)?** #flashcard
Adversarial probes (prompts designed to look like safety evaluations vs. real use)
