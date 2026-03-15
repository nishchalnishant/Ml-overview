# Interpretability

Interpretability research aims to understand **how** models compute their outputs. **Mechanistic interpretability** focuses on reverse-engineering circuits and representations inside networks.

---

## Mechanistic interpretability

- **Goal:** Identify subgraphs or "circuits" that implement specific behaviors (e.g. induction heads, factual recall). **Methods:** Ablation (remove components, observe effect); activation patching (replace activations with values from another run); causal tracing.
- **Interpretability as reverse engineering:** Treat the network as a program; find modules and data flow. **Challenges:** Scale (billions of parameters); superposition (multiple features in same direction); polysemantic neurons.

---

## Neuron analysis

- **Single-neuron:** Inspect what inputs maximally activate a neuron; visualize or describe. **Limitation:** Neurons can be polysemantic (respond to many unrelated features). **Probes:** Train linear (or small) classifiers on activations to predict concepts; measure "concept direction" in representation space.
- **Sparse autoencoders:** Decompose activations into sparse linear combinations of "features" (SAE dictionary); features often more interpretable than individual neurons. **Learned features:** Can correspond to concepts (e.g. "capital city", "code") and be used for editing or safety.

---

## Sparse autoencoders (SAE)

- **Setup:** Encoder maps activation to sparse code; decoder reconstructs activation from code. **Training:** Reconstruction loss + L1 on code; encourage sparse, interpretable features. **Use:** Discover interpretable directions; analyze model internals; potential for controllability (steering) and safety (detecting harmful features).
- **Scaling:** SAEs for LLMs (large layers, many features); ongoing work on quality and scalability.

---

## Quick revision

- **Mechanistic interpretability:** Reverse-engineer circuits; ablation, activation patching. **Neuron analysis:** Max-activation inputs, probes, concept directions. **SAE:** Sparse decomposition of activations into interpretable features; scaling to LLMs.
