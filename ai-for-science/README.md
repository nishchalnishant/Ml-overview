# AI for Science

Applications of AI and ML to scientific discovery: **protein folding**, **drug discovery**, and **materials science**. Post-2020, deep learning and foundation models have become central tools.

---

## Protein folding

- **AlphaFold (DeepMind):** Predicts 3D protein structure from amino acid sequence using deep learning (attention, MSA, structure module). **AlphaFold 2:** Major accuracy improvement; widely used in structural biology. **Impact:** Huge database of predicted structures; enables hypothesis generation and drug target identification.
- **Key ideas:** Evolutionary information (MSA); geometric and physical constraints; iterative refinement. **Limitations:** Dynamics, complexes, and some edge cases still challenging.

---

## Drug discovery

- **Molecular representation:** Graphs (GNN), SMILES/sequences (transformers), or 3D; pretrain on large molecular corpora. **Tasks:** Property prediction (e.g. binding affinity, toxicity); **de novo** molecule generation (generative models, diffusion); retrosynthesis (predict reaction pathways).
- **Examples:** GNNs for property prediction; diffusion or autoregressive models for molecule design; LLMs for protein design and reaction prediction. **Virtual screening:** Use ML to rank compounds for experimental validation.

---

## Materials science

- **Property prediction:** Predict properties (e.g. band gap, stability) from composition or structure; GNNs on crystal graphs, transformers on composition. **Discovery:** Generative models for new materials; optimization in latent space or with reinforcement learning. **Data:** Materials databases (e.g. Materials Project); active learning to suggest experiments.

---

## Quick revision

- **Protein folding:** AlphaFold; structure from sequence; MSA and structure module. **Drug discovery:** Molecular representations (graph, sequence); property prediction, generation, retrosynthesis. **Materials:** Property prediction, generative discovery, active learning.
