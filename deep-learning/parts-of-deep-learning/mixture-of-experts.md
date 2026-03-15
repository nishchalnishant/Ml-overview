# Mixture of Experts (MoE)

Mixture-of-Experts (MoE) models use **sparse activation**: for each input, only a subset of “expert” sub-networks is used. This increases model capacity without proportionally increasing compute per token, improving scalability.

---

## Sparse routing

- **Experts:** Multiple independent sub-networks (e.g. FFNs with same shape). **Router:** Per token (or per block), a small network or linear layer outputs a distribution over experts; typically **top-k** experts are selected (e.g. k=2) and their outputs are combined by the router weights.
- **Routing:** \(y = \sum_{i \in \text{top-k}} w_i \, \text{Expert}_i(x)\), where \(w_i\) from softmax over logits (often only over selected experts for stability). **Load balancing:** Auxiliary loss encourages uniform usage of experts to avoid collapse.

---

## Switch Transformer

- **Switch:** One expert per token (top-1 routing); simple and efficient. **Expert capacity:** Cap number of tokens per expert; overflow tokens can be dropped or passed to next preferred expert. **Scaling:** Scale number of experts (e.g. 64–2048) and expert size; total parameters grow while **active** parameters per forward pass stay manageable.

---

## MoE training

- **Challenges:** Load balancing (some experts underused); training instability; communication cost in distributed (different experts on different devices). **Solutions:** Load-balancing loss (e.g. auxiliary loss on router entropy); careful initialization; expert parallelism (assign experts to devices, all-gather for routing).
- **Efficiency:** FLOPs and latency are dominated by active experts; total parameter count can be 10× dense model with similar compute per step. Used in Mixtral, Google MoE models, and large-scale research.

---

## Quick revision

- **MoE:** Sparse activation; router selects top-k experts per token; combine outputs. **Switch:** Top-1; expert capacity. **Benefit:** Large capacity, similar compute per token. **Training:** Load balancing, expert parallelism.
