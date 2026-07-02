# Mixture of Experts

A Mixture-of-Experts (MoE) layer replaces a single dense feed-forward network with several parallel "expert" feed-forward networks, plus a router (gating network) that decides which experts process each token. In a typical top-k routing scheme, the router computes a score for every expert and activates only the top-k highest-scoring experts for each token, rather than all of them.

This gives a key efficiency property: the model can have a very large total parameter count (the sum of all experts' parameters) while the active parameter count per forward pass — and therefore the compute cost per token — stays much smaller, since only a handful of experts are actually invoked. For example, a model with 16 experts and top-2 routing activates only 2 experts per token, so its per-token compute resembles a much smaller dense model even though its total parameter count is far larger.

MoE models introduce new challenges: load balancing (ensuring tokens are spread roughly evenly across experts so no expert is starved or overloaded), routing instability during training, and increased memory requirements since all experts must be held in memory even though only a few are active per token. Auxiliary load-balancing losses are commonly added during training to encourage the router to distribute tokens more evenly across experts.
