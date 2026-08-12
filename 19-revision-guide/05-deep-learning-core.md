---
module: Deep Learning Core
topic: Revision Guide
subtopic: "05 Deep Learning Core"
status: unread
tags: [revision, cheatsheet, backpropagation, activations, initialization, loss-functions, optimizers, normalization, regularization, pytorch, transfer-learning]
---
# 05 Deep Learning Core — Revision Guide

Fast revision for the whole `05-deep-learning-core/` folder: an outline of every major concept, then graded Q&A (no answers — self-test). Use the outline to check you can explain each node aloud before drilling questions.

---

```markdown
- **05 Deep Learning Core**
  - **Backpropagation**
    - Chain rule<br/>credit assignment
    - Matrix form<br/>dL/dW = x^T delta
    - Gradient flow table<br/>per-layer local derivatives
    - Vanishing gradients<br/>sigmoid deriv <= 0.25
      - Residual +1 gradient term
    - Exploding gradients
    - Gradient clipping<br/>norm vs value
    - Autograd<br/>zero_grad/backward/step
    - Gradient checkpointing
  - **Activation Functions**
    - Sigmoid<br/>saturates, not zero-centered
    - Tanh<br/>zero-centered, still saturates
    - ReLU<br/>dying ReLU problem
    - Leaky ReLU / PReLU
    - ELU
    - GELU<br/>exact vs tanh approx
    - Swish/SiLU
    - SwiGLU<br/>gated FFN variant
    - Softmax<br/>numerical stability
      - Output activation by task
  - **Weight Initialization**
    - Zero init<br/>symmetry problem
    - Naive random<br/>variance blow-up/shrink
      - He/Kaiming<br/>ReLU, 2/n_in
        - Orthogonal init<br/>RNNs
    - Xavier/Glorot<br/>tanh/sigmoid, 1/n_in
    - Transformer residual scaling<br/>1/sqrt(2L)
    - GPT-2/LLaMA std=0.02
    - LSTM forget-gate bias=1
    - Bias init reference table
  - **Hidden Layers**
    - Depth vs width tradeoff
    - Universal Approximation Theorem
    - Layer types table
    - Residual connections<br/>H(x) = F(x) + x
    - Normalization layers overview
    - Practical sizing heuristics
    - Bottleneck architectures
  - **Loss Functions**
    - Binary cross-entropy
    - Multiclass cross-entropy
    - Weighted CE / Focal loss<br/>class imbalance
    - Label smoothing
      - Task-to-loss mapping table
    - MSE / MAE / Huber
    - Triplet loss
    - Contrastive / InfoNCE
    - KL divergence
  - **Optimizers**
    - SGD + Momentum + Nesterov
    - Adam<br/>1st/2nd moment estimates
    - AdamW<br/>decoupled weight decay
      - LR scheduling
        - Step / polynomial decay
    - RMSprop
    - Lion
    - Warmup
    - Cosine annealing
    - OneCycleLR
    - ReduceLROnPlateau
    - Gradient accumulation
    - Mixed precision / AMP<br/>fp16 vs bf16, GradScaler
      - Optimizer comparison table
  - **Normalization**
    - Internal covariate shift<br/>contested (Santurkar 2018)
    - BatchNorm<br/>per-channel, batch stats
      - BN as implicit regularizer
    - LayerNorm<br/>per-sample, batch-size invariant
      - Pre-Norm vs Post-Norm
    - RMSNorm<br/>no mean-centering
    - InstanceNorm / AdaIN
    - GroupNorm
    - WeightNorm
    - SpectralNorm
  - **Regularization**
    - L2 / weight decay<br/>w <- w(1-eta*lambda) - eta*grad
    - L1<br/>sparsity via sign(w)
    - Dropout<br/>inverted, 2^n ensemble view
    - Dropout + BatchNorm<br/>variance mismatch
    - Early stopping<br/>patience, val-set bias
    - Data augmentation
    - MixUp<br/>lambda ~ Beta(a,a)
    - CutMix
    - DropPath / Stochastic Depth
    - DropBlock<br/>contiguous spatial mask
    - Label smoothing<br/>KL-to-uniform
    - Regularization summary table
  - **PyTorch Fundamentals**
    - Tensors<br/>dtype, device, shape
    - view vs reshape<br/>contiguity
    - unsqueeze/squeeze/permute
    - NumPy interop<br/>shared memory
    - Autograd<br/>requires_grad, comp graph
    - Grad accumulation gotcha<br/>3x backward != single
    - detach vs no_grad vs inference_mode
    - Gradient hooks
    - nn.Module internals<br/>__setattr__ auto-registration
    - ModuleList vs plain list
    - parameters() vs state_dict()<br/>buffers = BN running stats
    - train()/eval() mode effects
    - Dataset/DataLoader<br/>num_workers, pin_memory
    - 6-step training loop
      - Custom loss functions<br/>focal, contrastive
    - Model saving/loading<br/>state_dict vs full model
    - Debugging<br/>NaN detection, shape hooks
  - **Transfer Learning**
    - Taxonomy<br/>fine-tune/domain-adapt/few-shot/zero-shot/meta
    - Discriminative fine-tuning<br/>per-layer LR
    - Gradual unfreezing<br/>catastrophic forgetting
    - Domain adaptation<br/>covariate/label shift/concept drift
    - DANN<br/>gradient reversal layer
    - CORAL<br/>covariance alignment
    - Instance reweighting
    - Test-Time Adaptation
    - Prototypical Networks
    - Matching Networks
    - MAML / FOMAML / Reptile
    - Relation Networks
    - Zero-shot<br/>attribute/CLIP-based
    - Self-supervised pretraining<br/>SimCLR/DINO/MAE/BERT/GPT
```

---

## Questions

### Easy

1. Write out the chain-rule expression for `dL/dw` for a single weight feeding into a pre-activation `z`, then activation `a`, then loss `L`.
2. Why does sigmoid saturate, and what is the maximum value its derivative can take?
3. Give one reason ReLU is preferred over sigmoid/tanh as a default hidden-layer activation, and one failure mode ReLU itself has.
4. What is the "dying ReLU" problem and which activation variants were designed to fix it?
5. Why does GELU behave differently from ReLU near zero, and where is it used by default?
6. What breaks if you initialize every weight in a network to exactly zero?
7. State the difference between Xavier/Glorot initialization and He/Kaiming initialization, including which activation each targets.
8. Why is the LSTM forget-gate bias commonly initialized to 1 rather than 0?
9. Name three factors that make a network prefer "deeper" over "wider" (or vice versa), per the Universal Approximation Theorem discussion.
10. What is the difference between binary cross-entropy and categorical cross-entropy, and when would multi-label classification make you pick BCE over softmax-CE?
11. Write the SGD-with-momentum update rule and explain what the momentum term physically represents.
12. What two moving averages does Adam maintain, and what does each one control?
13. Why does AdamW decouple weight decay from the gradient update instead of adding an L2 term to the loss?
14. What does a learning-rate warmup phase protect against early in training?
15. Explain, in one sentence each, what BatchNorm normalizes over versus what LayerNorm normalizes over.
16. Why is LayerNorm generally preferred over BatchNorm inside Transformer blocks?
17. What is inverted dropout, and why is the `1/(1-p)` rescaling necessary at all?
18. What visible symptom in train vs. validation loss curves signals overfitting, and name two mitigations.
19. In PyTorch, why must you call `optimizer.zero_grad()` before `loss.backward()` in a standard training loop?
20. What is the practical difference between `model.parameters()` and `model.state_dict()`?
21. What does `model.eval()` actually toggle, and what does it explicitly NOT do (i.e., what do you still need `torch.no_grad()` for)?
22. Define transfer learning in one sentence and explain why early layers of a pretrained network transfer better than later layers.
23. What is the core idea behind data augmentation as a regularizer, and give two examples for image data.
24. What is gradient clipping, and what is the difference between clipping by norm and clipping by value?

### Medium

1. Derive why stacking 10 sigmoid layers causes gradients at the first layer to shrink by roughly `0.25^10`, and explain concretely why ReLU avoids this multiplicative decay (and what new problem it introduces instead).
2. Compare He and Xavier initialization mathematically — why does He initialization use a factor of 2 instead of 1 in the variance term, tying it to what ReLU does to roughly half the activations?
3. You're training a Transformer and see loss spikes every few hundred steps that eventually recover. Walk through your triage: is this an LR problem, an init problem, a data problem, or a normalization-placement problem, and how would you distinguish between them?
4. Explain why residual connections mitigate vanishing gradients — write the gradient of `H(x) = F(x) + x` with respect to `x` and explain the role of the "+1" term across many stacked blocks.
5. Contrast focal loss with plain weighted cross-entropy for handling class imbalance — what does focal loss's modulating term `(1-p_t)^γ` do that a static class weight cannot?
6. Explain the Adam bias-correction step (`m̂_t`, `v̂_t`) — what would go wrong in the first few steps of training if bias correction were omitted?
7. Why does dropout combined with BatchNorm sometimes hurt rather than help, and what is the "variance shift" mechanism responsible for this?
8. Compare Pre-Norm and Post-Norm Transformer designs — which is easier to train at large depth without warmup, and why, in terms of gradient magnitude at initialization?
9. Walk through why weight decay in plain Adam is not equivalent to L2 regularization, but the two ARE equivalent under vanilla SGD.
10. Explain the difference between `.detach()`, `torch.no_grad()`, and `torch.inference_mode()` — when would you choose one over the others in a training vs. serving context?
11. You call `loss.backward()` three times in a row on the same graph without calling `optimizer.zero_grad()` in between and without freeing the graph. What happens to `.grad`, and what would you observe numerically for `y = x**2`?
12. Explain why `nn.ModuleList` is required instead of a plain Python list when storing submodules, tying it to how `nn.Module.__setattr__` performs registration.
13. Design a fine-tuning strategy for adapting an ImageNet-pretrained ResNet to a small (500-image) medical-imaging dataset — justify freeze vs. unfreeze choices and learning-rate strategy per layer group.
14. Explain the DANN gradient-reversal layer: what does it do on the forward pass versus the backward pass, and why does that make feature extraction domain-invariant?
15. Contrast covariate shift, label shift, and concept drift as three distinct causes of train/test distribution mismatch, and give one detection strategy for each.
16. Explain MixUp's data and label mixing formulas — why does training on non-integer, blended labels act as a regularizer, and what does it do to decision boundary smoothness?
17. What problem does gradient checkpointing solve, and what is the fundamental compute-vs-memory trade-off it makes?
18. Explain how Prototypical Networks classify a query example in a few-shot setting, including what a "prototype" is and how episodic training mimics the test-time task structure.
19. Compare RMSNorm to LayerNorm — what computation does RMSNorm skip, and why is that adequate in practice for large Transformer training?
20. Explain why mixed-precision training needs a `GradScaler` — what numerical problem does loss scaling solve, and where in the training step does "unscaling" need to happen relative to gradient clipping?
21. Contrast early stopping against explicit regularizers (L2, dropout) — why can early stopping be described as an implicit regularizer, and what's the risk of tuning it against a small validation set?
22. Walk through why SwiGLU-style gated FFN blocks in modern Transformers use three weight matrices instead of two, and what the sigmoid/SiLU gate is doing functionally.

### Hard

1. Derive the vanishing-gradient bound for a deep, fully-connected sigmoid network in terms of the product of per-layer weight-matrix spectral norms and per-layer activation derivative bounds. Under what condition on the weight matrices could gradients neither vanish nor explode?
2. You freeze a pretrained backbone by setting `requires_grad=False` on all its parameters, but during fine-tuning you notice the backbone's behavior still subtly drifts. What's the likely culprit, and why does it have nothing to do with `requires_grad`?
3. Derive the Adam update rule from first principles (first/second moment EMA, bias correction, and the final parameter update), then explain precisely why AdamW's decoupled weight decay produces mathematically different final weights than L2-regularized Adam, even at convergence.
4. A ResNet trained with BatchNorm reaches 95% train accuracy but its validation accuracy oscillates wildly across evaluation runs at the same checkpoint. List every plausible cause tied to BatchNorm's train/eval statistic-switching behavior, and how you'd isolate which one is responsible.
5. You're told internal covariate shift as BatchNorm's mechanism of action has been directly challenged empirically (Santurkar et al., 2018). Explain the alternative "smoother loss landscape" hypothesis, and describe an experiment that would distinguish the two explanations.
6. Explain why LayerNorm is placed differently (Pre-Norm inside the residual branch vs. Post-Norm after the residual sum) has a first-order effect on gradient norms at initialization as depth grows — derive informally why Post-Norm Transformers need learning-rate warmup but well-designed Pre-Norm ones often don't.
7. Given a loss surface with pathological curvature (a long narrow valley), explain why plain SGD oscillates while SGD+momentum and Adam behave differently — then explain a scenario where Adam's per-parameter adaptive scaling can actually hurt generalization compared to SGD+momentum, referencing the "adaptive gradient methods generalize worse" line of research.
8. Derive the gradient of the reparameterized dropout mask through inverted dropout, and explain formally why the "trains an ensemble of 2^n subnetworks with shared weights" claim is an approximation rather than an exact equivalence.
9. You implement gradient checkpointing on a Transformer and notice training now takes 30% longer wall-clock time to reach the same loss, despite fitting a larger batch size in memory. Explain the mechanism causing the slowdown, and under what memory-bound vs. compute-bound regimes checkpointing is or isn't worth it.
10. Walk through MAML's bi-level optimization: write out the inner-loop adaptation step and the outer-loop meta-update, then explain why computing the exact outer-loop gradient requires second-order derivatives (Hessian-vector products) and how FOMAML and Reptile each avoid this at different levels of approximation quality.
11. A model trained with label smoothing (ε=0.1) is later used as a teacher for knowledge distillation. Explain what label smoothing does to the entropy of the teacher's output distribution, and why this can actually hurt distillation quality — tying it to the "dark knowledge" argument for soft labels.
12. Explain precisely why gradient clipping by global norm versus per-parameter value clipping can change the effective direction of the update (not just its magnitude), and construct a toy 2-parameter example where the two clipping strategies send the optimizer in different directions.
13. You suspect a custom `ContrastiveLoss` implementation has a subtle bug because training loss decreases but downstream embedding quality (nearest-neighbor retrieval) doesn't improve. Describe your debugging strategy that isolates whether the fault is in the loss's math, the negative-sampling strategy, or the embedding normalization.
14. Explain CORAL's covariance-alignment loss for domain adaptation mathematically, and describe a failure case where aligning second-order statistics (covariance) is insufficient to close the domain gap even though first and second moments match exactly.
15. Derive why Xavier initialization's "1/n_in" variance-preserving assumption breaks down once you add residual connections without additional scaling, and explain why GPT-2/LLaMA-style models apply an extra `1/sqrt(2L)` scaling factor to residual-branch output projections specifically.
16. Explain why `torch.nn.utils.clip_grad_norm_` computes the norm across all parameters jointly by default rather than clipping each parameter tensor independently, and what training behavior would change if you clipped every layer's gradient independently to the same max norm instead.
17. Contrast the bias/variance implications of DropPath (stochastic depth) applied uniformly across all residual blocks versus a linearly increasing drop-probability schedule by depth — why does the linear schedule empirically outperform uniform dropping in very deep ResNets?
18. In test-time adaptation via entropy minimization, explain what failure mode occurs if the test batch's label distribution is heavily skewed compared to training — why can the model's predictions collapse to a single class, and what safeguard (e.g., augmented-view averaging) mitigates this?
