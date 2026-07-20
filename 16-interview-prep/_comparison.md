---
module: Interview Prep
topic: Comparison Cheat Sheet
status: unread
tags: [interviewprep, cheatsheet, comparison]
---
# Interview Prep — Comparison Cheat Sheet

Rapid final-scan artifact. Comparison-dense, terse. Framed around the specific "why X over Y in an interview answer" angles this folder's notes emphasize — not a re-derivation of theory (see 02-classical-ml, 03-deep-learning, 05-llms for that).

---

## Fundamentals

**Bias vs Variance**
- Bias: same mistakes regardless of training sample — model too simple. Fix: more capacity, more features, less regularization.
- Variance: different mistakes depending on training sample — model memorizes noise. Fix: regularization, more data, simpler model, dropout.
- Diagnostic question: "if retrained on a different sample from the same population, would it make the same mistakes?" Yes → bias. No → variance.
- Trap: applying regularization to a high-bias model makes it worse; assuming more data alone fixes variance (still need regularization).

**Loss Function vs Metric**
- Loss must be differentiable (gradient signal); metric must reflect what the business cares about. They can diverge — a model can minimize cross-entropy while missing the recall target (hedges near 0.5 on borderline cases).
- Pick metric from error-cost asymmetry (FP vs FN) before training starts, not after.

**L1 vs L2 Regularization**
- L1 (Lasso) gradient is constant `λ·sign(w)` → overcomes small weights' loss gradient → exact zeros → sparsity.
- L2 (Ridge) gradient is `2λw` → push weakens near zero → small but non-zero weights.
- Geometric reason: L1 penalty contours are diamonds (corners at axes = sparsity); L2 contours are spheres (no corners).
- Pick L1 when you suspect most features are noise and want interpretability/sparsity. Pick L2 (or Elastic Net) when features are correlated/grouped — L1 arbitrarily keeps one from a correlated group.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| L1 | Feature selection, sparse interpretable model | Correlated feature groups | Sparsity vs instability under correlation |
| L2 | Correlated features, general shrinkage | Need a truly sparse model | Smooth shrinkage, no zeros |
| Elastic Net | Correlated + want some sparsity | Simplicity is priority | Extra hyperparameter (mix ratio) |

**Dropout**
- Solves co-adaptation (neurons over-relying on specific co-neurons), not just "randomly turn off neurons."
- Equivalent to training an implicit ensemble of `2^n` thinned subnetworks.
- Trap: applying to already-underpowered (high-bias) nets makes things worse; forgetting `model.eval()` causes stochastic inference — a real production bug pattern.

**Cross-Validation**
- Reduces variance of the performance *estimate*, not a way to "use more data."
- High std across folds is itself a signal — model sensitive to which examples land in train vs val.
- Trap: preprocessing/feature-selection before the CV split leaks information; using CV score for model selection then reporting the same number as final performance is optimistic bias (need a separate untouched test set).

**Curse of Dimensionality**
- Three concrete failure modes, not a vague warning: (1) distance concentration — nearest/farthest neighbor ratio → 1, breaking distance-based methods; (2) volume concentration — most hypercube volume near faces/corners, not center; (3) statistical sparsity — need exponentially more data for the same coverage density.
- Trap: thinking regularization alone fixes it — dimensionality reduction or the right inductive bias is usually the real lever.

---

## Evaluation Metrics

**Precision vs Recall vs F1**
- Precision = TP/(TP+FP): use when false positives are costly (spam filter — don't block real email).
- Recall = TP/(TP+FN): use when false negatives are costly (cancer screening — don't miss real cases).
- F1 = harmonic mean, punishes lopsided precision/recall (100% precision + 0% recall → F1 = 0, correctly reflecting a useless model).
- Framing this folder uses: derive the metric from the *cost structure of errors* in the specific scenario, not from a memorized formula.

**Accuracy vs Precision/Recall on imbalanced data**
- Predict-all-negative on 99.9%-negative data scores 99.9% accuracy, 0% recall on the class you care about. Accuracy is the wrong headline metric under imbalance — always report precision/recall/F1 (or PR-AUC) instead.

**ROUGE / mAP / Perplexity — metrics that hide what they don't measure**
- ROUGE rewards n-gram overlap/copying; systematically undervalues abstractive summaries and cannot detect hallucination.
- mAP is symmetric — doesn't capture FP/FN cost asymmetry; must specify IoU threshold (mAP@0.5 vs mAP@0.5:0.95 are not comparable) and check size-stratified AP (AP_S can be terrible while overall mAP looks fine).
- Perplexity only comparable across models sharing the same tokenizer/vocab; low perplexity does not imply good generation quality (can reflect memorization, not reasoning).
- Shared framing: know what the metric structurally cannot see before quoting it as evidence of quality.

| Metric | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Precision | FP costly (spam, fraud alerts to limited reviewers) | Missing positives is the real cost | Conservative, more false negatives |
| Recall | FN costly (medical screening, fraud coverage) | False alarms are expensive/high-volume | Aggressive, more false positives |
| F1 | Need one balanced number, no clear cost asymmetry | Costs are actually asymmetric | Hides which error type dominates |
| Accuracy | Balanced classes | Any meaningful class imbalance | Misleading headline number |
| mAP / IoU | Detection localization + classification jointly | Need FP/FN cost split, or size-specific perf | Threshold choice changes the ranking of models |
| Perplexity | Comparing checkpoints, same tokenizer, domain fit | Comparing across tokenizers, or judging generation quality | Correlates with, but ≠, downstream quality |

---

## Classical Algorithms (interview framing)

**KNN failure at scale**
- In high dimensions (curse of dimensionality), all points become roughly equidistant — "nearest" neighbors carry no signal. This is the concrete mechanism interviewers want, not "KNN doesn't scale."

**Anomaly Detection: supervised classifier vs one-class/density estimation**
- Supervised classifier trained on known fraud/attack patterns → high recall on those patterns, ~zero recall on novel ones (never seen in training).
- One-class/density approaches (e.g., autoencoder trained on normal data only) model "what normal looks like" and flag deviation — generalizes to unseen anomaly types because they model absence-of-normalcy, not presence-of-known-pattern.
- Pick supervised when you have abundant labeled examples of the specific fraud/defect types you'll keep seeing. Pick one-class/density estimation when novel/adversarial patterns are expected and label coverage of "bad" is inherently incomplete.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Supervised classifier (anomaly) | Known, recurring anomaly types with labels | Novel/adversarial anomalies expected | Blind to unseen patterns |
| One-class / density estimation (e.g. autoencoder reconstruction error) | Rare, evolving, or adversarial anomalies | Rich labeled data on all anomaly types already exists | Threshold tuning is operational, not statistical |

---

## Deep Learning (interview framing)

**RNN/LSTM vs Transformer — the mechanism, not the verdict**
- Vanilla RNN: vanishing gradient — repeated multiplication by recurrent weight matrix; spectral radius <1 shrinks gradient exponentially over long sequences.
- LSTM: forget gate can sit near 1.0, letting cell-state gradient flow largely unchanged across many steps — but training is still inherently sequential (can't compute step t before t-1), capping parallelism and degrading very long-range signal (e.g., 2000-word docs).
- Transformer: attention gives constant-length path between any two positions, removing the sequential bottleneck at the cost of O(n²) compute/memory.
- Interview trap called out explicitly: saying "Transformers are just better" instead of naming the sequential-bottleneck + vanishing-gradient mechanism attention specifically removes.
- Still pick RNN/LSTM when: resource-constrained deployment where O(n²) attention is prohibitive, or short sequences where the bottleneck never bites.

**Residual connections — two distinct arguments, both worth stating**
- Optimization/gradient argument: identity shortcut means `∂y/∂x = ∂F/∂x + I` — even if the block's own gradient vanishes, the identity term still delivers signal to early layers.
- Representation argument: learning a near-zero residual `F(x) ≈ 0` is easier than learning a near-identity mapping `H(x) ≈ x` through stacked nonlinearities.
- Trap: conflating this with DenseNet — residuals *add* input to output (single shortcut); DenseNet *concatenates* all previous layer outputs (more sharing, more memory).

**BatchNorm vs LayerNorm vs GroupNorm — pick by batch-size and architecture constraint, not preference**
- BatchNorm: normalizes across the batch dimension; breaks down at small batch sizes (noisy batch statistics) and mismatches train (batch stats) vs inference (running stats) — a real production bug source.
- LayerNorm: normalizes across features per-example — batch-size independent, standard for Transformers/sequence models where batch statistics are unstable or variable-length.
- GroupNorm: normalizes within channel groups — used in vision models trained with small per-GPU batch sizes (e.g., detection/segmentation) where BatchNorm statistics would be unreliable.
- Pre-LN vs Post-LN in Transformers: Pre-LN (norm before sublayer) trains more stably at depth; Post-LN can outperform when it works but is harder to stabilize.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| BatchNorm | Large-batch CNN training | Small batch, variable-length sequences, RNNs/Transformers | Train/inference statistic mismatch risk |
| LayerNorm | Transformers, small/variable batch | Conv nets where batch stats are cheap & reliable | Slightly different regularizing effect than BatchNorm |
| GroupNorm | Vision models with small per-GPU batch (detection/segmentation) | Very large batch is available (BatchNorm is fine and simpler) | Extra hyperparameter: number of groups |

**Dropout vs BatchNorm as regularizers**
- Both are training/inference-mode-sensitive: forgetting the mode switch is a specific, named production bug in these notes (stochastic outputs in eval, or train-stat/inference-stat mismatch).

---

## Optimization (interview framing)

**Momentum vs Nesterov vs Adam vs AdamW**
- Momentum: accumulates velocity, damps oscillation in ravines.
- Nesterov: looks ahead before computing gradient — corrects overshoot momentum alone would cause.
- Adam: per-parameter adaptive learning rate + momentum, with bias correction for early-step estimate bias.
- AdamW: decouples weight decay from the gradient-adaptive update — plain Adam's L2-as-added-gradient-term interacts badly with adaptive scaling; AdamW applies decay directly to weights, restoring the regularization effect L2 is supposed to have.
- Interview framing: know *why* AdamW was introduced (Adam+L2 quietly under-regularizes large-gradient parameters) — not just "AdamW is the modern default."

**Vanishing vs Exploding gradients — different fixes for different directions**
- Vanishing: residual connections, better activations (ReLU family over sigmoid/tanh), normalization, careful init, LSTM-style gating.
- Exploding: gradient clipping, smaller learning rate, careful init.
- Trap: treating these as one problem with one fix — the notes explicitly frame them as requiring distinct remedies.

**Batch size tradeoffs**
- Larger batch: more accurate gradient estimate per step, more memory, typically needs a proportionally larger learning rate (linear scaling rule), and *can* converge to sharper minima that generalize worse — because small-batch noise acts as implicit regularization.
- Trap: increasing batch size without scaling the learning rate up — underfits because each step becomes too small relative to gradient quality.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Momentum | Ravine-shaped loss surfaces, general default | Need fastest theoretical convergence guarantees | Simple, minimal tuning |
| Nesterov | Momentum overshoot is a visible problem | Marginal benefit not worth the complexity | Lookahead gradient adds small overhead |
| Adam | Sparse gradients, fast prototyping, most deep nets | Want the decoupled-weight-decay regularization benefit | Can under-regularize with naive L2 |
| AdamW | Transformers and most modern large-model training | — (largely supersedes Adam+L2 for regularized training) | One more hyperparameter to tune correctly (decay) |

---

## NLP

**Bag-of-Words / TF-IDF vs Embeddings vs Contextual (Transformer) representations**
- BoW: loses order entirely; good baseline when word identity matters more than order (ticket classification).
- TF-IDF: recovers discriminative weighting (`tf·idf`, down-weights ubiquitous terms) — still no order/negation handling ("server is not responding" vs "is responding" look similar).
- Word2Vec/GloVe/FastText: dense vectors from distributional similarity, but **context-free** — "bank" gets one vector regardless of financial vs river context. FastText adds subword n-grams, so it handles unseen/rare words (product names) that Word2Vec cannot.
- Transformers (BERT/GPT-style): contextual — same word gets different representations depending on surrounding text; solves what static embeddings structurally cannot.
- Framing this folder pushes: don't skip the cheap baseline (TF-IDF+logistic regression) just because it's "unsophisticated" — it's a legitimate production answer when data is limited/latency is strict/interpretability matters.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Bag-of-Words / TF-IDF | Cheap, interpretable, small-data classification | Order/negation/syntax matters | Fast & interpretable vs no context modeling |
| Word2Vec | General-purpose static embeddings, larger corpora | Rare/novel words (OOV), small corpora (GloVe/Word2Vec compete here) | Context-free representation |
| FastText | Morphologically rich languages, frequent novel terms (product names) | Task needs contextual disambiguation | Subword robustness vs still context-free |
| Transformer contextual embeddings | Disambiguation-heavy tasks, SOTA accuracy | Extreme latency/interpretability constraints | Expensive, less interpretable |

**BERT (encoder) vs GPT (decoder) — objective drives architecture, not preference**
- BERT: bidirectional attention, masked-LM pretraining → best when full input is available at inference (classification, NER, extractive QA) — bidirectional context enriches each token's representation.
- GPT: causal attention, next-token pretraining → mandatory when the model must generate tokens it hasn't produced yet (can't peek at the future during training if it can't at inference).
- Trap: asserting "BERT is better for understanding" as a hard rule — scale can close/reverse this; also forgetting encoder-decoder (T5/BART) as the right answer for seq2seq tasks needing both rich input encoding and autoregressive output.

**Stemming vs Lemmatization**
- Stemming: mechanical suffix-chopping, fast, crude ("universal" → "univers").
- Lemmatization: dictionary-aware, respects POS, slower but correct.
- Both largely unnecessary in transformer pipelines (subword tokenization already handles morphology) — still relevant in classical/keyword-matching pipelines and search.

**Extractive vs Abstractive summarization**
- Extractive: copies spans — faithful by construction, choppier prose. Safer where hallucination risk is unacceptable (legal, dispute-sensitive transcripts).
- Abstractive: generates new text — more natural, real hallucination risk (plausible but unsupported claims).
- Pick by the application's hallucination tolerance, not by which sounds more advanced. ROUGE score is a poor referee here — rewards copying, penalizes legitimate paraphrase, blind to hallucination.

**Dependency parsing vs implicit (Transformer attention) structure**
- Explicit parsers still relevant when the application needs guaranteed, verifiable structured output (information extraction pipelines, semantic role labeling). Transformers often implicitly encode syntax in attention but don't guarantee it as inspectable/structured output.

---

## Computer Vision

**CNN vs Vision Transformer (ViT)**
- CNN: locality + translation equivariance are hard architectural constraints — strong inductive bias, better data efficiency on limited data (from-scratch ImageNet-1k: CNNs typically win).
- ViT: learns spatial relationships from scratch via self-attention — no built-in locality bias, needs web-scale pretraining to amortize that cost, but then dominates at scale.
- Framing: this is a data-efficiency vs scalability tradeoff, not "ViT is newer so better."

**One-stage vs Two-stage object detectors**
- Two-stage (Faster R-CNN): region proposals first, then classify/refine each — higher accuracy (esp. small/overlapping objects), higher latency (sequential RoI processing).
- One-stage (YOLO/SSD/RetinaNet): dense prediction from a grid in one pass — much faster, historically less accurate, gap closed by modern versions. RetinaNet's Focal Loss (`-(1-p_t)^γ log(p_t)`) fixes the specific problem of ~10,000 easy background anchors drowning the loss signal.
- Pick by deployment constraint first: 60fps conveyor-belt inspection → one-stage even at slightly lower mAP; offline medical imaging where recall matters more than throughput → two-stage.

**Classification vs Detection vs Semantic Segmentation vs Instance Segmentation vs Panoptic**
- Escalate only when the specific failure mode demands it (e.g., stacked/occluded objects require instance segmentation to count separately — not "segmentation is more sophisticated so use it").
- Segmentation costs ~15x detection's annotation cost (pixel masks vs boxes) — a real budget factor, not just a modeling one.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Classification | Single global label per image | Need location or count | Cheapest annotation, least information |
| Detection | Counting/locating discrete objects | Objects touch/occlude and must be separated per-pixel | Bounding box only, no pixel precision |
| Semantic segmentation | Per-pixel class, don't need instance identity | Need to count touching same-class objects | 15x annotation cost vs detection |
| Instance segmentation | Per-pixel + per-instance (e.g. stacked items) | Detection already solves the task | Heaviest annotation + compute cost |

**Max pooling vs Global Average Pooling vs strided convolution**
- Max pooling: local translation tolerance — strongest activation in a region, position-insensitive locally.
- Global Average Pooling: replaces FC classification heads — collapses spatial map to a vector, huge parameter reduction (e.g., ResNet-50 head ~8M params → one linear layer), implicit regularization, enables variable input size.
- Strided convolutions: learnable alternative to fixed pooling windows — modern architectures (ResNet, ViT patch embedding) increasingly prefer this over hand-fixed pooling.

**Data augmentation — must be label-preserving, not just "more data"**
- Geometric (flip/rotate/crop): spatial invariance — wrong for digit recognition (6→9 under flip) or lateralized medical anatomy.
- Color jitter: photometric invariance — wrong when color is diagnostic (pathology slides, skin lesions).
- CutMix/MixUp: occlusion robustness / smooth decision boundaries via label interpolation.
- The interview-relevant judgment call is *checking label-preservation per task*, not picking from the standard augmentation menu by default.

---

## System Design & MLOps

**Shadow vs Canary vs A/B vs Multi-armed bandit deployment**
- Shadow: new model runs alongside old, serves no real traffic, compares outputs — zero user risk, but doesn't measure real business-metric impact.
- Canary: small % of real traffic to new model, ramped if healthy — real risk exposure but bounded and reversible.
- A/B test: designed comparison for a specific statistical decision — needs a pre-registered MDE/sample size, vulnerable to peeking.
- Multi-armed bandit: adaptively shifts traffic toward the better-performing arm during the test — better cumulative outcome than a fixed-split A/B when you don't need a clean causal read for a report, worse when you need rigorous, unbiased statistical inference.
- Pick by what you need: is-it-safe (shadow) → does-it-work-at-small-scale (canary) → which-is-actually-better-statistically (A/B) → maximize-outcome-while-learning (bandit).

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Shadow | Validate new model is safe/non-crashing before any user exposure | Need to measure actual business-metric lift | No user risk, no efficacy signal |
| Canary | Bounded-risk rollout, catch regressions early | Need statistically rigorous comparison | Fast rollback vs weak statistical power |
| A/B test | Rigorous causal comparison for a decision | Urgency / can't wait for sample size | Clean inference vs slower, needs discipline (no peeking) |
| Multi-armed bandit | Maximize outcome while still learning | Need a clean, reportable causal effect size | Optimizes regret, sacrifices clean inference |

**Data drift vs Concept drift vs Label drift**
- Data drift: `P(X)` changes — input distribution shifts (e.g., new user demographic).
- Concept drift: `P(Y|X)` changes — same input, the *true relationship* to the label changes (e.g., fraud tactics evolve).
- Label drift: `P(Y)` changes — base rate of the outcome shifts (e.g., seasonal fraud rate change).
- Distinguishing these determines the fix: data drift → check feature pipeline/population; concept drift → retrain, the old model is stale by definition; label drift → recalibrate threshold, model may still be structurally correct.
- PSI (Population Stability Index) and KS test are the standard detection tools; retraining triggers should be tied to which drift type is diagnosed, not a blanket schedule.

**Feature store / train-serve skew — diagnose before retraining**
- Explicit "Quick Diagnostics" framing in these notes: check train-serve skew *first* when production accuracy drops — retraining papers over a pipeline bug and it will recur.
- Point-in-time correctness (no future feature leakage into training) is the specific failure mode feature stores are built to prevent.

**Online vs Batch inference**
- Online: low-latency, per-request, needed when a live user is waiting on the prediction.
- Batch: high-throughput, scheduled, appropriate when predictions can be precomputed (e.g., nightly recommendation refresh).
- Hybrid pattern: batch-precompute the expensive part, online-serve a cheap lookup/rerank — common in retrieval-ranking systems.

**Retrieval-Ranking two-stage systems**
- Retrieval stage: optimized for recall, cheap approximate methods (FAISS/BM25/collaborative filtering) over the full candidate pool.
- Ranking stage: optimized for precision, expensive precise methods (GBTs/DNNs/cross-encoders) over the small retrieved candidate set.
- This mirrors the one-stage/two-stage detector tradeoff in CV and the retrieval-vs-fine-tuning tradeoff in LLMs: cheap-and-broad first, expensive-and-narrow second.

**Quantization / Pruning / Distillation (inference optimization)**
- Quantization (PTQ vs QAT): PTQ is fast/cheap but can lose accuracy on sensitive layers; QAT bakes quantization noise into training for better final accuracy at higher training cost.
- Pruning (structured vs unstructured): structured (remove whole channels/heads) gives real speedup on standard hardware; unstructured (remove individual weights) gives higher theoretical sparsity but needs specialized sparse kernels to realize speedup.
- Distillation: compress into a smaller student model trained to match a teacher's outputs — best when you need a fundamentally smaller model, not just a faster-running same-size one.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| PTQ | Fast deployment, limited retraining budget | Accuracy-sensitive layers degrade too much | Cheap but riskier accuracy hit |
| QAT | Accuracy-critical quantized deployment | No budget for extra training | Better accuracy, more training cost |
| Structured pruning | Real hardware speedup on standard GPUs/CPUs | Need max theoretical sparsity | Easier to deploy, less aggressive |
| Unstructured pruning | Maximum sparsity, specialized inference stack | Standard hardware only | High sparsity but needs sparse kernel support |
| Distillation | Need a genuinely smaller model | Just need current model faster (use quantization instead) | Extra training pipeline, teacher dependency |

---

## LLMs

**RAG vs Fine-tuning vs Prompting — the decision framework these notes use**
- Knowledge/freshness problem (needs facts not in the base model, or facts that change) → RAG.
- Behavior/style problem (needs to consistently act/respond a certain way) → fine-tuning (or PEFT/LoRA for efficiency).
- Describable-task problem (can be specified well in context) → prompting.
- Explicit trap: "temperature doesn't fix knowledge" — sampling parameters cannot compensate for missing/stale knowledge; that's a retrieval or fine-tuning problem, not a decoding problem.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Prompting | Describable, in-context-solvable tasks | Task requires facts outside context or consistent behavior change | Cheapest, fastest iteration, weakest guarantee |
| RAG | Freshness/knowledge gaps, verifiable grounding | Task is about behavior/style, not facts | Adds retrieval latency + infra, keeps knowledge current |
| Fine-tuning / LoRA | Consistent behavior/style/format changes | Fast-changing facts (goes stale) | Higher upfront cost, stable behavior |

**Absolute (sinusoidal) vs Learned vs RoPE vs ALiBi positional encoding**
- Sinusoidal absolute: fixed, some extrapolation capability, no learned parameters.
- Learned absolute: flexible but typically fails to generalize beyond trained context length.
- RoPE: encodes relative position via rotation, better length generalization, used in most modern LLMs.
- ALiBi: linear recency bias baked directly into attention scores, strong extrapolation to longer contexts than trained on.
- Framing: pick based on how much context-length generalization the deployment actually needs — this is a length-extrapolation tradeoff table, not an arbitrary architecture choice.

**Top-k vs Top-p (nucleus) vs Beam search vs Temperature**
- Temperature: rescales logits before softmax (`P(w_i) = exp(z_i/T)/Σexp(z_j/T)`) — controls sharpness of the *entire* distribution, doesn't truncate it.
- Top-k: truncate to k highest-probability tokens — simple, but k is a fixed cutoff regardless of how peaked/flat the distribution is.
- Top-p (nucleus): truncate to the smallest set whose cumulative probability exceeds p — adapts to distribution shape (few tokens when confident, many when uncertain).
- Beam search: keeps multiple candidate sequences, maximizes joint sequence probability — good for tasks with one "correct" output (translation), bad for open-ended generation (produces generic, repetitive text).
- Explicit trap: temperature/sampling changes *how* the model expresses what it knows, not *what* it knows — cannot fix hallucination or knowledge gaps.

**Causal masking + KV-cache — prefill vs decode framing**
- Prefill: processing the prompt, compute-bound, fully parallel across prompt tokens.
- Decode: generating tokens one at a time, memory-bandwidth-bound, sequential — KV-cache avoids recomputing all previous keys/values (`O(T²)` naive → `O(n)` incremental per new token), trading memory for compute.
- This compute-bound vs memory-bandwidth-bound distinction is the basis for most LLM inference optimization decisions (batching helps decode more than prefill, quantization targets the memory-bound stage, etc).

**SFT vs RLHF vs DPO**
- SFT: directly imitate curated demonstrations — simplest, cheapest, ceiling limited by demonstration quality.
- RLHF: optimize a learned reward model with a KL penalty back to the SFT policy (`r_θ(x,y) - β·D_KL(π_φ‖π_SFT)`) — can exceed demonstration quality but introduces reward hacking risk and RL training complexity/instability.
- DPO: reformulates the RLHF objective as a direct classification loss on preference pairs, skipping the separate reward model and RL loop — gets much of RLHF's benefit without RL's instability/infra cost.
- Pick DPO over full RLHF when you want the alignment benefit without standing up a full RL pipeline; pick RLHF when you need the flexibility of an explicit, reusable reward model (e.g., iterative reward-model improvement).

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| SFT | Cheap baseline, clear demonstrations available | Need behavior beyond demonstration quality | Simple, but capped by data quality |
| RLHF | Push past demonstration ceiling, have infra for RL | Limited engineering resources for RL pipeline | Best ceiling, most complex/unstable |
| DPO | Want RLHF-like gains without RL infra | Need an explicit, reusable/iterable reward model | Simpler pipeline, less flexible than full RLHF |

**LLM evaluation: benchmark suites vs perplexity vs LLM-as-judge vs human eval**
- MMLU/HellaSwag/TruthfulQA/HumanEval: standardized, comparable, but can leak into pretraining data and don't capture your specific task.
- Perplexity: cheap proxy, not comparable across tokenizers, uncorrelated with generation quality.
- LLM-as-judge: scalable, cheap relative to human eval, but has known biases (favors verbosity, its own family's style).
- Human eval: gold standard for actual quality judgment, expensive/slow, the fallback when the other three don't resolve a decision.
- Framing: escalate from cheap/biased to expensive/reliable only as far as the decision actually requires.

**Hallucination — architecturally fundamental, not a fixable bug**
- Causes named explicitly: knowledge gaps, conflation of similar facts, context neglect, overconfidence from the training objective (next-token prediction rewards fluent completion, not calibrated uncertainty).
- Mitigations are layered, not a single fix: RAG (grounding), chain-of-thought (reasoning surface area), calibration, attribution/citation, verification steps — no single one is sufficient alone.

---

## Quick Cross-Domain Diagnostics (as framed in this folder)

- **Production accuracy drop:** check train-serve skew / pipeline consistency (resize interpolation, normalization, tokenizer mismatch) *before* assuming you need to retrain — retraining papers over a recurring bug.
- **"Model A vs Model B" question with no context:** always ask latency budget, data volume, interpretability need, and error-cost asymmetry before naming an architecture — these notes repeatedly frame architecture choice as the *last* step, not the first.
- **"Why did X replace Y" questions (RNN→Transformer, two-stage→one-stage, Adam→AdamW):** answer with the specific mechanism/failure being solved, not a general quality claim — this is the single most repeated "common trap" across every file in this folder.
