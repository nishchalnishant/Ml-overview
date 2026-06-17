---
module: Deep Learning
topic: Flashcards
subtopic: ""
status: unread
tags: [deeplearning, ml, flashcards]
---
# Deep Learning Flashcards

---

<!-- From mcp.md -->
**How does the model know what tools are available?** #flashcard
How does the model know what tools are available?

**How are tool inputs described and validated?** #flashcard
How are tool inputs described and validated?

**How are results returned and formatted?** #flashcard
How are results returned and formatted?

**How do you distinguish "actions the model should trigger" from "data the application should inject"?** #flashcard
How do you distinguish "actions the model should trigger" from "data the application should inject"?

**how clients discover available capabilities (capability negotiation)?** #flashcard
how clients discover available capabilities (capability negotiation)

**how tools, resources, and prompts are described (structured schemas)?** #flashcard
how tools, resources, and prompts are described (structured schemas)

**how tool calls are issued and results returned?** #flashcard
how tool calls are issued and results returned

**how errors are communicated?** #flashcard
how errors are communicated

**the lifecycle of a client-server connection?** #flashcard
the lifecycle of a client-server connection

**manages the lifecycle of one or more MCP clients?** #flashcard
manages the lifecycle of one or more MCP clients

**decides which servers to connect to?** #flashcard
decides which servers to connect to

**applies safety policies before passing tool definitions to the model?** #flashcard
applies safety policies before passing tool definitions to the model

**presents tool results to the user?** #flashcard
presents tool results to the user

**initializes the connection and negotiates capabilities?** #flashcard
initializes the connection and negotiates capabilities

**sends requests (tools/list, tools/call, resources/read, prompts/get)?** #flashcard
sends requests (tools/list, tools/call, resources/read, prompts/get)

**receives structured responses and returns them to the host?** #flashcard
receives structured responses and returns them to the host

**Is this tool allowed in this context?** #flashcard
Is this tool allowed in this context?

**Do the arguments look safe?** #flashcard
Do the arguments look safe?

**Does this action require user confirmation?** #flashcard
Does this action require user confirmation?

**Tied to a specific provider's API?** #flashcard
Tied to a specific provider's API

**Application code handles execution?** #flashcard
Application code handles execution

**No standardized server format or discovery?** #flashcard
No standardized server format or discovery

**Not reusable across applications?** #flashcard
Not reusable across applications

**Abstracts provider-specific function calling?** #flashcard
Abstracts provider-specific function calling

**Rich ecosystem of pre-built tools?** #flashcard
Rich ecosystem of pre-built tools

**Tools are embedded in application code?** #flashcard
Tools are embedded in application code

**Couples you to the LangChain ecosystem?** #flashcard
Couples you to the LangChain ecosystem

**Provider-agnostic?** #flashcard
any model that supports MCP works

**Tools are independently deployable services?** #flashcard
Tools are independently deployable services

**Reusable across hosts and applications?** #flashcard
Reusable across hosts and applications

**Standardized discovery, execution, and error handling?** #flashcard
Standardized discovery, execution, and error handling

**Higher deployment complexity than inline functions?** #flashcard
Higher deployment complexity than inline functions

**Return tool results as structured data, not raw text that could be mistaken for instructions?** #flashcard
Return tool results as structured data, not raw text that could be mistaken for instructions

**Use a distinct tool_result message role?** #flashcard
not interpolation into the system prompt

**Keep high-risk tools (delete, send, write) behind explicit user confirmation?** #flashcard
Keep high-risk tools (delete, send, write) behind explicit user confirmation

**Validate that results don't contain known injection patterns before returning them to the model?** #flashcard
Validate that results don't contain known injection patterns before returning them to the model

**Restrict all access to a root directory?** #flashcard
prevent path traversal

**Separate read and write tools?** #flashcard
easier to grant different permissions

**Return metadata (size, modified time) alongside content?** #flashcard
Return metadata (size, modified time) alongside content

**Implement a whitelist of allowed file types?** #flashcard
Implement a whitelist of allowed file types

<!-- From pytorch-fundamentals.md -->
**nn.Parameter sets it True automatically?** #flashcard
model weights are always tracked

**Input data tensors should be False?** #flashcard
you don't optimize inputs

**Intermediate tensors (non-leaves) don't retain .grad by default?** #flashcard
they're computed on demand

**Most schedulers (CosineAnnealingLR, StepLR, ReduceLROnPlateau)?** #flashcard
call scheduler.step() once per epoch, after validation.

**Batch-level schedulers (OneCycleLR)?** #flashcard
call scheduler.step() once per batch, inside the batch loop.

**Calling at the wrong cadence produces a wrong learning rate curve with no error message.?** #flashcard
Calling at the wrong cadence produces a wrong learning rate curve with no error message.

**log(0)?** #flashcard
add epsilon: torch.log(x + 1e-8)

**Division by zero?** #flashcard
guard denominators

**Exploding gradients?** #flashcard
use gradient clipping

**Bad initialization?** #flashcard
use Xavier/Kaiming

<!-- From methods/nlp-fundamentals.md -->
**Skip-gram: given a center word, predict surrounding context words?** #flashcard
P(context | word)

**CBOW: given surrounding context, predict the center word?** #flashcard
P(word | context)

**Negative sampling?** #flashcard
instead of softmax over the full vocabulary (expensive), train a binary classifier against K random "negative" words

**Subsampling: downsample frequent words ("the", "is", "a") proportionally?** #flashcard
they appear in so many contexts that they add noise

**Last hidden state?** #flashcard
one vector per token

**Concatenation/sum of last 4 layers?** #flashcard
richer features

**CLS token?** #flashcard
sentence-level representation

**B?** #flashcard
Beginning of entity

**I?** #flashcard
Inside entity (continuation)

**O?** #flashcard
Outside (not an entity)

**ROUGE-1?** #flashcard
unigram overlap

**ROUGE-2?** #flashcard
bigram overlap

**ROUGE-L?** #flashcard
longest common subsequence

**Context management?** #flashcard
concatenate conversation history into the prompt

**Persona consistency?** #flashcard
system prompt shapes behavior

**Grounding?** #flashcard
RAG to reduce hallucination in task-oriented dialogue

**NLU?** #flashcard
intent classification + slot filling (extract parameters: "date", "location", "cuisine")

**Dialogue State Tracking (DST)?** #flashcard
maintain what the user has specified so far

**Policy?** #flashcard
decide next system action given current state

**NLG?** #flashcard
generate natural language response

**BPE merges most frequent byte pairs iteratively; WordPiece merges by likelihood. Both create subword vocabularies. SentencePiece works on raw text with no whitespace assumption.?** #flashcard
BPE merges most frequent byte pairs iteratively; WordPiece merges by likelihood. Both create subword vocabularies. SentencePiece works on raw text with no whitespace assumption.

**Word2Vec?** #flashcard
local context windows. GloVe: global co-occurrence statistics. FastText: extends with character n-grams for OOV handling.

**ELMo = contextual embeddings from bidirectional LSTM. BERT = bidirectional Transformer with masked language modeling.?** #flashcard
ELMo = contextual embeddings from bidirectional LSTM. BERT = bidirectional Transformer with masked language modeling.

**Seq2Seq + Bahdanau attention: decoder attends over all encoder hidden states at each step?** #flashcard
this is the direct precursor to Transformer self-attention.

**Beam search?** #flashcard
maintain top-K hypotheses at each step. Better than greedy but still an approximation. Length penalty prevents preference for short sequences.

**BLEU measures precision of n-gram overlap; ROUGE measures recall; BERTScore uses contextual embedding similarity?** #flashcard
semantic not surface-level.

**NER?** #flashcard
BIO tagging converts span extraction to token classification. BERT + token classification head is the standard approach.

<!-- From methods/segmentation.md -->
**Efficient self-attention?** #flashcard
reshape K and V by reduction ratio R before computing attention, reducing cost from O(N²) to O(N²/R). R decreases across stages: {64, 16, 4, 1}.

**Mix-FFN: x + Conv_{3×3}(FFN(x))?** #flashcard
the 3×3 conv carries implicit positional signal.

**Top-down?** #flashcard
detect bounding boxes first → crop each person → run pose network. Accurate; inference time scales linearly with person count.

**Bottom-up?** #flashcard
detect all keypoints globally → group into individuals. Fixed inference time regardless of crowd size.

**A prediction is TP if its mask/box IoU with the matched ground-truth exceeds a threshold.?** #flashcard
A prediction is TP if its mask/box IoU with the matched ground-truth exceeds a threshold.

**COCO AP: average of AP at IoU thresholds {0.50, 0.55, ..., 0.95}?** #flashcard
10 thresholds.

**mAP?** #flashcard
mean AP over all categories.

**d_k?** #flashcard
Euclidean distance between predicted and ground-truth keypoint.

**s?** #flashcard
object scale (√bounding box area).

**σ_k?** #flashcard
per-keypoint constant reflecting annotation noise (hip σ=0.107, eye σ=0.025).

**v_k?** #flashcard
visibility flag.

<!-- From methods/video-understanding.md -->
**Temporal modeling?** #flashcard
which frames matter, and how do they relate to each other?

**Computational cost?** #flashcard
videos run at 25–60 fps over minutes. Processing every frame at full resolution with a deep network is prohibitive.

**Long-range dependencies: some actions (cooking a meal, playing a sport) require context from seconds or minutes apart?** #flashcard
well beyond the receptive field of any practical 3D CNN.

**Motion vs. appearance confusion?** #flashcard
some actions look identical as still images but differ only in motion direction or speed. A model that ignores temporal order cannot distinguish them.

**Optical flow is expensive?** #flashcard
dense flow computation adds significant preprocessing time and storage. It's feasible offline but not for real-time pipelines.

**Late fusion misses cross-stream interactions?** #flashcard
the two streams are trained independently and only communicate at the prediction level. An arm moving in a way that only makes sense in context of the adjacent body part cannot be captured.

**Flow estimation errors propagate?** #flashcard
inaccurate flow (from motion blur, rapid scene changes, occlusion) produces noisy motion features that hurt the temporal stream.

**Memory and compute?** #flashcard
3D convolutions are ~K_t × more expensive than 2D. A 3×3×3 conv is 3× more operations per layer than a 3×3 conv, and memory scales with temporal depth T.

**Pre-training data scarcity?** #flashcard
3D representations require video pre-training to be useful. Unlike 2D models with ImageNet, 3D models can't leverage the vast image pre-training ecosystem as directly.

**Short temporal range?** #flashcard
C3D and I3D process 16-frame clips. Long-range dependencies (minutes apart) require separate architectural mechanisms.

**Two-pathway cost?** #flashcard
the Fast pathway runs at 8× the frame rate but with 1/8 the channels. The total FLOP cost is comparable to a single-pathway model, but the two-stage design adds engineering complexity.

**Fixed ratio?** #flashcard
the 4:1 or 8:1 frame-rate ratio is a hyperparameter that may not be optimal across all action categories.

**Proposal + classification (BSN, BMN)?** #flashcard
generate temporal proposals of varying durations, then classify each.

**One-stage (AFSD, ActionFormer)?** #flashcard
directly predict start/end/class from dense temporal features.

**DETR-style (RTD-Net)?** #flashcard
end-to-end with action queries that decode to (start, end, class) tuples.

**Temporal stride > 1: read every other frame?** #flashcard
halves temporal resolution and compute.

**Knowledge distillation?** #flashcard
distill a 3D CNN teacher into an efficient 2D CNN student for edge deployment.

**Efficient attention: MViT pooling attention, TimeSformer factorized attention?** #flashcard
reduce O(T²H²W²) attention to manageable cost.

<!-- From methods/3d-vision.md -->
**No local structure: every point is processed independently before the global pool. The MLP cannot reason about how neighboring points relate to each other?** #flashcard
it has no spatial receptive field. Fine-grained local geometry (surface normals, local curvature) is not captured.

**Global context is a bottleneck?** #flashcard
the single 1024-d vector must encode the entire point cloud. Detailed spatial information about specific regions is compressed into this global descriptor and cannot be recovered for per-point tasks.

**Sensitivity to density?** #flashcard
LiDAR returns dense point clouds near the sensor and sparse clouds far away. PointNet has no mechanism to adapt to variable local density.

**Fixed radius?** #flashcard
ball query uses a single radius r per level. If the point cloud has highly variable density (dense near sensor, sparse far away), a fixed r may miss neighbors in sparse regions and include too many in dense regions. The multi-scale grouping (MSG) variant uses multiple radii at each level but at higher compute cost.

**FPS is sequential: finding the farthest point requires maintaining a distance matrix updated after each selection. FPS on N=8192 points takes O(N²) naive, or O(N log N) with priority queues?** #flashcard
still a training bottleneck.

**Discrete hierarchy?** #flashcard
the number of levels and their radii are fixed hyperparameters. The receptive field at each level is fixed; it cannot adapt to object scale.

**k-NN recomputation cost?** #flashcard
at each layer, computing pairwise distances and finding k-NN for N points is O(N² d) per layer. For N=2048, d=128, this is significant, especially during training with gradients flowing back through the graph topology (which itself depends on features). Approximate k-NN (e.g., FAISS) can help at the cost of gradient accuracy.

**Dynamic graph breaks gradient flow: the graph topology is a non-differentiable discrete selection (which k neighbors to use). Gradients cannot flow through the selection step itself?** #flashcard
only through the features of selected neighbors. This limits the expressiveness of the learned topology.

**Local-only attention?** #flashcard
attention is restricted to k local neighbors, so the model cannot capture long-range dependencies in a single layer. Stacking layers increases the effective receptive field, but it grows slowly (like convolutions).

**k-NN recomputation?** #flashcard
same cost as DGCNN. The k-NN graph is typically computed in 3D space (not recomputed in feature space at each layer), which recovers computational efficiency but loses DGCNN's feature-space adaptivity.

**Submanifold vs. regular sparse conv: if you use regular sparse conv, every input occupied voxel can activate output positions in its 3×3×3 neighborhood, causing the sparse tensor to grow denser after each layer (the "dilation" problem). Submanifold sparse convolutions only compute outputs at positions that were occupied in the input?** #flashcard
keeping the sparsity pattern fixed. This prevents feature dilation but limits the receptive field.

**Custom CUDA kernels required?** #flashcard
spconv operations cannot be expressed as standard PyTorch ops; they require specialized GPU kernels. Portability and debugging are harder.

**Loss of vertical structure?** #flashcard
compressing z into a single pillar feature loses fine-grained height information. This matters for distinguishing cyclists from pedestrians, or for detecting objects on slopes. CenterPoint and BEVFusion use sparse 3D backbones to retain height information before projecting to BEV.

**Fixed pillar resolution?** #flashcard
the (x, y) pillar grid has a fixed resolution. Fine-grained spatial precision requires small pillars → large feature maps → more memory. There is an explicit accuracy-speed tradeoff here.

**Occupancy Networks?** #flashcard
f_θ(x, z) → [0, 1], where z is a shape latent code from an encoder. The surface is the level set {x : f_θ(x, z) = 0.5}. Extract the surface via marching cubes on a dense query grid.

**DeepSDF?** #flashcard
f_θ(x, z) → ℝ, where the output is the signed distance to the surface. Negative inside, positive outside. The surface is {x : f_θ(x, z) = 0}.

**Per-scene optimization?** #flashcard
a trained NeRF represents a single scene. There is no generalization to new scenes; you must train a new NeRF from scratch for each scene. Training takes hours to days on an 8-GPU machine.

**Slow inference: naive NeRF queries the MLP at K×N ray samples per frame. Rendering a 1920×1080 image at K=128 samples requires ~264M MLP forward passes?** #flashcard
far too slow for real-time.

**View interpolation, not extrapolation?** #flashcard
NeRF needs training views that densely cover the scene. For novel viewpoints that are far outside the training view distribution, rendering quality degrades significantly.

**Instant-NGP (Müller et al., 2022)?** #flashcard
replaces the deep MLP with a small MLP combined with a multi-resolution hash grid of learned features. Hash grid queries are fast CUDA memory accesses; the MLP is shallow. Reduces training from hours to minutes.

**Mip-NeRF (Barron et al., 2021)?** #flashcard
models rays as cones (not infinitely thin lines), using integrated positional encoding over cone frustums. Handles anti-aliasing at different scales (close-up vs. distant views).

**Block-NeRF (Tancik et al., 2022)?** #flashcard
decomposes a city-scale scene into multiple NeRF blocks, each covering a spatial region, with appearance embeddings per block for consistency.

**μ_i ∈ ℝ³?** #flashcard
center position

**Σ_i ∈ ℝ^{3×3}?** #flashcard
covariance matrix (factored as Σ = RSS^T R^T where R is rotation, S is scale)

**α_i ∈ [0, 1]?** #flashcard
opacity

**SH coefficients?** #flashcard
view-dependent color via spherical harmonics

**Memory?** #flashcard
millions of Gaussians each storing position, covariance (6 floats), opacity, SH coefficients (~48 floats for degree-3 SH). A scene might require 3-6M Gaussians → several GB of GPU memory. Not viable on mobile devices.

**Editing artifacts: unlike NeRF where the geometry is implicitly encoded, 3D-GS explicitly stores Gaussians. Removing or moving objects requires identifying which Gaussians belong to the object?** #flashcard
there is no semantic segmentation of Gaussians by default.

**Initialization sensitivity?** #flashcard
optimization starting from random positions diverges. The COLMAP point cloud initialization is critical; scenes without good multi-view overlap fail to reconstruct.

**RangeNet++?** #flashcard
project the LiDAR point cloud onto a 2D range image (azimuth × elevation grid), apply a 2D CNN on this image, then project predictions back to 3D. Fast, handles non-uniform density naturally. What breaks: range image distorts 3D geometry; nearby points in 3D may be far apart in range image if at different elevations.

**Cylinder3D?** #flashcard
use cylindrical voxelization (r, θ, z) instead of Cartesian (x, y, z). Near the sensor, cylindrical cells are small and fine-grained; far from the sensor, cells are larger. This naturally adapts to LiDAR's variable density pattern.

<!-- From methods/metric-learning.md -->
**Collapsed negatives: most random pairs are easy negatives already beyond the margin?** #flashcard
they contribute zero gradient and waste computation.

**Pair construction cost?** #flashcard
enumerating all valid pairs is O(N²) in dataset size.

**Margin sensitivity?** #flashcard
the right margin depends on embedding scale and is dataset-specific; wrong values make the loss uninformative.

**Easy negatives (d(a,n) >> d(a,p)): already violate no constraint?** #flashcard
zero gradient, wasted computation.

**Hard negatives (d(a,n) < d(a,p))?** #flashcard
cause large gradient but collapse embeddings early in training if encountered exclusively.

**Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin?** #flashcard
the constraint is violated but the negative is not harder than the positive. Stable, informative gradient throughout training.

**Batch composition dependency?** #flashcard
semi-hard negatives only exist if each batch contains multiple classes and multiple examples per class. Batches too small or too class-homogeneous starve the miner.

**Collapsed training: fraction_positive_triplets approaching zero signals either convergence or mode collapse?** #flashcard
monitor it.

**Slow per-epoch learning?** #flashcard
triplets enumerate 3-tuples; with N samples there are O(N³) possible triplets. Online mining from a batch samples a tiny fraction.

**Large num_classes memory?** #flashcard
the weight matrix W is (C × D). At C=1M identities and D=512, W is a 2 GB parameter. Requires distributed class sharding in very large-scale face recognition.

**Margin tuning?** #flashcard
m=0.5 is standard for faces. For fine-grained retrieval with higher intra-class variance, a smaller margin avoids over-constraining the embedding.

**<1M vectors?** #flashcard
IndexFlatL2 (exact search).

**1M–50M?** #flashcard
IndexIVFFlat with n_list ~ √N, n_probe ~ n_list/8.

**>50M?** #flashcard
IndexIVFPQ for memory compression, or IndexHNSWFlat for low latency.

<!-- From methods/nlp-advanced.md -->
**Token masking?** #flashcard
replace tokens with [MASK]

**Token deletion?** #flashcard
remove random tokens; model must infer positions

**Text infilling?** #flashcard
replace arbitrary spans with a single [MASK] token

**Sentence permutation?** #flashcard
shuffle sentence order

**Document rotation?** #flashcard
rotate to start from a random token

**num_beams=4?** #flashcard
beam search width

**length_penalty=2.0?** #flashcard
penalize short summaries (>1 favors longer)

**no_repeat_ngram_size=3?** #flashcard
prevent repetitive output

**ROUGE-1?** #flashcard
unigram overlap (content coverage)

**ROUGE-2?** #flashcard
bigram overlap (fluency, phrase preservation)

**ROUGE-L?** #flashcard
longest common subsequence (order-sensitive)

**FactCC?** #flashcard
classify each generated sentence as supported/contradicted/neutral with respect to source

**SummaC: segment-level NLI?** #flashcard
compute entailment probability between each source segment and each summary sentence; aggregate

**QAEval?** #flashcard
generate QA pairs from the reference, check if the system summary answers them correctly

**PAWS?** #flashcard
adversarial pairs constructed by word swapping + back-translation. Hard for surface-overlap models.

**QQP (Quora Question Pairs)?** #flashcard
400K question pairs, binary label

**MRPC?** #flashcard
Microsoft Research Paraphrase Corpus

**SNLI (570K pairs)?** #flashcard
crowd-sourced from image captions. Clean but narrow domain.

**MultiNLI (433K pairs)?** #flashcard
10 genres (fiction, government, telephone…). More generalizable.

**ANLI?** #flashcard
adversarially collected; much harder, three rounds of increasing difficulty.

**Stanford CoreNLP: multi-pass sieve pipeline?** #flashcard
deterministic rules first, then statistical models

**SpanBERT (Lee et al., 2018)?** #flashcard
encode all spans, score pairs end-to-end. Current gold standard.

**Extractive?** #flashcard
score and select sentences; fast, faithful, but grammatically choppy at boundaries

**Abstractive?** #flashcard
generate new text; more fluent and flexible, but can hallucinate

**BART denoising: corrupts text (masking, deletion, permutation, shuffling) and trains decoder to reconstruct?** #flashcard
gives both strong document encoding and fluent generation

**ROUGE rewards n-gram surface overlap with a reference?** #flashcard
penalizes valid paraphrases, rewards verbatim copying

**High ROUGE does not imply factual correctness or fluency?** #flashcard
High ROUGE does not imply factual correctness or fluency

**Complement with BERTScore (semantic similarity), FactCC/SummaC (faithfulness to source)?** #flashcard
Complement with BERTScore (semantic similarity), FactCC/SummaC (faithfulness to source)

**Plain BERT?** #flashcard
O(N²) cross-encoder forward passes for N-sentence comparison

**SBERT?** #flashcard
encode once → cached embeddings → cosine similarity search in O(1) at query time

**Mean pooling over token embeddings outperforms CLS-only?** #flashcard
Mean pooling over token embeddings outperforms CLS-only

**Production pattern?** #flashcard
SBERT for recall (retrieve top-K), cross-encoder for re-ranking (precision)

**NLI models (MultiNLI fine-tuned) power zero-shot classification via hypothesis templates?** #flashcard
NLI models (MultiNLI fine-tuned) power zero-shot classification via hypothesis templates

**"This text is about {label}"?** #flashcard
entailment probability is the classification score

**Also used for fact-checking, summarization faithfulness (FactCC), paraphrase detection?** #flashcard
Also used for fact-checking, summarization faithfulness (FactCC), paraphrase detection

**Three stages?** #flashcard
mention detection → pairwise scoring → clustering

**SpanBERT end-to-end is the current standard?** #flashcard
SpanBERT end-to-end is the current standard

**Critical upstream of relation extraction?** #flashcard
"He acquired it" cannot be resolved without knowing who "he" is

**Every word has one head (except root); edges are labeled (nsubj, dobj, amod…)?** #flashcard
Every word has one head (except root); edges are labeled (nsubj, dobj, amod…)

**Universal Dependencies enables cross-lingual consistency?** #flashcard
Universal Dependencies enables cross-lingual consistency

**Used upstream of relation extraction, semantic role labeling, grammar correction?** #flashcard
Used upstream of relation extraction, semantic role labeling, grammar correction

**Supervised (TACRED)?** #flashcard
fixed schema, high precision, does not generalize to unseen relations

**Distant supervision?** #flashcard
scalable but noisy; multi-instance learning helps

**LLM-based: zero-shot, open-schema, flexible?** #flashcard
best for prototyping or low-resource scenarios

**Full pipeline?** #flashcard
NER → entity pair enumeration → RE model → knowledge graph population

<!-- From methods/generative-models.md -->
**Posterior collapse?** #flashcard
the encoder ignores the input and collapses to the prior; the decoder then ignores z entirely. Occurs when the decoder is too powerful or KL weight β is too high.

**Blurry reconstructions: the reconstruction term under a Gaussian decoder is MSE, which averages over modes?** #flashcard
sharp details are penalized unless the KL pressure forces the latent to carry them.

**β-VAE trade-off?** #flashcard
setting β > 1 increases disentanglement (each z dimension encodes one independent factor) at the cost of reconstruction fidelity. There is no free lunch between the two terms.

**Vanishing gradients: when D is too strong, log(1 − D(G(z))) ≈ 0?** #flashcard
G receives no gradient. Fix: train G to maximize log D(G(z)) (non-saturating objective) rather than minimizing log(1 − D(G(z))).

**Mode collapse?** #flashcard
G learns a small subset of high-scoring outputs and ignores the rest of the data distribution. High quality but zero diversity.

**Training instability?** #flashcard
the minimax game has no unique convergence path; it oscillates or diverges under most hyperparameter settings.

**No density estimate?** #flashcard
you can sample from G but cannot evaluate p(x) for a given x.

**Mapping network?** #flashcard
z → w (8-layer MLP); moves from spherical Gaussian to a more disentangled style space.

**AdaIN style injection?** #flashcard
at each synthesis layer, apply affine transforms of w to control feature statistics.

**Weight demodulation?** #flashcard
replaces instance normalization to remove characteristic droplet artifacts.

**Slow inference: the reverse process requires T=1000 sequential network evaluations per sample?** #flashcard
hundreds of forward passes for one image.

**No explicit latent structure?** #flashcard
unlike VAEs, there is no interpretable compressed representation; the "latent" is the full-resolution noisy image at step T.

**Exposure bias: the network is trained with ground-truth x_t but at inference uses its own predictions as inputs?** #flashcard
accumulated errors compound over 1000 steps.

**Quality degrades sharply below ~20 DDIM steps because the ODE approximation accumulates error with large step sizes. Higher-order ODE solvers (DPM-Solver++) recover some quality.?** #flashcard
Quality degrades sharply below ~20 DDIM steps because the ODE approximation accumulates error with large step sizes. Higher-order ODE solvers (DPM-Solver++) recover some quality.

**Determinism is a double edge: reproducibility is useful, but all diversity comes from the initial noise?** #flashcard
the model cannot inject stochasticity mid-trajectory to recover from early errors.

**Manifold drift: high guidance scale pushes samples toward regions of high conditional likelihood that may be out-of-distribution for the denoiser?** #flashcard
oversaturated colors, distorted anatomy.

**Diversity collapse?** #flashcard
at w >> 1, the denoising process converges to a few high-probability modes of p(x|y), sacrificing sample diversity.

**Inference cost: requires two forward passes per step (conditional + unconditional). FLUX-schnell solves this via guidance distillation?** #flashcard
a single unconditional model trained to match CFG output.

**VAE bottleneck quality?** #flashcard
any detail the VAE discards during encoding cannot be recovered by diffusion. The VAE's reconstruction quality sets a hard ceiling on output fidelity.

**KL regularization vs. reconstruction?** #flashcard
the KL penalty needed to make latents well-behaved for diffusion also forces the VAE to lose some high-frequency detail. SDXL uses a larger channel count (4→16) to partially mitigate this.

**Cross-attention alignment?** #flashcard
CLIP text embeddings may not capture all semantic detail in a long or complex prompt. SDXL concatenates two CLIP encoders (ViT-L and OpenCLIP ViT-G) to increase text representational capacity.

**Straight paths in expectation, not per-sample: the optimal transport pairing of noise and data is not straight lines between arbitrary pairs. If you pair x_0 and x_1 randomly, the learned velocity field averages over many crossing trajectories?** #flashcard
it becomes curved in practice. Reflow (iterating the procedure) straightens the paths further.

**ODE solver sensitivity?** #flashcard
with so few steps, step size matters more than in diffusion; an inappropriate step size or solver can produce visible artifacts.

**No spatial inductive bias?** #flashcard
unlike U-Nets, transformers have no built-in notion that nearby pixels are related. This is eventually learned from data, but requires more training to converge on small datasets.

**Quadratic attention cost?** #flashcard
global self-attention over 64 tokens is fine at 512×512, but scaling to 1024×1024 with smaller patches makes the attention prohibitive. Efficient attention variants are required.

**U-Net forward pass?** #flashcard
~35 ms/step

**VAE decode?** #flashcard
~150 ms (once at the end)

**At 20 steps?** #flashcard
35×20 + 150 = 850 ms end-to-end

<!-- From methods/dynamic-graphs.md -->
**Social networks?** #flashcard
friend/follow links appear and disappear

**Financial transactions?** #flashcard
detect fraud by modeling interaction bursts

**Traffic?** #flashcard
road-segment speeds change every minute

**Molecular dynamics?** #flashcard
atom bonds break and form during simulation

**Knowledge graphs?** #flashcard
facts have validity intervals

**EvolveGCN-H: uses the node embedding matrix H_{t-1} as input to the GRU?** #flashcard
better when node features are informative.

**EvolveGCN-O: uses only W_{t-1}?** #flashcard
handles graphs where the node set changes completely.

**O(n²) decoding: the inner-product decoder evaluates all node pairs?** #flashcard
infeasible for large graphs.

**No validity guarantee?** #flashcard
nothing prevents the decoder from producing chemically invalid graphs.

**No global graph structure?** #flashcard
the latent space has one vector per node; there is no graph-level variable that captures global properties (molecular weight, ring count, connectivity).

**Candidate block B_t = {v_{k+1}, ..., v_{k+b}}?** #flashcard
Candidate block B_t = {v_{k+1}, ..., v_{k+b}}

**Run GAT over current graph + candidates → score edges?** #flashcard
Run GAT over current graph + candidates → score edges

**Sample edges between B_t and existing nodes?** #flashcard
Sample edges between B_t and existing nodes

**Add B_t to graph, repeat?** #flashcard
Add B_t to graph, repeat

**Each atom satisfies its valency (Carbon?** #flashcard
degree ≤ 4, Nitrogen ≤ 3, etc.)

**The graph is connected?** #flashcard
The graph is connected

**No invalid bond types?** #flashcard
No invalid bond types

<!-- From methods/time-series.md -->
**Data leakage?** #flashcard
training on 2023 data to predict 2022. The model learns a perfectly calibrated cheat.

**Inflated confidence intervals?** #flashcard
assuming independence when adjacent observations are correlated inflates effective sample size.

**Leaked features?** #flashcard
computing rolling means on the full dataset before splitting injects future signal into past features.

**Additive trend?** #flashcard
y_t = α + βt + ε_t. Fluctuations have constant magnitude.

**Multiplicative trend?** #flashcard
y_t = α · e^{βt} · ε_t. Fluctuations scale with level. Apply log(y_t) to convert to additive.

**First-order differencing Δy_t = y_t − y_{t-1} removes a linear trend.?** #flashcard
First-order differencing Δy_t = y_t − y_{t-1} removes a linear trend.

**Augmented Dickey-Fuller (ADF)?** #flashcard
null = unit root (non-stationary). Reject at p < 0.05.

**KPSS?** #flashcard
null = stationary. Both tests together disambiguate.

**First differencing Δy_t = y_t − y_{t-1}?** #flashcard
removes linear trend

**Seasonal differencing y_t − y_{t-s}?** #flashcard
removes seasonal structure

**Log transform?** #flashcard
stabilizes exponentially growing variance

**Box-Cox?** #flashcard
general variance stabilization

**ACF (Autocorrelation Function)?** #flashcard
correlation between y_t and y_{t-k} for each lag k

**PACF (Partial Autocorrelation Function)?** #flashcard
correlation after removing effects of intermediate lags

**PACF cuts off sharply at lag p, ACF decays slowly → AR(p) process?** #flashcard
PACF cuts off sharply at lag p, ACF decays slowly → AR(p) process

**ACF cuts off at lag q, PACF decays slowly → MA(q) process?** #flashcard
ACF cuts off at lag q, PACF decays slowly → MA(q) process

**Both decay slowly → ARMA(p, q) or need differencing?** #flashcard
Both decay slowly → ARMA(p, q) or need differencing

**d?** #flashcard
differencing order to achieve stationarity (read from ADF)

**p: AR terms?** #flashcard
PACF cuts off at lag p

**q: MA terms?** #flashcard
ACF cuts off at lag q

**(P, D, Q)?** #flashcard
seasonal AR order, seasonal differencing, seasonal MA order

**[S]?** #flashcard
seasonal period (12 for monthly, 4 for quarterly, 7 for daily-with-weekly)

**Trend?** #flashcard
piecewise linear or logistic growth with automatic changepoint detection

**Seasonality?** #flashcard
Fourier series approximation at yearly, weekly, and daily periods

**Holidays?** #flashcard
user-supplied date ranges with learned offsets

**Holidays: binary or categorical flag?** #flashcard
critical for retail and energy

**Weather?** #flashcard
temperature is the dominant predictor for energy demand

**Known future values?** #flashcard
promotions, scheduled events (these are exogenous variables)

**Input normalization?** #flashcard
scale features to zero mean, unit variance using training statistics only

**Stacking?** #flashcard
2-3 layers usually sufficient; more causes overfitting

**Lookback window?** #flashcard
start with 2× the longest seasonal period; tune via validation

**Gradient clipping?** #flashcard
clip gradients to norm 1.0 (torch.nn.utils.clip_grad_norm_)

**Dropout?** #flashcard
between LSTM layers, not within (use nn.Dropout after LSTM)

**Layer 1 (d=1)?** #flashcard
sees 2 steps back

**Layer 2 (d=2)?** #flashcard
sees 4 steps back

**Layer 3 (d=4)?** #flashcard
sees 8 steps back

**Layer 4 (d=8)?** #flashcard
sees 16 steps back

**Distilling?** #flashcard
halve sequence length between encoder layers via max pooling

**Generative decoder?** #flashcard
predicts the full forecast horizon in one forward pass (not autoregressive)

**Trend stack?** #flashcard
polynomial basis [1, t, t², t³]

**Seasonality stack?** #flashcard
Fourier basis [cos(2πkt/P), sin(2πkt/P)]

**Patch tokenization?** #flashcard
patches of consecutive timesteps as tokens

**Distributional output?** #flashcard
Student-t distribution over future values

**Trained on?** #flashcard
large Chronos/Lotsa dataset collection

**Sudden drift?** #flashcard
COVID lockdowns changing energy demand overnight

**Gradual drift?** #flashcard
customer behavior shifting over months

**Recurring drift?** #flashcard
seasonal patterns returning each year (expected and handled by seasonal features)

**E-commerce demand?** #flashcard
days to weeks

**Energy demand?** #flashcard
weeks to months

**Financial markets?** #flashcard
hours to days

<!-- From methods/computer-vision.md -->
**ReLU: f(x) = max(0, x). Gradient is exactly 1 for positive inputs?** #flashcard
no vanishing. Cheap to compute.

**Leaky ReLU?** #flashcard
f(x) = max(alpha*x, x), alpha ≈ 0.01. Prevents dead neurons that permanently output zero.

**GELU?** #flashcard
f(x) = x * Phi(x) where Phi is the standard normal CDF. Smooth approximation to ReLU; used in Transformers and modern CNNs.

<!-- From components/rnn-lstm-gru.md -->
**$h_t \in \mathbb{R}^{d_h}$?** #flashcard
hidden state at time $t$

**$x_t \in \mathbb{R}^{d_x}$?** #flashcard
input at time $t$

**$W_h \in \mathbb{R}^{d_h \times d_h}$, $W_x \in \mathbb{R}^{d_h \times d_x}$?** #flashcard
shared across all timesteps

**If $|\lambda_{\max}| < 1$: gradients shrink geometrically?** #flashcard
vanishing gradients. Early timesteps receive near-zero gradient signal, so the model cannot learn long-range dependencies.

**If $|\lambda_{\max}| > 1$: gradients grow geometrically?** #flashcard
exploding gradients. Training becomes numerically unstable, with loss spiking to NaN.

**Sequence labeling (NER, POS tagging)?** #flashcard
each label benefits from full context

**Sentence classification?** #flashcard
the final representation captures both directions

**Machine translation encoder?** #flashcard
the encoder can be bidirectional; only the decoder must be causal

**Online / streaming inference where input arrives one token at a time and latency matters?** #flashcard
Online / streaming inference where input arrives one token at a time and latency matters

**Extremely long sequences where $O(n^2)$ attention is prohibitive and the full context isn't needed?** #flashcard
Extremely long sequences where $O(n^2)$ attention is prohibitive and the full context isn't needed

**Memory-constrained deployment (edge/embedded devices)?** #flashcard
Memory-constrained deployment (edge/embedded devices)

**Time series with strong local temporal structure?** #flashcard
Time series with strong local temporal structure

**Can be computed as a convolution during training (parallel like Transformers)?** #flashcard
Can be computed as a convolution during training (parallel like Transformers)

**Can be computed as a recurrence during inference (constant memory like RNNs)?** #flashcard
Can be computed as a recurrence during inference (constant memory like RNNs)

**Linear in sequence length?** #flashcard
$O(n \cdot d)$

**$W_x$?** #flashcard
Xavier/Glorot initialization (nn.init.xavier_uniform_)

**$W_h$?** #flashcard
Orthogonal initialization to avoid eigenvalue scaling issues at the start

**Forget gate bias?** #flashcard
initialize to 1.0 in LSTMs (b_f = 1.0)

**Output projection?** #flashcard
small normal initialization

<!-- From components/scaling-laws-and-chinchilla.md -->
**$N_{opt} \propto C^{0.45} \approx C^{0.5}$?** #flashcard
$N_{opt} \propto C^{0.45} \approx C^{0.5}$

**$D_{opt} \propto C^{0.55} \approx C^{0.5}$?** #flashcard
$D_{opt} \propto C^{0.55} \approx C^{0.5}$

**C4 heuristic?** #flashcard
remove docs with <3 sentences

**Perplexity filtering?** #flashcard
train small LM, keep moderate-perplexity docs

**Deduplication?** #flashcard
MinHash at document and ngram level

**Few-shot learning (not present at 1B)?** #flashcard
Few-shot learning (not present at 1B)

**Chain-of-thought reasoning (not present at 7B)?** #flashcard
Chain-of-thought reasoning (not present at 7B)

**Code generation?** #flashcard
Code generation

**N = 175B, D = 300B tokens?** #flashcard
N = 175B, D = 300B tokens

**C = 6 × 175B × 300B = 315 × 10²¹ FLOPs ≈ 3.1 × 10²³ FLOPs?** #flashcard
C = 6 × 175B × 300B = 315 × 10²¹ FLOPs ≈ 3.1 × 10²³ FLOPs

**At 312 TFLOP/s on V100 FP16?** #flashcard
≈ 10⁹ GPU-seconds ≈ 32 GPU-years

<!-- From components/distributed-training-and-parallelism.md -->
**Model weights (bf16)?** #flashcard
140 GB

**Adam optimizer states (fp32 master weights + m + v)?** #flashcard
840 GB

**Gradients (bf16)?** #flashcard
140 GB

**Activations (for a 4K seq batch)?** #flashcard
~80 GB

**Total: >1.2 TB?** #flashcard
requires ~15× A100 80GB GPUs minimum

**Stage 1?** #flashcard
all-reduce on gradients (same as DP)

**Stage 2?** #flashcard
reduce-scatter on gradients (slightly better than all-reduce)

**Stage 3?** #flashcard
all-gather before each layer's forward, reduce-scatter after each layer's backward

**TP degree = GPUs per node (typically 8)?** #flashcard
keep within NVLink domain

**PP degree = number of nodes if model is very large?** #flashcard
PP degree = number of nodes if model is very large

**DP = fill remaining GPU budget?** #flashcard
DP = fill remaining GPU budget

<!-- From components/regularization.md -->
**$\alpha = 0.2$: $\lambda$ mostly near 0 or 1?** #flashcard
nearly clean examples with mild mixing

**$\alpha = 1.0$: $\lambda \sim \text{Uniform}(0,1)$?** #flashcard
more aggressive blending

<!-- From components/backpropagation.md -->
**$z_1 = w_1 x = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$?** #flashcard
$z_1 = w_1 x = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$

**$z_2 = w_2 a_1 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$?** #flashcard
$z_2 = w_2 a_1 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$

**$L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$?** #flashcard
$L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$

**$\partial L / \partial \hat{y} = 2(\hat{y} - y) = 1.244$?** #flashcard
$\partial L / \partial \hat{y} = 2(\hat{y} - y) = 1.244$

**$\partial \hat{y} / \partial z_2 = \sigma'(z_2) = 0.622 \cdot 0.378 \approx 0.235$?** #flashcard
$\partial \hat{y} / \partial z_2 = \sigma'(z_2) = 0.622 \cdot 0.378 \approx 0.235$

**$\partial L / \partial w_2 = 1.244 \times 0.235 \times a_1 = 1.244 \times 0.235 \times 0.622 \approx 0.182$?** #flashcard
$\partial L / \partial w_2 = 1.244 \times 0.235 \times a_1 = 1.244 \times 0.235 \times 0.622 \approx 0.182$

**Continue?** #flashcard
$\partial L / \partial a_1 = 1.244 \times 0.235 \times w_2$, then multiply by $\sigma'(z_1)$ and $x$ to get $\partial L / \partial w_1$

**ReLU-family activations?** #flashcard
local derivative is 1 for positive activations

**Residual connections: $x_{l+1} = x_l + F(x_l)$?** #flashcard
gradient flows directly through the skip path, bypassing the activation's derivative entirely

**LayerNorm / BatchNorm?** #flashcard
prevents activations from drifting into saturation regions

**He initialization?** #flashcard
sets weight variance so signal stays at consistent scale at layer initialization

**Forget zero_grad()?** #flashcard
gradients accumulate across batches. Updates grow unboundedly. Loss spikes.

**Detach a tensor mid-computation?** #flashcard
the computation graph is cut. Gradients do not flow through the detached node. Weights upstream of the detach never update.

**Compute the loss, then modify it outside the graph?** #flashcard
the modified loss has no connection to the original graph. backward() computes gradients for the wrong thing.

<!-- From components/normalization.md -->
**$G = 1$?** #flashcard
equivalent to LayerNorm (over all channels)

**$G = C$?** #flashcard
equivalent to InstanceNorm (one channel per group)

**$G = 32$?** #flashcard
typical setting in detection/segmentation networks

<!-- From components/instruction-tuning-and-alignment.md -->
**Mask loss on user/system tokens (we don't optimize for predicting those)?** #flashcard
Mask loss on user/system tokens (we don't optimize for predicting those)

**Data quality >> data quantity?** #flashcard
1000 carefully written examples beats 100K low-quality

**InstructGPT used ~13K demonstrations (FLAN used millions but lower quality)?** #flashcard
InstructGPT used ~13K demonstrations (FLAN used millions but lower quality)

**β too small → reward hacking (exploit RM)?** #flashcard
β too small → reward hacking (exploit RM)

**β too large → policy barely moves from SFT (alignment wasted)?** #flashcard
β too large → policy barely moves from SFT (alignment wasted)

**Typical range?** #flashcard
β = 0.02–0.1

**No reward model needed?** #flashcard
reward is implicit in policy ratio

**No RL training loop (no advantage estimation, no critic)?** #flashcard
No RL training loop (no advantage estimation, no critic)

**Stable supervised training?** #flashcard
Stable supervised training

**Just two models needed (policy + frozen ref), not four?** #flashcard
Just two models needed (policy + frozen ref), not four

**"Is the response harmful, unethical, or deceptive?"?** #flashcard
"Is the response harmful, unethical, or deceptive?"

**"Does the response prioritize being helpful while avoiding harms?"?** #flashcard
"Does the response prioritize being helpful while avoiding harms?"

<!-- From components/loss-functions.md -->
**VAE loss?** #flashcard
penalize encoder distribution from deviating from Gaussian prior

**Knowledge distillation?** #flashcard
train student to match teacher's output distribution

**RLHF?** #flashcard
KL penalty keeping policy close to reference model

<!-- From components/transformers.md -->
**MQA (Multi-Query Attention)?** #flashcard
all heads share one $K$, $V$ projection. Cache divided by $n_\text{heads}$.

**GQA (Grouped-Query Attention)?** #flashcard
$g$ heads share one $K$, $V$ pair. Cache reduced by $n_\text{heads}/g$. Quality-memory tradeoff between MQA and full MHA. Used in LLaMA 3.

<!-- From components/quantization-pruning-detailed.md -->
**4-bit integers mapped to quantiles of normal distribution (not uniform)?** #flashcard
4-bit integers mapped to quantiles of normal distribution (not uniform)

**Better preserves weight distribution than uniform int4?** #flashcard
Better preserves weight distribution than uniform int4

**Base model (4-bit)?** #flashcard
~35GB

**Gradients + optimizer (only LoRA params)?** #flashcard
~1GB

**Activations?** #flashcard
~6GB

**Total: ~42GB?** #flashcard
fits on single A100 80GB

**Unstructured?** #flashcard
remove individual weights → sparse matrix → hard to accelerate (no hardware support for arbitrary sparsity)

**Structured?** #flashcard
remove entire neurons/heads/layers → dense smaller model → hardware-friendly

<!-- From components/optimisers.md -->
**Full-batch gradient descent?** #flashcard
use all data to compute gradient. Low variance, accurate gradient, but prohibitively slow for large datasets.

**SGD (stochastic gradient descent): use one example. Very noisy gradient?** #flashcard
updates jump around. Fast, but oscillates.

**Mini-batch SGD?** #flashcard
use 32–512 examples. Practical default: fast GPU parallelism, noisy enough to escape local minima, stable enough to converge.

<!-- From components/attention.md -->
**$Q \in \mathbb{R}^{n \times d_k}$?** #flashcard
queries (what each token is looking for)

**$K \in \mathbb{R}^{m \times d_k}$?** #flashcard
keys (what each token advertises)

**$V \in \mathbb{R}^{m \times d_v}$?** #flashcard
values (what each token actually returns)

**Output: $\mathbb{R}^{n \times d_v}$?** #flashcard
for each query, a weighted blend of values

**Encoder-decoder Transformers (decoder attends to encoder output)?** #flashcard
Encoder-decoder Transformers (decoder attends to encoder output)

**Multimodal models (text tokens attending to image patch embeddings)?** #flashcard
Multimodal models (text tokens attending to image patch embeddings)

**Retrieval-augmented generation (query tokens attending to retrieved document tokens)?** #flashcard
Retrieval-augmented generation (query tokens attending to retrieved document tokens)

**MQA (Multi-Query Attention)?** #flashcard
all heads share a single $K$, $V$ projection. Cache size divided by $n_\text{heads}$. Some quality loss.

**GQA (Grouped-Query Attention)?** #flashcard
groups of $g$ heads share one $K$, $V$ pair. Cache reduced by factor $n_\text{heads}/g$. Better quality-memory tradeoff than MQA. Used in LLaMA 3.

<!-- From components/hidden-layers.md -->
**Autoencoders (compression by design)?** #flashcard
Autoencoders (compression by design)

**ResNet bottleneck blocks (1×1 convolutions to reduce channels, then 3×3, then expand)?** #flashcard
ResNet bottleneck blocks (1×1 convolutions to reduce channels, then 3×3, then expand)

**Transformer FFN (2 linear layers with a 4× width expansion in the middle)?** #flashcard
Transformer FFN (2 linear layers with a 4× width expansion in the middle)

<!-- From components/model-compression.md -->
**FP16?** #flashcard
2× memory reduction, nearly zero accuracy loss. Start here.

**INT8?** #flashcard
4× reduction, $<1\%$ accuracy drop on most tasks with good calibration. Significant speed gains on CPUs and accelerators.

**INT4: 8× reduction. More aggressive?** #flashcard
accuracy drops more, requires careful calibration.
