# Model Compression

You trained a model that works beautifully on a 8-GPU cluster.

Now someone wants it to run on a phone, inside a browser, or with a latency budget of 20ms.

This file is about how to make that happen without burning everything down.

---

# 1. Why Model Compression Exists

There is a real gap between "this works in the lab" and "this ships to a billion users."

That gap is mostly about:

- memory footprint
- inference speed
- energy consumption
- hardware availability

## The Deployment Reality

Most impressive models do not live where they were trained.

**Edge devices**

A smartphone GPU has maybe 4GB of shared memory.
A microcontroller might have 512KB.
A model like GPT-4 would simply not load.

**Latency budgets**

Recommendation systems have < 50ms SLAs.
Voice assistants need to feel real-time.
Autocomplete feels broken at 500ms.

**Cost**

Running a large model for millions of inference requests per day is expensive.
At scale, even shaving 30% off compute can save tens of thousands of dollars monthly.

**The practical interview answer**

Model compression is about making models smaller and faster without catastrophically hurting accuracy.

It is not about making models worse.
It is about making them good enough for the actual deployment target.

---

# 2. Quantization

The core idea: numbers in a neural network are usually stored as 32-bit floats.

What if you used fewer bits?

Most computations do not actually need that precision.
A weight of 0.314159 and 0.314 behave nearly identically during inference.

---

## 2.1 Post-Training Quantization (PTQ)

You already have a trained model.
You convert the weights (and sometimes activations) to lower precision without retraining.

**FP16 (16-bit float)**

Half precision.
Most modern GPUs and accelerators handle this natively.
Almost no accuracy loss.
2x memory reduction.

This is the baseline.
If you are not running FP16, you should probably start here.

**INT8 (8-bit integer)**

Weights get mapped from a float range to integers in [-128, 127].

You need to know the range of values to do this mapping.
That is where calibration comes in (more on this below).

Typical accuracy drop: < 1% on most tasks if done well.
Speed gain: significant on CPUs and accelerators with INT8 support.

**INT4 (4-bit integer)**

Push the quantization further.
More aggressive, more savings, more care needed.

Used heavily in LLM deployment where memory is the bottleneck.

---

## 2.2 Quantization-Aware Training (QAT)

PTQ works well but has a ceiling.
At very aggressive quantization (INT4, binary), PTQ tends to degrade too much.

QAT simulates quantization during training.

**How it works**

During the forward pass, fake quantization nodes are inserted.
Weights and activations are rounded to simulate low-precision arithmetic.
But gradients still flow through at full precision.

This lets the model learn to be robust to quantization errors.

**The payoff**

Models trained with QAT can be quantized to INT4 or even lower with much less accuracy loss than PTQ.

**The cost**

You need to retrain or fine-tune the model.
That costs time, compute, and engineering effort.

---

## 2.3 GPTQ and AWQ for LLMs

Standard INT8/INT4 PTQ struggles with LLMs because:
- weights have very skewed distributions
- outlier values in activations are critical

Two methods became popular:

**GPTQ (Generalization of OBD)**

Uses second-order information (Hessian approximation) to determine which weights can be quantized aggressively and which matter more.

Quantizes layer by layer.
Compensates for quantization error in each weight by adjusting remaining weights.

Result: LLMs quantized to INT4 with surprisingly small accuracy loss.
Used heavily to get 70B models running on consumer hardware.

**AWQ (Activation-aware Weight Quantization)**

Observation: not all weights are equally important.
Weights that multiply large activations matter more.

AWQ scales channels based on activation magnitudes before quantizing.
Protects important weights from quantization damage.

Tends to outperform GPTQ on certain architectures.
Also faster to apply.

**Practical note**

If someone asks "how would you run Llama-3 70B on a single GPU" in an interview, GPTQ/AWQ quantization to 4-bit is the answer.

---

## 2.4 Calibration Datasets

PTQ needs to know the range of activations to set quantization parameters.

You pass a small representative dataset through the model, observe the activation distributions, and use that to set the scale and zero-point for each layer.

**What makes a good calibration set**

- Representative of real inputs (not random noise)
- Small: usually 128–1024 samples is enough
- Diverse: covers different modes of the data

**Why it matters**

Bad calibration = wrong quantization ranges = more accuracy loss.
Good calibration = the model behaves as if the precision drop never happened.

---

# 3. Knowledge Distillation

Quantization compresses numbers.
Distillation compresses knowledge.

---

## 3.1 The Teacher-Student Paradigm

**The analogy**

Think of an expert surgeon who has seen ten thousand cases.
They cannot clone themselves.
But they can train a resident.

The resident learns not just from textbooks (ground truth labels) but from watching the surgeon:
- how confident they were
- where they hesitated
- what alternatives they considered

That is knowledge distillation.

The large model (teacher) trains a smaller model (student) not just to match final predictions, but to mimic the teacher's internal reasoning expressed through its output distributions.

---

## 3.2 Soft Labels vs Hard Labels

**Hard labels**

The ground truth: cat = 1, dog = 0, bird = 0.

This is all-or-nothing.
The model gets no information about which wrong classes were close.

**Soft labels**

The teacher's output distribution: cat = 0.78, dog = 0.15, bird = 0.07.

This encodes much richer information:
- this image looked a bit like a dog too
- the model was uncertain, not confident
- there is structure in the error space

The student model trained on soft labels generalizes better because it learns the relationships between classes, not just the correct answer.

**Temperature scaling**

To make the teacher's output distribution softer (less peaked), you divide the logits by a temperature T before softmax.

At T = 1: normal distribution.
At T > 1: flatter distribution, more information in the tails.
At T → ∞: uniform distribution, no information.

The original distillation paper (Hinton et al., 2015) used T = 3 to 5.

**The loss function**

```
L = α * L_CE(student, hard_labels) + (1 - α) * L_KL(student, teacher_soft_labels)
```

Where:
- α controls how much you weigh ground truth vs teacher guidance
- L_KL is KL divergence between student and teacher distributions
- both are evaluated at temperature T (and then rescaled back)

---

## 3.3 Intermediate Layer Distillation

Matching only the final output ignores a lot of structure inside the teacher.

**Hint layers (FitNets)**

The intermediate feature maps of the teacher carry rich representations.
Force the student to also produce similar intermediate representations.

```
L_hint = ||f_student(x) - W * f_teacher(x)||^2
```

Where W is a learned projection (since teacher and student may have different hidden sizes).

**Attention transfer**

Transfer the attention maps layer by layer.
The student learns not just what the teacher outputs, but where it looks.

This is particularly useful for transformers: distilling attention distributions layer by layer gives the student a structural understanding of how to process sequences.

---

## 3.4 DistilBERT and TinyBERT

**DistilBERT**

HuggingFace's classic distilled BERT.
6 layers instead of 12.
40% smaller, 60% faster, retains ~97% of BERT's performance on GLUE.

Training approach:
- cosine embedding loss on hidden states
- MLM loss on student
- attention map transfer from teacher layers to student layers

**TinyBERT**

Goes further than DistilBERT.
Uses comprehensive distillation across:
- embedding layers
- transformer layers (attention + hidden states)
- prediction layer

Two-stage process:
1. General distillation on large corpus
2. Task-specific distillation on fine-tuned teacher

Result: 4-layer model that approaches BERT performance on many tasks.

---

## 3.5 Self-Distillation

No separate teacher.
The same model teaches itself.

**Born-Again Networks**

Train a model. Use it as a teacher to train an identical-size model (student).
Repeat this process (generation 1 teacher → generation 1 student → generation 2 student).

Surprisingly, the final model outperforms the original.

**Why this works**

Soft labels from the model itself act as a form of label smoothing.
They encode uncertainty that the hard labels did not.

**In modern training pipelines**

Self-distillation is used to regularize models during fine-tuning.
The model's own predictions from an earlier checkpoint guide training.

---

# 4. Pruning

If quantization compresses numbers and distillation compresses knowledge, pruning compresses structure.

You identify parts of the network that contribute little and remove them.

---

## 4.1 Unstructured vs Structured Pruning

**Unstructured pruning**

Remove individual weights.
Set small weights to zero.
The matrix becomes sparse.

Result: theoretical parameter reduction, but irregular sparsity is hard to accelerate on standard hardware.

To get real speed benefits from unstructured pruning, you need hardware that supports sparse computation (like NVIDIA's Ampere sparse tensor cores) or specific sparse inference libraries.

**Structured pruning**

Remove entire neurons, attention heads, or convolutional filters.

The model is literally smaller after pruning.
The weight matrices get smaller dimensions.
This directly reduces compute without needing special sparse hardware.

**Practical advice**

If you cannot control the hardware, prefer structured pruning.
If you are deploying on specialized hardware with sparsity support, unstructured pruning can be more flexible.

---

## 4.2 Magnitude-Based Pruning

The simplest approach: prune weights with the smallest absolute values.

**Intuition**

Small weights contribute little to the output.
Removing them should have minimal impact.

**Algorithm**

1. Train the model fully.
2. Compute a global (or per-layer) threshold.
3. Set all weights below the threshold to zero.
4. Fine-tune the pruned model to recover accuracy.

**Global vs local thresholds**

Global: pick a single threshold for the whole network.
Local: pick a threshold per layer, pruning a fixed fraction per layer.

Global thresholds can be too aggressive on some layers and too conservative on others.
Local thresholds give more control but require tuning.

---

## 4.3 Movement Pruning

Magnitude pruning looks at where weights are.
Movement pruning looks at where weights are going.

**Intuition**

A weight that starts at 0.1 and is moving toward zero is more disposable than a weight that starts at 0.5 and is moving toward 1.0.

Movement pruning scores weights based on their gradient-weighted sign: are they moving away from zero or toward it?

**Why this matters for fine-tuning**

When you fine-tune a pretrained model, weights that were important at pretraining may become less important.
Movement pruning adapts to the task during training.

This is particularly useful for NLP, where BERT-style models start with all weights "large" and fine-tuning makes only a subset relevant to the task.

---

## 4.4 The Lottery Ticket Hypothesis

One of the most cited ideas in recent pruning literature.

**The hypothesis (Frankle & Carlin, 2019)**

Large networks contain small subnetworks ("winning tickets") that, when trained from the same initial weights, converge faster and reach similar or better accuracy than the full network.

**The analogy**

A lottery has many tickets.
Most tickets lose.
A few tickets had the winning numbers from the start.

Training a large network = buying many tickets.
The winner was already in there; you just had to train to find it.

**How to find the winning ticket**

1. Train the full network until convergence.
2. Prune the smallest magnitude weights (say, 20–50%).
3. Reset the remaining weights to their original initialization.
4. Retrain from scratch with this sparse mask.

Repeat the iterative pruning process.

The result is often a subnetwork that trains well and performs comparably to the full model.

**Practical limitations**

The original paper was for small models and datasets.
Scaling this to LLMs is expensive.
But the theoretical insight influenced how we think about overparameterization and training dynamics.

---

## 4.5 Gradual Pruning Schedules

Pruning everything at once is risky.
The model's accuracy collapses and may not recover.

**Gradual pruning**

Start with a low sparsity target.
Increase it over training steps following a schedule.

Common schedule (cubic):

```
s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (n * Δt))^3
```

Where:
- s_i: initial sparsity (usually 0)
- s_f: final sparsity target (e.g., 90%)
- t_0: pruning start step
- n: number of pruning steps
- Δt: frequency of pruning updates

The model continuously adapts to increasing sparsity, rather than experiencing a sudden shock.

**Interleaving pruning with fine-tuning**

Prune → fine-tune a bit → prune more → fine-tune more.

This iterative approach is more expensive but recovers more accuracy at high sparsity levels.

---

# 5. Low-Rank Factorization

Every large weight matrix is expensive.
But many weight matrices are approximately low-rank.

**The idea**

If a matrix W of shape (m, n) has effective rank r << min(m, n), you can approximate it as:

```
W ≈ A * B   where A is (m, r) and B is (r, n)
```

Parameter count: m*n → m*r + r*n = r*(m+n)

When r is small, this is a massive reduction.

**LoRA (Low-Rank Adaptation)**

The famous fine-tuning technique is actually a low-rank factorization applied to weight updates during fine-tuning.

Instead of updating the full W (which is frozen), you learn:
```
ΔW = A * B
```

Where A and B are small matrices.
This is not model compression per se, but the same math is used in compression contexts.

**SVD-based compression**

Take the full weight matrix.
Compute the Singular Value Decomposition.
Keep only the top r singular values and their corresponding vectors.

The matrix is approximately reconstructed with much fewer parameters.

**When to use it**

Low-rank factorization works best when:
- weight matrices are large (wide or tall)
- the actual information in the matrix is low-dimensional
- the task does not need fine-grained distinctions in the compressed space

Attention projection matrices in transformers (Q, K, V projections) are common targets.

---

# 6. Neural Architecture Search (NAS)

All the previous techniques compress an existing model.

NAS asks: what if we designed a smaller model from scratch that is efficient by construction?

**The core idea**

Search over a space of architectures to find one that maximizes accuracy subject to hardware constraints (latency, memory, FLOPs).

**Methods**

**Differentiable NAS (DARTS)**

Relax the architecture search into a continuous optimization problem.
Architecture decisions become soft weights that are jointly trained with network weights.
After training, discretize the architecture choices.

**Evolutionary / Reinforcement Learning search**

Use RL or evolutionary algorithms to propose candidate architectures.
Train each candidate.
Use accuracy + efficiency as reward.

This is expensive but has found very competitive architectures (NASNet, EfficientNet).

**Efficiency-focused NAS**

MobileNet, EfficientNet, and MNASNet were all found or tuned via NAS with hardware-in-the-loop.

**Practical note for interviews**

NAS is often mentioned as a compression technique but is fundamentally different: you are searching for a better architecture, not compressing an existing one.

It is more expensive upfront but the resulting model may need less post-hoc compression.

---

# 7. Hardware-Aware Optimization

All the above techniques change the model.
Hardware optimization changes how the model runs.

---

## 7.1 ONNX

ONNX (Open Neural Network Exchange) is an open format for representing ML models.

**Why it matters**

Most models are trained in PyTorch or TensorFlow.
The target hardware may have a specialized runtime (TensorRT, CoreML, etc.).

ONNX is the translation layer.

```
PyTorch model → ONNX export → ONNX Runtime or other backend
```

ONNX Runtime can apply its own graph optimizations:
- operator fusion (fuse matmul + bias + activation into one kernel)
- constant folding (precompute static parts of the graph)
- memory planning (reduce peak memory)

---

## 7.2 TensorRT

NVIDIA's inference optimizer.

**What it does**

- Fuses layers into single CUDA kernels
- Applies INT8/FP16 quantization with hardware calibration
- Optimizes memory usage patterns
- Profiles against the actual GPU to select the fastest kernel implementations

**Typical gains**

3–10x inference speedup over raw PyTorch on NVIDIA hardware.

The optimization is hardware-specific: a TensorRT engine built for A100 will not run on a T4.

---

## 7.3 CoreML

Apple's deployment framework for iOS/macOS.

**What it does**

- Optimizes models for Apple Neural Engine (ANE), GPU, and CPU
- Handles FP16 conversion automatically
- Integrates with Xcode

If you are deploying anything on iPhone, CoreML is the standard path.

Models converted via `coremltools` from PyTorch or TensorFlow run with hardware-specific scheduling that no generic runtime can match on Apple silicon.

---

# 8. Speculative Decoding for LLMs

This one is specific to autoregressive language models.

**The problem**

LLM inference is slow because:
- Each token requires a full forward pass through all layers
- Tokens are generated one at a time (sequential dependency)
- The model is memory-bandwidth bound, not compute-bound

**The idea**

Use a small, fast draft model to propose multiple tokens at once.
Have the large model verify all of them in parallel with one forward pass.
Accept the tokens that match what the large model would have generated.

**Why this works**

Verification is cheaper than generation for the large model.
When multiple tokens are fed in together, the large model processes them in parallel.
Rejected tokens are re-generated, but in practice the acceptance rate is high.

**Typical speedup**

2–3x throughput improvement on greedy or temperature-sampling decoding.

**The analogy**

A junior employee drafts an email.
The manager reviews it quickly and accepts most of it.
Occasionally the manager rewrites a sentence.

The manager's total time spent is much less than if they had written the whole email themselves.

**Practical details**

- Draft model is typically 7–10x smaller than the target model
- Both models must share the same tokenizer and vocabulary
- Works best when the distribution of the draft and target models are similar

Used in production at Google and others to reduce inference cost for large models.

---

# 9. Combination Strategies

Real deployments do not use one technique.
They use several in sequence.

---

## 9.1 The Prune → Distill → Quantize Pipeline

This is a common production pattern.

**Step 1: Prune**

Remove neurons, heads, or layers that contribute least.
Fine-tune the pruned model.

This gives you a structurally smaller network.

**Step 2: Distill**

Use the original large model (or the pruned model) as teacher.
Train a smaller student model that learned from it.

This transfers knowledge to a potentially even smaller architecture.

**Step 3: Quantize**

Take the final model and apply PTQ (or QAT if you have budget).

INT8 for CPU-heavy deployments.
FP16 for GPU-heavy deployments.
INT4 if memory is the bottleneck.

**Why this order**

Pruning first removes dead capacity.
Distillation then transfers knowledge efficiently into the now-focused architecture.
Quantization at the end has the least accuracy-sensitive starting point.

---

## 9.2 LoRA + Quantization (QLoRA)

A popular modern combination for fine-tuning large models.

1. Quantize the base model to 4-bit (NF4 quantization, a format designed for neural weights)
2. Add LoRA adapters in BF16
3. Fine-tune only the adapters

This lets you fine-tune a 70B model on a single 48GB GPU.

The quantized base is frozen; only the small low-rank adapters are trained.

---

## 9.3 NAS + Distillation

Design a hardware-efficient architecture via NAS.
Use a large pretrained model to distill into it.

EfficientDet and MobileNetV3 variants were trained this way.

The NAS gives you an architecture designed for the hardware.
Distillation gives it the knowledge of a more powerful teacher.

---

# 10. Common Interview Questions

---

**Q: What is the difference between quantization and pruning?**

Quantization reduces the numerical precision of weights.
Pruning removes weights entirely.

Quantization keeps the model structure intact.
Pruning changes the model structure (or creates sparsity).

They complement each other and are often combined.

---

**Q: Why do soft labels help in knowledge distillation?**

Hard labels (one-hot) carry no information about the relationships between classes.

Soft labels from a teacher encode partial similarities: a cat image is a bit like a dog, not at all like an airplane.

This richer signal improves generalization, especially when the labeled dataset is small.

Temperature scaling amplifies this effect by making the distribution flatter.

---

**Q: What is the lottery ticket hypothesis and why does it matter?**

The hypothesis states that large networks contain small subnetworks (winning tickets) that, trained from their original initialization, can match the full network's performance.

Implication: overparameterization during training is a search process.
The big network is not needed for inference; it was needed to reliably find the good subnetwork.

Practical impact: justifies aggressive pruning if done correctly, and informs why smaller models trained well from scratch can sometimes match large models.

---

**Q: How would you deploy a 70B LLM on a single consumer GPU?**

Use 4-bit quantization (GPTQ or AWQ).
A 70B model at 4 bits requires ~35GB of VRAM.
Fits on a single 40GB or 48GB GPU.

For further improvement:
- Use speculative decoding with a small 7B draft model for faster generation
- Use flash attention to reduce memory usage during the attention computation
- Consider using bitsandbytes or llama.cpp for optimized int4 inference

---

**Q: What is speculative decoding and when does it help?**

A small draft model generates k tokens quickly.
The large target model verifies all k tokens in one parallel forward pass.
Tokens that match the target model's distribution are accepted.

It helps most when:
- generation is memory-bandwidth bound (not compute-bound)
- the draft model and target model have similar distributions (same family, different size)
- output sequences are long (more opportunities to parallelize)

---

**Q: What is QAT and when should you use it over PTQ?**

Quantization-Aware Training simulates quantization during training, letting the model adapt to lower precision.

Use PTQ when:
- you have a trained model and cannot retrain
- INT8 is your target (PTQ works well here)
- you want a fast solution

Use QAT when:
- you need INT4 or lower precision
- PTQ accuracy is unacceptable
- you have training compute available and the accuracy budget is tight

---

**Q: What is structured pruning and why is it preferred for hardware deployment?**

Structured pruning removes entire neurons, filters, or attention heads.

The result is a model with smaller matrices: fewer rows, fewer columns.
This directly reduces compute and memory without requiring sparse hardware support.

Unstructured pruning creates irregular sparsity.
Most hardware (CPU, GPU) is not efficient at irregular sparse operations unless specifically designed for it (e.g., NVIDIA 2:4 sparsity pattern).

---

**Q: What is ONNX and why is it used?**

ONNX is a model interchange format.
It represents the model as a computational graph in a hardware-agnostic way.

You export a PyTorch or TensorFlow model to ONNX.
Then run it via ONNX Runtime or convert it to TensorRT, CoreML, etc.

Benefits:
- operator fusion reduces kernel launch overhead
- constant folding pre-computes static subgraphs
- cross-platform compatibility

---

**Q: You are asked to compress a BERT model for production API serving. What steps do you take?**

1. Establish a baseline: current latency, memory, accuracy
2. Define constraints: latency budget, memory budget, accuracy threshold
3. Try FP16 conversion first: almost free, often enough
4. Structured pruning: remove redundant attention heads (analysis shows BERT has many redundant heads)
5. Distillation: use full BERT to distill a 6-layer student (DistilBERT-style)
6. INT8 quantization: apply PTQ with representative calibration data
7. Export to ONNX + TensorRT for the serving hardware
8. Benchmark and validate each step

This pipeline typically gets 3–5x speedup with < 2% accuracy drop on most NLP tasks.

---

**Q: What is knowledge distillation vs transfer learning?**

Transfer learning: take a pretrained model and fine-tune it on a new task.
The model itself is reused.

Knowledge distillation: use a larger model's predictions to train a smaller model.
The knowledge is transferred, not the weights.

They can be combined: distill from a fine-tuned large teacher to a smaller student.

---

**Q: How does AWQ differ from GPTQ?**

GPTQ uses second-order (Hessian) information to quantize weights layer by layer, compensating for quantization error in each weight by adjusting others.

AWQ identifies which weights are most important by looking at activation magnitudes.
Weights that multiply large activations are kept at higher precision or scaled protectively before quantization.

AWQ is faster to apply and tends to be more robust across architectures.
GPTQ achieves very high quality at the cost of more computation during quantization.

---

**Q: What is calibration in the context of quantization?**

Calibration determines the mapping from floating point values to integers.

You need to know:
- the range of values (min, max) for scale and zero-point
- the distribution shape for more advanced methods

A calibration dataset is a small representative set passed through the model before quantization.
Activation statistics are collected per layer.

Bad calibration = clipping or overflowing values during quantization = accuracy loss.

---

**Q: What compression technique would you recommend for a recommendation model vs a language model?**

**Recommendation model:**
- Usually embedding-heavy, not compute-heavy
- Low-rank factorization on embedding tables is highly effective
- INT8 quantization on the dense layers
- Structured pruning on overparameterized wide layers

**Language model (LLM):**
- Memory-bandwidth bound during inference
- INT4/INT8 weight quantization (GPTQ/AWQ) to reduce memory movement
- Speculative decoding for throughput
- Head pruning + layer dropping for smaller model sizes
- Distillation to a smaller architecture if retraining is possible
