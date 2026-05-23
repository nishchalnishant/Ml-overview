---
module: References
topic: Book Notes
subtopic: Deep Learning Deep Learning With Pytorch
status: unread
tags: [references, ml, book-notes-deep-learning]
---
# Deep Learning with PyTorch

## Chapter 1: Introducing Deep Learning and the PyTorch Library

**The problem the book is addressing**
Deep learning frameworks proliferated (Theano, Caffe, MXNet, TF, PyTorch), each with different APIs and trade-offs. Practitioners need to understand not just what PyTorch does but *why* it was designed the way it was — so they can use it effectively rather than cargo-culting examples.

**The core insight**
PyTorch uses dynamic computation graphs (define-by-run) rather than static graphs (define-then-run). This means the graph is rebuilt every forward pass, making debugging as natural as ordinary Python. The cost is that static graph optimizations (like TF's XLA) require an extra step (TorchScript/`torch.compile`).

**The mechanics**
- Core abstractions: `torch.Tensor` (multi-dim array with GPU support), `autograd` (automatic differentiation), `nn.Module` (composable model building block), `DataLoader`/`Dataset` (data pipeline)
- GPU acceleration: move tensors with `.to('cuda')`; operations on GPU tensors execute asynchronously on the GPU
- Dynamic graphs: `requires_grad=True` tensors record operations; `.backward()` computes all gradients in one reverse pass
- Deployment path: Python training → TorchScript/ONNX export → C++/production server

**What the book gets right / what to watch out for**
PyTorch is the right framework for research and is now also dominant in production. The dynamic graph advantage is real for debugging. For production performance, use `torch.compile()` (PyTorch 2.0) to get static-graph speedups while keeping dynamic semantics during development.

---

## Chapters 2–4: Tensors, the Core Data Structure

**The problem the book is addressing**
Neural network computations operate on multidimensional arrays. Getting shapes wrong silently produces incorrect results rather than errors. Practitioners need deep fluency with tensor manipulation to diagnose bugs.

**The core insight**
Tensors are typed multidimensional arrays with storage (the raw data in memory) and a view (shape, stride, offset) that describes how to interpret that storage. Many operations create views of the same storage — changes to a view affect the original.

**The mechanics**
- Creation: `torch.zeros/ones/rand/randn`, `torch.tensor([...])`, `from_numpy()`
- Shape ops: `.view(shape)`, `.reshape(shape)` (vs `.contiguous()`); `.squeeze/unsqueeze`; `.permute(dims)`
- Math: `+, -, *, /` (element-wise); `@` or `torch.mm` (matrix multiply); `torch.einsum`
- Type: `.float()`, `.double()`, `.half()`, `.to(dtype)` — always keep consistent dtype
- In-place ops: `_` suffix (`add_`, `relu_`) — modify tensor in-place; breaks autograd if applied to leaf variables

**What the book gets right / what to watch out for**
The stride/view explanation is the most important insight in these chapters. A common bug: `.view()` fails if tensor is not contiguous after `.permute()` — use `.contiguous().view()` or just `.reshape()`. Mixed dtype operations raise errors or silently downcast — always be explicit about dtype.

---

## Chapter 5: The Mechanics of Learning

**The problem the book is addressing**
Training a model requires more than running forward passes. You need to understand the full learning loop: compute loss, compute gradients via backward pass, update parameters, handle the model state for inference. These steps interact in subtle ways.

**The core insight**
PyTorch separates the three concerns: loss computation (your code), gradient computation (autograd, called once per backward pass), and parameter updates (optimizer). Each step is explicit and independently testable.

**The mechanics**
```python
# Standard training loop
model.train()
for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()          # clear previous gradients
    y_pred = model(x)              # forward pass
    loss = criterion(y_pred, y)    # compute loss
    loss.backward()                # compute gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()               # update parameters
```
- `optimizer.zero_grad()` must be called — gradients accumulate by default
- `model.train()` vs `model.eval()`: switches dropout and BatchNorm behavior
- `with torch.no_grad()`: disables gradient tracking for inference (saves memory)

**What the book gets right / what to watch out for**
The explicit `zero_grad()` requirement is a deliberate design choice that enables gradient accumulation (sum gradients over multiple mini-batches before stepping) — useful when GPU memory limits effective batch size. Forgetting it is a common bug that produces silently wrong training.

---

## Chapter 6: Using a Neural Network to Fit Data

**The problem the book is addressing**
`nn.Module` is PyTorch's building block for neural networks. Understanding how it tracks parameters, handles nested modules, and composes into complex architectures is necessary before you can debug or extend any real model.

**The core insight**
`nn.Module` is a container that tracks all `nn.Parameter` objects (trainable tensors) and sub-modules recursively. `model.parameters()` yields a flat iterator over all of them — this is what optimizers need. `forward()` defines the computation.

**The mechanics**
```python
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)  # registers as submodule
        self.bn = nn.BatchNorm1d(hidden)
        self.linear2 = nn.Linear(hidden, out_features)
    
    def forward(self, x):
        x = F.relu(self.bn(self.linear1(x)))
        return self.linear2(x)

model = TinyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- `nn.Sequential`: convenience wrapper for linear chains
- `nn.ModuleList/ModuleDict`: register lists/dicts of modules
- `model.state_dict()`: returns all parameter tensors as an ordered dict — use for checkpointing

**What the book gets right / what to watch out for**
The state_dict pattern is important: always save `model.state_dict()`, not the model object, to avoid pickling issues. The `strict=False` option in `load_state_dict` allows loading partial checkpoints (e.g., loading a pretrained backbone into a model with a different head).

---

## Chapters 7–9: Real-World Computer Vision (CT Scans)

**The problem the book is addressing**
The book's extended project — detecting malignant lung nodules in CT scans — illustrates the full DL engineering pipeline on a realistic medical problem. This is the gap between "it works on MNIST" and "it works in production": data loading, class imbalance, 3D volumetric data, and clinical metrics.

**The core insight**
Medical imaging is the hardest case study the book could have chosen: 3D inputs (not 2D), severe class imbalance (1 in 1000 slices contains a nodule), and clinical consequences of false negatives vs false positives. The lessons generalize to any high-stakes, imbalanced classification.

**The mechanics**
- 3D CT volumes: load with `SimpleITK` or `pydicom`; voxel intensities in Hounsfield units; resample to uniform voxel spacing
- DataLoader: custom `Dataset.__getitem__` handles disk I/O and augmentation lazily
- Class imbalance: balance training batches (equal positive/negative samples) in `__iter__`; SMOTE for tabular features
- U-Net segmentation: encoder-decoder with skip connections; Dice loss for pixel-level segmentation

**What the book gets right / what to watch out for**
The extended medical imaging project is one of the book's strongest contributions — it's unique among DL textbooks. The custom `Dataset` patterns for large disk-based datasets are directly applicable. The SMOTE for CT data is unusual (more common for tabular) — for images, augmentation is more effective.

---

## Chapter 10: Building a Dataset with Augmentation

**The problem the book is addressing**
Medical imaging models trained on small annotated datasets overfit. Collecting more labeled data is expensive. How do you expand effective dataset size and improve generalization without more annotations?

**The core insight**
Data augmentation applies label-preserving transformations (random flips, rotations, brightness changes) to training examples, forcing the model to be invariant to these variations. The transformations should reflect real-world variability, not arbitrary changes.

**The mechanics**
- Geometric: random horizontal/vertical flip, rotation (±30°), crop and resize
- Intensity: brightness ±20%, contrast ±20%, Gaussian noise
- For CT: random affine transforms on 3D volumes; be careful with left-right flips (anatomically meaningful asymmetry)
- Implementation: compose transforms in `torchvision.transforms.Compose`; apply only to training set (not validation/test)

**What the book gets right / what to watch out for**
Augmentation is one of the most effective regularization strategies. The key constraint — apply only during training — is critical and commonly violated. For medical images, domain-specific augmentation (realistic deformations) is more effective than standard transforms. MixUp and CutMix (blending two images) are modern augmentations that further improve generalization.

---

## Chapter 11: Training Classification Model

**The problem the book is addressing**
Training metrics (loss, accuracy) on the training set don't tell you if the model generalizes. For imbalanced problems, accuracy is misleading. How do you evaluate during training in a way that catches overfitting and measures the metrics that matter clinically?

**The core insight**
For medical classification, the relevant metric is sensitivity (recall) at a given specificity — captured by the ROC curve. AUC-ROC summarizes performance across all thresholds, decoupled from the operating point decision.

**The mechanics**
- Confusion matrix: TP, FP, TN, FN → precision = TP/(TP+FP), recall = TP/(TP+FN)
- ROC curve: plot sensitivity vs (1-specificity) as threshold varies from 0 to 1
- AUC: area under ROC; 0.5 = random, 1.0 = perfect; insensitive to class imbalance
- During training: log train loss, val loss, val AUC every N batches; save checkpoint at best val AUC

**What the book gets right / what to watch out for**
AUC-ROC is the right metric for imbalanced binary classification when the operating threshold will be tuned separately. For cases where false negatives are much worse than false positives (cancer detection), also track recall at 95% specificity explicitly. Don't use accuracy as the primary metric for imbalanced problems.

---

## Chapter 12: Improving Training with Metrics and Augmentation

**The problem the book is addressing**
The model from chapter 11 has decent AUC but poor recall at clinically acceptable specificity. How do you diagnose *why* it's failing and which interventions (more data, different architecture, better augmentation, balanced sampling) will help?

**The core insight**
Systematic ablation — changing one thing at a time and measuring impact — is the only reliable way to improve models. Adding three things simultaneously makes it impossible to know which one helped.

**The mechanics**
- Ablation: baseline → add augmentation → add balanced sampling → add deeper architecture; measure val AUC at each step
- Learning curves: plot val AUC vs training set size — still increasing = more data helps; plateaued = model is the bottleneck
- Error analysis: inspect false positives and false negatives — what do they have in common?
- Calibration: check if model probabilities match actual frequencies; use Platt scaling or isotonic regression if not

**What the book gets right / what to watch out for**
The systematic approach is correct. Error analysis on specific examples is often more informative than aggregate metrics — looking at the 50 worst false positives reveals systematic failure modes that metrics hide.

---

## Chapter 13: Using Segmentation to Find Suspected Nodules

**The problem the book is addressing**
Classification tells you whether a nodule is present but not where. Clinical use requires localization — which voxels belong to the nodule. Segmentation is the task of assigning a label (nodule/background) to every voxel.

**The core insight**
U-Net's encoder-decoder architecture with skip connections is the canonical solution for medical image segmentation. The encoder captures context (what kind of structure is present), the decoder provides localization (where exactly). Skip connections preserve fine-grained spatial detail that the encoder compresses away.

**The mechanics**
- U-Net: encoder (Conv → MaxPool, halves spatial, doubles channels) × 4; bottleneck; decoder (Upsample → Conv, halves channels, doubles spatial) × 4; skip connections concatenate encoder features to decoder at each scale
- Dice loss: L = 1 - (2 × |P∩T|) / (|P| + |T|); P=prediction, T=target; handles class imbalance naturally (background pixels don't dominate)
- BCE + Dice combined: BCE penalizes individual pixels; Dice penalizes structural overlap — combining improves convergence
- Evaluation: Dice score (segmentation overlap), IoU (intersection over union)

**What the book gets right / what to watch out for**
U-Net remains the standard for medical segmentation. Dice loss is correct and important — cross-entropy alone fails for highly imbalanced segmentation (tiny nodule vs large background). For 3D volumes, 3D U-Net extends naturally but requires significant GPU memory — patch-based training is standard.

---

## Chapters 14–15: End-to-End Nodule Analysis

**The problem the book is addressing**
Real clinical deployment requires combining segmentation (find nodules) and classification (assess malignancy) into a single pipeline. Each component has different failure modes, and errors compound across stages.

**The core insight**
Combining components: segmentation predicts candidate regions → classification scores each candidate → thresholding produces final predictions. The pipeline's recall is the product of each stage's recall — one weak stage bottlenecks the whole system.

**The mechanics**
- Stage 1: segmentation model produces voxel-level masks; extract connected components as candidate nodules
- Stage 2: for each candidate, crop a 3D patch centered on it; classify with a separate CNN
- Malignancy scoring: soft score output (probability) rather than hard threshold — defer threshold decision to clinical use
- False positive reduction: classifier trained to distinguish true nodules from segmentation false positives

**What the book gets right / what to watch out for**
The staged approach correctly separates detection from characterization. Error propagation across stages is the key risk — if segmentation misses 20% of nodules, the classifier never sees them. Multi-task learning (jointly optimize segmentation + classification) can improve recall at the detection stage.

---

## Chapter 15: Deploying to Production

**The problem the book is addressing**
A model trained in a Jupyter notebook is not a production system. Serving predictions requires an API, model loading, batching, and handling the operational reality that PyTorch models don't run natively in all environments (mobile, embedded, C++ services).

**The core insight**
The key production question is: what's the deployment target? Python server → use TorchServe or Flask; C++ service → export with TorchScript; mobile → export with TorchScript or ONNX; edge → quantize and optimize.

**The mechanics**
- Flask/Sanic: wrap `model(input_tensor)` in an HTTP endpoint; return prediction as JSON
- ONNX export: `torch.onnx.export(model, dummy_input, "model.onnx")` — framework-agnostic format; run with ONNXRuntime
- TorchScript: `torch.jit.script(model)` or `torch.jit.trace(model, example_input)` — serializable, runnable in C++ via LibTorch
- TorchServe: production model server; handles batching, versioning, logging; deploy with `torch-model-archiver`

**What the book gets right / what to watch out for**
The deployment chapter correctly covers the full production spectrum. TorchScript's `trace` mode (records operations for a specific input) vs `script` mode (compiles Python control flow) — use `script` for models with dynamic control flow (LSTMs, conditionals); `trace` for simple feedforward models. ONNX is best for cross-framework portability but not all PyTorch ops are supported.

## Flashcards

**Core abstractions?** #flashcard
torch.Tensor (multi-dim array with GPU support), autograd (automatic differentiation), nn.Module (composable model building block), DataLoader/Dataset (data pipeline)

**GPU acceleration?** #flashcard
move tensors with .to('cuda'); operations on GPU tensors execute asynchronously on the GPU

**Dynamic graphs?** #flashcard
requires_grad=True tensors record operations; .backward() computes all gradients in one reverse pass

**Deployment path?** #flashcard
Python training → TorchScript/ONNX export → C++/production server

**Creation?** #flashcard
torch.zeros/ones/rand/randn, torch.tensor([...]), from_numpy()

**Shape ops?** #flashcard
.view(shape), .reshape(shape) (vs .contiguous()); .squeeze/unsqueeze; .permute(dims)

**Math?** #flashcard
+, -, *, / (element-wise); @ or torch.mm (matrix multiply); torch.einsum

**Type: .float(), .double(), .half(), .to(dtype)?** #flashcard
always keep consistent dtype

**In-place ops: _ suffix (add_, relu_)?** #flashcard
modify tensor in-place; breaks autograd if applied to leaf variables

**optimizer.zero_grad() must be called?** #flashcard
gradients accumulate by default

**model.train() vs model.eval()?** #flashcard
switches dropout and BatchNorm behavior

**with torch.no_grad()?** #flashcard
disables gradient tracking for inference (saves memory)

**nn.Sequential?** #flashcard
convenience wrapper for linear chains

**nn.ModuleList/ModuleDict?** #flashcard
register lists/dicts of modules

**model.state_dict(): returns all parameter tensors as an ordered dict?** #flashcard
use for checkpointing

**3D CT volumes?** #flashcard
load with SimpleITK or pydicom; voxel intensities in Hounsfield units; resample to uniform voxel spacing

**DataLoader?** #flashcard
custom Dataset.__getitem__ handles disk I/O and augmentation lazily

**Class imbalance?** #flashcard
balance training batches (equal positive/negative samples) in __iter__; SMOTE for tabular features

**U-Net segmentation?** #flashcard
encoder-decoder with skip connections; Dice loss for pixel-level segmentation

**Geometric?** #flashcard
random horizontal/vertical flip, rotation (±30°), crop and resize

**Intensity?** #flashcard
brightness ±20%, contrast ±20%, Gaussian noise

**For CT?** #flashcard
random affine transforms on 3D volumes; be careful with left-right flips (anatomically meaningful asymmetry)

**Implementation?** #flashcard
compose transforms in torchvision.transforms.Compose; apply only to training set (not validation/test)

**Confusion matrix?** #flashcard
TP, FP, TN, FN → precision = TP/(TP+FP), recall = TP/(TP+FN)

**ROC curve?** #flashcard
plot sensitivity vs (1-specificity) as threshold varies from 0 to 1

**AUC?** #flashcard
area under ROC; 0.5 = random, 1.0 = perfect; insensitive to class imbalance

**During training?** #flashcard
log train loss, val loss, val AUC every N batches; save checkpoint at best val AUC

**Ablation?** #flashcard
baseline → add augmentation → add balanced sampling → add deeper architecture; measure val AUC at each step

**Learning curves: plot val AUC vs training set size?** #flashcard
still increasing = more data helps; plateaued = model is the bottleneck

**Error analysis: inspect false positives and false negatives?** #flashcard
what do they have in common?

**Calibration?** #flashcard
check if model probabilities match actual frequencies; use Platt scaling or isotonic regression if not

**U-Net?** #flashcard
encoder (Conv → MaxPool, halves spatial, doubles channels) × 4; bottleneck; decoder (Upsample → Conv, halves channels, doubles spatial) × 4; skip connections concatenate encoder features to decoder at each scale

**Dice loss?** #flashcard
L = 1 - (2 × |P∩T|) / (|P| + |T|); P=prediction, T=target; handles class imbalance naturally (background pixels don't dominate)

**BCE + Dice combined: BCE penalizes individual pixels; Dice penalizes structural overlap?** #flashcard
combining improves convergence

**Evaluation?** #flashcard
Dice score (segmentation overlap), IoU (intersection over union)

**Stage 1?** #flashcard
segmentation model produces voxel-level masks; extract connected components as candidate nodules

**Stage 2?** #flashcard
for each candidate, crop a 3D patch centered on it; classify with a separate CNN

**Malignancy scoring: soft score output (probability) rather than hard threshold?** #flashcard
defer threshold decision to clinical use

**False positive reduction?** #flashcard
classifier trained to distinguish true nodules from segmentation false positives

**Flask/Sanic?** #flashcard
wrap model(input_tensor) in an HTTP endpoint; return prediction as JSON

**ONNX export: torch.onnx.export(model, dummy_input, "model.onnx")?** #flashcard
framework-agnostic format; run with ONNXRuntime

**TorchScript: torch.jit.script(model) or torch.jit.trace(model, example_input)?** #flashcard
serializable, runnable in C++ via LibTorch

**TorchServe?** #flashcard
production model server; handles batching, versioning, logging; deploy with torch-model-archiver
