---
module: Domains
topic: Dl
subtopic: Computer Vision
status: unread
tags: [interviewprep, dl, dl-computer-vision, interview-framing]
---
# Computer Vision

**Primary reference:** [Computer Vision deep dive](04-computer-vision.md)

---

## What This File Is For

Every topic is structured around the four questions that matter in an interview:
1. What the interviewer is actually testing
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

---

## 1. Why Images Need Special Treatment

**What the interviewer is testing:** Whether you understand inductive bias — that encoding structural assumptions about a data type directly into the architecture reduces the amount of data needed to learn everything else. This is a proxy for whether you think about architecture as a choice, not a default.

**The reasoning structure:** A 224×224 RGB image has 150,528 numbers. A fully connected network treating each pixel as an independent input must learn from scratch that nearby pixels are related, that the same local pattern means the same thing wherever it appears, and that spatial hierarchies exist. That requires enormous data and produces enormous parameter counts.

Images have structure that can be encoded as architectural constraints rather than learned from data:
- **Locality:** the information needed to recognize any local region is mostly in that region and its immediate neighborhood
- **Translation equivariance:** a cat in the upper-left and a cat in the lower-right share the same local appearance features — the feature representation should not depend on where it is in the image
- **Hierarchy:** edges combine into shapes, shapes combine into parts, parts combine into objects

Building these assumptions into the architecture is not cheating — it is domain knowledge that reduces the sample complexity of learning everything else. The alternative is making the network learn these invariances from data, which is possible (Vision Transformers do exactly this at scale) but expensive.

**The pattern in action:** Train a fully connected network and a CNN on CIFAR-10 with 1,000 training examples. The CNN reaches approximately 60% accuracy; the MLP plateaus around 40%. The CNN needs less data to converge because it does not spend capacity learning that nearby pixels tend to be related — it already knows. Scale the dataset to 1.2 million images and the MLP gap closes significantly, because the MLP can learn those priors from data given enough examples.

**Common traps:**
- Saying "CNNs are better than MLPs for images" without explaining why. The answer is locality and translation equivariance as specific inductive biases that match the structure of natural images.
- Treating inductive bias as universally good. An inductive bias is an assumption about the data distribution. If the assumption is wrong — for example, applying translation equivariance to images where position is diagnostically important — the bias is harmful.
- Ignoring that Vision Transformers largely abandon these inductive biases and learn spatial relationships from scratch. The tradeoff is data efficiency (CNNs better) versus scalability with data (ViTs better).

---

## 2. Convolution: The Core Operation

**What the interviewer is testing:** Whether you understand what convolution actually computes — not just that "filters slide over images" — and can reason about the parameter count and spatial dimension consequences of each design choice.

**The reasoning structure:** A convolution filter is a small weight matrix — typically 3×3 — that slides over the input feature map, computing a dot product at each position. The same weights are applied at every spatial position. This is **weight sharing**, which implements translation equivariance and reduces parameters by orders of magnitude compared to a fully connected layer.

For an input of size $H \times W$ with $C_{\text{in}}$ channels, a single filter of size $k \times k$ has $k^2 \cdot C_{\text{in}}$ parameters regardless of input spatial dimensions. A fully connected layer connecting the same input to one output neuron would require $H \cdot W \cdot C_{\text{in}}$ parameters — and those parameters would learn a specific response for each spatial position rather than a position-independent pattern detector.

**Output size formula:**
$$H_{\text{out}} = \frac{H_{\text{in}} + 2p - k}{s} + 1$$
where $p$ = padding, $k$ = kernel size, $s$ = stride.

Early layers learn edge detectors, blob detectors, and color patches — these emerge from training because they are the most predictive low-level statistics of natural images. Deeper layers combine these into progressively more abstract representations.

**Depthwise separable convolutions** factorize a standard convolution into a depthwise step (one filter per input channel) and a pointwise 1×1 step. This reduces computation by roughly $k^2$, which is the foundation of MobileNet and EfficientNet. The key insight: a standard convolution mixes spatial filtering and channel mixing in one operation; separating them reduces FLOPs with minimal accuracy cost.

**The pattern in action:** Visualizing the first-layer filters of AlexNet reveals oriented Gabor-like edge detectors similar to those found in the primary visual cortex. This is not coincidence — the network learned the most informative low-level statistics of natural images given the architectural constraint of local weight sharing. A fully connected first layer learns a messier, less transferable representation because it can memorize position-specific responses.

**Common traps:**
- Confusing what stride and padding do independently. Stride reduces spatial resolution and increases the receptive field covered per parameter. Padding preserves spatial dimensions. A common interview question: "why would you increase stride?" — the answer is to reduce spatial resolution quickly and cheaply, as in stem layers of efficient architectures where you want to reduce computation before the expensive later layers.
- Treating depthwise separable convolutions as a separate topic rather than a factorization of standard convolutions. Interviewers asking about efficient architectures expect you to derive the FLOPs reduction: standard convolution has $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$ multiplications; depthwise separable has $k^2 \cdot C_{\text{in}} \cdot H \cdot W + C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$, a reduction by approximately $1/C_{\text{out}} + 1/k^2$.

---

## 3. Pooling

**What the interviewer is testing:** Whether you can articulate what pooling achieves beyond "makes the feature map smaller" — specifically the two distinct purposes pooling serves and when each is appropriate.

**The reasoning structure:** Pooling serves two distinct purposes worth separating:

1. **Spatial compression:** reduces feature map size, reducing computation in subsequent layers
2. **Local translation tolerance:** max pooling extracts the strongest activation in a region without caring about its exact sub-pixel position — small translations in the input produce the same pooled output

Max pooling retains the highest activation in each region, which is appropriate when you care about whether a feature is present somewhere in a neighborhood, not its precise location. Average pooling computes the mean, which is appropriate for aggregating distributed signals.

**Global average pooling (GAP)** reduces the entire spatial feature map to a single vector by averaging across all spatial positions. This replaces fully connected classification heads in modern architectures, dramatically reducing parameters and providing implicit regularization: the classification must emerge from a spatial average rather than a memorized mapping from flattened features. It also enables the model to process variable-sized inputs, since the spatial averaging is independent of the spatial dimensions.

**The pattern in action:** Replacing the fully connected head of a ResNet-50 with global average pooling drops the head from approximately 8 million parameters (2048 × 4096 + 4096 × num_classes) to one linear layer of size 2048 × num_classes. The model generalizes better on small datasets because overfitting the classification head is no longer possible — the head has too few parameters to memorize the training set.

**Common traps:**
- Treating pooling as mandatory. Strided convolutions achieve similar spatial compression and are learnable — the pooling window is fixed but a stride-2 convolution learns what to compress. Modern architectures (ResNet with stride-2 convolutions, ViT with patch embeddings) reduce or eliminate explicit pooling layers.
- Conflating max pooling (used within the feature extraction network for local translation tolerance) with global average pooling (used at the end for spatial collapse before classification). They are different operations with different purposes.

---

## 4. Deep Networks and Residual Connections

**What the interviewer is testing:** Whether you can explain the specific problem residual connections solve and the mechanism by which they solve it — not just say they "help deep networks train." The degradation problem and its gradient-flow solution are the core.

**The reasoning structure:** A deep network should in principle perform at least as well as a shallower one — worst case, the extra layers learn to be identity functions and add nothing. In practice, optimization cannot find this solution. Deeper networks trained with standard backpropagation often perform worse than shallower ones, not because they lack capacity but because the gradient signal degrades as it travels through many nonlinear layers. This is the degradation problem — empirically observed, not theoretically predicted.

Residual connections address this by rewriting what each layer must learn:

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

Instead of learning the full desired mapping $H(\mathbf{x})$, each block learns the residual $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$. If the optimal function is close to identity, learning $F(\mathbf{x}) \approx 0$ is far easier than learning $H(\mathbf{x}) \approx \mathbf{x}$ through many nonlinear layers — because forcing weights to zero is easier than finding a composition of nonlinearities that approximates the identity.

For backpropagation: the gradient through the skip connection is $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + I$. The identity term provides a direct gradient path from the loss to early layers, bypassing the block's nonlinearities. Even if $\frac{\partial F}{\partial \mathbf{x}}$ is small (vanishing gradient through the block), the identity term ensures a usable gradient still arrives at early layers.

**The pattern in action:** VGG-19 (19 layers, no skip connections) achieves approximately 74% top-1 on ImageNet. ResNet-50 (50 layers, residuals) achieves approximately 76% and trains more stably. ResNet-152 (152 layers) reaches approximately 78% — a depth completely untrainable without skip connections. The residual connection enabled depth scaling that was previously inaccessible, not by adding regularization but by changing the optimization landscape.

**Common traps:**
- Saying residual connections "help gradients flow" without the specific mechanism. The claim is that the identity shortcut provides a gradient path that bypasses the block's nonlinearities — the gradient of the sum includes the identity matrix, which prevents complete vanishing.
- Confusing the residual learning argument (easier to learn near-zero residuals than near-identity full mappings) with the gradient argument (identity shortcut prevents vanishing). Both are true and both are worth stating — they address different aspects of why the design works.
- Conflating residual connections with dense connections (DenseNet). Residuals add the block's input to its output via a single shortcut. DenseNet concatenates all previous layer outputs to every subsequent layer — more parameter sharing but also more memory.

---

## 5. Image Classification vs Object Detection vs Segmentation

**What the interviewer is testing:** Whether you understand what different spatial output formats require architecturally, and when each is the right problem formulation rather than the default one. The skill being tested is problem formulation, not architecture recall.

**The reasoning structure:** These are four different problem formulations with different output formats, different annotation costs, and different architectural requirements:

**Classification:** What class does this image belong to? Output: a single label or probability distribution. Architecture requirement: a global representation, achieved by pooling spatial features to a vector. Annotation cost: one label per image.

**Object detection:** What objects are present, and where are they? Output: a set of bounding boxes with class labels and confidence scores. Architecture requirement: predictions at multiple spatial locations and scales, because objects can be anywhere and any size. Annotation cost: one bounding box per object instance.

**Semantic segmentation:** Which class does each pixel belong to? Output: per-pixel class assignment. Does not distinguish instances — two dogs at the same location produce the same output. Architecture requirement: an encoder-decoder that recovers spatial resolution (U-Net, DeepLab). Annotation cost: pixel-level masks, approximately 15× more expensive than bounding boxes.

**Instance segmentation:** Which pixels belong to which individual object instance? Output: per-pixel class and instance identity. Must distinguish two dogs at the same location. Architecture requirement: extends detection with a mask head per detected region (Mask R-CNN).

**Panoptic segmentation:** Every pixel gets a class label; countable objects ("things") also get instance identities; background ("stuff") gets class only.

**The pattern in action:** An automated grocery checkout system. Classification ("is there an apple?") is insufficient — you need to count. Detection is the right formulation: one bounding box per item. If two apples are partially stacked and occluding each other, instance segmentation becomes necessary to count them separately. The escalation from detection to instance segmentation is justified by the specific failure mode of stacked items, not by a preference for more sophisticated methods.

**Common traps:**
- Defaulting to segmentation when detection suffices. Segmentation requires pixel-level annotation (15× more expensive to label), is computationally heavier, and provides no benefit if pixel-level location is not required.
- Treating semantic and instance segmentation as equivalent difficulty levels rather than different problem formulations. They differ on whether the model distinguishes individual instances of the same class — semantic segmentation cannot count two touching dogs; instance segmentation can.

---

## 6. One-Stage vs Two-Stage Object Detectors

**What the interviewer is testing:** Whether you can articulate the speed-accuracy tradeoff, explain the mechanism behind the difference, and reason about which is appropriate for a given deployment context.

**The reasoning structure:** The fundamental question is whether you separate "where might objects be?" from "what is in each candidate region?"

**Two-stage detectors (Faster R-CNN):**
Stage 1: A Region Proposal Network generates class-agnostic candidate bounding boxes likely to contain objects.
Stage 2: For each proposal, extract RoI-aligned features from the backbone and classify the box while refining its coordinates.

The two-stage approach lets stage 2 work on a small set of high-quality proposals with full contextual features, which is why it tends to have higher accuracy — especially on small or heavily overlapping objects. The cost is latency: extracting and processing hundreds of RoI crops is sequential and expensive.

**One-stage detectors (YOLO, SSD, RetinaNet):**
Predict boxes and classes directly from a dense grid of anchor points in a single forward pass. Much faster because there is no proposal extraction step. Historically traded some accuracy for speed, but modern one-stage detectors (YOLOv8, DINO) close much of the gap.

**RetinaNet's contribution — Focal Loss:** One-stage detectors predict from thousands of anchor locations. For every positive (actual object), there are ~10,000 easy background negatives. Standard cross-entropy loss gets dominated by easy negatives. Focal loss down-weights easy examples by their own prediction confidence:

$$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$

When $\gamma > 0$ (typically 2), examples the model already predicts correctly with high confidence contribute near-zero loss. The training signal concentrates on hard positives and hard negatives — the examples the model is actually uncertain about.

**The pattern in action:** A factory quality control system on a high-speed conveyor belt requires 60 fps inference. A YOLO-based one-stage detector is correct even if its mAP is slightly below a two-stage model, because the two-stage model cannot meet the throughput requirement. For offline medical imaging analysis where missed findings are costly and latency is irrelevant, the higher recall of a two-stage approach justifies the computational cost. The same architecture choice maps directly onto the deployment constraint.

**Common traps:**
- Ignoring the anchor design problem. One-stage detectors historically required carefully designed anchor boxes tuned to expected object aspect ratios. Anchor-free detectors (FCOS, CenterNet) eliminate this requirement — important when deploying on datasets with unusual object shapes where prior anchor distributions are unavailable.
- Treating mAP as the only relevant metric. Latency, hardware budget, and the cost asymmetry between false positives and false negatives often dominate the decision in practice.

---

## 7. IoU and mAP

**What the interviewer is testing:** Whether you can explain evaluation metrics precisely — what they summarize, what they hide, and what design choices in the metric (IoU threshold, size bins) affect the conclusions you can draw.

**The reasoning structure:**

**Intersection over Union** measures the geometric overlap between a predicted box and a ground truth box:
$$\text{IoU} = \frac{|\text{pred} \cap \text{gt}|}{|\text{pred} \cup \text{gt}|}$$

IoU = 1 is perfect overlap; IoU = 0 is no overlap. A detection is declared correct if IoU exceeds a threshold — 0.5 in PASCAL VOC (a fairly lenient standard), or averaged across thresholds 0.5:0.05:0.95 in COCO (a stricter standard that penalizes imprecise localization).

**Average Precision (AP)** for a single class: rank all detections by confidence score descending; at each rank position, compute precision (fraction of detections so far that are correct) and recall (fraction of all ground truth objects found so far). AP is the area under the precision-recall curve — it captures both whether the model finds objects (recall) and whether its confident predictions are correct (precision) across all possible confidence thresholds.

**Mean Average Precision (mAP)** averages AP across all classes. This normalizes for class frequency and detection difficulty differences across categories.

**The pattern in action:** A detector achieves mAP@0.5 of 0.80 but mAP@0.5:0.95 of 0.45. The large gap tells you the model finds objects in roughly the right location but does not localize them tightly — it gets the region right but not precise boxes. This matters for robotic grasping (needs tight boxes for reliable grip planning) but not for scene description (rough location is sufficient). Two metrics, same model, different deployment decisions.

**Common traps:**
- Reporting mAP without specifying the IoU threshold. mAP@0.5 and mAP@0.5:0.95 can differ dramatically, and comparisons across papers reporting different thresholds are invalid.
- Assuming high overall mAP means good performance on all object sizes. COCO breaks mAP into AP_S (small, area < 32²), AP_M (medium), AP_L (large, area > 96²). A model can have high overall mAP but terrible AP_S, which matters enormously for applications involving small objects — aerial surveillance, satellite imagery, microorganism detection.
- Treating mAP as sufficient when false positive and false negative costs are asymmetric. mAP is a symmetric metric; it does not capture whether your application cares more about missing objects (false negatives) or about raising false alarms (false positives).

---

## 8. Data Augmentation

**What the interviewer is testing:** Whether you understand augmentation as domain-knowledge-encoded regularization — not just a way to multiply training examples — and whether you can reason about which augmentations are label-preserving for a specific task.

**The reasoning structure:** Augmentation works by creating plausible variations of training examples that should produce the same label. Each augmentation encodes a specific invariance assumption about the problem. The key constraint is that augmentations must be **label-preserving** for the specific task — and this is domain-dependent:

- Horizontal flip is fine for ImageNet but wrong for digit recognition (a flipped "6" becomes a "9") and wrong for lateralized anatomy in medical images
- Aggressive color jitter is harmful when color is diagnostically relevant — pathology slides, food freshness assessment, skin lesion classification
- Rotation by 90° is appropriate for aerial imagery (no canonical orientation) but wrong for text recognition

Standard augmentations and the invariance each encodes:
- **Geometric (flip, rotation, crop, perspective):** spatial invariances — the class does not depend on exact orientation or cropping
- **Color (brightness, contrast, saturation, hue jitter):** photometric invariances — the class does not depend on exact lighting conditions
- **Cutout/CutMix:** occlusion robustness — teaches the model not to rely on any single distinctive patch; CutMix additionally interpolates labels proportional to the area swapped, which is a form of label smoothing
- **MixUp:** linear interpolation between two examples and their labels — $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$ — encourages smooth decision boundaries and reduces overconfidence

**The pattern in action:** A model for detecting manufacturing defects trained on 500 labeled images. Without augmentation: 72% recall at 90% precision. With geometric augmentations: 79% recall (camera angle variation). With geometric plus photometric: 83% recall (lighting variation). With MixUp added: 85% recall (appearance variation between defect types). Each augmentation type targets a specific source of variation in the imaging setup — the choice is diagnostic, not arbitrary.

**Common traps:**
- Applying augmentations without checking whether they are label-preserving for the specific task. This is the most important step and it is skipped constantly. The canonical failure: applying horizontal flip to digit recognition datasets because "flip is a standard augmentation."
- Treating more augmentation as always better. Aggressive augmentation on large datasets can damage fine-grained discriminative features — augmenting away the exact visual differences that distinguish similar classes.
- Not applying the same augmentation pipeline consistently between training and validation. Augmenting only training data is correct (the model should see varied training; evaluation should be clean). Applying training augmentations to the validation set inflates the difficulty of the evaluation.

---

## 9. Vision in Production

**What the interviewer is testing:** Whether you think beyond model architecture to the full system — specifically the failure modes that actually cause production problems, most of which are data distribution or pipeline issues, not model capacity issues.

**The reasoning structure:** A vision model that works in the notebook and fails in production almost always fails for one of a small number of reasons: the training distribution does not match the deployment distribution, preprocessing is inconsistent between training and serving, or the evaluation benchmark does not capture the hard cases the production system actually encounters.

Before deploying a vision system, these questions should be answered:
- How was training data labeled? What is the inter-annotator agreement? What are the edge case definitions in the labeling protocol?
- What camera hardware is used in production? Resolution, lens characteristics, color space, compression artifacts?
- What are the lighting conditions? Indoor, outdoor, mixed, artificial, time-of-day variation?
- What are the failure modes that matter? Are false positives or false negatives more costly?
- What is the monitoring plan? What signals detect when the model starts degrading?
- What is the rollback plan?

**The pattern in action:** A retail shelf-scanning robot is trained on images from high-quality research cameras in controlled lighting. Deployed with consumer-grade cameras in stores with mixed fluorescent lighting, accuracy drops 12 percentage points. The fix is retraining with images captured under production conditions — not architecture changes. The bottleneck was data distribution mismatch, not model capacity. This pattern repeats constantly in production computer vision systems: domain shift is the proximate cause; architecture optimization is irrelevant until the data distribution problem is addressed.

**Common traps:**
- Optimizing for benchmark metrics without evaluating performance on the specific visual categories or conditions relevant to the application. A model at 94% top-1 accuracy on ImageNet may have 60% accuracy on the objects relevant to the use case.
- Ignoring inference latency during architecture selection. A model taking 200ms per image is fine for batch processing but unusable for real-time video. This constraint should be established at the start of architecture search, not after.
- Neglecting preprocessing pipeline consistency. Training-serving skew in resizing interpolation method, normalization constants, or color channel ordering is a common cause of production performance drops that is completely invisible to offline evaluation. BILINEAR resizing in training and NEAREST in serving produces different input distributions to the model.

---

## Quick Diagnostics

**If asked to design a vision system for a new application:**
Anchor on task formulation first (classification, detection, segmentation), then latency and hardware constraints, then data situation and labeling cost, then error cost asymmetry. Choose architecture last. "What is your data situation and latency budget?" before "what architecture should we use?" signals strong engineering judgment.

**If asked about CNN vs Vision Transformer:**
CNNs encode locality and translation equivariance as hard architectural constraints — a useful prior when training data is limited. Vision Transformers learn spatial relationships from scratch via self-attention — appropriate when pretraining scale is large enough to amortize the cost of learning those priors from data. On ImageNet-1k trained from scratch, CNNs tend to outperform ViTs. At web-scale pretraining with hundreds of millions of image-text pairs, ViTs dominate. The tradeoff is data efficiency versus scalability.

**If asked why a model performs well on test but poorly in production:**
Enumerate the distribution shift sources: camera hardware differences, lighting differences, subject population differences, temporal drift in product appearances. Then enumerate pipeline consistency: same normalization? same resize interpolation? same color channel ordering? Offline accuracy gaps are data problems; pipeline bugs produce consistent systematic errors visible in specific input types.
