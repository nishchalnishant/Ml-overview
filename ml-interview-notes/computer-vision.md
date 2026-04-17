# Computer Vision

Computer vision is where ML learns to see.

And just like human styling, judging, or spotting quality, the trick is not only seeing pixels.
It is learning which visual details matter.

---

# 1. Why Images Need Special Treatment

An image is not just a long flat vector of numbers.

It has:

- spatial structure
- local relationships
- repeating patterns
- position-sensitive meaning

That is why image models need strong visual inductive bias.

If you ignore structure, the model has to learn everything the hard way.

Which is wasteful and usually disappointing.

---

# 2. Why CNNs Beat Plain MLPs on Images

CNNs are better suited to images because they build in two powerful assumptions:

- nearby pixels matter together
- the same pattern can appear in different locations

That is why convolutions use:

- local filters
- shared weights

instead of learning a separate parameter for every pixel relationship.

**Short answer**

CNNs work better than plain MLPs for images because they exploit locality and parameter sharing, which match the structure of visual data.

**Fashion analogy**

A stylist does not evaluate a look by memorizing every possible pixel arrangement.
They scan for recurring visual features:

- drape
- texture
- seam lines
- silhouette

CNNs do something similar.

---

# 3. Filters, Stride, and Padding

These are core CNN ideas.

## Filter

A small learned pattern detector.

Early filters often learn:

- edges
- corners
- textures

## Stride

How far the filter moves each step.

Larger stride:

- faster
- lower spatial resolution

## Padding

Extra border added so the filter can process edge regions without shrinking the image too quickly.

**Easy interview rule**

Filters detect.
Stride skips.
Padding protects borders.

---

# 4. Pooling

Pooling reduces spatial size while retaining strong signals.

Common versions:

- max pooling
- average pooling

Why use it:

- reduce compute
- build some translation tolerance
- focus on strongest activations

It is basically controlled compression.

---

# 5. Feature Extraction in Vision

Feature extraction in computer vision means turning raw pixels into representations that actually help the model reason.

Early layers learn simple details.
Later layers learn richer visual concepts.

**Fashion analogy**

Imagine examining a high-end outfit.

First you notice:

- fabric texture
- color
- sharpness of edges

Then:

- cut
- layering
- movement

Then:

- overall aesthetic
- occasion
- brand language

That is feature extraction in a nutshell.

---

# 6. Residual Connections and ResNet

As CNNs got deeper, training became harder.

Residual connections helped by adding shortcut paths:

- input goes forward
- transformed version also goes forward
- both are combined

That makes gradients flow more easily and allows much deeper networks to train well.

**Short answer**

Residual connections help very deep vision models train by making it easier to preserve and propagate information across layers.

---

# 7. CNN vs Vision Transformer

This is an increasingly common interview comparison.

## CNN

- strong visual inductive bias
- better data efficiency
- often stronger on smaller datasets

## Vision Transformer

- weaker built-in bias
- stronger global context modeling
- usually shines with large-scale pretraining

**Practical answer**

If data is limited, CNNs are often the safer starting point.
If pretraining scale is huge, ViTs can be extremely strong.

---

# 8. Image Classification vs Object Detection vs Segmentation

These are different tasks.

## Classification

What is in the image?

## Object Detection

What is in the image, and where?

## Segmentation

Which pixels belong to which class or object?

**Easy memory trick**

- classification = label
- detection = boxes
- segmentation = pixels

That one line is interview-friendly and very effective.

---

# 9. Object Detection: One-Stage vs Two-Stage

## One-Stage Detectors

Examples:

- YOLO
- SSD

They predict boxes and classes in one pass.

Pros:

- fast
- good for real-time use

## Two-Stage Detectors

Example:

- Faster R-CNN

They first propose candidate regions, then classify/refine them.

Pros:

- often stronger accuracy
- better for harder localization cases

**Short answer**

One-stage detectors trade some accuracy for speed; two-stage detectors trade speed for more precise detection.

---

# 10. IoU and mAP

## IoU

Intersection over Union measures how much a predicted box overlaps the true box.

Higher IoU means better localization.

## mAP

Mean Average Precision is the standard detection metric combining:

- precision-recall behavior
- class-wise performance
- localization quality through IoU thresholds

Do not overcomplicate this in an interview.

A strong answer is:

> "mAP summarizes detection quality by evaluating how accurately and consistently the model finds and localizes objects across classes."

That is clean and enough most of the time.

---

# 11. Data Augmentation in Vision

Vision models love augmentation, when used correctly.

Common types:

- flip
- crop
- rotate
- color jitter
- blur
- cutmix
- mixup

Why it helps:

- increases robustness
- improves generalization
- teaches useful invariances

But be careful.

Not every augmentation makes sense for every problem.

Flipping cat images is fine.
Flipping some medical imaging tasks or OCR pipelines may be a terrible idea.

---

# 12. OCR

OCR means Optical Character Recognition.

It turns text in images into machine-readable text.

Typical OCR pipeline:

1. detect text region
2. clean or align image
3. recognize characters/words
4. post-process

Used in:

- invoices
- IDs
- license plates
- scanned documents
- forms

---

# 13. Real-Time Tracking

Tracking is harder than it sounds.

The system must keep identity over time despite:

- occlusion
- motion blur
- lookalike objects
- missed detections
- camera shift

That is why tracking is not just "run detection on every frame."
Association logic matters too.

---

# 14. Vision in Production

Strong candidates mention more than architecture.

They ask:

- how was the data labeled?
- what are the slice failures?
- what happens under lighting shift?
- how does camera quality affect inference?
- what is the latency budget?
- how do we monitor drift?

That is the difference between "I know ResNet" and "I can ship a vision system."

---

# 15. Frameworks

Popular tooling includes:

- PyTorch
- TensorFlow / Keras
- OpenCV
- torchvision
- Detectron2

But this is rarely the main story.

Framework choice matters less than:

- data quality
- architecture fit
- deployment design

Same truth as most engineering, honestly.

---

# Quick Thought Experiment

You need a defect-detection model on a factory line with strict latency.

Would you prefer:

- a slower, heavier two-stage detector
- or a one-stage detector that is faster but slightly less precise

The answer depends on:

- cost of false negatives
- latency budget
- hardware constraints
- review workflow

That is the kind of tradeoff language interviewers love.

---

# How Would You Deploy This with Azure Pipelines?

For a vision model pipeline, I would validate:

- data version
- augmentation config
- model artifact version
- image preprocessing parity
- latency benchmark
- accuracy on critical slices
- rollback-ready previous version

Because a vision model that works beautifully in the notebook but collapses on real camera input is not a win.

It is just a glossy demo.
