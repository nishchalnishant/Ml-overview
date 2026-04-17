# Deep Learning for Computer Vision

Computer vision is what happens when deep learning learns to look at the world and decide which pixels are actually worth caring about.

The key challenge is not just seeing images.

It is learning:

- local structure
- spatial hierarchy
- invariance
- visual semantics

---

# 1. Main Vision Tasks

## Image Classification

What is in the image?

## Object Detection

What is in the image, and where?

## Segmentation

Which pixels belong to which object or class?

## Generative Vision

Can the model create or reconstruct plausible images?

That task split alone helps organize most vision discussions.

---

# 2. Why CNNs Were So Successful

CNNs work well because images have strong spatial structure.

They exploit:

- locality
- translation tolerance
- parameter sharing

That makes them much more natural for vision than flat dense networks.

**Fashion analogy**

A strong vision model examines an outfit the way a sharp stylist would:

- texture first
- silhouette next
- details and structure after that

That is hierarchical feature learning in couture form.

---

# 3. Vision Transformers

Vision Transformers treat image patches more like token sequences.

They are powerful because they model global relationships well.

But they often need:

- larger data
- stronger pretraining
- more compute

So the practical answer is:

- CNNs are often safer when data is limited
- ViTs shine when scale is available

---

# 4. Detection vs Segmentation Tradeoff

Detection gives boxes.
Segmentation gives pixel-level detail.

That means:

- detection is usually lighter
- segmentation is richer but heavier

Choosing between them depends on what the product actually needs.

Not what sounds cooler on a slide.
