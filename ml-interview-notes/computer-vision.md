# Computer Vision (CV)

This hub explores the core concepts of visual understanding, from convolutional inductive biases to modern Vision Transformers. Senior candidates are expected to bridge the gap between low-level pixel processing and high-level architectural decisions.

---

# 1. 🔹 Convolutional Neural Networks (CNNs)

## Q1: What makes CNNs inherently better for images than MLPs?

### 🔹 Direct Answer
CNNs use two key inductive biases that match the nature of images: **Locality** and **Translation Invariance**.
1. **Locality:** Pixels near each other are more related than distant pixels.
2. **Translation Invariance:** A "cat" is still a cat whether it is in the top-left or bottom-right corner.
Standard MLPs ignore spatial structure and lead to a parameter explosion ($H \times W \times C$ inputs). CNNs use **Weight Sharing**, where the same filter is applied across the whole image.

### 🔹 Intuition
Imagine looking for a specific needle in a haystack.
- **MLP Approach:** You memorize every possible position a needle could be in ($10,000$ positions).
- **CNN Approach:** You have a small magnet (Filter). You slide it over the whole haystack. It will find the needle wherever it is.

---

# 2. 🔹 Architectural Innovations

## Q2: How do Skip Connections (Residuals) solve the degradation problem?

### 🔹 Direct Answer
As neural networks get deeper, the gradients can "vanish" during backpropagation, causing accuracy to saturate or degrade. **Skip Connections** (introduced in ResNet) allow the gradient to flow directly through the identity mapping ($y = f(x) + x$).

### 🔹 Deep Dive
Mathematically, if the optimal mapping is an identity, it is easier for the residual block to push $f(x)$ to zero than to learn an identity from scratch. This allows for training networks with hundreds or thousands of layers.

---

# 3. 🔹 CNN vs. Vision Transformer (ViT)

## Q3: Compare CNNs and ViTs in terms of data efficiency.

### 🔹 Comparison Table

| Feature | CNN (e.g., ResNet) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Inductive Bias** | Strong (Locality/Invariance) | Weak (Global Attention) |
| **Data Requirements** | Lower (Learns well on small data) | Extreme (Needs huge pre-training) |
| **Scale Performance** | Plateaus earlier | Scales linearly with data/compute |
| **Context** | Local (limited by kernel size) | Global (every patch sees every patch) |

### 🔹 Practical Perspective
If you have a small dataset (e.g., 5,000 custom medical images), start with a **CNN**. If you have 100 million images (ImageNet-21k), a **ViT** will likely outperform the CNN by capturing complex, global relationships.

---

# 4. 🔹 Object Detection Paradigms

## Q4: One-Stage (YOLO) vs. Two-Stage (Faster R-CNN) Detectors.

### 🔹 Direct Answer
- **One-Stage (YOLO/SSD):** Treats detection as a single regression problem, predicting bounding boxes and class probabilities directly from the full image in one pass. It is **extremely fast** (Real-time).
- **Two-Stage (Faster R-CNN):** First proposes regions of interest (RPN) and then classifies/refines those regions. It is **more accurate** but much slower.

### 🔹 Comparison Table

| Feature | One-Stage | Two-Stage |
| :--- | :--- | :--- |
| **Speed** | Real-time (>60 FPS). | Slow (<10 FPS). |
| **Accuracy** | Lower (Higher localization error). | High (Better for small objects). |
| **Primary Use** | Robotics, Self-driving. | Satellite imagery, Medical analysis. |

---

# 5. 🔹 Evaluation Metrics

## Q5: How is mAP (mean Average Precision) calculated?

### 🔹 Direct Answer
**mAP** is the standard metric for object detection. It is calculated by:
1. Identifying **IoU (Intersection over Union)** between predicted and ground-truth boxes.
2. Setting a threshold (usually 0.5) to define a True Positive.
3. Calculating the **Average Precision (AP)** (Area under the Precision-Recall curve) for each class.
4. Taking the **Mean** of APs across all classes.

---

# 6. 🔹 Practical Perspective (Code)

### **Implementing a Skip Connection**
```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Identity mapping + Residual
        return self.relu(self.conv(x) + x)
```

---

## 🔹 Difficulty Tag: 🟡 Medium
