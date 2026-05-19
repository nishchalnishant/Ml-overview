# Computer Vision

---

## Table of Contents

1. CNN Building Blocks
2. Architecture Evolution
3. Skip Connections and Residual Learning
4. Object Detection (R-CNN → YOLO → DETR)
5. Semantic and Instance Segmentation
6. Vision Transformers (ViT, DeiT, Swin)
7. CLIP and Vision-Language Models (BLIP-2, LLaVA)
8. Image Generation: GANs, VAEs, Diffusion, Score Matching
9. Self-Supervised Learning (SimCLR, DINO, MAE)
10. Data Augmentation Strategies
11. Evaluation Metrics
12. Production Considerations
13. Common Interview Questions

---

## 1. CNN Building Blocks

### 1.1 Convolution Operation

**The problem**: a fully connected layer treating a 224×224 RGB image has 150,528 inputs per neuron. A single hidden layer of 1,000 neurons requires 150 million parameters — and the model treats pixel (0,0) and pixel (100,100) as completely unrelated, even though nearby pixels almost always share structure. This is both statistically wasteful and structurally wrong.

**The core insight**: natural images have translation-equivariant structure. An edge detector useful at the top-left is equally useful at the bottom-right. There is no reason to learn a separate detector for each spatial position — share the same weights everywhere.

**The mechanics**: a filter (kernel) of shape K×K×C slides over the input with stride s. At each position it computes a dot product with the local patch. The same filter weights are reused at every spatial position. Stacking F filters produces a feature map of shape H_out × W_out × F.

```
output[i, j] = sum_{m,n} input[i+m, j+n] * filter[m, n] + bias

H_out = floor((H_in + 2*pad - K) / stride) + 1
```

Parameter count for a conv layer: `K * K * C_in * C_out + C_out`.
A 3×3 conv 64→128 channels: 73,856 parameters. A fully-connected replacement on a 56×56 feature map: `56*56*64 * 56*56*128 = 1.4 billion`.

**What breaks**: weight sharing assumes translation equivariance — the same feature is equally useful everywhere. This is mostly true for images but fails badly on data with position-dependent structure (e.g., ECG signals where peak position matters). CNNs have no built-in rotation or scale invariance beyond what pooling provides, and they build global receptive fields only gradually through depth.

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 224, 224)
out = conv(x)   # (8, 64, 224, 224)
```

### 1.2 Activation Functions

**The problem**: stacking linear layers is still a linear transformation no matter how deep. Without nonlinearities, the whole network collapses to a single matrix multiply — depth buys nothing.

**The core insight**: the nonlinearity just needs to break linearity and let gradients flow. It does not need to saturate.

**The mechanics**:

- **ReLU**: `f(x) = max(0, x)`. Gradient is exactly 1 for positive inputs — no vanishing. Cheap to compute.
- **Leaky ReLU**: `f(x) = max(alpha*x, x)`, alpha ≈ 0.01. Prevents dead neurons that permanently output zero.
- **GELU**: `f(x) = x * Phi(x)` where Phi is the standard normal CDF. Smooth approximation to ReLU; used in Transformers and modern CNNs.

**What breaks**: ReLU neurons can permanently die if the bias drives their pre-activation negative for all training examples — they receive zero gradient forever (dead neuron problem). This is rare in practice with good initialization but more common with very high learning rates.

### 1.3 Pooling

**The problem**: after convolution, a small shift in the input produces a shifted feature map — the detector fires at a slightly different position. Classification must be robust to these small translations, but the conv layer is equivariant, not invariant.

**The core insight**: summarize over a local neighborhood rather than reporting exact position. Discard precise spatial information in exchange for local invariance.

**The mechanics**: max pooling takes the maximum over a k×k window, keeping the strongest activation regardless of exact position. Global Average Pooling (GAP) reduces each feature map to a single scalar — the average activation over the full spatial extent.

**What breaks**: pooling discards precise spatial information. This is fine for classification but harmful for localization tasks (detection, segmentation). Segmentation networks must reconstruct spatial resolution with upsampling.

```python
# GAP + classifier replaces flatten + large FC
gap = nn.AdaptiveAvgPool2d(1)
x = gap(feature_maps).flatten(1)   # (batch, channels)
logits = nn.Linear(channels, num_classes)(x)
```

### 1.4 Receptive Field

**The problem**: a single 3×3 conv layer can only see a 3×3 patch. Recognizing a face or a car requires integrating information across large regions. A network that cannot "see" the whole object cannot recognize it.

**The core insight**: stack layers. Each 3×3 layer adds 1 to the radius on each side. Two stacked 3×3 layers see a 5×5 region with fewer parameters and an extra nonlinearity compared to one 5×5 layer. Depth grows the receptive field without proportionally growing parameter count.

**The mechanics**: for a stack of L layers each with 3×3 kernels, the theoretical receptive field is 2L+1. Dilated convolutions insert gaps into the filter — a 3×3 filter with dilation d=2 covers a 5×5 region without adding parameters. Stacking dilated convolutions grows the receptive field exponentially.

**What breaks**: the *effective* receptive field — the region neurons actually depend on in practice — is much smaller than the theoretical one, roughly Gaussian-distributed around the center. Neurons do not automatically make equal use of their full theoretical receptive field. Very large receptive fields do not mean the model actually integrates that full range of information.

### 1.5 Feature Maps and Hierarchical Features

**The problem**: before deep learning, engineers spent years designing hand-crafted features (HOG, SIFT, LBP) for each visual task. Each new task or domain required new feature engineering — there was no generalization mechanism.

**The core insight**: if you train a deep network end-to-end on labeled data, the intermediate representations will automatically become useful features. Lower layers capture low-level patterns; upper layers capture semantically meaningful concepts. This hierarchy emerges from gradient descent alone.

**The mechanics**: each channel in a conv layer learns to detect a different pattern. Early layers detect edges, colors, and textures. Middle layers detect object parts. Late layers detect semantic concepts (eyes, wheels, windows). No programmer decided what each filter should detect.

**What breaks**: this hierarchy depends on large-scale supervised training. With limited data, upper-layer features may not form useful semantic representations. Transfer learning from ImageNet-pretrained models sidesteps this — the hierarchy has already been learned, and fine-tuning adapts it to the new task.

```python
# Grad-CAM: visualize which input regions drive a prediction
def grad_cam(model, x, target_class):
    activations = {}
    def hook(module, input, output):
        activations['feat'] = output
        output.retain_grad()

    handle = model.layer4.register_forward_hook(hook)
    out = model(x)
    handle.remove()

    model.zero_grad()
    out[0, target_class].backward()

    grads = activations['feat'].grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations['feat']).sum(dim=1, keepdim=True)
    cam = torch.nn.functional.relu(cam)
    cam = torch.nn.functional.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
    return cam.squeeze()
```

### 1.6 Depthwise Separable Convolution

**The problem**: standard convolution simultaneously mixes spatial filtering and channel mixing. This joint operation is parameter-inefficient, especially for mobile and edge hardware where every multiply-add counts.

**The core insight**: factorize. First filter spatially per channel (depthwise), then mix channels (pointwise). The two operations are independent enough that factorization loses little expressiveness while dramatically reducing cost.

**The mechanics**: a standard 3×3 conv with C input and C' output costs `3*3*C*C'` multiplications per spatial location. Depthwise separable splits this into depthwise (one 3×3 filter per input channel, cost `3*3*C`) then pointwise (1×1 across channels, cost `C*C'`). Total: `3*3*C + C*C'` — approximately 8–9× fewer operations when C is large.

**What breaks**: depthwise convolutions filter each channel independently and cannot capture cross-channel spatial patterns in a single operation. The pointwise convolution compensates, but the factorization is an approximation. For tasks requiring rich cross-channel spatial interactions early in the network, accuracy may degrade slightly versus standard convolution.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x
```

### 1.7 Batch Normalization

**The problem**: as gradients flow through deep networks, the distribution of each layer's inputs shifts with every parameter update. The next layer must continuously adapt to a moving target, slowing training. Higher learning rates amplify this instability.

**The core insight**: if you normalize each layer's activations to zero mean and unit variance, the input distribution stays stable. A learned scale (gamma) and shift (beta) let the network undo the normalization if the task requires it — so the normalization imposes no expressive constraint.

**The mechanics**: within a mini-batch, normalize per channel: `y = gamma * (x - mu_B) / sqrt(sigma_B^2 + eps) + beta`. At inference, use running statistics (exponential moving average of batch statistics) accumulated during training.

**What breaks**: BatchNorm fails when batch size is very small (1–2 samples) because batch statistics become too noisy to estimate reliably. It also behaves differently at training and inference — forgetting to call `model.eval()` before inference is a common source of accuracy drops. LayerNorm (normalizes over the feature dimension per sample) avoids the batch-size problem and is standard in Transformers.

```python
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU()
)
```

---

## 2. Architecture Evolution

### 2.1 LeNet-5 (1998)

**The problem it solved**: hand-designed feature pipelines for digit recognition were brittle and required expert knowledge. Could a network learn its own features from data?

LeNet-5 demonstrated that learned hierarchical convolution features could outperform hand-crafted ones on MNIST and postal code recognition. Structure: Conv → Pool → Conv → Pool → FC → FC → output. Limited to 32×32 grayscale inputs, but the concept was proven.

### 2.2 AlexNet (2012)

**The problem it solved**: ImageNet (1.2M images, 1,000 classes) was too large and complex for the feature engineering pipelines of its era — top-5 error was ~26%.

AlexNet reduced this to ~15% by scaling up convolution on GPUs. The key innovations were engineering, not architectural: ReLU (faster training than tanh), dropout in FC layers, data augmentation, and dual-GPU training. It proved that depth + data + compute beats hand-crafted features at scale.

### 2.3 VGG (2014)

**The problem it solved**: AlexNet used large kernels (11×11, 5×5). Were large kernels necessary?

**The core insight**: two 3×3 convolutions see the same 5×5 receptive field as one 5×5 convolution, but with fewer parameters (`2*9*C^2` vs `25*C^2`) and an extra nonlinearity. Depth from small kernels beats width from large kernels.

VGG-16 stacks 3×3 convolutions uniformly throughout. Its weakness: 138M parameters, most in FC layers — very memory-heavy.

### 2.4 Inception / GoogLeNet (2014)

**The problem it solved**: the optimal filter size is unknown ahead of time. A single filter size per layer may miss important patterns at other scales.

**The core insight**: apply multiple filter sizes in parallel and let the network decide what to use.

**The mechanics**: the Inception module applies 1×1, 3×3, 5×5 convolutions and 3×3 max pooling in parallel, then concatenates outputs along the channel dimension. 1×1 convolutions before larger filters act as bottlenecks, dramatically reducing channel count before the expensive operations. GoogLeNet achieves 4M parameters versus VGG's 138M.

### 2.5 ResNet (2015)

See Section 3 for full treatment. First model to surpass human-level top-5 accuracy on ImageNet (3.57% vs 5.1%). Enabled training of 50-, 101-, and 152-layer networks that were previously untrainable due to optimization failure.

### 2.6 DenseNet (2017)

**The problem it solved**: ResNet's skip connections go one block back. Features computed in early layers are still discarded by the time they reach the final classifier.

**The core insight**: connect every layer to every subsequent layer. Each layer receives feature maps from all preceding layers via concatenation.

```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

Low-level features (edges, colors) remain directly accessible to the final classifier — they do not have to be re-learned or pass unchanged through intermediate transformations. Transition layers (1×1 conv + 2×2 avg pool) reduce spatial size between dense blocks.

**What breaks**: memory grows quadratically with depth due to concatenation. DenseNet is harder to scale to very deep networks than ResNet.

### 2.7 EfficientNet (2019)

**The problem it solved**: when you have more compute budget, should you make the network deeper, wider, or use higher-resolution inputs? Prior work scaled these dimensions independently and arbitrarily.

**The core insight**: all three dimensions — depth, width, resolution — should be scaled together in a fixed ratio. Scaling only one dimension saturates quickly.

**The mechanics**: NAS finds a baseline architecture. Compound scaling multiplies depth, width, and resolution by `alpha^phi`, `beta^phi`, `gamma^phi` respectively, subject to `alpha * beta^2 * gamma^2 ≈ 2`. The coefficient `phi` controls the total scaling budget. EfficientNet-B7 achieves 84.3% top-1 with 8× fewer parameters than prior best models.

### 2.8 ConvNeXt (2022)

**The problem it solved**: Vision Transformers were outperforming CNNs on ImageNet — but was that because of the attention mechanism, or because of the training recipe and design choices (larger kernels, fewer normalization layers, GELU, etc.)?

ConvNeXt modernizes ResNet-50 by adopting ViT's design principles without attention: 7×7 depthwise conv, inverted bottleneck, GELU, LayerNorm, fewer activation/norm layers. It matches or exceeds ViT at equal compute with simpler training. The answer: much of ViT's gain came from design decisions, not attention itself.

### 2.9 Architecture Selection Guide

| Scenario | Recommended |
|---|---|
| Edge / mobile | MobileNetV3, EfficientNet-B0 |
| Moderate compute, high accuracy | ResNet-50, EfficientNet-B4 |
| Maximum accuracy | EfficientNet-B7, ConvNeXt-XL, ViT-L |
| Fine-tuning from ImageNet | ResNet-50 (stable, well-understood) |
| Detection backbone | ResNet + FPN, EfficientDet |

---

## 3. Skip Connections and Residual Learning

### 3.1 The Degradation Problem

**The problem**: as networks get deeper, training error increases — not from overfitting, but from optimization failure. A 56-layer plain network performs worse than a 20-layer one on CIFAR-10 training data.

This should not happen. A 56-layer network can always match a 20-layer one by setting the extra 36 layers to identity. But gradient-based optimization cannot find that solution. The signal that "identity is the right thing to do here" cannot propagate back 56 layers because gradients vanish through repeated matrix multiplications.

### 3.2 Residual Block

**The core insight**: reformulate what the layers must learn. Instead of asking them to learn the full transformation H(x) from scratch, ask them to learn the *residual* F(x) = H(x) - x. The identity x bypasses the transformation and is added back: `y = F(x) + x`.

Learning F(x) ≈ 0 (identity) is easy — initialize weights near zero and the block immediately functions as an identity. Learning H(x) ≈ x from scratch across a deep stack is hard — the optimization must find the right parameter combination through many nonlinear layers.

**The mechanics**: a direct path (skip connection) adds the input x to the layer's output. The gradient now has two paths back through this block: through F(x), and directly through the addition with gradient 1.

```
dL/dx = dL/dy * (dF/dx + I)
```

The identity term I means gradients never vanish even if dF/dx is small. In practice, ResNets train reliably at hundreds of layers.

**What breaks**: skip connections require the input and output of the residual block to have the same shape. When downsampling (stride=2) or changing channel count, a 1×1 projection convolution is used in the skip path.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
```

### 3.3 Bottleneck Block (ResNet-50+)

**The problem**: two 3×3 convolutions per block on wide feature maps is too expensive for very deep networks.

A bottleneck compresses channels first, then expands: 1×1 (256→64), 3×3 (64→64), 1×1 (64→256). Parameter count: 69,632 versus 1,179,648 for two 3×3 convolutions on 256-channel feature maps — a 17× reduction.

### 3.4 Why Residual Learning Works

Three complementary views:

1. **Gradient flow**: direct paths carry gradients from the loss to early layers without multiplicative decay.
2. **Ensemble of paths**: a ResNet implicitly contains exponentially many paths of different lengths from input to output. At test time it behaves like an ensemble of these paths.
3. **Loss landscape**: residual connections smooth the loss surface, making it less likely for optimization to get stuck in sharp local minima or flat plateaus.

---

## 4. Object Detection

### 4.1 Problem Definition

**The problem**: classification asks "what is in the image?" Detection asks "what is in the image, and where?" The where requires outputting bounding boxes as continuous coordinates — a regression problem layered on top of classification, with an unknown number of outputs per image.

### 4.2 Two-Stage Detectors: R-CNN Family

**The core insight**: separate the problem into two stages. First decide *where* there might be objects (region proposals). Then decide *what* is in each proposed region.

**R-CNN (2014)**: extract ~2,000 region proposals with Selective Search, warp each to fixed size, run CNN on each independently. Slow (~50s/image) because each region is processed separately.

**Fast R-CNN (2015)**: run CNN on the full image once. Project region proposals onto the feature map. ROI Pooling extracts a fixed-size feature for each proposed region — all regions share the convolution computation. Speed: ~0.3s/image. The new bottleneck is Selective Search (CPU-bound).

**Faster R-CNN (2016)**: replace Selective Search with a Region Proposal Network (RPN) — a small network that slides over the shared feature map and proposes boxes at each location. The entire pipeline now shares one backbone CNN and is trained end-to-end.

**What breaks**: two-stage detectors are slower than one-stage because of the explicit proposal generation and two classification passes. For real-time applications this is prohibitive.

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = [torch.randn(3, 800, 600)]
with torch.no_grad():
    predictions = model(image)
# predictions[0]: {'boxes', 'labels', 'scores'}
```

**Feature Pyramid Network (FPN)**: small objects vanish in deep feature maps. FPN builds a multi-scale pyramid by combining high-resolution early features (spatial detail) with low-resolution late features (semantic content) via lateral connections. Detection heads run at every scale.

### 4.3 Anchor Boxes

**The problem**: the network must predict bounding boxes as continuous coordinates. Predicting raw coordinates from scratch is an ill-conditioned regression — the targets span wildly different scales and aspect ratios, making the optimization landscape difficult.

**The core insight**: give the network reference points. At each spatial location in the feature map, predefine a set of boxes (anchors) at different scales and aspect ratios. The network predicts *offsets* from these anchors. The regression target is now a small, bounded correction rather than an absolute coordinate.

**What breaks**: anchors are hyperparameters — their scales and ratios must be tuned per dataset. Anchor-free detectors (CenterNet, FCOS) eliminate this by predicting object centers and extents directly.

### 4.4 IoU, NMS, and Focal Loss

**IoU** measures box overlap: `Area(Intersection) / Area(Union)`. Used to match predictions to ground truth during training and to filter redundant predictions during inference.

**NMS (Non-Maximum Suppression)**: the same object typically produces dozens of overlapping predictions. Sort by confidence, keep the highest, suppress all others with IoU above a threshold. Soft-NMS decays rather than removes overlapping boxes — better for heavily occluded scenes.

**Focal Loss** addresses class imbalance in one-stage detection. With ~100,000 anchors per image but fewer than 1,000 objects, more than 99% of anchors are background. Standard cross-entropy is dominated by easy background examples — the model gets little useful gradient from the objects it should be learning to detect.

**The core insight**: down-weight easy examples. An example the model classifies correctly with probability p_t contributes little information — down-weight its loss contribution. The factor `(1-p_t)^gamma` approaches 0 for well-classified examples and 1 for hard ones.

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

With gamma=2, a correctly-classified example with p_t=0.9 has its loss reduced by 100×. Hard examples with p_t=0.1 are barely affected.

### 4.5 One-Stage Detectors

**The core insight**: if the two-stage bottleneck is the proposal stage, eliminate it. Predict class and box directly from a dense grid, in a single forward pass.

**YOLO v1 (2016)**: divide the image into an S×S grid. Each cell predicts B boxes and C class scores directly. Single forward pass: 45 FPS. Trades some accuracy for speed.

**YOLO v3–v8**: multi-scale prediction (FPN-style), better anchor selection, mosaic augmentation. YOLOv8 is anchor-free with a decoupled head for classification and regression.

**What breaks**: one-stage detectors historically struggled with small objects and dense scenes because the grid forces each cell to detect at most one object.

### 4.6 DETR: Detection as Set Prediction

**The problem**: anchor-based and grid-based detectors require hand-designed priors (anchor scales/ratios, grid resolution) and post-processing (NMS). These are not learned — they encode human assumptions about object distributions.

**The core insight**: reformulate detection as a set prediction problem. Given an image, directly predict the set of (class, box) pairs — no anchors, no NMS.

**The mechanics**: DETR runs the image through a CNN backbone, flattens the feature map into a sequence, and passes it through a Transformer encoder. A Transformer decoder takes N learned object queries and attends to the encoder output. Each query independently produces one predicted class + box. Hungarian matching assigns each query to a ground truth object (or "no object") during training, providing an unambiguous loss signal.

**What breaks**: DETR is slow to converge (~500 epochs versus ~12 for Faster R-CNN). The quadratic attention cost over high-resolution feature maps limits small-object detection. Deformable DETR fixes convergence by replacing full attention with sparse deformable attention — each query attends to a small set of learned spatial positions rather than all positions.

---

## 5. Semantic and Instance Segmentation

### 5.1 From Classification to Dense Prediction

**The problem**: classification outputs one label per image. Segmentation needs a label for every pixel. A standard CNN progressively reduces spatial resolution — by the end, a 224×224 input becomes a 7×7 feature map. Recovering pixel-level predictions from this 32× downsampled representation requires reconstruction that standard architectures were not designed for.

### 5.2 Fully Convolutional Network (FCN, 2015)

**The core insight**: replace fully connected layers with 1×1 convolutions. The network then outputs a spatial feature map that can be upsampled to the input resolution.

Transposed convolutions (deconvolution) upsample by learning to distribute each input value across a larger output region. Skip connections from earlier (higher-resolution) layers restore spatial detail lost through pooling.

**What breaks**: FCN's upsampling is coarse — large pooling strides discard fine spatial structure that is expensive to recover. U-Net addresses this with symmetric skip connections.

### 5.3 U-Net (2015)

**The core insight**: the encoder compresses spatial information into semantics; the decoder must reconstruct spatial information. Every encoder layer retains information the corresponding decoder layer will need — connect them directly with skip connections.

**The mechanics**: the encoder halves spatial size at each stage. The decoder upsamples at each stage and concatenates (not adds) the corresponding encoder feature map. Concatenation preserves all channels from both paths. The bottleneck contains the most abstract semantic information; the skip connections carry fine spatial detail.

**What breaks**: skip connections double the number of channels at each decoder stage, increasing memory. For very large images (medical imaging at full resolution), memory becomes the bottleneck.

```python
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
```

### 5.4 DeepLab and Atrous Convolution

**The problem**: U-Net reconstructs resolution by upsampling. But what if we never lost it in the first place? Strided pooling discards spatial detail — can we build large receptive fields without striding?

**The core insight**: insert gaps into the filter kernel. An atrous (dilated) convolution with dilation rate r spaces filter elements r pixels apart. The receptive field grows to `(K-1)*r + 1` without adding parameters and without striding.

ASPP (Atrous Spatial Pyramid Pooling) applies dilated convolutions at multiple rates in parallel. This captures context at multiple scales simultaneously — important for segmenting objects of vastly different sizes in the same image.

### 5.5 Mask R-CNN (2017) — Instance Segmentation

**The problem**: semantic segmentation cannot distinguish between separate instances of the same class. Two touching cars both become a single "car" region with no separation between them.

**The core insight**: detect each object first (Faster R-CNN), then predict a mask for each detected instance. Treat each detection as a mini-segmentation problem.

**The mechanics**: extend Faster R-CNN with a third parallel head that outputs a K×K binary mask for each of the K classes. ROIAlign replaces ROI Pooling — instead of quantizing box coordinates to integer grid positions, it uses bilinear interpolation to extract features at exact floating-point coordinates. This eliminates spatial misalignment, which is critical for pixel-accurate mask prediction.

**What breaks**: Mask R-CNN requires detected bounding boxes first — it cannot segment objects that the detector misses. Heavily occluded objects may not be detected and therefore receive no mask.

---

## 6. Vision Transformers (ViT)

### 6.1 Core Architecture

**The problem**: CNNs have strong inductive biases (locality, translation equivariance) that help with limited data but constrain what they can learn. Long-range dependencies — a pixel's relationship with another on the opposite side of the image — require many layers to build up through the receptive field. This is a structural bottleneck the CNN architecture cannot escape.

**The core insight**: treat an image as a sequence of patches and apply the Transformer directly. Self-attention is O(N^2) in sequence length but gives every token direct access to every other token from layer 1.

**The mechanics**: split an H×W×C image into N = HW/P^2 patches of size P×P×C. Linearly project each patch to dimension D (patch embedding). Add learnable positional encodings (Transformers have no built-in spatial structure). Prepend a learnable [CLS] token. Run L Transformer encoder layers. Use the [CLS] output for classification.

**What breaks**: ViT has no inductive biases. Without large-scale pretraining, it underperforms CNNs on ImageNet-scale data (1.2M images) — it must learn from scratch that nearby patches are related, which CNNs get for free. ViT requires JFT-300M (300M images) or ImageNet-21K (14M images) to match CNN performance.

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, d_model=768, depth=12, heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim   = in_channels * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model*4,
                                       dropout=0.1, batch_first=True),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, H//P, P, W//P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*P*P)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])
```

### 6.2 DeiT: Data-Efficient ViT

**The problem**: ViT requires massive pretraining data that most practitioners do not have access to.

**The core insight**: use knowledge distillation from a CNN teacher. The student ViT learns from the teacher's soft predictions — not just hard labels — and a special distillation token attends specifically to the teacher's output.

DeiT trains ViT-Base on ImageNet-1K alone (no external data) and achieves 81.8% top-1, matching the original ViT-B that required JFT-300M pretraining.

### 6.3 Swin Transformer (2021)

**The problem**: ViT's O(N^2) attention cost makes it impractical for high-resolution images or dense prediction tasks (detection, segmentation). Also, ViT produces a single-resolution feature map — unusable as a backbone for FPN-style multi-scale detection.

**The core insight**: compute self-attention within local non-overlapping windows of size M×M. Attention complexity drops from O(N^2) to O(M^2 * N) — linear in image size. Between layers, shift the window partition so different windows interact (shifted windows). Add hierarchical patch merging (like CNN striding) to produce multi-scale representations.

**What breaks**: window attention cannot capture dependencies across distant windows in a single layer. It takes multiple shifted-window layers for information to propagate globally. Large objects spanning many windows require many layers to integrate.

---

## 7. CLIP and Vision-Language Models

### 7.1 CLIP (2021)

**The problem**: supervised ImageNet training requires expensive human-labeled data and fixes the set of categories at training time. Classifying a new category requires collecting new labels and retraining. There is no mechanism for zero-shot generalization to unseen categories.

**The core insight**: the internet contains billions of image-text pairs where the text describes the image. Train an image encoder and text encoder jointly so that matching (image, text) pairs have similar embeddings and non-matching pairs have dissimilar embeddings. Now the text encoder defines categories — you can classify anything expressible in language.

**The mechanics**: for a batch of N image-text pairs, the contrastive objective maximizes the diagonal of the N×N similarity matrix (matching pairs) and minimizes the off-diagonal (non-matching pairs):

```
L = -1/(2N) * [sum_i log(sim(v_i, t_i)/tau) + sum_i log(sim(t_i, v_i)/tau)]
```

where `sim(v,t) = v^T t / (||v|| ||t||)` and tau is a learned temperature.

Zero-shot classification: encode candidate class names as text, compute cosine similarity to the image embedding, softmax over classes.

**What breaks**: CLIP embeddings are not as accurate as fine-tuned classification models on in-domain data. Performance degrades on categories underrepresented in training (medical imaging, abstract art). Prompt engineering matters significantly — "a photo of a {class}" substantially outperforms just "{class}".

```python
import clip
from PIL import Image
import torch

model, preprocess = clip.load("ViT-B/32", device="cuda")

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to("cuda")
texts = clip.tokenize(["a cat", "a dog", "a car"]).to("cuda")

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(texts)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
```

### 7.2 BLIP-2 and LLaVA: Multimodal LLMs

**The problem**: CLIP produces visual embeddings aligned with text, but cannot generate language. Connecting vision to a language model enables visual question answering, captioning, and visual reasoning — but both the vision encoder and LLM are already enormous and expensive to train.

**The core insight (BLIP-2)**: freeze both the vision encoder and the LLM. Train only a lightweight bridge between them. This makes the problem tractable — both pretrained components are leveraged and only the connector is trained.

BLIP-2's bridge is a **Q-Former** (Querying Transformer): N learned query embeddings (e.g., 32) use cross-attention to extract task-relevant information from the frozen image encoder's output. These N visual tokens are prepended to the LLM's input as a soft prompt. Only the Q-Former (~188M parameters) is trained.

**The core insight (LLaVA)**: the Q-Former adds complexity. Can a single linear projection work? LLaVA maps CLIP's visual tokens directly into the LLM's embedding space with a single linear layer, then instruction-tunes the LLM on GPT-4-generated visual QA data. Despite architectural simplicity, LLaVA performs strongly — suggesting that instruction tuning data quality matters more than projection complexity.

**What breaks**: both architectures inherit the LLM's tendency to hallucinate. The model may confidently describe image content that is not there. Visual grounding remains an active research problem.

---

## 8. Image Generation: GANs, VAEs, Diffusion

### 8.1 Generative Adversarial Networks (GANs)

**The problem**: how do you train a generator to produce realistic images when there is no ground-truth target for each noise sample? Directly maximizing likelihood over image pixels is intractable in high dimensions.

**The core insight**: train an adversary (discriminator) to distinguish real from generated images. The generator's goal is to fool the discriminator. This creates a minimax game:

```
min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

The discriminator provides a training signal to the generator without requiring explicit likelihood computation.

**What breaks**: GAN training is notoriously unstable. Mode collapse — the generator converges to a few high-quality modes while ignoring the rest of the data distribution — is the primary failure mode. If the discriminator becomes too confident, it provides near-zero gradient to the generator. Techniques like gradient penalty (WGAN-GP), spectral normalization, and progressive growing stabilize training but require careful tuning.

### 8.2 Variational Autoencoders (VAEs)

**The problem**: standard autoencoders compress to a point in latent space. The latent space is unstructured — interpolating between two encodings does not produce meaningful intermediate images, because the space between learned codes is unexplored.

**The core insight**: instead of encoding to a point, encode to a *distribution* (mean and variance of a Gaussian). Force this distribution to stay close to a standard normal prior via KL divergence. The latent space becomes smooth and regularized — nearby points decode to similar images.

**The mechanics**: the ELBO loss balances reconstruction quality against latent regularization:

```
ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

The reparameterization trick `z = mu(x) + sigma(x) * epsilon, epsilon ~ N(0,I)` makes the sampling step differentiable.

**What breaks**: optimizing the ELBO causes blurry reconstructions. The reconstruction term treats each pixel independently under a Gaussian, which means blurring over uncertainty is cheaper than predicting a sharp but slightly wrong image. VAEs produce blurrier outputs than GANs for this reason.

### 8.3 Diffusion Models

**The problem**: GANs are unstable and mode-prone. VAEs are blurry. Is there a stable training procedure that produces sharp, diverse images?

**The core insight**: learn to denoise. If you progressively corrupt an image with Gaussian noise until it becomes pure noise, then train a network to reverse each small denoising step, you implicitly learn the data distribution. Denoising is a well-posed, stable supervised regression — it requires no adversary.

**The mechanics**: the forward process adds noise over T steps:
```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)
```

You can sample any intermediate step in closed form:
```
x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon,  epsilon ~ N(0,I)
```

The network is trained to predict the noise that was added:
```
L = E[||epsilon - epsilon_theta(x_t, t)||^2]
```

At generation time, start from Gaussian noise and run the reverse process T times.

**Score matching view**: the noise-prediction network is mathematically equivalent to estimating the score (gradient of log probability) of the noisy data distribution. Generation is gradient ascent in probability space with injected noise (Langevin dynamics). DDIM makes this deterministic and enables ~50-step generation instead of 1,000.

**What breaks**: sampling requires T forward passes through the network (T=1,000 for DDPM, T=20–50 for DDIM). This is 20–1,000× slower than a single GAN forward pass. Latent diffusion (Stable Diffusion) compresses the image into a small latent space first, running diffusion there — much faster while preserving quality.

---

## 9. Self-Supervised Learning for Vision

### 9.1 The Problem

**The problem**: labeling ImageNet took years and millions of dollars. The internet has billions of unlabeled images. Supervised pretraining caps out at what annotators have labeled. Can we extract useful representations from unlabeled images alone?

### 9.2 SimCLR (2020)

**The core insight**: two different augmented views of the same image should produce similar representations. Two different images should produce dissimilar representations. Define "similar" and "dissimilar" without any human labels — the augmentation pairs define the positives.

**The mechanics**: for each image, create two random augmentations (crop + color jitter + blur). Encode both through the same network. Maximize cosine similarity between same-image embeddings while minimizing similarity to all other images in the batch (NT-Xent loss).

**What breaks**: SimCLR requires very large batch sizes (4,096–8,192) to have enough negatives. Without many hard negatives, the model collapses — all embeddings converge to the same point.

```python
def simclr_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, labels)
```

### 9.3 MoCo (2020)

**The problem**: SimCLR's large batches are memory-intensive. Can we decouple the number of negatives from batch size?

**The core insight**: maintain a queue of recent embeddings as negatives. Use a momentum-updated "key" encoder to produce stable queue embeddings — the queue does not require backpropagation, only the online encoder does.

```
theta_k = m * theta_k + (1-m) * theta_q   (m=0.999)
```

The large queue provides many negatives without requiring a large batch.

### 9.4 DINO (2021)

**The core insight**: student and teacher, where the teacher is an exponential moving average of the student. The student processes local (small) crops; the teacher processes global (large) crops. The student must predict the teacher's representation from less context — this forces semantically meaningful representations.

**What emerges without supervision**: ViT trained with DINO produces attention heads that naturally segment objects. The network learns that a dog and its background are different things, and that "dog" should be one coherent region — without a single segmentation label.

### 9.5 MAE (Masked Autoencoders, 2022)

**The core insight**: masking 75% of image patches and predicting the masked patches from the visible 25% forces the model to understand global image structure. Predicting a missing patch requires understanding the scene — you cannot copy from nearby context when 75% is masked.

**The mechanics**: the encoder sees only the 25% visible patches (fast, because most tokens are excluded). A lightweight decoder takes the encoder's representations plus learned mask tokens and reconstructs raw pixel values of the masked patches.

**What breaks**: MAE representations are optimized for pixel reconstruction, which does not perfectly align with high-level semantic tasks. Fine-tuning on downstream tasks is required for competitive performance.

---

## 10. Data Augmentation Strategies

### 10.1 The Problem

**The problem**: a model trained only on original images memorizes specific appearances. It fails when test images are slightly different — different crop, different color balance, slightly occluded. Augmentation artificially increases dataset diversity to force robustness to these variations.

### 10.2 Standard Augmentations

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Random cropping with resize is the most impactful single augmentation — it forces the model to classify from partial views, preventing positional memorization.

### 10.3 Mixup

**The core insight**: the decision boundary between two classes should be smooth, not a hard discontinuity. Interpolating images and labels creates training examples along the boundary, explicitly regularizing the boundary region.

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**What breaks**: blended images are unrealistic — no natural image looks like two superimposed images. This can hurt if the model needs to learn realistic low-level texture features.

### 10.4 CutMix

**The core insight**: paste a realistic rectangular region from one image into another, rather than blending pixel values. Labels are proportional to the area of each image's contribution. The model learns to classify from local evidence — a patch of fur is enough to predict "cat," even without the full animal.

**What breaks**: cut patches can misalign semantics — pasting a car wheel onto a cat image with label "0.3 car, 0.7 cat" creates a confusing training signal if the wheel does not carry enough diagnostic information.

### 10.5 AutoAugment and RandAugment

**The problem**: standard augmentations are human-designed. Is the human's choice optimal for each dataset?

AutoAugment uses reinforcement learning to search for the best augmentation policy. RandAugment simplifies this — uniformly sample N operations and magnitudes from a predefined set, with only two hyperparameters (N and M). TrivialAugment goes further: uniformly sample one operation at a random magnitude. Despite its simplicity, TrivialAugment outperforms RandAugment on ImageNet.

---

## 11. Evaluation Metrics

### 11.1 Classification

**Top-1 accuracy**: fraction where the highest-probability class is correct.
**Top-5 accuracy**: fraction where the correct class appears in the top-5 predictions.

### 11.2 mAP for Detection

**The problem**: a detector outputs boxes with continuous confidence scores. A single number must capture both precision and recall across all confidence thresholds.

Average Precision (AP) for one class: plot precision versus recall as the confidence threshold varies; integrate the area under this curve. mAP averages AP over all classes.

COCO mAP averages over IoU thresholds 0.50:0.05:0.95 — more stringent than the earlier VOC standard (IoU=0.50 only). A model that localizes loosely scores worse under COCO mAP even if it identifies the class correctly.

### 11.3 Segmentation

**mIoU**: per-class IoU averaged over all classes.

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = mean over c of IoU_c
```

Pixel accuracy is misleading when classes are imbalanced — a model predicting only "background" on a street scene might achieve 90% pixel accuracy.

### 11.4 Generation

**FID (Frechet Inception Distance)**: compare the distribution of Inception features from real and generated images via Frechet distance between their Gaussian approximations. Lower is better. Captures both quality (sharpness) and diversity (variance). Requires ~10k generated samples for stable estimates.

**CLIP Score**: cosine similarity between CLIP embeddings of generated images and their text prompts. Used for text-conditional generation quality.

---

## 12. Production Considerations

### 12.1 Quantization

**The problem**: FP32 inference (4 bytes/weight) is memory-intensive and slow. INT8 quantization reduces to 1 byte/weight and activates faster integer hardware units.

**The mechanics**: calibrate per-channel scales using a small representative dataset, then convert weights and activations to INT8 at inference. Post-Training Quantization (PTQ) requires only ~100 calibration images. Quantization-Aware Training (QAT) simulates quantization during training and recovers more accuracy, but requires the full training loop.

**What breaks**: some operations (LayerNorm, softmax) are sensitive to quantization and may remain FP32 in mixed-precision inference. INT8 speedup only materializes on hardware with INT8 support (NVIDIA Tensor Cores, Apple Neural Engine).

### 12.2 Pruning

Remove weights below a magnitude threshold (unstructured pruning) or remove entire filters/channels (structured pruning). Structured pruning is hardware-friendly — it genuinely reduces the network width and produces measurable speedup without specialized sparse kernels.

### 12.3 Knowledge Distillation

**The core insight**: a small student model trained on the soft probability distribution from a large teacher learns more than one trained on hard one-hot labels. The teacher's near-zero probabilities for incorrect classes carry information about similarity structure — "this looks a bit like a cat but is definitely a dog" — that one-hot labels discard.

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    soft_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
        torch.nn.functional.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * temperature ** 2
    hard_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 12.4 Deployment

ONNX export enables cross-runtime deployment (TensorRT, OpenVINO, CoreML, ONNX Runtime). TensorRT further optimizes for NVIDIA GPU inference (kernel fusion, precision calibration). Always validate accuracy on the target hardware — quantization and compilation can introduce subtle accuracy regressions.

---

## 13. Common Interview Questions

---

**Q: Why do CNNs work so well for images compared to fully connected networks?**

Three properties of natural images match CNN inductive biases: (1) **locality** — nearby pixels share structure; processing local patches costs far fewer parameters than processing all pixels jointly; (2) **translation equivariance** — the same feature detector is useful at every spatial position; one learned filter handles the entire image; (3) **compositionality** — low-level features compose into higher-level ones layer by layer. A fully connected layer on a 224×224 image needs 150M parameters for a single hidden layer; a single 3×3 conv layer needs 64×3×3×3 = 1,728.

---

**Q: Explain how ResNet's skip connections help training.**

Without skip connections, backpropagating through 100 layers multiplies gradients by 100 weight matrices — if typical singular values are less than 1, the gradient signal vanishes exponentially before reaching early layers. ResNet's `y = F(x) + x` creates two gradient paths: through the residual F(x) and directly through the identity with gradient 1 regardless of depth. Early layers always receive at least a portion of the loss gradient. The skip connections also make it easy to learn identity mappings — the optimization target is "make a small correction" rather than "learn the full transformation from nothing."

---

**Q: What is the difference between semantic segmentation and instance segmentation?**

Semantic segmentation assigns a class label to every pixel but cannot distinguish between instances. Two overlapping cars both become "car" pixels with no separation. Instance segmentation assigns a unique identity to each object — each car gets its own mask. Mask R-CNN achieves this by running Faster R-CNN to detect individual objects, then predicting a per-instance binary mask. Panoptic segmentation unifies both: semantic labels for amorphous regions (sky, road) and instance masks for countable objects (people, cars).

---

**Q: How does the Vision Transformer differ from a CNN?**

ViT splits an image into P×P patches and processes them as a sequence with Transformer self-attention. Key differences: (1) **No spatial inductive bias** — ViT must learn from data that nearby patches are more correlated than distant ones; CNNs get this for free from local connectivity; (2) **Global receptive field from layer 1** — every patch attends to every other patch immediately; CNNs build global receptive fields gradually through depth; (3) **Data efficiency** — CNNs require less data because their inductive biases match image statistics; ViT needs massive pretraining; (4) **Quadratic attention cost** — O(N^2) in number of patches.

---

**Q: Explain focal loss and why it was needed.**

One-stage detectors generate ~100,000 anchors per image but at most a few hundred overlap with objects. More than 99% of anchors are background. With standard cross-entropy loss, easy background examples collectively dominate the gradient, preventing the model from learning from the rare hard examples. Focal loss adds the factor `(1-p_t)^gamma`, where p_t is the model's confidence for the correct class. For an easy example with p_t=0.9, this factor is (0.1)^2 = 0.01 — reducing its contribution by 100×. Hard examples with p_t=0.1 have factor (0.9)^2 = 0.81 — nearly unchanged.

---

**Q: Compare GANs, VAEs, and Diffusion models for image generation.**

**GANs**: adversarial training produces sharp images but suffers from training instability and mode collapse. Fast sampling (single forward pass). Hard to evaluate.

**VAEs**: stable training via ELBO maximization. Smooth, structured latent space enables interpolation and controlled generation. Images are blurry because the reconstruction loss averages over uncertainty pixel-by-pixel.

**Diffusion models**: stable training via noise prediction. Produce the sharpest and most diverse images. Slow sampling (hundreds of steps, though DDIM reduces to 20–50). Have largely replaced GANs for quality-critical generation tasks.

---

**Q: How would you adapt a pretrained ImageNet model to medical imaging with limited labeled data?**

1. Use the pretrained backbone — ImageNet features (edges, textures, shapes) generalize to medical images better than random initialization.
2. Progressive unfreezing: train the classifier head first, then unfreeze deeper layers gradually with lower learning rates for earlier layers (discriminative fine-tuning).
3. Domain augmentation: stain normalization for histopathology, random rotation for X-rays. Avoid augmentations that destroy diagnostically relevant structure.
4. If unlabeled medical images are available, run SSL pretraining (SimCLR or MAE) on them before supervised fine-tuning.
5. If labeled data is very scarce (<1k images), use a smaller backbone (ResNet-18 vs ResNet-50) to reduce overfitting risk.
