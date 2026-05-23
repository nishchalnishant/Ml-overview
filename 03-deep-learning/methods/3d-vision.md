---
module: Deep Learning
topic: Methods
subtopic: D Vision
status: unread
tags: [deeplearning, ml, methods-3d-vision]
---
# 3D Vision & Point Clouds

> Covers point cloud processing (PointNet, PointNet++, DGCNN, PointTransformer), voxel-based methods (SparseConv, PointPillars), implicit representations (NeRF, 3DGS), 3D object detection, and 3D semantic segmentation.

---

## Table of Contents

1. [3D Data Representations](#1-3d-data-representations)
2. [PointNet](#2-pointnet)
3. [PointNet++](#3-pointnet)
4. [DGCNN](#4-dgcnn)
5. [PointTransformer](#5-pointtransformer)
6. [Voxel-Based Methods](#6-voxel-based-methods)
7. [Mesh-Based Learning and Implicit Representations](#7-mesh-based-learning-and-implicit-representations)
8. [Neural Radiance Fields (NeRF)](#8-neural-radiance-fields-nerf)
9. [3D Gaussian Splatting](#9-3d-gaussian-splatting)
10. [3D Object Detection](#10-3d-object-detection)
11. [3D Semantic Segmentation](#11-3d-semantic-segmentation)
12. [Datasets](#12-datasets)
13. [Key Interview Points](#13-key-interview-points)

---

## 1. 3D Data Representations

**The problem:** a 2D image is a regular grid — every pixel has a fixed neighbor in four directions, so convolutions apply directly. Sensors that capture 3D geometry (LiDAR, depth cameras, structured light) produce data in formats that are not regular grids: scattered points in space, occupied cells in a 3D lattice, connected triangular surfaces, or learned functions that map positions to occupancy. Each format has different structural properties, and no single deep learning primitive handles all of them.

The choice of representation is the first architectural decision in any 3D vision pipeline.

| Representation | Description | Pro | Con |
|----------------|------------|-----|-----|
| **Point cloud** | Unordered set of (x,y,z) points | Compact, direct sensor output | No topology, permutation-sensitive |
| **Voxel grid** | 3D grid of cells (binary or occupancy) | Regular structure, 3D convolutions apply | Memory O(n³), extremely sparse |
| **Mesh** | Vertices + faces (triangles) | Compact surface encoding | Irregular connectivity, hard to apply DL |
| **Implicit (NeRF, SDF)** | Function f(x,y,z) → density or SDF | Continuous, differentiable | Slow rendering, per-scene optimization |
| **Multi-view images** | 2D projections from multiple viewpoints | Leverage 2D DL | Requires known camera poses |

Each downstream section picks one of these representations and explains why that choice was made.

---

## 2. PointNet

**The problem:** a LiDAR scanner returns N points in 3D space — (x, y, z) tuples with no fixed ordering. If you naively feed these into an MLP, the output changes if you shuffle the input, even though the physical object did not change. You need a function that is *permutation-invariant*: f({p_1, ..., p_N}) = f({p_σ(1), ..., p_σ(N)}) for any permutation σ. Standard convolutions require a fixed grid; recurrent models are order-sensitive.

**The core insight:** any permutation-invariant function on a set can be approximated by a composition of a symmetric function (one that treats all inputs equally, like max or sum) and a shared per-element function. Apply the same MLP to each point independently (shared weights, no inter-point communication), then take a global max pool across all points. Max pooling is symmetric: it does not matter what order you process the points in; the output is always the global maximum across the set.

**The mechanics:**

```
Input: N points × 3 (x, y, z)
├── [Optional] T-Net: learn a 3×3 transform to canonicalize input orientation
├── Point-wise MLP (shared weights): each point (x,y,z) → 64-d feature
├── [Optional] T-Net: learn a 64×64 transform to align feature space
├── Point-wise MLP (shared weights): 64-d → 1024-d feature per point
├── Global Max Pool: (N, 1024) → (1024,)   ← permutation-invariant bottleneck
└── MLP: 1024 → 512 → 256 → n_classes
```

For segmentation: concatenate the global feature vector back to each point's local 64-d feature, then apply a per-point MLP to classify each point. This way each point sees both its own local geometry and the global context.

```python
import torch
import torch.nn as nn

class PointNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)       # Point-wise MLP via 1D conv
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B, N, 3) → (B, 3, N) for Conv1d
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]           # Global max pool → (B, 1024)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**What breaks:**

- **No local structure:** every point is processed independently before the global pool. The MLP cannot reason about how neighboring points relate to each other — it has no spatial receptive field. Fine-grained local geometry (surface normals, local curvature) is not captured.
- **Global context is a bottleneck:** the single 1024-d vector must encode the entire point cloud. Detailed spatial information about specific regions is compressed into this global descriptor and cannot be recovered for per-point tasks.
- **Sensitivity to density:** LiDAR returns dense point clouds near the sensor and sparse clouds far away. PointNet has no mechanism to adapt to variable local density.

---

## 3. PointNet++

**The problem:** PointNet applies the same MLP to each point and pools globally. It has no notion of local neighborhoods — it cannot detect that three nearby points form an edge, or that a cluster of points outlines a wheel. Any 2D CNN learns hierarchical features: early layers detect edges, middle layers detect textures, late layers detect objects. PointNet has no hierarchy.

**The core insight:** 3D space has the same hierarchical structure as 2D images — small neighborhoods of nearby points encode local shape primitives; larger regions encode parts; the whole point cloud encodes the object. Build a hierarchy by recursively: (1) selecting representative centroids, (2) grouping nearby points around each centroid, (3) running a mini-PointNet on each group to produce a local feature. Repeat with sparser centroids and larger radii until you reach a single global feature.

**The mechanics:**

Each level consists of a *Set Abstraction* module with three steps:

1. **Farthest Point Sampling (FPS):** greedily select S centroids that maximize the minimum distance to the already-selected set. This gives a spatially uniform coverage of the point cloud — far better than random sampling for capturing fine structure.

2. **Ball Query:** for each centroid, collect all points within radius r. Pad or truncate to K points. Ball query gives translation-equivariant neighborhoods; k-NN would create neighborhoods of inconsistent spatial scale.

3. **Mini-PointNet:** apply a shared MLP + global max pool on each K-point neighborhood → one local feature per centroid.

```
Level 0: N=1024 points, 3-d
Level 1: S=256 centroids, r=0.2, K=32 neighbors → 256 × 128-d features
Level 2: S=64 centroids,  r=0.4, K=64 neighbors → 64  × 256-d features
Level 3: S=16 centroids,  r=0.8, K=128 neighbors → 16 × 512-d features
Level 4: Global pool → 1 × 1024-d
```

For segmentation, a *Feature Propagation* step interpolates features from sparse centroids back to the original N points using inverse-distance-weighted interpolation, then concatenates with skip-connected features from the encoding path (analogous to U-Net decoder).

**What breaks:**

- **Fixed radius:** ball query uses a single radius r per level. If the point cloud has highly variable density (dense near sensor, sparse far away), a fixed r may miss neighbors in sparse regions and include too many in dense regions. The multi-scale grouping (MSG) variant uses multiple radii at each level but at higher compute cost.
- **FPS is sequential:** finding the farthest point requires maintaining a distance matrix updated after each selection. FPS on N=8192 points takes O(N²) naive, or O(N log N) with priority queues — still a training bottleneck.
- **Discrete hierarchy:** the number of levels and their radii are fixed hyperparameters. The receptive field at each level is fixed; it cannot adapt to object scale.

---

## 4. DGCNN

**The problem:** PointNet++ constructs neighborhoods in 3D Euclidean space. But as the network deepens, the meaningful notion of "similarity" is not Euclidean distance between raw coordinates — it is distance in *feature space*. Two points that are far apart in 3D might have similar surface normals, similar curvature, and belong to the same semantic part. A fixed-radius ball query in 3D space cannot capture these feature-space relationships.

**The core insight:** construct the k-nearest-neighbor graph dynamically in the *current feature space* at each layer, not in fixed 3D space. For each point, aggregate features from its current k nearest neighbors in feature space. Update the feature representation. Recompute the k-NN graph in the new feature space. The graph is dynamic — it changes at every layer as features evolve.

**The mechanics:**

At each layer, for every point i with feature x_i, find its k nearest neighbors in feature space: j_1, ..., j_k. Compute an edge feature for each neighbor:

```
h(x_i, x_{j_m} - x_i) = MLP([x_i, x_{j_m} - x_i])
```

The concatenation of x_i (the point's own feature) and x_{j_m} - x_i (the relative feature difference) encodes both the global position in feature space and the local variation. Aggregate over the k neighbors:

```
x'_i = max_{m=1..k} h(x_i, x_{j_m} - x_i)
```

This is called *EdgeConv*. Stack multiple EdgeConv layers; at each layer, recompute k-NN in the current feature space.

**What breaks:**

- **k-NN recomputation cost:** at each layer, computing pairwise distances and finding k-NN for N points is O(N² d) per layer. For N=2048, d=128, this is significant, especially during training with gradients flowing back through the graph topology (which itself depends on features). Approximate k-NN (e.g., FAISS) can help at the cost of gradient accuracy.
- **Dynamic graph breaks gradient flow:** the graph topology is a non-differentiable discrete selection (which k neighbors to use). Gradients cannot flow through the selection step itself — only through the features of selected neighbors. This limits the expressiveness of the learned topology.

---

## 5. PointTransformer

**The problem:** DGCNN computes edge features by aggregating from k neighbors with a shared MLP. But this aggregation treats all k neighbors equally (post-max-pool). Some neighbors are more informative than others — a neighbor aligned with the surface normal carries more geometric signal than a neighbor in the interior of a flat region. You want each point to *selectively attend* to the neighbors most relevant to its current representation.

**The core insight:** apply self-attention to local 3D neighborhoods. Standard Transformer attention is quadratic in the sequence length, so full global attention over N=8192 points is prohibitive. But attention within a small k-NN neighborhood (k=16 or 32) is cheap. Add a 3D position encoding to the attention computation — the relative 3D offset between a point and each neighbor is embedded and added to both the attention weights and the values.

**The mechanics:**

For each point i with feature x_i and position p_i, compute attention over its k nearest neighbors:

```
δ = φ(p_i - p_j)           # position encoding of relative 3D offset
q = α(x_i)                  # query
k_j = β(x_j)                # key
v_j = γ(x_j)                # value

a_{ij} = softmax(ρ(q - k_j + δ))   # vector attention weights
x'_i = Σ_j a_{ij} ⊙ (v_j + δ)     # weighted sum with position bias
```

The subtraction q - k_j (rather than dot product q·k_j) is *vector self-attention* — it produces a per-channel weight rather than a scalar weight, giving the model channel-wise selectivity within each neighbor.

**What breaks:**

- **Local-only attention:** attention is restricted to k local neighbors, so the model cannot capture long-range dependencies in a single layer. Stacking layers increases the effective receptive field, but it grows slowly (like convolutions).
- **k-NN recomputation:** same cost as DGCNN. The k-NN graph is typically computed in 3D space (not recomputed in feature space at each layer), which recovers computational efficiency but loses DGCNN's feature-space adaptivity.

---

## 6. Voxel-Based Methods

### VoxNet

**The problem:** PointNet and its variants process unordered point sets, but the operations (shared MLP, pooling) are specialized primitives that don't reuse the massive engineering infrastructure built for image CNNs. Can you just convert a 3D scene to a regular 3D grid and apply standard 3D convolutions?

**The core insight:** discretize 3D space into a voxel grid. Each cell is 1 if it contains at least one point, 0 otherwise. A 3D CNN processes this binary occupancy grid exactly as a 2D CNN processes an image — convolution, pooling, batch norm, all apply unchanged.

**What breaks:** memory scales as O(n³). A 128³ voxel grid at float32 requires 8 MB per sample. A 256³ grid requires 64 MB. Most voxels are empty — a typical room scan is >99% empty space. You pay full O(n³) compute for O(M) occupied cells where M ≪ n³. This fundamentally limits resolution.

---

### Sparse 3D Convolutions (SparseConv)

**The problem:** VoxNet pays O(n³) compute for a grid that is >99% empty. The waste is structural: standard dense convolutions compute outputs at every grid location, including empty ones whose contributions are all zero.

**The core insight:** only compute convolutions at *occupied voxels*. Keep a list of (position, feature) pairs rather than a dense 3D tensor. For a sparse convolution kernel, only process input-output position pairs where the input position is occupied. The result is another sparse tensor containing only the occupied output positions.

**The mechanics:**

```python
import spconv.pytorch as spconv

# Sparse tensor: voxel coordinates + features at those coordinates
# coords: (M, 4) — (batch_idx, x, y, z) for M occupied voxels
# features: (M, C)

conv = spconv.SparseConv3d(in_channels=4, out_channels=64,
                            kernel_size=3, padding=1)
# Output is also a sparse tensor; only occupied positions are computed
```

SECOND (Sensor Fusion for 3D Object Detection), PointPillars, and CenterPoint all use sparse convolutions as their backbone. On typical LiDAR scans, sparse convolutions reduce compute by 10-100× compared to dense 3D CNNs.

**What breaks:**

- **Submanifold vs. regular sparse conv:** if you use regular sparse conv, every input occupied voxel can activate output positions in its 3×3×3 neighborhood, causing the sparse tensor to grow denser after each layer (the "dilation" problem). *Submanifold* sparse convolutions only compute outputs at positions that were occupied in the input — keeping the sparsity pattern fixed. This prevents feature dilation but limits the receptive field.
- **Custom CUDA kernels required:** spconv operations cannot be expressed as standard PyTorch ops; they require specialized GPU kernels. Portability and debugging are harder.

---

### PointPillars

**The problem:** sparse 3D convolutions still process data in a 3D volume. For autonomous driving, you care about objects in BEV (Bird's Eye View) — their (x, y) position and heading. The height dimension carries limited discriminative information (cars are ~1.5 m tall; pedestrians ~1.8 m). Why pay for 3D convolutions when a 2D BEV map would suffice?

**The core insight:** instead of voxels (x, y, z cells), use *pillars* — vertical columns covering the (x, y) plane with all z extent inside. Encode all points within each pillar using a simple PointNet → one feature vector per pillar → arrange feature vectors into a 2D BEV feature map → apply a standard 2D CNN detector (SSD-style). The 3D structure is compressed into 2D before any heavy convolution.

**The mechanics:**

```
LiDAR points (x, y, z, intensity)
  ↓ Bin into (x, y) pillars
  ↓ For each pillar: augment each point with (Δx, Δy, Δz) offset from pillar centroid
  ↓ Shared PointNet MLP on points within each pillar → (P, 64-d)
  ↓ Max-pool over points in each pillar → (P, 64) pillar features
  ↓ Scatter into 2D spatial grid → BEV feature map (H, W, 64)
  ↓ 2D CNN backbone + FPN
  ↓ SSD-style 3D bounding box head
```

Result: ~62 Hz inference on a single GPU — fast enough for real-time autonomous driving.

**What breaks:**

- **Loss of vertical structure:** compressing z into a single pillar feature loses fine-grained height information. This matters for distinguishing cyclists from pedestrians, or for detecting objects on slopes. CenterPoint and BEVFusion use sparse 3D backbones to retain height information before projecting to BEV.
- **Fixed pillar resolution:** the (x, y) pillar grid has a fixed resolution. Fine-grained spatial precision requires small pillars → large feature maps → more memory. There is an explicit accuracy-speed tradeoff here.

---

## 7. Mesh-Based Learning and Implicit Representations

### Graph CNNs on Meshes

**The problem:** meshes are the standard output format for 3D reconstruction pipelines, body model fitting (SMPL, FLAME), and CAD tools. But meshes have irregular connectivity — each vertex can have a variable number of neighbors. Standard convolutions do not apply. You want to learn features over mesh vertices that respect the surface topology.

**The core insight:** a mesh is naturally a graph: vertices are nodes, edges are connections. Apply graph convolutions (spectral or spatial) directly on the mesh graph. For body meshes with fixed topology (SMPL has 6890 vertices with the same edge structure for every body), the graph can be fixed and the Laplacian precomputed, making spectral GCNs efficient.

**What breaks:** mesh-based GCNs require the mesh to be valid (no self-intersections, consistent face orientations, manifold). Predicted meshes from reconstruction pipelines often violate these constraints. And for the general case, different shapes have different graph topology, making batching difficult without padding.

---

### Occupancy Networks and DeepSDF

**The problem:** point clouds are sparse and cannot represent watertight surfaces. Meshes require valid connectivity. Voxel grids are memory-limited. Is there a representation that can encode continuous, smooth 3D geometry at arbitrary resolution without these constraints?

**The core insight:** represent the 3D shape as a continuous implicit function — a neural network that takes a 3D position and outputs either occupancy probability (inside vs. outside) or signed distance to the nearest surface. The surface is the level set of this function. The network can be queried at any resolution without changing the representation.

**The mechanics:**

- **Occupancy Networks:** f_θ(x, z) → [0, 1], where z is a shape latent code from an encoder. The surface is the level set {x : f_θ(x, z) = 0.5}. Extract the surface via marching cubes on a dense query grid.
- **DeepSDF:** f_θ(x, z) → ℝ, where the output is the signed distance to the surface. Negative inside, positive outside. The surface is {x : f_θ(x, z) = 0}.

Both are trained with supervision: for Occupancy Networks, binary labels (is this 3D point inside or outside the shape?); for DeepSDF, ground-truth SDF values computed from watertight meshes.

**What breaks:** at test time, extracting the surface requires querying the network on a dense 3D grid (e.g., 256³ = 16M queries). This is slow. Instant-NGP and related methods replace the MLP with a hash-encoded feature grid to make queries faster, but the fundamental marching cubes step remains.

---

## 8. Neural Radiance Fields (NeRF)

**The problem:** given a set of 2D images of a scene taken from known camera positions, how do you synthesize a new image from a previously unseen viewpoint? Multi-view stereo can reconstruct a point cloud, but point clouds are sparse and don't handle view-dependent effects (reflections, specular highlights). Mesh-based methods require clean reconstruction. You need a representation that (1) accurately models the 3D geometry, (2) captures view-dependent appearance, and (3) can be trained directly from 2D image supervision with no 3D ground truth.

**The core insight:** represent the scene as a continuous volumetric radiance field — a function that maps any 3D position *and* viewing direction to color and volume density. Render any view by integrating color along each camera ray through this volume (differentiable volume rendering). Train by minimizing the photometric loss between rendered and actual pixel colors. No 3D supervision needed; the geometry emerges from the consistency constraints imposed by multiple views.

**The mechanics:**

The NeRF MLP takes a 5D input: (x, y, z, θ, φ) — 3D position and 2D viewing direction. It outputs (R, G, B, σ) — color and volume density.

Positional encoding expands the raw (x, y, z) coordinates into a high-frequency Fourier feature vector to let the MLP fit high-frequency scene details:

```
γ(p) = [p, sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^{L-1}πp), cos(2^{L-1}πp)]
```

Volume rendering integrates along a ray r(t) = o + t·d:

```
C(r) = ∫_{t_n}^{t_f} T(t) · σ(r(t)) · c(r(t), d) dt

T(t) = exp(-∫_{t_n}^{t} σ(r(s)) ds)   # transmittance: how much light reaches t
```

In practice, discretize the integral by sampling K points along the ray:

```
C(r) ≈ Σ_{i=1}^{K} T_i · (1 - exp(-σ_i δ_i)) · c_i

T_i = exp(-Σ_{j<i} σ_j δ_j)   # accumulated transmittance
```

where δ_i = t_{i+1} - t_i is the spacing between samples.

```python
class NeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4):
        super().__init__()
        pos_dim = 3 + 2 * L_pos * 3    # 3 raw + 2L×3 Fourier features
        dir_dim = 3 + 2 * L_dir * 3
        self.backbone = nn.Sequential(
            nn.Linear(pos_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.density_head = nn.Linear(256, 1)       # σ ≥ 0 (apply ReLU)
        self.color_net = nn.Sequential(
            nn.Linear(256 + dir_dim, 128), nn.ReLU(),
            nn.Linear(128, 3), nn.Sigmoid()         # RGB ∈ [0, 1]
        )

    def forward(self, pos_enc, dir_enc):
        h = self.backbone(pos_enc)
        sigma = torch.relu(self.density_head(h))
        rgb = self.color_net(torch.cat([h, dir_enc], dim=-1))
        return rgb, sigma
```

Training: for each pixel, sample a ray, sample K points along the ray, query NeRF at each point, integrate using volume rendering, compare rendered color to observed pixel color with L2 loss.

**What breaks:**

- **Per-scene optimization:** a trained NeRF represents a single scene. There is no generalization to new scenes; you must train a new NeRF from scratch for each scene. Training takes hours to days on an 8-GPU machine.
- **Slow inference:** naive NeRF queries the MLP at K×N ray samples per frame. Rendering a 1920×1080 image at K=128 samples requires ~264M MLP forward passes — far too slow for real-time.
- **View interpolation, not extrapolation:** NeRF needs training views that densely cover the scene. For novel viewpoints that are far outside the training view distribution, rendering quality degrades significantly.

**Variants:**

- **Instant-NGP (Müller et al., 2022):** replaces the deep MLP with a small MLP combined with a multi-resolution hash grid of learned features. Hash grid queries are fast CUDA memory accesses; the MLP is shallow. Reduces training from hours to minutes.
- **Mip-NeRF (Barron et al., 2021):** models rays as cones (not infinitely thin lines), using integrated positional encoding over cone frustums. Handles anti-aliasing at different scales (close-up vs. distant views).
- **Block-NeRF (Tancik et al., 2022):** decomposes a city-scale scene into multiple NeRF blocks, each covering a spatial region, with appearance embeddings per block for consistency.

---

## 9. 3D Gaussian Splatting

**The problem:** NeRF achieves high-quality novel view synthesis but is slow — querying an MLP millions of times per frame is inherently serial and expensive. Instant-NGP reduces training time but rendering is still far from real-time. For applications like VR, telepresence, and interactive 3D editing, you need real-time rendering (>30 fps at high resolution) with comparable quality to NeRF.

**The core insight:** represent the scene as an explicit set of millions of 3D Gaussians. Each Gaussian has a position (mean), a 3D covariance matrix (shape/orientation/size), opacity, and a set of spherical harmonic coefficients (view-dependent color). Render by *splatting* — projecting each Gaussian onto the 2D image plane as a 2D Gaussian and alpha-compositing them from front to back. Splatting can be done on the GPU with highly optimized rasterization — no MLP queries needed.

**The mechanics:**

Each Gaussian i is defined by:
- μ_i ∈ ℝ³: center position
- Σ_i ∈ ℝ^{3×3}: covariance matrix (factored as Σ = RSS^T R^T where R is rotation, S is scale)
- α_i ∈ [0, 1]: opacity
- SH coefficients: view-dependent color via spherical harmonics

Rendering a pixel:
1. Project each 3D Gaussian to a 2D Gaussian in screen space using the camera projection Jacobian.
2. Sort Gaussians by depth (front to back).
3. Alpha-composite contributions: C = Σ_i c_i α_i Π_{j<i} (1 - α_j)

Optimization: initialize from a sparse point cloud (e.g., from COLMAP SfM), then optimize all Gaussian parameters via gradient descent to minimize photometric loss. Adaptive density control: clone Gaussians in under-reconstructed regions, split Gaussians that are too large.

**What breaks:**

- **Memory:** millions of Gaussians each storing position, covariance (6 floats), opacity, SH coefficients (~48 floats for degree-3 SH). A scene might require 3-6M Gaussians → several GB of GPU memory. Not viable on mobile devices.
- **Editing artifacts:** unlike NeRF where the geometry is implicitly encoded, 3D-GS explicitly stores Gaussians. Removing or moving objects requires identifying which Gaussians belong to the object — there is no semantic segmentation of Gaussians by default.
- **Initialization sensitivity:** optimization starting from random positions diverges. The COLMAP point cloud initialization is critical; scenes without good multi-view overlap fail to reconstruct.

---

## 10. 3D Object Detection

**The problem:** autonomous vehicles need to detect, localize, and track 3D bounding boxes — (x, y, z, length, width, height, heading) — for all objects around them in real time, using LiDAR point clouds (and optionally camera images). The challenge is that objects are at varying distances (dense near, sparse far), have wide variation in size, and the detection must run at ≥10 Hz for a safety-critical system.

### Pipeline Approaches

| Method | Input Representation | Core idea |
|--------|---------------------|-----------|
| PointRCNN | Raw points | Generate proposals directly from per-point features; two-stage |
| VoxelNet | 3D voxels | VFE (voxel feature encoding) + 3D sparse conv + RPN |
| PointPillars | BEV pillars | Compress to 2D → fast 2D CNN; real-time |
| CenterPoint | BEV heatmap | Detect object centers as heatmap peaks; regress attributes |

**CenterPoint** is the current standard for LiDAR-only detection. It uses a sparse 3D backbone (VoxelNet-style), compresses to BEV, and then runs a center-based detection head: predict a Gaussian heatmap peak at each object center, then regress (z, size, heading, velocity) from the peak location. Center-based regression avoids the anchor matching problem and handles arbitrary heading without angle ambiguity.

### LiDAR + Camera Fusion

**The problem:** LiDAR provides accurate depth and 3D geometry but is low-resolution and cannot detect texture, color, or text. Cameras provide high-resolution texture and semantics but no reliable depth. Fusing both signals should give better detection than either alone.

**The core insight:** align LiDAR and camera features in a common representation space — BEV. Project image features into BEV using predicted depth distributions (BEVFusion), or use cross-attention between 3D queries and image features (TransFusion, DETR3D).

| Fusion strategy | When signals are combined | Notes |
|-----------------|--------------------------|-------|
| Early fusion | Project LiDAR points to image, append image features to each point | Simple; image resolution limits quality |
| Late fusion | Detect independently, then merge predictions via NMS | No cross-modal reasoning |
| Deep fusion (BEVFusion) | Align both in BEV feature space | Best accuracy; adds camera-to-BEV projection cost |

**What breaks in BEV fusion:** projecting image features into BEV requires depth estimation for each image pixel. Depth estimation errors cause camera features to be placed at wrong BEV locations, creating noise rather than signal. In adverse conditions (rain, fog, night), the camera signal degrades while LiDAR is often more robust — the fusion model must learn to down-weight the noisy modality.

---

## 11. 3D Semantic Segmentation

**The problem:** 3D object detection produces bounding boxes. But for robot manipulation, scene understanding, and HD map building, you need per-point semantic labels — which points belong to road, sidewalk, car, pedestrian, vegetation, building? This is the 3D analogue of 2D semantic segmentation: assign a class label to every point.

### Indoor (ScanNet, S3DIS)

**The problem:** indoor RGB-D scans produce dense meshes with millions of points covering walls, floors, furniture, and objects. The points are nearly uniformly distributed in 3D. You need to leverage both 3D geometry and 2D texture.

**PointTransformer** achieves state-of-the-art on S3DIS and ScanNet by applying local attention within neighborhoods, with a U-Net-style encoder-decoder to propagate features back to the original resolution.

**MinkowskiEngine** uses sparse 3D convolutions on the voxelized scan, giving a regular convolutional backbone that can use standard architectures (ResNet, U-Net). The sparse tensor format handles large scenes efficiently.

### Outdoor (SemanticKITTI, nuScenes)

**The problem:** outdoor LiDAR scans have radically non-uniform point density: 100+ points per square meter near the vehicle, 1 point per square meter at 50 m range. Standard 3D methods trained on uniform density fail on distant sparse regions.

- **RangeNet++:** project the LiDAR point cloud onto a 2D range image (azimuth × elevation grid), apply a 2D CNN on this image, then project predictions back to 3D. Fast, handles non-uniform density naturally. What breaks: range image distorts 3D geometry; nearby points in 3D may be far apart in range image if at different elevations.

- **Cylinder3D:** use cylindrical voxelization (r, θ, z) instead of Cartesian (x, y, z). Near the sensor, cylindrical cells are small and fine-grained; far from the sensor, cells are larger. This naturally adapts to LiDAR's variable density pattern.

---

## 12. Datasets

| Dataset | Task | Modality | Scale |
|---------|------|---------|-------|
| ModelNet40 | Classification | 3D CAD models | 40 classes, 12,311 models |
| ShapeNet | Part segmentation | 3D CAD models | 16 categories, 16,881 models |
| ScanNet | 3D scene segmentation | RGB-D scans | 1,513 scenes, 20 classes |
| S3DIS | Indoor segmentation | LiDAR + RGB | 6 areas, 271 rooms, 13 classes |
| KITTI | 3D detection | LiDAR + stereo | 7,481 training frames |
| nuScenes | 3D detection | LiDAR + 6 cameras | 1,000 scenes, 23 classes |
| Waymo Open | 3D detection | LiDAR + 5 cameras | 1,150 segments |

---

## 13. Key Interview Points

**Why does PointNet use global max pooling specifically, not average pooling or summation?**
Max pooling selects the most active feature across all points — it captures the *most distinctive* local signal anywhere in the point cloud, ignoring uninformative points. Average pooling would dilute the signal from salient points with contributions from background noise. Summation would make the output depend on the number of points (variable across samples). Max is the theoretically principled choice for approximating a permutation-invariant set function.

**What is the difference between ball query and k-NN in PointNet++, and why does it matter?**
Ball query returns all points within a fixed radius r — a spatially consistent neighborhood. k-NN returns exactly k points regardless of their distance — the neighborhood scale adapts to local density. Ball query is preferred for PointNet++ because it gives translation-equivariant neighborhoods with consistent scale, important for learning features that generalize across positions. k-NN can produce neighborhoods that are very large in sparse regions, degrading feature quality.

**Why are sparse 3D convolutions necessary for LiDAR-based detection?**
A typical LiDAR scan covers a 100m × 100m × 4m volume. At 0.1m voxel resolution, this is 1000 × 1000 × 40 = 40M voxels. A LiDAR scan has ~100,000 points → <0.25% occupancy. Dense 3D CNNs would compute 40M activations per layer, 99.75% of which are multiplying zeros. Sparse convolutions compute only the ~250,000 occupied voxel positions, achieving 100-200× speedup without approximation.

**How does PointPillars achieve real-time speed (~62 Hz) while PointNet-based methods are slower?**
PointPillars compresses 3D → 2D after a single PointNet per pillar. All subsequent processing uses 2D convolutions, which are highly optimized on modern GPUs (cuDNN, tensor cores). Full 3D or point-based backbones process irregular data structures (sparse tensors or scattered point lists) that are harder to parallelize. The 2D BEV backbone is the key architectural decision that enables real-time speed.

**Why does NeRF train slowly, and how does Instant-NGP fix this?**
Standard NeRF uses a large MLP (8 layers, 256 units) because it must implicitly encode all spatial variation in the network weights. Querying this MLP K×N times per training step (K samples per ray, N rays per batch) is slow. Instant-NGP replaces most of the MLP with a multi-resolution hash grid of learned feature vectors stored in GPU memory. A query at position (x, y, z) performs a hash table lookup (fast memory read) + interpolation + a tiny 2-layer MLP. Hash lookups are ~100× faster than large MLP evaluations, reducing training from hours to minutes.

**What is the key trade-off between NeRF and 3D Gaussian Splatting?**
NeRF is an implicit representation (an MLP); 3DGS is explicit (a list of Gaussians). NeRF is memory-efficient (the whole scene is encoded in network weights), but inference requires MLP queries — not real-time. 3DGS stores millions of Gaussians explicitly (~GB of GPU memory) but renders by rasterization — real-time at >100 fps. 3DGS is editable (you can move or delete Gaussians); NeRF requires retraining. For quality, both are comparable on in-distribution views; NeRF generalizes better to extreme novel viewpoints.

**On Something-Something, why do point cloud methods fail at temporal understanding?**
This is a 2D video benchmark — it doesn't involve point clouds. But the analogous 3D question is: for 4D LiDAR sequences, why do single-frame 3D detectors fail to estimate velocity? A single LiDAR frame gives position but not motion. CenterPoint explicitly adds velocity regression as a detection head and uses two concatenated frames as input to give the model temporal context. Multi-object tracking (MOT) methods propagate tracks across frames using Kalman filters, which explicitly model motion dynamics.

## Flashcards

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
