# 3D Vision & Point Clouds

3D vision processes spatial data in three dimensions: point clouds, voxel grids, meshes, and implicit representations. Key applications: autonomous driving (LiDAR), robotics, AR/VR, medical imaging (CT/MRI reconstruction).

---

## 3D Data Representations

| Representation | Description | Pro | Con |
|----------------|------------|-----|-----|
| **Point Cloud** | Unordered set of (x,y,z) points | Compact, direct sensor output | No topology, unordered |
| **Voxel Grid** | 3D grid of cells (binary or occupied) | Regular structure, 3D convolutions apply | Memory `O(n³)`, sparse |
| **Mesh** | Vertices + faces (triangles) | Compact surfaces | Complex to process with DL |
| **Implicit (NeRF, SDF)** | Function `f(x,y,z) → density/SDF` | Continuous, differentiable | Slow rendering |
| **Multi-view images** | 2D projections from different viewpoints | Leverage 2D DL | Requires known camera poses |

---

## Point Cloud Processing

### PointNet (Qi et al., 2017)

First deep learning method that operates directly on unordered point clouds.

**Key insight:** Permutation invariance via a symmetric function (global max pooling).

```
Input: N points × 3 (x, y, z)
├── Input Transform (T-Net: 3×3 transform to align input)
├── Point-wise MLP (64, 64)  — each point independently
├── Feature Transform (T-Net: 64×64 align feature space)
├── Point-wise MLP (64, 128, 1024)
├── Global Max Pool          — symmetric, permutation-invariant
└── MLP (512, 256, n_classes) → Classification
```

For segmentation: concatenate global features back to each point's local features, then per-point MLP.

```python
import torch
import torch.nn as nn

class PointNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)    # Point-wise MLP via 1D conv
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, n_classes)
    
    def forward(self, x):
        # x: (B, N, 3) → (B, 3, N)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]    # Global max pool: (B, 1024)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**Limitation:** No local structure — treats each point in global context only.

---

### PointNet++ (Qi et al., 2017)

Extends PointNet with hierarchical local feature extraction.

**Three components:**

1. **Farthest Point Sampling (FPS):** Select centroids that maximally cover the point cloud
2. **Ball Query:** For each centroid, find all points within radius r (neighborhood)
3. **PointNet:** Apply mini-PointNet on each local neighborhood → local feature

Stack multiple Set Abstraction layers to go from fine to coarse (like pooling in CNNs).

```
Level 0: N=1024 points
Level 1: N=256 centroids, radius=0.2, 32 neighbors per centroid → (256, 128-d features)
Level 2: N=64 centroids, radius=0.4, 64 neighbors per centroid → (64, 256-d features)
Level 3: Global → (1, 1024-d)
```

**For segmentation:** Add Feature Propagation (FP) layers — interpolate features back to original resolution.

---

### DGCNN — Dynamic Graph CNN

Constructs a k-NN graph in feature space (not just 3D space) and applies graph convolutions. The graph is updated dynamically after each layer.

**EdgeConv:** For each point, aggregate features from its k nearest neighbors:
`h_i = Σ_{j ∈ N(i)} h(x_i, x_j - x_i)`

Captures local shape structure adaptively.

---

### PointTransformer (Zhao et al., 2021)

Applies self-attention to local point neighborhoods. Includes vector self-attention with position encodings in 3D.

State-of-the-art on ModelNet40, ShapeNet, S3DIS (3D semantic segmentation).

---

## Voxel-Based Methods

### VoxNet

Occupancy grid → 3D CNNs. Intuitive extension of 2D CNN to 3D.

**Problem:** Memory explodes cubically. A 128³ voxel grid at float32 = 8 MB per sample.

### Sparse 3D Convolutions (SparseConv)

Only compute convolutions at occupied voxels. Reduces computation from `O(N³)` to `O(M)` where M = number of occupied voxels.

Used in: SECOND, PointPillars, CenterPoint (autonomous driving 3D detection).

```python
# Using spconv library
import spconv.pytorch as spconv

conv = spconv.SparseConv3d(in_channels=4, out_channels=64, 
                            kernel_size=3, padding=1)
# Input: sparse tensor (occupied voxel positions + features)
```

### PointPillars

Divide 3D space into vertical columns (pillars) instead of voxels. Points within each pillar → simplified PointNet → 2D BEV (Bird's Eye View) feature map → 2D CNN.

**Speed:** ~62 Hz inference for autonomous driving LiDAR.

---

## Mesh-Based Learning

### Graph Convolutional Networks on Meshes

Represent mesh as a graph: vertices = nodes, edges = connections. Apply GCNs.

Used in: 3D pose estimation, shape reconstruction, body model fitting (SMPL).

### Occupancy Networks / DeepSDF

Learn an implicit function:
- **Occupancy Networks:** `f_θ(x, z) → [0, 1]` (probability point x is inside shape)
- **DeepSDF:** `f_θ(x, z) → ℝ` (signed distance to surface)

z = shape latent code. At test time, query on a dense grid and extract isosurface (marching cubes).

---

## Neural Radiance Fields (NeRF)

NeRF represents a 3D scene as a function mapping 5D input (position + viewing direction) to volume density and color:

`(x, y, z, θ, φ) → (RGB, σ)`

Render by integrating along camera rays using volume rendering:
`C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt`

where `T(t) = exp(-∫ σ(r(s)) ds)` is transmittance.

**Training:** Minimize photometric loss between rendered and actual pixel colors.

```python
# Simplified NeRF forward
class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc_dim = 60   # positional encoding (L=10 frequencies × 3 × 2)
        self.dir_enc_dim = 24
        self.net = nn.Sequential(
            nn.Linear(self.pos_enc_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.density_head = nn.Linear(256, 1)      # σ (density)
        self.color_head = nn.Linear(256 + self.dir_enc_dim, 3)  # RGB
```

**Limitations:** Training is slow (hours), per-scene optimization, no generalization across scenes.

**Variants:**
- **Instant-NGP:** Hash grids for 100× faster training
- **Mip-NeRF:** Anti-aliased rendering, handles multi-scale views
- **Block-NeRF:** City-scale NeRF
- **NeRFstudio:** Modular toolkit

---

## 3D Gaussian Splatting (3DGS)

Represent scene as millions of 3D Gaussians, each with position, covariance, opacity, and spherical harmonics for color. Render by splatting (projecting) Gaussians onto 2D image plane.

**Advantages over NeRF:**
- Real-time rendering (>100 fps)
- Much faster training (~30 min vs hours)
- Explicit representation → editable

**Applications:** Novel view synthesis, telepresence, VR/AR, robotics scene understanding.

---

## 3D Object Detection (Autonomous Driving)

### Task

Given LiDAR point cloud (+ optional camera), detect 3D bounding boxes: `(x, y, z, l, w, h, θ)` for each object.

### Pipeline Approaches

| Method | Processing | Notes |
|--------|-----------|-------|
| PointRCNN | Point-based | Two-stage: generate proposals from points |
| VoxelNet | Voxel-based | Voxelize → 3D CNN → RPN |
| PointPillars | Pillar-based | Fast BEV → 2D detection |
| CenterPoint | BEV center-based | Detect object centers, then regress attributes |

### LiDAR + Camera Fusion

- **Early fusion:** Project points to image, append image features
- **Late fusion:** Detect separately, then fuse predictions (NMS)
- **Deep fusion:** BEVFusion, TransFusion — align in BEV space

---

## 3D Semantic Segmentation

Assign a class label to every point.

**Indoor (ScanNet, S3DIS):** PointTransformer, MinkowskiEngine (sparse 3D convolutions)  
**Outdoor (SemanticKITTI):** RangeNet++ (project to range image), Cylinder3D

---

## Datasets

| Dataset | Task | Modality |
|---------|------|---------|
| ModelNet40 | Classification | 3D CAD models |
| ShapeNet | Part segmentation | 3D CAD models |
| ScanNet | 3D scene segmentation | RGB-D scans |
| S3DIS | Indoor segmentation | LiDAR + RGB |
| KITTI | 3D detection (driving) | LiDAR + camera |
| nuScenes | 3D detection (driving) | LiDAR + 6 cameras |
| Waymo Open | 3D detection | LiDAR + camera |

---

## Key Interview Points

- PointNet achieves permutation invariance via global max pooling — the key architectural insight.
- PointNet++ adds local structure via FPS + ball query + hierarchical PointNets.
- Sparse convolutions are critical for efficiency: LiDAR point clouds are >95% empty in voxel space.
- PointPillars compresses 3D → 2D (BEV) to leverage fast 2D CNNs — preferred for real-time AD.
- NeRF is per-scene; 3D Gaussian Splatting is faster to train and renders in real-time.
- For autonomous driving detection: CenterPoint on BEV features + camera-LiDAR fusion (BEVFusion) is the current standard.
