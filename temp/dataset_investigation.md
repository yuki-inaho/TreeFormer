# TreeFormer Dataset Investigation Report

**Date:** 2025-11-14  
**Investigation Level:** Very Thorough  
**Purpose:** Comprehensive analysis of Guyot dataset structure and dataloader implementation

---

## Table of Contents

1. [Dataset Directory Structure](#1-dataset-directory-structure)
2. [Data File Format (.pt Files)](#2-data-file-format-pt-files)
3. [DataLoader Implementation](#3-dataloader-implementation)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Creating Custom Training Data](#5-creating-custom-training-data)
6. [Code Examples](#6-code-examples)
7. [Key Implementation Details](#7-key-implementation-details)

---

## 1. Dataset Directory Structure

### 1.1 Expected Directory Layout

The Guyot dataset follows a hierarchical structure with separate train/val/test splits:

```
guyot_data/
├── train/
│   ├── data/           # PyTorch tensor files (.pt) containing graph annotations
│   │   ├── Set02_IMG_3468.pt
│   │   ├── Set02_IMG_3469.pt
│   │   └── ...
│   ├── img/            # Original input images (PNG format)
│   │   ├── Set02_IMG_3468.png
│   │   ├── Set02_IMG_3469.png
│   │   └── ...
│   ├── unet/           # UNet predictions (auxiliary, not used in main loader)
│   │   └── images
│   └── check/          # Visualization/check images (auxiliary)
│       └── images
├── val/
│   ├── data/
│   ├── img/
│   ├── unet/
│   └── check/
└── test/
    ├── data/
    ├── img/
    ├── unet/
    └── check/
```

### 1.2 File Naming Convention

- **Image files**: `{name}.png` (e.g., `Set02_IMG_3468.png`)
- **Data files**: `{name}.pt` (e.g., `Set02_IMG_3468.pt`)
- **Correspondence**: Each `.pt` file must have a corresponding `.png` file with the same base name

### 1.3 Key Directories

| Directory | Purpose | Format | Required |
|-----------|---------|--------|----------|
| `data/` | Ground truth graph annotations | `.pt` PyTorch tensors | Yes |
| `img/` | Input RGB/grayscale images | `.png` | Yes |
| `unet/` | UNet segmentation outputs | `.png` | No (legacy) |
| `check/` | Visualization outputs | `.png` | No (debug) |

---

## 2. Data File Format (.pt Files)

### 2.1 File Structure

Each `.pt` file contains a **Python object** (not a dictionary) with the following attributes:

```python
class DataPoint:
    list_DETR_points_left_up: torch.Tensor  # Node coordinates (normalized)
    DETR_node_collections: torch.Tensor      # Edge connectivity
```

### 2.2 Attribute Details

#### 2.2.1 `list_DETR_points_left_up`

**Type:** `torch.Tensor`  
**Shape:** `[N, 2]` where N is the number of nodes  
**Data Type:** `torch.float32`  
**Coordinate System:** Normalized coordinates in range [0, 1]

- **Column 0:** x-coordinate (normalized by image width)
- **Column 1:** y-coordinate (normalized by image height)

**Example:**
```python
tensor([[0.2350, 0.4120],  # Node 0: x=0.235, y=0.412
        [0.3140, 0.5230],  # Node 1
        [0.4560, 0.6780],  # Node 2
        ...])
```

**To get pixel coordinates:**
```python
pixel_coords = list_DETR_points_left_up * torch.tensor([width, height])
```

#### 2.2.2 `DETR_node_collections`

**Type:** `torch.Tensor`  
**Shape:** `[E, 2]` where E is the number of edges  
**Data Type:** `torch.long`  
**Format:** Edge list representation

- **Column 0:** Source node index
- **Column 1:** Target node index

**Example:**
```python
tensor([[0, 1],   # Edge from node 0 to node 1
        [1, 2],   # Edge from node 1 to node 2
        [1, 3],   # Edge from node 1 to node 3 (branching)
        ...])
```

**Constraints:**
- Forms a **tree structure** (connected, acyclic graph)
- Node 0 is always the **root node**
- Edges represent parent-child relationships in the tree
- Must pass `nx.is_tree()` validation

### 2.3 Loading .pt Files

```python
import torch

# Load the data file
datapoint = torch.load('path/to/data/Set02_IMG_3468.pt')

# Access attributes
points = datapoint.list_DETR_points_left_up  # Shape: [N, 2]
edges = datapoint.DETR_node_collections       # Shape: [E, 2]

print(f"Number of nodes: {points.shape[0]}")
print(f"Number of edges: {edges.shape[0]}")
```

---

## 3. DataLoader Implementation

### 3.1 Dataset Class: `LoadCNNDataset`

**Location:** `train_mst.py` (lines 163-677) and `valid_smd_guyot_nx.py` (lines 182-665)

#### 3.1.1 Constructor Parameters

```python
class LoadCNNDataset(Dataset):
    def __init__(
        self,
        parent_path,              # Path to train/val/test directory
        max_size=1000,            # Maximum image dimension
        max_change_light_rate=0.3, # Brightness augmentation range
        is_train=True,            # Enable training augmentations
        is_rotate=False           # Enable rotation augmentation
    ):
```

#### 3.1.2 Initialization Process

```python
self.parent_path = parent_path
self.tgt_data_path = os.path.join(parent_path, "data")  # .pt files
self.img_path = os.path.join(parent_path, "img")         # .png files
self.file_list = self.processed_file_names               # List of .pt files

# Load all graph data at initialization
ids1, (list_DETR_points_left_up, list_DETR_node_collections) = \
    load_detr_dataset(self.tgt_data_path)

self.ids1 = ids1                                    # File names
self.list_DETR_points_left_up = list_DETR_points_left_up  # All node coords
self.list_DETR_node_collections = list_DETR_node_collections  # All edges
```

**Key Design:** All graph annotations are loaded into memory at initialization for faster access during training.

### 3.2 Data Loading Function

**Location:** `train_mst.py` lines 144-160

```python
def load_detr_dataset(tgt_data_path):
    """
    Load all .pt files from the data directory.
    
    Args:
        tgt_data_path: Path to the 'data' directory containing .pt files
        
    Returns:
        ids: List of .pt file names
        (list_DETR_points_left_up, list_DETR_node_collections): 
            Lists of tensors for all samples
    """
    path_list = []
    for file in os.listdir(tgt_data_path):
        path_list.append(file)

    list_DETR_points_left_up = []
    list_DETR_node_collections = []
    ids = path_list

    for id in ids:
        datapoint = torch.load(tgt_data_path + '/' + id)
        DETR_points_left_up = datapoint.list_DETR_points_left_up
        DETR_node_collections = datapoint.DETR_node_collections

        list_DETR_points_left_up.append(DETR_points_left_up)
        list_DETR_node_collections.append(DETR_node_collections)
        
    return ids, (list_DETR_points_left_up, list_DETR_node_collections)
```

### 3.3 `__getitem__` Method

**Purpose:** Retrieve and process a single sample

**Key Steps:**

1. **Load Graph Data** (from pre-loaded memory)
2. **Load Image** (from disk)
3. **Data Augmentation** (if training)
4. **Generate Auxiliary Targets** (PAFs, heatmaps, masks)
5. **Image Preprocessing** (normalization, resizing)

**Code Flow:**

```python
def __getitem__(self, idx):
    # 1. Get file names
    label_img_name = self.file_list[idx].split(".pt")[0] + ".png"
    label_img_name0 = label_img_name.split(".")[0]
    
    # 2. Get pre-loaded graph data
    list_DETR_points_left_up_idx = self.list_DETR_points_left_up[idx]
    list_DETR_node_collections_idx = self.list_DETR_node_collections[idx]
    
    # 3. Load image
    plt_img = plt.imread(os.path.join(self.img_path, label_img_name))
    plt_img = plt_img.astype(np.float32)
    
    # Handle RGBA images (remove alpha channel)
    if len(plt_img.shape) == 3 and plt_img.shape[2] == 4:
        plt_img = plt_img[:, :, :3]
    
    height, width, channels = plt_img.shape
    
    # 4. Data augmentation (training only)
    nodes_list = list_DETR_points_left_up_idx * torch.tensor([width, height])
    nodes_list = nodes_list.numpy()
    
    if self.is_train:
        result_list = self._augment_one_sample(input_img, nodes_list)
        feature_img, nodes = result_list[1], result_list[2]
        list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)
    else:
        feature_img = input_img
        list_DETR_points_left_up = list_DETR_points_left_up_idx
    
    # 5. Image normalization
    if len(feature_img.shape) == 3 and feature_img.shape[2] == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    feature_img = transform(feature_img)
    
    # 6. Image resizing
    C, height, width = feature_img.shape
    cut_height = height // 2
    cut_width = width // 2
    feature_img = TF.resize(feature_img, size=[cut_height, cut_width])
    
    # Further resize if exceeds max_size
    if max(cut_width, cut_height) > self.max_size:
        if cut_width > cut_height:
            scale = self.max_size / cut_width
            new_width = self.max_size
            new_height = int(cut_height * scale)
        else:
            scale = self.max_size / cut_height
            new_height = self.max_size
            new_width = int(cut_width * scale)
        feature_img = TF.resize(feature_img, size=[new_height, new_width])
    
    # 7. Generate auxiliary targets (PAFs, masks, heatmaps)
    feature_size = (feature_img.shape[1], feature_img.shape[2])
    PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
        list_DETR_node_collections_idx=list_DETR_node_collections_idx,
        list_DETR_points_left_up_idx=list_DETR_points_left_up,
        feature_size=feature_size,
        sigma=3, 
        unet_thickness=3, 
        mask_thickness=6
    )
    
    # 8. Return all components
    return (feature_img.contiguous(), 
            label_img_name0,
            list_DETR_points_left_up, 
            list_DETR_node_collections_idx,
            PAFs_idx, mask_idx, unet_idx, heatmap_idx,
            self.ids1[idx])
```

### 3.4 Data Augmentation Methods

#### 3.4.1 Brightness Adjustment

```python
def _changeLight(self, img):
    flag = random.uniform(
        1 - self.max_change_light_rate, 
        1 + self.max_change_light_rate
    )
    return exposure.adjust_gamma(img, flag)
```

#### 3.4.2 Gaussian Noise

```python
def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
    gasuss_img = image.astype(np.float32)
    noise = np.random.normal(mu, sigma, gasuss_img.shape)
    gauss_noise = gasuss_img + noise
    gauss_noise = np.clip(gauss_noise, 0.0, 1.0)
    return gauss_noise
```

#### 3.4.3 Horizontal Flip

```python
def _flip2(self, img, nodes_list):
    w = img.shape[1]
    img2 = cv2.flip(img, 1)  # Horizontal flip
    
    # Adjust node coordinates
    flip_new_nodes_list = []
    for x, y in nodes_list:
        flip_new_nodes_list.append([w - x, y])
    
    return img2, flip_new_nodes_list
```

#### 3.4.4 Rotation (Optional, Complex)

```python
def _rotate(self, img, nodes_tensor, connect_tensor):
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img2 = cv2.warpAffine(img, M, (width, height))
    
    # Transform node coordinates
    # Remove nodes outside image boundaries
    # Add edge branch extensions
    # Validate tree structure
    
    return img2_tensor, final_nodes_tensor, final_connect_tensor, M
```

### 3.5 Auxiliary Target Generation

#### 3.5.1 Part Affinity Fields (PAFs)

**Purpose:** Encode edge direction and location

```python
def generate_PAFs(height, width, points, paths, line_thickness=2):
    PAFs = np.zeros((height, width, 2), dtype=np.float32)
    
    for branch in paths:
        for idx in range(len(branch) - 1):
            start_point = points[branch[idx]]
            end_point = points[branch[idx + 1]]
            
            # Calculate unit vector
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            ux = (x2 - x1) / length
            uy = (y2 - y1) / length
            
            # Fill along line
            for t in np.linspace(0, 1, int(length)):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    PAFs[y-thickness:y+thickness, x-thickness:x+thickness, 0] = ux
                    PAFs[y-thickness:y+thickness, x-thickness:x+thickness, 1] = uy
    
    return PAFs
```

**Output Shape:** `[height, width, 2]`  
**Channel 0:** x-component of edge direction  
**Channel 1:** y-component of edge direction

#### 3.5.2 Heatmaps

**Purpose:** Encode node locations with Gaussian kernels

```python
def generate_heatmap(normalized_kpts, image_size, sigma):
    H, W = image_size
    heatmap = np.zeros((H, W))
    
    for keypoint in normalized_kpts:
        x_normalized, y_normalized = keypoint
        x = x_normalized * W
        y = y_normalized * H
        
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        gaussian = np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / sigma**2)
        gaussian[gaussian < 0.01] = 0
        heatmap = np.maximum(heatmap, gaussian)
    
    return heatmap
```

**Output Shape:** `[height, width]`  
**Values:** Gaussian peaks at node locations

#### 3.5.3 Masks

**Purpose:** Binary mask of skeleton structure

```python
def create_mask_with_polylines(image_shape, keypoints, segments, thickness=2):
    kpts = keypoints.copy()
    kpts[:, 0] *= image_shape[1]  # Scale to pixel coords
    kpts[:, 1] *= image_shape[0]
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for segment in segments:
        segment_points = kpts[segment].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [segment_points], isClosed=False, 
                      color=1, thickness=thickness)
    
    return mask
```

**Output Shape:** `[height, width]`  
**Values:** 0 (background), 1 (skeleton)

### 3.6 Custom Collate Function

**Location:** `train_mst.py` lines 679-712

**Purpose:** Batch multiple samples together with proper formatting

```python
def custom_collate_fn(batch):
    (feature_img, label_img_name0, list_DETR_points_left_up, 
     list_DETR_node_collections, list_PAFs, list_mask, 
     list_unet, list_heatmap, ids1) = zip(*batch)
    
    ACT_1 = 0.9999999
    ACT_0 = 0.0000001
    
    # Images: Keep as list (variable sizes)
    images = [item.to(torch.float32) for item in feature_img]
    
    # Graph data: Keep as list
    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]
    
    # Auxiliary targets: Concatenate into batch tensors
    PAFs_list_transformed = [PAFs.unsqueeze(0).permute(0, 3, 1, 2) 
                             for PAFs in list_PAFs]
    mask_list_transformed = [mask.unsqueeze(0).unsqueeze(0) 
                             for mask in list_mask]
    unet_list_transformed = [unet.unsqueeze(0).unsqueeze(0) 
                             for unet in list_unet]
    heatmap_list_transformed = [heatmap.unsqueeze(0).unsqueeze(0) 
                                for heatmap in list_heatmap]
    
    PAFs_concatenated = torch.cat(PAFs_list_transformed, 0)
    mask_concatenated = torch.cat(mask_list_transformed, 0).contiguous()
    unet_concatenated = torch.cat(unet_list_transformed, 0)
    heatmap_concatenated = torch.cat(heatmap_list_transformed, 0)
    
    # Clamp values
    PAFs_concatenated = torch.clamp(PAFs_concatenated, min=-ACT_1, max=ACT_1)
    unet_concatenated = torch.clamp(unet_concatenated, min=ACT_0, max=ACT_1)
    heatmap_concatenated = torch.clamp(heatmap_concatenated, min=ACT_0, max=ACT_1)
    
    detr_ids = list(ids1)
    
    return [images, points_left_up, edges,
            PAFs_concatenated, mask_concatenated, 
            unet_concatenated, heatmap_concatenated,
            detr_ids],
```

**Return Format:**
- `images`: List of tensors (variable sizes)
- `points_left_up`: List of tensors [N, 2]
- `edges`: List of tensors [E, 2]
- `PAFs_concatenated`: `[batch_size, 2, H, W]`
- `mask_concatenated`: `[batch_size, 1, H, W]`
- `unet_concatenated`: `[batch_size, 1, H, W]`
- `heatmap_concatenated`: `[batch_size, 1, H, W]`
- `detr_ids`: List of file names

### 3.7 DataLoader Instantiation

**Location:** `train_mst.py` lines 868-892

```python
# Training dataset
train_path = "/path/to/guyot_data/train"
dataset_train = LoadCNNDataset(
    parent_path=train_path,
    max_size=512,              # From config: DATA.MAX_SIZE
    max_change_light_rate=0.3,
    is_train=False,            # Set True for augmentation
    is_rotate=True             # Enable rotation augmentation
)

# Distributed sampler for multi-GPU training
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

# DataLoader
train_loader = DataLoader(
    dataset_train,
    batch_size=8,              # From config: DATA.BATCH_SIZE
    shuffle=False,             # Sampler handles shuffling
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=train_sampler
)

# Validation dataset
val_path = "/path/to/guyot_data/val"
dataset_val = LoadCNNDataset(
    parent_path=val_path,
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,            # No augmentation for validation
    is_rotate=False
)

valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)

val_loader = DataLoader(
    dataset_val,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=valid_sampler
)
```

---

## 4. Data Flow Pipeline

### 4.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATASET INITIALIZATION                                       │
├─────────────────────────────────────────────────────────────────┤
│  LoadCNNDataset(parent_path="guyot_data/train", ...)           │
│    ├─ Scan data/ directory for .pt files                       │
│    ├─ Load all graph annotations into memory                   │
│    └─ Store file list                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. __getitem__(idx) - Per Sample                                │
├─────────────────────────────────────────────────────────────────┤
│  a) Get graph data from memory                                  │
│     - points: [N, 2] normalized coordinates                     │
│     - edges: [E, 2] edge list                                   │
│                                                                  │
│  b) Load image from disk                                        │
│     - Read {name}.png                                           │
│     - Convert to float32, remove alpha if present               │
│                                                                  │
│  c) Data augmentation (if is_train=True)                        │
│     - Random brightness: ±30% gamma adjustment                  │
│     - Gaussian noise: σ=0.1                                     │
│     - Horizontal flip: 50% probability                          │
│     - Update node coordinates accordingly                       │
│                                                                  │
│  d) Image preprocessing                                          │
│     - Normalize: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]     │
│     - Resize: height//2, width//2                               │
│     - Further resize if > max_size (512)                        │
│                                                                  │
│  e) Generate auxiliary targets                                  │
│     - Segment extraction (DFS from root)                        │
│     - PAFs: Edge direction fields [H, W, 2]                     │
│     - Heatmap: Gaussian node locations [H, W]                   │
│     - Mask: Binary skeleton (thick=6) [H, W]                    │
│     - UNet: Binary skeleton (thick=3) [H, W]                    │
│                                                                  │
│  f) Return tuple                                                 │
│     (image, name, points, edges, PAFs, mask, unet, heatmap, id) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. COLLATE_FN - Batch Assembly                                  │
├─────────────────────────────────────────────────────────────────┤
│  Input: List of tuples from __getitem__                         │
│                                                                  │
│  Process:                                                        │
│  - Images: Keep as list (variable sizes)                        │
│  - Points, Edges: Keep as lists (variable graph sizes)          │
│  - PAFs: Stack → [B, 2, H, W], clamp to [-1, 1]                │
│  - Masks: Stack → [B, 1, H, W]                                 │
│  - UNet: Stack → [B, 1, H, W], clamp to [0, 1]                 │
│  - Heatmaps: Stack → [B, 1, H, W], clamp to [0, 1]             │
│                                                                  │
│  Output: [images, points, edges, PAFs, masks, unet,             │
│           heatmaps, ids]                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL FORWARD PASS                                           │
├─────────────────────────────────────────────────────────────────┤
│  TreeFormer model receives batch and processes                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Graph Segmentation Process

The dataset generates tree segments for PAF/mask generation using NetworkX:

```
Input: points, edges
         ↓
    Build NetworkX Graph
         ↓
    Identify branching nodes (degree > 2)
    Identify end nodes (degree = 1, except root)
         ↓
    DFS from root node (0) to extract segments
    - Stop at branching nodes
    - Stop at end nodes
         ↓
    Sort segments by angle (maintain tree order)
         ↓
    Generate PAFs, masks, heatmaps from segments
```

**Example:**
```
Tree structure:
       0
      / \
     1   2
    / \   \
   3   4   5

Edges: [[0,1], [0,2], [1,3], [1,4], [2,5]]
Branching nodes: [0, 1]
End nodes: [3, 4, 5]

Segments:
1. [0, 1]        # Root to first branch
2. [1, 3]        # Branch to leaf
3. [1, 4]        # Branch to leaf
4. [0, 2, 5]     # Root to leaf through branch
```

---

## 5. Creating Custom Training Data

### 5.1 Prerequisites

**Required Libraries:**
```python
import torch
import numpy as np
import cv2
from pathlib import Path
```

**Required Data:**
1. Input images (PNG format)
2. Tree skeleton annotations (nodes + edges)

### 5.2 Data Annotation Requirements

For each plant/tree image, you need to annotate:

1. **Node Positions**: Keypoints along the skeleton
   - Root node must be index 0
   - Coordinates in pixel space

2. **Edge Connectivity**: Parent-child relationships
   - Forms a tree structure (connected, acyclic)
   - Directed edges from root to leaves

**Annotation Guidelines:**
- Start from the root/base of the plant
- Place nodes at:
  - Root point
  - Branch junctions
  - Branch endpoints/tips
  - Regular intervals along long branches
- Connect nodes to form a tree structure
- Ensure no cycles exist

### 5.3 Step-by-Step Data Creation

#### Step 1: Prepare Directory Structure

```python
import os
from pathlib import Path

def create_dataset_structure(base_path):
    """Create the required directory structure."""
    splits = ['train', 'val', 'test']
    subdirs = ['data', 'img', 'unet', 'check']
    
    for split in splits:
        for subdir in subdirs:
            path = Path(base_path) / split / subdir
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path}")

# Usage
create_dataset_structure('guyot_data')
```

#### Step 2: Annotate Images

**Manual Annotation (Recommended Tools):**
- **LabelMe**: For polygon/point annotation
- **VGG Image Annotator (VIA)**: For point annotation
- **Custom GUI**: Using matplotlib or OpenCV

**Annotation Format (JSON example):**
```json
{
  "image_name": "Set02_IMG_3468.png",
  "image_size": [1920, 1080],
  "nodes": [
    {"id": 0, "x": 960, "y": 1000, "type": "root"},
    {"id": 1, "x": 950, "y": 800, "type": "junction"},
    {"id": 2, "x": 970, "y": 800, "type": "junction"},
    {"id": 3, "x": 940, "y": 600, "type": "tip"},
    {"id": 4, "x": 960, "y": 600, "type": "tip"},
    {"id": 5, "x": 980, "y": 600, "type": "tip"}
  ],
  "edges": [
    [0, 1],
    [0, 2],
    [1, 3],
    [1, 4],
    [2, 5]
  ]
}
```

#### Step 3: Convert Annotations to .pt Format

```python
import torch
import numpy as np
import json
import networkx as nx
from pathlib import Path

class DataPointObject:
    """Object to store in .pt file."""
    def __init__(self, points, edges):
        self.list_DETR_points_left_up = points  # torch.Tensor [N, 2]
        self.DETR_node_collections = edges       # torch.Tensor [E, 2]

def validate_tree_structure(edges):
    """Validate that edges form a valid tree."""
    G = nx.Graph()
    G.add_edges_from(edges.tolist())
    
    if not nx.is_tree(G):
        raise ValueError("Edge list does not form a valid tree!")
    
    if not nx.is_connected(G):
        raise ValueError("Graph is not connected!")
    
    return True

def create_pt_file(annotation_json_path, image_path, output_data_path, output_img_path):
    """
    Convert annotation JSON to TreeFormer .pt format.
    
    Args:
        annotation_json_path: Path to JSON annotation file
        image_path: Path to source image
        output_data_path: Output directory for .pt files
        output_img_path: Output directory for images
    """
    # Load annotation
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)
    
    image_name = annotation['image_name']
    base_name = Path(image_name).stem
    image_size = annotation['image_size']  # [width, height]
    width, height = image_size
    
    # Extract nodes and normalize coordinates
    nodes = annotation['nodes']
    num_nodes = len(nodes)
    
    # Sort nodes by ID to ensure consistent ordering
    nodes = sorted(nodes, key=lambda x: x['id'])
    
    # Create normalized coordinate array
    points_array = np.zeros((num_nodes, 2), dtype=np.float32)
    for node in nodes:
        idx = node['id']
        x = node['x']
        y = node['y']
        
        # Normalize to [0, 1]
        points_array[idx, 0] = x / width
        points_array[idx, 1] = y / height
    
    # Convert to tensor
    points_tensor = torch.tensor(points_array, dtype=torch.float32)
    
    # Extract edges
    edges = annotation['edges']
    edges_array = np.array(edges, dtype=np.int64)
    edges_tensor = torch.tensor(edges_array, dtype=torch.long)
    
    # Validate tree structure
    try:
        validate_tree_structure(edges_tensor)
        print(f"✓ Valid tree structure for {image_name}")
    except ValueError as e:
        print(f"✗ Invalid tree structure for {image_name}: {e}")
        return False
    
    # Create data object
    datapoint = DataPointObject(points_tensor, edges_tensor)
    
    # Save .pt file
    pt_filename = f"{base_name}.pt"
    pt_path = Path(output_data_path) / pt_filename
    torch.save(datapoint, pt_path)
    print(f"Saved: {pt_path}")
    
    # Copy image to output directory
    import shutil
    img_filename = f"{base_name}.png"
    img_output_path = Path(output_img_path) / img_filename
    shutil.copy(image_path, img_output_path)
    print(f"Copied: {img_output_path}")
    
    return True

# Usage example
annotation_file = "annotations/Set02_IMG_3468.json"
image_file = "images/Set02_IMG_3468.png"
output_data_dir = "guyot_data/train/data"
output_img_dir = "guyot_data/train/img"

create_pt_file(annotation_file, image_file, output_data_dir, output_img_dir)
```

#### Step 4: Batch Conversion Script

```python
import json
from pathlib import Path
from tqdm import tqdm

def batch_convert_annotations(
    annotation_dir,
    image_dir,
    output_base_dir,
    split='train'
):
    """
    Batch convert all annotations to .pt format.
    
    Args:
        annotation_dir: Directory containing JSON annotations
        image_dir: Directory containing source images
        output_base_dir: Base directory for output (e.g., 'guyot_data')
        split: 'train', 'val', or 'test'
    """
    annotation_dir = Path(annotation_dir)
    image_dir = Path(image_dir)
    output_data_dir = Path(output_base_dir) / split / 'data'
    output_img_dir = Path(output_base_dir) / split / 'img'
    
    # Create output directories
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files
    annotation_files = list(annotation_dir.glob('*.json'))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    success_count = 0
    fail_count = 0
    
    for ann_file in tqdm(annotation_files, desc=f"Converting {split}"):
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        image_name = annotation['image_name']
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            fail_count += 1
            continue
        
        try:
            result = create_pt_file(
                ann_file,
                image_path,
                output_data_dir,
                output_img_dir
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            fail_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(annotation_files)}")

# Usage
batch_convert_annotations(
    annotation_dir='annotations/train',
    image_dir='images/train',
    output_base_dir='guyot_data',
    split='train'
)

batch_convert_annotations(
    annotation_dir='annotations/val',
    image_dir='images/val',
    output_base_dir='guyot_data',
    split='val'
)

batch_convert_annotations(
    annotation_dir='annotations/test',
    image_dir='images/test',
    output_base_dir='guyot_data',
    split='test'
)
```

### 5.4 Annotation Tool (Simple Interactive Version)

```python
import cv2
import numpy as np
import json
from pathlib import Path

class SimpleTreeAnnotator:
    """
    Simple interactive tool for annotating tree structures.
    
    Controls:
    - Left click: Add node
    - Right click: Remove last node
    - 'e': Enter edge mode (click two nodes to connect)
    - 'u': Undo last edge
    - 's': Save annotation
    - 'q': Quit
    """
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        self.display_image = self.image.copy()
        
        self.height, self.width = self.image.shape[:2]
        
        self.nodes = []  # List of (x, y) tuples
        self.edges = []  # List of [node_idx1, node_idx2]
        
        self.edge_mode = False
        self.edge_start = None
        
        self.window_name = 'Tree Annotator'
        
    def draw_annotations(self):
        """Redraw image with current annotations."""
        self.display_image = self.image.copy()
        
        # Draw edges
        for edge in self.edges:
            pt1 = tuple(map(int, self.nodes[edge[0]]))
            pt2 = tuple(map(int, self.nodes[edge[1]]))
            cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw nodes
        for idx, (x, y) in enumerate(self.nodes):
            color = (0, 0, 255) if idx == 0 else (255, 0, 0)
            cv2.circle(self.display_image, (int(x), int(y)), 5, color, -1)
            cv2.putText(self.display_image, str(idx), 
                       (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show mode
        mode_text = "Edge Mode" if self.edge_mode else "Node Mode"
        cv2.putText(self.display_image, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.edge_mode:
                # Select node for edge
                for idx, (nx, ny) in enumerate(self.nodes):
                    dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                    if dist < 10:
                        if self.edge_start is None:
                            self.edge_start = idx
                            print(f"Edge start: node {idx}")
                        else:
                            if self.edge_start != idx:
                                self.edges.append([self.edge_start, idx])
                                print(f"Added edge: {self.edge_start} -> {idx}")
                            self.edge_start = None
                            self.edge_mode = False
                        break
            else:
                # Add node
                self.nodes.append((x, y))
                print(f"Added node {len(self.nodes)-1} at ({x}, {y})")
            
            self.draw_annotations()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last node
            if self.nodes:
                removed = self.nodes.pop()
                print(f"Removed node at {removed}")
                # Remove edges connected to this node
                self.edges = [e for e in self.edges 
                             if len(self.nodes)-1 not in e]
                self.draw_annotations()
    
    def run(self):
        """Run the annotation tool."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Tree Annotation Tool")
        print("====================")
        print("Controls:")
        print("  Left click: Add node")
        print("  Right click: Remove last node")
        print("  'e': Enter edge mode (click two nodes)")
        print("  'u': Undo last edge")
        print("  's': Save annotation")
        print("  'q': Quit")
        print("\nNote: First node (red) is the root!")
        
        self.draw_annotations()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.edge_mode = True
                self.edge_start = None
                print("Entering edge mode - click two nodes to connect")
                self.draw_annotations()
            elif key == ord('u'):
                if self.edges:
                    removed = self.edges.pop()
                    print(f"Removed edge: {removed}")
                    self.draw_annotations()
            elif key == ord('s'):
                self.save_annotation()
        
        cv2.destroyAllWindows()
    
    def save_annotation(self):
        """Save annotation to JSON file."""
        if len(self.nodes) < 2:
            print("Error: Need at least 2 nodes!")
            return
        
        if len(self.edges) < 1:
            print("Error: Need at least 1 edge!")
            return
        
        annotation = {
            'image_name': self.image_path.name,
            'image_size': [self.width, self.height],
            'nodes': [
                {
                    'id': idx,
                    'x': int(x),
                    'y': int(y),
                    'type': 'root' if idx == 0 else 'node'
                }
                for idx, (x, y) in enumerate(self.nodes)
            ],
            'edges': self.edges
        }
        
        output_path = self.image_path.stem + '_annotation.json'
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"Saved annotation to {output_path}")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Edges: {len(self.edges)}")

# Usage
if __name__ == '__main__':
    annotator = SimpleTreeAnnotator('path/to/image.png')
    annotator.run()
```

### 5.5 Validation Script

```python
import torch
import networkx as nx
from pathlib import Path

def validate_dataset(data_dir):
    """
    Validate all .pt files in a dataset directory.
    
    Args:
        data_dir: Path to 'data' directory containing .pt files
    """
    data_dir = Path(data_dir)
    pt_files = list(data_dir.glob('*.pt'))
    
    print(f"Validating {len(pt_files)} files in {data_dir}")
    print("=" * 60)
    
    valid_count = 0
    invalid_count = 0
    
    for pt_file in pt_files:
        try:
            # Load data
            datapoint = torch.load(pt_file)
            
            # Check attributes
            if not hasattr(datapoint, 'list_DETR_points_left_up'):
                print(f"✗ {pt_file.name}: Missing list_DETR_points_left_up")
                invalid_count += 1
                continue
            
            if not hasattr(datapoint, 'DETR_node_collections'):
                print(f"✗ {pt_file.name}: Missing DETR_node_collections")
                invalid_count += 1
                continue
            
            points = datapoint.list_DETR_points_left_up
            edges = datapoint.DETR_node_collections
            
            # Check shapes
            if points.dim() != 2 or points.shape[1] != 2:
                print(f"✗ {pt_file.name}: Invalid points shape {points.shape}")
                invalid_count += 1
                continue
            
            if edges.dim() != 2 or edges.shape[1] != 2:
                print(f"✗ {pt_file.name}: Invalid edges shape {edges.shape}")
                invalid_count += 1
                continue
            
            # Check coordinate range
            if (points < 0).any() or (points > 1).any():
                print(f"✗ {pt_file.name}: Points not in [0,1] range")
                invalid_count += 1
                continue
            
            # Check tree structure
            G = nx.Graph()
            G.add_edges_from(edges.tolist())
            
            if not nx.is_tree(G):
                print(f"✗ {pt_file.name}: Not a valid tree structure")
                invalid_count += 1
                continue
            
            # Check root node (0) exists
            if 0 not in G.nodes():
                print(f"✗ {pt_file.name}: Root node (0) not found")
                invalid_count += 1
                continue
            
            # All checks passed
            print(f"✓ {pt_file.name}: Valid "
                  f"(nodes={points.shape[0]}, edges={edges.shape[0]})")
            valid_count += 1
            
        except Exception as e:
            print(f"✗ {pt_file.name}: Error - {e}")
            invalid_count += 1
    
    print("=" * 60)
    print(f"Validation complete:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Total: {len(pt_files)}")
    
    return valid_count, invalid_count

# Usage
validate_dataset('guyot_data/train/data')
validate_dataset('guyot_data/val/data')
validate_dataset('guyot_data/test/data')
```

---

## 6. Code Examples

### 6.1 Loading and Inspecting a .pt File

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load data
datapoint = torch.load('guyot_data/train/data/Set02_IMG_3468.pt')

# Access attributes
points = datapoint.list_DETR_points_left_up  # [N, 2]
edges = datapoint.DETR_node_collections       # [E, 2]

print(f"Number of nodes: {points.shape[0]}")
print(f"Number of edges: {edges.shape[0]}")
print(f"\nFirst 5 nodes (normalized):")
print(points[:5])
print(f"\nFirst 5 edges:")
print(edges[:5])

# Load corresponding image
image = plt.imread('guyot_data/train/img/Set02_IMG_3468.png')
height, width = image.shape[:2]

# Convert normalized coordinates to pixels
pixel_coords = points * torch.tensor([width, height])

print(f"\nFirst 5 nodes (pixel coordinates):")
print(pixel_coords[:5])
```

### 6.2 Visualizing Annotations

```python
import matplotlib.pyplot as plt
import networkx as nx
import torch

def visualize_annotation(image_path, pt_path):
    """Visualize image with tree skeleton overlay."""
    # Load image
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    
    # Load annotations
    datapoint = torch.load(pt_path)
    points = datapoint.list_DETR_points_left_up
    edges = datapoint.DETR_node_collections
    
    # Convert to pixel coordinates
    pixel_coords = points.numpy() * np.array([width, height])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw edges
    for edge in edges:
        start_idx, end_idx = edge
        start = pixel_coords[start_idx]
        end = pixel_coords[end_idx]
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                'g-', linewidth=2, alpha=0.7)
    
    # Draw nodes
    ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], 
              c='red', s=50, zorder=10)
    
    # Highlight root
    ax.scatter([pixel_coords[0, 0]], [pixel_coords[0, 1]], 
              c='blue', s=100, zorder=11, marker='*')
    
    # Add node labels
    for idx, coord in enumerate(pixel_coords):
        ax.text(coord[0] + 10, coord[1] - 10, str(idx),
               color='white', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_title(f'Tree Skeleton Annotation\n'
                f'Nodes: {len(points)}, Edges: {len(edges)}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_annotation(
    'guyot_data/train/img/Set02_IMG_3468.png',
    'guyot_data/train/data/Set02_IMG_3468.pt'
)
```

### 6.3 Creating a Simple Dataset from Scratch

```python
import torch
import numpy as np
from pathlib import Path

class SimpleTreeDataCreator:
    """Helper class to create simple synthetic tree data."""
    
    @staticmethod
    def create_linear_tree(num_nodes=5):
        """Create a simple linear tree (no branching)."""
        # Nodes along a vertical line
        points = torch.zeros(num_nodes, 2)
        points[:, 0] = 0.5  # x = 0.5 (center)
        points[:, 1] = torch.linspace(0.1, 0.9, num_nodes)  # y varies
        
        # Edges: 0->1, 1->2, 2->3, etc.
        edges = torch.zeros(num_nodes - 1, 2, dtype=torch.long)
        for i in range(num_nodes - 1):
            edges[i] = torch.tensor([i, i + 1])
        
        return points, edges
    
    @staticmethod
    def create_branching_tree():
        """Create a tree with one branching point."""
        #      0
        #      |
        #      1
        #     / \
        #    2   3
        #    |   |
        #    4   5
        
        points = torch.tensor([
            [0.50, 0.90],  # 0: root
            [0.50, 0.70],  # 1: branch point
            [0.35, 0.50],  # 2: left branch
            [0.65, 0.50],  # 3: right branch
            [0.35, 0.30],  # 4: left tip
            [0.65, 0.30],  # 5: right tip
        ], dtype=torch.float32)
        
        edges = torch.tensor([
            [0, 1],  # root to branch
            [1, 2],  # branch to left
            [1, 3],  # branch to right
            [2, 4],  # left to tip
            [3, 5],  # right to tip
        ], dtype=torch.long)
        
        return points, edges
    
    @staticmethod
    def save_tree(points, edges, output_dir, name):
        """Save tree data to .pt file."""
        # Create data object
        class DataPoint:
            def __init__(self, pts, edg):
                self.list_DETR_points_left_up = pts
                self.DETR_node_collections = edg
        
        datapoint = DataPoint(points, edges)
        
        # Save
        output_path = Path(output_dir) / f"{name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(datapoint, output_path)
        
        print(f"Saved: {output_path}")
        return output_path

# Usage: Create sample data
creator = SimpleTreeDataCreator()

# Linear tree
points1, edges1 = creator.create_linear_tree(num_nodes=10)
creator.save_tree(points1, edges1, 'guyot_data/train/data', 'example_linear')

# Branching tree
points2, edges2 = creator.create_branching_tree()
creator.save_tree(points2, edges2, 'guyot_data/train/data', 'example_branch')
```

### 6.4 Testing the DataLoader

```python
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/user/TreeFormer')
from train_mst import LoadCNNDataset, custom_collate_fn

# Create dataset
dataset = LoadCNNDataset(
    parent_path='guyot_data/train',
    max_size=512,
    is_train=False,
    is_rotate=False
)

print(f"Dataset size: {len(dataset)}")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=0  # For debugging
)

# Test one batch
batch = next(iter(dataloader))
images, points, edges, PAFs, masks, unet, heatmaps, ids = batch

print(f"\nBatch contents:")
print(f"  Images: {len(images)} images")
print(f"    - Image 0 shape: {images[0].shape}")
print(f"  Points: {len(points)} graphs")
print(f"    - Graph 0 nodes: {points[0].shape}")
print(f"  Edges: {len(edges)} graphs")
print(f"    - Graph 0 edges: {edges[0].shape}")
print(f"  PAFs shape: {PAFs.shape}")
print(f"  Masks shape: {masks.shape}")
print(f"  UNet shape: {unet.shape}")
print(f"  Heatmaps shape: {heatmaps.shape}")
print(f"  IDs: {ids}")
```

---

## 7. Key Implementation Details

### 7.1 Important Constraints

1. **Tree Structure Requirements:**
   - Must be connected (single component)
   - Must be acyclic (no loops)
   - Node 0 must be the root
   - All nodes must be reachable from root

2. **Coordinate System:**
   - Points stored in **normalized** coordinates [0, 1]
   - Normalized by image dimensions: `(x/width, y/height)`
   - Must be converted to pixels for visualization/PAF generation

3. **File Naming:**
   - .pt and .png files must have matching base names
   - Example: `Set02_IMG_3468.pt` ↔ `Set02_IMG_3468.png`

4. **Data Types:**
   - Points: `torch.float32`
   - Edges: `torch.long`

### 7.2 Common Pitfalls

1. **Incorrect Object Format:**
   ```python
   # WRONG: Saving as dictionary
   data = {'points': points, 'edges': edges}
   torch.save(data, 'file.pt')
   
   # CORRECT: Saving as object with attributes
   class DataPoint:
       def __init__(self):
           self.list_DETR_points_left_up = points
           self.DETR_node_collections = edges
   torch.save(DataPoint(), 'file.pt')
   ```

2. **Forgetting to Normalize Coordinates:**
   ```python
   # WRONG: Pixel coordinates
   points = torch.tensor([[960, 540], [970, 530]])
   
   # CORRECT: Normalized coordinates
   points = torch.tensor([[960/1920, 540/1080], 
                          [970/1920, 530/1080]])
   ```

3. **Invalid Tree Structure:**
   ```python
   # WRONG: Creates a cycle
   edges = torch.tensor([[0,1], [1,2], [2,0]])  # Triangle
   
   # CORRECT: Tree structure
   edges = torch.tensor([[0,1], [1,2]])  # Linear
   ```

4. **Root Node Not Zero:**
   ```python
   # WRONG: Root is node 1
   edges = torch.tensor([[1,0], [1,2]])
   
   # CORRECT: Root is node 0
   edges = torch.tensor([[0,1], [0,2]])
   ```

### 7.3 Performance Considerations

1. **All Graph Data Loaded at Init:**
   - Pro: Fast training iteration
   - Con: High memory usage for large datasets
   - For huge datasets, modify `load_detr_dataset` to load on-demand

2. **Image Loading:**
   - Images loaded from disk in `__getitem__`
   - Consider preprocessing and caching if I/O is bottleneck

3. **Auxiliary Target Generation:**
   - PAFs, masks, heatmaps generated on-the-fly
   - Can be pre-computed and cached for faster loading

### 7.4 Extension Points

**To add new augmentations:**
```python
def _augment_one_sample(self, img, nodes_list):
    # Add your augmentation here
    if random.random() < 0.3:
        img, nodes_list = self._my_custom_augmentation(img, nodes_list)
    
    # ... existing augmentations ...
    return [1, img, nodes, 0]
```

**To add new auxiliary targets:**
```python
def __getitem__(self, idx):
    # ... existing code ...
    
    # Add custom target generation
    my_custom_target = self.generate_my_target(points, edges, image_size)
    
    return (feature_img, label_img_name0,
            points, edges,
            PAFs, mask, unet, heatmap,
            my_custom_target,  # Add here
            file_id)
```

**To modify collate function:**
```python
def custom_collate_fn(batch):
    # Unpack including new target
    (imgs, names, points, edges, PAFs, masks, 
     unet, heatmaps, custom_targets, ids) = zip(*batch)
    
    # Process custom targets
    custom_batch = [item for item in custom_targets]
    
    # Return extended batch
    return [images, points, edges,
            PAFs, masks, unet, heatmaps,
            custom_batch,  # Add here
            ids]
```

---

## Summary

### Quick Reference

**Dataset Structure:**
```
parent_path/
├── data/          # .pt files (graph annotations)
└── img/           # .png files (images)
```

**.pt File Format:**
```python
datapoint.list_DETR_points_left_up  # [N, 2] normalized float32
datapoint.DETR_node_collections      # [E, 2] edge list long
```

**DataLoader:**
```python
from train_mst import LoadCNNDataset, custom_collate_fn

dataset = LoadCNNDataset(
    parent_path='guyot_data/train',
    max_size=512,
    is_train=True,
    is_rotate=False
)

loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=custom_collate_fn,
    num_workers=4
)
```

**Creating Custom Data:**
1. Annotate images (nodes + edges forming tree)
2. Normalize coordinates to [0, 1]
3. Create object with attributes
4. Save with `torch.save()`
5. Validate tree structure
6. Place in correct directory structure

---

## References

**Key Files in TreeFormer:**
- `train_mst.py`: Main training script with dataset implementation
- `valid_smd_guyot_nx.py`: Validation script with dataset variant
- `README.md`: Dataset structure documentation
- `configs/tree_2D_use_mst_only1.yaml`: Configuration file

**External Dependencies:**
- PyTorch: Tensor operations and data loading
- NetworkX: Graph/tree structure validation
- OpenCV: Image processing and augmentation
- NumPy: Numerical operations
- Matplotlib: Visualization

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-14  
**Generated by:** TreeFormer Dataset Investigation
