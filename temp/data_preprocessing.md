# TreeFormer Data Preprocessing and Augmentation Pipeline

## Table of Contents
1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Complete Preprocessing Pipeline](#complete-preprocessing-pipeline)
4. [Data Augmentation Techniques](#data-augmentation-techniques)
5. [Annotation Transformation](#annotation-transformation)
6. [Auxiliary Representations](#auxiliary-representations)
7. [Collate Function for Batching](#collate-function-for-batching)
8. [Applying to Custom Data](#applying-to-custom-data)

---

## Overview

TreeFormer uses a sophisticated data preprocessing and augmentation pipeline designed for tree-structured graph extraction from images. The pipeline handles:
- Tree/root system images with keypoint annotations
- Graph structure preservation during transformations
- Multi-modal representations (PAFs, heatmaps, masks)
- Variable-sized batching

**Main Files:**
- `train_mst.py` / `train_unmst.py`: Contains `LoadCNNDataset` class and preprocessing logic
- `epoch.py`: Contains training/validation loops
- `utils.py`: Contains collate functions

---

## Dataset Structure

### Expected Directory Layout
```
parent_path/
├── data/           # Annotation files (.pt format)
│   ├── data_1.pt
│   ├── data_2.pt
│   └── ...
└── img/            # Image files (.png format)
    ├── data_1.png
    ├── data_2.png
    └── ...
```

### Annotation Format (.pt files)
Each `.pt` file contains:
```python
datapoint.list_DETR_points_left_up    # Tensor of normalized keypoints [N, 2]
datapoint.DETR_node_collections        # List of edge connections
```

Example:
```python
# Keypoints (normalized to [0, 1])
list_DETR_points_left_up = torch.tensor([
    [0.5, 0.3],    # node 0
    [0.4, 0.5],    # node 1
    [0.6, 0.7],    # node 2
    # ... more nodes
])

# Edge connections (node collections)
DETR_node_collections = [
    [0, 1, 2],     # path from node 0 -> 1 -> 2
    [2, 3],        # branch from node 2 -> 3
    # ... more paths
]
```

---

## Complete Preprocessing Pipeline

### Step-by-Step Process

```python
class LoadCNNDataset(Dataset):
    def __init__(self, parent_path, max_size=1000,
                 max_change_light_rate=0.3, is_train=True, is_rotate=False):
        """
        Args:
            parent_path: Path to parent directory containing 'data/' and 'img/'
            max_size: Maximum dimension for resized images (default: 1000)
            max_change_light_rate: Range for brightness adjustment (default: 0.3)
            is_train: Whether to apply augmentations (default: True)
            is_rotate: Whether to apply rotation augmentation (default: False)
        """
```

### Pipeline Flow Diagram

```
Load Image & Annotations
         ↓
Apply Augmentations (if is_train=True)
  ├─→ Brightness Adjustment (20% chance)
  ├─→ Gaussian Noise (10% chance)
  └─→ Flip + Light/Noise (70% chance)
         ↓
Convert to Tensor & Normalize
  ├─→ ToTensor()
  └─→ Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
         ↓
Resize (half size)
         ↓
Constraint to Max Size
         ↓
Apply Rotation (if is_rotate=True)
  ├─→ Random angle [-15, 15] degrees
  ├─→ Transform keypoints
  ├─→ Remove out-of-bounds nodes
  ├─→ Extend edge branches to boundaries
  └─→ Validate tree structure
         ↓
Generate Auxiliary Representations
  ├─→ PAFs (Part Affinity Fields)
  ├─→ Heatmaps (Gaussian blobs)
  └─→ Masks (Polylines)
         ↓
Return Batch Item
```

### Detailed Implementation

```python
def __getitem__(self, idx):
    # 1. Load image
    label_img_name = self.file_list[idx].split(".pt")[0] + ".png"
    plt_img = plt.imread(os.path.join(self.img_path, label_img_name)).astype(np.float32)
    
    # Handle RGBA images (convert to RGB)
    if len(plt_img.shape) == 3 and plt_img.shape[2] == 4:
        plt_img = plt_img[:, :, :3]
    
    height, width, channels = plt_img.shape
    
    # 2. Load and denormalize keypoints
    list_DETR_points_left_up_idx = self.list_DETR_points_left_up[idx]
    nodes_list = list_DETR_points_left_up_idx * torch.tensor([width, height])
    nodes_list = nodes_list.numpy()
    
    # 3. Apply augmentations
    if self.is_train:
        result_list = self._augment_one_sample(plt_img, nodes_list)
        feature_img, nodes = result_list[1], result_list[2]
    else:
        feature_img = plt_img
        nodes = list_DETR_points_left_up_idx
    
    list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)
    
    # 4. Normalize image
    if len(feature_img.shape) == 3 and feature_img.shape[2] == 3:
        transform_feature = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform_feature = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    feature_img = transform_feature(feature_img)
    
    # 5. Resize (half size)
    C, height, width = feature_img.shape
    cut_height = height // 2
    cut_width = width // 2
    feature_img = TF.resize(feature_img, size=[cut_height, cut_width])
    
    # 6. Constraint to max_size
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
    
    # 7. Apply rotation (if enabled)
    if self.is_rotate:
        feature_img, list_DETR_points_left_up, list_DETR_node_collections_idx, M = \
            self._rotate(feature_img, list_DETR_points_left_up, list_DETR_node_collections_idx)
        
        # Validate that structure is still a tree
        G_tree = nx.Graph()
        G_tree.add_edges_from(list_DETR_node_collections_idx.tolist())
        if not nx.is_tree(G_tree):
            # Revert to non-rotated version
            feature_img = old_save_img
            list_DETR_points_left_up = old_save_list_DETR_points_left_up
            list_DETR_node_collections_idx = old_save_list_DETR_node_collections_idx
    
    # 8. Generate auxiliary representations
    feature_size = (feature_img.shape[1], feature_img.shape[2])
    PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
        list_DETR_node_collections_idx=list_DETR_node_collections_idx,
        list_DETR_points_left_up_idx=list_DETR_points_left_up,
        feature_size=feature_size,
        sigma=3,
        unet_thickness=3,
        mask_thickness=6
    )
    
    return (feature_img.contiguous(), label_img_name0,
            list_DETR_points_left_up, list_DETR_node_collections_idx,
            PAFs_idx, mask_idx, unet_idx, heatmap_idx,
            self.ids1[idx])
```

---

## Data Augmentation Techniques

### 1. Gaussian Noise Addition

Adds random noise sampled from Gaussian distribution to simulate sensor noise.

```python
def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image (normalized to [0, 1])
        mu: Mean of Gaussian distribution (default: 0.0)
        sigma: Standard deviation (default: 0.1)
    
    Returns:
        Image with added Gaussian noise, clipped to [0, 1]
    """
    gasuss_img = copy.deepcopy(image)
    gasuss_img = gasuss_img.astype(np.float32)
    
    # Generate noise
    noise = np.random.normal(mu, sigma, gasuss_img.shape)
    
    # Add noise and clip
    gauss_noise = gasuss_img + noise
    gauss_noise = np.clip(gauss_noise, 0.0, 1.0)
    
    return gauss_noise

def _addNoise(self, img):
    """Wrapper for noise addition."""
    return self._gasuss_noise(img)
```

**Usage Example:**
```python
# Apply to your image
noisy_image = dataset._addNoise(your_image)
```

---

### 2. Brightness/Gamma Adjustment

Uses gamma correction to adjust image brightness.

```python
def _changeLight(self, img):
    """
    Adjust image brightness using gamma correction.
    
    Args:
        img: Input image
    
    Returns:
        Image with adjusted brightness
    
    The gamma value is randomly sampled from:
        [1 - max_change_light_rate, 1 + max_change_light_rate]
    
    gamma > 1: Darkens the image
    gamma < 1: Brightens the image
    """
    from skimage import exposure
    
    # Random gamma in [0.7, 1.3] if max_change_light_rate=0.3
    flag = random.uniform(
        1 - self.max_change_light_rate,
        1 + self.max_change_light_rate
    )
    
    light_img = copy.deepcopy(img)
    return exposure.adjust_gamma(light_img, flag)
```

**Usage Example:**
```python
# Create dataset with different brightness ranges
dataset = LoadCNNDataset(
    parent_path='./data',
    max_change_light_rate=0.3  # ±30% brightness adjustment
)

# Apply to your image
adjusted_image = dataset._changeLight(your_image)
```

---

### 3. Horizontal Flipping

Flips image and keypoints horizontally.

```python
def _flip2(self, img, nodes_list):
    """
    Flip image and keypoints horizontally.
    
    Args:
        img: Input image (H, W, C)
        nodes_list: List of keypoints [(x, y), ...]
    
    Returns:
        Flipped image and transformed keypoints
    """
    flip_nodes_list = copy.deepcopy(nodes_list)
    flip_img = copy.deepcopy(img)
    w = flip_img.shape[1]
    
    # Flip image (1 = horizontal, 0 = vertical)
    img2 = cv2.flip(flip_img, 1)
    
    # Transform keypoints
    flip_new_nodes_list = list()
    for x, y in flip_nodes_list:
        flip_new_nodes_list.append([w - x, y])
    
    return img2, flip_new_nodes_list
```

**Transformation:**
```
Original point: (x, y)
Flipped point:  (width - x, y)
```

**Usage Example:**
```python
# Apply horizontal flip
flipped_img, flipped_keypoints = dataset._flip2(img, keypoints)
```

---

### 4. Rotation with Graph Preservation

The most sophisticated augmentation - rotates image while preserving tree structure.

```python
def _rotate(self, img, nodes_tensor, connect_tensor):
    """
    Rotate image and transform graph structure.
    
    Args:
        img: Input image tensor (C, H, W)
        nodes_tensor: Normalized keypoints [N, 2]
        connect_tensor: Edge connections [E, 2]
    
    Returns:
        Rotated image, transformed keypoints, updated edges, transformation matrix
    
    Process:
        1. Generate random rotation angle in [-15, 15] degrees
        2. Apply affine transformation to image
        3. Transform all keypoints using affine matrix
        4. Remove nodes outside image boundaries
        5. Extend edge branches to image boundaries
        6. Validate tree structure
    """
    rotate_nodes_tensor = copy.deepcopy(nodes_tensor)
    rotate_img = copy.deepcopy(img).cpu().numpy()
    C, height, width = rotate_img.shape
    
    # Convert from (C, H, W) to (H, W, C)
    rotate_img = np.transpose(rotate_img, (1, 2, 0))
    
    # Random rotation angle
    angle = random.randint(-15, 15)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Apply rotation to image
    img2 = cv2.warpAffine(
        rotate_img, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    # Transform keypoints
    rotate_new_nodes_list = list()
    for x, y in (rotate_nodes_tensor * torch.tensor([width, height])).cpu().numpy():
        x1 = M[0][0] * x + M[0][1] * y + M[0][2]
        y1 = M[1][0] * x + M[1][1] * y + M[1][2]
        rotate_new_nodes_list.append([int(x1), int(y1)])
    
    # Build graph
    G = nx.Graph()
    for i, point in enumerate(rotate_new_nodes_list):
        G.add_node(i, point=point)
    
    for connection in connect_tensor.cpu().numpy().tolist():
        start, end = connection
        if start in G.nodes and end in G.nodes:
            G.add_edge(start, end)
    
    # Remove out-of-bounds nodes
    for node in rotate_new_nodes_list:
        x, y = node
        if not (0 <= x < width and 0 <= y < height):
            index_id = rotate_new_nodes_list.index(node)
            if index_id in G.nodes:
                G.remove_node(index_id)
    
    # Extract remaining nodes
    nodes_data = [G.nodes[node]['point'] for node in G.nodes]
    
    # Normalize and convert to tensor
    final_nodes_tensor = torch.tensor(nodes_data, dtype=torch.float32) / \
                        torch.tensor([width, height], dtype=torch.float32)
    
    # Update edge indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(G.nodes)}
    updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in G.edges]
    final_connect_tensor = torch.tensor(updated_edges, dtype=torch.long)
    
    # Convert image back to (C, H, W)
    img2 = np.transpose(img2, (2, 0, 1))
    img2_tensor = torch.tensor(img2, dtype=torch.float32)
    
    # Extend edge branches to boundaries
    final_nodes_tensor, final_connect_tensor = self._add_edge_branch(
        img, nodes_tensor, connect_tensor,
        final_nodes_tensor, final_connect_tensor, M
    )
    
    return img2_tensor, final_nodes_tensor, final_connect_tensor, M
```

**Rotation Matrix:**
```
M = [cos(θ)  -sin(θ)  tx]
    [sin(θ)   cos(θ)  ty]

Transformed point:
x' = M[0,0] * x + M[0,1] * y + M[0,2]
y' = M[1,0] * x + M[1,1] * y + M[1,2]
```

**Usage Example:**
```python
# Enable rotation during training
dataset = LoadCNNDataset(
    parent_path='./data',
    is_train=True,
    is_rotate=True  # Enable rotation
)

# Rotation is applied automatically in __getitem__
```

---

### 5. Augmentation Orchestration

The main augmentation function that combines different techniques probabilistically.

```python
def _augment_one_sample(self, check_img, nodes_list):
    """
    Apply augmentations with probabilistic selection.
    
    Augmentation Probabilities:
    - 20%: Brightness adjustment only
    - 10%: Gaussian noise only
    - 70%: Combination with flip
        - 56%: Brightness + Flip
        - 7%: Noise + Flip
        - 7%: Brightness + Noise + Flip
    
    Args:
        check_img: Input image (H, W, C)
        nodes_list: Keypoints in pixel coordinates
    
    Returns:
        [success_flag, augmented_img, normalized_nodes, 0]
    """
    height, width, channels = check_img.shape
    a = random.random()
    
    if a < 0.2:
        # 20%: Only brightness
        crop_img = self._changeLight(check_img)
        nodes_list_check = copy.deepcopy(nodes_list)
        
    elif 0.2 <= a < 0.3:
        # 10%: Only noise
        crop_img = self._addNoise(check_img)
        nodes_list_check = copy.deepcopy(nodes_list)
        
    else:
        # 70%: Combinations with flip
        c = random.random()
        if c < 0.8:
            # 56%: Brightness + Flip
            crop_img = self._changeLight(check_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
        elif 0.8 <= c < 0.9:
            # 7%: Noise + Flip
            crop_img = self._addNoise(check_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
        else:
            # 7%: Brightness + Noise + Flip
            crop_img = self._changeLight(check_img)
            crop_img = self._addNoise(crop_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
    
    # Normalize keypoints to [0, 1]
    output_nodes = np.array(nodes_list_check)
    if crop_img.shape[0] == height and crop_img.shape[1] == width:
        output_nodes = output_nodes / np.array([width, height])
        return [1, crop_img, output_nodes, 0]
    else:
        new_height, new_width = crop_img.shape[0], crop_img.shape[1]
        output_nodes = output_nodes / np.array([new_width, new_height])
        new_img = cv2.resize(crop_img, (width, height))
        return [1, new_img, output_nodes, 0]
```

**Augmentation Probability Tree:**
```
100%
├── 20% → Brightness only
├── 10% → Noise only
└── 70% → With Flip
    ├── 56% → Brightness + Flip
    ├── 7%  → Noise + Flip
    └── 7%  → Brightness + Noise + Flip
```

---

## Annotation Transformation

### Coordinate System

TreeFormer uses normalized coordinates in [0, 1] range:

```python
# Pixel coordinates → Normalized
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height

# Normalized → Pixel coordinates
pixel_x = normalized_x * image_width
pixel_y = normalized_y * image_height
```

### Transformation Examples

#### 1. Flip Transformation
```python
# Original
point = (0.7, 0.3)  # normalized coordinates

# After horizontal flip
flipped_point = (1.0 - 0.7, 0.3) = (0.3, 0.3)

# In pixel space (width=800)
original_pixel = (560, 240)
flipped_pixel = (800 - 560, 240) = (240, 240)
```

#### 2. Rotation Transformation
```python
# Given rotation matrix M for 15-degree rotation
M = cv2.getRotationMatrix2D((width/2, height/2), 15, 1)

# Transform point (400, 300) with width=800, height=600
x_new = M[0,0] * 400 + M[0,1] * 300 + M[0,2]
y_new = M[1,0] * 400 + M[1,1] * 300 + M[1,2]
```

### Edge Branch Extension

After rotation, edge branches that exit the image are intelligently extended to boundaries:

```python
def _add_edge_branch(self, img, nodes_tensor, connect_tensor, 
                     final_nodes_tensor, final_connect_tensor, M):
    """
    Extend edge branches to image boundaries after rotation.
    
    This prevents information loss when rotation causes branches
    to exit the image boundaries.
    
    Process:
        1. Identify end nodes (degree = 1)
        2. Find nearest image edge for each end node's child
        3. Calculate intersection point with that edge
        4. Add new node at intersection if valid
        5. Update graph connections
    
    Args:
        img: Original image
        nodes_tensor: Original keypoints
        connect_tensor: Original edges
        final_nodes_tensor: Rotated keypoints
        final_connect_tensor: Rotated edges
        M: Rotation matrix
    
    Returns:
        Updated nodes and edges with extended branches
    """
    # Define image edges as line equations
    edge_func = {
        'top': [0, 1, 0],           # y = 0
        'bottom': [0, 1, -height],  # y = height
        'left': [1, 0, 0],          # x = 0
        'right': [1, 0, -width]     # x = width
    }
    
    # ... (implementation details in code)
    
    return final_rotate_nodes_tensor, final_rotate_connect_tensor
```

**Example:**
```
Before rotation:          After rotation (with extension):
     
  ●---●---●                   ●
   \                           \
    ●                           ●---●---●
                                         \
                                          ●---● (extended to boundary)
```

### Graph Structure Validation

```python
# After rotation, validate tree structure
G_tree = nx.Graph()
G_tree.add_edges_from(list_DETR_node_collections_idx.tolist())

if not nx.is_tree(G_tree):
    # Revert to original if structure is corrupted
    feature_img = old_save_img
    list_DETR_points_left_up = old_save_list_DETR_points_left_up
    list_DETR_node_collections_idx = old_save_list_DETR_node_collections_idx
```

**Tree Properties Checked:**
- Connected graph
- No cycles
- N nodes → N-1 edges

---

## Auxiliary Representations

TreeFormer generates multiple representations to aid training:

### 1. Part Affinity Fields (PAFs)

Encodes edge direction and location as vector fields.

```python
def generate_PAFs(height, width, points, paths, line_thickness=2):
    """
    Generate Part Affinity Fields for edge representation.
    
    PAFs encode both the location and direction of edges in the graph.
    Each pixel along an edge contains a unit vector pointing along the edge.
    
    Args:
        height, width: Image dimensions
        points: Normalized keypoints
        paths: List of paths (sequences of node indices)
        line_thickness: Thickness of PAF region (default: 2)
    
    Returns:
        PAFs: Array of shape (H, W, 2) containing unit vectors
    
    For each edge (node_i → node_j):
        - Calculate unit direction vector: (ux, uy)
        - Paint this vector in a thick line between the nodes
        - PAFs[y, x, :] = (ux, uy) for all pixels in the line
    """
    PAFs = np.zeros((height, width, 2), dtype=np.float32)
    
    for branch in paths:
        for idx in range(len(branch) - 1):
            start_point = points[branch[idx]]
            end_point = points[branch[idx + 1]]
            
            # Convert to pixel coordinates
            x1, y1 = int(start_point[0] * width), int(start_point[1] * height)
            x2, y2 = int(end_point[0] * width), int(end_point[1] * height)
            
            # Calculate unit vector
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length == 0:
                continue
            ux = (x2 - x1) / length
            uy = (y2 - y1) / length
            
            # Paint PAF along the line
            for t in np.linspace(0, 1, int(length)):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    PAFs[y-line_thickness:y+line_thickness, 
                         x-line_thickness:x+line_thickness, 0] = ux
                    PAFs[y-line_thickness:y+line_thickness, 
                         x-line_thickness:x+line_thickness, 1] = uy
    
    return PAFs
```

**Visual Representation:**
```
Image with edges:        PAF Visualization:

    ●                        ↗
     \                      ↗
      ●---●              →→→
           \            ↘
            ●          ↘

(Arrows show direction and magnitude of PAF vectors)
```

**Usage:**
```python
PAFs = generate_PAFs(
    height=512,
    width=512,
    points=keypoints,  # Normalized [0, 1]
    paths=sorted_segments,
    line_thickness=2
)

# PAFs.shape = (512, 512, 2)
# PAFs[:, :, 0] = x-component of unit vectors
# PAFs[:, :, 1] = y-component of unit vectors
```

---

### 2. Gaussian Heatmaps

Represents keypoint locations as Gaussian blobs.

```python
def generate_heatmap(normalized_kpts, image_size, sigma):
    """
    Generate Gaussian heatmap for keypoint localization.
    
    Each keypoint is represented as a 2D Gaussian blob.
    Multiple overlapping Gaussians use max pooling.
    
    Args:
        normalized_kpts: Keypoints in [0, 1] range
        image_size: (height, width)
        sigma: Standard deviation of Gaussian (controls blob size)
    
    Returns:
        Heatmap: Array of shape (H, W) with values in [0, 1]
    
    Gaussian formula:
        G(x, y) = exp(-0.5 * ((x - x_kp)^2 + (y - y_kp)^2) / sigma^2)
    """
    H, W = image_size
    heatmap = np.zeros((H, W))
    
    for keypoint in normalized_kpts:
        x_normalized, y_normalized = keypoint
        x = x_normalized * W
        y = y_normalized * H
        
        # Create meshgrid
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        
        # Compute Gaussian
        gaussian = np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / sigma**2)
        
        # Threshold small values
        gaussian[gaussian < 0.01] = 0
        
        # Max pooling for overlapping Gaussians
        heatmap = np.maximum(heatmap, gaussian)
    
    return heatmap
```

**Visual Representation:**
```
Keypoints:              Heatmap:

   ●                      ▓▓▓
                         ▓███▓
       ●               ▓▓▓███▓▓▓
                              ▓███▓
                              ▓▓▓▓▓

(Brightness indicates Gaussian intensity)
```

**Sigma Parameter Effect:**
```
sigma=1 (tight):        sigma=3 (medium):       sigma=5 (wide):
    ██                      ▓▓▓▓                  ░░▓▓▓▓░░
    ██                    ▓▓████▓▓              ░░▓▓████▓▓░░
                          ▓▓████▓▓            ░░▓▓████████▓▓░░
                            ▓▓▓▓              ░░▓▓████████▓▓░░
                                                ░░▓▓▓▓▓▓░░
```

---

### 3. Polyline Masks

Binary masks representing tree structure as thick lines.

```python
def create_mask_with_polylines(image_shape, keypoints, segments, thickness=2):
    """
    Create binary mask with polylines for tree structure.
    
    Args:
        image_shape: (height, width)
        keypoints: Normalized keypoints [N, 2]
        segments: List of paths (sequences of node indices)
        thickness: Line thickness in pixels
    
    Returns:
        Binary mask of shape (H, W) with 1s along tree structure
    
    Uses cv2.polylines to draw connected segments.
    """
    kpts = copy.deepcopy(keypoints)
    
    # Scale keypoints to image dimensions
    kpts[:, 0] *= image_shape[1]
    kpts[:, 1] *= image_shape[0]
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for segment in segments:
        # Extract points for this segment
        segment_points = kpts[segment].reshape((-1, 1, 2)).astype(np.int32)
        
        # Draw polyline
        cv2.polylines(
            mask,
            [segment_points],
            isClosed=False,
            color=1,
            thickness=thickness
        )
    
    return mask
```

**Thickness Comparison:**
```
thickness=2:            thickness=4:            thickness=6:
    ●                       ●                       ●
    ║                      ║║                      ║║║
    ●══●                  ●════●                  ●══════●
        ║                     ║║                     ║║║
        ●                     ●                      ●
```

**Multiple Representations:**
```python
# Generated in generate_PAFs_by_idx()

# 1. Mask for loss computation (thick)
PAFs_mask = create_mask_with_polylines(
    orig_size, kpts, segments, thickness=6
)
mask_tensor = torch.tensor(PAFs_mask, dtype=torch.bool)

# 2. UNet auxiliary target (medium)
PAFs_unet = create_mask_with_polylines(
    orig_size, kpts, segments, thickness=2
)
unet_tensor = torch.tensor(PAFs_unet, dtype=torch.float32)
```

---

### 4. Graph Segmentation

Segments are identified using DFS traversal:

```python
def find_segments_v2(start_node, node_collections, branching_nodes, end_nodes):
    """
    Find all path segments in the tree using DFS.
    
    A segment is a path from:
        - Start node → branching node
        - Branching node → branching node
        - Branching node → end node
        - Start node → end node (if no branching)
    
    Args:
        start_node: Root of the tree (usually node 0)
        node_collections: List of edge connections
        branching_nodes: Nodes with degree > 2
        end_nodes: Leaf nodes with degree = 1
    
    Returns:
        List of segments, each segment is a list of node indices
    """
    segments = []
    visited_nodes = set()
    
    def dfs(node, path):
        visited_nodes.add(node)
        path.append(node)
        
        if node in branching_nodes:
            # Save path up to branching point
            segments.append(path.copy())
            # Start new paths from branching node
            for collection in node_collections:
                if node in collection:
                    for neighbor in collection:
                        if neighbor not in visited_nodes:
                            dfs(neighbor, [node])
            return
        
        if node in end_nodes:
            # Reached a leaf
            segments.append(path.copy())
            return
        
        # Continue along path
        for collection in node_collections:
            if node in collection:
                for neighbor in collection:
                    if neighbor not in visited_nodes:
                        dfs(neighbor, path.copy())
    
    dfs(start_node, [])
    return segments
```

**Example Segmentation:**
```
Tree Structure:
        0 (start)
        |
        1
       / \
      2   3 (branching)
     / \   \
    4   5   6

Segments:
[0, 1, 3]        # Start → branching
[3, 2]           # Branching → branching  
[2, 4]           # Branching → end
[2, 5]           # Branching → end
[3, 6]           # Branching → end
```

---

## Collate Function for Batching

TreeFormer handles variable-sized images and graphs using a custom collate function:

```python
def custom_collate_fn(batch):
    """
    Custom collate function for batching variable-sized samples.
    
    Args:
        batch: List of items from __getitem__
               Each item is (feature_img, label_img_name0, 
                           list_DETR_points_left_up, list_DETR_node_collections,
                           PAFs_idx, mask_idx, unet_idx, heatmap_idx, ids1)
    
    Returns:
        Batched data:
        - images: List of tensors (variable sizes)
        - points_left_up: List of keypoint tensors
        - edges: List of edge tensors
        - PAFs_concatenated: [B, 2, H, W]
        - mask_concatenated: [B, 1, H, W]
        - unet_concatenated: [B, 1, H, W]
        - heatmap_concatenated: [B, 1, H, W]
        - detr_ids: List of IDs
    """
    (feature_img, label_img_name0, list_DETR_points_left_up, list_DETR_node_collections,
     list_PAFs, list_mask, list_unet, list_heatmap, ids1) = zip(*batch)
    
    # Numerical stability constants
    ACT_1 = 0.9999999  # Max value for clamping
    ACT_0 = 0.0000001  # Min value for clamping
    
    # 1. Keep images as list (variable sizes)
    images = [item.to(torch.float32) for item in feature_img]
    
    # 2. Keep keypoints and edges as lists
    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]
    
    # 3. Transform and concatenate PAFs
    # PAFs: (H, W, 2) → (1, 2, H, W)
    PAFs_list_transformed = [PAFs.unsqueeze(0).permute(0, 3, 1, 2) 
                             for PAFs in list_PAFs]
    
    # 4. Transform and concatenate masks
    # Masks: (H, W) → (1, 1, H, W)
    mask_list_transformed = [mask.unsqueeze(0).unsqueeze(0) 
                            for mask in list_mask]
    unet_list_transformed = [unet.unsqueeze(0).unsqueeze(0) 
                            for unet in list_unet]
    heatmap_list_transformed = [heatmap.unsqueeze(0).unsqueeze(0) 
                               for heatmap in list_heatmap]
    
    # 5. Concatenate along batch dimension
    PAFs_concatenated = torch.cat(PAFs_list_transformed, 0)
    mask_concatenated = torch.cat(mask_list_transformed, 0).contiguous()
    unet_concatenated = torch.cat(unet_list_transformed, 0)
    heatmap_concatenated = torch.cat(heatmap_list_transformed, 0)
    
    # 6. Clamp values for numerical stability
    PAFs_concatenated = torch.clamp(PAFs_concatenated, min=-ACT_1, max=ACT_1)
    unet_concatenated = torch.clamp(unet_concatenated, min=ACT_0, max=ACT_1)
    heatmap_concatenated = torch.clamp(heatmap_concatenated, min=ACT_0, max=ACT_1)
    
    # 7. Prepare IDs
    detr_ids = list(ids1)
    
    return [images, points_left_up, edges,
            PAFs_concatenated, mask_concatenated, unet_concatenated, heatmap_concatenated,
            detr_ids],
```

### Batch Structure

```python
# Example batch with batch_size=2

batch = {
    'images': [
        torch.Tensor([3, 512, 384]),  # Image 1: 512x384
        torch.Tensor([3, 480, 640]),  # Image 2: 480x640 (different size!)
    ],
    
    'points_left_up': [
        torch.Tensor([25, 2]),  # Image 1: 25 keypoints
        torch.Tensor([18, 2]),  # Image 2: 18 keypoints (different count!)
    ],
    
    'edges': [
        torch.Tensor([24, 2]),  # Image 1: 24 edges
        torch.Tensor([17, 2]),  # Image 2: 17 edges
    ],
    
    'PAFs_concatenated': torch.Tensor([2, 2, 512, 384]),
    'mask_concatenated': torch.Tensor([2, 1, 512, 384]),
    'unet_concatenated': torch.Tensor([2, 1, 512, 384]),
    'heatmap_concatenated': torch.Tensor([2, 1, 512, 384]),
    
    'detr_ids': ['data_1', 'data_2']
}
```

### DataLoader Setup

```python
from torch.utils.data import DataLoader

# Training dataset
dataset_train = LoadCNNDataset(
    parent_path='./train',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=True,
    is_rotate=True
)

# Training loader
train_loader = DataLoader(
    dataset_train,
    batch_size=8,
    shuffle=False,  # Using DistributedSampler
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=train_sampler  # DistributedSampler
)

# Validation dataset (no augmentation)
dataset_val = LoadCNNDataset(
    parent_path='./val',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,   # No augmentation
    is_rotate=False   # No rotation
)

# Validation loader
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

## Applying to Custom Data

### Step 1: Prepare Your Data

#### Directory Structure
```bash
your_dataset/
├── train/
│   ├── data/
│   │   ├── sample_001.pt
│   │   ├── sample_002.pt
│   │   └── ...
│   └── img/
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
├── val/
│   ├── data/
│   │   └── ...
│   └── img/
│       └── ...
└── test/
    ├── data/
    │   └── ...
    └── img/
        └── ...
```

#### Create Annotation Files

```python
import torch

class AnnotationData:
    """Container for annotation data."""
    def __init__(self, keypoints, edges):
        self.list_DETR_points_left_up = keypoints
        self.DETR_node_collections = edges

# Example: Creating annotation for a tree structure
def create_annotation_example():
    """
    Create annotation for this tree:
        0 (root)
        |
        1
       / \
      2   3
      |   |
      4   5
    """
    # Keypoints (normalized to [0, 1])
    keypoints = torch.tensor([
        [0.5, 0.1],   # Node 0: root at top-center
        [0.5, 0.3],   # Node 1: below root
        [0.3, 0.6],   # Node 2: left branch
        [0.7, 0.6],   # Node 3: right branch
        [0.3, 0.9],   # Node 4: left leaf
        [0.7, 0.9],   # Node 5: right leaf
    ], dtype=torch.float32)
    
    # Edge collections (paths in the tree)
    # Each path is a sequence of connected nodes
    edges = [
        [0, 1, 2, 4],  # Path: root → node1 → node2 → leaf4
        [1, 3, 5],     # Path: node1 → node3 → leaf5
    ]
    
    # Create annotation object
    annotation = AnnotationData(keypoints, edges)
    
    # Save to file
    torch.save(annotation, 'sample_001.pt')
    
    return annotation

# Create multiple annotations
for i in range(100):
    annotation = create_annotation_for_sample(i)
    torch.save(annotation, f'train/data/sample_{i:03d}.pt')
```

#### Annotation Format Details

```python
# Keypoints format
keypoints.shape = (N, 2)  # N = number of nodes
keypoints.dtype = torch.float32
# Values in range [0, 1] (normalized)
# keypoints[i] = [x_normalized, y_normalized]

# Example
keypoints = torch.tensor([
    [0.25, 0.30],  # Node 0 at 25% width, 30% height
    [0.50, 0.60],  # Node 1 at 50% width, 60% height
    [0.75, 0.90],  # Node 2 at 75% width, 90% height
])

# Edges format (node collections)
# List of paths, each path is a list of node indices
edges = [
    [0, 1, 2],     # Path connecting nodes 0→1→2
    [1, 3],        # Branch from node 1→3
]

# Important: Edges must form a valid tree structure
# - Connected (all nodes reachable from root)
# - Acyclic (no loops)
# - Root is typically node 0
```

---

### Step 2: Create Dataset

```python
from train_mst import LoadCNNDataset, custom_collate_fn
from torch.utils.data import DataLoader

# Training dataset with augmentation
train_dataset = LoadCNNDataset(
    parent_path='./your_dataset/train',
    max_size=512,              # Max image dimension
    max_change_light_rate=0.3, # ±30% brightness
    is_train=True,             # Enable augmentation
    is_rotate=True             # Enable rotation
)

# Validation dataset without augmentation
val_dataset = LoadCNNDataset(
    parent_path='./your_dataset/val',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,   # Disable augmentation
    is_rotate=False   # Disable rotation
)

# Test dataset
test_dataset = LoadCNNDataset(
    parent_path='./your_dataset/test',
    max_size=512,
    is_train=False,
    is_rotate=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

---

### Step 3: Create DataLoaders

```python
# Training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Validation loader
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)

# Test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)
```

---

### Step 4: Iterate and Visualize

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch_item(batch, idx=0):
    """
    Visualize a single item from a batch.
    
    Args:
        batch: Output from DataLoader
        idx: Index of item to visualize (default: 0)
    """
    images, points, edges, PAFs, masks, unets, heatmaps, ids = batch[0]
    
    # Get specific item
    img = images[idx].cpu().numpy()
    pts = points[idx].cpu().numpy()
    edg = edges[idx].cpu().numpy()
    paf = PAFs[idx].cpu().numpy()
    mask = masks[idx].cpu().numpy()
    heatmap = heatmaps[idx].cpu().numpy()
    
    # Denormalize image
    img = (img * 0.5) + 0.5  # Reverse normalization
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Image with keypoints
    axes[0, 1].imshow(img)
    H, W = img.shape[:2]
    # Convert normalized to pixel coords
    pixel_pts = pts * np.array([W, H])
    axes[0, 1].scatter(pixel_pts[:, 0], pixel_pts[:, 1], 
                       c='red', s=50, marker='o')
    for i, pt in enumerate(pixel_pts):
        axes[0, 1].text(pt[0], pt[1], str(i), 
                       color='white', fontsize=8)
    axes[0, 1].set_title('Keypoints')
    axes[0, 1].axis('off')
    
    # 3. Image with graph structure
    axes[0, 2].imshow(img)
    axes[0, 2].scatter(pixel_pts[:, 0], pixel_pts[:, 1], 
                       c='red', s=50, marker='o')
    # Draw edges
    for edge in edg:
        pt1 = pixel_pts[int(edge[0])]
        pt2 = pixel_pts[int(edge[1])]
        axes[0, 2].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
                       'b-', linewidth=2)
    axes[0, 2].set_title('Graph Structure')
    axes[0, 2].axis('off')
    
    # 4. Heatmap
    axes[1, 0].imshow(heatmap.squeeze(), cmap='hot')
    axes[1, 0].set_title('Keypoint Heatmap')
    axes[1, 0].axis('off')
    
    # 5. Mask
    axes[1, 1].imshow(mask.squeeze(), cmap='gray')
    axes[1, 1].set_title('Tree Structure Mask')
    axes[1, 1].axis('off')
    
    # 6. PAF magnitude
    paf_magnitude = np.sqrt(paf[0]**2 + paf[1]**2)
    axes[1, 2].imshow(paf_magnitude, cmap='viridis')
    axes[1, 2].set_title('PAF Magnitude')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'visualization_{ids[idx]}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Visualize first batch
for batch in train_loader:
    visualize_batch_item(batch, idx=0)
    break
```

---

### Step 5: Custom Augmentation Configuration

```python
class CustomDataset(LoadCNNDataset):
    """Custom dataset with modified augmentation."""
    
    def _augment_one_sample(self, check_img, nodes_list):
        """
        Override augmentation function.
        
        Custom augmentation probabilities:
        - 30%: Brightness only
        - 20%: Noise only  
        - 50%: Combinations
        """
        height, width, channels = check_img.shape
        a = random.random()
        
        if a < 0.3:
            # 30%: Brightness
            crop_img = self._changeLight(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
            
        elif 0.3 <= a < 0.5:
            # 20%: Noise
            crop_img = self._addNoise(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
            
        else:
            # 50%: Combinations
            crop_img = self._changeLight(check_img)
            crop_img = self._addNoise(crop_img)
            crop_img, nodes_list_check = self._flip2(
                img=crop_img, 
                nodes_list=nodes_list
            )
        
        # Normalize
        output_nodes = np.array(nodes_list_check)
        output_nodes = output_nodes / np.array([width, height])
        
        return [1, crop_img, output_nodes, 0]

# Use custom dataset
custom_train = CustomDataset(
    parent_path='./your_dataset/train',
    max_size=512,
    is_train=True,
    is_rotate=True
)
```

---

### Step 6: Advanced Usage - Add Custom Augmentations

```python
class AdvancedDataset(LoadCNNDataset):
    """Dataset with additional augmentation techniques."""
    
    def _add_blur(self, img):
        """Add Gaussian blur."""
        import cv2
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def _add_contrast(self, img, alpha=None):
        """Adjust contrast."""
        if alpha is None:
            alpha = random.uniform(0.7, 1.3)
        return np.clip(img * alpha, 0, 1)
    
    def _elastic_transform(self, img, nodes_list):
        """Apply elastic deformation (advanced)."""
        # Implementation of elastic transform
        # This would require careful handling of keypoints
        pass
    
    def _augment_one_sample(self, check_img, nodes_list):
        """Enhanced augmentation pipeline."""
        height, width, channels = check_img.shape
        a = random.random()
        
        # Apply base augmentations
        if a < 0.2:
            crop_img = self._changeLight(check_img)
            crop_img = self._add_contrast(crop_img)
            nodes_list_check = copy.deepcopy(nodes_list)
            
        elif 0.2 <= a < 0.4:
            crop_img = self._addNoise(check_img)
            crop_img = self._add_blur(crop_img)
            nodes_list_check = copy.deepcopy(nodes_list)
            
        else:
            # Complex combinations
            crop_img = self._changeLight(check_img)
            crop_img = self._add_contrast(crop_img)
            
            if random.random() < 0.5:
                crop_img = self._add_blur(crop_img)
            
            crop_img, nodes_list_check = self._flip2(
                img=crop_img,
                nodes_list=nodes_list
            )
        
        # Normalize
        output_nodes = np.array(nodes_list_check)
        output_nodes = output_nodes / np.array([width, height])
        
        return [1, crop_img, output_nodes, 0]

# Use advanced dataset
advanced_train = AdvancedDataset(
    parent_path='./your_dataset/train',
    max_size=512,
    is_train=True,
    is_rotate=True
)
```

---

### Step 7: Validation and Debugging

```python
def validate_dataset(dataset, num_samples=10):
    """
    Validate dataset annotations and preprocessing.
    
    Checks:
    - Images load correctly
    - Keypoints are in valid range [0, 1]
    - Edges form valid tree structure
    - No NaN or Inf values
    - Augmentations preserve graph structure
    """
    import networkx as nx
    
    issues = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            # Get sample
            sample = dataset[i]
            img, name, kpts, edges, paf, mask, unet, heatmap, id_ = sample
            
            # Check image
            if torch.isnan(img).any() or torch.isinf(img).any():
                issues.append(f"Sample {i}: Image contains NaN or Inf")
            
            # Check keypoints range
            if (kpts < 0).any() or (kpts > 1).any():
                issues.append(f"Sample {i}: Keypoints out of range [0, 1]")
            
            # Check tree structure
            G = nx.Graph()
            G.add_edges_from(edges.numpy().tolist())
            if not nx.is_tree(G):
                issues.append(f"Sample {i}: Edges don't form a valid tree")
            
            # Check for disconnected nodes
            if kpts.shape[0] != len(G.nodes):
                issues.append(f"Sample {i}: Mismatch between keypoints and graph nodes")
            
            # Check auxiliary representations
            if torch.isnan(paf).any():
                issues.append(f"Sample {i}: PAF contains NaN")
            if torch.isnan(heatmap).any():
                issues.append(f"Sample {i}: Heatmap contains NaN")
                
        except Exception as e:
            issues.append(f"Sample {i}: Error - {str(e)}")
    
    # Print results
    if not issues:
        print(f"✓ All {num_samples} samples validated successfully!")
    else:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return issues

# Validate your dataset
issues = validate_dataset(train_dataset, num_samples=100)
```

---

## Summary

### Key Features of TreeFormer Preprocessing

1. **Flexible Image Handling**
   - Supports variable image sizes
   - Automatic RGBA to RGB conversion
   - Configurable maximum size constraint

2. **Sophisticated Augmentation**
   - Brightness adjustment via gamma correction
   - Gaussian noise addition
   - Horizontal flipping with keypoint transformation
   - Rotation with graph structure preservation
   - Probabilistic combination of techniques

3. **Graph-Aware Transformations**
   - Keypoints and edges transformed together
   - Out-of-bounds nodes removed
   - Edge branches extended to boundaries
   - Tree structure validation after rotation

4. **Multi-Modal Representations**
   - Part Affinity Fields (PAFs) for edge encoding
   - Gaussian heatmaps for keypoint localization
   - Binary masks for structure representation
   - Multiple thickness variants

5. **Efficient Batching**
   - Handles variable-sized images as lists
   - Variable number of nodes/edges per sample
   - Fixed-size auxiliary representations
   - Numerical stability through clamping

### Configuration Parameters

```python
# Key parameters to tune
LoadCNNDataset(
    parent_path='./data',
    max_size=512,              # Image size constraint (default: 1000)
    max_change_light_rate=0.3, # Brightness range ±30% (default: 0.3)
    is_train=True,             # Enable augmentation
    is_rotate=False            # Enable rotation (default: False)
)

# Rotation parameters (hardcoded in _rotate)
angle = random.randint(-15, 15)  # Rotation range

# Auxiliary representation parameters
generate_PAFs_by_idx(
    ...,
    sigma=3,           # Heatmap Gaussian std (default: 3)
    unet_thickness=3,  # UNet mask thickness (default: 2)
    mask_thickness=6   # Loss mask thickness (default: 6)
)
```

### Common Pitfalls and Solutions

| Issue | Solution |
|-------|----------|
| Keypoints out of bounds after flip | Ensure keypoints are normalized before flip |
| Tree structure broken after rotation | Check `nx.is_tree()` validation is enabled |
| NaN in PAFs | Verify edges don't have zero length |
| Memory issues with large batches | Reduce batch size or max_size parameter |
| Slow data loading | Increase num_workers in DataLoader |
| Augmentation too strong | Reduce max_change_light_rate, disable rotation |

---

## References

- **Main training script**: `/home/user/TreeFormer/train_mst.py`
- **Epoch functions**: `/home/user/TreeFormer/epoch.py`
- **Utilities**: `/home/user/TreeFormer/utils.py`
- **Config example**: `/home/user/TreeFormer/configs/tree_2D_use_mst_only1.yaml`

---

**Document created:** 2025-11-14  
**TreeFormer version:** Based on repository code analysis  
**Author:** Automated documentation system
