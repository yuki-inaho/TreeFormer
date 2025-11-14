# TreeFormer Data Directory

This directory contains datasets for training and evaluating TreeFormer models on plant skeleton estimation tasks.

## Available Datasets

### guyot_200_20

Stratified subset of the 3D2cut Single Guyot Dataset for efficient training and testing.

**Structure:**
```
guyot_200_20/
├── 01-TrainAndValidationSet/    # 200 grapevine images
│   ├── Set*.jpeg                 # Original resolution images (4032x3024)
│   └── Set*_annotation.json      # Tree structure annotations
├── 02-IndependentTestSet/        # 20 test images
│   ├── Set*.jpeg
│   └── Set*_annotation.json
└── sampling_metadata.json        # Sampling details
```

**Statistics:**
- Train/Validation: 200 images
- Test: 20 images
- Sampling method: Stratified random by Set (maintaining original distribution)
- Random seed: 42 (reproducible)
- Set distribution: Set00 (0.9%), Set01 (6.6%), Set04 (30.7%), Set05 (31.7%), Set06 (30.1%)

**Annotation Format:**
- Standard: 3D2cut Grapevine Annotation v1.0.0
- Format: JSON with tree structure information
- Contents:
  - VineFeature: Node coordinates, types, parent relationships
  - Feature types: rootCrown, branchNode, pruningCut, branchToPhotoEdge, growingTip
  - Branch labels: root, mainTrunk, oldWood, cane, courson, shoot, lateralShoot

### guyot_dataset_sample_5

Small sample dataset (5 images) for quick testing and development.

- Images: Set00_IMG_3283-3287.jpeg
- Same annotation format as guyot_200_20

### guyot_dataset_quarter

Quarter-resolution version of guyot_dataset_sample_5 (1008x756 pixels).

- Created using tools/resize_guyot_dataset.py
- Annotations scaled proportionally

## Source Dataset

**3D2cut Single Guyot Dataset**
- Provider: Idiap Research Institute, 3D2cut SA
- URL: https://www.idiap.ch/en/scientific-research/data/3d2cut
- Total images: 1,511 grapevine images
  - Training/Validation: 1,254 images
  - Independent Test: 257 images
- Resolution: Typically 4032x3024 pixels
- Background: Artificial white/blue sheets for isolation

## Usage with TreeFormer

### Visualization

Visualize annotations using the provided tool:

```bash
# View with feature type colors
uv run python tools/visualize_guyot_annotations.py \
  --image data/guyot_200_20/01-TrainAndValidationSet/Set04_IMG_0001.jpeg \
  --annotation data/guyot_200_20/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json

# Resize for easier viewing
uv run python tools/visualize_guyot_annotations.py \
  --image data/guyot_200_20/01-TrainAndValidationSet/Set04_IMG_0001.jpeg \
  --annotation data/guyot_200_20/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json \
  --max-height 800 \
  --color-mode branch-label
```