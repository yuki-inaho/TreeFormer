#!/usr/bin/env python3
"""
Guyot Dataset Loader
Minimal implementation for TreeFormer training with Guyot grapevine dataset

Follows DRY/KISS/SOLID principles:
- Single Responsibility: Only handles Guyot data loading
- KISS: Minimal necessary functionality
- No implicit fallbacks: Explicit error handling
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class GuyotDataset(Dataset):
    """
    Guyot Grapevine Dataset Loader

    Dataset structure:
        parent_path/
            01-TrainAndValidationSet/ or 02-IndependentTestSet/
                *.jpeg (images, 1008x756)
                *_annotation.json (VineFeature annotations)

    Args:
        parent_path: Path to guyot_200_20_resized directory
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        target_size: Target image size (width, height), default (512, 512)
    """

    def __init__(
        self,
        parent_path: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """Initialize Guyot dataset"""
        self.parent_path = Path(parent_path)
        self.split = split
        self.target_size = target_size

        # Determine subdirectory based on split
        if split == 'train':
            self.data_dir = self.parent_path / '01-TrainAndValidationSet'
        elif split == 'test':
            self.data_dir = self.parent_path / '02-IndependentTestSet'
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        # Verify directory exists (no implicit fallback)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                f"Expected structure: {parent_path}/01-TrainAndValidationSet/ or /02-IndependentTestSet/"
            )

        # Get list of image files
        self.image_files = sorted(list(self.data_dir.glob('*.jpeg')))
        if not self.image_files:
            raise ValueError(f"No .jpeg files found in {self.data_dir}")

        # Setup transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_files)

    def _load_annotation(self, image_path: Path) -> Dict:
        """
        Load annotation JSON file corresponding to image

        Args:
            image_path: Path to image file

        Returns:
            Annotation dictionary

        Raises:
            FileNotFoundError: If annotation file does not exist
            json.JSONDecodeError: If annotation file is invalid JSON
        """
        # Construct annotation filename
        anno_path = image_path.parent / f"{image_path.stem}_annotation.json"

        # No implicit fallback: raise error if not found
        if not anno_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {anno_path}\n"
                f"Expected for image: {image_path}"
            )

        try:
            with open(anno_path, 'r') as f:
                anno = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in annotation file: {anno_path}",
                e.doc, e.pos
            )

        return anno

    def _parse_vine_features(self, anno: Dict) -> Tuple[List, List]:
        """
        Parse VineFeature annotations into nodes and edges

        Args:
            anno: Annotation dictionary

        Returns:
            (nodes, edges) where:
                nodes: List of [x, y] coordinates
                edges: List of [parent_id, child_id] pairs
        """
        try:
            vine_features = anno['VineImage'][0]['VineFeature'][0]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid annotation structure: {e}")

        nodes = []
        edges = []

        for feature in vine_features:
            feature_id = feature['FeatureID']
            coords = feature['FeatureCoordinates']  # [x, y]
            parent_id = feature['ParentID']

            nodes.append(coords)

            # Create edge if parent exists (ParentID is not null)
            if parent_id is not None:
                edges.append([parent_id, feature_id])

        return nodes, edges

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'image': Transformed image tensor [C, H, W]
                - 'nodes': Node coordinates tensor [N, 2]
                - 'edges': Edge list tensor [E, 2]
                - 'filename': Image filename
        """
        # Load image
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image {image_path}: {e}")

        # Load annotation
        anno = self._load_annotation(image_path)

        # Parse VineFeatures
        nodes, edges = self._parse_vine_features(anno)

        # Transform image
        image_tensor = self.transform(image)

        # Convert to tensors
        nodes_tensor = torch.tensor(nodes, dtype=torch.float32)
        edges_tensor = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 2), dtype=torch.long)

        return {
            'image': image_tensor,
            'nodes': nodes_tensor,
            'edges': edges_tensor,
            'filename': image_path.name
        }


def test_guyot_dataset():
    """Test function for GuyotDataset"""
    dataset = GuyotDataset(
        parent_path='data/guyot_200_20_resized',
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")

    # Test first sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Nodes shape: {sample['nodes'].shape}")
    print(f"Edges shape: {sample['edges'].shape}")
    print(f"Filename: {sample['filename']}")

    return True


if __name__ == '__main__':
    test_guyot_dataset()
