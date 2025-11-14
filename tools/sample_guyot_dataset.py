#!/usr/bin/env python3
"""
Guyot Dataset Sampler

Creates stratified random subsets from the 3D2cut Single Guyot dataset.
Maintains Set distribution to preserve dataset characteristics.

Usage:
    python sample_guyot_dataset.py --input <source_dir> --output <output_dir> --train-val 200 --test 20 --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def collect_image_files(dataset_dir: Path, subdirs: List[str]) -> Dict[str, List[Path]]:
    """
    Collect image files organized by Set name.

    Args:
        dataset_dir: Root directory of dataset
        subdirs: List of subdirectory names to scan

    Returns:
        Dictionary mapping Set names to lists of image paths
    """
    images_by_set = defaultdict(list)

    for subdir in subdirs:
        subdir_path = dataset_dir / subdir
        if not subdir_path.exists():
            continue

        for image_path in sorted(subdir_path.glob('*.jpeg')):
            set_name = image_path.stem.split('_')[0]
            images_by_set[set_name].append(image_path)

    return dict(images_by_set)


def stratified_sample(images_by_set: Dict[str, List[Path]],
                     total_count: int,
                     seed: int = 42) -> List[Path]:
    """
    Perform stratified random sampling maintaining Set proportions.

    Args:
        images_by_set: Dictionary mapping Set names to image lists
        total_count: Total number of images to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled image paths
    """
    random.seed(seed)

    total_available = sum(len(images) for images in images_by_set.values())
    sampled_images = []

    for set_name, images in sorted(images_by_set.items()):
        proportion = len(images) / total_available
        count = round(total_count * proportion)
        count = min(count, len(images))

        sampled = random.sample(images, count)
        sampled_images.extend(sampled)
        print(f"  {set_name}: sampled {count}/{len(images)} images ({proportion*100:.1f}%)")

    if len(sampled_images) < total_count:
        remaining = total_count - len(sampled_images)
        all_images = [img for images in images_by_set.values() for img in images]
        available = [img for img in all_images if img not in sampled_images]
        additional = random.sample(available, min(remaining, len(available)))
        sampled_images.extend(additional)
        print(f"  Added {len(additional)} additional images to reach target")

    return sampled_images[:total_count]


def copy_files(image_paths: List[Path], output_dir: Path) -> int:
    """
    Copy images and their annotation files to output directory.

    Args:
        image_paths: List of image file paths
        output_dir: Destination directory

    Returns:
        Number of file pairs copied
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for image_path in image_paths:
        annotation_name = image_path.stem + '_annotation.json'
        annotation_path = image_path.parent / annotation_name

        if not annotation_path.exists():
            print(f"  Warning: No annotation for {image_path.name}, skipping")
            continue

        shutil.copy2(image_path, output_dir / image_path.name)
        shutil.copy2(annotation_path, output_dir / annotation_name)
        copied += 1

    return copied


def save_metadata(output_dir: Path,
                 train_val_samples: List[Path],
                 test_samples: List[Path],
                 seed: int) -> None:
    """
    Save sampling metadata to JSON file.

    Args:
        output_dir: Output directory
        train_val_samples: List of train/val sample paths
        test_samples: List of test sample paths
        seed: Random seed used
    """
    train_val_by_set = defaultdict(list)
    test_by_set = defaultdict(list)

    for img in train_val_samples:
        set_name = img.stem.split('_')[0]
        train_val_by_set[set_name].append(img.name)

    for img in test_samples:
        set_name = img.stem.split('_')[0]
        test_by_set[set_name].append(img.name)

    metadata = {
        'dataset': '3D2cut Single Guyot Dataset',
        'sampling_method': 'stratified_random',
        'random_seed': seed,
        'train_validation': {
            'total': len(train_val_samples),
            'by_set': {k: len(v) for k, v in sorted(train_val_by_set.items())}
        },
        'test': {
            'total': len(test_samples),
            'by_set': {k: len(v) for k, v in sorted(test_by_set.items())}
        }
    }

    metadata_path = output_dir / 'sampling_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")


def sample_dataset(input_dir: Path,
                  output_dir: Path,
                  train_val_count: int,
                  test_count: int,
                  seed: int = 42) -> None:
    """
    Create stratified subset of Guyot dataset.

    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for subset
        train_val_count: Number of train/validation images
        test_count: Number of test images
        seed: Random seed
    """
    print(f"Sampling Guyot dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}\n")

    train_val_images = collect_image_files(input_dir, ['01-TrainAndValidationSet'])
    test_images = collect_image_files(input_dir, ['02-IndependentTestSet'])

    print("Train/Validation sampling:")
    train_val_samples = stratified_sample(train_val_images, train_val_count, seed)

    print(f"\nTest sampling:")
    test_samples = stratified_sample(test_images, test_count, seed + 1)

    print(f"\nCopying train/validation files...")
    train_val_dir = output_dir / '01-TrainAndValidationSet'
    copied_train_val = copy_files(train_val_samples, train_val_dir)
    print(f"  Copied {copied_train_val} image-annotation pairs")

    print(f"\nCopying test files...")
    test_dir = output_dir / '02-IndependentTestSet'
    copied_test = copy_files(test_samples, test_dir)
    print(f"  Copied {copied_test} image-annotation pairs")

    save_metadata(output_dir, train_val_samples, test_samples, seed)

    print(f"\nSampling complete!")
    print(f"Total images: {copied_train_val + copied_test}")
    print(f"  Train/Val: {copied_train_val}")
    print(f"  Test: {copied_test}")


def main():
    """Main function to parse arguments and run sampling."""
    parser = argparse.ArgumentParser(
        description='Sample stratified subset from Guyot dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 200 train/val + 20 test subset
  python sample_guyot_dataset.py --input /path/to/3D2cut_Single_Guyot --output data/guyot_200_20 --train-val 200 --test 20

  # With custom seed
  python sample_guyot_dataset.py --input /path/to/3D2cut_Single_Guyot --output data/guyot_200_20 --train-val 200 --test 20 --seed 123
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing 3D2cut Single Guyot dataset'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for sampled subset'
    )

    parser.add_argument(
        '--train-val',
        type=int,
        default=200,
        help='Number of train/validation images (default: 200)'
    )

    parser.add_argument(
        '--test',
        type=int,
        default=20,
        help='Number of test images (default: 20)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    if not (input_dir / '01-TrainAndValidationSet').exists():
        print(f"Error: Expected subdirectory '01-TrainAndValidationSet' not found in {input_dir}")
        return 1

    if not (input_dir / '02-IndependentTestSet').exists():
        print(f"Error: Expected subdirectory '02-IndependentTestSet' not found in {input_dir}")
        return 1

    try:
        sample_dataset(input_dir, output_dir, args.train_val, args.test, args.seed)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
