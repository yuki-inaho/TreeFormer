#!/usr/bin/env python3
"""
Guyot Dataset Resizer

Resizes images and rescales annotation coordinates for the 3D2cut Guyot dataset.
Useful for creating smaller datasets for faster processing and testing.

Usage:
    python resize_guyot_dataset.py --input <input_dir> --output <output_dir> --scale 0.25
"""

import argparse
import json
import cv2
from pathlib import Path
from typing import Dict, Any


def resize_image(image_path: Path, output_path: Path, scale: float) -> None:
    """
    Resize an image by the given scale factor.

    Args:
        image_path: Path to input image
        output_path: Path to save resized image
        scale: Scale factor (e.g., 0.25 for 1/4 size)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized)


def scale_coordinates(coords: list, scale: float) -> list:
    """
    Scale coordinate values by the given factor.

    Args:
        coords: List of [x, y] coordinates
        scale: Scale factor

    Returns:
        Scaled coordinates as list
    """
    return [coords[0] * scale, coords[1] * scale]


def resize_annotation(annotation_path: Path, output_path: Path, scale: float) -> None:
    """
    Resize annotation coordinates by the given scale factor.

    Args:
        annotation_path: Path to input annotation JSON
        output_path: Path to save resized annotation
        scale: Scale factor (e.g., 0.25 for 1/4 size)
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    if 'VineImage' not in data or len(data['VineImage']) == 0:
        raise ValueError(f"Invalid annotation format in {annotation_path}")

    vine_image = data['VineImage'][0]
    features = vine_image['VineFeature'][0]

    for feature in features:
        original_coords = feature['FeatureCoordinates']
        feature['FeatureCoordinates'] = scale_coordinates(original_coords, scale)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_dataset(input_dir: Path, output_dir: Path, scale: float) -> Dict[str, int]:
    """
    Process all images and annotations in a directory.

    Args:
        input_dir: Input directory containing images and annotations
        output_dir: Output directory for resized data
        scale: Scale factor

    Returns:
        Dictionary with processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob('*.jpeg'))
    processed_images = 0
    processed_annotations = 0

    for image_path in image_files:
        image_name = image_path.name
        annotation_name = image_path.stem + '_annotation.json'
        annotation_path = input_dir / annotation_name

        if not annotation_path.exists():
            print(f"Warning: No annotation found for {image_name}, skipping")
            continue

        output_image_path = output_dir / image_name
        output_annotation_path = output_dir / annotation_name

        try:
            resize_image(image_path, output_image_path, scale)
            processed_images += 1

            resize_annotation(annotation_path, output_annotation_path, scale)
            processed_annotations += 1

            print(f"Processed: {image_name}")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    return {
        'images': processed_images,
        'annotations': processed_annotations
    }


def main():
    """Main function to parse arguments and run dataset resizing."""
    parser = argparse.ArgumentParser(
        description='Resize Guyot dataset images and annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize to 1/4 size (0.25 scale)
  python resize_guyot_dataset.py --input data/guyot_dataset_sample_5 --output data/guyot_dataset_quarter --scale 0.25

  # Resize to 1/2 size (0.5 scale)
  python resize_guyot_dataset.py --input data/guyot_dataset_sample_5 --output data/guyot_dataset_half --scale 0.5
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images and annotations'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for resized data'
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=0.25,
        help='Scale factor for resizing (default: 0.25 for 1/4 size)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1

    if args.scale <= 0 or args.scale > 1:
        print(f"Error: Scale must be between 0 and 1, got: {args.scale}")
        return 1

    print(f"Processing dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor: {args.scale} ({args.scale * 100:.0f}%)")
    print()

    try:
        stats = process_dataset(input_dir, output_dir, args.scale)
        print()
        print(f"Processing complete!")
        print(f"Images processed: {stats['images']}")
        print(f"Annotations processed: {stats['annotations']}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
