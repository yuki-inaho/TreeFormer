#!/usr/bin/env python3
"""
Guyot Dataset Annotation Visualizer

This script visualizes 3D2cut Grapevine Annotation data using OpenCV.
It displays the tree structure with nodes and edges, color-coded by either
FeatureType or BranchLabel.

Usage:
    python visualize_guyot_annotations.py --image <image_path> --annotation <json_path> [--color-mode <mode>]
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


# Color definitions (BGR format for OpenCV)
FEATURE_TYPE_COLORS = {
    'rootCrown': (0, 0, 255),           # Red
    'branchNode': (255, 0, 0),          # Blue
    'pruningCut': (0, 255, 0),          # Green
    'branchToPhotoEdge': (0, 255, 255), # Yellow
    'growingTip': (255, 0, 255),        # Magenta
}

BRANCH_LABEL_COLORS = {
    'root': (0, 0, 128),                # Dark red
    'mainTrunk': (42, 42, 165),         # Brown
    'oldWood': (128, 128, 128),         # Gray
    'courson': (0, 165, 255),           # Orange
    'cane': (0, 255, 128),              # Yellow-green
    'shoot': (0, 255, 0),               # Bright green
    'lateralShoot': (255, 255, 0),      # Cyan
}


def load_annotation(annotation_path):
    """
    Load Guyot annotation JSON file.

    Args:
        annotation_path: Path to the annotation JSON file

    Returns:
        dict: Parsed annotation data
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    if 'VineImage' not in data or len(data['VineImage']) == 0:
        raise ValueError("Invalid annotation format: VineImage not found")

    return data['VineImage'][0]


def build_tree_structure(vine_image):
    """
    Build tree structure from annotation data.

    Args:
        vine_image: VineImage annotation object

    Returns:
        tuple: (nodes dict, edges list)
            nodes: {feature_id: {coords, type, label, ...}}
            edges: [(parent_id, child_id), ...]
    """
    nodes = {}
    edges = []

    # VineFeature is a nested list [[features...]]
    features = vine_image['VineFeature'][0]

    for feature in features:
        feature_id = feature['FeatureID']
        nodes[feature_id] = {
            'coords': tuple(map(int, feature['FeatureCoordinates'])),
            'type': feature['FeatureType'],
            'label': feature['BranchLabel'],
            'parent_id': feature['ParentID']
        }

        # Add edge if parent exists
        if feature['ParentID'] is not None:
            edges.append((feature['ParentID'], feature_id))

    return nodes, edges


def draw_legend(image, color_map, title, start_y=30):
    """
    Draw color legend on the image.

    Args:
        image: Image to draw on
        color_map: Dictionary mapping labels to colors
        title: Legend title
        start_y: Starting y position

    Returns:
        int: Next available y position
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Draw title
    cv2.putText(image, title, (10, start_y), font, font_scale, (255, 255, 255), thickness + 1)
    cv2.putText(image, title, (10, start_y), font, font_scale, (0, 0, 0), thickness)

    y = start_y + 25
    for label, color in color_map.items():
        # Draw color box
        cv2.rectangle(image, (10, y - 10), (30, y + 5), color, -1)
        cv2.rectangle(image, (10, y - 10), (30, y + 5), (0, 0, 0), 1)

        # Draw label text
        cv2.putText(image, label, (35, y), font, font_scale, (255, 255, 255), thickness + 1)
        cv2.putText(image, label, (35, y), font, font_scale, (0, 0, 0), thickness)

        y += 20

    return y


def resize_image_and_coords(image, nodes, max_height=None):
    """
    Resize image and scale node coordinates while maintaining aspect ratio.

    Args:
        image: Input image
        nodes: Dictionary of nodes with coordinates
        max_height: Maximum height for resized image (None = no resize)

    Returns:
        tuple: (resized_image, scaled_nodes)
    """
    if max_height is None or image.shape[0] <= max_height:
        return image, nodes

    height, width = image.shape[:2]
    scale = max_height / height
    new_width = int(width * scale)

    resized_image = cv2.resize(image, (new_width, max_height))

    scaled_nodes = {}
    for node_id, node_data in nodes.items():
        x, y = node_data['coords']
        scaled_nodes[node_id] = {
            **node_data,
            'coords': (int(x * scale), int(y * scale))
        }

    return resized_image, scaled_nodes


def visualize_annotations(image_path, annotation_path, color_mode='feature-type', max_height=None):
    """
    Visualize Guyot annotations on the image.

    Args:
        image_path: Path to the image file
        annotation_path: Path to the annotation JSON file
        color_mode: Color coding mode ('feature-type' or 'branch-label')
        max_height: Maximum height for display (None = original size)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Load annotation
    vine_image = load_annotation(annotation_path)
    nodes, edges = build_tree_structure(vine_image)

    # Resize image and scale coordinates
    image, nodes = resize_image_and_coords(image, nodes, max_height)

    # Select color map based on mode
    if color_mode == 'feature-type':
        color_map = FEATURE_TYPE_COLORS
        node_color_key = 'type'
        legend_title = "Feature Types"
    elif color_mode == 'branch-label':
        color_map = BRANCH_LABEL_COLORS
        node_color_key = 'label'
        legend_title = "Branch Labels"
    else:
        raise ValueError(f"Invalid color mode: {color_mode}")

    # Create a copy for drawing
    vis_image = image.copy()

    # Draw edges (parent-child connections)
    for parent_id, child_id in edges:
        parent_coords = nodes[parent_id]['coords']
        child_coords = nodes[child_id]['coords']

        # Get color based on child node
        node_key = nodes[child_id][node_color_key]
        color = color_map.get(node_key, (128, 128, 128))  # Default to gray

        # Draw line
        cv2.line(vis_image, parent_coords, child_coords, color, 2)

    # Draw nodes
    for node_id, node_data in nodes.items():
        coords = node_data['coords']
        node_key = node_data[node_color_key]
        color = color_map.get(node_key, (128, 128, 128))  # Default to gray

        # Draw node as filled circle
        cv2.circle(vis_image, coords, 5, color, -1)
        # Draw black outline
        cv2.circle(vis_image, coords, 5, (0, 0, 0), 1)

    # Draw legend
    draw_legend(vis_image, color_map, legend_title)

    # Add instruction text at bottom
    height, width = vis_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Press ESC to exit"
    cv2.putText(vis_image, text, (10, height - 10), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_image, text, (10, height - 10), font, 0.6, (0, 0, 0), 1)

    # Display image
    window_name = f"Guyot Annotation Visualization - {Path(image_path).name}"
    cv2.imshow(window_name, vis_image)

    print(f"Displaying annotation visualization for: {Path(image_path).name}")
    print(f"Color mode: {color_mode}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    print("\nPress ESC to exit...")

    # Wait for ESC key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


def main():
    """Main function to parse arguments and run visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize 3D2cut Guyot Grapevine Annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize with feature type colors
  python visualize_guyot_annotations.py --image data/Set00_IMG_3283.jpeg --annotation data/Set00_IMG_3283_annotation.json

  # Visualize with branch label colors
  python visualize_guyot_annotations.py --image data/Set00_IMG_3283.jpeg --annotation data/Set00_IMG_3283_annotation.json --color-mode branch-label

  # Resize display to max height of 800 pixels
  python visualize_guyot_annotations.py --image data/Set00_IMG_3283.jpeg --annotation data/Set00_IMG_3283_annotation.json --max-height 800
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the image file'
    )

    parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='Path to the annotation JSON file'
    )

    parser.add_argument(
        '--color-mode',
        type=str,
        default='feature-type',
        choices=['feature-type', 'branch-label'],
        help='Color coding mode: "feature-type" or "branch-label" (default: feature-type)'
    )

    parser.add_argument(
        '--max-height',
        type=int,
        default=None,
        help='Maximum height for display in pixels (maintains aspect ratio, default: original size)'
    )

    args = parser.parse_args()

    # Validate paths
    image_path = Path(args.image)
    annotation_path = Path(args.annotation)

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_path}")
        return 1

    try:
        visualize_annotations(image_path, annotation_path, args.color_mode, args.max_height)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
