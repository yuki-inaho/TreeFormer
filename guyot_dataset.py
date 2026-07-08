from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def parse_guyot_annotation(
    annotation: dict[str, Any], image_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a raw Guyot annotation into normalized nodes and indexed edges."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must contain positive width and height, got {image_size!r}")

    try:
        features = annotation["VineImage"][0]["VineFeature"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("annotation must contain VineImage[0].VineFeature[0]") from exc

    id_to_index: dict[int, int] = {}
    normalized_nodes: list[list[float]] = []

    for index, feature in enumerate(features):
        try:
            feature_id = feature["FeatureID"]
            x, y = feature["FeatureCoordinates"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid VineFeature at index {index}") from exc

        if feature_id in id_to_index:
            raise ValueError(f"duplicate FeatureID: {feature_id!r}")

        id_to_index[feature_id] = index
        normalized_nodes.append([float(x) / width, float(y) / height])

    edges: list[list[int]] = []
    for index, feature in enumerate(features):
        feature_id = feature["FeatureID"]
        parent_id = feature.get("ParentID")

        if parent_id is None or parent_id == feature_id:
            continue
        if parent_id not in id_to_index:
            raise ValueError(f"unknown ParentID {parent_id!r} for FeatureID {feature_id!r}")

        edges.append([id_to_index[parent_id], index])

    nodes_tensor = torch.tensor(normalized_nodes, dtype=torch.float32)
    if edges:
        edges_tensor = torch.tensor(edges, dtype=torch.long)
    else:
        edges_tensor = torch.empty((0, 2), dtype=torch.long)

    return nodes_tensor, edges_tensor


class GuyotDataset(Dataset):
    _SPLIT_DIRS = {
        "train": "01-TrainAndValidationSet",
        "test": "02-IndependentTestSet",
    }
    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in self._SPLIT_DIRS:
            raise ValueError(f"invalid split {split!r}; expected one of {sorted(self._SPLIT_DIRS)}")

        self.root = Path(root)
        self.split = split
        self.data_dir = self.root / self._SPLIT_DIRS[split]
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Guyot split directory not found: {self.data_dir}")

        self.image_files = sorted(
            path for path in self.data_dir.iterdir() if path.is_file() and path.suffix.lower() in self._IMAGE_SUFFIXES
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_files[index]
        annotation_path = image_path.with_name(f"{image_path.stem}_annotation.json")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"annotation file not found for image {image_path.name}: {annotation_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image_bytes = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            image_tensor = image_bytes.view(height, width, 3).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)

        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        nodes, edges = parse_guyot_annotation(annotation, image_size=(width, height))

        return {
            "image": image_tensor,
            "nodes": nodes,
            "edges": edges,
            "filename": image_path.name,
        }


class GuyotTrainingAdapter(Dataset):
    """Adapt raw Guyot samples to the legacy TreeFormer training tuple."""

    def __init__(self, dataset: Dataset, max_size: int = 1000) -> None:
        self.dataset = dataset
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        image = sample["image"]
        nodes = sample["nodes"].to(dtype=torch.float32)
        edges = sample["edges"].to(dtype=torch.long)
        filename = str(sample["filename"])
        name = Path(filename).stem

        image = _resize_image_for_training(image, max_size=self.max_size)
        image = image.mul(2.0).sub(1.0).clamp(-1.0, 1.0).contiguous()
        height, width = image.shape[1:]

        pafs, mask, unet, heatmap = _build_auxiliary_targets(nodes, edges, image_size=(height, width))

        return (
            image,
            name,
            nodes,
            edges,
            pafs,
            mask,
            unet,
            heatmap,
            filename,
        )


def _resize_image_for_training(image: torch.Tensor, max_size: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"image must have shape [C,H,W], got {tuple(image.shape)}")
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    _, height, width = image.shape
    scale = min(1.0, float(max_size) / max(height, width))
    if scale == 1.0:
        return image.to(dtype=torch.float32)

    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.resize(image.to(dtype=torch.float32), [new_height, new_width], antialias=True)


def _build_auxiliary_targets(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_size
    pafs = torch.zeros((height, width, 2), dtype=torch.float32)
    mask = torch.zeros((height, width), dtype=torch.bool)
    unet = torch.zeros((height, width), dtype=torch.float32)
    heatmap = torch.zeros((height, width), dtype=torch.float32)

    if nodes.numel() == 0:
        return pafs, mask, unet, heatmap

    pixel_nodes = _nodes_to_pixels(nodes, image_size=(height, width))
    for x, y in pixel_nodes:
        _draw_disk(mask, x, y, radius=2, value=True)
        _draw_disk(unet, x, y, radius=2, value=1.0)
        _draw_disk(heatmap, x, y, radius=2, value=1.0)

    for start, end in edges.tolist():
        if start < 0 or end < 0 or start >= len(pixel_nodes) or end >= len(pixel_nodes):
            raise ValueError(f"edge index out of range: {(start, end)!r}")
        x0, y0 = pixel_nodes[start]
        x1, y1 = pixel_nodes[end]
        _draw_line_targets(pafs, mask, unet, x0, y0, x1, y1)

    return pafs, mask, unet, heatmap


def _nodes_to_pixels(nodes: torch.Tensor, image_size: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = image_size
    points: list[tuple[int, int]] = []
    for x_norm, y_norm in nodes.tolist():
        x = int(round(float(x_norm) * max(width - 1, 1)))
        y = int(round(float(y_norm) * max(height - 1, 1)))
        points.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))
    return points


def _draw_line_targets(
    pafs: torch.Tensor,
    mask: torch.Tensor,
    unet: torch.Tensor,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    ux = dx / steps
    uy = dy / steps
    norm = (ux * ux + uy * uy) ** 0.5
    if norm > 0:
        ux /= norm
        uy /= norm

    for step in range(steps + 1):
        x = int(round(x0 + dx * step / steps))
        y = int(round(y0 + dy * step / steps))
        _draw_disk(mask, x, y, radius=2, value=True)
        _draw_disk(unet, x, y, radius=2, value=1.0)
        _draw_disk(pafs[..., 0], x, y, radius=2, value=float(ux))
        _draw_disk(pafs[..., 1], x, y, radius=2, value=float(uy))


def _draw_disk(tensor: torch.Tensor, x: int, y: int, radius: int, value) -> None:
    height, width = tensor.shape[:2]
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    tensor[y_min:y_max, x_min:x_max] = value
