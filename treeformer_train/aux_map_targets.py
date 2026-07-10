from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class AuxMapTargetConfig:
    mode: str = "seg_only"
    heatmap_sigma: float = 3.0
    heatmap_cutoff: float = 0.01
    heatmap_target_stride: int = 1
    paf_line_thickness: int = 2
    paf_mask_thickness: int = 6
    direction_target_source: str = "graph_edges"
    direction_encoding: str = "vector"
    direction_tangent_radius: int = 8
    direction_junction_exclusion_radius: int = 6

    @property
    def trains_heatmap(self) -> bool:
        return self.mode in {"seg_heatmap", "seg_heatmap_paf", "aux_maps"}

    @property
    def trains_paf(self) -> bool:
        return self.mode in {"seg_heatmap_paf", "aux_maps"}


def normalize_aux_target_mode(mode: str | None) -> str:
    normalized = str(mode or "seg_only").lower().replace("-", "_")
    aliases = {
        "seg": "seg_only",
        "segmentation": "seg_only",
        "segmentation_only": "seg_only",
        "segmentation_heatmap": "seg_heatmap",
        "seg_heatmap_only": "seg_heatmap",
        "segmentation_heatmap_paf": "seg_heatmap_paf",
        "full_aux": "aux_maps",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"seg_only", "seg_heatmap", "seg_heatmap_paf", "aux_maps"}
    if normalized not in allowed:
        raise ValueError(f"AUX_TARGET_MODE must be one of {sorted(allowed)}, got {mode!r}")
    return normalized


def make_aux_map_target_config(
    *,
    mode: str | None = None,
    heatmap_sigma: float = 3.0,
    heatmap_cutoff: float = 0.01,
    heatmap_target_stride: int = 1,
    paf_line_thickness: int = 2,
    paf_mask_thickness: int = 6,
    direction_target_source: str = "graph_edges",
    direction_encoding: str = "vector",
    direction_tangent_radius: int = 8,
    direction_junction_exclusion_radius: int = 6,
) -> AuxMapTargetConfig:
    if int(heatmap_target_stride) < 1:
        raise ValueError(
            "AUX_HEATMAP_TARGET_STRIDE must be a positive integer, "
            f"got {heatmap_target_stride}"
        )
    direction_target_source = str(direction_target_source).lower().replace("-", "_")
    if direction_target_source not in {"graph_edges", "mask_skeleton"}:
        raise ValueError(
            "AUX_DIRECTION_TARGET_SOURCE must be one of ['graph_edges', 'mask_skeleton'], "
            f"got {direction_target_source!r}"
        )
    direction_encoding = str(direction_encoding).lower().replace("-", "_")
    if direction_encoding not in {"vector", "double_angle"}:
        raise ValueError(
            "AUX_DIRECTION_ENCODING must be one of ['vector', 'double_angle'], "
            f"got {direction_encoding!r}"
        )
    if direction_target_source == "mask_skeleton" and direction_encoding != "double_angle":
        raise ValueError("mask_skeleton direction targets require AUX_DIRECTION_ENCODING='double_angle'")
    if direction_tangent_radius < 1:
        raise ValueError(f"AUX_DIRECTION_TANGENT_RADIUS must be positive, got {direction_tangent_radius}")
    if direction_junction_exclusion_radius < 0:
        raise ValueError(
            "AUX_DIRECTION_JUNCTION_EXCLUSION_RADIUS must be non-negative, "
            f"got {direction_junction_exclusion_radius}"
        )
    return AuxMapTargetConfig(
        mode=normalize_aux_target_mode(mode),
        heatmap_sigma=float(heatmap_sigma),
        heatmap_cutoff=float(heatmap_cutoff),
        heatmap_target_stride=int(heatmap_target_stride),
        paf_line_thickness=int(paf_line_thickness),
        paf_mask_thickness=int(paf_mask_thickness),
        direction_target_source=direction_target_source,
        direction_encoding=direction_encoding,
        direction_tangent_radius=int(direction_tangent_radius),
        direction_junction_exclusion_radius=int(direction_junction_exclusion_radius),
    )


def heatmap_target_size(image_size: tuple[int, int], *, stride: int) -> tuple[int, int]:
    """Return the native heatmap grid size for an image-space target stride."""

    height, width = image_size
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"heatmap target stride must be positive, got {stride}")
    return ((int(height) + stride - 1) // stride, (int(width) + stride - 1) // stride)


def make_node_heatmap(
    nodes: torch.Tensor,
    image_size: tuple[int, int],
    *,
    sigma: float = 3.0,
    cutoff: float = 0.01,
) -> torch.Tensor:
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    if nodes.numel() == 0:
        return torch.from_numpy(heatmap)

    yy, xx = np.mgrid[0:height, 0:width]
    for x_norm, y_norm in nodes.detach().cpu().float().tolist():
        x = float(x_norm) * max(width - 1, 1)
        y = float(y_norm) * max(height - 1, 1)
        gaussian = np.exp(-0.5 * (((xx - x) ** 2 + (yy - y) ** 2) / max(float(sigma), 1e-6) ** 2))
        if cutoff > 0.0:
            gaussian[gaussian < cutoff] = 0.0
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return torch.from_numpy(np.clip(heatmap, 0.0, 1.0))


def make_paf_targets(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    image_size: tuple[int, int],
    *,
    line_thickness: int = 2,
    mask_thickness: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    paf_x = np.zeros((height, width), dtype=np.float32)
    paf_y = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if nodes.numel() == 0 or edges.numel() == 0:
        paf = np.stack([paf_x, paf_y], axis=-1)
        return torch.from_numpy(paf), torch.from_numpy(mask.astype(bool))

    points = _nodes_to_pixel_points(nodes, image_size)
    for start, end in _iter_edge_segments(edges):
        if start < 0 or end < 0 or start >= len(points) or end >= len(points):
            continue
        x0, y0 = points[start]
        x1, y1 = points[end]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = float((dx * dx + dy * dy) ** 0.5)
        if length <= 0.0:
            continue
        ux = dx / length
        uy = dy / length
        cv2.line(paf_x, (x0, y0), (x1, y1), float(ux), thickness=max(1, int(line_thickness)))
        cv2.line(paf_y, (x0, y0), (x1, y1), float(uy), thickness=max(1, int(line_thickness)))
        cv2.line(mask, (x0, y0), (x1, y1), 1, thickness=max(1, int(mask_thickness)))
    paf = np.stack([paf_x, paf_y], axis=-1)
    return torch.from_numpy(np.clip(paf, -1.0, 1.0)), torch.from_numpy(mask.astype(bool))


def make_mask_skeleton_direction_targets(
    segmentation: torch.Tensor,
    *,
    tangent_radius: int = 8,
    junction_exclusion_radius: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a full-width, double-angle local orientation field from a binary mask.

    The field is supervised only in unambiguous foreground.  Endpoints and branch
    junctions are excluded because a single local orientation cannot represent
    multiple outgoing tangents there. Elsewhere, each foreground pixel receives
    its nearest valid skeleton orientation via an EDT index map. The output is
    ``[cos(2*theta), sin(2*theta)]`` so opposite tangent signs are equivalent.
    """

    mask = _as_binary_mask(segmentation)
    height, width = mask.shape
    direction = np.zeros((height, width, 2), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    if not mask.any():
        return torch.from_numpy(direction), torch.from_numpy(valid_mask)

    skeleton = skeletonize(mask)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, neighbor_kernel)
    neighbor_count -= skeleton.astype(np.int16)
    ambiguous_skeleton = np.logical_and(skeleton, neighbor_count != 2)
    ambiguous_region = _dilate_binary(ambiguous_skeleton, radius=junction_exclusion_radius)
    valid_skeleton = np.logical_and(skeleton, ~ambiguous_region)
    if not valid_skeleton.any():
        return torch.from_numpy(direction), torch.from_numpy(valid_mask)

    tangent_seed = _skeleton_tangent_seeds(valid_skeleton, radius=tangent_radius)
    has_tangent = np.linalg.vector_norm(tangent_seed, axis=-1) > 0.0
    if not has_tangent.any():
        return torch.from_numpy(direction), torch.from_numpy(valid_mask)

    # EDT returns nearest zero locations.  Setting tangent seeds to zero makes
    # the index map a direct nearest-tangent lookup for every foreground pixel.
    distances, nearest = distance_transform_edt(~has_tangent, return_indices=True)
    nearest_y, nearest_x = nearest
    propagated_tangent = tangent_seed[nearest_y, nearest_x]
    propagated = _encode_double_angle(propagated_tangent)

    component_labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1]
    seed_components = component_labels[nearest_y, nearest_x]
    valid_mask = np.logical_and(mask, ~ambiguous_region)
    valid_mask &= seed_components == component_labels
    valid_mask &= distances >= 0.0
    direction[valid_mask] = propagated[valid_mask]
    return torch.from_numpy(direction), torch.from_numpy(valid_mask)


def make_aux_map_targets(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    image_size: tuple[int, int],
    config: AuxMapTargetConfig,
    *,
    segmentation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_size
    paf = torch.zeros((height, width, 2), dtype=torch.float32)
    paf_mask = torch.zeros((height, width), dtype=torch.bool)
    heatmap = torch.zeros(heatmap_target_size(image_size, stride=config.heatmap_target_stride), dtype=torch.float32)

    if config.trains_heatmap:
        heatmap = make_node_heatmap(
            nodes,
            tuple(heatmap.shape),
            sigma=config.heatmap_sigma,
            cutoff=config.heatmap_cutoff,
        )
    if config.trains_paf:
        if config.direction_target_source == "mask_skeleton":
            if segmentation is None:
                raise ValueError("mask_skeleton direction targets require the external segmentation mask")
            paf, paf_mask = make_mask_skeleton_direction_targets(
                segmentation,
                tangent_radius=config.direction_tangent_radius,
                junction_exclusion_radius=config.direction_junction_exclusion_radius,
            )
        else:
            paf, paf_mask = make_paf_targets(
                nodes,
                edges,
                image_size,
                line_thickness=config.paf_line_thickness,
                mask_thickness=config.paf_mask_thickness,
            )
    return paf.contiguous(), paf_mask.contiguous(), heatmap.contiguous()


def _as_binary_mask(segmentation: torch.Tensor) -> np.ndarray:
    mask = segmentation.detach().cpu().float().numpy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"segmentation mask must reduce to [H,W], got shape {tuple(segmentation.shape)}")
    return mask > 0.5


def _dilate_binary(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _skeleton_tangent_seeds(skeleton: np.ndarray, *, radius: int) -> np.ndarray:
    height, width = skeleton.shape
    tangents = np.zeros((height, width, 2), dtype=np.float32)
    for y, x in np.argwhere(skeleton):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        local_y, local_x = np.nonzero(skeleton[y0:y1, x0:x1])
        local_y = local_y.astype(np.float32) + y0 - y
        local_x = local_x.astype(np.float32) + x0 - x
        within_radius = local_x * local_x + local_y * local_y <= radius * radius
        points = np.column_stack((local_x[within_radius], local_y[within_radius]))
        if len(points) < 2:
            continue
        centered = points - points.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        tangent = right[0]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-6:
            continue
        tangent = tangent / norm
        tangents[y, x] = tangent.astype(np.float32)
    return tangents


def _encode_double_angle(tangents: np.ndarray) -> np.ndarray:
    x = tangents[..., 0]
    y = tangents[..., 1]
    return np.stack((x * x - y * y, 2.0 * x * y), axis=-1).astype(np.float32)


def _nodes_to_pixel_points(nodes: torch.Tensor, image_size: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = image_size
    points: list[tuple[int, int]] = []
    for x_norm, y_norm in nodes.detach().cpu().float().tolist():
        x = int(round(float(x_norm) * max(width - 1, 1)))
        y = int(round(float(y_norm) * max(height - 1, 1)))
        points.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))
    return points


def _iter_edge_segments(edges: torch.Tensor) -> list[tuple[int, int]]:
    edges_cpu = edges.detach().cpu().long()
    if edges_cpu.ndim == 1:
        values = [int(item) for item in edges_cpu.tolist()]
        return list(zip(values, values[1:], strict=False))

    segments: list[tuple[int, int]] = []
    for row in edges_cpu.tolist():
        values = [int(item) for item in row if int(item) >= 0]
        if len(values) < 2:
            continue
        segments.extend((start, end) for start, end in zip(values, values[1:], strict=False))
    return segments
