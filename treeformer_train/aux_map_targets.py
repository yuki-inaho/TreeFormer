from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class AuxMapTargetConfig:
    mode: str = "seg_only"
    heatmap_sigma: float = 3.0
    heatmap_cutoff: float = 0.01
    paf_line_thickness: int = 2
    paf_mask_thickness: int = 6

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
    paf_line_thickness: int = 2,
    paf_mask_thickness: int = 6,
) -> AuxMapTargetConfig:
    return AuxMapTargetConfig(
        mode=normalize_aux_target_mode(mode),
        heatmap_sigma=float(heatmap_sigma),
        heatmap_cutoff=float(heatmap_cutoff),
        paf_line_thickness=int(paf_line_thickness),
        paf_mask_thickness=int(paf_mask_thickness),
    )


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


def make_aux_map_targets(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    image_size: tuple[int, int],
    config: AuxMapTargetConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_size
    paf = torch.zeros((height, width, 2), dtype=torch.float32)
    paf_mask = torch.zeros((height, width), dtype=torch.bool)
    heatmap = torch.zeros((height, width), dtype=torch.float32)

    if config.trains_heatmap:
        heatmap = make_node_heatmap(
            nodes,
            image_size,
            sigma=config.heatmap_sigma,
            cutoff=config.heatmap_cutoff,
        )
    if config.trains_paf:
        paf, paf_mask = make_paf_targets(
            nodes,
            edges,
            image_size,
            line_thickness=config.paf_line_thickness,
            mask_thickness=config.paf_mask_thickness,
        )
    return paf.contiguous(), paf_mask.contiguous(), heatmap.contiguous()


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
