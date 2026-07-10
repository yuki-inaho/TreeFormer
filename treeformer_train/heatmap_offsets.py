from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def make_native_offset_targets(
    nodes_by_image: Sequence[torch.Tensor],
    *,
    target_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sub-cell node offsets for a native heatmap grid.

    Each normalized node is assigned to its nearest heatmap cell.  The target
    stores the remaining offset in native-grid coordinates, constrained to
    ``[-0.5, 0.5]``.  Nodes that collide in one cell cannot be distinguished by
    a single heatmap peak, so only the first node owns that cell.
    """

    height, width = (int(target_size[0]), int(target_size[1]))
    if height < 1 or width < 1:
        raise ValueError(f"target_size must be positive, got {target_size}")
    batch_size = len(nodes_by_image)
    offsets = torch.zeros((batch_size, 2, height, width), device=device, dtype=dtype)
    valid = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.bool)

    for batch_index, raw_nodes in enumerate(nodes_by_image):
        nodes = torch.as_tensor(raw_nodes, device=device, dtype=dtype)
        if nodes.numel() == 0:
            continue
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"nodes must be [N,2], got {tuple(nodes.shape)}")
        coordinates = nodes.clamp(0.0, 1.0)
        x = coordinates[:, 0] * max(width - 1, 1)
        y = coordinates[:, 1] * max(height - 1, 1)
        cell_x = torch.floor(x + 0.5).long().clamp(0, width - 1)
        cell_y = torch.floor(y + 0.5).long().clamp(0, height - 1)
        for node_index in range(nodes.shape[0]):
            row = int(cell_y[node_index])
            column = int(cell_x[node_index])
            if valid[batch_index, 0, row, column]:
                continue
            offsets[batch_index, 0, row, column] = x[node_index] - float(column)
            offsets[batch_index, 1, row, column] = y[node_index] - float(row)
            valid[batch_index, 0, row, column] = True
    return offsets, valid


def bounded_offset_prediction(offset_logits: torch.Tensor) -> torch.Tensor:
    """Map offset logits to the nearest-cell coordinate range."""

    if offset_logits.ndim != 4 or offset_logits.shape[1] != 2:
        raise ValueError(f"offset logits must be [N,2,H,W], got {tuple(offset_logits.shape)}")
    return torch.tanh(offset_logits) * 0.5


def decode_native_heatmap_peaks(
    heatmap_logits: torch.Tensor,
    offset_logits: torch.Tensor | None,
    *,
    threshold: float,
    valid_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Apply 3x3 NMS and optional sub-cell offsets on native heatmap logits.

    Returned tensors are shaped ``[K, 3]`` with ``x, y, confidence`` in native
    heatmap-grid coordinates.  ``valid_mask`` can restrict candidates to a
    predicted segmentation foreground. The caller decides whether to scale
    coordinates to an input image for rendering or graph post-processing.
    """

    if heatmap_logits.ndim != 4 or heatmap_logits.shape[1] != 1:
        raise ValueError(f"heatmap logits must be [N,1,H,W], got {tuple(heatmap_logits.shape)}")
    if offset_logits is not None and offset_logits.shape != (
        heatmap_logits.shape[0],
        2,
        heatmap_logits.shape[2],
        heatmap_logits.shape[3],
    ):
        raise ValueError(
            "offset logits must share batch/spatial shape with heatmap logits, "
            f"got {tuple(offset_logits.shape)} and {tuple(heatmap_logits.shape)}"
        )

    probabilities = torch.sigmoid(heatmap_logits)
    local_maxima = probabilities == F.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
    keep = local_maxima & (probabilities >= float(threshold))
    if valid_mask is not None:
        if valid_mask.shape != heatmap_logits.shape:
            raise ValueError(
                "valid_mask must match heatmap logits shape, "
                f"got {tuple(valid_mask.shape)} and {tuple(heatmap_logits.shape)}"
            )
        keep = keep & valid_mask.to(dtype=torch.bool)
    offsets = bounded_offset_prediction(offset_logits) if offset_logits is not None else None
    decoded: list[torch.Tensor] = []
    for batch_index in range(probabilities.shape[0]):
        rows, columns = torch.nonzero(keep[batch_index, 0], as_tuple=True)
        confidence = probabilities[batch_index, 0, rows, columns]
        x = columns.to(dtype=probabilities.dtype)
        y = rows.to(dtype=probabilities.dtype)
        if offsets is not None:
            x = x + offsets[batch_index, 0, rows, columns]
            y = y + offsets[batch_index, 1, rows, columns]
        decoded.append(torch.stack((x, y, confidence), dim=1))
    return decoded
