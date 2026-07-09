from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_nchw_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().float()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [H,W], [1,H,W] or [N,1,H,W], got {tuple(mask.shape)}")
    return mask.clamp(0.0, 1.0)


def make_stdc_detail_boundary_target(
    mask: torch.Tensor,
    *,
    threshold: float = 0.1,
    scales: tuple[int, ...] = (1, 2, 4),
    support_kernel_size: int = 3,
) -> torch.Tensor:
    """Create an STDC-style multi-scale boundary target from a dense mask.

    The target is derived from the segmentation mask by applying the Laplacian
    boundary operator at multiple scales.  A small fine-scale support keeps the
    target close to the actual boundary so it does not become a second broad
    foreground mask on thin plant stems.
    """

    mask = _as_nchw_mask(mask)
    if not scales:
        raise ValueError("scales must contain at least one positive integer")
    if any(int(scale) <= 0 for scale in scales):
        raise ValueError(f"scales must be positive integers, got {scales!r}")
    if support_kernel_size < 1 or support_kernel_size % 2 == 0:
        raise ValueError(f"support_kernel_size must be a positive odd integer, got {support_kernel_size}")

    kernel = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        device=mask.device,
        dtype=mask.dtype,
    ).view(1, 1, 3, 3)
    height, width = mask.shape[-2:]
    edges = []
    for scale in tuple(int(scale) for scale in scales):
        scaled = (
            mask
            if scale <= 1
            else F.interpolate(mask, scale_factor=1.0 / float(scale), mode="bilinear", align_corners=False)
        )
        edge = F.conv2d(scaled, kernel, padding=1).abs()
        edge = (edge > threshold).to(dtype=mask.dtype)
        if edge.shape[-2:] != (height, width):
            edge = F.interpolate(edge, size=(height, width), mode="nearest")
        edges.append(edge)

    combined = torch.stack(edges, dim=0).amax(dim=0)
    if support_kernel_size > 1:
        fine_support = F.max_pool2d(
            edges[0], kernel_size=support_kernel_size, stride=1, padding=support_kernel_size // 2
        )
        combined = combined * fine_support
    return combined.clamp(0.0, 1.0)
